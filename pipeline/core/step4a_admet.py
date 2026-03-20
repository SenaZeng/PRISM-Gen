# File: core/step4a_admet.py
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED
from rdkit.Chem import AllChem, DataStructs

import joblib  # For loading the hERG prediction model

# Input: Step 3C output (xTB physics-re-ranked results)
INPUT_FILE = "../results/step3c_dft_refined.csv"
OUTPUT_FILE = "../results/step4a_admet_final.csv"

# hERG model location (placed in the results directory)
HERG_MODEL_PATH = "../results/herg_model/herg_rf_model.pkl"
HERG_THRESHOLD = 0.5  # hERG high-risk probability threshold (adjustable)


# ---------- hERG utility functions ---------- #

def smiles_to_fp(smi, radius=2, n_bits=2048):
    """
    Convert a SMILES string to a Morgan fingerprint for hERG model input.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((n_bits,), dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def predict_herg_risk(smiles, herg_model):
    """
    Predict cardiac toxicity risk using a pretrained hERG model.
    Returns:
        prob:    predicted probability of hERG channel blockade (float or None)
        is_risk: whether the molecule is classified as high-risk (bool or None)
    """
    if herg_model is None:
        return None, None

    fp = smiles_to_fp(smiles)
    proba = herg_model.predict_proba(fp.reshape(1, -1))[0, 1]
    is_risk = bool(proba >= HERG_THRESHOLD)
    return float(proba), is_risk


# ---------- ADMET property calculation ---------- #

def calc_admet_props(smiles, herg_model=None):
    """
    Compute basic ADMET properties for a single molecule:
      - Lipinski Rule of Five descriptors (MW, LogP, HBD, HBA, RotBonds)
      - TPSA
      - QED (quantitative estimate of drug-likeness, 0-1)
      - hERG cardiac toxicity prediction (if model is available)
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # 1. Lipinski Rule of Five descriptors
    mw = Descriptors.MolWt(mol)        # Molecular weight (< 500)
    logp = Crippen.MolLogP(mol)        # Lipophilicity (< 5)
    hbd = Lipinski.NumHDonors(mol)     # H-bond donors (< 5)
    hba = Lipinski.NumHAcceptors(mol)  # H-bond acceptors (< 10)
    rot_bonds = Lipinski.NumRotatableBonds(mol)  # Rotatable bonds (< 10)
    
    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    is_lipinski_pass = (violations <= 1)

    # 2. Additional drug-likeness descriptors
    tpsa = Descriptors.TPSA(mol)       # Topological polar surface area (< 140 Å²)
    qed_val = QED.qed(mol)             # Quantitative drug-likeness score (0-1, higher is better)

    # 3. hERG cardiac toxicity prediction (AI-based safety screening)
    herg_prob, herg_risk = predict_herg_risk(smiles, herg_model)

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": int(hbd),
        "HBA": int(hba),
        "RotBonds": int(rot_bonds),
        "TPSA": round(tpsa, 2),
        "QED": round(qed_val, 3),
        "Lipinski_Pass": is_lipinski_pass,
        "Violations": int(violations),
        "hERG_Prob": herg_prob,   # Probability in [0, 1]
        "hERG_Risk": herg_risk,   # True = high risk, False = low risk
    }


def compute_r_admet_and_global(df_final,
                               alpha_lip=0.6,
                               alpha_safety=0.4,
                               beta=2.0):
    """
    Append R_ADMET and R_global columns to df_final.

    Design:
      Lipinski_score = 1.0 (pass) or 0.0 (fail / missing)

      safety_score:
        - Preferred: (1 - hERG_Prob), clipped to [0, 1]
        - Fallback:  QED (0-1) if hERG_Prob is absent
        - Default:   0.5 if both are missing

      R_ADMET = alpha_lip * Lipinski_score + alpha_safety * safety_score

      R_base:
        - Preferred: R_total (Step 3C composite score)
        - Fallback:  Reward
        - Default:   0.0

      R_global = R_base + beta * R_ADMET
    """
    def _row_score(row):
        # Lipinski component
        lip_pass = row.get("Lipinski_Pass")
        lip_score = 1.0 if (lip_pass is True) else 0.0

        # Safety component: prefer hERG, fall back to QED
        safety_score = None
        if "hERG_Prob" in row.index and pd.notna(row["hERG_Prob"]):
            try:
                safety_score = 1.0 - float(row["hERG_Prob"])
            except Exception:
                safety_score = None

        if safety_score is None:
            if "QED" in row.index and pd.notna(row["QED"]):
                try:
                    safety_score = float(row["QED"])
                except Exception:
                    safety_score = None

        if safety_score is None:
            safety_score = 0.5  # Neutral default

        # Clip to [0, 1]
        safety_score = max(0.0, min(1.0, safety_score))

        r_admet = alpha_lip * lip_score + alpha_safety * safety_score

        # Upstream base score
        if "R_total" in row.index and pd.notna(row["R_total"]):
            base = float(row["R_total"])
        elif "Reward" in row.index and pd.notna(row["Reward"]):
            base = float(row["Reward"])
        else:
            base = 0.0

        r_global = base + beta * r_admet

        return pd.Series({
            "R_ADMET": round(r_admet, 3),
            "R_global": round(r_global, 3),
        })

    scores = df_final.apply(_row_score, axis=1)
    df_with_scores = pd.concat([df_final, scores], axis=1)
    return df_with_scores


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(current_dir, INPUT_FILE)
    out_path = os.path.join(current_dir, OUTPUT_FILE)
    herg_model_path = os.path.join(current_dir, HERG_MODEL_PATH)

    if not os.path.exists(in_path):
        print(f"❌ Input file not found: {in_path}")
        return

    # Attempt to load the hERG model
    herg_model = None
    if os.path.exists(herg_model_path):
        print(f"🧪 Loading hERG prediction model: {herg_model_path}")
        try:
            herg_model = joblib.load(herg_model_path)
            print("✅ hERG model loaded. AI-based cardiac toxicity screening enabled.")
        except Exception as e:
            print(f"⚠️ Failed to load hERG model; hERG prediction will be skipped. Error: {e}")
            herg_model = None
    else:
        print(f"⚠️ hERG model file not found: {herg_model_path}. hERG prediction will be skipped.")

    print(f"Reading Step 3C results: {in_path}")
    df = pd.read_csv(in_path)
    
    print(f"Computing ADMET properties for {len(df)} molecules...")

    admet_data = []
    empty_admet = {
        "MW": None,
        "LogP": None,
        "HBD": None,
        "HBA": None,
        "RotBonds": None,
        "TPSA": None,
        "QED": None,
        "Lipinski_Pass": None,
        "Violations": None,
        "hERG_Prob": None,
        "hERG_Risk": None,
    }

    for idx, row in df.iterrows():
        smiles = row.get("smiles", None)
        if pd.isna(smiles):
            admet_data.append(empty_admet.copy())
            continue

        props = calc_admet_props(smiles, herg_model=herg_model)
        if props:
            admet_data.append(props)
        else:
            admet_data.append(empty_admet.copy())

    # Merge ADMET data
    df_admet = pd.DataFrame(admet_data)
    df_final = pd.concat([df, df_admet], axis=1)

    # Compute R_ADMET and R_global
    df_final = compute_r_admet_and_global(df_final)

    # Lipinski pass statistics
    df_lipinski_pass = df_final[df_final["Lipinski_Pass"] == True].copy()

    # hERG statistics (if prediction was run)
    if "hERG_Risk" in df_final.columns:
        df_herg_risk_true = df_final[df_final["hERG_Risk"] == True]
        df_lipinski_herg_pass = df_final[
            (df_final["Lipinski_Pass"] == True) &
            ((df_final["hERG_Risk"] == False) | (df_final["hERG_Risk"].isna()))
        ].copy()
    else:
        df_herg_risk_true = pd.DataFrame(columns=df_final.columns)
        df_lipinski_herg_pass = df_lipinski_pass.copy()

    # === Compute Active_Set flag (ADMET Pass) ===
    # Serves as the unified entry point for Steps 4B / 5A / 5B.
    # Rule: Active_Set = True iff Lipinski_Pass == True AND hERG is not high-risk.
    if "Lipinski_Pass" in df_final.columns:
        lip_ok = (df_final["Lipinski_Pass"] == True)
    else:
        lip_ok = pd.Series([True] * len(df_final))

    if "hERG_Risk" in df_final.columns:
        herg_ok = (df_final["hERG_Risk"].isna()) | (df_final["hERG_Risk"] == False)
    elif "hERG_Prob" in df_final.columns:
        herg_ok = (df_final["hERG_Prob"].isna()) | (df_final["hERG_Prob"] < HERG_THRESHOLD)
    else:
        herg_ok = pd.Series([True] * len(df_final))

    df_final["Active_Set"] = (lip_ok & herg_ok)
    try:
        print(f"✅ Active_Set (ADMET Pass) count = {int(df_final['Active_Set'].sum())}")
    except Exception:
        pass

    df_final.to_csv(out_path, index=False)
    
    print("-" * 30)
    print(f"✅ ADMET evaluation complete. Results saved to: {out_path}")
    print(f"📊 Total molecules: {len(df)}")
    print(f"💊 Molecules passing Lipinski rules: {len(df_lipinski_pass)}")

    if herg_model is not None:
        print(f"❤️ Molecules predicted as hERG high-risk: {len(df_herg_risk_true)}")
        print(f"🛡️ Molecules passing both Lipinski and hERG screening: {len(df_lipinski_herg_pass)}")
    else:
        print("⚠️ hERG prediction was not enabled; only Lipinski filtering was applied.")

    print("-" * 30)
    
    # Display top 5 final candidates (sorted by R_global, then R_total)
    if not df_lipinski_herg_pass.empty:
        df_for_top = df_lipinski_herg_pass.copy()

        sort_col = None
        if "R_global" in df_for_top.columns:
            sort_col = "R_global"
        elif "R_total" in df_for_top.columns:
            sort_col = "R_total"

        if sort_col is not None:
            df_for_top = df_for_top.sort_values(sort_col, ascending=False)

        cols_to_show = ["smiles"]
        for col in ["pIC50", "gap_ev", "LogP", "QED", "R_total", "R_ADMET", "R_global", "Lipinski_Pass", "hERG_Prob"]:
            if col in df_for_top.columns:
                cols_to_show.append(col)

        print("🏆 Top 5 final candidates (composite activity + electronic stability + drug-likeness/safety):")
        print(df_for_top[cols_to_show].head(5))


if __name__ == "__main__":
    main()
