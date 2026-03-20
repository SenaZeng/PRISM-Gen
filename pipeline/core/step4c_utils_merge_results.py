# File: core/step4c_utils_merge_results.py
# -*- coding: utf-8 -*-
"""
Step 4C: Master Summary Generator

Role:
- Acts as the pipeline's central data warehouse, consolidating results from
  Step 3C (RL + xTB/DFT), Step 4A (ADMET + R_ADMET + R_global),
  and optionally Step 4B (PySCF high-accuracy DFT).
- Applies unified SMILES canonicalization to prevent merge mismatches.
- Attaches Filter_Status / Data_Source_Status labels to each molecule
  for downstream funnel visualization and statistical analysis.

Key improvements:
1. [Critical] SMILES canonicalization prevents merge failures caused by
   different string representations of the same molecule.
2. [Critical] Automatically locates the latest CSV matching a filename
   pattern and modification time, rather than hard-coding filenames.
3. [Critical] Adds Filter_Status / Is_Final_Top / Data_Source_Status labels
   to make the filtering logic fully transparent.
"""

import os
import glob
import pandas as pd
from rdkit import Chem
from typing import Optional


def _results_dir():
    """Return the absolute path to ../results (relative to the core/ directory)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "results"))


# ===================== SMILES canonicalization ===================== #

def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string using RDKit (isomericSmiles=True).
    Prevents merge mismatches caused by different representations
    of the same molecule (e.g. "CCO" vs "OCC").
    """
    if not isinstance(smiles, str):
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles
    except Exception:
        return smiles


# ===================== Automatic file discovery ===================== #

def find_latest_file(base_dir: str, pattern: str, required: bool = True) -> Optional[str]:
    """
    Find the most recently modified CSV in base_dir matching the glob pattern.
    Example: pattern="step3c_dft_refined*.csv"
    """
    search_path = os.path.join(base_dir, pattern)
    files = glob.glob(search_path)
    if not files:
        if required:
            raise FileNotFoundError(f"❌ No file found matching pattern: {search_path}")
        else:
            print(f"⚠️ Optional file not found: {search_path}")
            return None

    latest_file = max(files, key=os.path.getmtime)
    print(f"✅ Located file: {os.path.basename(latest_file)} (pattern={pattern})")
    return latest_file


# ===================== Main merge logic ===================== #

def merge_all_steps():
    base_dir = _results_dir()
    print(f">>> Building Step 4C Master Summary (base dir: {base_dir})...")

    # ---------- 1. Load Step 3C (required) ----------
    path_3c = find_latest_file(base_dir, "step3c_dft_refined*.csv", required=True)
    df_3c = pd.read_csv(path_3c)

    if "smiles" not in df_3c.columns:
        raise ValueError(f"'smiles' column missing from {path_3c}")

    print("   -> Canonicalizing Step 3C SMILES...")
    df_3c["merge_key"] = df_3c["smiles"].apply(canonicalize_smiles)

    # ---------- 2. Load Step 4A ADMET (required) ----------
    path_4a = find_latest_file(base_dir, "step4a_admet*.csv", required=True)
    df_4a = pd.read_csv(path_4a)

    if "smiles" not in df_4a.columns:
        raise ValueError(f"'smiles' column missing from {path_4a}")

    print("   -> Canonicalizing Step 4A SMILES...")
    df_4a["merge_key"] = df_4a["smiles"].apply(canonicalize_smiles)

    # ---------- 3. Load Step 4B PySCF (optional) ----------
    path_4b = find_latest_file(base_dir, "*pyscf*.csv", required=False)
    df_4b = None
    if path_4b:
        df_4b = pd.read_csv(path_4b)
        if "smiles" not in df_4b.columns:
            print(f"⚠️ 'smiles' column missing from {path_4b}; PySCF data will be ignored.")
            df_4b = None
        else:
            print("   -> Canonicalizing Step 4B SMILES...")
            df_4b["merge_key"] = df_4b["smiles"].apply(canonicalize_smiles)

    # ============ 4. Merge steps ============ #

    print(">>> Merging Step 3C (base) + Step 4A (ADMET)...")

    # Only bring in new columns from Step 4A to avoid overwriting Step 3C columns
    admet_extra_cols = [
        c for c in df_4a.columns
        if c not in df_3c.columns and c not in ("smiles",)
    ]
    if "merge_key" not in admet_extra_cols:
        admet_extra_cols.append("merge_key")

    df_merge = pd.merge(
        df_3c,
        df_4a[admet_extra_cols],
        on="merge_key",
        how="left",
    )

    # Merge PySCF results if available
    if df_4b is not None:
        print(">>> Step 4B PySCF results found; merging...")
        pyscf_extra_cols = [
            c for c in df_4b.columns
            if c not in df_merge.columns and c not in ("smiles",)
        ]
        if "merge_key" not in pyscf_extra_cols:
            pyscf_extra_cols.append("merge_key")

        df_merge = pd.merge(
            df_merge,
            df_4b[pyscf_extra_cols],
            on="merge_key",
            how="left",
        )
    else:
        print("⚠️ PySCF data not merged (file missing or invalid). Downstream filtering will rely on ADMET / R_global only.")

    # ============ 5. Data_Source_Status: which pipeline stages completed? ============ #

    def determine_data_source_status(row: pd.Series) -> str:
        """
        Label each molecule according to how far it progressed in the pipeline:
          - Step3c_Only:     Only RL + DFT (Step 3C); no ADMET data
          - Step3c+4a:       Has ADMET, but no PySCF high-accuracy DFT
          - Step3c+4a+4b:    Has both ADMET and PySCF results
        """
        has_admet = pd.notna(row.get("Lipinski_Pass"))
        has_pyscf = pd.notna(row.get("PySCF_Gap_eV")) if "PySCF_Gap_eV" in row.index else False

        if has_admet and has_pyscf:
            return "Step3c+4a+4b"
        elif has_admet:
            return "Step3c+4a"
        else:
            return "Step3c_Only"

    df_merge["Data_Source_Status"] = df_merge.apply(determine_data_source_status, axis=1)

    # ============ 6. Filter_Status / Is_Final_Top ============ #

    HERG_THRESHOLD = 0.5      # Consistent with Step 4A
    PYS_CF_GAP_MIN = 4.0      # Reasonable lower bound for PySCF gap (eV)
    PYS_CF_GAP_MAX = 7.0      # Reasonable upper bound for PySCF gap (eV)

    def determine_filter_status(row: pd.Series) -> str:
        """
        Assign a filter label to each molecule based on ADMET, hERG, and PySCF data:
          - Fail_ADMET_Missing : No ADMET result available
          - Fail_Lipinski      : Lipinski rules not satisfied
          - Fail_hERG_HighRisk : hERG blockade probability exceeds threshold
          - Fail_PySCF_Gap     : PySCF result present but gap is outside acceptable range
          - Pass               : Passes all active filters
        """
        lip = row.get("Lipinski_Pass")

        # 1) ADMET data missing
        if pd.isna(lip):
            return "Fail_ADMET_Missing"

        # 2) Lipinski not satisfied
        if lip is False:
            return "Fail_Lipinski"

        # 3) hERG high risk
        if "hERG_Prob" in row.index and pd.notna(row["hERG_Prob"]):
            try:
                hp = float(row["hERG_Prob"])
                if hp >= HERG_THRESHOLD:
                    return "Fail_hERG_HighRisk"
            except Exception:
                pass  # Parsing failure: treat as no hERG data

        # 4) PySCF gap check (if available)
        if "PySCF_Gap_eV" in row.index and pd.notna(row["PySCF_Gap_eV"]):
            try:
                g = float(row["PySCF_Gap_eV"])
                if not (PYS_CF_GAP_MIN <= g <= PYS_CF_GAP_MAX):
                    return "Fail_PySCF_Gap"
            except Exception:
                pass  # Conversion failure: skip PySCF check, treat as ADMET-only pass

        # 5) All checks passed
        return "Pass"

    df_merge["Filter_Status"] = df_merge.apply(determine_filter_status, axis=1)
    df_merge["Is_Final_Top"] = df_merge["Filter_Status"] == "Pass"

    # ============ 7. Clean auxiliary columns and order output columns ============ #

    if "merge_key" in df_merge.columns:
        df_merge.drop(columns=["merge_key"], inplace=True)

    priority_cols = [
        "smiles",
        "Is_Final_Top",
        "Filter_Status",
        "Active_Set",
        "Data_Source_Status",
        "Reward",
        "R0",
        "R_DFT",
        "R_total",
        "R_PySCF",
        "R_total2",
        "R_total2_base",
        "R_total2_base_col",
        "gamma_pyscf",
        "dft_effect",
        "pIC50",
        "QED",
        "Lipinski_Pass",
        "Violations",
        "MW",
        "LogP",
        "HBD",
        "HBA",
        "RotBonds",
        "TPSA",
        "hERG_Prob",
        "hERG_Risk",
        "R_ADMET",
        "R_global",
        "EHOMO_ev",
        "ELUMO_ev",
        "gap_ev",
        "PySCF_Gap_eV",
        "PySCF_Dipole_Debye",
        "PySCF_HOMO_eV",
        "PySCF_LUMO_eV",
        "PySCF_Energy_Eh",
    ]

    ordered_cols = [c for c in priority_cols if c in df_merge.columns]
    rest_cols = [c for c in df_merge.columns if c not in ordered_cols]
    cols_all = ordered_cols + rest_cols

    df_master = df_merge[cols_all]

    out_path = os.path.join(base_dir, "step4c_master_summary.csv")
    df_master.to_csv(out_path, index=False)

    print("==============================================")
    print(f"✅ Step 4C Master Summary generated: {out_path}")
    print(f"   Total molecules: {len(df_master)}")
    print(f"   Is_Final_Top=True count: {df_master['Is_Final_Top'].sum()}")
    print("==============================================")

    print("\n[Funnel statistics - Filter_Status]")
    print(df_master["Filter_Status"].value_counts())
    print("==============================================")


if __name__ == "__main__":
    merge_all_steps()
