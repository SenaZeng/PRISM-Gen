# -*- coding: utf-8 -*-
"""
Step 5A (Patched - Ranking Friendly, No Tier Dependency)
---------------------------------------------------------
Functionality (consistent with original version):
- Select top-N molecules from step4c_master_summary.csv (default sort: R_global)
- Dock each molecule against multiple coronavirus Mpro receptors
- Parse per-target binding energies and compute Broad_Spectrum_Score
- Write step5a_broadspectrum_docking.csv
- Add Broad_Rank / Broad_Rank_Pct columns (ranked by Broad_Spectrum_Score)

Key changes in this patch:
1) Broad_Spectrum_Score computed more faithfully: all finite values across
   all targets contribute to the worst-target (max) score, without the
   previous score < -0.1 pre-filter.
2) Ranking columns are included in the output CSV so that Step 5B can take
   the top-K directly by rank, avoiding empty output from strict tier thresholds.

Note: this script does not use Gold/Silver/Bronze tiers. Ranking is determined
naturally by the score, giving Top1 / TopK.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import concurrent.futures
from typing import Dict, Any, Optional, List
import glob

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# ----------------- Base path setup ----------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

DEFAULT_INPUT_CSV = os.path.join(project_root, "results", "step4c_master_summary.csv")
DEFAULT_OUT_CSV = os.path.join(project_root, "results", "step5a_broadspectrum_docking.csv")
DEFAULT_RECEPTOR_DIR = os.path.join(project_root, "data", "receptors")


# ----------------- Grid box configuration (consistent with current version) ----------------- #
TARGET_CONFIG: Dict[str, Dict[str, Any]] = {
    "SARS_CoV_2": {
        "file": "6W63_gast_clean.pdbqt",
        "box": {"center_x": -17.369, "center_y": 18.129, "center_z": -31.220, "size_x": 24.0, "size_y": 24.0, "size_z": 24.0},
    },
    "SARS_CoV_1": {
        "file": "3V3M_gast_clean.pdbqt",
        "box": {"center_x": 21.845, "center_y": -31.086, "center_z": -4.129, "size_x": 24.0, "size_y": 24.0, "size_z": 24.0},
    },
    "MERS_CoV": {
        "file": "4YLU_gast_clean.pdbqt",
        "box": {"center_x": 27.1, "center_y": -26.2, "center_z": 69.1, "size_x": 24.0, "size_y": 24.0, "size_z": 24.0},
    },
}


# ===================== Stage 2 guardrails configuration =====================
# Purpose: after predE3-based ranking, apply a small set of interpretable
# physicochemical filters to remove structurally problematic molecules from
# the docking queue, improving TopN stability (especially avoiding broad-tail
# outliers).
#
# These parameters encode cross-batch structural knowledge, not batch-specific tuning.
#
# Parameter definitions:
# - pool:              Take the top `pool` rows from the ranked result as the
#                      candidate pool (larger pool reduces risk of running short
#                      after ADMET / gate filtering)
# - mw_min:            Minimum molecular weight; too-small molecules often lack
#                      sufficient anchor points for stable docking (empirical lower
#                      bound: 260 Da)
# - tpsa_min:          Minimum TPSA; very low TPSA correlates with poor pocket
#                      occupancy and broad-tail scores (validated: TPSA >= 35 cuts tails)
# - hba_min:           Minimum H-bond acceptor count; too few HBAs are associated
#                      with hard-negative docking outcomes
# - soft_tpsa_target:  Soft target: TPSA below this value incurs a ranking penalty
#                      rather than hard exclusion (default: 45)
# - soft_logp_max:     Soft target: LogP above this value incurs a ranking penalty
#                      to reduce pose instability and ADMET edge cases (default: 4.8)
# - penalty_tpsa / penalty_logp:
#                      Penalty strength (larger = stricter); generally stable across batches
#
# Three preset profiles:
# - strict  : tighter tail removal (more stable, fewer candidates)
# - balanced: default recommendation
# - loose   : relaxed (use when pool is too small after filtering)
STAGE2_PROFILES = {
    "strict": {
        "pool": 1000,
        "mw_min": 280.0,
        "tpsa_min": 40.0,
        "hba_min": 2,
        "soft_tpsa_target": 55.0,
        "soft_logp_max": 4.6,
        "penalty_tpsa": 0.04,
        "penalty_logp": 0.35,
    },
    "balanced": {
        "pool": 800,
        "mw_min": 260.0,
        "tpsa_min": 35.0,
        "hba_min": 2,
        "soft_tpsa_target": 45.0,
        "soft_logp_max": 4.8,
        "penalty_tpsa": 0.03,
        "penalty_logp": 0.30,
    },
    "loose": {
        "pool": 800,
        "mw_min": 260.0,
        "tpsa_min": 30.0,
        "hba_min": 2,
        "soft_tpsa_target": 40.0,
        "soft_logp_max": 5.2,
        "penalty_tpsa": 0.02,
        "penalty_logp": 0.20,
    },
}
DEFAULT_STAGE2_PROFILE = "balanced"


# ===================== predE3 surrogate model training configuration =====================
# Controls training speed and stability; default parameters are lightweight.
PRED_E3_DEFAULT = {
    "n_estimators": 300,   # Increase to 600-800 once training is stable
    "cv_splits": 3,        # Increase to 5 once training is stable
    "min_samples_leaf": 2,
    "random_state": 0,
}


def resolve_receptor_path(receptor_dir: str, base_filename: str) -> Optional[str]:
    """Prefer *_gast_clean.pdbqt; fall back to the original filename if not found."""
    candidates: List[str] = []
    if base_filename.endswith("_gast_clean.pdbqt"):
        candidates.append(base_filename)
    else:
        stem = base_filename[:-6] if base_filename.endswith(".pdbqt") else base_filename
        candidates.append(f"{stem}_gast_clean.pdbqt")
        candidates.append(base_filename)

    for fn in candidates:
        p = os.path.join(receptor_dir, fn)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def pocket_center_from_pdbqt(rec_pdbqt: str, his_resi: int = 41, cys_resi: int = 145) -> Optional[Dict[str, float]]:
    """
    Parse the Mpro binding pocket center from a receptor PDBQT file.
    Center is computed as the midpoint between His41 (NE2 or ND1) and Cys145 (SG).
    - Does not require a separate PDB file; coordinates are read directly from the PDBQT
    - Falls back from NE2 to ND1 if NE2 is absent
    Returns: {"center_x": ..., "center_y": ..., "center_z": ...} or None
    """
    his_resnames = {"HIS", "HIE", "HID", "HIP"}
    hx = hy = hz = None
    cx = cy = cz = None

    def try_find(his_atom: str) -> bool:
        nonlocal hx, hy, hz, cx, cy, cz
        hx = hy = hz = None
        cx = cy = cz = None
        try:
            with open(rec_pdbqt, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                        continue
                    parts = ln.split()
                    # Expected format: ATOM serial atom resname chain resi x y z ...
                    if len(parts) < 9:
                        continue
                    atom = parts[2]
                    resn = parts[3]
                    try:
                        resi = int(parts[5])
                    except Exception:
                        continue
                    try:
                        x = float(parts[6]); y = float(parts[7]); z = float(parts[8])
                    except Exception:
                        continue

                    if resi == cys_resi and resn == "CYS" and atom == "SG":
                        cx, cy, cz = x, y, z
                    if resi == his_resi and resn in his_resnames and atom == his_atom:
                        hx, hy, hz = x, y, z

                    if (hx is not None) and (cx is not None):
                        break
        except Exception:
            return False

        return (hx is not None) and (cx is not None)

    ok = try_find("NE2")
    if not ok:
        ok = try_find("ND1")
    if not ok:
        return None

    return {
        "center_x": (hx + cx) / 2.0,
        "center_y": (hy + cy) / 2.0,
        "center_z": (hz + cz) / 2.0,
    }


def build_resolved_target_config(receptor_dir: str) -> Dict[str, Dict[str, Any]]:
    """Build a TARGET_CONFIG copy with resolved receptor paths and auto-computed pocket centers."""
    resolved: Dict[str, Dict[str, Any]] = {}
    for virus, conf in TARGET_CONFIG.items():
        base_file = conf["file"]
        rec_path = resolve_receptor_path(receptor_dir, base_file)

        box = dict(conf["box"])  # copy
        if rec_path is None:
            print(f"⚠️ Receptor missing: {virus} | expected {base_file} or *_gast_clean.pdbqt")
            resolved[virus] = {"path": None, "box": box, "file": base_file}
            continue

        # Auto-compute pocket center from His41/Cys145; fall back to config center on failure
        cen = pocket_center_from_pdbqt(rec_path)
        if cen is not None:
            box.update(cen)
            print(
                f"🧭 {virus} box center (auto His41/Cys145) = "
                f"({box['center_x']:.3f}, {box['center_y']:.3f}, {box['center_z']:.3f}) "
                f"| receptor={os.path.basename(rec_path)}"
            )
        else:
            print(
                f"⚠️ {virus}: could not parse His41/Cys145 from receptor; using config center"
                f" | receptor={os.path.basename(rec_path)}"
            )

        resolved[virus] = {"path": rec_path, "box": box, "file": os.path.basename(rec_path)}
    return resolved


def choose_sort_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for c in preferred:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def smiles_to_3d_pdb(smiles: str, out_pdb: str) -> bool:
    """Generate a 3D PDB structure from a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDG()
        ps.randomSeed = 42
        if AllChem.EmbedMolecule(mol, ps) != 0:
            return False
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.MolToPDBFile(mol, out_pdb)
        return True
    except Exception:
        return False


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def pdb_to_pdbqt_with_obabel(in_pdb: str, out_pdbqt: str) -> bool:
    """
    Convert PDB to PDBQT using OpenBabel with pH 7.4 protonation.
    Requires OpenBabel (obabel) to be installed on the system.
    """
    try:
        cmd = ["obabel", in_pdb, "-O", out_pdbqt, "--partialcharge", "gasteiger", "-p", "7.4"]
        res = run_cmd(cmd)
        return res.returncode == 0 and os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 0
    except Exception:
        return False


def run_single_docking(lig_pdbqt: str, rec_pdbqt: str, box: Dict[str, float], cpu_per_task: int = 1) -> float:
    """
    Run AutoDock Vina for a single ligand-receptor pair.
    Returns the best affinity (kcal/mol).
    Requires Vina to be installed on the system.
    """
    out_pdbqt = lig_pdbqt.replace(".pdbqt", f"_out_{os.path.basename(rec_pdbqt)}")
    log_txt = lig_pdbqt.replace(".pdbqt", f"_{os.path.basename(rec_pdbqt)}.log")

    cmd = [
        "vina",
        "--receptor", rec_pdbqt,
        "--ligand", lig_pdbqt,
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--cpu", str(cpu_per_task),
        "--exhaustiveness", "8",
        "--num_modes", "1",
        "--out", out_pdbqt,
        "--log", log_txt,
    ]

    res = run_cmd(cmd)
    if res.returncode != 0:
        return np.nan

    # Parse best affinity from Vina log
    try:
        with open(log_txt, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        # The first mode affinity appears on a line starting with "1 "
        for ln in lines:
            if ln.strip().startswith("1 "):
                parts = ln.split()
                return float(parts[1])
    except Exception:
        pass

    return np.nan


def process_one_molecule(row: pd.Series, target_conf: Dict[str, Dict[str, Any]], tmp_root: str, cpu_per_task: int) -> Optional[Dict[str, Any]]:
    name = str(row.get("name", row.get("id", "")))
    if not name:
        name = f"mol_{int(row.name)}"
    smiles = row.get("smiles", "")
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol_dir = os.path.join(tmp_root, name)
    os.makedirs(mol_dir, exist_ok=True)

    lig_pdb = os.path.join(mol_dir, f"{name}.pdb")
    lig_pdbqt = os.path.join(mol_dir, f"{name}.pdbqt")

    # 1) RDKit -> PDB
    if not smiles_to_3d_pdb(smiles, lig_pdb):
        return None

    # 2) obabel -> PDBQT
    if not pdb_to_pdbqt_with_obabel(lig_pdb, lig_pdbqt):
        return None

    # 3) Dock against each target
    rec_scores: Dict[str, float] = {}
    finite_scores: List[float] = []

    for virus, conf in target_conf.items():
        rec_path = conf.get("path")
        if not rec_path or (not os.path.exists(rec_path)):
            rec_scores[virus] = np.nan
            continue

        score = run_single_docking(
            lig_pdbqt=lig_pdbqt,
            rec_pdbqt=rec_path,
            box=conf["box"],
            cpu_per_task=cpu_per_task,
        )
        rec_scores[virus] = score
        if score is not None and np.isfinite(score):
            finite_scores.append(float(score))

    # 4) Broad_Spectrum_Score: worst-target score (max; more negative is better)
    if not finite_scores:
        broad_score = np.nan
    else:
        broad_score = max(finite_scores)

    record: Dict[str, Any] = {
        "name": name,
        "smiles": smiles,
        "Broad_Spectrum_Score": broad_score,
    }

    # 5) Carry forward upstream metadata if available
    for key in ["pIC50", "Reward", "R_total2", "R_total", "R_ADMET", "R_global"]:
        if key in row.index:
            record[key] = row[key]

    # 6) Per-target scores
    for virus, sc in rec_scores.items():
        record[f"E_{virus}"] = sc

    return record


def compute_rank_pct(scores: pd.Series) -> pd.Series:
    n = int(scores.notna().sum())
    if n <= 0:
        return pd.Series([np.nan] * len(scores), index=scores.index)
    rank = scores.rank(method="min", ascending=True)
    return rank / float(n)


# ===================== predE3 + Stage2 auto-ranking / filtering =====================
def _load_docking_labels(dock_csv_glob: str) -> pd.DataFrame:
    """
    Load historical docking results as supervised labels.
    Expected columns: smiles, E_SARS_CoV_2, E_SARS_CoV_1, E_MERS_CoV, Broad_Spectrum_Score
    When a SMILES appears multiple times, keep the row with the best Broad_Spectrum_Score (most negative).
    """
    import glob

    need = ["smiles", "E_SARS_CoV_2", "E_SARS_CoV_1", "E_MERS_CoV", "Broad_Spectrum_Score"]
    dfs: List[pd.DataFrame] = []
    for p in glob.glob(dock_csv_glob):
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if all(c in d.columns for c in need):
            dfs.append(d[need].copy())
    if not dfs:
        return pd.DataFrame(columns=need)

    dock = pd.concat(dfs, ignore_index=True)
    dock = dock.sort_values("Broad_Spectrum_Score").drop_duplicates("smiles", keep="first")
    return dock


def _train_predE3_and_rank(step4c_df: pd.DataFrame,
                          dock_labels: pd.DataFrame,
                          n_estimators: int,
                          cv_splits: int,
                          n_jobs: int,
                          min_samples_leaf: int = 2,
                          random_state: int = 0) -> pd.DataFrame:
    """
    Train a multi-output RandomForest to predict binding energies for all three targets simultaneously.
    Returns the input df augmented with pred_* columns, sorted by pred_broad (ascending = better).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    if dock_labels.empty:
        raise RuntimeError("No docking labels loaded (dock_labels is empty).")

    m = step4c_df.merge(dock_labels, on="smiles", how="inner")
    if len(m) < 50:
        raise RuntimeError(f"Too few labeled rows to train predE3: {len(m)}")

    # Use numeric Step 4C columns as features; exclude obvious label/status/reward columns
    drop_cols = set([
        "Reward", "R_total", "R_global", "R_total2",
        "Is_Final_Top", "Filter_Status", "Active_Set", "Data_Source_Status",
        "status", "Calc_Method",
    ])
    num_cols = [c for c in step4c_df.columns if pd.api.types.is_numeric_dtype(step4c_df[c])]
    feat_cols = [c for c in num_cols if c not in drop_cols]

    X = m[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    Y = m[["E_SARS_CoV_2", "E_SARS_CoV_1", "E_MERS_CoV"]].astype(float)

    # Lightweight cross-validation (sanity check only)
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    maes = []
    for tr, te in kf.split(X):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            min_samples_leaf=min_samples_leaf,
        )
        model.fit(X.iloc[tr], Y.iloc[tr])
        pred = model.predict(X.iloc[te])
        maes.append(float(np.mean(np.abs(pred - Y.iloc[te].values))))
    print(f"[predE3] CV MAE (mean over 3 targets) = {np.mean(maes):.3f} ± {np.std(maes):.3f}")

    # Full-data training
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X, Y)

    # Full-data prediction
    X_all = step4c_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(X_all.median(numeric_only=True))
    pred_all = model.predict(X_all)

    df_out = step4c_df.copy()
    df_out["pred_E_SARS_CoV_2"] = pred_all[:, 0]
    df_out["pred_E_SARS_CoV_1"] = pred_all[:, 1]
    df_out["pred_E_MERS_CoV"] = pred_all[:, 2]
    df_out["pred_broad"] = df_out[["pred_E_SARS_CoV_2", "pred_E_SARS_CoV_1", "pred_E_MERS_CoV"]].max(axis=1)

    # For compatibility with Step 5A TopN selection: temporarily replace R_global with -pred_broad
    if "R_global" in df_out.columns:
        df_out["R_global_bak_predE3"] = df_out["R_global"]
    df_out["R_global"] = -df_out["pred_broad"]
    df_out = df_out.sort_values("R_global", ascending=False)
    return df_out


def _apply_stage2_guardrails(
    df_ranked: pd.DataFrame,
    cfg: Dict[str, Any],
    use_strict_gate: bool,
    rphys_min: float
) -> pd.DataFrame:
    """
    Stage 2: post-ranking filtering and re-ordering to remove hard-negative tail molecules.
    - Apply the same Active_Set / Is_Final_Top / strict gate logic as main()
    - Take the top `pool` rows
    - Hard gates: MW / TPSA / HBA
    - Soft penalties: TPSA below soft_tpsa_target, LogP above soft_logp_max
    """
    df = df_ranked.copy()

    if use_strict_gate and all(c in df.columns for c in ["Data_Source_Status", "Physical_HardFail", "R_phys"]):
        df = df[
            (df["Data_Source_Status"] == "Step3c+4a+4b") &
            (df["Physical_HardFail"] == False) &
            (df["R_phys"] >= float(rphys_min))
        ].copy()
    elif "Is_Final_Top" in df.columns:
        df = df[df["Is_Final_Top"] == True].copy()
    elif "Active_Set" in df.columns:
        df = df[(df["Active_Set"] == True) | (df["Active_Set"] == 1)].copy()

    pool = int(cfg["pool"])
    df = df.head(pool).copy()

    # Hard gates
    df = df[(df["MW"] >= float(cfg["mw_min"])) &
            (df["TPSA"] >= float(cfg["tpsa_min"])) &
            (df["HBA"] >= int(cfg["hba_min"]))].copy()

    # Soft penalties (affect ranking only, not hard exclusion)
    tpsa = df["TPSA"].astype(float)
    logp = df["LogP"].astype(float) if "LogP" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

    soft_tpsa_target = float(cfg["soft_tpsa_target"])
    soft_logp_max = float(cfg["soft_logp_max"])
    penalty_tpsa = float(cfg["penalty_tpsa"])
    penalty_logp = float(cfg["penalty_logp"])

    pen = penalty_tpsa * np.maximum(0.0, soft_tpsa_target - tpsa) + penalty_logp * np.maximum(0.0, logp - soft_logp_max)

    base = -df["pred_broad"].astype(float) if "pred_broad" in df.columns else df["R_global"].astype(float)
    df["R_global_bak_stage2"] = df["R_global"]
    df["R_global"] = base - pen
    df = df.sort_values("R_global", ascending=False)
    return df


def save_top_n_structures(df_results, tmp_root, top_n=20):
    """
    Extract and save free-ligand and docked poses for the top-N molecules.
    """
    save_dir = os.path.join(project_root, "results", "step5a_top_structures")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Sort ascending: more negative Broad_Spectrum_Score is better
    df_top = df_results.sort_values("Broad_Spectrum_Score", ascending=True).head(top_n)

    print(f"\n>>> Extracting 3D structures for top {top_n} molecules...")
    for _, row in df_top.iterrows():
        mol_name = str(row["name"])
        mol_src_dir = os.path.join(tmp_root, mol_name)
        mol_dst_dir = os.path.join(save_dir, mol_name)
        os.makedirs(mol_dst_dir, exist_ok=True)

        # 1) Free ligand: {name}.pdbqt
        lig = os.path.join(mol_src_dir, f"{mol_name}.pdbqt")
        if os.path.exists(lig):
            shutil.copy(lig, os.path.join(mol_dst_dir, f"{mol_name}_free_ligand.pdbqt"))

        # 2) Docked poses: {name}_out_*.pdbqt
        for p in glob.glob(os.path.join(mol_src_dir, f"{mol_name}_out_*.pdbqt")):
            shutil.copy(p, os.path.join(mol_dst_dir, os.path.basename(p)))

    print(f"✅ Structures saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Step 5A: Broad-Spectrum Docking (Ranking-friendly)")
    parser.add_argument("--use_strict_gate", action="store_true",
                help="Apply strict gate: require PySCF data, Physical_HardFail=False, and R_phys >= threshold")
    parser.add_argument("--rphys_min", type=float, default=0.85,
                help="R_phys lower bound for strict gate (default: 0.85)")
    
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV,
                        help=f"Input Step 4C master summary (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--out_csv", type=str, default=DEFAULT_OUT_CSV,
                        help=f"Output docking results CSV (default: {DEFAULT_OUT_CSV})")
    parser.add_argument("--receptor_dir", type=str, default=DEFAULT_RECEPTOR_DIR,
                        help=f"Directory containing receptor PDBQT files (default: {DEFAULT_RECEPTOR_DIR})")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of molecules from Step 4C to dock (default: 20)")
    
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel molecule workers (default: 4)")
    parser.add_argument("--vina_cpu", type=int, default=1,
                        help="Number of CPU cores per Vina process (default: 1)")
    parser.add_argument("--auto_predE3_stage2", action="store_true",
                        help="Enable: train predE3 from historical docking labels, apply Stage2 guardrails, then dock (recommended)")
    parser.add_argument("--step4c_csv", type=str, default=None,
                        help="Step 4C master CSV (defaults to --input_csv if not set)")
    parser.add_argument("--dock_csv_glob", type=str,
                        default=os.path.join(project_root, "results", "step5a_broadspectrum_docking*.csv"),
                        help="Glob pattern for historical docking CSVs used to train predE3 (default: results/step5a_broadspectrum_docking*.csv)")

    # predE3 training parameters (lightweight defaults)
    parser.add_argument("--predE3_n_estimators", type=int, default=PRED_E3_DEFAULT["n_estimators"])
    parser.add_argument("--predE3_cv_splits", type=int, default=PRED_E3_DEFAULT["cv_splits"])
    parser.add_argument("--predE3_n_jobs", type=int, default=40,
                        help="Parallel cores for predE3 training (recommended: <= CPU quota)")

    # Stage2 parameters: profile first, then individual overrides
    parser.add_argument("--stage2_profile", type=str, default=DEFAULT_STAGE2_PROFILE,
                        choices=list(STAGE2_PROFILES.keys()),
                        help="Stage2 preset: strict / balanced / loose (default: balanced)")
    parser.add_argument("--stage2_pool", type=int, default=None, help="Override profile.pool")
    parser.add_argument("--stage2_mw_min", type=float, default=None, help="Override profile.mw_min")
    parser.add_argument("--stage2_tpsa_min", type=float, default=None, help="Override profile.tpsa_min")
    parser.add_argument("--stage2_hba_min", type=int, default=None, help="Override profile.hba_min")
    parser.add_argument("--stage2_soft_tpsa_target", type=float, default=None, help="Override profile.soft_tpsa_target")
    parser.add_argument("--stage2_soft_logp_max", type=float, default=None, help="Override profile.soft_logp_max")
    parser.add_argument("--stage2_penalty_tpsa", type=float, default=None, help="Override profile.penalty_tpsa")
    parser.add_argument("--stage2_penalty_logp", type=float, default=None, help="Override profile.penalty_logp")

    parser.add_argument("--save_top_structures", action="store_true", default=True,
                        help="Extract and save 3D structure files for the top-N molecules")
    parser.add_argument("--top_n_save", type=int, default=20,
                        help="Number of top molecules whose structures are saved")

    parser.add_argument("--write_intermediate_csv", action="store_true",
                        help="Under auto_predE3_stage2, write results/step4c_master_summary_SORTBY_predE3*.csv for reproducibility")
    parser.add_argument("--no_write_intermediate_csv", action="store_true",
                        help="Under auto_predE3_stage2, suppress intermediate CSV output (default: write)")
    args = parser.parse_args()

    # ----------------- Read Step 4C master summary ----------------- #
    step4c_csv = args.step4c_csv or args.input_csv
    if not os.path.exists(step4c_csv):
        raise FileNotFoundError(f"Input not found: {step4c_csv}")
    df4 = pd.read_csv(step4c_csv)
    if df4.empty:
        print("⚠️ Step 4C input is empty; exiting.")
        return

    # ----------------- Optional: auto predE3 ranking + Stage2 guardrails ----------------- #
    if args.auto_predE3_stage2:
        # 1) Build Stage2 config (profile + overrides)
        cfg = STAGE2_PROFILES.get(args.stage2_profile, STAGE2_PROFILES[DEFAULT_STAGE2_PROFILE]).copy()
        if args.stage2_pool is not None: cfg["pool"] = args.stage2_pool
        if args.stage2_mw_min is not None: cfg["mw_min"] = args.stage2_mw_min
        if args.stage2_tpsa_min is not None: cfg["tpsa_min"] = args.stage2_tpsa_min
        if args.stage2_hba_min is not None: cfg["hba_min"] = args.stage2_hba_min
        if args.stage2_soft_tpsa_target is not None: cfg["soft_tpsa_target"] = args.stage2_soft_tpsa_target
        if args.stage2_soft_logp_max is not None: cfg["soft_logp_max"] = args.stage2_soft_logp_max
        if args.stage2_penalty_tpsa is not None: cfg["penalty_tpsa"] = args.stage2_penalty_tpsa
        if args.stage2_penalty_logp is not None: cfg["penalty_logp"] = args.stage2_penalty_logp

        print(f"🧠 auto_predE3_stage2=ON | stage2_profile={args.stage2_profile} | cfg={cfg}")

        # 2) Load historical docking labels
        dock_labels = _load_docking_labels(args.dock_csv_glob)
        print(f"🧪 predE3 labels loaded: {len(dock_labels)} unique SMILES (glob={args.dock_csv_glob})")

        try:
            # 3) predE3 ranking
            df_ranked = _train_predE3_and_rank(
                step4c_df=df4,
                dock_labels=dock_labels,
                n_estimators=int(args.predE3_n_estimators),
                cv_splits=int(args.predE3_cv_splits),
                n_jobs=int(args.predE3_n_jobs),
                min_samples_leaf=int(PRED_E3_DEFAULT["min_samples_leaf"]),
                random_state=int(PRED_E3_DEFAULT["random_state"]),
            )

            # 4) Stage2 guardrails
            df = _apply_stage2_guardrails(
                df_ranked,
                cfg,
                use_strict_gate=bool(args.use_strict_gate),
                rphys_min=float(args.rphys_min)
            )

            # 5) Optionally write intermediate CSVs (default: write; suppress with --no_write_intermediate_csv)
            do_write = (not args.no_write_intermediate_csv) or args.write_intermediate_csv
            if do_write:
                out1 = os.path.join(project_root, "results", "step4c_master_summary_SORTBY_predE3.csv")
                out2 = os.path.join(project_root, "results", "step4c_master_summary_SORTBY_predE3_stage2.csv")
                df_ranked.to_csv(out1, index=False)
                df.to_csv(out2, index=False)
                print(f"📝 Intermediate CSVs written: {out1} | {out2}")

            print(f"🧱 Stage2 retained rows: {len(df)} (after ADMET + pool + MW/TPSA/HBA + soft-penalty reorder)")
            if df.empty:
                print("⚠️ Stage2 filtering produced empty result; falling back to original R_global ranking.")
                df = df4.copy()

        except Exception as e:
            print(f"⚠️ auto_predE3_stage2 failed ({e}); falling back to original ranking (R_global).")
            df = df4.copy()
    else:
        df = df4.copy()

    # === Final candidate gate (physical veto) ===
    # Apply in order: strict DFT gate > Is_Final_Top > Active_Set
    if args.use_strict_gate and all(c in df.columns for c in ["Data_Source_Status", "Physical_HardFail", "R_phys"]):
        df = df[
            (df["Data_Source_Status"] == "Step3c+4a+4b") &
            (df["Physical_HardFail"] == False) &
            (df["R_phys"] >= float(args.rphys_min))
        ].copy()
        print(f"✅ Strict gate applied: {len(df)} molecules passed PySCF/DFT + R_phys >= {args.rphys_min} for docking.")

    elif "Is_Final_Top" in df.columns:
        df = df[df["Is_Final_Top"] == True].copy()
        print(f"✅ Using Is_Final_Top: {len(df)} molecules for docking.")

    elif "Active_Set" in df.columns:
        df = df[(df["Active_Set"] == True) | (df["Active_Set"] == 1)].copy()
        print(f"✅ Using Active_Set: {len(df)} molecules for docking.")
        if df.empty:
            print("⚠️ Active_Set filtering produced empty result; exiting.")
            return

    # Select TopN (docking is expensive; top_n limits the queue)
    preferred = ["R_global2", "R_total2", "R_total", "R_global", "Reward", "pIC50"]
    sort_col = choose_sort_col(df, preferred)
    
    if sort_col is None:
        sort_col = df.columns[0]
        df_sorted = df.copy()
    else:
        df_sorted = df.sort_values(sort_col, ascending=False).copy()

    df_top = df_sorted.head(args.top_n).copy()
    if "name" not in df_top.columns:
        df_top.insert(0, "name", [f"mol_{i}" for i in range(len(df_top))])

    print(f"📥 Input molecules: {len(df)} | Docking TopN={len(df_top)} | sort_by={sort_col}")

    # Prefer *_gast_clean.pdbqt receptors and auto-compute box centers from His41/Cys145
    resolved_targets = build_resolved_target_config(args.receptor_dir)

    tmp_root = tempfile.mkdtemp(prefix="step5a_docking_")

    results: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for _, row in df_top.iterrows():
            futures.append(ex.submit(process_one_molecule, row, resolved_targets, tmp_root, args.vina_cpu))

        total = len(futures)
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futures):
            done_cnt += 1
            res = fut.result()
            if res is not None:
                results.append(res)
                bs = res.get("Broad_Spectrum_Score")
                bs_str = f"{bs:.2f}" if bs is not None and np.isfinite(bs) else "NaN"
                print(f"[{done_cnt}/{total}] {res.get('name','')} BroadScore={bs_str}")
            else:
                print(f"[{done_cnt}/{total}] Molecule processing failed")

    # Extract structures before cleaning up the temp directory
    if not results:
        print("⚠️ No successful docking results; output file not generated")
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass
        return

    df_res = pd.DataFrame(results)

    if args.save_top_structures:
        save_top_n_structures(df_res, tmp_root, top_n=args.top_n_save)

    # Clean up temporary directory
    try:
        shutil.rmtree(tmp_root)
    except Exception:
        pass

    # Generate ranking columns (lower Broad_Spectrum_Score is better -> ascending)
    df_res["Broad_Rank"] = df_res["Broad_Spectrum_Score"].rank(method="min", ascending=True)
    df_res["Broad_Rank_Pct"] = compute_rank_pct(df_res["Broad_Spectrum_Score"])

    # Arrange columns: fixed base columns first, then extras
    base_cols = ["name", "smiles", "Broad_Spectrum_Score", "Broad_Rank", "Broad_Rank_Pct",
                 "E_SARS_CoV_2", "E_SARS_CoV_1", "E_MERS_CoV"]
    extra_cols = [c for c in df_res.columns if c not in base_cols]
    df_res = df_res[base_cols + extra_cols]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_res.to_csv(args.out_csv, index=False)

    print("\n========================================")
    print(f"✅ Broad-spectrum docking complete. Results saved to: {args.out_csv}")
    print(f"   Molecules docked: {len(df_res)}")
    if df_res["Broad_Spectrum_Score"].notna().any():
        s = df_res["Broad_Spectrum_Score"].dropna()
        print(f"   Broad_Spectrum_Score: min={s.min():.3f} mean={s.mean():.3f} max={s.max():.3f}")
        best = df_res.sort_values("Broad_Spectrum_Score").iloc[0]
        print(f"   Top1: {best['name']} score={best['Broad_Spectrum_Score']:.3f}")
    print("========================================")


if __name__ == "__main__":
    main()
