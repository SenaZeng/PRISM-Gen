# -*- coding: utf-8 -*-
"""
Step 5B (Patched - Ranking Only)
---------------------------------
Goal: remove Gold/Silver/Bronze tiering (which caused empty output when
thresholds were too strict) in favour of a simple ranking approach.

Pipeline:
1) Merge the Step 4C master summary with Step 5A docking results using
   canonical SMILES keys.
2) For rows with a valid docking score, sort by Broad_Spectrum_Score
   (lower is better) and generate Docking_Rank / Docking_Rank_Pct columns.
3) Write the merged master summary (with ranking columns).
4) Write the final candidates file: by default, the docking top-K rows
   (at least Top 1 is always guaranteed). If no docking scores exist,
   fall back to sorting by an upstream composite score (also non-empty).

Default paths are unchanged and compatible with the existing pipeline:
- Step 4C master:  results/step4c_master_summary.csv
- Step 5A docking: results/step5a_broadspectrum_docking.csv
- Output master:   results/step5b_master_summary.csv
- Output final:    results/step5b_final_candidates.csv
"""

import os
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
from rdkit import Chem

# ----------------- Base path setup ----------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

DEFAULT_MASTER_4C = os.path.join(project_root, "results", "step4c_master_summary.csv")
DEFAULT_DOCK_5A = os.path.join(project_root, "results", "step5a_broadspectrum_docking.csv")

DEFAULT_OUT_MASTER = os.path.join(project_root, "results", "step5b_master_summary.csv")
DEFAULT_OUT_FINAL = os.path.join(project_root, "results", "step5b_final_candidates.csv")


# ----------------- SMILES canonicalization ----------------- #
def canonicalize_smiles(smiles: str) -> str:
    if not isinstance(smiles, str):
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles
    except Exception:
        return smiles


def compute_rank_pct(scores: pd.Series) -> pd.Series:
    """
    Compute rank percentile: rank / n (best ~ 1/n).
    Only rows with non-null docking scores contribute to n.
    """
    n = int(scores.notna().sum())
    if n <= 0:
        return pd.Series([np.nan] * len(scores), index=scores.index)
    rank = scores.rank(method="min", ascending=True)  # Lower score is better
    pct = rank / float(n)
    return pct


def choose_fallback_sort_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for c in preferred:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="Step 5B: merge + ranking (no tiers)")
    ap.add_argument("--master_4c", type=str, default=DEFAULT_MASTER_4C, help="Step 4C master summary")
    ap.add_argument("--dock_5a", type=str, default=DEFAULT_DOCK_5A, help="Step 5A docking results")
    ap.add_argument("--out_master", type=str, default=DEFAULT_OUT_MASTER, help="Output master summary path")
    ap.add_argument("--out_final", type=str, default=DEFAULT_OUT_FINAL, help="Output final candidates path (top-K by rank)")

    ap.add_argument("--final_top_k", type=int, default=20, help="Number of final candidates to output (default 20; at least 1 is always written)")
    ap.add_argument("--fallback_top_k", type=int, default=None, help="Top-K used when no docking scores exist (defaults to final_top_k)")
    ap.add_argument("--fallback_sort_cols", nargs="+",
                    default=["R_global2", "R_total2", "R_total", "R_global", "Reward", "pIC50"],
                    help="Fallback sort column priority when no docking scores are available (higher is better)")

    args = ap.parse_args()
    if args.fallback_top_k is None:
        args.fallback_top_k = args.final_top_k

    if not os.path.exists(args.master_4c):
        raise FileNotFoundError(f"Step 4C master not found: {args.master_4c}")
    if not os.path.exists(args.dock_5a):
        raise FileNotFoundError(f"Step 5A docking results not found: {args.dock_5a}")

    df_master = pd.read_csv(args.master_4c)
    df_dock = pd.read_csv(args.dock_5a)

    if "smiles" not in df_master.columns:
        raise ValueError("Step 4C master is missing the 'smiles' column")
    if "smiles" not in df_dock.columns:
        raise ValueError("Step 5A docking results are missing the 'smiles' column")

    # If docking output lacks Broad_Spectrum_Score, attempt to derive it from E_* columns
    if "Broad_Spectrum_Score" not in df_dock.columns:
        e_cols = [c for c in df_dock.columns if c.startswith("E_")]
        if not e_cols:
            raise ValueError("Step 5A docking output lacks both Broad_Spectrum_Score and E_* columns")
        # Worst-target score across all receptors (max, since more negative is better)
        df_dock["Broad_Spectrum_Score"] = df_dock[e_cols].max(axis=1)

    # Canonical merge keys
    df_master["merge_key"] = df_master["smiles"].apply(canonicalize_smiles)
    df_dock["merge_key"] = df_dock["smiles"].apply(canonicalize_smiles)

    # Only bring in new columns from docking (avoid overwriting master columns)
    dock_cols = [c for c in df_dock.columns if c not in df_master.columns and c != "smiles"]
    if "merge_key" not in dock_cols:
        dock_cols.append("merge_key")

    df_merge = pd.merge(df_master, df_dock[dock_cols], on="merge_key", how="left")

    # Ranking and percentile (meaningful only for rows with a docking score)
    df_merge["Docking_Rank"] = df_merge["Broad_Spectrum_Score"].rank(method="min", ascending=True)
    df_merge["Docking_Rank_Pct"] = compute_rank_pct(df_merge["Broad_Spectrum_Score"])

    # Remove auxiliary column
    df_merge.drop(columns=["merge_key"], inplace=True)

    # Write master summary
    os.makedirs(os.path.dirname(args.out_master), exist_ok=True)
    df_merge.to_csv(args.out_master, index=False)

    # Write final candidates (guaranteed non-empty)
    df_valid = df_merge[df_merge["Broad_Spectrum_Score"].notna()].copy()

    # Apply quality gate: prefer strict physical/DFT criteria, then Is_Final_Top, then Active_Set
    if all(c in df_valid.columns for c in ["Data_Source_Status", "Physical_HardFail", "R_phys"]):
        df_valid = df_valid[
            (df_valid["Data_Source_Status"] == "Step3c+4a+4b") &
            (df_valid["Physical_HardFail"] == False) &
            (df_valid["R_phys"] >= 0.85)
        ].copy()
    elif "Is_Final_Top" in df_valid.columns:
        df_valid = df_valid[df_valid["Is_Final_Top"] == True].copy()
    elif "Active_Set" in df_valid.columns:
        df_valid = df_valid[df_valid["Active_Set"] == True].copy()

    selection_mode = ""

    if not df_valid.empty:
        # Sort by docking score (lower is better), then by rank for tie-breaking
        df_valid = df_valid.sort_values(["Broad_Spectrum_Score", "Docking_Rank"], ascending=[True, True])
        k = max(1, int(args.final_top_k))
        df_final = df_valid.head(k).copy()
        selection_mode = f"docking_rank_top{k}"
    else:
        # Fallback: no valid docking scores (e.g. Step 5A failed or was not run)
        fallback_col = choose_fallback_sort_col(df_merge, args.fallback_sort_cols)
        k = max(1, int(args.fallback_top_k))
        if fallback_col is None:
            # Last resort: return first k rows
            df_final = df_merge.head(k).copy()
            selection_mode = f"fallback_first{k}"
        else:
            df_final = df_merge.sort_values(fallback_col, ascending=False).head(k).copy()
            selection_mode = f"fallback_{fallback_col}_top{k}"

    df_final["Selection_Mode"] = selection_mode
    os.makedirs(os.path.dirname(args.out_final), exist_ok=True)
    df_final.to_csv(args.out_final, index=False)

    # Print summary
    print("==============================================")
    print(f"✅ Step 5B master summary: {args.out_master}")
    print(f"   Total molecules: {len(df_merge)}")
    print(f"   Molecules with docking scores: {int(df_merge['Broad_Spectrum_Score'].notna().sum())}")
    if df_merge['Broad_Spectrum_Score'].notna().any():
        print("   Docking Broad_Spectrum_Score statistics:")
        s = df_merge['Broad_Spectrum_Score'].dropna()
        print(f"   min={s.min():.3f} mean={s.mean():.3f} max={s.max():.3f}")
    print("----------------------------------------------")
    print(f"✅ Step 5B final candidates: {args.out_final}")
    print(f"   Selection_Mode={selection_mode} | count: {len(df_final)}")
    if 'Broad_Spectrum_Score' in df_final.columns and df_final['Broad_Spectrum_Score'].notna().any():
        best = df_final.iloc[0]
        print(f"   Top1 Broad_Spectrum_Score={best['Broad_Spectrum_Score']:.3f} | smiles={best.get('smiles','')[:60]}...")
    print("==============================================")


if __name__ == "__main__":
    main()
