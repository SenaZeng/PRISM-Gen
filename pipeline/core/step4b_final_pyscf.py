# File: core/step4b_final_pyscf.py
# -*- coding: utf-8 -*-
"""
Step 4B: Parallel PySCF Calculation (Aligned + Top20 Strategy)
--------------------------------------------------------------
Goal: introduce the "Top20 = 10 exploit + 10 explore (diversity)" selection
strategy as a built-in upgrade, while keeping all externally visible behaviour
identical to the previous version:

Unchanged from previous version:
- CLI arguments: --input_file / --output_file / --top_k / --workers (names and defaults)
- Log format: 📥 / 🔍 / 🚀 / ⏳ / ✅ emoji-prefixed messages
- Default output filename: ../results/step4b_top_molecules_pyscf.csv (Step 4C requires no changes)
- Column order rule: ["smiles", "PySCF_Gap_eV", "PySCF_Dipole_Debye", "R_global"] placed first

New in this version ("silent upgrade"):
- Lipinski + hERG filtering is retained (if columns are present), but TopK selection
  now uses a CandidatePool (default 200 = 100/50/50) followed by
  Exploitation / Exploration combination, padded to top_k.
"""

import os
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from pyscf import gto, scf, dft


# ================= Default configuration =================
DEFAULT_INPUT_FILE = "../results/step4a_admet_final.csv"
DEFAULT_OUTPUT_FILE = "../results/step4b_top_molecules_pyscf.csv"
DEFAULT_TOP_K = 20
DEFAULT_WORKERS = 40
# =========================================================


def canonicalize_smiles(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return Chem.MolToSmiles(m, isomericSmiles=True)
        return smiles.strip()
    except Exception:
        return smiles.strip()


def choose_sort_col(df: pd.DataFrame, sort_cols: List[str]) -> Optional[str]:
    for col in sort_cols:
        if col in df.columns:
            return col
    return None


def generate_xyz_string(mol: Chem.Mol):
    """Generate 3D coordinates with RDKit and convert to an XYZ string for PySCF."""
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDG()
    ps.randomSeed = 42
    if AllChem.EmbedMolecule(mol, ps) == -1:
        raise RuntimeError("RDKit EmbedMolecule returned -1")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    xyz_lines = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        xyz_lines.append(f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}")
    return "\n".join(xyz_lines)


def run_pyscf_task(task: Tuple[str, int]):
    """
    Single-molecule PySCF calculation task (intended for parallel invocation).
    task: (smiles, row_idx)
    Returns a dict on success; None on failure.
    """
    smiles, row_idx = task
    try:
        mol_rd = Chem.MolFromSmiles(smiles)
        if mol_rd is None:
            return None

        xyz_str = generate_xyz_string(mol_rd)

        mol = gto.M(
            atom=xyz_str,
            basis="6-31g*",
            charge=0,
            spin=0,
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        mf = scf.newton(mf)

        energy = mf.kernel()
        if not mf.converged:
            return None

        mo_energies = mf.mo_energy
        nocc = mol.nelectron // 2
        homo = mo_energies[nocc - 1] * 27.2114
        lumo = mo_energies[nocc] * 27.2114
        gap = lumo - homo

        dipole_vec = mf.dip_moment(mol, unit="Debye", verbose=0)
        dipole_mag = np.linalg.norm(dipole_vec)

        return {
            "_row_idx": int(row_idx),              # Internal alignment key (not written to output)
            "smiles": smiles,
            "PySCF_Energy_Eh": float(energy),
            "PySCF_HOMO_eV": round(float(homo), 3),
            "PySCF_LUMO_eV": round(float(lumo), 3),
            "PySCF_Gap_eV": round(float(gap), 3),
            "PySCF_Dipole_Debye": round(float(dipole_mag), 3),
            "Calc_Method": "B3LYP/6-31G*",
        }

    except Exception:
        return None


# ------------------ TopK selection strategy (built-in) ------------------

def _ecfp4_fps(smiles_list: List[str], n_bits: int = 2048, radius: int = 2):
    fps = []
    ok = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            fps.append(None)
            ok.append(False)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits))
            ok.append(True)
    return fps, ok


def _butina_clusters(fps: List, cutoff_dist: float = 0.4) -> List[List[int]]:
    dists = []
    n = len(fps)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, n, cutoff_dist, isDistData=True)
    return [list(c) for c in clusters]


def select_candidates_topk(df_clean: pd.DataFrame, top_k: int, used_sort_col: str,
                           pool_n1: int = 100, pool_n2: int = 50, pool_n3: int = 50,
                           exploit_k: int = 10, cluster_cutoff_dist: float = 0.4) -> pd.DataFrame:
    """
    Select top_k candidates using a three-stage strategy:
    1. Build candidate pool (N1 by main score + N2 by activity + N3 by drug-likeness), deduplicated
    2. Exploitation: top exploit_k by used_sort_col
    3. Exploration: Butina cluster representatives for diversity
    4. Pad to top_k if needed
    """
    df = df_clean.copy()
    df["_canon"] = df["smiles"].apply(canonicalize_smiles)

    # N1: top by main score
    part1 = df.sort_values(used_sort_col, ascending=False).head(pool_n1) if used_sort_col else df.head(0)

    # N2: top by activity
    act_col = None
    for c in ["Reward", "pIC50"]:
        if c in df.columns:
            act_col = c
            break
    part2 = df.sort_values(act_col, ascending=False).head(pool_n2) if act_col else df.head(0)

    # N3: drug-likeness (QED high + SA low) split
    part3 = df.head(0)
    n3a = pool_n3 // 2
    n3b = pool_n3 - n3a
    if "QED" in df.columns and df["QED"].notna().any() and n3a > 0:
        part3 = pd.concat([part3, df.sort_values("QED", ascending=False).head(n3a)], ignore_index=False)
    if "SA" in df.columns and df["SA"].notna().any() and n3b > 0:
        part3 = pd.concat([part3, df.sort_values("SA", ascending=True).head(n3b)], ignore_index=False)

    pool = pd.concat([part1, part2, part3], ignore_index=False).drop_duplicates(subset=["_canon"], keep="first")

    # Exploitation
    exploit = pool.sort_values(used_sort_col, ascending=False).head(min(exploit_k, top_k)).copy()
    exploit_keys = set(exploit["_canon"].tolist())

    # Exploration (cluster representatives)
    need_explore = max(0, top_k - len(exploit))
    remain = pool[~pool["_canon"].isin(exploit_keys)].copy()

    explore = remain.head(0)
    if need_explore > 0 and not remain.empty:
        smi_list = remain["smiles"].astype(str).tolist()
        fps, ok = _ecfp4_fps(smi_list)
        keep_idx = [i for i, flag in enumerate(ok) if flag]
        if len(keep_idx) == 0:
            explore = remain.sort_values(used_sort_col, ascending=False).head(need_explore)
        else:
            remain_ok = remain.iloc[keep_idx].copy()
            fps_ok = [fps[i] for i in keep_idx]
            clusters = _butina_clusters(fps_ok, cutoff_dist=cluster_cutoff_dist)

            picks = []
            for cl in clusters:
                sub = remain_ok.iloc[cl].sort_values(used_sort_col, ascending=False)
                picks.append(sub.iloc[0])
            explore = pd.DataFrame(picks).sort_values(used_sort_col, ascending=False).head(need_explore)

    candidates = pd.concat([exploit, explore], ignore_index=False)
    candidates = candidates.drop_duplicates(subset=["_canon"], keep="first")

    # Padding: first from pool remainder, then from the full df
    if len(candidates) < top_k:
        need = top_k - len(candidates)
        pool_rem = pool[~pool["_canon"].isin(set(candidates["_canon"].tolist()))].sort_values(used_sort_col, ascending=False)
        candidates = pd.concat([candidates, pool_rem.head(need)], ignore_index=False)

    if len(candidates) < top_k:
        need = top_k - len(candidates)
        df_rem = df[~df["_canon"].isin(set(candidates["_canon"].tolist()))].sort_values(used_sort_col, ascending=False)
        candidates = pd.concat([candidates, df_rem.head(need)], ignore_index=False)

    candidates = candidates.head(top_k).copy()
    candidates.drop(columns=["_canon"], inplace=True, errors="ignore")
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Step 4B: Parallel PySCF Calculation")
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE, help="Input file (Step 4A output)")
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE, help="Output file")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of top molecules to compute")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel processes")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return

    df = pd.read_csv(args.input_file)
    print(f"📥 Candidate molecules loaded: {len(df)}")

    # Filter (Lipinski + hERG) — consistent with previous version
    df_clean = df.copy()

    # Active_Set filter (unified entry point): if Step 4A wrote Active_Set, use it
    if "Active_Set" in df_clean.columns:
        df_clean = df_clean[df_clean["Active_Set"] == True].copy()

    # Lipinski filter
    if "Lipinski_Pass" in df_clean.columns:
        df_clean = df_clean[df_clean["Lipinski_Pass"] == True].copy()

    # hERG filter (exclude high-risk)
    if "hERG_Risk" in df_clean.columns:
        df_clean = df_clean[
            (df_clean["hERG_Risk"] == False) | (df_clean["hERG_Risk"].isna())
        ].copy()

    print(f"🔍 Molecules remaining after ADMET filtering: {len(df_clean)}")

    if df_clean.empty:
        print("⚠️ No candidates after filtering; writing empty output file")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        pd.DataFrame(columns=["smiles"]).to_csv(args.output_file, index=False)
        return

    # Sorting criterion (prefer R_global)
    sort_cols = ["R_global", "R_total", "R0", "Reward", "pIC50"]
    used_sort_col = choose_sort_col(df_clean, sort_cols)
    if used_sort_col:
        df_clean = df_clean.sort_values(used_sort_col, ascending=False)

    # Select top K (built-in upgrade: CandidatePool + exploit/explore)
    candidates = select_candidates_topk(df_clean, top_k=args.top_k, used_sort_col=used_sort_col or "smiles")

    print(f"🚀 [Step 4B] Starting PySCF calculation: Top {len(candidates)} (sorted by: {used_sort_col})")
    print(f"    Parallel workers: {args.workers}")
    print("    Recommended: set OMP_NUM_THREADS=1 externally to prevent PySCF threading conflicts with multiprocessing")

    # Assign internal row indices to handle duplicate SMILES safely (not written to output)
    candidates = candidates.reset_index(drop=True).copy()
    candidates["_row_idx"] = np.arange(len(candidates), dtype=int)

    tasks = list(zip(candidates["smiles"].astype(str).tolist(), candidates["_row_idx"].astype(int).tolist()))

    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, res in enumerate(executor.map(run_pyscf_task, tasks), 1):
            if res:
                # Merge back original metadata (pIC50, R_global, etc.)
                orig_row = candidates[candidates["_row_idx"] == res["_row_idx"]].iloc[0]
                for key in ["pIC50", "QED", "SA", "hERG_Prob", "hERG_Risk", "R0", "R_total", "R_ADMET", "R_global"]:
                    if key in orig_row:
                        res[key] = orig_row[key]
                results.append(res)

            # Progress reporting
            if i % 5 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r⏳ Progress: {i}/{len(tasks)} | Success: {len(results)} | Elapsed: {elapsed:.1f}s")
                sys.stdout.flush()

    print("\n")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if results:
        df_final = pd.DataFrame(results)

        # Remove internal key from output
        if "_row_idx" in df_final.columns:
            df_final = df_final.drop(columns=["_row_idx"])

        def _mad(x: np.ndarray) -> float:
            med = np.median(x)
            return float(np.median(np.abs(x - med)))

        def _robust_z(x: np.ndarray, med: float, mad: float, eps: float = 1e-9) -> np.ndarray:
            return 0.6745 * (x - med) / (mad + eps)

        # ===== Robust physical gating (Gap + Dipole) =====
        if "PySCF_Gap_eV" in df_final.columns and "PySCF_Dipole_Debye" in df_final.columns:
            gap = df_final["PySCF_Gap_eV"].astype(float).values
            dip = df_final["PySCF_Dipole_Debye"].astype(float).values

            # --- Gap robust statistics ---
            gap_med = float(np.median(gap))
            gap_mad = _mad(gap)
            gap_rz = _robust_z(gap, gap_med, gap_mad)

            df_final["Gap_Median"] = gap_med
            df_final["Gap_MAD"] = gap_mad
            df_final["Gap_RZ"] = gap_rz

            # Hard fail: extreme outliers only
            df_final["Gap_HardFail"] = (np.abs(df_final["Gap_RZ"]) >= 3.5)

            # Soft penalty
            z0, z1 = 1.5, 3.5
            gap_pen = np.clip((np.abs(gap_rz) - z0) / (z1 - z0), 0, 1)
            df_final["Gap_Penalty"] = gap_pen
            df_final["R_gap"] = 1 - gap_pen

            # --- Dipole robust statistics ---
            dip_med = float(np.median(dip))
            dip_mad = _mad(dip)
            dip_p90 = float(np.percentile(dip, 90))
            dip_p75 = float(np.percentile(dip, 75))

            dip_hard_thr = max(dip_p90, dip_med + 2 * dip_mad)
            dip_soft_thr = max(dip_p75, dip_med + 1 * dip_mad)

            df_final["Dip_Median"] = dip_med
            df_final["Dip_MAD"] = dip_mad
            df_final["Dip_P90"] = dip_p90
            df_final["Dip_SoftThr"] = dip_soft_thr
            df_final["Dip_HardThr"] = dip_hard_thr

            df_final["Dipole_HardFail"] = (df_final["PySCF_Dipole_Debye"].astype(float) > dip_hard_thr)

            dip_pen = np.clip(
                (dip - dip_soft_thr) / (dip_hard_thr - dip_soft_thr + 1e-9),
                0, 1
            )
            df_final["Dip_Penalty"] = dip_pen
            df_final["R_dip"] = 1 - dip_pen

            # --- Integrate physical gate ---
            df_final["Physical_HardFail"] = df_final["Gap_HardFail"] | df_final["Dipole_HardFail"]

            # TPSA/LogP not available in Step 4B; set R_conf = 1
            df_final["R_conf"] = 1.0

            df_final["R_phys"] = (df_final["R_gap"] ** 1.0) * (df_final["R_dip"] ** 1.0) * (df_final["R_conf"] ** 0.5)

            if "R_global" in df_final.columns:
                df_final["R_global2"] = df_final["R_global"].astype(float) * df_final["R_phys"].astype(float)

            # Reorder columns: priority columns first
            cols = list(df_final.columns)
            head_cols = ["smiles", "PySCF_Gap_eV", "PySCF_Dipole_Debye", "R_global"]
            sorted_cols = [c for c in head_cols if c in cols] + [c for c in cols if c not in head_cols]
            df_final = df_final[sorted_cols]

            df_final.to_csv(args.output_file, index=False)
            print(f"✅ PySCF results saved to: {args.output_file}")
        else:
            # Write output even if physical gating columns are missing
            df_final.to_csv(args.output_file, index=False)
            print(f"✅ PySCF results saved to: {args.output_file} (some columns missing; physical gating skipped)")
    else:
        print("⚠️ All PySCF calculations failed or did not converge")
        pd.DataFrame(columns=["smiles"]).to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
