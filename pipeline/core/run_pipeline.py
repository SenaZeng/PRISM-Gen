#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified end-to-end pipeline script: Step 1 -> Step 5 (located in the core/ directory).

Design goals:
1) Step 3A executes normally (generates/optimizes molecules, produces step3a_*.csv)
2) All downstream steps (Step 3B onwards) continue after Step 3A completes
3) All key CSV paths are unified under <project_root>/results/ (absolute paths)
4) For step scripts that internally use ../results relative paths, temporarily
   chdir to core/ before invoking them to ensure path resolution is correct
"""

import os
import sys
# --- Optional: handle GLIBCXX path conflict on HPC systems ---
# If your conda environment provides a newer libstdc++ than the system default,
# set the environment variable CONDA_LIB_PATH to its full path before running,
# e.g.:
#   export CONDA_LIB_PATH=/path/to/conda/envs/your_env/lib/libstdc++.so.6
# This is only needed when you encounter GLIBCXX version errors at runtime.
conda_lib_path = os.environ.get("CONDA_LIB_PATH", "")
if conda_lib_path and os.path.exists(conda_lib_path):
    os.environ["LD_PRELOAD"] = conda_lib_path
# -------------------------------------------------------------


import shutil
import pandas as pd
import time
import glob
import datetime
import builtins
from contextlib import contextmanager
# --- Optional: prepend conda lib directory to LD_LIBRARY_PATH ---
# Set CONDA_LIB_DIR to your conda environment's lib/ directory if needed,
# e.g.:
#   export CONDA_LIB_DIR=/path/to/conda/envs/your_env/lib
conda_lib_dir = os.environ.get("CONDA_LIB_DIR", "")
if conda_lib_dir:
    os.environ["LD_LIBRARY_PATH"] = conda_lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
# ----------------------------------------------------------------


# ----------------- 0. Path and parallelism configuration -----------------

# Directory containing this script: core/
current_dir = os.path.dirname(os.path.abspath(__file__))
# Project root: one level above core/
project_root = os.path.abspath(os.path.join(current_dir, ".."))


# Ensure project root is on sys.path so that 'import core' works
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# All results are written to <project_root>/results/
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Target number of parallel workers (logical cores)
TARGET_WORKERS = 40

# -----------------------------------------------------------------
# Step 3A search parameters — edit these three lines to change run scale.
#
# Preset examples:
#   Smoke test  :  RESTARTS=10,  STEPS=50,  TOP_K_3A=100
#   Full run    :  RESTARTS=100, STEPS=100, TOP_K_3A=200
#
# Rule of thumb: TOP_K_3A should be >= RESTARTS to avoid losing candidates.
# -----------------------------------------------------------------
RESTARTS  = 10    # number of random restarts
STEPS     = 50    # hill-climbing steps per restart
TOP_K_3A  = 100   # top-K molecules passed to downstream steps

# Limit per-task thread counts to avoid thread over-subscription in parallel libs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---- Key CSV paths: fixed under <project_root>/results/ ----
STEP3A_RAW_CSV = os.path.join(RESULTS_DIR, "step3a_optimized_molecules_raw.csv")
STEP3A_TOP2000_CSV = os.path.join(RESULTS_DIR, "step3a_top200.csv")
STEP3A_STD_CSV = os.path.join(RESULTS_DIR, "step3a_optimized_molecules.csv")  # Expected input for Step 3B


def get_effective_workers():
    """Return the number of workers to use, capped by available CPU cores."""
    n_cores = os.cpu_count() or 1
    workers = min(TARGET_WORKERS, n_cores)
    print(
        f"[Parallel Config] Detected {n_cores} CPU cores, "
        f"pipeline will use up to {workers} workers."
    )
    return workers


@contextmanager
def pushd(path: str):
    """Temporarily change the working directory; restore it on exit."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ----------------- 1. Logging: simultaneous terminal and file output -----------------

def setup_logging():
    """
    Lightweight logging system:
    - All print() output continues to appear in the terminal
    - Output is also appended to results/pipeline_YYYYmmdd_HHMMSS.log
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"pipeline_{ts}.log")

    # Open log file with line buffering
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    orig_print = builtins.print

    def tee_print(*args, **kwargs):
        orig_print(*args, **kwargs)
        if "file" in kwargs and kwargs["file"] is not None:
            return
        text = " ".join(str(a) for a in args)
        log_fp.write(text + "\n")

    builtins.print = tee_print
    orig_print(f" Pipeline log file: {log_path}")
    return log_path


# ----------------- 2. Utility functions: file glue -----------------

def ensure_file_match(source_pattern, target_name):
    """
    Filename adapter:
    If a step produces a file with a non-standard name (e.g. step3a_top2000.csv)
    but the next step hard-codes a standard name (e.g. step3a_optimized_molecules.csv),
    this function copies the latest matching file to the expected standard name,
    preventing pipeline breaks due to filename mismatches.
    """
    target_path = os.path.join(RESULTS_DIR, target_name)

    # 1) If the target already exists and is non-empty, nothing to do
    if os.path.exists(target_path) and os.path.getsize(target_path) > 10:
        return

    # 2) Otherwise, search results/ for the latest file matching the pattern
    search_path = os.path.join(RESULTS_DIR, source_pattern)
    candidates = glob.glob(search_path)

    if candidates:
        latest = max(candidates, key=os.path.getmtime)
        print(f" [Pipeline Fix] Copy {os.path.basename(latest)} -> {target_name}")
        shutil.copy(latest, target_path)
    else:
        print(
            f" [Pipeline Warning] Not found pattern={source_pattern}. "
            f"Next step may fail due to missing input."
        )


def require_file(path: str, hint: str = ""):
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        raise FileNotFoundError(f"Missing required file: {path}\n{hint}")


# ----------------- 3. Import step modules -----------------

try:
    from core import (
        step1_vae,
        step2_surrogate,
        step2b_train_herg_model,
        step3a_optimizer,          # Step 3A: RL-based generation and optimization
        step3b_run_dft,
        step3c_dft_refine,
        step4a_admet,
        step4b_final_pyscf,
        step4c_utils_merge_results,
        step5a_docking,
        step5b_utils_merge,
    )
    print(" Imported all core modules")
except ImportError as e:
    print(f" Import failed: {e}")
    print("Run it from project root, e.g.:")
    print("  cd /path/to/Mpro_AI_Design")
    print("  python core/run_pipeline.py")
    sys.exit(1)


# ----------------- 4. Main pipeline -----------------

def main():
    log_path = setup_logging()

    print(" Start pipeline (core/run_pipeline.py)")
    print(f"   Project root: {project_root}")
    print(f"   Results dir : {RESULTS_DIR}")
    print(f"   Log file    : {log_path}")

    start_time = time.time()
    workers = get_effective_workers()

    # --- Step 1: VAE training / loading ---
    print("\n" + "=" * 50)
    print(">>> Step 1: Fragment-based VAE (FRATTVAE) train/init")
    try:
        if hasattr(step1_vae, "main"):
            step1_vae.main()
        else:
            print(" step1_vae has no main(), imported only.")
    except Exception as e:
        print(f" Step 1 exception (often ignorable if already trained): {e}")

    # --- Step 2: Surrogate model training / loading ---
    print("\n" + "=" * 50)
    print(">>> Step 2: Uni-Mol surrogate train/init")
    try:
        if hasattr(step2_surrogate, "main"):
            step2_surrogate.main()
        else:
            print(" step2_surrogate has no main(), imported only.")
    except Exception as e:
        print(f" Step 2 exception (often ignorable if already trained): {e}")

    # --- Step 2B: hERG model training ---
    print("\n" + "=" * 50)
    print(">>> Step 2B: hERG model train/update")
    try:
        if hasattr(step2b_train_herg_model, "main"):
            step2b_train_herg_model.main()
        else:
            print(" step2b_train_herg_model has no main(), imported only.")
    except Exception as e:
        print(f" Step 2B exception (often ignorable if already trained): {e}")

    # =========================================================
    # Step 3A: RL-based generation and optimization
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 3A: RL optimization / generation (run)")
    step3a_ok = False
    try:
        with pushd(current_dir):
            if hasattr(step3a_optimizer, "main"):
                argv_backup = sys.argv[:]
                sys.argv = [
                    "step3a_optimizer.py",
                    "--n_restarts", str(RESTARTS),   # 10 (quick test) or 100 (production)
                    "--steps",      str(STEPS),       # 10 (quick test) or 100 (production)
                    "--top_k",      str(TOP_K_3A),    # 50  (quick test) or 200 (production)
                    "--n_jobs",     str(workers),
                ]
                step3a_optimizer.main()
                sys.argv = argv_backup
                step3a_ok = True
            else:
                raise RuntimeError("step3a_optimizer has no main()")
    except Exception as e:
        print(f" Step 3A failed: {e}")

    # Regardless of whether Step 3A raised an exception, verify output files exist.
    # If files are present (e.g. exception was non-fatal), continue downstream.
    # If files are absent, halt immediately since all downstream steps depend on them.
    try:
        require_file(STEP3A_STD_CSV, hint="Expected from Step3A: step3a_optimized_molecules.csv")
        require_file(STEP3A_TOP2000_CSV, hint="Expected from Step3A: step3a_top200.csv")
        print(f" Step3A output ready: {STEP3A_STD_CSV}")
        print(f" Step3A output ready: {STEP3A_TOP2000_CSV}")
    except Exception as e:
        print(f" Step 3A outputs missing: {e}")
        return
    # Ensure standard filenames are present for downstream steps
    ensure_file_match("step3a_optimized_molecules*.csv", "step3a_optimized_molecules.csv")
    ensure_file_match("step3a_top200*.csv", "step3a_top200.csv")

    # =========================================================
    # Step 3B: xTB electronic structure pre-screening
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 3B: xTB pre-screen (MOCK/Real)")

    # step3b_run_dft.py internally references ../results/step3a_optimized_molecules.csv
    # Temporarily chdir to core/ so that ../results resolves to <project_root>/results/
    try:
        with pushd(current_dir):
            step3b_run_dft.main()
    except Exception as e:
        print(f" Step 3B failed: {e}")

    # =========================================================
    # Step 3C: DFT-guided re-ranking
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 3C: DFT refine & R_total ranking")
    try:
        with pushd(current_dir):
            step3c_dft_refine.main(top_k=2000)
    except Exception as e:
        print(f" Step 3C failed: {e}")

    # =========================================================
    # Step 4A: ADMET and hERG screening
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 4A: ADMET & hERG screening")
    try:
        with pushd(current_dir):
            if hasattr(step4a_admet, "main"):
                step4a_admet.main()
            else:
                raise RuntimeError("step4a_admet has no main()")
    except Exception as e:
        print(f" Step 4A failed: {e}")

    # Ensure Step 4B can locate Step 4A output under the standard filename
    ensure_file_match("step4a_admet*.csv", "step4a_admet_final.csv")

    # === Report Active_Set count (ADMET Pass) ===
    try:
        step4a_out = os.path.join(RESULTS_DIR, "step4a_admet_final.csv")
        if os.path.exists(step4a_out):
            _df4a = pd.read_csv(step4a_out)
            if "Active_Set" in _df4a.columns:
                print(f" Active_Set(ADMET Pass) in Step4A = {int(_df4a['Active_Set'].sum())} / {len(_df4a)}")
    except Exception as _e:
        print(f"⚠️ Active_Set count failed (pipeline continues): {_e}")

    # =========================================================
    # Step 4B: High-accuracy DFT refinement with PySCF
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 4B: PySCF DFT refine (B3LYP/6-31G*)")
    print("    Note: typically only refine top few molecules.")
    try:
        argv_backup = sys.argv[:]
        # Reduce top_k if PySCF runtime is a concern
        sys.argv = ["step4b_final_pyscf.py", "--top_k", "50", "--workers", str(workers)]
        with pushd(current_dir):
            step4b_final_pyscf.main()
        sys.argv = argv_backup
    except Exception as e:
        print(f" Step 4B failed: {e}")
        sys.argv = argv_backup

    # =========================================================
    # Step 4C: Merge master summary table (pre-docking)
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 4C: merge master summary (pre-docking)")
    try:
        with pushd(current_dir):
            step4c_utils_merge_results.merge_all_steps()
    except Exception as e:
        print(f" Step 4C failed: {e}")

    # =========================================================
    # Step 5A: Multi-target broad-spectrum docking
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 5A: broad-spectrum docking")
    try:
        argv_backup = sys.argv[:]
        sys.argv = [
            "step5a_docking.py",
            "--use_strict_gate",
            "--rphys_min", "0.85",
            "--auto_predE3_stage2",
            "--stage2_profile", "balanced",
            "--predE3_n_jobs", str(min(40, int(workers))),
            "--top_n", "300",
            "--workers", str(workers),
            "--vina_cpu", "1",
            "--vina_seed",  "42",
            "--rdkit_seed", "42",
        ]
        with pushd(current_dir):
            step5a_docking.main()
        sys.argv = argv_backup
    except Exception as e:
        print(f" Step 5A failed: {e}")
        sys.argv = argv_backup

    # =========================================================
    # Step 5B: Final merge and report
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 5B: final merge/report")
    try:
        argv_backup = sys.argv[:]
        sys.argv = ["step5b_utils_merge.py", "--final_top_k", "8"]
        with pushd(current_dir):
            step5b_utils_merge.main()
        sys.argv = argv_backup        
    except Exception as e:
        print(f" Step 5B failed: {e}")

    # =========================================================
    # Final extraction: export top-ranked candidates
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Pipeline Final: export final candidates list")

    summary_path = os.path.join(RESULTS_DIR, "step5b_master_summary.csv")
    final_out_path = os.path.join(RESULTS_DIR, "final_candidates_paper.csv")

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        if "Broad_Spectrum_Score" in df.columns:
            df_final = df[df["Broad_Spectrum_Score"].notna()].copy()

            if "Is_Broad_Top" in df_final.columns:
                df_gold = df_final[df_final["Is_Broad_Top"] == True]
                if len(df_gold) > 0:
                    print(f"   Found {len(df_gold)} Broad_Top molecules, prefer them.")
                    df_final = df_gold

            df_final = df_final.sort_values("Broad_Spectrum_Score", ascending=True)

            if not df_final.empty:
                df_final.to_csv(final_out_path, index=False)
                print(f" Exported: {final_out_path}")
                print(f"   Molecules: {len(df_final)}")
                cols = ["smiles", "Broad_Spectrum_Score", "R_global", "Broad_Tier", "pIC50"]
                print(df_final[[c for c in cols if c in df_final.columns]].head())
            else:
                print(" Broad_Spectrum_Score exists but all empty.")
        else:
            print(" No Broad_Spectrum_Score column, cannot export final candidates.")
    else:
        print(f" Missing summary: {summary_path} (check Step 5B).")

    print("\n" + "=" * 50)
    print(f" Pipeline finished. Total time: {time.time() - start_time:.1f} sec")
    print(f" Log file: {log_path}")


if __name__ == "__main__":
    main()
