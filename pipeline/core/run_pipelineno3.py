#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一从 Step1 -> Step5 的全流程流水线脚本（位于 core/ 目录中）。

本版本修改目标：
1) Step3A 不再执行（直接复用 results/step3a_*.csv 的现成输出）
2) Step3B 之后全部继续执行
3) 所有关键 CSV 路径统一改为 “项目根目录/results/xxx.csv”（绝对路径）
4) 兼容某些 step 脚本内部使用 ../results 的相对路径：调用这些 step 时临时 chdir 到 core/
"""

import os
import sys
import shutil
import pandas as pd
import time
import glob
import datetime
import builtins
from contextlib import contextmanager


# ----------------- 0. 路径与并行配置 -----------------

# 当前脚本所在目录：core/
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：core 的上一层
project_root = os.path.abspath(os.path.join(current_dir, ".."))


# 确保项目根目录在 sys.path 中，这样才能 import core
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 结果目录统一放在项目根目录下的 results/
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 目标并行核数（逻辑核）
TARGET_WORKERS = 40

# 为了避免某些并行库把单个任务开太多线程，这里适度限制 OMP/MKL 线程数
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---- 关键 CSV：固定为项目根目录 results 下 ----
STEP3A_RAW_CSV = os.path.join(RESULTS_DIR, "step3a_optimized_molecules_raw.csv")
STEP3A_TOP2000_CSV = os.path.join(RESULTS_DIR, "step3a_top2000.csv")
STEP3A_STD_CSV = os.path.join(RESULTS_DIR, "step3a_optimized_molecules.csv")  # Step3B 默认要吃这个


def get_effective_workers():
    """根据机器实际核数，返回合适的 workers 数。"""
    n_cores = os.cpu_count() or 1
    workers = min(TARGET_WORKERS, n_cores)
    print(
        f"[Parallel Config] Detected {n_cores} CPU cores, "
        f"pipeline will use up to {workers} workers."
    )
    return workers


@contextmanager
def pushd(path: str):
    """临时切换工作目录，用完自动切回。"""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ----------------- 1. 日志系统：终端 + 文件双写 -----------------

def setup_logging():
    """
    简易日志系统：
    - 所有 print 输出仍然会显示在 terminal
    - 同时写入 results/pipeline_YYYYmmdd_HHMMSS.log
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"pipeline_{ts}.log")

    # 打开日志文件（行缓冲）
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    orig_print = builtins.print

    def tee_print(*args, **kwargs):
        orig_print(*args, **kwargs)
        if "file" in kwargs and kwargs["file"] is not None:
            return
        text = " ".join(str(a) for a in args)
        log_fp.write(text + "\n")

    builtins.print = tee_print
    orig_print(f"📝 Pipeline log file: {log_path}")
    return log_path


# ----------------- 2. 工具函数：文件胶水 -----------------

def ensure_file_match(source_pattern, target_name):
    """
    文件名适配器：
    如果上一步生成的文件名带有后缀（如 step3a_top2000.csv），
    而下一步代码写死读取标准名（如 step3a_optimized_molecules.csv），
    则自动复制一份标准名文件，确保流程不断裂。
    """
    target_path = os.path.join(RESULTS_DIR, target_name)

    # 1) 如果目标文件已存在且不为空，直接用
    if os.path.exists(target_path) and os.path.getsize(target_path) > 10:
        return

    # 2) 否则去 results 找匹配 pattern 的最新文件
    search_path = os.path.join(RESULTS_DIR, source_pattern)
    candidates = glob.glob(search_path)

    if candidates:
        latest = max(candidates, key=os.path.getmtime)
        print(f"🔄 [Pipeline Fix] Copy {os.path.basename(latest)} -> {target_name}")
        shutil.copy(latest, target_path)
    else:
        print(
            f"⚠️ [Pipeline Warning] Not found pattern={source_pattern}. "
            f"Next step may fail due to missing input."
        )


def require_file(path: str, hint: str = ""):
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        raise FileNotFoundError(f"Missing required file: {path}\n{hint}")


# ----------------- 3. 导入各 Step 模块 -----------------

try:
    from core import (
        step1_vae,
        step2_surrogate,
        step2b_train_herg_model,
        step3a_optimizer,          # 仍保留 import，但 main 不再调用
        step3b_run_dft,
        step3c_dft_refine,
        step4a_admet,
        step4b_final_pyscf,
        step4c_utils_merge_results,
        step5a_docking,
        step5b_utils_merge,
    )
    print("✅ Imported all core modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run it from project root, e.g.:")
    print("  cd /path/to/Mpro_AI_Design")
    print("  python core/run_pipeline.py")
    sys.exit(1)


# ----------------- 4. 主流程 -----------------

def main():
    log_path = setup_logging()

    print("🚀 Start pipeline (core/run_pipeline.py)")
    print(f"   Project root: {project_root}")
    print(f"   Results dir : {RESULTS_DIR}")
    print(f"   Log file    : {log_path}")

    start_time = time.time()
    workers = get_effective_workers()

    # --- Step 1: VAE 训练 / 载入 ---
    print("\n" + "=" * 50)
    print(">>> Step 1: Fragment-based VAE (FRATTVAE) train/init")
    try:
        if hasattr(step1_vae, "main"):
            step1_vae.main()
        else:
            print("ℹ️ step1_vae has no main(), imported only.")
    except Exception as e:
        print(f"⚠️ Step 1 exception (often ignorable if already trained): {e}")

    # --- Step 2: Surrogate 训练 / 载入 ---
    print("\n" + "=" * 50)
    print(">>> Step 2: Uni-Mol surrogate train/init")
    try:
        if hasattr(step2_surrogate, "main"):
            step2_surrogate.main()
        else:
            print("ℹ️ step2_surrogate has no main(), imported only.")
    except Exception as e:
        print(f"⚠️ Step 2 exception (often ignorable if already trained): {e}")

    # --- Step 2B: hERG 模型训练 ---
    print("\n" + "=" * 50)
    print(">>> Step 2B: hERG model train/update")
    try:
        if hasattr(step2b_train_herg_model, "main"):
            step2b_train_herg_model.main()
        else:
            print("ℹ️ step2b_train_herg_model has no main(), imported only.")
    except Exception as e:
        print(f"⚠️ Step 2B exception (often ignorable if already trained): {e}")

    # =========================================================
    # ✅ Step 3A: 跳过（直接复用已有 CSV）
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 3A: SKIPPED (reuse existing optimization CSVs)")
    try:
        # 你这三个文件已经在 results/ 下生成了，就直接检查
        require_file(STEP3A_STD_CSV, hint="Expected from Step3A: step3a_optimized_molecules.csv")
        require_file(STEP3A_TOP2000_CSV, hint="Expected from Step3A: step3a_top2000.csv")
        print(f"✅ Found: {STEP3A_STD_CSV}")
        print(f"✅ Found: {STEP3A_TOP2000_CSV}")
    except Exception as e:
        print(f"❌ Step 3A skipped but required CSV missing: {e}")
        print("   Please make sure Step3A outputs exist in results/ before running Step3B+.")
        return

    # 额外保险：如果存在 step3a_top*.csv，自动拷贝/补齐标准文件名
    # 原版逻辑是把 step3a_top*.csv 复制为 step3a_optimized_molecules.csv（见 :contentReference[oaicite:1]{index=1}）
    ensure_file_match("step3a_optimized_molecules*.csv", "step3a_optimized_molecules.csv")
    ensure_file_match("step3a_top2000*.csv", "step3a_top2000.csv")

    # =========================================================
    # Step 3B: xTB 电子结构预筛
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 3B: xTB pre-screen (MOCK/Real)")

    # 关键点：step3b_run_dft.py 里很可能用的是 ../results/step3a_optimized_molecules.csv
    # 为了让 ../results 指向 “项目根/results”，我们临时切到 core/ 目录执行
    try:
        with pushd(current_dir):
            step3b_run_dft.main()
    except Exception as e:
        print(f"❌ Step 3B failed: {e}")

    # =========================================================
    # Step 3C: 结合 DFT 结果进行重排
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 3C: DFT refine & R_total ranking")
    try:
        with pushd(current_dir):
            step3c_dft_refine.main(top_k=2000)
    except Exception as e:
        print(f"❌ Step 3C failed: {e}")

    # =========================================================
    # Step 4A: ADMET 筛选  （修复：这里必须跑 step4a_admet）
    # 原文件这里误跑了 step4b_final_pyscf（见 :contentReference[oaicite:2]{index=2}）
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
        print(f"❌ Step 4A failed: {e}")

    # 确保 Step 4B 能找到 Step 4A 的输出（原文件也有这段胶水逻辑）
    ensure_file_match("step4a_admet*.csv", "step4a_admet_final.csv")

    # === 新增：Active_Set 统计（ADMET Pass） ===
    try:
        step4a_out = os.path.join(RESULTS_DIR, "step4a_admet_final.csv")
        if os.path.exists(step4a_out):
            _df4a = pd.read_csv(step4a_out)
            if "Active_Set" in _df4a.columns:
                print(f"✅ Active_Set(ADMET Pass) in Step4A = {int(_df4a['Active_Set'].sum())} / {len(_df4a)}")
    except Exception as _e:
        print(f"⚠️ Active_Set 统计失败（不影响流程继续）：{_e}")

    # =========================================================
    # Step 4B: PySCF 高精度 DFT 精修
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 4B: PySCF DFT refine (B3LYP/6-31G*)")
    print("    Note: typically only refine top few molecules.")
    try:
        argv_backup = sys.argv[:]
        # 你也可以把 top_k 从 20 改小一些（比如 5），否则 PySCF 很耗时
        sys.argv = ["step4b_final_pyscf.py", "--top_k", "20", "--workers", str(workers)]
        with pushd(current_dir):
            step4b_final_pyscf.main()
        sys.argv = argv_backup
    except Exception as e:
        print(f"❌ Step 4B failed: {e}")
        sys.argv = argv_backup

    # =========================================================
    # Step 4C: 汇总主表
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 4C: merge master summary (pre-docking)")
    try:
        with pushd(current_dir):
            step4c_utils_merge_results.merge_all_steps()
    except Exception as e:
        print(f"❌ Step 4C failed: {e}")

    # =========================================================
    # Step 5A: 多靶点广谱对接
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 5A: broad-spectrum docking")
    try:
        argv_backup = sys.argv[:]
        sys.argv = [
            "step5a_docking.py",
            "--auto_predE3_stage2",
            "--stage2_profile", "balanced",
            "--predE3_n_jobs", str(min(40, int(workers))),
            "--top_n", "100",
            "--workers", str(workers),
            "--vina_cpu", "1",
        ]
        with pushd(current_dir):
            step5a_docking.main()
        sys.argv = argv_backup
    except Exception as e:
        print(f"❌ Step 5A failed: {e}")
        sys.argv = argv_backup

    # =========================================================
    # Step 5B: 终极汇总
    # =========================================================
    print("\n" + "=" * 50)
    print(">>> Step 5B: final merge/report")
    try:
        with pushd(current_dir):
            step5b_utils_merge.main()
    except Exception as e:
        print(f"❌ Step 5B failed: {e}")

    # =========================================================
    # Final Extraction: 提取真正的优胜候选
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
                print(f"🏆 Exported: {final_out_path}")
                print(f"   Molecules: {len(df_final)}")
                cols = ["smiles", "Broad_Spectrum_Score", "R_global", "Broad_Tier", "pIC50"]
                print(df_final[[c for c in cols if c in df_final.columns]].head())
            else:
                print("⚠️ Broad_Spectrum_Score exists but all empty.")
        else:
            print("⚠️ No Broad_Spectrum_Score column, cannot export final candidates.")
    else:
        print(f"⚠️ Missing summary: {summary_path} (check Step 5B).")

    print("\n" + "=" * 50)
    print(f"✅ Pipeline finished. Total time: {time.time() - start_time:.1f} sec")
    print(f"📝 Log file: {log_path}")


if __name__ == "__main__":
    main()