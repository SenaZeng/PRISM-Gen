# -*- coding: utf-8 -*-
"""
Step5B (Patched - Ranking Only)
--------------------------------
目标：不再使用 Gold/Silver/Bronze 分级（避免阈值导致“全灭/输出为空”）。

改为：
1) 将 Step4C master 与 Step5A docking 按 canonical SMILES 合并；
2) 对有 docking 分数的行，按 Broad_Spectrum_Score（越负越好）排序并生成 Docking_Rank / Docking_Rank_Pct；
3) 输出 master summary（含排名列）；
4) 输出 final candidates：默认取 docking 排名 TopK（保证至少有 Top1）；若没有任何 docking 分数，则回退按上游综合分数排序取 TopK（也保证不为空）。

默认路径保持不变，兼容你原有 pipeline：
- Step4C master:  results/step4c_master_summary.csv
- Step5A docking: results/step5a_broadspectrum_docking.csv
- out master:      results/step5b_master_summary.csv
- out final:       results/step5b_final_candidates.csv
"""

import os
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
from rdkit import Chem

# ----------------- 路径基础设置 ----------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

DEFAULT_MASTER_4C = os.path.join(project_root, "results", "step4c_master_summary.csv")
DEFAULT_DOCK_5A = os.path.join(project_root, "results", "step5a_broadspectrum_docking.csv")

DEFAULT_OUT_MASTER = os.path.join(project_root, "results", "step5b_master_summary.csv")
DEFAULT_OUT_FINAL = os.path.join(project_root, "results", "step5b_final_candidates.csv")


# ----------------- SMILES 归一化 ----------------- #
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
    百分位：rank / n（最优≈1/n）
    仅对非空 docking 分数参与 n 统计
    """
    n = int(scores.notna().sum())
    if n <= 0:
        return pd.Series([np.nan] * len(scores), index=scores.index)
    rank = scores.rank(method="min", ascending=True)  # 越负越好 -> 升序
    pct = rank / float(n)
    return pct


def choose_fallback_sort_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for c in preferred:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="Step5B: merge + ranking (NO tiers)")
    ap.add_argument("--master_4c", type=str, default=DEFAULT_MASTER_4C, help="Step4C master summary")
    ap.add_argument("--dock_5a", type=str, default=DEFAULT_DOCK_5A, help="Step5A docking results")
    ap.add_argument("--out_master", type=str, default=DEFAULT_OUT_MASTER, help="输出 master summary")
    ap.add_argument("--out_final", type=str, default=DEFAULT_OUT_FINAL, help="输出最终候选（按排名 TopK）")

    ap.add_argument("--final_top_k", type=int, default=20, help="最终候选输出 TopK（默认 20；至少会输出 1 条）")
    ap.add_argument("--fallback_top_k", type=int, default=None, help="当无 docking 分数时的 TopK（默认与 final_top_k 相同）")
    ap.add_argument("--fallback_sort_cols", nargs="+",
                    default=["R_global2", "R_total2", "R_total", "R_global", "Reward", "pIC50"],
                    help="无 docking 时的回退排序列优先级（越大越好）")

    args = ap.parse_args()
    if args.fallback_top_k is None:
        args.fallback_top_k = args.final_top_k

    if not os.path.exists(args.master_4c):
        raise FileNotFoundError(f"找不到 Step4C master: {args.master_4c}")
    if not os.path.exists(args.dock_5a):
        raise FileNotFoundError(f"找不到 Step5A docking: {args.dock_5a}")

    df_master = pd.read_csv(args.master_4c)
    df_dock = pd.read_csv(args.dock_5a)

    if "smiles" not in df_master.columns:
        raise ValueError("Step4C master 缺少 smiles 列")
    if "smiles" not in df_dock.columns:
        raise ValueError("Step5A docking 缺少 smiles 列")

    # 如果 docking 文件没有 Broad_Spectrum_Score，则尝试从 E_* 计算（兼容旧输出）
    if "Broad_Spectrum_Score" not in df_dock.columns:
        e_cols = [c for c in df_dock.columns if c.startswith("E_")]
        if not e_cols:
            raise ValueError("Step5A docking 缺少 Broad_Spectrum_Score 且无 E_* 列可推算")
        # 取各靶点最差值作为广谱短板（越负越好 -> 最大值最差）
        df_dock["Broad_Spectrum_Score"] = df_dock[e_cols].max(axis=1)

    # canonical key
    df_master["merge_key"] = df_master["smiles"].apply(canonicalize_smiles)
    df_dock["merge_key"] = df_dock["smiles"].apply(canonicalize_smiles)

    # 只保留 docking 的新增列，避免覆盖 master（保留 Broad_Spectrum_Score 及 E_* 等）
    dock_cols = [c for c in df_dock.columns if c not in df_master.columns and c != "smiles"]
    if "merge_key" not in dock_cols:
        dock_cols.append("merge_key")

    df_merge = pd.merge(df_master, df_dock[dock_cols], on="merge_key", how="left")

    # 排名与百分位（只对有 docking 分数的行有意义）
    df_merge["Docking_Rank"] = df_merge["Broad_Spectrum_Score"].rank(method="min", ascending=True)
    df_merge["Docking_Rank_Pct"] = compute_rank_pct(df_merge["Broad_Spectrum_Score"])

    # 清理辅助列
    df_merge.drop(columns=["merge_key"], inplace=True)

    # 输出 master
    os.makedirs(os.path.dirname(args.out_master), exist_ok=True)
    df_merge.to_csv(args.out_master, index=False)

    # 输出 final（保证不空）
    df_valid = df_merge[df_merge["Broad_Spectrum_Score"].notna()].copy()
    # === Active_Set (ADMET Pass) 统一起点：final candidates 仅从 Active_Set==True 中产生（若存在该列） ===
    
    # 优先严格门槛/终审通过（如果列存在）
    if all(c in df_valid.columns for c in ["Data_Source_Status","Physical_HardFail","R_phys"]):
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
        # docking 排名：越负越好 -> 升序；若并列，用 Docking_Rank（min）保持一致
        df_valid = df_valid.sort_values(["Broad_Spectrum_Score", "Docking_Rank"], ascending=[True, True])
        k = max(1, int(args.final_top_k))
        df_final = df_valid.head(k).copy()
        selection_mode = f"docking_rank_top{k}"
    else:
        # 回退：没有任何 docking 分数（例如 Step5A 全失败或没跑）
        fallback_col = choose_fallback_sort_col(df_merge, args.fallback_sort_cols)
        k = max(1, int(args.fallback_top_k))
        if fallback_col is None:
            # 退无可退：直接输出前 k 行
            df_final = df_merge.head(k).copy()
            selection_mode = f"fallback_first{k}"
        else:
            df_final = df_merge.sort_values(fallback_col, ascending=False).head(k).copy()
            selection_mode = f"fallback_{fallback_col}_top{k}"

    df_final["Selection_Mode"] = selection_mode
    os.makedirs(os.path.dirname(args.out_final), exist_ok=True)
    df_final.to_csv(args.out_final, index=False)

    # 打印摘要
    print("==============================================")
    print(f"✅ Step5B master summary: {args.out_master}")
    print(f"   总分子数: {len(df_merge)}")
    print(f"   有 docking 记录: {int(df_merge['Broad_Spectrum_Score'].notna().sum())}")
    if df_merge['Broad_Spectrum_Score'].notna().any():
        print("   docking Broad_Spectrum_Score:")
        s = df_merge['Broad_Spectrum_Score'].dropna()
        print(f"   min={s.min():.3f} mean={s.mean():.3f} max={s.max():.3f}")
    print("----------------------------------------------")
    print(f"✅ Step5B final candidates: {args.out_final}")
    print(f"   Selection_Mode={selection_mode} | 条目数: {len(df_final)}")
    if 'Broad_Spectrum_Score' in df_final.columns and df_final['Broad_Spectrum_Score'].notna().any():
        best = df_final.iloc[0]
        print(f"   Top1 Broad_Spectrum_Score={best['Broad_Spectrum_Score']:.3f} | smiles={best.get('smiles','')[:60]}...")
    print("==============================================")


if __name__ == "__main__":
    main()