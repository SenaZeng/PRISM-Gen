# 文件: core/step3c_dft_refine.py
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


def score_gap(gap_ev, mu_gap=5.0, sigma_gap=2.0):
    """
    基于 HOMO-LUMO 能隙的高斯型得分，gap 越接近 mu_gap 得分越高，范围 (0, 1]。

    - gap_ev:  xTB 计算得到的能隙（eV）
    - mu_gap:  目标能隙中心（默认 5 eV）
    - sigma_gap: 容忍宽度（默认 2 eV）

    返回值: float, 位于 [0, 1] 的得分（越高越好）
    """
    if pd.isna(gap_ev):
        return 0.0
    x = (gap_ev - mu_gap) / sigma_gap
    return float(np.exp(-0.5 * x * x))


def score_esp(esp_min, esp_min_ref=-0.4, delta_neg=0.6, delta_pos=0.4):
    """
    基于 "最负原子电荷 proxy 的 ESP 分数"。

    设计假设：
    - esp_min 来自 xTB 的 "最负原子电荷"（例如 Mulliken / xTB charges）；
      通常位于 [-1, 0] 区间，越负表示局部电子越富集。
    - 我们希望 esp_min 适度为负：
      * 太不负（接近 0）：说明电子分布过于均匀，缺少明显 H-bond 受体/局部电子富集位点；
      * 适度为负（如约 -0.4）：认为较为合理；
      * 过度负（<-1）：视作极端情况，得分不再继续增加。

    参数：
    - esp_min: xTB 提供的最负原子电荷（负值，越负电子越富集）
    - esp_min_ref: 理想参考值（默认 -0.4）
    - delta_neg: 在 esp_min_ref 左侧允许的负向偏移宽度（默认 0.6，对应 ~[-1.0, -0.4]）
    - delta_pos: 在 esp_min_ref 右侧允许的正向偏移宽度（默认 0.4，对应 ~[-0.4, 0.0]）

    实现：
    - 对 [esp_min_ref - delta_neg, esp_min_ref + delta_pos] 做一个简单的三角形/梯形评分：
      * 在 esp_min_ref 处取得近似 1 分；
      * 在两侧边界线性衰减到 0；
      * 超出边界则 clip 为 0。
    """
    if pd.isna(esp_min):
        return 0.0

    # 确保是 float
    esp = float(esp_min)

    left = esp_min_ref - delta_neg   # 比这个更负，按最差处理
    right = esp_min_ref + delta_pos  # 比这个更不负，按最差处理

    if esp <= left or esp >= right:
        # 远离合理区间，得分 0
        return 0.0

    if esp <= esp_min_ref:
        # 在 [left, esp_min_ref] 区间线性从 0 -> 1
        score = (esp - left) / (esp_min_ref - left)
    else:
        # 在 [esp_min_ref, right] 区间线性从 1 -> 0
        score = (right - esp) / (right - esp_min_ref)

    # 保险起见 clip 到 [0,1]
    score = max(0.0, min(1.0, score))
    return float(score)


def main(
    top_k=200,
    gamma=0.4,
    w_gap=0.7,
    w_esp=0.3,
):
    """
    Step 3c: 基于 xTB 电子结构（gap + ESP proxy）对 RL 结果进行物理重排。

    核心逻辑：
    - 从 Step 3a 的优化结果中取前 top_k 个候选（按 Reward 排序）；
    - 读取 Step 3b 的 xTB 结果（EHOMO_ev, ELUMO_ev, gap_ev, esp_min）；
    - 对 gap_ev 计算 gap_score，对 esp_min 计算 esp_score；
    - 合成 R_DFT = w_gap * gap_score + w_esp * esp_score；
    - 定义 R0 = 原始 Reward，R_total = R0 + gamma * R_DFT；
    - 对比 R0 排名和 R_total 排名，分析电子结构对排序的影响。

    注意：
    - 当前实现中，esp_min 默认假定为 “最负原子电荷”的 proxy；
      请确保在 Step 3b 中已将其写入 CSV。
    """
    # 1. 读取 Step3 优化结果 (RL baseline)
    rl_path = os.path.abspath(os.path.join(current_dir, "../results/step3a_optimized_molecules.csv"))
    if not os.path.exists(rl_path):
        raise FileNotFoundError(f"找不到 Step3 输出文件: {rl_path}")
    df_rl = pd.read_csv(rl_path)

    # 确保有 Reward 列
    if "Reward" not in df_rl.columns:
        raise ValueError("step3a_optimized_molecules.csv 中找不到 'Reward' 列，请检查 Step3 输出。")

    # 取 Top K
    df_rl = df_rl.sort_values("Reward", ascending=False).reset_index(drop=True)
    df_top = df_rl.iloc[:top_k].copy()
    print(f"Step3C: 从 Step3 中取前 {top_k} 个候选用于 xTB 物理重排。")

    # 2. 读取 xTB 结果 (Step 3b)
    dft_path = os.path.abspath(os.path.join(current_dir, "../results/step3b_dft_results.csv"))
    if not os.path.exists(dft_path):
        raise FileNotFoundError(
            f"找不到 DFT/xTB 结果文件: {dft_path}\n"
            f"请先运行 step3b_run_dft.py 生成 step3b_dft_results.csv。"
        )
    df_dft = pd.read_csv(dft_path)

    required_cols = ["smiles", "EHOMO_ev", "ELUMO_ev", "gap_ev", "esp_min"]
    for c in required_cols:
        if c not in df_dft.columns:
            raise ValueError(f"step3b_dft_results.csv 中缺少必需列: '{c}'")

    # 如果存在 status 列，只保留成功的记录
    if "status" in df_dft.columns:
        df_dft = df_dft[df_dft["status"].isin(["xtb_success", "mock_success"])].copy()

    # 3. 合并 RL 结果与 xTB 结果
    df_merge = pd.merge(df_top, df_dft, on="smiles", how="left", suffixes=("", "_dft"))
    print(f"合并后共有 {len(df_merge)} 条记录。")

    # 4. 计算 xTB-based 物理评分 (gap_score + esp_score)
    gap_scores = []
    esp_scores = []
    r_dft_list = []

    for _, row in df_merge.iterrows():
        gap = row.get("gap_ev", np.nan)
        esp_min = row.get("esp_min", np.nan)

        g_score = score_gap(gap)
        e_score = score_esp(esp_min)

        r_dft = w_gap * g_score + w_esp * e_score

        gap_scores.append(g_score)
        esp_scores.append(e_score)
        r_dft_list.append(r_dft)

    df_merge["gap_score"] = gap_scores
    df_merge["esp_score"] = esp_scores
    df_merge["R_DFT"] = r_dft_list

    # 5. 计算总评分 R_total = R0 + gamma * R_DFT
    df_merge["R0"] = df_merge["Reward"]
    df_merge["R_total"] = df_merge["R0"] + gamma * df_merge["R_DFT"]

    # 6. 分别根据 R0 和 R_total 排名，比较变化
    df_merge = df_merge.sort_values("R0", ascending=False).reset_index(drop=True)
    df_merge["rank_R0"] = df_merge.index + 1  # 原始 rank

    df_merge = df_merge.sort_values("R_total", ascending=False).reset_index(drop=True)
    df_merge["rank_total"] = df_merge.index + 1  # 加物理后 rank

    # delta_rank > 0 表示排名下降，< 0 表示排名上升
    df_merge["delta_rank"] = df_merge["rank_total"] - df_merge["rank_R0"]

    # 标记: 物理“剔除”/“提升”
    def classify_row(row):
        gap = row["gap_ev"]
        r_dft = row["R_DFT"]
        delta = row["delta_rank"]

        # gap 明显异常（过小或过大），且排名下降
        bad_gap = (pd.notna(gap) and (gap < 2.0 or gap > 8.0))
        if bad_gap and delta > 0:
            return "physically_filtered"  # 电子结构不合理 → 排名下降

        # R_DFT 较高且排名有明显提升
        if (r_dft > 0.7) and (delta < -5):
            return "physically_promoted"  # 物理评分强烈加分 → 排名提升

        return "neutral"

    df_merge["dft_effect"] = df_merge.apply(classify_row, axis=1)

    # 7. 保存结果
    out_path = os.path.abspath(os.path.join(current_dir, "../results/step3c_dft_refined.csv"))
    df_merge.to_csv(out_path, index=False)
    print(f"Step3C: 带 gap+ESP 物理重排的结果已保存到: {out_path}")

    # 8. 简单统计：看看被物理“剔除”和“提升”的数量
    print("\n=== xTB 物理评分影响统计 ===")
    print(df_merge["dft_effect"].value_counts())

    print("\n示例：被物理认为“不太好”的前 5 个：")
    print(
        df_merge[df_merge["dft_effect"] == "physically_filtered"][
            [
                "rank_R0",
                "rank_total",
                "delta_rank",
                "smiles",
                "R0",
                "R_total",
                "gap_ev",
                "esp_min",
            ]
        ].head(5)
    )

    print("\n示例：被物理评分明显“加分”的前 5 个：")
    print(
        df_merge[df_merge["dft_effect"] == "physically_promoted"][
            [
                "rank_R0",
                "rank_total",
                "delta_rank",
                "smiles",
                "R0",
                "R_total",
                "gap_ev",
                "esp_min",
            ]
        ].head(5)
    )


if __name__ == "__main__":
    # 你可以在这里调整 top_k / gamma / w_gap / w_esp
    main(
        top_k=200,
        gamma=0.5,
        w_gap=0.7,
        w_esp=0.3,
    )
