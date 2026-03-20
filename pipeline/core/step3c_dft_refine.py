# File: core/step3c_dft_refine.py
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
    Gaussian-shaped score based on the HOMO-LUMO gap.
    Score is highest when gap_ev is closest to mu_gap; range is (0, 1].

    Parameters:
    - gap_ev:    HOMO-LUMO gap computed by xTB (eV)
    - mu_gap:    Target gap center (default: 5 eV)
    - sigma_gap: Tolerance width (default: 2 eV)

    Returns: float in [0, 1] (higher is better)
    """
    if pd.isna(gap_ev):
        return 0.0
    x = (gap_ev - mu_gap) / sigma_gap
    return float(np.exp(-0.5 * x * x))


def score_esp(esp_min, esp_min_ref=-0.4, delta_neg=0.6, delta_pos=0.4):
    """
    Score based on a proxy for the electrostatic potential minimum
    (most negative atomic charge from xTB).

    Design rationale:
    - esp_min is taken from xTB's most negative atomic charge (e.g. Mulliken/xTB charges),
      typically in the range [-1, 0]; more negative indicates greater local electron density.
    - A moderately negative value is preferred:
      * Near 0: overly uniform electron distribution; weak H-bond acceptor character
      * Around -0.4: considered favorable
      * Below -1: extreme case; score does not increase further

    Parameters:
    - esp_min:     Most negative atomic charge from xTB (negative value)
    - esp_min_ref: Ideal reference value (default: -0.4)
    - delta_neg:   Allowed negative deviation from esp_min_ref (default: 0.6, covering ~[-1.0, -0.4])
    - delta_pos:   Allowed positive deviation from esp_min_ref (default: 0.4, covering ~[-0.4, 0.0])

    Implementation:
    - Triangular/trapezoidal scoring over [esp_min_ref - delta_neg, esp_min_ref + delta_pos]:
      * Score ~1 at esp_min_ref
      * Linear decay to 0 on both sides
      * Clipped to 0 outside the range
    """
    if pd.isna(esp_min):
        return 0.0

    esp = float(esp_min)

    left = esp_min_ref - delta_neg   # Below this: treated as worst case
    right = esp_min_ref + delta_pos  # Above this: treated as worst case

    if esp <= left or esp >= right:
        # Outside the acceptable range: score 0
        return 0.0

    if esp <= esp_min_ref:
        # Linear ramp from 0 -> 1 over [left, esp_min_ref]
        score = (esp - left) / (esp_min_ref - left)
    else:
        # Linear ramp from 1 -> 0 over [esp_min_ref, right]
        score = (right - esp) / (right - esp_min_ref)

    # Safety clip to [0, 1]
    score = max(0.0, min(1.0, score))
    return float(score)


def main(
    top_k=200,
    gamma=0.4,
    w_gap=0.7,
    w_esp=0.3,
):
    """
    Step 3C: Physics-informed re-ranking of RL candidates using xTB electronic structure
             (HOMO-LUMO gap and ESP proxy).

    Core logic:
    - Take the top top_k candidates from Step 3A output (sorted by Reward)
    - Read Step 3B xTB results (EHOMO_ev, ELUMO_ev, gap_ev, esp_min)
    - Compute gap_score from gap_ev and esp_score from esp_min
    - Form R_DFT = w_gap * gap_score + w_esp * esp_score
    - Define R0 = original Reward; R_total = R0 + gamma * R_DFT
    - Compare R0 ranking against R_total ranking to assess the influence of
      electronic structure on candidate prioritization

    Note:
    - esp_min is assumed to represent the most negative atomic charge proxy;
      ensure Step 3B has written this column to the CSV.
    """
    # 1. Read Step 3A RL optimization results (baseline)
    rl_path = os.path.abspath(os.path.join(current_dir, "../results/step3a_optimized_molecules.csv"))
    if not os.path.exists(rl_path):
        raise FileNotFoundError(f"Step 3A output file not found: {rl_path}")
    df_rl = pd.read_csv(rl_path)

    # Verify Reward column is present
    if "Reward" not in df_rl.columns:
        raise ValueError("'Reward' column not found in step3a_optimized_molecules.csv. Check Step 3A output.")

    # Select top K
    df_rl = df_rl.sort_values("Reward", ascending=False).reset_index(drop=True)
    df_top = df_rl.iloc[:top_k].copy()
    print(f"Step3C: Using top {top_k} candidates from Step 3A for xTB-based physics re-ranking.")

    # 2. Read xTB results (Step 3B)
    dft_path = os.path.abspath(os.path.join(current_dir, "../results/step3b_dft_results.csv"))
    if not os.path.exists(dft_path):
        raise FileNotFoundError(
            f"DFT/xTB result file not found: {dft_path}\n"
            f"Please run step3b_run_dft.py first to generate step3b_dft_results.csv."
        )
    df_dft = pd.read_csv(dft_path)

    required_cols = ["smiles", "EHOMO_ev", "ELUMO_ev", "gap_ev", "esp_min"]
    for c in required_cols:
        if c not in df_dft.columns:
            raise ValueError(f"Required column missing from step3b_dft_results.csv: '{c}'")

    # If a status column is present, retain only successful records
    if "status" in df_dft.columns:
        df_dft = df_dft[df_dft["status"].isin(["xtb_success", "mock_success"])].copy()

    # 3. Merge RL results with xTB results
    df_merge = pd.merge(df_top, df_dft, on="smiles", how="left", suffixes=("", "_dft"))
    print(f"Merged table contains {len(df_merge)} records.")

    # 4. Compute xTB-based physics scores (gap_score + esp_score)
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

    # 5. Compute total score: R_total = R0 + gamma * R_DFT
    df_merge["R0"] = df_merge["Reward"]
    df_merge["R_total"] = df_merge["R0"] + gamma * df_merge["R_DFT"]

    # 6. Assign ranks under R0 and R_total; compute rank change
    df_merge = df_merge.sort_values("R0", ascending=False).reset_index(drop=True)
    df_merge["rank_R0"] = df_merge.index + 1  # Original rank

    df_merge = df_merge.sort_values("R_total", ascending=False).reset_index(drop=True)
    df_merge["rank_total"] = df_merge.index + 1  # Physics-adjusted rank

    # delta_rank > 0: rank decreased; delta_rank < 0: rank improved
    df_merge["delta_rank"] = df_merge["rank_total"] - df_merge["rank_R0"]

    # Classify physics effect
    def classify_row(row):
        gap = row["gap_ev"]
        r_dft = row["R_DFT"]
        delta = row["delta_rank"]

        # Anomalous gap (too small or too large) and rank degradation
        bad_gap = (pd.notna(gap) and (gap < 2.0 or gap > 8.0))
        if bad_gap and delta > 0:
            return "physically_filtered"  # Unfavorable electronic structure -> rank decreased

        # High R_DFT and substantial rank improvement
        if (r_dft > 0.7) and (delta < -5):
            return "physically_promoted"  # Strong physics bonus -> rank improved

        return "neutral"

    df_merge["dft_effect"] = df_merge.apply(classify_row, axis=1)

    # 7. Save results
    out_path = os.path.abspath(os.path.join(current_dir, "../results/step3c_dft_refined.csv"))
    df_merge.to_csv(out_path, index=False)
    print(f"Step3C: Physics re-ranked results (gap + ESP) saved to: {out_path}")

    # 8. Summary statistics: count of physically filtered vs. promoted candidates
    print("\n=== xTB Physics Score Effect Summary ===")
    print(df_merge["dft_effect"].value_counts())

    print("\nExamples: top 5 molecules flagged as physically unfavorable:")
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

    print("\nExamples: top 5 molecules strongly promoted by physics score:")
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
    # Adjust top_k / gamma / w_gap / w_esp as needed
    main(
        top_k=200,
        gamma=0.5,
        w_gap=0.7,
        w_esp=0.3,
    )
