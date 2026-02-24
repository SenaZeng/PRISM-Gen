# 文件: core/step4b_final_pyscf.py
# -*- coding: utf-8 -*-
"""
Step 4B: Parallel PySCF Calculation (Aligned + Top20 Strategy)
--------------------------------------------------------------
✅ 目标：在不改变项目现有 step4b_final_pyscf.py 的【参数命名 / 日志风格 / 输出列顺序 / 输出路径】前提下，
把 “Top20=10冲分+10保险（多样性）” 的选择策略内置进来。

保持无感的点（与旧版一致）：
- CLI 参数：--input_file / --output_file / --top_k / --workers（名字与默认值不变）
- 日志输出：📥 / 🔍 / 🚀 / ⏳ / ✅ 的格式与语句保持一致
- 输出文件名默认仍为：../results/step4b_top_molecules_pyscf.csv（Step4C 不需要改）
- 输出列顺序规则仍为：把 ["smiles","PySCF_Gap_eV","PySCF_Dipole_Debye","R_global"] 放最前，其余列按原样追加

变化点（“无感升级”）：
- 仍然会做 Lipinski + hERG 的过滤（若列存在），但 TopK 的选择不再是“单一分数截断”：
  先构建 Candidate Pool（默认 200=100/50/50），再取 Exploitation/Exploration 组合，最后补齐到 top_k。
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


# ================= 默认配置 =================
DEFAULT_INPUT_FILE = "../results/step4a_admet_final.csv"
DEFAULT_OUTPUT_FILE = "../results/step4b_top_molecules_pyscf.csv"
DEFAULT_TOP_K = 20  # 默认值，会被命令行参数覆盖
DEFAULT_WORKERS = 40
# ============================================


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
    """RDKit 生成 3D 坐标并转换为 PySCF 可用 XYZ 字符串"""
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
    task: (smiles, row_idx)
    返回 dict；失败返回 None（保持旧版逻辑风格）
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
            "_row_idx": int(row_idx),              # 内部对齐键（最终不输出）
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


# ------------------ TopK 选择策略（内置，无需改参数）------------------

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
    选 TopK：先候选池（N1/N2/N3 合并去重） -> Exploitation（按 used_sort_col） -> Exploration（Butina 多样性） -> 补齐到 top_k
    """
    df = df_clean.copy()
    df["_canon"] = df["smiles"].apply(canonicalize_smiles)

    # N1: main score top
    part1 = df.sort_values(used_sort_col, ascending=False).head(pool_n1) if used_sort_col else df.head(0)

    # N2: activity top
    act_col = None
    for c in ["Reward", "pIC50"]:
        if c in df.columns:
            act_col = c
            break
    part2 = df.sort_values(act_col, ascending=False).head(pool_n2) if act_col else df.head(0)

    # N3: developability (QED high + SA low) split
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

    # Exploration (cluster reps)
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

    # 补齐：先用 pool 剩余，再用 df 全表
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
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE, help="输入文件 (Step4A输出)")
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE, help="输出文件")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="筛选前 K 个分子进行计算")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行进程数")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ 找不到输入文件: {args.input_file}")
        return

    df = pd.read_csv(args.input_file)
    print(f"📥 读取候选分子: {len(df)} 个")

    # 2. 过滤逻辑 (Lipinski + hERG) —— 保持旧版行为
    df_clean = df.copy()

    # Active_Set 过滤（统一起点）：如果 Step4A 已写入 Active_Set，则优先以其为准。
    if "Active_Set" in df_clean.columns:
        df_clean = df_clean[df_clean["Active_Set"] == True].copy()

    # Lipinski 过滤
    if "Lipinski_Pass" in df_clean.columns:
        df_clean = df_clean[df_clean["Lipinski_Pass"] == True].copy()

    # hERG 过滤 (排除高风险)
    if "hERG_Risk" in df_clean.columns:
        df_clean = df_clean[
            (df_clean["hERG_Risk"] == False) | (df_clean["hERG_Risk"].isna())
        ].copy()

    print(f"🔍 经过 ADMET 过滤后剩余: {len(df_clean)} 个")

    if df_clean.empty:
        print("⚠️ 过滤后无候选分子，输出空文件")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        pd.DataFrame(columns=["smiles"]).to_csv(args.output_file, index=False)
        return

    # 3. 排序依据 (优先 R_global) —— 保持旧版输出文案
    sort_cols = ["R_global", "R_total", "R0", "Reward", "pIC50"]
    used_sort_col = choose_sort_col(df_clean, sort_cols)
    if used_sort_col:
        df_clean = df_clean.sort_values(used_sort_col, ascending=False)

    # 4. 选取 Top K（内置升级：CandidatePool + 10冲分+10保险）
    candidates = select_candidates_topk(df_clean, top_k=args.top_k, used_sort_col=used_sort_col or "smiles")

    print(f"🚀 [Step 4B] 启动 PySCF 计算: Top {len(candidates)} (排序依据: {used_sort_col})")
    print(f"    并行核心: {args.workers}")
    print("    建议外部设置 OMP_NUM_THREADS=1 防止 PySCF 内部线程与多进程冲突")

    # 给候选加内部行号，避免 smiles 重复导致回填错行（内部使用，不输出）
    candidates = candidates.reset_index(drop=True).copy()
    candidates["_row_idx"] = np.arange(len(candidates), dtype=int)

    tasks = list(zip(candidates["smiles"].astype(str).tolist(), candidates["_row_idx"].astype(int).tolist()))

    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, res in enumerate(executor.map(run_pyscf_task, tasks), 1):
            if res:
                # 补全原始信息 (将 pIC50, R_global 等合并回去) —— 保持旧版字段集合
                orig_row = candidates[candidates["_row_idx"] == res["_row_idx"]].iloc[0]
                for key in ["pIC50", "QED", "SA", "hERG_Prob", "hERG_Risk", "R0", "R_total", "R_ADMET", "R_global"]:
                    if key in orig_row:
                        res[key] = orig_row[key]
                results.append(res)

            # 进度显示（保持旧版风格）
            if i % 5 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r⏳ 进度: {i}/{len(tasks)} | 成功: {len(results)} | 耗时: {elapsed:.1f}s")
                sys.stdout.flush()

    print("\n")

    # 6. 保存结果（保持旧版输出形式与列顺序策略）
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if results:
        df_final = pd.DataFrame(results)

        # 内部键不输出
        if "_row_idx" in df_final.columns:
            df_final = df_final.drop(columns=["_row_idx"])

        # 20260105 hongmei
        def _mad(x: np.ndarray) -> float:
            med = np.median(x)
            return float(np.median(np.abs(x - med)))

        def _robust_z(x: np.ndarray, med: float, mad: float, eps: float = 1e-9) -> np.ndarray:
            return 0.6745 * (x - med) / (mad + eps)

        # ===== Robust physical gating (Gap + Dipole) =====
        if "PySCF_Gap_eV" in df_final.columns and "PySCF_Dipole_Debye" in df_final.columns:
            gap = df_final["PySCF_Gap_eV"].astype(float).values
            dip = df_final["PySCF_Dipole_Debye"].astype(float).values

            # --- Gap robust stats ---
            gap_med = float(np.median(gap))
            gap_mad = _mad(gap)
            gap_rz = _robust_z(gap, gap_med, gap_mad)

            df_final["Gap_Median"] = gap_med
            df_final["Gap_MAD"] = gap_mad
            df_final["Gap_RZ"] = gap_rz

            # Hard fail: only extreme outliers
            df_final["Gap_HardFail"] = (np.abs(df_final["Gap_RZ"]) >= 3.5)

            # Soft penalty
            z0, z1 = 1.5, 3.5
            gap_pen = np.clip((np.abs(gap_rz) - z0) / (z1 - z0), 0, 1)
            df_final["Gap_Penalty"] = gap_pen
            df_final["R_gap"] = 1 - gap_pen

            # --- Dipole robust stats ---
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

            # --- Integrate ---
            df_final["Physical_HardFail"] = df_final["Gap_HardFail"] | df_final["Dipole_HardFail"]

            # In Step4B we don't have TPSA/LogP, so set R_conf=1
            df_final["R_conf"] = 1.0

            df_final["R_phys"] = (df_final["R_gap"] ** 1.0) * (df_final["R_dip"] ** 1.0) * (df_final["R_conf"] ** 0.5)

            if "R_global" in df_final.columns:
                df_final["R_global2"] = df_final["R_global"].astype(float) * df_final["R_phys"].astype(float)

            # 20260105 hongmei 
            # 简单整理一下列顺序，把重要的放前面（保持旧版）
            cols = list(df_final.columns)
            head_cols = ["smiles", "PySCF_Gap_eV", "PySCF_Dipole_Debye", "R_global"]
            sorted_cols = [c for c in head_cols if c in cols] + [c for c in cols if c not in head_cols]
            df_final = df_final[sorted_cols]

            df_final.to_csv(args.output_file, index=False)
            print(f"✅ PySCF 结果已保存至: {args.output_file}")
        else:
            # （可选）如果 df_final 存在但缺字段，也输出一个文件，至少不崩
            df_final.to_csv(args.output_file, index=False)
            print(f"✅ PySCF 结果已保存至: {args.output_file} (缺少部分列，未做物理门控)")
    else:
        print("⚠️ PySCF 计算全部失败或未收敛")
        pd.DataFrame(columns=["smiles"]).to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()