# -*- coding: utf-8 -*-
"""
Step 5A (Patched - Ranking Friendly, No Tier Dependency)
--------------------------------------------------------
功能（保持与原版一致）：
- 从 step4c_master_summary.csv 中选 Top N 分子（默认按 R_global 优先）
- 对每个分子对接到多种冠状病毒 Mpro
- 解析每个靶点结合能，计算 Broad_Spectrum_Score
- 输出 step5a_broadspectrum_docking.csv
- 额外：在输出 CSV 中加入 Broad_Rank / Broad_Rank_Pct（按 Broad_Spectrum_Score 排名）

本补丁的关键改动：
1) Broad_Spectrum_Score 计算更“诚实”：对所有靶点的有效数值都计入最差靶点（max），不再用 score<-0.1 的过滤。
2) 输出自带排名列，便于 Step5B 直接按排名取 TopK，避免阈值分级导致“全灭”。

说明：本脚本不使用 Gold/Silver/Bronze；排名逻辑由分数自然给出 Top1/TopK。
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

# ----------------- 路径基础设置 ----------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

DEFAULT_INPUT_CSV = os.path.join(project_root, "results", "step4c_master_summary.csv")
DEFAULT_OUT_CSV = os.path.join(project_root, "results", "step5a_broadspectrum_docking.csv")
DEFAULT_RECEPTOR_DIR = os.path.join(project_root, "data", "receptors")


# ----------------- Grid Box 配置（保持与你当前版本一致）----------------- #
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



# ===================== Stage2 Guardrails 配置块 =====================
# 目标：在“代理模型 predE3 排序”之后，用少量可解释的理化闸门把明显掉队的结构类型挡在 docking 之外，
#      以提升 TopN 的稳定性（尤其是避免出现 broad 尾巴分子）。
#
# 你可以把它理解为：**不针对某一批分子调参**，而是固化“跨批次都成立”的结构常识。
#
# 参数含义（大白话）：
# - pool:         先从排序结果里取前 pool 个作为候选池（pool 越大，越不容易因为 ADMET/闸门过滤后不够 100 个）
# - mw_min:       分子别太小；太小往往锚点不够、对接不稳（你这里经验上 260 是个安全下限）
# - tpsa_min:     极性/氢键表面积下限；太低常见“抓不住口袋” → broad 尾巴（你已经验证 TPSA>=35 能显著剪尾巴）
# - hba_min:      受体氢键受体数下限；HBA 太少（例如 1）容易出现 hard-negative（如 mol_95 类型）
# - soft_tpsa_target:
#                软目标：TPSA 低于该值时，不直接砍掉，而是给一点“排序惩罚”，把它往后排（默认 45）
# - soft_logp_max:
#                软目标：LogP 高于该值时给一点惩罚，避免过疏水导致 pose 不稳定/ADMET 边缘（默认 4.8）
# - penalty_tpsa / penalty_logp:
#                惩罚强度（越大越“严格”）；一般不建议频繁改，除非你发现尾巴又回潮。
#
# 提供 3 个预设 profile：
# - strict  : 更严格剪尾巴（更稳，但可能可用分子更少）
# - balanced: 默认推荐（你目前验证最接近这个）
# - loose   : 更宽松（适合你担心过滤后不够 100 个时）
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


# ===================== predE3 代理模型训练配置块（轻量默认） =====================
# 说明：这部分主要控制训练速度/稳定性。默认参数是“很快能跑完”的版本。
PRED_E3_DEFAULT = {
    "n_estimators": 300,   # 稳定后可升到 600/800
    "cv_splits": 3,        # 稳定后可升到 5
    "min_samples_leaf": 2,
    "random_state": 0,
}


def resolve_receptor_path(receptor_dir: str, base_filename: str) -> Optional[str]:
    """优先使用 *_gast_clean.pdbqt；否则回退到原文件名。"""
    # 如果原本已经是 gast_clean，则直接使用
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
    从受体 PDBQT 解析 Mpro 口袋中心：取 His41(NE2/ND1) 与 Cys145(SG) 的中点。
    - 不依赖 PDB 文件（直接用 pdbqt 内的残基/原子/坐标字段）
    - 若 NE2 不存在，则尝试 ND1
    返回: {"center_x":..., "center_y":..., "center_z":...} 或 None
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
                    # 期望格式：ATOM serial atom res chain resi x y z ...
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
    """构建一个“可直接用于 docking”的 TARGET_CONFIG 副本：解析受体路径 + 自动口袋中心。"""
    resolved: Dict[str, Dict[str, Any]] = {}
    for virus, conf in TARGET_CONFIG.items():
        base_file = conf["file"]
        rec_path = resolve_receptor_path(receptor_dir, base_file)

        box = dict(conf["box"])  # copy
        if rec_path is None:
            print(f"⚠️ 受体缺失: {virus} | 期望 {base_file} 或 *_gast_clean.pdbqt")
            resolved[virus] = {"path": None, "box": box, "file": base_file}
            continue

        # 自动计算口袋中心；失败则回退配置中心
        cen = pocket_center_from_pdbqt(rec_path)
        if cen is not None:
            box.update(cen)
            print(
                f"🧭 {virus} box center(auto His41/Cys145) = "
                f"({box['center_x']:.3f}, {box['center_y']:.3f}, {box['center_z']:.3f}) "
                f"| receptor={os.path.basename(rec_path)}"
            )
        else:
            print(
                f"⚠️ {virus} 无法从受体解析 His41/Cys145（保持配置中心）"
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
    """RDKit 生成 3D PDB"""
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
    用 obabel 转 PDBQT（保持你原脚本的“pH 7.4 质子化”思路）
    需要系统安装 OpenBabel（obabel）
    """
    try:
        cmd = ["obabel", in_pdb, "-O", out_pdbqt, "--partialcharge", "gasteiger", "-p", "7.4"]
        res = run_cmd(cmd)
        return res.returncode == 0 and os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 0
    except Exception:
        return False


def run_single_docking(lig_pdbqt: str, rec_pdbqt: str, box: Dict[str, float], cpu_per_task: int = 1) -> float:
    """
    调用 vina，返回 best affinity（kcal/mol）
    需要系统安装 vina
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

    # 从 log 中解析 affinity
    try:
        with open(log_txt, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        # vina log 中通常有表格，第一条 mode 的 affinity 在某行
        # 这里做一个稳健解析：找包含 "1 " 且有浮点数的行
        for ln in lines:
            if ln.strip().startswith("1 "):
                parts = ln.split()
                # parts[1] 通常是 affinity
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

    # 3) 对接到多个靶点
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

    # 4) Broad_Spectrum_Score：取“最差靶点”（max，越负越好）
    if not finite_scores:
        broad_score = np.nan
    else:
        broad_score = max(finite_scores)

    record: Dict[str, Any] = {
        "name": name,
        "smiles": smiles,
        "Broad_Spectrum_Score": broad_score,
    }

    # 5) 回填上游信息（如果存在）
    for key in ["pIC50", "Reward", "R_total2", "R_total", "R_ADMET", "R_global"]:
        if key in row.index:
            record[key] = row[key]

    # 6) 各靶点分数
    for virus, sc in rec_scores.items():
        record[f"E_{virus}"] = sc

    return record


def compute_rank_pct(scores: pd.Series) -> pd.Series:
    n = int(scores.notna().sum())
    if n <= 0:
        return pd.Series([np.nan] * len(scores), index=scores.index)
    rank = scores.rank(method="min", ascending=True)
    return rank / float(n)


# ===================== predE3 + Stage2 自动排序/过滤 =====================
def _load_docking_labels(dock_csv_glob: str) -> pd.DataFrame:
    '''
    读取历史 docking 结果作为监督标签。
    期望列：smiles, E_SARS_CoV_2, E_SARS_CoV_1, E_MERS_CoV, Broad_Spectrum_Score
    同一个 smiles 多次出现时，保留 Broad_Spectrum_Score 最好的（最负的那条）。
    '''
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
    '''
    训练多输出 RF：同时预测三个靶点能量。
    返回带 pred_* 列、并按 pred_broad（越小越好）排序的 df。
    '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    if dock_labels.empty:
        raise RuntimeError("No docking labels loaded (dock_labels is empty).")

    m = step4c_df.merge(dock_labels, on="smiles", how="inner")
    if len(m) < 50:
        raise RuntimeError(f"Too few labeled rows to train predE3: {len(m)}")

    # 用 step4c 的数值列做特征；剔除明显“标签/状态/奖励”列
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

    # 轻量 CV（主要 sanity check）
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
    print(f"[predE3] CV MAE(mean over 3 targets) = {np.mean(maes):.3f} ± {np.std(maes):.3f}")

    # 全量训练
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X, Y)

    # 全量预测
    X_all = step4c_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(X_all.median(numeric_only=True))
    pred_all = model.predict(X_all)

    df_out = step4c_df.copy()
    df_out["pred_E_SARS_CoV_2"] = pred_all[:, 0]
    df_out["pred_E_SARS_CoV_1"] = pred_all[:, 1]
    df_out["pred_E_MERS_CoV"] = pred_all[:, 2]
    df_out["pred_broad"] = df_out[["pred_E_SARS_CoV_2", "pred_E_SARS_CoV_1", "pred_E_MERS_CoV"]].max(axis=1)

    # 兼容 step5a 的 TopN 选择：把 R_global 临时替换为 -pred_broad（越大越好）
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

    '''
    Stage2：在“排序后”再过滤/重排（避免 hard-negative 尾巴）。
    - 先按 step5a 逻辑保持 Active_Set==True（如果有）
    - 取前 pool
    - 硬闸门：MW/TPSA/HBA
    - 软惩罚：TPSA 低于 soft_tpsa_target、LogP 高于 soft_logp_max
    '''
    df = df_ranked.copy()

    
    # 与 main() 逻辑保持一致：只有开启 --use_strict_gate 才启用严格门槛
    if use_strict_gate and all(c in df.columns for c in ["Data_Source_Status", "Physical_HardFail", "R_phys"
    ]):
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

    # 硬闸门
    df = df[(df["MW"] >= float(cfg["mw_min"])) &
            (df["TPSA"] >= float(cfg["tpsa_min"])) &
            (df["HBA"] >= int(cfg["hba_min"]))].copy()

    # 软惩罚（只影响排序）
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
    保存 Top N 分子的自由态与对接态结构（匹配当前脚本实际生成的文件名）
    """
    save_dir = os.path.join(project_root, "results", "step5a_top_structures")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Broad_Spectrum_Score 越负越好（升序）
    df_top = df_results.sort_values("Broad_Spectrum_Score", ascending=True).head(top_n)

    print(f"\n>>> 正在提取 Top {top_n} 分子的 3D 结构...")
    for _, row in df_top.iterrows():
        mol_name = str(row["name"])
        mol_src_dir = os.path.join(tmp_root, mol_name)
        mol_dst_dir = os.path.join(save_dir, mol_name)
        os.makedirs(mol_dst_dir, exist_ok=True)

        # 1) 自由态 ligand：实际是 {name}.pdbqt
        lig = os.path.join(mol_src_dir, f"{mol_name}.pdbqt")
        if os.path.exists(lig):
            shutil.copy(lig, os.path.join(mol_dst_dir, f"{mol_name}_free_ligand.pdbqt"))

        # 2) 对接态：实际是 {name}_out_*.pdbqt
        for p in glob.glob(os.path.join(mol_src_dir, f"{mol_name}_out_*.pdbqt")):
            shutil.copy(p, os.path.join(mol_dst_dir, os.path.basename(p)))

    print(f"✅ 结构已保存至: {save_dir}")



def main():
    parser = argparse.ArgumentParser(description="Step 5A: Broad-Spectrum Docking (Ranking-friendly)")
    parser.add_argument("--use_strict_gate", action="store_true",
                help="使用严格MD门槛筛选：必须有PySCF且Physical_HardFail=False且R_phys>=阈值")
    parser.add_argument("--rphys_min", type=float, default=0.85,
                help="严格门槛的 R_phys 下限（默认 0.85；可改 0.80）")
    
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV, help=f"输入 Step4C 总表 (默认: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--out_csv", type=str, default=DEFAULT_OUT_CSV, help=f"输出 docking 结果 CSV (默认: {DEFAULT_OUT_CSV})")
    parser.add_argument("--receptor_dir", type=str, default=DEFAULT_RECEPTOR_DIR, help=f"受体 pdbqt 所在目录 (默认: {DEFAULT_RECEPTOR_DIR})")
    parser.add_argument("--top_n", type=int, default=20, help="从 Step4C 中选前 top_n 个分子做 docking (默认: 20)")
    
    parser.add_argument("--workers", type=int, default=4, help="并行处理的分子数 (默认: 4)")
    parser.add_argument("--vina_cpu", type=int, default=1, help="每个 Vina 进程使用的 CPU 数 (默认: 1)")
    # --- 固化模式：predE3 代理排序 + Stage2 guardrails（可选启用） ---
    parser.add_argument("--auto_predE3_stage2", action="store_true",
                        help="启用：自动用历史 docking 训练 predE3，并执行 Stage2 guardrails 后再 dock（推荐）")
    parser.add_argument("--step4c_csv", type=str, default=None,
                        help="Step4C master csv（不填则沿用 --input_csv）")
    parser.add_argument("--dock_csv_glob", type=str, default=os.path.join(project_root, "results", "step5a_broadspectrum_docking*.csv"),
                        help="历史 docking 结果 glob，用于训练 predE3（默认: results/step5a_broadspectrum_docking*.csv）")

    # predE3 训练参数（默认轻量快速）
    parser.add_argument("--predE3_n_estimators", type=int, default=PRED_E3_DEFAULT["n_estimators"])
    parser.add_argument("--predE3_cv_splits", type=int, default=PRED_E3_DEFAULT["cv_splits"])
    parser.add_argument("--predE3_n_jobs", type=int, default=40, help="predE3 训练用并行核数（建议<=CPU配额）")

    # Stage2 参数：优先使用 profile，再用单项参数覆盖
    parser.add_argument("--stage2_profile", type=str, default=DEFAULT_STAGE2_PROFILE, choices=list(STAGE2_PROFILES.keys()),
                        help="Stage2 预设：strict/balanced/loose（默认 balanced）")
    parser.add_argument("--stage2_pool", type=int, default=None, help="覆盖 profile.pool（不填则用 profile 默认）")
    parser.add_argument("--stage2_mw_min", type=float, default=None, help="覆盖 profile.mw_min")
    parser.add_argument("--stage2_tpsa_min", type=float, default=None, help="覆盖 profile.tpsa_min")
    parser.add_argument("--stage2_hba_min", type=int, default=None, help="覆盖 profile.hba_min")
    parser.add_argument("--stage2_soft_tpsa_target", type=float, default=None, help="覆盖 profile.soft_tpsa_target")
    parser.add_argument("--stage2_soft_logp_max", type=float, default=None, help="覆盖 profile.soft_logp_max")
    parser.add_argument("--stage2_penalty_tpsa", type=float, default=None, help="覆盖 profile.penalty_tpsa")
    parser.add_argument("--stage2_penalty_logp", type=float, default=None, help="覆盖 profile.penalty_logp")

    # === 在这里插入您新增的参数 ===
    parser.add_argument("--save_top_structures", action="store_true", default=True, 
                        help="是否提取并保存 Top N 分子的 3D 结构文件")
    parser.add_argument("--top_n_save", type=int, default=20, 
                        help="指定保存前多少个分子的结构")

    parser.add_argument("--write_intermediate_csv", action="store_true",
                        help="在 auto_predE3_stage2 下，写出 results/step4c_master_summary_SORTBY_predE3*.csv 以便复现")
    parser.add_argument("--no_write_intermediate_csv", action="store_true",
                        help="在 auto_predE3_stage2 下，不写中间 CSV（默认会写）")
    args = parser.parse_args()

    # ----------------- 读取 Step4C 总表 ----------------- #
    step4c_csv = args.step4c_csv or args.input_csv
    if not os.path.exists(step4c_csv):
        raise FileNotFoundError(f"找不到输入: {step4c_csv}")
    df4 = pd.read_csv(step4c_csv)
    if df4.empty:
        print("⚠️ Step4C 输入为空，退出。")
        return

    # ----------------- 可选：自动 predE3 排序 + Stage2 guardrails ----------------- #
    if args.auto_predE3_stage2:
        # 1) 组装 Stage2 配置（profile + override）
        cfg = STAGE2_PROFILES.get(args.stage2_profile, STAGE2_PROFILES[DEFAULT_STAGE2_PROFILE]).copy()
        # overrides
        if args.stage2_pool is not None: cfg["pool"] = args.stage2_pool
        if args.stage2_mw_min is not None: cfg["mw_min"] = args.stage2_mw_min
        if args.stage2_tpsa_min is not None: cfg["tpsa_min"] = args.stage2_tpsa_min
        if args.stage2_hba_min is not None: cfg["hba_min"] = args.stage2_hba_min
        if args.stage2_soft_tpsa_target is not None: cfg["soft_tpsa_target"] = args.stage2_soft_tpsa_target
        if args.stage2_soft_logp_max is not None: cfg["soft_logp_max"] = args.stage2_soft_logp_max
        if args.stage2_penalty_tpsa is not None: cfg["penalty_tpsa"] = args.stage2_penalty_tpsa
        if args.stage2_penalty_logp is not None: cfg["penalty_logp"] = args.stage2_penalty_logp

        print(f"🧠 auto_predE3_stage2=ON | stage2_profile={args.stage2_profile} | cfg={cfg}")

        # 2) 读取历史 docking 标签
        dock_labels = _load_docking_labels(args.dock_csv_glob)
        print(f"🧪 predE3 labels loaded: {len(dock_labels)} unique smiles (glob={args.dock_csv_glob})")

        try:
            # 3) predE3 排序
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

            # 5) 可选：写中间 CSV（默认写，除非显式 --no_write_intermediate_csv）
            do_write = (not args.no_write_intermediate_csv) or args.write_intermediate_csv
            if do_write:
                out1 = os.path.join(project_root, "results", "step4c_master_summary_SORTBY_predE3.csv")
                out2 = os.path.join(project_root, "results", "step4c_master_summary_SORTBY_predE3_stage2.csv")
                df_ranked.to_csv(out1, index=False)
                df.to_csv(out2, index=False)
                print(f"📝 wrote intermediates: {out1} | {out2}")

            print(f"🧱 Stage2 kept rows: {len(df)} (after ADMET+pool+MW/TPSA/HBA + soft-penalty reorder)")
            if df.empty:
                print("⚠️ Stage2 过滤后为空；回退到原始排序（R_global）。")
                df = df4.copy()

        except Exception as e:
            print(f"⚠️ auto_predE3_stage2 failed ({e}); fallback to original ranking (R_global).")
            df = df4.copy()
    else:
        df = df4.copy()

    # === 决赛名单筛选 (物理否决权核心) ===
    # 优先使用 Is_Final_Top，确保只有通过 DFT 终审且符合物理/药代标准的分子进入对接
    # 1) 入口筛选：严格门槛优先（得到 36），否则回退 Is_Final_Top / Active_Set
    if args.use_strict_gate and all(c in df.columns for c in ["Data_Source_Status", "Physical_HardFail", "R_phys"]):
        df = df[
            (df["Data_Source_Status"] == "Step3c+4a+4b") &
            (df["Physical_HardFail"] == False) &
            (df["R_phys"] >= float(args.rphys_min))
        ].copy()
        print(f"✅ 严格门槛生效：仅对通过 PySCF+DFT 且 R_phys>={args.rphys_min} 的 {len(df)} 个分子 docking。")

    elif "Is_Final_Top" in df.columns:
        df = df[df["Is_Final_Top"] == True].copy()
        print(f"✅ 使用 Is_Final_Top：{len(df)} 个分子 docking。")

    elif "Active_Set" in df.columns:
        df = df[(df["Active_Set"] == True) | (df["Active_Set"] == 1)].copy()
        print(f"✅ 使用 Active_Set：{len(df)} 个分子 docking。")
        if df.empty:
            print("⚠️ Active_Set 筛选后为空，退出。")
            return

    # 选择 TopN（这里仍保留 TopN：docking 成本太高；你也可以把 top_n 改大）
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

    print(f"📥 输入分子数: {len(df)} | 选择 docking TopN={len(df_top)} | sort_by={sort_col}")

    # 受体优先使用 *_gast_clean.pdbqt，并自动按 His41/Cys145 计算 box center
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
                print(f"[{done_cnt}/{total}] 该分子处理失败")

    # --- 插入点：在清理 tmp 之前提取结构 ---
    # 先检查有没有结果
    if not results:
        print("⚠️ 没有任何成功的 docking 结果，未生成输出文件")
        # 可选：失败时也清理 tmp
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass
        return

    # 先创建 df_res（关键：后面保存结构/排名/保存CSV都依赖它）
    df_res = pd.DataFrame(results)

    # --- 在清理 tmp 之前提取结构（此时 df_res 已存在） ---
    if args.save_top_structures:
        save_top_n_structures(df_res, tmp_root, top_n=args.top_n_save)

    # 清理临时目录（可按需要保留）
    try:
        shutil.rmtree(tmp_root)
    except Exception:
        pass


    # 生成排名列（越负越好 -> 升序）
    df_res["Broad_Rank"] = df_res["Broad_Spectrum_Score"].rank(method="min", ascending=True)
    df_res["Broad_Rank_Pct"] = compute_rank_pct(df_res["Broad_Spectrum_Score"])

    # 整理列顺序
    base_cols = ["name", "smiles", "Broad_Spectrum_Score", "Broad_Rank", "Broad_Rank_Pct",
                 "E_SARS_CoV_2", "E_SARS_CoV_1", "E_MERS_CoV"]
    extra_cols = [c for c in df_res.columns if c not in base_cols]
    df_res = df_res[base_cols + extra_cols]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_res.to_csv(args.out_csv, index=False)

    print("\n========================================")
    print(f"✅ 广谱对接完成，结果已保存: {args.out_csv}")
    print(f"   docking 分子数: {len(df_res)}")
    if df_res["Broad_Spectrum_Score"].notna().any():
        s = df_res["Broad_Spectrum_Score"].dropna()
        print(f"   Broad_Spectrum_Score: min={s.min():.3f} mean={s.mean():.3f} max={s.max():.3f}")
        best = df_res.sort_values("Broad_Spectrum_Score").iloc[0]
        print(f"   Top1: {best['name']} score={best['Broad_Spectrum_Score']:.3f}")
    print("========================================")


if __name__ == "__main__":
    main()