# 文件: core/step4c_utils_merge_results.py
# -*- coding: utf-8 -*-
"""
Step 4C: Master Summary Generator (Professional Edition)

角色：
- 作为整个流水线的“数据仓库 (Data Warehouse)”，
  统一汇总 Step3c (RL + xTB/DFT)、Step4a (ADMET + R_ADMET + R_global)、
  以及可选的 Step4b (PySCF) 结果。
- 对 SMILES 做统一规范化，避免合并时错配。
- 给每个分子增加 Filter_Status / Data_Source_Status 等标签，
  方便后续画漏斗图、做统计分析。

提升点：
1. [关键] SMILES 归一化 (canonicalization)，防止因为字符串写法不同导致 merge 失败。
2. [关键] 自动按文件名模式 + 修改时间找到最新 CSV，而不是手写 alt_names。
3. [关键] 增加 Filter_Status / Is_Final_Top / Data_Source_Status 等标签，透明化筛选逻辑。
"""

import os
import glob
import pandas as pd
from rdkit import Chem
from typing import Optional


def _results_dir():
    """返回 ../results 的绝对路径（相对于 core 目录）"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "results"))


# ===================== SMILES 归一化 ===================== #

def canonicalize_smiles(smiles: str) -> str:
    """
    使用 RDKit 对 SMILES 做规范化（canonicalization）：
      - 统一成同一套规则 (isomericSmiles=True)
      - 防止 "CCO" vs "OCC" 等等写法不同导致的 merge mismatch
    """
    if not isinstance(smiles, str):
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles
    except Exception:
        return smiles


# ===================== 文件自动发现 ===================== #

def find_latest_file(base_dir: str, pattern: str, required: bool = True) -> Optional[str]:
    """
    在 base_dir 里根据通配符 pattern 查找 CSV：
      - 例如 pattern="step3c_dft_refined*.csv"
      - 按修改时间排序，取最新版本
    """
    search_path = os.path.join(base_dir, pattern)
    files = glob.glob(search_path)
    if not files:
        if required:
            raise FileNotFoundError(f"❌ 未找到符合模式的文件: {search_path}")
        else:
            print(f"⚠️ 可选文件未找到: {search_path}")
            return None

    latest_file = max(files, key=os.path.getmtime)
    print(f"✅ 锁定文件: {os.path.basename(latest_file)} (pattern={pattern})")
    return latest_file


# ===================== 主合并逻辑 ===================== #

def merge_all_steps():
    base_dir = _results_dir()
    print(f">>> 正在构建 Step4C Master Summary (Base: {base_dir})...")

    # ---------- 1. 读取 Step3c (必需) ----------
    path_3c = find_latest_file(base_dir, "step3c_dft_refined*.csv", required=True)
    df_3c = pd.read_csv(path_3c)

    if "smiles" not in df_3c.columns:
        raise ValueError(f"{path_3c} 中缺少 'smiles' 列")

    print("   -> 正在归一化 Step3c SMILES...")
    df_3c["merge_key"] = df_3c["smiles"].apply(canonicalize_smiles)

    # ---------- 2. 读取 Step4a ADMET (必需) ----------
    path_4a = find_latest_file(base_dir, "step4a_admet*.csv", required=True)
    df_4a = pd.read_csv(path_4a)

    if "smiles" not in df_4a.columns:
        raise ValueError(f"{path_4a} 中缺少 'smiles' 列")

    print("   -> 正在归一化 Step4a SMILES...")
    df_4a["merge_key"] = df_4a["smiles"].apply(canonicalize_smiles)

    # ---------- 3. 读取 Step4b PySCF (可选) ----------
    path_4b = find_latest_file(base_dir, "*pyscf*.csv", required=False)
    df_4b = None
    if path_4b:
        df_4b = pd.read_csv(path_4b)
        if "smiles" not in df_4b.columns:
            print(f"⚠️ {path_4b} 中没有 'smiles' 列，将忽略 PySCF 数据。")
            df_4b = None
        else:
            print("   -> 正在归一化 Step4b SMILES...")
            df_4b["merge_key"] = df_4b["smiles"].apply(canonicalize_smiles)

    # ============ 4. 开始合并 ============ #

    print(">>> 合并 Step3c (base) + Step4a (ADMET)...")

    # 从 Step4a 中只挑选“新增信息列”，避免覆盖 Step3c 中已有的列
    admet_extra_cols = [
        c for c in df_4a.columns
        if c not in df_3c.columns and c not in ("smiles",)
    ]
    if "merge_key" not in admet_extra_cols:
        admet_extra_cols.append("merge_key")

    df_merge = pd.merge(
        df_3c,
        df_4a[admet_extra_cols],
        on="merge_key",
        how="left",
    )

    # 合并 PySCF 结果（如果有）
    if df_4b is not None:
        print(">>> 发现 Step4b PySCF 结果，正在合并...")
        pyscf_extra_cols = [
            c for c in df_4b.columns
            if c not in df_merge.columns and c not in ("smiles",)
        ]
        if "merge_key" not in pyscf_extra_cols:
            pyscf_extra_cols.append("merge_key")

        df_merge = pd.merge(
            df_merge,
            df_4b[pyscf_extra_cols],
            on="merge_key",
            how="left",
        )
    else:
        print("⚠️ 未能合并 PySCF 数据（文件缺失或不合法），后续筛选仅基于 ADMET / R_global 等。")

    # ============ 5. Data_Source_Status：跑通了哪些步骤？ ============ #

    def determine_data_source_status(row: pd.Series) -> str:
        """
        标记该分子在流水线中跑到了哪一步：
          - Step3c_Only: 只有 RL+DFT (Step3c)，没有 ADMET
          - Step3c+4a:   有 ADMET，但没有 PySCF
          - Step3c+4a+4b:既有 ADMET，又有 PySCF 高精度结果
        """
        has_admet = pd.notna(row.get("Lipinski_Pass"))
        has_pyscf = pd.notna(row.get("PySCF_Gap_eV")) if "PySCF_Gap_eV" in row.index else False

        if has_admet and has_pyscf:
            return "Step3c+4a+4b"
        elif has_admet:
            return "Step3c+4a"
        else:
            return "Step3c_Only"

    df_merge["Data_Source_Status"] = df_merge.apply(determine_data_source_status, axis=1)

    # ============ 6. Filter_Status / Is_Final_Top：筛选标签 ============ #

    HERG_THRESHOLD = 0.5      # 与 Step4a 保持一致
    PYS_CF_GAP_MIN = 4.0      # 可调：PySCF gap 合理下限 (eV)
    PYS_CF_GAP_MAX = 7.0      # 可调：PySCF gap 合理上限 (eV)

    def determine_filter_status(row: pd.Series) -> str:
        """
        根据 ADMET + hERG + PySCF 等信息，给每个分子一个筛选标签：
          - Fail_ADMET_Missing : 没有 ADMET 结果
          - Fail_Lipinski      : Lipinski 规则不合格
          - Fail_hERG_HighRisk : hERG 阻断概率过高
          - Fail_PySCF_Gap     : 有 PySCF 结果但 gap 明显不在合理范围
          - Pass               : 通过上述筛选
        """
        lip = row.get("Lipinski_Pass")

        # 1) ADMET 缺失
        if pd.isna(lip):
            return "Fail_ADMET_Missing"

        # 2) Lipinski 不通过
        if lip is False:
            return "Fail_Lipinski"

        # 3) hERG 高风险
        if "hERG_Prob" in row.index and pd.notna(row["hERG_Prob"]):
            try:
                hp = float(row["hERG_Prob"])
                if hp >= HERG_THRESHOLD:
                    return "Fail_hERG_HighRisk"
            except Exception:
                pass  # 解析失败则忽略，按无 hERG 数据处理

        # 4) PySCF gap 检查（如果有数值）
        if "PySCF_Gap_eV" in row.index and pd.notna(row["PySCF_Gap_eV"]):
            try:
                g = float(row["PySCF_Gap_eV"])
                if not (PYS_CF_GAP_MIN <= g <= PYS_CF_GAP_MAX):
                    return "Fail_PySCF_Gap"
            except Exception:
                # gap 转换失败就忽略 PySCF 这一条，只按 ADMET 通过处理
                pass

        # 5) 其它情况视为通过
        return "Pass"

    df_merge["Filter_Status"] = df_merge.apply(determine_filter_status, axis=1)
    df_merge["Is_Final_Top"] = df_merge["Filter_Status"] == "Pass"

    # ============ 7. 清理辅助列、整理列顺序 ============ #

    if "merge_key" in df_merge.columns:
        df_merge.drop(columns=["merge_key"], inplace=True)

    priority_cols = [
        "smiles",
        "Is_Final_Top",
        "Filter_Status",
        "Active_Set",
        "Data_Source_Status",
        "Reward",
        "R0",
        "R_DFT",
        "R_total",
        "R_PySCF",
        "R_total2",
        "R_total2_base",
        "R_total2_base_col",
        "gamma_pyscf",
        "dft_effect",
        "pIC50",
        "QED",
        "Lipinski_Pass",
        "Violations",
        "MW",
        "LogP",
        "HBD",
        "HBA",
        "RotBonds",
        "TPSA",
        "hERG_Prob",
        "hERG_Risk",
        "R_ADMET",
        "R_global",
        "EHOMO_ev",
        "ELUMO_ev",
        "gap_ev",
        "PySCF_Gap_eV",
        "PySCF_Dipole_Debye",
        "PySCF_HOMO_eV",
        "PySCF_LUMO_eV",
        "PySCF_Energy_Eh",
    ]

    ordered_cols = [c for c in priority_cols if c in df_merge.columns]
    rest_cols = [c for c in df_merge.columns if c not in ordered_cols]
    cols_all = ordered_cols + rest_cols

    df_master = df_merge[cols_all]

    out_path = os.path.join(base_dir, "step4c_master_summary.csv")
    df_master.to_csv(out_path, index=False)

    print("==============================================")
    print(f"✅ Step4C Master Summary 已生成: {out_path}")
    print(f"   分子总数: {len(df_master)}")
    print(f"   Is_Final_Top=True 的条目数: {df_master['Is_Final_Top'].sum()}")
    print("==============================================")

    print("\n[漏斗统计 - Filter_Status]")
    print(df_master["Filter_Status"].value_counts())
    print("==============================================")


if __name__ == "__main__":
    merge_all_steps()