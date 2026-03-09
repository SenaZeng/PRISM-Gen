# 文件: core/step4a_admet.py
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED
from rdkit.Chem import AllChem, DataStructs

import joblib  # 用于加载 hERG 预测模型

# 输入文件：Step 3C (xTB 物理重排后) 的结果
INPUT_FILE = "../results/step3c_dft_refined.csv"
OUTPUT_FILE = "../results/step4a_admet_final.csv"

# hERG 模型位置（方案 A：放在 results 目录下）
HERG_MODEL_PATH = "../results/herg_model/herg_rf_model.pkl"
HERG_THRESHOLD = 0.5  # hERG 高风险阈值，可调


# ---------- hERG 相关工具函数 ---------- #

def smiles_to_fp(smi, radius=2, n_bits=2048):
    """
    将 SMILES 转为 Morgan 指纹，用于 hERG 模型输入。
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((n_bits,), dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def predict_herg_risk(smiles, herg_model):
    """
    使用预训练的 hERG 模型预测心脏毒性风险。
    返回:
        prob: 阻断 hERG 的预测概率 (float 或 None)
        is_risk: 是否为高风险 (bool 或 None)
    """
    if herg_model is None:
        return None, None

    fp = smiles_to_fp(smiles)
    proba = herg_model.predict_proba(fp.reshape(1, -1))[0, 1]
    is_risk = bool(proba >= HERG_THRESHOLD)
    return float(proba), is_risk


# ---------- ADMET 计算 ---------- #

def calc_admet_props(smiles, herg_model=None):
    """
    对单个分子计算基础 ADMET 属性：
      - Lipinski 五规则相关 (MW, LogP, HBD, HBA, RotBonds)
      - TPSA
      - QED (定量成药性，0~1)
      - hERG 心脏毒性预测 (如果有模型)
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # 1. Lipinski 五规则指标
    mw = Descriptors.MolWt(mol)        # 分子量 (<500)
    logp = Crippen.MolLogP(mol)        # 脂溶性 (<5)
    hbd = Lipinski.NumHDonors(mol)     # 氢键供体 (<5)
    hba = Lipinski.NumHAcceptors(mol)  # 氢键受体 (<10)
    rot_bonds = Lipinski.NumRotatableBonds(mol)  # 可旋转键 (<10)
    
    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    is_lipinski_pass = (violations <= 1)

    # 2. 其他成药性指标
    tpsa = Descriptors.TPSA(mol)       # 极性表面积 (透膜性, <140 Å²)
    # QED: 定量成药性评分（0~1，越高越“药”）
    qed_val = QED.qed(mol)

    # 3. hERG 心脏毒性预测（AI 毒性筛查）
    herg_prob, herg_risk = predict_herg_risk(smiles, herg_model)

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": int(hbd),
        "HBA": int(hba),
        "RotBonds": int(rot_bonds),
        "TPSA": round(tpsa, 2),
        "QED": round(qed_val, 3),
        "Lipinski_Pass": is_lipinski_pass,
        "Violations": int(violations),
        "hERG_Prob": herg_prob,   # 0~1 概率
        "hERG_Risk": herg_risk,   # True=高风险, False=低风险
    }


def compute_r_admet_and_global(df_final,
                               alpha_lip=0.6,
                               alpha_safety=0.4,
                               beta=2.0):
    """
    在已有 df_final 上增加：
      - R_ADMET: 基于 Lipinski + (1 - hERG_Prob 或 QED) 的 ADMET 综合分
      - R_global: 综合活性-物理-ADMET 分数

    设计：
      Lipinski_score = 1.0 (通过) or 0.0 (不通过/缺失)
      safety_score:
        - 优先使用 (1 - hERG_Prob), 范围截断到 [0,1]
        - 如果没有 hERG_Prob，则退回使用 QED (0~1)
        - 如果两者都没有，则给一个中性值 0.5

      R_ADMET = alpha_lip * Lipinski_score + alpha_safety * safety_score

      R_base:
        - 优先使用 R_total (Step 3c 的活性+电子结构总分)
        - 若不存在，则退回 Reward
        - 若仍不存在，则视为 0

      R_global = R_base + beta * R_ADMET
    """
    def _row_score(row):
        # Lipinski 部分
        lip_pass = row.get("Lipinski_Pass")
        lip_score = 1.0 if (lip_pass is True) else 0.0

        # 安全性部分：hERG 优先，其次 QED
        safety_score = None
        if "hERG_Prob" in row.index and pd.notna(row["hERG_Prob"]):
            try:
                safety_score = 1.0 - float(row["hERG_Prob"])
            except Exception:
                safety_score = None

        if safety_score is None:
            # 退回使用 QED（0~1）
            if "QED" in row.index and pd.notna(row["QED"]):
                try:
                    safety_score = float(row["QED"])
                except Exception:
                    safety_score = None

        if safety_score is None:
            safety_score = 0.5  # 中性默认值

        # clamp 到 [0,1]
        safety_score = max(0.0, min(1.0, safety_score))

        r_admet = alpha_lip * lip_score + alpha_safety * safety_score

        # 上游基准分
        if "R_total" in row.index and pd.notna(row["R_total"]):
            base = float(row["R_total"])
        elif "Reward" in row.index and pd.notna(row["Reward"]):
            base = float(row["Reward"])
        else:
            base = 0.0

        r_global = base + beta * r_admet

        return pd.Series({
            "R_ADMET": round(r_admet, 3),
            "R_global": round(r_global, 3),
        })

    scores = df_final.apply(_row_score, axis=1)
    df_with_scores = pd.concat([df_final, scores], axis=1)
    return df_with_scores


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(current_dir, INPUT_FILE)
    out_path = os.path.join(current_dir, OUTPUT_FILE)
    herg_model_path = os.path.join(current_dir, HERG_MODEL_PATH)

    if not os.path.exists(in_path):
        print(f"❌ 找不到输入文件: {in_path}")
        return

    # 尝试加载 hERG 模型
    herg_model = None
    if os.path.exists(herg_model_path):
        print(f"🧪 加载 hERG 预测模型: {herg_model_path}")
        try:
            herg_model = joblib.load(herg_model_path)
            print("✅ hERG 模型加载成功，将启用 AI 心脏毒性筛查。")
        except Exception as e:
            print(f"⚠️ hERG 模型加载失败，将跳过 hERG 预测。错误信息: {e}")
            herg_model = None
    else:
        print(f"⚠️ 未找到 hERG 模型文件: {herg_model_path}，将跳过 hERG 预测。")

    print(f"读取 Step3C 结果: {in_path}")
    df = pd.read_csv(in_path)
    
    print(f"开始计算 ADMET 属性 (共 {len(df)} 个分子)...")

    admet_data = []
    empty_admet = {
        "MW": None,
        "LogP": None,
        "HBD": None,
        "HBA": None,
        "RotBonds": None,
        "TPSA": None,
        "QED": None,
        "Lipinski_Pass": None,
        "Violations": None,
        "hERG_Prob": None,
        "hERG_Risk": None,
    }

    for idx, row in df.iterrows():
        smiles = row.get("smiles", None)
        if pd.isna(smiles):
            admet_data.append(empty_admet.copy())
            continue

        props = calc_admet_props(smiles, herg_model=herg_model)
        if props:
            admet_data.append(props)
        else:
            admet_data.append(empty_admet.copy())

    # 合并 ADMET 数据
    df_admet = pd.DataFrame(admet_data)
    df_final = pd.concat([df, df_admet], axis=1)

    # === 新增：计算 R_ADMET 和 R_global ===
    df_final = compute_r_admet_and_global(df_final)

    # 先看 Lipinski 通过情况
    df_lipinski_pass = df_final[df_final["Lipinski_Pass"] == True].copy()

    # 如果有 hERG 预测，再统计 hERG 情况
    if "hERG_Risk" in df_final.columns:
        df_herg_risk_true = df_final[df_final["hERG_Risk"] == True]
        df_lipinski_herg_pass = df_final[
            (df_final["Lipinski_Pass"] == True) &
            ((df_final["hERG_Risk"] == False) | (df_final["hERG_Risk"].isna()))
        ].copy()
    else:
        df_herg_risk_true = pd.DataFrame(columns=df_final.columns)
        df_lipinski_herg_pass = df_lipinski_pass.copy()

    # 保存完整结果
    # === Active_Set (ADMET Pass) ===
    # 作为后续 Step4B/5A/5B 的统一起点：只要是 ADMET 通过，就标记为 True。
    # 规则尽量与本脚本实际过滤口径一致：
    #   Active_Set = Lipinski_Pass == True 且 ((hERG_Risk == False) 或缺失)
    # 若只有 hERG_Prob，则使用 HERG_THRESHOLD 判定。
    if "Lipinski_Pass" in df_final.columns:
        lip_ok = (df_final["Lipinski_Pass"] == True)
    else:
        lip_ok = pd.Series([True] * len(df_final))

    if "hERG_Risk" in df_final.columns:
        herg_ok = (df_final["hERG_Risk"].isna()) | (df_final["hERG_Risk"] == False)
    elif "hERG_Prob" in df_final.columns:
        herg_ok = (df_final["hERG_Prob"].isna()) | (df_final["hERG_Prob"] < HERG_THRESHOLD)
    else:
        herg_ok = pd.Series([True] * len(df_final))

    df_final["Active_Set"] = (lip_ok & herg_ok)
    try:
        print(f"✅ Active_Set(ADMET Pass) count = {int(df_final['Active_Set'].sum())}")
    except Exception:
        pass

    df_final.to_csv(out_path, index=False)
    
    print("-" * 30)
    print(f"✅ ADMET 评估完成！结果已保存至: {out_path}")
    print(f"📊 原始分子数: {len(df)}")
    print(f"💊 符合 Lipinski 规则的分子数: {len(df_lipinski_pass)}")

    if herg_model is not None:
        print(f"❤️ 预测为 hERG 高风险的分子数: {len(df_herg_risk_true)}")
        print(f"🛡️ 同时通过 Lipinski + hERG 筛查的分子数: {len(df_lipinski_herg_pass)}")
    else:
        print("⚠️ 未启用 hERG 预测，仅进行了 Lipinski 规则筛查。")

    print("-" * 30)
    
    # 展示 Top 5 最终候选分子（按 R_global 优先，其次 R_total）
    if not df_lipinski_herg_pass.empty:
        df_for_top = df_lipinski_herg_pass.copy()

        sort_col = None
        if "R_global" in df_for_top.columns:
            sort_col = "R_global"
        elif "R_total" in df_for_top.columns:
            sort_col = "R_total"

        if sort_col is not None:
            df_for_top = df_for_top.sort_values(sort_col, ascending=False)

        cols_to_show = ["smiles"]
        for col in ["pIC50", "gap_ev", "LogP", "QED", "R_total", "R_ADMET", "R_global", "Lipinski_Pass", "hERG_Prob"]:
            if col in df_for_top.columns:
                cols_to_show.append(col)

        print("🏆 最终 Top 5 候选分子 (综合活性 + 电子稳定 + 成药性/安全性):")
        print(df_for_top[cols_to_show].head(5))


if __name__ == "__main__":
    main()