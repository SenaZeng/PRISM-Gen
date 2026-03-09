# 文件: core/step2b_train_herg_model.py
import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pandas as pd
import joblib

RADIUS = 2
N_BITS = 2048
N_ESTIMATORS = 300
RANDOM_STATE = 42

def smiles_to_fp(smi, radius=RADIUS, n_bits=N_BITS):
    """将 SMILES 转为 Morgan 指纹(bit向量)。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((n_bits,), dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 读取本地导出的 hERG 数据
    data_path = os.path.join(current_dir, "../data/herg_tdc_full.csv")
    if not os.path.exists(data_path):
        print(f"❌ 找不到 hERG 数据文件: {data_path}")
        print("请确认已从有网环境导出 herg_tdc_full.csv 并放到 data/ 目录。")
        return

    print(f"📥 读取 hERG 数据: {data_path}")
    df = pd.read_csv(data_path)

    # 期待列名为 'Drug' 和 'Y'
    if "Drug" not in df.columns or "Y" not in df.columns:
        print("❌ 数据格式不符合预期，应包含 'Drug' 和 'Y' 两列。")
        print("实际列名:", df.columns.tolist())
        return

    smiles_list = df["Drug"].astype(str).tolist()
    labels = df["Y"].values

    print(f"数据条数: {len(df)}")

    # 2. 划分 train/test
    X_train_smi, X_test_smi, y_train, y_test = train_test_split(
        smiles_list, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    # 3. 特征编码
    print("🔬 特征编码 (Morgan 指纹)...")
    X_train = np.array([smiles_to_fp(s) for s in X_train_smi])
    X_test = np.array([smiles_to_fp(s) for s in X_test_smi])

    # 4. 训练 RandomForest 分类器
    print("🌲 训练 RandomForest hERG 分类模型...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    X_tr, y_tr = shuffle(X_train, y_train, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)

    # 5. 在测试集上评估
    print("📊 在测试集上评估模型表现...")
    y_proba_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba_test)
    auprc = average_precision_score(y_test, y_proba_test)

    print(f"✅ hERG RF 模型 AUROC: {auc:.3f}")
    print(f"✅ hERG RF 模型 AUPRC: {auprc:.3f}")

    # 6. 保存模型到 ../results/herg_model/herg_rf_model.pkl
    model_dir = os.path.join(current_dir, "../results/herg_model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "herg_rf_model.pkl")

    joblib.dump(clf, model_path)
    print(f"💾 模型已保存到: {model_path}")
    print("🎉 hERG 模型训练完成！")

if __name__ == "__main__":
    main()
