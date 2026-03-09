# 文件: core/step2_surrogate.py
# -*- coding: utf-8 -*-
"""
Step 2: Uni-Mol 代理模型封装
- 使用 UniMolRepr 将 SMILES 编码为 CLS embedding
- 使用 RandomForestRegressor 做活性回归 (pIC50)
- 暴露 SurrogateModel.predict(smiles_list) / predict_single(smiles)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. 路径挂载 unimol_source ---

current_dir = os.path.dirname(os.path.abspath(__file__))
unimol_path = os.path.join(current_dir, 'unimol_source')
if unimol_path not in sys.path:
    sys.path.append(unimol_path)

try:
    # 调用 Uni-Mol 的官方接口
    from unimol_tools import UniMolRepr
except ImportError:
    print("【错误】无法导入 unimol_tools，请检查 core/unimol_source 是否存在且可用。")
    raise


class SurrogateModel:
    """
    基于 Uni-Mol 表征 + sklearn RandomForest 的代理模型
    - 用于快速预测 Mpro 抑制活性 (pIC50)
    """
    def __init__(self, work_dir="../results/surrogate_model"):
        # 工作目录，用于保存 sklearn 模型
        self.work_dir = os.path.abspath(os.path.join(current_dir, work_dir))
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)

        self.sklearn_model_path = os.path.join(self.work_dir, "activity_predictor.pkl")

        # 初始化 Uni-Mol 特征提取器
        print("【初始化】加载 Uni-Mol 预训练模型 (代理表征)...")
        # data_type='molecule' 表示处理小分子；remove_hs=True 对应 mol_pre_no_h 预训练
        self.clf = UniMolRepr(data_type='molecule', remove_hs=True)
        self.model = None

    # ------------- 内部: Uni-Mol 表征提取 ------------- #

    def _get_embeddings(self, smiles_list):
        """
        调用 Uni-Mol 将 SMILES 转为高维向量 (N, 512)
        - UniMolRepr.get_repr: 会自动完成 3D 构象生成 + Transformer 前向
        - 返回 dict, 其中 'cls_repr' 是 [N, 512]
        """
        print(f"【特征提取】正在处理 {len(smiles_list)} 个分子 (自动生成 3D 构象)...")
        reprs = self.clf.get_repr(smiles_list, return_atomic_reprs=False)
        # reprs['cls_repr'] 是一个 tensor-like/array-like，转成 numpy
        X = np.array(reprs['cls_repr'])
        return X

    # ------------- 训练流程 ------------- #

    def train(self, data_path):
        """
        训练流程：
        - 读取 CSV
        - 提取 Uni-Mol CLS embedding
        - 训练 RandomForest 回归模型预测 pIC50
        """
        print(f"【训练】读取数据: {data_path}")
        df = pd.read_csv(data_path)

        if 'pchembl_value' in df.columns:
            y = df['pchembl_value'].values
        elif 'Standard Value' in df.columns:
            vals = pd.to_numeric(df['Standard Value'], errors='coerce')
            vals = np.where(vals <= 0, 1e-9, vals)
            y = 9 - np.log10(vals)  # pIC50 = 9 - log10(nM)
        else:
            raise ValueError("CSV 中找不到活性列 (pchembl_value 或 Standard Value)")

        X_smiles = df['smiles'].astype(str).tolist()

        # 1. 抽取 Uni-Mol 表征
        X_emb = self._get_embeddings(X_smiles)

        # 2. 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_emb, y, test_size=0.2, random_state=42
        )

        # 3. 训练 RandomForest 回归
        print("【训练】开始训练下游回归模型 (RandomForestRegressor)...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        # 4. 简单评估
        score = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        print(f"【评估】R2 Score: {score:.4f}, RMSE: {rmse:.4f}")

        # 5. 保存 sklearn 模型
        joblib.dump(self.model, self.sklearn_model_path)
        print(f"【保存】模型已保存至 {self.sklearn_model_path}")

    # ------------- 加载与预测接口 ------------- #

    def load_model(self):
        """加载训练好的 sklearn 模型"""
        if os.path.exists(self.sklearn_model_path):
            self.model = joblib.load(self.sklearn_model_path)
            print(f"✅ 代理模型加载成功: {self.sklearn_model_path}")
        else:
            print("⚠️ 未找到代理模型，请先运行 train()")

    def predict(self, smiles_list):
        """
        预测接口：
        - 输入: 一组 SMILES 字符串
        - 输出: 对应的 pIC50 预测值 (list[float])
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                return [0.0] * len(smiles_list)

        X_emb = self._get_embeddings(smiles_list)
        preds = self.model.predict(X_emb)
        return preds

    def predict_pIC50(self, smiles_list):
        """语义更明确的别名：预测 pIC50"""
        return self.predict(smiles_list)

    def predict_single(self, smiles: str) -> float:
        """方便 Step 3 调用的单个预测接口"""
        return float(self.predict([smiles])[0])


# --- 测试 / 训练脚本 --- #

if __name__ == "__main__":
    agent = SurrogateModel()

    # 使用基于当前脚本位置的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.abspath(
        os.path.join(script_dir, "../data/processed/mpro_actives_clean.csv")
    )

    print(f"数据路径锁定: {data_file}")

    # 如果没有训练过，就训练一次
    if not os.path.exists(agent.sklearn_model_path):
        if os.path.exists(data_file):
            print(">>> 首次运行，开始训练代理模型...")
            agent.train(data_file)
        else:
            print(
                f"❌ 错误：找不到训练数据！\n"
                f"请检查 {data_file} 是否存在。\n"
                f"提示：请先运行 'python tools/data_cleaner.py'"
            )

    # 简单预测测试
    test_mols = [
        "CC(=O)Nc1ccc(O)cc1",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]
    print("\n>>> 测试预测 (Mpro 活性 pIC50):")
    scores = agent.predict(test_mols)
    for smi, score in zip(test_mols, scores):
        print(f"SMILES: {smi[:30]}... -> pIC50 预测: {score:.2f}")
