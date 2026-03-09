# 文件: core/step1_vae.py
# -*- coding: utf-8 -*-
"""
Step 1: FRATTVAE 生成器封装
- 提供三个关键接口给 Step 3 使用:
  1) latent_dim: 潜空间维度
  2) sample_latent(n): 采样 n 个 latent 向量
  3) decode_from_latent(latent): 从 latent 解码为 SMILES 列表
"""

import sys
import os
import torch
import torch.nn as nn
import yaml
import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# --- 1. 路径与导入 ---

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, 'frattvae_source')
if source_dir not in sys.path:
    sys.path.append(source_dir)

try:
    from models.frattvae import FRATTVAE
except ImportError:
    print("【错误】无法导入 FRATTVAE，请检查 core/frattvae_source 路径是否正确。")
    raise


class VAE_Generator(nn.Module):
    """
    FRATTVAE 外层封装:
    - 负责加载配置、片段词表、片段指纹、模型权重
    - 提供 sample / decode_from_latent / sample_latent 等接口
    """
    def __init__(self, result_dir="../results/pretrained_model", device="cpu"):
        super().__init__()

        # 结果目录（包含 input_data / model 等）
        # 默认相对于 core 目录
        default_dir = os.path.join(os.path.dirname(current_dir), "results", "pretrained_model")
        if os.path.exists(os.path.abspath(os.path.join(current_dir, result_dir))):
            self.result_dir = os.path.abspath(os.path.join(current_dir, result_dir))
        elif os.path.exists(default_dir):
            self.result_dir = default_dir
        else:
            # 兜底：以当前工作目录为根
            self.result_dir = os.path.join(os.getcwd(), "results", "pretrained_model")

        self.device = torch.device(device)

        # 1. 加载配置
        config_path = os.path.join(self.result_dir, "input_data/params.yml")
        self.params = self._load_yaml(config_path)

        # 2. 加载片段词表
        frag_path = os.path.join(self.result_dir, "input_data/fragments.csv")
        self.vocab_list = self._load_fragments(frag_path)
        num_tokens = len(self.vocab_list)
        print(f"【信息】加载片段词表成功，大小: {num_tokens}")

        # 3. 抽取模型和分解参数
        model_cfg = self.params.get('model', {})
        decomp_cfg = self.params.get('decomp', {})

        # 4. 计算片段特征 (ECFP + dummy 计数)
        print("【初始化】正在计算片段的化学特征 (ECFP 指纹)...")
        self.frag_ecfps, self.ndummys = self._compute_fragment_features(
            self.vocab_list,
            radius=decomp_cfg.get('radius', 2),
            nbits=decomp_cfg.get('n_bits', 2048),
        )

        # 5. 初始化 FRATTVAE 模型
        print("【初始化】构建 FRATTVAE 模型结构...")
        self.model = FRATTVAE(
            num_tokens=num_tokens,
            depth=decomp_cfg.get('max_depth', 16),
            width=decomp_cfg.get('max_degree', 8),
            feat_dim=model_cfg.get('feat', 2048),
            latent_dim=model_cfg.get('latent', 256),
            d_model=model_cfg.get('d_model', 512),
            d_ff=model_cfg.get('d_ff', 2048),
            num_layers=model_cfg.get('nlayer', 6),
            nhead=model_cfg.get('nhead', 8),
            n_jobs=1,
        )

        self.model.set_labels(np.array(self.vocab_list))
        self.model.to(self.device)

        # 6. 默认不自动加载权重，交给调用方显式调用 load_weights()

    # ---------------- 内部工具函数 ----------------

    def _load_yaml(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        print(f"【警告】找不到配置文件 {path}")
        return {}

    def _load_fragments(self, path):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'smiles' in df.columns:
                    return df['smiles'].tolist()
                return df.iloc[:, 0].tolist()
            except Exception as e:
                print(f"【警告】读取片段文件失败 {path}: {e}")
        print("【警告】使用默认片段词表占位")
        return ["<pad>", "C", "CC"]

    def _compute_fragment_features(self, vocab_list, radius=2, nbits=2048):
        """
        计算每个片段的:
        - ECFP 指纹 (nbits 维)
        - dummy '*' 原子数量 (连接点数)
        """
        ecfps = []
        ndummys = []

        for smi in vocab_list:
            # 1. 统计 dummy 原子数量
            ndummys.append(smi.count('*'))

            # 2. 计算 ECFP 指纹
            if smi in ['<pad>', '<start>', '<end>']:
                arr = np.zeros((nbits,), dtype=np.int8)
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
                    arr = np.zeros((nbits,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                else:
                    arr = np.zeros((nbits,), dtype=np.int8)
            ecfps.append(arr[None, :])  # [1, nbits]

        frag_ecfps = torch.FloatTensor(np.vstack(ecfps))  # [V, nbits]
        ndummys = torch.LongTensor(ndummys)
        return frag_ecfps, ndummys

    # ---------------- 权重加载 & 采样接口 ----------------

    def load_weights(self):
        weight_path = os.path.join(self.result_dir, "model/model_best.pth")
        if os.path.exists(weight_path):
            print(f"【加载权重】{weight_path}")
            state_dict = torch.load(weight_path, map_location=self.device)
            new_state = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state[name] = v

            try:
                self.model.load_state_dict(new_state, strict=True)
                print("✅ 权重加载成功 (strict=True)")
            except RuntimeError as e:
                print(f"⚠️ 非严格模式加载: {e}")
                self.model.load_state_dict(new_state, strict=False)
                print("✅ 权重加载成功 (strict=False)")
        else:
            print(f"❌ 未找到权重文件: {weight_path}")

    @property
    def latent_dim(self) -> int:
        """暴露 latent 维度，供 Step 3 使用"""
        return int(self.model.latent_dim)

    def sample_latent(self, num_samples: int = 1) -> torch.Tensor:
        """从标准正态分布采样 latent 向量"""
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return z

    def decode_from_latent(self, latent: torch.Tensor, max_nfrags: int = 16):
        """
        从 latent 解码为 SMILES 列表
        latent: [N, latent_dim]
        返回: 长度为 N 的 SMILES 列表
        """
        self.model.eval()
        latent = latent.to(self.device)
        with torch.no_grad():
            outputs = self.model.sequential_decode(
                latent,
                self.frag_ecfps.to(self.device),
                self.ndummys.to(self.device),
                asSmiles=True,
                max_nfrags=max_nfrags,
            )
        # FRATTVAE 一般返回 list[str]
        return outputs

    def sample(self, num_samples=3):
        """
        兼容之前的接口: 直接在潜空间采样并解码
        """
        print(f"【生成】正在从潜空间采样并解码 {num_samples} 个分子...")
        z = self.sample_latent(num_samples)
        return self.decode_from_latent(z)


# --- 测试代码 ---
if __name__ == "__main__":
    gen = VAE_Generator()
    gen.load_weights()

    print("\n>>> 见证奇迹的时刻：")
    mols = gen.sample(3)
    for i, m in enumerate(mols):
        print(f"[{i+1}] {m}")
