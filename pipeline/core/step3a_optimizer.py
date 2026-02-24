# 文件: core/step3a_optimizer.py
# -*- coding: utf-8 -*-
"""
Step 3a: 基于潜空间爬山 (Latent Space Hill Climbing) 的分子优化闭环 + 去重 + top_k 导出

- 生成器: VAE_Generator (FRATTVAE, Step 1)
- 评分器: SurrogateModel (Uni-Mol 代理, Step 2)
- 策略:
    在 FRATTVAE 的 latent space 中加噪声 (z -> z + noise)，
    解码为 SMILES，调用 SurrogateModel + QED + SA 形成复合 Reward，
    如果 Reward 提升就接受，形成 hill-climbing 轨迹。
- 优化 (v2):
    引入分子量 (MW) 软约束奖励，鼓励生成分子量在目标区间 (默认 320~520) 的分子，
    解决生成分子过小的问题。
- 输出:
    1) 所有被接受的分子轨迹：step3a_optimized_molecules_raw.csv
    2) 去重后按 Reward 排序的前 top_k 分子：
       - step3a_top{top_k}.csv
       - step3a_optimized_molecules.csv  (供后续 DFT 使用)
       
--调用：
unset LD_LIBRARY_PATH
python core/step3a_optimizer.py  --steps 50  --step_size 0.5  --n_restarts 10 --top_k 50 --n_jobs 40 --mw_weight 1.0
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import QED, Descriptors, RDConfig

# 为了避免每个线程再开很多 BLAS 线程，限制为 1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 添加 core 路径以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 Step 1 和 Step 2 的模型
try:
    from core.step1_vae import VAE_Generator
    from core.step2_surrogate import SurrogateModel
except ImportError:
    # 尝试直接从当前目录导入（如果脚本直接在 core/ 下运行）
    try:
        from step1_vae import VAE_Generator
        from step2_surrogate import SurrogateModel
    except ImportError as e:
        print(f"❌ 无法导入 VAE_Generator 或 SurrogateModel: {e}")
        sys.exit(1)

# SA Score 路径 (RDKit Contrib)
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
except ImportError:
    print("⚠️ 无法导入 sascorer，将使用默认 SA=0")
    sascorer = None


def calc_sa_score(mol):
    if sascorer is None:
        return 0.0
    try:
        return sascorer.calculateScore(mol)
    except:
        return 10.0

def calc_mw_bonus(mw, target_min, target_max, sigma=50.0):
    """
    计算分子量奖励:
    - 在 [target_min, target_max] 区间内: bonus = 1.0
    - 在区间外: 高斯衰减 exp(-0.5 * ((delta)/sigma)^2)
    """
    if target_min <= mw <= target_max:
        return 1.0
    elif mw < target_min:
        return np.exp(-0.5 * ((target_min - mw) / sigma) ** 2)
    else:  # mw > target_max
        return np.exp(-0.5 * ((mw - target_max) / sigma) ** 2)


class MoleculeOptimizer:
    def __init__(self, results_dir="../results", device="cpu", 
                 mw_weight=1.0, mw_min=320.0, mw_max=520.0, mw_sigma=50.0):
        print(">>> [系统启动] 正在初始化优化闭环...")

        # 为每个进程/线程限制 torch 内部线程数，避免过度并行
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        self.device = torch.device(device)

        # --- Step1: 初始化生成器 ---
        print(">>> 加载 VAE_Generator (FRATTVAE)...")
        self.generator = VAE_Generator()
        self.generator.load_weights()
        self.generator.model.eval()
        print("✅ 生成器 (FRATTVAE) 就绪")

        # --- Step2: 初始化 Surrogate 代理 ---
        print(">>> 加载 SurrogateModel (Uni-Mol 代理)...")
        self.scorer = SurrogateModel()
        print("✅ 评价器 (Uni-Mol 代理) 就绪")

        # --- MW (molecular weight) bonus settings ---
        # 这里的 mw_weight 现在可以从函数参数中读取到了
        self.mw_weight = float(mw_weight)
        self.mw_min = float(mw_min)
        self.mw_max = float(mw_max)
        self.mw_sigma = float(mw_sigma)

        if self.mw_weight != 0.0:
            print(
                f">>> MW 奖励启用: weight={self.mw_weight} | "
                f"target_range=[{self.mw_min}, {self.mw_max}] | sigma={self.mw_sigma}"
            )
        else:
            print(">>> MW 奖励关闭 (mw_weight=0.0)，行为与旧版一致")

        # 输出目录
        self.results_dir = os.path.abspath(os.path.join(current_dir, results_dir))
        os.makedirs(self.results_dir, exist_ok=True)

        # 全局最佳记录
        self.best_molecule = None
        self.best_score = -1e9
        self.best_info = {}

        # 轨迹记录（所有接受的点）
        self.history = []

        # 全局去重集合 + 候选列表（只记录每个新 SMILES 最好的一次）
        self.seen_smiles = set()
        self.candidates = []

    # -------------- Reward 计算 -------------- #

    def get_composite_reward(self, smiles: str):
        """
        复合奖励:
        Reward = pIC50 + 0.5 * QED - 0.1 * SA_score + mw_weight * MW_bonus

        说明：
        - MW_bonus ∈ [0,1]，鼓励分子量落在 [mw_min, mw_max]（默认 320~520）
        - 这样可以缓解生成分子普遍偏小（例如 MW≈270）的现象
        - 若想完全复现旧版行为，将 --mw_weight 设为 0
        """
        if smiles is None or len(smiles) == 0:
            return -10.0, {}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -10.0, {}
        
        # --- [新增] 计算 LogP 以评估 hERG 风险 ---
        logp = Descriptors.MolLogP(mol)
        # 设置 LogP 惩罚阈值（通常 LogP > 4.5 会显著增加 hERG 风险）
        logp_penalty = 0.0
        if logp > 4.5:
            # 每超过 1 个单位，奖励分扣除 1 分
            logp_penalty = (logp - 4.5) * 1.0        

        # 1. pIC50 预测
        try:
            pIC50 = float(self.scorer.predict_single(smiles))
        except Exception as e:
            print(f"⚠️ 代理模型预测失败: {e}")
            pIC50 = 0.0

        # 2. QED
        try:
            qed = float(QED.qed(mol))
        except Exception:
            qed = 0.0

        # 3. SA_score
        sa = calc_sa_score(mol)

        # 4. MW bonus (鼓励分子量不要过小)
        mw = float(Descriptors.MolWt(mol))
        mw_bonus = calc_mw_bonus(mw, self.mw_min, self.mw_max, self.mw_sigma)

        # reward = pIC50 + 0.5 * qed - 0.1 * sa + self.mw_weight * mw_bonus
        # --- [改进] 复合奖励公式：减去 LogP 惩罚 ---
        # 这种做法会引导 VAE 生成结构更紧凑、水溶性更好的分子
        reward = pIC50 + 0.5 * qed - 0.1 * sa + self.mw_weight * mw_bonus - logp_penalty        

        info = {
            "pIC50": pIC50,
            "QED": qed,
            "SA": sa,
            "MW": mw,
            "LogP": logp,  # 记录在日志中以便观察
            "LogP_Penalty": logp_penalty,
            "Reward": reward,
        }
        return reward, info

    # -------------- 解码接口封装 -------------- #

    def decode_from_latent(self, z):
        """将 latent vector 解码为 SMILES"""
        try:
            # VAE Generator 需要 Tensor 输入
            z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)
            # 假设 generator.decode 返回的是 list of smiles
            smiles_list = self.generator.decode_from_latent(z_tensor)
            if smiles_list and len(smiles_list) > 0:
                return smiles_list[0]
            return None
        except Exception as e:
            # print(f"Decode error: {e}")
            return None

    def get_random_latent(self):
        """获取一个随机的 latent vector (从先验分布采样)"""
        # 假设 latent dim = 256 (FRATTVAE default)
        dim = 256
        return np.random.normal(0, 1, (1, dim))

    # -------------- 核心: 爬山算法 (单次重启) -------------- #

    def run_single_restart(self, restart_idx, steps=50, step_size=0.5, T=1.0):
        """
        运行单次随机重启的 Hill Climbing
        """
        # 1. 随机初始化
        current_latent = self.get_random_latent()
        current_smiles = self.decode_from_latent(current_latent)
        
        if current_smiles is None:
            return []  # 初始化失败

        current_score, cinfo = self.get_composite_reward(current_smiles)
        
        # 记录轨迹
        trajectory = []
        
        # 初始点记录
        entry = {
            "restart": restart_idx,
            "step": 0,
            "smiles": current_smiles,
            **cinfo
        }
        trajectory.append(entry)

        # 2. 迭代优化
        accepted = 0
        proposed = 0

        for step_idx in range(1, steps + 1):
            # 2.1 扰动 latent (加高斯噪声)
            noise = np.random.normal(0, step_size, current_latent.shape)
            candidate_latent = current_latent + noise
            
            # 为了提高效率，可以批量采样几次，选有效的 (这里简化为单次)
            # 实际生产中可以一次 decode 多个近邻
            
            # 尝试多次解码以获得有效分子
            best_candidate_smiles = None
            best_candidate_score = -1e9
            
            batch_proposals = 5 # 每次扰动尝试解码 5 次（微小变动或不同采样）
            
            for _ in range(batch_proposals):
                # 微调噪声或重新采样
                noise_i = np.random.normal(0, step_size, current_latent.shape)
                cand_lat_i = current_latent + noise_i
                cand_smi = self.decode_from_latent(cand_lat_i)
                
                if cand_smi and Chem.MolFromSmiles(cand_smi):
                    candidate_score, cinfo = self.get_composite_reward(cand_smi)
                    if candidate_score > best_candidate_score:
                        best_candidate_score = candidate_score
                        best_candidate_latent = cand_lat_i
                        best_candidate_smiles = cand_smi
                        best_candidate_info = cinfo

            if best_candidate_smiles is None:
                # print(f"[Restart {restart_idx} | Step {step_idx}] 本批次解码无效，跳过")
                continue

            proposed += 1
            delta = best_candidate_score - current_score

            # 模拟退火接受准则 (或简单的贪婪/Metropolis)
            accept = False
            if delta >= 0:
                accept = True
            else:
                # Metropolis
                if T > 1e-8:
                    prob = float(np.exp(delta / T))
                    prob = max(min(prob, 1.0), 0.0)
                else:
                    prob = 0.0
                u = float(np.random.rand())
                accept = (u < prob)

            if accept:
                accepted += 1
                current_latent = best_candidate_latent
                current_score = best_candidate_score
                current_smiles = best_candidate_smiles
                
                # 记录
                entry = {
                    "restart": restart_idx,
                    "step": step_idx,
                    "smiles": current_smiles,
                    **best_candidate_info
                }
                trajectory.append(entry)
                
                # 简单的温度衰减
                T *= 0.95
            
        return trajectory

    # -------------- 并行优化主入口 -------------- #

    def hill_climbing(self, steps=50, step_size=0.5, n_restarts=10, top_k=200, n_jobs=4):
        print(f"\n🚀 开始并行优化: Restarts={n_restarts} | Steps={steps} | Jobs={n_jobs} | MW_Weight={self.mw_weight}")
        
        start_time = time.time()
        all_results = []

        # 使用 ThreadPoolExecutor 进行并行
        # 注意：由于 Python GIL，如果是计算密集型，ProcessPool 可能更好
        # 但这里涉及模型加载 (CUDA/Torch)，多进程需小心 context，Thread 更稳妥且只要 I/O 或 C++ call 够多就行
        # rdkit 和 pytorch 内部通常会释放 GIL
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self.run_single_restart, i, steps, step_size) 
                for i in range(n_restarts)
            ]
            
            for future in as_completed(futures):
                try:
                    traj = future.result()
                    if traj:
                        all_results.extend(traj)
                except Exception as e:
                    print(f"❌ 某个 restart 发生异常: {e}")

        # 汇总结果
        print(f"\n>>> 优化完成，共收集到 {len(all_results)} 个轨迹点。")
        
        # 存 Raw Data
        df_raw = pd.DataFrame(all_results)
        raw_path = os.path.join(self.results_dir, "step3a_optimized_molecules_raw.csv")
        df_raw.to_csv(raw_path, index=False)
        print(f"📁 原始轨迹已保存至: {raw_path}")

        # 去重并选 Top K
        if not df_raw.empty:
            # 按 SMILES 去重，保留 Reward 最高的那个
            df_unique = df_raw.sort_values("Reward", ascending=False).drop_duplicates("smiles").copy()
            
            # 取 Top K
            df_top = df_unique.head(top_k)
            
            # 保存 Top K (带详细信息)
            top_path = os.path.join(self.results_dir, f"step3a_top{top_k}.csv")
            df_top.to_csv(top_path, index=False)
            print(f"📁 去重后按 Reward 排序的前 {top_k} 个分子已保存至: {top_path}")

            # 保存标准输入格式供下一步 (Step 3b/3c) 使用
            # 只需要 smiles 列 (或者 id, smiles)
            # 为了方便后续，我们保留 smiles 和 Reward
            canonical_path = os.path.join(self.results_dir, "step3a_optimized_molecules.csv")
            df_top.to_csv(canonical_path, index=False)
            print(f"📁 标准候选文件已保存至: {canonical_path}")

            # 打印最佳的一个
            best = df_top.iloc[0]
            print(f"\n🏆 全局最佳分子: {best['smiles']}")
            print(f"🏆 最佳综合得分: Reward={best['Reward']:.4f} | pIC50={best['pIC50']:.2f}, MW={best.get('MW', 0):.1f}")
            
        else:
            print("⚠️ 未生成任何有效分子！请检查 VAE 或 Surrogate 模型。")

        print(f"⏱️ 总耗时: {time.time() - start_time:.1f} 秒")


def main():
    parser = argparse.ArgumentParser(description="Step3a: 潜空间爬山 + 去重 + top_k 导出")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="结果输出目录（默认: ../results）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="运行设备（默认: cpu）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="每次重启的 hill-climbing 步数",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.5,
        help="潜空间扰动的步长",
    )
    parser.add_argument(
        "--n_restarts",
        type=int,
        default=10,
        help="随机重启次数",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="最终保留的候选数量（按 Reward 排序）",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="并行重启的线程数（建议 <= CPU 核数，例如 40）",
    )
    
    # === 新增以下 MW 相关参数 ===
    parser.add_argument("--mw_weight", type=float, default=1.0, help="分子量奖励权重 (0=关闭)")
    parser.add_argument("--mw_min", type=float, default=320.0, help="目标分子量下限")
    parser.add_argument("--mw_max", type=float, default=520.0, help="目标分子量上限")
    parser.add_argument("--mw_sigma", type=float, default=50.0, help="高斯奖励的宽度 sigma")

    args = parser.parse_args()

    optimizer = MoleculeOptimizer(
        results_dir=args.results_dir, 
        device=args.device, 
        mw_weight=args.mw_weight, 
        mw_min=args.mw_min, 
        mw_max=args.mw_max, 
        mw_sigma=args.mw_sigma
    )
    
    optimizer.hill_climbing(
        steps=args.steps,
        step_size=args.step_size,
        n_restarts=args.n_restarts,
        top_k=args.top_k,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()