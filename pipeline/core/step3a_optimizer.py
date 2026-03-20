# File: core/step3a_optimizer.py
# -*- coding: utf-8 -*-
"""
Step 3A: Latent-space hill climbing for molecule optimization with deduplication and top_k export

- Generator:  VAE_Generator (FRATTVAE, Step 1)
- Scorer:     SurrogateModel (Uni-Mol surrogate, Step 2)
- Strategy:
    Perturb latent vectors in FRATTVAE's latent space (z -> z + noise),
    decode to SMILES, evaluate with SurrogateModel + QED + SA to form a
    composite Reward, and accept if Reward improves (hill-climbing trajectory).
- Optimization (v2):
    Introduces a molecular weight (MW) soft-constraint bonus that encourages
    molecules within a target MW range (default 320-520 Da), addressing the
    tendency of the generator to produce overly small molecules.
- Outputs:
    1) All accepted trajectory points: step3a_optimized_molecules_raw.csv
    2) Deduplicated, Reward-sorted top_k molecules:
       - step3a_top{top_k}.csv
       - step3a_optimized_molecules.csv  (canonical input for downstream DFT steps)

-- Usage:
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

# Limit per-thread BLAS thread count to avoid thread over-subscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Add core/ to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Step 1 and Step 2 models
try:
    from core.step1_vae import VAE_Generator
    from core.step2_surrogate import SurrogateModel
except ImportError:
    # Fallback: try importing directly if the script is run from inside core/
    try:
        from step1_vae import VAE_Generator
        from step2_surrogate import SurrogateModel
    except ImportError as e:
        print(f"❌ Cannot import VAE_Generator or SurrogateModel: {e}")
        sys.exit(1)

# SA Score path (RDKit Contrib)
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
except ImportError:
    print("⚠️ Cannot import sascorer; SA score will default to 0")
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
    Molecular weight bonus:
    - Returns 1.0 if MW is within [target_min, target_max]
    - Outside the range: Gaussian decay exp(-0.5 * ((delta)/sigma)^2)
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
        print(">>> [Startup] Initializing optimization loop...")

        # Limit torch internal thread count per process/thread to avoid over-subscription
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        self.device = torch.device(device)

        # --- Step 1: Initialize generator ---
        print(">>> Loading VAE_Generator (FRATTVAE)...")
        self.generator = VAE_Generator()
        self.generator.load_weights()
        self.generator.model.eval()
        print("✅ Generator (FRATTVAE) ready")

        # --- Step 2: Initialize surrogate scorer ---
        print(">>> Loading SurrogateModel (Uni-Mol surrogate)...")
        self.scorer = SurrogateModel()
        print("✅ Scorer (Uni-Mol surrogate) ready")

        # --- Molecular weight bonus settings ---
        self.mw_weight = float(mw_weight)
        self.mw_min = float(mw_min)
        self.mw_max = float(mw_max)
        self.mw_sigma = float(mw_sigma)

        if self.mw_weight != 0.0:
            print(
                f">>> MW bonus enabled: weight={self.mw_weight} | "
                f"target_range=[{self.mw_min}, {self.mw_max}] | sigma={self.mw_sigma}"
            )
        else:
            print(">>> MW bonus disabled (mw_weight=0.0); behavior matches original version")

        # Output directory
        self.results_dir = os.path.abspath(os.path.join(current_dir, results_dir))
        os.makedirs(self.results_dir, exist_ok=True)

        # Global best record
        self.best_molecule = None
        self.best_score = -1e9
        self.best_info = {}

        # Trajectory record (all accepted points)
        self.history = []

        # Global deduplication set + candidate list (one entry per unique SMILES, best Reward kept)
        self.seen_smiles = set()
        self.candidates = []

    # -------------- Reward computation -------------- #

    def get_composite_reward(self, smiles: str):
        """
        Composite reward:
        Reward = pIC50 + 0.5 * QED - 0.1 * SA_score + mw_weight * MW_bonus - logp_penalty

        Notes:
        - MW_bonus in [0, 1]: encourages MW within [mw_min, mw_max] (default 320-520 Da)
          to counteract the generator's tendency to produce small molecules (~270 Da)
        - logp_penalty: penalizes LogP > 4.5 to reduce hERG liability risk
          (penalty = (LogP - 4.5) * 1.0 per unit above threshold)
        - To reproduce original behavior without MW bonus, set --mw_weight 0
        """
        if smiles is None or len(smiles) == 0:
            return -10.0, {}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -10.0, {}
        
        # --- LogP penalty to reduce hERG risk ---
        logp = Descriptors.MolLogP(mol)
        logp_penalty = 0.0
        if logp > 4.5:
            # Penalize by 1 point per unit above threshold
            logp_penalty = (logp - 4.5) * 1.0        

        # 1. pIC50 prediction
        try:
            pIC50 = float(self.scorer.predict_single(smiles))
        except Exception as e:
            print(f"⚠️ Surrogate model prediction failed: {e}")
            pIC50 = 0.0

        # 2. QED
        try:
            qed = float(QED.qed(mol))
        except Exception:
            qed = 0.0

        # 3. SA score
        sa = calc_sa_score(mol)

        # 4. MW bonus (discourages overly small molecules)
        mw = float(Descriptors.MolWt(mol))
        mw_bonus = calc_mw_bonus(mw, self.mw_min, self.mw_max, self.mw_sigma)

        # Composite reward formula with LogP penalty
        reward = pIC50 + 0.5 * qed - 0.1 * sa + self.mw_weight * mw_bonus - logp_penalty        

        info = {
            "pIC50": pIC50,
            "QED": qed,
            "SA": sa,
            "MW": mw,
            "LogP": logp,
            "LogP_Penalty": logp_penalty,
            "Reward": reward,
        }
        return reward, info

    # -------------- Decode interface -------------- #

    def decode_from_latent(self, z):
        """Decode a latent vector to a SMILES string."""
        try:
            # VAE_Generator expects a Tensor input
            z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)
            smiles_list = self.generator.decode_from_latent(z_tensor)
            if smiles_list and len(smiles_list) > 0:
                return smiles_list[0]
            return None
        except Exception as e:
            return None

    def get_random_latent(self):
        """Sample a random latent vector from the prior distribution."""
        # Latent dim = 256 (FRATTVAE default)
        dim = 256
        return np.random.normal(0, 1, (1, dim))

    # -------------- Core: hill climbing (single restart) -------------- #

    def run_single_restart(self, restart_idx, steps=50, step_size=0.5, T=1.0):
        """
        Run one random-restart hill climbing trajectory.
        """
        # 1. Random initialization
        current_latent = self.get_random_latent()
        current_smiles = self.decode_from_latent(current_latent)
        
        if current_smiles is None:
            return []  # Initialization failed

        current_score, cinfo = self.get_composite_reward(current_smiles)
        
        # Trajectory record
        trajectory = []
        
        # Record initial point
        entry = {
            "restart": restart_idx,
            "step": 0,
            "smiles": current_smiles,
            **cinfo
        }
        trajectory.append(entry)

        # 2. Iterative optimization
        accepted = 0
        proposed = 0

        for step_idx in range(1, steps + 1):
            # 2.1 Perturb latent (add Gaussian noise)
            noise = np.random.normal(0, step_size, current_latent.shape)
            candidate_latent = current_latent + noise
            
            # Try multiple decodes per perturbation to increase the chance of a valid molecule
            best_candidate_smiles = None
            best_candidate_score = -1e9
            
            batch_proposals = 5  # Attempt 5 decodes per perturbation step
            
            for _ in range(batch_proposals):
                # Resample noise slightly for each attempt
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
                continue

            proposed += 1
            delta = best_candidate_score - current_score

            # Simulated annealing acceptance criterion (Metropolis)
            accept = False
            if delta >= 0:
                accept = True
            else:
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
                
                # Record accepted point
                entry = {
                    "restart": restart_idx,
                    "step": step_idx,
                    "smiles": current_smiles,
                    **best_candidate_info
                }
                trajectory.append(entry)
                
                # Simple temperature decay
                T *= 0.95
            
        return trajectory

    # -------------- Parallel optimization main entry -------------- #

    def hill_climbing(self, steps=50, step_size=0.5, n_restarts=10, top_k=200, n_jobs=4):
        print(f"\n🚀 Starting parallel optimization: Restarts={n_restarts} | Steps={steps} | Jobs={n_jobs} | MW_Weight={self.mw_weight}")
        
        start_time = time.time()
        all_results = []

        # Use ThreadPoolExecutor for parallelism.
        # Note: for compute-bound work ProcessPool may be faster, but threading is safer
        # when sharing loaded models (CUDA/Torch). RDKit and PyTorch release the GIL
        # in their C++ internals, so threading provides effective parallelism here.
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
                    print(f"❌ Exception in a restart: {e}")

        # Aggregate results
        print(f"\n>>> Optimization complete. Collected {len(all_results)} trajectory points.")
        
        # Save raw trajectory
        df_raw = pd.DataFrame(all_results)
        raw_path = os.path.join(self.results_dir, "step3a_optimized_molecules_raw.csv")
        df_raw.to_csv(raw_path, index=False)
        print(f"📁 Raw trajectory saved to: {raw_path}")

        # Deduplicate and select top K
        if not df_raw.empty:
            # Deduplicate by SMILES, keeping the entry with the highest Reward
            df_unique = df_raw.sort_values("Reward", ascending=False).drop_duplicates("smiles").copy()
            
            # Select top K
            df_top = df_unique.head(top_k)
            
            # Save top K with full details
            top_path = os.path.join(self.results_dir, f"step3a_top{top_k}.csv")
            df_top.to_csv(top_path, index=False)
            print(f"📁 Deduplicated top {top_k} molecules (by Reward) saved to: {top_path}")

            # Save canonical input file for Step 3B/3C
            canonical_path = os.path.join(self.results_dir, "step3a_optimized_molecules.csv")
            df_top.to_csv(canonical_path, index=False)
            print(f"📁 Canonical candidate file saved to: {canonical_path}")

            # Report global best
            best = df_top.iloc[0]
            print(f"\n🏆 Global best molecule: {best['smiles']}")
            print(f"🏆 Best composite score: Reward={best['Reward']:.4f} | pIC50={best['pIC50']:.2f}, MW={best.get('MW', 0):.1f}")
            
        else:
            print("⚠️ No valid molecules generated. Check the VAE or surrogate model.")

        print(f"⏱️ Total elapsed time: {time.time() - start_time:.1f} s")


def main():
    parser = argparse.ArgumentParser(description="Step 3A: Latent-space hill climbing + deduplication + top_k export")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Output directory for results (default: ../results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of hill-climbing steps per restart",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.5,
        help="Perturbation step size in latent space",
    )
    parser.add_argument(
        "--n_restarts",
        type=int,
        default=10,
        help="Number of random restarts",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Number of top candidates to retain (sorted by Reward)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel restart threads (recommended: <= CPU core count, e.g. 40)",
    )
    
    # MW bonus parameters
    parser.add_argument("--mw_weight", type=float, default=1.0, help="MW bonus weight (0 = disabled)")
    parser.add_argument("--mw_min", type=float, default=320.0, help="Lower bound of target MW range")
    parser.add_argument("--mw_max", type=float, default=520.0, help="Upper bound of target MW range")
    parser.add_argument("--mw_sigma", type=float, default=50.0, help="Gaussian bonus width sigma")

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
