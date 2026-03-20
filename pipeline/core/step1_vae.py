# File: core/step1_vae.py
# -*- coding: utf-8 -*-
"""
Step 1: FRATTVAE generator wrapper
- Provides three key interfaces for Step 3:
  1) latent_dim: dimensionality of the latent space
  2) sample_latent(n): sample n latent vectors
  3) decode_from_latent(latent): decode latent vectors into a list of SMILES strings
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

# --- 1. Path setup and imports ---

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, 'frattvae_source')
if source_dir not in sys.path:
    sys.path.append(source_dir)

try:
    from models.frattvae import FRATTVAE
except ImportError:
    print("ERROR: Cannot import FRATTVAE. Check that core/frattvae_source path is correct.")
    raise


class VAE_Generator(nn.Module):
    """
    Wrapper around FRATTVAE:
    - Loads configuration, fragment vocabulary, fragment fingerprints, and model weights
    - Provides sample / decode_from_latent / sample_latent interfaces
    """
    def __init__(self, result_dir="../results/pretrained_model", device="cpu"):
        super().__init__()

        # Result directory (contains input_data/ and model/ subdirectories)
        # Default is relative to the core/ directory
        default_dir = os.path.join(os.path.dirname(current_dir), "results", "pretrained_model")
        if os.path.exists(os.path.abspath(os.path.join(current_dir, result_dir))):
            self.result_dir = os.path.abspath(os.path.join(current_dir, result_dir))
        elif os.path.exists(default_dir):
            self.result_dir = default_dir
        else:
            # Fallback: resolve relative to the current working directory
            self.result_dir = os.path.join(os.getcwd(), "results", "pretrained_model")

        self.device = torch.device(device)

        # 1. Load configuration
        config_path = os.path.join(self.result_dir, "input_data/params.yml")
        self.params = self._load_yaml(config_path)

        # 2. Load fragment vocabulary
        frag_path = os.path.join(self.result_dir, "input_data/fragments.csv")
        self.vocab_list = self._load_fragments(frag_path)
        num_tokens = len(self.vocab_list)
        print(f"[Info] Fragment vocabulary loaded, size: {num_tokens}")

        # 3. Extract model and decomposition parameters
        model_cfg = self.params.get('model', {})
        decomp_cfg = self.params.get('decomp', {})

        # 4. Compute fragment features (ECFP fingerprints + dummy atom counts)
        print("[Init] Computing fragment chemical features (ECFP fingerprints)...")
        self.frag_ecfps, self.ndummys = self._compute_fragment_features(
            self.vocab_list,
            radius=decomp_cfg.get('radius', 2),
            nbits=decomp_cfg.get('n_bits', 2048),
        )

        # 5. Instantiate FRATTVAE model
        print("[Init] Building FRATTVAE model architecture...")
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

        # 6. Weights are not loaded automatically; caller must invoke load_weights() explicitly

    # ---------------- Internal utility functions ----------------

    def _load_yaml(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        print(f"[Warning] Config file not found: {path}")
        return {}

    def _load_fragments(self, path):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'smiles' in df.columns:
                    return df['smiles'].tolist()
                return df.iloc[:, 0].tolist()
            except Exception as e:
                print(f"[Warning] Failed to read fragment file {path}: {e}")
        print("[Warning] Using placeholder fragment vocabulary")
        return ["<pad>", "C", "CC"]

    def _compute_fragment_features(self, vocab_list, radius=2, nbits=2048):
        """
        Compute per-fragment features:
        - ECFP fingerprint (nbits dimensions)
        - Number of dummy '*' atoms (connection points)
        """
        ecfps = []
        ndummys = []

        for smi in vocab_list:
            # 1. Count dummy atoms
            ndummys.append(smi.count('*'))

            # 2. Compute ECFP fingerprint
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

    # ---------------- Weight loading and sampling interfaces ----------------

    def load_weights(self):
        weight_path = os.path.join(self.result_dir, "model/model_best.pth")
        if os.path.exists(weight_path):
            print(f"[Load weights] {weight_path}")
            state_dict = torch.load(weight_path, map_location=self.device)
            new_state = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state[name] = v

            try:
                self.model.load_state_dict(new_state, strict=True)
                print("✅ Weights loaded successfully (strict=True)")
            except RuntimeError as e:
                print(f"⚠️ Falling back to non-strict loading: {e}")
                self.model.load_state_dict(new_state, strict=False)
                print("✅ Weights loaded successfully (strict=False)")
        else:
            print(f"❌ Weight file not found: {weight_path}")

    @property
    def latent_dim(self) -> int:
        """Expose latent dimensionality for use in Step 3."""
        return int(self.model.latent_dim)

    def sample_latent(self, num_samples: int = 1) -> torch.Tensor:
        """Sample latent vectors from a standard normal distribution."""
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return z

    def decode_from_latent(self, latent: torch.Tensor, max_nfrags: int = 16):
        """
        Decode latent vectors into a list of SMILES strings.
        latent: [N, latent_dim]
        Returns: list of N SMILES strings
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
        # FRATTVAE typically returns list[str]
        return outputs

    def sample(self, num_samples=3):
        """
        Convenience interface: sample from the latent space and decode directly.
        """
        print(f"[Generate] Sampling and decoding {num_samples} molecules from latent space...")
        z = self.sample_latent(num_samples)
        return self.decode_from_latent(z)


# --- Test / training entry point ---
if __name__ == "__main__":
    gen = VAE_Generator()
    gen.load_weights()

    print("\n>>> Sample decoded molecules:")
    mols = gen.sample(3)
    for i, m in enumerate(mols):
        print(f"[{i+1}] {m}")
