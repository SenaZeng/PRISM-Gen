# File: core/step2_surrogate.py
# -*- coding: utf-8 -*-
"""
Step 2: Uni-Mol surrogate model wrapper
- Encodes SMILES into CLS embeddings using UniMolRepr
- Uses a RandomForestRegressor for activity regression (pIC50)
- Exposes SurrogateModel.predict(smiles_list) / predict_single(smiles)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Mount unimol_source onto sys.path ---

current_dir = os.path.dirname(os.path.abspath(__file__))
unimol_path = os.path.join(current_dir, 'unimol_source')
if unimol_path not in sys.path:
    sys.path.append(unimol_path)

try:
    # Use the official Uni-Mol interface
    from unimol_tools import UniMolRepr
except ImportError:
    print("ERROR: Cannot import unimol_tools. Check that core/unimol_source exists and is accessible.")
    raise


class SurrogateModel:
    """
    Surrogate model based on Uni-Mol representations + sklearn RandomForest.
    Used for rapid prediction of Mpro inhibitory activity (pIC50).
    """
    def __init__(self, work_dir="../results/surrogate_model"):
        # Working directory for saving the sklearn model
        self.work_dir = os.path.abspath(os.path.join(current_dir, work_dir))
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)

        self.sklearn_model_path = os.path.join(self.work_dir, "activity_predictor.pkl")

        # Initialize Uni-Mol feature extractor
        print("[Init] Loading Uni-Mol pretrained model (surrogate representation)...")
        # data_type='molecule' for small molecules; remove_hs=True matches mol_pre_no_h pretraining
        self.clf = UniMolRepr(data_type='molecule', remove_hs=True)
        self.model = None

    # ------------- Internal: Uni-Mol embedding extraction ------------- #

    def _get_embeddings(self, smiles_list):
        """
        Encode SMILES into high-dimensional vectors (N, 512) using Uni-Mol.
        - UniMolRepr.get_repr: automatically generates 3D conformers and runs the Transformer forward pass
        - Returns a dict; 'cls_repr' contains [N, 512]
        """
        print(f"[Feature extraction] Processing {len(smiles_list)} molecules (3D conformers generated automatically)...")
        reprs = self.clf.get_repr(smiles_list, return_atomic_reprs=False)
        # reprs['cls_repr'] is tensor-like/array-like; convert to numpy
        X = np.array(reprs['cls_repr'])
        return X

    # ------------- Training pipeline ------------- #

    def train(self, data_path):
        """
        Training pipeline:
        - Read CSV
        - Extract Uni-Mol CLS embeddings
        - Train a RandomForest regression model to predict pIC50
        """
        print(f"[Train] Reading data: {data_path}")
        df = pd.read_csv(data_path)

        if 'pchembl_value' in df.columns:
            y = df['pchembl_value'].values
        elif 'Standard Value' in df.columns:
            vals = pd.to_numeric(df['Standard Value'], errors='coerce')
            vals = np.where(vals <= 0, 1e-9, vals)
            y = 9 - np.log10(vals)  # pIC50 = 9 - log10(nM)
        else:
            raise ValueError("Activity column not found in CSV (expected 'pchembl_value' or 'Standard Value')")

        X_smiles = df['smiles'].astype(str).tolist()

        # 1. Extract Uni-Mol representations
        X_emb = self._get_embeddings(X_smiles)

        # 2. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_emb, y, test_size=0.2, random_state=42
        )

        # 3. Train RandomForest regressor
        print("[Train] Fitting downstream regression model (RandomForestRegressor)...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        # 4. Quick evaluation
        score = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        print(f"[Eval] R2 Score: {score:.4f}, RMSE: {rmse:.4f}")

        # 5. Save sklearn model
        joblib.dump(self.model, self.sklearn_model_path)
        print(f"[Save] Model saved to {self.sklearn_model_path}")

    # ------------- Load and predict interfaces ------------- #

    def load_model(self):
        """Load a previously trained sklearn model."""
        if os.path.exists(self.sklearn_model_path):
            self.model = joblib.load(self.sklearn_model_path)
            print(f"✅ Surrogate model loaded: {self.sklearn_model_path}")
        else:
            print("⚠️ Surrogate model not found. Please run train() first.")

    def predict(self, smiles_list):
        """
        Prediction interface:
        - Input:  a list of SMILES strings
        - Output: corresponding predicted pIC50 values (list[float])
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                return [0.0] * len(smiles_list)

        X_emb = self._get_embeddings(smiles_list)
        preds = self.model.predict(X_emb)
        return preds

    def predict_pIC50(self, smiles_list):
        """Semantically explicit alias for predict(): returns predicted pIC50 values."""
        return self.predict(smiles_list)

    def predict_single(self, smiles: str) -> float:
        """Single-molecule prediction interface for use in Step 3."""
        return float(self.predict([smiles])[0])


# --- Test / training entry point --- #

if __name__ == "__main__":
    agent = SurrogateModel()

    # Resolve data path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.abspath(
        os.path.join(script_dir, "../data/processed/mpro_actives_clean.csv")
    )

    print(f"Data path resolved to: {data_file}")

    # Train if no saved model exists
    if not os.path.exists(agent.sklearn_model_path):
        if os.path.exists(data_file):
            print(">>> First run: starting surrogate model training...")
            agent.train(data_file)
        else:
            print(
                f"❌ Error: training data not found!\n"
                f"Expected: {data_file}\n"
                f"Hint: run 'python tools/data_cleaner.py' first."
            )

    # Quick prediction test
    test_mols = [
        "CC(=O)Nc1ccc(O)cc1",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]
    print("\n>>> Prediction test (Mpro activity pIC50):")
    scores = agent.predict(test_mols)
    for smi, score in zip(test_mols, scores):
        print(f"SMILES: {smi[:30]}... -> predicted pIC50: {score:.2f}")
