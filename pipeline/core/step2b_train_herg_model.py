# File: core/step2b_train_herg_model.py
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
    """Convert a SMILES string to a Morgan fingerprint (bit vector)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((n_bits,), dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Read locally exported hERG data
    data_path = os.path.join(current_dir, "../data/herg_tdc_full.csv")
    if not os.path.exists(data_path):
        print(f"❌ hERG data file not found: {data_path}")
        print("Please export herg_tdc_full.csv from a networked environment and place it in the data/ directory.")
        return

    print(f"📥 Reading hERG data: {data_path}")
    df = pd.read_csv(data_path)

    # Expected column names: 'Drug' and 'Y'
    if "Drug" not in df.columns or "Y" not in df.columns:
        print("❌ Unexpected data format. Expected columns 'Drug' and 'Y'.")
        print("Actual columns:", df.columns.tolist())
        return

    smiles_list = df["Drug"].astype(str).tolist()
    labels = df["Y"].values

    print(f"Total records: {len(df)}")

    # 2. Train / test split
    X_train_smi, X_test_smi, y_train, y_test = train_test_split(
        smiles_list, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    # 3. Feature encoding
    print("🔬 Encoding features (Morgan fingerprints)...")
    X_train = np.array([smiles_to_fp(s) for s in X_train_smi])
    X_test = np.array([smiles_to_fp(s) for s in X_test_smi])

    # 4. Train RandomForest classifier
    print("🌲 Training RandomForest hERG classifier...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    X_tr, y_tr = shuffle(X_train, y_train, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)

    # 5. Evaluate on test set
    print("📊 Evaluating model on test set...")
    y_proba_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba_test)
    auprc = average_precision_score(y_test, y_proba_test)

    print(f"✅ hERG RF model AUROC: {auc:.3f}")
    print(f"✅ hERG RF model AUPRC: {auprc:.3f}")

    # 6. Save model to ../results/herg_model/herg_rf_model.pkl
    model_dir = os.path.join(current_dir, "../results/herg_model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "herg_rf_model.pkl")

    joblib.dump(clf, model_path)
    print(f"💾 Model saved to: {model_path}")
    print("🎉 hERG model training complete.")

if __name__ == "__main__":
    main()
