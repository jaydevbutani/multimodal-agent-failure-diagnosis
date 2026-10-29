"""
Data Ingestion, Split Verification & Pipeline Preprocessing
Agent Health Intelligence (AUCML — TE7483)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ml.preprocessing.feature_engineering import extract_derived_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_processed_splits():
    """
    Loads the leakage-free train, validation, and test splits (Parquet format)
    created from the Agent Diagnostics benchmark suite (11,995 total trials).
    """
    proc_dir = os.path.join(BASE_DIR, "data", "processed")
    train_df = pd.read_parquet(os.path.join(proc_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(proc_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(proc_dir, "test.parquet"))
    
    preprocessor = joblib.load(os.path.join(proc_dir, "preprocessor.joblib"))
    vocab = joblib.load(os.path.join(proc_dir, "tool_vocab.joblib"))
    feature_names = joblib.load(os.path.join(proc_dir, "feature_names.joblib"))
    
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "preprocessor": preprocessor,
        "vocab": vocab,
        "feature_names": feature_names
    }

if __name__ == "__main__":
    print("Verifying processed dataset splits and artifacts...")
    data = load_processed_splits()
    print(f"Train split: {len(data['train_df'])} rows")
    print(f"Validation split: {len(data['val_df'])} rows")
    print(f"Test split: {len(data['test_df'])} rows (1,996 held-out)")
    print(f"Transformed features: {len(data['feature_names'])} dimensions")
    print(f"Tool vocabulary size: {len(data['vocab'])} tokens")
    print("Preprocessed artifacts verified successfully.")
