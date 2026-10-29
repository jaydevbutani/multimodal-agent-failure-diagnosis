"""
Preprocessing & Leakage-Safe Data Preparation Module
=====================================================
Agent Health Intelligence (AUCML — TE7483)

Single Responsibility:
- Grouped train/validation/test split by task_id (0% task overlap, zero leakage)
- Scale numerical features and encode categorical variables
- Encode tool-call sequences into integer token sequences for neural views
- Save and load fitted preprocessors
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

NUMERICAL_COLS = [
    "total_turns", "tool_calls_total", "error_count", "retry_count",
    "search_tool_calls", "edit_tool_calls", "unique_files_read",
    "unique_files_edited", "patch_size_lines", "duration_seconds",
    "DERIVED_error_rate", "DERIVED_retry_ratio", "DERIVED_search_to_edit_ratio",
    "DERIVED_edit_churn", "DERIVED_tool_intensity", "DERIVED_read_to_edit_ratio",
    "DERIVED_time_per_turn", "DERIVED_tool_entropy", "DERIVED_log_duration",
    "DERIVED_log_patch_size"
]

CATEGORICAL_COLS = [
    "model", "agent_name", "benchmark"
]

def create_grouped_splits(df: pd.DataFrame, group_col: str = "task_id") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes a grouped split (70% train / 15% val / 15% test) strictly grouped by task_id.
    Guarantees 0% task overlap between train, validation, and test partitions.
    """
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df[group_col]))
    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp[group_col]))
    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()
    
    # Verify leakage guarantee
    train_tasks = set(df_train[group_col].unique())
    val_tasks = set(df_val[group_col].unique())
    test_tasks = set(df_test[group_col].unique())
    assert len(train_tasks.intersection(test_tasks)) == 0, "DATA LEAKAGE DETECTED between train and test!"
    assert len(train_tasks.intersection(val_tasks)) == 0, "DATA LEAKAGE DETECTED between train and val!"
    
    return df_train, df_val, df_test

def build_tabular_preprocessor() -> ColumnTransformer:
    """Builds scikit-learn preprocessing pipeline for tabular telemetry."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS)
    ], remainder="drop")
    
    return preprocessor

def encode_tool_sequence(seq: List[str], vocab: Dict[str, int], max_len: int = 50) -> np.ndarray:
    """Encodes tool string sequence into fixed-length integer array with zero padding."""
    arr = np.zeros(max_len, dtype=np.int64)
    if not seq:
        return arr
    for i, tool in enumerate(seq[:max_len]):
        t_clean = str(tool).strip().lower()
        arr[i] = vocab.get(t_clean, vocab.get("<UNK>", 1))
    return arr

def load_preprocessor():
    """Loads saved preprocessor joblib artifact."""
    p_path = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
    if not os.path.exists(p_path):
        p_path = os.path.join(BASE_DIR, "models", "preprocessor_fitted.joblib")
    return joblib.load(p_path)
