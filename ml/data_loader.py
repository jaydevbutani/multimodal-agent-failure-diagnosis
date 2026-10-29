"""
Data Loader Module
==================
Agent Health Intelligence (AUCML — TE7483)

Single Responsibility:
- Load agent execution trajectory datasets from disk
- Validate required telemetry and label columns
- Return clean Pandas DataFrames for downstream preprocessing
"""

import os
import pandas as pd
from typing import Tuple, Optional, Dict, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQUIRED_COLUMNS = [
    "task_id",
    "total_turns",
    "tool_calls_total",
    "error_count",
    "retry_count",
    "duration_seconds",
    "passed"
]

def load_dataset(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test splits from parquet/csv storage.
    
    Returns:
        (df_train, df_val, df_test)
    """
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, "data", "processed")
        if not os.path.exists(os.path.join(data_dir, "train.parquet")):
            data_dir = os.path.join(BASE_DIR, "data")
            
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Processed dataset partitions not found at {data_dir}")
        
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path) if os.path.exists(val_path) else pd.DataFrame()
    df_test = pd.read_parquet(test_path)
    
    # Validate required columns
    for col in REQUIRED_COLUMNS:
        assert col in df_train.columns, f"Required column '{col}' missing from training data."
        assert col in df_test.columns, f"Required column '{col}' missing from test data."
        
    return df_train, df_val, df_test

def get_dataset_summary() -> Dict[str, Any]:
    """Returns dataset summary statistics for presentation and viva audit."""
    df_train, df_val, df_test = load_dataset()
    total_samples = len(df_train) + len(df_val) + len(df_test)
    fail_count = sum((df["passed"] == 0).sum() for df in [df_train, df_val, df_test] if not df.empty)
    
    return {
        "dataset_name": "Agent Diagnostics & SWE-bench Execution Traces",
        "total_trials": total_samples,
        "train_trials": len(df_train),
        "val_trials": len(df_val),
        "test_trials": len(df_test),
        "target_variable": "passed (0 = Failure, 1 = Success)",
        "failure_ratio": fail_count / max(1, total_samples),
        "leakage_strategy": "GroupShuffleSplit by task_id (0% task overlap)"
    }

if __name__ == "__main__":
    summary = get_dataset_summary()
    print("==================================================================")
    print("DATA LOADER AUDIT SUMMARY")
    print("==================================================================")
    for k, v in summary.items():
        print(f"{k:20s}: {v}")
    print("==================================================================")
