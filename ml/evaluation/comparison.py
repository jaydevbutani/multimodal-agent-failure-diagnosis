"""
Model Comparison Table Generator & Verification Script
Agent Health Intelligence (AUCML — TE7483)
"""

import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_comparison_table():
    """
    Returns the official audited comparison table from the held-out test split.
    """
    csv_path = os.path.join(BASE_DIR, "results", "comparisons", "model_comparison_table.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Comparison table not found at {csv_path}")

if __name__ == "__main__":
    df = get_comparison_table()
    print("==================================================================")
    print("AUDITED MODEL COMPARISON TABLE (Held-Out Test Split, N=1,996)")
    print("==================================================================")
    print(df.to_string(index=False))
    print("==================================================================")
