"""
Feature Importance & Attribution Analysis
Agent Health Intelligence (AUCML — TE7483)
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_top_predictive_features():
    """
    Retrieves the audited top predictive behavioral features ranked by importance.
    """
    csv_path = os.path.join(BASE_DIR, "results", "metrics", "top_predictive_features.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # Fallback to feature importance extracted from XGBoost / Random Forest
        rf = joblib.load(os.path.join(BASE_DIR, "models_saved", "random_forest.joblib"))
        feat_names = joblib.load(os.path.join(BASE_DIR, "data", "processed", "feature_names.joblib"))
        importances = rf.feature_importances_
        idx = np.argsort(importances)[::-1][:15]
        return pd.DataFrame({
            "feature": [feat_names[i] for i in idx],
            "importance": importances[idx]
        })

if __name__ == "__main__":
    df = get_top_predictive_features()
    print("==================================================================")
    print("TOP PREDICTIVE FEATURES (Audited Importance)")
    print("==================================================================")
    print(df.head(10).to_string(index=False))
    print("==================================================================")
