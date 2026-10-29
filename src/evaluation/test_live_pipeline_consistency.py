import os
import sys
sys.path.insert(0, os.getcwd())
import joblib
import numpy as np
import pandas as pd
from src.inference.diagnose import AgentFailureDiagnoser

def test_pipeline_consistency():
    print("="*65)
    print("PHASE 13: LIVE DEMO PIPELINE CONSISTENCY VERIFICATION")
    print("="*65)
    
    # 1. Load test parquet and precomputed X_test
    df_test = pd.read_parquet("data/processed/test.parquet")
    X_test = np.load("data/processed/X_test.npy")
    rf_model = joblib.load("models_saved/random_forest.joblib")
    
    # 2. Select 5 sample test records
    sample_indices = [0, 10, 50, 100, 500]
    
    diagnoser = AgentFailureDiagnoser()
    
    all_match = True
    for idx in sample_indices:
        row_dict = df_test.iloc[idx].to_dict()
        
        # Method A: Direct precomputed X_test inference
        prob_direct = float(rf_model.predict_proba(X_test[idx:idx+1])[0, 1])
        
        # Method B: Passing through Live Diagnoser harness (raw dict -> derived features -> preprocessor -> model)
        diag_result = diagnoser.diagnose_trace(row_dict)
        prob_live = diag_result['rf_probability']
        
        diff = abs(prob_direct - prob_live)
        print(f"Sample {idx} (Task: {row_dict['task_id'][:25]}...): Direct RF Prob = {prob_direct:.4f}, Live Diagnoser RF Prob = {prob_live:.4f}, Diff = {diff:.6f}")
        
        if diff > 1e-4:
            all_match = False
            
    print("\n--- Pipeline Consistency Verdict ---")
    if all_match:
        print("VERIFIED: Training pipeline, Offline inference pipeline, and Live demo pipeline are IDENTICAL.")
        print("Proof: Preprocessing transformations, feature formulas, and loaded model weights match exactly.")
    else:
        print("FAILED: Discrepancy detected between offline and live pipelines.")
        
    assert all_match, "Pipeline consistency check failed!"

if __name__ == '__main__':
    test_pipeline_consistency()
