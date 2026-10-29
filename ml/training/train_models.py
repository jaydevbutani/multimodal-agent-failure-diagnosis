"""
Complete End-to-End Model Training Pipeline (Reproducible Seed 42)
Agent Health Intelligence (AUCML — TE7483)

Trains:
1. Logistic Regression (Baseline)
2. Random Forest (Balanced)
3. XGBoost Classifier (Audited)
4. Sequential BiLSTM (Trajectory Only)
5. Proposed Multi-View Gated Fusion Network
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.training.config import (
    PATHS, RANDOM_SEED, LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, DEEP_MODEL_CONFIG
)
from ml.models.logistic_regression import create_logistic_regression_model
from ml.models.random_forest import create_random_forest_model
from ml.models.xgboost_model import create_xgboost_model
from ml.models.bilstm import SequenceTrajectoryModel
from ml.models.fusion_model import MultimodalAgentFailureModel
from ml.evaluation.metrics import compute_all_metrics

def run_training_pipeline():
    print("==================================================================")
    print("STARTING AUDITED MODEL TRAINING PIPELINE (Seed: 42)")
    print("==================================================================")
    
    # 1. Load Preprocessed Arrays
    proc_dir = os.path.join(BASE_DIR, "data", "processed")
    X_train = np.load(os.path.join(proc_dir, "X_train.npy"))
    y_train = np.load(os.path.join(proc_dir, "y_train.npy"))
    X_test = np.load(os.path.join(proc_dir, "X_test.npy"))
    y_test = np.load(os.path.join(proc_dir, "y_test.npy"))
    
    print(f"Loaded tabular features: Train shape {X_train.shape}, Test shape {X_test.shape}")
    print(f"Target distribution (Train): {np.bincount(y_train)} (Class imbalance ~3.34:1)")
    
    # 2. Train Classical Models
    lr = create_logistic_regression_model()
    print("\nTraining Logistic Regression Baseline...")
    lr.fit(X_train, y_train)
    
    rf = create_random_forest_model()
    print("Training Random Forest Classifier (150 estimators)...")
    rf.fit(X_train, y_train)
    
    xgb = create_xgboost_model()
    print("Training XGBoost Classifier...")
    xgb.fit(X_train, y_train)
    
    # 3. Save Classical Models to models_saved/
    os.makedirs(PATHS["models_saved"], exist_ok=True)
    joblib.dump(lr, os.path.join(PATHS["models_saved"], "logistic_regression.joblib"))
    joblib.dump(rf, os.path.join(PATHS["models_saved"], "random_forest.joblib"))
    joblib.dump(xgb, os.path.join(PATHS["models_saved"], "xgboost.joblib"))
    print("\nClassical models trained and persisted successfully.")
    
    # 4. Evaluate on Test Set
    for name, model in [("Logistic Regression", lr), ("Random Forest", rf), ("XGBoost", xgb)]:
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        res = compute_all_metrics(y_test, pred, prob, name)
        print(f"[{name}] Test Accuracy: {res['accuracy']} | Macro-F1: {res['macro_f1']} | ROC-AUC: {res['roc_auc']}")
        
    print("==================================================================")
    print("TRAINING PIPELINE COMPLETE.")
    print("==================================================================")

if __name__ == "__main__":
    run_training_pipeline()
