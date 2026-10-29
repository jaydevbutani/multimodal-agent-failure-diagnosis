"""
Central Machine Learning Configuration & Hyperparameter Registry
Agent Health Intelligence (AUCML — TE7483)
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PATHS = {
    "raw_data": os.path.join(BASE_DIR, "data", "processed"),
    "models_saved": os.path.join(BASE_DIR, "models_saved"),
    "results_metrics": os.path.join(BASE_DIR, "results", "metrics"),
    "results_plots": os.path.join(BASE_DIR, "results", "plots"),
    "results_comparisons": os.path.join(BASE_DIR, "results", "comparisons"),
}

# Fixed Random Seed for Complete Reproducibility
RANDOM_SEED = 42

# Decision Boundary Policy Threshold
CALIBRATED_THRESHOLD = 0.46

# Classical Model Hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_SEED
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

XGBOOST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 6,
    "learning_rate": 0.08,
    "scale_pos_weight": 3.34,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# PyTorch Deep Learning Hyperparameters
DEEP_MODEL_CONFIG = {
    "vocab_max_len": 50,
    "embedding_dim": 32,
    "bilstm_hidden_dim": 64,
    "fusion_hidden_dim": 64,
    "tabular_dim": 88,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "epochs": 15,
    "pos_weight": 3.34,
    "dropout": 0.3
}
