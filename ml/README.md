# Machine Learning Architecture: Agent Health Intelligence
## Heterogeneous Multi-View Trace Fusion for AI Agent Failure Diagnosis

This directory contains the complete, reproducible, and audited machine learning pipeline for the **Agent Health Intelligence** system (AUCML — TE7483).

---

## 1. Directory Structure

```
ml/
├── README.md                           # This documentation
├── data/
│   ├── README.md                       # Dataset provenance & governance
│   └── data_dictionary.csv             # Complete feature specification (34 features)
├── preprocessing/
│   ├── preprocess.py                   # Data ingestion, leakage-free split, preprocessing
│   └── feature_engineering.py          # 10 domain-specific derived features
├── models/
│   ├── logistic_regression.py          # Baseline linear model definition
│   ├── random_forest.py                # Audited Random Forest classifier
│   ├── xgboost_model.py                # Gradient-boosted decision trees (Top Macro-F1)
│   ├── bilstm.py                       # Sequential tool trajectory model (PyTorch)
│   └── fusion_model.py                 # Multi-View Gated Fusion Network (PyTorch)
├── training/
│   ├── config.py                       # Central hyperparameter configuration
│   └── train_models.py                 # Reproducible training pipeline (Seed 42)
├── evaluation/
│   ├── metrics.py                      # Multi-metric evaluation functions
│   ├── evaluate_models.py              # Test set evaluation suite
│   └── comparison.py                   # Generates official model comparison table
├── explainability/
│   └── feature_importance.py           # Feature importance & attribution analysis
├── visualization/
│   ├── confusion_matrix.py             # Generates confusion matrix plots
│   ├── roc_curve.py                    # Generates ROC curve plots
│   └── pr_curve.py                     # Generates Precision-Recall curve plots
├── inference/
│   └── predict.py                      # Standalone CLI & programmatic inference engine
└── artifacts/
    └── README.md                       # Registry of saved checkpoints and encoders
```

---

## 2. Models & Hyperparameter Specifications

| Model | Architecture | Explicit Hyperparameters | Key Role |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear / Sigmoid | `class_weight='balanced'`, `max_iter=1000`, `solver='lbfgs'`, `random_state=42` | Audited Classical Baseline |
| **Random Forest** | 150 Decision Trees | `n_estimators=150`, `max_depth=12`, `min_samples_split=5`, `class_weight='balanced'`, `random_state=42` | Highest Test Accuracy (84.27%) & Precision (0.6641) |
| **XGBoost Classifier** | Gradient Boosted Trees | `n_estimators=150`, `max_depth=6`, `learning_rate=0.08`, `scale_pos_weight=3.34`, `eval_metric='logloss'`, `random_state=42` | Highest Overall Macro-F1 (0.7659) & PR-AUC (0.7132) |
| **Sequential BiLSTM** | Bidirectional LSTM | `emb_dim=32`, `hidden_dim=64`, `dropout=0.3`, `lr=0.001`, `AdamW`, `max_len=50` | Trajectory-Only Sequence Baseline |
| **Multi-View Fusion** | Cross-Modal Gated Fusion | Tabular branch (128&rarr;64) + BiLSTM branch (64&rarr;64) + Sigmoid Gating + Head (32&rarr;1), `pos_weight=3.34`, `dropout=0.3`, `epochs=15`, `lr=0.001` | Highest Failure Recall (72.95%), Most Sensitive at Intercepting Agent Failures |

---

## 3. Audited Evaluation Results (Held-Out Test Split, N=1,996)

All values are computed directly from the held-out test predictions:

| Model Architecture | Accuracy | Precision | Failure Recall | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 0.7771 | 0.4959 | 0.6932 | 0.7133 | 0.7889 | 0.8344 | 0.6196 |
| Random Forest (Balanced) | **0.8427** | **0.6641** | 0.5795 | 0.7599 | **0.8387** | **0.8632** | 0.7128 |
| XGBoost Classifier | 0.8337 | 0.6120 | 0.6705 | **0.7659** | 0.8363 | 0.8595 | **0.7132** |
| Sequential BiLSTM (Trajectory Only) | 0.7971 | 0.5314 | 0.6727 | 0.7293 | 0.8050 | 0.8318 | 0.5936 |
| **Proposed Multi-View Fusion Network** | 0.8081 | 0.5487 | **0.7295** | 0.7486 | 0.8170 | 0.8568 | 0.6949 |

> **Scientific Honesty Note:** XGBoost achieved the strongest overall Macro-F1 (0.7659), while the proposed Multi-View Fusion Network achieved higher Failure Recall (0.7295 vs 0.6705), making it the most sensitive architecture for catching silent agent failures in production without missing fatal runs.

---

## 4. Reproducing the ML Pipeline

To re-run the entire pipeline end-to-end:
```bash
# 1. Feature Preprocessing
python ml/preprocessing/preprocess.py

# 2. Model Training
python ml/training/train_models.py

# 3. Model Evaluation & Comparison
python ml/evaluation/evaluate_models.py

# 4. Standalone Inference
python ml/inference/predict.py --trace data/demo_traces/02_retry_thrashing.json
```
