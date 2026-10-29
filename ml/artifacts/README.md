# Artifacts Registry
## Agent Health Intelligence (AUCML — TE7483)

This registry documents all persisted trained checkpoints, preprocessor transformers, vocabularies, and metric results.

### 1. Trained Model Checkpoints (`models_saved/`)
- `multimodal_fusion.pt`: PyTorch weights for the Heterogeneous Multi-View Gated Fusion Network (tab_dim=88, vocab_size=32, emb_dim=32, fusion_dim=64).
- `sequence_bilstm.pt`: PyTorch weights for the Sequential Trajectory BiLSTM model.
- `xgboost.joblib`: Audited XGBoost Classifier (n_estimators=150, max_depth=6).
- `random_forest.joblib`: Audited Random Forest Classifier (n_estimators=150, max_depth=12).
- `logistic_regression.joblib`: Audited Logistic Regression Baseline.

### 2. Preprocessing & Feature Encoders (`data/processed/`)
- `preprocessor.joblib`: Scikit-Learn `ColumnTransformer` (StandardScaler on 18 numericals, OneHotEncoder on categoricals).
- `tool_vocab.joblib`: Discrete token vocabulary mapping for agent tool calls.
- `feature_names.joblib`: 88 transformed tabular feature column names.
- `train.parquet`, `val.parquet`, `test.parquet`: Leakage-free split partitions (11,995 total trials).

### 3. Evaluation Artifacts (`results/`)
- `comparisons/model_comparison_table.csv`: Official test set metric comparison across all 5 architectures.
- `metrics/top_predictive_features.csv`: Ranked empirical feature importance scores.
- `plots/`: Confusion matrices, ROC curves, and SHAP summary plots.
