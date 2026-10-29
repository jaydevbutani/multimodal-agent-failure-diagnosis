# RESULTS AUDIT & RECALCULATION VERIFICATION

**Course:** Applications and Use Cases of Machine Learning (AUCML — TE7483)  
**Evaluation:** Independent Audit of Test Metrics & Saved Predictions  
**Test Set Size:** 1,996 Unseen Trials (Zero Task Overlap with Train/Val Sets)  
**Predictions Storage:** `data/processed/test_predictions.parquet`  

---

## 1. Traceability Chain: Data → Code → Model → Predictions → Metrics

Every reported metric in this project is directly traceable through the following artifact chain:

```
Raw Parquet: data/raw/agent_diagnostics/signals.parquet (11,995 trials)
     │
     ↓ (src/preprocessing/prepare_data.py)
Grouped Split: data/processed/test.parquet (1,996 samples, 869 unique tasks)
Feature Matrix: data/processed/X_test.npy (Shape: 1,996 × 88)
Ground Truth: data/processed/y_test.npy (Class 0: 1,556, Class 1: 440)
     │
     ↓ (src/models/classical_models.py & src/models/deep_models.py)
Model Weights: models_saved/ (RF, XGB, LR, BiLSTM, Multimodal Fusion)
     │
     ↓ (src/evaluation/verify_predictions.py)
Exact Predictions: data/processed/test_predictions.parquet
     │
     ↓
Verified Metrics: results/comparisons/model_comparison_table.csv
```

---

## 2. Recalculated Test Performance Benchmark Table

Recalculated directly from `data/processed/test_predictions.parquet` across all 1,996 test trials:

| Model Architecture | Input Modality / Representation | Test Accuracy | Precision | Recall (Failure) | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | Tabular Telemetry (Linear) | 0.7771 | 0.4959 | 0.6932 | 0.7133 | 0.7889 | 0.8344 | 0.6196 |
| **Random Forest (Balanced)** | Tabular Telemetry (Tree Ensemble) | **0.8427** | **0.6641** | 0.5795 | 0.7599 | **0.8387** | **0.8632** | 0.7128 |
| **XGBoost Classifier** | Tabular Telemetry (Gradient Boosted) | 0.8337 | 0.6120 | 0.6705 | **0.7659** | 0.8363 | 0.8595 | **0.7132** |
| **Sequential BiLSTM** | Tool-Call Sequence Only | 0.7971 | 0.5314 | 0.6727 | 0.7293 | 0.8050 | 0.8318 | 0.5936 |
| **Proposed Multi-View Fusion Network** | **Telemetry + Tool Sequence** | 0.8081 | 0.5487 | **0.7295** | 0.7486 | 0.8170 | 0.8568 | 0.6949 |

---

## 3. Dissected Confusion Matrices (1,996 Test Trials)

### Random Forest (Balanced):
- True Normal (TN): 1,427 | False Positive (FP): 129
- False Negative (FN): 185 | True Positive (TP): 255
- *Interpretation:* High specificity ($91.7\%$) and low false alarm rate ($8.3\%$), but misses 185 actual failures.

### XGBoost Classifier:
- True Normal (TN): 1,369 | False Positive (FP): 187
- False Negative (FN): 145 | True Positive (TP): 295
- *Interpretation:* Strong balance between precision ($61.2\%$) and recall ($67.1\%$), producing the top Macro-F1 score ($0.7659$).

### Proposed Multi-View Fusion Network:
- True Normal (TN): 1,292 | False Positive (FP): 264
- False Negative (FN): 119 | True Positive (TP): 321
- *Interpretation:* Maximizes sensitivity to agent failure; achieves the lowest False Negative count (119 misses), catching 321 out of 440 true failure cases.

---

## 4. Scientifically Correct Performance Narrative & Corrections Made

1. **Reframing Model Comparisons:**  
   We do NOT claim the Fusion Network is the universal winner across all metrics. Instead, the experimental trade-off is explicitly stated:
   - **XGBoost** is the best model for overall balanced classification (Macro-F1: 0.7659, PR-AUC: 0.7132).
   - **Random Forest** achieves the highest overall accuracy (0.8427) and ROC-AUC (0.8632).
   - **Proposed Multi-View Fusion Network** achieves the highest failure-class recall (0.7295), which is critical for SRE safety monitoring where undetected failures lead to silent system outages.
2. **Failure Recall Precision:**  
   - Absolute improvement: $+15.0$ percentage points over Random Forest ($72.95\% - 57.95\%$).
   - Relative improvement: $+25.9\%$ relative gain ($\frac{0.7295 - 0.5795}{0.5795}$).
   - We avoid unscientific claims such as "26% more catastrophic failures" since the raw label represents general task failure.
