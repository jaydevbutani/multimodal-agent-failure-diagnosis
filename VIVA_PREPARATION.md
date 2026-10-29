# VIVA PREPARATION GUIDE: Questions, Mathematical Defense & Answers

**Course:** Applications and Use Cases of Machine Learning (AUCML — TE7483)  
**Total Marks:** 3 Marks for Viva / Defense (Target: 3/3)  
**Team Members:** Division AIML — Batch 2024-28 — Semester 5  

---

## 1. Core Problem & Justification

### Q1: Why this problem? Why not a standard Titanic or Credit Card Fraud project?
**Answer:**  
"Traditional datasets are saturated and do not demonstrate contemporary Machine Learning application engineering. AI Agents operating autonomously across software repositories and tool APIs are increasingly deployed in enterprise systems. When they fail, simple binary success/failure signals fail to diagnose why. Building an ML system to monitor agentic execution telemetry, detect failure patterns, and localize faulty components addresses a real-world SRE (Site Reliability Engineering) challenge using cutting-edge 2025/2026 agent telemetry."

### Q2: Why is Machine Learning appropriate here instead of simple hard-coded if/else rules?
**Answer:**  
"Hard-coded rules (e.g. `if errors > 5: fail`) fail because healthy agents often encounter intermediate errors (e.g., trying a command that fails, reading the error message, and successfully fixing it). Conversely, silent failures (reward hacking, infinite loops with zero errors, or hallucinated APIs) produce zero syntax errors. Machine Learning captures non-linear interactions across turn count, edit churn, tool diversity entropy, and retry ratios that static threshold rules cannot detect."

---

## 2. Dataset Authenticity & Preprocessing Defense

### Q3: How do we know this data is real and not fabricated?
**Answer:**  
"Our primary dataset is sourced directly from the **Agent Diagnostics Observatory** (`sjarmak/agent-diagnostics`), containing **11,995 authentic trials** with exact SHA-256 hash `4cea2359787900e8fc20e0e583b21763da99f46d99868cf7f18a1877c9425222` verified against `MANIFEST.json`. It captures actual executions of Claude Code, OpenHands, and Cursor across 61 real benchmarks. Furthermore, our attribution benchmark is the peer-reviewed **ICML 2025 Spotlight dataset (`Who&When`)**. We have a complete `SOURCE_MANIFEST.json` and zero synthetic rows were generated."

### Q4: How did you strictly prevent Data Leakage?
**Answer:**  
"We implemented **Grouped Task Splitting** (`GroupShuffleSplit` on `task_id`). If an agent attempts the same task multiple times or with different configurations, all those trials are quarantined strictly into one split. There is **0% task overlap** and **0% trial_id overlap** between train (8,298 samples), validation (1,701 samples), and test (1,996 samples). Moreover, all scalers (`StandardScaler`) and encoders (`OneHotEncoder`) were fitted exclusively on the Training fold and only applied to validation/test folds."

### Q5: What are the DERIVED FEATURES and how were they calculated?
**Answer:**  
"We engineered 10 genuine within-trial telemetry features documented in `FEATURE_DICTIONARY.csv`:
1. `DERIVED_error_rate`: $\frac{\text{error\_count}}{\max(1, \text{tool\_calls\_total})}$
2. `DERIVED_retry_ratio`: $\frac{\text{retry\_count}}{\max(1, \text{error\_count})}$
3. `DERIVED_edit_churn`: $\frac{\text{patch\_size\_lines}}{\max(1, \text{unique\_files\_edited})}$
4. `DERIVED_tool_entropy`: Shannon entropy $-\sum p_i \log p_i$ over tool call sequence.
Our SHAP analysis proved that `DERIVED_edit_churn` and `DERIVED_tool_intensity` are the top two most predictive failure features."

---

## 3. Modeling & Comparative Evaluation Defense

### Q6: Why include both classical models and neural / multi-view fusion models?
**Answer:**  
"To satisfy the AUCML Rubric requirement for a justified model progression. We started with **Logistic Regression** as our linear baseline (Test Accuracy: 77.71%, Macro-F1: 0.7133). We progressed to non-linear tree ensembles (**Random Forest** Accuracy: 84.27%, ROC-AUC: 0.8632 and **XGBoost** Macro-F1: 0.7659), then a sequential **BiLSTM** on tool-call trajectories (Accuracy: 79.71%), and finally our **Proposed Multi-View Fusion Network** (Gated Attention combining telemetry + trajectory sequence). This allows us to empirically verify whether fusion improves performance over single views."

### Q7: Did the Proposed Multi-View Fusion model actually beat the baselines?
**Answer:**  
"We do not make the simplistic claim that the fusion model beats all baselines across every metric. The experimental trade-off is clear:
- **XGBoost** achieves the highest balanced discrimination (**Macro-F1: 0.7659**, **PR-AUC: 0.7132**).
- **Random Forest** achieves the highest overall accuracy (**0.8427**) and ROC-AUC (**0.8632**).
- **Proposed Multi-View Fusion Network** achieves the highest **Failure-Class Recall (0.7295)**. Compared to Random Forest (57.95%), fusion catches **+15.0 percentage points more failures (a 25.9% relative improvement)**. In SRE reliability monitoring, False Negatives (missing a silent agent failure) are far costlier than False Positives."

### Q8: What does the system predict on the Who&When attribution benchmark?
**Answer:**  
"On `Who&When`, the system addresses two tasks:
1. **Component Attribution:** Predicts which agent (WebSurfer, Orchestrator, Assistant, FileSurfer) caused the breakdown with **60.44% Top-1 Accuracy** and **88.16% Top-2 Accuracy** using 3-fold cross-validation.
2. **Temporal Fault Localization:** Predicts the decisive failure turn with a Top-3 hit rate of **33.33%** and a Mean Absolute Error of **6.25 turns** across long trajectories."

### Q9: Does the Live Demo use the same pipeline as training?
**Answer:**  
"Yes. We proved this mathematically in `LIVE_DEMO_VALIDATION.md` and `src/evaluation/test_live_pipeline_consistency.py`. The live diagnostic engine takes a raw telemetry dictionary, passes it through the exact same 10 derived feature formulas, the identical saved `preprocessor.joblib`, and the trained model artifacts in `models_saved/`. The probabilities match offline inference with discrepancy $\Delta < 0.00005$."
