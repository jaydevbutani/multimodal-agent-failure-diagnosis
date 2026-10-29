# UI PRESENTATION & VERIFICATION CHECKLIST
**Heterogeneous Multi-View Trace Fusion for AI Agent Failure Diagnosis and Root-Cause Attribution**  
*AUCML TE7483 — Symbiosis Institute of Technology (SIT), Pune*

---

## 1. System Launch & Initialization Checklist

| Item | Verification Step | Expected Status |
| :--- | :--- | :---: |
| **Python Environment** | Active Python 3.11 environment with dependencies installed | ✅ Ready |
| **Artifacts Present** | `data/processed/preprocessor.joblib`, `models_saved/random_forest.joblib`, `models_saved/xgboost.joblib`, `models_saved/multimodal_fusion.pt` | ✅ Verified |
| **Data Partitions Present** | `data/processed/test.parquet` (1,996 records), `results/comparisons/model_comparison_table.csv` | ✅ Verified |
| **Streamlit App Launch Command** | `streamlit run app/main.py --server.port 8502` | ✅ Verified |
| **App Startup Time** | Loads cleanly in < 3 seconds; all cached resources cached via `@st.cache_resource` | ✅ Verified |

---

## 2. Section-by-Section Feature Verification Checklist

### Section 1: Overview & Model Performance
- [x] **Project Metadata:** Course (*AUCML TE7483*), Institution (*SIT Pune*), and clear one-sentence problem statement displayed at top.
- [x] **Dataset Metrics Cards:** Displays 11,995 total trials, 1,996 held-out test trials, 21.9% failure rate, and 88 engineered features.
- [x] **Grouped Splitting Statement:** Explicit note confirming GroupKFold grouped by `task_id` with 0% task leakage.
- [x] **Model Comparison Table:** All 5 models rendered with max values highlighted:
  - Random Forest: Accuracy **0.8427**, ROC-AUC **0.8632**
  - XGBoost: Macro-F1 **0.7659**, PR-AUC **0.7132**
  - Proposed Multi-View Fusion: Recall **0.7295** (+15.0% absolute / +25.9% relative gain over Random Forest)
- [x] **Balanced Metric Chart:** Grouped bar chart comparing Macro-F1, ROC-AUC, and PR-AUC.
- [x] **Failure Sensitivity Chart:** Highlight bar chart emphasizing the Fusion Network's 0.7295 failure recall.
- [x] **Feature Importance Chart:** Horizontal bar chart of top 10 features, with distinct orange labeling for `DERIVED FEATURE` and blue for original telemetry.
- [x] **Scientific Model Trade-Off Box:** Crystal-clear explanation of when to deploy XGBoost (balanced SRE monitoring) vs Fusion Network (mission-critical safety tripwire).

### Section 2: Interactive Trace Diagnosis
- [x] **Dual Benchmark Switch:** Radio buttons to toggle between Primary 11,995-trial benchmark and Who&When attribution benchmark.
- [x] **Outcome Filtering:** Dropdown allows filtering by *All Traces*, *Only True Failures*, or *Only True Passes*.
- [x] **Trace Selector:** Dropdown displays Task ID, Model, and Ground Truth.
- [x] **Side-by-Side Verification Cards:**
  - 🤖 **Model Prediction Output:** Predicted status, failure probability, confidence, active model name, and threshold cutoff.
  - 🏷️ **Ground-Truth Dataset Annotation:** Actual benchmark status, task ID, and benchmark origin.
  - 🎯 **Concordance:** Visual indicator showing whether the prediction is *Correct Match*, *False Alarm*, or *Missed Failure*.
- [x] **Execution Telemetry:** JSON block with turns, tool calls, errors, retries, patch lines, code churn, retry ratio, and tool entropy.
- [x] **Tool Sequence Flow:** Visual chronological string of invoked tools.
- [x] **Diagnostic Evidence Signals:** Plain English bullet points flagging retry thrashing, high error rates, or code churn.
- [x] **Who&When Attribution View:**
  - Displays validated 3-fold cross-validation metrics (Top-1: 60.44%, Top-2: 88.16%, MAE: 6.25 turns).
  - Trajectory selector showing responsible agent and mistake turn.
  - Ground-truth root-cause explanation.
  - Collapsible multi-agent turn history highlighting the decisive fault turn.

### Section 3: Live Agent Demonstration
- [x] **Controlled Scenarios:** Dropdown with 3 scenarios: Nominal Healthy, Command Thrashing Failure, and Information Over-Exploration.
- [x] **Execution Button:** Single-click execution (`🚀 Execute Live Agent & Run Diagnosis`) with live spinner.
- [x] **Live Telemetry Capture:** Generates real-time trace JSON via `LiveAgentTracer`.
- [x] **Live Model Inference:** Real-time diagnosis via `AgentFailureDiagnoser` using the saved preprocessor and models.
- [x] **Status & Evidence Cards:** Diagnostic status, failure probability, diagnostic confidence, and plain English evidence points.
- [x] **Actionable SRE Remediation:** Clear recommendation (e.g., interrupt retry loop, review diff blast radius).
- [x] **Consistency Guarantee Note:** Cites empirical validation ($\Delta < 0.00005$) proving offline-online schema consistency.

---

## 3. Step-by-Step Viva Presentation Script (5-Minute Walkthrough)

### Minute 1: The Problem & Dataset Provenance (Section 1)
- **Say:** *"Examiners, AI agents operating in autonomous environments frequently encounter silent failures, cyclic retry loops, and unconstrained code modifications that pass simple exit codes but break systems."*
- **Show:** The summary cards in Section 1. Point out the **11,995 authentic trials** from SWE-bench, WebArena, and GAIA. Emphasize that **zero rows are synthetic**. Point out that the test split of 1,996 trials used **GroupKFold grouped splitting by `task_id`**, guaranteeing 0% task leakage.

### Minute 2: Benchmark Results & Architecture Trade-Offs (Section 1)
- **Say:** *"We evaluated 5 models from Logistic Regression to our Proposed Multi-View Gated Fusion Network."*
- **Show:** The comparison table and dual bar charts.
- **Explain the Key Trade-off:**
  - If the operational objective is balanced general monitoring: **XGBoost** is optimal with **Macro-F1 of 0.7659** and **PR-AUC of 0.7132**.
  - If the operational objective is safety-critical failure interception: our **Proposed Multi-View Fusion Network** is superior with a **Failure Recall of 0.7295** (+15.0% absolute gain over Random Forest), detecting 321 out of 440 true failures.
- **Highlight Feature Importance:** Point out how `DERIVED_` features (edit churn, retry ratio, tool entropy) dominate the top predictive features.

### Minute 3: Interactive Trace Diagnosis (Section 2)
- **Say:** *"Let's look at a concrete unseen test trace in Section 2."*
- **Action:** Select "Only True Failures" and choose a failure trace.
- **Show:** Point to the two side-by-side cards:
  - Left card: **Model-Predicted Output** (Probability and failure status).
  - Right card: **Ground-Truth Dataset Annotation** (Actual outcome from benchmark).
- **Show Evidence:** Point to the diagnostic evidence: high error count, retry thrashing, and high edit churn.

### Minute 4: Who&When Attribution Localization (Section 2)
- **Say:** *"Detection is only half the problem; in multi-agent systems, engineers need to know Who failed, When they failed, and Why."*
- **Action:** Switch the radio button to *Who&When Multi-Agent Failure Attribution*.
- **Show:** The 3-fold cross-validation metrics (Top-1: 60.44%, Top-2: 88.16%, MAE: 6.25 turns).
- **Show:** A sample trajectory identifying the decisive mistake turn, responsible sub-agent, and ground-truth reasoning, expanding the conversation turns to show the highlighted failure point.

### Minute 5: Live Instrumentation & Operational Consistency (Section 3)
- **Say:** *"To prove our models generalize to live production systems without training-serving skew, Section 3 instruments an active agent run."*
- **Action:** Choose "Command Thrashing Failure" and click `🚀 Execute Live Agent & Run Diagnosis`.
- **Show:** The telemetry is captured in real-time, transformed via the exact offline preprocessor, and flagged with >85% failure probability, citing retry thrashing and recommending prompt intervention.
- **Conclude:** *"As verified in our test scripts, the feature drift between offline and live vectors is strictly $\Delta < 0.00005$, proving full end-to-end operational consistency."*
