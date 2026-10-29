# FINAL UI AUDIT REPORT
**Heterogeneous Multi-View Trace Fusion for AI Agent Failure Diagnosis and Root-Cause Attribution**  
*AUCML TE7483 — Symbiosis Institute of Technology (SIT), Pune*  
*Date of Audit: October 2026*

---

## 1. Executive Summary & Design Principle
To guarantee maximum rubric compliance, scientific validity, and defensibility during the final viva examination, the Streamlit interface (`app/main.py`) was audited under a strict rubric-first mandate:
> **"Eliminate all decorative, redundant, or misleading UI elements; ensure every displayed metric is directly connected to authentic saved artifacts; and organize the workflow into exactly 3 presentation-ready sections."**

Every feature in the application was evaluated against the criterion: *"Why does the examiner or user need to see this, and is the underlying data genuinely verifiable?"*

---

## 2. Feature Classification Table

| Feature / UI Component | Purpose in Diagnostic System | Working? | Data / Model Source | Audit Status | Justification / Action Taken |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **App Title & Header Context** | Immediate orientation for faculty: course, institution, problem statement | ✅ Yes | Static Metadata | **KEEP** | Retained at the top of the interface. |
| **Sidebar 3-Section Nav** | Single, intuitive navigational switch across the 3 core views | ✅ Yes | Streamlit `st.sidebar.radio` | **KEEP** | Restructured into exactly 3 numbered sections. |
| **Decision Threshold Slider** | Allows examiner to interactively explore sensitivity/specificity trade-offs | ✅ Yes | Dynamic probability threshold (0.20 - 0.80) | **KEEP** | Default set to 0.46 (optimal cutoff for Fusion model). |
| **Model Selection Dropdown** | Switch inference between Fusion, XGBoost, Random Forest, and Ensemble | ✅ Yes | `models_saved/` artifacts | **KEEP** | Demonstrates multi-model comparative capability live. |
| **Dataset Architecture Cards** | Summarize trial counts, held-out split, failure distribution, features | ✅ Yes | `data/processed/test.parquet` (11,995 total, 1,996 test) | **KEEP** | Replaces narrative text with clear empirical cards. |
| **Model Comparison Table** | Full benchmark comparison across 5 model architectures | ✅ Yes | `results/comparisons/model_comparison_table.csv` | **KEEP** | Dynamically rendered with max value highlighting; no hardcoded rows. |
| **Balanced Metrics Chart** | Visual comparison of Macro-F1, ROC-AUC, and PR-AUC | ✅ Yes | Plotly grouped bar chart from CSV | **KEEP** | Shows balanced discriminative ability clearly. |
| **Failure Recall Sensitivity Chart** | Visual proof that Fusion model detects +15.0% more failures | ✅ Yes | Plotly bar chart from CSV | **KEEP** | Directly defends the necessity of multi-view sequence fusion. |
| **Telemetry Feature Importance** | Gini feature importance highlighting `DERIVED_` features | ✅ Yes | `results/metrics/top_predictive_features.csv` | **KEEP** | Distinguishes original telemetry from engineered behavioral signals. |
| **Scientific Conclusion Summary** | Explicit guidance on which model to choose for which operational goal | ✅ Yes | Verified test set conclusions | **KEEP** | Answers faculty questions on model selection trade-offs. |
| **Trace Selection Dropdown** | Allows examiner to pick authentic benchmark execution sessions | ✅ Yes | `data/processed/test.parquet` | **KEEP** | 1,996 held-out trials with zero task leakage. |
| **Trace Status Filter** | Filter traces by True Failure vs True Pass | ✅ Yes | Dynamic DataFrame slice | **KEEP** | Enables testing edge cases and true failure traces. |
| **Model Prediction Card** | Display predicted class, probability, confidence, and cutoff | ✅ Yes | Real-time model inference | **KEEP** | Clean visual separation with alert badges. |
| **Ground-Truth Annotation Card** | Display actual benchmark outcome from dataset | ✅ Yes | Ground truth column `is_failed` | **KEEP** | Crucial: visually separates prediction from ground truth. |
| **Concordance Indicator** | Verify whether model prediction matches ground-truth | ✅ Yes | Match boolean logic (`is_pred == is_true`) | **KEEP** | Highlights correct detections, false alarms, and missed failures. |
| **Session Telemetry JSON** | Full tabular telemetry and derived metrics for the selected session | ✅ Yes | Raw session record | **KEEP** | Transparent inspection of raw data. |
| **Chronological Tool Trajectory** | Visual sequence of tools called during the session | ✅ Yes | `tool_call_sequence` column | **KEEP** | Demonstrates sequential tool input to the BiLSTM/Fusion branch. |
| **Rule-Based Evidence Signals** | Identify which telemetry indicators triggered the failure flag | ✅ Yes | Behavioral thresholds on churn, retries, and errors | **KEEP** | Provides explainable root-cause evidence. |
| **Who&When Attribution Metrics** | Quantified 3-fold CV scores (Top-1: 60.44%, Top-2: 88.16%, MAE: 6.25) | ✅ Yes | Empirical CV evaluation (`who_when_cv_evaluation.csv`) | **KEEP** | Proves the attribution module is validated, not anecdotal. |
| **Who&When Instance Explorer** | Inspect responsible agent, mistake turn, and reason | ✅ Yes | `data/raw/who_when/who_and_when_hc.parquet` | **KEEP** | Real ICML 2025 multi-agent trajectories. |
| **Turn-by-Turn Dialog Expander** | Collapsible conversation history highlighting the decisive error turn | ✅ Yes | Parsed JSON history from Who&When | **KEEP** | Shows exact fault turn in context. |
| **Live Scenario Selector** | Controlled scenarios (Healthy, Thrashing Failure, Over-Exploration) | ✅ Yes | `src/live_tracing/agent_tracer.py` | **KEEP** | Test harness for live evaluation. |
| **Live Execution & Diagnosis Button**| Triggers live tracing and real-time ML diagnosis | ✅ Yes | `AgentFailureDiagnoser` | **KEEP** | Fully functional live demonstration. |
| **Live Diagnostic Metrics & Telemetry**| Shows live probability, confidence, telemetry JSON, and evidence | ✅ Yes | Output of live inference pipeline | **KEEP** | Live proof of end-to-end functionality. |
| **Actionable SRE Remediation** | Specific recommended operational intervention based on failure mode | ✅ Yes | Automated rule dispatch on telemetry | **KEEP** | Bridges ML diagnosis with practical DevOps/SRE operations. |
| **Pipeline Consistency Notice** | Verification note proving $\Delta < 0.00005$ offline-online consistency | ✅ Yes | Reference to `test_live_pipeline_consistency.py` | **KEEP** | Defends against examiner questions on training-serving skew. |
| *Decorative Gauge Charts* | Uncalibrated circular gauges with arbitrary color cutoffs | ❌ No | Redundant CSS styling | **REMOVE** | Replaced with clean numeric cards and Plotly comparison bars. |
| *Simulated Multi-Agent Chat* | Mock dynamic conversation generator with synthetic dialog | ❌ Fake | Procedural string generator | **REMOVE** | Prohibited under data integrity guidelines. Only authentic traces used. |
| *Redundant Confusion Matrix Tabs* | Fragmented secondary tabs repeating test set numbers | ⚠️ Redundant| Duplicate evaluation CSV | **REMOVE** | Unified into the single, clear benchmark comparison table. |
| *Uncalibrated LLM Explainer* | Hallucinatory prompt querying external API for generic explanations | ❌ Unverified| External LLM call | **REMOVE** | Replaced with grounded feature attribution and verified Who&When annotations. |

---

## 3. Structural Simplification Verification
The application has been consolidated into exactly **3 comprehensive, highly-focused sections**:

1. **`1. 📊 Overview & Model Performance`**:
   - High-level problem statement and system scope.
   - Genuine dataset architecture (11,995 total trials, 1,996 independent test trials, 0% task leakage).
   - Dynamic comparison table across 5 model architectures.
   - Dual comparison charts: balanced performance (Macro-F1, ROC-AUC, PR-AUC) and failure sensitivity (Recall).
   - Top predictive feature importance chart (Random Forest Gini impurity) distinguishing derived behavioral features.
   - Scientific conclusion summarizing operational model selection trade-offs.

2. **`2. 🔬 Interactive Trace Diagnosis`**:
   - Dual source selector: Primary 11,995-trial benchmark vs Who&When ICML 2025 attribution benchmark.
   - Trace filtering by true outcome (All, Only Failures, Only Passes).
   - Side-by-side **Model Prediction Output** vs **Ground-Truth Dataset Annotation** with concordance status.
   - Session telemetry breakdown and chronological tool sequence trajectory.
   - Diagnostic evidence signals (churn, retry ratio, error frequency).
   - Empirical Who&When attribution view with verified 3-fold cross-validation performance.

3. **`3. ⚡ Live Agent Demonstration`**:
   - Controlled scenarios reflecting genuine agent failure archetypes (healthy nominal, command thrashing, over-exploration).
   - Live execution via `LiveAgentTracer` generating real-time telemetry.
   - Instant diagnostic inference via `AgentFailureDiagnoser` utilizing the exact saved preprocessor and models.
   - Captured telemetry, failure probability, confidence score, and actionable SRE recommendations.
   - Formal live pipeline consistency guarantee ($\Delta < 0.00005$) proving zero feature skew.

---

## 4. UI Error Handling & Resilience
- **Artifact Caching:** Wrapped in `@st.cache_resource` with try/except fallbacks to prevent reloading on user interactions.
- **Graceful Error Handling:** All prediction and trace extraction logic is encapsulated in try/except blocks with user-friendly `st.error()` messages; raw Python tracebacks are completely suppressed.
- **Data Integrity:** No hardcoded metrics, no fabricated rows, and no synthetic chat sessions exist anywhere in the UI code. Every metric directly references saved parquet and CSV artifacts.
