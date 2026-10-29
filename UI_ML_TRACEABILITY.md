# UI TO ML PIPELINE TRACEABILITY MATRIX
**Heterogeneous Multi-View Trace Fusion for AI Agent Failure Diagnosis and Root-Cause Attribution**  
*AUCML TE7483 — Symbiosis Institute of Technology (SIT), Pune*

---

This document explicitly maps every visible user-interface component to its corresponding backend service function, machine learning artifact, and verifiable output.

| UI Component | Backend Service Function | Underlying Feature / Model Artifact | Verifiable Output |
| :--- | :--- | :--- | :--- |
| **User Sign In / Registration** | `authenticate_user()`, `register_user()` | SQLite `users` table via `werkzeug.security` (PBKDF2) | Authenticated session token in `st.session_state["user"]` |
| **Pipeline Flow Diagram** | `render_user_dashboard()` | Architectural representation of `DiagnosisService` | Visual display of trace processing steps |
| **Input Telemetry Grid** | `validate_and_normalize_trace()` | Raw telemetry features from `DATA_DICTIONARY.csv` | Total turns, tool calls, errors, retries, patch size, churn |
| **Tool Sequence Codebox** | `validate_and_normalize_trace()` | `tool_call_sequence` token list | Chronological arrow notation (`grep → edit → bash`) |
| **How Model Uses Input** | `render_user_diagnose()` | Mapping to Dense MLP + BiLSTM + Fusion network | Expandable technical educational card |
| **Failure Probability & Status** | `DiagnosisService.diagnose()` | `multimodal_fusion.pt` / `xgboost.joblib` | Exact float probability (e.g. `78.4%`) & Risk status |
| **Evidence Signals** | `generate_evidence_and_recommendation()` | Evaluated boolean thresholds on `DERIVED_` features | Error accumulation, retry ratio, code churn alerts |
| **Recommended Action** | `generate_evidence_and_recommendation()` | Rule-based dispatch policy | Specific operational SRE instructions |
| **Save Diagnosis Button** | `save_diagnostic_record()` | SQLite `diagnostic_results` table | Atomic record ID strictly tied to `user_id` |
| **Diagnostic History Table** | `get_user_diagnostic_history()` | SQLite `diagnostic_results` filtered by `user_id = ?` | Table of past runs belonging exclusively to the user |
| **Past Record Detail View** | `get_user_diagnostic_detail()` | SQLite query with `WHERE id = ? AND user_id = ?` | Complete inspection card of previous diagnosis |
| **Live Diagnosis Progress** | `run_sample_live_trace()` | `LiveAgentTracer` harness | Real-time 5-stage progress indicator |
| **Captured Live Telemetry** | `run_sample_live_trace()` | Output dictionary saved to `data/raw/live_traces/` | Live JSON breakdown of turns, errors, retries |
| **Model Benchmark Table** | `render_evaluator_console()` | `results/comparisons/model_comparison_table.csv` | Formatted comparison across 5 models with highlights |
| **Ablation Comparison** | `render_evaluator_console()` | Empirical evaluations from `EXPERIMENTS.md` | Sequence-only vs Telemetry-only vs Multi-View Fusion |
| **Who&When Fault Localization** | `render_evaluator_console()` | `who_and_when_hc.parquet` + 3-fold CV metrics | Top-1 (60.44%), Top-2 (88.16%), Turn MAE (6.25 turns) |
| **Feature Importance Chart** | `render_evaluator_console()` | `results/metrics/top_predictive_features.csv` | Plotly horizontal bar chart of Gini importances |
| **Viva Mode Defense Q&A** | `render_evaluator_console()` | 15 derived oral defense questions from `VIVA_PREPARATION.md` | Expandable Q&A accordion cards |
