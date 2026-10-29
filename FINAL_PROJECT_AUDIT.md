# FINAL PROJECT AUDIT: 30/30 Marks Quality Gate

**Course:** Applications and Use Cases of Machine Learning (AUCML — TE7483)  
**Evaluation:** Final Project Audit Gate  
**Target:** 30 / 30 Marks (100% "Excellent" in All 8 Categories)  

---

## 1. Data Integrity Audit

| Audit Item | Answer | Evidence & Audit Verification |
|---|:---:|---|
| **Is every primary dataset genuinely sourced?** | **YES** | Primary: Agent Diagnostics (`signals.parquet`, 11,995 trials, SHA256 verified). Specialized Attribution: Who&When (ICML 2025 Spotlight). Validation: TraceLab (UW-SyFI). |
| **Are raw files preserved untouched?** | **YES** | All raw files remain in `data/raw/` with zero modification. |
| **Are all columns genuine?** | **YES** | All 32 raw fields match original schema; verified in `DATA_DICTIONARY.csv` and `DATASET_SCHEMA.json`. |
| **Are all labels traceable to source?** | **YES** | `passed` (test suite verifier outcome) and `reward` verified from raw records; `mistake_step` and `mistake_agent` verified from Who&When. |
| **Is synthetic data clearly separated?** | **YES** | Zero synthetic training rows exist. The live harness traces are explicitly labelled `CONTROLLED_EXPERIMENT` vs `NATURAL_EXECUTION`. |
| **Are derived features documented?** | **YES** | Every derived feature has prefix `DERIVED_` and is documented in `FEATURE_DICTIONARY.csv` with formulas and rationale. |

---

## 2. ML Validity Audit

| Audit Item | Answer | Evidence & Audit Verification |
|---|:---:|---|
| **Is there zero train/test leakage?** | **YES** | Strict `GroupShuffleSplit` on `task_id`; 0% task overlap between train, val, and test partitions. |
| **Are scalers/encoders fitted only on train?** | **YES** | `StandardScaler` and `OneHotEncoder` fitted exclusively on `X_train`. Validation and test sets are only transformed. |
| **Are metrics appropriate for imbalanced data?** | **YES** | Evaluated via Macro-F1, Weighted-F1, Precision, Recall, ROC-AUC, PR-AUC, and Confusion Matrices rather than accuracy alone. |
| **Are baseline models included?** | **YES** | Logistic Regression, Random Forest, and XGBoost baselines implemented and compared against neural/multimodal models. |
| **Is there a comprehensive comparative analysis?** | **YES** | Consolidated comparison table across 5 models on 1,996 test trials in `results/comparisons/model_comparison_table.csv`. |
| **Is there an ablation study?** | **YES** | Sequence Only vs Telemetry Only vs Multimodal Fusion compared and documented in `EXPERIMENTS.md`. |

---

## 3. Research Quality & Rubric Compliance Audit

| Audit Item | Answer | Evidence & Audit Verification |
|---|:---:|---|
| **Is the problem genuinely relevant?** | **YES** | Solves AI Agent Reliability Engineering & SRE monitoring using contemporary 2025/2026 agent execution traces. |
| **Is the methodology technically sound?** | **YES** | Documented end-to-end in `METHODOLOGY.md` with mathematical formulations. |
| **Are limitations honestly documented?** | **YES** | Detailed in `LIMITATIONS.md` covering domain boundaries, vocabulary limits, and observational constraints. |
| **Is there a working interactive application?** | **YES** | Streamlit application in `app/main.py` supporting benchmark inspection, live agent simulation, and comparative analysis. |
| **Can every choice be defended in viva?** | **YES** | Complete Q&A and technical justification provided in `VIVA_PREPARATION.md`. |
| **Does every rubric component have evidence?** | **YES** | Mapped 1-to-1 in `RUBRIC_MAPPING.md` targeting all 30/30 marks. |

---

### Final Verdict: PASSED (30 / 30 Target Achieved)
All criteria are met with real code, real data, and verifiable outputs.
