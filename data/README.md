# Dataset Provenance & Data Architecture

This directory contains the execution traces, derived behavioral metrics, and benchmark partitions utilized in **Multimodal AI for Agentic System Failure Diagnosis and Root-Cause Reasoning**.

---

## 1. Primary Supervised Dataset: Agent Diagnostics Observatory

- **Dataset Name:** Agent Diagnostics (`signals.parquet`)
- **Official Repository:** [https://github.com/sjarmak/agent-diagnostics](https://github.com/sjarmak/agent-diagnostics)
- **Source Commit:** `0206772a2a7e536951972e9f336d0c0b96461fc9`
- **License:** Apache License 2.0
- **Total Records:** 11,995 verified execution trials
- **Unique Tasks:** 1,200+ distinct coding tasks across 61 software development benchmarks
- **Target Variable:** `passed` (boolean) $\to$ mapped to `is_failed` ($0$ = Success, $1$ = Failure)
  - `is_failed == 0` (Task Succeeded): **9,231 trials (76.96%)**
  - `is_failed == 1` (Task Failed): **2,764 trials (23.04%)**
  - Class Imbalance: 3.34 : 1

### Data Collection & Provenance
The execution traces were generated across frontier language model backends (`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`) interacting with software engineering environments (`OpenHands`, `sdlc_fix`, `org_migration`, `sdlc_feature`, `org_security`). Each execution trial logs discrete turn metrics, tool invocation sequences, runtime error counters, retry attempts, modified patch lines, and final verification suite status.

---

## 2. Temporal Attribution Benchmark: Who&When

- **Dataset Name:** Who&When (`who_and_when_ag.parquet`, `who_and_when_hc.parquet`)
- **Official Repository:** [https://github.com/ag2ai/Agents_Failure_Attribution](https://github.com/ag2ai/Agents_Failure_Attribution)
- **Paper Reference:** *Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems* (ICML 2025 Spotlight)
- **License:** MIT License
- **Total Records:** 184 full multi-agent trajectories (4,092 turns)
- **Target Attributes:** `mistake_step` (fault index), `mistake_agent` (failing component/agent), `mistake_reason` (ground-truth root cause)

---

## 3. Directory Layout

```text
data/
├── README.md                               # This file
├── demo_traces/                            # 7 Curated failure scenario traces for live diagnosis
│   ├── 01_healthy_nominal.json
│   ├── 02_retry_thrashing.json
│   ├── 03_environment_failure.json
│   ├── 04_high_code_churn.json
│   ├── 05_over_exploration.json
│   ├── 06_tool_execution_failure.json
│   └── 07_compound_failure.json
├── processed/                              # Leakage-safe grouped splits and preprocessors
│   ├── train.parquet                       # Train set (8,298 samples, 0 task overlap)
│   ├── val.parquet                         # Validation set (1,701 samples, 0 task overlap)
│   ├── test.parquet                        # Test set (1,996 samples, 0 task overlap)
│   ├── test_predictions.parquet           # Precomputed model predictions on test split
│   ├── X_train.npy, X_val.npy, X_test.npy  # Scaled feature matrices (88 features)
│   ├── y_train.npy, y_val.npy, y_test.npy  # Target label arrays
│   ├── preprocessor.joblib                 # ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── feature_names.joblib                # 88 feature column headers
│   └── tool_vocab.joblib                   # Sequential vocabulary dictionary
└── raw/
    ├── agent_diagnostics/
    │   ├── signals.parquet                 # Primary raw dataset (1.48 MB)
    │   └── MANIFEST.json                   # SHA-256 integrity hashes & metadata
    └── who_when/
        ├── who_and_when_ag.parquet         # Algorithm-generated benchmark (0.6 MB)
        └── who_and_when_hc.parquet         # Hand-crafted benchmark (1.2 MB)
```

---

## 4. Leakage Prevention Methodology

All preprocessing and feature scaling are fitted **strictly on the training split** using grouped task splitting:
- Grouping Column: `task_id`
- Train / Val / Test Partition: **8,298 / 1,701 / 1,996**
- Overlap Verification: **0% task overlap** between splits ($\text{Train} \cap \text{Test} = \emptyset$).
