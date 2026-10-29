# Multimodal AI for Agentic System Failure Diagnosis and Root-Cause Reasoning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course: AUCML](https://img.shields.io/badge/Course-AUCML--TE7483-blue.svg)](https://sitpune.edu.in)
[![Institution: SIT Pune](https://img.shields.io/badge/Institution-Symbiosis%20Institute%20of%20Technology-800000.svg)](https://sitpune.edu.in)

> **Academic Course Project & Research Framework**  
> **Course:** Applications and Use Cases of Machine Learning (AUCML — TE7483)  
> **Institution:** Symbiosis Institute of Technology (SIT), Pune  
> **Focus:** Multimodal failure detection, behavioral signature profiling, and telemetry-guided root-cause attribution for autonomous AI agents.

---

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Contributions](#contributions)
5. [System Architecture](#system-architecture)
6. [Dataset](#dataset)
7. [Dataset Provenance and License](#dataset-provenance-and-license)
8. [Data Preprocessing](#data-preprocessing)
9. [Feature Engineering](#feature-engineering)
10. [Model Architecture](#model-architecture)
11. [Experimental Setup](#experimental-setup)
12. [Results](#results)
13. [Failure Diagnosis and Attribution](#failure-diagnosis-and-attribution)
14. [Interactive Applications](#interactive-applications)
15. [Project Structure](#project-structure)
16. [Installation](#installation)
17. [Usage](#usage)
18. [Reproducibility](#reproducibility)
19. [Limitations](#limitations)
20. [Future Work](#future-work)
21. [Citation](#citation)
22. [License](#license)

---

## Overview

As Large Language Model (LLM) agents are increasingly entrusted with autonomous multi-step software development, environment navigation, and tool execution, runtime reliability remains a major barrier to production deployment. Unlike deterministic software, autonomous agents fail silently through complex behavioral pathologies, including cyclic retry loops (thrashing), context drift, environment permission exhaustion, and unconstrained exploration.

This project introduces a **multimodal machine learning framework** that jointly models **scalar execution telemetry** (error counters, durations, code churn, turn statistics) and **sequential tool invocation trajectories** (Markovian tool transitions) to:
1. Accurately detect task execution failures before final verifier execution.
2. Characterize root-cause failure signatures (Retry Thrashing, Environment Crash, High Churn, Over-Exploration).
3. Attribute failures to specific execution turns and tool calls to accelerate human-in-the-loop debugging.

---

## Problem Statement

Given an autonomous agent execution trial $\mathcal{T} = (\mathbf{x}_{\text{tabular}}, \mathbf{s}_{\text{sequence}})$, where $\mathbf{x}_{\text{tabular}} \in \mathbb{R}^{d}$ denotes cumulative execution telemetry and $\mathbf{s}_{\text{sequence}} = (t_1, t_2, \dots, t_L)$ denotes the discrete sequence of invoked tools across turns:
1. **Binary Failure Classification:** Predict whether the agent trial resulted in task failure ($y \in \{0, 1\}$), where $y=1$ denotes that the agent failed to satisfy the task verifier.
2. **Behavioral Attribution:** Infer the underlying failure mechanism and identify the decisive turn $k \in \{1, \dots, L\}$ responsible for task disruption.

Traditional static analysis and rule-based thresholding struggle because agent failure is emergent: an error count of 3 may be nominal during exploratory test execution, but catastrophic during initial environment configuration.

---

## Motivation

Existing observability tools treat LLM agents as standard web services, monitoring request latency and token usage while ignoring agent-specific execution dynamics:
- **Tabular Models (XGBoost, Random Forest)** capture static statistical aggregations (e.g., error rate, churn) but lose the temporal ordering of actions.
- **Pure Sequence Models (RNNs, LSTMs)** capture temporal tool order but neglect continuous resource consumption and aggregate environment feedback.
- **Multimodal Fusion** addresses this gap by combining continuous telemetry with sequence dynamics through a gated cross-modal mechanism, allowing the model to dynamically weight telemetry versus sequence context depending on the failure mode.

---

## Contributions

1. **Leakage-Safe Benchmarking on 11,995 Real Trials:** Evaluated classical and neural architectures on real software engineering agent executions across 61 benchmarks with strict grouped task splitting (zero cross-task leakage).
2. **Domain-Specific Feature Engineering:** Formulated 10 derived behavioral metrics, including Shannon tool entropy, retry ratios, and normalized edit churn.
3. **Cross-Modal Gated Fusion Architecture:** Designed a two-stream deep neural network that fuses dense tabular telemetry embeddings with bidirectional LSTM sequence representations using a learnable gating mechanism.
4. **Honest Multi-Metric Evaluation:** Demonstrated that while tree ensembles (XGBoost, Random Forest) achieve the strongest overall accuracy and Macro-F1 on tabular signals, the Multimodal Fusion Network achieves the highest **Failure Recall ($72.95\%$)**, reducing undetected agent failures by $+15.0$ percentage points over Random Forest.
5. **Interactive Observability Applications:** Implemented both a full-featured Flask/SVG Trace Canvas observability platform and a Streamlit research console with live session recording, sandbox simulation, and trace deep-linking.

---

## System Architecture

```text
                               AGENT EXECUTION TRIAL
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     Structured Telemetry (x)                       Tool Sequence (s)
[Turns, Errors, Churn, Duration, Entropy]          [read_file → grep → bash → ...]
                 │                                               │
                 ▼ (StandardScaler + OneHot)                     ▼ (Embedding + Padding)
      Dense Feature Vector (R^88)                     Token Matrix (R^{50 x 32})
                 │                                               │
                 ▼                                               ▼
       Tabular MLP Encoder                            BiLSTM Sequence Encoder
    Linear(88→128) → BN → ReLU                      BiLSTM(32→32, Bidirectional)
         Linear(128→64)                               Max-Pooling Over Time
                 │                                               │
                 ▼                                               ▼
      Telemetry Embedding (h_tab)                     Sequence Embedding (h_seq)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                            Cross-Modal Gated Fusion
                         α = σ(W_g · [h_tab ; h_seq])
                         h_fused = α · h_tab + (1 - α) · h_seq
                                         │
                                         ▼
                                  Classifier Head
                             Linear(64→32) → ReLU → Linear(32→1)
                                         │
                                         ▼
                        Failure Probability P(Failure | x, s)
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 Thresholding (τ=0.46)          Root-Cause Attribution
                 [HEALTHY / FAILURE]           [Thrashing, Churn, Crash]
```

---

## Dataset

The primary supervised dataset is derived from the **Agent Diagnostics Observatory**, capturing end-to-end execution trajectories of autonomous coding agents operating on standardized development benchmarks.

### Summary Statistics
| Attribute | Value | Notes |
|:---|:---|:---|
| **Total Records** | **11,995** | Complete verified agent execution trials |
| **Unique Tasks** | 1,200+ | Distinct coding and debugging problems |
| **Source Benchmarks** | 61 | `openhands`, `sdlc_fix`, `org_migration`, `sdlc_feature`, etc. |
| **Agent Frameworks** | Multiple | OpenHands, SWE-Agent, Aider, Modular ReAct |
| **Foundation Models** | Multiple | Claude-3.5-Sonnet, Claude-Haiku, Claude-Opus, GPT-4o |
| **Target Variable** | `is_failed` | Binary ($0$: Task Succeeded, $1$: Task Failed) |
| **Success Count (`is_failed = 0`)** | 9,231 ($76.96\%$) | Verified by sandbox test suite pass |
| **Failure Count (`is_failed = 1`)** | 2,764 ($23.04\%$) | Test suite failed, rate limited, or crashed |
| **Class Imbalance** | 3.34 : 1 | Handled via threshold calibration & balanced weights |

---

## Dataset Provenance and License

To ensure academic reproducibility, all data sources and licenses are documented and verified:

1. **Primary Supervised Dataset — Agent Diagnostics (`signals.parquet`):**
   - **Repository:** [https://github.com/sjarmak/agent-diagnostics](https://github.com/sjarmak/agent-diagnostics)
   - **Commit Hash:** `0206772a2a7e536951972e9f336d0c0b96461fc9`
   - **License:** Apache License 2.0
   - **File Size:** 1.48 MB (`data/raw/agent_diagnostics/signals.parquet`)
   - **Integrity SHA-256:** `4cea2359787900e8fc20e0e583b21763da99f46d99868cf7f18a1877c9425222`

2. **Temporal Attribution Benchmark — Who&When:**
   - **Repository:** [https://github.com/ag2ai/Agents_Failure_Attribution](https://github.com/ag2ai/Agents_Failure_Attribution)
   - **Reference:** *Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems* (ICML 2025 Spotlight)
   - **License:** MIT License
   - **File Size:** 0.6 MB (`who_and_when_ag.parquet`), 1.2 MB (`who_and_when_hc.parquet`)
   - **Role:** Evaluates turn-level failure attribution and multi-agent fault localization.

---

## Data Preprocessing

### Strict Leakage Prevention via Grouped Partitioning
Standard random train/test splits cause severe optimistic bias because multiple trials solve the same underlying software task. To guarantee zero data leakage, we utilize **`GroupShuffleSplit` on `task_id`**:
- All trials solving the same task are isolated strictly into the same split.
- **Cross-task overlap is verified to be 0%** ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$).

| Split | Sample Count | Unique Tasks | Failure Rate |
|:---|:---:|:---:|:---:|
| **Train** | **8,298** ($69.2\%$) | 845 | $22.95\%$ |
| **Validation** | **1,701** ($14.2\%$) | 172 | $23.16\%$ |
| **Test** | **1,996** ($16.6\%$) | 869 | $22.04\%$ |

### Feature Transformation
All transformations are **fitted exclusively on the training split** and applied out-of-sample:
1. **Numerical Telemetry (21 features):** Scaled using `StandardScaler()`.
2. **Categorical Metadata (3 features: `model`, `agent_name`, `benchmark`):** Transformed using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` producing 67 one-hot indicators.
3. **Sequence Pipeline:** Tool call sequences are mapped to integer token indices using a vocabulary constructed from training tool names (`<PAD>: 0`, `<UNK>: 1`), truncated/padded to $L=50$.

---

## Feature Engineering

In addition to 11 raw telemetry variables, we engineer **10 domain-specific behavioral features** derived strictly from trial execution:

| Feature Name | Mathematical Definition | Domain Motivation |
|:---|:---|:---|
| `DERIVED_error_rate` | $\frac{\text{error\_count}}{\max(1, \text{tool\_calls\_total})}$ | Frequency of tool execution failures per action. |
| `DERIVED_retry_ratio` | $\frac{\text{retry\_count}}{\max(1, \text{error\_count})}$ | Proportion of errors triggering immediate command retries. |
| `DERIVED_search_to_edit_ratio` | $\frac{\text{search\_tool\_calls}}{\max(1, \text{edit\_tool\_calls})}$ | Balances exploration (search/grep) vs exploitation (code edit). |
| `DERIVED_edit_churn` | $\frac{\text{patch\_size\_lines}}{\max(1, \text{unique\_files\_edited})}$ | Average code modifications per edited file (detects thrashing). |
| `DERIVED_tool_intensity` | $\frac{\text{tool\_calls\_total}}{\max(1, \text{total\_turns})}$ | Action density per conversational turn. |
| `DERIVED_read_to_edit_ratio` | $\frac{\text{unique\_files\_read}}{\max(1, \text{unique\_files\_edited})}$ | Information intake relative to code modification. |
| `DERIVED_time_per_turn` | $\frac{\text{duration\_seconds}}{\max(1, \text{total\_turns})}$ | Latency per turn (detects hangs or timeout thrashing). |
| `DERIVED_tool_entropy` | $-\sum_{i} p_i \ln p_i$ | Shannon entropy over the tool sequence; low entropy ($<0.5$) indicates repetitive loops. |
| `DERIVED_log_duration` | $\ln(1 + \text{duration\_seconds})$ | Stabilizes extreme right-skewness of execution duration. |
| `DERIVED_log_patch_size` | $\ln(1 + \text{patch\_size\_lines})$ | Stabilizes extreme right-skewness of patch size lines. |

---

## Model Architecture

We systematically evaluate five models representing diverse inductive biases:

1. **Logistic Regression (Baseline):** Linear classifier with $L_2$ regularization on tabular telemetry; provides a baseline for linear separability.
2. **Random Forest (Balanced):** Ensemble of 200 decision trees trained with balanced class weights; handles nonlinear threshold interactions.
3. **XGBoost Classifier:** Gradient boosted decision trees optimized with histogram-based splitting; state-of-the-art benchmark for tabular data.
4. **Sequential BiLSTM:** 2-layer Bidirectional LSTM processing the discrete tool sequence only ($L=50$, embedding dimension $32$, hidden dimension $64$) with max-pooling over time; isolates purely sequential information.
5. **Proposed Multimodal Cross-Modal Attention Fusion Network:** Dual-stream architecture combining a tabular MLP encoder ($88 \to 128 \to 64$) and a sequence BiLSTM encoder ($50 \to 32 \to 64$), unified through a learnable gating mechanism:
   $$\alpha = \sigma\left(\mathbf{W}_g [\mathbf{h}_{\text{tab}} \,;\, \mathbf{h}_{\text{seq}}] + \mathbf{b}_g\right)$$
   $$\mathbf{h}_{\text{fused}} = \alpha \odot \mathbf{h}_{\text{tab}} + (1 - \alpha) \odot \mathbf{h}_{\text{seq}}$$

---

## Experimental Setup

- **Hardware:** Intel Core i7 / NVIDIA RTX GPU / CPU fallback enabled.
- **Evaluation Set:** 1,996 completely unseen test trials (zero task overlap with train/val).
- **Decision Threshold:** Calibrated on the validation set ($\tau = 0.46$) to optimize failure detection under class imbalance.
- **Metrics Tracked:** Accuracy, Precision, Failure Recall, Macro-F1, Weighted-F1, ROC-AUC, and PR-AUC.
- **Artifact Verification:** Predictions on the 1,996 test trials are persisted in [`data/processed/test_predictions.parquet`](file:///c:/Users/jayde/Downloads/AUCML/data/processed/test_predictions.parquet).

---

## Results

### Performance Comparison on 1,996 Unseen Test Trials
All reported numbers are verified directly from test predictions and evaluation logs:

| Model Architecture | Input Modality | Accuracy | Precision | Failure Recall | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | Tabular Telemetry | $0.7771$ | $0.4959$ | $0.6932$ | $0.7133$ | $0.7889$ | $0.8344$ | $0.6196$ |
| **Random Forest (Balanced)** | Tabular Telemetry | **0.8427** | **0.6641** | $0.5795$ | $0.7599$ | **0.8387** | **0.8632** | $0.7128$ |
| **XGBoost Classifier** | Tabular Telemetry | $0.8337$ | $0.6120$ | $0.6705$ | **0.7659** | $0.8363$ | $0.8595$ | **0.7132** |
| **Sequential BiLSTM** | Tool Sequence Only | $0.7971$ | $0.5314$ | $0.6727$ | $0.7293$ | $0.8050$ | $0.8318$ | $0.5936$ |
| **Proposed Multimodal Fusion Network** | **Telemetry + Tool Sequence** | $0.8081$ | $0.5487$ | **0.7295** | $0.7486$ | $0.8170$ | $0.8568$ | $0.6949$ |

### Scientific Trade-Off & Analysis

1. **XGBoost** achieves the highest **Macro-F1 ($0.7659$)** and **PR-AUC ($0.7132$)**, making it the best balanced classifier for structured telemetry.
2. **Random Forest** delivers the highest **Accuracy ($0.8427$)** and **ROC-AUC ($0.8632$)** with the highest specificity ($91.7\%$), but suffers from lower sensitivity ($57.95\%$ recall), missing 185 out of 440 failures.
3. **Proposed Multimodal Fusion Network** achieves the highest **Failure Recall ($72.95\%$)**, correctly catching 321 out of 440 failure cases (only 119 false negatives).
   - Absolute gain: **$+15.0$ percentage points** over Random Forest ($72.95\% - 57.95\%$).
   - Relative gain: **$+25.9\%$ relative improvement** in caught failures.
   - For mission-critical SRE and automated safety harnesses, this recall advantage is essential because silent failures carry disproportionately high operational cost.

### Confusion Matrices (1,996 Test Trials)
- **Random Forest:** True Normal: 1,427 | False Positive: 129 | False Negative: 185 | True Positive: 255
- **XGBoost:** True Normal: 1,369 | False Positive: 187 | False Negative: 145 | True Positive: 295
- **Multimodal Fusion:** True Normal: 1,292 | False Positive: 264 | False Negative: 119 | True Positive: 321

---

## Failure Diagnosis and Attribution

In this framework, **prediction** and **diagnosis** represent distinct capabilities:
- **Prediction:** The statistical model estimates failure likelihood $P(\text{failure} \mid \mathbf{x}, \mathbf{s})$.
- **Diagnosis:** The attribution engine performs trace-based evidence localization to identify the behavioral signature and pinpoint the responsible turn.

### Diagnostic Taxonomy & Signatures
1. **Cyclic Retry Thrashing:** Repeated execution of identical commands with escalating latency (detected via low tool entropy and high retry ratio).
2. **Environment & Sandbox Crash:** Permission denials, broken Docker mounts, or container exit errors (detected via early bash failures and high error rates).
3. **High Code Churn:** Excessive lines modified across files without progress (detected via high `DERIVED_edit_churn`).
4. **Information Over-Exploration:** Unbounded file searches and directory walks without code modification (detected via elevated search-to-edit ratio).
5. **Tool Instability:** Frequent non-zero exit codes across diverse tool types.

---

## Interactive Applications

The repository includes two fully functional interfaces:

### 1. Modern Agent Observability Dashboard (Flask)
```bash
python run.py
```
- Available at `http://127.0.0.1:8502`
- **Interactive Trace Canvas:** Bounded execution DAG with Zoom In, Zoom Out, and Fit controls.
- **7-Card Scenario Selection Grid:** One-click evaluation of healthy, thrashing, crash, and churn scenarios.
- **Temporal Telemetry Charts:** Latency area curves with failure pin markers and tool frequency distributions.
- **Forensic Assistant & Raw JSON View:** Explains what happened, what went wrong, and recommended resolutions.
- **History Deep-Linking:** Inspects saved historical traces with exact state restoration.

### 2. Streamlit Research Console
```bash
streamlit run app/main.py
```
- Multi-page evaluation suite with benchmark comparison tables, SHAP feature importance plots, and diagnostic simulators.

---

## Project Structure

```text
multimodal-agent-failure-diagnosis/
│
├── README.md                               # Academic research README (this document)
├── LICENSE                                 # MIT License
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusion rules
├── run.py                                  # Flask entrypoint (Port 8502)
│
├── app/                                    # Streamlit Research Console
│   └── main.py                             # Streamlit entrypoint
│
├── configs/                                # Experiment and hyperparameter configs
│
├── data/                                   # Data directory (see data/README.md)
│   ├── README.md                           # Dataset documentation
│   ├── demo_traces/                        # 7 curated scenario traces
│   ├── processed/                          # Train/val/test splits & preprocessors
│   └── raw/                                # Raw benchmarks (signals.parquet, who&when)
│
├── ml/                                     # Machine Learning Pipeline Modules
│   ├── data_loader.py                      # Data ingestion & schema validation
│   ├── features.py                         # Derived feature definitions
│   ├── preprocessing.py                    # ColumnTransformer pipeline
│   ├── models/                             # Model class definitions
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── bilstm.py
│   │   └── fusion_model.py
│   ├── training/                           # Model training scripts
│   ├── evaluation/                         # Metric calculations & confusion matrices
│   ├── explainability/                     # Feature importance & SHAP attribution
│   └── inference/                          # Production inference engine
│
├── models_saved/                           # Serialized model weights & checkpoints
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── sequence_bilstm.pt
│   └── multimodal_fusion.pt
│
├── results/                                # Empirical results & figures
│   ├── comparisons/                        # Model comparison CSV tables
│   ├── metrics/                            # JSON test metrics
│   └── plots/                              # Confusion matrices, ROC curves, feature plots
│
├── src/                                    # Core Platform & Services
│   ├── database/                           # SQLite database & history storage
│   ├── live_tracing/                       # Agent telemetry collector
│   ├── services/                           # Diagnosis & scenario registry
│   ├── ui/                                 # UI view components
│   └── web/                                # Flask application & REST API routes
│
├── templates/                              # Jinja2 HTML templates
│   ├── diagnose.html                       # Trace Canvas & Scenario Grid
│   ├── dashboard.html                      # System Overview
│   ├── model_evaluation.html               # Comparative Benchmark View
│   └── history.html                        # Diagnostic Trace History
│
├── static/                                 # Static assets (CSS, JS)
│
└── tests/                                  # Automated Test Suites
    ├── test_final_ui.py                    # Comprehensive UI & routing tests (15/15)
    ├── test_flask_app.py                   # Integration & API contract tests (12/12)
    └── test_ui_smoke.py                    # Telemetry & smoke tests (14/14)
```

---

## Installation

### Prerequisites
- Python 3.10 or 3.11
- Git

### Clone and Setup
```bash
git clone https://github.com/jaydevbutani/multimodal-agent-failure-diagnosis.git
cd multimodal-agent-failure-diagnosis

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Launch Observability Platform (Flask)
```bash
python run.py
```
Open [http://127.0.0.1:8502](http://127.0.0.1:8502) in your browser. Click **Demo Operator Access** to explore:
- Interactive trace DAG with zoom, pan, and fit.
- Clickable scenario cards for nominal and failure runs.
- Model selector to compare diagnostic outputs across all 5 architectures.

### 2. Launch Streamlit Research Console
```bash
streamlit run app/main.py
```

### 3. Run Automated Tests
```bash
# Run UI & routing test suite
python -m pytest tests/test_final_ui.py -q

# Run Flask integration & API suite
python tests/test_flask_app.py

# Run telemetry smoke test suite
python tests/test_ui_smoke.py
```

---

## Reproducibility

To re-run the full pipeline from raw data to model evaluation:

```bash
# 1. Preprocess raw data and create grouped splits
python src/preprocessing/prepare_data.py

# 2. Train classical baseline models (LR, RF, XGBoost)
python src/models/classical_models.py

# 3. Train deep sequence and multimodal models (BiLSTM, Fusion)
python src/models/deep_models.py

# 4. Generate comparison tables and verify test metrics
python src/evaluation/verify_predictions.py
```

---

## Limitations

1. **Domain Focus:** Evaluated primarily on software engineering and coding agent benchmarks (OpenHands, SWE-bench). Generalization to embodied robotics or multimodal vision agents requires domain-specific telemetry features.
2. **Fixed Sequence Window:** The BiLSTM branch currently truncates tool sequences to $L=50$. Ultra-long agent trajectories ($>200$ turns) may benefit from hierarchical recurrence or state-space models (Mamba/S4).
3. **Correlation vs Causality:** While telemetry signals (e.g., error rate, churn) strongly correlate with failure, they represent behavioral manifestations rather than formal causal proofs.

---

## Future Work

- **Dynamic Sequence Adaptation:** Incorporate state-space models (Mamba) to support unbounded turn trajectories without memory explosion.
- **Online Failure Interventions:** Deploy the model as an active circuit-breaker in agent runtimes to pause execution, prune retry thrashing, and prompt human-in-the-loop intervention before token budgets are exhausted.
- **Cross-Agent Zero-Shot Generalization:** Test transferability to emerging proprietary agentic harnesses without retraining.

---

## Citation

If you find this work useful in your research or educational projects, please cite:

```bibtex
@misc{butani2026multimodal,
  author = {Jaydev Butani},
  title = {Multimodal AI for Agentic System Failure Diagnosis and Root-Cause Reasoning},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jaydevbutani/multimodal-agent-failure-diagnosis}},
  institution = {Symbiosis Institute of Technology (SIT), Pune}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
The underlying `Agent Diagnostics` dataset is licensed under [Apache License 2.0](https://github.com/sjarmak/agent-diagnostics).  
The `Who&When` benchmark is licensed under [MIT License](https://github.com/ag2ai/Agents_Failure_Attribution).
