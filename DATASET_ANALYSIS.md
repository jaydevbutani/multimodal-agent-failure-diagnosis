# Dataset Analysis & Profiling Report

## Project Title
**Multimodal AI for Agentic System Failure Diagnosis and Root-Cause Reasoning**  
**Course:** Applications and Use Cases of Machine Learning (AUCML - TE7483)  
**Institution:** Symbiosis Institute of Technology (SIT), Pune  

---

## 1. Dataset Identification & Sources

To support a rigorous, reproducible Machine Learning pipeline without fabricating execution traces or synthetic labels, this project integrates two verified, open-source agent execution benchmarks:

| Benchmark / Dataset | Primary Repository & Source | Paper / Reference | Modalities Captured | Role in Our Pipeline |
|---|---|---|---|---|
| **ToolFailBench Traces** | Hugging Face: `SoHarshh/toolfailbench-traces` | *ToolFailBench: A Diagnostic Benchmark for Tool-Use Failures in LLMs* | Textual prompts, tool telemetry, API mock returns, agent output text | **Task A:** Execution Failure Detection (Binary)<br>**Task B:** Failure Mode Classification (Multi-class) |
| **Who&When Benchmark** | GitHub: `ag2ai/Agents_Failure_Attribution`<br>Hugging Face: `Kevin355/Who_and_When` | *Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems* (ICML 2025 Spotlight) | Step-level sequential turns, agent speaker roles, tool/code execution traces, textual reasoning | **Task C:** Step-Level Temporal Fault Localization<br>**Task D:** Responsible Component Attribution<br>**Task E:** Root-Cause Grounded Reasoning |

---

## 2. Dataset 1: ToolFailBench (Execution Failure Diagnosis)

### 2.1 Overview & Schema
- **Total Samples:** 2,000 real agent execution traces evaluated across state-of-the-art backbones (`claude-haiku-4-5` and `gpt-5.4-mini`).
- **Domain Coverage:** 5 critical enterprise domains (400 samples each):
  1. Finance (e.g., P/E ratios, earnings, market telemetry)
  2. Medical (e.g., clinical guidelines, dosing, pharmacology)
  3. Legal (e.g., case law, compliance, statutes)
  4. Cybersecurity (e.g., CVE audits, port scans, access logs)
  5. Real Estate (e.g., property valuation, zoning, mortgages)

### 2.2 Features & Attributes
| Attribute Name | Data Type | Modality | Description |
|---|---|---|---|
| `task_id` | String (Categorical ID) | Metadata | Unique identifier for task instance (e.g., `RI-FIN-001`) |
| `domain` | String (Categorical) | Categorical Context | Operational domain (5 classes) |
| `turn_type` | String (Categorical) | Metadata | Turn interaction type (`single` vs `multi`) |
| `conflict_type` | String (Categorical) | Contextual Condition | Prompt condition (`none`, `contradicting`, `trap`) |
| `user_message` | Text (String) | Textual Modality | User objective / prompt input (mean length: 76.8 chars) |
| `num_tool_calls` | Integer (Discrete Numerical) | Execution Telemetry | Number of tools executed by agent (range: 0 to 8, mean: 0.73) |
| `has_tool_call` | Boolean (Binary Numerical) | Execution Telemetry | Whether tool calling was triggered (0 or 1) |
| `mock_tool_return` | JSON / Text | Execution Telemetry | Structured telemetry returned by external tools/APIs |
| `agent_answer` | Text (String) | Textual Modality | Agent generated output text (mean length: 970.2 chars) |
| `model_id` | String (Categorical) | System Context | Underlying model generator |

### 2.3 Target Variables & Class Distributions
1. **Binary Target (`is_failed`):**
   - `False` (Normal / Successful Execution): **1,536 samples (76.8%)**
   - `True` (Failed Execution): **464 samples (23.2%)**
   - *Class Imbalance Ratio:* ~3.3 : 1 (Handled via stratified splitting and class-weighted cross-entropy / focal loss).

2. **Multi-Class Target (`classification`):**
   - `correct` (Normal execution): **1,536 samples (76.8%)**
   - `tool_skip` (Agent failed to call mandatory tool): **256 samples (12.8%)**
   - `result_ignore` (Agent called tool but ignored returned telemetry): **108 samples (5.4%)**
   - `output_fabrication` (Agent hallucinated nonexistent data): **77 samples (3.85%)**
   - `wrong_answer` (Agent suffered reasoning / calculation failure): **23 samples (1.15%)**

### 2.4 Data Quality & Missing Value Assessment
- **Missing values:** 0 across all evaluated columns (`task_id`, `domain`, `classification`, `user_message`, `agent_answer`, `tool_calls`).
- **Data integrity:** Every row contains paired inputs, actual telemetry traces, and verified ground-truth labels.

---

## 3. Dataset 2: Who&When (Temporal Fault Localization & Component Attribution)

### 3.1 Overview & Schema
- **Benchmark Source:** ICML 2025 Spotlight (`ag2ai/Agents_Failure_Attribution`).
- **Total Multi-Agent Trajectories:** 184 full task executions (58 Hand-Crafted Magentic-One + 126 Algorithm-Generated CaptainAgent).
- **Total Execution Steps / Sequential Turns:** **4,092 turns** (average 22.2 turns per execution).
- **Core Strengths:** Contains step-by-step sequential interaction logs with human-verified annotations pinpointing:
  1. Exactly *when* the failure started (`mistake_step`).
  2. Exactly *which* agent / component was responsible (`mistake_agent`).
  3. Ground-truth natural language root-cause reasoning (`mistake_reason`).

### 3.2 Step-Level Sequential Representation
For each trajectory of length $T$, turns $t \in [0, T-1]$ are modeled as temporal sequences:
| Step Feature | Type | Description |
|---|---|---|
| `step_idx` | Integer | Position along the trajectory timeline ($t = 0, 1, \dots, T-1$) |
| `normalized_pos` | Float $\in [0, 1]$ | Relative temporal progression ($\frac{t}{T-1}$) |
| `role` / `agent` | Categorical | Active agent module (e.g., Orchestrator, WebSurfer, FileSurfer, Assistant) |
| `content_len` | Integer | Token/character volume of step output |
| `has_code` | Boolean | Whether step executed terminal / Python code |
| `content` | Text | Raw message, thought trace, tool action, or environment observation |

### 3.3 Component Localization Classes (`mistake_agent`)
In the Hand-Crafted system (Microsoft Magentic-One architecture):
- `WebSurfer`: **33 instances** (Autonomous web interaction / browser tool failures)
- `Orchestrator`: **18 instances** (Planning, sub-goal delegation, and synthesis failures)
- `Assistant`: **4 instances** (LLM code generation / response failures)
- `FileSurfer`: **3 instances** (Local file inspection and file extraction failures)

In the Algorithm-Generated system (CaptainAgent dynamic teams):
- `Verification_Expert`: **18 instances**
- `PythonDebugging_Expert`: **7 instances**
- `DataAnalysis_Expert`: **6 instances**
- `DataVerification_Expert`: **5 instances**
- `Validation_Expert`: **5 instances**
- Other domain specialists: **85 instances**

### 3.4 Temporal Failure Phase Distribution
Across the 2,993 evaluated hand-crafted steps:
- `pre_failure` (Normal collaborative execution): **888 steps (29.7%)**
- `failure_point` (The decisive critical mistake step $t^*$): **55 steps (1.8%)**
- `post_failure` (Cascading error propagation): **2,050 steps (68.5%)**

This mirrors real-world telemetry where a single silent fault cascades across subsequent agent turns until task abort.

---

## 4. Preprocessing & Feature Engineering Pipeline

### 4.1 Modality 1: Structured Telemetry & Execution Features
1. `num_tool_calls`: Scaled using `StandardScaler` (fitted on train split only).
2. `has_tool_call`: Binary indicator.
3. `user_msg_len`, `agent_ans_len`: Log-transformed ($\log(1 + x)$) and standard scaled.
4. `domain`: One-Hot Encoded with fixed vocabulary.
5. `conflict_type`: One-Hot Encoded.
6. Temporal sequence features (for sequential models):
   - Relative step index: $t / T$
   - Turn length dynamics: $\Delta \text{len}_t = \text{len}_t - \text{len}_{t-1}$
   - Speaker transition indicator: $I(\text{role}_t \neq \text{role}_{t-1})$

### 4.2 Modality 2: Textual Context & Error Representation
- Preprocessed text: Tokenized, lowercased, stripped of non-printable Unicode artifacts.
- Encoders:
  - TF-IDF + n-gram vectorizer ($n \in \{1, 2\}$, 500 max features) for Classical ML baseline.
  - Sentence-Transformer / Pretrained Transformer embeddings (`all-MiniLM-L6-v2` / `distilbert-base-uncased`, 384 / 768 dimensions) for deep models.

### 4.3 Multimodal Fusion
- Concatenation of textual embedding vector $\mathbf{e}_{text} \in \mathbb{R}^{d_{text}}$ with telemetry feature vector $\mathbf{x}_{telemetry} \in \mathbb{R}^{d_{num}}$ followed by non-linear projection and multi-head self-attention / MLP classification head.

---

## 5. Train / Validation / Test Strategy & Data Leakage Prevention

### 5.1 Strict Data Partitioning
- **Split Ratio:** 70% Train, 15% Validation, 15% Test.
- **Stratified Partitioning:** Stratified on target labels (`classification` and `mistake_agent`) to maintain identical class ratios across all partitions.
- **Grouped Task Splitting:** Partitioning is performed at the `task_id` / `question_ID` level so that multiple steps or turns from the same execution instance NEVER appear across different splits (guaranteeing zero cross-split leakage).

### 5.2 Transformation Isolation
- All encoders, scalers, imputers, and vectorizers are fitted **strictly on the Training split**.
- The validation and test splits are transformed using the fitted training parameters without re-fitting.
