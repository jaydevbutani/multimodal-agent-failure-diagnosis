# Dataset Card: AI Agent Execution Failure Observatory

## 1. Dataset Summary
- **Dataset Name:** Agent Diagnostics Observatory Signals (`signals.parquet`) + Who&When Attribution Benchmark
- **Primary Source:** `https://github.com/sjarmak/agent-diagnostics`
- **Secondary Source:** `https://github.com/ag2ai/Agents_Failure_Attribution` (ICML 2025 Spotlight)
- **Validation Source:** `https://github.com/uw-syfi/TraceLab` (UW-SyFI)
- **Domain:** Autonomous AI Agent Reliability, Coding Agents, Multi-Agent Tool Telemetry
- **Primary Task:** Automated Detection of Agent Execution Failures, Diagnostic Profiling, and Temporal/Component Attribution

---

## 2. Dataset Structure & Dimensions

### Primary Supervised Dataset: Agent Diagnostics
- **Total Records:** 11,995 verified execution trials
- **Unique Tasks:** 1,200+ distinct task problems across 61 benchmarks
- **Total Features:** 32 columns (tabular execution telemetry, tool call sequences, metadata, verifier outcomes)
- **Outcome Target (`passed`):**
  - `True` (Task Succeeded): **9,231 trials (76.96%)**
  - `False` (Task Failed): **2,764 trials (23.04%)**
  - Imbalance Ratio: 3.34 : 1
- **Continuous Score (`reward`):** Range $[0.0, 1.0]$, mean $0.483$, standard deviation $0.370$.

### Multi-Agent Benchmark: Who&When
- **Total Trajectories:** 184 full multi-agent task executions (58 Hand-Crafted Magentic-One + 126 Algorithm-Generated CaptainAgent)
- **Total Turns:** 4,092 execution turns (mean: 22.2 turns per task)
- **Target Attributes:** `mistake_step` (temporal fault index), `mistake_agent` (failing agent role), `mistake_reason` (ground-truth explanation)

---

## 3. Data Collection & Provenance
1. **Agent Diagnostics:** Captured from real open-source and proprietary agent runs executing on standardized software development benchmarks (`openhands`, `sdlc_fix`, `org_migration`, `sdlc_feature`, `org_security`, etc.) across multiple frontier LLMs (`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`). All outcomes are validated by sandbox test suite execution.
2. **Who&When:** Trajectories generated on GAIA and AssistantBench benchmarks, with dual-annotator human verification identifying the decisive mistake step and responsible agent.
3. **TraceLab:** Telemetry logged from in-the-wild developer coding sessions utilizing Claude Code and Codex command executions.

---

## 4. Ethical Considerations & Privacy
- All datasets are released under open-source licenses (Apache 2.0, MIT, CC BY 4.0).
- TraceLab and Agent Diagnostics strip private tokens, credentials, and API keys.
- No synthetic hallucinations are presented as empirical facts.

---

## 5. Intended Use & Limitations
- **Intended Use:** Training and evaluating Machine Learning models for failure detection, error pattern classification, and telemetry-based root-cause diagnosis.
- **Limitations:** Focuses primarily on software engineering, web surfing, and API-interaction agents. Generalization to purely physical robotics or unconstrained multi-modal vision agents requires further domain adaptation.
