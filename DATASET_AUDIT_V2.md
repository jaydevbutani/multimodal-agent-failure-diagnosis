# DATASET AUDIT V2: Authoritative Source & Provenance Verification

**Project:** Heterogeneous Multi-View Trace Fusion for AI Agent Failure Detection & Root-Cause Reasoning  
**Course:** Applications and Use Cases of Machine Learning (AUCML — TE7483)  
**Date of Verification:** October 24, 2026  
**Auditor:** Lead ML Engineer  

---

## 1. Terminology & Provenance Definitions

To uphold strict scientific integrity in our AUCML evaluation, we establish explicit definitions:
1. **Public Benchmark Agent Execution Data:** Real execution traces recorded while autonomous agents execute standardized benchmark tasks in sandbox environments (e.g. OpenHands, SDLC benchmarks, GAIA). *Our primary and specialized datasets belong to this category.*
2. **In-the-Wild Developer Traces:** Telemetry recorded from human developers using agentic CLI tools (e.g. Claude Code, Codex in TraceLab).
3. **Production Logs:** Proprietary telemetry from closed commercial infrastructure (we do NOT make unverified claims that our benchmark data is internal enterprise production data).
4. **Synthetic Data:** Artificially generated mock samples. *No synthetic samples are used for model training in this project.*

---

## 2. In-Depth Dataset Verification Matrix

### 2.1 Primary Dataset: Agent Diagnostics Observatory
- **Official Source:** `https://github.com/sjarmak/agent-diagnostics`
- **Owner / Author:** `sjarmak`
- **Release Commit:** `0206772a2a7e536951972e9f336d0c0b96461fc9`
- **License:** Apache License 2.0
- **File Local Path:** `data/raw/agent_diagnostics/signals.parquet`
- **Verified SHA-256 Checksum:** `4cea2359787900e8fc20e0e583b21763da99f46d99868cf7f18a1877c9425222` (matches `MANIFEST.json`)
- **Total Records:** Exactly 11,995 trials.
- **Unique Tasks:** 5,793 unique benchmark tasks across 61 benchmark suites (`openhands`, `sdlc_fix`, `org_migration`, etc.).
- **Models Represented:** `claude-haiku-4-5-20251001` (6,443), `claude-sonnet-4-6` (4,564), `claude-opus-4-6` (677), `claude-opus-4-5-20251101` (253).
- **Agents Represented:** `claude-code` (9,421), `openhands` (2,483), `cursor-cli` (91).
- **Primary Objective Target:** `passed` (Boolean: 9,231 True [76.96%], 2,764 False [23.04%]).
- **Continuous Score:** `reward` (Float: mean $0.4832$, std $0.3701$).
- **Target Provenance:** Objective verification outcomes produced by sandbox unit test runners (`has_verifier_result == True`).
- **Failure Taxonomy:** Accompanied by `taxonomy_v3.yaml` (40 behavioral failure categories across 11 dimensions). Note: The taxonomy is an analytical framework and heuristic/classifier overlay; our primary supervised task is anchored on the objective ground truth `passed`.
- **Role in Project:** Primary supervised dataset for **Task A: Execution Failure Detection**.

### 2.2 Specialized Benchmark: Who&When (ICML 2025 Spotlight)
- **Official Source:** `https://github.com/ag2ai/Agents_Failure_Attribution` & `https://huggingface.co/datasets/Kevin355/Who_and_When`
- **Paper Reference:** *"Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems"* (ICML 2025 Spotlight)
- **License:** MIT License
- **Total Trajectories:** 184 full task executions (58 Hand-Crafted Magentic-One + 126 Algorithm-Generated CaptainAgent) comprising 4,092 execution turns.
- **SHA-256 Checksums:**
  - `who_and_when_ag.parquet`: `1d1720f4a8416ee0ed64bc13fa3f75655891e760a917924a071deac443133cf8`
  - `who_and_when_hc.parquet`: `d28ffe10b124b2ba46a662be3b438fbe0de7c4c4fcba4f46302c0a4b2575f62f`
- **Target Labels:**
  - `mistake_step`: Integer index indicating the decisive error step along the trajectory.
  - `mistake_agent`: Responsible failing agent role (`WebSurfer`, `Orchestrator`, `Assistant`, `FileSurfer`).
  - `mistake_reason`: Detailed natural language root-cause explanation.
- **Label Provenance:** Dual human annotator consensus on failure trajectories from GAIA and AssistantBench.
- **Role in Project:** Specialized benchmark for **Task C: Temporal Fault Localization** and **Task D: Responsible Component Attribution**.

### 2.3 Observational & Out-of-Domain Dataset: TraceLab (UW-SyFI)
- **Official Source:** `https://github.com/uw-syfi/TraceLab`
- **Release Tag:** `v2026-06-08-syfi-trace`
- **License:** Apache License 2.0 / CC BY 4.0
- **File Local Path:** `data/raw/tracelab/syfi_coding_trace.jsonl.gz` (53,601,226 bytes)
- **Verified SHA-256 Checksum:** `9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b`
- **Content:** 8,058 sessions, 665,453 agent steps from real developers using Claude Code and Codex.
- **Available Signals:** Step-level `is_error` (tool execution failure), `tool_wall_latency_ms`, token usage events.
- **Role in Project:** Observational validation of tool error distributions. It does NOT contain end-of-task verified pass/fail labels, and is therefore properly treated as an observational dataset rather than forced into supervised task classification.

---

## 3. Investigation of CMU Agent Trajectories (`cx-cmu/agent_trajectories`)

- **Author / HF ID:** `cx-cmu/agent_trajectories`
- **Recheck Finding:** Direct HTTP download requests to files such as `search.parquet` return:
  ```json
  {
    "X-Error-Code": "GatedRepo",
    "X-Error-Message": "Access to dataset cx-cmu/agent_trajectories is restricted. You must have access to it and be authenticated to access it. Please log in."
  }
  ```
- **Hugging Face API Metadata:** The dataset is formally tagged as `Gated: auto`.
- **Decision:** Because this dataset requires personal Hugging Face user login credentials and manual access approval, incorporating it as a primary pipeline dependency would violate the strict reproducibility requirement of the AUCML rubric (evaluators would encounter HTTP 401 errors unless they supplied an approved personal token). It is excluded from the primary pipeline for reproducibility.

---

## 4. Final Dataset Architectural Decision

```
Primary Supervised Training & Testing:   Agent Diagnostics (11,995 trials)
Specialized Root-Cause Attribution:       Who&When (184 trajectories, 4,092 turns)
Observational Cross-Domain Telemetry:     TraceLab (8,058 sessions)
Live Verification & Demonstration:        Instrumented Live Agent Harness (local execution)
```
This architecture provides 100% genuine data, full reproducibility, zero authentication friction, and clear separation of task scopes.
