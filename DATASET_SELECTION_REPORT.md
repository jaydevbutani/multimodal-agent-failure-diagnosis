# DATASET SELECTION REPORT: Multi-Source Agent Failure Diagnosis

**Project:** Machine Learning for AI Agent Failure Detection, Attribution and Root-Cause Analysis  
**Subject:** Applications and Use Cases of Machine Learning (AUCML — TE7483)  
**Evaluation Date:** October 24, 2026  
**Auditor:** Lead ML Engineer  

---

## 1. Executive Summary & Objective

In accordance with Section 5 of the Master Specification, this report presents an empirical, non-fabricated evaluation of all 6 candidate datasets for AI-agent execution failure analysis. 

Rather than forcing an artificial single-file dataset, we adopt a **multi-tier research design**:
1. **Primary Supervised Training & In-Domain Validation:** **Agent Diagnostics (`sjarmak/agent-diagnostics`)** — 11,995 verified real agent trials across 61 benchmarks, with ground-truth execution outcomes (`passed`, `reward`), comprehensive execution telemetry, and sequential tool trajectories.
2. **Specialized Temporal & Component Attribution Benchmark:** **Who&When (`ag2ai/Agents_Failure_Attribution` / `Kevin355/Who_and_When`)** — 184 full multi-agent trajectories (4,092 execution turns) with ground-truth decisive error step (`mistake_step`), responsible agent (`mistake_agent`), and root-cause reasoning (`mistake_reason`).
3. **Observational & Cross-Domain Validation:** **TraceLab (`uw-syfi/TraceLab`)** — 8,058 real-world developer coding sessions with tool execution error signals (`is_error`) and latency.
4. **Live Instrumented Evaluation Suite:** A lightweight execution harness in `src/live_tracing/` capturing real agent runs for external demonstration.

---

## 2. Comparative Evaluation Matrix across 6 Candidate Datasets

| Evaluation Criterion | Candidate 1: TraceLab (UW-SyFI) | Candidate 2: Agent Diagnostics (Primary) | Candidate 3: CMU Agent Trajectories | Candidate 4: Trace Commons | Candidate 5: FreeInference | Candidate 6: Who&When (Attribution) |
|---|---|---|---|---|---|---|
| **1. Source** | `github.com/uw-syfi/TraceLab` | `github.com/sjarmak/agent-diagnostics` | `hf.co/datasets/cx-cmu/agent_trajectories` | `hf.co/datasets/trace-commons/agent-traces` | `github.com/HarvardMadSys/freeinference_agentic_trace` | `github.com/ag2ai/Agents_Failure_Attribution` |
| **2. Real vs Synthetic** | **Real:** captured from human developers using Claude Code / Codex | **Real:** actual coding agents (Claude Code, OpenHands, Cursor) running benchmark tasks | **Real/Benchmark:** Agent trajectories on math/mcp/search tasks | **Real:** Human developer sessions using Claude Code | **Real:** Serving traces from freeinference.org | **Real:** Trajectories from GAIA & AssistantBench tasks |
| **3. Number of records** | ~665,453 steps across 8,058 sessions | **11,995 full trials** (Verified in MANIFEST.json) | ~8,653 trajectories | 30 sessions (current public release) | 12,002 sessions, 1.2M requests | **4,092 turns** across 184 trajectories |
| **4. Number of sessions / traces** | 8,058 sessions | **11,995 independent trials** | ~8,653 trajectories | 30 sessions | 12,002 sessions | 184 trajectories (58 Hand-Crafted + 126 Alg-Generated) |
| **5. Number of fields** | 26 columns | **32 columns** | 14 columns | 11 columns | System / cache metrics | 8 to 9 columns |
| **6. Text available?** | Yes (command skeletons, event text) | Yes (`trial_path`, `task_id`, file paths) | Yes (conversation turns) | Yes (prompts & messages) | No (token counts only) | Yes (full conversation text, queries, thoughts) |
| **7. Tool calls available?** | Yes (743,819 tool calls with name, latency, error) | Yes (`tool_calls_total`, tool subsets, `tool_call_sequence`) | Yes (global tool inventory) | Yes (`num_tool_calls`, tool specs) | Yes (tool call frequency) | Yes (code execution, terminal, browser tools) |
| **8. Temporal information?** | Yes (millisecond event timestamps) | Yes (`duration_seconds`, sequential `tool_call_sequence`, turns) | Yes (turn sequence index) | Yes (`sent_at` timestamp) | Yes (hourly/weekly timestamps) | Yes (`step_idx`, decisive failure step $t^*$) |
| **9. Failure labels?** | Step-level `is_error: True/False` only (no final task outcome) | **Yes:** `passed: False` (2,764 trials, 23.04%) + 40 failure categories in taxonomy | Pass/Fail reward available | No failure labels (raw dialogues) | No task failure labels (serving performance only) | **Yes:** Ground-truth `mistake_reason` and error step |
| **10. Outcome labels?** | No task outcome label | **Yes:** `passed: True/False` (9,231 Pass / 2,764 Fail) + continuous `reward` | Task completion score / reward | None | Serving cache hit/miss only | Task failure verified |
| **11. Agent identity?** | Claude Code, Codex | `claude-code` (9,421), `openhands` (2,483), `cursor-cli` (91) | Multiple agent architectures | `claude_code` | Generic serving agent | `WebSurfer`, `Orchestrator`, `Assistant`, `FileSurfer`, `Verification_Expert`, etc. |
| **12. Model identity?** | Claude Opus 4.7, Claude Sonnet | `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-opus-4-5` | GPT-4o, Claude, Qwen, etc. | Claude Code harness | Mixed LLM serving | GPT-4, specialized LLMs |
| **13. Benchmark / domain?** | Real-world software engineering | **61 benchmarks:** `openhands`, `org_migration`, `sdlc_fix`, `org_security`, `sdlc_feature`, `sdlc_debug`, etc. | MathHay, MCPBench, Search | Open-ended developer tasks | LLM serving workloads | GAIA, AssistantBench |
| **14. License** | CC BY 4.0 / Apache 2.0 | Apache License 2.0 | Research use (Gated / 401 Auth required) | Open public dataset | Custom open research | MIT License |
| **15. Missing values** | Minimal (<1% latency) | **0 missing values** across 31 features; `reward` has 300 nulls (2.5%) where only binary `passed` was logged | Varies by split | None | None | 0 missing values in core features |
| **16. Duplicate records** | None (unique session IDs) | **0 duplicate `trial_id` records** (11,995 unique IDs) | None | None | None | 0 duplicate `question_ID` records |
| **17. Data leakage risk** | Session-level split required | Prevented by grouping on `task_id` or `trial_id` | Trajectory grouping required | Session grouping required | Workload grouping required | Task ID grouping required |
| **18. ML tasks supported** | Tool-level Error Prediction, Latency Modeling | **Task A:** Execution Failure Detection<br>**Task B:** Error Severity & Reward Regression<br>**Task C:** Failure Mode Profiling | Pass/Fail Classification | Unsupervised Representation | Infrastructure Simulation Only | **Task D:** Temporal Fault Localization<br>**Task E:** Component Localization & Attribution |
| **19. Suitability for AUCML** | High (Observational / Cross-Domain) | **Highest (Ideal Primary ML Dataset)** | Low (Gated behind HF 401 token authentication) | Low (Only 30 sessions in public release) | Poor (No task failure ground truth) | **Highest (Specialized Attribution Benchmark)** |

---

## 3. Dataset Ranking & Selection Justification

### Rank 1 (Primary Supervised Training & Evaluation): Agent Diagnostics (`signals.parquet`)
- **Justification:** It provides 11,995 authentic agent execution trials with verified ground truth (`passed: True/False`), continuous `reward`, 30+ execution telemetry signals, and tool-call sequence trajectories. It has 0 missing values in all telemetry features, full provenance with cryptographic SHA-256 validation, and an established behavioral failure taxonomy.
- **Role:** Backbone for **Task 1 (Failure Detection)** and **Task 2 (Failure Severity & Diagnostic Prediction)**.

### Rank 2 (Specialized Attribution Benchmark): Who&When (ICML 2025 Spotlight)
- **Justification:** Sole peer-reviewed benchmark with human-verified annotations for decisive failure step index (`mistake_step`), responsible agent module (`mistake_agent`), and evidence-grounded reasoning (`mistake_reason`).
- **Role:** Backbone for **Task 3 (Temporal Fault Localization)** and **Task 4 (Component Attribution)**.

### Rank 3 (Observational & Cross-Domain Validation): TraceLab (UW-SyFI)
- **Justification:** Large-scale (51 MB compressed, 8,058 sessions) developer trace dataset with step-level tool execution telemetry (`is_error`, latency, token counts).
- **Role:** Observational out-of-domain validation for tool error dynamics.

### Rank 4 (Excluded from Supervised Failure Modeling):
- **Candidate 3 (CMU Agent Trajectories):** Restricted behind Hugging Face 401 authentication; non-reproducible for external evaluators without personal API tokens.
- **Candidate 4 (Trace Commons):** Only 30 sessions in public release; statistically insufficient for ML training.
- **Candidate 5 (FreeInference):** Contains serving infrastructure metrics (cache hits, token throughput) but zero task failure or completion labels.
