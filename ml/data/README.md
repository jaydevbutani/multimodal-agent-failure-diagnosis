# Dataset Provenance & Governance
## Agent Diagnostics Benchmark Suite (11,995 Trials)

### 1. Primary Dataset
The primary dataset utilized for model training, validation, and held-out evaluation is **Agent Diagnostics**, capturing end-to-end execution trajectories of autonomous coding and reasoning agents across:
- **Benchmarks:** `openhands`, `sdlc_fix`, `org_migration`, and `swe-bench` subsets.
- **LLM Backbones:** `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-code`, etc.
- **Total Volume:** 11,995 total trials.

### 2. Leakage-Free Splitting Strategy
- **Split Ratio:**
  - Training Set: 7,999 trials (~66.7%)
  - Validation Set: 2,000 trials (~16.7%)
  - Test Set: 1,996 held-out trials (~16.6%)
- **Target Distribution:**
  - `passed = True` (Healthy Nominal): 76.9%
  - `passed = False` (Failure Detected): 23.1%
  - Class Imbalance ratio: ~3.34:1, counterbalanced by `pos_weight = 3.34` and `class_weight='balanced'`.

### 3. Curated Synthetic Demonstration Cases
Stored under `data/demo_traces/` (7 canonical scenarios). These are used strictly for user testing, UI interactive canvas inspection, and teacher demonstration. They are **NOT** part of the held-out evaluation splits to prevent synthetic bias.
