# LIMITATIONS & FUTURE RESEARCH DIRECTIONS

**Course:** Applications and Use Cases of Machine Learning (AUCML — TE7483)  
**Institution:** Symbiosis Institute of Technology (SIT), Pune  

---

## 1. Dataset & Domain Boundaries
1. **Software Engineering Task Preponderance:**  
   The primary dataset (`signals.parquet`, 11,995 trials) is sourced predominantly from coding benchmarks (`sdlc_fix`, `openhands`, `org_migration`). While software agents represent the most complex multi-tool workflows, behaviors in purely conversational or robotic physical agents may exhibit distinct failure dynamics.
2. **Benchmark Environment Constraints:**  
   Benchmark tasks operate within sandbox containers with explicit unit test verifiers (`has_verifier_result == True`). In production enterprise environments, objective automated ground truth is often delayed or noisy.

---

## 2. Modality & Sequence Modeling Constraints
1. **Text Modality Truncation:**  
   Due to memory constraints on student hardware and to prevent excessive latency, full text logs were abstracted into structured telemetry and tool sequences rather than passing entire 100,000-token session logs through multi-billion parameter foundation models.
2. **Tool Vocabulary Dynamics:**  
   The tool-call sequence model utilizes a fixed vocabulary of 69 observed tool tokens. When an agent invokes novel or custom third-party MCP tools not present in the training set, they are mapped to `<UNK>` (Unknown), which slightly degrades sequence embedding fidelity.

---

## 3. Attribution & Causal Discovery Boundaries
1. **Attribution vs True Causality:**  
   Our model identifies **predictive correlations and empirical evidence contributions** using SHAP and attention weights. We explicitly do NOT claim true counterfactual causal discovery, as unmeasured environment latencies or non-deterministic LLM temperature variations can introduce confounding factors.
2. **Sample Size on Component Localization:**  
   While the primary dataset has 11,995 trials, the specialized `Who&When` attribution benchmark currently has 184 full multi-agent trajectories. While adequate for benchmarking, expanding human-annotated multi-agent failure traces remains an active research direction.

---

## 4. Future Work
- **Online Streaming Inference:** Transitioning from batch post-mortem diagnosis to step-by-step real-time early warning before an agent exhausts its API budget.
- **Multimodal Visual Telemetry:** Incorporating screenshot observations from GUI-based computer-use agents (e.g., Anthropic Computer Use / OS-World).
