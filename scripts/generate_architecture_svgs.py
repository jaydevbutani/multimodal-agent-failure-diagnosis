import os

os.makedirs('docs/figures', exist_ok=True)

# 1. Figure 1: End-to-End Multi-View Pipeline
fig1_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 360" width="100%" height="auto" style="background:#ffffff; font-family:'Times New Roman',serif;">
  <defs>
    <linearGradient id="gradHeader" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
    <linearGradient id="gradTab" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#0369a1"/>
    </linearGradient>
    <linearGradient id="gradSeq" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#6d28d9"/>
    </linearGradient>
    <linearGradient id="gradFuse" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#e11d48"/>
      <stop offset="100%" stop-color="#be123c"/>
    </linearGradient>
    <linearGradient id="gradDiag" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#047857"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
    </marker>
    <marker id="arrowRed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#be123c"/>
    </marker>
  </defs>

  <!-- Title -->
  <text x="500" y="28" font-size="16" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 1: End-to-End Heterogeneous Execution-Trace Diagnosis & Attribution Architecture</text>

  <!-- Input Trial -->
  <rect x="20" y="110" width="160" height="120" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <rect x="20" y="110" width="160" height="32" rx="8" fill="url(#gradHeader)"/>
  <text x="100" y="132" font-size="13" font-weight="bold" fill="#ffffff" text-anchor="middle">Agent Execution</text>
  <text x="35" y="162" font-size="11" fill="#334155">• Turn Telemetry</text>
  <text x="35" y="180" font-size="11" fill="#334155">• Tool Exit Codes</text>
  <text x="35" y="198" font-size="11" fill="#334155">• Code Modification</text>
  <text x="35" y="216" font-size="11" fill="#334155">• Action Trajectory</text>

  <!-- Fork arrows -->
  <path d="M 180 150 L 230 150 L 230 95 L 260 95" fill="none" stroke="#475569" stroke-width="1.8" marker-end="url(#arrow)"/>
  <path d="M 180 190 L 230 190 L 230 245 L 260 245" fill="none" stroke="#475569" stroke-width="1.8" marker-end="url(#arrow)"/>

  <!-- Tabular Stream -->
  <rect x="260" y="55" width="220" height="85" rx="8" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1.5"/>
  <rect x="260" y="55" width="220" height="26" rx="8" fill="url(#gradTab)"/>
  <text x="370" y="73" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Structured Telemetry Stream</text>
  <text x="272" y="98" font-size="11" fill="#0369a1">21 Scaled Telemetry Metrics + 67 One-Hot</text>
  <text x="272" y="115" font-size="11" fill="#0f172a" font-weight="bold">Dense Tabular MLP Encoder [R^88 → R^64]</text>
  <text x="272" y="130" font-size="10.5" fill="#475569">Linear(88→128) → BatchNorm → ReLU → Linear(64)</text>

  <!-- Sequence Stream -->
  <rect x="260" y="200" width="220" height="85" rx="8" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.5"/>
  <rect x="260" y="200" width="220" height="26" rx="8" fill="url(#gradSeq)"/>
  <text x="370" y="218" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Tool Action Sequence Stream</text>
  <text x="272" y="243" font-size="11" fill="#6d28d9">Tokenized Tool Sequence: [t_1, t_2, ..., t_50]</text>
  <text x="272" y="260" font-size="11" fill="#0f172a" font-weight="bold">Bidirectional LSTM Encoder [R^50 → R^64]</text>
  <text x="272" y="275" font-size="10.5" fill="#475569">Embedding(32) → BiLSTM(32) → Max-Pooling</text>

  <!-- Connectors to Fusion -->
  <path d="M 480 97 L 530 97 L 530 145 L 550 145" fill="none" stroke="#475569" stroke-width="1.8" marker-end="url(#arrow)"/>
  <path d="M 480 242 L 530 242 L 530 195 L 550 195" fill="none" stroke="#475569" stroke-width="1.8" marker-end="url(#arrow)"/>

  <!-- Cross-Modal Gated Fusion -->
  <rect x="550" y="115" width="200" height="110" rx="8" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.5"/>
  <rect x="550" y="115" width="200" height="28" rx="8" fill="url(#gradFuse)"/>
  <text x="650" y="134" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Cross-Modal Gated Fusion</text>
  <text x="560" y="160" font-size="11" fill="#be123c">Gating: α = σ(W_g · [h_tab ; h_seq])</text>
  <text x="560" y="178" font-size="11" fill="#9f1239">Fused: h = α ⊙ h_tab + (1-α) ⊙ h_seq</text>
  <text x="560" y="196" font-size="11" fill="#0f172a" font-weight="bold">Classification Head: Linear(64→32→1)</text>
  <text x="560" y="214" font-size="10.5" fill="#475569">Failure Probability P(Failure | x, s)</text>

  <!-- Connector to Diagnosis -->
  <path d="M 750 170 L 790 170" fill="none" stroke="#be123c" stroke-width="2" marker-end="url(#arrowRed)"/>

  <!-- Diagnostic Attribution Engine -->
  <rect x="790" y="85" width="190" height="170" rx="8" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1.5"/>
  <rect x="790" y="85" width="190" height="28" rx="8" fill="url(#gradDiag)"/>
  <text x="885" y="104" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Forensic Diagnosis Engine</text>
  <text x="802" y="130" font-size="11" font-weight="bold" fill="#047857">1. Binary Status (τ = 0.46):</text>
  <text x="815" y="146" font-size="10.5" fill="#15803d">HEALTHY / FAILURE DETECTED</text>
  <text x="802" y="168" font-size="11" font-weight="bold" fill="#047857">2. Behavioral Attribution:</text>
  <text x="815" y="184" font-size="10.5" fill="#334155">• Cyclic Retry Thrashing</text>
  <text x="815" y="200" font-size="10.5" fill="#334155">• Environment Crash</text>
  <text x="815" y="216" font-size="10.5" fill="#334155">• Excessive Churn / Exploration</text>
  <text x="802" y="238" font-size="11" font-weight="bold" fill="#047857">3. Turn Localization (Turn k)</text>
</svg>
"""

with open('docs/figures/fig1_research_workflow.svg', 'w', encoding='utf-8') as f:
    f.write(fig1_svg)
print("Saved docs/figures/fig1_research_workflow.svg")

# 2. Figure 6: Detailed Neural Gated Fusion Diagram
fig6_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 420" width="100%" height="auto" style="background:#ffffff; font-family:'Times New Roman',serif;">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#334155"/>
    </marker>
  </defs>

  <text x="450" y="28" font-size="16" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 6: Cross-Modal Gated Multi-View Fusion Architecture</text>

  <!-- Left: Tabular Branch -->
  <rect x="50" y="60" width="340" height="240" rx="8" fill="#f8fafc" stroke="#94a3b8" stroke-dasharray="4 4"/>
  <text x="220" y="85" font-size="13" font-weight="bold" fill="#0369a1" text-anchor="middle">Branch 1: Dense Telemetry Representation</text>

  <rect x="80" y="105" width="280" height="32" rx="4" fill="#e0f2fe" stroke="#0284c7"/>
  <text x="220" y="126" font-size="11" fill="#0369a1" text-anchor="middle">Input: Scaled Telemetry &amp; Metadata [x ∈ R^88]</text>

  <path d="M 220 137 L 220 155" stroke="#334155" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="80" y="155" width="280" height="32" rx="4" fill="#e0f2fe" stroke="#0284c7"/>
  <text x="220" y="176" font-size="11" fill="#0369a1" text-anchor="middle">Linear(88 → 128) + BatchNorm1d + ReLU + Dropout(0.3)</text>

  <path d="M 220 187 L 220 205" stroke="#334155" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="80" y="205" width="280" height="32" rx="4" fill="#e0f2fe" stroke="#0284c7"/>
  <text x="220" y="226" font-size="11" fill="#0369a1" text-anchor="middle">Linear(128 → 64) + ReLU</text>

  <path d="M 220 237 L 220 255" stroke="#334155" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="110" y="255" width="220" height="32" rx="4" fill="#0284c7"/>
  <text x="220" y="276" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Telemetry Embedding: h_tab ∈ R^64</text>

  <!-- Right: Sequence Branch -->
  <rect x="510" y="60" width="340" height="240" rx="8" fill="#f8fafc" stroke="#94a3b8" stroke-dasharray="4 4"/>
  <text x="680" y="85" font-size="13" font-weight="bold" fill="#6d28d9" text-anchor="middle">Branch 2: Sequential Trajectory Representation</text>

  <rect x="540" y="105" width="280" height="32" rx="4" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="680" y="126" font-size="11" fill="#6d28d9" text-anchor="middle">Input: Integer Tool Tokens [s ∈ {0..V}^50]</text>

  <path d="M 680 137 L 680 155" stroke="#334155" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="540" y="155" width="280" height="32" rx="4" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="680" y="176" font-size="11" fill="#6d28d9" text-anchor="middle">Embedding(Vocab_size → 32, padding_idx=0)</text>

  <path d="M 680 187 L 680 205" stroke="#334155" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="540" y="205" width="280" height="32" rx="4" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="680" y="226" font-size="11" fill="#6d28d9" text-anchor="middle">BiLSTM(32 → 32) + Temporal Max-Pooling [R^64]</text>

  <path d="M 680 237 L 680 255" stroke="#334155" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="570" y="255" width="220" height="32" rx="4" fill="#7c3aed"/>
  <text x="680" y="276" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Sequence Embedding: h_seq ∈ R^64</text>

  <!-- Center Gated Fusion -->
  <path d="M 220 287 L 220 330 L 370 330" stroke="#334155" stroke-width="1.8" fill="none" marker-end="url(#arr)"/>
  <path d="M 680 287 L 680 330 L 530 330" stroke="#334155" stroke-width="1.8" fill="none" marker-end="url(#arr)"/>

  <rect x="370" y="310" width="160" height="50" rx="6" fill="#be123c"/>
  <text x="450" y="331" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Learnable Gating Mechanism</text>
  <text x="450" y="348" font-size="10.5" fill="#fecdd3" text-anchor="middle">α = σ(W_g · [h_tab ; h_seq])</text>

  <path d="M 450 360 L 450 380" stroke="#334155" stroke-width="1.8" marker-end="url(#arr)"/>

  <rect x="320" y="380" width="260" height="28" rx="4" fill="#0f172a"/>
  <text x="450" y="399" font-size="11.5" font-weight="bold" fill="#ffffff" text-anchor="middle">Classifier Head: Linear(64 → 32) → ReLU → Linear(32 → 1)</text>
</svg>
"""

with open('docs/figures/fig6_fusion_architecture.svg', 'w', encoding='utf-8') as f:
    f.write(fig6_svg)
print("Saved docs/figures/fig6_fusion_architecture.svg")

# 3. Figure 13: Diagnostic Reasoning & Localization Pipeline
fig13_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 950 200" width="100%" height="auto" style="background:#ffffff; font-family:'Times New Roman',serif;">
  <defs>
    <marker id="marr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0f172a"/>
    </marker>
  </defs>

  <text x="475" y="24" font-size="15" font-weight="bold" text-anchor="middle" fill="#0f172a">Figure 13: Decoupled ML Failure Prediction &amp; Evidence-Based Diagnostic Attribution Flow</text>

  <!-- Step 1 -->
  <rect x="15" y="55" width="165" height="115" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <rect x="15" y="55" width="165" height="26" rx="6" fill="#0284c7"/>
  <text x="97" y="73" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">1. Telemetry Capture</text>
  <text x="25" y="98" font-size="10" fill="#334155">• Turn error counters</text>
  <text x="25" y="114" font-size="10" fill="#334155">• Command retry loops</text>
  <text x="25" y="130" font-size="10" fill="#334155">• Step duration &amp; churn</text>
  <text x="25" y="146" font-size="10" fill="#334155">• Tool transition log</text>

  <path d="M 180 112 L 205 112" stroke="#0f172a" stroke-width="1.8" marker-end="url(#marr)"/>

  <!-- Step 2 -->
  <rect x="205" y="55" width="165" height="115" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <rect x="205" y="55" width="165" height="26" rx="6" fill="#7c3aed"/>
  <text x="287" y="73" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">2. Multi-View Prediction</text>
  <text x="215" y="98" font-size="10" fill="#334155">• Compute derived ratios</text>
  <text x="215" y="114" font-size="10" fill="#334155">• Dense MLP inference</text>
  <text x="215" y="130" font-size="10" fill="#334155">• BiLSTM trajectory pass</text>
  <text x="215" y="146" font-size="10" fill="#6d28d9" font-weight="bold">• P(Failure) ≥ 0.46</text>

  <path d="M 370 112 L 395 112" stroke="#0f172a" stroke-width="1.8" marker-end="url(#marr)"/>

  <!-- Step 3 -->
  <rect x="395" y="55" width="165" height="115" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <rect x="395" y="55" width="165" height="26" rx="6" fill="#be123c"/>
  <text x="477" y="73" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">3. Signal Extraction</text>
  <text x="405" y="98" font-size="10" fill="#334155">• Tool entropy calculation</text>
  <text x="405" y="114" font-size="10" fill="#334155">• Retry-to-error quotient</text>
  <text x="405" y="130" font-size="10" fill="#334155">• Search vs edit density</text>
  <text x="405" y="146" font-size="10" fill="#334155">• Behavioral anomalies</text>

  <path d="M 560 112 L 585 112" stroke="#0f172a" stroke-width="1.8" marker-end="url(#marr)"/>

  <!-- Step 4 -->
  <rect x="585" y="55" width="165" height="115" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <rect x="585" y="55" width="165" height="26" rx="6" fill="#ea580c"/>
  <text x="667" y="73" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">4. Signature Matching</text>
  <text x="595" y="98" font-size="10" fill="#334155">• Cyclic Retry Thrashing</text>
  <text x="595" y="114" font-size="10" fill="#334155">• Environment Crash</text>
  <text x="595" y="130" font-size="10" fill="#334155">• High Code Churn</text>
  <text x="595" y="146" font-size="10" fill="#334155">• Over-Exploration Loop</text>

  <path d="M 750 112 L 775 112" stroke="#0f172a" stroke-width="1.8" marker-end="url(#marr)"/>

  <!-- Step 5 -->
  <rect x="775" y="55" width="160" height="115" rx="6" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>
  <rect x="775" y="55" width="160" height="26" rx="6" fill="#059669"/>
  <text x="855" y="73" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">5. Forensic Output</text>
  <text x="785" y="98" font-size="10" fill="#15803d" font-weight="bold">• What Happened</text>
  <text x="785" y="114" font-size="10" fill="#15803d" font-weight="bold">• Why it Failed</text>
  <text x="785" y="130" font-size="10" fill="#15803d" font-weight="bold">• Responsible Step</text>
  <text x="785" y="146" font-size="10" fill="#15803d" font-weight="bold">• Actionable Remedy</text>
</svg>
"""

with open('docs/figures/fig13_diagnostic_reasoning.svg', 'w', encoding='utf-8') as f:
    f.write(fig13_svg)
print("Saved docs/figures/fig13_diagnostic_reasoning.svg")
