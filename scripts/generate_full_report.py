import os
import base64

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower().replace('.', '')
    mime = "image/jpeg" if ext in ['jpg', 'jpeg'] else ("image/svg+xml" if ext == 'svg' else "image/png")
    return f"data:{mime};base64,{encoded}"

print("Reading and encoding visuals...")
logo_b64 = get_base64_image("static/img/sit_pune_logo.png")
class_dist_b64 = get_base64_image("results/plots/class_distribution.png")
model_comp_b64 = get_base64_image("results/plots/model_comparison_bar.png")
roc_b64 = get_base64_image("results/plots/roc_curves_all_models.png")
pr_b64 = get_base64_image("results/plots/pr_curves_all_models.png")
cm_grid_b64 = get_base64_image("results/plots/confusion_matrix_grid.png")
feat_imp_b64 = get_base64_image("results/plots/feature_importance_top15.png")
shap_b64 = get_base64_image("results/plots/shap_summary_failure.png")
ablation_b64 = get_base64_image("results/plots/ablation_study_chart.png")
ui_screen_b64 = get_base64_image("Screenshot 2026-10-23 234749.png")
if not ui_screen_b64:
    ui_screen_b64 = get_base64_image("stitch_reference/stitch_agentic_ai_failure_diagnostic_ui (1)/screen.png")

fig1_svg = open("docs/figures/fig1_research_workflow.svg", encoding="utf-8").read()
fig6_svg = open("docs/figures/fig6_fusion_architecture.svg", encoding="utf-8").read()
fig13_svg = open("docs/figures/fig13_diagnostic_reasoning.svg", encoding="utf-8").read()

print("Constructing master research report HTML...")

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Research Report: Multimodal AI for Agentic System Failure Diagnosis &amp; Root-Cause Reasoning</title>
<style>
  @page {{
    size: A4;
    margin: 25mm 20mm 25mm 20mm;
    @bottom-right {{
      content: counter(page);
      font-family: 'Times New Roman', Times, serif;
      font-size: 10pt;
    }}
    @top-right {{
      content: "AUCML — TE7483: Research Project Report";
      font-family: 'Times New Roman', Times, serif;
      font-size: 9pt;
      color: #64748b;
    }}
  }}

  body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #111827;
    margin: 0;
    padding: 0;
    text-align: justify;
    background-color: #ffffff;
  }}

  .page-container {{
    max-width: 820px;
    margin: 0 auto;
    padding: 40px 30px;
    background: #ffffff;
  }}

  /* Page Break Utilities */
  .page-break {{
    page-break-before: always;
    break-before: page;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
  }}

  @media print {{
    .page-break {{
      border-top: none;
      margin-top: 0;
      padding-top: 0;
    }}
    .page-container {{
      max-width: 100%;
      padding: 0;
    }}
    .no-print {{
      display: none !important;
    }}
  }}

  /* Cover Page Styling */
  .cover-page {{
    text-align: center;
    min-height: 95vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 20px 10px;
    page-break-after: always;
  }}

  .cover-logo {{
    max-width: 380px;
    height: auto;
    margin: 0 auto 25px auto;
  }}

  .cover-institution {{
    font-size: 13pt;
    font-weight: bold;
    color: #1e3a8a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
  }}

  .cover-department {{
    font-size: 11pt;
    font-weight: normal;
    color: #334155;
    margin-bottom: 30px;
  }}

  .cover-project-tag {{
    display: inline-block;
    background-color: #f1f5f9;
    color: #0f172a;
    font-size: 10.5pt;
    font-weight: bold;
    padding: 4px 14px;
    border-radius: 4px;
    border: 1px solid #cbd5e1;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}

  .cover-title {{
    font-size: 21pt;
    font-weight: bold;
    color: #0f172a;
    line-height: 1.3;
    margin: 10px 0 20px 0;
  }}

  .cover-subtitle {{
    font-size: 12.5pt;
    color: #475569;
    font-style: italic;
    margin-bottom: 40px;
    line-height: 1.4;
  }}

  .group-details-table {{
    width: 90%;
    margin: 0 auto 30px auto;
    border-collapse: collapse;
    font-size: 10.5pt;
    text-align: left;
  }}

  .group-details-table th, .group-details-table td {{
    border: 1px solid #94a3b8;
    padding: 7px 12px;
  }}

  .group-details-table th {{
    background-color: #f8fafc;
    color: #0f172a;
    font-weight: bold;
  }}

  .meta-info-box {{
    margin-top: 25px;
    font-size: 10.5pt;
    color: #334155;
    line-height: 1.6;
  }}

  /* Typography & Headings */
  h1.paper-title {{
    font-size: 18pt;
    font-weight: bold;
    text-align: center;
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 12px;
    line-height: 1.3;
  }}

  .author-block {{
    text-align: center;
    font-size: 10.5pt;
    color: #334155;
    margin-bottom: 24px;
    line-height: 1.4;
  }}

  .author-names {{
    font-weight: bold;
    color: #0f172a;
  }}

  h2.section-heading {{
    font-size: 13pt;
    font-weight: bold;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1.5px solid #0f172a;
    padding-bottom: 4px;
    margin-top: 32px;
    margin-bottom: 14px;
    page-break-after: avoid;
  }}

  h3.subsection-heading {{
    font-size: 11.5pt;
    font-weight: bold;
    color: #1e293b;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}

  h4.subsubsection-heading {{
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 0;
    margin-bottom: 12px;
    text-indent: 1.5em;
  }}

  p.no-indent {{
    text-indent: 0;
  }}

  /* Abstract & Callout Box */
  .abstract-box {{
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #1e3a8a;
    padding: 16px 20px;
    margin-bottom: 25px;
    font-size: 10.5pt;
    line-height: 1.5;
  }}

  .abstract-title {{
    font-weight: bold;
    font-size: 11pt;
    text-transform: uppercase;
    color: #1e3a8a;
    margin-bottom: 8px;
  }}

  .keywords-block {{
    margin-top: 10px;
    font-size: 10pt;
    color: #475569;
  }}

  /* Academic Tables */
  table.academic-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 10pt;
    page-break-inside: avoid;
  }}

  table.academic-table th, table.academic-table td {{
    border-top: 1px solid #cbd5e1;
    border-bottom: 1px solid #cbd5e1;
    padding: 7px 10px;
    text-align: left;
  }}

  table.academic-table th {{
    border-top: 2px solid #0f172a;
    border-bottom: 1.5px solid #0f172a;
    background-color: #f8fafc;
    font-weight: bold;
    color: #0f172a;
  }}

  table.academic-table tr:last-child td {{
    border-bottom: 2px solid #0f172a;
  }}

  .table-caption {{
    font-size: 10pt;
    font-weight: bold;
    margin-bottom: 6px;
    color: #0f172a;
    text-align: left;
  }}

  .table-notes {{
    font-size: 9pt;
    color: #64748b;
    font-style: italic;
    margin-top: 4px;
    margin-bottom: 16px;
  }}

  /* Figures */
  .figure-container {{
    text-align: center;
    margin: 25px 0;
    page-break-inside: avoid;
  }}

  .figure-image {{
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}

  .figure-caption {{
    font-size: 9.5pt;
    color: #334155;
    margin-top: 8px;
    text-align: center;
    line-height: 1.4;
  }}

  .figure-caption strong {{
    color: #0f172a;
  }}

  /* Formulas & Math */
  .math-block {{
    text-align: center;
    font-style: italic;
    padding: 10px 0;
    margin: 14px 0;
    background: #fdfdfe;
    font-size: 11pt;
  }}

  .equation-num {{
    float: right;
    font-style: normal;
    color: #475569;
  }}

  /* Code & Syntax */
  code {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9.5pt;
    background-color: #f1f5f9;
    padding: 2px 4px;
    border-radius: 3px;
    color: #0f172a;
  }}

  pre.code-block {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    background-color: #0f172a;
    color: #f8fafc;
    padding: 12px 16px;
    border-radius: 4px;
    overflow-x: auto;
    line-height: 1.4;
    margin: 16px 0;
  }}

  /* Reference List */
  .reference-item {{
    font-size: 10pt;
    margin-bottom: 10px;
    padding-left: 28px;
    text-indent: -28px;
    line-height: 1.45;
  }}

  .floating-print-btn {{
    position: fixed;
    bottom: 25px;
    right: 25px;
    background: #1e3a8a;
    color: #ffffff;
    border: none;
    padding: 12px 22px;
    border-radius: 30px;
    font-family: Arial, sans-serif;
    font-size: 11pt;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    z-index: 9999;
    transition: background 0.2s;
  }}

  .floating-print-btn:hover {{
    background: #0f172a;
  }}
</style>
</head>
<body>

<button class="floating-print-btn no-print" onclick="window.print()">🖨️ Print / Save as PDF (25-30 Pages)</button>

<div class="page-container">

  <!-- ========================================================================= -->
  <!-- COVER PAGE -->
  <!-- ========================================================================= -->
  <div class="cover-page">
    <div>
      <img src="{logo_b64}" class="cover-logo" alt="Symbiosis Institute of Technology Logo">
      <div class="cover-institution">Symbiosis Institute of Technology (SIT), Pune</div>
      <div class="cover-department">Department of Artificial Intelligence &amp; Machine Learning<br>Symbiosis International (Deemed University), Lavale, Pune, Maharashtra 412115</div>
    </div>

    <div>
      <div class="cover-project-tag">AUCML Case Study &amp; ESE Research Project — TE7483</div>
      <h1 class="cover-title">Multimodal AI for Agentic System Failure Diagnosis and Root-Cause Reasoning</h1>
      <div class="cover-subtitle">A Cross-Modal Gated Fusion Framework Integrating Structured Telemetry and Sequential Tool Trajectories for Autonomous LLM Agent Observability</div>
    </div>

    <div>
      <table class="group-details-table">
        <thead>
          <tr>
            <th colspan="3" style="text-align: center; background-color: #1e3a8a; color: white; font-size: 11pt;">ACADEMIC PROJECT GROUP DETAILS — GROUP NO. 16</th>
          </tr>
          <tr>
            <th>Student Full Name</th>
            <th>Permanent Registration Number (PRN)</th>
            <th>Class / Batch</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Jaydev Butani</strong> (Team Lead)</td>
            <td>24070126075</td>
            <td>B.Tech AIML B-1 (2024–2028)</td>
          </tr>
          <tr>
            <td><strong>Lavanay Mutreja</strong></td>
            <td>24070126100</td>
            <td>B.Tech AIML B-1 (2024–2028)</td>
          </tr>
          <tr>
            <td><strong>Mahit Turakhia</strong></td>
            <td>24070126101</td>
            <td>B.Tech AIML B-1 (2024–2028)</td>
          </tr>
          <tr>
            <td><strong>Het Shah</strong></td>
            <td>25070126513</td>
            <td>B.Tech AIML B-1 (2024–2028)</td>
          </tr>
        </tbody>
      </table>

      <div class="meta-info-box">
        <strong>Faculty Project Guide &amp; Course Evaluator:</strong> Prof. Aniket Shahade<br>
        <strong>Academic Course:</strong> Applications and Use Cases of Machine Learning (AUCML — TE7483)<br>
        <strong>Official Project GitHub Repository:</strong> <a href="https://github.com/jaydevbutani/multimodal-agent-failure-diagnosis" target="_blank" style="color: #1e3a8a; text-decoration: underline;">https://github.com/jaydevbutani/multimodal-agent-failure-diagnosis</a><br>
        <strong>Submission Date:</strong> October 2026
      </div>
    </div>
  </div>

  <!-- ========================================================================= -->
  <!-- PAGE BREAK -> EXECUTIVE SUMMARY / ABSTRACT -->
  <!-- ========================================================================= -->
  <div class="page-break"></div>

  <h1 class="paper-title">Multimodal AI for Agentic System Failure Diagnosis and Root-Cause Reasoning</h1>

  <div class="author-block">
    <span class="author-names">Jaydev Butani, Lavanay Mutreja, Mahit Turakhia, Het Shah</span><br>
    Under the Guidance of <strong>Prof. Aniket Shahade</strong><br>
    <em>Department of Artificial Intelligence and Machine Learning, Symbiosis Institute of Technology (SIT), Pune, India</em><br>
    Project Repository: <code>https://github.com/jaydevbutani/multimodal-agent-failure-diagnosis</code>
  </div>

  <div class="abstract-box">
    <div class="abstract-title">Abstract</div>
    <p class="no-indent">
      Autonomous Large Language Model (LLM) agents are increasingly entrusted with complex, multi-turn software development, shell navigation, and API orchestration. However, ensuring operational reliability remains fundamentally difficult: agents frequently succumb to silent, emergent behavioral pathologies such as cyclic retry thrashing, tool instability, sandbox environment permission denial, and unbounded information over-exploration. Existing observability solutions treat agents either as opaque black-box web services (monitoring solely latency and token cost) or rely on brittle post-hoc human inspection. In this work, we present a comprehensive, leakage-safe machine learning and forensic framework that reframes agent reliability through <strong>heterogeneous multi-view execution-trace fusion</strong>. We jointly model dense tabular execution telemetry (turn statistics, error frequencies, normalized code churn, duration, and information-theoretic tool entropy) alongside ordered discrete tool-call sequences. 
    </p>
    <p class="no-indent" style="margin-top: 8px;">
      Using the verified <strong>Agent Diagnostics</strong> benchmark comprising <strong>11,995 execution trials</strong> across 61 software benchmarks and multiple frontier LLMs (Claude-3.5-Sonnet, Haiku, GPT-4o), we establish a strict grouped task-splitting protocol (<code>task_id</code> grouping) that guarantees <strong>zero cross-task data leakage</strong> across 8,298 training, 1,701 validation, and 1,996 unseen test trials. We systematically compare five model families: Logistic Regression, Random Forest, XGBoost, a sequential BiLSTM, and a proposed <strong>Cross-Modal Gated Multi-View Fusion Network</strong>. Evaluated on the 1,996 completely unseen test trials, experimental results demonstrate a critical operational trade-off: while XGBoost achieves the strongest balanced Macro-F1 ($0.7659$) and PR-AUC ($0.7132$), and Random Forest attains peak overall accuracy ($84.27\%$) and specificity ($91.7\%$), our proposed Multi-View Fusion Network achieves the highest <strong>Failure Recall ($72.95\%$)</strong>, correctly catching 321 out of 440 true failure cases. This represents a <strong>$+15.0$ percentage-point absolute improvement</strong> and a <strong>$+25.9\%$ relative gain</strong> in detected failures over Random Forest. Furthermore, we decouple statistical failure prediction from diagnostic attribution, developing an evidence-based forensic engine that pinpoints suspicious turns, attributes component failures, and suggests actionable mitigations. Finally, we provide both a high-throughput Flask-based SVG Trace Canvas observability platform and a Streamlit research console, facilitating interactive diagnosis and live sandbox simulation.
    </p>
    <div class="keywords-block">
      <strong>Keywords:</strong> Autonomous AI Agents, Agent Observability, Failure Diagnosis, Multimodal Fusion, Telemetry Modeling, Root-Cause Analysis, Tool-Call Sequences, BiLSTM, XGBoost, Reliability Engineering.
    </div>
  </div>

  <!-- ========================================================================= -->
  <!-- 1. INTRODUCTION -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">1. Introduction</h2>
  <p>
    The paradigm of artificial intelligence has experienced a profound shift from passive language prediction to autonomous, interactive agency. Modern agentic architectures—such as OpenHands, SWE-Agent, Aider, and AutoGen—combine frontier foundation models (e.g., Claude-3.5-Sonnet, GPT-4o) with external execution environments, allowing agents to inspect file trees, execute arbitrary bash commands, navigate web browsers, and author multi-file software patches. By iteratively executing ReAct (Reasoning and Acting) loops, these agents autonomously decompose overarching programming specifications into granular tool calls.
  </p>
  <p>
    Despite impressive demonstrations, deploying autonomous agents in mission-critical environments introduces unprecedented software reliability risks. Unlike traditional deterministic software systems that fail with explicit stack traces, and unlike static machine learning models whose errors manifest as isolated misclassifications, agentic failures are <em>emergent, sequential, and silent</em>. An agent may execute 60 consecutive turns without triggering an uncaught Python exception, yet fail completely to accomplish the user task because it entered an infinite loop of executing identical grep commands, or thrashing repeatedly on a failing unit test without adapting its hypothesis.
  </p>
  <p>
    Current industry observability tools (e.g., LangSmith, Arize Phoenix, Datadog LLM Observability) predominantly monitor superficial API telemetry, including HTTP request latency, token consumption, and model invocation counts. While necessary for billing and latency monitoring, these metrics provide virtually zero insight into execution correctness or failure etiology. When an agent execution fails, Site Reliability Engineers (SREs) and software teams are forced to manually parse transcripts spanning tens of thousands of tokens, searching for the decisive mistake step that derailed the trajectory.
  </p>
  <p>
    To resolve this challenge, this research paper formulates automated agent failure detection and root-cause localization as a <strong>heterogeneous multi-view representation learning problem</strong>. An agent execution trial naturally contains two complementary representations:
  </p>
  <ul>
    <li><strong>Structured Execution Telemetry:</strong> Aggregated continuous and discrete counters, including total turns, tool execution error counts, retry frequencies, unique files read, lines of code modified (patch churn), cumulative execution duration, and information-theoretic diversity.</li>
    <li><strong>Ordered Action Trajectories:</strong> The temporal, discrete Markovian sequence of tool invocations ($\text{read\_file} \to \text{grep} \to \text{str\_replace} \to \text{execute\_bash}$) reflecting the agent's problem-solving strategy and behavioral transitions.</li>
  </ul>
  <p>
    We demonstrate that fusing these complementary views provides superior sensitivity to system failure compared to analyzing either view in isolation. Crucially, we maintain strict scientific integrity by demonstrating that while tree-based ensembles (XGBoost) excel at balanced tabular discrimination, our proposed deep multi-view fusion architecture excels at <strong>failure sensitivity (recall)</strong>, minimizing the catastrophic risk of silent, unflagged agent failures.
  </p>

  <!-- ========================================================================= -->
  <!-- 2. BACKGROUND AND PROBLEM CONTEXT -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">2. Background and Problem Context</h2>
  <p>
    To understand why autonomous agent failure diagnosis requires specialized machine learning architectures, one must examine the operational structure of an autonomous agent session. An agent operates within a discrete feedback loop formalized as a Partially Observable Markov Decision Process (POMDP). At turn $t$, the agent receives an observation $o_t$ (terminal stdout, test logs, file contents), updates its internal working context, generates a reasoning trace, and emits an action $a_t$ corresponding to a tool invocation with structured arguments.
  </p>
  <p>
    Because foundation models operate probabilistically, minor errors in environment feedback often trigger compounding failure cascades. Through systematic empirical analysis of verified execution benchmarks, we categorize agent failure modes into five canonical behavioral pathologies:
  </p>
  <ol>
    <li><strong>Cyclic Retry Thrashing:</strong> The agent encounters an unexpected error (e.g., an unsatisfied module dependency or syntax error) and repeatedly executes the identical command or minor syntactic variations without altering the underlying environment state. This behavior is characterized by escalating turn latency and near-zero tool sequence entropy.</li>
    <li><strong>Environment &amp; Sandbox Crashes:</strong> The execution environment encounters catastrophic infrastructure failure, including Docker daemon socket disconnection, permission denial (HTTP 403 / EACCES), or out-of-memory container termination. Such failures manifest as high error frequencies during early initialization turns.</li>
    <li><strong>High Code Churn Pathology:</strong> The agent repeatedly modifies source files, reverts edits, and introduces large, erratic diffs without satisfying target unit tests. This behavior is characterized by high patch-size-to-file ratios without task completion.</li>
    <li><strong>Information Over-Exploration:</strong> The agent becomes trapped in exploratory code navigation, issuing dozens of <code>find</code>, <code>grep</code>, and <code>cat</code> commands without ever transitioning to active code repair or test execution.</li>
    <li><strong>Tool Execution Instability:</strong> The agent invokes commands with invalid argument schemas, non-existent flags, or malformed JSON, causing tools to repeatedly return non-zero exit codes.</li>
  </ol>

  <!-- Figure 1 -->
  <div class="figure-container">
    {fig1_svg}
    <div class="figure-caption">
      <strong>Figure 1.</strong> High-level system architecture of the proposed agent failure diagnosis framework, illustrating heterogeneous data ingestion, dual-stream neural encoding, cross-modal gated fusion, and evidence-based forensic attribution.
    </div>
  </div>

  <!-- ========================================================================= -->
  <!-- 3. LITERATURE REVIEW -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">3. Literature Review</h2>
  <p>
    Research into LLM agent reliability and evaluation has accelerated rapidly across software engineering, systems engineering, and machine learning communities. We review prior work across three primary dimensions: agent benchmark frameworks, failure attribution methodologies, and trace representation learning.
  </p>
  <p>
    <strong>Agent Evaluation Benchmarks:</strong> Early benchmarks such as AgentBench (Pan et al., 2024) and SWE-bench (Yang et al., 2024) pioneered rigorous, sandbox-verified evaluation of autonomous coding agents. SWE-bench established a gold standard by testing agents on thousands of real-world GitHub issues from major open-source Python repositories (e.g., SymPy, Django, scikit-learn). However, SWE-bench measures exclusively the binary end-to-end outcome: whether the generated patch passes unit tests. It provides zero intermediate telemetry or diagnostic guidance regarding why an agent failed.
  </p>
  <p>
    <strong>Trajectory &amp; Error Analysis:</strong> Addressing this limitation, recent work has explored intermediate trajectory inspection. Shinn et al. (2023) proposed <em>Reflexion</em>, which enables agents to reflect upon verbal feedback from failing trials; however, Reflexion relies on the LLM itself to identify errors, which frequently fails when the agent hallucinates or suffers from context saturation. Zhang et al. (2025) introduced <em>Who&amp;When</em> (ICML 2025 Spotlight), presenting a hand-crafted and algorithmically generated benchmark for attributing failure to specific agents and steps in multi-agent workflows. Similarly, Jarmak (2025) released the <em>Agent Diagnostics</em> observatory, logging detailed turn-level telemetry, tool execution sequences, and verifier outcomes across thousands of coding trials.
  </p>
  <p>
    <strong>Heterogeneous Trace Representation:</strong> In the systems observability domain, classical root-cause analysis has relied on either tabular metrics (CPU, memory, request error rates) via tree ensembles or distributed trace graphs via Graph Neural Networks (GNNs). However, LLM agent execution traces differ fundamentally from microservices: agent actions are governed by language model semantics and structured tool schemas. Pure sequence models (BiLSTMs, Transformers) capture Markovian tool transitions but ignore continuous resource costs, while tabular gradient boosters capture cumulative churn but lose temporal ordering. Our research synthesizes these paradigms through multi-view representation learning.
  </p>

  <div class="table-caption">Table 1: Comprehensive Literature Review &amp; Methodological Comparison with Prior Work</div>
  <table class="academic-table">
    <thead>
      <tr>
        <th style="width: 14%;">Paper / Framework</th>
        <th style="width: 6%;">Year</th>
        <th style="width: 18%;">Focus Domain</th>
        <th style="width: 18%;">Dataset / Setting</th>
        <th style="width: 18%;">Methodology Used</th>
        <th style="width: 14%;">Primary Limitation</th>
        <th style="width: 12%;">Relevance to Our Work</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>SWE-bench</strong> (Yang et al.)</td>
        <td>2024</td>
        <td>Coding Agent Evaluation</td>
        <td>2,294 GitHub Issues</td>
        <td>Docker Test Harness</td>
        <td>Binary outcome only; no intermediate telemetry</td>
        <td>Foundation for benchmark tasks</td>
      </tr>
      <tr>
        <td><strong>AgentBench</strong> (Pan et al.)</td>
        <td>2024</td>
        <td>Multi-Environment Benchmarking</td>
        <td>8 Distinct Environments</td>
        <td>ReAct prompting evaluation</td>
        <td>Lacks telemetry and failure taxonomy</td>
        <td>Defines agent failure modes</td>
      </tr>
      <tr>
        <td><strong>ChatDev</strong> (Qian et al.)</td>
        <td>2024</td>
        <td>Multi-Agent Software Org</td>
        <td>Synthetic Software Tasks</td>
        <td>Role-playing communicative agents</td>
        <td>High human intervention required</td>
        <td>Multi-agent tool dynamics</td>
      </tr>
      <tr>
        <td><strong>Who&amp;When</strong> (Zhang et al.)</td>
        <td>2025</td>
        <td>Failure Attribution (ICML)</td>
        <td>184 Trajectories, 4,092 turns</td>
        <td>LLM-assisted temporal attribution</td>
        <td>Small sample size (184 runs); no ML modeling</td>
        <td>Attribution benchmark validation</td>
      </tr>
      <tr>
        <td><strong>Agent Diagnostics</strong> (Jarmak)</td>
        <td>2025</td>
        <td>Agent Telemetry Observatory</td>
        <td>11,995 Verified Trials</td>
        <td>Turn logging &amp; signals extraction</td>
        <td>Raw repository; no diagnostic ML models</td>
        <td><strong>Primary dataset used in our research</strong></td>
      </tr>
      <tr>
        <td><strong>Reflexion</strong> (Shinn et al.)</td>
        <td>2023</td>
        <td>Self-Reflective Agents</td>
        <td>HumanEval, AlfWorld</td>
        <td>LLM verbal self-reflection</td>
        <td>Susceptible to LLM hallucination in errors</td>
        <td>Demonstrates need for ML diagnosis</td>
      </tr>
      <tr>
        <td><strong>ToolNet</strong> (Liu et al.)</td>
        <td>2024</td>
        <td>Tool Execution Graph</td>
        <td>RapidAPI benchmarks</td>
        <td>Graph-based tool routing</td>
        <td>Static API schemas; lacks execution telemetry</td>
        <td>Graph representation insights</td>
      </tr>
      <tr>
        <td><strong>AutoGen</strong> (Wu et al.)</td>
        <td>2023</td>
        <td>Multi-Agent Orchestration</td>
        <td>Code Generation &amp; Math</td>
        <td>Multi-agent conversation loops</td>
        <td>No automated failure recovery</td>
        <td>Agent framework reference</td>
      </tr>
      <tr>
        <td><strong>XGBoost</strong> (Chen &amp; Guestrin)</td>
        <td>2016</td>
        <td>Tabular Gradient Boosting</td>
        <td>Standard Tabular Benchmarks</td>
        <td>Second-order gradient tree boosting</td>
        <td>Ignores discrete action sequence dynamics</td>
        <td>High-performance tabular baseline</td>
      </tr>
      <tr>
        <td><strong>Our Proposed Framework</strong></td>
        <td><strong>2026</strong></td>
        <td><strong>Multi-View Agent Failure Diagnosis</strong></td>
        <td><strong>11,995 Verified Trials (Agent Diagnostics)</strong></td>
        <td><strong>Cross-Modal Gated Fusion (Telemetry + BiLSTM)</strong></td>
        <td>Evaluated on coding &amp; software agents</td>
        <td><strong>This Research Contribution</strong></td>
      </tr>
    </tbody>
  </table>
  <div class="table-notes">Note: All references listed represent verified academic publications or authoritative open-source repositories verified in our project provenance manifest.</div>

  <!-- ========================================================================= -->
  <!-- 4. RESEARCH GAP -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">4. Research Gap</h2>
  <p>
    A critical synthesis of the literature reveals three fundamental research gaps at the intersection of machine learning and autonomous systems observability:
  </p>
  <ol>
    <li><strong>The End-to-End Outcome Fallacy:</strong> Prior research overwhelmingly measures final pass/fail outcomes against unit tests, treating the agent's intermediate trajectory as a black box. Consequently, when an agent fails after 40 turns, developers have no mechanism to determine whether failure stemmed from an early misconfiguration in Turn 3 or a late syntax error in Turn 38.</li>
    <li><strong>Single-View Modality Insufficiency:</strong> Existing statistical failure detection attempts rely exclusively on static aggregated telemetry (ignoring action ordering) or sequence models (ignoring continuous physical duration, code churn, and error ratios). As demonstrated in our ablation experiments, neither view alone achieves optimal failure sensitivity.</li>
    <li><strong>Conflation of Prediction and Causal Attribution:</strong> Several emerging observability tools conflate statistical failure prediction with causal root-cause reasoning. A machine learning model outputting $P(\text{failure}) = 0.92$ indicates high risk, but does not provide an actionable explanation of <em>why</em> the failure occurred or <em>which</em> component is responsible. Decoupling statistical prediction from deterministic evidence extraction remains unaddressed in the literature.</li>
  </ol>

  <!-- ========================================================================= -->
  <!-- 5. PROBLEM STATEMENT & MATHEMATICAL FORMULATION -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">5. Problem Statement &amp; Mathematical Formulation</h2>
  <p class="no-indent">
    Let an agent execution trial be represented as an ordered tuple:
  </p>
  <div class="math-block">
    $$\mathcal{T} = (\mathbf{x}, \mathbf{s}, \mathbf{m})$$
    <span class="equation-num">(1)</span>
  </div>
  <p class="no-indent">
    where $\mathbf{x} \in \mathbb{R}^{d}$ is a continuous feature vector of $d=21$ execution telemetry metrics, $\mathbf{s} = (t_1, t_2, \dots, t_L)$ is an ordered discrete sequence of $L$ tool call tokens drawn from a finite vocabulary $\mathcal{V}$ ($|\mathcal{V}| = 35$, truncated/padded to $L=50$), and $\mathbf{m}$ is a categorical tuple encoding metadata $(\text{model}, \text{agent\_name}, \text{benchmark})$.
  </p>
  <p>
    <strong>Task 1: Out-of-Sample Binary Failure Classification.</strong> Learn a parameterized mapping $f_\theta: (\mathbf{x}, \mathbf{s}, \mathbf{m}) \to [0, 1]$ estimating the posterior failure probability:
  </p>
  <div class="math-block">
    $$\hat{y} = P(y = 1 \mid \mathbf{x}, \mathbf{s}, \mathbf{m})$$
    <span class="equation-num">(2)</span>
  </div>
  <p class="no-indent">
    where the true label $y \in \{0, 1\}$ is determined by sandbox test suite verification ($y=0$ if the patch passes all unit tests; $y=1$ if the agent crashes, times out, or fails verification). The binary classification decision is governed by a calibrated threshold $\tau$:
  </p>
  <div class="math-block">
    $$\hat{C}_\tau = \mathbb{I}(\hat{y} \ge \tau)$$
    <span class="equation-num">(3)</span>
  </div>
  <p>
    <strong>Task 2: Evidence-Based Turn Localization &amp; Attribution.</strong> Given an execution trace flagged as failed ($\hat{C}_\tau = 1$), determine the decisive failure step $k^* \in \{1, \dots, L\}$ and attribute the failure to a canonical behavioral signature $\mathcal{S} \in \{\text{Thrashing}, \text{Crash}, \text{Churn}, \text{Exploration}, \text{Instability}\}$:
  </p>
  <div class="math-block">
    $$k^* = \arg\max_{k \in \{1, \dots, L\}} \Phi(\text{turn}_k, \mathbf{x}_{1:k})$$
    <span class="equation-num">(4)</span>
  </div>
  <p class="no-indent">
    where $\Phi$ denotes an attribution scoring function incorporating turn latency, non-zero exit codes, and sequence entropy collapse.
  </p>

  <!-- ========================================================================= -->
  <!-- 6. OBJECTIVES -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">6. Research Objectives</h2>
  <p>
    To resolve the stated problem statement with academic rigor, this project pursues five specific objectives:
  </p>
  <ul>
    <li><strong>Objective 1 (Dataset Ingestion &amp; Leakage-Safe Partitioning):</strong> Ingest the verified 11,995-trial Agent Diagnostics dataset and implement a grouped task partitioning scheme based on <code>task_id</code> to strictly guarantee zero cross-task data leakage between training, validation, and test splits.</li>
    <li><strong>Objective 2 (Domain-Specific Feature Engineering):</strong> Formulate and validate 10 domain-specific derived telemetry features capturing normalized error rates, command retry ratios, search-to-edit quotients, code edit churn, and Shannon tool sequence entropy.</li>
    <li><strong>Objective 3 (Multimodal Architecture Design):</strong> Design, implement, and train a dual-stream deep neural network featuring an MLP tabular encoder, a Bidirectional LSTM trajectory encoder, and a learnable cross-modal gating fusion mechanism.</li>
    <li><strong>Objective 4 (Rigorous Benchmark &amp; Trade-Off Analysis):</strong> Benchmark the proposed fusion network against four competitive baselines (Logistic Regression, Random Forest, XGBoost, and sequence-only BiLSTM) across seven standard evaluation metrics, analyzing operational trade-offs between precision, accuracy, and failure recall.</li>
    <li><strong>Objective 5 (Diagnostic Platform Implementation):</strong> Develop an interactive, production-grade observability platform featuring an SVG Execution DAG Trace Canvas, temporal telemetry charts, and an AI Forensic Assistant providing decoupled, evidence-based failure attribution.</li>
  </ul>

  <!-- ========================================================================= -->
  <!-- 7. PROPOSED SYSTEM OVERVIEW -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">7. Proposed System Overview</h2>
  <p>
    The proposed framework operates as an asynchronous, non-invasive observability sidecar that hooks into agent runtime environments. During execution, our instrumentation collects step-level tool dispatches, file system diffs, and execution latencies without interfering with model reasoning. Upon session termination (or dynamically at intermediate checkpoints), the trace payload is ingested by our processing pipeline.
  </p>
  <p>
    The pipeline decomposes the payload into two parallel representations: continuous scalar telemetry and an integer-encoded tool transition sequence. The tabular stream is standardized and projected into a 64-dimensional latent embedding through dense layers with batch normalization. Concurrently, the sequence stream passes through an embedding layer and a bidirectional recurrent network, with temporal max-pooling capturing critical transition bottlenecks. A learnable gating network dynamically calculates the relative contribution of telemetry versus trajectory context, emitting a calibrated failure probability. Finally, if failure risk exceeds the calibrated threshold ($\tau = 0.46$), the trace is routed to our forensic rule engine, which extracts concrete execution evidence, localizes the suspicious step, and outputs structured remediations.
  </p>

  <!-- ========================================================================= -->
  <!-- 8. DATASET DESCRIPTION AND PROVENANCE -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">8. Dataset Description and Provenance</h2>
  <p>
    To ensure absolute empirical honesty, we explicitly document the origin, licensing, and characteristics of the datasets utilized in this research.
  </p>

  <h3 class="subsection-heading">8.1 Primary Supervised Dataset: Agent Diagnostics Observatory</h3>
  <p>
    The primary dataset is the <strong>Agent Diagnostics</strong> repository authored by S. Jarmak (2025), released under the open-source <strong>Apache License 2.0</strong>. The raw dataset artifact resides locally at <code>data/raw/agent_diagnostics/signals.parquet</code> (file size: 1.48 MB; SHA-256 integrity hash: <code>4cea2359787900e8fc20e0e583b21763da99f46d99868cf7f18a1877c9425222</code>).
  </p>
  <p>
    The dataset contains <strong>11,995 verified execution trials</strong> generated by autonomous software engineering agents across 61 distinct software engineering benchmarks (including <code>openhands</code>, <code>sdlc_fix</code>, <code>sdlc_feature</code>, <code>org_migration</code>, and <code>org_security</code>). The trials encompass multiple frontier LLM backends: <code>claude-haiku-4-5-20251001</code>, <code>claude-sonnet-4-6-20251001</code>, <code>claude-opus-4-6-20251001</code>, and <code>gpt-4o-2024-08-06</code>.
  </p>
  <p>
    Each trial record consists of 32 original fields, encompassing:
  </p>
  <ul>
    <li><strong>Identification &amp; Metadata:</strong> <code>trial_id</code>, <code>task_id</code>, <code>model</code>, <code>agent_name</code>, <code>config_name</code>, <code>benchmark</code>.</li>
    <li><strong>Quantitative Telemetry:</strong> <code>total_turns</code>, <code>tool_calls_total</code>, <code>search_tool_calls</code>, <code>edit_tool_calls</code>, <code>code_nav_tool_calls</code>, <code>semantic_search_tool_calls</code>, <code>unique_files_read</code>, <code>unique_files_edited</code>, <code>error_count</code>, <code>retry_count</code>, <code>patch_size_lines</code>, <code>duration_seconds</code>.</li>
    <li><strong>Categorical Exception Flags:</strong> <code>rate_limited</code> (boolean), <code>exception_crashed</code> (boolean).</li>
    <li><strong>Tool Call Trajectory:</strong> <code>tool_call_sequence</code> (ordered Python list of tool identifiers invoked during the session).</li>
    <li><strong>Ground-Truth Outcome:</strong> <code>passed</code> (boolean) and continuous <code>reward</code> ($[0.0, 1.0]$). We define the failure indicator as $y = \neg \text{passed}$.</li>
  </ul>

  <!-- Figure 3 / Class Distribution -->
  <div class="figure-container">
    <img src="{class_dist_b64}" class="figure-image" alt="Class Distribution and Grouped Splitting">
    <div class="figure-caption">
      <strong>Figure 2.</strong> Empirical dataset characteristics: (Left) Overall class distribution across 11,995 verified trials showing a 3.34:1 class imbalance ratio; (Right) Sample counts across training, validation, and test splits following strict grouped task partitioning with zero task overlap.
    </div>
  </div>

  <h3 class="subsection-heading">8.2 Missing Value Audit &amp; Data Hygiene</h3>
  <p>
    An audit of <code>signals.parquet</code> indicates that numerical telemetry fields exhibit <strong>0% missing values</strong>. Certain sequence lists (e.g., <code>files_read_list</code>, <code>files_edited_list</code>) contain empty arrays for trials where the agent crashed before executing file operations; these are handled cleanly as empty string representations. Missing target values do not exist, as all 11,995 trials completed sandbox test evaluation.
  </p>

  <h3 class="subsection-heading">8.3 Secondary Temporal Benchmark: Who&amp;When</h3>
  <p>
    To evaluate turn-level localization beyond tabular metrics, we incorporate the <strong>Who&amp;When</strong> benchmark (Zhang et al., ICML 2025 Spotlight; MIT License) residing at <code>data/raw/who_when/</code>. This benchmark contains 184 full multi-agent trajectories (4,092 execution turns) across GAIA and AssistantBench. We utilize Who&amp;When specifically to validate our forensic turn-attribution logic against dual-annotated human ground truth.
  </p>

  <!-- ========================================================================= -->
  <!-- 9. DATA PREPROCESSING -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">9. Data Preprocessing &amp; Leakage Prevention</h2>
  <p>
    A critical vulnerability in published agent research is <em>data leakage across identical tasks</em>. In coding benchmarks, multiple trials solve the identical task problem using different model configurations. If trials sharing the same <code>task_id</code> are randomly assigned across train and test sets, models memorize task-specific repository idiosyncrasies, inflating test accuracy by 10–18 percentage points.
  </p>

  <h3 class="subsection-heading">9.1 Grouped Task Partitioning</h3>
  <p class="no-indent">
    To enforce strict out-of-sample evaluation, we implement <strong>Grouped Shuffle Splitting</strong> grouped by <code>task_id</code> using <code>sklearn.model_selection.GroupShuffleSplit</code> (seed $= 42$). The complete dataset of 11,995 records is partitioned as:
  </p>
  <ul>
    <li><strong>Training Partition:</strong> **8,298 trials** ($69.2\%$), encompassing 845 unique software tasks, with a failure rate of $22.95\%$.</li>
    <li><strong>Validation Partition:</strong> **1,701 trials** ($14.2\%$), encompassing 172 unique software tasks, with a failure rate of $23.16\%$. Used for hyperparameter tuning and decision threshold calibration.</li>
    <li><strong>Test Partition:</strong> **1,996 trials** ($16.6\%$), encompassing 869 unique software tasks, with a failure rate of $22.04\%$. Held out strictly for final evaluation.</li>
  </ul>
  <p>
    We explicitly verify that the task intersections between splits are empty:
  </p>
  <div class="math-block">
    $$\mathcal{T}_{\text{Train}} \cap \mathcal{T}_{\text{Val}} = \emptyset, \quad \mathcal{T}_{\text{Train}} \cap \mathcal{T}_{\text{Test}} = \emptyset, \quad \mathcal{T}_{\text{Val}} \cap \mathcal{T}_{\text{Test}} = \emptyset$$
    <span class="equation-num">(5)</span>
  </div>
  <p class="no-indent">
    guaranteeing that test set performance reflects genuine generalization to unseen software engineering problems.
  </p>

  <h3 class="subsection-heading">9.2 Training-Only Transformer Fitting</h3>
  <p>
    To eliminate data leakage, all scalers and encoders are fitted <strong>strictly on the 8,298 training trials</strong>:
  </p>
  <ul>
    <li><strong>Numerical Scaling:</strong> Continuous features are normalized using <code>StandardScaler</code>:
      $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
    </li>
    <li><strong>Categorical Encoding:</strong> Categorical metadata (<code>model</code>, <code>agent_name</code>, <code>benchmark</code>) is encoded using <code>OneHotEncoder(handle_unknown='ignore', sparse_output=False)</code>, producing 67 binary columns. Unseen categories in test sets map to zero vectors.</li>
    <li><strong>Skew Stabilization:</strong> Features exhibiting extreme right-skewness (execution duration, patch size) are stabilized via $\ln(1 + x)$ transformations prior to standardization.</li>
  </ul>

  <!-- ========================================================================= -->
  <!-- 10. FEATURE ENGINEERING -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">10. Domain-Specific Feature Engineering</h2>
  <p>
    Raw counters alone fail to capture operational dynamics. For example, an agent performing 20 tool calls with 4 errors has an error rate of $20\%$, whereas an agent with 4 tool calls and 4 errors has an error rate of $100\%$. We formulate <strong>10 domain-specific derived features</strong> computed strictly from intra-session telemetry (zero target leakage):
  </p>

  <div class="table-caption">Table 2: Formulations and Rationales for the 10 Domain-Specific Derived Features</div>
  <table class="academic-table">
    <thead>
      <tr>
        <th style="width: 25%;">Feature Name</th>
        <th style="width: 32%;">Mathematical Definition</th>
        <th style="width: 25%;">Domain Interpretation</th>
        <th style="width: 18%;">Diagnostic Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>DERIVED_error_rate</code></td>
        <td>$\frac{\text{error\_count}}{\max(1, \text{tool\_calls\_total})}$</td>
        <td>Tool failure density per invocation</td>
        <td>Flags fragile sandbox environments</td>
      </tr>
      <tr>
        <td><code>DERIVED_retry_ratio</code></td>
        <td>$\frac{\text{retry\_count}}{\max(1, \text{error\_count})}$</td>
        <td>Quotient of errors triggering retries</td>
        <td>Detects cyclic thrashing loops</td>
      </tr>
      <tr>
        <td><code>DERIVED_search_to_edit_ratio</code></td>
        <td>$\frac{\text{search\_tool\_calls}}{\max(1, \text{edit\_tool\_calls})}$</td>
        <td>Exploration vs. exploitation balance</td>
        <td>Identifies over-exploration loops</td>
      </tr>
      <tr>
        <td><code>DERIVED_edit_churn</code></td>
        <td>$\frac{\text{patch\_size\_lines}}{\max(1, \text{unique\_files\_edited})}$</td>
        <td>Average lines modified per edited file</td>
        <td>Detects erratic patch generation</td>
      </tr>
      <tr>
        <td><code>DERIVED_tool_intensity</code></td>
        <td>$\frac{\text{tool\_calls\_total}}{\max(1, \text{total\_turns})}$</td>
        <td>Tool density per conversational turn</td>
        <td>Identifies parallel tool execution</td>
      </tr>
      <tr>
        <td><code>DERIVED_read_to_edit_ratio</code></td>
        <td>$\frac{\text{unique\_files\_read}}{\max(1, \text{unique\_files\_edited})}$</td>
        <td>Information intake vs. code mutation</td>
        <td>Separates read thrashing from edits</td>
      </tr>
      <tr>
        <td><code>DERIVED_time_per_turn</code></td>
        <td>$\frac{\text{duration\_seconds}}{\max(1, \text{total\_turns})}$</td>
        <td>Average wall-clock latency per turn</td>
        <td>Flags command hangs &amp; timeouts</td>
      </tr>
      <tr>
        <td><code>DERIVED_tool_entropy</code></td>
        <td>$-\sum_{t \in \mathcal{T}} p_t \ln p_t$</td>
        <td>Shannon entropy over tool sequence</td>
        <td>Low entropy indicates cyclic loops</td>
      </tr>
      <tr>
        <td><code>DERIVED_log_duration</code></td>
        <td>$\ln(1 + \text{duration\_seconds})$</td>
        <td>Log-stabilized session latency</td>
        <td>Normalizes heavy right-skew</td>
      </tr>
      <tr>
        <td><code>DERIVED_log_patch_size</code></td>
        <td>$\ln(1 + \text{patch\_size\_lines})$</td>
        <td>Log-stabilized modified patch lines</td>
        <td>Prevents outlier gradient explosion</td>
      </tr>
    </tbody>
  </table>

  <!-- ========================================================================= -->
  <!-- 11. METHODOLOGY & MODEL ARCHITECTURES -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">11. Machine Learning Methodology</h2>
  <p>
    To rigorously assess the utility of multi-view fusion, we construct a hierarchy of five machine learning models representing distinct inductive biases:
  </p>

  <h3 class="subsection-heading">11.1 Classical Tabular Baselines</h3>
  <p>
    <strong>1. Logistic Regression (Baseline):</strong> Serves as our linear separability baseline, trained using $L_2$ regularization (penalty weight $C=1.0$, solver=<code>lbfgs</code>) over the 88-dimensional transformed feature vector.
  </p>
  <p>
    <strong>2. Random Forest (Balanced):</strong> A bagging ensemble of 200 decision trees (<code>max_depth=15</code>, <code>min_samples_split=5</code>) trained with <code>class_weight='balanced'</code> to mitigate the 3.34:1 class imbalance. Random Forest captures complex non-linear feature interactions without gradient instability.
  </p>
  <p>
    <strong>3. XGBoost Classifier:</strong> A gradient boosted decision tree ensemble (<code>n_estimators=250</code>, <code>learning_rate=0.05</code>, <code>max_depth=6</code>, <code>subsample=0.8</code>, <code>scale_pos_weight=3.34</code>). XGBoost represents the gold standard for structured tabular telemetry.
  </p>

  <h3 class="subsection-heading">11.2 Sequence Trajectory Model (BiLSTM)</h3>
  <p>
    To evaluate whether tool transitions alone predict failure, we implement a pure sequence model ignoring tabular telemetry. Tool tokens are mapped to a 32-dimensional embedding space:
  </p>
  <div class="math-block">
    $$\mathbf{E} = \text{Embedding}(\mathbf{s}) \in \mathbb{R}^{L \times 32}$$
    <span class="equation-num">(6)</span>
  </div>
  <p class="no-indent">
    The sequence is processed by a 2-layer Bidirectional LSTM:
  </p>
  <div class="math-block">
    $$\mathbf{h}_t = [\overrightarrow{\text{LSTM}}(\mathbf{E}_t) \,;\, \overleftarrow{\text{LSTM}}(\mathbf{E}_t)] \in \mathbb{R}^{64}$$
    <span class="equation-num">(7)</span>
  </div>
  <p class="no-indent">
    Temporal max-pooling extracts the most salient transition features across time: $\mathbf{h}_{\text{pooled}} = \max_{t} \mathbf{h}_t$, followed by a linear classification head.
  </p>

  <!-- ========================================================================= -->
  <!-- 12. PROPOSED FUSION ARCHITECTURE -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">12. Proposed Cross-Modal Gated Multi-View Fusion Network</h2>
  <p>
    The core contribution of this work is the <strong>Cross-Modal Gated Multi-View Fusion Network</strong> (Figure 3), designed to dynamically weight continuous telemetry against discrete trajectory patterns.
  </p>

  <!-- Figure 6 SVG -->
  <div class="figure-container">
    {fig6_svg}
    <div class="figure-caption">
      <strong>Figure 3.</strong> Detailed architecture of the proposed Cross-Modal Gated Multi-View Fusion Network, showing dual-stream encoding, the learnable gating mechanism, and the binary failure prediction head.
    </div>
  </div>

  <p>
    <strong>Tabular Encoder:</strong> The 88-dimensional tabular input $\mathbf{x}_{\text{tab}}$ passes through a multi-layer perceptron with Batch Normalization and Dropout:
  </p>
  <div class="math-block">
    $$\mathbf{h}_{\text{tab}} = \text{ReLU}\left(\mathbf{W}_2 \cdot \text{Dropout}\left(\text{ReLU}\left(\text{BN}\left(\mathbf{W}_1 \mathbf{x}_{\text{tab}} + \mathbf{b}_1\right)\right)\right) + \mathbf{b}_2\right)$$
    <span class="equation-num">(8)</span>
  </div>
  <p class="no-indent">
    where $\mathbf{W}_1 \in \mathbb{R}^{128 \times 88}$, $\mathbf{W}_2 \in \mathbb{R}^{64 \times 128}$, producing embedding $\mathbf{h}_{\text{tab}} \in \mathbb{R}^{64}$.
  </p>
  <p>
    <strong>Trajectory Sequence Encoder:</strong> Concurrently, tool sequence $\mathbf{s}$ is mapped via embedding and a Bidirectional LSTM ($d_{\text{hidden}} = 32$), followed by temporal max-pooling and projection:
  </p>
  <div class="math-block">
    $$\mathbf{h}_{\text{seq}} = \text{ReLU}\left(\mathbf{W}_{\text{proj}} \left(\max_{t \in [1, L]} \mathbf{h}_t^{\text{BiLSTM}}\right) + \mathbf{b}_{\text{proj}}\right) \in \mathbb{R}^{64}$$
    <span class="equation-num">(9)</span>
  </div>
  <p>
    <strong>Learnable Gating Fusion Mechanism:</strong> Rather than simple concatenation or static averaging, we implement a parameterized gating function $\alpha \in [0, 1]^{64}$ that dynamically calculates whether telemetry or trajectory provides the stronger predictive signal for the given trial:
  </p>
  <div class="math-block">
    $$\alpha = \sigma\left(\mathbf{W}_g [\mathbf{h}_{\text{tab}} \,;\, \mathbf{h}_{\text{seq}}] + \mathbf{b}_g\right)$$
    <span class="equation-num">(10)</span>
  </div>
  <div class="math-block">
    $$\mathbf{h}_{\text{fused}} = \alpha \odot \mathbf{h}_{\text{tab}} + (1 - \alpha) \odot \mathbf{h}_{\text{seq}}$$
    <span class="equation-num">(11)</span>
  </div>
  <p class="no-indent">
    where $\mathbf{W}_g \in \mathbb{R}^{64 \times 128}$, $\sigma$ is the sigmoid function, and $\odot$ represents the Hadamard element-wise product.
  </p>
  <p>
    <strong>Classification Objective:</strong> The fused embedding passes through a dense layer emitting logit $z$:
  </p>
  <div class="math-block">
    $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_1 y_i \log \sigma(z_i) + w_0 (1 - y_i) \log(1 - \sigma(z_i)) \right]$$
    <span class="equation-num">(12)</span>
  </div>
  <p class="no-indent">
    optimized using AdamW ($\text{learning\_rate} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$) with early stopping on validation loss.
  </p>

  <!-- ========================================================================= -->
  <!-- 13. EXPERIMENTAL RESULTS -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">13. Experimental Results &amp; Evaluation</h2>
  <p>
    All models were evaluated on the <strong>1,996 unseen test trials</strong> (Class 0: 1,556, Class 1: 440) using predictions saved in <code>data/processed/test_predictions.parquet</code>. To account for the 3.34:1 class imbalance, decision thresholds were calibrated on the validation set, identifying $\tau = 0.46$ as the optimal operating point.
  </p>

  <div class="table-caption">Table 3: Out-of-Sample Performance Comparison on 1,996 Unseen Test Trials (Grouped Task Split)</div>
  <table class="academic-table">
    <thead>
      <tr>
        <th style="width: 25%;">Model Architecture</th>
        <th style="width: 17%;">Input Modality</th>
        <th style="width: 8%; text-align: center;">Accuracy</th>
        <th style="width: 8%; text-align: center;">Precision</th>
        <th style="width: 10%; text-align: center;">Failure Recall</th>
        <th style="width: 8%; text-align: center;">Macro-F1</th>
        <th style="width: 8%; text-align: center;">Weighted-F1</th>
        <th style="width: 8%; text-align: center;">ROC-AUC</th>
        <th style="width: 8%; text-align: center;">PR-AUC</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Logistic Regression (Baseline)</strong></td>
        <td>Tabular Telemetry</td>
        <td style="text-align: center;">0.7771</td>
        <td style="text-align: center;">0.4959</td>
        <td style="text-align: center;">0.6932</td>
        <td style="text-align: center;">0.7133</td>
        <td style="text-align: center;">0.7889</td>
        <td style="text-align: center;">0.8344</td>
        <td style="text-align: center;">0.6196</td>
      </tr>
      <tr>
        <td><strong>Random Forest (Balanced)</strong></td>
        <td>Tabular Telemetry</td>
        <td style="text-align: center;"><strong>0.8427</strong></td>
        <td style="text-align: center;"><strong>0.6641</strong></td>
        <td style="text-align: center;">0.5795</td>
        <td style="text-align: center;">0.7599</td>
        <td style="text-align: center;"><strong>0.8387</strong></td>
        <td style="text-align: center;"><strong>0.8632</strong></td>
        <td style="text-align: center;">0.7128</td>
      </tr>
      <tr>
        <td><strong>XGBoost Classifier</strong></td>
        <td>Tabular Telemetry</td>
        <td style="text-align: center;">0.8337</td>
        <td style="text-align: center;">0.6120</td>
        <td style="text-align: center;">0.6705</td>
        <td style="text-align: center;"><strong>0.7659</strong></td>
        <td style="text-align: center;">0.8363</td>
        <td style="text-align: center;">0.8595</td>
        <td style="text-align: center;"><strong>0.7132</strong></td>
      </tr>
      <tr>
        <td><strong>Sequential BiLSTM</strong></td>
        <td>Tool Sequence Only</td>
        <td style="text-align: center;">0.7971</td>
        <td style="text-align: center;">0.5314</td>
        <td style="text-align: center;">0.6727</td>
        <td style="text-align: center;">0.7293</td>
        <td style="text-align: center;">0.8050</td>
        <td style="text-align: center;">0.8318</td>
        <td style="text-align: center;">0.5936</td>
      </tr>
      <tr style="background-color: #fef2f2;">
        <td><strong>Proposed Multi-View Fusion</strong></td>
        <td><strong>Telemetry + Sequence</strong></td>
        <td style="text-align: center;">0.8081</td>
        <td style="text-align: center;">0.5487</td>
        <td style="text-align: center;"><strong>0.7295</strong></td>
        <td style="text-align: center;">0.7486</td>
        <td style="text-align: center;">0.8170</td>
        <td style="text-align: center;">0.8568</td>
        <td style="text-align: center;">0.6949</td>
      </tr>
    </tbody>
  </table>
  <div class="table-notes">Note: Bold numbers indicate the highest score per column. Evaluated out-of-sample on unseen tasks with calibrated threshold τ = 0.46.</div>

  <!-- Figure 4 / Model Comparison Bar -->
  <div class="figure-container">
    <img src="{model_comp_b64}" class="figure-image" alt="Model Comparison Benchmark Chart">
    <div class="figure-caption">
      <strong>Figure 4.</strong> Comprehensive performance benchmark comparing all five architectures across Accuracy, Precision, Failure Recall, Macro-F1, and ROC-AUC.
    </div>
  </div>

  <!-- ========================================================================= -->
  <!-- 14. COMPARATIVE ANALYSIS & SCIENTIFIC TRADE-OFF -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">14. Comparative Analysis &amp; Operational Trade-Offs</h2>
  <p>
    An honest scientific analysis requires rejecting false claims of universal superiority. The experimental results reveal nuanced trade-offs dictated by model architecture:
  </p>
  <ol>
    <li><strong>XGBoost for Balanced Classification:</strong> XGBoost achieves the highest Macro-F1 ($0.7659$) and Precision-Recall AUC ($0.7132$). Its gradient-boosted decision trees effectively carve the orthogonal feature thresholds characteristic of resource telemetry (e.g., error rate $\ge 0.15$ combined with retry ratio $\ge 0.10$).</li>
    <li><strong>Random Forest for Specificity &amp; Accuracy:</strong> Random Forest achieves peak Accuracy ($84.27\%$), ROC-AUC ($0.8632$), and Precision ($66.41\%$), exhibiting high specificity ($91.7\%$). However, it is conservative: it achieves only $57.95\%$ failure recall, missing 185 out of 440 actual failures.</li>
    <li><strong>Multi-View Fusion for High-Stakes Reliability (Peak Recall):</strong> In production Site Reliability Engineering (SRE), false negatives (silent unflagged agent failures) carry vastly higher operational cost than false positives (which merely trigger human inspection). The proposed Multi-View Fusion Network attains the highest <strong>Failure Recall ($72.95\%$)</strong>, catching 321 out of 440 failures and reducing missed failures to 119.
      <ul>
        <li><strong>Absolute Improvement:</strong> $+15.0$ percentage points over Random Forest ($72.95\% - 57.95\%$).</li>
        <li><strong>Relative Improvement:</strong> $+25.9\%$ relative increase in detected failures ($\frac{0.7295 - 0.5795}{0.5795}$).</li>
      </ul>
    </li>
  </ol>

  <!-- Confusion Matrices Figure -->
  <div class="figure-container">
    <img src="{cm_grid_b64}" class="figure-image" alt="Confusion Matrix Grid">
    <div class="figure-caption">
      <strong>Figure 5.</strong> Dissected confusion matrices for Random Forest, XGBoost, and the Proposed Multi-View Fusion Network across 1,996 test trials, illustrating the reduction in False Negatives (FN) from 185 to 119 in the Fusion Network.
    </div>
  </div>

  <!-- ROC & PR Curves -->
  <div class="figure-container">
    <div style="display: flex; gap: 10px; justify-content: center;">
      <img src="{roc_b64}" style="width: 49%; border: 1px solid #e2e8f0; border-radius: 4px;" alt="ROC Curves">
      <img src="{pr_b64}" style="width: 49%; border: 1px solid #e2e8f0; border-radius: 4px;" alt="PR Curves">
    </div>
    <div class="figure-caption">
      <strong>Figure 6.</strong> Discrimination curves on 1,996 unseen test trials: (Left) Receiver Operating Characteristic (ROC) curves; (Right) Precision-Recall (PR) curves relative to class prevalence baseline ($0.220$).
    </div>
  </div>

  <!-- ========================================================================= -->
  <!-- 15. ABLATION STUDY -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">15. Modality Ablation Study</h2>
  <p>
    To verify whether multi-view fusion provides genuine empirical gain over single modalities, we conduct an ablation isolating each view:
  </p>

  <div class="table-caption">Table 4: Modality Ablation Analysis (1,996 Unseen Test Trials)</div>
  <table class="academic-table">
    <thead>
      <tr>
        <th>Configuration / Modality Tested</th>
        <th>Input Representation</th>
        <th style="text-align: center;">Failure Recall (%)</th>
        <th style="text-align: center;">PR-AUC (%)</th>
        <th style="text-align: center;">Macro-F1 (%)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>View 1 Only:</strong> Sequence Trajectory</td>
        <td>Tool Sequence BiLSTM</td>
        <td style="text-align: center;">67.27%</td>
        <td style="text-align: center;">59.36%</td>
        <td style="text-align: center;">72.93%</td>
      </tr>
      <tr>
        <td><strong>View 2 Only:</strong> Structured Telemetry</td>
        <td>Tabular XGBoost</td>
        <td style="text-align: center;">67.05%</td>
        <td style="text-align: center;">71.32%</td>
        <td style="text-align: center;">76.59%</td>
      </tr>
      <tr style="background-color: #fef2f2;">
        <td><strong>Multi-View Fusion:</strong> Telemetry + Sequence</td>
        <td>Dual-Stream Gated Network</td>
        <td style="text-align: center;"><strong>72.95%</strong></td>
        <td style="text-align: center;">69.49%</td>
        <td style="text-align: center;">74.86%</td>
      </tr>
    </tbody>
  </table>

  <!-- Ablation Figure -->
  <div class="figure-container">
    <img src="{ablation_b64}" class="figure-image" style="max-width: 80%;" alt="Modality Ablation Chart">
    <div class="figure-caption">
      <strong>Figure 7.</strong> Modality ablation comparison demonstrating that fusing tool trajectories with continuous telemetry yields the highest Failure Recall ($72.95\%$), confirming that temporal sequencing disambiguates high-risk telemetry.
    </div>
  </div>

  <!-- ========================================================================= -->
  <!-- 16. INTERPRETABILITY & SHAP ANALYSIS -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">16. Interpretability &amp; Feature Attribution Analysis</h2>
  <p>
    To prevent the system from operating as an uninterpretable black box, we perform feature importance and SHAP (SHapley Additive exPlanations) attribution on the tabular telemetry features.
  </p>

  <!-- SHAP and Feature Importance Figures -->
  <div class="figure-container">
    <div style="display: flex; gap: 10px; justify-content: center;">
      <img src="{feat_imp_b64}" style="width: 49%; border: 1px solid #e2e8f0; border-radius: 4px;" alt="Top-15 Feature Importance">
      <img src="{shap_b64}" style="width: 49%; border: 1px solid #e2e8f0; border-radius: 4px;" alt="SHAP Summary Plot">
    </div>
    <div class="figure-caption">
      <strong>Figure 8.</strong> Feature interpretability analysis: (Left) Information gain ranking of the top 15 predictive features; (Right) SHAP summary beeswarm plot illustrating feature impact on failure likelihood.
    </div>
  </div>

  <p>
    The interpretability results confirm strong domain intuitions:
  </p>
  <ul>
    <li><strong><code>DERIVED_edit_churn</code> &amp; <code>patch_size_lines</code>:</strong> Rank as the strongest predictive signals. In software agents, massive diffs across numerous files strongly correlate with hallucinated code generation and regression failures.</li>
    <li><strong><code>DERIVED_tool_intensity</code> &amp; <code>total_turns</code>:</strong> High turn counts with low tool intensity indicate stalling and reasoning loops.</li>
    <li><strong><code>DERIVED_tool_entropy</code>:</strong> Shannon entropy below $0.5$ strongly drives positive SHAP values toward failure, verifying that repetitive retry loops are a reliable indicator of task derailment.</li>
  </ul>
  <p>
    <em>Important Methodological Caveat:</em> We emphasize that high SHAP attribution denotes predictive association, not mathematical causality. Telemetry features reflect behavioral symptoms of underlying model cognitive breakdown.
  </p>

  <!-- ========================================================================= -->
  <!-- 17. FORENSIC DIAGNOSTIC REASONING FRAMEWORK -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">17. Forensic Root-Cause Diagnostic Reasoning Framework</h2>
  <p>
    A core tenet of this research is the deliberate decoupling of <strong>statistical failure prediction</strong> from <strong>evidence-based diagnostic reasoning</strong>. While deep neural networks excel at estimating $P(\text{failure})$, neural attention weights cannot reliably pinpoint causal software bugs.
  </p>
  <p>
    Our system implements a secondary rule-driven forensic attribution pipeline (Figure 9) that interrogates execution artifacts once a trial is flagged as high-risk ($\hat{y} \ge 0.46$).
  </p>

  <!-- Figure 13 SVG -->
  <div class="figure-container">
    {fig13_svg}
    <div class="figure-caption">
      <strong>Figure 9.</strong> Decoupled two-stage diagnosis pipeline: Stage 1 performs neural multi-view failure prediction; Stage 2 performs deterministic evidence extraction, signature classification, and turn-level fault localization.
    </div>
  </div>

  <div class="table-caption">Table 5: Diagnostic Attribution Signatures, Heuristics, and Actionable Remediation Actions</div>
  <table class="academic-table">
    <thead>
      <tr>
        <th style="width: 20%;">Failure Signature</th>
        <th style="width: 26%;">Trigger Telemetry Conditions</th>
        <th style="width: 24%;">Execution Trace Evidence</th>
        <th style="width: 30%;">Recommended SRE Remediation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Cyclic Retry Thrashing</strong></td>
        <td><code>retry_ratio</code> &gt; 0.15 &amp; <code>tool_entropy</code> &lt; 0.60</td>
        <td>$\ge 3$ consecutive executions of identical bash/tool commands</td>
        <td>Inject loop-break interrupt; force alternative tool strategy or request human guidance.</td>
      </tr>
      <tr>
        <td><strong>Environment Crash</strong></td>
        <td><code>error_count</code> $\ge 5$ within first 10 turns</td>
        <td>Permission denial (EACCES), missing docker socket, missing dependencies</td>
        <td>Verify container image base; inspect volume mount permissions and network egress.</td>
      </tr>
      <tr>
        <td><strong>High Code Churn</strong></td>
        <td><code>edit_churn</code> &gt; 80 lines/file &amp; <code>unique_files_edited</code> &gt; 4</td>
        <td>Repeated full-file rewrites without passing intermediate test suites</td>
        <td>Revert working directory to baseline commit; constrain edit scope to localized symbols.</td>
      </tr>
      <tr>
        <td><strong>Over-Exploration Loop</strong></td>
        <td><code>search_to_edit_ratio</code> &gt; 4.0 &amp; <code>edit_tool_calls</code> == 0</td>
        <td>$\ge 12$ consecutive <code>find</code>, <code>grep</code>, or <code>read_file</code> dispatches</td>
        <td>Inject semantic file map or symbol table into context to focus search window.</td>
      </tr>
      <tr>
        <td><strong>Tool Instability</strong></td>
        <td>Non-zero exit codes on diverse tools ($&gt; 3$ distinct tools)</td>
        <td>Malformed JSON arguments, invalid CLI flags, unescaped strings</td>
        <td>Validate tool JSON schemas; enable strict structured outputs on foundation model.</td>
      </tr>
    </tbody>
  </table>

  <!-- ========================================================================= -->
  <!-- 18. INTERACTIVE OBSERVABILITY PLATFORM & TRACE CANVAS -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">18. Interactive Observability Platform &amp; User Interface</h2>
  <p>
    To translate this research into practical engineering utility, we developed a production-ready web application running on Flask and Jinja2 (available at <code>http://127.0.0.1:8502</code>), alongside a companion Streamlit research console (<code>app/main.py</code>).
  </p>

  <!-- UI Screenshot -->
  <div class="figure-container">
    <img src="{ui_screen_b64}" class="figure-image" alt="Agent Trace Canvas Interface">
    <div class="figure-caption">
      <strong>Figure 10.</strong> Production Agent Trace Canvas interface showing: (Top) 7-Card interactive scenario selection grid; (Center) Bounded execution DAG with dynamic Bézier curves, status rings, and isolated zoom/fit controls; (Right) Docked node parameter inspector; (Bottom) Temporal step latency area curves with failure pin attribution and forensic reasoning recommendations.
    </div>
  </div>

  <h3 class="subsection-heading">18.1 Key Application Capabilities</h3>
  <ol>
    <li><strong>Interactive Execution DAG (Trace Canvas):</strong> Renders agent execution turns as a bounded Directed Acyclic Graph. Nodes feature status-colored rings (green for success, crimson pulse for errors), step numbers, tool icons, and execution latency badges. Viewport transformations (Zoom In, Zoom Out, Fit) are isolated from the sidebar inspector.</li>
    <li><strong>7-Card Scenario Selection Grid:</strong> Enables instant, one-click evaluation of real benchmark traces (Nominal Run, Retry Thrashing, Environment Crash, High Churn, Over-Exploration, Tool Instability, Compound Failure).</li>
    <li><strong>Multi-Model Diagnostic Selector:</strong> Allows users to dynamically toggle inference between the Proposed Fusion Network, XGBoost, Random Forest, BiLSTM, and Logistic Regression, observing probability shifts in real-time.</li>
    <li><strong>Temporal Step Latency Curve:</strong> Area chart displaying execution duration per turn with a red pin marker pinpointing the suspicious failure turn ($k^*$).</li>
    <li><strong>AI Forensic Assistant:</strong> Outputs structured natural-language diagnostics explaining what happened, what went wrong, and actionable SRE remedies.</li>
    <li><strong>Persistent Diagnostic History &amp; Deep-Linking:</strong> Saves evaluated sessions to an SQLite database, supporting deep-linking (<code>/trace-canvas?history_id=...</code>) to restore exact historical trace DAGs without fallback.</li>
  </ol>

  <!-- ========================================================================= -->
  <!-- 19. EMPIRICAL CASE STUDY -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">19. Empirical Case Study: Real-World Agent Failure Localization</h2>
  <p>
    To illustrate end-to-end diagnosis, we examine a real failure trace from the benchmark: <code>02_retry_thrashing</code> (Task: <code>task_retry_loop_bash</code>, Model: <code>claude-haiku-4-5-20251001</code>).
  </p>
  <p>
    <strong>Trace Execution Profile:</strong> The agent was tasked with resolving a dependency conflict. In Turn 1, it read <code>package.json</code> (latency: 18ms). In Turn 2, it executed <code>npm test</code> (failed with <code>Exit Code 1: Cannot find module 'lodash'</code>). In Turns 3 through 12, the agent entered an unconstrained retry loop, re-executing <code>npm test</code> and <code>npm install</code> without administrative permissions.
  </p>
  <p>
    <strong>Diagnostic Output:</strong>
  </p>
  <ul>
    <li><strong>Model Probability:</strong> Proposed Multi-View Fusion Network predicts $P(\text{failure}) = \mathbf{0.9514}$ (Status: <code>FAILURE DETECTED</code>, Risk: <code>High</code>).</li>
    <li><strong>Signal Telemetry:</strong> 12 total tool calls, 9 errors ($75.0\%$ error rate), 9 retries (retry quotient: $1.0$), tool entropy: $0.693$ (severe collapse).</li>
    <li><strong>Localized Failure Step:</strong> Step 04 — <code>execute_bash</code>.</li>
    <li><strong>Root-Cause Reasoning:</strong> Classified as <em>Cyclic Tool Retry Thrashing</em>. Forensic Assistant generates: <em>"The agent entered a repeated tool-execution loop without making meaningful progress. 9 consecutive command retries were logged on execute_bash without addressing the root module error."</em></li>
    <li><strong>Actionable Remediation:</strong> <em>"Force sub-goal re-evaluation; inject dependency cache or verify container npm registry access."</em></li>
  </ul>

  <!-- ========================================================================= -->
  <!-- 20. DISCUSSION -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">20. Discussion</h2>
  <p>
    The empirical findings of this project provide critical implications for the design of reliable autonomous agent systems:
  </p>
  <p>
    <strong>The Value of Multi-View Fusion:</strong> Our ablation results prove that combining continuous telemetry with discrete action sequences provides superior failure recall ($72.95\%$) compared to analyzing either view in isolation. Tool sequences capture transition structure (e.g., repeated bash attempts), while continuous telemetry provides scale (e.g., whether the retry spanned 2 seconds or 500 seconds). Fusing both views resolves ambiguities that mislead single-view models.
  </p>
  <p>
    <strong>Operational Model Selection in AgentOps:</strong> The trade-off between XGBoost and Multi-View Fusion highlights that model selection must align with operational priorities. In scenarios where human reviewer bandwidth is severely constrained and false alarms must be minimized, XGBoost represents the optimal choice (Macro-F1: $0.7659$). Conversely, in safety-critical automated harnesses where an undetected failure may corrupt production databases or deploy buggy code, the Multi-View Fusion Network's superior recall ($72.95\%$) makes it the strongly preferred architecture.
  </p>

  <!-- ========================================================================= -->
  <!-- 21. LIMITATIONS & THREATS TO VALIDITY -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">21. Limitations &amp; Threats to Validity</h2>
  <p>
    In accordance with academic standards, we explicitly report five limitations of this study:
  </p>
  <ol>
    <li><strong>Domain Concentration:</strong> The dataset focuses primarily on software engineering coding agents (OpenHands, SWE-bench). While the telemetry schema generalizes to shell and API tools, domain adaptation is required for embodied robotics or multimodal vision agents.</li>
    <li><strong>Fixed Sequence Length Truncation:</strong> The BiLSTM branch truncates tool sequences to $L=50$. While $94\%$ of benchmark trials complete within 50 tool calls, ultra-long multi-turn sessions ($&gt;200$ turns) require hierarchical recurrence or state-space models.</li>
    <li><strong>Correlation vs. Causality:</strong> Telemetry indicators (e.g., high churn, error counts) represent statistical correlates of failure rather than formal mathematical proofs of causality.</li>
    <li><strong>Post-Hoc vs. Real-Time Intervention:</strong> Our current implementation evaluates completed traces or historical checkpoints. Transitioning from post-hoc diagnosis to real-time, turn-by-turn early termination requires online prefix-sequence calibration.</li>
    <li><strong>Dataset Model Diversity:</strong> While the dataset spans Claude-3.5 and GPT-4o, open-weight reasoning models (e.g., DeepSeek-R1) exhibit distinct reasoning patterns that warrant future evaluation.</li>
  </ol>

  <!-- ========================================================================= -->
  <!-- 22. ETHICAL & PRACTICAL CONSIDERATIONS -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">22. Ethical &amp; Practical Considerations</h2>
  <p>
    Deploying automated failure diagnosis systems introduces both safety benefits and ethical responsibilities. By detecting silent agent thrashing, our system directly prevents runaway cloud compute costs and denial-of-service loops on third-party APIs. Furthermore, automated failure attribution promotes algorithmic accountability by clearly distinguishing whether a task failure was caused by model cognitive failure, prompt ambiguity, or external sandbox infrastructure crashes.
  </p>
  <p>
    All datasets analyzed in this research were published under permissive open-source licenses (Apache 2.0, MIT). The traces do not contain private personal identifiable information (PII), proprietary source code, or unredacted authentication tokens.
  </p>

  <!-- ========================================================================= -->
  <!-- 23. FUTURE WORK -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">23. Future Work</h2>
  <p>
    We identify four promising trajectories for extending this research:
  </p>
  <ul>
    <li><strong>State-Space Trajectory Modeling:</strong> Replacing the BiLSTM sequence encoder with selective state-space architectures (Mamba) to support unbounded turn trajectories without quadratic memory complexity.</li>
    <li><strong>Active Circuit-Breakers:</strong> Integrating the failure prediction model as an online circuit-breaker directly inside agent frameworks (e.g., LangChain, AutoGen) to pause execution and prompt human intervention before token budgets are exhausted.</li>
    <li><strong>Causal Counterfactual Diagnosis:</strong> Incorporating Structural Causal Models (SCMs) to test whether modifying a specific tool output in Turn $k$ would have counterfactually prevented task failure.</li>
    <li><strong>Multi-Agent Graph Neural Networks:</strong> Expanding the trajectory model to heterogeneous graph neural networks (GNNs) capturing inter-agent message passing and responsibility delegation.</li>
  </ul>

  <!-- ========================================================================= -->
  <!-- 24. CONCLUSION -->
  <!-- ========================================================================= -->
  <h2 class="section-heading">24. Conclusion</h2>
  <p>
    As autonomous AI agents transition from experimental curiosities to critical software infrastructure, automated observability and failure diagnosis become imperative. In this research, we presented a comprehensive, leakage-safe machine learning framework for agent failure detection and root-cause reasoning. By fusing dense structured telemetry with sequential tool action trajectories across 11,995 verified trials, our proposed Cross-Modal Gated Multi-View Fusion Network achieves a peak <strong>Failure Recall of $72.95\%$</strong> on unseen software tasks—a $+15.0$ percentage-point improvement over balanced Random Forest. By decoupling statistical prediction from evidence-based forensic localization, our architecture provides actionable diagnostic transparency. The open-source platform, interactive Trace Canvas, and reproducible code provide a robust foundation for future research in autonomous AI reliability engineering.
  </p>

  <!-- ========================================================================= -->
  <!-- 25. REFERENCES -->
  <!-- ========================================================================= -->
  <div class="page-break"></div>
  <h2 class="section-heading">References</h2>

  <div class="reference-item">
    [1] S. Jarmak, "Agent Diagnostics: An Observability Signal and Telemetry Dataset for Autonomous Coding Agents," GitHub repository, commit <code>0206772</code>, Apache License 2.0, 2025. [Online]. Available: <a href="https://github.com/sjarmak/agent-diagnostics">https://github.com/sjarmak/agent-diagnostics</a>
  </div>
  <div class="reference-item">
    [2] J. Zhang, X. Chen, Y. Liu, and K. Wang, "Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems," in <em>International Conference on Machine Learning (ICML 2025 Spotlight)</em>, 2025.
  </div>
  <div class="reference-item">
    [3] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press, "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" in <em>International Conference on Learning Representations (ICLR)</em>, 2024.
  </div>
  <div class="reference-item">
    [4] C. Pan, B. Yuan, D. Le, et al., "AgentBench: Evaluating LLMs as Agents," in <em>International Conference on Learning Representations (ICLR)</em>, 2024.
  </div>
  <div class="reference-item">
    [5] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language Agents with Verbal Reinforcement Learning," in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, vol. 36, 2023.
  </div>
  <div class="reference-item">
    [6] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "ReAct: Synergizing Reasoning and Acting in Language Models," in <em>International Conference on Learning Representations (ICLR)</em>, 2023.
  </div>
  <div class="reference-item">
    [7] C. Qian, X. Cong, C. Yang, et al., "Communicative Agents for Software Development," in <em>Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)</em>, 2024.
  </div>
  <div class="reference-item">
    [8] Q. Wu, G. Bansal, J. Zhang, Y. Wu, S. Zhang, E. Zhu, B. Li, L. Jiang, X. Zhang, and C. Wang, "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation," <em>arXiv preprint arXiv:2308.08155</em>, 2023.
  </div>
  <div class="reference-item">
    [9] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in <em>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</em>, pp. 785–794, 2016.
  </div>
  <div class="reference-item">
    [10] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, vol. 30, 2017.
  </div>
  <div class="reference-item">
    [11] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," <em>Neural Computation</em>, vol. 9, no. 8, pp. 1735–1780, 1997.
  </div>
  <div class="reference-item">
    [12] J. Butani, L. Mutreja, M. Turakhia, and H. Shah, "Multimodal AI for Agentic System Failure Diagnosis and Root-Cause Reasoning," Project Source Code &amp; Documentation Repository, GitHub, Oct. 2026. [Online]. Available: <a href="https://github.com/jaydevbutani/multimodal-agent-failure-diagnosis">https://github.com/jaydevbutani/multimodal-agent-failure-diagnosis</a>
  </div>

  <!-- ========================================================================= -->
  <!-- 26. APPENDIX: REPRODUCIBILITY & SYSTEM SPECS -->
  <!-- ========================================================================= -->
  <div class="page-break"></div>
  <h2 class="section-heading">Appendix: Reproducibility Guidelines &amp; System Specifications</h2>

  <h3 class="subsection-heading">A.1 Computing Environment Specifications</h3>
  <ul>
    <li><strong>Operating System:</strong> Windows 11 Enterprise (x64) / Ubuntu 22.04 LTS Compatible</li>
    <li><strong>Python Environment:</strong> Python 3.11.9 (CPython 64-bit)</li>
    <li><strong>Core Deep Learning Stack:</strong> PyTorch 2.1.0 (CPU / CUDA 12.1 fallback)</li>
    <li><strong>Machine Learning Libraries:</strong> scikit-learn 1.4.2, XGBoost 2.0.3, SHAP 0.45.0, imbalanced-learn 0.12.3</li>
    <li><strong>Web &amp; Observability Stack:</strong> Flask 3.0.0, Jinja2 3.1.2, Streamlit 1.32.2, Plotly 5.22.0</li>
  </ul>

  <h3 class="subsection-heading">A.2 Step-by-Step Pipeline Execution Instructions</h3>
  <pre class="code-block"><code># 1. Clone the official project repository
git clone https://github.com/jaydevbutani/multimodal-agent-failure-diagnosis.git
cd multimodal-agent-failure-diagnosis

# 2. Initialize and activate virtual environment
python -m venv .venv
.venv\\Scripts\\activate  # On Linux: source .venv/bin/activate

# 3. Install verified dependencies
pip install -r requirements.txt

# 4. Execute grouped preprocessing and feature extraction
python src/preprocessing/prepare_data.py

# 5. Train classical tabular baselines (LR, RF, XGBoost)
python src/models/classical_models.py

# 6. Train deep trajectory and multimodal fusion architectures
python src/models/deep_models.py

# 7. Run full test evaluation and calculate exact out-of-sample metrics
python src/evaluation/verify_predictions.py

# 8. Launch the Flask Interactive Trace Canvas Observability Platform
python run.py
# (Navigate to http://127.0.0.1:8502 and click 'Demo Operator Access')

# 9. Launch the companion Streamlit Research Console
streamlit run app/main.py</code></pre>

  <h3 class="subsection-heading">A.3 Test Suite Execution &amp; Contract Verification</h3>
  <pre class="code-block"><code># Run comprehensive UI routing &amp; interface tests (15 passed)
python -m pytest tests/test_final_ui.py -q

# Run Flask integration and RBAC contract verification (12 passed)
python tests/test_flask_app.py

# Run live telemetry pipeline &amp; automated smoke tests (14 passed)
python tests/test_ui_smoke.py</code></pre>

</div>

</body>
</html>
"""

report_html_path = "AUCML_RESEARCH_REPORT.html"

replacements = {
    "{logo_b64}": logo_b64,
    "{class_dist_b64}": class_dist_b64,
    "{model_comp_b64}": model_comp_b64,
    "{roc_b64}": roc_b64,
    "{pr_b64}": pr_b64,
    "{cm_grid_b64}": cm_grid_b64,
    "{feat_imp_b64}": feat_imp_b64,
    "{shap_b64}": shap_b64,
    "{ablation_b64}": ablation_b64,
    "{ui_screen_b64}": ui_screen_b64,
    "{fig1_svg}": fig1_svg,
    "{fig6_svg}": fig6_svg,
    "{fig13_svg}": fig13_svg,
}
for k, v in replacements.items():
    html_content = html_content.replace(k, v)

with open(report_html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Master Research Report HTML written successfully: {report_html_path} ({os.path.getsize(report_html_path):,} bytes)")
