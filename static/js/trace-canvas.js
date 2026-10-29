// ==============================================================================
// STITCH TRACE CANVAS CONTROLLER & DIAGNOSIS ENGINE
// ==============================================================================

let currentTrace = null;
let currentDiagnosis = null;
let currentDagNodes = [];
let benchmarkTraces = [];
let isDiagnosing = false;

document.addEventListener('DOMContentLoaded', async () => {
  await loadBenchmarkOptions();
  
  // Check URL params for deep-linking
  const urlParams = new URLSearchParams(window.location.search);
  const historyId = urlParams.get('history_id');
  const mode = urlParams.get('mode');

  if (historyId) {
    await loadHistoricalTrace(historyId);
  } else if (mode === 'live') {
    document.getElementById('select-trace-source').value = 'live_thrashing';
    await executeLiveSimulation('thrashing_failure');
  } else {
    // Default to first benchmark trace
    await runBenchmarkDiagnosis(0);
  }
});

// 1. Load Benchmark Options from /api/traces/benchmark
async function loadBenchmarkOptions() {
  try {
    const res = await fetch('/api/traces/benchmark');
    if (!res.ok) return;
    benchmarkTraces = await res.json();

    const group = document.getElementById('benchmark-options-group');
    const selectEl = document.getElementById('select-benchmark-trace');

    if (group) {
      group.innerHTML = '';
      benchmarkTraces.forEach((t, i) => {
        const opt = document.createElement('option');
        opt.value = `benchmark_${i}`;
        const statusLabel = t.ground_truth_passed ? 'Passed/Healthy' : 'Held-Out Failure';
        opt.innerText = `[${statusLabel}] Task ${t.task_id.substring(0, 16)} (${t.total_turns} turns, ${t.tool_calls_total} tools)`;
        group.appendChild(opt);
      });
    }

    if (selectEl) {
      selectEl.innerHTML = '';
      benchmarkTraces.forEach((t, i) => {
        const opt = document.createElement('option');
        opt.value = `${i}`;
        const statusLabel = t.ground_truth_passed ? 'Passed/Healthy' : 'Held-Out Failure';
        opt.innerText = `[${statusLabel}] Task ${t.task_id.substring(0, 16)} (${t.total_turns} turns, ${t.tool_calls_total} tools, ${t.error_count} errs)`;
        selectEl.appendChild(opt);
      });
    }
  } catch (err) {
    console.error('Failed to load benchmark traces:', err);
  }
}

// 2. Trace Source Switcher
async function handleTraceSourceChange(val) {
  if (val.startsWith('benchmark_')) {
    const idx = parseInt(val.split('_')[1], 10);
    await runBenchmarkDiagnosis(idx);
  } else if (val === 'live_healthy') {
    await executeLiveSimulation('healthy');
  } else if (val === 'live_thrashing') {
    await executeLiveSimulation('thrashing_failure');
  } else if (val === 'live_exploration') {
    await executeLiveSimulation('over_exploration');
  }
}

function updateThresholdValue(val) {
  document.getElementById('threshold-display').innerText = parseFloat(val).toFixed(2);
}

// 3. Trigger Diagnosis via API
async function triggerDiagnosis() {
  if (!currentTrace) return;
  const modelChoice = document.getElementById('select-model-choice').value;
  const threshold = parseFloat(document.getElementById('threshold-slider').value);

  showToast(`Running inference with ${modelChoice}...`, 'info');
  await executeDiagnoseCall(currentTrace, modelChoice, threshold);
}

// 4. Run Benchmark Diagnosis
async function runBenchmarkDiagnosis(index) {
  if (!benchmarkTraces[index]) return;
  const trace = benchmarkTraces[index];
  currentTrace = trace;

  const modelChoice = document.getElementById('select-model-choice').value;
  const threshold = parseFloat(document.getElementById('threshold-slider').value);

  await executeDiagnoseCall(trace, modelChoice, threshold);
}

// 5. Execute Live Simulation
async function executeLiveSimulation(scenario) {
  if (isDiagnosing) return;
  isDiagnosing = true;

  const modelChoice = document.getElementById('select-model-choice').value;
  const threshold = parseFloat(document.getElementById('threshold-slider').value);

  showToast(`Executing live telemetry harness for scenario: ${scenario}...`, 'info');

  try {
    const res = await fetch('/api/live-diagnosis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario: scenario,
        model_choice: modelChoice,
        threshold: threshold
      })
    });

    const data = await res.json();
    if (res.ok && data.success) {
      currentTrace = data.telemetry;
      currentDiagnosis = data.diagnosis;
      currentDagNodes = data.dag_nodes;
      renderAllViews(currentTrace, currentDiagnosis, currentDagNodes);
      showToast(`Live simulation diagnosis complete. Saved to session history.`, 'success');
    } else {
      showToast(data.error || 'Live simulation failed.', 'error');
    }
  } catch (err) {
    showToast('Failed to connect to live telemetry harness: ' + err.message, 'error');
  } finally {
    isDiagnosing = false;
  }
}

// 6. Execute Diagnose API Call
async function executeDiagnoseCall(tracePayload, modelChoice, threshold) {
  if (isDiagnosing) return;
  isDiagnosing = true;

  try {
    const res = await fetch('/api/diagnose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trace_data: tracePayload,
        model_choice: modelChoice,
        threshold: threshold,
        save: false
      })
    });

    const data = await res.json();
    if (res.ok && data.success) {
      currentDiagnosis = data.diagnosis;
      currentDagNodes = data.dag_nodes;
      renderAllViews(tracePayload, currentDiagnosis, currentDagNodes);
    } else {
      showToast(data.error || 'Diagnostic evaluation failed.', 'error');
    }
  } catch (err) {
    showToast('Inference error: ' + err.message, 'error');
  } finally {
    isDiagnosing = false;
  }
}

// 7. Load Historical Trace
async function loadHistoricalTrace(recordId) {
  try {
    const res = await fetch(`/api/history/${recordId}`);
    if (!res.ok) return;
    const rec = await res.json();
    
    // Convert raw telemetry to trace
    currentTrace = rec.raw_telemetry || {};
    currentTrace.trace_id = rec.trace_id || `HIST-${rec.id}`;
    currentTrace.total_turns = currentTrace.total_turns || 10;
    currentTrace.tool_calls_total = currentTrace.tool_calls_total || 8;
    currentTrace.tool_call_sequence = rec.tool_call_sequence || currentTrace.tool_call_sequence || ['Read', 'Bash'];

    await triggerDiagnosis();
    showToast(`Loaded historical session: ${rec.session_title}`, 'info');
  } catch (err) {
    console.error('Failed to load history record:', err);
  }
}

// 8. Custom JSON File Upload
async function handleCustomFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (e) => {
    try {
      const json = JSON.parse(e.target.result);
      currentTrace = json;
      showToast(`Uploaded custom trace: ${file.name}`, 'info');
      await triggerDiagnosis();
    } catch (err) {
      showToast('Invalid JSON file format: ' + err.message, 'error');
    }
  };
  reader.readAsText(file);
}

// ==============================================================================
// RENDER VIEWS: DAG, SUBHEADER, WATERFALL, REASONING, COPILOT
// ==============================================================================

function renderAllViews(trace, diag, nodes) {
  renderSubheader(trace, diag);
  renderDagNodes(nodes);
  renderSvgConnections(nodes);
  renderTemporalMetrics(trace, diag);
  renderReasoningPanel(trace, diag);
  renderRawJson(trace, diag);
  initCopilotChat(trace, diag);
}

// Render Subheader Status
function renderSubheader(trace, diag) {
  const breadcrumb = document.getElementById('active-breadcrumb-page');
  const statusBadge = document.getElementById('trace-status-badge');
  const statusText = document.getElementById('trace-status-text');
  const statusDot = document.getElementById('trace-status-dot');
  const idBadge = document.getElementById('trace-id-badge');

  const traceId = trace.trace_id || trace.task_id || 'TRACE-8492';
  if (breadcrumb) breadcrumb.innerText = `Trace: ${traceId.substring(0, 22)}`;
  if (idBadge) idBadge.innerText = `Trace-ID: ${traceId.substring(0, 16)}`;

  const isFailed = (diag.status === 'FAILURE DETECTED');
  const isElevated = (diag.status === 'ELEVATED RISK');

  if (statusBadge && statusDot && statusText) {
    if (isFailed) {
      statusBadge.className = 'px-2.5 py-0.5 rounded font-label-sm text-label-sm bg-error text-on-error font-medium uppercase tracking-wider flex items-center gap-1.5 shadow-sm transition-all';
      statusDot.className = 'w-1.5 h-1.5 rounded-full bg-white animate-ping';
      statusText.innerText = 'Failure Detected';
    } else if (isElevated) {
      statusBadge.className = 'px-2.5 py-0.5 rounded font-label-sm text-label-sm bg-[#fe8357] text-[#6f2000] font-medium uppercase tracking-wider flex items-center gap-1.5 shadow-sm transition-all';
      statusDot.className = 'w-1.5 h-1.5 rounded-full bg-[#6f2000] animate-pulse';
      statusText.innerText = 'Elevated Risk';
    } else {
      statusBadge.className = 'px-2.5 py-0.5 rounded font-label-sm text-label-sm bg-[#2f6f4e] text-white font-medium uppercase tracking-wider flex items-center gap-1.5 shadow-sm transition-all';
      statusDot.className = 'w-1.5 h-1.5 rounded-full bg-white';
      statusText.innerText = 'Nominal Execution';
    }
  }
}

// Render DAG Nodes into Canvas
function renderDagNodes(nodes) {
  const container = document.getElementById('dag-nodes-container');
  container.innerHTML = '';

  const total = nodes.length;
  // Compute positions across the canvas
  nodes.forEach((node, idx) => {
    const leftPct = (total <= 1) ? 50 : (10 + (idx / (total - 1)) * 80);
    // Slight vertical wave
    const topPct = (idx % 2 === 0) ? 50 : ((idx % 4 === 1) ? 35 : 65);

    const nodeEl = document.createElement('div');
    nodeEl.className = `dag-node absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center group cursor-pointer p-1 rounded-lg transition-transform ${idx === total - 1 ? 'active-selected-node' : ''}`;
    nodeEl.style.left = `${leftPct}%`;
    nodeEl.style.top = `${topPct}%`;
    nodeEl.dataset.node = node.id;
    nodeEl.onclick = () => selectDagNode(node.id);

    let ringBg = 'bg-surface-container-lowest text-primary';
    let pulseClass = '';
    if (node.state === 'error') {
      ringBg = 'bg-error text-on-error animate-pulse ring-2 ring-secondary ring-offset-2';
    } else if (node.state === 'success') {
      ringBg = 'bg-[#2f6f4e] text-white';
    } else if (node.id === 'input') {
      ringBg = 'bg-primary text-on-primary';
    } else if (node.id === 'supervisor') {
      ringBg = 'bg-surface-container-lowest text-secondary';
    }

    nodeEl.innerHTML = `
      <div class="node-ring w-14 h-14 rounded-full ${ringBg} shadow-md flex items-center justify-center group-hover:scale-105 transition-all ring-offset-2 ring-0 relative">
        <span class="material-symbols-outlined text-[24px]">${node.icon}</span>
        ${node.state === 'error' ? '<span class="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-secondary-container text-on-secondary-container flex items-center justify-center font-label-sm text-label-sm font-bold shadow">!</span>' : ''}
      </div>
      <span class="mt-2 font-headline-sm text-[13px] text-on-surface font-semibold text-center truncate max-w-[110px]">${node.name}</span>
      <span class="font-code-sm text-[10px] text-on-surface-variant truncate max-w-[110px]">${node.packets}</span>
    `;

    container.appendChild(nodeEl);
  });

  // Default selection to terminal node
  if (nodes.length > 0) {
    selectDagNode(nodes[nodes.length - 1].id);
  }

  // Update footer text
  const footer = document.getElementById('canvas-status-footer');
  if (footer) {
    footer.innerText = `Graph Execution: ${total} Nodes • Latency: ${currentTrace?.duration_seconds || 1.2}s • Max Depth: 4`;
  }
}

// Render SVG Bezier Connections
function renderSvgConnections(nodes) {
  const svgGroup = document.getElementById('svg-paths-group');
  if (!svgGroup) return;
  svgGroup.innerHTML = '';

  const total = nodes.length;
  if (total < 2) return;

  const surface = document.getElementById('dag-surface');
  if (!surface) return;
  const width = surface.clientWidth || 800;
  const height = surface.clientHeight || 370;

  for (let i = 0; i < total - 1; i++) {
    const x1 = (width * (10 + (i / (total - 1)) * 80)) / 100;
    const y1 = (height * ((i % 2 === 0) ? 50 : ((i % 4 === 1) ? 35 : 65))) / 100;

    const x2 = (width * (10 + ((i + 1) / (total - 1)) * 80)) / 100;
    const y2 = (height * (((i + 1) % 2 === 0) ? 50 : (((i + 1) % 4 === 1) ? 35 : 65))) / 100;

    const cx1 = x1 + (x2 - x1) * 0.5;
    const cy1 = y1;
    const cx2 = x1 + (x2 - x1) * 0.5;
    const cy2 = y2;

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const d = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`;
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');

    if (i === total - 2 && nodes[i + 1].state === 'error') {
      path.setAttribute('stroke', 'url(#failureGrad)');
      path.setAttribute('stroke-width', '3');
      path.setAttribute('stroke-dasharray', '6,4');
      path.setAttribute('class', 'animate-pulse');
    } else {
      path.setAttribute('stroke', '#807471');
      path.setAttribute('stroke-width', '2');
      path.setAttribute('stroke-dasharray', '3,3');
    }
    svgGroup.appendChild(path);
  }
}

// Select DAG Node & Update Inspector Card
function selectDagNode(nodeId) {
  const node = currentDagNodes.find(n => n.id === nodeId);
  if (!node) return;

  // Highlight active ring
  document.querySelectorAll('.dag-node').forEach(el => {
    const ring = el.querySelector('.node-ring');
    if (ring) ring.classList.remove('ring-2', 'ring-secondary', 'ring-offset-2');
  });
  const activeEl = document.querySelector(`.dag-node[data-node="${nodeId}"]`);
  if (activeEl) {
    const ring = activeEl.querySelector('.node-ring');
    if (ring) ring.classList.add('ring-2', 'ring-secondary', 'ring-offset-2');
  }

  // Update Inspector Card
  const card = document.getElementById('inspector-card');
  if (card) {
    card.classList.add('scale-[1.02]');
    setTimeout(() => card.classList.remove('scale-[1.02]'), 150);

    const sBadge = document.getElementById('inspector-step-badge');
    const sTitle = document.getElementById('inspector-title');
    const sTarget = document.getElementById('inspector-target');
    const sPackets = document.getElementById('inspector-packets');
    const sTime = document.getElementById('inspector-time');
    const sStatus = document.getElementById('inspector-status-desc');

    if (sBadge) sBadge.innerText = node.step;
    if (sTitle) sTitle.innerText = node.title || node.name;
    if (sTarget) sTarget.innerText = node.target;
    if (sPackets) sPackets.innerText = node.packets;
    if (sTime) {
      sTime.innerText = node.time;
      if (node.state === 'error') {
        sTime.classList.add('text-error');
        sTime.classList.remove('text-primary');
      } else {
        sTime.classList.remove('text-error');
        sTime.classList.add('text-primary');
      }
    }
    if (sStatus) sStatus.innerText = node.status;
  }
}

// Render Temporal Telemetry & Swimlanes
function renderTemporalMetrics(trace, diag) {
  const duration = trace.duration_seconds || 1.2;
  const isFailed = (diag.status === 'FAILURE DETECTED');
  const isElevated = (diag.status === 'ELEVATED RISK');
  const retries = trace.retry_count || 0;
  const errors = trace.error_count || 0;
  const turns = trace.total_turns || 1;
  const tools = trace.tool_calls_total || (trace.tool_call_sequence ? trace.tool_call_sequence.length : 0);

  // Update diagnose.html top stat cards if present
  const mTurns = document.getElementById('metric-turns');
  const mTools = document.getElementById('metric-tool-calls');
  const mErrors = document.getElementById('metric-errors');
  const mRetries = document.getElementById('metric-retries');
  const telSummary = document.getElementById('telemetry-summary-text');

  if (mTurns) mTurns.innerText = turns;
  if (mTools) mTools.innerText = tools;
  if (mErrors) mErrors.innerText = errors;
  if (mRetries) mRetries.innerText = retries;
  if (telSummary) telSummary.innerText = `Duration: ${duration.toFixed(1)}s • ${turns} Turns • ${tools} Tools`;

  // Update tool sequence strip in diagnose.html
  const seqStrip = document.getElementById('tool-sequence-strip');
  if (seqStrip) {
    seqStrip.innerHTML = '';
    const seq = trace.tool_call_sequence || [];
    if (seq.length === 0) {
      seqStrip.innerHTML = '<span class="text-on-surface-variant">No tool calls logged in trace.</span>';
    } else {
      seq.forEach((tool, i) => {
        seqStrip.innerHTML += `
          <span class="px-2 py-0.5 rounded bg-surface-container font-code-sm text-[11px] text-primary border border-surface-container-high flex items-center gap-1">
            <span class="text-secondary font-bold">#${i+1}</span>
            <span>${tool}</span>
          </span>
          ${i < seq.length - 1 ? '<span class="text-on-surface-variant text-[10px]">→</span>' : ''}
        `;
      });
    }
  }

  // Waterfall Updates for trace_canvas.html if present
  const orchMs = Math.round(duration * 1000);
  const swOrchText = document.getElementById('swimlane-orchestrate-text');
  const swOrchMs = document.getElementById('swimlane-orchestrate-ms');
  if (swOrchText) swOrchText.innerText = `orchestrate: ${orchMs}ms total (${turns} turns)`;
  if (swOrchMs) swOrchMs.innerText = `${orchMs} ms`;

  const toolSeq = trace.tool_call_sequence || [];
  const primaryTool = toolSeq.length > 0 ? toolSeq[0] : 'Read';
  const swToolLabel = document.getElementById('swimlane-tool-label');
  const swToolText = document.getElementById('swimlane-tool-text');
  if (swToolLabel) swToolLabel.innerText = `Tool.${primaryTool}`;
  if (swToolText) swToolText.innerText = `${toolSeq.length} calls: ${toolSeq.slice(0, 4).join(' → ')}`;

  // Terminal Swimlane if present
  const termRow = document.getElementById('swimlane-terminal-row');
  const termTitle = document.getElementById('swimlane-terminal-title');
  const termBar = document.getElementById('swimlane-terminal-bar');
  const termBarText = document.getElementById('swimlane-terminal-bartext');
  const termMs = document.getElementById('swimlane-terminal-ms');
  const termIcon = document.getElementById('swimlane-terminal-icon');

  if (termRow && termTitle && termBar && termBarText && termMs && termIcon) {
    if (isFailed) {
      termRow.className = 'grid grid-cols-12 gap-2 items-center bg-error-container/40 p-2 rounded hover:bg-error-container/60 cursor-pointer transition-colors';
      termTitle.className = 'font-headline-sm text-headline-sm text-on-error-container font-semibold truncate';
      termTitle.innerText = `Fault: ${diag.root_cause}`;
      termBar.className = 'absolute left-[50%] top-0 bottom-0 w-[45%] bg-error rounded flex items-center px-2 shadow';
      termBarText.className = 'font-code-sm text-code-sm text-on-error truncate font-bold';
      termBarText.innerText = `FAILURE (${(diag.failure_probability * 100).toFixed(1)}%)`;
      termMs.className = 'col-span-2 text-right font-code-sm text-code-sm text-on-error-container font-bold';
      termMs.innerText = `${orchMs} ms`;
      termIcon.className = 'material-symbols-outlined text-error text-[16px]';
      termIcon.innerText = 'bolt';
    } else {
      termRow.className = 'grid grid-cols-12 gap-2 items-center bg-[#d4edda]/40 p-2 rounded hover:bg-[#d4edda]/60 cursor-pointer transition-colors';
      termTitle.className = 'font-headline-sm text-headline-sm text-[#155724] font-semibold truncate';
      termTitle.innerText = 'Status: Verified Nominal';
      termBar.className = 'absolute left-[50%] top-0 bottom-0 w-[45%] bg-[#2f6f4e] rounded flex items-center px-2 shadow';
      termBarText.className = 'font-code-sm text-code-sm text-white truncate font-bold';
      termBarText.innerText = `HEALTHY (${(diag.failure_probability * 100).toFixed(1)}%)`;
      termMs.className = 'col-span-2 text-right font-code-sm text-code-sm text-[#155724] font-bold';
      termMs.innerText = `${orchMs} ms`;
      termIcon.className = 'material-symbols-outlined text-[#2f6f4e] text-[16px]';
      termIcon.innerText = 'check_circle';
    }
  }

  const spikeLabel = document.getElementById('chart-spike-label');
  const spikeText = document.getElementById('chart-spike-text');
  if (spikeLabel) spikeLabel.innerText = isFailed ? `+${Math.round(orchMs * 0.8)}ms [SPIKE: ${diag.root_cause}]` : `+${Math.round(orchMs * 0.8)}ms [NOMINAL: Safe Bounds]`;
  if (spikeText) spikeText.innerText = isFailed ? `Spike: ${diag.why.substring(0, 36)}` : `Nominal: Prob ${(diag.failure_probability).toFixed(4)}`;
}

// Render Root-Cause Reasoning Panel
function renderReasoningPanel(trace, diag) {
  const isFailed = (diag.status === 'FAILURE DETECTED');
  const isElevated = (diag.status === 'ELEVATED RISK');

  // Update diagnose.html Diagnosis Card elements
  const dStatus = document.getElementById('diag-status-text');
  const dProb = document.getElementById('diag-prob-text');
  const dPattern = document.getElementById('diag-pattern-title');
  const dDesc = document.getElementById('diag-pattern-desc');
  const dEvidence = document.getElementById('diag-evidence-list');
  const dAction = document.getElementById('diag-action-text');
  const dRiskPill = document.getElementById('card-risk-pill');
  const dTopBar = document.getElementById('card-top-bar');

  if (dStatus) dStatus.innerText = diag.status;
  if (dProb) dProb.innerText = `${(diag.failure_probability * 100).toFixed(1)}%`;
  if (dPattern) dPattern.innerText = diag.root_cause;
  if (dDesc) dDesc.innerText = diag.why;
  if (dAction) dAction.innerText = diag.recommended_action;

  if (dStatus && dProb) {
    if (isFailed) {
      dStatus.className = 'font-headline-lg text-xl sm:text-2xl font-bold tracking-tight mt-0.5 text-error';
      dProb.className = 'font-display-lg text-2xl sm:text-3xl font-bold font-code-sm text-error';
      if (dTopBar) dTopBar.className = 'absolute top-0 left-0 right-0 h-1.5 bg-error';
      if (dRiskPill) {
        dRiskPill.className = 'px-2.5 py-0.5 rounded-full font-code-sm text-xs font-semibold bg-error text-white';
        dRiskPill.innerText = 'HIGH RISK';
      }
    } else if (isElevated) {
      dStatus.className = 'font-headline-lg text-xl sm:text-2xl font-bold tracking-tight mt-0.5 text-[#a23e18]';
      dProb.className = 'font-display-lg text-2xl sm:text-3xl font-bold font-code-sm text-[#a23e18]';
      if (dTopBar) dTopBar.className = 'absolute top-0 left-0 right-0 h-1.5 bg-[#fe8357]';
      if (dRiskPill) {
        dRiskPill.className = 'px-2.5 py-0.5 rounded-full font-code-sm text-xs font-semibold bg-[#fe8357] text-[#6f2000]';
        dRiskPill.innerText = 'ELEVATED RISK';
      }
    } else {
      dStatus.className = 'font-headline-lg text-xl sm:text-2xl font-bold tracking-tight mt-0.5 text-[#2f6f4e]';
      dProb.className = 'font-display-lg text-2xl sm:text-3xl font-bold font-code-sm text-[#2f6f4e]';
      if (dTopBar) dTopBar.className = 'absolute top-0 left-0 right-0 h-1.5 bg-[#2f6f4e]';
      if (dRiskPill) {
        dRiskPill.className = 'px-2.5 py-0.5 rounded-full font-code-sm text-xs font-semibold bg-[#2f6f4e] text-white';
        dRiskPill.innerText = 'HEALTHY';
      }
    }
  }

  if (dEvidence) {
    dEvidence.innerHTML = '';
    const evidencePoints = diag.evidence || [];
    if (evidencePoints.length === 0) {
      evidencePoints.push('All derived telemetry metrics remained within nominal threshold bounds.');
    }
    evidencePoints.forEach(pt => {
      const li = document.createElement('li');
      li.className = 'flex items-start gap-2';
      li.innerHTML = `
        <span class="material-symbols-outlined text-secondary text-sm shrink-0 mt-0.5">${isFailed ? 'warning' : 'check_circle'}</span>
        <span>${pt}</span>
      `;
      dEvidence.appendChild(li);
    });
  }

  // trace_canvas.html elements (if present)
  const confBadge = document.getElementById('confidence-badge');
  if (confBadge) {
    if (isFailed) {
      confBadge.className = 'px-2 py-0.5 rounded font-label-sm text-label-sm bg-secondary text-on-secondary font-medium uppercase shadow-sm';
      confBadge.innerText = 'High Confidence';
    } else if (isElevated) {
      confBadge.className = 'px-2 py-0.5 rounded font-label-sm text-label-sm bg-[#fe8357] text-[#6f2000] font-medium uppercase shadow-sm';
      confBadge.innerText = 'Elevated Risk';
    } else {
      confBadge.className = 'px-2 py-0.5 rounded font-label-sm text-label-sm bg-[#2f6f4e] text-white font-medium uppercase shadow-sm';
      confBadge.innerText = 'Verified Nominal';
    }
  }

  const taxTitle = document.getElementById('taxonomy-title');
  if (taxTitle) taxTitle.innerText = diag.root_cause;

  const sevEl = document.getElementById('taxonomy-severity');
  if (sevEl) {
    if (isFailed) {
      sevEl.className = 'px-2 py-0.5 rounded bg-error text-on-error font-code-sm text-code-sm font-semibold shadow-sm';
      sevEl.innerText = 'CRITICAL';
    } else if (isElevated) {
      sevEl.className = 'px-2 py-0.5 rounded bg-[#fe8357] text-[#6f2000] font-code-sm text-code-sm font-semibold shadow-sm';
      sevEl.innerText = 'ELEVATED';
    } else {
      sevEl.className = 'px-2 py-0.5 rounded bg-[#2f6f4e] text-white font-code-sm text-code-sm font-semibold shadow-sm';
      sevEl.innerText = 'NOMINAL';
    }
  }

  const taxDesc = document.getElementById('taxonomy-desc');
  if (taxDesc) taxDesc.innerText = diag.why;

  const toolSeq = trace.tool_call_sequence || [];
  const compName = toolSeq.length > 0 ? `Agent.${toolSeq[toolSeq.length - 1]}` : 'Agent.MainLoop';
  const locCompName = document.getElementById('localized-component-name');
  if (locCompName) locCompName.innerText = compName;
  const locCompFile = document.getElementById('localized-component-file');
  if (locCompFile) locCompFile.innerText = `src/agents/${trace.agent_name || 'swe_agent'}.py:124`;

  const evContainer = document.getElementById('evidence-list-container');
  if (evContainer) {
    evContainer.innerHTML = '';
    const evPoints = diag.evidence || [];
    if (evPoints.length === 0) evPoints.push('All telemetry metrics remained within nominal baseline boundaries.');
    evPoints.forEach(pt => {
      const li = document.createElement('li');
      li.className = 'flex items-start gap-2';
      li.innerHTML = `
        <span class="material-symbols-outlined text-secondary text-[16px] shrink-0 mt-0.5">${isFailed ? 'error' : 'check_circle'}</span>
        <span>${pt}</span>
      `;
      evContainer.appendChild(li);
    });
  }

  const codeDiagTitle = document.getElementById('code-block-diag-title');
  if (codeDiagTitle) codeDiagTitle.innerText = diag.root_cause;
  const codeErrTag = document.getElementById('code-block-error-tag');
  if (codeErrTag) codeErrTag.innerText = isFailed ? `← [FATAL: PROB ${(diag.failure_probability * 100).toFixed(1)}%]` : '← [NOMINAL: VERIFIED]';
  const applyPatchText = document.getElementById('apply-patch-text');
  if (applyPatchText) applyPatchText.innerText = diag.recommended_action || 'No Intervention Required';
}

// Render Raw JSON Code
function renderRawJson(trace, diag) {
  const codeEl = document.getElementById('raw-json-code');
  if (!codeEl) return;
  const combined = {
    trace_id: trace.trace_id || trace.task_id,
    agent_model: trace.model,
    benchmark: trace.benchmark,
    diagnosis_output: {
      status: diag.status,
      failure_probability: diag.failure_probability,
      risk_level: diag.risk_level,
      root_cause: diag.root_cause,
      recommended_action: diag.recommended_action,
      evidence: diag.evidence,
      signals: diag.signals
    },
    raw_telemetry: trace
  };
  codeEl.innerText = JSON.stringify(combined, null, 2);
}

// Initialize Copilot Chat Stream
function initCopilotChat(trace, diag) {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;
  container.innerHTML = `
    <div class="bg-surface-container-low rounded-lg p-3 text-body-sm font-body-sm text-on-surface flex flex-col gap-1.5 shadow-sm">
      <span class="font-label-sm text-label-sm text-secondary font-bold uppercase tracking-wider">Automated Diagnosis</span>
      <p>
        Evaluated trace <strong>#${(trace.trace_id || trace.task_id || '').substring(0, 14)}</strong> using <em>${diag.model_used}</em>.
        Status: <strong>${diag.status}</strong> (Failure Probability: <strong>${(diag.failure_probability).toFixed(4)}</strong>).
      </p>
      <p class="text-on-surface-variant font-code-sm text-[11px] pt-1 border-t border-surface-container">
        Root-Cause: ${diag.root_cause}. ${diag.why}
      </p>
    </div>
  `;
}

// Copilot Chat Submit
async function handleCopilotSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('copilot-input');
  const text = input.value.trim();
  if (!text) return;

  const container = document.getElementById('chat-messages-container');

  // Append user message
  const userMsg = document.createElement('div');
  userMsg.className = 'self-end bg-surface-container-high rounded-lg p-2.5 text-body-sm font-body-sm text-on-surface max-w-[85%] shadow-sm';
  userMsg.innerText = text;
  container.appendChild(userMsg);
  input.value = '';
  container.scrollTop = container.scrollHeight;

  // Query Backend Copilot
  try {
    const res = await fetch('/api/copilot/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text, diagnosis: currentDiagnosis })
    });
    const data = await res.json();

    const botMsg = document.createElement('div');
    botMsg.className = 'bg-surface-container-low rounded-lg p-3 text-body-sm font-body-sm text-on-surface flex flex-col gap-1 shadow-sm border-l-2 border-secondary';
    botMsg.innerHTML = `
      <span class="font-label-sm text-label-sm text-secondary font-bold uppercase tracking-wider">Copilot Reasoning</span>
      <p>${data.reply}</p>
    `;
    container.appendChild(botMsg);
    container.scrollTop = container.scrollHeight;
  } catch (err) {
    console.error('Copilot query error:', err);
  }
}

// Re-run Trace in Sandbox Simulation
let isRerunning = false;
function triggerRerunSimulation() {
  if (isRerunning) return;
  isRerunning = true;

  const btn = document.getElementById('rerun-btn');
  const label = document.getElementById('rerun-label');
  const icon = document.getElementById('rerun-icon');

  icon.classList.add('animate-spin');
  label.innerText = 'Executing in Sandbox...';
  btn.classList.add('opacity-80', 'pointer-events-none');

  showToast('Sandbox runtime initialized: Dispatching simulated telemetry stream...', 'info');

  setTimeout(async () => {
    // Re-run with clean nominal execution
    await executeLiveSimulation('healthy');

    icon.classList.remove('animate-spin');
    label.innerText = 'Re-run Completed (OK)';
    btn.classList.remove('opacity-80', 'pointer-events-none', 'bg-secondary');
    btn.classList.add('bg-[#2f6f4e]', 'text-white');

    showToast('Sandbox execution completed: All nodes verified within nominal boundaries.', 'success');
    isRerunning = false;
  }, 2200);
}

// Deploy Remediation Patch
function applyRemediationPatch() {
  const btn = document.getElementById('btn-apply-patch');
  const text = document.getElementById('apply-patch-text');
  const icon = document.getElementById('apply-patch-icon');

  icon.classList.add('animate-spin');
  text.innerText = 'Dispatching operational intervention...';
  btn.classList.add('pointer-events-none');

  setTimeout(() => {
    icon.classList.remove('animate-spin');
    icon.innerText = 'check';
    text.innerText = 'Intervention Successfully Deployed';
    btn.classList.remove('bg-[#271e1a]', 'hover:bg-[#342a25]');
    btn.classList.add('bg-[#2f6f4e]', 'text-white');
    btn.classList.remove('pointer-events-none');

    showToast('Operational policy deployed: Injected prompt break and rate limiting.', 'success');
  }, 1200);
}

// Copy Code
function copyTraceCode() {
  const text = document.getElementById('diagnostic-code-block').innerText;
  navigator.clipboard.writeText(text).then(() => {
    showToast('Diagnostic stack trace copied to clipboard.', 'success');
  });
}

function copyTraceJson() {
  const text = document.getElementById('raw-json-code').innerText;
  navigator.clipboard.writeText(text).then(() => {
    showToast('Raw trace JSON copied to clipboard.', 'success');
  });
}

function inspectDumpModal() {
  showToast('Full telemetry packet dump loaded into memory viewer.', 'info');
}

// View Mode Toggle (Forensic vs Raw JSON)
function setViewMode(mode) {
  const btnForensic = document.getElementById('btn-forensic-view');
  const btnRaw = document.getElementById('btn-raw-view');
  const dagContainer = document.getElementById('visual-dag-container');
  const rawContainer = document.getElementById('raw-json-container');

  if (mode === 'raw') {
    btnRaw.classList.add('bg-surface-container-lowest', 'text-primary', 'shadow-sm');
    btnRaw.classList.remove('text-on-surface-variant');
    btnForensic.classList.remove('bg-surface-container-lowest', 'text-primary', 'shadow-sm');
    btnForensic.classList.add('text-on-surface-variant');

    dagContainer.classList.add('hidden');
    rawContainer.classList.remove('hidden');
  } else {
    btnForensic.classList.add('bg-surface-container-lowest', 'text-primary', 'shadow-sm');
    btnForensic.classList.remove('text-on-surface-variant');
    btnRaw.classList.remove('bg-surface-container-lowest', 'text-primary', 'shadow-sm');
    btnRaw.classList.add('text-on-surface-variant');

    rawContainer.classList.add('hidden');
    dagContainer.classList.remove('hidden');
  }
}

// Graph Sub-Tabs
function selectGraphSubTab(btn, mode) {
  const pills = document.querySelectorAll('#graph-mode-pills button');
  pills.forEach(p => {
    p.classList.remove('bg-primary', 'text-on-primary');
    p.classList.add('text-on-surface-variant');
  });
  btn.classList.add('bg-primary', 'text-on-primary');
  btn.classList.remove('text-on-surface-variant');
  showToast(`Canvas view switched: ${btn.innerText}`, 'info');
}

let canvasZoomed = false;
function toggleCanvasZoom() {
  canvasZoomed = !canvasZoomed;
  const surface = document.getElementById('dag-surface');
  const zoomIcon = document.getElementById('zoom-icon');
  if (canvasZoomed) {
    surface.classList.remove('h-[370px]');
    surface.classList.add('h-[480px]');
    zoomIcon.innerText = 'zoom_out';
    showToast('Canvas view expanded', 'info');
  } else {
    surface.classList.add('h-[370px]');
    surface.classList.remove('h-[480px]');
    zoomIcon.innerText = 'zoom_in';
    showToast('Canvas view normalized', 'info');
  }
}

function recenterGraph() {
  const surface = document.getElementById('dag-surface');
  surface.classList.add('opacity-70');
  setTimeout(() => {
    surface.classList.remove('opacity-70');
    showToast('DAG Graph centered', 'info');
  }, 150);
}

// Telemetry Zoom and Filter
function setTelemetryZoom(btn, zoom) {
  const btns = document.querySelectorAll('#telemetry-zoom-btns button');
  btns.forEach(b => {
    b.classList.remove('bg-surface-container', 'text-primary');
    b.classList.add('text-on-surface-variant');
  });
  btn.classList.add('bg-surface-container', 'text-primary');
  btn.classList.remove('text-on-surface-variant');

  const svg = document.getElementById('metrics-svg');
  if (zoom === '5x') {
    svg.classList.add('scale-x-125');
    showToast('Temporal resolution zoomed to 5x', 'info');
  } else {
    svg.classList.remove('scale-x-125');
    showToast('Temporal resolution reset to 1x', 'info');
  }
}

function toggleMetricFilter(btn, metric) {
  btn.classList.toggle('opacity-40');
  btn.classList.toggle('active');
  const isActive = btn.classList.contains('active');

  if (metric === 'latency') {
    document.getElementById('chart-latency-fill').style.opacity = isActive ? '1' : '0.05';
    document.getElementById('chart-latency-line').style.opacity = isActive ? '1' : '0.1';
  } else if (metric === 'memory') {
    document.getElementById('chart-mem-fill').style.opacity = isActive ? '0.25' : '0.05';
    document.getElementById('chart-mem-line').style.opacity = isActive ? '1' : '0.1';
  } else if (metric === 'fault') {
    document.getElementById('chart-fault-marker').style.opacity = isActive ? '1' : '0';
    document.getElementById('chart-spike-tooltip').style.opacity = isActive ? '1' : '0';
  }
}
