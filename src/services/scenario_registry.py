"""
Centralized Scenario Registry & In-Memory Trace Store
=====================================================
Single source of truth for demonstration scenarios, canonical scenario mappings,
metadata, operational recommendations, and runtime trace caching.
"""

import os
import json
import threading
from typing import Dict, Any, Optional, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ------------------------------------------------------------------------------
# 1. CANONICAL SCENARIO REGISTRY
# ------------------------------------------------------------------------------
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "healthy": {
        "id": "healthy",
        "demo_id": "01_healthy_nominal",
        "filename": "01_healthy_nominal.json",
        "title": "Nominal SWE Run",
        "full_name": "Nominal Execution (Clean Task Completion)",
        "expected_status": "HEALTHY",
        "failure_type": "Nominal Execution",
        "responsible_component": "Execution Orchestrator",
        "suspicious_step": "None (Nominal Execution)",
        "what_happened": "The agent completed task ingestion, file inspection, targeted diff editing, and verified tests cleanly.",
        "description": "Standard autonomous software engineering task. Reads files, drafts bug patch, verifies green test suite with zero errors.",
        "turns": 32,
        "tools": 18,
        "errors": 0,
        "retries": 0,
        "aliases": [
            "healthy", "nominal", "01_healthy_nominal", "01_healthy_nominal.json",
            "DEMO-01", "clean_run"
        ],
        "contributing_signals": [
            "Tool call diversity within nominal entropy bounds (> 1.0)",
            "Zero command retries or syntax execution loops",
            "Targeted file edits matching task scope",
            "All environment checks verified nominal"
        ],
        "recommended_resolutions": [
            "No intervention required.",
            "Execution completed within expected operational parameters.",
            "Deploy agent artifacts to staging / pull request pipeline."
        ]
    },
    "thrashing_failure": {
        "id": "thrashing_failure",
        "demo_id": "02_retry_thrashing",
        "filename": "02_retry_thrashing.json",
        "title": "Retry Thrashing",
        "full_name": "Cyclic Tool Retry Thrashing",
        "expected_status": "FAILURE DETECTED",
        "failure_type": "Retry Thrashing",
        "responsible_component": "Bash Command Shell",
        "suspicious_step": "Step 04 — execute_bash",
        "what_happened": "The agent entered a repeated tool-execution loop without making meaningful progress.",
        "description": "Agent repeatedly re-executes failing bash test commands in a tight cyclic loop without editing code or adapting parameters.",
        "turns": 45,
        "tools": 38,
        "errors": 12,
        "retries": 11,
        "aliases": [
            "thrashing_failure", "retry_thrashing", "cyclic_retry_thrashing",
            "02_retry_thrashing", "02_retry_thrashing.json", "DEMO-02", "thrashing"
        ],
        "contributing_signals": [
            "High retry ratio (repeated identical tool invocations)",
            "Elevated tool intensity per conversation turn",
            "Consecutive nonzero exit codes from bash execution",
            "Depressed tool diversity (Shannon entropy < 0.5)"
        ],
        "recommended_resolutions": [
            "Stop the repeated tool-call loop immediately.",
            "Re-evaluate the failed bash command syntax and environment state.",
            "Inject corrective prompt instructing agent to alter strategy.",
            "Reset execution context before resuming tool dispatch."
        ]
    },
    "environment_failure": {
        "id": "environment_failure",
        "demo_id": "03_environment_failure",
        "filename": "03_environment_failure.json",
        "title": "Environment Crash",
        "full_name": "Environment / Tool Execution Failure",
        "expected_status": "FAILURE DETECTED",
        "failure_type": "Environment Failure",
        "responsible_component": "Tool Sandbox Runtime",
        "suspicious_step": "Step 03 — execute_bash",
        "what_happened": "The agent failed to dispatch tools due to missing container dependencies or permission denial.",
        "description": "Agent encounters container sandbox failure, broken paths, missing library dependencies, or permission exceptions.",
        "turns": 55,
        "tools": 20,
        "errors": 15,
        "retries": 2,
        "aliases": [
            "environment_failure", "env_failure", "03_environment_failure",
            "03_environment_failure.json", "DEMO-03", "sandbox_crash"
        ],
        "contributing_signals": [
            "Critical unhandled environment exception / stderr exit codes",
            "Missing system libraries or container filesystem permissions",
            "Tool execution aborting before test dispatch",
            "Elevated error-to-turn ratio (> 0.35)"
        ],
        "recommended_resolutions": [
            "Verify sandbox environment configuration and docker socket mount.",
            "Install missing system dependencies before re-executing agent.",
            "Check filesystem read/write permissions inside the container.",
            "Validate tool schemas and external connection endpoints."
        ]
    },
    "high_code_churn": {
        "id": "high_code_churn",
        "demo_id": "04_high_code_churn",
        "filename": "04_high_code_churn.json",
        "title": "High Code Churn",
        "full_name": "High Modification Churn",
        "expected_status": "FAILURE DETECTED",
        "failure_type": "High Code Churn",
        "responsible_component": "Code Editor Engine",
        "suspicious_step": "Step 03 — str_replace_editor",
        "what_happened": "The agent generated continuous large file rewrites across multiple modules without test convergence.",
        "description": "Agent generates massive code diff churn across many files, oscillating between contradictory edits without passing tests.",
        "turns": 55,
        "tools": 42,
        "errors": 5,
        "retries": 4,
        "aliases": [
            "high_code_churn", "code_churn", "04_high_code_churn",
            "04_high_code_churn.json", "DEMO-04", "diff_churn"
        ],
        "contributing_signals": [
            "Very high edit churn (> 250 modified lines per file)",
            "Repeated modification of identical source files across turns",
            "High diff blast radius without incremental verification",
            "Patch size entropy exceeding operational thresholds"
        ],
        "recommended_resolutions": [
            "Reduce repeated modifications and validate changes incrementally.",
            "Roll back unbounded diff churn to last known compiling state.",
            "Require unit test execution between atomic code edits.",
            "Constrain agent modification scope to targeted source files."
        ]
    },
    "over_exploration": {
        "id": "over_exploration",
        "demo_id": "05_over_exploration",
        "filename": "05_over_exploration.json",
        "title": "Over-Exploration",
        "full_name": "Information Over-Exploration",
        "expected_status": "FAILURE DETECTED",
        "failure_type": "Over-Exploration",
        "responsible_component": "File System Navigator",
        "suspicious_step": "Step 05 — read_file",
        "what_happened": "The agent navigated through unrelated files searching endlessly without drafting any code changes.",
        "description": "Agent wanders through directory tree reading 18+ unrelated files, blowing context budget without writing any code.",
        "turns": 40,
        "tools": 25,
        "errors": 0,
        "retries": 1,
        "aliases": [
            "over_exploration", "information_over_exploration", "05_over_exploration",
            "05_over_exploration.json", "DEMO-05", "search_loop"
        ],
        "contributing_signals": [
            "Extremely high search-to-edit ratio (> 10x read vs write)",
            "15+ consecutive read/grep operations without code generation",
            "Context window saturation risk from sprawling file ingestion",
            "Zero progress toward task verification criteria"
        ],
        "recommended_resolutions": [
            "Reduce unnecessary information retrieval and return to task objective.",
            "Provide focused file path guidance to terminate search loops.",
            "Prune retrieved context buffer to prevent attention degradation.",
            "Enforce a maximum search turn limit before requiring a patch proposal."
        ]
    },
    "tool_execution_failure": {
        "id": "tool_execution_failure",
        "demo_id": "06_tool_execution_failure",
        "filename": "06_tool_execution_failure.json",
        "title": "Tool Instability",
        "full_name": "Tool Execution Instability",
        "expected_status": "FAILURE DETECTED",
        "failure_type": "Tool Instability",
        "responsible_component": "External Tool Connector",
        "suspicious_step": "Step 04 — execute_tool",
        "what_happened": "External tools and language servers timed out, returning malformed output and parse exceptions.",
        "description": "External language server or test runner times out, emitting truncated stdout and unparseable JSON responses.",
        "turns": 35,
        "tools": 18,
        "errors": 8,
        "retries": 3,
        "aliases": [
            "tool_execution_failure", "tool_instability", "06_tool_execution_failure",
            "06_tool_execution_failure.json", "DEMO-06", "lsp_timeout"
        ],
        "contributing_signals": [
            "Tool response latency spikes exceeding timeout thresholds (> 500ms)",
            "Intermittent tool execution errors and invalid return payloads",
            "Tool parse exceptions logged by telemetry supervisor",
            "Sub-threshold execution instability escalating to fault state"
        ],
        "recommended_resolutions": [
            "Check tool availability, timeout behavior, and error responses.",
            "Verify tool JSON schemas and parameter contracts.",
            "Increase timeout margins for external language server calls.",
            "Implement automatic tool failover / fallback strategies."
        ]
    },
    "compound_failure": {
        "id": "compound_failure",
        "demo_id": "07_compound_failure",
        "filename": "07_compound_failure.json",
        "title": "Compound Cascade Failure",
        "full_name": "Compound Execution Failure",
        "expected_status": "FAILURE DETECTED",
        "failure_type": "Compound Failure",
        "responsible_component": "Execution Orchestrator",
        "suspicious_step": "Step 06 — execute_bash",
        "what_happened": "An initial environment failure triggered repeated retry attempts, cascading across multiple tool subsystems.",
        "description": "Initial environment failure triggers frantic retry loops and broken edits, cascading across multiple tool subsystems.",
        "turns": 60,
        "tools": 45,
        "errors": 18,
        "retries": 14,
        "aliases": [
            "compound_failure", "compound_cascade_failure", "07_compound_failure",
            "07_compound_failure.json", "DEMO-07", "cascade_failure"
        ],
        "contributing_signals": [
            "Simultaneous occurrence of high retries, high churn, and errors",
            "Cascading fault propagation from environment to command shell",
            "Severe telemetry degradation across multiple behavioral axes",
            "Failure probability exceeding 80% across all evaluated views"
        ],
        "recommended_resolutions": [
            "Halt agent session immediately to prevent cascading state corruption.",
            "Isolate the root environment failure before attempting patch repairs.",
            "Revert all unverified code edits and restore clean git working tree.",
            "Execute full container sanity check before re-launching agent."
        ]
    }
}

# ------------------------------------------------------------------------------
# 2. RESOLVER & NORMALIZATION UTILITIES
# ------------------------------------------------------------------------------
def normalize_scenario_id(identifier: str) -> Optional[str]:
    """
    Normalizes any incoming scenario identifier (key, demo ID, filename, alias)
    to its canonical scenario key in SCENARIOS.
    """
    if not identifier:
        return None
    raw = str(identifier).strip().lower()
    
    # Direct match on key
    if raw in SCENARIOS:
        return raw
        
    # Match on demo_id or filename
    for key, sc in SCENARIOS.items():
        if raw == sc["id"].lower() or raw == sc["demo_id"].lower() or raw == sc["filename"].lower():
            return key
        if raw in [a.lower() for a in sc.get("aliases", [])]:
            return key
            
    # Fuzzy match by stripping extension or numeric prefix
    clean = raw.replace(".json", "")
    for key, sc in SCENARIOS.items():
        if clean == sc["id"].lower() or clean == sc["demo_id"].lower():
            return key
        if clean in sc["demo_id"].lower() or key in clean:
            return key
            
    return None

def get_scenario(identifier: str) -> Optional[Dict[str, Any]]:
    """Returns scenario metadata dict for any alias or identifier."""
    canonical_key = normalize_scenario_id(identifier)
    if canonical_key:
        return SCENARIOS.get(canonical_key)
    return None

def get_demo_trace_path(identifier: str) -> Optional[str]:
    """
    Resolves the exact file path to the demo JSON trace on disk.
    Guarantees no 'demo trace not found' errors.
    """
    sc = get_scenario(identifier)
    if not sc:
        return None
    filename = sc["filename"]
    path = os.path.join(BASE_DIR, "data", "demo_traces", filename)
    if os.path.exists(path):
        return path
    return None

def get_all_demo_cases() -> List[Dict[str, Any]]:
    """Returns list of all demo cases for the UI."""
    return [
        {
            "id": sc["demo_id"],
            "canonical_id": sc["id"],
            "filename": sc["filename"],
            "title": sc["title"],
            "full_name": sc["full_name"],
            "expected_status": sc["expected_status"],
            "description": sc["description"],
            "turns": sc["turns"],
            "tools": sc["tools"],
            "errors": sc["errors"],
            "retries": sc["retries"],
            "responsible_component": sc["responsible_component"],
            "suspicious_step": sc["suspicious_step"]
        }
        for sc in SCENARIOS.values()
    ]

# ------------------------------------------------------------------------------
# 3. THREAD-SAFE IN-MEMORY TRACE STORE
# ------------------------------------------------------------------------------
class TraceStore:
    """
    Centralized in-memory trace store for live agent simulations,
    demo executions, and uploaded traces.
    """
    _lock = threading.Lock()
    _traces: Dict[str, Dict[str, Any]] = {}
    _recent_order: List[str] = []
    _MAX_CAPACITY = 50

    @classmethod
    def save(cls, trace_id: str, trace_data: Dict[str, Any], diagnosis: Optional[Dict[str, Any]] = None, dag: Optional[Dict[str, Any]] = None) -> str:
        with cls._lock:
            cls._traces[trace_id] = {
                "trace_id": trace_id,
                "trace_data": trace_data,
                "diagnosis": diagnosis,
                "dag": dag
            }
            if trace_id in cls._recent_order:
                cls._recent_order.remove(trace_id)
            cls._recent_order.append(trace_id)
            
            # Prune capacity
            while len(cls._recent_order) > cls._MAX_CAPACITY:
                oldest = cls._recent_order.pop(0)
                cls._traces.pop(oldest, None)
                
            return trace_id

    @classmethod
    def get(cls, trace_id: str) -> Optional[Dict[str, Any]]:
        with cls._lock:
            # Check direct match
            if trace_id in cls._traces:
                return cls._traces[trace_id]
            # Check prefix match (e.g. LIVE-THRASHING)
            for tid, entry in cls._traces.items():
                if tid.startswith(trace_id) or trace_id in tid:
                    return entry
            return None

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._traces.clear()
            cls._recent_order.clear()
