"""
Feature Engineering Module
==========================
Agent Health Intelligence (AUCML — TE7483)

Single Responsibility:
- Extract domain-specific derived behavioral features from agent execution telemetry
- Compute tool diversity (Shannon entropy) from tool-call sequences
- Return feature dictionary with both raw telemetry and derived behavioral signals
"""

import math
from typing import Dict, Any, List

def compute_tool_entropy(sequence: List[str]) -> float:
    """
    Computes Shannon entropy of tool call sequences:
        H(tools) = - sum(p_i * log2(p_i))
    Low entropy (< 0.5) indicates cyclic retry loops or thrashing.
    High entropy (> 1.2) indicates diverse, exploratory tool utilization.
    """
    if not sequence:
        return 1.0
    counts: Dict[str, int] = {}
    for tool in sequence:
        t_clean = str(tool).strip().lower()
        counts[t_clean] = counts.get(t_clean, 0) + 1
        
    total = len(sequence)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(float(entropy), 4)

def extract_behavioral_features(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts 10 domain-specific behavioral features from agent telemetry.
    Used before model inference across all models.
    """
    d = dict(trace_dict)
    
    turns = max(1, int(d.get("total_turns", 1)))
    tool_calls = max(0, int(d.get("tool_calls_total", 0)))
    errors = max(0, int(d.get("error_count", 0)))
    retries = max(0, int(d.get("retry_count", 0)))
    search_calls = max(0, int(d.get("search_tool_calls", 0)))
    edit_calls = max(0, int(d.get("edit_tool_calls", 0)))
    files_read = max(0, int(d.get("unique_files_read", 0)))
    files_edited = max(0, int(d.get("unique_files_edited", 0)))
    patch_size = max(0, int(d.get("patch_size_lines", 0)))
    duration = max(0.1, float(d.get("duration_seconds", 1.0)))
    seq = d.get("tool_call_sequence", [])
    
    # 1. Error rate: proportion of tool invocations yielding exceptions
    d["DERIVED_error_rate"] = float(errors / max(1, tool_calls))
    
    # 2. Retry ratio: proportion of errors immediately followed by re-execution
    d["DERIVED_retry_ratio"] = float(retries / max(1, errors))
    
    # 3. Search-to-edit ratio: exploratory reading relative to code modification
    d["DERIVED_search_to_edit_ratio"] = float(search_calls / max(1, edit_calls))
    
    # 4. Edit churn: lines of code modified per touched file
    d["DERIVED_edit_churn"] = float(patch_size / max(1, files_edited))
    
    # 5. Tool intensity: tool actions executed per conversation turn
    d["DERIVED_tool_intensity"] = float(tool_calls / max(1, turns))
    
    # 6. Read-to-edit ratio: file foraging breadth vs modification focus
    d["DERIVED_read_to_edit_ratio"] = float(files_read / max(1, files_edited))
    
    # 7. Time per turn: latency per agent-environment step
    d["DERIVED_time_per_turn"] = float(duration / max(1, turns))
    
    # 8. Tool diversity entropy: Shannon entropy of tool distribution
    d["DERIVED_tool_entropy"] = compute_tool_entropy(seq)
    
    # 9. Log-duration: dampens extreme long-tail runtimes
    d["DERIVED_log_duration"] = float(math.log1p(max(0.0, duration)))
    
    # 10. Log-patch size: dampens large refactoring diffs
    d["DERIVED_log_patch_size"] = float(math.log1p(max(0.0, patch_size)))
    
    return d
