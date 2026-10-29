"""
Domain-Specific Derived Feature Engineering for Agent Failure Diagnosis
Agent Health Intelligence (AUCML — TE7483)
"""

import math
import numpy as np
from collections import Counter

def compute_tool_entropy(seq):
    """
    Computes Shannon entropy over the discrete sequence of tool calls:
    H(T) = - sum(p_i * log(p_i))
    Low entropy (< 0.5) indicates cyclic retry loops or repetitive tool execution.
    """
    if seq is None or len(seq) == 0:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p)
    return float(entropy)

def extract_derived_features(trace_dict):
    """
    Extracts the 10 audited domain-specific features:
    1. DERIVED_error_rate = error_count / tool_calls_total
    2. DERIVED_retry_ratio = retry_count / error_count
    3. DERIVED_search_to_edit_ratio = search_tool_calls / edit_tool_calls
    4. DERIVED_edit_churn = patch_size_lines / unique_files_edited
    5. DERIVED_tool_intensity = tool_calls_total / total_turns
    6. DERIVED_read_to_edit_ratio = unique_files_read / unique_files_edited
    7. DERIVED_time_per_turn = duration_seconds / total_turns
    8. DERIVED_tool_entropy = Shannon entropy of tool sequence
    9. DERIVED_log_duration = log1p(duration_seconds)
    10. DERIVED_log_patch_size = log1p(patch_size_lines)
    """
    d = trace_dict.copy()
    turns = max(1, int(d.get('total_turns', 1)))
    tool_calls = max(0, int(d.get('tool_calls_total', 0)))
    err_count = max(0, int(d.get('error_count', 0)))
    retries = max(0, int(d.get('retry_count', 0)))
    search_calls = max(0, int(d.get('search_tool_calls', 0)))
    edit_calls = max(0, int(d.get('edit_tool_calls', 0)))
    files_read = max(0, int(d.get('unique_files_read', 0)))
    files_edited = max(0, int(d.get('unique_files_edited', 0)))
    patch_size = max(0, int(d.get('patch_size_lines', 0)))
    duration = max(0.1, float(d.get('duration_seconds', 1.0)))
    seq = d.get('tool_call_sequence', [])
    
    d['DERIVED_error_rate'] = float(err_count / max(1, tool_calls))
    d['DERIVED_retry_ratio'] = float(retries / max(1, err_count))
    d['DERIVED_search_to_edit_ratio'] = float(search_calls / max(1, edit_calls))
    d['DERIVED_edit_churn'] = float(patch_size / max(1, files_edited))
    d['DERIVED_tool_intensity'] = float(tool_calls / max(1, turns))
    d['DERIVED_read_to_edit_ratio'] = float(files_read / max(1, files_edited))
    d['DERIVED_time_per_turn'] = float(duration / max(1, turns))
    d['DERIVED_tool_entropy'] = compute_tool_entropy(seq)
    d['DERIVED_log_duration'] = float(np.log1p(max(0.0, duration)))
    d['DERIVED_log_patch_size'] = float(np.log1p(max(0.0, patch_size)))
    
    return d
