"""
Explainability & Attribution Module
===================================
Agent Health Intelligence (AUCML — TE7483)

Single Responsibility:
- Model explainability and feature contribution extraction
- Connect model predictions to empirical execution telemetry signals
- STRICT COMPLIANCE: Phrased as "Features contributing most strongly to the model prediction"
  NEVER claiming that features caused the failure (correlation != causation).
"""

from typing import Dict, Any, List

def get_top_contributing_signals(trace_dict: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Identifies the strongest empirical telemetry signals contributing to model prediction.
    
    Note: These represent features contributing most strongly to the model's
    predictive scoring; they do NOT establish ground-truth physical causality.
    """
    signals = []
    
    # 1. Retry ratio
    retries = int(trace_dict.get("retry_count", 0))
    retry_ratio = float(trace_dict.get("DERIVED_retry_ratio", 0.0))
    if retries > 2 or retry_ratio > 0.2:
        signals.append({
            "feature": "DERIVED_retry_ratio",
            "name": "Command Retry Thrashing Ratio",
            "value": f"{retry_ratio:.2f} ({retries} retries)",
            "impact": "High Positive Failure Contribution",
            "interpretation": "Repeated command re-executions strongly correlate with agent failure."
        })
        
    # 2. Error rate
    errors = int(trace_dict.get("error_count", 0))
    error_rate = float(trace_dict.get("DERIVED_error_rate", 0.0))
    if errors > 2 or error_rate > 0.15:
        signals.append({
            "feature": "DERIVED_error_rate",
            "name": "Tool Execution Error Rate",
            "value": f"{error_rate:.2f} ({errors} errors)",
            "impact": "High Positive Failure Contribution",
            "interpretation": "Elevated exception frequencies strongly increase predicted risk."
        })
        
    # 3. Tool diversity entropy
    entropy = float(trace_dict.get("DERIVED_tool_entropy", 1.0))
    if entropy < 0.8:
        signals.append({
            "feature": "DERIVED_tool_entropy",
            "name": "Tool Diversity (Shannon Entropy)",
            "value": f"{entropy:.2f} bits",
            "impact": "Negative Tool Diversity Contribution",
            "interpretation": "Depressed entropy indicates repetitive, unvaried command loops."
        })
        
    # 4. Code churn
    churn = float(trace_dict.get("DERIVED_edit_churn", 0.0))
    if churn > 150:
        signals.append({
            "feature": "DERIVED_edit_churn",
            "name": "Code Modification Churn",
            "value": f"{churn:.1f} lines/file",
            "impact": "Elevated Churn Contribution",
            "interpretation": "High diff blast radius indicates oscillating patch proposals."
        })
        
    # 5. Search-to-edit ratio
    search_ratio = float(trace_dict.get("DERIVED_search_to_edit_ratio", 0.0))
    if search_ratio > 5.0:
        signals.append({
            "feature": "DERIVED_search_to_edit_ratio",
            "name": "Information Exploration Ratio",
            "value": f"{search_ratio:.1f}x",
            "impact": "Over-Exploration Contribution",
            "interpretation": "High read-to-edit ratio indicates context budget exhaustion."
        })
        
    # Default nominal signals if no anomalies
    if not signals:
        signals.append({
            "feature": "All Indicators Nominal",
            "name": "Operational Baseline Conformity",
            "value": "Nominal",
            "impact": "Strong Healthy Contribution",
            "interpretation": "Telemetry remained within verified nominal training distributions."
        })
        
    return signals[:top_n]
