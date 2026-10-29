import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from collections import Counter

def compute_tool_entropy(seq):
    if seq is None or len(seq) == 0:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p)
    return float(entropy)

class AgentFailureDiagnoser:
    def __init__(self, models_dir="models_saved", data_dir="data/processed"):
        self.rf_model = joblib.load(os.path.join(models_dir, "random_forest.joblib"))
        self.xgb_model = joblib.load(os.path.join(models_dir, "xgboost.joblib"))
        self.preprocessor = joblib.load(os.path.join(data_dir, "preprocessor.joblib"))
        self.feature_names = joblib.load(os.path.join(data_dir, "feature_names.joblib"))
        self.numerical_cols = joblib.load(os.path.join(data_dir, "numerical_cols.joblib"))
        self.categorical_cols = joblib.load(os.path.join(data_dir, "categorical_cols.joblib"))
        
    def diagnose_trace(self, trace_dict):
        """
        Diagnoses an agent execution trace dictionary.
        Returns a structured dictionary with failure probability, status, evidence, and actions.
        """
        df = pd.DataFrame([trace_dict])
        
        # Ensure base columns exist
        defaults = {
            'total_turns': 1, 'tool_calls_total': 0, 'search_tool_calls': 0, 'edit_tool_calls': 0,
            'code_nav_tool_calls': 0, 'semantic_search_tool_calls': 0, 'unique_files_read': 0,
            'unique_files_edited': 0, 'error_count': 0, 'retry_count': 0, 'patch_size_lines': 0,
            'duration_seconds': 1.0, 'model': 'claude-sonnet-4-6', 'agent_name': 'claude-code',
            'benchmark': 'openhands', 'tool_call_sequence': []
        }
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v
                
        # Engineer Derived Features
        df['DERIVED_error_rate'] = df['error_count'] / np.maximum(1, df['tool_calls_total'])
        df['DERIVED_retry_ratio'] = df['retry_count'] / np.maximum(1, df['error_count'])
        df['DERIVED_search_to_edit_ratio'] = df['search_tool_calls'] / np.maximum(1, df['edit_tool_calls'])
        df['DERIVED_edit_churn'] = df['patch_size_lines'] / np.maximum(1, df['unique_files_edited'])
        df['DERIVED_tool_intensity'] = df['tool_calls_total'] / np.maximum(1, df['total_turns'])
        df['DERIVED_read_to_edit_ratio'] = df['unique_files_read'] / np.maximum(1, df['unique_files_edited'])
        df['DERIVED_time_per_turn'] = df['duration_seconds'] / np.maximum(1, df['total_turns'])
        df['DERIVED_tool_entropy'] = df['tool_call_sequence'].apply(compute_tool_entropy)
        df['DERIVED_log_duration'] = np.log1p(np.maximum(0.0, df['duration_seconds']))
        df['DERIVED_log_patch_size'] = np.log1p(np.maximum(0.0, df['patch_size_lines']))
        
        # Transform features
        X_trans = self.preprocessor.transform(df[self.numerical_cols + self.categorical_cols])
        
        # Predict with ensemble / models
        rf_prob = float(self.rf_model.predict_proba(X_trans)[0, 1])
        xgb_prob = float(self.xgb_model.predict_proba(X_trans)[0, 1])
        ensemble_prob = round((rf_prob + xgb_prob) / 2.0, 4)
        
        is_failure = ensemble_prob >= 0.5
        
        # Extract Evidence Signals
        evidence = []
        if df['error_count'].iloc[0] > 5:
            evidence.append(f"Elevated error count: {int(df['error_count'].iloc[0])} environment errors logged.")
        if df['DERIVED_retry_ratio'].iloc[0] > 0.5:
            evidence.append(f"High retry loop propensity: {df['DERIVED_retry_ratio'].iloc[0]:.2f} retries per error.")
        if df['DERIVED_edit_churn'].iloc[0] > 500:
            evidence.append(f"Severe code churn detected: {df['DERIVED_edit_churn'].iloc[0]:.1f} lines modified per file.")
        if df['DERIVED_tool_entropy'].iloc[0] < 0.5 and df['tool_calls_total'].iloc[0] > 4:
            evidence.append("Low tool diversity (entropy < 0.5): agent stuck repeatedly calling the same command.")
        if df['DERIVED_search_to_edit_ratio'].iloc[0] > 10:
            evidence.append("Information over-exploration: agent issued extensive searches without implementing edits.")
        if not evidence:
            evidence.append("Telemetry metrics remain within nominal operational boundaries.")
            
        # Recommendation
        if is_failure:
            if df['DERIVED_retry_ratio'].iloc[0] > 0.5:
                rec_action = "Interrupt execution; break cyclic retry loop by injecting clarification prompt."
            elif df['DERIVED_edit_churn'].iloc[0] > 500:
                rec_action = "Review diff blast radius; revert unbounded file rewrites."
            else:
                rec_action = "Flag session for human SRE inspection; verify tool permissions and environment dependencies."
        else:
            rec_action = "Nominal execution. Allow session to continue without intervention."
            
        return {
            'trace_id': trace_dict.get('trace_id', 'UNKNOWN'),
            'session_type': trace_dict.get('session_type', 'NATURAL_EXECUTION'),
            'status': "FAILURE DETECTED" if is_failure else "NORMAL EXECUTION",
            'failure_probability': ensemble_prob,
            'rf_probability': round(rf_prob, 4),
            'xgb_probability': round(xgb_prob, 4),
            'confidence': round(abs(ensemble_prob - 0.5) * 2.0, 3),
            'key_evidence': evidence,
            'recommended_action': rec_action
        }

if __name__ == '__main__':
    diagnoser = AgentFailureDiagnoser()
    
    # Test on Healthy live trace
    with open("data/raw/live_traces/LIVE-HEALTHY-1792782658.json") as f:
        t_healthy = json.load(f)
    print("--- Diagnostic Result (Healthy Trace) ---")
    print(json.dumps(diagnoser.diagnose_trace(t_healthy), indent=2))
    
    # Test on Thrashing Failure live trace
    with open("data/raw/live_traces/LIVE-THRASHING_FAILURE-1792782658.json") as f:
        t_thrash = json.load(f)
    print("\n--- Diagnostic Result (Thrashing Failure Trace) ---")
    print(json.dumps(diagnoser.diagnose_trace(t_thrash), indent=2))
