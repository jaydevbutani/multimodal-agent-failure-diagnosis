"""
Standalone CLI & Programmatic Inference Engine
Agent Health Intelligence (AUCML — TE7483)

Usage:
    python ml/inference/predict.py --trace data/demo_traces/02_retry_thrashing.json
"""

import os
import sys
import json
import argparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.web.app import get_diagnosis_service

def run_trace_prediction(trace_path, model_name="Proposed Multi-View Fusion Network"):
    """
    Loads trace JSON, executes diagnosis through the validated pipeline,
    and returns failure probability, pattern, and failure signature.
    """
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
        
    with open(trace_path, "r", encoding="utf-8") as fp:
        raw_trace = json.load(fp)
        
    service = get_diagnosis_service()
    result = service.diagnose(raw_trace, model_choice=model_name)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run failure diagnosis inference on an agent trace.")
    parser.add_argument("--trace", type=str, default="data/demo_traces/02_retry_thrashing.json",
                        help="Path to JSON trace file")
    parser.add_argument("--model", type=str, default="Proposed Multi-View Fusion Network",
                        help="Model to use for inference")
    args = parser.parse_args()
    
    print(f"\nEvaluating: {args.trace} using '{args.model}'...")
    res = run_trace_prediction(args.trace, args.model)
    
    print("\n==================================================================")
    print("AGENT HEALTH INTELLIGENCE — INFERENCE REPORT")
    print("==================================================================")
    print(f"Status:              {res['status']} ({res['risk_level']} Risk)")
    print(f"Failure Probability: {res['failure_probability']*100:.1f}%")
    print(f"Likely Pattern:      {res['root_cause']}")
    print(f"Suspicious Step:     {res['suspicious_step']}")
    print(f"Why:                 {res['why']}")
    print(f"Recommended Action:  {res['recommended_action']}")
    print("==================================================================")
    sig = res.get("failure_signature", {})
    if sig:
        print("\nFAILURE SIGNATURE:")
        print(f"Pattern:           {sig.get('pattern')}")
        print("Observed Signals:")
        for s in sig.get("observed_signals", []):
            print(f"  - {s}")
        chain = str(sig.get('execution_evidence', '')).replace('\u2192', '->')
        print(f"Evidence Chain:    {chain}")
    print("==================================================================\n")
