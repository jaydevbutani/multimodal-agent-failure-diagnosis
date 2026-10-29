"""
Trace Prediction Module
=======================
Agent Health Intelligence (AUCML — TE7483)

Single Responsibility:
- Centralized single-trace inference pipeline:
    Raw Trace -> Feature Extraction -> Preprocessing -> Model Inference -> Calibrated Attribution
- Consistent interface used by Trace Diagnosis, Live Simulation, and CLI
"""

import os
import sys
import json
from typing import Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.web.app import get_diagnosis_service

def predict_trace(
    trace_data: Dict[str, Any],
    model_name: str = "Proposed Multi-View Fusion Network",
    threshold: float = 0.46
) -> Dict[str, Any]:
    """
    Executes end-to-end inference on a single agent execution trace.
    Returns structured diagnosis report:
    - Status (HEALTHY / FAILURE DETECTED)
    - Failure probability
    - Root cause pattern
    - Responsible component
    - Suspicious step
    - Contributing signals
    - Recommended action
    """
    service = get_diagnosis_service()
    diagnosis = service.diagnose(trace_data, model_choice=model_name, threshold=threshold)
    return diagnosis

if __name__ == "__main__":
    demo_file = os.path.join(BASE_DIR, "data", "demo_traces", "02_retry_thrashing.json")
    if os.path.exists(demo_file):
        with open(demo_file, "r", encoding="utf-8") as f:
            sample_trace = json.load(f)
        result = predict_trace(sample_trace)
        print("==================================================================")
        print("TRACE PREDICTION AUDIT")
        print("==================================================================")
        print(f"Status:               {result['status']}")
        print(f"Probability:          {result['failure_probability']:.4f}")
        print(f"Root Cause:           {result['root_cause']}")
        print(f"Responsible Component:{result.get('responsible_component')}")
        print(f"Suspicious Step:      {result.get('suspicious_step')}")
        print(f"Action:               {result['recommended_action']}")
        print("==================================================================")
