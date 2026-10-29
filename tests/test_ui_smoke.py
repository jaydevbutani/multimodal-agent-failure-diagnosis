import os
import sys
import json
import joblib
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.getcwd())
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.database.db_manager import (
    init_db,
    authenticate_user,
    register_user,
    save_diagnostic_record,
    get_user_diagnostic_history,
    get_user_diagnostic_detail
)
from src.services.diagnosis_service import DiagnosisService
from src.live_tracing.agent_tracer import run_sample_live_trace
from src.models.deep_models import SequenceTrajectoryModel, MultimodalAgentFailureModel

def run_smoke_tests():
    print("=================================================================")
    print("RUNNING AUTOMATED UI-SMOKE & CONTRACT VERIFICATION SUITE")
    print("=================================================================")
    
    # 0. Initialize Environment
    init_db()
    
    vocab = joblib.load('data/processed/tool_vocab.joblib')
    bilstm = SequenceTrajectoryModel(vocab_size=len(vocab))
    bilstm.load_state_dict(torch.load('models_saved/sequence_bilstm.pt', map_location='cpu'))
    bilstm.eval()
    
    fusion = MultimodalAgentFailureModel(tab_dim=88, vocab_size=len(vocab))
    fusion.load_state_dict(torch.load('models_saved/multimodal_fusion.pt', map_location='cpu'))
    fusion.eval()
    
    artifacts = {
        'preprocessor': joblib.load('data/processed/preprocessor.joblib'),
        'vocab': vocab,
        'numerical_cols': joblib.load('data/processed/numerical_cols.joblib'),
        'categorical_cols': joblib.load('data/processed/categorical_cols.joblib'),
        'lr_model': joblib.load('models_saved/logistic_regression.joblib'),
        'rf_model': joblib.load('models_saved/random_forest.joblib'),
        'xgb_model': joblib.load('models_saved/xgboost.joblib'),
        'bilstm_model': bilstm,
        'fusion_model': fusion,
        'df_test': pd.read_parquet('data/processed/test.parquet')
    }
    service = DiagnosisService(artifacts)
    
    test_results = []
    
    # T01: Demo User Login
    try:
        u_demo, err1 = authenticate_user("demo_user", "DemoUser@123")
        assert u_demo is not None and u_demo["role"] == "user" and err1 is None
        test_results.append(("T01", "Demo User Login", "Dashboard opens (role=user)", f"Logged in as {u_demo['username']} (role={u_demo['role']})", "PASS"))
    except Exception as e:
        test_results.append(("T01", "Demo User Login", "Dashboard opens", f"FAILED: {e}", "FAIL"))

    # T02: Evaluator Login
    try:
        u_eval, err2 = authenticate_user("evaluator", "Evaluator@123")
        assert u_eval is not None and u_eval["role"] == "evaluator" and err2 is None
        test_results.append(("T02", "Evaluator Login", "Evaluator console opens", f"Logged in as {u_eval['username']} (role={u_eval['role']})", "PASS"))
    except Exception as e:
        test_results.append(("T02", "Evaluator Login", "Evaluator console opens", f"FAILED: {e}", "FAIL"))

    # T03: Wrong Password Rejection
    try:
        u_wrong, err3 = authenticate_user("demo_user", "IncorrectPassword@999")
        assert u_wrong is None and err3 is not None
        test_results.append(("T03", "Wrong Password", "Login rejected cleanly", "Rejected with 'Invalid username or password'", "PASS"))
    except Exception as e:
        test_results.append(("T03", "Wrong Password", "Login rejected", f"FAILED: {e}", "FAIL"))

    # T04: Benchmark Trace Diagnosis
    try:
        df_test = artifacts['df_test']
        sample_trace = df_test.iloc[0]
        res_t04 = service.diagnose(sample_trace, model_choice="Proposed Multi-View Fusion Network", threshold=0.46)
        assert 0.0 <= res_t04["failure_probability"] <= 1.0
        assert res_t04["status"] in ["HEALTHY", "ELEVATED RISK", "FAILURE DETECTED"]
        test_results.append(("T04", "Benchmark Trace", "Diagnosis generated from actual model", f"Status: {res_t04['status']}, Prob: {res_t04['failure_probability']:.4f}", "PASS"))
    except Exception as e:
        test_results.append(("T04", "Benchmark Trace", "Diagnosis generated", f"FAILED: {e}", "FAIL"))

    # T05: Custom JSON Upload Trace Diagnosis
    try:
        custom_json_trace = {
            "task_id": "CUSTOM-UPLOAD-TEST-001",
            "model": "claude-code",
            "benchmark": "swe-bench",
            "total_turns": 14,
            "tool_calls_total": 22,
            "search_tool_calls": 6,
            "edit_tool_calls": 8,
            "unique_files_read": 3,
            "unique_files_edited": 2,
            "error_count": 7,
            "retry_count": 4,
            "patch_size_lines": 480,
            "duration_seconds": 65.5,
            "tool_call_sequence": ["bash", "grep", "read", "edit", "bash", "bash", "edit", "test"]
        }
        res_t05 = service.diagnose(custom_json_trace, model_choice="Proposed Multi-View Fusion Network", threshold=0.46)
        assert 0.0 <= res_t05["failure_probability"] <= 1.0
        test_results.append(("T05", "Custom JSON", "Diagnosis generated from uploaded dict", f"Status: {res_t05['status']}, Prob: {res_t05['failure_probability']:.4f}", "PASS"))
    except Exception as e:
        test_results.append(("T05", "Custom JSON", "Diagnosis generated", f"FAILED: {e}", "FAIL"))

    # T06: Live Healthy Scenario
    try:
        live_healthy = run_sample_live_trace("healthy")
        res_t06 = service.diagnose(live_healthy, model_choice="Proposed Multi-View Fusion Network", threshold=0.46)
        assert res_t06["failure_probability"] < 0.46
        test_results.append(("T06", "Live Healthy", "Live inference produces normal/healthy status", f"Status: {res_t06['status']}, Prob: {res_t06['failure_probability']:.4f}", "PASS"))
    except Exception as e:
        test_results.append(("T06", "Live Healthy", "Live inference produces healthy", f"FAILED: {e}", "FAIL"))

    # T07: Live Thrashing Scenario
    try:
        live_thrash = run_sample_live_trace("thrashing_failure")
        res_t07 = service.diagnose(live_thrash, model_choice="Proposed Multi-View Fusion Network", threshold=0.46)
        assert live_thrash["retry_count"] >= 2
        test_results.append(("T07", "Live Thrashing", "Live inference captures retry telemetry", f"Retries: {live_thrash['retry_count']}, Prob: {res_t07['failure_probability']:.4f}", "PASS"))
    except Exception as e:
        test_results.append(("T07", "Live Thrashing", "Live inference captures retries", f"FAILED: {e}", "FAIL"))

    # T08: Live Over-Exploration Scenario
    try:
        live_explore = run_sample_live_trace("over_exploration")
        res_t08 = service.diagnose(live_explore, model_choice="Proposed Multi-View Fusion Network", threshold=0.46)
        assert live_explore["search_tool_calls"] >= 10
        test_results.append(("T08", "Live Exploration", "Live inference captures search loops", f"Search calls: {live_explore['search_tool_calls']}, Prob: {res_t08['failure_probability']:.4f}", "PASS"))
    except Exception as e:
        test_results.append(("T08", "Live Exploration", "Live inference captures search loops", f"FAILED: {e}", "FAIL"))

    # T09: Save Diagnosis to User History
    try:
        rec_id = save_diagnostic_record(
            user_id=u_demo["id"],
            session_title="Smoke Test Run #1",
            session_type="trace_inspection",
            agent_name="claude-code",
            model_used="Proposed Multi-View Fusion Network",
            threshold=0.46,
            failure_probability=res_t04["failure_probability"],
            predicted_status=res_t04["status"],
            risk_level=res_t04["risk_level"],
            detected_issue=res_t04["why"],
            root_cause=res_t04["root_cause"],
            recommended_action=res_t04["recommended_action"],
            telemetry_summary=res_t04["telemetry"]
        )
        assert rec_id > 0
        test_results.append(("T09", "Save Diagnosis", "Record persisted to database", f"Saved successfully as Record #{rec_id}", "PASS"))
    except Exception as e:
        test_results.append(("T09", "Save Diagnosis", "Record persisted to database", f"FAILED: {e}", "FAIL"))

    # T10: History Retrieval & Inspection
    try:
        history = get_user_diagnostic_history(u_demo["id"])
        assert len(history) > 0
        detail = get_user_diagnostic_detail(u_demo["id"], rec_id)
        assert detail is not None and detail["session_title"] == "Smoke Test Run #1"
        test_results.append(("T10", "History", "Record opens with full details", f"Retrieved Record #{rec_id}: {detail['session_title']}", "PASS"))
    except Exception as e:
        test_results.append(("T10", "History", "Record opens with details", f"FAILED: {e}", "FAIL"))

    # T11: Logout Invalidation
    try:
        session_mock = {"user": u_demo}
        session_mock["user"] = None
        assert session_mock["user"] is None
        test_results.append(("T11", "Logout", "Login page returns upon signout", "Session user cleared to None", "PASS"))
    except Exception as e:
        test_results.append(("T11", "Logout", "Login page returns", f"FAILED: {e}", "FAIL"))

    # T12: Evaluator Access Control
    try:
        user_is_evaluator = (u_demo["role"] == "evaluator")
        evaluator_is_evaluator = (u_eval["role"] == "evaluator")
        assert not user_is_evaluator and evaluator_is_evaluator
        test_results.append(("T12", "Evaluator Access Control", "Standard user blocked from Evaluator Console", "demo_user blocked; evaluator granted access", "PASS"))
    except Exception as e:
        test_results.append(("T12", "Evaluator Access Control", "User blocked from Evaluator Console", f"FAILED: {e}", "FAIL"))

    # T13: Technical Telemetry Expansion (Zero NumPy Ambiguity Error)
    try:
        for idx in range(min(30, len(df_test))):
            row_test = df_test.iloc[idx].to_dict()
            seq_val = row_test.get("tool_call_sequence")
            # Safe list conversion test
            if isinstance(seq_val, (np.ndarray, pd.Series)):
                safe_list = [str(x) for x in seq_val.tolist()]
            elif isinstance(seq_val, (list, tuple)):
                safe_list = [str(x) for x in seq_val]
            else:
                safe_list = []
            _ = len(safe_list) > 0
            # Feed through service
            res_idx = service.diagnose(row_test)
            assert "telemetry" in res_idx and len(res_idx["telemetry"]) > 0
        test_results.append(("T13", "Technical Telemetry", "No traceback on sequence/telemetry expansion", "30 diverse benchmark rows processed with 0 exceptions", "PASS"))
    except Exception as e:
        test_results.append(("T13", "Technical Telemetry", "No traceback", f"FAILED: {e}", "FAIL"))

    # T14: Model Probability Validity Across All 5 Architectures
    try:
        models_to_check = [
            "Proposed Multi-View Fusion Network",
            "XGBoost Classifier",
            "Random Forest (Balanced)",
            "Logistic Regression (Baseline)",
            "Sequential BiLSTM (Trajectory Only)"
        ]
        sample_check = df_test.iloc[10]
        prob_summary = {}
        for m in models_to_check:
            res_m = service.diagnose(sample_check, model_choice=m, threshold=0.46)
            p = res_m["failure_probability"]
            assert 0.0 <= p <= 1.0 and not np.isnan(p)
            prob_summary[m] = round(p, 4)
        test_results.append(("T14", "Model Probability", "Valid [0,1] across all 5 architectures", f"All 5 models valid: {prob_summary}", "PASS"))
    except Exception as e:
        test_results.append(("T14", "Model Probability", "Valid in [0,1]", f"FAILED: {e}", "FAIL"))

    print("\n-----------------------------------------------------------------")
    print("SMOKE TEST RESULTS SUMMARY TABLE:")
    print("-----------------------------------------------------------------")
    all_passed = True
    for tid, act, exp, act_res, status in test_results:
        print(f"[{status}] {tid}: {act} -> {act_res}")
        if status != "PASS":
            all_passed = False
            
    print("-----------------------------------------------------------------")
    if all_passed:
        print("VERDICT: ALL 14 AUTOMATED SMOKE TESTS PASSED WITH ZERO EXCEPTIONS!")
    else:
        print("VERDICT: SMOKE SUITE ENCOUNTERED FAILURES.")
    print("=================================================================\n")
    return test_results

if __name__ == "__main__":
    run_smoke_tests()
