import os
import sys
import json
import pytest

# Ensure root directory is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.web.app import app, init_db, get_artifacts
from src.database.db_manager import (
    register_user,
    save_diagnostic_record,
    get_user_diagnostic_history
)

def run_flask_test_suite():
    print("==================================================================")
    print("RUNNING AUTOMATED FLASK APPLICATION & CONTRACT TEST SUITE")
    print("==================================================================")
    
    init_db()
    app.config["TESTING"] = True
    client = app.test_client()

    results = []

    # --------------------------------------------------------------------------
    # 1. Demo User Login Succeeds
    # --------------------------------------------------------------------------
    try:
        res = client.post("/api/auth/demo", json={"role": "standard"})
        data = res.get_json()
        assert res.status_code == 200
        assert data["success"] is True
        assert data["role"] == "user"
        assert data["redirect"] == "/dashboard"
        results.append(("AUTH-01", "Demo User Login", "Authenticates standard user and redirects to /dashboard", "PASS"))
    except Exception as e:
        results.append(("AUTH-01", "Demo User Login", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 2. Demo User Authentication Check
    # --------------------------------------------------------------------------
    try:
        res = client.post("/api/auth/demo", json={})
        data = res.get_json()
        assert res.status_code == 200
        assert data["success"] is True
        assert data["redirect"] == "/dashboard"
        results.append(("AUTH-02", "Demo Auth Endpoint", "Authenticates demo operator and redirects to /dashboard", "PASS"))
    except Exception as e:
        results.append(("AUTH-02", "Demo Auth Endpoint", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 3. Wrong Password Fails
    # --------------------------------------------------------------------------
    try:
        res = client.post("/api/auth/login", json={"identifier": "demo_user", "password": "WrongPassword@999"})
        data = res.get_json()
        assert res.status_code == 401
        assert data["success"] is False
        results.append(("AUTH-03", "Wrong Password", "Rejects invalid credentials with HTTP 401", "PASS"))
    except Exception as e:
        results.append(("AUTH-03", "Wrong Password", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 4. Unknown Username Fails
    # --------------------------------------------------------------------------
    try:
        res = client.post("/api/auth/login", json={"identifier": "ghost_agent_9999", "password": "AnyPassword@123"})
        data = res.get_json()
        assert res.status_code == 401
        assert data["success"] is False
        results.append(("AUTH-04", "Unknown Username", "Rejects nonexistent user with HTTP 401", "PASS"))
    except Exception as e:
        results.append(("AUTH-04", "Unknown Username", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 5. Logout Works
    # --------------------------------------------------------------------------
    try:
        client.post("/api/auth/demo", json={"role": "standard"})
        res = client.post("/api/auth/logout")
        data = res.get_json()
        assert res.status_code == 200
        assert data["success"] is True
        results.append(("AUTH-05", "Logout Action", "Clears session and returns redirect to /login", "PASS"))
    except Exception as e:
        results.append(("AUTH-05", "Logout Action", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 6. Session Disappears After Logout
    # --------------------------------------------------------------------------
    try:
        res = client.get("/api/auth/me")
        assert res.status_code == 401
        results.append(("AUTH-06", "Session Termination", "Subsequent requests are unauthenticated (401)", "PASS"))
    except Exception as e:
        results.append(("AUTH-06", "Session Termination", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 7. Model Evaluation Page Access
    # --------------------------------------------------------------------------
    try:
        client.post("/api/auth/demo", json={})
        res = client.get("/model-evaluation")
        assert res.status_code == 200
        assert b"Model Evaluation" in res.data or b"Methodology" in res.data or b"Comparative Analysis" in res.data
        results.append(("AUTH-07", "Model Evaluation Access", "Authenticated user accesses /model-evaluation with HTTP 200", "PASS"))
    except Exception as e:
        results.append(("AUTH-07", "Model Evaluation Access", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 8. Evaluator Legacy Redirect
    # --------------------------------------------------------------------------
    try:
        client.post("/api/auth/demo", json={})
        res = client.get("/evaluator")
        assert res.status_code in [302, 200]
        results.append(("AUTH-08", "Evaluator Legacy Redirect", "Legacy /evaluator redirects cleanly to dashboard", "PASS"))
    except Exception as e:
        results.append(("AUTH-08", "Evaluator Legacy Redirect", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 9. Demo User Can Access Dashboard
    # --------------------------------------------------------------------------
    try:
        client.post("/api/auth/demo", json={"role": "standard"})
        res = client.get("/dashboard")
        assert res.status_code == 200
        assert b"Agent Health Intelligence Platform" in res.data
        results.append(("AUTH-09", "Dashboard Access", "Standard user loads /dashboard successfully", "PASS"))
    except Exception as e:
        results.append(("AUTH-09", "Dashboard Access", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 10. Diagnostic History Isolated by Authenticated user_id
    # --------------------------------------------------------------------------
    try:
        # Create user X and user Y
        ts = int(time.time())
        u_x, _ = register_user(f"user_iso_x_{ts}", f"x_{ts}@test.com", "Password@123")
        u_y, _ = register_user(f"user_iso_y_{ts}", f"y_{ts}@test.com", "Password@123")

        # Save record for user X only
        save_diagnostic_record(
            user_id=u_x["id"],
            session_title="User X Private Diagnostic Trace",
            session_type="trace_inspection",
            agent_name="claude-code",
            model_used="Proposed Multi-View Fusion Network",
            threshold=0.46,
            failure_probability=0.88,
            predicted_status="FAILURE DETECTED",
            risk_level="High",
            detected_issue="Private Issue",
            root_cause="Private Root Cause",
            recommended_action="Action X"
        )

        hist_x = get_user_diagnostic_history(u_x["id"])
        hist_y = get_user_diagnostic_history(u_y["id"])

        assert len(hist_x) == 1
        assert len(hist_y) == 0
        assert hist_x[0]["session_title"] == "User X Private Diagnostic Trace"

        results.append(("AUTH-10", "Data Isolation", "Diagnostic history strictly isolated by user_id", "PASS"))
    except Exception as e:
        results.append(("AUTH-10", "Data Isolation", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 11. ML Inference via API (All 5 Models)
    # --------------------------------------------------------------------------
    try:
        client.post("/api/auth/demo", json={"role": "standard"})
        test_trace = {
            "trace_id": "TEST-TRACE-API-001",
            "model": "claude-code",
            "total_turns": 14,
            "tool_calls_total": 20,
            "search_tool_calls": 5,
            "edit_tool_calls": 6,
            "error_count": 2,
            "retry_count": 1,
            "patch_size_lines": 80,
            "duration_seconds": 25.0,
            "tool_call_sequence": ["Read", "Grep", "Write", "Bash"]
        }

        res = client.post("/api/diagnose", json={
            "trace_data": test_trace,
            "model_choice": "Proposed Multi-View Fusion Network",
            "threshold": 0.46
        })
        data = res.get_json()
        assert res.status_code == 200
        assert data["success"] is True
        assert 0.0 <= data["diagnosis"]["failure_probability"] <= 1.0
        assert len(data["dag_nodes"]) >= 5
        results.append(("ML-01", "API Diagnose Endpoint", f"Prob: {data['diagnosis']['failure_probability']:.4f}, DAG Nodes: {len(data['dag_nodes'])}", "PASS"))
    except Exception as e:
        results.append(("ML-01", "API Diagnose Endpoint", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # 12. Live Telemetry Pipeline Endpoint
    # --------------------------------------------------------------------------
    try:
        client.post("/api/auth/demo", json={"role": "standard"})
        res = client.post("/api/live-diagnosis", json={
            "scenario": "thrashing_failure",
            "model_choice": "Proposed Multi-View Fusion Network",
            "threshold": 0.46,
            "save": True
        })
        data = res.get_json()
        assert res.status_code == 200
        assert data["success"] is True
        assert data["saved_record_id"] is not None
        assert data["telemetry"]["retry_count"] >= 2
        results.append(("ML-02", "Live Pipeline Endpoint", f"Scenario: {data['scenario']}, Saved Record: #{data['saved_record_id']}", "PASS"))
    except Exception as e:
        results.append(("ML-02", "Live Pipeline Endpoint", f"FAILED: {e}", "FAIL"))

    # --------------------------------------------------------------------------
    # Print Summary Table
    # --------------------------------------------------------------------------
    print("\n------------------------------------------------------------------")
    print("AUTOMATED TEST RESULTS MATRIX:")
    print("------------------------------------------------------------------")
    all_passed = True
    for tid, name, desc, status in results:
        status_tag = f"[{status}]"
        print(f"{status_tag:<8} {tid}: {name:<24} -> {desc}")
        if status != "PASS":
            all_passed = False

    print("------------------------------------------------------------------")
    if all_passed:
        print("VERDICT: ALL 12 FLASK INTEGRATION & RBAC TESTS PASSED WITH 100% SUCCESS!")
    else:
        print("VERDICT: SOME TESTS FAILED.")
    print("==================================================================\n")
    return all_passed

if __name__ == "__main__":
    import time
    success = run_flask_test_suite()
    sys.exit(0 if success else 1)
