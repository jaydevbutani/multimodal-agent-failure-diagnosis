import os
import sys
import io
import json
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.web.app import app
from src.database.db_manager import init_db, get_connection

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client

def test_demo_login_operator(client):
    """Test that Try Demo as Operator reliably logs in demo@agenthealth.ai"""
    res = client.post("/api/auth/demo", json={})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["redirect"] == "/dashboard"
    assert "user" in data["role"] or data["role"] == "user"

def test_registration_and_login_flow(client):
    """Test registration with full name, email, and password confirmation, then login"""
    import uuid
    rand_email = f"operator_{uuid.uuid4().hex[:6]}@testcompany.org"
    
    # 1. Reject mismatched passwords
    res_bad = client.post("/api/auth/register", json={
        "full_name": "Test Operator",
        "email": rand_email,
        "password": "Password@123",
        "confirm_password": "WrongPassword@456"
    })
    assert res_bad.status_code == 400
    assert "match" in res_bad.get_json()["error"].lower()

    # 2. Register successfully
    res_reg = client.post("/api/auth/register", json={
        "full_name": "Test Operator",
        "email": rand_email,
        "password": "Password@123",
        "confirm_password": "Password@123"
    })
    assert res_reg.status_code == 200
    assert res_reg.get_json()["success"] is True

    # 3. Sign in with the registered email & password
    res_login = client.post("/api/auth/login", json={
        "identifier": rand_email,
        "password": "Password@123"
    })
    assert res_login.status_code == 200
    data = res_login.get_json()
    assert data["success"] is True
    assert data["redirect"] == "/dashboard"

def test_navigation_routes(client):
    """Ensure all platform navigation views respond with HTTP 200 for authenticated user"""
    # Login first
    client.post("/api/auth/demo", json={})

    # Test each primary page
    routes = [
        "/dashboard",
        "/diagnose",
        "/model-evaluation",
        "/history",
        "/settings",
        "/live"
    ]
    for r in routes:
        res = client.get(r)
        assert res.status_code == 200, f"Route {r} failed with status {res.status_code}"
        # Assert no faculty evaluator in title or sidebar
        assert b"Faculty Portal" not in res.data
        assert b"Evaluator Console" not in res.data

    # Test route aliases and legacy redirects to /model-evaluation
    assert client.get("/models").status_code == 200
    assert client.get("/dataset").status_code == 302
    assert client.get("/dataset-methodology").status_code == 302
    # Verify redirected destination resolves to 200
    assert client.get("/dataset-methodology", follow_redirects=True).status_code == 200

def test_evaluator_removal(client):
    """Ensure evaluator route redirects to dashboard and has no special access"""
    client.post("/api/auth/demo", json={})
    res = client.get("/evaluator", follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["Location"].endswith("/dashboard")

def test_diagnose_benchmark_and_save(client):
    """Test diagnosing a benchmark trace and saving to isolated history"""
    client.post("/api/auth/demo", json={})

    # 1. Get benchmark traces
    res_bench = client.get("/api/traces/benchmark")
    assert res_bench.status_code == 200
    benchmarks = res_bench.get_json()
    assert len(benchmarks) >= 6

    # 2. Run diagnosis on benchmark 0
    res_diag = client.post("/api/diagnose", json={
        "benchmark_index": 0,
        "model_choice": "Proposed Multi-View Fusion Network",
        "threshold": 0.46,
        "save": True
    })
    assert res_diag.status_code == 200
    data = res_diag.get_json()
    assert data["success"] is True
    assert "failure_probability" in data["diagnosis"]
    assert "status" in data["diagnosis"]
    assert "root_cause" in data["diagnosis"]
    assert "dag_nodes" in data
    assert len(data["dag_nodes"]) >= 4

    # 3. Check history
    res_hist = client.get("/api/history")
    assert res_hist.status_code == 200
    history = res_hist.get_json()
    assert len(history) >= 1

def test_live_simulation_harness(client):
    """Test live telemetry instrumentation and diagnosis for thrashing scenario with strict save behavior"""
    client.post("/api/auth/demo", json={})

    # 1. Run live simulation without saving -> must NOT pollute history
    hist_before = len(client.get("/api/history").get_json())
    res = client.post("/api/live-diagnosis", json={
        "scenario": "thrashing_failure",
        "model_choice": "Proposed Multi-View Fusion Network",
        "threshold": 0.46,
        "save": False
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["diagnosis"]["status"] == "FAILURE DETECTED"
    assert data["diagnosis"]["failure_probability"] >= 0.46
    assert data["saved_record_id"] is None
    hist_after = len(client.get("/api/history").get_json())
    assert hist_after == hist_before, "Running live simulation without explicit save must not pollute history"

    # 2. Run live simulation with explicit save -> must create history record
    res_save = client.post("/api/live-diagnosis", json={
        "scenario": "thrashing_failure",
        "model_choice": "Proposed Multi-View Fusion Network",
        "threshold": 0.46,
        "save": True
    })
    assert res_save.status_code == 200
    data_save = res_save.get_json()
    assert data_save["success"] is True
    assert data_save["saved_record_id"] is not None

def test_diagnostic_history_limit_10(client):
    """Verify that a user's diagnostic history is capped at exactly 10 records and prunes oldest"""
    import uuid
    cap_user_email = f"cap_test_{uuid.uuid4().hex[:6]}@limit.org"
    client.post("/api/auth/register", json={
        "full_name": "Cap Tester",
        "email": cap_user_email,
        "password": "Password@123",
        "confirm_password": "Password@123"
    })
    client.post("/api/auth/login", json={"identifier": cap_user_email, "password": "Password@123"})

    # Save 12 records consecutively
    for i in range(12):
        res = client.post("/api/diagnose", json={
            "demo_id": "01_healthy_nominal",
            "save": True
        })
        assert res.status_code == 200

    hist = client.get("/api/history").get_json()
    assert len(hist) == 10, f"Expected exactly 10 history records, got {len(hist)}"

def test_csv_upload_diagnose(client):
    """Test multipart CSV upload to /api/diagnose"""
    client.post("/api/auth/demo", json={})

    csv_content = """total_turns,tool_calls_total,error_count,retry_count,unique_files_read,unique_files_edited,patch_size_lines,duration_seconds,tool_call_sequence
12,10,4,3,2,1,10,3.2,"['bash', 'bash', 'read_file', 'bash']"
"""
    data = {
        'file': (io.BytesIO(csv_content.encode("utf-8")), 'trace_sample.csv'),
        'model_choice': 'Proposed Multi-View Fusion Network',
        'threshold': '0.46'
    }

    res = client.post("/api/diagnose", data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    diag_data = res.get_json()
    assert diag_data["success"] is True
    assert "failure_probability" in diag_data["diagnosis"]

def test_user_data_isolation(client):
    """Verify User A cannot see User B's diagnostic records"""
    import uuid
    user1_email = f"user1_{uuid.uuid4().hex[:6]}@iso.org"
    user2_email = f"user2_{uuid.uuid4().hex[:6]}@iso.org"

    # Register User 1
    client.post("/api/auth/register", json={
        "full_name": "User One",
        "email": user1_email,
        "password": "Password@123",
        "confirm_password": "Password@123"
    })
    # Login User 1 and save diagnosis
    client.post("/api/auth/login", json={"identifier": user1_email, "password": "Password@123"})
    res1 = client.post("/api/diagnose", json={"benchmark_index": 1, "save": True})
    rec1_id = res1.get_json()["saved_record_id"]

    # Logout User 1
    client.post("/api/auth/logout")

    # Register and Login User 2
    client.post("/api/auth/register", json={
        "full_name": "User Two",
        "email": user2_email,
        "password": "Password@123",
        "confirm_password": "Password@123"
    })
    client.post("/api/auth/login", json={"identifier": user2_email, "password": "Password@123"})

    # User 2 list history should NOT contain User 1's record
    hist2 = client.get("/api/history").get_json()
    assert all(h["result_id"] != rec1_id for h in hist2)

    # User 2 direct detail request on User 1's record should be 404
    res_direct = client.get(f"/api/history/{rec1_id}")
    assert res_direct.status_code == 404

def test_demo_cases_api(client):
    """Test GET /api/traces/demo returns the 7 curated synthetic demo cases"""
    client.post("/api/auth/demo", json={})
    res = client.get("/api/traces/demo")
    assert res.status_code == 200
    demos = res.get_json()
    assert len(demos) == 7
    ids = [d["id"] for d in demos]
    assert "01_healthy_nominal" in ids
    assert "02_retry_thrashing" in ids
    assert "03_environment_failure" in ids
    assert "04_high_code_churn" in ids
    assert "05_over_exploration" in ids
    assert "06_tool_execution_failure" in ids
    assert "07_compound_failure" in ids

def test_all_seven_demo_diagnoses_without_history_pollution(client):
    """Verify each of the 7 synthetic demo cases diagnoses accurately and does NOT pollute history"""
    client.post("/api/auth/demo", json={})
    
    # Check baseline history count
    hist_before = len(client.get("/api/history").get_json())
    
    expected_results = {
        "01_healthy_nominal": ("HEALTHY", "Nominal Execution"),
        "02_retry_thrashing": ("FAILURE DETECTED", "Cyclic Tool Retry Thrashing"),
        "03_environment_failure": ("FAILURE DETECTED", "Environment / Tool Execution Failure"),
        "04_high_code_churn": ("FAILURE DETECTED", "High Modification Churn"),
        "05_over_exploration": ("FAILURE DETECTED", "Information Over-Exploration"),
        "06_tool_execution_failure": ("FAILURE DETECTED", "Tool Execution Instability"),
        "07_compound_failure": ("FAILURE DETECTED", "Compound Execution Failure")
    }
    
    for demo_id, (expected_status, expected_pattern) in expected_results.items():
        res = client.post("/api/diagnose", json={"demo_id": demo_id, "save": False})
        assert res.status_code == 200, f"Demo {demo_id} failed: {res.data}"
        data = res.get_json()
        assert data["success"] is True
        assert data["diagnosis"]["status"] == expected_status, f"Mismatch for {demo_id}: {data['diagnosis']['status']}"
        assert data["diagnosis"]["root_cause"] == expected_pattern, f"Pattern mismatch for {demo_id}: {data['diagnosis']['root_cause']}"
        assert data["dag"]["has_graph"] is True
        assert len(data["dag"]["nodes"]) >= 3
        # Ensure not saved to history
        assert data.get("saved_record_id") is None
        
    hist_after = len(client.get("/api/history").get_json())
    assert hist_after == hist_before, "Running demo traces must not pollute user history without explicit save!"

def test_trace_upload_validation(client):
    """Test rejection of uploaded trace when required fields are missing"""
    client.post("/api/auth/demo", json={})
    
    # Missing total_turns and tool_calls_total
    bad_trace = {
        "error_count": 2,
        "retry_count": 1
    }
    res = client.post("/api/diagnose", json={"trace_data": bad_trace, "save": False})
    assert res.status_code == 400
    err_data = res.get_json()
    assert "missing required" in err_data["error"].lower()

def test_model_selection_inference_routing(client):
    """Verify selecting different saved models routes to actual saved model inference"""
    client.post("/api/auth/demo", json={})
    
    # 1. Multi-View Fusion
    res_fusion = client.post("/api/diagnose", json={
        "demo_id": "02_retry_thrashing",
        "model_choice": "Proposed Multi-View Fusion Network",
        "save": False
    })
    assert res_fusion.status_code == 200
    data_f = res_fusion.get_json()
    assert data_f["diagnosis"]["model_used"] == "Proposed Multi-View Fusion Network"
    
    # 2. XGBoost
    res_xgb = client.post("/api/diagnose", json={
        "demo_id": "02_retry_thrashing",
        "model_choice": "XGBoost Classifier",
        "save": False
    })
    assert res_xgb.status_code == 200
    data_x = res_xgb.get_json()
    assert data_x["diagnosis"]["model_used"] == "XGBoost Classifier"
    assert "failure_probability" in data_x["diagnosis"]
    
    # 3. Random Forest
    res_rf = client.post("/api/diagnose", json={
        "demo_id": "02_retry_thrashing",
        "model_choice": "Random Forest (Balanced)",
        "save": False
    })
    assert res_rf.status_code == 200
    data_rf = res_rf.get_json()
    assert data_rf["diagnosis"]["model_used"] == "Random Forest (Balanced)"

def test_novelty_failure_signature_and_behaviour_deviation(client):
    """Verify novelty features: Failure Signature & Behavioural Baseline Deviation"""
    client.post("/api/auth/demo", json={})
    res = client.post("/api/diagnose", json={"demo_id": "02_retry_thrashing", "save": False})
    assert res.status_code == 200
    diag = res.get_json()["diagnosis"]
    
    # Verify Failure Signature
    assert "failure_signature" in diag
    sig = diag["failure_signature"]
    assert "pattern" in sig
    assert "observed_signals" in sig
    assert len(sig["observed_signals"]) >= 2
    assert "execution_evidence" in sig
    assert "suspicious_step" in sig
    
    # Verify Behavioural Deviation
    assert "behaviour_deviation" in diag
    dev = diag["behaviour_deviation"]
    assert "tool_diversity" in dev
    assert "retry_ratio" in dev
    assert "error_rate" in dev
    assert "execution_duration" in dev
    assert dev["retry_ratio"]["level"] in ["critical", "high", "moderate", "nominal"]

def test_direct_deep_link_history_retrieval(client):
    """Verify saved record contains raw telemetry and deep-links directly"""
    client.post("/api/auth/demo", json={})
    
    # Save a diagnosis
    res_save = client.post("/api/diagnose", json={
        "demo_id": "03_environment_failure",
        "save": True
    })
    assert res_save.status_code == 200
    rec_id = res_save.get_json()["saved_record_id"]
    assert rec_id is not None
    
    # Direct retrieval by record ID
    res_detail = client.get(f"/api/history/{rec_id}")
    assert res_detail.status_code == 200
    rec = res_detail.get_json()
    assert "raw_telemetry" in rec
    assert rec["raw_telemetry"]["total_turns"] == 55
    assert "failure_signature" in rec
    assert "behaviour_deviation" in rec
    assert rec["source_type"] == "Curated Demo Case"
    
    # Direct retrieval by trace ID prefix
    res_trace = client.get("/api/history/by-trace/DEMO-03")
    assert res_trace.status_code == 200
    assert res_trace.get_json()["result_id"] == rec_id


