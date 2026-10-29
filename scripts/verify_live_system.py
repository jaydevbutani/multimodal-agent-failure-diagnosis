import requests

BASE_URL = "http://127.0.0.1:8502"
session = requests.Session()

print("--- 1. Testing Demo Authentication ---")
res_auth = session.post(f"{BASE_URL}/api/auth/demo", json={})
assert res_auth.status_code == 200, f"Auth failed: {res_auth.text}"
auth_data = res_auth.json()
print("Auth success:", auth_data.get("username"), auth_data.get("role"))

print("\n--- 2. Testing Model Evaluation & Methodology ---")
res_eval = session.get(f"{BASE_URL}/model-evaluation")
assert res_eval.status_code == 200
assert "Model Evaluation &amp; Methodology" in res_eval.text or "Model Evaluation & Methodology" in res_eval.text
assert "11,995" in res_eval.text
assert "0% Task Overlap" in res_eval.text or "GroupShuffleSplit" in res_eval.text
print("Model Evaluation & Methodology page verified!")

print("\n--- 3. Testing Legacy Redirect /dataset-methodology ---")
res_redir = session.get(f"{BASE_URL}/dataset-methodology", allow_redirects=False)
assert res_redir.status_code == 302
assert res_redir.headers["Location"].endswith("/model-evaluation")
print("Redirect verified (302 -> /model-evaluation)!")

print("\n--- 4. Testing Diagnose API on environment_failure ---")
res_diag = session.post(f"{BASE_URL}/api/diagnose", json={
    "demo_case": "environment_failure",
    "model_choice": "Proposed Multi-View Fusion Network",
    "threshold": 0.46,
    "save": False
})
assert res_diag.status_code == 200, f"Diagnose error: {res_diag.text}"
diag_data = res_diag.json()
assert diag_data["success"] is True
diag = diag_data["diagnosis"]
print("Diagnose environment_failure result:")
print("  Status:", diag.get("status"))
print("  Root Cause:", diag.get("root_cause"))
print("  What Happened:", diag.get("what_happened"))
print("  What Went Wrong:", diag.get("what_went_wrong"))
print("  Responsible Component:", diag.get("responsible_component"))
print("  Suspicious Step:", diag.get("suspicious_step"))

print("\n--- 5. Testing Live Simulation API on thrashing_failure ---")
res_live = session.post(f"{BASE_URL}/api/live-diagnosis", json={
    "scenario": "thrashing_failure",
    "model_choice": "Proposed Multi-View Fusion Network",
    "threshold": 0.46,
    "save": False
})
assert res_live.status_code == 200, f"Live simulation error: {res_live.text}"
live_data = res_live.json()
assert live_data["success"] is True
print("Live thrashing_failure result:")
print("  Trace ID:", live_data.get("trace_id"))
print("  Scenario Expected Status:", live_data.get("scenario_expected_status"))
print("  ML Predicted Status:", live_data.get("ml_predicted_status"))
print("  ML Confidence:", live_data.get("ml_confidence"))
print("  What Happened:", live_data.get("what_happened"))
print("  Responsible Component:", live_data.get("responsible_component"))
print("  Suspicious Step:", live_data.get("suspicious_step"))

print("\n--- 6. Testing Canvas Deep-link with Cached Live Trace ---")
t_id = live_data.get("trace_id")
res_canvas = session.post(f"{BASE_URL}/api/diagnose", json={
    "trace_id": t_id,
    "model_choice": "Proposed Multi-View Fusion Network",
    "threshold": 0.46
})
assert res_canvas.status_code == 200
canvas_data = res_canvas.json()
assert canvas_data["success"] is True
print(f"Canvas successfully retrieved cached live trace {t_id} with {len(canvas_data.get('dag_nodes', []))} DAG nodes!")

print("\n================ ALL LIVE TESTS PASSED! ================")
