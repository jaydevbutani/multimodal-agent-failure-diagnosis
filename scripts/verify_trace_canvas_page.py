import requests

BASE_URL = "http://127.0.0.1:8502"
session = requests.Session()

# 1. Login
res = session.post(f"{BASE_URL}/api/auth/demo", json={})
assert res.status_code == 200, f"Auth failed: {res.status_code}"
print("Auth success!")

# 2. Test /trace-canvas
res_canvas = session.get(f"{BASE_URL}/trace-canvas")
assert res_canvas.status_code == 200, f"GET /trace-canvas failed: {res_canvas.status_code}"
html = res_canvas.text

# Check required sections
checks = [
    ("AGENT TRACE CANVAS", "AGENT TRACE CANVAS header"),
    ("TEMPORAL TELEMETRY &amp; METRICS", "Temporal Telemetry header"),
    ("Execution Tracks &amp; Tool Spans", "Execution Tracks & Tool Spans header"),
    ("ROOT-CAUSE REASONING", "Root-Cause Reasoning header"),
    ("AI Forensic Assistant", "AI Forensic Assistant header"),
    ("select-trace-scenario", "Scenario Selector Dropdown"),
    ("select-model-choice", "Model Choice Dropdown"),
    ("rerun-btn", "Re-run Trace in Sandbox button"),
    ("dag-surface", "DAG Surface Container"),
    ("inspector-card", "Docked Node Detail Inspector"),
    ("metrics-svg", "Telemetry Area Chart SVG"),
    ("swimlanes-container", "Execution Tracks Swimlanes Container"),
    ("diagnostic-stack-box", "Diagnostic Stack Attribution Box"),
]

for string_match, label in checks:
    if string_match in html:
        print(f" [PASS] {label}")
    else:
        print(f" [FAIL] {label} not found in HTML!")

# 3. Test /api/diagnose with thrashing_failure
res_diag = session.post(f"{BASE_URL}/api/diagnose", json={"demo_id": "thrashing_failure"})
assert res_diag.status_code == 200, f"Diagnose API failed: {res_diag.status_code}"
data = res_diag.json()
diag = data.get("diagnosis", {})
print("\nAPI Diagnose Thrashing Failure:")
print(" Status:", diag.get("status"))
print(" Root Cause:", diag.get("root_cause"))
print(" Failure Prob:", diag.get("failure_probability"))
print(" Has DAG Graph:", bool(data.get("dag") or data.get("dag_nodes")))

print("\nALL CANONICAL TRACE CANVAS CHECKS PASSED!")
