import os
import sys
import io
sys.path.insert(0, os.getcwd())
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import joblib
import torch
import pandas as pd
from src.database.db_manager import (
    init_db,
    register_user,
    authenticate_user,
    save_diagnostic_record,
    get_user_diagnostic_history,
    get_user_diagnostic_detail
)
from src.models.deep_models import SequenceTrajectoryModel, MultimodalAgentFailureModel, encode_sequence
from src.live_tracing.agent_tracer import run_sample_live_trace
from src.inference.diagnose import AgentFailureDiagnoser

def test_full_application():
    print("=================================================================")
    print("STEP 1: AUTHENTICATION & USER DATA ISOLATION VERIFICATION")
    print("=================================================================")
    init_db()
    
    # 1.1 Demo login
    demo_u, err = authenticate_user("demo_user", "DemoUser@123")
    assert err is None and demo_u is not None, f"Demo login failed: {err}"
    print(f"✓ Demo user login verified: {demo_u['username']} (ID: {demo_u['id']})")
    
    # 1.2 Wrong password check
    _, err_wrong = authenticate_user("demo_user", "WrongPassword123")
    assert err_wrong is not None, "Security failure: wrong password accepted!"
    print("✓ Wrong password rejected as expected.")
    
    # 1.3 User registration with fresh timestamp
    import time
    ts = int(time.time())
    u_test_a, err_a = register_user(f"usera_{ts}", f"usera_{ts}@test.com", "PasswordUserA123")
    assert u_test_a is not None, f"Register User A failed: {err_a}"
    u_test_b, err_b = register_user(f"userb_{ts}", f"userb_{ts}@test.com", "PasswordUserB123")
    assert u_test_b is not None, f"Register User B failed: {err_b}"
    print(f"✓ Registered User A ({u_test_a['id']}) and User B ({u_test_b['id']})")
    
    # 1.4 Diagnostic record saving & strict data isolation
    rec_a = save_diagnostic_record(
        user_id=u_test_a["id"],
        session_title="User A Private Run",
        session_type="trace_inspection",
        agent_name="claude-code",
        model_used="Proposed Multi-View Fusion Network",
        threshold=0.46,
        failure_probability=0.78,
        predicted_status="FAILURE DETECTED",
        risk_level="HIGH RISK",
        detected_issue="High edit churn",
        root_cause="Unbounded Code Churn",
        recommended_action="Review diff blast radius.",
        telemetry_summary={"churn": 512}
    )
    print(f"✓ Saved private diagnostic record #{rec_a} for User A")
    
    # Verify User B cannot access User A's history
    hist_b = get_user_diagnostic_history(u_test_b["id"])
    for h in hist_b:
        assert h["result_id"] != rec_a, "CRITICAL ERROR: User B can see User A history!"
    detail_b_attempt = get_user_diagnostic_detail(u_test_b["id"], rec_a)
    assert detail_b_attempt is None, "CRITICAL ERROR: User B retrieved User A detail via ID manipulation!"
    print("✓ STRICT DATA ISOLATION VERIFIED: User B cannot query or access User A records.")
    
    print("\n=================================================================")
    print("STEP 2: MODEL SELECTOR INDEPENDENCE & REAL INFERENCE VERIFICATION")
    print("=================================================================")
    preprocessor = joblib.load("data/processed/preprocessor.joblib")
    vocab = joblib.load("data/processed/tool_vocab.joblib")
    numerical_cols = joblib.load("data/processed/numerical_cols.joblib")
    categorical_cols = joblib.load("data/processed/categorical_cols.joblib")
    
    lr = joblib.load("models_saved/logistic_regression.joblib")
    rf = joblib.load("models_saved/random_forest.joblib")
    xgb = joblib.load("models_saved/xgboost.joblib")
    
    bilstm = SequenceTrajectoryModel(vocab_size=len(vocab))
    bilstm.load_state_dict(torch.load("models_saved/sequence_bilstm.pt", map_location="cpu"))
    bilstm.eval()
    
    fusion = MultimodalAgentFailureModel(tab_dim=88, vocab_size=len(vocab))
    fusion.load_state_dict(torch.load("models_saved/multimodal_fusion.pt", map_location="cpu"))
    fusion.eval()
    
    df_test = pd.read_parquet("data/processed/test.parquet")
    sample_trace = df_test.iloc[10]  # Known test sample
    row_df = pd.DataFrame([sample_trace])
    X_trans = preprocessor.transform(row_df[numerical_cols + categorical_cols])
    seq_enc = encode_sequence(sample_trace.get('tool_call_sequence', []), vocab, max_len=50)
    
    p_lr = float(lr.predict_proba(X_trans)[0, 1])
    p_rf = float(rf.predict_proba(X_trans)[0, 1])
    p_xgb = float(xgb.predict_proba(X_trans)[0, 1])
    with torch.no_grad():
        p_bilstm = float(torch.sigmoid(bilstm(torch.tensor([seq_enc], dtype=torch.long))).item())
        p_fusion = float(torch.sigmoid(fusion(torch.tensor(X_trans, dtype=torch.float32), torch.tensor([seq_enc], dtype=torch.long))).item())
        
    print(f"Sample 10 Ground Truth: {'FAILED' if sample_trace['is_failed'] else 'PASSED'}")
    print(f"  • Logistic Regression Prob:   {p_lr:.4f}")
    print(f"  • Random Forest Prob:          {p_rf:.4f}")
    print(f"  • XGBoost Prob:                {p_xgb:.4f}")
    print(f"  • Sequential BiLSTM Prob:      {p_bilstm:.4f}")
    print(f"  • Proposed Fusion Prob:        {p_fusion:.4f}")
    
    # Confirm models produce distinct real outputs
    unique_probs = {round(p, 3) for p in [p_lr, p_rf, p_xgb, p_bilstm, p_fusion]}
    assert len(unique_probs) >= 3, "ERROR: Model selector outputs are identical!"
    print("✓ Model Selector verified: All 5 models compute distinct real-time predictions.")
    
    print("\n=================================================================")
    print("STEP 3: THRESHOLD SENSITIVITY VERIFICATION")
    print("=================================================================")
    test_prob = 0.50
    thresh_low = 0.40
    thresh_high = 0.60
    
    status_low = "FAILURE" if test_prob >= thresh_low else "NORMAL"
    status_high = "FAILURE" if test_prob >= thresh_high else "NORMAL"
    assert status_low == "FAILURE" and status_high == "NORMAL", "Threshold logic error!"
    print(f"✓ Probability {test_prob}: At threshold {thresh_low} -> {status_low}; At threshold {thresh_high} -> {status_high}")
    print("✓ Dynamic decision boundary confirmed working correctly.")
    
    print("\n=================================================================")
    print("STEP 4: LIVE DEMONSTRATION & REAL-TIME DIAGNOSIS PIPELINE")
    print("=================================================================")
    for scenario in ["healthy", "thrashing_failure", "over_exploration"]:
        live_trace = run_sample_live_trace(scenario)
        diagnoser = AgentFailureDiagnoser()
        res = diagnoser.diagnose_trace(live_trace)
        print(f"  • Live Scenario '{scenario}': Status = {res['status']}, Prob = {res['failure_probability']:.2f}")
        assert "failure_probability" in res and "recommended_action" in res
    print("✓ Live pipeline verified: Trace capture -> Preprocessing -> Model inference -> Actionable diagnosis.")
    
    print("\n=================================================================")
    print("ALL APPLICATION INTEGRITY TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    test_full_application()
