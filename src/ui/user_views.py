import json
import streamlit as st
import pandas as pd
import numpy as np
from src.database.db_manager import (
    save_diagnostic_record,
    get_user_diagnostic_history,
    get_user_diagnostic_detail
)
from src.services.diagnosis_service import DiagnosisService
from src.live_tracing.agent_tracer import run_sample_live_trace

# ------------------------------------------------------------------------------
# 1. USER DASHBOARD
# ------------------------------------------------------------------------------
def render_user_dashboard(user, artifacts):
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Agent Health Intelligence</div>
        <div class="hero-subtitle">
            Diagnose agent execution failures using runtime telemetry and sequential tool behavior.
        </div>
        <div style="background: rgba(255,255,255,0.08); padding: 12px 18px; border-radius: 8px; font-size: 0.95rem; margin-top: 14px; border: 1px solid rgba(255,255,255,0.12);">
            Upload or select an agent execution trace. The system extracts execution telemetry and tool-call behavior,
            passes them through the trained ML pipeline, estimates failure risk, identifies supporting signals,
            and recommends an operational action.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual Pipeline Flow Card
    st.markdown("### 🔄 Diagnostic System Flow")
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; font-weight: 600; color: #1e293b; text-align: center;">
            <div style="flex: 1; min-width: 120px; padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="font-size: 1.3rem;">📋</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">Agent Trace</div>
            </div>
            <div style="color: #2563eb; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 120px; padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="font-size: 1.3rem;">⚙️</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">Feature Extraction</div>
            </div>
            <div style="color: #2563eb; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 120px; padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="font-size: 1.3rem;">🧠</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">ML Multi-View Model</div>
            </div>
            <div style="color: #2563eb; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 120px; padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="font-size: 1.3rem;">🎯</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">Risk Prediction</div>
            </div>
            <div style="color: #2563eb; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 120px; padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="font-size: 1.3rem;">💡</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">Explanation</div>
            </div>
            <div style="color: #2563eb; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 120px; padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="font-size: 1.3rem;">🛠️</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">Operational Action</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # User Account History Metrics
    history = get_user_diagnostic_history(user["id"])
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("""
        <div class="clean-metric">
            <div class="clean-metric-label">Active Diagnostic Model</div>
            <div class="clean-metric-value" style="font-size: 1.25rem; color: #2563eb;">Multi-View Fusion</div>
            <div style="color: #64748b; font-size: 0.82rem; margin-top: 4px;">Tabular Telemetry + Sequence LSTM</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Your Saved Diagnoses</div>
            <div class="clean-metric-value">{len(history)}</div>
            <div style="color: #64748b; font-size: 0.82rem; margin-top: 4px;">User: <strong>{user['username']}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    with col_d3:
        fail_count = sum(1 for h in history if "FAILURE" in h.get("predicted_status", "") or "FAILED" in h.get("predicted_status", ""))
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Intercepted Failures</div>
            <div class="clean-metric-value" style="color: #dc2626;">{fail_count}</div>
            <div style="color: #64748b; font-size: 0.82rem; margin-top: 4px;">Prevented silent agent outages</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Recent Diagnoses List
    st.subheader("📋 Recent Personal Diagnoses")
    if not history:
        st.info("No diagnoses recorded yet for your account. Go to **Diagnose Agent** or **Live Trace Diagnosis** to run your first evaluation.")
    else:
        for item in history[:4]:
            badge = "badge-healthy" if "HEALTHY" in item['predicted_status'] else "badge-failure"
            st.markdown(f"""
            <div class="product-card" style="padding: 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-weight: 600; font-size: 0.95rem;">{item['session_title']}</span>
                    <span class="{badge}">{item['predicted_status']}</span>
                </div>
                <div style="color: #64748b; font-size: 0.85rem;">
                    Agent: <strong>{item['agent_name']}</strong> &bull; Failure Prob: <strong>{item['failure_probability']*100:.1f}%</strong> &bull; {item['created_at'][:16]}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. DIAGNOSE AGENT (THE MOST IMPORTANT PAGE)
# ------------------------------------------------------------------------------
def render_user_diagnose(user, artifacts):
    st.header("🔍 Diagnose Agent Execution")
    st.markdown("""
    Select or upload an agent execution trace to evaluate operational stability and predict failure probability.
    """)
    
    service = DiagnosisService(artifacts)
    df_test = artifacts["df_test"]
    
    # 3 Input Modes
    input_mode = st.radio(
        "Select Input Mode:",
        ["Mode 1: Benchmark Trace", "Mode 2: Upload Trace (JSON/CSV)", "Mode 3: Live Trace Scenario"],
        horizontal=True
    )
    
    selected_trace_data = None
    
    # --- Mode 1: Benchmark Trace ---
    if input_mode == "Mode 1: Benchmark Trace":
        st.subheader("Mode 1: Select Real Benchmark Trace")
        filter_status = st.selectbox("Filter Benchmark Traces:", ["All Traces", "Known Failure Traces Only", "Known Nominal Traces Only"])
        
        if filter_status == "Known Failure Traces Only":
            active_df = df_test[df_test['is_failed'] == 1]
        elif filter_status == "Known Nominal Traces Only":
            active_df = df_test[df_test['is_failed'] == 0]
        else:
            active_df = df_test
            
        selected_idx = st.selectbox(
            "Select Agent Diagnostics Session:",
            active_df.index[:100],
            format_func=lambda x: f"Trial: {str(active_df.loc[x, 'task_id'])[:24]} | Agent: {active_df.loc[x, 'model']} | Status: {'FAILED' if active_df.loc[x, 'is_failed'] else 'PASSED'}"
        )
        selected_trace_data = active_df.loc[selected_idx].to_dict()
        
    # --- Mode 2: Upload Trace ---
    elif input_mode == "Mode 2: Upload Trace (JSON/CSV)":
        st.subheader("Mode 2: Upload Agent Execution Trace")
        uploaded_file = st.file_uploader("Upload JSON or CSV trace file:", type=["json", "csv"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = json.load(uploaded_file)
                    selected_trace_data = content if isinstance(content, dict) else content[0]
                else:
                    up_df = pd.read_csv(uploaded_file)
                    selected_trace_data = up_df.iloc[0].to_dict()
                st.success(f"Trace uploaded successfully: **{uploaded_file.name}**")
            except Exception as e:
                st.error(f"Unable to parse uploaded trace: {str(e)}")
        else:
            st.info("Upload a JSON or CSV trace matching the agent telemetry schema, or load an example below.")
            if st.button("Load Example High-Risk Trace"):
                selected_trace_data = {
                    "task_id": "UPLOAD-SAMPLE-RISK-01",
                    "model": "claude-code",
                    "benchmark": "synthetic-test",
                    "total_turns": 16,
                    "tool_calls_total": 28,
                    "search_tool_calls": 8,
                    "edit_tool_calls": 6,
                    "error_count": 9,
                    "retry_count": 5,
                    "unique_files_read": 3,
                    "unique_files_edited": 1,
                    "patch_size_lines": 720,
                    "duration_seconds": 94.2,
                    "tool_call_sequence": ["bash", "grep", "view", "edit", "bash", "bash", "bash", "edit", "bash"]
                }
                st.success("Loaded Example High-Risk Trace.")
                
    # --- Mode 3: Live Trace Scenario ---
    else:
        st.subheader("Mode 3: Run Controlled Live Demo Scenario")
        sc_choice = st.selectbox(
            "Select Scenario to Execute in Harness:",
            [
                ("healthy", "🟢 Nominal Healthy Execution (Read → Search → Edit → Tests pass)"),
                ("thrashing_failure", "🔴 Command Retry Thrashing (Repeated Bash errors → Cyclic retries → Large churn)"),
                ("over_exploration", "🟡 Information Over-Exploration (12 consecutive search queries with zero edits)")
            ],
            format_func=lambda x: x[1]
        )[0]
        
        if st.button("⚡ Execute Live Agent & Capture Telemetry", type="primary"):
            with st.spinner("Executing live agent harness..."):
                selected_trace_data = run_sample_live_trace(sc_choice)
                st.session_state["live_captured_trace"] = selected_trace_data
                st.success(f"Captured live telemetry: **{selected_trace_data['trace_id']}**")
        elif "live_captured_trace" in st.session_state:
            selected_trace_data = st.session_state["live_captured_trace"]
            
    if selected_trace_data is None:
        st.warning("Please select or upload a trace above to proceed with diagnosis.")
        return
        
    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # DISPLAY MODEL INPUT
    # --------------------------------------------------------------------------
    st.subheader("📥 Model Input: Extracted Features & Telemetry")
    st.markdown("These exact execution telemetry features and sequence transitions are passed to the ML pipeline:")
    
    # Extracted Telemetry Grid
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Total Turns", int(selected_trace_data.get('total_turns', 1)))
        st.metric("Tool Invocations", int(selected_trace_data.get('tool_calls_total', 0)))
    with c_m2:
        st.metric("Environment Errors", int(selected_trace_data.get('error_count', 0)))
        st.metric("Command Retries", int(selected_trace_data.get('retry_count', 0)))
    with c_m3:
        st.metric("Files Edited", int(selected_trace_data.get('unique_files_edited', 0)))
        st.metric("Patch Size (Lines)", int(selected_trace_data.get('patch_size_lines', 0)))
    with c_m4:
        churn_val = float(selected_trace_data.get('DERIVED_edit_churn', 0.0))
        if churn_val == 0.0 and int(selected_trace_data.get('unique_files_edited', 0)) > 0:
            churn_val = float(selected_trace_data.get('patch_size_lines', 0)) / max(1, int(selected_trace_data.get('unique_files_edited', 1)))
        st.metric("Edit Churn (Lines/File)", f"{churn_val:.1f}")
        st.metric("Duration", f"{float(selected_trace_data.get('duration_seconds', 1.0)):.1f}s")
        
    # Tool-Call Sequence Display (Genuinely safe Python list conversion)
    st.markdown("##### 🔎 Chronological Tool-Call Sequence")
    raw_seq = selected_trace_data.get('tool_call_sequence', [])
    if isinstance(raw_seq, (np.ndarray, pd.Series)):
        seq_list = [str(x) for x in raw_seq.tolist()]
    elif isinstance(raw_seq, (list, tuple)):
        seq_list = [str(x) for x in raw_seq]
    else:
        seq_list = []
        
    if len(seq_list) > 0:
        st.code(" → ".join(seq_list[:25]) + (" ..." if len(seq_list) > 25 else ""))
    else:
        st.info("No tool calls recorded in this session trajectory.")
        
    with st.expander("ℹ️ How does the ML model use this input?", expanded=False):
        st.markdown("""
        - **Execution Telemetry → Tabular ML Branch:** 21 numerical features + 3 categorical variables are scaled and passed through tabular classifiers (XGBoost / Random Forest) and the tabular dense projection layer.
        - **Tool-Call Sequence → Sequential LSTM Branch:** The sequence of tools is mapped into integer token embeddings ($d=32$) and processed via a Bidirectional LSTM ($h=64$) to capture temporal loop dynamics.
        - **Telemetry + Sequence → Multi-View Fusion Network:** Learned cross-modal gating combines tabular behavioral state with temporal sequence state to predict the overall failure probability.
        """)
        
    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # RUN DIAGNOSIS
    # --------------------------------------------------------------------------
    st.subheader("🎯 ML Diagnosis Output")
    
    result = service.diagnose(selected_trace_data, model_choice="Proposed Multi-View Fusion Network", threshold=0.46)
    
    # Status Banner
    prob_pct = result["failure_probability"] * 100.0
    risk_level = result["risk_level"]
    status_label = result["status"]
    
    if risk_level == "High":
        hero_class = "diag-hero-failure"
        hero_icon = "🔴"
    elif risk_level == "Medium":
        hero_class = "diag-hero-warning"
        hero_icon = "🟡"
    else:
        hero_class = "diag-hero-healthy"
        hero_icon = "🟢"
        
    st.markdown(f"""
    <div class="diag-hero-card {hero_class}">
        <div style="font-size: 2.8rem;">{hero_icon}</div>
        <div class="diag-status-text">{status_label}</div>
        <div class="diag-prob-text">Estimated Failure Probability: <strong>{prob_pct:.1f}%</strong> &bull; <strong>{risk_level.upper()} RISK</strong></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Supporting Evidence & Recommendation Cards
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown("### 💡 Why Did the Model Make This Decision?")
        st.markdown(f"**Primary Assessment:** {result['why']}")
        st.markdown(f"**Identified Root Cause:** `{result['root_cause']}`")
        
        st.markdown("#### Evidence Signals")
        for ev in result["evidence"]:
            st.markdown(f"• {ev}")
            
        st.caption("ℹ️ *These features are predictive contributors associated with the model's decision. They do not establish causal relationships.*")
        
    with col_out2:
        st.markdown("### 🛠️ Recommended Operational Action")
        st.info(result['recommended_action'])
        
        st.markdown("#### Operational Activity Signals")
        sig = result["signals"]
        st.markdown(f"""
        <div class="product-card" style="padding: 16px;">
            <p style="margin-bottom: 6px;"><strong>Tool Activity:</strong> {sig['tool_activity']} ({sig['tool_count']} invocations)</p>
            <p style="margin-bottom: 6px;"><strong>Environment Error State:</strong> {sig['errors']}</p>
            <p style="margin-bottom: 6px;"><strong>Code Modification Velocity:</strong> {sig['code_churn']}</p>
            <p style="margin-bottom: 0;"><strong>Command Retry Tendency:</strong> {sig['retries']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Save Diagnosis Button
    if st.button("💾 Save Diagnosis to My Account History", type="primary"):
        rec_id = save_diagnostic_record(
            user_id=user["id"],
            session_title=f"Diagnosis: {str(result['task_id'])[:20]}",
            session_type="trace_inspection",
            agent_name=str(result["agent_model"]),
            model_used=result["model_used"],
            threshold=result["threshold"],
            failure_probability=result["failure_probability"],
            predicted_status=result["status"],
            risk_level=result["risk_level"],
            detected_issue=result["why"],
            root_cause=result["root_cause"],
            recommended_action=result["recommended_action"],
            telemetry_summary=result["telemetry"]
        )
        st.success(f"Diagnosis saved to your personal history (Record #{rec_id}).")

# ------------------------------------------------------------------------------
# 3. LIVE TRACE DIAGNOSIS PAGE
# ------------------------------------------------------------------------------
def render_user_live_demo(user, artifacts):
    st.header("⚡ Live Trace Diagnosis")
    st.markdown("""
    Execute a controlled agent scenario, capture its telemetry, and pass the resulting trace through
    the same diagnostic pipeline used for benchmark analysis.
    """)
    
    scenario_choice = st.selectbox(
        "Select Controlled Agent Execution Scenario:",
        [
            ("healthy", "🟢 Nominal Healthy Execution (Read → Search → Edit → Unit tests pass)"),
            ("thrashing_failure", "🔴 Command Retry Thrashing (Repeated Bash errors → Cyclic retries → Large patch churn)"),
            ("over_exploration", "🟡 Information Over-Exploration (12 consecutive search queries with zero edits)")
        ],
        format_func=lambda x: x[1]
    )[0]
    
    if st.button("🚀 Run Live Diagnosis", type="primary"):
        st.markdown("### Execution Pipeline Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("1/5: Agent running scenario in environment...")
        progress_bar.progress(20)
        live_trace = run_sample_live_trace(scenario_choice)
        
        status_text.text("2/5: Capturing execution telemetry and tool transitions...")
        progress_bar.progress(40)
        
        status_text.text("3/5: Extracting 10 DERIVED_ behavioral features...")
        progress_bar.progress(60)
        
        status_text.text("4/5: Applying trained preprocessing transformation...")
        progress_bar.progress(80)
        
        service = DiagnosisService(artifacts)
        result = service.diagnose(live_trace, model_choice="Proposed Multi-View Fusion Network", threshold=0.46)
        
        status_text.text("5/5: Running trained Multi-View Fusion model inference...")
        progress_bar.progress(100)
        
        # Save to database for authenticated user
        rec_id = save_diagnostic_record(
            user_id=user["id"],
            session_title=f"Live Demo: {scenario_choice.capitalize()}",
            session_type="live_demonstration",
            agent_name="claude-code (live)",
            model_used=result["model_used"],
            threshold=result["threshold"],
            failure_probability=result["failure_probability"],
            predicted_status=result["status"],
            risk_level=result["risk_level"],
            detected_issue=result["why"],
            root_cause=result["root_cause"],
            recommended_action=result["recommended_action"],
            telemetry_summary=result["telemetry"]
        )
        
        st.markdown("---")
        
        # Display Results
        prob_pct = result["failure_probability"] * 100.0
        risk_level = result["risk_level"]
        hero_class = "diag-hero-failure" if risk_level == "High" else ("diag-hero-warning" if risk_level == "Medium" else "diag-hero-healthy")
        hero_icon = "🔴" if risk_level == "High" else ("🟡" if risk_level == "Medium" else "🟢")
        
        st.markdown(f"""
        <div class="diag-hero-card {hero_class}">
            <div style="font-size: 2.8rem;">{hero_icon}</div>
            <div class="diag-status-text">{result['status']}</div>
            <div class="diag-prob-text">Estimated Failure Probability: <strong>{prob_pct:.1f}%</strong> &bull; <strong>{risk_level.upper()} RISK</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📋 Captured Live Telemetry")
            tel = result["telemetry"]
            st.json({
                "trace_id": live_trace['trace_id'],
                "total_turns": tel['total_turns'],
                "tool_calls": tel['tool_calls_total'],
                "errors": tel['error_count'],
                "retries": tel['retry_count'],
                "duration_seconds": tel['duration_seconds'],
                "edit_churn": tel['DERIVED_edit_churn'],
                "retry_ratio": tel['DERIVED_retry_ratio']
            })
        with c2:
            st.markdown("### 💡 Root Cause & Action")
            st.markdown(f"**Identified Root Cause:** `{result['root_cause']}`")
            st.markdown(f"**Supporting Evidence:** {result['why']}")
            st.markdown("#### Recommended SRE Action")
            st.info(result['recommended_action'])
            
        st.success(f"Live diagnosis completed and automatically saved to your history (Record #{rec_id}).")

# ------------------------------------------------------------------------------
# 4. DIAGNOSTIC HISTORY
# ------------------------------------------------------------------------------
def render_user_history(user):
    st.header("📋 Diagnostic History")
    st.markdown(f"All saved agent diagnoses belonging to **{user['username']}** (strictly isolated at database layer).")
    
    history = get_user_diagnostic_history(user["id"])
    
    if not history:
        st.info("You have not saved any diagnostic runs yet. Run a diagnosis or live demo to build your history.")
        return
        
    h_df = pd.DataFrame(history)
    display_df = h_df[["created_at", "session_title", "agent_name", "predicted_status", "failure_probability", "root_cause"]].copy()
    display_df.columns = ["Timestamp", "Session", "Agent Model", "Status", "Failure Prob", "Root Cause"]
    display_df["Failure Prob"] = display_df["Failure Prob"].apply(lambda p: f"{p*100:.1f}%")
    
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("### 🔍 Inspect Complete Diagnostic Record")
    selected_id = st.selectbox(
        "Select History Entry to Inspect:",
        [h["result_id"] for h in history],
        format_func=lambda x: next(f"Record #{x}: {h['session_title']} — {h['predicted_status']} ({h['created_at'][:16]})" for h in history if h['result_id'] == x)
    )
    
    detail = get_user_diagnostic_detail(user["id"], selected_id)
    if detail:
        badge = "badge-healthy" if "HEALTHY" in detail['predicted_status'] else "badge-failure"
        st.markdown(f"""
        <div class="product-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0;">{detail['session_title']}</h4>
                <span class="{badge}">{detail['predicted_status']}</span>
            </div>
            <p><strong>Agent:</strong> {detail['agent_name']} &bull; <strong>Model Engine:</strong> {detail['model_used']}</p>
            <p><strong>Failure Probability:</strong> {detail['failure_probability']*100:.1f}% &bull; <strong>Risk Level:</strong> {detail['risk_level']}</p>
            <p><strong>Identified Root Cause:</strong> {detail['root_cause']}</p>
            <p><strong>Recommended Action:</strong> {detail['recommended_action']}</p>
            <p style="color: #64748b; font-size: 0.85rem;">Saved on: {detail['created_at']}</p>
        </div>
        """, unsafe_allow_html=True)
