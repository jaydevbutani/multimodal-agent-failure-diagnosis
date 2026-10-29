import os
import sys
sys.path.insert(0, os.getcwd())
import joblib
import pandas as pd
import torch
import streamlit as st

# Import Database & Auth Modules
from src.database.db_manager import init_db
from src.ui.styles import get_custom_css
from src.ui.auth_views import render_login_view, render_user_profile
from src.ui.user_views import (
    render_user_dashboard,
    render_user_diagnose,
    render_user_history,
    render_user_live_demo
)
from src.ui.evaluator_views import render_evaluator_console
from src.models.deep_models import MultimodalAgentFailureModel, SequenceTrajectoryModel

# Initialize SQLite database on startup
init_db()

# Streamlit Page Setup
st.set_page_config(
    page_title="Agent Health Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Modern UI Styles
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Cache-Load All Artifacts & Models
@st.cache_resource
def load_all_artifacts():
    preprocessor = joblib.load("data/processed/preprocessor.joblib")
    feature_names = joblib.load("data/processed/feature_names.joblib")
    numerical_cols = joblib.load("data/processed/numerical_cols.joblib")
    categorical_cols = joblib.load("data/processed/categorical_cols.joblib")
    vocab = joblib.load("data/processed/tool_vocab.joblib")
    
    # 5 Verified Model Architectures
    lr_model = joblib.load("models_saved/logistic_regression.joblib")
    rf_model = joblib.load("models_saved/random_forest.joblib")
    xgb_model = joblib.load("models_saved/xgboost.joblib")
    
    bilstm_model = SequenceTrajectoryModel(vocab_size=len(vocab))
    bilstm_model.load_state_dict(torch.load("models_saved/sequence_bilstm.pt", map_location=torch.device('cpu')))
    bilstm_model.eval()
    
    fusion_model = MultimodalAgentFailureModel(tab_dim=88, vocab_size=len(vocab))
    fusion_model.load_state_dict(torch.load("models_saved/multimodal_fusion.pt", map_location=torch.device('cpu')))
    fusion_model.eval()
    
    df_test = pd.read_parquet("data/processed/test.parquet")
    df_comp = pd.read_csv("results/comparisons/model_comparison_table.csv")
    df_top_feat = pd.read_csv("results/metrics/top_predictive_features.csv")
    df_who = pd.read_parquet("data/raw/who_when/who_and_when_hc.parquet")
    
    return {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "vocab": vocab,
        "lr_model": lr_model,
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "bilstm_model": bilstm_model,
        "fusion_model": fusion_model,
        "df_test": df_test,
        "df_comp": df_comp,
        "df_top_feat": df_top_feat,
        "df_who": df_who
    }

artifacts = load_all_artifacts()

# Session State Initialization
if "user" not in st.session_state:
    st.session_state["user"] = None
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "📊 Dashboard"

user = st.session_state["user"]

# If not authenticated, display login view
if not user:
    render_login_view()
    st.stop()

# ==============================================================================
# AUTHENTICATED SIDEBAR NAVIGATION (STRICT ROLE-BASED ISOLATION)
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff;">🛡️ Agent Intelligence</div>
        <div style="font-size: 0.8rem; color: #94a3b8;">Failure Diagnosis & Telemetry Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # User Profile Pill
    is_evaluator = (user.get("role") == "evaluator")
    role_badge = "badge-warning" if is_evaluator else "badge-healthy"
    role_text = "EVALUATOR" if is_evaluator else "STANDARD USER"
    
    st.markdown(f"""
    <div style="background-color: #1e293b; padding: 10px 14px; border-radius: 8px; margin-bottom: 20px;">
        <div style="color: #f8fafc; font-weight: 600; font-size: 0.95rem;">👤 {user['username']}</div>
        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 2px;">Role: <span class="{role_badge}">{role_text}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # User Navigation Options
    nav_options = [
        "📊 Dashboard",
        "🔍 Diagnose Agent",
        "⚡ Live Trace Diagnosis",
        "📋 Diagnostic History"
    ]
    
    # Evaluator Console is ONLY visible to Evaluator Role
    if is_evaluator:
        nav_options.append("🔬 Evaluator & Research Console")
        
    nav_options.append("👤 Account Profile")
    
    # Determine default index based on active_nav
    current_nav = st.session_state.get("active_nav", "📊 Dashboard")
    default_idx = nav_options.index(current_nav) if current_nav in nav_options else 0
    
    active_nav = st.radio("Navigation Menu:", nav_options, index=default_idx, key="sidebar_nav_radio", label_visibility="collapsed")
    st.session_state["active_nav"] = active_nav
    
    st.markdown("---")
    if st.button("🚪 Sign Out", key="btn_signout_sidebar", use_container_width=True):
        st.session_state["user"] = None
        st.session_state["active_nav"] = "📊 Dashboard"
        st.rerun()

# ==============================================================================
# VIEW DISPATCHER
# ==============================================================================
if active_nav == "📊 Dashboard":
    render_user_dashboard(user, artifacts)
    
elif active_nav == "🔍 Diagnose Agent":
    render_user_diagnose(user, artifacts)
    
elif active_nav == "⚡ Live Trace Diagnosis":
    render_user_live_demo(user, artifacts)
    
elif active_nav == "📋 Diagnostic History":
    render_user_history(user)
    
elif active_nav == "🔬 Evaluator & Research Console":
    # Extra safety check: non-evaluators blocked
    if not is_evaluator:
        st.error("Access Restricted: Evaluator Console is reserved for faculty/evaluator accounts.")
        st.info("Log in with an Evaluator account to access research artifacts.")
    else:
        render_evaluator_console(artifacts)
        
elif active_nav == "👤 Account Profile":
    render_user_profile()
