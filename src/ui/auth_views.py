import streamlit as st
from src.database.db_manager import authenticate_user, register_user

def render_login_view():
    st.markdown("""
    <div style="max-width: 520px; margin: 20px auto 10px auto; text-align: center;">
        <h1 style="color: #0f172a; font-weight: 700; margin-bottom: 6px; font-size: 2.1rem;">🛡️ Agent Health Intelligence</h1>
        <p style="color: #64748b; font-size: 0.98rem; margin-bottom: 20px;">
            Multimodal Failure Diagnosis & Root-Cause Attribution Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        # ======================================================================
        # PROMINENT ONE-CLICK DEMO LOGIN CARD (REQUIRED BY USER)
        # ======================================================================
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 22px; color: white;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                ⚡ ONE-CLICK DEMO ACCESS
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 14px;">
                Instant Role-Based Examination Login
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_demo1, c_demo2 = st.columns(2)
        with c_demo1:
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 8px;">
                <div style="font-size: 1.2rem;">👤</div>
                <div style="font-weight: 700; color: #0f172a; font-size: 0.9rem;">Standard User</div>
                <div style="font-size: 0.78rem; color: #64748b;">demo_user</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("LOGIN AS DEMO USER", key="btn_login_demo_user", type="primary", use_container_width=True):
                user, err = authenticate_user("demo_user", "DemoUser@123")
                if user:
                    st.session_state["user"] = user
                    st.session_state["active_nav"] = "📊 Dashboard"
                    st.success("Authenticated as demo_user (Standard User). Loading Dashboard...")
                    st.rerun()
                else:
                    st.error(f"Demo authentication error: {err}")
                    
        with c_demo2:
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 8px;">
                <div style="font-size: 1.2rem;">🔬</div>
                <div style="font-weight: 700; color: #0f172a; font-size: 0.9rem;">Faculty Evaluator</div>
                <div style="font-size: 0.78rem; color: #64748b;">evaluator</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("LOGIN AS EVALUATOR", key="btn_login_evaluator", type="secondary", use_container_width=True):
                user, err = authenticate_user("evaluator", "Evaluator@123")
                if user:
                    st.session_state["user"] = user
                    st.session_state["active_nav"] = "🔬 Evaluator & Research Console"
                    st.success("Authenticated as evaluator (Faculty Role). Loading Research Console...")
                    st.rerun()
                else:
                    st.error(f"Evaluator authentication error: {err}")

        st.markdown("<div style='text-align: center; margin: 18px 0 14px 0; color: #94a3b8; font-size: 0.85rem; font-weight: 600;'>— OR SIGN IN WITH CUSTOM ACCOUNT —</div>", unsafe_allow_html=True)

        # Standard Authentication Tabs
        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Create Account"])
        
        with tab_login:
            login_id = st.text_input("Username or Email", key="login_id_input")
            login_pwd = st.text_input("Password", type="password", key="login_pwd_input")
            
            if st.button("Sign In with Custom Account", key="btn_custom_signin", use_container_width=True):
                if not login_id or not login_pwd:
                    st.error("Please enter both username/email and password.")
                else:
                    user, err = authenticate_user(login_id, login_pwd)
                    if err:
                        st.error(f"Authentication Failed: {err}")
                    else:
                        st.session_state["user"] = user
                        st.session_state["active_nav"] = "🔬 Evaluator & Research Console" if user.get("role") == "evaluator" else "📊 Dashboard"
                        st.success(f"Welcome back, {user['username']}!")
                        st.rerun()
                        
        with tab_register:
            reg_username = st.text_input("Username", key="reg_username_input")
            reg_email = st.text_input("Email Address", key="reg_email_input")
            reg_pwd = st.text_input("Password (min 6 characters)", type="password", key="reg_pwd_input")
            reg_role = st.selectbox(
                "Select Role",
                ["user", "evaluator"],
                key="reg_role_select",
                format_func=lambda x: "Standard AI Engineer (User Application)" if x == "user" else "Faculty / Evaluator (Research Console Access)"
            )
            
            if st.button("Register & Sign In", key="btn_register_submit", use_container_width=True):
                user, err = register_user(reg_username, reg_email, reg_pwd, reg_role)
                if err:
                    st.error(err)
                else:
                    st.session_state["user"] = user
                    st.session_state["active_nav"] = "🔬 Evaluator & Research Console" if reg_role == "evaluator" else "📊 Dashboard"
                    st.success("Account created successfully! Loading your session...")
                    st.rerun()

def render_user_profile():
    user = st.session_state.get("user")
    if not user:
        return
        
    st.markdown("### 👤 User Account & Profile")
    role_class = "badge-healthy" if user["role"] == "user" else "badge-warning"
    role_display = "Standard User (Application Mode)" if user["role"] == "user" else "Faculty Evaluator (Research Console Access)"
    
    st.markdown(f"""
    <div class="product-card">
        <h4 style="margin: 0 0 10px 0;">{user['username']}</h4>
        <p style="margin-bottom: 6px;"><strong>Email:</strong> {user['email']}</p>
        <p style="margin-bottom: 6px;"><strong>Access Role:</strong> <span class="{role_class}">{role_display}</span></p>
        <p style="margin-bottom: 0;"><strong>Database User ID:</strong> #{user['id']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Sign Out of Platform", key="btn_signout_profile", type="secondary"):
        st.session_state["user"] = None
        st.session_state["active_nav"] = "📊 Dashboard"
        st.rerun()
