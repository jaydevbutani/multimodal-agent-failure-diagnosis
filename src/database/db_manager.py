import os
import sqlite3
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join("data", "app_database.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Users Table (with case-insensitive uniqueness and full_name)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL COLLATE NOCASE,
        email TEXT UNIQUE NOT NULL COLLATE NOCASE,
        full_name TEXT DEFAULT '',
        password_hash TEXT NOT NULL,
        salt TEXT DEFAULT '',
        role TEXT NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Check if full_name column exists for backwards compatibility
    cursor.execute("PRAGMA table_info(users);")
    columns = [col[1] for col in cursor.fetchall()]
    if "full_name" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT DEFAULT '';")
    
    # 2. Diagnostic Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diagnostic_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_title TEXT NOT NULL,
        session_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # 3. Diagnostic Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diagnostic_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        agent_name TEXT NOT NULL,
        model_used TEXT NOT NULL,
        threshold REAL NOT NULL,
        failure_probability REAL NOT NULL,
        predicted_status TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        detected_issue TEXT NOT NULL,
        root_cause TEXT NOT NULL,
        recommended_action TEXT NOT NULL,
        telemetry_summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES diagnostic_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    
    # Ensure standard demo operator account exists
    def ensure_operator_account(uname, mail, pwd, full_name, role="user"):
        cursor.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?);", (uname, mail))
        row = cursor.fetchone()
        hashed = generate_password_hash(pwd, method="pbkdf2:sha256")
        if not row:
            cursor.execute(
                "INSERT INTO users (username, email, full_name, password_hash, salt, role) VALUES (?, ?, ?, ?, ?, ?);",
                (uname, mail, full_name, hashed, "", role)
            )
        else:
            cursor.execute(
                "UPDATE users SET email = ?, full_name = ?, password_hash = ?, role = ? WHERE id = ?;",
                (mail, full_name, hashed, role, row[0])
            )
        conn.commit()

    # The canonical single demo operator account
    ensure_operator_account("demo", "demo@agenthealth.ai", "Demo@123", "Demo Operator", "user")
    # Also support demo_user / demo@aucml.ai for backward compat smoothly
    ensure_operator_account("demo_user", "demo@aucml.ai", "DemoUser@123", "Demo User", "user")
    
    conn.close()

def register_user(full_name: str, email: str, password: str, role: str = "user"):
    full_name = (full_name or "").strip()
    email = (email or "").strip().lower()
    password = (password or "").strip()
    
    if not full_name:
        return None, "Full Name is required."
    if "@" not in email or "." not in email:
        return None, "Please enter a valid email address."
    if len(password) < 6:
        return None, "Password must be at least 6 characters long."
        
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE LOWER(email) = LOWER(?);", (email,))
    if cursor.fetchone() is not None:
        conn.close()
        return None, "An account with this email address already exists. Please sign in."
        
    # Derive a base username from the email prefix
    base_username = email.split("@")[0].replace(".", "_")
    username = base_username
    counter = 1
    while True:
        cursor.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(?);", (username,))
        if cursor.fetchone() is None:
            break
        username = f"{base_username}_{counter}"
        counter += 1
        
    pwd_hash = generate_password_hash(password, method="pbkdf2:sha256")
    try:
        cursor.execute(
            "INSERT INTO users (username, email, full_name, password_hash, salt, role) VALUES (?, ?, ?, ?, ?, ?);",
            (username, email, full_name, pwd_hash, "", role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {
            "id": user_id,
            "username": username,
            "full_name": full_name,
            "email": email,
            "role": role
        }, None
    except Exception as e:
        conn.close()
        return None, f"Registration failed: {str(e)}"

def authenticate_user(username_or_email: str, password: str):
    term = (username_or_email or "").strip().lower()
    password = (password or "").strip()
    
    if not term or not password:
        return None, "Email and password are required."
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, username, email, full_name, password_hash, role FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?);",
        (term, term)
    )
    user_row = cursor.fetchone()
    conn.close()
    
    if not user_row:
        return None, "Invalid email or password."
        
    stored_hash = user_row["password_hash"]
    
    # Check with werkzeug check_password_hash
    if check_password_hash(stored_hash, password):
        return {
            "id": user_row["id"],
            "username": user_row["username"],
            "full_name": user_row["full_name"] or user_row["username"],
            "email": user_row["email"],
            "role": user_row["role"]
        }, None
    else:
        return None, "Invalid email or password."

def save_diagnostic_record(user_id: int, session_title: str, session_type: str,
                           agent_name: str, model_used: str, threshold: float,
                           failure_probability: float, predicted_status: str,
                           risk_level: str, detected_issue: str, root_cause: str,
                           recommended_action: str, telemetry_summary: dict = None):
    conn = get_connection()
    cursor = conn.cursor()
    
    def _json_default(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            try:
                return obj.item()
            except Exception:
                pass
        return str(obj)
        
    tel_str = json.dumps(telemetry_summary, default=_json_default) if telemetry_summary else "{}"
    
    # Enforce maximum 10 saved records per user (Requirement 10)
    cursor.execute("""
        SELECT r.id, s.id FROM diagnostic_results r
        JOIN diagnostic_sessions s ON r.session_id = s.id
        WHERE r.user_id = ?
        ORDER BY r.created_at DESC;
    """, (user_id,))
    existing_records = cursor.fetchall()
    if len(existing_records) >= 10:
        # Keep 9 most recent so adding the new record keeps exactly 10
        records_to_prune = existing_records[9:]
        for old_r_id, old_s_id in records_to_prune:
            cursor.execute("DELETE FROM diagnostic_results WHERE id = ?;", (old_r_id,))
            cursor.execute("DELETE FROM diagnostic_sessions WHERE id = ?;", (old_s_id,))

    cursor.execute(
        "INSERT INTO diagnostic_sessions (user_id, session_title, session_type) VALUES (?, ?, ?);",
        (user_id, session_title, session_type)
    )
    session_id = cursor.lastrowid
    
    cursor.execute("""
    INSERT INTO diagnostic_results (
        session_id, user_id, agent_name, model_used, threshold,
        failure_probability, predicted_status, risk_level,
        detected_issue, root_cause, recommended_action, telemetry_summary
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        session_id, user_id, agent_name, model_used, threshold,
        failure_probability, predicted_status, risk_level,
        detected_issue, root_cause, recommended_action, tel_str
    ))
    
    result_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return result_id

def get_user_diagnostic_history(user_id: int):
    """
    Returns only the authenticated user's diagnostic history.
    Strict data isolation: user_id = ?
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT 
        r.id as result_id,
        s.id as session_id,
        s.session_title,
        s.session_type,
        r.agent_name,
        r.model_used,
        r.threshold,
        r.failure_probability,
        r.predicted_status,
        r.risk_level,
        r.detected_issue,
        r.root_cause,
        r.recommended_action,
        r.telemetry_summary,
        r.created_at
    FROM diagnostic_results r
    JOIN diagnostic_sessions s ON r.session_id = s.id
    WHERE r.user_id = ?
    ORDER BY r.created_at DESC
    LIMIT 10;
    """, (user_id,))
    
    rows = cursor.fetchall()
    history = [dict(r) for r in rows]
    conn.close()
    return history

def get_user_diagnostic_detail(user_id: int, result_id: int):
    """
    Enforces authorization check: User A cannot retrieve User B's result.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT 
        r.id as result_id,
        s.id as session_id,
        s.session_title,
        s.session_type,
        r.agent_name,
        r.model_used,
        r.threshold,
        r.failure_probability,
        r.predicted_status,
        r.risk_level,
        r.detected_issue,
        r.root_cause,
        r.recommended_action,
        r.telemetry_summary,
        r.created_at
    FROM diagnostic_results r
    JOIN diagnostic_sessions s ON r.session_id = s.id
    WHERE r.id = ? AND r.user_id = ?;
    """, (result_id, user_id))
    
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
