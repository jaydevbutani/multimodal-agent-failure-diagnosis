# Modern, polished styling for AUCML AI Product

def get_custom_css():
    return """
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Overall App Background */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }

    /* Sidebar Styling: Dark Navy / Charcoal */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        background-color: transparent;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #1e293b;
    }

    /* Product Cards */
    .product-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .product-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }

    /* Status Badges */
    .badge-healthy {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }

    .badge-warning {
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }

    .badge-failure {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 36px 32px;
        border-radius: 16px;
        margin-bottom: 28px;
        border: 1px solid #334155;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        color: #ffffff !important;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8 !important;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    .hero-stat {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.88rem;
        color: #cbd5e1;
    }

    /* Diagnosis Hero Card */
    .diag-hero-card {
        text-align: center;
        padding: 36px 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 2px solid transparent;
    }

    .diag-hero-healthy {
        background: #f0fdf4;
        border-color: #86efac;
        color: #14532d;
    }

    .diag-hero-warning {
        background: #fffbeb;
        border-color: #fcd34d;
        color: #78350f;
    }

    .diag-hero-failure {
        background: #fef2f2;
        border-color: #fca5a5;
        color: #7f1d1d;
    }

    .diag-status-text {
        font-size: 2rem;
        font-weight: 700;
        margin: 12px 0;
        letter-spacing: -0.01em;
    }

    .diag-prob-text {
        font-size: 1.25rem;
        font-weight: 600;
        opacity: 0.9;
    }

    /* Primary Accent Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Clean Metric Box */
    .clean-metric {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        text-align: left;
    }

    .clean-metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .clean-metric-value {
        font-size: 1.6rem;
        color: #0f172a;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Workflow Pipeline Step */
    .workflow-step {
        border-left: 3px solid #2563eb;
        padding-left: 18px;
        margin-bottom: 28px;
    }
    
    .workflow-step-num {
        font-size: 0.8rem;
        font-weight: 700;
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .workflow-step-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin: 4px 0 10px 0;
    }
    </style>
    """
