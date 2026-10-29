# APPLICATION ARCHITECTURE SPECIFICATION
**Multimodal AI for Agentic System Failure Diagnosis and Root-Cause Reasoning**  
**Canonical Stitch UI Presentation Layer + Validated ML Service Architecture**  
*AUCML TE7483 — Symbiosis Institute of Technology (SIT), Pune*

---

## 1. System Architecture Overview

The system strictly decouples the visual presentation layer from the machine learning inference engine and persistent database layer:

```text
Stitch Canonical Web UI (HTML, Tailwind CSS, Space Grotesk, JetBrains Mono, Vanilla JS)
                                     ↓
                     Flask Application & REST API (src/web/app.py)
                                     ↓
                    Service Layer (DiagnosisService)
                   ↙                                 ↘
Tabular Telemetry Preprocessing (88 Dimensions)    Tool Sequence Tokenizer (Vocab: 69, L=50)
                   ↘                                 ↙
               Trained Models (LR, RF, XGBoost, BiLSTM, Multi-View Fusion)
                                     ↓
                   Calibrated Failure Probability & Risk Decision
                                     ↓
                 Evidence-Based Root-Cause Reasoning & Remediation
                                     ↓
                 User-Isolated SQLite Persistence (data/app_database.db)
```

No ML preprocessing or model scoring is duplicated inside client-side JavaScript. All diagnostic evaluations flow through `DiagnosisService.diagnose()`.

---

## 2. Presentation Layer Migration (Stitch Canonical Design System)

The visual interface is a direct translation of the canonical Stitch design artifacts (`DESIGN.md`, `code.html`, and reference screenshots):

### Design Tokens & Palette
- **Primary Workspace Canvas**: Warm ivory / cream (`#fef9f0`)
- **Top Masthead & Navigation Shell**: Deep espresso / mocha (`#342a25` and `#271e1a`)
- **Primary Action & Accents**: Terracotta / burnt orange (`#a23e18` and `#c85a32`)
- **Failure Marker & Badges**: Crimson (`#ba1a1a` / error container `#ffdad6`)
- **Warning & Secondary**: Warm amber / ochre (`#fcba5f` / `#fe8357`)
- **Nominal / Healthy Execution**: Muted forest green / sage (`#2f6f4e` / `#d4edda`)

### Typography
- **Headings & Badges**: Space Grotesk
- **Body & Form Inputs**: Plus Jakarta Sans
- **Telemetry, Code & Trace IDs**: JetBrains Mono

---

## 3. Real Authentication & Role-Based Access Control (RBAC)

The Stitch mockup's simulated client-side `setTimeout` routines were replaced with verified server-side endpoints backed by SQLite:

```text
Browser Client
      ↓
POST /api/auth/demo  {"role": "standard"} or {"role": "evaluator"}
      ↓
SQLite Lookup (data/app_database.db -> users table)
      ↓
Werkzeug pbkdf2:sha256 Hash Verification
      ↓
Signed Flask HttpOnly Cookie Session
      ↓
Server-Side Redirection:
  - role: 'user'      → /dashboard (Dashboard, Canvas, Telemetry, Forensics, History)
  - role: 'evaluator' → /evaluator (13-Section Research & Rubric Console)
```

### Route-Level Enforcement
- Standard users attempting to load `/evaluator` receive an HTTP 403 Forbidden page.
- Diagnostic history is strictly isolated via `WHERE user_id = ?`.

---

## 4. Primary Web Views

| Route | View Description | Source Template |
|---|---|---|
| `/login` | Canonical Stitch login with instant one-click demo cards + custom sign-in | `templates/login.html` |
| `/dashboard` | Executive Warm Canvas dashboard with 6-stage diagnostic pipeline flow & metrics | `templates/dashboard.html` |
| `/trace-canvas` | Canonical Interactive Agent Trace Canvas (DAG SVG + dual sparklines + Root-Cause Reasoning) | `templates/trace_canvas.html` |
| `/telemetry` | Temporal Telemetry & Derived Metrics (Formulas for 10 `DERIVED_` features) | `templates/telemetry.html` |
| `/forensics` | Reasoning & Forensics (Cross-modal gating, SHAP Gini importance, Who&When attribution) | `templates/forensics.html` |
| `/history` | User-isolated incident history with search, status filters, and modal inspections | `templates/history.html` |
| `/models` | 5-Model Comparative Benchmark Registry & evaluation metrics | `templates/model_registry.html` |
| `/settings` | Account details, cluster configuration, and decision threshold policy | `templates/settings.html` |
| `/evaluator` | 13-section academic research console (Rubric 30/30, ablation, viva defense) | `templates/evaluator.html` |

---

## 5. Port Configuration & Execution

- **ResolveX AI**: Active and untouched on port `8501`.
- **AUCML Stitch Application**: Active on port `8502`.

```powershell
# Launch AUCML Flask Web Platform:
python run.py

# Automated Test Suite:
python tests/test_flask_app.py
```
