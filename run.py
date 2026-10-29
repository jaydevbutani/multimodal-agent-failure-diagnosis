import os
import sys

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.web.app import app, init_db

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 8502))
    print("==================================================================")
    print(f"AUCML Agent Health Intelligence Platform")
    print(f"Canonical Stitch UI Frontend + Validated ML Diagnostics Pipeline")
    print(f"Serving at: http://localhost:{port}")
    print(f"ResolveX AI (port 8501) untouched.")
    print("==================================================================")
    app.run(host="0.0.0.0", port=port, debug=False)
