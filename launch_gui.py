"""Convenience launcher for the ForgeAI GUI.

Run from the repo root:
    python launch_gui.py                 # desktop window (pywebview)
    python launch_gui.py --browser       # open in the default browser
    python launch_gui.py --no-window     # serve only (headless API+UI)
    python launch_gui.py --dev           # API only; Vite dev on :5173
or:
    venv\\Scripts\\python.exe launch_gui.py

The web UI is built once from forge_ui/ (`npm run build`) and served by
the FastAPI backend in forge_gui_server/. Extra args are forwarded to
the server's CLI.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")  # engine prints use → / · (cp1252 breaks)

from forge_gui_server.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
