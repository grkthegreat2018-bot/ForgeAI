"""ForgeAI GUI launcher.

    python -m forge_gui_server            # serve + desktop window
    python -m forge_gui_server --browser  # serve + open in browser
    python -m forge_gui_server --no-window# serve only (headless)
    python -m forge_gui_server --dev      # API only; Vite dev on :5173

The web UI is served from forge_ui/dist (run `npm run build` in
forge_ui/ first) or, with --dev, the Vite dev server proxies /api + /ws
here.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser

import uvicorn

DEFAULT_PORT = 8741


def _serve(port: int, ready: threading.Event) -> None:
    config = uvicorn.Config(
        "forge_gui_server.app:app", host="127.0.0.1", port=port,
        log_level="warning", ws="auto")
    server = uvicorn.Server(config)
    # signal readiness once the server reports started
    orig = server.startup

    async def startup(*a, **kw):
        await orig(*a, **kw)
        ready.set()
    server.startup = startup  # type: ignore[assignment]
    server.run()


def main() -> int:
    # Opt-in runtime configuration (import of `forge` is side-effect-free).
    from forge.runtime.configure import configure
    configure()

    ap = argparse.ArgumentParser(prog="forge_gui_server")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--browser", action="store_true",
                    help="open in the default browser instead of a "
                         "desktop window")
    ap.add_argument("--no-window", action="store_true",
                    help="serve only, don't open any UI")
    ap.add_argument("--dev", action="store_true",
                    help="API only — use the Vite dev server for the UI")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from forge_gui_server.auth import token as _gui_token
    # The per-launch token rides the launch URL (?token=) — the SPA lifts
    # it into sessionStorage and sends X-Forge-Token / ?token= afterwards.
    url = f"http://{args.host}:{args.port}/?token={_gui_token()}"

    if args.dev or args.no_window:
        # foreground serve
        config = uvicorn.Config(
            "forge_gui_server.app:app", host=args.host, port=args.port,
            log_level="info", ws="auto")
        if args.dev:
            print(f"API on http://{args.host}:{args.port} — Vite dev "
                  "server expected on :5173 — open "
                  f"http://localhost:5173/?token={_gui_token()}")
        else:
            print(f"ForgeAI GUI on {url}")
        uvicorn.Server(config).run()
        return 0

    # background serve + window
    ready = threading.Event()
    t = threading.Thread(target=_serve, args=(args.port, ready),
                         daemon=True)
    t.start()
    if not ready.wait(timeout=30):
        print("server failed to start", file=sys.stderr)
        return 1

    if args.browser:
        webbrowser.open(url)
        print(f"ForgeAI GUI on {url} — Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    try:
        import webview
        webview.create_window(
            "ForgeAI", url, width=1480, height=920,
            min_size=(1080, 680), background_color="#0b0e14")
        webview.start()
        return 0
    except Exception as e:
        print(f"desktop window unavailable ({e}); opening browser")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
