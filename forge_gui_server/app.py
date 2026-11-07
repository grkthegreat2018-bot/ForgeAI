"""ForgeAI GUI backend — FastAPI app: REST API + WebSocket event stream
+ static SPA hosting for forge_ui/dist.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import auth
from .deps import services
from .hub import dumps
from .routes import chat, core, system, train

logger = logging.getLogger(__name__)

_UI_DIST = Path(__file__).resolve().parents[1] / "forge_ui" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await services.start()
    yield
    await services.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="ForgeAI GUI", version="1.0.0",
                  lifespan=lifespan)
    app.add_middleware(
        # Loopback origins only — wildcard CORS on a localhost API lets any
        # web page drive-by call /api/* (incl. agent tool execution) and
        # read the responses through the victim's browser.
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"])

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):
        # Every /api route mutates or reads real engine/workspace state —
        # gate on the per-launch token (header preferred; ?token= covers
        # WS-upgrade and bare-link clients that can't set headers).
        if request.url.path.startswith("/api"):
            tok = (request.headers.get("x-forge-token")
                   or request.query_params.get("token"))
            if not auth.verify(tok):
                return JSONResponse(
                    {"detail": "missing or invalid GUI token"},
                    status_code=401)
        return await call_next(request)

    app.include_router(core.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(train.router, prefix="/api")
    app.include_router(system.router, prefix="/api")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        # CSWSH guard: browsers send Origin on WS handshakes — reject
        # anything that isn't a loopback UI (missing Origin = non-browser
        # client like the desktop webview, which is fine).
        import re as _re
        origin = websocket.headers.get("origin")
        if origin and not _re.fullmatch(
                r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?", origin):
            await websocket.close(code=1008)
            return
        tok = (websocket.headers.get("x-forge-token")
               or websocket.query_params.get("token"))
        if not auth.verify(tok):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        q = services.hub.subscribe(replay=True)
        try:
            while True:
                evt = await q.get()
                await websocket.send_text(dumps(evt))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            services.hub.unsubscribe(q)

    # ── SPA static hosting ────────────────────────────────────────────
    if _UI_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=_UI_DIST / "assets"),
                  name="assets")

        @app.get("/{full_path:path}")
        async def spa(full_path: str):
            # SPA fallback — every non-API route serves index.html
            if full_path.startswith(("api/", "ws")):
                return {"error": "not found"}
            index = _UI_DIST / "index.html"
            return FileResponse(index)

    return app


app = create_app()
