"""Per-launch auth token for the ForgeAI GUI server.

The loopback API exposes engine load/unload, agent runs, file-backed
writes and task kills — previously reachable UNAUTHENTICATED from any
local process (drive-by was already blocked by loopback CORS + WS Origin
checks). The launcher mints a per-launch token (or honors an explicit
FORGE_GUI_TOKEN env var for scripted use), passes it in the window URL,
and the SPA lifts it into sessionStorage — it is never served back over
HTTP, so another local process cannot learn it by fetching /.

Enforcement: X-Forge-Token header or ?token= query on /api/* and /ws.
Static assets and the SPA shell stay open (they carry no token).
"""
from __future__ import annotations

import hmac
import os
import secrets

_TOKEN = os.environ.get("FORGE_GUI_TOKEN") or secrets.token_urlsafe(24)


def token() -> str:
    return _TOKEN


def verify(presented: str | None) -> bool:
    return bool(presented) and hmac.compare_digest(presented, _TOKEN)
