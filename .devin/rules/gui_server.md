# GUI/Server Rules — trust boundaries

## Entry points
- `forge_gui/` (client UI) · `forge_ui/` · `forge_gui_server/`
  (backend, binds 127.0.0.1, per-launch token `FORGE_GUI_TOKEN` carried
  as `X-Forge-Token` header / `?token=`) · `launch_gui.py` (root).
- `forge/engine/forge_server.py` — OpenAI-compatible API
  (127.0.0.1 default; `FORGE_API_KEY` gates ALL /v1 methods incl. GETs).

## Hard rules (security posture — full list in AGENTS.md)
- Loopback only, never `0.0.0.0`. Loopback-only CORS + WS Origin check.
- `forge_gui_server` token is a local-trust boundary — do not
  port-forward or serve on LAN.
- `run_python`/`run_tests` in `forge_gui/api/agent_tools.py` stay
  env-scrubbed + pytest flag allowlist.
- `lora_adapter`/`adapter_path` args confined to repo + FORGE_LORA_ROOTS.
- `tool_security.py` default blacklist (.env/.ssh/.git/sandbox.json) is
  a non-removable floor — PATCH can only add patterns.
- `library.py` entry_id is validated `[A-Za-z0-9_-]{1,128}` — keep it.
- torch.load `weights_only=False` sites remain for pickle-required
  formats — load only self-produced artifacts, all warn.
