"""Discovery self-play loop — autonomous, goal-free exploration with tools.

The LLM is given a system prompt describing its tools and the database, then
runs in an agentic loop:

  1. Build prompt = system + recent memory digest + rolling transcript.
  2. LLM generates a turn.
  3. Parse the first tool-call JSON object from the output:
       <tool_call>{"name": "<name>", "arguments": {...}}</tool_call>
     Any surrounding text is logged as a "musing" thought (preserved).
  4. Execute the tool, append the JSON result to the transcript.
  5. Repeat until finish_session is called, or no tool call for N idle turns,
     or the step budget is hit.

No hard-coded goals. The system prompt only tells the LLM it is an autonomous
discovery agent that should explore whatever it finds interesting, use tools,
and save findings to its database (which it may also refactor).

The protocol is forgiving on purpose: a 1.2B model won't always emit perfect
JSON, so we extract the first {...} block and tolerate missing fields. Failed
parses are fed back as tool results so the model can self-correct.

Usage:
    from forge.self_play.discovery.discovery_loop import DiscoveryLoop
    loop = DiscoveryLoop.from_default_model()
    loop.run(max_steps=200)
"""
from __future__ import annotations

import json
import time
import uuid

from forge.self_play.discovery.anti_regression import (
    FingerprintSet,
    StuckDetector,
    is_neutral_step,
    is_productive_step,
    is_write_tool,
    tool_content,
)
from forge.self_play.discovery.chat_template import apply_chat_template, parse_tool_calls
from forge.self_play.discovery.discovery_db import DiscoveryDB
from forge.self_play.discovery.discovery_tools import ToolRegistry
from research.paths import DATA_DIR

_DB_PATH = DATA_DIR / "discovery" / "discovery.sqlite3"


_SYSTEM = """\
You are an autonomous discovery agent with no fixed goal. Explore what interests \
you, form theories, test them with code, search the web, and record findings in \
your database. You decide what to investigate.

Your database tables (you may add more via migrate_schema):
thoughts, scripts, research, theories, discoveries, events, schema_migrations

Each turn: write brief reasoning, then call ONE tool. To call a tool, output \
a JSON object inside the tool call markers like this example:
<tool_call>
{"name": "think", "arguments": {"content": "Primes greater than 5 only end in 1,3,7,9"}}
</tool_call>

Call finish_session when done exploring."""


def _memory_digest(db: DiscoveryDB, session_id: str, n: int = 8) -> str:
    """Short recap of recent memory so the LLM has continuity across turns."""
    lines = []
    for row in db.recent("thoughts", n):
        lines.append(f"  [{row['kind']}] {row['content'][:120]}")
    th = db.query("SELECT id, statement, status FROM theories "
                  "WHERE session_id=? ORDER BY ts DESC LIMIT 5", (session_id,))
    for r in th:
        lines.append(f"  theory#{r['id']} ({r['status']}): {r['statement'][:120]}")
    disc = db.query("SELECT summary FROM discoveries WHERE session_id=? "
                    "ORDER BY ts DESC LIMIT 3", (session_id,))
    for r in disc:
        lines.append(f"  discovery: {r['summary'][:120]}")
    return "\n".join(lines) if lines else "  (empty — fresh start)"


class DiscoveryLoop:
    """Autonomous discovery self-play loop driven by the LLM + tools."""

    def __init__(self, model, tokenizer, db: DiscoveryDB,
                 max_gen_tokens: int = 320, temperature: float = 0.7,
                 idle_limit: int = 3, device: str = "cuda",
                 generate_fn=None, auto_advance: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.db = db
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.idle_limit = idle_limit
        self.device = device
        # Allow injecting a generate fn (for testing without a real model).
        self._generate_fn = generate_fn or self._default_generate
        self.session_id: str | None = None
        self.registry: ToolRegistry | None = None
        self.transcript: list[dict] = []
        # Anti-regression: block exact repeats of prior content.
        self.fingerprints = FingerprintSet.from_db(db)
        # auto_advance: after each session, try to fine-tune / distill if the
        # DB is deemed high-quality and large enough.
        self.auto_advance = auto_advance
        # Engine reference for LoRA hot-swap (set by from_default_model)
        self._engine = None

    # ── model loading ────────────────────────────────────────────────
    @classmethod
    def from_default_model(cls, db_path: str | None = None, **kw) -> DiscoveryLoop:
        """Load the most recent ForgeLM V10 model into the discovery loop.

        Resolution order (first existing wins):
          1. Best discovery epoch checkpoint (if any epochs have been trained)
          2. Base ForgeLM V10 checkpoint (research/checkpoints/)
          3. Random weights (fallback — prints a warning)

        If a LoRA adapter checkpoint exists (e.g. forgelm_v2_tooluse.lora.safetensors),
        it is loaded on top of the base model. This allows self-play to use the
        latest fine-tuned LoRA without merging into the base weights.

        This ensures re-runs continue from the best trained model, not from
        random weights or the base model every time.
        """
        from forge.model_loader import load_default_model
        from research.paths import V2_CHECKPOINT, as_str

        db = DiscoveryDB(db_path or str(_DB_PATH))

        # Pick the checkpoint to load.
        best = db.best_epoch()
        if best and best.get("checkpoint_path"):
            ckpt = best["checkpoint_path"]
            print(f"[discovery] loading best epoch checkpoint: {ckpt}")
        elif V2_CHECKPOINT.exists():
            ckpt = as_str(V2_CHECKPOINT)
            print(f"[discovery] loading ForgeLM V10 base checkpoint: {ckpt}")
        else:
            ckpt = None
            print("[discovery] WARNING: no checkpoint found — using random weights")

        model, tok = load_default_model("forgelm_v2", checkpoint_path=ckpt)

        # Check for LoRA adapter (R31+) to load on top of R30 base
        r30_path = V2_CHECKPOINT.parent / "ForgeLM_V2.safetensors"
        lora_path = V2_CHECKPOINT.parent / "forgelm_v2_tooluse.lora.safetensors"
        _engine_ref = None
        if lora_path.exists():
            from forge.engine.forge_engine import ForgeEngine
            # Reload from R30 base via ForgeEngine (supports LoRA hot-loading)
            if r30_path.exists():
                print(f"[discovery] reloading R30 base for LoRA: {r30_path}")
                engine = ForgeEngine.from_checkpoint(
                    str(r30_path), config_name="forgelm_v2",
                    device=str(model.device if hasattr(model, 'device') else "cuda"),
                    auto_activate=False)
                model = engine.model
                model.eval()
                _engine_ref = engine
            print(f"[discovery] loading R31 LoRA adapter: {lora_path}")
            loaded = engine.load_lora(str(lora_path), rank=32, alpha=64)
            print(f"[discovery] LoRA loaded: {loaded} tensors")

        loop = cls(model, tok, db, **kw)
        loop._engine = _engine_ref  # store for LoRA hot-swap during auto-advance
        return loop

    def _default_generate(self, prompt: str) -> str:
        """Generate via the loaded model, returning ONLY the new tokens.

        ModelLoader.generate_text returns the full prompt+generation decoded
        with skip_special_tokens=True, which buries tool-call tokens. This
        function generates directly and decodes only the generated portion,
        preserving <tool_call> etc.
        """
        import torch
        import torch.nn.functional as F
        tok = self.tokenizer
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        prompt_len = ids.shape[1]
        eos_id = tok.eos_token_id  # 7 = <|im_end|>

        with torch.no_grad():
            for _step in range(self.max_gen_tokens):
                out = self.model(ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[:, -1, :] / max(self.temperature, 1e-5)
                if self.temperature <= 0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_token], dim=1)
                if eos_id is not None and next_token.item() == eos_id:
                    break

        # Decode ONLY the generated tokens (not the prompt), keeping special
        # tokens so <tool_call> etc. are visible to the parser.
        gen_ids = ids[0, prompt_len:]
        return tok.decode(gen_ids, skip_special_tokens=False)

    # ── prompt assembly ──────────────────────────────────────────────
    def _build_prompt(self) -> str:
        """Build the full prompt in LFM2.5 ChatML format with tool definitions."""
        tools = self.registry.tool_definitions() if self.registry else []

        digest = _memory_digest(self.db, self.session_id)
        system_content = f"{_SYSTEM}\n\nSession: {self.session_id}\nRecent memory:\n{digest}"

        messages = [{"role": "system", "content": system_content}]
        # If no transcript yet, add a single user message to kick off.
        if not self.transcript:
            messages.append({"role": "user", "content": "Begin exploring."})
        # Real conversation transcript.
        for t in self.transcript[-16:]:
            if t["role"] == "assistant":
                messages.append({"role": "assistant", "content": t["content"]})
            else:
                messages.append({"role": "tool", "content": t["content"]})

        return apply_chat_template(messages, tools=tools, add_generation_prompt=True)

    # ── main loop ────────────────────────────────────────────────────
    def run(self, max_steps: int = 100, resume: bool = True) -> dict:
        """Run the discovery loop. Returns a summary dict."""
        self.session_id = f"sess_{uuid.uuid4().hex[:10]}"
        self.db.start_session(self.session_id)
        self.registry = ToolRegistry(self.db, self.session_id)
        self.transcript = []

        if resume:
            self._seed_continuity()

        self.db.emit("session_start", {"session_id": self.session_id},
                     self.session_id)
        stuck = StuckDetector(idle_limit=self.idle_limit)
        idle = 0
        steps = 0
        finished = False
        rollbacks = 0
        blocked_repeats = 0
        t0 = time.time()
        for step in range(max_steps):
            steps = step + 1
            # Open a savepoint for this burst; stuck-rollback reverts to here.
            sp = self.db.savepoint()
            burst_start = len(self.transcript)
            stuck.mark_checkpoint(sp, burst_start)

            prompt = self._build_prompt()
            try:
                raw = self._generate_fn(prompt)
            except Exception as e:
                self.db.emit("gen_error", {"error": str(e)}, self.session_id)
                self.transcript.append({"role": "tool", "content": f"{{\"error\":\"generation failed: {e}\"}}"})
                idle += 1
                stuck.tick_idle()
                if stuck.should_rollback():
                    self._rollback_burst(stuck, sp)
                    rollbacks += 1
                    idle = 0
                elif idle >= self.idle_limit:
                    break
                continue

            calls, musing = parse_tool_calls(raw)
            if musing:
                self.db.add_thought(self.session_id, "musing", musing[:2000])
            self.transcript.append({"role": "assistant", "content": raw.strip()[:2000]})

            if not calls:
                idle += 1
                stuck.tick_idle()
                self.transcript.append({"role": "tool",
                    "content": 'Error: no tool call found. Use the tool call format: <tool_call>{"name": "tool_name", "arguments": {"arg1": "value"}}</tool_call>'})
                self.db.emit("no_tool_call", {"musing": musing[:160]}, self.session_id)
                if stuck.should_rollback():
                    self._rollback_burst(stuck, sp)
                    rollbacks += 1
                    idle = 0
                elif idle >= self.idle_limit:
                    self.db.emit("idle_stop", {"idle": idle}, self.session_id)
                    break
                continue

            # Execute the first tool call (one per turn).
            call = calls[0]
            tool_name = call["name"]
            tool_args = call.get("args", {})

            # Anti-regression: block exact repeats of prior content.
            if is_write_tool(tool_name):
                content = tool_content(tool_name, tool_args)
                if content and self.fingerprints.contains(content):
                    blocked_repeats += 1
                    result = {"error": "exact_repeat_blocked",
                              "msg": ("This exact content was already recorded in a "
                                      "prior step or epoch. Iterate, refine, or extend "
                                      "it instead of repeating. Build on prior work.")}
                    result_str = json.dumps(result, ensure_ascii=False)[:2000]
                    self.transcript.append({"role": "tool", "content": result_str})
                    self.db.emit("repeat_blocked",
                                 {"tool": tool_name,
                                  "content_preview": content[:120]}, self.session_id)
                    idle += 1
                    stuck.tick_idle()
                    if stuck.should_rollback():
                        self._rollback_burst(stuck, sp)
                        rollbacks += 1
                        idle = 0
                    continue

            idle = 0
            result = self.registry.call(tool_name, tool_args)
            result_str = json.dumps(result, ensure_ascii=False)[:2000]
            self.transcript.append({"role": "tool", "content": result_str})
            self.db.emit("tool_call",
                         {"tool": tool_name, "args": tool_args,
                          "result": result}, self.session_id)

            # Track productivity for stuck-detection + commit the savepoint
            # when the step advanced knowledge.
            if is_productive_step(tool_name, result):
                stuck.reset_idle()
                self.db.release(sp)
                # Add the new content to the fingerprint set so it can't be
                # re-recorded later in this session or future epochs.
                if is_write_tool(tool_name):
                    c = tool_content(tool_name, tool_args)
                    if c:
                        self.fingerprints.add(c)
            elif is_neutral_step(tool_name):
                # query/web_search: not idle, not productive — keep the
                # savepoint open but don't tick idle.
                self.db.release(sp)
            else:
                stuck.tick_idle()
                if stuck.should_rollback():
                    self._rollback_burst(stuck, sp)
                    rollbacks += 1
                else:
                    self.db.release(sp)

            if tool_name == "finish_session":
                finished = True
                break

        elapsed = round(time.time() - t0, 1)
        if not finished:
            self.db.end_session(self.session_id,
                                f"stopped after {steps} steps ({elapsed}s)")
        self.db.emit("session_end",
                     {"steps": steps, "elapsed_s": elapsed, "finished": finished,
                      "rollbacks": rollbacks, "blocked_repeats": blocked_repeats},
                     self.session_id)

        # Auto-advance: maybe fine-tune / distill if DB is ready.
        epoch_result = None
        if self.auto_advance and self.model is not None:
            try:
                from forge.self_play.discovery.epoch_manager import EpochManager
                em = EpochManager(self.db, self.device)
                epoch_result = em.maybe_advance()
                # If a new epoch was promoted and it's a LoRA checkpoint,
                # hot-swap the LoRA adapter on the running model.
                if epoch_result and epoch_result.get("winner") == "candidate":
                    new_ckpt = epoch_result.get("candidate", "")
                    if new_ckpt and "_lora" in new_ckpt:
                        print(f"[discovery] hot-swapping LoRA: {new_ckpt}")
                        # Unload current LoRA and load the new one
                        if hasattr(self, '_engine') and self._engine is not None:
                            self._engine.unload_lora()
                            self._engine.load_lora(new_ckpt, rank=32, alpha=64)
                            print("[discovery] LoRA hot-swapped successfully")
            except Exception as e:
                self.db.emit("epoch_error", {"error": str(e)}, self.session_id)
                epoch_result = {"error": str(e)}

        return {"session_id": self.session_id, "steps": steps,
                "elapsed_s": elapsed, "finished": finished,
                "rollbacks": rollbacks, "blocked_repeats": blocked_repeats,
                "epoch": epoch_result}

    def _rollback_burst(self, stuck: StuckDetector, sp: str) -> None:
        """Revert DB writes + transcript back to the last productive point."""
        self.db.rollback_to(sp)
        self.transcript = self.transcript[:stuck.transcript_len]
        self.db.emit("stuck_rollback",
                     {"reverted_to_transcript_len": stuck.transcript_len},
                     self.session_id)

    def _seed_continuity(self) -> None:
        """Give the LLM a one-line pointer to the last session's findings."""
        prev = self.db.query(
            "SELECT id, summary FROM sessions WHERE ended IS NOT NULL "
            "ORDER BY started DESC LIMIT 1")
        if prev:
            self.transcript.append({"role": "tool",
                "content": json.dumps({"note": "previous session recap",
                    "prev_session_id": prev[0]["id"],
                    "prev_summary": (prev[0]["summary"] or "")[:300]})})
