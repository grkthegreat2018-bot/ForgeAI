"""LoRA training trigger — allows the agent to kick off LoRA training.

Provides a tool definition and execution wrapper that the unified harness
dispatches. The agent can request training of a new LoRA adapter for a
specific skill category, using chat history or a dataset as training data.

The training runs as a subprocess via ProcessManager (same as the FineTune
page), so it doesn't block the agent loop. The agent can poll the training
status using the `check_training` tool.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def lora_training_tool_defs() -> list[dict]:
    """Tool definitions for LoRA training triggers."""
    return [
        {
            "type": "function",
            "function": {
                "name": "train_lora",
                "description": (
                    "Launch a LoRA training run to create a new skill "
                    "adapter. Training runs in the background. Use this "
                    "when you identify a skill gap and want to improve "
                    "the model for a specific category (coding, math, "
                    "reasoning, etc.). Training data can come from chat "
                    "history (good-rated turns) or a JSONL file."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": (
                                "Skill category for the adapter (coding, "
                                "math, reasoning, tool_use, agentic, "
                                "chat_assist, self_play, vision)"),
                            "enum": ["coding", "math", "reasoning",
                                     "tool_use", "agentic", "chat_assist",
                                     "self_play", "vision"],
                        },
                        "rank": {
                            "type": "integer",
                            "description": "LoRA rank (default 32)",
                        },
                        "data_source": {
                            "type": "string",
                            "enum": ["chat_history", "file"],
                            "description": (
                                "Source of training data: 'chat_history' "
                                "uses good-rated chat turns, 'file' uses "
                                "a JSONL file path"),
                        },
                        "data_file": {
                            "type": "string",
                            "description": (
                                "Path to JSONL training data (required "
                                "if data_source='file')"),
                        },
                        "max_steps": {
                            "type": "integer",
                            "description": (
                                "Training optimizer steps (default 500). "
                                "sft_train is step-budgeted — it has no "
                                "epoch flag."),
                        },
                        "lr": {
                            "type": "number",
                            "description": "Learning rate (default 5e-5)",
                        },
                    },
                    "required": ["category"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_training",
                "description": (
                    "Check the status of a LoRA training run. Returns "
                    "progress, current epoch, loss, and whether it's "
                    "still running."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID from train_lora response",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        },
    ]


class LoraTrainingTrigger:
    """Executes LoRA training tool calls via ProcessManager.

    Args:
        proc_mgr: ProcessManager instance for launching subprocesses.
        chat_store: ChatStore for extracting good-rated turns.
        checkpoint: Base checkpoint path for training.
    """

    def __init__(self, proc_mgr=None, chat_store=None,
                 checkpoint: str = "") -> None:
        self.proc_mgr = proc_mgr
        self.chat_store = chat_store
        self.checkpoint = checkpoint
        self._training_tasks: dict[str, dict] = {}  # task_id → metadata

    def execute(self, name: str, args: dict) -> dict:
        """Dispatch a training tool call."""
        if name == "train_lora":
            return self._train_lora(args)
        elif name == "check_training":
            return self._check_training(args)
        return {"ok": False, "result": {"error": f"unknown training tool: {name}"}}

    def _train_lora(self, args: dict) -> dict:
        """Launch a LoRA training run."""
        category = args.get("category", "")
        if not category:
            return {"ok": False, "result": {"error": "category required"}}

        rank = args.get("rank", 32)
        # sft_train budgets optimizer steps, not epochs.
        max_steps = int(args.get("max_steps", 500) or 500)
        max_steps = max(1, min(max_steps, 200000))
        lr = args.get("lr", 5e-5)
        data_source = args.get("data_source", "chat_history")

        # prepare training data
        if data_source == "chat_history":
            if self.chat_store is None:
                return {"ok": False, "result": {"error": "no chat_store available"}}
            data_path = self._export_chat_data(category)
            if data_path is None:
                return {"ok": False, "result": {
                    "error": "no good-rated chat turns found for training"}}
        else:
            data_file = args.get("data_file", "")
            if not data_file:
                return {"ok": False, "result": {
                    "error": "data_file required when data_source='file'"}}
            data_path = Path(data_file)
            if not data_path.is_file():
                return {"ok": False, "result": {
                    "error": f"data file not found: {data_file}"}}

        # build output adapter name — sft_train snapshots the un-merged
        # LoRA tensors to Path(--save).with_suffix(".lora.safetensors"),
        # which the LoRA scan picks up under research/checkpoints/.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        adapter_name = f"ForgeLM_V11_{category}_R{rank}_{timestamp}"
        save_path = (Path("research/checkpoints/lora")
                     / f"{adapter_name}.safetensors")
        adapter_path = save_path.with_suffix(".lora.safetensors")

        # build training command — real sft_train flags:
        # --lora-r / --lora-alpha / --max-steps / --save; there is no
        # --epochs / --lora-rank / --output (the old argv TypeError'd or
        # silently mis-routed the adapter path).
        from forge_gui.api.status_reader import project_root
        root = project_root()
        venv_py = root / "venv" / "Scripts" / "python.exe"
        cmd = [
            str(venv_py if venv_py.is_file() else "python"),
            "-u", "-m", "forge.training.runners.sft_train",
            "--checkpoint", self.checkpoint,
            "--data", str(data_path),
            "--save", str(save_path),
            "--save-lora-adapter",
            "--lora-r", str(rank),
            "--lora-alpha", str(rank * 2),
            "--max-steps", str(max_steps),
            "--lr", str(lr),
        ]

        # launch via ProcessManager
        if self.proc_mgr is None:
            return {"ok": False, "result": {
                "error": "no ProcessManager available to launch training"}}

        task_id = self.proc_mgr.launch(f"LoRA Training ({category})", cmd)
        self._training_tasks[task_id] = {
            "category": category,
            "rank": rank,
            "adapter_name": adapter_name,
            "output_dir": str(adapter_path),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_path": str(data_path),
        }

        return {"ok": True, "result": {
            "task_id": task_id,
            "category": category,
            "adapter_name": adapter_name,
            "rank": rank,
            "max_steps": max_steps,
            "adapter_path": str(adapter_path),
            "message": (
                f"LoRA training launched for '{category}' category. "
                f"Adapter will be saved as {adapter_path}. "
                f"Use check_training(task_id='{task_id}') to monitor progress.")}}

    def _check_training(self, args: dict) -> dict:
        """Check training status."""
        task_id = args.get("task_id", "")
        if task_id not in self._training_tasks:
            return {"ok": False, "result": {"error": f"unknown task_id: {task_id}"}}

        meta = self._training_tasks[task_id]
        # check if process is still running — ProcessService exposes
        # all_tasks() (there is no processes() method; the old call
        # AttributeError'd every status check).
        running = False
        if self.proc_mgr is not None:
            tasks = self.proc_mgr.all_tasks()
            running = any(t.get("id") == task_id and t.get("is_live")
                          for t in tasks)

        return {"ok": True, "result": {
            "task_id": task_id,
            "running": running,
            "category": meta["category"],
            "adapter_name": meta["adapter_name"],
            "started": meta["started"],
            "output_dir": meta["output_dir"],
            "message": (
                "Training in progress..." if running else
                "Training completed (or stopped). Check output directory.")}}

    def _export_chat_data(self, category: str) -> Path | None:
        """Export good-rated chat turns as JSONL for training."""
        if self.chat_store is None:
            return None
        try:
            export_path = Path("data/sft") / f"lora_{category}_{int(time.time())}.jsonl"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            # export_training_data(conv_ids, out_path, ...) returns
            # (path, n_examples) — the old call passed the path positionally
            # as conv_ids and compared the tuple to 0.
            _path, n = self.chat_store.export_training_data(
                out_path=export_path)
            if n == 0:
                return None
            return export_path
        except Exception as e:
            logger.warning("chat data export failed: %s", e)
            return None

    @property
    def training_tasks(self) -> dict[str, dict]:
        return self._training_tasks
