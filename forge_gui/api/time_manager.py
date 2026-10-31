"""Time management tools — get_time, timers, alarms, and conditional flags.

Provides the agent with time awareness and scheduling:
- **get_time**: current time, date, timezone, uptime
- **set_timer**: countdown timer (fires after N seconds/minutes)
- **set_alarm**: absolute time alarm (fires at HH:MM)
- **check_timer**: check timer/alarm status
- **cancel_timer**: cancel a timer or alarm
- **list_timers**: list all active timers/alarms

Timer condition flags:
- `on_process_exit`: timer auto-cancels if a named process exits
- `on_user_prompt`: timer auto-cancels if user sends a new prompt
- `repeat`: timer repeats after firing

When a timer fires, it emits a signal. The agent loop can check fired
timers between rounds and inject the notification into the next round.
Timers do NOT block the agent — they run in the background via
``threading.Timer`` daemon threads.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as dtime

from ._signal import SimpleSignal

logger = logging.getLogger(__name__)


@dataclass
class TimerEntry:
    """A single timer or alarm entry."""
    timer_id: str
    kind: str = "timer"        # "timer" (countdown) or "alarm" (absolute)
    label: str = ""
    fire_at: float = 0.0       # unix timestamp when it should fire
    interval_s: float = 0.0    # for countdown timers
    repeat: bool = False
    on_process_exit: str = ""  # process name to watch; cancel timer if it exits
    on_user_prompt: bool = False  # cancel timer if user sends a prompt
    status: str = "active"     # active, fired, cancelled, expired
    fired_count: int = 0
    created_at: float = 0.0
    message: str = ""          # message to deliver when fired


class TimeManager:
    """Manages timers, alarms, and time queries for the agent.

    Signals (SimpleSignal):
        timer_fired(timer_id, label, message): a timer/alarm fired
        timer_cancelled(timer_id): a timer was cancelled
    """

    def __init__(self) -> None:
        self.timer_fired = SimpleSignal()
        self.timer_cancelled = SimpleSignal()
        self._timers: dict[str, TimerEntry] = {}
        self._thread_timers: dict[str, threading.Timer] = {}
        self._counter = 0
        self._start_time = time.time()

    @property
    def uptime_s(self) -> float:
        return time.time() - self._start_time

    def _next_id(self) -> str:
        self._counter += 1
        return f"timer_{self._counter:03d}"

    # ── get_time ──────────────────────────────────────────────────────
    def get_time(self) -> dict:
        """Get current time info."""
        now = datetime.now()
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": now.strftime("%A"),
            "timezone": time.tzname[0] if time.tzname else "unknown",
            "unix_timestamp": time.time(),
            "uptime_seconds": round(self.uptime_s, 1),
            "uptime_human": _human_duration(self.uptime_s),
        }

    # ── set_timer (countdown) ─────────────────────────────────────────
    def set_timer(self, seconds: float, label: str = "",
                  message: str = "", repeat: bool = False,
                  on_process_exit: str = "",
                  on_user_prompt: bool = False) -> str:
        """Set a countdown timer that fires after N seconds."""
        timer_id = self._next_id()
        fire_at = time.time() + seconds
        entry = TimerEntry(
            timer_id=timer_id, kind="timer", label=label or f"Timer {seconds:.0f}s",
            fire_at=fire_at, interval_s=seconds, repeat=repeat,
            on_process_exit=on_process_exit, on_user_prompt=on_user_prompt,
            created_at=time.time(), message=message or f"Timer '{label}' fired")
        self._timers[timer_id] = entry
        self._start_timer(timer_id, seconds)
        logger.info("Timer set: %s (%.0fs, label=%s)", timer_id, seconds, label)
        return timer_id

    # ── set_alarm (absolute time) ─────────────────────────────────────
    def set_alarm(self, time_str: str, label: str = "",
                  message: str = "", repeat: bool = False,
                  on_process_exit: str = "",
                  on_user_prompt: bool = False) -> str:
        """Set an alarm for a specific time (HH:MM format, 24h or 12h with AM/PM).

        If the time has already passed today, it fires tomorrow.
        With repeat=True, it fires every day at that time.
        """
        target = _parse_time(time_str)
        if target is None:
            return ""
        now = datetime.now()
        fire_dt = now.replace(hour=target.hour, minute=target.minute,
                              second=0, microsecond=0)
        if fire_dt <= now:
            fire_dt += timedelta(days=1)
        delay_s = (fire_dt - now).total_seconds()
        timer_id = self._next_id()
        entry = TimerEntry(
            timer_id=timer_id, kind="alarm", label=label or f"Alarm {time_str}",
            fire_at=fire_dt.timestamp(), interval_s=delay_s, repeat=repeat,
            on_process_exit=on_process_exit, on_user_prompt=on_user_prompt,
            created_at=time.time(),
            message=message or f"Alarm '{label}' fired at {time_str}")
        self._timers[timer_id] = entry
        self._start_timer(timer_id, delay_s)
        logger.info("Alarm set: %s (%s, label=%s)", timer_id, time_str, label)
        return timer_id

    # ── timer management ──────────────────────────────────────────────
    def _start_timer(self, timer_id: str, seconds: float) -> None:
        t = threading.Timer(max(seconds, 0.001), self._on_fire,
                            args=(timer_id,))
        t.daemon = True
        t.start()
        self._thread_timers[timer_id] = t

    def _on_fire(self, timer_id: str) -> None:
        entry = self._timers.get(timer_id)
        if entry is None or entry.status != "active":
            return
        entry.fired_count += 1
        entry.status = "fired"
        self.timer_fired.emit(entry.timer_id, entry.label, entry.message)
        logger.info("Timer fired: %s (%s)", timer_id, entry.label)
        if entry.repeat:
            entry.status = "active"
            entry.fire_at = time.time() + entry.interval_s
            # schedule the next fire — threading.Timer is single-shot
            self._start_timer(timer_id, entry.interval_s)

    def check_timer(self, timer_id: str) -> dict | None:
        """Check the status of a timer."""
        entry = self._timers.get(timer_id)
        if entry is None:
            return None
        remaining = max(0, entry.fire_at - time.time()) if entry.status == "active" else 0
        return {
            "timer_id": entry.timer_id, "kind": entry.kind,
            "label": entry.label, "status": entry.status,
            "fired_count": entry.fired_count,
            "remaining_seconds": round(remaining, 1),
            "fire_at": datetime.fromtimestamp(entry.fire_at).strftime(
                "%Y-%m-%d %H:%M:%S") if entry.fire_at else "",
            "repeat": entry.repeat,
            "on_process_exit": entry.on_process_exit,
            "on_user_prompt": entry.on_user_prompt,
        }

    def cancel_timer(self, timer_id: str) -> bool:
        """Cancel a timer or alarm."""
        entry = self._timers.get(timer_id)
        if entry is None:
            return False
        entry.status = "cancelled"
        t = self._thread_timers.pop(timer_id, None)
        if t:
            t.cancel()
        self.timer_cancelled.emit(timer_id)
        logger.info("Timer cancelled: %s", timer_id)
        return True

    def list_timers(self) -> list[dict]:
        """List all timers/alarms with their status."""
        return [self.check_timer(tid) for tid in self._timers
                if self.check_timer(tid) is not None]

    # ── condition flags ───────────────────────────────────────────────
    def check_conditions(self, active_processes: list[str] | None = None,
                         user_prompted: bool = False) -> list[str]:
        """Check timer condition flags and cancel timers whose conditions are met.

        Args:
            active_processes: list of running process names (for on_process_exit)
            user_prompted: True if the user just sent a new prompt

        Returns list of cancelled timer_ids.
        """
        cancelled = []
        for tid, entry in list(self._timers.items()):
            if entry.status != "active":
                continue
            # on_process_exit: cancel if the named process is no longer running
            if entry.on_process_exit and active_processes is not None:
                if entry.on_process_exit not in active_processes:
                    self.cancel_timer(tid)
                    cancelled.append(tid)
                    continue
            # on_user_prompt: cancel if user sent a new prompt
            if entry.on_user_prompt and user_prompted:
                self.cancel_timer(tid)
                cancelled.append(tid)
                continue
        return cancelled

    def get_fired_timers(self) -> list[dict]:
        """Get all timers that have fired (and clear their fired status)."""
        fired = []
        for _tid, entry in list(self._timers.items()):
            if entry.status == "fired" and not entry.repeat:
                fired.append({
                    "timer_id": entry.timer_id, "label": entry.label,
                    "message": entry.message, "kind": entry.kind})
                entry.status = "expired"
        return fired

    def clear_expired(self) -> int:
        """Remove expired/cancelled timers. Returns count removed."""
        to_remove = [tid for tid, e in self._timers.items()
                     if e.status in ("expired", "cancelled")]
        for tid in to_remove:
            t = self._thread_timers.pop(tid, None)
            if t:
                t.cancel()
            del self._timers[tid]
        return len(to_remove)

    def shutdown(self) -> None:
        """Stop all timers."""
        for t in self._thread_timers.values():
            t.cancel()
        self._thread_timers.clear()
        self._timers.clear()


# ── helper functions ────────────────────────────────────────────────────

def _parse_time(time_str: str) -> dtime | None:
    """Parse a time string in HH:MM or HH:MM AM/PM format."""
    import re
    s = time_str.strip().upper()
    # 12h format: "5:00 PM", "5 PM", "5:30AM"
    m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3)
        if ampm == "PM" and hour < 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
        if 0 <= hour < 24 and 0 <= minute < 60:
            return dtime(hour, minute)
    # 24h format: "17:00", "5:30"
    m = re.match(r"(\d{1,2}):(\d{2})$", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return dtime(hour, minute)
    return None


def _human_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


# ── tool definitions ────────────────────────────────────────────────────

def time_tool_defs() -> list[dict]:
    """Tool definitions for time management."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": (
                    "Get the current time, date, weekday, timezone, and "
                    "system uptime. Use when the user asks about time or "
                    "when you need to schedule timers/alarms."),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": (
                    "Set a countdown timer that fires after N seconds. "
                    "The timer runs in the background and does not block. "
                    "When it fires, the notification is injected into the "
                    "next agent round. Supports repeat and condition flags."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "number",
                            "description": "Seconds until the timer fires.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Human-readable name for the timer.",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message to deliver when timer fires.",
                        },
                        "repeat": {
                            "type": "boolean",
                            "description": "If true, timer repeats after firing.",
                        },
                        "on_process_exit": {
                            "type": "string",
                            "description": (
                                "Process name to watch. Timer auto-cancels "
                                "if this process exits."),
                        },
                        "on_user_prompt": {
                            "type": "boolean",
                            "description": (
                                "If true, timer auto-cancels when the user "
                                "sends a new prompt."),
                        },
                    },
                    "required": ["seconds"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_alarm",
                "description": (
                    "Set an alarm for a specific time (HH:MM or HH:MM AM/PM). "
                    "If the time has passed today, fires tomorrow. "
                    "Supports repeat (daily) and condition flags."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "Target time, e.g. '5:00 PM' or '17:00'.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Human-readable name for the alarm.",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message to deliver when alarm fires.",
                        },
                        "repeat": {
                            "type": "boolean",
                            "description": "If true, alarm fires every day at this time.",
                        },
                        "on_process_exit": {
                            "type": "string",
                            "description": "Process name to watch for auto-cancel.",
                        },
                        "on_user_prompt": {
                            "type": "boolean",
                            "description": "Auto-cancel if user sends a new prompt.",
                        },
                    },
                    "required": ["time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_timer",
                "description": (
                    "Check the status of a timer or alarm. Returns status "
                    "(active/fired/cancelled), remaining time, and fired count."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timer_id": {
                            "type": "string",
                            "description": "Timer ID from set_timer or set_alarm.",
                        },
                    },
                    "required": ["timer_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_timer",
                "description": "Cancel an active timer or alarm.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timer_id": {
                            "type": "string",
                            "description": "Timer ID to cancel.",
                        },
                    },
                    "required": ["timer_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_timers",
                "description": (
                    "List all active and recently fired timers/alarms with "
                    "their status and remaining time."),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
