"""Event hub — fan-out pub/sub feeding the WebSocket stream.

Services publish events as ``{"type": ..., "payload": ...}`` dicts.
Each connected websocket gets its own queue; slow consumers are dropped
rather than allowed to backpressure publishers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_QUEUE_MAX = 512


class EventHub:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._seq = 0
        # ring buffer of recent events so a freshly-connected client can
        # catch up on current state (engine status, agent run, tasks)
        self._recent: list[dict] = []
        self._recent_max = 200
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the event loop — called once from the app lifespan."""
        self._loop = loop

    def publish(self, type_: str, payload: Any = None) -> dict:
        """Thread-safe: callable from the loop thread or worker threads."""
        with self._lock:
            self._seq += 1
            evt = {"seq": self._seq, "ts": time.time(), "type": type_,
                   "payload": payload}
            self._recent.append(evt)
            if len(self._recent) > self._recent_max:
                del self._recent[: len(self._recent) - self._recent_max]
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._fanout, evt)
        else:
            # No running loop → nobody can consume the queues anyway, and
            # asyncio.Queue is not thread-safe: fanning out here touched
            # subscriber deques from a foreign thread. The event stays in
            # _recent, so late subscribers still replay it.
            logger.debug("publish: no running loop; event kept for replay")
        return evt

    def _fanout(self, evt: dict) -> None:
        dead = []
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                dead.append(q)
        if dead:
            with self._lock:
                for q in dead:
                    self._subs.discard(q)
            logger.debug("dropped slow ws subscriber")

    def subscribe(self, replay: bool = True) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        with self._lock:
            if replay:
                for evt in self._recent:
                    try:
                        q.put_nowait(evt)
                    except asyncio.QueueFull:
                        break
            self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


def dumps(evt: dict) -> str:
    return json.dumps(evt, ensure_ascii=False, default=str)


hub = EventHub()
