"""Minimal signal shim for the de-Qt'd domain layer.

The forge_gui api package no longer depends on PySide6 — GUI-facing
signals are replaced by per-instance SimpleSignal objects with the same
connect/emit surface the old code used.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class SimpleSignal:
    """Per-instance signal: connect(fn) subscribes, emit(*args) calls all."""

    def __init__(self) -> None:
        self._subs: list[Callable] = []

    def connect(self, fn: Callable) -> None:
        self._subs.append(fn)

    def disconnect(self, fn: Callable) -> None:
        with contextlib.suppress(ValueError):
            self._subs.remove(fn)

    def emit(self, *args) -> None:
        for fn in list(self._subs):
            try:
                fn(*args)
            except Exception:
                logger.debug("signal subscriber failed", exc_info=True)
