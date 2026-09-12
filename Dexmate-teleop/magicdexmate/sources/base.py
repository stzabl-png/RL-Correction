from __future__ import annotations

import abc
import threading

from magicdexmate.skeleton import HandFrame


class LatestSlot:
    """Thread-safe single-value slot: producers overwrite, consumers read newest."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value: HandFrame | None = None

    def put(self, value: HandFrame) -> None:
        with self._lock:
            self._value = value

    def get(self) -> HandFrame | None:
        with self._lock:
            return self._value


class GloveSource(abc.ABC):
    """A source of HandFrame samples (real glove, mock, or replay)."""

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def get_latest(self) -> HandFrame | None:
        """Newest frame, or None if nothing received yet. Never blocks."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
