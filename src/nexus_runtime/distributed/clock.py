"""Clock port used by every time-dependent distributed operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Protocol

from ..models import utcnow


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return utcnow()


@dataclass(slots=True)
class ManualClock:
    """Thread-safe controllable clock for deterministic simulations."""

    current: datetime
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def now(self) -> datetime:
        with self._lock:
            return self.current

    def advance(self, duration: timedelta) -> datetime:
        if duration < timedelta(0):
            raise ValueError("manual clock cannot move backwards")
        with self._lock:
            self.current += duration
            return self.current
