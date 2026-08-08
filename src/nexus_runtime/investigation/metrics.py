"""Scientific observability for autonomous investigation sessions."""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Protocol


class InvestigationMetrics(Protocol):
    def increment(self, name: str, amount: int = 1) -> None: ...

    def observe(self, name: str, value: float) -> None: ...


class InMemoryInvestigationMetrics:
    """Small deterministic sink suitable for adapters and test baselines."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._observations: dict[str, list[float]] = defaultdict(list)
        self._lock = RLock()

    def increment(self, name: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("metric increments cannot be negative")
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._observations[name].append(float(value))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            counters = dict(sorted(self._counters.items()))
            observations = {
                name: {
                    "count": len(values),
                    "sum": sum(values),
                    "average": sum(values) / len(values) if values else 0.0,
                    "minimum": min(values) if values else 0.0,
                    "maximum": max(values) if values else 0.0,
                }
                for name, values in sorted(self._observations.items())
            }
        executed = counters.get("investigations_executed", 0)
        succeeded = counters.get("investigation_successes", 0)
        gauges = {
            "investigation_success_rate": succeeded / executed if executed else 0.0,
        }
        return {"counters": counters, "observations": observations, "gauges": gauges}
