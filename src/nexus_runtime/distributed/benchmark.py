"""Reproducible local benchmark for comparing scheduler/runtime changes."""

from __future__ import annotations

import json
from dataclasses import asdict

from .simulator import DeterministicHarness, LocalDistributedSimulator


def run_benchmark(task_count: int = 1_000, worker_count: int = 10) -> dict[str, object]:
    simulator = LocalDistributedSimulator()
    harness = DeterministicHarness()
    for index in range(worker_count):
        simulator.add_worker(f"worker-{index}", frozenset({"agent.execute"}), harness)
    simulator.submit(task_count)
    report = simulator.run_until_terminal()
    return {
        "methodology": {
            "backend": "in-memory atomic task store",
            "clock": "deterministic manual UTC clock",
            "execution": "single-process coordinator plus cooperative workers",
            "task_count": task_count,
            "worker_count": worker_count,
        },
        "results": asdict(report),
    }


def main() -> int:
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
