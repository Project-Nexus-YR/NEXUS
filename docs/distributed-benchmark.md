# Distributed runtime benchmark

The reference benchmark submits 1,000 independent `agent.execute` tasks to ten
cooperative workers. It uses the real Coordinator, Worker, priority scheduler,
TaskQueue, harness port, and atomic in-memory TaskStore with a deterministic UTC clock.
Only the harness implementation is deterministic and immediate.

Run it with:

```bash
nexus-runtime-bench
```

Reference run on 8 August 2026 using CPython 3.12 on macOS:

| Metric | Result |
| --- | ---: |
| Tasks | 1,000 |
| Workers | 10 |
| Scheduling cycles | 100 |
| Successful | 1,000 |
| Dead-lettered | 0 |
| Retries | 0 |
| Logical worker utilization | 100% |
| Wall-clock throughput | 126.40 tasks/s |

Wall-clock throughput depends on machine and Python version; correctness counts,
scheduling cycles, and logical utilization are deterministic. Queue wait and execution
latency are zero logical seconds because the manual clock advances only when work is
waiting for a future retry. Future changes should compare reports in the same
environment rather than treating this number as a service-level objective.
