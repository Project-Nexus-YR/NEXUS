# Distributed execution

Workers register capabilities and a concurrency limit, heartbeat, acquire compatible
priority leases, execute, checkpoint through the runtime, and acknowledge completion.
Draining workers receive no new work. A lease is capped by both scheduler visibility
duration and the task timeout.

Tasks form a dynamic DAG. New tasks may be added during execution, but every dependency
must exist and the graph is validated for cycles. Independent ready tasks can lease to
different workers concurrently. The scheduler enforces backpressure, worker
concurrency, timeouts, retry policy, cancellation, and failed-dependency cancellation.
On restart it reloads snapshots and deliberately reclaims former worker leases.
