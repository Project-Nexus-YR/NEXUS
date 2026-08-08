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

The worker invokes the Agent Harness with an immutable `HarnessExecutionContext`:
`run_id`, `correlation_id`, `task_id`, `attempt_id`, `lease_id`, `worker_id`, and the
task metadata produced by the investigation plan. Harness implementations use this
public context to construct evidence lineage; they do not query coordinator storage or
infer the active attempt from in-process worker state.
