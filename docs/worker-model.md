# Worker model

Workers move through `STARTING`, `READY`, `BUSY`, `DRAINING`, `UNHEALTHY`, and
`STOPPED`. Registration accepts a `WorkerIdentity` issued by the deployment boundary;
workers cannot add capabilities after identity issuance.

Each record tracks version, start/heartbeat time, declared capacity, active task IDs,
and status. Available slots are `max_concurrency - current_concurrency`. Saturated,
draining, unhealthy, and stopped workers receive no new tasks.

The worker loop is register → heartbeat → claim → start → harness execute/resume →
complete/fail → repeat. The harness port accepts only `run_id`, correlation context,
and a cancellation probe. The worker never handles model providers, prompts, tools,
context compaction, or knowledge-engine internals.

Draining stops new claims. Existing tasks finish or checkpoint/cancel through the
harness, after which the worker transitions to stopped.
