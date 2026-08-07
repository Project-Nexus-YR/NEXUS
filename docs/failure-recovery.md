# Failure recovery

## Worker failure

After heartbeat loss, the worker becomes unhealthy. Its active task leases expire;
recovery records the failed attempt and either schedules retry or dead-letters it. A
replacement worker receives a new attempt/lease and asks the harness to resume the same
AgentRun checkpoint.

## ACK loss and duplicate delivery

If execution finishes but completion is not committed, the lease eventually expires.
A later attempt can repeat the harness call. The old worker's late ACK is rejected
because its lease is stale. The first completion committed with the current valid lease
wins. External actions must be idempotent.

## Coordinator restart

SQLite persists task state, lease, attempt history, and cancellation request. A new
coordinator uses the same TaskStore, advances due retries, expires stale leases, and
continues scheduling. Workers re-register; registry loss is safe because task ownership
is validated from durable lease data.

## Cancellation

Cancellation is a durable task state, not an event-only signal. Queued/retrying work can
cancel immediately. Active work becomes `CANCEL_REQUESTED`; the worker observes it,
calls `harness.cancel_run(run_id)`, and acknowledges cancellation. If that worker is
gone, recovery preserves the request and completes cancellation without re-execution.
