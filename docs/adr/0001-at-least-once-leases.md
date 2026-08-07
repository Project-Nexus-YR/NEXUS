# ADR 0001: lease-based at-least-once delivery

## Decision

Use capability-aware worker leases, visibility timeouts, retries, and caller supplied
idempotency keys. Do not claim exactly-once task execution.

## Consequences

A crash can result in duplicate handler invocation after a lease expires. This is safer
than silently discarding work and compatible with distributed workers. Every
side-effecting adapter uses the task/tool idempotency key at its transaction boundary.
Scheduler recovery after restart treats pre-restart worker ownership as untrusted.
