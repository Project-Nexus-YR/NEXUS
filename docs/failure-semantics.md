# Failure semantics

NEXUS provides **at-least-once task delivery**, not exactly once. A task remains owned
only until its visibility lease expires. If a worker crashes, stops heartbeating, times
out, or a scheduler restarts, active work transitions to retry or terminal failure
according to `RetryPolicy`; it never silently vanishes.

| Concern | Behavior |
| --- | --- |
| Duplicate delivery | Possible after failure; side-effect handlers use idempotency keys. |
| Successful duplicate key | A newly enqueued completed key is not delivered. |
| Lease expiration | The attempt is recorded, then work retries or fails. |
| Task/tool failure | The error is evented; work retries while attempts remain. |
| Bus fault | In-memory subscribers are dead-lettered; production adapters persist acknowledgements/DLQ. |
| Cancellation | Active work is marked cancelled; workers must cooperate and stop side effects. |
| Budget exhaustion | AgentRun pauses with a checkpoint and partial structured state. |

Durable state is separate from external side effects. The task/tool idempotency key is
the handler's consistency boundary, not a claim the scheduler can atomically commit an
external system.
