# Distributed execution semantics

| Property | Initial guarantee |
| --- | --- |
| Delivery | At least once |
| Atomic claim | One successful claim per task version in one TaskStore |
| Completion | First valid current-lease completion wins |
| Ordering | Priority FIFO with aging; no global event order |
| Retry | Transient only, bounded exponential backoff and optional deterministic jitter |
| Cancellation | Durable request; cooperative for active harness execution |
| Coordinator recovery | Durable tasks/attempts restored; workers re-register |
| Backpressure | Bounded non-terminal task count and per-worker capacity |
| Traceability | Correlation ID propagates Task → Worker → Harness → emitted events |

There is no exactly-once guarantee, cross-service transaction, consensus, distributed
lock service, or promise that a timed-out external side effect stopped. Production
adapters must provide isolation equivalent to the atomic TaskStore operations and must
preserve task, attempt, lease, and correlation identifiers.
