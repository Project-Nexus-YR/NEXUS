# Distributed task lifecycle

A distributed Task contains only scheduling data and a reference to an AgentRun:
`task_id`, `run_id`, correlation ID, priority, state, attempt count, worker/lease IDs,
timestamps, availability/deadline, retry policy, capability requirements, and validated
metadata. The AgentRun itself remains in the harness store.

```mermaid
stateDiagram-v2
    [*] --> QUEUED
    QUEUED --> CLAIMED: atomic claim
    CLAIMED --> RUNNING: valid lease starts
    RUNNING --> SUCCEEDED: valid completion wins
    CLAIMED --> FAILED: execution/start failure
    RUNNING --> FAILED: execution failure
    FAILED --> RETRY_WAIT: retryable and attempts remain
    RETRY_WAIT --> QUEUED: backoff elapsed
    FAILED --> DEAD_LETTERED: permanent or exhausted
    QUEUED --> CANCEL_REQUESTED
    CLAIMED --> CANCEL_REQUESTED
    RUNNING --> CANCEL_REQUESTED
    RETRY_WAIT --> CANCEL_REQUESTED
    CANCEL_REQUESTED --> CANCELLED: harness cancellation acknowledged
```

Lease expiry follows the same failure accounting path and either enters `RETRY_WAIT` or
`DEAD_LETTERED`. Transitions are centralized and invalid edges raise a domain error.
Each attempt has a distinct `attempt_id`; each claim has a distinct `lease_id`.

The dead-letter record retains the task, attempt history, workers, errors, timestamps,
last checkpoint reference, and last known state for inspection and explicit retry.
