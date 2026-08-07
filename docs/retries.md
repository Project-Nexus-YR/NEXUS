# Retry and dead-letter semantics

Failures are classified as `TRANSIENT`, `PERMANENT`, `CANCELLED`,
`BUDGET_EXHAUSTED`, or `POLICY_VIOLATION`. Only transient failures are retried
automatically. Permanent, budget, and policy failures are dead-lettered; cancellation
finishes as cancelled.

Retry policy contains maximum attempts, initial/max backoff, multiplier, and optional
deterministic jitter. Backoff is computed from the completed attempt number and the
stable task ID, making tests reproducible while avoiding synchronized retries.

An operator may explicitly retry a dead-lettered task. That action preserves prior
attempt history, clears current ownership, sets a new availability time, and emits a
retry event. It does not erase diagnostic evidence.
