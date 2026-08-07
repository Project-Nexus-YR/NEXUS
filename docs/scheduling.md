# Scheduling and backpressure

The initial policy is capability-aware priority FIFO with aging. A task is eligible
when it is queued, its `available_at` has passed, its deadline has not passed, and its
required capabilities are a subset of the worker identity. Among eligible tasks the
highest effective priority wins; equal scores use creation time and task ID.

Effective priority increases by one aging step for each configured aging interval in
the queue. This bounds low-priority starvation while retaining HIGH/NORMAL/LOW intent.
The scheduler is a stateless policy object and can later be replaced by locality-,
cost-, or load-aware policies.

Backpressure is enforced at submission by a configured bound on non-terminal tasks.
Worker capacity is enforced both by WorkerRegistry and assignment accounting. Queue
statistics expose queued, claimed, running, retrying, succeeded, failed, cancelled,
dead-lettered, and total depth.
