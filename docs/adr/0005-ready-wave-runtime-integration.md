# ADR 0005: Submit investigation DAGs in dependency-ready waves

## Status

Accepted for Phase 4. Superseded in part: plans no longer compile through the legacy
`TaskDAG` (removed in the architectural consolidation). `PlanExecutionController`
submits dependency-ready waves from the plan's `to_distributed_tasks` contract, and
the remainder of this decision is unchanged.

## Context

`InvestigationPlan` needs dependencies, while the Phase 3 `DistributedTask` contract
intentionally owns placement and delivery but has no dependency field. Adding a second
scheduler in the investigation layer or reaching into coordinator storage would break
the subsystem boundary.

## Decision

Plans compile through the existing `TaskDAG`. `PlanExecutionController` persists the
plan-to-AgentRun-to-distributed-task mapping and submits only nodes whose parent tasks
have succeeded. The normal `RuntimeApplication.submit_task` boundary remains the only
submission path. Capabilities, priorities, retries, worker assignment, leases,
cancellation, and recovery remain owned by the distributed runtime.

## Consequences

Independent nodes are queued together and remain parallelizable. Dependent nodes are
submitted in later waves. Permanent parent failure blocks descendants explicitly.
Session restart can reconstruct the mapping without depending on Python object identity.
