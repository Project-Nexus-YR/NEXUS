# ADR 0003: investigation planning is a deterministic orchestration boundary

## Status

Superseded in part: the legacy `TaskDAG` was removed during the architectural
consolidation. The decision that planning never creates or executes an agent loop,
and that `InvestigationPlan` retains explicit investigation identifiers and
dependency mappings for the integration application, remains in force. Dependency-
ready waves are submitted by `PlanExecutionController` through
`RuntimeApplication.submit_task`; the plan itself no longer compiles to
`DistributedTask` records.

## Context

NEXUS already has knowledge-gap detection and scoring, a durable Agent Harness, a
validated task DAG, and a fault-tolerant distributed runtime. The autonomous
investigation layer needs to decide what work is valuable without duplicating those
systems or coupling knowledge reasoning to worker internals.

## Decision

Track A uses explicit, serializable domain records from objective through plan. It
consumes existing `KnowledgeGap` objects and delegates the base gain/cost signal to the
existing knowledge scorer. Candidate generation, extended scoring, selection,
dependency validation, information-gain forecasting, and termination are
deterministic for deterministic inputs.

An `InvestigationPlan` retains explicit investigation identifiers and dependency
mappings. AgentRun IDs are supplied explicitly by the integration application;
planning never creates or executes an agent loop and never constructs
`DistributedTask` records directly. Because the current distributed task schema has
no dependency field, the integration layer submits ready DAG waves through the
public runtime API instead of changing coordinator internals.

## Consequences

Planning decisions are reproducible and explainable, while knowledge retrieval,
distributed ownership, retries, leases, and AgentRun execution retain their existing
owners. The integration layer must persist Track A records and coordinate DAG
readiness with runtime terminal states. A future public batch/DAG submission API can
replace wave submission without changing the plan domain contract.
