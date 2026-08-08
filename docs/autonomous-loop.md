# Closed Epistemic Loop

The investigation engine advances one explicit stage at a time. There is no
`while True` research loop and no prompt that substitutes for control flow.

```mermaid
stateDiagram-v2
    [*] --> PLANNING
    PLANNING --> EXECUTING: selected plan
    EXECUTING --> EVALUATING: complete or accepted partial results
    EVALUATING --> UPDATING: verification report
    UPDATING --> PLANNING: continue after progress measurement
    PLANNING --> COMPLETED: no valuable work / objective satisfied
    UPDATING --> COMPLETED: stopping criterion
    PLANNING --> PAUSED
    EXECUTING --> PAUSED
    EVALUATING --> PAUSED
    UPDATING --> PAUSED
    PAUSED --> PLANNING
    PAUSED --> EXECUTING
    PAUSED --> EVALUATING
    PAUSED --> UPDATING
    PLANNING --> CANCELLED
    EXECUTING --> CANCELLED
    EVALUATING --> FAILED
    UPDATING --> FAILED
```

## One iteration

```mermaid
flowchart TD
    O["Observe retrieval, GraphRAG, gaps, contradictions"] --> G["Generate grounded candidates"]
    G --> S["Score gain, importance, uncertainty, availability, cost, time, risk, redundancy"]
    S --> P["Budget/capacity selection and TaskDAG plan"]
    P --> R["Submit ready AgentRuns to distributed runtime"]
    R --> E["Collect complete and partial evidence with lineage"]
    E --> F["Fuse duplicates, support, contradictions, and low-quality evidence"]
    F --> V["Verify provenance, independent sources, quality, confidence, conflict policy"]
    V --> U["Commit eligible updates through KnowledgeEngine"]
    U --> N["Observe new knowledge and calculate progress"]
    N --> T{"Termination policy"}
    T -->|"continue"| O
    T -->|"bounded stop"| X["Completed / Cancelled / Failed"]
```

## Bounded termination

Every session has maximum iterations, investigations, AgentRuns, cost, and execution
time. The deterministic termination policy handles objective satisfaction, confidence,
budget exhaustion, maximum iterations, no valuable candidate, unresolvable conflict,
user cancellation, and system failure in a documented precedence order.

An uncertain result is valid. Verification preserves `confirmed`, `probable`,
`uncertain`, `contradicted`, and `insufficient_evidence`; only policy-eligible claims
reach the knowledge-update boundary.

## Partial results and failure

Distributed retry and lease recovery stay in the coordinator. The investigation layer
sees the eventual task state and can evaluate completed or partial
`InvestigationResult` values while other tasks remain running. Failed dependencies are
recorded as blocked investigations. They are not silently counted as success.

Collected results are persisted before evaluation. After restart,
`resume_collected_iteration` reloads the evidence, continues evaluation and update, and
does not invoke the Agent Harness again. At-least-once distributed execution can still
duplicate delivery; evidence fingerprints and stable result lineage make fusion
idempotent and inspectable.

`resume_iteration` also reconstructs sessions interrupted during `EVALUATING` or
`UPDATING`. The update proposal is persisted before calling the knowledge service. If
the process fails after that call but before the receipt is stored, recovery may submit
the same proposal again. Claim and evidence identifiers remain stable, so repository
adapters must provide idempotent save-by-ID behavior. This is an explicit at-least-once
knowledge-update boundary; the application does not claim a distributed transaction
across its SQLite record and an external knowledge store.

## Progress

`ProgressMeasurer` compares before/after gap and contradiction snapshots and reports:

- gaps resolved, remaining, and newly discovered;
- uncertainty reduced;
- contradictions introduced and resolved;
- evidence and updates;
- information gain;
- cost per resolved gap.

This report is the input to the next stopping decision and is persisted with the full
iteration history.
