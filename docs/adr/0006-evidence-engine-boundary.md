# ADR 0006: Citation verification is a bounded context over existing runtime boundaries

- Status: accepted
- Date: 2026-08-13

## Context

NEXUS already has knowledge-domain evidence and investigation-runtime evidence.
The former supports uncertainty and graph updates; the latter proves complete
session → task → attempt → tool → source → document → chunk lineage. Final prose
also needs claim-level citation coverage, exact locators, numerical validation,
and repair. Folding all three meanings into one mutable `Evidence` record would
couple unrelated lifecycles and weaken invariants.

## Decision

Create `nexus_evidence` as a bounded context with explicit adapters:

- consume retrieval through a new public `KnowledgeEngine.retrieve_evidence`
  operation;
- compile distributed work to existing `DistributedTask` records and the standard
  Agent Harness;
- persist checkpoints as existing investigation artifacts;
- expose an optional citation-verification port on `InvestigationApplication`;
- keep deterministic numerical and citation policy in the evidence domain;
- treat external content as untrusted at every adapter boundary.

The evidence layer may reference stable source/document/chunk identifiers, but it
must not access knowledge repositories through the runtime or create another
scheduler/state machine.

## Consequences

Citation workflows inherit runtime durability and failure semantics. Existing
knowledge and investigation APIs remain compatible. The model contains an
intentional bridge—shared stable provenance identifiers—without aliasing their
different evidence aggregates. Distributed stage handlers remain responsible for
idempotent result persistence under at-least-once delivery.
