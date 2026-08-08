# ADR 0003: verify evidence before promoting knowledge

## Context

Investigation agents can return duplicated, weak, or conflicting claims. Agent
confidence alone cannot establish truth, and the investigation layer must not access
knowledge storage or recreate contradiction persistence.

## Decision

Track B uses structured evidence with complete execution and source lineage. It fuses
evidence deterministically, requires independent high-quality support, and preserves
unresolved conflicts. Eligible updates use the existing `KnowledgeUpdate` service
contract. New claims enter as unverified; the existing contradiction analyzer runs
before conflict-free claims pass through the existing uncertainty verifier.

## Consequences

Knowledge remains explainable from session through source chunk, and incomplete or
conflicted evidence cannot become verified merely because an agent reported high
confidence. The service's current contradiction scan requires candidate claims to be
committed before cross-batch conflicts can be discovered, so those claims remain
unverified and the explicit contradiction is returned. A future pure preflight method
on the knowledge service could reject them before commit without changing Track B's
domain contracts.
