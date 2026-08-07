# Runtime lifecycle

```text
goal -> gap discovery -> investigation -> hypothesis -> task DAG
     -> leased execution -> evidence -> experiment -> critique -> synthesis
     -> knowledge-update proposal -> verification by KnowledgeService
```

An AgentRun progresses `CREATED -> RUNNING -> WAITING|PAUSED|COMPLETED`; failures may
transition to `RETRYING` then `RUNNING`. Invalid transitions are rejected. Its loop
phases are individually recorded: observe, retrieve context, reason, choose action,
execute action, and update/checkpoint. A provider cannot replace this with an opaque
orchestration call.

Structured `HypothesisProposal` and `KnowledgeUpdateProposal` records reject missing
fields and invalid confidence. The runtime only proposes updates; KnowledgeService
verification is required before a commit.
