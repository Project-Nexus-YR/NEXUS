# Architecture

NEXUS is a runtime domain with adapters around it. The domain owns investigation
records, agent-run state, distributed task lifecycle, lease scheduling, policy
decisions, and checkpoints. It does not own retrieval or a knowledge graph.

```text
Transport API -> Runtime domain -> Coordinator / EventBus / TaskStore / workers
                      |                       |
                      |                       +-> Event broker adapter
                      +-> providers (model, memory, tools)
                      +-> Knowledge Intelligence Engine boundary
```

`Coordinator` exclusively owns distributed task-state transitions; `AgentExecutor`
owns AgentRun transitions. Both emit events and can persist records through durable
stores. SQLite is the durable reference adapter; production deployments should supply
a transactional store and broker suited to the selected topology.

The Knowledge Intelligence Engine remains external. `nexus_runtime` consumes it
through the knowledge-service boundary (retrieval, GraphRAG, gaps, contradictions,
verification) and returns knowledge-update proposals; the engine's database is never
read directly.
