# Architecture

NEXUS is a runtime domain with adapters around it. The domain owns investigation
records, agent-run state, task DAG validation, lease scheduling, policy decisions,
checkpoints, and replay. It does not own retrieval or a knowledge graph.

```text
Transport API -> Runtime domain -> EventBus / StateStore / workers
                      |              |
                      |              +-> Event broker adapter
                      +-> providers (model, memory, tools, workflow, search)
                      +-> KnowledgeService contract -> Knowledge Intelligence Engine
```

`Scheduler` exclusively owns task-state transitions; `AgentExecutor` owns AgentRun
transitions. Both emit events and can persist records through `StateStore`. SQLite is
the durable reference adapter; production deployments should supply a transactional
store and broker suited to the selected topology.

The Knowledge Intelligence Engine remains external. Its only integration surface is
`KnowledgeService` in `src/nexus_runtime/contracts.py`.
