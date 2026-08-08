# Event contract

Every event has an opaque event ID, event type, schema version, UTC timestamp, producer,
trace ID, correlation ID, optional causation ID, and structured payload. Stable event
namespaces are `agent`, `task`, `tool`, `worker`, `investigation`, `hypothesis`,
`experiment`, and `knowledge_update`.

The autonomous investigation application emits these stable lifecycle events:

- `investigation.session_created`
- `investigation.planning_started`
- `investigation.gaps_identified`
- `investigation.candidates_generated`
- `investigation.selected`
- `investigation.plan_created`
- `investigation.execution_started`
- `investigation.evidence_collected`
- `investigation.evaluated`
- `investigation.verification_started`
- `investigation.knowledge_updated`
- `investigation.iteration_completed`
- `investigation.completed`
- `investigation.failed`

Their payloads include the session, objective, and iteration. The session is the trace
identifier and the objective is the correlation identifier, preserving a stable chain
from planning through distributed execution and knowledge update.

`EventBus` is broker-independent: `publish`, `subscribe`, `acknowledge`, and
`dead_letter`. `InMemoryEventBus` is deterministic for tests. Production adapters must
define consumer-group, retention, authentication, ordering, acknowledgement, retry, and
dead-letter guarantees rather than leaking a broker API into the runtime domain.
