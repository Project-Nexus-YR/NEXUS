# Event contract

Every event has an opaque event ID, event type, schema version, UTC timestamp, producer,
trace ID, correlation ID, optional causation ID, and structured payload. Stable event
namespaces are `agent`, `task`, `tool`, `worker`, `investigation`, `hypothesis`,
`experiment`, and `knowledge_update`.

`EventBus` is broker-independent: `publish`, `subscribe`, `acknowledge`, and
`dead_letter`. `InMemoryEventBus` is deterministic for tests. Production adapters must
define consumer-group, retention, authentication, ordering, acknowledgement, retry, and
dead-letter guarantees rather than leaking a broker API into the runtime domain.
