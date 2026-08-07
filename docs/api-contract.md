# API contract

The application service is transport independent. HTTP, gRPC, queue consumers, or CLI
adapters call these operations without exposing domain internals:

- `create_run`, `get_run`, `cancel_run`
- `get_tasks`, `get_events`
- `create_agent`, `list_agents`
- `create_investigation`, `get_investigation`

The initial `RuntimeAPI` exposes them in-process. Adapters propagate trace/correlation
IDs, authenticate callers, authorize cancellation and creation, and translate
`DomainError` to their transport's validation response.
