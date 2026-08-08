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

Phase 4 adds a separate `InvestigationApplication` boundary. Its explicit stage
operations are `create`, `observe`, `generate`, `score`, `select`, `build_plan`,
`start_execution`, `advance_execution`, `collect_execution_results`,
`collect_evidence`, `evaluate`, `verify`, `update_knowledge`, and
`finish_iteration`. `status`, `pause`, `resume`, `cancel`, `explain`, and
`resume_iteration` provide lifecycle and recovery operations. The CLI delegates to
this boundary and contains no planning, execution, or verification policy.
