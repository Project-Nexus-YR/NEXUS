# Security

Tools declare input/output schemas, capability, permissions, timeout, side-effect
classification, and idempotency behavior. `ToolRegistry` asks `PolicyEngine` before
each invocation and returns `ALLOW`, `DENY`, or `REQUIRE_APPROVAL`.

This is an enforcement point, not a prompt convention. An LLM cannot grant itself
`filesystem.write`, `process.execute`, `knowledge.write`, or any other capability.
Deployments should use a policy adapter backed by the host identity provider and pass
idempotency keys to side-effecting tools. Secrets and environment files are ignored by
Git.
