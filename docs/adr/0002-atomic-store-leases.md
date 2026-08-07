# ADR 0002: atomic store operations and expiring leases

## Context

The durable agent harness protects execution progress, but it cannot decide distributed
ownership. JSON snapshots are suitable for local harness state and unsafe as a
multi-process claim primitive.

## Decision

Distributed tasks use explicit atomic TaskStore operations. The local durable adapter
uses SQLite `BEGIN IMMEDIATE`; the simulator uses an `RLock`. Claims create unique
attempt and lease identities. Ownership expires and delivery is at least once.

## Consequences

The coordinator can restart and workers can fail without losing tasks. Duplicate
execution remains possible around ACK loss, so side effects require idempotency. SQLite
is intentionally a single-node adapter; a future transactional database may replace it
without changing coordinator or worker semantics.
