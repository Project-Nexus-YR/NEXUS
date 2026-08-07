# NEXUS Autonomous Research Runtime

NEXUS turns a research goal into a durable, inspectable investigation: planning,
dynamic task decomposition, distributed execution, experiments, critique, synthesis,
and a **proposal** for a knowledge update. It deliberately does not implement the
knowledge graph, retrieval, embeddings, entity extraction, or ranking subsystem.

## Status

This initial milestone provides an executable, provider-neutral runtime core:

- explicit agent, task, hypothesis, and experiment state machines;
- a dynamic, cycle-safe task DAG and priority scheduler with worker leases;
- at-least-once task delivery, idempotency-key deduplication, retries, timeouts,
  cancellation, backpressure, and failed-worker recovery;
- versioned domain events, in-memory event bus, durable SQLite checkpoint/event store,
  deterministic replay, capability policy checks, and structured outputs;
- interfaces for model, tools, memory, search, workflow, and knowledge services;
- mock-provider end-to-end and failure-injection tests.

The public service boundary is intentionally `KnowledgeService`; NEXUS does not read
the knowledge engine's database.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
make verify
```

Run the mock end-to-end test without optional development dependencies:

```bash
PYTHONPATH=src python3 -m unittest tests.test_end_to_end -v
```

## Guarantees

Task delivery is **at least once**. A completed idempotency key is not scheduled
again; task handlers that cause external side effects must use that key in their own
side-effect boundary. A worker owns a task only while its lease is valid. Expired or
dead-worker leases are requeued (or exhausted according to retry policy), so a task is
never silently discarded. See [failure semantics](docs/failure-semantics.md).

## Project layout

```text
src/nexus_runtime/  runtime domain and adapters
tests/              unit, integration, and fault-injection coverage
docs/               contracts, guarantees, and architecture decisions
```

## Development

`make format`, `make lint`, `make typecheck`, and `make test` require the optional
development dependencies. `make verify` runs only standard-library compilation and
the complete test suite, so the project remains runnable with no external runtime
dependencies.

See [architecture](docs/architecture.md), [runtime](docs/runtime.md), and
[API contract](docs/api-contract.md) for the integration boundary.
