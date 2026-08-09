# NEXUS

NEXUS is an autonomous knowledge-discovery platform. It pairs a **knowledge
intelligence engine** (`nexus_knowledge`) with a **fault-tolerant autonomous
research runtime** (`nexus_runtime`). The engine ingests heterogeneous sources
into a knowledge graph and exposes hybrid retrieval, GraphRAG evidence
extraction, and uncertainty/contradiction/gap analysis. The runtime turns a
research goal into a durable, inspectable investigation — planning, task
decomposition, distributed execution, critique, synthesis — and proposes
knowledge updates through the engine's knowledge-service boundary.

## Components

### Nexus Knowledge (`nexus_knowledge`)

Ingests heterogeneous sources into a knowledge graph, exposes hybrid retrieval,
GraphRAG evidence extraction, and uncertainty/contradiction/gap analysis with
deterministic, reproducible evaluation.

**Install**

Requires Python 3.11+ and NumPy.

```bash
pip install -e ".[test]"
```

**Quick start**

The CLI wires up all default adapters (gazetteer + relation-pattern entity
extraction, recursive chunking, lexical/vector/entity/graph retrieval, local
embedding provider).

```bash
# ingest a text file or directory
nexus-knowledge ingest data/report.md --kind markdown --title "report"

# hybrid retrieval for a query
nexus-knowledge retrieve "What does Acme Corp develop?" --top-k 10

# GraphRAG: evidence graph for a query
nexus-knowledge graphrag "Ada Lovelace at Acme Corp"

# knowledge gaps and scored candidate investigations
nexus-knowledge gaps
nexus-knowledge score --top-k 20

# graph statistics
nexus-knowledge stats

# deterministic benchmarks (report is reproducible; --output writes JSON)
nexus-knowledge bench --output bench-report.json
```

**Library API**

```python
from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.service.factory import Adapters, create_engine

engine = create_engine(Adapters())
engine.ingest(Source(title="report", kind=SourceKind.TEXT, reference="r"), "Ada works at Acme.")
result = engine.retrieve("Ada Acme")
evidence = engine.graphrag("Ada Acme")
gaps = engine.find_knowledge_gaps()
scored = engine.score_investigation(top_k=20)
```

### Nexus Runtime (`nexus_runtime`)

Provider-neutral, fault-tolerant autonomous research runtime. It deliberately
does not implement the knowledge graph, retrieval, embeddings, entity
extraction, or ranking subsystem — it consumes the engine via the
knowledge-service boundary and returns proposals for knowledge updates.

**Status**

The Phase 3 milestone provides a provider-neutral distributed execution layer:

- explicit agent and distributed task state machines;
- a dynamic, cycle-safe distributed task queue with priority aging and worker
  leases;
- at-least-once task delivery, idempotency-key deduplication, retries, timeouts,
  cancellation, backpressure, and failed-worker recovery;
- versioned domain events, in-memory event bus, durable SQLite checkpoint/event store,
  capability policy checks, and structured outputs;
- interfaces for model, tools, and memory providers;
- mock-provider end-to-end and failure-injection tests.
- atomic in-memory and SQLite TaskStore adapters, worker identities and capacity,
  priority aging, durable cancellation, dead letters, metrics, and coordinator restart;
- a deterministic local multi-worker simulator using the same Coordinator, Worker,
  scheduler, queue, and harness interfaces as production adapters.

The public service boundary is intentionally the knowledge engine; NEXUS does not read
the knowledge engine's database.

### Autonomous investigation (Phase 4)

Phase 4 closes the epistemic loop: structured objectives are observed against current
knowledge, measurable gaps become scored and budget-aware investigation plans, those
plans execute through the distributed Agent Harness, and provenance-complete evidence
is fused, verified, and applied through the knowledge service. Progress is measured
before another bounded iteration is considered.

```bash
nexus-research create "What is Acme's verified status?" \
  --criterion "two independent sources support the answer"
nexus-research status SESSION_ID
nexus-research explain SESSION_ID
nexus-investigation-bench
```

See [autonomous investigation](docs/autonomous-investigation.md) and the
[closed epistemic loop](docs/autonomous-loop.md).

**Quick start**

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
make verify
```

Run the production-path investigation test without optional development dependencies:

```bash
PYTHONPATH=src python3 -m unittest tests.test_investigation_application -v
```

Submit and inspect durable distributed tasks:

```bash
nexus-runtime --db .nexus/runtime.sqlite submit run-123 \
  --correlation-id investigation-123 --capability agent.execute
nexus-runtime --db .nexus/runtime.sqlite queue
nexus-runtime --db .nexus/runtime.sqlite task TASK_ID
```

Run the 1,000-task/10-worker local benchmark:

```bash
nexus-runtime-bench
```

**Guarantees**

Task delivery is **at least once**. A completed idempotency key is not scheduled
again; task handlers that cause external side effects must use that key in their own
side-effect boundary. A worker owns a task only while its lease is valid. Expired or
dead-worker leases are requeued (or exhausted according to retry policy), so a task is
never silently discarded. See [failure semantics](docs/failure-semantics.md).

## Evaluation

`nexus-knowledge bench` runs deterministic benchmarks over a synthetic corpus
(`nexus_knowledge.eval.fixtures`) with labeled relations, evidence and claims.
Every run reproduces the same numbers.

Graph extraction:

| metric              | value  |
|---------------------|--------|
| entity recall       | 1.000  |
| relation recall     | 1.000  |
| evidence precision  | 0.548  |

Knowledge (claim verification + provenance):

| metric                  | value  |
|-------------------------|--------|
| claim accuracy          | 1.000  |
| calibration error       | 0.133  |
| provenance correctness  | 1.000  |

Retrieval at k=5:

| method | MRR   | nDCG@5 | P@5  | R@5 |
|--------|-------|--------|------|-----|
| lexical| 1.000 | 1.000  | 0.37 | 1.0 |
| vector | 1.000 | 0.984  | 0.36 | 1.0 |
| graph  | 0.900 | 0.926  | 0.65 | 1.0 |
| hybrid | 1.000 | 1.000  | 0.36 | 1.0 |

## Project layout

```text
src/nexus_knowledge/  knowledge intelligence engine (domain, graph, retrieval, eval)
src/nexus_runtime/    autonomous research runtime (distributed execution, policy, agent loop)
tests/                unit, integration, and fault-injection coverage
docs/                 contracts, guarantees, and architecture decisions
```

## Development

`make format`, `make lint`, `make typecheck`, and `make test` require the optional
development dependencies. `make verify` runs only standard-library compilation and
the complete test suite, so the project remains runnable with no external runtime
dependencies.

```bash
pytest -ra                                   # run all tests
pytest --cov=nexus_knowledge --cov-fail-under=80   # knowledge engine coverage gate
```

The investigation evidence pipeline is covered by sectioned verification
suites (grounding, provenance, identity, duplicates, lifecycle, adversarial,
replay, boundaries, closed loop, LLM-source trust, performance). See
[investigation verification test plan](docs/investigation-verification-tests.md).

See [architecture](docs/architecture.md), [runtime](docs/runtime.md), and
[distributed runtime](docs/distributed-runtime.md) for the integration boundary.
