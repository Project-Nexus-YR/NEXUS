# Nexus Knowledge

Knowledge Intelligence Engine for the NEXUS autonomous knowledge-discovery
platform. Ingests heterogeneous sources into a knowledge graph, exposes hybrid
retrieval, GraphRAG evidence extraction, and uncertainty/contradiction/gap
analysis with deterministic, reproducible evaluation.

## Install

Requires Python 3.10+ and NumPy.

```bash
pip install -e ".[test]"
```

## Quick start

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

## Library API

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

## Development

```bash
pytest -ra                                   # run tests
pytest --cov=nexus_knowledge --cov-fail-under=80   # coverage gate
```
