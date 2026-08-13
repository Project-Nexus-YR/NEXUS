# NEXUS visual knowledge explorer

The visual explorer is an Obsidian-style workspace over NEXUS's existing public
boundaries. The graph remains the primary navigation surface; list views for gaps,
contradictions, sources, investigations, and runtime state focus objects back into
that graph instead of becoming separate CRUD pages.

## Architecture

```text
React / TypeScript / Sigma.js
          │ JSON over /api
          ▼
FastAPI transport (nexus_gui.api)
          │
          ▼
NexusGuiService ── KnowledgeExplorer ── KnowledgeEngine
          │                                │
          └──── RuntimeMonitor ────────────┼── InvestigationRepository
                         │                 └── KnowledgeRepository / KnowledgeGraph
                         └──────────────────── RuntimeApplication
```

`KnowledgeExplorer` is a transport-neutral read application service. It uses the
existing `KnowledgeEngine` and repository ports to project a heterogeneous,
bounded graph; the HTTP layer does not read SQLite tables. `RuntimeMonitor` uses
the public runtime application and investigation repository contracts. Creating
an investigation from a gap calls `InvestigationApplication.create`, so the UI
action creates a real durable session.

The server polls no internal state. The browser polls the graph and overview APIs
at a configurable interval (8 seconds by default). A stable snapshot identifier
prevents unnecessary graph reconstruction; newly observed node identifiers are
highlighted for five seconds. This provides live refresh without claiming that
the current runtime has a streaming transport.

## Domain-to-visual mapping

| NEXUS model | Visual representation |
| --- | --- |
| `Entity` | neutral, connectivity-sized entity node |
| `Relation` | directed edge whose type is the real relation predicate |
| `Claim` | confidence/verification-aware claim node |
| `Evidence` | compact node linked to its claim and document |
| `Document` / `Source` | provenance nodes and document-to-source edges |
| `KnowledgeGap` | warning-colored, importance-sized node exposing priority and uncertainty |
| `Contradiction` | red edge between conflicting claims, with both evidence sides in inspectors |
| investigation session | active-accent node linked to its target gap |
| distributed task / worker | runtime and investigation activity panels |

Node size combines a stable per-kind base with graph degree and, where the model
has them, importance and confidence. Colors are deliberately restrained: neutral
knowledge, supported knowledge, gaps, contradictions, and active investigations.
The inspector presents only serialized domain fields and provides Overview,
Evidence, Connections, Provenance, and History tabs.

## Why Sigma.js and Graphology

The canvas uses [Sigma.js](https://www.sigmajs.org/docs/) 3 with Graphology and
the ForceAtlas2 worker. Sigma renders through WebGL and is designed for interactive
network exploration with thousands of visible elements, while Graphology supplies
a well-supported graph data model and layout ecosystem. This fits NEXUS better than
a DOM-oriented node editor. Scalability is also enforced before rendering: the API
defaults to 750 nodes / 2,500 edges and hard-caps requests at 2,000 / 10,000.

Global graph, server-filtered slices, fuzzy search, and depth 1–3 neighborhood
queries are separate operations. This lets a future persistent graph adapter
optimize projection/query execution without changing the frontend contract.

## Install and run

Python 3.11+ and Node.js 20+ are required.

```bash
python3 -m pip install -e '.[gui]'
npm --prefix frontend install
npm --prefix frontend run build
nexus-gui --snapshot .nexus/knowledge.json
```

The built application is served at <http://127.0.0.1:8000>. Without `--snapshot`,
the GUI opens against an empty in-memory knowledge engine while investigation and
runtime state remain durable in `.nexus/*.sqlite`.

Create a real snapshot while ingesting:

```bash
nexus-knowledge ingest data/report.md --kind markdown \
  --title "Research report" --output .nexus/knowledge.json
```

For a deterministic development corpus, isolated from production components:

```bash
python3 scripts/generate_gui_demo.py .nexus/demo-knowledge.json
nexus-gui --snapshot .nexus/demo-knowledge.json
```

Useful server options:

```bash
nexus-gui --help
nexus-gui --host 127.0.0.1 --port 8000 \
  --investigations-db .nexus/investigations.sqlite \
  --runtime-db .nexus/runtime.sqlite
```

## Development workflow

Run the API and Vite separately for hot reload:

```bash
# terminal 1
nexus-gui --snapshot .nexus/demo-knowledge.json --frontend-dir /nonexistent

# terminal 2 (proxies /api to 127.0.0.1:8000)
npm --prefix frontend run dev
```

Verification commands:

```bash
pytest -q
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
```

The Makefile also exposes `gui-install`, `gui-test`, `gui-build`, and `gui`.

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | service and knowledge health summary |
| `GET /api/graph` | bounded global graph with server-side filters |
| `GET /api/graph/neighborhood/{id}` | local graph at depth 1–3 |
| `GET /api/nodes/{id}` | inspector data, evidence, provenance, connections and history |
| `GET /api/search?q=...` | fuzzy search across knowledge and investigations |
| `GET /api/gaps` | ranked measurable gaps and attached sessions |
| `POST /api/gaps/{id}/investigations` | create a durable investigation session |
| `GET /api/contradictions` | first-class conflicting knowledge records |
| `GET /api/sources` | source navigation independent of the active graph slice |
| `GET /api/investigations[/{id}]` | investigation summaries/details |
| `GET /api/runtime` | queue, worker and durable task state |

Graph query filters include repeated `kinds`, `relation_types`, and
`verification_states` parameters plus `min_confidence`, `max_nodes`, and
`max_edges`. The frontend never imports Python persistence schemas.

## Current limits

- Projection currently materializes the in-process repository before applying the
  bounded slice. The response/render path is bounded, but very large deployments
  should implement equivalent query pushdown in a persistent graph adapter.
- Updates use polling because the existing public runtime exposes read methods but
  no event stream suitable for a browser.
- `Investigate` creates the same durable `PLANNING` session as the research CLI;
  execution still requires the configured investigation runner/providers and is not
  simulated by the GUI.
- The local server composes one knowledge snapshot with its SQLite investigation
  and task stores. Multi-tenant authentication and remote deployment policy are
  intentionally outside this local exploration milestone.
