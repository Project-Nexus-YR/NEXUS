# Evidence Engine

The NEXUS Evidence Engine turns draft prose into atomic claims, retrieves source
material, verifies the relationship between every claim and exact evidence span,
emits citations, and audits the result. It is a bounded context in
`nexus_evidence`; it does not replace either the Knowledge Intelligence Engine's
knowledge claims or the Investigation Runtime's provenance-complete agent evidence.

## Architecture audit and reuse decisions

The implementation reuses four existing boundaries:

| Existing boundary | Evidence Engine use |
|---|---|
| `KnowledgeEngine` | `retrieve_evidence` exposes ranked chunks plus source → document → chunk locators without repository access. |
| `RuntimeApplication` / `Worker` | Evidence DAG nodes are ordinary distributed tasks, retaining leases, bounded retries, cancellation, dead letters, backpressure, worker capabilities, and at-least-once delivery. |
| `InvestigationRepository` | `InvestigationArtifactEvidenceStore` appends immutable workflow checkpoints to the existing in-memory or SQLite investigation record. |
| `InvestigationApplication` | The optional `CitationVerificationPort` audits synthesized prose and records the report as a `citation_audit` artifact. |

No second scheduler, task state machine, event bus, or SQLite schema was added.
The earlier investigation `Evidence` remains the strict runtime lineage record
used to decide whether knowledge may be updated. `EvidenceSpan` is instead an
exact source locator used to decide whether final prose is cited correctly.

## Domain model

- `AtomicClaim` has a stable content/span identity, claim type, importance,
  source offsets, optional subject/predicate/object, and qualifiers.
- `EvidenceSource` is the origin. It is not evidence by itself.
- `EvidenceSpan` is exact content within a document/chunk and records the search
  query, retrieval strategy, rank, page/section/paragraph/character locator,
  content versions, and arbitrary JSON metadata.
- `EvidenceAssessment` links one claim to one span as `supported`,
  `partially_supported`, `contradicted`, `insufficient`, or `unverifiable`.
- `ClaimAssessment` aggregates independent-source evidence and preserves
  `conflicting_evidence` rather than forcing a false consensus.
- `Citation` is a validated claim → evidence → source path with an exact locator.
- `CitationAudit` contains issues, metrics, the repair round, and the complete
  claim/citation assessment set.

The queryable `CitationGraph` supports claim-to-evidence, claim-to-citation,
source-to-claim, and adjacency queries. It rejects dangling or mismatched edges.

## Workflow

1. Deterministic extraction splits prose into assertive sentences and only
   decomposes conjunctions when both sides contain independent verb phrases.
   An optional extraction provider is supported, but its outputs must point to
   exact input spans and conform to the same schema.
2. Claims are prioritized by importance. The planner emits exact, semantic,
   primary-source, type-specific, and contradiction queries.
3. Retrieval adapters use the public knowledge service or a provider-neutral
   search port. Results are normalized into source/span records and deduplicated
   by stable evidence identity.
4. Verification first applies deterministic rules. Numerical values are parsed
   with decimal arithmetic, units are normalized, and a configurable relative
   tolerance is applied. Quotations require exact words. Negation polarity and
   conflicting comparable quantities can yield contradiction.
5. Supported paths become citations. The auditor reports unsupported claims,
   missing/weak/partial/contradictory/mismatched/overreaching/duplicate/stale
   citations and computes coverage, weighted coverage, citation precision,
   evidence support, unsupported, contradiction, and source-quality rates.
6. Repair is bounded by `max_repair_rounds`. Only repairable claim IDs receive a
   targeted authoritative-source query. The loop stops on success, no new
   evidence, no quality improvement, or its configured bound.

Every workflow has a deterministic identity derived from session and draft hash.
Completed reports are returned idempotently without repeating retrieval. The
artifact adapter checkpoints extracted claims, collected evidence, repair rounds,
and the completed report, so a SQLite-backed investigation retains state across
process restarts.

## Distributed execution

`EvidenceTaskCompiler` emits a stable DAG of:

- `claim_extraction`;
- parallel `evidence_retrieval` nodes for each claim/query;
- `evidence_verification` nodes after their retrieval wave;
- a final `citation_audit` node.

`EvidenceExecutionController` submits only ready nodes. It recovers existing tasks
by workflow/node identity, blocks descendants of cancelled/dead-lettered parents,
and delegates cancellation to the runtime. Task metadata carries a schema version,
stable idempotency key, dependency IDs, stage payload, and an explicit marker that
source content is untrusted. `EvidenceHarness` adapts provider-specific stage
handlers to the standard runtime worker without changing worker semantics.

At-least-once delivery means stage handlers must persist their result under the
metadata idempotency key before returning a result reference. A transient handler
failure follows the runtime retry policy; malformed task payloads are permanent.

## Source security

External content is data, never an instruction. `untrusted_context` adds a clear
model boundary around source text. Retrieval normalization removes transport
control characters and enforces a size limit. The web adapter accepts only
absolute HTTP(S) references and rejects credentials, localhost names, private,
loopback, link-local, multicast, and reserved literal IP targets before a provider
is allowed to dereference them. Production HTTP adapters must additionally pin DNS
resolution and re-check every redirect target to prevent DNS-rebinding SSRF.

## CLI and API

```bash
# inspect claim decomposition
nexus-evidence extract --draft-file answer.md

# audit against an existing knowledge snapshot
nexus-evidence audit --draft-file answer.md --data .nexus/knowledge.json \
  --session-id investigation-123 --output citation-report.json

# deterministic evaluation
nexus-evidence-bench --output evidence-benchmark.json
```

Library composition:

```python
from nexus_evidence import EvidenceEngine, KnowledgeEvidenceRetriever

evidence = EvidenceEngine(KnowledgeEvidenceRetriever(knowledge_engine))
report = evidence.verify_answer(draft, session_id=session_id)
if report.audit.passes:
    publish(report.audit.citations)
```

## Metrics and known limits

`EvidenceMetrics` records workflow/query/evidence/citation/repair counters, issue
counts, and every audit metric. Distributed task/worker health remains in the
existing runtime metrics; investigation invocations also publish citation metrics
under the `citation_` prefix.

The deterministic extractor and verifier are intentionally conservative. They do
not resolve arbitrary coreference, entailment, causal validity, or implicit
baselines. Complex claims should use a schema-constrained provider followed by the
same deterministic locator and numerical checks. Source-quality scores are adapter
inputs, not claims of institutional authority. Freshness is only evaluated when a
parseable publication timestamp is available.
