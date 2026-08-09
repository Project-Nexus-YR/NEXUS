# Investigation evidence verification test plan

The `nexus_runtime.investigation` pipeline (extract → fuse → evaluate → verify
→ acquire → update) is validated by sectioned suites that lock in the
epistemic invariants below. Every suite is deterministic and ruff-clean; the
full set runs with:

```bash
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m ruff check src tests
```

## Sections and suites

| Section | Scope | Suite |
|---------|-------|-------|
| S4  | Grounding: candidates must reference real observations | `tests/test_grounding_provenance.py` |
| S5  | Provenance: complete, non-empty, round-trippable, resolveable | `tests/test_grounding_provenance.py` |
| S6  | Deterministic identity: claim/evidence/candidate ids and fingerprints | `tests/test_claim_identity.py` |
| S7  | Duplicates and contradictions: no inflation, no false merge | `tests/test_duplicates_contradictions.py` |
| S9  | Deferred lifecycle: single-source and probable claims defer | `tests/test_acquisition_lifecycle.py` |
| S10 | Rejected lifecycle: contradicted and no-decision claims reject | `tests/test_acquisition_lifecycle.py` |
| S11 | Mixed lifecycle: only verified claims are submitted | `tests/test_acquisition_lifecycle.py` |
| S12 | Adversarial output: hallucinated refs, forged ids, status-as-claim | `tests/test_adversarial_output.py` |
| S13 | Replay safety: identical semantics after crashes at every boundary | `tests/test_replay_safety.py` |
| S14 | Architecture regression: no synthetic claim construction | `tests/test_no_synthetic_claims.py` |
| S15 | Evidentiary strength boundaries and clamping | `tests/test_evidentiary_boundaries.py` |
| S16 | Closed loop: learn, persist, retrieve, re-verify on the real engine | `tests/test_closed_loop.py` |
| S17 | LLM-source trust boundary: low quality cannot bypass gates | `tests/test_llm_source_boundary.py` |
| S18 | Pipeline performance sanity at scale | `tests/test_pipeline_performance.py` |

## Invariants locked in

- **Grounding.** A candidate only exists if its conclusion references known
  observations. Unknown references surface as `unknown_observation_reference`
  diagnostics naming the missing id; extraction never mutates the input result.
- **Provenance.** `EvidenceProvenance` requires all ten fields non-empty;
  `is_complete` and round-trips via `to_dict`/`from_dict`. Evidence must match
  its provenance (investigation id, source reference) and the enclosing result's
  lineage; replay recomputes `Evaluation` from the persisted `EvidenceSet`.
- **Determinism.** Claim ids are `_stable_id("claim", *identity)` over the
  case-folded, whitespace-normalized identity; fingerprints ignore confidence
  but not the source, excerpt, payload, or role. Replay must compare these
  semantic ids, never auto-generated timestamps/verification ids.
- **Fusion.** Duplicate evidence never inflates the independent-source count;
  identical claims merge, opposite claims never merge; contradictions surface
  as deterministic `conflict_id`s.
- **Gates.** An item must pass `min_evidence_confidence` and
  `min_source_quality` to count; verification requires `min_independent_sources`
  and an average quality floor. A single inflated signal cannot carry an item
  (geometric mean collapses to zero). The integration layer only submits
  eligible claims whose provenance resolves through the knowledge service.
- **Lifecycle.** Eligible → verified; contradicted → rejected; probable,
  uncertain, and insufficient → deferred; no decision → rejected. Only verified
  claims are committed; re-application is idempotent on the real engine.
- **Closed loop.** A verified claim is persisted by `commit_knowledge_update`,
  the engine's own `verify_claim` confirms it, and a later retrieval surfaces
  its grounding corpus.
- **Performance.** `EvidenceEvaluator.evaluate` builds the acceptable-supporting
  index once rather than per claim, keeping the pipeline linear-ish; the scale
  suite guards against superlinear regressions.

## Known behavior notes

- Two independent acceptable sources yield `EpistemicStatus.CONFIRMED`, not
  "verified"; `CandidateStatus.VERIFIED` is the acquisition status.
- A source with `source_quality` below 0.5 is excluded entirely from
  supporting evidence, regardless of agent confidence.
- The lexical/vector retrieval index is corpus-backed; learned claims are read
  back through the claims repository, `query_graph`-style relations, and
  `provenance()`, not through chunk retrieval.
