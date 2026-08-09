"""LLM-backed evidence source: contracts, adapters, and loop integration (Section 8)."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from nexus_runtime.agent import AgentExecutor
from nexus_runtime.distributed.worker import (
    HarnessExecutionContext,
    HarnessStatus,
)
from nexus_runtime.investigation.agent_harness import AgentHarness
from nexus_runtime.investigation.candidate_claims import CandidateClaimExtractor
from nexus_runtime.investigation.evidence import InvestigationResultState
from nexus_runtime.investigation.llm_source import (
    DeterministicLLMSource,
    LLMSourceResult,
    LLMSourceTool,
    OpenAICompatibleLLMSource,
)
from nexus_runtime.investigation.results import InMemoryInvestigationResultRepository
from nexus_runtime.models import Agent, Budget
from nexus_runtime.policy import PolicyEngine
from nexus_runtime.tools import ToolRegistry


def _chat_response(content: str) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 10}}
    ).encode("utf-8")


class TestLLMSourceResult:
    def test_requires_text_and_identifier_fields(self):
        with pytest.raises(ValueError, match="non-empty string"):
            LLMSourceResult("", "source", "doc", "chunk", "ref")
        with pytest.raises(ValueError, match="non-empty string"):
            LLMSourceResult("text", "source", "doc", "", "ref")

    def test_quality_and_confidence_are_bounded(self):
        with pytest.raises(ValueError, match="between zero and one"):
            LLMSourceResult("text", "s", "d", "c", "r", source_quality=1.5)

    def test_serialization_round_trip(self):
        result = LLMSourceResult(
            "Atlas is in London",
            "llm:test",
            "doc-1",
            "chunk-1",
            "llm://test",
            source_quality=0.8,
            metadata={"provider": "test"},
        )
        restored = LLMSourceResult.from_dict(result.to_dict())
        assert restored == result


class TestDeterministicLLMSource:
    def test_exact_match_and_default_fallback(self):
        source = DeterministicLLMSource(
            {"acme hq": "Acme Corp is based in Initech Park."},
            default="No relevant record.",
        )
        hit = source.query("acme hq")
        assert len(hit) == 1
        assert hit[0].source_reference == "llm://deterministic"
        fallback = source.query("unknown question")
        assert len(fallback) == 1
        assert fallback[0].text == "No relevant record."

    def test_no_match_without_default_returns_empty(self):
        source = DeterministicLLMSource({})
        assert source.query("anything") == ()

    def test_identifiers_are_stable_across_calls(self):
        source = DeterministicLLMSource({"q": "answer"})
        first = source.query("q")[0]
        second = source.query("q")[0]
        assert first.source_id == second.source_id
        assert first.document_id == second.document_id
        assert first.chunk_id == second.chunk_id
        assert first.to_dict() == second.to_dict()


class TestOpenAICompatibleLLMSource:
    def test_parses_chat_completion_response(self):
        captured: dict[str, Any] = {}

        def transport(url: str, payload: dict[str, Any], timeout: float) -> bytes:
            captured["url"] = url
            captured["payload"] = payload
            assert "api_key" not in payload
            return _chat_response("Atlas is headquartered in London.")

        source = OpenAICompatibleLLMSource(
            "https://api.example.test/v1/chat/completions",
            "test-model",
            api_key="secret",
            transport=transport,
        )
        results = source.query("Where is Atlas?")
        assert len(results) == 1
        result = results[0]
        assert result.text == "Atlas is headquartered in London."
        assert result.source_id == "llm:test-model"
        assert result.source_reference == "llm://test-model"
        assert result.document_id
        assert result.chunk_id.endswith(":0")
        assert captured["url"] == "https://api.example.test/v1/chat/completions"
        assert captured["payload"]["model"] == "test-model"

    def test_limit_truncates_results(self):
        body = {"choices": [{"message": {"content": f"answer {index}"}} for index in range(3)]}

        def transport(url: str, payload: dict[str, Any], timeout: float) -> bytes:
            return json.dumps(body).encode("utf-8")

        source = OpenAICompatibleLLMSource(
            "https://api.example.test/v1/chat/completions", "test-model", transport=transport
        )
        assert len(source.query("q", limit=2)) == 2

    def test_malformed_response_raises_runtime_error(self):
        source = OpenAICompatibleLLMSource(
            "https://api.example.test/v1/chat/completions",
            "test-model",
            transport=lambda url, payload, timeout: b"not json",
        )
        with pytest.raises(RuntimeError, match="malformed JSON"):
            source.query("q")

    def test_response_without_choices_raises_runtime_error(self):
        source = OpenAICompatibleLLMSource(
            "https://api.example.test/v1/chat/completions",
            "test-model",
            transport=lambda url, payload, timeout: b'{"error": "nope"}',
        )
        with pytest.raises(RuntimeError, match="lacks choices"):
            source.query("q")

    def test_transport_failure_is_wrapped(self):
        def broken(url: str, payload: dict[str, Any], timeout: float) -> bytes:
            raise ConnectionError("down")

        source = OpenAICompatibleLLMSource(
            "https://api.example.test/v1/chat/completions", "test-model", transport=broken
        )
        with pytest.raises(RuntimeError, match="request failed"):
            source.query("q")

    def test_default_transport_sends_authorization_header(self, monkeypatch):
        captured: dict[str, Any] = {}

        class FakeResponse:
            def __init__(self) -> None:
                self.body = _chat_response("Atlas in London.")

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return self.body

        def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
            captured["headers"] = dict(request.headers)
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        source = OpenAICompatibleLLMSource(
            "https://api.example.test/v1/chat/completions", "test-model", api_key="secret"
        )
        assert len(source.query("q")) == 1
        assert captured["headers"]["Authorization"] == "Bearer secret"


class TestLLMSourceTool:
    def _tool(self, **kwargs: Any) -> LLMSourceTool:
        return LLMSourceTool(DeterministicLLMSource({"q": "Atlas is in London."}, **kwargs))

    def test_flattens_primary_result_for_harness_lineage(self):
        output = self._tool().execute({"query": "q", "limit": 1}, "idem-1")
        assert output["excerpt"] == "Atlas is in London."
        assert output["source_id"].startswith("llm:")
        assert output["document_id"]
        assert output["chunk_id"]
        assert output["source_reference"] == "llm://deterministic"
        assert len(output["results"]) == 1

    def test_empty_source_returns_empty_results(self):
        tool = LLMSourceTool(DeterministicLLMSource({}))
        assert tool.execute({"query": "q"}, None) == {"results": []}

    def test_rejects_missing_or_blank_query(self):
        tool = self._tool()
        with pytest.raises(ValueError, match="non-empty query"):
            tool.execute({}, None)
        with pytest.raises(ValueError, match="non-empty query"):
            tool.execute({"query": "  "}, None)

    def test_rejects_invalid_limit(self):
        tool = self._tool()
        with pytest.raises(ValueError, match="positive integer"):
            tool.execute({"query": "q", "limit": 0}, None)


class TestLLMSourceInHarness:
    """LLM-sourced evidence flows through the harness into resolvable lineage."""

    class _Model:
        def __init__(self) -> None:
            self.steps = 0

        def complete(self, prompt: str, response_schema: dict[str, object]) -> dict[str, Any]:
            self.steps += 1
            if self.steps == 1:
                return {
                    "action": "tool",
                    "tool": "llm_source",
                    "input": {"query": "Where is Atlas HQ?", "limit": 1},
                }
            observation_id = ""
            for line in prompt.splitlines():
                if line.startswith("[observation:"):
                    observation_id = line.split("]", 1)[0][len("[observation:") :]
                    break
            return {
                "action": "finish",
                "output": {
                    "final_answer": "Atlas is headquartered in London.",
                    "conclusions": [
                        {
                            "claim": {
                                "text": "Atlas is headquartered in London",
                                "subject": "Atlas",
                                "predicate": "headquartered_in",
                                "object": "London",
                            },
                            "supporting_observation_ids": [observation_id],
                            "confidence": 0.9,
                            "conclusion_id": "conclusion-llm-hq",
                        }
                    ],
                },
            }

        def recall(self, query: str, limit: int = 10) -> list[dict[str, str]]:
            return []

    def _context(self) -> HarnessExecutionContext:
        return HarnessExecutionContext(
            run_id="run-llm-1",
            correlation_id="session-llm-1",
            task_id="task-llm-1",
            attempt_id="attempt-llm-1",
            lease_id="lease-1",
            worker_id="worker-1",
            metadata={
                "investigation_id": "investigation-llm-1",
                "question": "Where is Atlas HQ?",
            },
        )

    def test_harness_builds_complete_evidence_from_llm_source(self, ingested_engine):
        agent = Agent("Researcher", "Researcher", frozenset({"llm_source.query"}))
        policy = PolicyEngine({agent.agent_id: frozenset({"llm_source.query"})})
        registry = ToolRegistry(policy)
        registry.register(
            LLMSourceTool(
                DeterministicLLMSource({"Where is Atlas HQ?": "Atlas is headquartered in London."})
            )
        )
        executor = AgentExecutor(self._Model(), self._Model(), registry)
        results = InMemoryInvestigationResultRepository()
        harness = AgentHarness(
            executor,
            results,
            agent,
            budget=Budget(100, timedelta(minutes=1), 5, 1, 1),
        )
        outcome = harness.execute_or_resume(self._context(), cancellation_requested=lambda: False)

        assert outcome.status == HarnessStatus.SUCCEEDED
        result = results.get(outcome.result_ref or "")
        assert result is not None
        assert result.state == InvestigationResultState.COMPLETED
        assert result.conclusions

        extraction = CandidateClaimExtractor().extract(result)
        assert len(extraction.candidates) == 1
        evidence = extraction.evidence_set.evidence[0]
        assert evidence.provenance.is_complete
        assert evidence.provenance.source_id == "llm:deterministic"
        assert evidence.provenance.source_reference == "llm://deterministic"
        assert evidence.provenance.chunk_id
        assert evidence.provenance.tool_call_id
        assert evidence.evidentiary_strength == pytest.approx((0.9 * 0.7) ** 0.5)

        # The strict lineage gate applies uniformly to LLM-sourced evidence:
        # ids not present in the knowledge store are rejected at preparation.
        from nexus_runtime.investigation.evaluation import EvidenceEvaluator
        from nexus_runtime.investigation.knowledge_update import KnowledgeUpdateIntegrator
        from nexus_runtime.investigation.verification import ClaimVerifier, VerificationPolicy

        verification = ClaimVerifier(
            VerificationPolicy(min_independent_sources=1, allow_probable_updates=True)
        ).verify(EvidenceEvaluator().evaluate(extraction.evidence_set))
        assert verification.decisions[0].eligible_for_update
        integrator = KnowledgeUpdateIntegrator(ingested_engine)
        with pytest.raises(ValueError, match="does not resolve"):
            integrator.prepare(verification, extraction.evidence_set)
