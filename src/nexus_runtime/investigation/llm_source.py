"""LLM-backed evidence sources for the investigation loop.

A :class:`LLMSource` is a read-only, source-grounded evidence adapter: given a
prompt it returns :class:`LLMSourceResult` items whose stable identifiers
(source/document/chunk) feed the existing provenance chain so downstream
verification and knowledge update resolve without special casing.  The runtime
``Tool`` adapter (:class:`LLMSourceTool`) exposes one to agents through the
capability-based :class:`ToolRegistry`; providers are injected, never imported.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "DeterministicLLMSource",
    "LLMSource",
    "LLMSourceResult",
    "LLMSourceTool",
    "OpenAICompatibleLLMSource",
]


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[
        :24
    ]
    return f"{prefix}_{digest}"


class LLMSource(Protocol):
    """Read-only protocol satisfied by real and deterministic sources."""

    name: str

    def query(self, prompt: str, *, limit: int = 1) -> tuple[LLMSourceResult, ...]: ...


@dataclass(frozen=True, slots=True)
class LLMSourceResult:
    """One source-grounded snippet with the lineage the loop can resolve."""

    text: str
    source_id: str
    document_id: str
    chunk_id: str
    source_reference: str
    source_quality: float = 0.6
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("source result text must be a non-empty string")
        for value, name in (
            (self.source_id, "source_id"),
            (self.document_id, "document_id"),
            (self.chunk_id, "chunk_id"),
            (self.source_reference, "source_reference"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source result {name} must be a non-empty string")
        for number, name in (
            (self.source_quality, "source_quality"),
            (self.confidence, "confidence"),
        ):
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"source result {name} must be numeric")
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"source result {name} must be between zero and one")
        if not isinstance(self.metadata, dict):
            raise ValueError("source result metadata must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_reference": self.source_reference,
            "source_quality": self.source_quality,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LLMSourceResult:
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("malformed source result")
        quality = payload.get("source_quality", 0.6)
        confidence = payload.get("confidence", 0.5)
        if isinstance(quality, bool) or not isinstance(quality, (int, float)):
            raise ValueError("malformed source result")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("malformed source result")
        try:
            return cls(
                text=_persisted_string(payload, "text"),
                source_id=_persisted_string(payload, "source_id"),
                document_id=_persisted_string(payload, "document_id"),
                chunk_id=_persisted_string(payload, "chunk_id"),
                source_reference=_persisted_string(payload, "source_reference"),
                source_quality=float(quality),
                confidence=float(confidence),
                metadata=dict(metadata),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed source result") from exc


def _persisted_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


class DeterministicLLMSource:
    """Offline source for tests and deterministic evaluation runs.

    Responses are looked up by exact prompt match; a ``default`` response is
    returned when configured and no exact match exists.  Results are stable:
    identical prompts yield identical identifiers, so evidence lineage stays
    reproducible across retries.
    """

    def __init__(
        self,
        responses: Mapping[str, str],
        *,
        default: str = "",
        name: str = "deterministic",
        source_quality: float = 0.7,
    ) -> None:
        self._responses = dict(responses)
        self._default = default
        self.name = name
        self.source_quality = source_quality

    def query(self, prompt: str, *, limit: int = 1) -> tuple[LLMSourceResult, ...]:
        if limit < 1:
            raise ValueError("limit must be at least one")
        text = self._responses.get(prompt, self._default)
        if not text:
            return ()
        return (self._result(prompt, text),)

    def _result(self, prompt: str, text: str) -> LLMSourceResult:
        return LLMSourceResult(
            text=text,
            source_id=f"llm:{self.name}",
            document_id=_stable_id("llm_document", prompt),
            chunk_id=f"{_stable_id('llm_document', prompt)}:0",
            source_reference=f"llm://{self.name}",
            source_quality=self.source_quality,
            confidence=0.7,
            metadata={"provider": self.name},
        )


Transport = Callable[[str, Mapping[str, Any], float], bytes]


def _make_transport(api_key: str | None) -> Transport:
    def transport(url: str, payload: Mapping[str, Any], timeout_seconds: float) -> bytes:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data: bytes = response.read()
            return data

    return transport


class OpenAICompatibleLLMSource:
    """Real adapter for any OpenAI-compatible ``/chat/completions`` endpoint.

    The transport is injectable so network behaviour can be tested without a
    live endpoint; by default it uses the standard library (``urllib``) and
    sends the API key in the ``Authorization`` header.  The adapter never
    retries or re-sends payloads — idempotency is the caller's responsibility
    via ``LLMSourceTool``.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        name: str | None = None,
        timeout_seconds: float = 60.0,
        source_quality: float = 0.6,
        transport: Transport | None = None,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("LLM endpoint must be a non-empty URL")
        if not model.strip():
            raise ValueError("LLM model must be a non-empty string")
        self._endpoint = endpoint
        self._model = model
        self.name = name or f"llm:{model}"
        self._timeout_seconds = timeout_seconds
        self._source_quality = source_quality
        self._transport = transport or _make_transport(api_key)

    def query(self, prompt: str, *, limit: int = 1) -> tuple[LLMSourceResult, ...]:
        if limit < 1:
            raise ValueError("limit must be at least one")
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "n": 1,
        }
        try:
            raw = self._transport(self._endpoint, payload, self._timeout_seconds)
        except Exception as exc:
            raise RuntimeError(f"LLM source request failed: {self._endpoint}") from exc
        results = self._parse(raw, prompt)
        return tuple(results[:limit])

    def _parse(self, raw: bytes, prompt: str) -> tuple[LLMSourceResult, ...]:
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("LLM source returned malformed JSON") from exc
        choices = body.get("choices")
        if not isinstance(choices, list):
            raise RuntimeError("LLM source response lacks choices")
        results: list[LLMSourceResult] = []
        for index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                continue
            document_id = _stable_id("llm_document", prompt, str(index))
            results.append(
                LLMSourceResult(
                    text=content.strip(),
                    source_id=f"llm:{self._model}",
                    document_id=document_id,
                    chunk_id=f"{document_id}:{index}",
                    source_reference=f"llm://{self._model}",
                    source_quality=self._source_quality,
                    confidence=0.6,
                    metadata={"provider": self.name, "choice_index": index},
                )
            )
        return tuple(results)


class LLMSourceTool:
    """Runtime ``Tool`` adapter exposing an :class:`LLMSource` to agents.

    The output flattens the primary result's provenance identifiers at the top
    level so the agent harness can build :class:`Evidence` with complete,
    resolvable lineage without parsing provider-specific response shapes.
    """

    name = "llm_source"
    description = "Query an LLM-backed source for source-grounded evidence"
    capability = "llm_source.query"
    input_schema = {"required": ["query"]}
    output_schema = {"required": ["results"]}
    permissions = frozenset({"llm_source.query"})
    timeout_seconds = 60.0
    side_effect = "none"
    idempotency = "safe"

    def __init__(self, source: LLMSource, *, timeout_seconds: float = 60.0) -> None:
        self._source = source
        self.timeout_seconds = timeout_seconds

    def execute(self, input: dict[str, Any], idempotency_key: str | None) -> dict[str, Any]:
        prompt = input.get("query")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("llm_source requires a non-empty query string")
        limit = input.get("limit", 1)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("llm_source limit must be a positive integer")
        results = self._source.query(prompt, limit=limit)
        serialized = [item.to_dict() for item in results]
        primary = results[0] if results else None
        if primary is None:
            return {"results": serialized}
        return {
            "results": serialized,
            "excerpt": primary.text,
            "source_id": primary.source_id,
            "document_id": primary.document_id,
            "chunk_id": primary.chunk_id,
            "source_reference": primary.source_reference,
            "source_quality": primary.source_quality,
        }
