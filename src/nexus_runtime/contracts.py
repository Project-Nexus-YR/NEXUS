"""Provider contracts. The runtime never reaches through these interfaces."""

from __future__ import annotations

from typing import Any, Protocol


class ModelProvider(Protocol):
    def complete(self, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]: ...


class Tool(Protocol):
    name: str
    description: str
    capability: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permissions: frozenset[str]
    timeout_seconds: float
    side_effect: str
    idempotency: str

    def execute(self, input: dict[str, Any], idempotency_key: str | None) -> dict[str, Any]: ...


class MemoryProvider(Protocol):
    def recall(self, query: str, limit: int = 10) -> list[dict[str, Any]]: ...

    def remember(self, record: dict[str, Any]) -> str: ...


class Planner(Protocol):
    def plan(self, goal: str, context: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class Evaluator(Protocol):
    def evaluate(self, candidate: dict[str, Any]) -> dict[str, Any]: ...


class SearchProvider(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...

    def search_batch(self, queries: list[str]) -> list[list[dict[str, Any]]]: ...

    def retrieve_document(self, document_id: str) -> dict[str, Any]: ...


class WorkflowExecutor(Protocol):
    def submit_workflow(self, specification: dict[str, Any]) -> str: ...

    def get_status(self, workflow_id: str) -> str: ...

    def cancel_workflow(self, workflow_id: str) -> None: ...

    def get_outputs(self, workflow_id: str) -> dict[str, Any]: ...


class KnowledgeService(Protocol):
    """The sole boundary to the separately-owned Knowledge Intelligence Engine."""

    def retrieve(self, query: str) -> list[dict[str, Any]]: ...

    def query_graph(self, query: dict[str, Any]) -> dict[str, Any]: ...

    def get_subgraph(self, seed_ids: list[str]) -> dict[str, Any]: ...

    def find_knowledge_gaps(self, goal: str) -> list[dict[str, Any]]: ...

    def score_investigation(self, investigation: dict[str, Any]) -> float: ...

    def propose_claim(self, proposal: dict[str, Any]) -> str: ...

    def verify_claim(self, claim_id: str) -> dict[str, Any]: ...

    def commit_knowledge_update(self, verified_proposal: dict[str, Any]) -> str: ...
