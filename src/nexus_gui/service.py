"""Transport-independent composition of knowledge and runtime read models."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from difflib import SequenceMatcher
from typing import Any

from nexus_knowledge.service.explorer import ExplorerFilters, KnowledgeExplorer
from nexus_runtime.investigation.application import InvestigationApplication
from nexus_runtime.investigation.objective import ResearchObjective
from nexus_runtime.investigation.session import InvestigationBudget
from nexus_runtime.monitoring import RuntimeMonitor


class NexusGuiService:
    """Stable application boundary consumed by HTTP and test adapters."""

    def __init__(
        self,
        knowledge: KnowledgeExplorer,
        runtime: RuntimeMonitor,
        investigations: InvestigationApplication,
    ) -> None:
        self._knowledge = knowledge
        self._runtime = runtime
        self._investigations = investigations

    def graph(self, filters: ExplorerFilters | None = None) -> dict[str, Any]:
        filters = filters or ExplorerFilters()
        graph = self._knowledge.graph(filters).to_dict()
        if "investigation" in filters.node_kinds:
            self._add_runtime_overlay(graph, filters)
        return graph

    def neighborhood(
        self,
        node_id: str,
        *,
        depth: int,
        filters: ExplorerFilters | None = None,
    ) -> dict[str, Any]:
        filters = filters or ExplorerFilters(max_nodes=500, max_edges=2_000)
        try:
            graph = self._knowledge.neighborhood(node_id, depth=depth, filters=filters).to_dict()
        except KeyError:
            investigation = self._runtime.investigation(node_id)
            graph = self._investigation_neighborhood(investigation, depth, filters)
        if "investigation" in filters.node_kinds:
            self._add_runtime_overlay(graph, filters, related_to=set(self._node_ids(graph)))
        return graph

    def node(self, node_id: str) -> dict[str, Any]:
        try:
            return self._knowledge.node_detail(node_id).to_dict()
        except KeyError:
            investigation = self._runtime.investigation(node_id)
            return self._investigation_detail(investigation)

    def search(self, query: str, *, limit: int = 25) -> list[dict[str, Any]]:
        results = self._knowledge.search(query, limit=limit)
        text = " ".join(query.casefold().split())
        for item in self._runtime.list_investigations():
            haystack = f"{item['question']} {item['state']} investigation".casefold()
            if text in haystack:
                score = 2.0 + len(text) / max(1, len(haystack))
            else:
                score = SequenceMatcher(None, text, haystack).ratio()
            if score < 0.32:
                continue
            results.append(
                {
                    "score": round(score, 4),
                    "id": item["session_id"],
                    "kind": "investigation",
                    "label": item["question"],
                    "subtitle": item["phase"],
                    "confidence": None,
                    "uncertainty": None,
                    "importance": None,
                    "verification_state": None,
                    "status": item["state"],
                    "degree": len(item["target_gap_ids"]),
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "metadata": {},
                }
            )
        results.sort(key=lambda item: (-float(item["score"]), str(item["label"])))
        return results[:limit]

    def gaps(self) -> list[dict[str, Any]]:
        sessions_by_gap: dict[str, list[dict[str, Any]]] = {}
        for session in self._runtime.list_investigations():
            for gap_id in session["target_gap_ids"]:
                sessions_by_gap.setdefault(gap_id, []).append(session)
        return [
            {**gap, "sessions": sessions_by_gap.get(str(gap["id"]), [])}
            for gap in self._knowledge.gaps()
        ]

    def contradictions(self) -> list[dict[str, Any]]:
        return self._knowledge.contradictions()

    def sources(self) -> list[dict[str, Any]]:
        return self._knowledge.sources()

    def investigations(self) -> list[dict[str, Any]]:
        return self._runtime.list_investigations()

    def investigation(self, session_id: str) -> dict[str, Any]:
        return self._runtime.investigation(session_id)

    def runtime(self) -> dict[str, Any]:
        return self._runtime.runtime()

    def create_gap_investigation(
        self,
        gap_id: str,
        *,
        question: str | None = None,
        success_criteria: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        gap = next((item for item in self._knowledge.gaps() if item["id"] == gap_id), None)
        if gap is None:
            raise KeyError(gap_id)
        resolved_question = question or f"Resolve this knowledge gap: {gap['description']}"
        criteria = success_criteria or (
            f"The knowledge gap {gap_id} is resolved with provenance-complete verified evidence",
        )
        objective = ResearchObjective(
            question=resolved_question,
            success_criteria=criteria,
            scope=tuple(str(item) for item in gap.get("affected_entities", [])),
            constraints=("preserve uncertainty and contradictory evidence",),
            metadata={"target_gap_id": gap_id, "created_via": "nexus_gui"},
        )
        session = self._investigations.create(
            objective,
            InvestigationBudget(
                max_iterations=3,
                max_investigations=10,
                max_agent_runs=10,
                max_cost=50.0,
                max_execution_time=timedelta(hours=1),
            ),
        )
        return self._runtime.investigation(session.session_id)

    def health(self) -> dict[str, Any]:
        graph = self._knowledge.graph(ExplorerFilters(max_nodes=1, max_edges=1))
        return {
            "status": "ok",
            "knowledge": graph.statistics,
            "investigation_count": len(self._runtime.list_investigations()),
            "runtime_available": bool(self._runtime.runtime()["available"]),
        }

    def _add_runtime_overlay(
        self,
        graph: dict[str, Any],
        filters: ExplorerFilters,
        *,
        related_to: set[str] | None = None,
    ) -> None:
        nodes, edges = self._runtime.graph_overlay()
        existing = self._node_ids(graph)
        available = existing | {str(item["id"]) for item in nodes}
        if related_to is not None:
            related_sessions = {
                str(edge["source"]) for edge in edges if str(edge["target"]) in related_to
            }
            related_sessions.update(item for item in related_to if item in available)
            nodes = [item for item in nodes if str(item["id"]) in related_sessions]
            available = existing | {str(item["id"]) for item in nodes}
        overlay_node_total = len(nodes)
        eligible_edges = [
            item
            for item in edges
            if str(item["source"]) in available and str(item["target"]) in available
        ]
        room = max(0, filters.max_nodes - len(existing))
        nodes = nodes[:room]
        available = existing | {str(item["id"]) for item in nodes}
        edges = [
            item
            for item in eligible_edges
            if str(item["source"]) in available and str(item["target"]) in available
        ]
        edge_room = max(0, filters.max_edges - len(graph["edges"]))
        graph["nodes"].extend(nodes)
        graph["edges"].extend(edges[:edge_room])
        graph["total_nodes"] = int(graph["total_nodes"]) + overlay_node_total
        graph["total_edges"] = int(graph["total_edges"]) + len(eligible_edges)
        graph["truncated"] = (
            bool(graph["truncated"])
            or overlay_node_total > room
            or len(edges) > edge_room
        )
        graph["snapshot_id"] = self._combined_snapshot(str(graph["snapshot_id"]))

    def _investigation_neighborhood(
        self,
        investigation: dict[str, Any],
        depth: int,
        filters: ExplorerFilters,
    ) -> dict[str, Any]:
        target_gaps = investigation["target_gap_ids"]
        if target_gaps:
            pieces = [
                self._knowledge.neighborhood(gap_id, depth=depth, filters=filters).to_dict()
                for gap_id in target_gaps
            ]
            graph = self._merge_graphs(pieces, filters)
        else:
            graph = self._knowledge.graph(
                ExplorerFilters(
                    node_kinds=filters.node_kinds,
                    relation_types=filters.relation_types,
                    verification_states=filters.verification_states,
                    min_confidence=filters.min_confidence,
                    max_nodes=min(filters.max_nodes, 50),
                    max_edges=min(filters.max_edges, 100),
                )
            ).to_dict()
        return graph

    def _investigation_detail(self, item: dict[str, Any]) -> dict[str, Any]:
        connections = [
            {
                "edge": {
                    "id": f"investigation-gap:{item['session_id']}:{gap_id}",
                    "source": item["session_id"],
                    "target": gap_id,
                    "kind": "investigation_target",
                    "label": "investigates",
                    "directed": True,
                    "metadata": {},
                },
                "node": {"id": gap_id, "kind": "gap", "label": gap_id, "subtitle": ""},
            }
            for gap_id in item["target_gap_ids"]
        ]
        return {
            "node": {
                "id": item["session_id"],
                "kind": "investigation",
                "label": item["question"],
                "subtitle": item["phase"],
                "confidence": None,
                "uncertainty": None,
                "importance": None,
                "verification_state": None,
                "status": item["state"],
                "degree": len(connections),
                "created_at": item["created_at"],
                "updated_at": item["updated_at"],
                "metadata": {},
            },
            "attributes": item,
            "connections": connections,
            "evidence": [] if item["evidence"] is None else [item["evidence"]],
            "provenance": {},
            "contradictions": [],
            "gaps": [{"id": gap_id} for gap_id in item["target_gap_ids"]],
            "history": item.get("timeline", []),
        }

    def _combined_snapshot(self, knowledge_snapshot: str) -> str:
        encoded = f"{knowledge_snapshot}:{self._runtime.version()}"
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _node_ids(graph: dict[str, Any]) -> set[str]:
        return {str(item["id"]) for item in graph["nodes"]}

    @staticmethod
    def _merge_graphs(graphs: list[dict[str, Any]], filters: ExplorerFilters) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        statistics: dict[str, int | float] = {}
        truncated = False
        for graph in graphs:
            statistics = graph["statistics"]
            truncated = truncated or bool(graph["truncated"])
            for node in graph["nodes"]:
                nodes.setdefault(str(node["id"]), node)
            for edge in graph["edges"]:
                edges.setdefault(str(edge["id"]), edge)
        node_items = list(nodes.values())[: filters.max_nodes]
        node_ids = {str(item["id"]) for item in node_items}
        edge_items = [
            item
            for item in edges.values()
            if str(item["source"]) in node_ids and str(item["target"]) in node_ids
        ][: filters.max_edges]
        state = json.dumps(
            ([item["id"] for item in node_items], [item["id"] for item in edge_items]),
            sort_keys=True,
        )
        return {
            "nodes": node_items,
            "edges": edge_items,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "truncated": truncated or len(node_items) < len(nodes) or len(edge_items) < len(edges),
            "snapshot_id": hashlib.sha256(state.encode("utf-8")).hexdigest()[:24],
            "statistics": statistics,
        }
