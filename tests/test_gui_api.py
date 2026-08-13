"""HTTP contract tests for the NEXUS visual API."""

import pytest
from fastapi.testclient import TestClient

from nexus_gui.api import create_app
from nexus_gui.service import NexusGuiService
from nexus_knowledge.service.explorer import KnowledgeExplorer
from nexus_runtime.distributed.coordinator import Coordinator
from nexus_runtime.distributed.service import RuntimeApplication
from nexus_runtime.distributed.store import InMemoryTaskStore
from nexus_runtime.investigation.application import InvestigationApplication
from nexus_runtime.investigation.repository import InMemoryInvestigationRepository
from nexus_runtime.monitoring import RuntimeMonitor


@pytest.fixture
def gui_client(ingested_engine):
    investigations = InMemoryInvestigationRepository()
    runtime = RuntimeApplication(Coordinator(InMemoryTaskStore()))
    service = NexusGuiService(
        KnowledgeExplorer(ingested_engine),
        RuntimeMonitor(investigations, runtime),
        InvestigationApplication(ingested_engine, repository=investigations),
    )
    with TestClient(create_app(service)) as client:
        yield client


def test_health_graph_search_and_node_endpoints(gui_client):
    health = gui_client.get("/api/health")
    graph = gui_client.get("/api/graph", params={"max_nodes": 12, "max_edges": 20})

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert graph.status_code == 200
    payload = graph.json()
    assert len(payload["nodes"]) <= 12
    assert len(payload["edges"]) <= 20
    node_id = payload["nodes"][0]["id"]
    assert gui_client.get(f"/api/nodes/{node_id}").status_code == 200
    assert gui_client.get("/api/search", params={"q": "Acme"}).json()
    assert gui_client.get("/api/sources").json()


def test_graph_validation_and_not_found_contracts(gui_client):
    assert gui_client.get("/api/graph", params={"kinds": "fiction"}).status_code == 422
    assert gui_client.get("/api/graph", params={"max_nodes": 2_001}).status_code == 422
    missing = gui_client.get("/api/nodes/not-a-real-node")
    assert missing.status_code == 404
    assert "unknown graph node" in missing.json()["detail"]


def test_local_graph_is_bounded_and_contains_focus(gui_client):
    global_graph = gui_client.get("/api/graph").json()
    focus = next(node for node in global_graph["nodes"] if node["kind"] == "entity")

    response = gui_client.get(
        f"/api/graph/neighborhood/{focus['id']}",
        params={"depth": 2, "max_nodes": 10, "max_edges": 15},
    )

    assert response.status_code == 200
    payload = response.json()
    assert focus["id"] in {node["id"] for node in payload["nodes"]}
    assert len(payload["nodes"]) <= 10
    assert len(payload["edges"]) <= 15


def test_gap_action_creates_real_investigation_and_runtime_overlay(gui_client):
    gaps = gui_client.get("/api/gaps").json()
    gap_id = gaps[0]["id"]

    created = gui_client.post(
        f"/api/gaps/{gap_id}/investigations",
        json={"question": "Resolve this evidence gap with two independent sources"},
    )

    assert created.status_code == 201
    investigation = created.json()
    assert gap_id in investigation["target_gap_ids"]
    assert gui_client.get("/api/investigations").json()[0]["session_id"] == investigation[
        "session_id"
    ]
    graph = gui_client.get("/api/graph").json()
    assert investigation["session_id"] in {node["id"] for node in graph["nodes"]}
    assert any(
        edge["kind"] == "investigation_target" and edge["target"] == gap_id
        for edge in graph["edges"]
    )
    assert gui_client.get("/api/runtime").json()["available"] is True
