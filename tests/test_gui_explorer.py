"""Read-model tests for the heterogeneous GUI knowledge projection."""

from nexus_knowledge.service.explorer import ExplorerFilters, KnowledgeExplorer


def test_graph_projection_is_heterogeneous_and_referentially_sound(ingested_engine):
    explorer = KnowledgeExplorer(ingested_engine)

    graph = explorer.graph()

    assert {"entity", "claim", "evidence", "document", "source", "gap"} <= {
        node.kind for node in graph.nodes
    }
    node_ids = {node.id for node in graph.nodes}
    assert all(edge.source in node_ids and edge.target in node_ids for edge in graph.edges)
    assert any(edge.kind == "provenance" for edge in graph.edges)
    assert graph.snapshot_id == explorer.graph().snapshot_id


def test_graph_slices_by_real_relation_predicate_and_bounds(ingested_engine):
    explorer = KnowledgeExplorer(ingested_engine)
    predicate = ingested_engine.repository.relations.all()[0].predicate

    filtered = explorer.graph(
        ExplorerFilters(
            node_kinds=frozenset({"entity"}),
            relation_types=frozenset({predicate}),
            max_nodes=100,
            max_edges=100,
        )
    )
    bounded = explorer.graph(ExplorerFilters(max_nodes=4, max_edges=2))

    assert filtered.edges
    assert {edge.kind for edge in filtered.edges} == {predicate}
    assert {node.kind for node in filtered.nodes} == {"entity"}
    assert len(bounded.nodes) <= 4
    assert len(bounded.edges) <= 2
    assert bounded.truncated is True
    assert bounded.total_nodes > len(bounded.nodes)


def test_neighborhood_search_and_claim_inspector_use_domain_data(ingested_engine):
    explorer = KnowledgeExplorer(ingested_engine)
    claim = ingested_engine.repository.claims.all()[0]

    neighborhood = explorer.neighborhood(claim.id, depth=1)
    results = explorer.search(claim.subject)
    detail = explorer.node_detail(claim.id)

    assert claim.id in {node.id for node in neighborhood.nodes}
    assert any(result["id"] == claim.id for result in results)
    assert detail.node.id == claim.id
    assert detail.attributes["text"] == claim.text
    assert detail.evidence
    assert detail.provenance["source_references"]
    assert detail.history


def test_contradictions_become_edges_and_first_class_records(ingested_engine):
    first = ingested_engine.propose_claim(
        "Acme Corp is headquartered in London",
        "Acme Corp",
        "headquartered_in",
        "London",
        confidence=0.9,
    )
    second = ingested_engine.propose_claim(
        "Acme Corp is headquartered in Paris",
        "Acme Corp",
        "headquartered_in",
        "Paris",
        confidence=0.8,
    )
    explorer = KnowledgeExplorer(ingested_engine)

    contradictions = explorer.contradictions()
    graph = explorer.graph()

    assert any(
        second.id in {item["claim_a_id"], item["claim_b_id"]}
        for item in contradictions
    )
    assert any(
        edge.kind == "contradiction" and second.id in {edge.source, edge.target}
        for edge in graph.edges
    )
    assert first.id in {node.id for node in graph.nodes}


def test_gap_ids_are_stable_for_polling_and_gap_detail_is_available(ingested_engine):
    explorer = KnowledgeExplorer(ingested_engine)

    first = explorer.gaps()
    second = explorer.gaps()
    detail = explorer.node_detail(first[0]["id"])

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert detail.node.kind == "gap"
    assert detail.gaps[0]["id"] == first[0]["id"]
