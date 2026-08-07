"""Graph abstraction and in-memory backend tests."""

import pytest

from nexus_knowledge.domain.entity import Entity, Relation
from nexus_knowledge.graph.memory import InMemoryGraph


def _graph_with_data():
    graph = InMemoryGraph()
    entities = [
        Entity(name="Ada", id="ada"),
        Entity(name="Acme", id="acme"),
        Entity(name="Alan", id="alan"),
        Entity(name="London", id="london"),
        Entity(name="Alone", id="alone"),
    ]
    for entity in entities:
        graph.add_entity(entity)
    relations = [
        Relation(subject_id="ada", predicate="works_at", object_id="acme", id="r1"),
        Relation(subject_id="alan", predicate="works_at", object_id="acme", id="r2"),
        Relation(subject_id="acme", predicate="located_in", object_id="london", id="r3"),
        Relation(subject_id="acme", predicate="develops", object_id="acme", id="r4"),
    ]
    for relation in relations:
        graph.add_relation(relation)
    return graph


class TestEntities:
    def test_add_and_get(self):
        graph = _graph_with_data()
        assert graph.get_entity("ada").name == "Ada"
        assert graph.get_entity("nope") is None

    def test_add_is_idempotent(self):
        graph = InMemoryGraph()
        entity = Entity(name="X", id="x")
        assert graph.add_entity(entity) is graph.add_entity(entity)
        assert len(graph) == 1

    def test_update_and_remove(self):
        graph = _graph_with_data()
        updated = Entity(name="Ada Lovelace", id="ada")
        graph.update_entity(updated)
        assert graph.get_entity("ada").name == "Ada Lovelace"
        assert graph.remove_entity("london") is True
        assert graph.get_entity("london") is None

    def test_remove_entity_removes_incident_relations(self):
        graph = _graph_with_data()
        graph.remove_entity("acme")
        remaining = [r for r in graph.all_relations()]
        assert remaining == []


class TestRelations:
    def test_add_and_dedup_by_tuple(self):
        graph = InMemoryGraph()
        graph.add_entity(Entity(name="A", id="a"))
        graph.add_entity(Entity(name="B", id="b"))
        first = Relation(subject_id="a", predicate="p", object_id="b", id="r1")
        second = Relation(subject_id="a", predicate="p", object_id="b", id="r2")
        result = graph.add_relation(first)
        result = graph.add_relation(second)
        assert result.id == first.id  # merged into the existing relation
        assert len(graph.all_relations()) == 1

    def test_remove_relation(self):
        graph = _graph_with_data()
        assert graph.remove_relation("r1") is True
        assert graph.get_relation("r1") is None
        assert graph.remove_relation("r1") is False


class TestTraversal:
    def test_neighbors_out_in_both(self):
        graph = _graph_with_data()
        assert {e.object_id for e in graph.neighbors("ada", "out")} == {"acme"}
        assert {e.subject_id for e in graph.neighbors("london", "in")} == {"acme"}
        assert len(graph.neighbors("acme", "both")) >= 3

    def test_traversal_bounded(self):
        graph = _graph_with_data()
        visited, edges = graph.traversal(["ada"], depth=2)
        assert "acme" in visited
        assert "london" in visited
        assert len(edges) >= 2

    def test_paths_found(self):
        graph = _graph_with_data()
        paths = graph.paths("ada", "london", max_length=3)
        assert any(p.nodes == ("ada", "acme", "london") for p in paths)

    def test_subgraph(self):
        graph = _graph_with_data()
        subgraph = graph.subgraph(["ada"], depth=2)
        assert {"ada", "acme", "london", "alan"} <= set(subgraph.entity_ids())

    def test_invalid_direction(self):
        graph = _graph_with_data()
        with pytest.raises(ValueError):
            graph.neighbors("ada", "sideways")


class TestAnalytics:
    def test_pagerank_is_probability_distribution(self):
        graph = _graph_with_data()
        ranks = graph.pagerank()
        assert abs(sum(ranks.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in ranks.values())

    def test_pagerank_isolated_node_gets_base_mass(self):
        graph = _graph_with_data()
        ranks = graph.pagerank()
        assert ranks["alone"] > 0

    def test_personalized_pagerank_prefers_seeds(self):
        graph = _graph_with_data()
        ppr = graph.personalized_pagerank(["ada"])
        assert ppr["ada"] > ppr["alone"]
        assert ppr["ada"] > ppr["alan"]

    def test_degree_centrality(self):
        graph = _graph_with_data()
        centrality = graph.degree_centrality()
        assert centrality["acme"] > centrality["alone"]

    def test_betweenness_centrality_deterministic(self):
        graph = _graph_with_data()
        assert graph.betweenness_centrality() == graph.betweenness_centrality()

    def test_communities_deterministic(self):
        graph = _graph_with_data()
        communities = graph.communities()
        assert set(communities) == {e.id for e in graph.all_entities()}

    def test_statistics(self):
        graph = _graph_with_data()
        stats = graph.statistics()
        assert stats.num_entities == 5
        assert stats.num_relations == 4
        assert stats.num_components >= 2
        assert stats.num_isolated >= 1
