"""Graph algorithms implemented over a plain adjacency structure.

``Adjacency`` is ``dict[node_id, list[(neighbor_id, weight)]]`` for the
outgoing edges of each node. The algorithms are backend-agnostic: any
graph storage implementing :class:`KnowledgeGraph` can feed its
adjacency here.

All algorithms are deterministic.
"""

from __future__ import annotations

from collections import defaultdict, deque

Adjacency = dict[str, list[tuple[str, float]]]

__all__ = [
    "pagerank",
    "personalized_pagerank",
    "degree_centrality",
    "betweenness_centrality",
    "connected_components",
    "label_propagation_communities",
    "enumerate_paths",
    "density",
    "average_degree",
]


def _nodes(adj: Adjacency) -> list[str]:
    nodes: set[str] = set(adj)
    for neighbors in adj.values():
        for neighbor, _ in neighbors:
            nodes.add(neighbor)
    return sorted(nodes)


def _in_adjacency(adj: Adjacency, nodes: list[str]) -> dict[str, list[str]]:
    incoming: dict[str, list[str]] = {n: [] for n in nodes}
    for node, neighbors in adj.items():
        for neighbor, _ in neighbors:
            incoming[neighbor].append(node)
    return incoming


def pagerank(
    adj: Adjacency,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    personalization: dict[str, float] | None = None,
) -> dict[str, float]:
    """PageRank over the weighted directed adjacency.

    ``personalization`` biases the random restart towards a subset of
    nodes (used for personalized PageRank).
    """
    nodes = _nodes(adj)
    n = len(nodes)
    if n == 0:
        return {}
    index = {node: i for i, node in enumerate(nodes)}

    out_degree = [0] * n
    edges: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for node, neighbors in adj.items():
        i = index[node]
        out_degree[i] = len(neighbors)
        total = sum(abs(weight) for _, weight in neighbors) or 1.0
        for neighbor, weight in neighbors:
            edges[i].append((index[neighbor], max(0.0, weight) / total))

    if personalization:
        total = sum(personalization.values()) or 1.0
        base = [(personalization.get(node, 0.0) / total) for node in nodes]
    else:
        base = [1.0 / n] * n

    rank = [1.0 / n] * n
    for _ in range(max_iter):
        new_rank = [0.0] * n
        dangling_mass = sum(rank[i] for i in range(n) if out_degree[i] == 0)
        for i in range(n):
            new_rank[i] = (1.0 - damping) * base[i] + damping * dangling_mass / n
        for i in range(n):
            if out_degree[i] == 0:
                continue
            mass = damping * rank[i]
            for j, weight in edges[i]:
                new_rank[j] += mass * weight
        diff = sum(abs(a - b) for a, b in zip(rank, new_rank))
        rank = new_rank
        if diff < tol:
            break
    return {nodes[i]: rank[i] for i in range(n)}


def personalized_pagerank(
    adj: Adjacency,
    seed_ids: list[str],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Personalized PageRank seeded on ``seed_ids``."""
    seeds = {node: 1.0 for node in seed_ids}
    return pagerank(adj, damping=damping, max_iter=max_iter, tol=tol, personalization=seeds)


def degree_centrality(adj: Adjacency) -> dict[str, float]:
    """Normalized total degree (in + out) per node."""
    nodes = _nodes(adj)
    out_count = {n: len(neighbors) for n, neighbors in adj.items()}
    in_count: dict[str, int] = defaultdict(int)
    for neighbors in adj.values():
        for neighbor, _ in neighbors:
            in_count[neighbor] += 1
    maximum = max(len(nodes) - 1, 1)
    return {node: (out_count.get(node, 0) + in_count[node]) / maximum for node in nodes}


def betweenness_centrality(adj: Adjacency) -> dict[str, float]:
    """Brandes betweenness centrality for the (unweighted) directed graph.

    Returns a dict mapping node id to normalized centrality.
    """
    nodes = _nodes(adj)
    outgoing = {node: [nb for nb, _ in adj.get(node, [])] for node in nodes}
    centrality = defaultdict(float)
    for source in nodes:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {n: [] for n in nodes}
        sigma: dict[str, float] = {n: 0.0 for n in nodes}
        sigma[source] = 1.0
        distance: dict[str, int] = {n: -1 for n in nodes}
        distance[source] = 0
        queue: deque[str] = deque([source])
        while queue:
            node = queue.popleft()
            stack.append(node)
            for neighbor in outgoing.get(node, []):
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
                if distance[neighbor] == distance[node] + 1:
                    sigma[neighbor] += sigma[node]
                    predecessors[neighbor].append(node)
        dependency: dict[str, float] = defaultdict(float)
        while stack:
            node = stack.pop()
            for pred in predecessors[node]:
                dependency[pred] += (sigma[pred] / sigma[node]) * (1.0 + dependency[node])
            if node != source:
                centrality[node] += dependency[node]
    maximum = max((len(nodes) - 1) * (len(nodes) - 2), 1)
    return {node: centrality[node] / maximum for node in nodes}


def connected_components(adj: Adjacency) -> list[set[str]]:
    """Weakly connected components (undirected reachability)."""
    nodes = _nodes(adj)
    neighbors: dict[str, set[str]] = {n: set() for n in nodes}
    for node, edges in adj.items():
        for neighbor, _ in edges:
            neighbors[node].add(neighbor)
            neighbors[neighbor].add(node)
    seen: set[str] = set()
    components: list[set[str]] = []
    for node in nodes:
        if node in seen:
            continue
        component: set[str] = set()
        queue = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            component.add(current)
            for nb in neighbors[current]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        components.append(component)
    return components


def label_propagation_communities(adj: Adjacency, max_iter: int = 20) -> dict[str, int]:
    """Deterministic label-propagation community detection."""
    nodes = _nodes(adj)
    neighbors: dict[str, list[str]] = {n: [] for n in nodes}
    for node, edges in adj.items():
        for neighbor, _ in edges:
            if neighbor not in neighbors[node]:
                neighbors[node].append(neighbor)
            if node not in neighbors[neighbor]:
                neighbors[neighbor].append(node)

    label = {node: i for i, node in enumerate(nodes)}
    for _ in range(max_iter):
        changed = False
        for node in nodes:
            counter: dict[int, int] = defaultdict(int)
            for nb in neighbors[node]:
                counter[label[nb]] += 1
            if not counter:
                continue
            best = min(counter, key=lambda k: (-counter[k], k))
            if best != label[node]:
                label[node] = best
                changed = True
        if not changed:
            break
    # canonicalize community ids
    community_map = {label[node] for node in nodes}
    canonical = {old: i for i, old in enumerate(sorted(community_map))}
    return {node: canonical[label[node]] for node in nodes}


def enumerate_paths(
    adj: Adjacency,
    source: str,
    target: str,
    max_length: int = 4,
    max_paths: int = 20,
) -> list[list[str]]:
    """Simple DFS paths (no repeated nodes) from source to target."""
    results: list[list[str]] = []

    def dfs(current: str, visited: list[str]) -> None:
        if len(results) >= max_paths:
            return
        if current == target:
            results.append(list(visited))
            return
        if len(visited) > max_length:
            return
        for neighbor, _ in adj.get(current, []):
            if neighbor not in visited:
                visited.append(neighbor)
                dfs(neighbor, visited)
                visited.pop()

    dfs(source, [source])
    return results


def density(adj: Adjacency) -> float:
    """Edge density of the directed graph."""
    n = len(_nodes(adj))
    if n <= 1:
        return 0.0
    edge_count = sum(len(neighbors) for neighbors in adj.values())
    return edge_count / (n * (n - 1))


def average_degree(adj: Adjacency) -> float:
    """Mean total degree (in + out) over all nodes."""
    nodes = _nodes(adj)
    if not nodes:
        return 0.0
    out_count = {n: len(neighbors) for n, neighbors in adj.items()}
    in_count: dict[str, int] = defaultdict(int)
    for neighbors in adj.values():
        for neighbor, _ in neighbors:
            in_count[neighbor] += 1
    return sum(out_count.get(n, 0) + in_count[n] for n in nodes) / len(nodes)
