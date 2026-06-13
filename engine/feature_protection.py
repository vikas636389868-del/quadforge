"""Feature-aware filtering for quad candidates."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

Edge = Tuple[int, int]
Quad = Sequence[int]


def _edge_key(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _quad_edges(quad: Quad) -> Set[Edge]:
    if len(quad) != 4:
        return set()
    return {_edge_key(quad[i], quad[(i + 1) % 4]) for i in range(4)}


def filter_quads_by_features(quads: Iterable[Quad], feature_edges: Set[Edge]) -> List[list[int]]:
    """Keep quads that do not cut across protected feature edges.

    A quad is rejected if it contains both endpoints of a feature edge, but
    that feature edge is not one of the quad boundary edges.
    """
    if not feature_edges:
        return [list(q) for q in quads]

    protected = {_edge_key(a, b) for a, b in feature_edges}
    filtered: List[list[int]] = []

    for quad in quads:
        if len(quad) != 4:
            continue

        quad_list = list(quad)
        q_edges = _quad_edges(quad_list)
        q_vertices = set(quad_list)

        if any(e in q_edges for e in protected):
            # Feature-aligned quads are allowed.
            filtered.append(quad_list)
            continue

        # Conservative test: reject if the quad spans any protected edge endpoints
        if any(a in q_vertices and b in q_vertices for a, b in protected):
            continue

        filtered.append(quad_list)

    return filtered
