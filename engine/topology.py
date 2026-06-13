"""Topology analysis for QuadForge meshes.

Provides manifold checking, Euler characteristic, genus computation,
boundary detection, and connected component enumeration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Set, Tuple

from .halfedge import HalfEdgeMesh, NONE


@dataclass
class TopologyInfo:
    """Complete topological analysis of a mesh."""
    num_vertices: int
    num_edges: int
    num_faces: int
    num_boundary_loops: int
    boundary_loops: List[List[int]]
    euler_characteristic: int        # V - E + F
    genus: int                       # (2 - χ - B) / 2 for orientable surfaces
    num_components: int
    components: List[Set[int]]       # sets of face indices per component
    is_manifold: bool
    is_closed: bool                  # no boundary
    non_manifold_edges: List[Tuple[int, int]]
    non_manifold_vertices: List[int]
    isolated_vertices: List[int]


def analyze_topology(mesh: HalfEdgeMesh) -> TopologyInfo:
    """Run a full topological analysis on the given half-edge mesh."""
    V = mesh.num_vertices
    E = mesh.num_edges
    F = mesh.num_faces

    # --- Boundary loops ---
    boundary_loops = mesh.boundary_loops()
    B = len(boundary_loops)

    # --- Euler characteristic ---
    chi = V - E + F

    # --- Genus (for orientable surface with boundaries) ---
    # χ = 2 - 2g - B  →  g = (2 - χ - B) / 2
    genus = max(0, (2 - chi - B) // 2)

    # --- Connected components (face-based BFS) ---
    # Build face adjacency from the half-edge structure
    face_adj: list[set] = [set() for _ in range(F)]
    for he in range(mesh.num_halfedges):
        fi = mesh.he_face(he)
        if fi == NONE:
            continue
        twin = mesh.he_twin(he)
        if twin == NONE:
            continue
        fj = mesh.he_face(twin)
        if fj == NONE:
            continue
        if fi != fj:
            face_adj[fi].add(fj)
            face_adj[fj].add(fi)

    visited_faces: set = set()
    components: List[Set[int]] = []

    for seed in range(F):
        if seed in visited_faces:
            continue
        comp: Set[int] = set()
        stack = [seed]
        while stack:
            f = stack.pop()
            if f in visited_faces:
                continue
            visited_faces.add(f)
            comp.add(f)
            for nb in face_adj[f]:
                if nb not in visited_faces:
                    stack.append(nb)
        if comp:
            components.append(comp)

    # --- Non-manifold detection ---
    # An edge is non-manifold if more than 2 faces share it.
    # Build edge → face count from face lists
    edge_face_count: dict[Tuple[int, int], int] = {}
    for fi, fv in enumerate(mesh._face_lists):
        n = len(fv)
        for i in range(n):
            ek = (min(fv[i], fv[(i + 1) % n]), max(fv[i], fv[(i + 1) % n]))
            edge_face_count[ek] = edge_face_count.get(ek, 0) + 1

    non_manifold_edges = [ek for ek, cnt in edge_face_count.items() if cnt > 2]

    # A vertex is non-manifold if its face fan is disconnected (in the
    # combinatorial sense). Simple heuristic: check if vertex one-ring
    # from the half-edge structure has fewer neighbors than expected.
    # For a thorough check we'd do a fan-connectivity BFS; here we use
    # a simpler criterion — vertex appears in more faces than its valence + 1
    # (works for most practical cases).
    vert_face_count = np.zeros(V, dtype=np.int32)
    for fi, fv in enumerate(mesh._face_lists):
        for v in fv:
            vert_face_count[v] += 1

    non_manifold_vertices: List[int] = []
    for vi in range(V):
        val = mesh.vertex_valence(vi)
        is_bnd = mesh.is_boundary_vertex(vi)
        expected_max = val if is_bnd else val
        # Non-manifold if face count exceeds valence (shouldn't happen on manifold)
        if vert_face_count[vi] > expected_max + 1 and val > 0:
            non_manifold_vertices.append(vi)

    # Isolated vertices (no incident faces)
    isolated = [vi for vi in range(V) if vert_face_count[vi] == 0]

    is_manifold = len(non_manifold_edges) == 0 and len(non_manifold_vertices) == 0
    is_closed = B == 0

    return TopologyInfo(
        num_vertices=V,
        num_edges=E,
        num_faces=F,
        num_boundary_loops=B,
        boundary_loops=boundary_loops,
        euler_characteristic=chi,
        genus=genus,
        num_components=len(components),
        components=components,
        is_manifold=is_manifold,
        is_closed=is_closed,
        non_manifold_edges=non_manifold_edges,
        non_manifold_vertices=non_manifold_vertices,
        isolated_vertices=isolated,
    )


def print_topology_report(info: TopologyInfo) -> str:
    """Return a human-readable topology report string."""
    lines = [
        f"Topology Report",
        f"  Vertices:      {info.num_vertices}",
        f"  Edges:         {info.num_edges}",
        f"  Faces:         {info.num_faces}",
        f"  Euler char:    {info.euler_characteristic}",
        f"  Genus:         {info.genus}",
        f"  Boundary loops:{info.num_boundary_loops}",
        f"  Components:    {info.num_components}",
        f"  Manifold:      {info.is_manifold}",
        f"  Closed:        {info.is_closed}",
    ]
    if info.non_manifold_edges:
        lines.append(f"  Non-manifold edges: {len(info.non_manifold_edges)}")
    if info.non_manifold_vertices:
        lines.append(f"  Non-manifold verts: {len(info.non_manifold_vertices)}")
    if info.isolated_vertices:
        lines.append(f"  Isolated verts:     {len(info.isolated_vertices)}")
    return "\n".join(lines)
