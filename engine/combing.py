"""Cross-field combing for QuadForge.

Combs the 4-RoSy cross-field to produce a consistent 2-direction field
(choosing one arm of the cross per face) across the surface. Where
consistency breaks, seam edges are introduced.

Algorithm:
  1. BFS from a seed face across the mesh
  2. At each edge, choose the matching (out of 4 rotations) that best
     aligns the neighbor's field with the current face's combed direction
  3. Edges where no good match exists become seam edges
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Set, Tuple


def comb_field(
    num_faces: int,
    field_angles: np.ndarray,
    face_frames_e1: np.ndarray,
    face_frames_e2: np.ndarray,
    face_normals: np.ndarray,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    transport_angles: Dict[Tuple[int, int], float],
    feature_edges: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Set[Tuple[int, int]]]:
    """Comb the 4-RoSy field into a consistent 2-direction field.

    Parameters
    ----------
    num_faces : int
    field_angles : (F,) field angle per face in local frame
    face_frames_e1, face_frames_e2, face_normals : (F, 3) per-face frames
    edge_to_faces : edge → face list adjacency
    transport_angles : edge → parallel transport angle
    feature_edges : optional set of feature edge keys

    Returns
    -------
    combed_angles : (F,) combed field angles (consistent orientation)
    period_jumps : (E_seam,) integer period jump at each seam edge
    seam_edges : set of edge keys where seams were cut
    """
    feature_edges = feature_edges or set()

    # Build face adjacency (face_i → [(face_j, edge_key), ...])
    face_adj: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {}
    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fi, fj = face_list
        face_adj.setdefault(fi, []).append((fj, edge_key))
        face_adj.setdefault(fj, []).append((fi, edge_key))

    combed = np.copy(field_angles)
    visited = np.zeros(num_faces, dtype=bool)
    # Per-face rotation index k ∈ {0,1,2,3}: how many π/2 rotations applied
    rotation_k = np.zeros(num_faces, dtype=np.int32)

    seam_edges: Set[Tuple[int, int]] = set()

    # BFS from face 0
    queue = deque()
    seed = 0
    visited[seed] = True
    queue.append(seed)

    while queue:
        fi = queue.popleft()
        theta_i = combed[fi]

        for fj, edge_key in face_adj.get(fi, []):
            phi = transport_angles.get(edge_key, 0.0)
            theta_j_raw = field_angles[fj]

            if not visited[fj]:
                # Find best k ∈ {0,1,2,3} such that
                # theta_j_raw + k*π/2 best matches theta_i + phi
                target = theta_i + phi
                best_k = 0
                best_diff = float('inf')

                for k in range(4):
                    candidate = theta_j_raw + k * np.pi / 2.0
                    diff = candidate - target
                    # Wrap to [-π, π]
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                    if abs(diff) < best_diff:
                        best_diff = abs(diff)
                        best_k = k

                combed[fj] = theta_j_raw + best_k * np.pi / 2.0
                rotation_k[fj] = best_k
                visited[fj] = True
                queue.append(fj)
            else:
                # Already visited — check consistency
                theta_j_combed = combed[fj]
                expected = theta_i + phi
                diff = theta_j_combed - expected
                diff = (diff + np.pi) % (2 * np.pi) - np.pi

                # If not close to a multiple of π/2, this is a seam
                k_jump = round(diff / (np.pi / 2.0))
                residual = abs(diff - k_jump * np.pi / 2.0)

                if residual > 0.1 or abs(k_jump) > 0:
                    seam_edges.add(edge_key)

    # Handle unvisited faces (disconnected components)
    for fi in range(num_faces):
        if not visited[fi]:
            visited[fi] = True

    return combed, rotation_k, seam_edges


def compute_seam_cut(
    num_faces: int,
    seam_edges: Set[Tuple[int, int]],
    singularity_vertices: np.ndarray,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Set[Tuple[int, int]]:
    """Compute a minimal seam-cut graph connecting all singularities.

    The cut graph ensures the surface is topologically a disk (for
    genus-0) by connecting singularity vertices through shortest paths
    on the dual graph.

    Parameters
    ----------
    num_faces : int
    seam_edges : initial seam edges from combing
    singularity_vertices : array of singular vertex indices
    edge_to_faces : adjacency
    vertices, faces : mesh data

    Returns
    -------
    final_seams : set of edge keys forming the complete seam cut
    """
    final_seams = set(seam_edges)

    if len(singularity_vertices) < 2:
        return final_seams

    # Build vertex adjacency with edge weights (edge lengths)
    vert_adj: Dict[int, List[Tuple[int, float, Tuple[int, int]]]] = {}
    for edge_key in edge_to_faces:
        v0, v1 = edge_key
        dist = float(np.linalg.norm(vertices[v1] - vertices[v0]))
        vert_adj.setdefault(v0, []).append((v1, dist, edge_key))
        vert_adj.setdefault(v1, []).append((v0, dist, edge_key))

    # Connect singularities via shortest paths (Dijkstra)
    sing_set = set(int(v) for v in singularity_vertices)

    # Simple approach: connect each singularity to the nearest one
    connected = set()
    if len(sing_set) > 0:
        start = next(iter(sing_set))
        connected.add(start)

        while connected != sing_set:
            best_path = None
            best_dist = float('inf')

            for src in connected:
                path, dist = _dijkstra_to_set(
                    src, sing_set - connected, vert_adj, len(vertices)
                )
                if path is not None and dist < best_dist:
                    best_path = path
                    best_dist = dist

            if best_path is None:
                break

            # Add path edges to seams
            for i in range(len(best_path) - 1):
                v0, v1 = best_path[i], best_path[i + 1]
                ek = (min(v0, v1), max(v0, v1))
                final_seams.add(ek)
            connected.add(best_path[-1])

    return final_seams


def _dijkstra_to_set(
    src: int,
    targets: Set[int],
    vert_adj: Dict[int, List[Tuple[int, float, Tuple[int, int]]]],
    num_verts: int,
) -> Tuple[Optional[List[int]], float]:
    """Dijkstra from src to nearest vertex in targets."""
    import heapq

    dist = np.full(num_verts, np.inf, dtype=np.float64)
    prev = np.full(num_verts, -1, dtype=np.int32)
    dist[src] = 0.0
    heap = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u in targets and u != src:
            # Reconstruct path
            path = []
            v = u
            while v != -1:
                path.append(v)
                v = int(prev[v])
            return list(reversed(path)), d
        for v, w, _ in vert_adj.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return None, float('inf')
