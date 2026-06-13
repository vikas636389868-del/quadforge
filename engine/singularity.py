"""Singularity relocation and cancellation for QuadForge.

After the cross-field solve detects singularities via holonomy,
this module improves their placement:

  1. Pair Cancellation: Cancel unnecessary +1/4 / -1/4 pairs whose
     geodesic distance is below a threshold. Each cancellation reduces
     the number of irregular vertices in the output quad mesh by 2.

  2. Feature Corner Relocation: Move singularities to nearby feature
     corners where irregular vertices are expected and look natural
     (e.g. at the tip of a cone or corner of a box).

  3. Symmetric Pairing: When symmetry is active, ensure that
     singularities appear in symmetric pairs across the symmetry plane.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import heapq


def cancel_singularity_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    sing_vertices: np.ndarray,
    sing_indices: np.ndarray,
    max_cancel_distance_factor: float = 3.0,
    base_edge_length: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cancel unnecessary singularity pairs.

    Pairs a +1/4 singularity with a nearby -1/4 singularity and removes
    both. The result has fewer irregular vertices in the output mesh.

    Parameters
    ----------
    vertices : (V, 3)
    faces : (F, 3)
    sing_vertices : array of singular vertex indices
    sing_indices : array of singularity indices (e.g. +0.25, -0.25)
    max_cancel_distance_factor : max geodesic distance for cancellation,
        as a multiple of base_edge_length
    base_edge_length : typical edge length for distance scaling

    Returns
    -------
    new_sing_vertices : remaining singularities after cancellation
    new_sing_indices : their indices
    """
    if len(sing_vertices) < 2:
        return sing_vertices, sing_indices

    max_dist = max_cancel_distance_factor * base_edge_length

    # Separate positive and negative singularities
    pos_mask = sing_indices > 0
    neg_mask = sing_indices < 0

    pos_verts = sing_vertices[pos_mask]
    pos_idx = sing_indices[pos_mask]
    neg_verts = sing_vertices[neg_mask]
    neg_idx = sing_indices[neg_mask]

    if len(pos_verts) == 0 or len(neg_verts) == 0:
        return sing_vertices, sing_indices

    # Build vertex adjacency for geodesic distance approximation
    edge_adj = _build_vertex_adjacency(vertices, faces)

    # Find closest positive-negative pairs
    cancelled_pos: Set[int] = set()
    cancelled_neg: Set[int] = set()
    pairs_found = []

    for pi, pv in enumerate(pos_verts):
        if pi in cancelled_pos:
            continue
        best_ni = -1
        best_dist = max_dist

        for ni, nv in enumerate(neg_verts):
            if ni in cancelled_neg:
                continue
            # Approximate geodesic distance (Euclidean as proxy)
            dist = np.linalg.norm(vertices[pv] - vertices[nv])
            if dist < best_dist:
                best_dist = dist
                best_ni = ni

        if best_ni >= 0:
            pairs_found.append((pi, best_ni, best_dist))

    # Sort by distance — cancel closest pairs first
    pairs_found.sort(key=lambda x: x[2])

    for pi, ni, dist in pairs_found:
        if pi in cancelled_pos or ni in cancelled_neg:
            continue
        cancelled_pos.add(pi)
        cancelled_neg.add(ni)

    # Build remaining singularity lists
    remaining_verts = []
    remaining_idx = []

    for pi in range(len(pos_verts)):
        if pi not in cancelled_pos:
            remaining_verts.append(pos_verts[pi])
            remaining_idx.append(pos_idx[pi])

    for ni in range(len(neg_verts)):
        if ni not in cancelled_neg:
            remaining_verts.append(neg_verts[ni])
            remaining_idx.append(neg_idx[ni])

    n_cancelled = len(cancelled_pos)
    if n_cancelled > 0:
        print(f"[QuadForge] Singularity cancellation: {n_cancelled} pairs cancelled, "
              f"{len(remaining_verts)} singularities remaining")

    if not remaining_verts:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    return np.array(remaining_verts, dtype=np.int32), np.array(remaining_idx, dtype=np.float64)


def relocate_to_feature_corners(
    vertices: np.ndarray,
    faces: np.ndarray,
    sing_vertices: np.ndarray,
    sing_indices: np.ndarray,
    feature_edges: Set[Tuple[int, int]],
    relocation_radius_factor: float = 2.0,
    base_edge_length: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Relocate singularities to nearby feature corners.

    Feature corners are vertices where 3+ feature edges meet. These are
    natural locations for irregular vertices (e.g. valence 3 or 5), so
    placing singularities there produces better visual results.

    Parameters
    ----------
    vertices : (V, 3)
    faces : (F, 3)
    sing_vertices : current singularity vertex indices
    sing_indices : their indices
    feature_edges : set of feature edge keys
    relocation_radius_factor : max distance for relocation as multiple of edge length
    base_edge_length : typical edge length

    Returns
    -------
    relocated_vertices : singularity vertices (some may have moved)
    sing_indices : unchanged
    """
    if len(sing_vertices) == 0 or not feature_edges:
        return sing_vertices, sing_indices

    max_radius = relocation_radius_factor * base_edge_length

    # Find feature corners: vertices where 3+ feature edges meet
    vert_feature_count: Dict[int, int] = defaultdict(int)
    for v0, v1 in feature_edges:
        vert_feature_count[v0] += 1
        vert_feature_count[v1] += 1

    feature_corners = set(v for v, count in vert_feature_count.items() if count >= 3)

    if not feature_corners:
        return sing_vertices, sing_indices

    corner_positions = np.array([vertices[v] for v in sorted(feature_corners)], dtype=np.float64)
    corner_indices = np.array(sorted(feature_corners), dtype=np.int32)

    # For each singularity, check if a feature corner is nearby
    new_sing = sing_vertices.copy()
    used_corners: Set[int] = set()
    relocated_count = 0

    for si in range(len(sing_vertices)):
        sv = sing_vertices[si]
        sv_pos = vertices[sv]

        best_corner = -1
        best_dist = max_radius

        for ci in range(len(corner_indices)):
            cv = corner_indices[ci]
            if cv in used_corners:
                continue
            dist = np.linalg.norm(sv_pos - corner_positions[ci])
            if dist < best_dist:
                best_dist = dist
                best_corner = ci

        if best_corner >= 0:
            new_sing[si] = corner_indices[best_corner]
            used_corners.add(corner_indices[best_corner])
            relocated_count += 1

    if relocated_count > 0:
        print(f"[QuadForge] Singularity relocation: {relocated_count} moved to feature corners")

    return new_sing, sing_indices


def enforce_symmetric_singularities(
    vertices: np.ndarray,
    sing_vertices: np.ndarray,
    sing_indices: np.ndarray,
    symmetry_pairs: Dict[int, int],
    symmetry_axis: int,
    tolerance: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure singularities appear in symmetric pairs.

    For each singularity, if its mirror vertex is not also a singularity
    of the same type, add it as one.

    Parameters
    ----------
    vertices : (V, 3)
    sing_vertices : current singularity vertices
    sing_indices : their indices
    symmetry_pairs : vertex → mirror vertex mapping
    symmetry_axis : 0=X, 1=Y, 2=Z
    tolerance : plane tolerance for on-plane singularities

    Returns
    -------
    augmented_vertices : singularities with symmetric additions
    augmented_indices : their indices
    """
    if len(sing_vertices) == 0 or not symmetry_pairs:
        return sing_vertices, sing_indices

    sing_set = set(int(v) for v in sing_vertices)
    sing_map = {int(v): float(idx) for v, idx in zip(sing_vertices, sing_indices)}

    new_verts = list(sing_vertices)
    new_idx = list(sing_indices)

    for sv, si in zip(sing_vertices, sing_indices):
        sv = int(sv)
        # Check if on symmetry plane
        if abs(vertices[sv, symmetry_axis]) < tolerance:
            continue  # On-plane singularities don't need mirrors

        mirror = symmetry_pairs.get(sv)
        if mirror is not None and mirror not in sing_set:
            new_verts.append(mirror)
            new_idx.append(si)  # Same type as original
            sing_set.add(mirror)

    if len(new_verts) > len(sing_vertices):
        added = len(new_verts) - len(sing_vertices)
        print(f"[QuadForge] Symmetric singularities: {added} mirror singularities added")

    return np.array(new_verts, dtype=np.int32), np.array(new_idx, dtype=np.float64)


def _build_vertex_adjacency(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[int, List[Tuple[int, float]]]:
    """Build vertex adjacency with edge lengths."""
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    faces = np.asarray(faces, dtype=np.int32)

    for fi in range(len(faces)):
        tri = faces[fi]
        for i in range(3):
            v0, v1 = int(tri[i]), int(tri[(i + 1) % 3])
            dist = float(np.linalg.norm(vertices[v1] - vertices[v0]))
            adj[v0].append((v1, dist))
            adj[v1].append((v0, dist))

    return adj


def optimise_singularities(
    vertices: np.ndarray,
    faces: np.ndarray,
    sing_vertices: np.ndarray,
    sing_indices: np.ndarray,
    feature_edges: Optional[Set[Tuple[int, int]]] = None,
    symmetry_pairs: Optional[Dict[int, int]] = None,
    symmetry_axis: Optional[int] = None,
    base_edge_length: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full singularity optimisation pipeline.

    1. Cancel unnecessary pairs
    2. Relocate to feature corners
    3. Enforce symmetry

    Parameters
    ----------
    vertices, faces : mesh geometry
    sing_vertices, sing_indices : detected singularities
    feature_edges : optional feature edge set
    symmetry_pairs : optional vertex mirror map
    symmetry_axis : optional axis for symmetry enforcement
    base_edge_length : typical edge length

    Returns
    -------
    optimised_vertices, optimised_indices
    """
    sv, si = sing_vertices, sing_indices

    print(f"[QuadForge] Singularity optimisation: {len(sv)} initial singularities")

    # Step 1: Cancel pairs
    sv, si = cancel_singularity_pairs(
        vertices, faces, sv, si,
        base_edge_length=base_edge_length,
    )

    # Step 2: Relocate to feature corners
    if feature_edges:
        sv, si = relocate_to_feature_corners(
            vertices, faces, sv, si,
            feature_edges, base_edge_length=base_edge_length,
        )

    # Step 3: Enforce symmetry
    if symmetry_pairs and symmetry_axis is not None:
        sv, si = enforce_symmetric_singularities(
            vertices, sv, si,
            symmetry_pairs, symmetry_axis,
        )

    print(f"[QuadForge] Singularity optimisation: {len(sv)} final singularities")
    return sv, si
