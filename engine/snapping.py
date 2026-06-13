"""Feature curve snapping for QuadForge.

Snaps output mesh vertices near detected feature curves onto those curves.
This preserves sharp creases, material boundaries, and UV seams that would
otherwise be smoothed away by the post-processing stage.

Algorithm:
  1. Build polyline chains from feature edges
  2. For each output vertex within snap distance of a chain, project
     it onto the nearest point on that chain
  3. Apply tangential sliding so the vertex sits at the closest
     position along the curve
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from .spatial import KDTree


def build_feature_chains(
    feature_edges: Set[Tuple[int, int]],
    vertices: np.ndarray,
) -> List[List[int]]:
    """Assemble feature edges into ordered polyline chains.

    Each chain is a list of vertex indices forming a connected feature curve.
    """
    if not feature_edges:
        return []

    # Build adjacency for feature graph
    adj: Dict[int, List[int]] = {}
    for v0, v1 in feature_edges:
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    visited_edges: Set[Tuple[int, int]] = set()
    chains: List[List[int]] = []

    for start_v in sorted(adj.keys()):
        for nb in adj[start_v]:
            ek = (min(start_v, nb), max(start_v, nb))
            if ek in visited_edges:
                continue

            # Trace chain from start_v through nb
            chain = [start_v]
            current = start_v
            next_v = nb

            while True:
                ek = (min(current, next_v), max(current, next_v))
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                chain.append(next_v)
                current = next_v

                # Continue to next unvisited neighbor
                neighbors = adj.get(current, [])
                found = False
                for nn in neighbors:
                    nek = (min(current, nn), max(current, nn))
                    if nek not in visited_edges:
                        next_v = nn
                        found = True
                        break
                if not found:
                    break

            if len(chain) >= 2:
                chains.append(chain)

    return chains


def _closest_point_on_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Find closest point on line segment ab to point p.

    Returns (closest_point, parameter_t).
    """
    ab = b - a
    ab_len_sq = np.dot(ab, ab)

    if ab_len_sq < 1e-15:
        return a.copy(), 0.0

    t = np.dot(p - a, ab) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return closest, float(t)


def snap_to_features(
    output_vertices: np.ndarray,
    input_vertices: np.ndarray,
    feature_edges: Set[Tuple[int, int]],
    snap_distance: float = 0.1,
    snap_strength: float = 1.0,
) -> Tuple[np.ndarray, Set[int]]:
    """Snap output vertices near feature curves onto those curves.

    Parameters
    ----------
    output_vertices : (V_out, 3) output mesh vertices
    input_vertices : (V_in, 3) original input mesh vertices
    feature_edges : set of (v0, v1) feature edge keys (indices into input_vertices)
    snap_distance : maximum distance for snapping
    snap_strength : 0.0 = no snap, 1.0 = full snap

    Returns
    -------
    snapped_vertices : (V_out, 3) vertices with snapping applied
    snapped_set : set of output vertex indices that were snapped
    """
    if not feature_edges or snap_strength <= 0.0:
        return output_vertices.copy(), set()

    output_verts = output_vertices.copy().astype(np.float64)
    input_verts = np.asarray(input_vertices, dtype=np.float64)
    snapped_set: Set[int] = set()

    # Collect all feature edge segments as (point_a, point_b) pairs
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for v0, v1 in feature_edges:
        if v0 < len(input_verts) and v1 < len(input_verts):
            segments.append((input_verts[v0], input_verts[v1]))

    if not segments:
        return output_verts.astype(np.float32), snapped_set

    # Build KDTree from feature edge midpoints for fast proximity query
    midpoints = np.array([(a + b) * 0.5 for a, b in segments], dtype=np.float64)
    seg_lengths = np.array([np.linalg.norm(b - a) for a, b in segments], dtype=np.float64)

    # For each output vertex, find nearest feature segment
    for vi in range(len(output_verts)):
        p = output_verts[vi]

        best_dist = snap_distance
        best_point = None

        # Check segments near this vertex (brute force for now; BVH for >100K)
        for si, (a, b) in enumerate(segments):
            # Quick reject: distance to midpoint > snap_distance + half segment length
            mid_dist = np.linalg.norm(p - midpoints[si])
            if mid_dist > snap_distance + 0.5 * seg_lengths[si]:
                continue

            closest, t = _closest_point_on_segment(p, a, b)
            dist = np.linalg.norm(p - closest)

            if dist < best_dist:
                best_dist = dist
                best_point = closest

        if best_point is not None:
            output_verts[vi] = p * (1.0 - snap_strength) + best_point * snap_strength
            snapped_set.add(vi)

    return output_verts.astype(np.float32), snapped_set
