"""Spatial indexing for QuadForge.

Provides:
  - KDTree: wrapper around a simple numpy-based KD-tree for nearest-neighbor
    queries on point clouds.
  - TriangleBVH: axis-aligned bounding box hierarchy for triangle meshes,
    supporting closest-point-on-surface and ray intersection queries.

These are pure-Python/numpy implementations optimised for correctness
and moderate mesh sizes (up to ~500K triangles). For larger meshes the
C++ engine with nanoflann will take over.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ======================================================================
# KD-Tree (brute-force for now; fine for <1M points in Python)
# ======================================================================

class KDTree:
    """Simple wrapper for nearest-neighbor queries on 3D point sets.

    Uses a brute-force approach backed by numpy vectorisation.
    For the Python fallback path this is adequate up to ~200K points.
    """

    def __init__(self, points: np.ndarray):
        """
        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
        """
        self.points = np.asarray(points, dtype=np.float64)

    def query(self, point: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find the k nearest neighbors to *point*.

        Returns
        -------
        distances : np.ndarray, shape (k,)
        indices   : np.ndarray, shape (k,)
        """
        point = np.asarray(point, dtype=np.float64)
        diffs = self.points - point[np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=1)
        if k == 1:
            idx = np.argmin(dists)
            return np.array([dists[idx]]), np.array([idx])
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]
        return dists[idx], idx

    def query_ball(self, point: np.ndarray, radius: float) -> List[int]:
        """Return all point indices within *radius* of *point*."""
        point = np.asarray(point, dtype=np.float64)
        diffs = self.points - point[np.newaxis, :]
        dists_sq = np.sum(diffs * diffs, axis=1)
        return list(np.where(dists_sq <= radius * radius)[0])


# ======================================================================
# Triangle BVH
# ======================================================================

@dataclass
class ClosestPointResult:
    """Result of a closest-point-on-surface query."""
    point: np.ndarray     # closest point on the surface
    distance: float       # distance from query point to closest point
    face_index: int       # triangle index
    bary: np.ndarray      # barycentric coordinates (u, v, w)


def _closest_point_on_triangle(
    p: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute closest point on triangle (v0, v1, v2) to point p.

    Returns (closest_point, barycentric_coords).
    Algorithm from Real-Time Collision Detection, Ericson 2005.
    """
    ab = v1 - v0
    ac = v2 - v0
    ap = p - v0

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return v0.copy(), np.array([1.0, 0.0, 0.0])

    bp = p - v1
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return v1.copy(), np.array([0.0, 1.0, 0.0])

    cp = p - v2
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return v2.copy(), np.array([0.0, 0.0, 1.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v_param = d1 / (d1 - d3)
        return v0 + v_param * ab, np.array([1.0 - v_param, v_param, 0.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return v0 + w * ac, np.array([1.0 - w, 0.0, w])

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return v1 + w * (v2 - v1), np.array([0.0, 1.0 - w, w])

    denom = 1.0 / (va + vb + vc)
    v_param = vb * denom
    w_param = vc * denom
    return v0 + v_param * ab + w_param * ac, np.array([1.0 - v_param - w_param, v_param, w_param])


class _BVHNode:
    __slots__ = ('bbox_min', 'bbox_max', 'left', 'right', 'tri_indices')

    def __init__(self):
        self.bbox_min: np.ndarray = np.zeros(3)
        self.bbox_max: np.ndarray = np.zeros(3)
        self.left: Optional[_BVHNode] = None
        self.right: Optional[_BVHNode] = None
        self.tri_indices: Optional[List[int]] = None  # leaf: list of triangle indices


class TriangleBVH:
    """Bounding Volume Hierarchy for triangle meshes.

    Supports:
      - closest_point(query_point) → ClosestPointResult
      - project_point(query_point) → projected point on surface
    """

    MAX_LEAF_SIZE = 8

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Parameters
        ----------
        vertices : np.ndarray, shape (V, 3)
        faces : np.ndarray, shape (F, 3), int — triangle vertex indices
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.num_triangles = len(self.faces)

        # Precompute triangle centroids and bboxes
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        self._tri_centroids = (v0 + v1 + v2) / 3.0
        self._tri_bbox_min = np.minimum(np.minimum(v0, v1), v2)
        self._tri_bbox_max = np.maximum(np.maximum(v0, v1), v2)

        # Build the tree
        self._root = self._build(list(range(self.num_triangles)))

    def _build(self, indices: List[int]) -> _BVHNode:
        node = _BVHNode()
        idx_arr = np.array(indices)
        node.bbox_min = np.min(self._tri_bbox_min[idx_arr], axis=0)
        node.bbox_max = np.max(self._tri_bbox_max[idx_arr], axis=0)

        if len(indices) <= self.MAX_LEAF_SIZE:
            node.tri_indices = indices
            return node

        # Split along longest axis
        extent = node.bbox_max - node.bbox_min
        axis = int(np.argmax(extent))
        centroids = self._tri_centroids[idx_arr, axis]
        median = np.median(centroids)

        left_idx = [i for i in indices if self._tri_centroids[i, axis] <= median]
        right_idx = [i for i in indices if self._tri_centroids[i, axis] > median]

        # Fallback if split is degenerate
        if len(left_idx) == 0 or len(right_idx) == 0:
            mid = len(indices) // 2
            sorted_idx = sorted(indices, key=lambda i: self._tri_centroids[i, axis])
            left_idx = sorted_idx[:mid]
            right_idx = sorted_idx[mid:]

        node.left = self._build(left_idx)
        node.right = self._build(right_idx)
        return node

    def closest_point(self, query: np.ndarray) -> ClosestPointResult:
        """Find the closest point on the triangle mesh to *query*."""
        query = np.asarray(query, dtype=np.float64)
        best = ClosestPointResult(
            point=np.zeros(3),
            distance=np.inf,
            face_index=-1,
            bary=np.zeros(3),
        )
        self._closest_recurse(self._root, query, best)
        return best

    def _closest_recurse(self, node: _BVHNode, query: np.ndarray, best: ClosestPointResult):
        # Distance from query to AABB
        clamped = np.clip(query, node.bbox_min, node.bbox_max)
        box_dist = np.linalg.norm(query - clamped)
        if box_dist > best.distance:
            return  # prune

        if node.tri_indices is not None:
            # Leaf node — test each triangle
            for ti in node.tri_indices:
                v0 = self.vertices[self.faces[ti, 0]]
                v1 = self.vertices[self.faces[ti, 1]]
                v2 = self.vertices[self.faces[ti, 2]]
                pt, bary = _closest_point_on_triangle(query, v0, v1, v2)
                dist = np.linalg.norm(query - pt)
                if dist < best.distance:
                    best.point = pt
                    best.distance = dist
                    best.face_index = ti
                    best.bary = bary
            return

        # Internal node — recurse into closer child first
        left_clamped = np.clip(query, node.left.bbox_min, node.left.bbox_max)
        right_clamped = np.clip(query, node.right.bbox_min, node.right.bbox_max)
        left_dist = np.linalg.norm(query - left_clamped)
        right_dist = np.linalg.norm(query - right_clamped)

        if left_dist < right_dist:
            self._closest_recurse(node.left, query, best)
            self._closest_recurse(node.right, query, best)
        else:
            self._closest_recurse(node.right, query, best)
            self._closest_recurse(node.left, query, best)

    def project_point(self, query: np.ndarray) -> np.ndarray:
        """Project *query* onto the nearest surface point."""
        return self.closest_point(query).point

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Project an array of points onto the surface.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)

        Returns
        -------
        projected : np.ndarray, shape (N, 3)
        """
        points = np.asarray(points, dtype=np.float64)
        result = np.empty_like(points)
        for i in range(len(points)):
            result[i] = self.project_point(points[i])
        return result
