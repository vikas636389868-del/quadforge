"""Adaptive sizing field for QuadForge.

Computes per-vertex target edge lengths based on:
  - Base edge length from target quad count and total surface area
  - Curvature modulation (high curvature → smaller quads)
  - Vertex color density map (red = denser, green = sparser)
  - Feature proximity (denser near feature edges)

The sizing field feeds into the cross-field solver's metric tensor
and the parametrization's gradient scaling.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Set, Tuple


def compute_base_edge_length(total_area_or_verts, target_quad_count_or_faces=None,
                              target_count: int = None, target_quad_count: int = None) -> float:
    """Compute uniform target edge length from area and quad count.

    Supports two calling conventions:
      compute_base_edge_length(total_area: float, target_quad_count: int)
      compute_base_edge_length(vertices: np.ndarray, faces, target_count=N)

    Each quad has area ≈ L², so L = sqrt(total_area / target_quad_count).
    """
    import numpy as np

    # Detect which calling convention is being used
    if isinstance(total_area_or_verts, (int, float)):
        # Classic call: compute_base_edge_length(total_area, target_quad_count)
        total_area = float(total_area_or_verts)
        count = int(target_quad_count_or_faces or target_quad_count or 1000)
    else:
        # Array call: compute_base_edge_length(vertices, faces, target_count=N)
        vertices = np.asarray(total_area_or_verts, dtype=np.float64)
        faces_raw = target_quad_count_or_faces
        count = int(target_count or target_quad_count or 1000)

        # Compute total surface area from the mesh
        if faces_raw is not None and len(faces_raw) > 0:
            fa = np.asarray(faces_raw, dtype=np.int32)
            if fa.ndim == 2 and fa.shape[1] == 3:
                v0 = vertices[fa[:, 0]]
                v1 = vertices[fa[:, 1]]
                v2 = vertices[fa[:, 2]]
                crosses = np.cross(v1 - v0, v2 - v0)
                total_area = float(np.sum(np.linalg.norm(crosses, axis=1)) * 0.5)
            else:
                total_area = 1.0
        else:
            total_area = 1.0

    if count <= 0 or total_area <= 0:
        return 1.0
    return float(np.sqrt(total_area / max(count, 1)))


def compute_sizing_field(
    arg1,
    arg2=None,
    arg3=None,
    arg4=None,
    total_area: float = None,
    target_quad_count: int = None,
    curvature_adaptivity: float = 0.0,
    adaptivity: float = None,
    principal_curvatures: Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
    feature_edges: Optional[Set[Tuple[int, int]]] = None,
    feature_proximity_factor: float = 0.5,
) -> np.ndarray:
    """Compute per-vertex target edge length.

    Supports multiple calling conventions:
      (1) compute_sizing_field(vertices, faces, total_area, target_quad_count, ...)
          — original pipeline form
      (2) compute_sizing_field(he, curv, target_quad_count=N, adaptivity=A)
          — HalfEdgeMesh + CurvatureData form (used in tests)
      (3) compute_sizing_field(vertices, faces, curv, L_base, adaptivity=A)
          — array form with CurvatureData and precomputed base edge length

    Parameters
    ----------
    """
    import numpy as np

    # Detect calling convention by inspecting arg types
    from .curvature import CurvatureData
    from .halfedge import HalfEdgeMesh

    # Resolve adaptivity alias
    if adaptivity is not None and curvature_adaptivity == 0.0:
        curvature_adaptivity = adaptivity

    # Convention (2): (HalfEdgeMesh, CurvatureData, ...)
    if isinstance(arg1, HalfEdgeMesh):
        he = arg1
        vertices = np.asarray(he.vertices, dtype=np.float64)
        if hasattr(he, 'faces'):
            faces_list = list(he.faces)
        elif hasattr(he, '_face_lists'):
            faces_list = list(he._face_lists)
        else:
            faces_list = []
        faces_arr = np.asarray(faces_list, dtype=np.int32)

        curv = arg2 if isinstance(arg2, CurvatureData) else None
        # arg3 may be target_quad_count
        if isinstance(arg3, int):
            target_quad_count = arg3
        if isinstance(arg4, float):
            curvature_adaptivity = arg4

        if curv is not None:
            principal_curvatures = curv.max_abs_curvature

        v0 = vertices[faces_arr[:, 0]]
        v1 = vertices[faces_arr[:, 1]]
        v2 = vertices[faces_arr[:, 2]]
        crosses = np.cross(v1 - v0, v2 - v0)
        computed_area = float(np.sum(np.linalg.norm(crosses, axis=1)) * 0.5)
        total_area = computed_area if total_area is None else total_area
        target_quad_count = target_quad_count or 1000

    # Convention (3): (vertices_arr, faces, CurvatureData, L_base, ...)
    elif isinstance(arg2, (list, np.ndarray)) and not isinstance(arg2, CurvatureData) and \
         isinstance(arg3, CurvatureData):
        vertices = np.asarray(arg1, dtype=np.float64)
        faces_list = list(arg2) if not isinstance(arg2, np.ndarray) else arg2.tolist()
        faces_arr = np.asarray(faces_list, dtype=np.int32)
        curv = arg3
        principal_curvatures = curv.max_abs_curvature

        # arg4 = L_base — convert to total_area via total_area = L_base² * count
        if isinstance(arg4, float):
            L_base = arg4
            target_quad_count = target_quad_count or 1000
            total_area = (L_base ** 2) * target_quad_count
        else:
            v0 = vertices[faces_arr[:, 0]]
            v1 = vertices[faces_arr[:, 1]]
            v2 = vertices[faces_arr[:, 2]]
            crosses = np.cross(v1 - v0, v2 - v0)
            total_area = float(np.sum(np.linalg.norm(crosses, axis=1)) * 0.5)
            target_quad_count = target_quad_count or 1000

    else:
        # Convention (1): classic (vertices, faces, total_area, target_quad_count, ...)
        vertices = np.asarray(arg1, dtype=np.float64)
        if isinstance(arg2, (list, np.ndarray)):
            faces_list = list(arg2) if isinstance(arg2, np.ndarray) else arg2
        else:
            faces_list = []
        faces_arr = np.asarray(faces_list, dtype=np.int32) if faces_list else np.zeros((0,3), dtype=np.int32)

        if arg3 is not None and total_area is None:
            total_area = float(arg3)
        if arg4 is not None and target_quad_count is None:
            target_quad_count = int(arg4)
        total_area = total_area or 1.0
        target_quad_count = target_quad_count or 1000

    # --- Core computation ---
    num_verts = len(vertices)
    L_base = compute_base_edge_length(total_area, target_quad_count)
    sizing = np.full(num_verts, L_base, dtype=np.float64)

    # --- Curvature modulation ---
    if principal_curvatures is not None and curvature_adaptivity > 0.0:
        kappa = np.abs(principal_curvatures).astype(np.float64)
        alpha = curvature_adaptivity
        # L(v) = L_base / (1 + α * κ(v) * L_base)
        sizing = L_base / (1.0 + alpha * kappa * L_base)

    # --- Vertex color density map ---
    if vertex_colors is not None:
        colors = np.asarray(vertex_colors, dtype=np.float64)
        if colors.shape[0] == num_verts and colors.shape[1] >= 3:
            red = colors[:, 0]
            green = colors[:, 1]

            # Red > 0.5 → multiply density (smaller quads)
            # density_multiplier = 4^(2*red - 1) for red region
            red_mask = red > 0.5
            red_factor = np.ones(num_verts, dtype=np.float64)
            red_factor[red_mask] = np.power(4.0, 2.0 * red[red_mask] - 1.0)

            # Green > 0.5 → divide density (bigger quads)
            # density_multiplier = 0.25^(2*green - 1) for green region
            green_mask = green > 0.5
            green_factor = np.ones(num_verts, dtype=np.float64)
            green_factor[green_mask] = np.power(0.25, 2.0 * green[green_mask] - 1.0)

            # Combine: higher density_multiplier → smaller edge length
            # sizing /= density_multiplier, but we need the inverse:
            # edge_length = L / sqrt(density_mult)  (since area = L², density = 1/L²)
            density = red_factor * green_factor
            sizing /= np.sqrt(np.clip(density, 0.01, 100.0))

    # --- Feature proximity ---
    if feature_edges and feature_proximity_factor > 0.0:
        # Mark vertices on feature edges
        faces = faces_arr
        feature_verts: set = set()
        for v0, v1 in feature_edges:
            feature_verts.add(v0)
            feature_verts.add(v1)

        # Also mark their one-ring neighbors (approximate proximity)
        neighbor_verts: set = set()
        vert_to_faces: dict = {}
        for fi, face in enumerate(faces_arr):
            for v in face:
                vert_to_faces.setdefault(v, []).append(fi)

        for fv in feature_verts:
            for fi in vert_to_faces.get(fv, []):
                for v in faces_arr[fi]:
                    neighbor_verts.add(v)

        # Feature vertices: reduce sizing by feature_proximity_factor
        feature_arr = np.array(list(feature_verts), dtype=np.int32)
        if len(feature_arr) > 0:
            sizing[feature_arr] *= (1.0 - 0.5 * feature_proximity_factor)

        # Neighbor ring: gentler reduction
        neighbor_arr = np.array(list(neighbor_verts - feature_verts), dtype=np.int32)
        if len(neighbor_arr) > 0:
            sizing[neighbor_arr] *= (1.0 - 0.25 * feature_proximity_factor)

    # Clamp to reasonable range
    min_size = L_base * 0.1
    max_size = L_base * 5.0
    sizing = np.clip(sizing, min_size, max_size)

    return sizing
