"""Integer iso-line tracing and quad extraction for QuadForge.

Given a UV parametrization on a triangle mesh, traces all integer
iso-lines of U and V, finds their intersections, and constructs
quad faces from the resulting grid.

Algorithm:
  1. For each triangle, find where integer U/V values cross its edges
  2. Chain intersection points into polylines (iso-lines)
  3. At each U-line × V-line intersection, create a quad vertex
  4. Walk the grid cell-by-cell to construct quad faces
  5. Clean up degenerate faces and short edges
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


def _find_edge_crossings(
    uv: np.ndarray,
    faces: np.ndarray,
    axis: int,
) -> List[Tuple[int, int, int, float, np.ndarray]]:
    """Find where integer iso-lines of uv[:, axis] cross triangle edges.

    Returns list of (face_idx, edge_v0, edge_v1, integer_value, 3D_point).
    """
    crossings = []

    for fi in range(len(faces)):
        tri = faces[fi]
        for i in range(3):
            v0 = int(tri[i])
            v1 = int(tri[(i + 1) % 3])

            u0 = uv[v0, axis]
            u1 = uv[v1, axis]

            if abs(u1 - u0) < 1e-12:
                continue

            # Find all integers between u0 and u1
            lo = min(u0, u1)
            hi = max(u0, u1)
            i_lo = int(np.ceil(lo))
            i_hi = int(np.floor(hi))

            for k in range(i_lo, i_hi + 1):
                # Interpolation parameter
                t = (float(k) - u0) / (u1 - u0)
                if t < -1e-10 or t > 1.0 + 1e-10:
                    continue
                t = np.clip(t, 0.0, 1.0)

                crossings.append((fi, v0, v1, float(k), t))

    return crossings


def _interpolate_3d(vertices: np.ndarray, v0: int, v1: int, t: float) -> np.ndarray:
    """Linearly interpolate between vertices v0 and v1."""
    return (1.0 - t) * vertices[v0] + t * vertices[v1]


def trace_isolines(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Trace integer iso-lines and construct a quad mesh.

    Parameters
    ----------
    vertices : (V, 3) original mesh vertices
    faces : (F, 3) int, triangle indices
    uv : (V, 2) UV parametrization

    Returns
    -------
    quad_vertices : (Q_V, 3) quad mesh vertex positions
    quad_faces : list of [v0, v1, v2, v3] quad face indices
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    uv = np.asarray(uv, dtype=np.float64)

    # Step 1: Find all edge crossings for U and V iso-lines
    u_crossings = _find_edge_crossings(uv, faces, axis=0)
    v_crossings = _find_edge_crossings(uv, faces, axis=1)

    if not u_crossings and not v_crossings:
        print("[QuadForge] No iso-line crossings found — UV range too small")
        return np.zeros((0, 3)), []

    # Step 2: Grid-based quad construction
    # Determine the integer UV grid range
    u_min = int(np.floor(uv[:, 0].min()))
    u_max = int(np.ceil(uv[:, 0].max()))
    v_min = int(np.floor(uv[:, 1].min()))
    v_max = int(np.ceil(uv[:, 1].max()))

    # For each grid cell (i, j) to (i+1, j+1), find the quad vertex
    # at each corner by looking up which triangle contains (i, j) and
    # interpolating the 3D position.

    # Build a point lookup: for each integer (u, v) pair, find its 3D position
    grid_points: Dict[Tuple[int, int], np.ndarray] = {}

    for fi in range(len(faces)):
        tri = faces[fi]
        uv0 = uv[tri[0]]
        uv1 = uv[tri[1]]
        uv2 = uv[tri[2]]

        # Bounding box of this triangle in UV space
        uv_lo = np.minimum(np.minimum(uv0, uv1), uv2)
        uv_hi = np.maximum(np.maximum(uv0, uv1), uv2)

        i_lo = int(np.floor(uv_lo[0]))
        i_hi = int(np.ceil(uv_hi[0]))
        j_lo = int(np.floor(uv_lo[1]))
        j_hi = int(np.ceil(uv_hi[1]))

        for iu in range(i_lo, i_hi + 1):
            for jv in range(j_lo, j_hi + 1):
                key = (iu, jv)
                if key in grid_points:
                    continue

                # Check if (iu, jv) is inside this triangle in UV space
                pt = np.array([float(iu), float(jv)])
                bary = _barycentric_2d(uv0, uv1, uv2, pt)

                if bary is not None and np.all(bary >= -0.01):
                    # Interpolate 3D position
                    p3d = (bary[0] * vertices[tri[0]] +
                           bary[1] * vertices[tri[1]] +
                           bary[2] * vertices[tri[2]])
                    grid_points[key] = p3d

    if not grid_points:
        print("[QuadForge] No grid points found in UV space")
        return np.zeros((0, 3)), []

    # Step 3: Construct quads from grid cells
    # Assign indices to grid points
    point_to_idx: Dict[Tuple[int, int], int] = {}
    quad_verts_list: List[np.ndarray] = []

    for key, pos in grid_points.items():
        idx = len(quad_verts_list)
        point_to_idx[key] = idx
        quad_verts_list.append(pos)

    quad_faces: List[List[int]] = []

    for iu in range(u_min, u_max):
        for jv in range(v_min, v_max):
            # Four corners of this grid cell
            c00 = (iu, jv)
            c10 = (iu + 1, jv)
            c11 = (iu + 1, jv + 1)
            c01 = (iu, jv + 1)

            if (c00 in point_to_idx and c10 in point_to_idx and
                    c11 in point_to_idx and c01 in point_to_idx):
                i00 = point_to_idx[c00]
                i10 = point_to_idx[c10]
                i11 = point_to_idx[c11]
                i01 = point_to_idx[c01]

                # Avoid degenerate quads
                if len({i00, i10, i11, i01}) == 4:
                    quad_faces.append([i00, i10, i11, i01])

    quad_vertices = np.array(quad_verts_list, dtype=np.float64) if quad_verts_list else np.zeros((0, 3))

    print(f"[QuadForge] Iso-line extraction: {len(quad_vertices)} verts, "
          f"{len(quad_faces)} quads from UV grid [{u_min}..{u_max}]×[{v_min}..{v_max}]")

    return quad_vertices, quad_faces


def _barycentric_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute barycentric coordinates of point p in 2D triangle (a, b, c).

    Returns None if the triangle is degenerate.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return None

    inv = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * inv
    w = (d00 * d21 - d01 * d20) * inv
    u = 1.0 - v - w

    return np.array([u, v, w])


def cleanup_quad_mesh(
    vertices: np.ndarray,
    faces: List[List[int]],
    min_edge_length: float = 1e-6,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Clean up the extracted quad mesh.

    - Remove degenerate faces (zero area)
    - Collapse very short edges
    - Remove duplicate vertices
    """
    if len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    vertices = np.asarray(vertices, dtype=np.float64)

    # BUG-O FIX (v118): Remove duplicate vertices using O(V) grid-hash.
    # The previous O(V²) double loop iterated all vertex pairs — at 5 K
    # quad vertices that is 12.5 million distance checks in pure Python.
    # Replaced with the same spatial grid-hash strategy as BUG-F (cleanup.cpp).
    nv_raw = len(vertices)
    inv_tol = 1.0 / max(min_edge_length, 1e-15)

    from collections import defaultdict
    grid: dict = defaultdict(list)   # cell → list of already-merged new indices
    new_verts_list: list = []
    unique_map = np.empty(nv_raw, dtype=np.int32)

    for i in range(nv_raw):
        p = vertices[i]
        cx = int(np.floor(p[0] * inv_tol))
        cy = int(np.floor(p[1] * inv_tol))
        cz = int(np.floor(p[2] * inv_tol))

        found = -1
        for dx in (-1, 0, 1):
            if found >= 0: break
            for dy in (-1, 0, 1):
                if found >= 0: break
                for dz in (-1, 0, 1):
                    for cand in grid[(cx+dx, cy+dy, cz+dz)]:
                        pc = new_verts_list[cand]
                        dd = (pc[0]-p[0])**2 + (pc[1]-p[1])**2 + (pc[2]-p[2])**2
                        if dd < min_edge_length * min_edge_length:
                            found = cand; break
                    if found >= 0: break

        if found >= 0:
            unique_map[i] = found
        else:
            nidx = len(new_verts_list)
            new_verts_list.append(p)
            unique_map[i] = nidx
            grid[(cx, cy, cz)].append(nidx)

    # Remap faces
    cleaned_faces = []
    for face in faces:
        remapped = [int(unique_map[v]) for v in face]
        # Remove faces with duplicate vertices
        if len(set(remapped)) == len(remapped):
            cleaned_faces.append(remapped)

    # Compact vertex indices
    used = set()
    for face in cleaned_faces:
        used.update(face)

    if not used:
        return np.zeros((0, 3)), []

    old_to_new = {}
    new_verts = []
    for old_idx in sorted(used):
        old_to_new[old_idx] = len(new_verts)
        new_verts.append(vertices[old_idx])

    final_faces = [[old_to_new[v] for v in face] for face in cleaned_faces]
    final_verts = np.array(new_verts, dtype=np.float64)

    return final_verts, final_faces
