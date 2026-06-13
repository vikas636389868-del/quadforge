"""QuadForge v1.4 — Live Preview Mode.

Generates a lightweight cross-field wireframe overlay for real-time preview
as the user adjusts parameters, without running the full remesh pipeline.

The preview runs only the fast stages:
  1. Curvature computation
  2. Cross-field solve (optionally simplified)
  3. Approximate isoline density visualisation (no extraction)

The field directions are returned as a list of (start, end) point pairs
that the operator can display as a temporary mesh object in the viewport.

Public API
----------
compute_field_preview(vertices, faces, params, n_arrows=None)
    → FieldPreview

FieldPreview.to_edge_mesh()
    → (verts, edges) ready for bpy mesh construction
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class FieldPreview:
    """Result of a field preview computation.

    Attributes
    ----------
    face_centers : (F, 3) float64  centre of each face
    direction_u  : (F, 3) float64  first  cross-field direction (unit vec)
    direction_v  : (F, 3) float64  second cross-field direction (unit vec)
    arrow_scale  : float            recommended arrow half-length
    elapsed      : float            computation time (seconds)
    """
    face_centers: np.ndarray
    direction_u:  np.ndarray
    direction_v:  np.ndarray
    arrow_scale:  float
    elapsed:      float = 0.0

    def to_edge_mesh(
        self,
        scale_override: Optional[float] = None,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
        """Convert the field preview to vertex + edge lists for Blender mesh.

        Each face contributes 4 vertices and 2 edges (the two arms of the cross).
        The scale is `arrow_scale` unless `scale_override` is provided.

        Returns
        -------
        verts : list of (x, y, z) tuples
        edges : list of (i, j) vertex-index pairs
        """
        scale = scale_override if scale_override is not None else self.arrow_scale
        fc = self.face_centers
        du = self.direction_u * scale
        dv = self.direction_v * scale

        verts: List[Tuple[float, float, float]] = []
        edges: List[Tuple[int, int]] = []

        n = len(fc)
        for i in range(n):
            c  = fc[i]
            u  = du[i]
            v  = dv[i]
            # 4 endpoint vertices per face
            base = len(verts)
            verts.append(tuple(c - u))   # base + 0
            verts.append(tuple(c + u))   # base + 1
            verts.append(tuple(c - v))   # base + 2
            verts.append(tuple(c + v))   # base + 3
            edges.append((base + 0, base + 1))
            edges.append((base + 2, base + 3))

        return verts, edges


def compute_field_preview(
    vertices: np.ndarray,
    faces:    np.ndarray,
    params,
    n_arrows: Optional[int] = None,
    fast: bool = True,
) -> FieldPreview:
    """Compute a lightweight cross-field preview without full remeshing.

    This runs only the curvature + field stages (< 0.5 s on medium meshes)
    and skips parametrisation, extraction, and post-processing entirely.

    Parameters
    ----------
    vertices  : (V, 3) float64 mesh vertex positions
    faces     : (F, 3) int32   triangulated faces
    params    : QFParams-like namespace (target_quad_count, curvature_adaptivity…)
    n_arrows  : if set, sub-sample to at most this many arrows (for viewport perf)
    fast      : if True, use the curvature-only field (fastest preview)

    Returns
    -------
    FieldPreview with face-centre directions.
    """
    import time
    t0 = time.perf_counter()

    vertices = np.asarray(vertices, dtype=np.float64)
    faces    = np.asarray(faces,    dtype=np.int32)

    # --- Face centres --------------------------------------------------------
    fc = vertices[faces].mean(axis=1)   # (F, 3)

    # --- Compute face normals ------------------------------------------------
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms   = np.where(norms < 1e-12, 1.0, norms)
    normals = normals / norms           # (F, 3) unit normals

    # --- Cross-field directions ----------------------------------------------
    if fast:
        # Fast path: use curvature for direction hints, else arbitrary tangent
        du, dv = _fast_tangent_field(vertices, faces, normals, params)
    else:
        # Full cross-field solve (used when fast=False from the operator)
        try:
            from .curvature import compute_curvature
            from .field import compute_cross_field
            from .halfedge import HalfEdgeMesh

            he = HalfEdgeMesh(vertices, faces)
            curv = compute_curvature(he)
            field_data = compute_cross_field(
                he, curv, None,
                field_solver=getattr(params, 'field_solver', 'CURVATURE'),
            )
            du = field_data.u_dir   # (F, 3)
            dv = field_data.v_dir   # (F, 3)
        except Exception:
            du, dv = _fast_tangent_field(vertices, faces, normals, params)

    # --- Sub-sample for viewport performance ---------------------------------
    if n_arrows is not None and n_arrows < len(fc):
        idx = np.linspace(0, len(fc) - 1, n_arrows, dtype=int)
        fc      = fc[idx]
        du      = du[idx]
        dv      = dv[idx]
        normals = normals[idx]

    # --- Arrow scale: proportional to average edge length --------------------
    total_area = float(np.sum(
        np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    )) * 0.5
    arrow_scale = 0.5 * float(np.sqrt(total_area / max(len(faces), 1)))

    elapsed = time.perf_counter() - t0

    return FieldPreview(
        face_centers=fc,
        direction_u=du,
        direction_v=dv,
        arrow_scale=arrow_scale,
        elapsed=elapsed,
    )


def _fast_tangent_field(
    vertices: np.ndarray,
    faces:    np.ndarray,
    normals:  np.ndarray,
    params,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cheap tangent field using per-face curvature hints.

    Falls back to an arbitrary tangent frame if curvature fails.
    """
    F = len(faces)

    # Try curvature-based direction
    try:
        from .curvature import compute_curvature

        # Simple per-face curvature: use the max-curvature direction at each face
        vert_kappa1 = np.zeros((len(vertices), 3), dtype=np.float64)
        curv = compute_curvature(vertices, faces)

        # Per-vertex principal direction
        if hasattr(curv, 'principal_dir1'):
            vert_kappa1 = np.asarray(curv.principal_dir1, dtype=np.float64)
            if vert_kappa1.shape == (len(vertices), 3):
                # Average over face vertices
                d1 = (vert_kappa1[faces[:, 0]] +
                      vert_kappa1[faces[:, 1]] +
                      vert_kappa1[faces[:, 2]]) / 3.0   # (F, 3)
                nrm = np.linalg.norm(d1, axis=1, keepdims=True)
                valid = (nrm > 1e-8).ravel()
                d1_unit = np.zeros_like(d1)
                d1_unit[valid] = d1[valid] / nrm[valid]

                # For invalid faces, fall back to arbitrary tangent
                _fill_arbitrary_tangent(d1_unit, normals, ~valid)

                d2 = np.cross(normals, d1_unit)
                nrm2 = np.linalg.norm(d2, axis=1, keepdims=True)
                valid2 = (nrm2 > 1e-8).ravel()
                d2[valid2] /= nrm2[valid2]
                _fill_arbitrary_tangent(d2, normals, ~valid2)

                return d1_unit, d2
    except Exception:
        pass

    # Fallback: arbitrary tangent frame from face normals
    du = np.zeros((F, 3), dtype=np.float64)
    dv = np.zeros((F, 3), dtype=np.float64)
    _fill_arbitrary_tangent(du, normals, np.ones(F, dtype=bool))
    dv = np.cross(normals, du)
    nrm = np.linalg.norm(dv, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-8, 1.0, nrm)
    dv /= nrm
    return du, dv


def _fill_arbitrary_tangent(
    out:     np.ndarray,  # (F, 3) — modified in-place
    normals: np.ndarray,  # (F, 3)
    mask:    np.ndarray,  # (F,) bool — which rows to fill
) -> None:
    """Fill rows of `out` with an arbitrary tangent perpendicular to `normals`."""
    if not mask.any():
        return
    n = normals[mask]
    # Choose an "up" vector not parallel to the normal
    up = np.zeros_like(n)
    up[:, 2] = 1.0
    # For normals nearly parallel to Z, use X instead
    dot_z = np.abs(n[:, 2])
    use_x = dot_z > 0.9
    up[use_x, 0] = 1.0
    up[use_x, 2] = 0.0

    t = np.cross(n, up)
    nrm = np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-8, 1.0, nrm)
    t /= nrm
    out[mask] = t
