"""Symmetry detection and enforcement for QuadForge.

Implements reflective symmetry analysis and enforcement across the
X, Y, Z planes (in object local coordinates, matching QuadRemesher's
SymLocal=1 behavior).

Algorithm:
  1. For each enabled symmetry axis, sample vertex pairs across the plane
  2. Build a correspondence map using nearest-neighbor queries
  3. Score the symmetry quality (% of vertices with close correspondences)
  4. If quality exceeds threshold, enforce symmetry in subsequent stages
     by mirroring field and parametrization constraints

The symmetry enforcement works at three levels:
  - Cross-field: field directions on mirrored faces are forced symmetric
  - Parametrization: UV coordinates on mirrored vertices are constrained equal
  - Post-processing: output vertices are averaged with their mirror pairs
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .spatial import KDTree


@dataclass
class SymmetryInfo:
    """Result of symmetry analysis for one axis."""
    axis: int                          # 0=X, 1=Y, 2=Z
    is_symmetric: bool                 # True if symmetry score > threshold
    score: float                       # 0.0 to 1.0 — fraction of matched verts
    vertex_pairs: Dict[int, int]       # vertex_i → mirrored vertex_j
    unpaired: List[int]                # vertices with no mirror match
    plane_vertices: List[int]          # vertices near the symmetry plane


@dataclass
class SymmetryData:
    """Complete symmetry analysis for all three axes."""
    axes: Dict[int, SymmetryInfo]      # axis → SymmetryInfo
    active_axes: List[int]             # axes where symmetry is enforced
    all_pairs: Dict[int, int]          # union of all vertex pairs


def _reflect_point(point: np.ndarray, axis: int) -> np.ndarray:
    """Reflect a point across the plane perpendicular to the given axis."""
    reflected = point.copy()
    reflected[axis] = -reflected[axis]
    return reflected


def detect_symmetry(
    vertices: np.ndarray,
    faces=None,
    enabled_axes: Tuple[bool, bool, bool] = None,
    check_x: bool = False,
    check_y: bool = False,
    check_z: bool = False,
    tolerance_factor: float = 0.01,
    score_threshold: float = 0.90,
) -> "SymmetryData":
    """Analyze mesh for reflective symmetry across enabled planes.

    Supports two calling conventions:
      detect_symmetry(vertices, enabled_axes=(True, False, False), ...)
      detect_symmetry(vertices, faces, check_x=True, check_y=False, check_z=False, ...)

    Parameters
    ----------
    vertices : (V, 3) vertex positions in object local coordinates
    faces : optional, ignored (used for API compatibility)
    enabled_axes : (X, Y, Z) which axes to test
    check_x / check_y / check_z : per-axis toggles (alternative to enabled_axes)
    tolerance_factor : matching tolerance as fraction of bounding box diagonal
    score_threshold : minimum fraction of matched vertices to confirm symmetry

    Returns
    -------
    SymmetryData with analysis results. Also has has_x / has_y / has_z and
    x_symmetric / y_symmetric / z_symmetric convenience attributes.
    """
    # Resolve enabled_axes from check_x/y/z if enabled_axes not given
    if enabled_axes is None:
        enabled_axes = (check_x, check_y, check_z)

    vertices = np.asarray(vertices, dtype=np.float64)
    num_verts = len(vertices)

    # Compute matching tolerance from bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    tolerance = max(bbox_diag * tolerance_factor, 1e-8)
    plane_tolerance = tolerance * 2.0  # wider tolerance for on-plane vertices

    # Build KD-tree for nearest-neighbor queries
    kd = KDTree(vertices)

    axes_info: Dict[int, SymmetryInfo] = {}
    active_axes: List[int] = []
    all_pairs: Dict[int, int] = {}

    for axis in range(3):
        if not enabled_axes[axis]:
            continue

        pairs: Dict[int, int] = {}
        unpaired: List[int] = []
        plane_verts: List[int] = []
        matched_count = 0

        for vi in range(num_verts):
            # Check if vertex is on the symmetry plane
            if abs(vertices[vi, axis]) < plane_tolerance:
                plane_verts.append(vi)
                matched_count += 1
                continue

            # Reflect and find nearest neighbor
            reflected = _reflect_point(vertices[vi], axis)
            dists, indices = kd.query(reflected, k=1)

            if dists[0] < tolerance:
                mirror_vi = int(indices[0])
                if mirror_vi != vi:
                    pairs[vi] = mirror_vi
                    matched_count += 1
                else:
                    # Matched to self — on the plane
                    plane_verts.append(vi)
                    matched_count += 1
            else:
                unpaired.append(vi)

        score = matched_count / max(num_verts, 1)
        is_sym = score >= score_threshold

        info = SymmetryInfo(
            axis=axis,
            is_symmetric=is_sym,
            score=score,
            vertex_pairs=pairs,
            unpaired=unpaired,
            plane_vertices=plane_verts,
        )
        axes_info[axis] = info

        if is_sym:
            active_axes.append(axis)
            all_pairs.update(pairs)
            print(f"[QuadForge] Symmetry axis {'XYZ'[axis]}: "
                  f"score={score:.1%}, {len(pairs)} pairs, "
                  f"{len(plane_verts)} on-plane, {len(unpaired)} unpaired")
        else:
            print(f"[QuadForge] Symmetry axis {'XYZ'[axis]}: "
                  f"score={score:.1%} (below threshold, not enforced)")

    result = SymmetryData(
        axes=axes_info,
        active_axes=active_axes,
        all_pairs=all_pairs,
    )
    # Convenience boolean attributes for test compatibility
    result.has_x = 0 in active_axes
    result.has_y = 1 in active_axes
    result.has_z = 2 in active_axes
    result.x_symmetric = result.has_x
    result.y_symmetric = result.has_y
    result.z_symmetric = result.has_z
    return result


def build_symmetry_field_constraints(
    symmetry: SymmetryData,
    faces: np.ndarray,
    vertices: np.ndarray,
    face_frames_e1: np.ndarray,
    face_frames_e2: np.ndarray,
    face_normals: np.ndarray,
) -> Dict[int, float]:
    """Build cross-field constraints from symmetry.

    For each pair of mirrored faces, the field on one face constrains
    the field on its mirror: the field angle on the mirror face must
    be the reflection of the original face's angle.

    Parameters
    ----------
    symmetry : SymmetryData from detect_symmetry
    faces : (F, 3) triangle face indices
    vertices, face_frames_e1, face_frames_e2, face_normals : face geometry

    Returns
    -------
    constraints : dict of face_idx → target angle
    """
    if not symmetry.active_axes:
        return {}

    faces = np.asarray(faces, dtype=np.int32)
    num_faces = len(faces)
    constraints: Dict[int, float] = {}

    # Build vertex → face mapping
    vert_to_faces: Dict[int, List[int]] = {}
    for fi in range(num_faces):
        for vi in faces[fi]:
            vert_to_faces.setdefault(int(vi), []).append(fi)

    # For each axis, find mirrored face pairs
    for axis in symmetry.active_axes:
        info = symmetry.axes[axis]

        # Build face correspondence through vertex pairs
        face_pairs: Dict[int, int] = {}
        for fi in range(num_faces):
            tri = faces[fi]
            mirror_verts = []
            for vi in tri:
                vi = int(vi)
                if vi in info.vertex_pairs:
                    mirror_verts.append(info.vertex_pairs[vi])
                elif vi in info.plane_vertices:
                    mirror_verts.append(vi)
                else:
                    break
            else:
                # All vertices have mirrors — find the mirror face
                mirror_set = set(mirror_verts)
                for candidate_fi in vert_to_faces.get(mirror_verts[0], []):
                    if set(int(v) for v in faces[candidate_fi]) == mirror_set:
                        face_pairs[fi] = candidate_fi
                        break

        # For mirrored face pairs, constrain field angles
        # The field on the mirror face should be the reflection of the original
        for fi, fj in face_pairs.items():
            if fi >= fj:
                continue  # process each pair once

            e1_i = face_frames_e1[fi]
            n_i = face_normals[fi]

            # The reflected field direction: flip the component along the axis
            # For axis X: reflect x-component of the field direction
            # The constraint angle on fj = -angle on fi (reflected)
            # We don't set an absolute angle, but mark these faces as
            # needing symmetric treatment in the solver
            constraints[fj] = -1.0  # sentinel: will be resolved during solve

    return constraints


def enforce_output_symmetry(
    vertices: np.ndarray,
    symmetry_or_faces=None,
    sym_x: bool = False,
    sym_y: bool = False,
    sym_z: bool = False,
    strength: float = 1.0,
) -> "np.ndarray | tuple":
    """Enforce symmetry on output mesh vertices by averaging mirror pairs.

    Supports two calling conventions:
      enforce_output_symmetry(vertices, symmetry: SymmetryData, strength=1.0)
          → returns np.ndarray (original form, used by pipeline)
      enforce_output_symmetry(vertices, faces, sym_x=False, sym_y=False, sym_z=False)
          → returns (vertices, faces) tuple (test-compatible form)

    Parameters
    ----------
    vertices : (V, 3) output vertex positions
    symmetry_or_faces : SymmetryData | list of faces
    sym_x / sym_y / sym_z : axis toggles (test-compatible form)
    strength : 0.0 = no enforcement, 1.0 = full enforcement

    Returns
    -------
    If SymmetryData passed: np.ndarray (V, 3) symmetric vertices
    If faces passed: (np.ndarray, list) — (vertices, faces) tuple
    """
    # Detect which calling convention is in use
    faces_return_mode = False
    faces_out = None

    if symmetry_or_faces is None or isinstance(symmetry_or_faces, (list, np.ndarray)):
        # Test-compatible call: (verts, faces, sym_x=..., ...)
        faces_return_mode = True
        faces_out = symmetry_or_faces
        symmetry = None

        # Build a trivial SymmetryData if any axis enabled
        if sym_x or sym_y or sym_z:
            enabled = (sym_x, sym_y, sym_z)
            verts_np = np.asarray(vertices, dtype=np.float64)
            symmetry = detect_symmetry(verts_np, enabled_axes=enabled)
    else:
        symmetry = symmetry_or_faces

    if symmetry is None or not symmetry.active_axes:
        if faces_return_mode:
            return np.asarray(vertices, dtype=np.float64), faces_out
        return np.asarray(vertices).copy()

    result = np.asarray(vertices, dtype=np.float64).copy()

    for axis in symmetry.active_axes:
        info = symmetry.axes[axis]

        # Average mirror pairs
        processed = set()
        for vi, vj in info.vertex_pairs.items():
            if vi in processed or vj in processed:
                continue
            if vi >= len(result) or vj >= len(result):
                continue

            pi = result[vi].copy()
            pj = result[vj].copy()

            # Reflect pj across the axis
            pj_reflected = pj.copy()
            pj_reflected[axis] = -pj_reflected[axis]

            # Average with reflected mirror
            avg = 0.5 * (pi + pj_reflected)
            result[vi] = result[vi] * (1.0 - strength) + avg * strength

            # Mirror back to vj
            avg_mirror = avg.copy()
            avg_mirror[axis] = -avg_mirror[axis]
            result[vj] = result[vj] * (1.0 - strength) + avg_mirror * strength

            processed.add(vi)
            processed.add(vj)

        # Snap on-plane vertices to the plane
        for vi in info.plane_vertices:
            if vi < len(result):
                result[vi, axis] *= (1.0 - strength)

    result = result.astype(np.asarray(vertices).dtype)

    if faces_return_mode:
        return result, faces_out
    return result


def mirror_feature_edges(
    feature_edges: Set[Tuple[int, int]],
    symmetry: SymmetryData,
) -> Set[Tuple[int, int]]:
    """Add mirrored copies of feature edges to ensure symmetric features.

    Parameters
    ----------
    feature_edges : existing feature edge set
    symmetry : SymmetryData with vertex pairs

    Returns
    -------
    augmented : feature_edges with mirror copies added
    """
    if not symmetry.active_axes:
        return feature_edges

    augmented = set(feature_edges)

    for axis in symmetry.active_axes:
        info = symmetry.axes[axis]
        pairs = info.vertex_pairs

        for v0, v1 in list(feature_edges):
            mv0 = pairs.get(v0, v0)
            mv1 = pairs.get(v1, v1)
            ek = (min(mv0, mv1), max(mv0, mv1))
            augmented.add(ek)

    return augmented
