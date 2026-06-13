"""Catmull-Clark subdivision compatibility checker for QuadForge.

A quad mesh is subdivision-compatible when every vertex has valence exactly 4
(i.e. it is a regular mesh).  In practice some irregular vertices
(valence ≠ 4) are always present at feature corners and topology
singularities, but the ratio should be low.

This module checks a remeshed quad output for subdivision compatibility and
provides a human-readable report, matching the postprocess/subdiv.h / subdiv.cpp
contract on the C++ side.

Public API
----------
check_subdiv_compatibility(vertices, faces) -> SubdivReport
    Main entry point.  Analyses a quad mesh and returns a report object.

SubdivReport
    Dataclass holding all compatibility metrics.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubdivReport:
    """Result of a Catmull-Clark compatibility analysis.

    Attributes
    ----------
    is_compatible : bool
        True if the mesh passes all compatibility checks at the given
        ``strict`` threshold.
    quad_percentage : float
        Percentage of faces that are quads (should be 100 % for pure CC input).
    regular_vertex_ratio : float
        Fraction of interior vertices with valence exactly 4.
    irregular_vertices : List[int]
        Indices of interior vertices with valence ≠ 4.
    valence_histogram : Dict[int, int]
        Maps valence → count of interior vertices with that valence.
    boundary_vertex_count : int
        Number of boundary vertices (valence computation differs on boundary).
    non_manifold_edges : List[Tuple[int,int]]
        Edges shared by more than two faces (incompatible with CC subdivision).
    has_non_quad_faces : bool
        True if any face has != 4 vertices.
    min_quad_angle_deg : float
        Minimum interior angle across all quad faces (degrees).
        Very small angles (<5°) indicate degenerate quads that will explode
        on subdivision.
    mean_scaled_jacobian : float
        Mean Scaled Jacobian quality metric (1.0 = perfect rectangle).
    warnings : List[str]
        Human-readable warnings sorted by severity.
    """
    is_compatible: bool = False
    quad_percentage: float = 0.0
    regular_vertex_ratio: float = 0.0
    irregular_vertices: List[int] = field(default_factory=list)
    valence_histogram: Dict[int, int] = field(default_factory=dict)
    boundary_vertex_count: int = 0
    non_manifold_edges: List[Tuple[int, int]] = field(default_factory=list)
    has_non_quad_faces: bool = False
    min_quad_angle_deg: float = 90.0
    mean_scaled_jacobian: float = 1.0
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a compact one-line summary string."""
        status = "✅ CC-compatible" if self.is_compatible else "⚠️  Not CC-compatible"
        return (
            f"{status} | "
            f"quads={self.quad_percentage:.1f}% | "
            f"regular={self.regular_vertex_ratio * 100:.1f}% | "
            f"irregular={len(self.irregular_vertices)} | "
            f"min_angle={self.min_quad_angle_deg:.1f}° | "
            f"mean_SJ={self.mean_scaled_jacobian:.3f}"
        )

    def full_report(self) -> str:
        """Return a multi-line human-readable report."""
        lines = [
            "─" * 60,
            "  QuadForge — Catmull-Clark Subdivision Compatibility Report",
            "─" * 60,
            f"  Status            : {'COMPATIBLE ✅' if self.is_compatible else 'NOT COMPATIBLE ⚠️'}",
            f"  Quad faces        : {self.quad_percentage:.2f}%",
            f"  Regular vertices  : {self.regular_vertex_ratio * 100:.2f}%",
            f"  Irregular verts   : {len(self.irregular_vertices)}",
            f"  Boundary verts    : {self.boundary_vertex_count}",
            f"  Non-manifold edges: {len(self.non_manifold_edges)}",
            f"  Min quad angle    : {self.min_quad_angle_deg:.2f}°",
            f"  Mean Scaled Jacob.: {self.mean_scaled_jacobian:.4f}",
            "",
            "  Valence histogram (interior vertices):",
        ]
        for val in sorted(self.valence_histogram):
            count = self.valence_histogram[val]
            marker = " ← ideal" if val == 4 else ""
            lines.append(f"    valence {val:2d}: {count:6d}{marker}")

        if self.warnings:
            lines.append("")
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    • {w}")

        lines.append("─" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_vertex_valences(
    vertices: np.ndarray,
    faces: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Compute per-vertex valence (number of incident edges) and detect
    boundary / non-manifold edges.

    Returns
    -------
    valences : (V,) int array
    is_boundary : (V,) bool array — True for boundary vertices
    non_manifold : list of (v0, v1) edge pairs incident on >2 faces
    """
    num_verts = len(vertices)
    edge_face_count: Dict[Tuple[int, int], int] = {}
    edge_incident: Dict[int, set] = {i: set() for i in range(num_verts)}

    for face in faces:
        n = len(face)
        for i in range(n):
            v0 = int(face[i])
            v1 = int(face[(i + 1) % n])
            key = (min(v0, v1), max(v0, v1))
            edge_face_count[key] = edge_face_count.get(key, 0) + 1
            edge_incident[v0].add(v1)
            edge_incident[v1].add(v0)

    # Boundary edges (count == 1)
    boundary_verts = set()
    non_manifold = []
    for (v0, v1), cnt in edge_face_count.items():
        if cnt == 1:
            boundary_verts.add(v0)
            boundary_verts.add(v1)
        elif cnt > 2:
            non_manifold.append((v0, v1))

    valences = np.array([len(edge_incident[i]) for i in range(num_verts)], dtype=np.int32)
    is_boundary = np.zeros(num_verts, dtype=bool)
    for v in boundary_verts:
        is_boundary[v] = True

    return valences, is_boundary, non_manifold


def _compute_quad_angles(
    vertices: np.ndarray,
    faces: List[List[int]],
) -> Tuple[float, float]:
    """Compute minimum and mean interior angles across all quad faces.

    Returns (min_angle_deg, mean_angle_deg).
    Only considers faces with exactly 4 vertices.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    min_angle = 180.0
    angle_sum = 0.0
    count = 0

    for face in faces:
        if len(face) != 4:
            continue
        pts = verts[face]  # (4, 3)
        for i in range(4):
            p_prev = pts[(i - 1) % 4]
            p_curr = pts[i]
            p_next = pts[(i + 1) % 4]
            e1 = p_prev - p_curr
            e2 = p_next - p_curr
            l1 = np.linalg.norm(e1)
            l2 = np.linalg.norm(e2)
            if l1 < 1e-12 or l2 < 1e-12:
                continue
            cos_a = np.clip(np.dot(e1, e2) / (l1 * l2), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_a))
            min_angle = min(min_angle, angle_deg)
            angle_sum += angle_deg
            count += 1

    mean_angle = (angle_sum / count) if count > 0 else 90.0
    return float(min_angle), float(mean_angle)


def _compute_scaled_jacobians(
    vertices: np.ndarray,
    faces: List[List[int]],
) -> Tuple[float, float]:
    """Compute min and mean Scaled Jacobian for quad faces.

    The Scaled Jacobian at corner i of a quad is:
        SJ_i = det(e1, e2, n) / (|e1| * |e2|)
    where e1, e2 are the two edges meeting at corner i, and n is the
    face normal.  Range: [-1, 1].  1.0 = perfect rectangle, <0 = inverted.

    Returns (min_sj, mean_sj).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    min_sj = 1.0
    sj_sum = 0.0
    count = 0

    for face in faces:
        if len(face) != 4:
            continue
        pts = verts[face]

        # Face normal (average of two diagonal cross products)
        d1 = pts[2] - pts[0]
        d2 = pts[3] - pts[1]
        n = np.cross(d1, d2)
        n_len = np.linalg.norm(n)
        if n_len < 1e-15:
            continue
        n = n / n_len

        for i in range(4):
            e1 = pts[(i + 1) % 4] - pts[i]
            e2 = pts[(i - 1) % 4] - pts[i]
            l1 = np.linalg.norm(e1)
            l2 = np.linalg.norm(e2)
            if l1 < 1e-12 or l2 < 1e-12:
                continue
            sj = np.dot(np.cross(e1, e2), n) / (l1 * l2)
            min_sj = min(min_sj, sj)
            sj_sum += sj
            count += 1

    mean_sj = (sj_sum / count) if count > 0 else 1.0
    return float(min_sj), float(mean_sj)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_subdiv_compatibility(
    vertices: np.ndarray,
    faces: "List[List[int]]",
    strict: bool = False,
) -> SubdivReport:
    """Analyse a quad mesh for Catmull-Clark subdivision compatibility.

    Parameters
    ----------
    vertices : (V, 3) float array of vertex positions
    faces : list of face index lists (should all be length-4 for a quad mesh)
    strict : bool
        When True, require >99 % regular vertices and >99.5 % quad faces.
        When False (default), allow up to 5 % irregular vertices and
        require >97 % quad faces — matching QuadForge's quality targets.

    Returns
    -------
    SubdivReport
    """
    verts = np.asarray(vertices, dtype=np.float64)
    num_verts = len(verts)
    num_faces = len(faces)
    warnings: List[str] = []

    if num_verts == 0 or num_faces == 0:
        return SubdivReport(
            is_compatible=False,
            warnings=["Mesh is empty — no vertices or faces."],
        )

    # ---- 1. Quad percentage ------------------------------------------------
    quad_faces = [f for f in faces if len(f) == 4]
    non_quad   = [f for f in faces if len(f) != 4]
    quad_pct   = 100.0 * len(quad_faces) / num_faces if num_faces > 0 else 0.0
    has_non_quad = len(non_quad) > 0

    if has_non_quad:
        warnings.append(
            f"{len(non_quad)} non-quad face(s) ({100 - quad_pct:.1f}%) — "
            "Catmull-Clark requires all-quad input."
        )

    # ---- 2. Valence analysis -----------------------------------------------
    valences, is_boundary, non_manifold = _build_vertex_valences(verts, faces)

    interior_mask   = ~is_boundary
    interior_verts  = np.where(interior_mask)[0]
    boundary_count  = int(np.sum(is_boundary))

    valence_hist: Dict[int, int] = {}
    irregular: List[int] = []
    for vi in interior_verts:
        val = int(valences[vi])
        valence_hist[val] = valence_hist.get(val, 0) + 1
        if val != 4:
            irregular.append(vi)

    num_interior = len(interior_verts)
    regular_ratio = (
        (num_interior - len(irregular)) / num_interior
        if num_interior > 0 else 1.0
    )

    if len(irregular) > 0:
        irreg_pct = 100.0 * len(irregular) / max(num_interior, 1)
        # List the valences present
        bad_vals = sorted(set(int(valences[vi]) for vi in irregular))
        warnings.append(
            f"{len(irregular)} irregular interior vertex/vertices ({irreg_pct:.1f}%) "
            f"with valences: {bad_vals}. "
            "Each will produce an 'extraordinary point' under CC subdivision."
        )

    if non_manifold:
        warnings.append(
            f"{len(non_manifold)} non-manifold edge(s) — subdivision will fail on these."
        )

    # ---- 3. Angle / Jacobian quality ---------------------------------------
    min_angle, _ = _compute_quad_angles(verts, faces)
    min_sj, mean_sj = _compute_scaled_jacobians(verts, faces)

    if min_angle < 5.0:
        warnings.append(
            f"Minimum quad angle {min_angle:.2f}° is dangerously small — "
            "degenerate quads will explode on subdivision."
        )
    elif min_angle < 15.0:
        warnings.append(
            f"Minimum quad angle {min_angle:.2f}° is low — "
            "consider running an extra smoothing pass before subdividing."
        )

    if min_sj < 0.0:
        warnings.append(
            f"Minimum Scaled Jacobian {min_sj:.4f} < 0 — "
            "inverted / self-intersecting quads present."
        )
    elif min_sj < 0.1:
        warnings.append(
            f"Minimum Scaled Jacobian {min_sj:.4f} is very low — "
            "near-degenerate quads present."
        )

    # ---- 4. Compatibility decision -----------------------------------------
    quad_threshold   = 99.5 if strict else 97.0
    regular_threshold = 0.99 if strict else 0.95

    is_compatible = (
        quad_pct >= quad_threshold
        and regular_ratio >= regular_threshold
        and len(non_manifold) == 0
        and min_sj >= 0.0
        and min_angle >= 5.0
    )

    return SubdivReport(
        is_compatible=is_compatible,
        quad_percentage=quad_pct,
        regular_vertex_ratio=regular_ratio,
        irregular_vertices=irregular,
        valence_histogram=valence_hist,
        boundary_vertex_count=boundary_count,
        non_manifold_edges=non_manifold,
        has_non_quad_faces=has_non_quad,
        min_quad_angle_deg=min_angle,
        mean_scaled_jacobian=mean_sj,
        warnings=warnings,
    )
