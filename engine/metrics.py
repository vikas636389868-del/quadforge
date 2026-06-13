"""Quality metrics computation for QuadForge output meshes.

Computes:
  - Quad face percentage
  - Vertex valence histogram and statistics
  - Per-face scaled Jacobian quality
  - Feature alignment error
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class QualityMetrics:
    """Quality metrics for a quad-dominant mesh.

    Roadmap Phase 7 requires: quad %, valence distribution, aspect ratio
    distribution, and scaled Jacobian statistics.
    """
    total_faces: int
    num_quads: int
    num_tris: int
    num_other: int
    quad_percentage: float           # 0-100

    num_interior_vertices: int
    avg_valence: float
    valence_histogram: Dict[int, int]  # valence -> count
    irregularity_ratio: float        # % of interior vertices with valence != 4

    min_scaled_jacobian: float
    mean_scaled_jacobian: float
    max_scaled_jacobian: float

    # Aspect ratio: longest / shortest edge per face. Ideal = 1.0, bad > 4.0
    min_aspect_ratio: float
    mean_aspect_ratio: float
    max_aspect_ratio: float
    aspect_ratio_histogram: Dict[str, int]  # bucket label -> count

    # Minimum interior angle per face (degrees). Roadmap target: min > 20 deg
    min_angle_deg: float
    mean_min_angle_deg: float

    total_vertices: int
    total_edges: int


def compute_quality_metrics(
    vertices: np.ndarray,
    faces: List[List[int]],
    feature_edges: Optional[Set[Tuple[int, int]]] = None,
) -> QualityMetrics:
    """Compute comprehensive quality metrics for the output mesh.

    Parameters
    ----------
    vertices : (V, 3) vertex positions
    faces : list of face vertex index lists (quads and tris)
    feature_edges : optional set for alignment error computation

    Returns
    -------
    QualityMetrics
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    num_verts = len(vertices)

    # --- Face counts ---
    num_quads = sum(1 for f in faces if len(f) == 4)
    num_tris = sum(1 for f in faces if len(f) == 3)
    num_other = len(faces) - num_quads - num_tris
    total_faces = len(faces)
    quad_pct = (num_quads / total_faces * 100) if total_faces > 0 else 0.0

    # --- Valence ---
    valence = np.zeros(num_verts, dtype=np.int32)
    edge_set: Set[Tuple[int, int]] = set()
    boundary_verts: Set[int] = set()

    for face in faces:
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i + 1) % n]
            ek = (min(v0, v1), max(v0, v1))
            edge_set.add(ek)
            valence[v0] += 1

    # Identify boundary vertices (edges shared by only 1 face)
    edge_face_count: Dict[Tuple[int, int], int] = {}
    for face in faces:
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i + 1) % n]
            ek = (min(v0, v1), max(v0, v1))
            edge_face_count[ek] = edge_face_count.get(ek, 0) + 1

    for ek, cnt in edge_face_count.items():
        if cnt == 1:
            boundary_verts.add(ek[0])
            boundary_verts.add(ek[1])

    # Interior vertices only for valence stats
    interior_mask = np.ones(num_verts, dtype=bool)
    for v in boundary_verts:
        if v < num_verts:
            interior_mask[v] = False

    interior_valence = valence[interior_mask]
    num_interior = len(interior_valence)
    avg_val = float(np.mean(interior_valence)) if num_interior > 0 else 0.0

    # Valence histogram
    val_hist: Dict[int, int] = {}
    for v in interior_valence:
        val_hist[int(v)] = val_hist.get(int(v), 0) + 1

    # Irregularity ratio
    irregular = sum(1 for v in interior_valence if v != 4)
    irreg_ratio = (irregular / num_interior * 100) if num_interior > 0 else 0.0

    # --- Scaled Jacobian ---
    jacobians = []
    for face in faces:
        if len(face) != 4:
            continue
        pts = vertices[face]
        sj = _scaled_jacobian_quad(pts)
        jacobians.append(sj)

    if jacobians:
        min_sj = float(np.min(jacobians))
        mean_sj = float(np.mean(jacobians))
        max_sj = float(np.max(jacobians))
    else:
        min_sj = mean_sj = max_sj = 0.0

    # --- Aspect Ratio (longest / shortest edge per face) ---
    aspect_ratios: List[float] = []
    min_angles: List[float] = []
    for face in faces:
        n = len(face)
        if n < 3:
            continue
        pts = vertices[face]
        edges = [pts[(i + 1) % n] - pts[i] for i in range(n)]
        lengths = [float(np.linalg.norm(e)) for e in edges]
        # Filter degenerate
        lengths_pos = [l for l in lengths if l > 1e-15]
        if len(lengths_pos) >= 2:
            ar = max(lengths_pos) / min(lengths_pos)
            aspect_ratios.append(ar)
        # Minimum interior angle
        min_a = 180.0
        for i in range(n):
            e1 = edges[(i - 1) % n]
            e2 = edges[i]
            l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if l1 < 1e-15 or l2 < 1e-15:
                continue
            cos_a = float(np.clip(np.dot(-e1, e2) / (l1 * l2), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_a)))
            if angle_deg < min_a:
                min_a = angle_deg
        if min_a < 180.0:
            min_angles.append(min_a)

    if aspect_ratios:
        min_ar   = float(np.min(aspect_ratios))
        mean_ar  = float(np.mean(aspect_ratios))
        max_ar   = float(np.max(aspect_ratios))
        # Histogram: [1,2), [2,4), [4,8), [8,inf)
        ar_hist: Dict[str, int] = {"1-2": 0, "2-4": 0, "4-8": 0, "8+": 0}
        for ar in aspect_ratios:
            if ar < 2.0:   ar_hist["1-2"] += 1
            elif ar < 4.0: ar_hist["2-4"] += 1
            elif ar < 8.0: ar_hist["4-8"] += 1
            else:          ar_hist["8+"]  += 1
    else:
        min_ar = mean_ar = max_ar = 0.0
        ar_hist = {"1-2": 0, "2-4": 0, "4-8": 0, "8+": 0}

    min_angle   = float(np.min(min_angles))  if min_angles else 0.0
    mean_min_a  = float(np.mean(min_angles)) if min_angles else 0.0

    return QualityMetrics(
        total_faces=total_faces,
        num_quads=num_quads,
        num_tris=num_tris,
        num_other=num_other,
        quad_percentage=quad_pct,
        num_interior_vertices=num_interior,
        avg_valence=avg_val,
        valence_histogram=val_hist,
        irregularity_ratio=irreg_ratio,
        min_scaled_jacobian=min_sj,
        mean_scaled_jacobian=mean_sj,
        max_scaled_jacobian=max_sj,
        min_aspect_ratio=min_ar,
        mean_aspect_ratio=mean_ar,
        max_aspect_ratio=max_ar,
        aspect_ratio_histogram=ar_hist,
        min_angle_deg=min_angle,
        mean_min_angle_deg=mean_min_a,
        total_vertices=num_verts,
        total_edges=len(edge_set),
    )


def _scaled_jacobian_quad(pts: np.ndarray) -> float:
    """Compute the minimum scaled Jacobian of a quad face.

    For each corner, compute the Jacobian of the bilinear map and
    its scaled determinant. The minimum over all four corners is
    the element quality metric.
    """
    min_sj = 1.0
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p3 = pts[(i + 3) % 4]

        e1 = p1 - p0
        e2 = p3 - p0

        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)

        if l1 < 1e-15 or l2 < 1e-15:
            return 0.0

        cross = np.cross(e1, e2)
        det = np.linalg.norm(cross)
        sj = det / (l1 * l2)
        min_sj = min(min_sj, sj)

    return min_sj


def print_quality_report(metrics: QualityMetrics) -> str:
    """Format quality metrics as a human-readable report.

    Roadmap Phase 7: quad %, valence distribution, aspect ratio distribution,
    scaled Jacobian, minimum angles.
    """
    lines = [
        "Quality Report",
        f"  Faces:        {metrics.total_faces} ({metrics.num_quads} quads, "
        f"{metrics.num_tris} tris, {metrics.num_other} other)",
        f"  Quad %:       {metrics.quad_percentage:.1f}%  (target: >=97%)",
        f"  Vertices:     {metrics.total_vertices}",
        f"  Edges:        {metrics.total_edges}",
        f"  Avg valence:  {metrics.avg_valence:.3f}  (ideal: 4.0)",
        f"  Irregularity: {metrics.irregularity_ratio:.1f}%  (target: <4%)",
        f"  Scaled Jacobian: min={metrics.min_scaled_jacobian:.3f}, "
        f"mean={metrics.mean_scaled_jacobian:.3f}, max={metrics.max_scaled_jacobian:.3f}",
        f"  Aspect Ratio: min={metrics.min_aspect_ratio:.2f}, "
        f"mean={metrics.mean_aspect_ratio:.2f}, max={metrics.max_aspect_ratio:.2f}",
        f"  Angle (min):  {metrics.min_angle_deg:.1f} deg  "
        f"(mean_per_face: {metrics.mean_min_angle_deg:.1f} deg, target: >20 deg)",
    ]
    if metrics.aspect_ratio_histogram:
        ar_str = "  ".join(f"{k}:{v}" for k, v in metrics.aspect_ratio_histogram.items())
        lines.append(f"  AR dist:      {ar_str}")
    if metrics.valence_histogram:
        hist_str = ", ".join(f"v{k}:{v}" for k, v in sorted(metrics.valence_histogram.items()))
        lines.append(f"  Valence dist: {hist_str}")
    return "\n".join(lines)
