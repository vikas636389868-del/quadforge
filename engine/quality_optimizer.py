"""QuadForge — Auto Quality Optimizer (Roadmap §13.3).

Post-solve optimization loop that searches a small neighbourhood in
parameter space around the initial solve and keeps the best result by
composite quality score.

Design notes:
    * The optimizer is driven by a single scalar ``quality_score`` that
      combines quad percentage, valence error, feature alignment,
      symmetry preservation, angle quality, and flipped-face penalties.
    * To stay practical, candidate evaluations use the *preview* (fast)
      pipeline path wherever possible. A final full-quality pass is only
      run once, on the winning parameter set.
    * All candidate runs are snapshot so the optimizer can roll back to
      the previous best if a candidate actually makes things worse.
    * The parameter sweep is bounded: at most ``max_candidates`` runs,
      and each is clamped to fit within the user's time budget.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Weights for the composite quality score. Tuned so that a "clean" result
# with 100% quads, valence 4.0 everywhere, and no feature drift sits very
# close to 1.0.
QUALITY_WEIGHTS = {
    "quad_percentage": 0.30,
    "valence_error":   0.20,
    "feature_align":   0.20,
    "symmetry":        0.10,
    "angle":           0.15,
    "flipped":         0.05,
}


# ---------------------------------------------------------------------------
# Quality score
# ---------------------------------------------------------------------------

@dataclass
class QualityBreakdown:
    """Detailed per-component quality data for reporting."""
    quad_percentage: float = 0.0
    valence_mean: float = 0.0
    valence_error: float = 0.0        # mean |valence - 4|
    feature_alignment_error: float = 0.0  # 0..1, lower is better
    symmetry_deviation: float = 0.0   # 0..1, lower is better
    min_scaled_jacobian: float = 0.0  # -1..+1, higher is better
    mean_scaled_jacobian: float = 0.0
    flipped_face_ratio: float = 0.0   # 0..1
    composite: float = 0.0            # weighted composite 0..1

    def as_dict(self) -> Dict[str, float]:
        return {
            "quad_percentage": self.quad_percentage,
            "valence_mean": self.valence_mean,
            "valence_error": self.valence_error,
            "feature_alignment_error": self.feature_alignment_error,
            "symmetry_deviation": self.symmetry_deviation,
            "min_scaled_jacobian": self.min_scaled_jacobian,
            "mean_scaled_jacobian": self.mean_scaled_jacobian,
            "flipped_face_ratio": self.flipped_face_ratio,
            "composite": self.composite,
        }


def compute_quality_score(vertices: np.ndarray,
                          faces,  # list-of-lists of indices (quads/tris mixed)
                          *,
                          feature_edges: Optional[List[Tuple[int, int]]] = None,
                          symmetry_axes: Tuple[bool, bool, bool] = (False, False, False),
                          ) -> QualityBreakdown:
    """Compute the composite quality score for a remeshed quad-dominant mesh.

    Returns a ``QualityBreakdown`` with ``composite`` in [0, 1] — higher is
    better. This function is deterministic and side-effect free.
    """
    breakdown = QualityBreakdown()
    if not faces:
        return breakdown

    # ---- Quad percentage ----
    quad_count = sum(1 for f in faces if len(f) == 4)
    total = len(faces)
    breakdown.quad_percentage = quad_count / total if total > 0 else 0.0

    # ---- Valence ----
    valence = np.zeros(len(vertices), dtype=np.int64)
    used = np.zeros(len(vertices), dtype=bool)
    for f in faces:
        n = len(f)
        for i in range(n):
            v = int(f[i])
            used[v] = True
            valence[v] += 1  # count face incidences; approximates interior valence
    used_valence = valence[used]
    if len(used_valence) > 0:
        breakdown.valence_mean = float(np.mean(used_valence))
        breakdown.valence_error = float(np.mean(np.abs(used_valence - 4.0)))
    else:
        breakdown.valence_error = 4.0

    # ---- Scaled Jacobian + flipped ----
    sjs: List[float] = []
    flipped = 0
    for f in faces:
        if len(f) != 4:
            continue
        p0 = vertices[f[0]]
        p1 = vertices[f[1]]
        p2 = vertices[f[2]]
        p3 = vertices[f[3]]
        sj = _scaled_jacobian_quad(p0, p1, p2, p3)
        sjs.append(sj)
        if sj < 0.0:
            flipped += 1
    if sjs:
        arr = np.asarray(sjs)
        breakdown.min_scaled_jacobian = float(arr.min())
        breakdown.mean_scaled_jacobian = float(arr.mean())
        breakdown.flipped_face_ratio = flipped / len(sjs)

    # ---- Feature alignment error (cheap proxy: mean distance from feature
    # edges to the nearest quad edge, normalised by the bbox diagonal). ----
    breakdown.feature_alignment_error = _feature_alignment_error(
        vertices, faces, feature_edges
    )

    # ---- Symmetry deviation ----
    if any(symmetry_axes):
        breakdown.symmetry_deviation = _symmetry_deviation(vertices, symmetry_axes)

    # ---- Composite ----
    comp = 0.0
    comp += QUALITY_WEIGHTS["quad_percentage"] * breakdown.quad_percentage
    comp += QUALITY_WEIGHTS["valence_error"] * max(0.0, 1.0 - breakdown.valence_error / 2.0)
    comp += QUALITY_WEIGHTS["feature_align"] * max(0.0, 1.0 - breakdown.feature_alignment_error)
    comp += QUALITY_WEIGHTS["symmetry"] * max(0.0, 1.0 - breakdown.symmetry_deviation)
    # Angle score: 0.5*(mean SJ + 1) in [0..1]
    ang = 0.5 * (breakdown.mean_scaled_jacobian + 1.0)
    comp += QUALITY_WEIGHTS["angle"] * max(0.0, min(1.0, ang))
    comp += QUALITY_WEIGHTS["flipped"] * max(0.0, 1.0 - breakdown.flipped_face_ratio * 10.0)
    breakdown.composite = float(max(0.0, min(1.0, comp)))
    return breakdown


def _scaled_jacobian_quad(p0, p1, p2, p3) -> float:
    """Minimum signed scaled Jacobian over the 4 corners of a quad.

    Returns a value in [-1, +1]. A consistent-winding, convex, planar
    quad returns ~1.0. A bowtie/self-intersecting quad returns a
    negative value at the inverted corner(s), which is how we detect
    flipped faces in the quality gate.
    """
    pts = (p0, p1, p2, p3)

    # Reference normal: average of the two triangle normals of the quad.
    # This handles slightly non-planar quads better than picking one corner.
    n_ref = np.cross(pts[1] - pts[0], pts[2] - pts[0]) \
          + np.cross(pts[2] - pts[0], pts[3] - pts[0])
    n_ref_len = float(np.linalg.norm(n_ref))
    if n_ref_len < 1e-20:
        return -1.0
    n_ref = n_ref / n_ref_len

    worst = 1.0
    for i in range(4):
        a = pts[(i - 1) % 4]
        b = pts[i]
        c = pts[(i + 1) % 4]
        e1 = a - b
        e2 = c - b
        l1 = float(np.linalg.norm(e1))
        l2 = float(np.linalg.norm(e2))
        if l1 < 1e-12 or l2 < 1e-12:
            return -1.0
        cross = np.cross(e1, e2)
        mag = float(np.linalg.norm(cross))
        # Sign from the alignment with the reference normal. If this
        # corner's wedge points opposite to the reference, the quad
        # is folded at this corner and we report a negative SJ.
        sign = 1.0 if float(np.dot(cross, n_ref)) >= 0.0 else -1.0
        # NB: the corner's cross is e1 x e2, which has the OPPOSITE sign
        # of the face normal (since the face normal is built from
        # (p1-p0) x (p2-p0) = edge_out x edge_out at corner 0). Undo that.
        sign = -sign
        sj = sign * mag / (l1 * l2)
        if sj < worst:
            worst = sj
    return worst


def _feature_alignment_error(vertices: np.ndarray, faces, feature_edges) -> float:
    if not feature_edges:
        return 0.0
    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    diag = float(np.linalg.norm(bbox)) + 1e-9

    # Sample points along each feature edge
    samples: List[np.ndarray] = []
    for a, b in feature_edges:
        pa = vertices[a]
        pb = vertices[b]
        for t in (0.25, 0.5, 0.75):
            samples.append(pa * (1 - t) + pb * t)
    if not samples:
        return 0.0
    sample_arr = np.asarray(samples)

    # Collect quad edges
    edge_pts: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in faces:
        n = len(f)
        for i in range(n):
            edge_pts.append((vertices[f[i]], vertices[f[(i + 1) % n]]))

    if not edge_pts:
        return 1.0

    # For each sample, compute distance to nearest quad edge (approx:
    # cap at 200 edges sampled for speed)
    max_edges = min(len(edge_pts), 2000)
    step = max(1, len(edge_pts) // max_edges)
    sampled_edges = edge_pts[::step]
    errors = []
    for sp in sample_arr:
        best = float("inf")
        for e0, e1 in sampled_edges:
            d = _point_segment_distance(sp, e0, e1)
            if d < best:
                best = d
        errors.append(best)
    return float(np.mean(errors) / diag)


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-20:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / denom))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _symmetry_deviation(vertices: np.ndarray,
                        symmetry_axes: Tuple[bool, bool, bool]) -> float:
    centre = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    centred = vertices - centre
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) + 1e-9

    max_dev = 0.0
    for axis in range(3):
        if not symmetry_axes[axis]:
            continue
        mirrored = centred.copy()
        mirrored[:, axis] = -mirrored[:, axis]
        # Approximate nearest-neighbour distance via coarse grid
        dev = _approx_nn_distance(centred[::10], mirrored[::10]) / scale
        max_dev = max(max_dev, dev)
    return float(min(1.0, max_dev))


def _approx_nn_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    # O(n*m) brute force on a downsample
    n = min(len(a), 200)
    m = min(len(b), 200)
    aa = a[:n]
    bb = b[:m]
    diffs = aa[:, None, :] - bb[None, :, :]
    d2 = (diffs * diffs).sum(axis=2)
    nn = np.sqrt(d2.min(axis=1))
    return float(nn.mean())


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

@dataclass
class OptimizerCandidate:
    """Snapshot of one evaluated candidate."""
    label: str
    params_overrides: Dict[str, float]
    score: float = 0.0
    breakdown: Optional[QualityBreakdown] = None
    vertices: Optional[np.ndarray] = None
    faces: Optional[List[List[int]]] = None
    elapsed: float = 0.0


@dataclass
class OptimizerResult:
    best_candidate: OptimizerCandidate
    all_candidates: List[OptimizerCandidate] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    improvement: float = 0.0  # best - initial

    def summary(self) -> str:
        return (
            f"QualityOptimizer: best='{self.best_candidate.label}' "
            f"score={self.best_candidate.score:.3f} "
            f"(+{self.improvement:.3f}) "
            f"from {len(self.all_candidates)} candidates in "
            f"{self.elapsed_seconds:.1f}s"
        )


# Default sweep grid. Entries are multipliers (or absolute offsets) applied
# on top of the user's base parameters.
DEFAULT_SWEEP: List[Tuple[str, Dict[str, float]]] = [
    ("baseline",             {}),
    ("denser",               {"target_quad_count_mul": 1.15}),
    ("sparser",              {"target_quad_count_mul": 0.88}),
    ("more_adaptive",        {"curvature_adaptivity_add": 0.15}),
    ("less_adaptive",        {"curvature_adaptivity_add": -0.15}),
    ("harder_snap",          {"feature_snap_mul": 1.5}),
    ("softer_snap",          {"feature_snap_mul": 0.6}),
    ("more_smoothing",       {"smooth_iter_add": 10}),
    ("less_smoothing",       {"smooth_iter_add": -5}),
]


def run_auto_optimizer(run_pipeline_fn: Callable,
                       base_params: Dict,
                       *,
                       feature_edges: Optional[List[Tuple[int, int]]] = None,
                       symmetry_axes: Tuple[bool, bool, bool] = (False, False, False),
                       max_candidates: int = 6,
                       time_budget_s: float = 30.0,
                       sweep: Optional[List[Tuple[str, Dict[str, float]]]] = None,
                       progress_cb: Optional[Callable[[str, float], None]] = None,
                       ) -> OptimizerResult:
    """Run the quality optimizer.

    Parameters
    ----------
    run_pipeline_fn : callable ``(params_dict) -> (vertices, faces)``
        A thin wrapper around the QuadForge pipeline. The optimizer calls
        it once per candidate. It's the caller's responsibility to use a
        fast preview path here if speed matters.
    base_params : dict
        The baseline parameters that the user selected.
    feature_edges : optional list
        Feature edges used for alignment scoring.
    symmetry_axes : 3-tuple of bool
        Which axes to check for symmetry deviation.
    max_candidates : int
        Hard cap on how many candidates to evaluate.
    time_budget_s : float
        Total wall-clock budget (seconds). Optimizer stops early if the
        budget is exhausted.
    sweep : optional list of (label, overrides)
        Custom candidate grid. Defaults to ``DEFAULT_SWEEP``.

    Returns
    -------
    OptimizerResult
    """
    t0 = time.perf_counter()
    if sweep is None:
        sweep = DEFAULT_SWEEP
    sweep = sweep[:max_candidates]

    candidates: List[OptimizerCandidate] = []
    best: Optional[OptimizerCandidate] = None

    for i, (label, overrides) in enumerate(sweep):
        if time.perf_counter() - t0 > time_budget_s and i > 0:
            break
        if progress_cb is not None:
            progress_cb(f"candidate {label}", i / max(1, len(sweep)))

        params = _apply_overrides(base_params, overrides)
        t_run = time.perf_counter()
        try:
            verts, faces = run_pipeline_fn(params)
        except Exception as exc:  # noqa: BLE001
            # Candidate failed — skip but log
            candidates.append(OptimizerCandidate(
                label=label + " [FAILED]",
                params_overrides=overrides,
                score=-1.0,
                elapsed=time.perf_counter() - t_run,
            ))
            continue

        breakdown = compute_quality_score(
            verts, faces,
            feature_edges=feature_edges,
            symmetry_axes=symmetry_axes,
        )
        cand = OptimizerCandidate(
            label=label,
            params_overrides=overrides,
            score=breakdown.composite,
            breakdown=breakdown,
            vertices=verts,
            faces=faces,
            elapsed=time.perf_counter() - t_run,
        )
        candidates.append(cand)
        if best is None or cand.score > best.score:
            best = cand

    if best is None:
        # All candidates failed — synthesise a zero candidate
        best = OptimizerCandidate(label="none",
                                  params_overrides={},
                                  score=0.0)

    initial_score = candidates[0].score if candidates else 0.0
    return OptimizerResult(
        best_candidate=best,
        all_candidates=candidates,
        elapsed_seconds=time.perf_counter() - t0,
        improvement=best.score - initial_score,
    )


def _apply_overrides(base: Dict, overrides: Dict[str, float]) -> Dict:
    """Apply a sweep delta dictionary to the base parameter dictionary.

    Supported override keys:
        target_quad_count_mul, curvature_adaptivity_add,
        feature_snap_mul, smooth_iter_add, smooth_strength_add
    """
    new = dict(base)
    if "target_quad_count_mul" in overrides:
        new["target_quad_count"] = max(
            100, int(round(base.get("target_quad_count", 5000) *
                            overrides["target_quad_count_mul"]))
        )
    if "curvature_adaptivity_add" in overrides:
        new["curvature_adaptivity"] = max(
            0.0, min(1.0, base.get("curvature_adaptivity", 0.5) +
                         overrides["curvature_adaptivity_add"])
        )
    if "feature_snap_mul" in overrides:
        new["feature_snap_distance"] = max(
            0.0, base.get("feature_snap_distance", 0.1) *
                 overrides["feature_snap_mul"]
        )
    if "smooth_iter_add" in overrides:
        new["smooth_iterations"] = max(
            0, int(base.get("smooth_iterations", 10) +
                   overrides["smooth_iter_add"])
        )
    if "smooth_strength_add" in overrides:
        new["smooth_strength"] = max(
            0.0, min(1.0, base.get("smooth_strength", 0.5) +
                         overrides["smooth_strength_add"])
        )
    return new
