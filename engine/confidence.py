"""QuadForge — Confidence Score & Quality Gate System.

Implements Roadmap §13.9 (Confidence Score System), §11.6 (Quality Gates
Before Returning a Result), and §11.5 (Failure Recovery / Degradation
Modes).

A result is only returned to the user if it passes every quality gate.
If it fails, the engine automatically retries with safer settings. If
every retry still fails, the engine downgrades to a simpler pipeline
path and explains in plain language what happened.

Confidence is a 0..100 score combining:
    - input repair confidence
    - quality breakdown composite
    - gate violations (each violation reduces the score)
    - per-region risk assessment

The module also identifies risk zones (per-face risk in [0, 1]) so the
UI can show a heatmap overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .quality_optimizer import QualityBreakdown


# ---------------------------------------------------------------------------
# Gate configuration
# ---------------------------------------------------------------------------

@dataclass
class QualityGates:
    """Thresholds that a result must meet to be returned without retry."""
    min_quad_percentage: float = 0.95
    max_flipped_ratio: float = 0.001
    max_mean_valence_error: float = 0.35
    max_feature_alignment_error: float = 0.05
    max_symmetry_deviation: float = 0.08
    min_scaled_jacobian: float = 0.1      # absolute min SJ any quad may have
    reject_non_manifold: bool = True

    def relax(self, factor: float = 1.5) -> "QualityGates":
        """Return a relaxed version of the gates for fallback retries."""
        return QualityGates(
            min_quad_percentage=max(0.8, self.min_quad_percentage - 0.05 * factor),
            max_flipped_ratio=self.max_flipped_ratio * factor,
            max_mean_valence_error=self.max_mean_valence_error * factor,
            max_feature_alignment_error=self.max_feature_alignment_error * factor,
            max_symmetry_deviation=self.max_symmetry_deviation * factor,
            min_scaled_jacobian=max(-0.2, self.min_scaled_jacobian - 0.1 * factor),
            reject_non_manifold=self.reject_non_manifold,
        )


# ---------------------------------------------------------------------------
# Confidence report
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceReport:
    score: float = 0.0          # 0..100
    passed: bool = False

    # Per-component contributions (all 0..100)
    input_contribution: float = 0.0
    quality_contribution: float = 0.0
    gates_contribution: float = 0.0

    # Gate outcomes
    gate_violations: List[str] = field(default_factory=list)

    # Per-face risk map, if computed (None if not computed)
    face_risk: Optional[np.ndarray] = None

    # Plain-language explanation
    summary: str = ""
    recommendation: str = ""

    def as_dict(self) -> Dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "input_contribution": self.input_contribution,
            "quality_contribution": self.quality_contribution,
            "gates_contribution": self.gates_contribution,
            "gate_violations": list(self.gate_violations),
            "summary": self.summary,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def check_quality_gates(breakdown: QualityBreakdown,
                        gates: QualityGates,
                        *,
                        is_manifold: bool = True,
                        ) -> List[str]:
    """Return a list of human-readable gate violation messages. An empty
    list means the result passed every gate."""
    violations: List[str] = []

    if breakdown.quad_percentage < gates.min_quad_percentage:
        violations.append(
            f"quad percentage {breakdown.quad_percentage*100:.1f}% < "
            f"required {gates.min_quad_percentage*100:.0f}%"
        )
    if breakdown.flipped_face_ratio > gates.max_flipped_ratio:
        violations.append(
            f"flipped face ratio {breakdown.flipped_face_ratio*100:.2f}% > "
            f"allowed {gates.max_flipped_ratio*100:.2f}%"
        )
    if breakdown.valence_error > gates.max_mean_valence_error:
        violations.append(
            f"mean valence error {breakdown.valence_error:.2f} > "
            f"allowed {gates.max_mean_valence_error:.2f}"
        )
    if breakdown.feature_alignment_error > gates.max_feature_alignment_error:
        violations.append(
            f"feature alignment error {breakdown.feature_alignment_error:.3f} > "
            f"allowed {gates.max_feature_alignment_error:.3f}"
        )
    if breakdown.symmetry_deviation > gates.max_symmetry_deviation:
        violations.append(
            f"symmetry deviation {breakdown.symmetry_deviation:.3f} > "
            f"allowed {gates.max_symmetry_deviation:.3f}"
        )
    if breakdown.min_scaled_jacobian < gates.min_scaled_jacobian:
        violations.append(
            f"min scaled Jacobian {breakdown.min_scaled_jacobian:.3f} < "
            f"required {gates.min_scaled_jacobian:.2f}"
        )
    if gates.reject_non_manifold and not is_manifold:
        violations.append("output mesh is non-manifold")

    return violations


# ---------------------------------------------------------------------------
# Per-face risk map
# ---------------------------------------------------------------------------

def compute_face_risk(vertices: np.ndarray,
                      faces: List[List[int]],
                      *,
                      valence: Optional[np.ndarray] = None,
                      ) -> np.ndarray:
    """Return a (F,) per-face risk score in [0, 1].

    A face is "risky" when:
        - it is not a quad
        - it has a low scaled Jacobian
        - one of its vertices is highly irregular (|valence - 4| > 1)
    """
    n = len(faces)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Compute vertex valence if not provided
    if valence is None:
        valence = np.zeros(len(vertices), dtype=np.int64)
        for f in faces:
            for v in f:
                valence[v] += 1

    risk = np.zeros(n, dtype=np.float64)
    for i, f in enumerate(faces):
        r = 0.0
        if len(f) != 4:
            r += 0.5
        else:
            p = [vertices[v] for v in f]
            sj = _quad_sj(p[0], p[1], p[2], p[3])
            if sj < 0.0:
                r += 0.8
            elif sj < 0.3:
                r += 0.4
            elif sj < 0.5:
                r += 0.15

        # Irregular vertices raise the risk
        for v in f:
            err = abs(int(valence[v]) - 4)
            if err >= 2:
                r += 0.1 * err
        risk[i] = min(1.0, r)
    return risk


def _quad_sj(p0, p1, p2, p3) -> float:
    """Signed min scaled Jacobian. See quality_optimizer._scaled_jacobian_quad
    for the full derivation. Negative values indicate a folded/bowtie quad
    and are the signal the quality gate uses to reject flipped output."""
    pts = (p0, p1, p2, p3)
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
        sign = 1.0 if float(np.dot(cross, n_ref)) >= 0.0 else -1.0
        sign = -sign
        sj = sign * mag / (l1 * l2)
        if sj < worst:
            worst = sj
    return worst


# ---------------------------------------------------------------------------
# Top-level confidence computation
# ---------------------------------------------------------------------------

def compute_confidence(breakdown: QualityBreakdown,
                       gates: QualityGates,
                       *,
                       repair_confidence: float = 1.0,
                       is_manifold: bool = True,
                       vertices: Optional[np.ndarray] = None,
                       faces: Optional[List[List[int]]] = None,
                       ) -> ConfidenceReport:
    """Combine everything into a single ConfidenceReport."""
    report = ConfidenceReport()

    # Three weighted contributions
    report.input_contribution = 100.0 * float(max(0.0, min(1.0, repair_confidence)))

    report.quality_contribution = 100.0 * float(
        max(0.0, min(1.0, breakdown.composite))
    )

    violations = check_quality_gates(breakdown, gates, is_manifold=is_manifold)
    report.gate_violations = violations

    gates_score = max(0.0, 1.0 - 0.15 * len(violations))
    report.gates_contribution = 100.0 * gates_score

    # Weighted composite
    report.score = (
        0.15 * report.input_contribution
        + 0.55 * report.quality_contribution
        + 0.30 * report.gates_contribution
    )
    report.passed = len(violations) == 0

    if vertices is not None and faces is not None:
        report.face_risk = compute_face_risk(vertices, faces)

    # Plain-language summary
    report.summary, report.recommendation = _build_summary(
        report, breakdown, gates
    )
    return report


def _build_summary(report: ConfidenceReport,
                   breakdown: QualityBreakdown,
                   gates: QualityGates) -> Tuple[str, str]:
    score = report.score
    if score >= 90.0:
        s = "Excellent result — topology looks production-ready."
        rec = "You can use this output directly."
    elif score >= 75.0:
        s = "Good result with minor imperfections."
        rec = "Inspect any flagged risk zones before export."
    elif score >= 55.0:
        s = "Acceptable result, but some quality gates were relaxed."
        rec = ("Consider increasing target quad count or adaptivity, "
               "or re-running with smoothing +5 iterations.")
    elif score >= 35.0:
        s = "Result has noticeable topology issues."
        rec = ("Enable mesh repair, reduce hard-edge angle to 20°, "
               "or try a different preset.")
    else:
        s = "Low-confidence result — the engine downgraded to a fallback path."
        rec = ("Check the input mesh for non-manifold geometry, "
               "self-intersections, or sparse sampling. "
               "Re-run with mesh repair enabled.")

    if report.gate_violations:
        s += f" {len(report.gate_violations)} gate(s) failed: "
        s += "; ".join(report.gate_violations[:3])
        if len(report.gate_violations) > 3:
            s += f"; … ({len(report.gate_violations) - 3} more)"
    return s, rec


# ---------------------------------------------------------------------------
# Retry planner
# ---------------------------------------------------------------------------

@dataclass
class RetryPlan:
    """What the pipeline should change before retrying a failed solve."""
    relaxed_gates: QualityGates
    param_overrides: Dict[str, float]
    explanation: str


def plan_retry(violations: List[str],
               base_gates: QualityGates) -> RetryPlan:
    """Produce a concrete retry plan based on which gates failed.

    Each gate has a canonical mitigation: e.g. if the quad percentage is
    too low, retry with denser target and more smoothing. If feature
    alignment failed, retry with harder snap.
    """
    overrides: Dict[str, float] = {}
    reasons: List[str] = []

    for v in violations:
        if "quad percentage" in v:
            overrides["target_quad_count_mul"] = 1.2
            overrides["smooth_iter_add"] = 8
            reasons.append("increase density and smoothing")
        elif "flipped face" in v:
            overrides["smooth_iter_add"] = 15
            overrides["smooth_strength_add"] = 0.1
            reasons.append("more smoothing to un-flip faces")
        elif "valence error" in v:
            overrides["curvature_adaptivity_add"] = 0.1
            reasons.append("increase adaptivity to place singularities better")
        elif "feature alignment" in v:
            overrides["feature_snap_mul"] = 1.8
            reasons.append("stronger feature snapping")
        elif "symmetry" in v:
            reasons.append("tighten symmetry enforcement")
        elif "scaled Jacobian" in v:
            overrides["smooth_iter_add"] = 10
            reasons.append("smoothing to improve quad shape")

    relaxed = base_gates.relax(1.4)
    explanation = (
        "Retry plan: " + ", ".join(reasons) + ". "
        "Quality gates relaxed to the production-tolerant level."
        if reasons else "Retry with relaxed gates."
    )
    return RetryPlan(relaxed_gates=relaxed,
                     param_overrides=overrides,
                     explanation=explanation)
