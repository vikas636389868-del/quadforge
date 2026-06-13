"""QuadForge — Dominance Pipeline Integrator.

Top-level orchestrator that wires together every module from Part XI
and Part XIII of the roadmap into a single callable:

    repair -> classify -> (progressive preview) -> dispatch ->
    solve -> confidence gate -> retry -> auto-optimize -> edge-flow
    refine -> final confidence report

Every stage is optional and guarded by a flag on the incoming settings
object, so existing callers of the plain ``pipeline.run_pipeline`` are
not affected. The dominance pipeline is opt-in via
``settings.enable_dominance_mode = True`` (see ``properties.py``).

Call signature is deliberately identical to ``run_pipeline`` so the
existing operator can swap in via a single ``if`` branch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import pipeline as core_pipeline
from .mesh_repair import repair_mesh, RepairReport, voxel_rescue_remesh
from .determinism import DeterministicContext, canonicalize_mesh
from .artist_intelligence import run_artist_intelligence, MeshClass, SemanticFeatures
from .quality_optimizer import (
    compute_quality_score,
    run_auto_optimizer,
    QualityBreakdown,
    OptimizerResult,
)
from .confidence import (
    QualityGates,
    compute_confidence,
    ConfidenceReport,
    plan_retry,
)
from .edge_flow_refine import refine_edge_flow, EdgeFlowReport
from .progressive_preview import (
    run_progressive_preview,
    PreviewCache,
)
from .mesh_classifier import (
    decide_tier,
    dispatch_with_fallback,
    DispatchDecision,
    TierRunResult,
    SolverTier,
)


ProgressCallback = Callable[[str, float], None]


def _noop_progress(stage: str, progress: float) -> None:
    print(f"[QuadForge/dominance] {stage}: {progress*100:.0f}%")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DominanceResult:
    """Everything the dominance pipeline produces in a single object."""

    vertices: np.ndarray
    faces: List[List[int]]

    repair_report: Optional[RepairReport] = None
    mesh_class: Optional[MeshClass] = None
    semantic_features: Optional[SemanticFeatures] = None
    dispatch_decision: Optional[DispatchDecision] = None
    tier_attempts: List[TierRunResult] = field(default_factory=list)
    quality_breakdown: Optional[QualityBreakdown] = None
    confidence_report: Optional[ConfidenceReport] = None
    optimizer_result: Optional[OptimizerResult] = None
    edge_flow_report: Optional[EdgeFlowReport] = None
    preview_cache: Optional[PreviewCache] = None

    elapsed_seconds: float = 0.0
    retries_used: int = 0

    def summary(self) -> str:
        lines: List[str] = [
            f"QuadForge Dominance result: {len(self.vertices)}v / {len(self.faces)}f "
            f"in {self.elapsed_seconds:.2f}s",
        ]
        if self.repair_report is not None:
            lines.append("  " + self.repair_report.summary())
        if self.mesh_class is not None:
            lines.append(f"  classified: {self.mesh_class.describe()}")
        if self.dispatch_decision is not None:
            lines.append(f"  dispatch: {self.dispatch_decision.describe()}")
        if self.quality_breakdown is not None:
            lines.append(
                f"  quality: composite={self.quality_breakdown.composite:.3f}, "
                f"quads={self.quality_breakdown.quad_percentage*100:.1f}%, "
                f"val_err={self.quality_breakdown.valence_error:.2f}"
            )
        if self.confidence_report is not None:
            lines.append(
                f"  confidence: {self.confidence_report.score:.1f}/100 "
                f"{'PASS' if self.confidence_report.passed else 'GATED'}"
            )
        if self.optimizer_result is not None:
            lines.append("  " + self.optimizer_result.summary())
        if self.edge_flow_report is not None:
            lines.append("  " + self.edge_flow_report.summary())
        if self.retries_used:
            lines.append(f"  retries: {self.retries_used}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Settings normalization
# ---------------------------------------------------------------------------

def _get(settings, name: str, default):
    """Safely read an attribute from a settings object (PropertyGroup or
    dict-like) with a fallback default."""
    if settings is None:
        return default
    if hasattr(settings, name):
        return getattr(settings, name)
    if isinstance(settings, dict):
        return settings.get(name, default)
    return default


def _extract_params_dict(settings) -> dict:
    """Build a plain dict of the tunable parameters the optimizer / retry
    planner can mutate. Mirrors the overrides supported by
    ``quality_optimizer._apply_overrides``.
    """
    return {
        "target_quad_count":    int(_get(settings, "target_quad_count", 5000)),
        "curvature_adaptivity": float(_get(settings, "curvature_adaptivity", 0.5)),
        "feature_snap_distance": float(_get(settings, "feature_snap_distance", 0.1)),
        "smooth_iterations":    int(_get(settings, "smooth_iterations", 10)),
        "smooth_strength":      float(_get(settings, "smooth_strength", 0.5)),
    }


# ---------------------------------------------------------------------------
# Tier runners (primary / secondary / tertiary) — thin wrappers over
# the existing core pipeline with progressively-simpler settings.
# ---------------------------------------------------------------------------

class _SettingsSnapshot:
    """A mutable copy of the user's settings that the retry planner can
    tweak without touching the actual Blender PropertyGroup."""

    # Sane defaults so the core pipeline never sees None on a field it
    # expects to unpack as int/bool/str. These mirror the defaults in
    # ``properties.QFSettingsPropertyGroup`` AND ``bridge.QFParams`` so
    # either consumer can read from a snapshot safely.
    _DEFAULTS = {
        # --- Target & sizing ---
        "target_quad_count": 5000,
        "curvature_adaptivity": 0.5,
        "exact_quad_count": False,
        # --- Feature detection ---
        "auto_detect_hard_edges": True,
        "hard_edge_angle_deg": 30.0,
        "use_normals": False,
        "use_materials": False,
        "use_vertex_colors": False,
        "use_uv_seams": False,
        "feature_snap_distance": 0.1,
        # --- Symmetry (snapshot form: three booleans) ---
        "symmetry_x": False,
        "symmetry_y": False,
        "symmetry_z": False,
        # --- Quality ---
        "smooth_iterations": 10,
        "smooth_strength": 0.5,
        # --- Performance ---
        "num_threads": 0,
        "use_gpu": False,
        # --- Algorithm selection ---
        "field_solver": 'KNOPPEL',
        "param_method": 'MIQ',
        "extraction_method": 'ISO',
        "preset": 'CUSTOM',
        # --- Output ---
        "shade_smooth_output": True,
        "keep_original": True,
        # --- Dominance flags ---
        "enable_dominance_mode": False,
        "enable_mesh_repair": True,
        "enable_auto_optimizer": False,
        "enable_edge_flow_refine": True,
        "enable_artist_intelligence": True,
        "reproducible_mode": False,
        "reproducible_seed": 0xC0FFEE,
        "confidence_target": 75.0,
        "time_budget_seconds": 60.0,
    }

    def __init__(self, base_settings):
        for attr, default in self._DEFAULTS.items():
            val = _get(base_settings, attr, default)
            if val is None:
                val = default
            setattr(self, attr, val)
        # Critical: force the dominance flag OFF so recursive calls into
        # core_pipeline.run_pipeline do not re-enter this module and cause
        # infinite recursion.
        self.enable_dominance_mode = False


# ---------------------------------------------------------------------------
# Local QFParams-shaped params object
# ---------------------------------------------------------------------------
# The core pipeline accesses ``params`` as an *object with attributes*
# (``params.target_quad_count``, ``params.smooth_iterations``, …). We can't
# import ``bridge.QFParams`` here because ``bridge`` imports ``bpy`` at
# module load time and that would break the dominance pipeline's
# standalone-import tests. So we duck-type a local replacement with the
# same attribute shape.

@dataclass
class _LocalQFParams:
    target_quad_count: int = 5000
    curvature_adaptivity: float = 0.5
    exact_quad_count: bool = False
    auto_detect_hard_edges: bool = True
    hard_edge_angle_deg: float = 30.0
    use_normals: bool = False
    use_materials: bool = False
    use_vertex_colors: bool = False
    use_uv_seams: bool = False
    symmetry: Tuple[bool, bool, bool] = (False, False, False)
    smooth_iterations: int = 10
    smooth_strength: float = 0.5
    feature_snap_distance: float = 0.1
    num_threads: int = 0
    use_gpu: bool = False
    preset: str = 'CUSTOM'
    field_solver: str = 'KNOPPEL'
    param_method: str = 'MIQ'
    extraction_method: str = 'ISO'
    shade_smooth_output: bool = True
    keep_original: bool = True


def _build_qf_params(snap: _SettingsSnapshot,
                     overrides: dict) -> _LocalQFParams:
    """Build a real QFParams-shaped object from a snapshot + the
    optimizer/retry override dict. This is what gets passed to
    ``core_pipeline.run_pipeline`` — a dict would crash at the first
    attribute access."""
    p = _LocalQFParams(
        target_quad_count     = int(overrides.get("target_quad_count",
                                                  snap.target_quad_count)),
        curvature_adaptivity  = float(overrides.get("curvature_adaptivity",
                                                    snap.curvature_adaptivity)),
        exact_quad_count      = bool(snap.exact_quad_count),
        auto_detect_hard_edges= bool(snap.auto_detect_hard_edges),
        hard_edge_angle_deg   = float(snap.hard_edge_angle_deg),
        use_normals           = bool(snap.use_normals),
        use_materials         = bool(snap.use_materials),
        use_vertex_colors     = bool(snap.use_vertex_colors),
        use_uv_seams          = bool(snap.use_uv_seams),
        symmetry              = (bool(snap.symmetry_x),
                                 bool(snap.symmetry_y),
                                 bool(snap.symmetry_z)),
        smooth_iterations     = int(overrides.get("smooth_iterations",
                                                  snap.smooth_iterations)),
        smooth_strength       = float(overrides.get("smooth_strength",
                                                    snap.smooth_strength)),
        feature_snap_distance = float(overrides.get("feature_snap_distance",
                                                    snap.feature_snap_distance)),
        num_threads           = int(snap.num_threads),
        use_gpu               = bool(snap.use_gpu),
        preset                = str(snap.preset),
        field_solver          = str(snap.field_solver),
        param_method          = str(snap.param_method),
        extraction_method     = str(snap.extraction_method),
        shade_smooth_output   = bool(snap.shade_smooth_output),
        keep_original         = bool(snap.keep_original),
    )
    return p


class _InputMeshShim:
    """Minimal input mesh object understood by ``core_pipeline.run_pipeline``
    (it expects attributes ``vertices`` and ``faces``).
    """
    __slots__ = ("vertices", "faces")

    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        self.vertices = verts
        self.faces = faces


def _primary_runner(input_vertices: np.ndarray,
                    input_faces: np.ndarray,
                    params_dict: dict,
                    *,
                    settings,
                    obj=None,
                    progress_cb: Optional[ProgressCallback] = None,
                    ) -> Tuple[np.ndarray, List[List[int]]]:
    """Full-quality primary solver: delegate to the existing
    ``core_pipeline.run_pipeline``. Builds a real ``_LocalQFParams``
    object from the snapshot + overrides so that the core pipeline,
    which reads params via attribute access (``params.target_quad_count``
    etc.), doesn't crash on a plain dict."""
    snap = _SettingsSnapshot(settings)
    qf_params = _build_qf_params(snap, params_dict)
    shim = _InputMeshShim(input_vertices, input_faces)
    return core_pipeline.run_pipeline(
        shim, qf_params, obj=obj, settings=snap, progress_cb=progress_cb
    )


def _secondary_runner(input_vertices: np.ndarray,
                      input_faces: np.ndarray,
                      params_dict: dict,
                      *,
                      settings,
                      obj=None,
                      progress_cb: Optional[ProgressCallback] = None,
                      ) -> Tuple[np.ndarray, List[List[int]]]:
    """Robust simplified solver. Forces the safer algorithm choices:
        field_solver      = CURVATURE (curvature-only, no global solve)
        param_method      = POISSON   (no MIQ rounding)
        extraction_method = ISO       (simplest robust tracer)
        smoothing         = doubled

    Everything else stays the same. Still uses core_pipeline internally
    so we inherit every quality-gate-friendly behaviour already there.
    """
    snap = _SettingsSnapshot(settings)
    # Downgrade algorithms before building the QFParams. Values are
    # strings because ``bridge.QFParams`` and the EnumProperty fields in
    # ``properties.py`` both use the string form.
    snap.field_solver      = 'CURVATURE'
    snap.param_method      = 'POISSON'
    snap.extraction_method = 'ISO'
    snap.smooth_iterations = max(15, int(snap.smooth_iterations or 10) * 2)

    qf_params = _build_qf_params(snap, params_dict)
    shim = _InputMeshShim(input_vertices, input_faces)
    return core_pipeline.run_pipeline(
        shim, qf_params, obj=obj, settings=snap, progress_cb=progress_cb
    )


def _tertiary_runner(input_vertices: np.ndarray,
                     input_faces: np.ndarray,
                     params_dict: dict,
                     *,
                     settings=None,
                     obj=None,
                     progress_cb: Optional[ProgressCallback] = None,
                     ) -> Tuple[np.ndarray, List[List[int]]]:
    """Fast preview tier: voxel surface-nets quad extraction from the
    ``progressive_preview`` module. This is guaranteed to produce
    *something* even for severely broken inputs because it doesn't
    depend on the primary solver stack at all.
    """
    from .progressive_preview import build_tier2_preview

    target_qc = int(params_dict.get("target_quad_count", 2000))
    tier = build_tier2_preview(
        input_vertices,
        input_faces,
        target_quad_count=target_qc,
    )
    if tier.vertices is None or not tier.faces:
        # Absolute last resort: voxel rescue + trivial quadrangulation
        v, f = voxel_rescue_remesh(input_vertices, input_faces)
        # Wrap the tris as single-face lists to satisfy the quad-dominant
        # output contract — not ideal but non-empty.
        return v, [list(tri) for tri in f]
    return tier.vertices, tier.faces


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dominance_pipeline(input_mesh,
                           params,
                           *,
                           obj=None,
                           settings=None,
                           progress_cb: Optional[ProgressCallback] = None,
                           max_retries: int = 2,
                           ) -> DominanceResult:
    """Run the full Dominance Layer pipeline.

    This is a drop-in replacement for ``core_pipeline.run_pipeline`` that
    additionally performs: mesh repair, classification, adaptive
    dispatch with fallback, quality gate + retry, auto quality
    optimization, and edge-flow refinement.

    Returns a ``DominanceResult`` with the final mesh and every report
    produced along the way.
    """
    t_start = time.perf_counter()
    cb = progress_cb or _noop_progress

    # Extract base params + flags
    want_repair = bool(_get(settings, "enable_mesh_repair", True))
    want_artist = bool(_get(settings, "enable_artist_intelligence", True))
    want_optimizer = bool(_get(settings, "enable_auto_optimizer", False))
    want_edge_flow = bool(_get(settings, "enable_edge_flow_refine", True))
    reproducible = bool(_get(settings, "reproducible_mode", False))
    seed = int(_get(settings, "reproducible_seed", 0xC0FFEE))
    time_budget = float(_get(settings, "time_budget_seconds", 60.0))
    confidence_target = float(_get(settings, "confidence_target", 75.0))

    # Extract raw input
    in_verts = np.asarray(input_mesh.vertices, dtype=np.float64)
    in_faces_raw = np.asarray(input_mesh.faces)

    # Enter deterministic context if requested
    if reproducible:
        ctx = DeterministicContext(seed=seed)
    else:
        # Null context manager
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        # -----------------------------------------------------------
        # STAGE 0 — Mesh repair
        # -----------------------------------------------------------
        cb("Repair", 0.02)
        if want_repair:
            rep_verts, rep_faces, repair_report = repair_mesh(
                in_verts, in_faces_raw, aggressive=True
            )
        else:
            rep_verts = in_verts
            # Coerce to (F,3) if already triangular
            if (isinstance(in_faces_raw, np.ndarray)
                    and in_faces_raw.ndim == 2
                    and in_faces_raw.shape[1] == 3):
                rep_faces = in_faces_raw.astype(np.int64)
            else:
                from .mesh_repair import _triangulate_faces
                rep_faces = _triangulate_faces(in_faces_raw)
            repair_report = RepairReport(
                input_vertices=len(in_verts),
                input_faces=len(rep_faces),
                output_vertices=len(in_verts),
                output_faces=len(rep_faces),
                confidence=1.0,
            )

        # Canonicalize for determinism if requested
        if reproducible and len(rep_verts) > 0 and len(rep_faces) > 0:
            rep_verts, rep_faces, _ = canonicalize_mesh(rep_verts, rep_faces)

        if len(rep_verts) == 0 or len(rep_faces) == 0:
            # Nothing to remesh — return empty result
            result = DominanceResult(
                vertices=rep_verts,
                faces=[],
                repair_report=repair_report,
                elapsed_seconds=time.perf_counter() - t_start,
            )
            return result

        # -----------------------------------------------------------
        # STAGE 1 — Classify + artist intelligence
        # -----------------------------------------------------------
        cb("Classify", 0.08)
        mesh_class: Optional[MeshClass] = None
        semantic_features: Optional[SemanticFeatures] = None
        if want_artist:
            try:
                mesh_class, semantic_features, _template = run_artist_intelligence(
                    rep_verts, rep_faces
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[QuadForge/dominance] artist intelligence failed: {exc}")
                mesh_class = MeshClass(category="unknown", confidence=0.0)
        else:
            mesh_class = MeshClass(category="unknown", confidence=0.0)

        # -----------------------------------------------------------
        # STAGE 2 — Dispatch decision
        # -----------------------------------------------------------
        cb("Dispatch", 0.12)
        decision = decide_tier(
            mesh_class,
            num_vertices=len(rep_verts),
            num_faces=len(rep_faces),
            repair_confidence=repair_report.confidence,
            time_budget_s=time_budget,
            user_request_fast_preview=False,
        )
        print(f"[QuadForge/dominance] {decision.describe()}")

        # -----------------------------------------------------------
        # STAGE 3 — Solve with gate+retry loop
        # -----------------------------------------------------------
        base_params = _extract_params_dict(settings)
        gates = QualityGates()
        # Loosen if the user asked for a lower confidence target
        if confidence_target < 75.0:
            gates = gates.relax(1.0 + (75.0 - confidence_target) / 100.0)

        retries_used = 0
        tier_attempts: List[TierRunResult] = []
        working_params = dict(base_params)
        final_verts: Optional[np.ndarray] = None
        final_faces: Optional[List[List[int]]] = None
        final_quality: Optional[QualityBreakdown] = None
        final_confidence: Optional[ConfidenceReport] = None

        for retry in range(max_retries + 1):
            cb(f"Solve (attempt {retry + 1})", 0.15 + 0.5 * (retry / (max_retries + 1)))

            def _primary(v, f, p, _settings=settings, _obj=obj):
                return _primary_runner(v, f, p, settings=_settings, obj=_obj)

            def _secondary(v, f, p, _settings=settings, _obj=obj):
                return _secondary_runner(v, f, p, settings=_settings, obj=_obj)

            def _tertiary(v, f, p, _settings=settings, _obj=obj):
                return _tertiary_runner(v, f, p, settings=_settings, obj=_obj)

            tier_result, attempts = dispatch_with_fallback(
                decision, rep_verts, rep_faces, working_params,
                primary_fn=_primary,
                secondary_fn=_secondary,
                tertiary_fn=_tertiary,
            )
            tier_attempts.extend(attempts)

            if not tier_result.success:
                # All tiers failed — give up and return the repair-only mesh
                print("[QuadForge/dominance] all solver tiers failed")
                break

            # Score and gate
            final_verts = tier_result.vertices
            final_faces = tier_result.faces
            final_quality = compute_quality_score(
                final_verts, final_faces,
                feature_edges=None,
                symmetry_axes=(
                    bool(_get(settings, "symmetry_x", False)),
                    bool(_get(settings, "symmetry_y", False)),
                    bool(_get(settings, "symmetry_z", False)),
                ),
            )
            final_confidence = compute_confidence(
                final_quality, gates,
                repair_confidence=repair_report.confidence,
                is_manifold=True,
                vertices=final_verts,
                faces=final_faces,
            )

            if final_confidence.passed or retry == max_retries:
                break

            # Plan a retry
            retry_plan = plan_retry(final_confidence.gate_violations, gates)
            print(f"[QuadForge/dominance] retry {retry + 1}: {retry_plan.explanation}")

            # Apply overrides by bouncing through the optimizer helper
            from .quality_optimizer import _apply_overrides
            working_params = _apply_overrides(working_params, retry_plan.param_overrides)
            gates = retry_plan.relaxed_gates
            retries_used += 1

        # -----------------------------------------------------------
        # STAGE 4 — Auto quality optimizer (optional)
        # -----------------------------------------------------------
        optimizer_result: Optional[OptimizerResult] = None
        if (want_optimizer and final_verts is not None
                and final_faces is not None):
            cb("Optimize", 0.7)

            def _run_candidate(p):
                r = _primary_runner(
                    rep_verts, rep_faces, p, settings=settings, obj=obj
                )
                return r

            try:
                optimizer_result = run_auto_optimizer(
                    _run_candidate,
                    working_params,
                    symmetry_axes=(
                        bool(_get(settings, "symmetry_x", False)),
                        bool(_get(settings, "symmetry_y", False)),
                        bool(_get(settings, "symmetry_z", False)),
                    ),
                    max_candidates=5,
                    time_budget_s=max(5.0, time_budget * 0.4),
                )
                if (optimizer_result.best_candidate.vertices is not None
                        and optimizer_result.best_candidate.score
                        > final_quality.composite):
                    final_verts = optimizer_result.best_candidate.vertices
                    final_faces = optimizer_result.best_candidate.faces
                    final_quality = optimizer_result.best_candidate.breakdown
            except Exception as exc:  # noqa: BLE001
                print(f"[QuadForge/dominance] optimizer failed: {exc}")

        # -----------------------------------------------------------
        # STAGE 5 — Edge flow refinement
        # -----------------------------------------------------------
        edge_flow_report: Optional[EdgeFlowReport] = None
        if (want_edge_flow and final_verts is not None
                and final_faces is not None):
            cb("Refine Edge Flow", 0.88)
            try:
                ef_verts, ef_faces, edge_flow_report = refine_edge_flow(
                    final_verts, final_faces,
                    feature_vertices=None,
                    enable_doublet_removal=True,
                    enable_collapse=True,
                    enable_relaxation=True,
                    relaxation_iters=2,
                )
                final_verts = ef_verts
                final_faces = ef_faces
            except Exception as exc:  # noqa: BLE001
                print(f"[QuadForge/dominance] edge flow refine failed: {exc}")

        # -----------------------------------------------------------
        # STAGE 6 — Final confidence recompute (after optimizer + refine)
        # -----------------------------------------------------------
        if final_verts is not None and final_faces is not None:
            final_quality = compute_quality_score(
                final_verts, final_faces,
                symmetry_axes=(
                    bool(_get(settings, "symmetry_x", False)),
                    bool(_get(settings, "symmetry_y", False)),
                    bool(_get(settings, "symmetry_z", False)),
                ),
            )
            final_confidence = compute_confidence(
                final_quality, gates,
                repair_confidence=repair_report.confidence,
                vertices=final_verts,
                faces=final_faces,
            )

        cb("Done", 1.0)

        # -----------------------------------------------------------
        # Assemble result
        # -----------------------------------------------------------
        if final_verts is None or final_faces is None:
            final_verts = rep_verts
            final_faces = [list(tri) for tri in rep_faces]

        result = DominanceResult(
            vertices=final_verts,
            faces=final_faces,
            repair_report=repair_report,
            mesh_class=mesh_class,
            semantic_features=semantic_features,
            dispatch_decision=decision,
            tier_attempts=tier_attempts,
            quality_breakdown=final_quality,
            confidence_report=final_confidence,
            optimizer_result=optimizer_result,
            edge_flow_report=edge_flow_report,
            elapsed_seconds=time.perf_counter() - t_start,
            retries_used=retries_used,
        )

        print(result.summary())
        return result


def _write_result_to_settings(settings, result: "DominanceResult") -> None:
    """Copy selected fields from a DominanceResult onto the live Blender
    settings PropertyGroup so the 'Last Run' block in the Dominance panel
    can display them. Fails silently if ``settings`` is a plain dict or
    lacks the attribute (which is the case in unit tests)."""
    if settings is None:
        return

    def _set(name: str, value) -> None:
        if hasattr(settings, name):
            try:
                setattr(settings, name, value)
            except (TypeError, ValueError):
                pass  # type mismatch — ignore

    if result.confidence_report is not None:
        _set("last_confidence_score", float(result.confidence_report.score))
        gates = result.confidence_report.gate_violations
        if gates:
            _set("last_gate_status", f"{len(gates)} violation(s)")
        else:
            _set("last_gate_status", "PASS")

    if result.mesh_class is not None:
        cat = result.mesh_class.category
        if result.mesh_class.sub_category:
            cat += f"/{result.mesh_class.sub_category}"
        _set("last_mesh_category", cat[:63])  # Blender string prop length safety

    if result.dispatch_decision is not None:
        _set("last_solver_tier", result.dispatch_decision.tier.name)

    _set("last_retries_used", int(result.retries_used))

    if result.repair_report is not None:
        _set("last_repair_welded", int(result.repair_report.welded_vertices))
        _set("last_repair_fixed", int(
            result.repair_report.degenerate_faces_removed +
            result.repair_report.duplicate_faces_removed +
            result.repair_report.flipped_faces +
            result.repair_report.non_manifold_vertices_split
        ))


def run_dominance_pipeline_compat(input_mesh, params, obj=None, settings=None,
                                  progress_cb=None):
    """Compatibility wrapper returning ``(vertices, faces)`` tuple so
    existing callers can switch to the dominance pipeline without
    touching their unpacking code. Also writes the DominanceResult
    back into the settings PropertyGroup so the UI can display it.
    """
    res = run_dominance_pipeline(
        input_mesh, params, obj=obj, settings=settings, progress_cb=progress_cb
    )
    _write_result_to_settings(settings, res)
    return res.vertices, res.faces
