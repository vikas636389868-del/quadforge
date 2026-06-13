"""QuadForge — Adaptive Pipeline Dispatcher (Roadmap §11.3 + §13.5).

Implements the "Hybrid Solver Stack" and "Adaptive Pipeline Switching".

The dispatcher picks one of three solver paths based on a combination of
the mesh classification (from ``artist_intelligence.classify_mesh``),
the input mesh size, the user's time budget, and the repair confidence.

Three tiers:
    PRIMARY   — globally-optimal cross-field + MIQ + motorcycle extraction
                (highest quality, highest cost)
    SECONDARY — robust simplified solver (curvature-only field + Poisson
                parametrization + iso-line extraction), chosen for noisy
                or difficult meshes where the primary solver would
                numerically degrade.
    TERTIARY  — fast preview path (voxel surface-nets from
                ``progressive_preview.build_tier2_preview``) used only
                when the user requires instant feedback or when both
                higher tiers failed.

Fallback chain: on exception or gate failure in the primary solver, the
dispatcher automatically downgrades one tier and re-runs. The user is
never left without *some* output (unless the repair stage itself
produced an empty mesh).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional, Tuple

import numpy as np

from .artist_intelligence import MeshClass


class SolverTier(IntEnum):
    PRIMARY = 1
    SECONDARY = 2
    TERTIARY = 3


# ---------------------------------------------------------------------------
# Dispatch decision
# ---------------------------------------------------------------------------

@dataclass
class DispatchDecision:
    tier: SolverTier
    reason: str
    estimated_seconds: float = 0.0
    fallback_chain: List[SolverTier] = field(default_factory=list)

    def describe(self) -> str:
        chain = " -> ".join(t.name for t in self.fallback_chain)
        return f"{self.tier.name} ({self.reason}) [chain: {chain}]"


def decide_tier(mesh_class: MeshClass,
                *,
                num_vertices: int,
                num_faces: int,
                repair_confidence: float = 1.0,
                time_budget_s: float = 60.0,
                user_request_fast_preview: bool = False,
                ) -> DispatchDecision:
    """Pick the initial solver tier and build the fallback chain.

    Heuristics (in priority order):
        1. If the user explicitly asked for a fast preview -> TERTIARY.
        2. If repair confidence is below 0.55 -> SECONDARY (robust path).
        3. If mesh is "scan" or "damaged" -> SECONDARY.
        4. If mesh has > 2M faces AND time budget < 30 s -> SECONDARY.
        5. If mesh has > 5M faces -> SECONDARY (primary won't finish).
        6. Otherwise -> PRIMARY.
    """
    # Estimate rough primary-solver cost: O(F log F) with a ~1.8us/face
    # constant for the MIQ + motorcycle path on a modern desktop.
    est_primary = max(0.5, num_faces * 1.8e-6 + num_vertices * 0.5e-6)
    est_secondary = est_primary * 0.35
    est_tertiary = est_primary * 0.05

    if user_request_fast_preview:
        return DispatchDecision(
            tier=SolverTier.TERTIARY,
            reason="user requested fast preview",
            estimated_seconds=est_tertiary,
            fallback_chain=[SolverTier.TERTIARY],
        )

    if repair_confidence < 0.55:
        return DispatchDecision(
            tier=SolverTier.SECONDARY,
            reason=f"low repair confidence ({repair_confidence:.2f})",
            estimated_seconds=est_secondary,
            fallback_chain=[SolverTier.SECONDARY, SolverTier.TERTIARY],
        )

    if mesh_class.category in ("scan", "damaged"):
        return DispatchDecision(
            tier=SolverTier.SECONDARY,
            reason=f"{mesh_class.category} category",
            estimated_seconds=est_secondary,
            fallback_chain=[SolverTier.SECONDARY, SolverTier.TERTIARY],
        )

    if num_faces > 5_000_000:
        return DispatchDecision(
            tier=SolverTier.SECONDARY,
            reason=f"very large mesh ({num_faces:,} faces)",
            estimated_seconds=est_secondary,
            fallback_chain=[SolverTier.SECONDARY, SolverTier.TERTIARY],
        )

    if num_faces > 2_000_000 and est_primary > time_budget_s:
        return DispatchDecision(
            tier=SolverTier.SECONDARY,
            reason=f"primary solver exceeds time budget "
                   f"({est_primary:.1f}s > {time_budget_s:.1f}s)",
            estimated_seconds=est_secondary,
            fallback_chain=[SolverTier.SECONDARY, SolverTier.TERTIARY],
        )

    return DispatchDecision(
        tier=SolverTier.PRIMARY,
        reason=f"clean {mesh_class.category} mesh, primary solver OK",
        estimated_seconds=est_primary,
        fallback_chain=[SolverTier.PRIMARY,
                        SolverTier.SECONDARY,
                        SolverTier.TERTIARY],
    )


# ---------------------------------------------------------------------------
# Solver tier runners
# ---------------------------------------------------------------------------

@dataclass
class TierRunResult:
    tier: SolverTier
    success: bool
    vertices: Optional[np.ndarray] = None
    faces: Optional[List[List[int]]] = None
    elapsed_seconds: float = 0.0
    error_message: str = ""


def run_tier(tier: SolverTier,
             vertices: np.ndarray,
             faces: np.ndarray,
             params: dict,
             *,
             primary_fn: Optional[Callable] = None,
             secondary_fn: Optional[Callable] = None,
             tertiary_fn: Optional[Callable] = None,
             ) -> TierRunResult:
    """Execute a single solver tier. Each ``*_fn`` is a
    ``(vertices, faces, params) -> (new_verts, new_faces)`` callable.

    The dispatcher knows nothing about the actual solver internals —
    the three callables are injected by ``dominance_pipeline.py`` so
    this module stays dependency-free.
    """
    t0 = time.perf_counter()
    result = TierRunResult(tier=tier, success=False)

    fn = {
        SolverTier.PRIMARY: primary_fn,
        SolverTier.SECONDARY: secondary_fn,
        SolverTier.TERTIARY: tertiary_fn,
    }.get(tier)

    if fn is None:
        result.error_message = f"no callable registered for {tier.name}"
        result.elapsed_seconds = time.perf_counter() - t0
        return result

    try:
        v, f = fn(vertices, faces, params)
        result.vertices = v
        result.faces = f
        result.success = (v is not None and f is not None and len(f) > 0)
        if not result.success:
            result.error_message = f"{tier.name} produced empty output"
    except Exception as exc:  # noqa: BLE001 — fallback on any failure
        result.error_message = f"{tier.name} raised: {type(exc).__name__}: {exc}"
    result.elapsed_seconds = time.perf_counter() - t0
    return result


def dispatch_with_fallback(decision: DispatchDecision,
                           vertices: np.ndarray,
                           faces: np.ndarray,
                           params: dict,
                           *,
                           primary_fn: Optional[Callable] = None,
                           secondary_fn: Optional[Callable] = None,
                           tertiary_fn: Optional[Callable] = None,
                           ) -> Tuple[TierRunResult, List[TierRunResult]]:
    """Walk the fallback chain from the decision. Returns the first
    successful result and the list of every attempt for logging.
    """
    attempts: List[TierRunResult] = []
    for tier in decision.fallback_chain:
        res = run_tier(tier, vertices, faces, params,
                       primary_fn=primary_fn,
                       secondary_fn=secondary_fn,
                       tertiary_fn=tertiary_fn)
        attempts.append(res)
        if res.success:
            return res, attempts
    # All failed — return the last attempt so caller can see the error
    return attempts[-1], attempts
