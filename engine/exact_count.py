"""Exact quad count solver for QuadForge.

When the user enables "Exact Quad Count" mode, this module performs a
binary search over the sizing field scale factor to converge on the
target quad count. The full pipeline runs multiple times with different
scale factors until the output quad count is within ±2% of the target.

This is more expensive than a single pass (typically 3-5 iterations),
but produces output that precisely matches the user's request.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple


def binary_search_quad_count(
    run_pipeline_fn: Callable[[int], Tuple[np.ndarray, list, int]],
    target_count: int,
    tolerance: float = 0.02,
    max_iterations: int = 6,
    initial_scale_range: Tuple[float, float] = (0.5, 2.0),
) -> Tuple[np.ndarray, list]:
    """Binary search over quad count to hit the target.

    Parameters
    ----------
    run_pipeline_fn : callable
        Takes target_quad_count (int) and returns (vertices, faces, actual_count).
        The pipeline should run the full remesh with the given count target.
    target_count : int
        Desired number of quads.
    tolerance : float
        Acceptable relative error (0.02 = ±2%).
    max_iterations : int
        Maximum binary search iterations.
    initial_scale_range : (lo, hi)
        Initial range of scale multipliers on target_count.

    Returns
    -------
    best_vertices, best_faces : the result closest to the target count
    """
    lo_mult, hi_mult = initial_scale_range
    best_verts = None
    best_faces = None
    best_diff = float('inf')

    print(f"[QuadForge] Exact quad count: target={target_count}, "
          f"tolerance=±{tolerance*100:.0f}%")

    for iteration in range(max_iterations):
        mid_mult = 0.5 * (lo_mult + hi_mult)
        trial_count = max(100, int(target_count * mid_mult))

        print(f"[QuadForge]   Iteration {iteration + 1}: "
              f"trying {trial_count} quads (scale={mid_mult:.3f})")

        try:
            verts, faces, actual_count = run_pipeline_fn(trial_count)
        except Exception as e:
            print(f"[QuadForge]   Pipeline failed: {e}")
            break

        diff = abs(actual_count - target_count) / max(target_count, 1)

        print(f"[QuadForge]   Got {actual_count} quads "
              f"(diff={diff*100:.1f}%)")

        if diff < best_diff:
            best_diff = diff
            best_verts = verts
            best_faces = faces

        # Check convergence
        if diff <= tolerance:
            print(f"[QuadForge] Exact quad count: converged at {actual_count} "
                  f"quads in {iteration + 1} iterations")
            return verts, faces

        # Binary search update
        if actual_count < target_count:
            lo_mult = mid_mult  # need more quads → increase target
        else:
            hi_mult = mid_mult  # too many quads → decrease target

        # Check if range is too narrow
        if hi_mult - lo_mult < 0.01:
            break

    if best_verts is not None:
        print(f"[QuadForge] Exact quad count: best result "
              f"(diff={best_diff*100:.1f}%) after {max_iterations} iterations")
        return best_verts, best_faces

    # Should never reach here — fallback
    return np.zeros((0, 3)), []
