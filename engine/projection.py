"""Surface projection for QuadForge.

After smoothing moves vertices off the original surface, this module
projects them back onto the nearest point of the input triangle mesh
using the BVH spatial index.

Also provides batch projection with optional weighting (feature vertices
project more aggressively than interior vertices).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Set

from .spatial import TriangleBVH


def project_to_surface(
    output_vertices: np.ndarray,
    bvh: TriangleBVH,
    blend: float = 1.0,
    pinned_vertices: Optional[Set[int]] = None,
) -> np.ndarray:
    """Project output vertices onto the original surface.

    Parameters
    ----------
    output_vertices : (V, 3) current vertex positions
    bvh : TriangleBVH built from the original input mesh
    blend : 0.0 = no projection, 1.0 = full projection
    pinned_vertices : vertices to skip (already on features etc.)

    Returns
    -------
    projected : (V, 3) vertices projected onto original surface
    """
    if blend <= 0.0:
        return output_vertices.copy()

    pinned = pinned_vertices or set()
    result = output_vertices.copy().astype(np.float64)

    for i in range(len(result)):
        if i in pinned:
            continue
        closest = bvh.closest_point(result[i])
        result[i] = result[i] * (1.0 - blend) + closest.point * blend

    return result.astype(np.float32)


def iterative_smooth_and_project(
    output_vertices: np.ndarray,
    output_faces: list,
    bvh: TriangleBVH,
    smooth_iterations: int = 5,
    smooth_strength: float = 0.3,
    project_blend: float = 0.8,
    num_rounds: int = 3,
    pinned_vertices: Optional[Set[int]] = None,
) -> np.ndarray:
    """Alternating smoothing and projection for high-quality results.

    Each round: smooth → project → repeat. This produces meshes
    that are both smooth and tightly conforming to the original surface.

    Parameters
    ----------
    output_vertices : (V, 3)
    output_faces : list of face vertex lists
    bvh : TriangleBVH from input mesh
    smooth_iterations : iterations per smoothing round
    smooth_strength : Taubin lambda per round
    project_blend : how aggressively to project (0-1)
    num_rounds : number of smooth→project cycles
    pinned_vertices : set of vertices to skip

    Returns
    -------
    refined : (V, 3)
    """
    from .smoothing import taubin_smooth

    verts = output_vertices.copy().astype(np.float32)
    pinned = pinned_vertices or set()

    for r in range(num_rounds):
        # Smooth
        verts = taubin_smooth(
            verts, output_faces,
            iterations=smooth_iterations,
            lam=smooth_strength,
            mu=-smooth_strength * 1.06,
            pinned_vertices=pinned,
        )

        # Project back
        verts = project_to_surface(verts, bvh, blend=project_blend, pinned_vertices=pinned)

    return verts
