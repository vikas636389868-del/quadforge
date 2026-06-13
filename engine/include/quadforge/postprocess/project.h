#pragma once
/* quadforge/postprocess/project.h — BVH-accelerated surface projection.
 *
 * After smoothing moves vertices off the original surface, this projects
 * each vertex back to the nearest point on the input mesh.
 */

#ifndef QUADFORGE_POSTPROCESS_PROJECT_H
#define QUADFORGE_POSTPROCESS_PROJECT_H

#include "../types.h"
#include "../mesh/spatial.h"

namespace qf {

/**
 * Project all vertices of quad_mesh onto the reference surface via BVH.
 *
 * @param quad_mesh  Vertices modified in place.
 * @param bvh        BVH built from the input triangle mesh.
 * @param num_threads 0 = OpenMP auto.
 */
void project_to_surface(QuadMesh&         quad_mesh,
                        const TriangleBVH& bvh,
                        int               num_threads = 0);

/**
 * Alternate between smoothing and projection for a given number of
 * rounds to reduce surface drift while maintaining regularity.
 *
 * @param rounds                Number of smooth+project cycles.
 * @param lambda                Taubin positive step (shrink).
 * @param mu                    Taubin negative step (counter-shrink).
 * @param num_threads           0 = OpenMP auto-detect.
 * @param smooth_iters_per_round
 *   BUG FIX (v83 / Bug 4): Number of Taubin iterations inside each round.
 *   When 0 (backward-compatible default), falls back to the original
 *   hardcoded formula: lround(10.0 / rounds).  When > 0, the caller's
 *   explicit value is used, which is how QFParams::smooth_iterations is
 *   finally forwarded all the way to the actual smoothing passes.
 *   Previously this parameter did not exist: engine.cpp computed
 *   sp.iterations = max(1, smooth_iterations / 3) but had no way to pass
 *   it here, so the user's "Smooth Iterations" slider had zero effect.
 */
void iterative_smooth_and_project(
    QuadMesh&          quad_mesh,
    const TriangleBVH& bvh,
    int                rounds,
    double             lambda,
    double             mu,
    int                num_threads          = 0,
    int                smooth_iters_per_round = 0
);

} // namespace qf

#endif // QUADFORGE_POSTPROCESS_PROJECT_H
