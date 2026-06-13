#pragma once
/* quadforge/param/igm.h — Integer Grid Maps (IGM) parametrization.
 *
 * Implements Bommes et al. (2013) "Integer-Grid Maps for Reliable Quad
 * Meshing" — an improvement over MIQ that enforces globally consistent
 * period jumps across seam cuts via constraint propagation and a
 * spanning-tree-based rounding strategy.
 *
 * Key differences from MIQ:
 *   1. Period jumps at seam edges are rounded GLOBALLY (not greedily)
 *      using Union-Find + offset propagation for consistency.
 *   2. Topological consistency: sum of integer jumps around each
 *      singularity is verified to match the singularity index (±1).
 *   3. Strict global snapping: ALL vertices (not just seam vertices)
 *      are snapped to the nearest integer grid after rounding.
 *   4. A constrained re-solve forces the parametrization to honour
 *      the chosen integer transitions exactly.
 *
 * The result is a more globally consistent UV grid that is especially
 * beneficial for CAD / architectural meshes (Architecture preset).
 *
 * References:
 *   Bommes et al. (2013), "Integer-Grid Maps for Reliable Quad Meshing,"
 *   ACM Trans. Graph. 32(4) (SIGGRAPH 2013).
 */

#ifndef QUADFORGE_PARAM_IGM_H
#define QUADFORGE_PARAM_IGM_H

#include <vector>
#include "../types.h"
#include "../mesh/halfedge.h"
#include "combing.h"    // CombingResult, EdgeSet
#include "miq.h"        // build_poisson_system (shared helper)

namespace qf {

/** Configuration for the IGM parametrization solve. */
struct IGMParams {
    int    max_cg_iterations  = 1000;  ///< Max CG iterations for the Poisson solve
    double cg_tolerance       = 1e-6;  ///< CG convergence tolerance
    double jump_snap_tol      = 0.20;  ///< Fraction threshold to snap jump to nearest int
    bool   strict_global_snap = true;  ///< After solve, snap ALL verts to nearest int grid
    double global_snap_tol    = 0.15;  ///< Max fractional deviation to apply global snapping
    int    num_threads        = 0;     ///< 0 = OpenMP auto-detect
};

/**
 * Compute UV parametrization using Integer Grid Maps (IGM).
 *
 * The key improvement over MIQ is that period jumps at seam cuts are
 * rounded in a globally consistent way using a Union-Find spanning-tree
 * approach, ensuring that the UV transitions are exactly integer across
 * every seam edge.  A constrained re-solve then produces a parametrization
 * that honours these integer transitions exactly, resulting in a cleaner,
 * more regular quad layout — especially for hard-surface and CAD meshes.
 *
 * @param mesh     Triangle mesh (half-edge structure)
 * @param combing  Combed cross-field result (angles + seam cuts)
 * @param sizing   Per-vertex target edge length (from sizing field)
 * @param params   IGM solver configuration
 */
UVParam compute_igm_parametrization(
    const HalfEdgeMesh&    mesh,
    const CombingResult&   combing,
    const Eigen::VectorXd& sizing,
    const IGMParams&       params = IGMParams{}
);

/**
 * Verify global period-jump consistency.
 *
 * For each interior vertex, checks that the sum of integer period jumps
 * on its surrounding seam edges matches the expected value from the
 * singularity structure.
 *
 * @param mesh         Triangle mesh
 * @param seam_edges   Seam edge pairs (v_lo, v_hi)
 * @param jump_u       Integer U-jump per seam edge
 * @param jump_v       Integer V-jump per seam edge
 * @param tol          Tolerance for "close to integer"
 * @return             Number of inconsistency violations found
 */
int verify_jump_consistency(
    const HalfEdgeMesh&              mesh,
    const EdgeSet&                   seam_edges,
    const std::vector<int>&          jump_u,
    const std::vector<int>&          jump_v,
    double                           tol = 0.1
);

} // namespace qf

#endif // QUADFORGE_PARAM_IGM_H
