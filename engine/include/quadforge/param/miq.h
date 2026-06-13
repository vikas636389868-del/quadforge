#pragma once
/* quadforge/param/miq.h — Mixed-Integer Quadrangulation.
 *
 * Implements:
 *   1. Poisson parametrization: minimise |∇U - X|² + |∇V - Y|²
 *      using CHOLMOD sparse Cholesky.
 *   2. MIQ rounding: enforce integer period jumps at seam edges
 *      via greedy rounding with Cholesky rank-1 updates.
 *   3. Adaptive metric tensor: sizing field modulates the Poisson system.
 *
 * References:
 *   Bommes et al. (2009), "Mixed-Integer Quadrangulation," SIGGRAPH.
 *   Bommes et al. (2013), "Integer-Grid Maps for Reliable Quad Meshing."
 */

#ifndef QUADFORGE_PARAM_MIQ_H
#define QUADFORGE_PARAM_MIQ_H

#include <vector>
#include "../types.h"
#include "../mesh/halfedge.h"
#include "combing.h"

namespace qf {

/** How to solve the parametrization. */
enum class ParamMethod {
    MIQ,            ///< Full MIQ with integer rounding (best quality)
    IGM,            ///< Integer Grid Maps — globally consistent rounding (best for CAD/architecture)
    POISSON_SIMPLE  ///< Poisson only, no rounding (fastest)
};

struct ParamParams {
    ParamMethod method         = ParamMethod::MIQ;
    int         max_miq_rounds = 30;          ///< Max integer rounding iterations
    double      miq_tolerance  = 0.05;        ///< Rounding confidence threshold
    double      sizing_weight  = 1.0;         ///< Weight of the adaptive metric
    int         num_threads    = 0;
};

/**
 * Compute a (U,V) parametrization from the combed cross-field.
 *
 * The parametrization is globally smooth and has integer period jumps
 * across seam edges (after MIQ rounding).
 *
 * @param mesh      Triangle mesh
 * @param combing   Combed cross-field result (angles + seam cuts)
 * @param sizing    Per-vertex target edge length (from sizing field)
 * @param params    Parametrization configuration
 */
UVParam compute_parametrization(
    const HalfEdgeMesh&   mesh,
    const CombingResult&  combing,
    const Eigen::VectorXd& sizing,
    const ParamParams&    params = ParamParams{}
);

/**
 * Assemble the Poisson system for U (or V) parametrization.
 * Returns the SPD matrix A and right-hand side b.
 *
 * Public for unit testing.
 */
void build_poisson_system(
    const HalfEdgeMesh&    mesh,
    const std::vector<double>& field_angles,  ///< Combed angles (U or V direction)
    const Eigen::VectorXd& sizing,
    SparseMat&             A_out,
    VecX&                  b_out
);

} // namespace qf

#endif // QUADFORGE_PARAM_MIQ_H
