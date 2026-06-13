#pragma once
/* quadforge/field/smoothing.h — Post-solve 4-RoSy field smoothing.
 *
 * After the globally optimal cross-field solve (Knöppel 2013), the raw
 * eigenvector output can have residual noise caused by:
 *   – degenerate or near-degenerate triangles perturbing the Laplacian
 *   – hard constraint penalty terms introducing local discontinuities
 *   – partial convergence of the power iteration
 *
 * This module provides a fast, feature-preserving smoothing pass that
 * operates directly on the complex per-face field representation.
 * It does NOT alter constrained faces beyond a user-specified tolerance.
 *
 * Algorithm:
 *   For each face f and each Gauss-Seidel iteration:
 *     1. Accumulate the weighted average of neighbouring fields:
 *          sum_j  w_ij * conj(r_ij) * u_j
 *        where r_ij = exp(4i * phi_ij) is the parallel transport between
 *        faces i and j, and w_ij is the cotangent-area weight.
 *     2. If |accumulated| > eps, set u_f = accumulated / |accumulated|
 *        (project onto unit circle in the 4-RoSy representation).
 *     3. If face f is feature-constrained, blend the result with the
 *        constraint target by (1 - alpha) * smoothed + alpha * target.
 *
 * Complexity: O(iter * F) where F is the face count.  Embarrassingly
 * parallel within each iteration (read-only previous, write-only current).
 *
 * References:
 *   Knöppel et al. (2013), §4 — smoothness energy minimisation
 *   Ray et al. (2008), §5    — iterative field smoothing
 */

#ifndef QUADFORGE_FIELD_SMOOTHING_H
#define QUADFORGE_FIELD_SMOOTHING_H

#include <complex>
#include <utility>          // std::pair — required for compute_cotangent_weights() return type
#include <vector>
#include "../types.h"
#include "../mesh/halfedge.h"
#include "cross_field.h"     // CrossField, CrossFieldParams
#include "constraints.h"     // FieldConstraint

namespace qf {

// -----------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------

struct FieldSmoothingParams {
    int    iterations       = 5;      ///< Gauss-Seidel passes (5 is usually sufficient)
    double constraint_alpha = 0.95;   ///< How strongly to hold feature constraints [0,1]
    double convergence_tol  = 1e-6;   ///< Stop early if max field change < this
    int    num_threads      = 0;      ///< 0 = OpenMP auto-detect
    bool   use_cotangent_weights = true; ///< Cotangent vs. uniform (area) weights
};

// -----------------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------------

/**
 * Smooth a 4-RoSy cross-field using iterated Gauss-Seidel on the
 * connection Laplacian energy, preserving feature-edge constraints.
 *
 * The input field is modified in place.  The function returns the
 * maximum field change in the last iteration (useful for convergence
 * diagnostics).
 *
 * @param mesh        The triangle mesh (read-only)
 * @param field       The cross-field to smooth (modified in place)
 * @param constraints Hard constraints from feature/boundary/symmetry edges
 *                    (their face_idx entries are protected by constraint_alpha)
 * @param params      Smoothing configuration
 * @return            Max |Δu| in the final iteration (0 = fully converged)
 */
double smooth_cross_field(
    const HalfEdgeMesh&              mesh,
    CrossField&                      field,
    const std::vector<FieldConstraint>& constraints,
    const FieldSmoothingParams&      params = FieldSmoothingParams{}
);

// -----------------------------------------------------------------------
// Utilities (exposed for unit testing)
// -----------------------------------------------------------------------

/**
 * Compute per-edge cotangent weights for the cross-field smoothing.
 *
 * Returns a vector of (half_edge_index, weight) pairs for every interior
 * half-edge.  The weight w_ij = (cot α + cot β) / 2 where α and β are
 * the two angles opposite the shared edge.
 *
 * These are the same cotangent weights used in the Poisson parametrization
 * (connection Laplacian in Knöppel 2013, equation 3).
 */
std::vector<std::pair<int,double>> compute_cotangent_weights(
    const HalfEdgeMesh& mesh
);

/**
 * Re-extract face_frames (reference angles) from the complex field after
 * smoothing, so that singularity detection sees consistent data.
 *
 * face_frames[f] = arg(field.face_field[f]) / 4.0
 */
void recompute_face_frames(CrossField& field);

} // namespace qf

#endif // QUADFORGE_FIELD_SMOOTHING_H
