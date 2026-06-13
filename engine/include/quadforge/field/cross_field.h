#pragma once
/* quadforge/field/cross_field.h — Globally optimal 4-RoSy direction field.
 *
 * Implements:
 *   - Connection Laplacian assembly (complex-valued, per-face)
 *   - Knöppel et al. 2013: smallest eigenvector of connection Laplacian
 *   - Feature/boundary/symmetry constraint integration
 *   - Fallback: simple curvature-aligned field (no solve required)
 *
 * References:
 *   Knöppel et al. (2013), "Globally Optimal Direction Fields," SIGGRAPH.
 *   Ray et al. (2008), "N-Symmetry Direction Field Design."
 */

#ifndef QUADFORGE_FIELD_CROSS_FIELD_H
#define QUADFORGE_FIELD_CROSS_FIELD_H

#include <complex>
#include <vector>
#include "../types.h"
#include "../mesh/halfedge.h"
#include "../mesh/features.h"
#include "curvature.h"

namespace qf {

/** Solver algorithm selection.
 *
 *  Maps to QFParams::field_solver:
 *    0 = EIGEN_SMOOTH   — fast power-iteration (50-iter cap, no convergence check).
 *                         Good balance: faster than KNOPPEL_2013, smoother than
 *                         CURVATURE_ONLY.  Skips the post-solve Gauss-Seidel pass.
 *    1 = KNOPPEL_2013   — globally optimal eigenvector (up to 500 iters, full
 *                         convergence check + post-solve smoothing).  Best quality.
 *    2 = CURVATURE_ONLY — curvature-aligned field, no solve at all.  Fastest.
 *
 *  BUG FIX (v83): EIGEN_SMOOTH was declared in api.h (field_solver=0) but was
 *  missing from this enum entirely.  engine.cpp mapped field_solver==0 and ==1
 *  both to KNOPPEL_2013, silently ignoring the user's choice.  The enum is now
 *  ordered to match the integer values in QFParams::field_solver.
 */
enum class FieldSolver {
    EIGEN_SMOOTH   = 0, ///< Fast power-iteration (50 iters) — balanced speed/quality
    KNOPPEL_2013   = 1, ///< Globally optimal eigensolver — best quality
    CURVATURE_ONLY = 2  ///< Simple curvature-aligned field — fastest
};

/** Parameters for the cross-field solve. */
struct CrossFieldParams {
    FieldSolver solver          = FieldSolver::KNOPPEL_2013;
    int         max_iterations  = 500;
    double      convergence_tol = 1e-8;
    double      constraint_weight = 1e6;   ///< Penalty for hard constraints
    bool        align_to_features = true;
    bool        enforce_symmetry_x = false;
    bool        enforce_symmetry_y = false;
    bool        enforce_symmetry_z = false;
    int         num_threads = 0;           ///< 0 = OpenMP auto
};

/**
 * Compute a globally smooth 4-RoSy direction field on the triangle mesh.
 *
 * The field is represented as one complex number per face:
 *   u_f = r_f * exp(4i * theta_f)
 * where theta_f is the angle of one cross arm relative to a local frame.
 *
 * @param mesh      The triangle mesh
 * @param curvature Principal curvature directions (used as initial guess
 *                  and for curvature-only mode)
 * @param features  Detected feature edges (used as hard constraints)
 * @param params    Solver configuration
 */
CrossField compute_cross_field(
    const HalfEdgeMesh&  mesh,
    const CurvatureData& curvature,
    const FeatureData&   features,
    const CrossFieldParams& params = CrossFieldParams{}
);

/**
 * Build the connection Laplacian for a 4-RoSy field.
 * Public for unit testing.
 *
 * @param mesh         Triangle mesh
 * @param face_frames  Per-face reference angle in local coordinate frame
 * @return             Complex sparse matrix of size F×F
 */
SparseMatC build_connection_laplacian(
    const HalfEdgeMesh&        mesh,
    const std::vector<double>& face_frames
);

/**
 * Compute the parallel transport angle between adjacent faces
 * sharing half-edge he.
 */
double parallel_transport_angle(const HalfEdgeMesh& mesh, int he);

} // namespace qf

#endif // QUADFORGE_FIELD_CROSS_FIELD_H
