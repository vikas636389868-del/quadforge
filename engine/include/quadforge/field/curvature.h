#pragma once
/* quadforge/field/curvature.h — Discrete principal curvature computation.
 *
 * Implements the Rusinkiewicz (2004) shape operator on triangle meshes.
 * Per-edge curvature estimates are assembled into a per-vertex quadratic
 * form whose eigenvectors are the principal curvature directions.
 *
 * Reference: S. Rusinkiewicz, "Estimating curvatures and their derivatives
 * on triangle meshes," 3DPVT 2004.
 */

#ifndef QUADFORGE_FIELD_CURVATURE_H
#define QUADFORGE_FIELD_CURVATURE_H

#include "../types.h"
#include "../mesh/halfedge.h"

namespace qf {

/** Principal curvature data for every vertex. */
struct CurvatureData {
    Eigen::VectorXd kappa1;      ///< Max principal curvature magnitude [V]
    Eigen::VectorXd kappa2;      ///< Min principal curvature magnitude [V]
    Eigen::MatrixXd dir1;        ///< Max curvature direction [V×3]
    Eigen::MatrixXd dir2;        ///< Min curvature direction [V×3]
    Eigen::VectorXd mean_curv;   ///< (κ1+κ2)/2
    Eigen::VectorXd gauss_curv;  ///< κ1*κ2
    Eigen::VectorXd max_abs_curv;///< max(|κ1|,|κ2|) — used for sizing field
};

/**
 * Compute principal curvatures for all vertices.
 *
 * Uses Rusinkiewicz shape operator: for each edge, estimate the
 * normal curvature κ_e = 2*(n1-n2)·e / |e|², then assemble a 3×3
 * symmetric tensor per vertex and eigen-decompose it.
 *
 * Computation is embarrassingly parallel (independent per vertex).
 *
 * @param mesh         Input triangle mesh
 * @param num_threads  0 = auto-detect (OpenMP)
 */
CurvatureData compute_curvature(const HalfEdgeMesh& mesh,
                                int num_threads = 0);

} // namespace qf

#endif // QUADFORGE_FIELD_CURVATURE_H
