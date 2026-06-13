#pragma once
#ifndef QUADFORGE_PARAM_POISSON_H
#define QUADFORGE_PARAM_POISSON_H

#include <cstdint>
#include <vector>
#include <unordered_set>

namespace qf {

/**
 * Poisson-based global UV parametrization.
 *
 * Given a combed 4-RoSy field and a seam cut, solves for (U,V)
 * coordinates that minimise:
 *
 *   E = ∫ |∇U - X|² + |∇V - Y|² dA
 *
 * where X, Y are the two perpendicular directions of the combed cross-
 * field.  This is discretised as a sparse linear system (Cotangent
 * Laplacian) solved with a sparse Cholesky factorisation.
 *
 * The sizing field modulates the Poisson equation's coefficients so
 * that the iso-line spacing matches the desired local quad size.
 */

/**
 * Build the cotangent Laplacian for a triangulated mesh.
 *
 * Returns the matrix in CSR format (symmetric, SPD after Dirichlet BC).
 *
 * @param positions   [3*N]
 * @param tris        [3*T]
 * @param nv, nt
 * @param sizing      [N] target edge length per vertex (nullptr = uniform)
 * @param out_row_ptr CSR row pointers [nv+1]
 * @param out_col_idx CSR column indices
 * @param out_vals    CSR values (cotangent weights)
 */
void build_cotangent_laplacian(
    const float*   positions,
    const int32_t* tris,
    int32_t nv, int32_t nt,
    const float* sizing,
    std::vector<int32_t>& out_row_ptr,
    std::vector<int32_t>& out_col_idx,
    std::vector<double>&  out_vals
);

/**
 * Build the right-hand side for the Poisson system.
 *
 * The RHS encodes the cross-field directions X (for U) and Y (for V)
 * as divergence of the target gradient field.
 *
 * @param positions     [3*N]
 * @param tris          [3*T]
 * @param nv, nt
 * @param combed_angles [T] per-face combed field angle (radians)
 * @param face_e1       [3*T] local X-axis per face
 * @param face_e2       [3*T] local Y-axis per face
 * @param sizing        [N] target edge length (nullptr = uniform)
 * @param out_rhs_u     Output RHS for U component [nv]
 * @param out_rhs_v     Output RHS for V component [nv]
 */
void build_poisson_rhs(
    const float*   positions,
    const int32_t* tris,
    int32_t nv, int32_t nt,
    const double* combed_angles,
    const float*  face_e1,
    const float*  face_e2,
    const float*  sizing,
    std::vector<double>& out_rhs_u,
    std::vector<double>& out_rhs_v
);

/**
 * Solve the Poisson system for both U and V simultaneously.
 *
 * Uses Eigen SparseLU (handles asymmetric system after Dirichlet BC
 * modification) with SimplicialLDLT fallback on the symmetric system.
 *
 * @param row_ptr    CSR row pointers
 * @param col_idx    CSR column indices
 * @param vals       CSR values (cotangent Laplacian)
 * @param rhs_u      RHS for U component [nv]
 * @param rhs_v      RHS for V component [nv]
 * @param seam_verts Set of seam vertex indices (one DOF pinned for BC)
 * @param out_u      Output U coordinates [nv]
 * @param out_v      Output V coordinates [nv]
 * @return           True if solve succeeded
 */
bool solve_poisson_parametrization(
    const std::vector<int32_t>& row_ptr,
    const std::vector<int32_t>& col_idx,
    const std::vector<double>&  vals,
    const std::vector<double>&  rhs_u,
    const std::vector<double>&  rhs_v,
    const std::unordered_set<int32_t>& seam_verts,
    std::vector<double>& out_u,
    std::vector<double>& out_v
);

/**
 * Conjugate Gradient solver for symmetric positive definite systems.
 *
 * Iterative solver suitable for large sparse SPD matrices.
 *
 * @param row_ptr, col_idx, vals  CSR matrix
 * @param n      System dimension
 * @param rhs    Right-hand side [n]
 * @param x      Initial guess / output solution [n]
 * @param max_iter   Maximum CG iterations
 * @param tol        Convergence tolerance (relative residual)
 * @return           Number of iterations taken; negative if failed to converge
 */
int conjugate_gradient(
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double*  vals,
    int32_t n,
    const double* rhs,
    double* x,
    int max_iter,
    double tol
);

} // namespace qf

#endif // QUADFORGE_PARAM_POISSON_H
