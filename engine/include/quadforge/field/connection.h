#pragma once
#ifndef QUADFORGE_FIELD_CONNECTION_H
#define QUADFORGE_FIELD_CONNECTION_H

#include <cstdint>
#include <vector>
#include <complex>

namespace qf {  // was "quadforge" — unified to match all other field files

/**
 * 4-RoSy connection Laplacian for globally-optimal direction fields
 * (Knöppel et al. 2013).
 *
 * The connection Laplacian L is a complex-valued sparse matrix of size
 * F×F (one variable per face).  Its smallest eigenvector gives the
 * globally smoothest 4-RoSy direction field on the surface.
 *
 * Internally stored as COO (coordinate) format for easy assembly,
 * then converted to CSR for the eigensolver.
 */

struct ConnectionLaplacian {
    int32_t n;                       // matrix size (number of faces)
    std::vector<int32_t> row_idx;    // COO row indices
    std::vector<int32_t> col_idx;    // COO col indices
    std::vector<std::complex<double>> values; // COO values

    // CSR form (populated by finalize())
    std::vector<int32_t> csr_row_ptr;
    std::vector<int32_t> csr_col_idx;
    std::vector<std::complex<double>> csr_vals;

    void finalize();  // convert COO → CSR, sort within rows
};

/**
 * Compute the parallel transport angle between two adjacent faces.
 *
 * Given edge shared by face i and face j, this is the rotation angle
 * φ_ij such that exp(4iφ_ij) maps the 4-RoSy frame of face i to that
 * of face j through parallel transport along the shared edge.
 *
 * @param e1_i, e2_i  Local frame of face i (tangent basis vectors)
 * @param n_i         Normal of face i (unused — kept for ABI compatibility)
 * @param e1_j, e2_j  Local frame of face j
 * @param n_j         Normal of face j (unused — kept for ABI compatibility)
 * @param edge_vec    Vector along the shared edge (v1 - v0)
 * @return            Parallel transport angle in [-π/4, π/4)
 */
double parallel_transport_angle(
    const float* e1_i, const float* e2_i, const float* n_i,
    const float* e1_j, const float* e2_j, const float* n_j,
    const float* edge_vec
);

/**
 * Assemble the connection Laplacian for a triangulated mesh.
 *
 * @param positions     [3*N] vertex positions
 * @param tris          [3*T] triangle indices
 * @param nt            Triangle count
 * @param face_e1       [3*T] local X-axis per face
 * @param face_e2       [3*T] local Y-axis per face
 * @param face_normals  [3*T] face normals
 * @param feature_edges Flat edge pairs [v0,v1,...] that are hard edges
 * @param constraint_weight  Penalty weight for feature edge constraints
 *                           (large value, e.g. 1e6)
 * @param out_L         Output connection Laplacian
 */
void assemble_connection_laplacian(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    const float* face_e1,
    const float* face_e2,
    const float* face_normals,
    const std::vector<int32_t>& feature_edges,
    double constraint_weight,
    ConnectionLaplacian& out_L
);

/**
 * Solve for the smallest eigenvector of the connection Laplacian.
 *
 * Uses a shift-and-invert power iteration:
 *   1. Apply a small positive regularization shift ε (e.g. 1e-10) to the
 *      diagonal to ensure numerical invertibility:
 *        L_reg = L + ε * I
 *      This is necessary because a 4-RoSy connection Laplacian is positive
 *      SEMI-definite: λ_min may be exactly 0 on a perfectly smooth mesh,
 *      making L singular.  Solving L \ z with σ = 0 would be ill-conditioned
 *      or fail entirely on such meshes.
 *      BUG FIX (v97): the previous comment stated "Apply a shift σ = 0" which
 *      implied L is invertible at σ = 0.  That is wrong for semi-definite L.
 *      The implementation always adds ε > 0 before the solve.
 *   2. Iterate: z_new = L_reg \ z_old,  z_new /= |z_new|
 *      This converges to the eigenvector for the smallest eigenvalue of L.
 * For small to medium meshes (< 200 K faces) this converges quickly.
 *
 * @param L           The assembled connection Laplacian (CSR form required)
 * @param max_iter    Maximum power iterations (default 200)
 * @param tol         Convergence tolerance (default 1e-8)
 * @param out_field   Output eigenvector (complex, size L.n), unit complex
 *                    numbers giving the per-face field direction
 * @return            True on convergence, false if max_iter exceeded
 */
bool solve_connection_laplacian_eigenvector(
    const ConnectionLaplacian& L,
    int max_iter,
    double tol,
    std::vector<std::complex<double>>& out_field
);

/**
 * Compute per-face local orthonormal frames from flat triangle arrays.
 *
 * This is the public flat-array companion to the internal
 * compute_face_frames() helper in cross_field.cpp.  It is exposed here
 * so that unit tests and tools that work with raw arrays (rather than a
 * HalfEdgeMesh) can build the per-face tangent bases needed before
 * calling assemble_connection_laplacian() or detect_singularities().
 *
 * Algorithm (mirrors cross_field.cpp compute_face_frames + v59/v60 guards):
 *   1. Compute the face normal n = (v1-v0) × (v2-v0), normalised.
 *      If the face is degenerate (|n| < 1e-12) fall back to (0,0,1).
 *   2. Project the first edge (v1-v0) onto the tangent plane n⊥ to get e1.
 *      If the projection is too short fall back to n.unitOrthogonal().
 *   3. e2 = n × e1 (always orthogonal, always unit-length).
 *
 * @param positions        [3*N] vertex XYZ (row-major)
 * @param tris             [3*T] triangle vertex indices
 * @param nt               Triangle count T
 * @param out_face_normals [3*T] output per-face unit normals  (may be NULL)
 * @param out_e1           [3*T] output per-face local X-axis
 * @param out_e2           [3*T] output per-face local Y-axis
 *
 * BUG FIX (v94): Previously this function existed only as a static
 * helper inside cross_field.cpp, making it unreachable from unit tests
 * and the flat-array detect_singularities() overload.  Both
 * test_crossfield.cpp and the new singularity.cpp overload call it as
 * compute_face_frames_cpp() — the _cpp suffix distinguishes it from the
 * internal static in cross_field.cpp.
 */
void compute_face_frames_cpp(
    const float*   positions,
    const int32_t* tris,
    int32_t        nt,
    float*         out_face_normals,
    float*         out_e1,
    float*         out_e2
);

} // namespace qf  // was "quadforge" — unified

#endif // QUADFORGE_FIELD_CONNECTION_H
