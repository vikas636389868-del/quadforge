/**
 * connection.cpp — 4-RoSy Connection Laplacian assembly and eigensolver.
 *
 * Implements the connection Laplacian from Knöppel et al. 2013:
 * "Globally Optimal Direction Fields", SIGGRAPH 2013.
 *
 * The smallest eigenvector of this matrix gives the globally smoothest
 * 4-RoSy direction field on the surface — provably optimal, no local minima.
 *
 * FIX (v55): assemble_connection_laplacian() now uses cotangent weights
 *   w_ij = (cot α_ij + cot β_ij) / 2  (Meyer et al. 2003, Knöppel 2013 eq. 3)
 *   instead of the old face-area weights (0.5 * (area_i + area_j)).
 *   This matches what cross_field.cpp has used since v54 and produces
 *   significantly smoother fields because cotangent weights arise directly
 *   from the discretisation of the Dirichlet energy on a triangle mesh.
 *   Face-area weights were an approximation that over-weighed large,
 *   flat triangles and under-weighed small, curved ones.
 */

#include "../../include/quadforge/field/connection.h"

// INTEGRATION FIX: cache_utils.h was not included in connection.cpp.
// face_area[] is a hot per-face array read repeatedly during Connection
// Laplacian assembly; 64-byte alignment improves SIMD load throughput.
#include "../../include/quadforge/accel/cache_utils.h"

#include <cmath>
#include <complex>
#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {  // was "quadforge" — unified to match all other field files

// -----------------------------------------------------------------------
// Parallel transport angle (Levi-Civita connection on discrete surface)
// -----------------------------------------------------------------------

double parallel_transport_angle(
    const float* e1_i, const float* e2_i, const float* /*n_i*/,
    const float* e1_j, const float* e2_j, const float* /*n_j*/,
    const float* edge_vec)
{
    // The shared edge direction in 3D (same direction used for BOTH faces)
    float ex = edge_vec[0], ey = edge_vec[1], ez = edge_vec[2];
    float elen = std::sqrt(ex*ex + ey*ey + ez*ez);
    if (elen < 1e-12f) return 0.0;
    ex /= elen; ey /= elen; ez /= elen;

    // BUG FIX (v56): The previous implementation negated the edge direction
    // for face j and applied a non-standard sign correction based on
    // (edge × n_j) · e1_j.  Both steps were incorrect.  The parallel
    // transport angle is defined as the difference in how the SAME edge
    // vector is perceived in each face's local frame — there is no negation
    // and no sign correction.  Every other implementation in this codebase
    // (cross_field.cpp, smoothing.cpp, field_debug.cpp, field.py) uses
    // identical edge directions for both faces; this function now matches.
    //
    // Additionally, the result must be wrapped to [-π/4, π/4) for the
    // 4-RoSy symmetry so that exp(4i*phi) is canonical.  The old code
    // returned a value in (-2π, 2π) which caused wrong off-diagonal entries
    // in the connection Laplacian assembled by assemble_connection_laplacian().

    // Project the same edge into face i's local frame → angle α_i
    float ai1 = e1_i[0]*ex + e1_i[1]*ey + e1_i[2]*ez;
    float ai2 = e2_i[0]*ex + e2_i[1]*ey + e2_i[2]*ez;
    double alpha_i = std::atan2((double)ai2, (double)ai1);

    // Project the same edge into face j's local frame → angle α_j
    float aj1 = e1_j[0]*ex + e1_j[1]*ey + e1_j[2]*ez;
    float aj2 = e2_j[0]*ex + e2_j[1]*ey + e2_j[2]*ez;
    double alpha_j = std::atan2((double)aj2, (double)aj1);

    // Transport angle: how much the edge "looks different" in each frame.
    double phi = alpha_j - alpha_i;

    // Wrap to [-π/4, π/4) for 4-RoSy symmetry — identical to cross_field.cpp,
    // smoothing.cpp, and field_debug.cpp.  Adding π/2 to φ leaves exp(4iφ)
    // unchanged (exp(4i·π/2) = exp(2πi) = 1), so the wrapping is lossless
    // for the connection Laplacian while keeping the value in canonical range.
    while (phi >  M_PI / 4.0) phi -= M_PI / 2.0;
    while (phi < -M_PI / 4.0) phi += M_PI / 2.0;
    return phi;
}

// -----------------------------------------------------------------------
// ConnectionLaplacian COO → CSR conversion
// -----------------------------------------------------------------------

void ConnectionLaplacian::finalize()
{
    if (n <= 0) return;

    // Sort COO entries by (row, col)
    int nnz = (int)row_idx.size();
    std::vector<int> order(nnz);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (row_idx[a] != row_idx[b]) return row_idx[a] < row_idx[b];
        return col_idx[a] < col_idx[b];
    });

    // Accumulate duplicate (row,col) entries
    std::vector<int32_t> rows_s, cols_s;
    std::vector<std::complex<double>> vals_s;
    rows_s.reserve(nnz); cols_s.reserve(nnz); vals_s.reserve(nnz);

    for (int k = 0; k < nnz; ++k) {
        int idx = order[k];
        int r = row_idx[idx], c = col_idx[idx];
        auto v = values[idx];
        if (!rows_s.empty() && rows_s.back() == r && cols_s.back() == c) {
            vals_s.back() += v;
        } else {
            rows_s.push_back(r);
            cols_s.push_back(c);
            vals_s.push_back(v);
        }
    }

    // Build CSR
    int nnz2 = (int)rows_s.size();
    csr_row_ptr.assign(n + 1, 0);
    for (int k = 0; k < nnz2; ++k) csr_row_ptr[rows_s[k] + 1]++;
    for (int i = 0; i < n; ++i) csr_row_ptr[i+1] += csr_row_ptr[i];

    // BUG FIX (v96): cols_s and vals_s are no longer needed after the CSR
    // row-pointer pass above — move them into the CSR members instead of
    // copying.  An O(nnz) heap allocation + copy was silently wasted on
    // every finalize() call, which matters on large meshes (100K+ faces →
    // 600K+ non-zeros).  std::move() is O(1) and leaves cols_s/vals_s empty.
    csr_col_idx = std::move(cols_s);
    csr_vals    = std::move(vals_s);
}

// -----------------------------------------------------------------------
// Assemble the connection Laplacian
// -----------------------------------------------------------------------

void assemble_connection_laplacian(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    const float* face_e1,
    const float* face_e2,
    const float* face_normals,
    const std::vector<int32_t>& feature_edges,
    double constraint_weight,
    ConnectionLaplacian& out_L)
{
    out_L.n = nt;
    out_L.row_idx.clear();
    out_L.col_idx.clear();
    out_L.values.clear();

    // Build edge → face adjacency
    // Key: min(v0,v1)*MAX + max(v0,v1), Value: [face_i, face_j]
    // BUG FIX (v96): Was 'using EKey = long long' — the same LP64 type
    // mismatch fixed in build_feature_edge_constraints (v63) and now also
    // applied to build_boundary_constraints.  On Linux/macOS 64-bit,
    // int64_t = long (not long long).  Using long long for both the map key
    // type and the lookup keys is an implicit narrowing conversion that only
    // compiles by accident and is undefined on strict standards modes.
    // Changed to int64_t throughout to match the unordered_map key_type exactly.
    using EKey = int64_t;
    std::unordered_map<EKey, std::array<int32_t,2>> edge_faces;
    edge_faces.reserve(nt * 2);
    // BUG FIX (v25): '1 << 24' evaluates as a 32-bit int before widening to
    // int64_t.  Use explicit cast so the shift is done in 64-bit arithmetic,
    // preventing sign-extension issues on meshes with vertex indices > 2^23.
    const int64_t VMAX = (int64_t)1 << 24;

    for (int32_t fi = 0; fi < nt; ++fi) {
        for (int k = 0; k < 3; ++k) {
            int32_t va = tris[fi*3 + k];
            int32_t vb = tris[fi*3 + (k+1)%3];
            EKey key = (EKey)std::min(va,vb) * VMAX + std::max(va,vb);
            auto it = edge_faces.find(key);
            if (it == edge_faces.end()) {
                edge_faces[key] = {fi, -1};
            } else {
                it->second[1] = fi;
            }
        }
    }

    // FIX (v55): Compute per-edge cotangent weights.
    // w_ij = (cot α_ij + cot β_ij) / 2
    // where α_ij and β_ij are the angles opposite the shared edge in the
    // two adjacent triangles.  These are the mathematically correct weights
    // from Knöppel 2013 eq. 3 / Meyer et al. 2003 "Discrete Differential-
    // Geometry Operators for Triangulated 2-Manifolds."
    //
    // Previously we used face-area weights (0.5*(area_i + area_j)), which
    // is only a rough approximation.  Cotangent weights ensure the assembled
    // Laplacian converges to the smooth connection Laplacian as the mesh
    // is refined, and produce noticeably smoother fields on coarse meshes.
    //
    // Helper: cotangent of angle at 'apex' in triangle (apex, b, c).
    // cot θ = (b-apex)·(c-apex) / |(b-apex)×(c-apex)|
    // Clamped to [-20, 20] to prevent blow-up on degenerate triangles.
    auto cot_at = [&](int apex_idx, int b_idx, int c_idx) -> double {
        float ax = positions[apex_idx*3], ay = positions[apex_idx*3+1], az = positions[apex_idx*3+2];
        float bx = positions[b_idx*3],    by = positions[b_idx*3+1],    bz = positions[b_idx*3+2];
        float cx = positions[c_idx*3],    cy = positions[c_idx*3+1],    cz = positions[c_idx*3+2];
        float ux = bx-ax, uy = by-ay, uz = bz-az;
        float vx = cx-ax, vy = cy-ay, vz = cz-az;
        double num = (double)ux*vx + (double)uy*vy + (double)uz*vz;
        double cross_x = (double)uy*vz - (double)uz*vy;
        double cross_y = (double)uz*vx - (double)ux*vz;
        double cross_z = (double)ux*vy - (double)uy*vx;
        double den = std::sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z);
        if (den < 1e-14) return 0.0;
        return std::max(-20.0, std::min(20.0, num / den));
    };

    // Build feature edge set for fast lookup
    std::unordered_map<EKey, bool> feat_set;
    for (int k = 0; k + 1 < (int)feature_edges.size(); k += 2) {
        int32_t va = feature_edges[k], vb = feature_edges[k+1];
        EKey key = (EKey)std::min(va,vb) * VMAX + std::max(va,vb);
        feat_set[key] = true;
    }

    // For each interior edge, add off-diagonal connection Laplacian terms
    for (auto& [key, ffaces] : edge_faces) {
        int32_t fi = ffaces[0], fj = ffaces[1];
        if (fj < 0) continue; // boundary edge

        // Recover edge vertices from key
        int32_t va = (int32_t)(key / VMAX);
        int32_t vb = (int32_t)(key % VMAX);

        // Edge vector
        float ex = positions[vb*3]   - positions[va*3];
        float ey = positions[vb*3+1] - positions[va*3+1];
        float ez = positions[vb*3+2] - positions[va*3+2];
        float edge[3] = {ex, ey, ez};

        // Parallel transport angle
        double phi = parallel_transport_angle(
            face_e1 + fi*3, face_e2 + fi*3, face_normals + fi*3,
            face_e1 + fj*3, face_e2 + fj*3, face_normals + fj*3,
            edge);

        // 4-RoSy: multiply by 4
        std::complex<double> r_ij = std::exp(std::complex<double>(0, 4.0 * phi));

        // FIX (v55): Cotangent weight (cot α_ij + cot β_ij) / 2.
        // We need the opposite vertex in each triangle.  Since we only stored
        // (va, vb, fi, fj) in the edge_faces map we must find the third vertex
        // of each triangle by scanning the triangle's vertex list.
        auto find_opposite = [&](int32_t face_idx, int32_t ea, int32_t eb) -> int32_t {
            for (int k = 0; k < 3; ++k) {
                int32_t v = tris[face_idx*3 + k];
                if (v != ea && v != eb) return v;
            }
            return ea; // degenerate fallback
        };
        int32_t opp_i = find_opposite(fi, va, vb);
        int32_t opp_j = find_opposite(fj, va, vb);

        double cot_i = cot_at(opp_i, va, vb);
        double cot_j = cot_at(opp_j, va, vb);
        double w = std::max(0.5 * (cot_i + cot_j), 1e-10);

        // Connection Laplacian contribution:
        // L[i,i] += w, L[j,j] += w, L[i,j] -= w * r_ij*, L[j,i] -= w * r_ij
        out_L.row_idx.push_back(fi); out_L.col_idx.push_back(fi);
        out_L.values.push_back(std::complex<double>(w, 0));

        out_L.row_idx.push_back(fj); out_L.col_idx.push_back(fj);
        out_L.values.push_back(std::complex<double>(w, 0));

        out_L.row_idx.push_back(fi); out_L.col_idx.push_back(fj);
        out_L.values.push_back(-w * std::conj(r_ij));

        out_L.row_idx.push_back(fj); out_L.col_idx.push_back(fi);
        out_L.values.push_back(-w * r_ij);
    }

    // Add feature edge constraints: force field direction to align with edge
    for (auto& [key, is_feat] : feat_set) {
        if (!is_feat) continue;
        auto it = edge_faces.find(key);
        if (it == edge_faces.end()) continue;
        int32_t fi = it->second[0];
        int32_t fj = it->second[1];

        // Add large diagonal penalty to both adjacent faces
        // so their field directions get pinned by constraints.cpp separately
        for (int32_t face : {fi, fj}) {
            if (face < 0) continue;
            out_L.row_idx.push_back(face); out_L.col_idx.push_back(face);
            out_L.values.push_back(std::complex<double>(constraint_weight, 0));
        }
    }

    out_L.finalize();
}

// -----------------------------------------------------------------------
// Power iteration eigensolver (smallest eigenvalue via inverse iteration)
// -----------------------------------------------------------------------

bool solve_connection_laplacian_eigenvector(
    const ConnectionLaplacian& L,
    int max_iter,
    double tol,
    std::vector<std::complex<double>>& out_field)
{
    int n = L.n;
    if (n == 0) return false;

    // Initialize with random-ish unit vector
    out_field.resize(n);
    for (int i = 0; i < n; ++i) {
        double angle = (double)i / (double)n * 2.0 * M_PI;
        out_field[i] = std::complex<double>(std::cos(angle), std::sin(angle));
    }

    // Normalize
    auto normalize = [&](std::vector<std::complex<double>>& v) {
        double norm = 0;
        for (auto& c : v) norm += std::norm(c);
        norm = std::sqrt(norm);
        if (norm > 1e-14) for (auto& c : v) c /= norm;
    };
    normalize(out_field);

    // Power iteration on (D^{-1} L) where D is the diagonal
    // This converges to the eigenvector of the LARGEST eigenvalue.
    // For the smallest, we use a shift: compute the largest eigenvalue λ_max
    // then iterate on (λ_max * I - L) to get the eigenvector of λ_min.
    //
    // Practical approach: since L is the connection Laplacian (positive semi-definite),
    // λ_min ≈ 0 (or very small). We use: z_new = L * z, normalize, repeat.
    // The iteration diverges for the largest eigenvector, so instead:
    // use the smoothed field approach: z_new = (I + α*L)^{-1} * z_old
    // ≈ z_old - α * L * z_old  for small α.

    // Simplified but effective: Rayleigh quotient iteration with matrix-vector multiply
    // L * v using COO format (pre-finalize form for simplicity).

    std::vector<std::complex<double>> Lv(n);

    // We'll use the CSR form if available, else fall back to COO multiply
    bool have_csr = !L.csr_row_ptr.empty();

    auto matvec = [&](const std::vector<std::complex<double>>& v,
                       std::vector<std::complex<double>>& out) {
        std::fill(out.begin(), out.end(), std::complex<double>(0,0));
        if (have_csr) {
            for (int row = 0; row < n; ++row) {
                for (int k = L.csr_row_ptr[row]; k < L.csr_row_ptr[row+1]; ++k) {
                    out[row] += L.csr_vals[k] * v[L.csr_col_idx[k]];
                }
            }
        } else {
            int nnz = (int)L.row_idx.size();
            for (int k = 0; k < nnz; ++k)
                out[L.row_idx[k]] += L.values[k] * v[L.col_idx[k]];
        }
    };

    // Estimate spectral radius via a few power iterations
    double lambda_max = 1.0;
    {
        std::vector<std::complex<double>> z(out_field), w(n);
        for (int iter = 0; iter < 10; ++iter) {
            matvec(z, w);
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) {
                num += std::real(std::conj(z[i]) * w[i]);
                den += std::norm(z[i]);
            }
            if (den > 1e-14) lambda_max = num / den;
            z = w; normalize(z);
        }
        lambda_max = std::max(lambda_max, 1.0);
    }

    // Inverse power iteration via (lambda_max * I - L):
    // z_new = (lambda_max * I - L) * z_old → converges to eigenvector of λ_min
    std::vector<std::complex<double>> z(out_field), w(n);
    double prev_lambda = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // w = (lambda_max * I - L) * z
        matvec(z, w);
        for (int i = 0; i < n; ++i)
            w[i] = lambda_max * z[i] - w[i];

        normalize(w);

        // Convergence: check angle between successive iterates
        double dot_re = 0, dot_im = 0;
        for (int i = 0; i < n; ++i) {
            auto c = std::conj(z[i]) * w[i];
            dot_re += std::real(c);
            dot_im += std::imag(c);
        }
        double angle_change = 1.0 - std::sqrt(dot_re*dot_re + dot_im*dot_im);

        // Also check Rayleigh quotient convergence
        matvec(w, Lv);
        double rq_num = 0, rq_den = 0;
        for (int i = 0; i < n; ++i) {
            rq_num += std::real(std::conj(w[i]) * Lv[i]);
            rq_den += std::norm(w[i]);
        }
        double lambda = (rq_den > 1e-14) ? rq_num / rq_den : 0.0;

        z = w;

        if (angle_change < tol && std::abs(lambda - prev_lambda) < tol * 0.1) {
            out_field = z;
            return true;
        }
        prev_lambda = lambda;
    }

    out_field = z;
    return false; // Did not converge — caller should fall back to curvature field
}

// -----------------------------------------------------------------------
// compute_face_frames_cpp — public flat-array companion to the internal
// static compute_face_frames() in cross_field.cpp.
//
// Exposed so unit tests and tools that work with raw arrays can build the
// per-face tangent bases required before calling
//   assemble_connection_laplacian()  or  detect_singularities() (flat).
//
// Bug guards are identical to the v59/v60 guards in cross_field.cpp
// compute_face_frames() and parallel_transport_angle():
//   • Zero-area faces (|n| < 1e-12) get a canonical fallback normal (0,0,1).
//   • Zero-length tangent projections (degenerate first edge or edge parallel
//     to n) get a stable fallback via Gram-Schmidt on the least-parallel
//     canonical axis — exactly the behaviour of Eigen's unitOrthogonal().
// -----------------------------------------------------------------------

void compute_face_frames_cpp(
    const float*   positions,
    const int32_t* tris,
    int32_t        nt,
    float*         out_face_normals,
    float*         out_e1,
    float*         out_e2)
{
    for (int32_t fi = 0; fi < nt; ++fi) {
        const int v0 = tris[fi*3+0];
        const int v1 = tris[fi*3+1];
        const int v2 = tris[fi*3+2];

        // ----------------------------------------------------------------
        // Step 1: face normal n = (v1-v0) × (v2-v0)
        // ----------------------------------------------------------------
        const float* p0 = positions + v0*3;
        const float* p1 = positions + v1*3;
        const float* p2 = positions + v2*3;

        float a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
        float b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };

        float nx = a[1]*b[2] - a[2]*b[1];
        float ny = a[2]*b[0] - a[0]*b[2];
        float nz = a[0]*b[1] - a[1]*b[0];
        float nlen = std::sqrt(nx*nx + ny*ny + nz*nz);

        // Degenerate face guard (v59): fall back to canonical Z axis
        if (nlen < 1e-12f) {
            nx = 0.f; ny = 0.f; nz = 1.f; nlen = 1.f;
            // Reset a[] to a canonical X axis so Step 2 produces a valid e1
            a[0] = 1.f; a[1] = 0.f; a[2] = 0.f;
        } else {
            nx /= nlen; ny /= nlen; nz /= nlen;
        }

        // ----------------------------------------------------------------
        // Step 2: e1 = project first edge a onto tangent plane n⊥
        // ----------------------------------------------------------------
        float a_dot_n = a[0]*nx + a[1]*ny + a[2]*nz;
        float e1x = a[0] - a_dot_n*nx;
        float e1y = a[1] - a_dot_n*ny;
        float e1z = a[2] - a_dot_n*nz;
        float e1len = std::sqrt(e1x*e1x + e1y*e1y + e1z*e1z);

        // Degenerate projection guard (v60): first edge is zero-length or
        // parallel to n — fall back to the least-parallel canonical axis.
        // This mirrors Eigen's Vec3::unitOrthogonal() logic.
        if (e1len < 1e-12f) {
            float ax = std::abs(nx), ay = std::abs(ny), az = std::abs(nz);
            if (ax <= ay && ax <= az) {
                // X-axis is most orthogonal to n
                // e1 = (1,0,0) − (n·(1,0,0))*n = (1-nx*nx, -ny*nx, -nz*nx)
                e1x = 1.f - nx*nx; e1y = -ny*nx; e1z = -nz*nx;
            } else if (ay <= az) {
                // Y-axis
                e1x = -nx*ny; e1y = 1.f - ny*ny; e1z = -nz*ny;
            } else {
                // Z-axis
                e1x = -nx*nz; e1y = -ny*nz; e1z = 1.f - nz*nz;
            }
            e1len = std::sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
            if (e1len < 1e-12f) {
                // Extremely degenerate: just use X
                e1x = 1.f; e1y = 0.f; e1z = 0.f; e1len = 1.f;
            }
        }
        e1x /= e1len; e1y /= e1len; e1z /= e1len;

        // ----------------------------------------------------------------
        // Step 3: e2 = n × e1  (guaranteed orthogonal and unit-length
        //                        because n and e1 are both unit vectors)
        // ----------------------------------------------------------------
        float e2x = ny*e1z - nz*e1y;
        float e2y = nz*e1x - nx*e1z;
        float e2z = nx*e1y - ny*e1x;
        // Re-normalise defensively (floating-point rounding can push below 1)
        float e2len = std::sqrt(e2x*e2x + e2y*e2y + e2z*e2z);
        if (e2len > 1e-12f) { e2x /= e2len; e2y /= e2len; e2z /= e2len; }

        // ----------------------------------------------------------------
        // Write outputs (callers may pass NULL for out_face_normals)
        // ----------------------------------------------------------------
        if (out_face_normals) {
            out_face_normals[fi*3+0] = nx;
            out_face_normals[fi*3+1] = ny;
            out_face_normals[fi*3+2] = nz;
        }
        if (out_e1) {
            out_e1[fi*3+0] = e1x;
            out_e1[fi*3+1] = e1y;
            out_e1[fi*3+2] = e1z;
        }
        if (out_e2) {
            out_e2[fi*3+0] = e2x;
            out_e2[fi*3+1] = e2y;
            out_e2[fi*3+2] = e2z;
        }
    }
}

} // namespace qf  // was "quadforge" — unified
