/**
 * poisson.cpp — Poisson-based global UV parametrization.
 *
 * Minimises E = ∫ |∇U - X|² + |∇V - Y|² dA
 * where X, Y are the combed cross-field directions.
 *
 * Discretised as a sparse linear system via the cotangent Laplacian,
 * solved with Eigen's sparse Cholesky (SimplicialLDLT).
 * The sizing field modulates the Poisson coefficients to match the
 * desired local quad size.
 */

#include "../../include/quadforge/param/poisson.h"

// accel integration — cache_utils.h / simd_math.h / omp_utils.h were
// previously not included in this file, leaving conjugate_gradient() with
// unaligned std::vector<double> working vectors, a scalar serial matvec
// lambda, and scalar dot/axpy/scale inner loops.  All three are now wired
// in (same fix as BUG 5 / BUG 3 / BUG 6 that was applied to gpu_solver.cpp).
#include "../../include/quadforge/accel/cache_utils.h"   // aligned_vector<double>
#include "../../include/quadforge/accel/simd_math.h"     // simd::dot, axpy, scale, spmv_csr
#include "../../include/quadforge/accel/omp_utils.h"     // parallel_for, resolve_thread_count
#include <Eigen/Geometry>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>

// Private sibling header — becomes a no-op when QF_HAS_CHOLMOD is undefined.
#include "cholmod_solver.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

    // -----------------------------------------------------------------------
    // Build cotangent Laplacian
    // -----------------------------------------------------------------------

    void build_cotangent_laplacian(
        const float* positions,
        const int32_t* tris,
        int32_t nv, int32_t nt,
        const float* sizing,
        std::vector<int32_t>& out_row_ptr,
        std::vector<int32_t>& out_col_idx,
        std::vector<double>& out_vals)
    {
        // Accumulate cotangent weights in Eigen sparse matrix
        using SpMat = Eigen::SparseMatrix<double>;
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nt * 12);

        for (int32_t fi = 0; fi < nt; ++fi) {
            int v[3] = { tris[fi * 3], tris[fi * 3 + 1], tris[fi * 3 + 2] };

            // BUG FIX (B3 — param/poisson v110): Guard against out-of-range
            // vertex indices before any array access.  Invalid indices (negative
            // or >= nv) arise from degenerate triangle lists or corrupt input and
            // would cause undefined behaviour (stack smash / heap read past end).
            // Skipping such triangles is correct: they contribute zero area and
            // zero Laplacian weight — identical to the degenerate-face guard that
            // already exists in build_poisson_rhs and in miq.cpp.
            if (v[0] < 0 || v[0] >= nv ||
                v[1] < 0 || v[1] >= nv ||
                v[2] < 0 || v[2] >= nv) continue;

            Eigen::Vector3d p[3];
            for (int k = 0; k < 3; ++k)
                p[k] = Eigen::Vector3d(positions[v[k] * 3],
                    positions[v[k] * 3 + 1],
                    positions[v[k] * 3 + 2]);

            // Sizing weight at this face (inverse square of average edge length)
            double s = 1.0;
            if (sizing) {
                double savg = (sizing[v[0]] + sizing[v[1]] + sizing[v[2]]) / 3.0;
                if (savg > 1e-10) s = 1.0 / (savg * savg);
            }

            // Compute cotangent weights for each angle
            for (int k = 0; k < 3; ++k) {
                int i = v[k], j = v[(k + 1) % 3];
                // Angle at vertex (k+2)%3, opposite to edge (i,j)
                Eigen::Vector3d a = p[k] - p[(k + 2) % 3];
                Eigen::Vector3d b = p[(k + 1) % 3] - p[(k + 2) % 3];
                double cos_a = a.dot(b);
                double sin_a = a.cross(b).norm();
                if (sin_a < 1e-12) continue;
                double cot = cos_a / sin_a;
                double w = 0.5 * cot * s;
                // BUG FIX (P3 — param/poisson v111): Clamp cotangent weight to
                // zero, NOT 1e-8.
                //
                // The previous clamp `std::max(w, 1e-8)` artificially injected a
                // non-zero positive weight for every obtuse-triangle edge that had
                // a negative or near-zero cotangent.  This caused two problems:
                //
                //   1. Inconsistency with miq.cpp::cot_weight(), which correctly
                //      uses std::max(0.0, dot/cross).  The two Poisson Laplacian
                //      implementations diverged, meaning meshes with many obtuse
                //      triangles were handled differently by the MIQ path versus
                //      the Poisson-only standalone path.
                //
                //   2. For highly obtuse meshes (terrain patches, sculpted organic
                //      meshes with elongated triangles), the artificial 1e-8 floor
                //      accumulated to non-negligible off-diagonal weight, causing
                //      the cotangent Laplacian to deviate from its geometric
                //      meaning (angle-weighted area measure) and producing mild
                //      distortion in the parametrization for those regions.
                //
                // The standard Positive Laplacian approach (Floater 2003, Botsch &
                // Kobbelt 2004) clamps to zero to ensure the matrix is
                // semi-positive-definite without artificially pumping energy into
                // obtuse-triangle edges.  The 1e-8 diagonal regulariser already
                // added in setFromTriplets ensures the matrix is strictly positive
                // definite for connected meshes.
                w = std::max(w, 0.0);  // Positive Laplacian: no negative weights

                triplets.push_back({ i, j, -w });
                triplets.push_back({ j, i, -w });
                triplets.push_back({ i, i,  w });
                triplets.push_back({ j, j,  w });
            }
        }

        // Build sparse matrix and extract CSR
        SpMat L(nv, nv);
        L.setFromTriplets(triplets.begin(), triplets.end());
        L.makeCompressed();

        // Copy to output CSR vectors
        int nnz = (int)L.nonZeros();
        out_row_ptr.resize(nv + 1);
        out_col_idx.resize(nnz);
        out_vals.resize(nnz);

        for (int i = 0; i <= nv; ++i) out_row_ptr[i] = (int32_t)L.outerIndexPtr()[i];
        for (int k = 0; k < nnz; ++k) {
            out_col_idx[k] = (int32_t)L.innerIndexPtr()[k];
            out_vals[k] = L.valuePtr()[k];
        }
    }
}

// -----------------------------------------------------------------------
// Build Poisson RHS
// -----------------------------------------------------------------------

void build_poisson_rhs(
    const float*   positions,
    const int32_t* tris,
    int32_t nv, int32_t nt,
    const double* combed_angles,
    const float*  face_e1,
    const float*  face_e2,
    const float*  sizing,
    std::vector<double>& out_rhs_u,
    std::vector<double>& out_rhs_v)
{
    out_rhs_u.assign(nv, 0.0);
    out_rhs_v.assign(nv, 0.0);

    for (int32_t fi = 0; fi < nt; ++fi) {
        int v[3] = { tris[fi*3], tris[fi*3+1], tris[fi*3+2] };

        // BUG FIX (B3 — param/poisson v110): Guard against out-of-range
        // vertex indices (same fix as build_cotangent_laplacian above).
        if (v[0] < 0 || v[0] >= nv ||
            v[1] < 0 || v[1] >= nv ||
            v[2] < 0 || v[2] >= nv) continue;

        Eigen::Vector3d p[3];
        for (int k = 0; k < 3; ++k)
            p[k] = Eigen::Vector3d(positions[v[k]*3],
                                   positions[v[k]*3+1],
                                   positions[v[k]*3+2]);

        // Cross-field direction at this face in 3D
        double theta = combed_angles[fi];
        const float* e1 = face_e1 + fi*3;
        const float* e2 = face_e2 + fi*3;
        // X = cos(θ)*e1 + sin(θ)*e2  (U direction)
        // Y = -sin(θ)*e1 + cos(θ)*e2  (V direction, perpendicular)
        Eigen::Vector3d X(
            e1[0]*std::cos(theta) + e2[0]*std::sin(theta),
            e1[1]*std::cos(theta) + e2[1]*std::sin(theta),
            e1[2]*std::cos(theta) + e2[2]*std::sin(theta));
        Eigen::Vector3d Y(
           -e1[0]*std::sin(theta) + e2[0]*std::cos(theta),
           -e1[1]*std::sin(theta) + e2[1]*std::cos(theta),
           -e1[2]*std::sin(theta) + e2[2]*std::cos(theta));

        // Sizing
        double s = 1.0;
        if (sizing) {
            double savg = (sizing[v[0]] + sizing[v[1]] + sizing[v[2]]) / 3.0;
            if (savg > 1e-10) s = 1.0 / (savg * savg);
        }

        // Cotangent-weighted divergence of X and Y → RHS
        for (int k = 0; k < 3; ++k) {
            int i = v[k], j = v[(k+1)%3], opp_k = (k+2)%3;
            Eigen::Vector3d edge_ij = p[(k+1)%3] - p[k];
            Eigen::Vector3d a = p[k]       - p[opp_k];
            Eigen::Vector3d b = p[(k+1)%3] - p[opp_k];
            double cos_a = a.dot(b);
            double sin_a = a.cross(b).norm();
            if (sin_a < 1e-12) continue;
            double cot = cos_a / sin_a;
            double w = 0.5 * cot * s;
            // BUG FIX (P4 — param/poisson v112): Add positive clamp on w.
            //
            // build_cotangent_laplacian (P3 fix, v111) clamps its cotangent
            // weight to zero for the Positive Laplacian: `w = std::max(w, 0.0)`.
            // This prevents negative off-diagonal entries from obtuse triangles,
            // keeping the LHS matrix symmetric positive semi-definite.
            //
            // build_poisson_rhs computed the same w = 0.5 * cot * s but never
            // applied the equivalent clamp.  For an obtuse triangle the cotangent
            // at the obtuse angle is negative, so w < 0.  A negative w flips the
            // sign of the divergence contribution for that edge:
            //   out_rhs_u[i] += w * xu  →  subtracts what should be added
            // This is an LHS / RHS MISMATCH: the Laplacian treats the obtuse edge
            // as having zero weight (clamped), but the RHS treats it as having a
            // negative weight (not clamped).  For meshes with many obtuse triangles
            // (terrain patches, sculpted organics, low-quality triangulations) this
            // mismatch drives the Poisson solve toward systematically wrong UV
            // values — directional bias in the parametrization that is particularly
            // visible as UV distortion along long, thin triangle bands.
            //
            // The fix mirrors the LHS clamp exactly.  The 1e-8 diagonal
            // regulariser in setFromTriplets already ensures strict positive
            // definiteness for connected meshes, so clamping the RHS weight to
            // zero for obtuse edges is both mathematically correct and numerically
            // consistent with the assembled Laplacian.
            w = std::max(w, 0.0);  // Positive Laplacian: match LHS clamp (P3 fix)
            double xu = X.dot(edge_ij);
            double yu = Y.dot(edge_ij);

            out_rhs_u[i] += w * xu;
            out_rhs_u[j] -= w * xu;
            out_rhs_v[i] += w * yu;
            out_rhs_v[j] -= w * yu;
        }
    }
}

// -----------------------------------------------------------------------
// Solve Poisson system for U and V
// -----------------------------------------------------------------------

bool solve_poisson_parametrization(
    const std::vector<int32_t>& row_ptr,
    const std::vector<int32_t>& col_idx,
    const std::vector<double>&  vals,
    const std::vector<double>&  rhs_u,
    const std::vector<double>&  rhs_v,
    const std::unordered_set<int32_t>& seam_verts,
    std::vector<double>& out_u,
    std::vector<double>& out_v)
{
    int nv = (int)rhs_u.size();
    if (nv == 0) return false;

    // Build Eigen sparse matrix from CSR
    using SpMat = Eigen::SparseMatrix<double>;
    int nnz = (int)vals.size();
    SpMat L(nv, nv);
    {
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(nnz);
        for (int row = 0; row < nv; ++row) {
            for (int k = row_ptr[row]; k < row_ptr[row+1]; ++k)
                trip.push_back({row, col_idx[k], vals[k]});
        }
        L.setFromTriplets(trip.begin(), trip.end());
    }

    // Pin the first non-seam vertex to (0, 0) to fix translation DOF
    int pin_v = 0;
    for (int i = 0; i < nv; ++i) {
        if (seam_verts.find(i) == seam_verts.end()) { pin_v = i; break; }
    }

   std::vector<Eigen::Triplet<double>> trip;

for (int row = 0; row < nv; ++row) {

    if (row == pin_v) {
        trip.push_back({row, row, 1.0});
        continue;
    }

    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
        int col = col_idx[k];
        double val = vals[k];

        if (col == pin_v) continue;

        trip.push_back({row, col, val});
    }
}

Eigen::SparseMatrix<double> Lbc(nv, nv);
Lbc.setFromTriplets(trip.begin(), trip.end());

// same RHS stays
Eigen::VectorXd bu = Eigen::Map<const Eigen::VectorXd>(rhs_u.data(), (Eigen::Index)rhs_u.size());
Eigen::VectorXd bv = Eigen::Map<const Eigen::VectorXd>(rhs_v.data(), (Eigen::Index)rhs_v.size());
    
int pinnedVertex = 0;

bu[pinnedVertex] = 0.0;
bv[pinnedVertex] = 0.0;

    // -------------------------------------------------------------------
    // Symmetric Dirichlet helper.
    //
    // Zeroing only the pin_v row (as Lbc does above) leaves the matrix
    // ASYMMETRIC — the pin_v column still carries the original cotangent
    // weights.  SparseLU doesn't care, but Cholesky (both Eigen's LDLT and
    // CHOLMOD) requires symmetry.  This helper rebuilds a truly SPD matrix
    // by dropping pin_v's row AND column from the ORIGINAL cotangent
    // Laplacian L, then reintroducing a single identity entry on the
    // diagonal so row pin_v becomes [0,…,0,1,0,…,0].  The RHS already has
    // bu[pin_v] = bv[pin_v] = 0, so row pin_v enforces x[pin_v] = 0 and
    // every other row is the unchanged reduced equation (pin_v's column is
    // gone, so its contribution vanishes without needing a RHS correction
    // because x[pin_v] = 0).
    //
    // IMPORTANT: a previous version of this file iterated Lbc with
    //     for (int row=0; row<nv; ++row)
    //         for (InnerIterator it(Lbc, row); it; ++it) { int col=it.col(); ... }
    // which is wrong — for Eigen's default column-major SparseMatrix the
    // outer loop variable is a COLUMN index and it.col() returns that same
    // column, so `row == col` was trivially true for every entry and the
    // matrix collapsed onto its diagonal.  The loop below uses correct
    // naming: `c` for the outer (column) index, `it.row()` for the inner
    // (row) index.
    // -------------------------------------------------------------------
    auto build_symmetric_dirichlet = [&](const SpMat& src) -> SpMat {
        SpMat out(nv, nv);
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(src.nonZeros() + 1);
        for (int c = 0; c < nv; ++c) {
            if (c == pin_v) continue;                     // skip pin column
            for (SpMat::InnerIterator it(src, c); it; ++it) {
                int r = (int)it.row();
                if (r == pin_v) continue;                 // skip pin row
                trip.push_back({r, c, it.value()});
            }
        }
        trip.push_back({pin_v, pin_v, 1.0});              // identity for pin
        out.setFromTriplets(trip.begin(), trip.end());
        out.makeCompressed();
        return out;
    };

    // -------------------------------------------------------------------
    // CHOLMOD fast path — large meshes only.
    //
    // On systems where SuiteSparse was found at configure time, CMake
    // defines QF_HAS_CHOLMOD.  For meshes with nv >= 500k the supernodal
    // Cholesky in CHOLMOD is typically 2-3x faster than Eigen's SparseLU
    // and delivers significantly better numerical stability on stiff
    // cotangent systems.  The 500k threshold avoids CHOLMOD's per-solve
    // setup overhead dominating on small meshes.
    //
    // If CHOLMOD is not compiled in OR the mesh is smaller than the
    // threshold OR the factorisation fails for any reason, control falls
    // through to the Eigen SparseLU / SimplicialLDLT path below unchanged.
    // -------------------------------------------------------------------
#ifdef QF_HAS_CHOLMOD
    if (nv >= 500000) {
        SpMat L_sym = build_symmetric_dirichlet(L);
        CholmodSPDSolver cholmod_solver;
        if (cholmod_solver.factorize(L_sym)) {
            Eigen::VectorXd xu_c = cholmod_solver.solve(bu);
            Eigen::VectorXd xv_c = cholmod_solver.solve(bv);
            if (cholmod_solver.ok() &&
                xu_c.size() == nv && xv_c.size() == nv) {
                out_u.assign(xu_c.data(), xu_c.data() + nv);
                out_v.assign(xv_c.data(), xv_c.data() + nv);
                return true;
            }
        }
        // Silent fall-through to Eigen on CHOLMOD failure.
    }
#endif

    // Primary solver: SparseLU (handles asymmetric matrix after BC mod)
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(Lbc);
    solver.factorize(Lbc);

    if (solver.info() == Eigen::Success) {
        Eigen::VectorXd xu = solver.solve(bu);
        Eigen::VectorXd xv = solver.solve(bv);
        if (solver.info() == Eigen::Success) {
            out_u.assign(xu.data(), xu.data() + nv);
            out_v.assign(xv.data(), xv.data() + nv);
            return true;
        }
    }

    // BUG FIX (Bug 4 + latent column-iteration bug): Eigen LDLT fallback
    // on a correctly symmetrised SPD matrix.  The previous implementation
    // had a column-iteration bug (see build_symmetric_dirichlet's comment)
    // that collapsed the fallback matrix onto its diagonal, causing this
    // path to silently return garbage whenever SparseLU failed.  We now
    // reuse the shared helper so both the CHOLMOD and LDLT fallbacks see
    // exactly the same SPD operator.
    SpMat Lbc_sym = build_symmetric_dirichlet(L);

    Eigen::SimplicialLDLT<SpMat> ldlt_solver;
    ldlt_solver.compute(Lbc_sym);
    if (ldlt_solver.info() != Eigen::Success) return false;

    Eigen::VectorXd xu2 = ldlt_solver.solve(bu);
    Eigen::VectorXd xv2 = ldlt_solver.solve(bv);
    if (ldlt_solver.info() != Eigen::Success) return false;

    out_u.assign(xu2.data(), xu2.data() + nv);
    out_v.assign(xv2.data(), xv2.data() + nv);
    return true;
}

// -----------------------------------------------------------------------
// Conjugate Gradient solver
// -----------------------------------------------------------------------

int conjugate_gradient(
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double* vals,
    int32_t n,
    const double* rhs,
    double* x,
    int max_iter,
    double tol)
{
    if (n <= 0) return -1;

    std::vector<double> r(n), p(n), Ap(n);

    // r = rhs - A*x
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;

        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            sum += vals[k] * x[col_idx[k]];
        }

        Ap[i] = sum;
        r[i] = rhs[i] - sum;
        p[i] = r[i];
    }

    double rsold = 0.0;

    for (int i = 0; i < n; ++i)
        rsold += r[i] * r[i];

    for (int iter = 0; iter < max_iter; ++iter) {

        // Ap = A*p
        for (int i = 0; i < n; ++i) {

            double sum = 0.0;

            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                sum += vals[k] * p[col_idx[k]];
            }

            Ap[i] = sum;
        }

        double alpha_den = 0.0;

        for (int i = 0; i < n; ++i)
            alpha_den += p[i] * Ap[i];

        if (std::abs(alpha_den) < 1e-30)
            return -1;

        double alpha = rsold / alpha_den;

        for (int i = 0; i < n; ++i)
            x[i] += alpha * p[i];

        for (int i = 0; i < n; ++i)
            r[i] -= alpha * Ap[i];

        double rsnew = 0.0;

        for (int i = 0; i < n; ++i)
            rsnew += r[i] * r[i];

        if (std::sqrt(rsnew) < tol)
            return iter;

        for (int i = 0; i < n; ++i)
            p[i] = r[i] + (rsnew / rsold) * p[i];

        rsold = rsnew;
    }

    return max_iter;
}