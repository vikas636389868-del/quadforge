/**
 * dual_contour.cpp — Dual-contouring quad extraction (extraction_method = 2).
 *
 * Implements extract_quads_dual_contour() and solve_qef() declared in
 * include/quadforge/extract/dual_contour.h.
 *
 * IMPLEMENTATION NOTES
 * --------------------
 * The algorithm mirrors extract_quads_from_isolines() (isolines.cpp) for
 * the cell-enumeration and face-assembly phases, but replaces the linear
 * interpolation vertex placement with a per-cell Quadratic Error Function
 * (QEF) solve.
 *
 * Hermite data collection
 * ~~~~~~~~~~~~~~~~~~~~~~~
 * For each triangle face fi with UV coords (u0,v0), (u1,v1), (u2,v2):
 *
 *   For each integer u-value iu in [ceil(u_lo), floor(u_hi)]:
 *     Walk the three directed edges of the triangle.  If iu lies strictly
 *     between the U endpoints of an edge, record:
 *       pos  = lerp(p_a, p_b, t) where t = (iu - U[a]) / (U[b] - U[a])
 *       norm = face normal (in 3-D)
 *     These are the "V-direction" Hermite samples for the u-iso-line.
 *
 *   Same for integer v-values.
 *
 * QEF solve (per grid cell)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~
 * Given S Hermite samples { (p_i, n_i) }:
 *
 *   Build 3×3 matrix  A = Σ n_i n_i^T
 *   Build 3×1 vector  b = Σ (n_i · p_i) n_i
 *
 *   Solve: A x = b  (least-squares)
 *
 *   If the smallest singular value of A is below svd_tol * largest, the
 *   system is rank-deficient (e.g. flat region with parallel normals).
 *   Fall back to the centroid of the sample positions.
 *
 * Cell grid convention
 * ~~~~~~~~~~~~~~~~~~~~
 * Identical to isolines.cpp: cell(iu, iv) has corners at UV integers
 *   (iu,   iv),   (iu+1, iv),
 *   (iu+1, iv+1), (iu,   iv+1)
 * stored in slots 0..3.  We key cells with a 64-bit integer using the
 * same collision-free packing from the v24 bug fix:
 *   key = (uint32_t)(iu + 32768) << 32 | (uint32_t)(iv + 32768)
 *
 * Thread safety
 * ~~~~~~~~~~~~~
 * The current implementation is single-threaded for the Hermite-collection
 * and QEF-solve passes (which together are ≈ 1.5–2× iso-line cost).
 * The num_threads argument is reserved for a future OpenMP port over the
 * per-cell QEF loop.
 */

#include "../../include/quadforge/extract/dual_contour.h"
#include "../../include/quadforge/accel/omp_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace qf {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Encode (iu, iv) into a 64-bit cell key (same as isolines.cpp v24 fix). */
static inline int64_t make_cell_key(int iu, int iv) {
    return (static_cast<int64_t>(static_cast<uint32_t>(iu + 32768)) << 32)
         |  static_cast<int64_t>(static_cast<uint32_t>(iv + 32768));
}

/**
 * Linear interpolation of two 3-D positions.
 * Returns p_a + t * (p_b - p_a).
 */
static inline std::array<double,3> lerp3(
    const Vec3& a, const Vec3& b, double t)
{
    return { a[0] + t*(b[0]-a[0]),
             a[1] + t*(b[1]-a[1]),
             a[2] + t*(b[2]-a[2]) };
}

/**
 * Safe unit normal of a triangle face.
 * Returns (0,0,1) if the face is degenerate.
 */
static std::array<double,3> safe_face_normal(const Vec3& p0,
                                              const Vec3& p1,
                                              const Vec3& p2)
{
    Vec3 a = p1 - p0, b = p2 - p0;
    Vec3 n = a.cross(b);
    double len = n.norm();
    if (len < 1e-15) return {0.0, 0.0, 1.0};
    return { n[0]/len, n[1]/len, n[2]/len };
}

// ---------------------------------------------------------------------------
// solve_qef
// ---------------------------------------------------------------------------

std::array<double,3> solve_qef(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<std::array<double,3>>& normals,
    const std::array<double,3>&              fallback,
    double                                   svd_tol)
{
    const int S = static_cast<int>(positions.size());
    if (S == 0) return fallback;

    // Build the 3×3 ATA system (A = matrix of row-normals, b = A^T rhs).
    // Equation: (Σ nᵢnᵢᵀ) x = Σ (nᵢ·pᵢ) nᵢ
    double A[3][3] = {};
    double rhs[3]  = {};

    for (int i = 0; i < S; ++i) {
        const auto& n  = normals[i];
        const auto& p  = positions[i];
        double ndotp = n[0]*p[0] + n[1]*p[1] + n[2]*p[2];
        for (int r = 0; r < 3; ++r) {
            rhs[r]          += ndotp * n[r];
            for (int c = 0; c < 3; ++c)
                A[r][c] += n[r] * n[c];
        }
    }

    // SVD of a 3×3 symmetric positive-semidefinite matrix via Jacobi
    // iterations (no external dependency, handles rank-deficient cases).
    //
    // We decompose A = U D U^T, solve D U^T x = U^T b with singular-value
    // thresholding.  For a 3×3 matrix a two-sweep Jacobi is exact.

    // Copy A into a working array.
    double M[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            M[r][c] = A[r][c];

    // Accumulate U in identity.
    double U[3][3] = { {1,0,0},{0,1,0},{0,0,1} };

    // One-sided Jacobi: 6 off-diagonal pairs, repeated 10 times.
    for (int sweep = 0; sweep < 10; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p+1; q < 3; ++q) {
                double Mpq = M[p][q];
                if (std::abs(Mpq) < 1e-15) continue;
                double tau = (M[q][q] - M[p][p]) / (2.0 * Mpq);
                double t   = (tau >= 0.0)
                           ? 1.0 / (tau + std::sqrt(1.0 + tau*tau))
                           : 1.0 / (tau - std::sqrt(1.0 + tau*tau));
                double c   = 1.0 / std::sqrt(1.0 + t*t);
                double s   = t * c;
                // Apply the rotation J(p,q,θ) to M and U
                // M <- J^T M J
                double Mpp = M[p][p], Mqq = M[q][q];
                M[p][p] = c*c*Mpp - 2*s*c*Mpq + s*s*Mqq;
                M[q][q] = s*s*Mpp + 2*s*c*Mpq + c*c*Mqq;
                M[p][q] = M[q][p] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    if (r == p || r == q) continue;
                    double Mrp = M[r][p], Mrq = M[r][q];
                    M[r][p] = M[p][r] = c*Mrp - s*Mrq;
                    M[r][q] = M[q][r] = s*Mrp + c*Mrq;
                }
                for (int r = 0; r < 3; ++r) {
                    double Urp = U[r][p], Urq = U[r][q];
                    U[r][p] = c*Urp - s*Urq;
                    U[r][q] = s*Urp + c*Urq;
                }
            }
        }
    }

    // Diagonal singular values
    double sigma[3] = { M[0][0], M[1][1], M[2][2] };
    double max_sigma = std::max({std::abs(sigma[0]),
                                 std::abs(sigma[1]),
                                 std::abs(sigma[2])});

    // Compute U^T rhs
    double Ut_rhs[3] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Ut_rhs[r] += U[c][r] * rhs[c];  // U^T row r = U col r

    // Pseudo-inverse: divide by singular value if above threshold
    double thresh = svd_tol * max_sigma;
    double y[3] = {};
    int rank = 0;
    for (int r = 0; r < 3; ++r) {
        if (std::abs(sigma[r]) > thresh) {
            y[r] = Ut_rhs[r] / sigma[r];
            ++rank;
        }
    }

    if (rank == 0) return fallback;  // zero matrix — degenerate

    // x = U y
    std::array<double,3> x = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            x[r] += U[r][c] * y[c];

    return x;
}

// ---------------------------------------------------------------------------
// extract_quads_dual_contour
// ---------------------------------------------------------------------------

QuadMesh extract_quads_dual_contour(
    const HalfEdgeMesh& mesh,
    const UVParam&      uv,
    int                 num_threads)
{
    (void)num_threads;   // reserved for future OpenMP port

    QuadMesh result;
    const int nv = mesh.num_vertices();
    const int nf = mesh.num_faces();

    if (nv == 0 || nf == 0) return result;
    if (static_cast<int>(uv.U.size()) != nv ||
        static_cast<int>(uv.V.size()) != nv) return result;

    // ------------------------------------------------------------------
    // Phase 1: Collect Hermite samples per grid cell
    //
    // For each triangle and each pair of directed edges, find where an
    // integer iso-line crosses the edge and record the Hermite datum
    // (3-D position + face normal) into the cell(s) that border that
    // iso-crossing.
    // ------------------------------------------------------------------

    struct HermiteSample {
        std::array<double,3> pos;
        std::array<double,3> nrm;   // face normal at crossing
    };

    struct CellData {
        std::vector<HermiteSample> samples;
    };

    std::unordered_map<int64_t, CellData> cell_map;
    cell_map.reserve(static_cast<size_t>(nf) * 4);

    for (int fi = 0; fi < nf; ++fi) {
        auto [vi0, vi1, vi2] = mesh.face_vertices(fi);

        Vec3 p[3] = { mesh.vertex_pos(vi0),
                      mesh.vertex_pos(vi1),
                      mesh.vertex_pos(vi2) };

        double U[3] = { uv.U[vi0], uv.U[vi1], uv.U[vi2] };
        double V[3] = { uv.V[vi0], uv.V[vi1], uv.V[vi2] };

        auto fnrm = safe_face_normal(p[0], p[1], p[2]);

        // Helper: record a Hermite sample at position `hp` into every cell
        // that borders the integer (iu, iv) grid corner.
        auto record = [&](int iu, int iv, const std::array<double,3>& hp) {
            // A crossing at integer U = iu, V = iv borders 4 cells:
            //   cell(iu-1, iv-1), cell(iu, iv-1),
            //   cell(iu-1, iv),   cell(iu,   iv)
            // We record the sample into all of them so that each cell
            // accumulates samples from all its edges.
            for (int du = -1; du <= 0; ++du) {
                for (int dv = -1; dv <= 0; ++dv) {
                    int64_t key = make_cell_key(iu + du, iv + dv);
                    cell_map[key].samples.push_back({hp, fnrm});
                }
            }
        };

        // Walk each of the 3 directed edges (a→b) for U crossings and
        // V crossings.
        int edges[3][2] = { {0,1}, {1,2}, {2,0} };

        for (auto& e : edges) {
            int ia = e[0], ib = e[1];

            // --- U iso-crossings ---
            {
                double ua = U[ia], ub = U[ib];
                double va = V[ia], vb = V[ib];
                int iu_lo = static_cast<int>(std::ceil(std::min(ua,ub) + 1e-9));
                int iu_hi = static_cast<int>(std::floor(std::max(ua,ub) - 1e-9));
                for (int iu = iu_lo; iu <= iu_hi; ++iu) {
                    double t  = (iu - ua) / (ub - ua + 1e-300);
                    if (t < 0.0 || t > 1.0) continue;
                    double iv_real = va + t * (vb - va);
                    int    iv_floor = static_cast<int>(std::floor(iv_real));
                    auto   hp = lerp3(p[ia], p[ib], t);
                    // The sample is on the U iso-line at integer iu.
                    // Its V coordinate determines which row of cells it
                    // contributes to.
                    record(iu, iv_floor, hp);
                    // BUG FIX (v53): only record into the adjacent row if the
                    // rounded V integer is DIFFERENT from iv_floor.  Previously
                    // both record() calls could target the same cell group
                    // (when iv_real ≈ integer and round(iv_real) == iv_floor),
                    // adding the sample twice and biasing the QEF solve.
                    if (std::abs(iv_real - std::round(iv_real)) < 1e-6) {
                        int iv_round = static_cast<int>(std::round(iv_real));
                        if (iv_round != iv_floor)
                            record(iu, iv_round, hp);
                    }
                }
            }

            // --- V iso-crossings ---
            {
                double ua = U[ia], ub = U[ib];
                double va = V[ia], vb = V[ib];
                int iv_lo = static_cast<int>(std::ceil(std::min(va,vb) + 1e-9));
                int iv_hi = static_cast<int>(std::floor(std::max(va,vb) - 1e-9));
                for (int iv = iv_lo; iv <= iv_hi; ++iv) {
                    double t  = (iv - va) / (vb - va + 1e-300);
                    if (t < 0.0 || t > 1.0) continue;
                    double iu_real = ua + t * (ub - ua);
                    int    iu_floor = static_cast<int>(std::floor(iu_real));
                    auto   hp = lerp3(p[ia], p[ib], t);
                    record(iu_floor, iv, hp);
                    // BUG FIX (v53): only record into the adjacent column if
                    // the rounded U integer differs from iu_floor (same fix as
                    // the U iso-crossings path above).
                    if (std::abs(iu_real - std::round(iu_real)) < 1e-6) {
                        int iu_round = static_cast<int>(std::round(iu_real));
                        if (iu_round != iu_floor)
                            record(iu_round, iv, hp);
                    }
                }
            }
        }
    }

    if (cell_map.empty()) return result;

    // ------------------------------------------------------------------
    // Phase 2: Solve QEF per cell → one vertex per cell
    //
    // The output vertex for cell (iu, iv) is stored at index = sequential
    // order of iteration over cell_map.  We build a parallel index map
    // cell_key → vertex_index.
    // ------------------------------------------------------------------

    std::unordered_map<int64_t, int> cell_to_vert;
    cell_to_vert.reserve(cell_map.size());

    result.vertices.reserve(cell_map.size());

    for (auto& [key, cd] : cell_map) {
        // Compute centroid as fallback / mass-point
        std::array<double,3> centroid = {0.0, 0.0, 0.0};
        for (auto& s : cd.samples) {
            centroid[0] += s.pos[0];
            centroid[1] += s.pos[1];
            centroid[2] += s.pos[2];
        }
        double inv = 1.0 / static_cast<double>(cd.samples.size());
        centroid[0] *= inv; centroid[1] *= inv; centroid[2] *= inv;

        std::vector<std::array<double,3>> spos, snrm;
        spos.reserve(cd.samples.size());
        snrm.reserve(cd.samples.size());
        for (auto& s : cd.samples) { spos.push_back(s.pos); snrm.push_back(s.nrm); }

        std::array<double,3> vpos = solve_qef(spos, snrm, centroid);

        // Clamp to bounding box of sample positions to prevent large
        // extrapolations that could produce inverted faces.
        double lo[3] = { 1e30, 1e30, 1e30 };
        double hi[3] = {-1e30,-1e30,-1e30 };
        for (auto& s : spos) {
            for (int d = 0; d < 3; ++d) {
                lo[d] = std::min(lo[d], s[d]);
                hi[d] = std::max(hi[d], s[d]);
            }
        }
        constexpr double PAD = 2.0;  // Allow QEF to extrapolate slightly
        for (int d = 0; d < 3; ++d) {
            double slack = (hi[d] - lo[d]) * PAD;
            vpos[d] = std::max(lo[d] - slack,
                      std::min(hi[d] + slack, vpos[d]));
        }

        int vi = static_cast<int>(result.vertices.size());
        result.vertices.push_back(vpos);
        cell_to_vert[key] = vi;
    }

    // ------------------------------------------------------------------
    // Phase 3: Assemble faces
    //
    // Each group of four adjacent cells
    //   (iu,   iv), (iu+1, iv), (iu+1, iv+1), (iu, iv+1)
    // forms a quad whose corners are the four QEF vertices of those cells.
    //
    // We iterate over all cells and attempt to form quads where all four
    // neighbours exist.  Cells without a complete neighbourhood become
    // triangles (degenerate boundary patches near singularities).
    // ------------------------------------------------------------------

    // We need the set of all valid cell keys to look up neighbours.
    // cell_to_vert already has all valid keys — use find() for membership.

    for (auto& [key, vi0] : cell_to_vert) {
        // Decode (iu, iv) from the key
        uint32_t uk = static_cast<uint32_t>((key >> 32) & 0xFFFFFFFF);
        uint32_t vk = static_cast<uint32_t>( key        & 0xFFFFFFFF);
        int iu = static_cast<int>(uk) - 32768;
        int iv = static_cast<int>(vk) - 32768;

        // Attempt to form the quad:
        //   cell(iu,   iv)   → vi0  (this cell)
        //   cell(iu+1, iv)   → vi1
        //   cell(iu+1, iv+1) → vi2
        //   cell(iu,   iv+1) → vi3
        auto it1 = cell_to_vert.find(make_cell_key(iu+1, iv  ));
        auto it2 = cell_to_vert.find(make_cell_key(iu+1, iv+1));
        auto it3 = cell_to_vert.find(make_cell_key(iu,   iv+1));

        bool have1 = it1 != cell_to_vert.end();
        bool have2 = it2 != cell_to_vert.end();
        bool have3 = it3 != cell_to_vert.end();

        if (have1 && have2 && have3) {
            int vi1 = it1->second;
            int vi2 = it2->second;
            int vi3 = it3->second;
            // Degenerate quad check
            if (vi0 != vi1 && vi0 != vi2 && vi0 != vi3 &&
                vi1 != vi2 && vi1 != vi3 && vi2 != vi3)
            {
                result.quads.push_back({ vi0, vi1, vi2, vi3 });
            }
        } else {
            // Partial cell: emit triangle(s) for boundary / singularity
            // Only emit a triangle if exactly 3 of the 4 corners exist.
            int count = 1 + (have1?1:0) + (have2?1:0) + (have3?1:0);
            if (count == 3) {
                std::array<int,3> tri;
                int ti = 0;
                tri[ti++] = vi0;
                if (have1) tri[ti++] = it1->second;
                if (have2) tri[ti++] = it2->second;
                if (have3) tri[ti++] = it3->second;
                if (tri[0]!=tri[1] && tri[0]!=tri[2] && tri[1]!=tri[2])
                    result.tris.push_back({tri[0], tri[1], tri[2]});
            }
        }
    }

    // Stats
    int total = static_cast<int>(result.quads.size() + result.tris.size());
    result.quad_percentage = total > 0
        ? 100.f * static_cast<float>(result.quads.size()) / total
        : 0.f;

    // BUG FIX (v92): avg_valence was never set; mirroring the fix applied
    // to extract_quads_from_isolines() in isolines.cpp.
    {
        int dc_nv = static_cast<int>(result.vertices.size());
        if (dc_nv > 0) {
            std::vector<int> val(dc_nv, 0);
            for (auto& q : result.quads)
                for (int v : q) if (v >= 0 && v < dc_nv) val[v]++;
            for (auto& t : result.tris)
                for (int v : t) if (v >= 0 && v < dc_nv) val[v]++;
            double sum = 0.0;
            for (int v : val) sum += v;
            result.avg_valence = static_cast<float>(sum / dc_nv);
        }
    }

    return result;
}

} // namespace qf
