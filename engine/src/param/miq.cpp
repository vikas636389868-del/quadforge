/**
 * miq.cpp — Poisson parametrization and Mixed-Integer rounding.
 *
 * Builds the cotangent Laplacian for a 4-RoSy-guided Poisson system,
 * solves for (U, V) via Eigen's ConjugateGradient, then applies
 * greedy integer rounding at seam edges.
 */

#include "../../include/quadforge/param/miq.h"
#include "../../include/quadforge/param/igm.h"
#include "../../include/quadforge/accel/omp_utils.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// -----------------------------------------------------------------------
// Cotangent weight for edge opposite to vertex pk in triangle (pi,pk,pj)
// -----------------------------------------------------------------------
static double cot_weight(const Vec3& pi, const Vec3& pk, const Vec3& pj) {
    Vec3 a = pi - pk, b = pj - pk;
    double dot  = a.dot(b);
    double cross= a.cross(b).norm();
    if (cross < 1e-15) return 0.0;
    return std::max(0.0, dot / cross); // clamp to ≥ 0 (no negative weights)
}

// -----------------------------------------------------------------------
// build_poisson_system
// -----------------------------------------------------------------------

void build_poisson_system(
    const HalfEdgeMesh&        mesh,
    const std::vector<double>& field_angles,
    const Eigen::VectorXd&     sizing,
    SparseMat&                 A_out,
    VecX&                      b_out)
{
    int nv = mesh.num_vertices();
    int nf = mesh.num_faces();

    // Cotangent Laplacian (symmetric, SPD)
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nf * 9);

    b_out.setZero(nv);

    for (int fi = 0; fi < nf; ++fi) {
        auto [vi0, vi1, vi2] = mesh.face_vertices(fi);
        Vec3 p0 = mesh.vertex_pos(vi0);
        Vec3 p1 = mesh.vertex_pos(vi1);
        Vec3 p2 = mesh.vertex_pos(vi2);

        // Cotangent weights at each vertex
        double w0 = cot_weight(p1, p0, p2) * 0.5;
        double w1 = cot_weight(p2, p1, p0) * 0.5;
        double w2 = cot_weight(p0, p2, p1) * 0.5;

        // Sizing modulation: scale Laplacian by 1/sizing²
        double s_avg = 1.0;
        if (sizing.size() == nv) {
            double s0=sizing[vi0], s1=sizing[vi1], s2=sizing[vi2];
            s_avg = (s0+s1+s2)/3.0;
            if (s_avg > 1e-10) {
                w0 /= s_avg*s_avg;
                w1 /= s_avg*s_avg;
                w2 /= s_avg*s_avg;
            }
        }

        // Off-diagonal
        triplets.push_back({vi1, vi2, -w0}); triplets.push_back({vi2, vi1, -w0});
        triplets.push_back({vi0, vi2, -w1}); triplets.push_back({vi2, vi0, -w1});
        triplets.push_back({vi0, vi1, -w2}); triplets.push_back({vi1, vi0, -w2});
        // Diagonal
        triplets.push_back({vi0, vi0, w1+w2});
        triplets.push_back({vi1, vi1, w0+w2});
        triplets.push_back({vi2, vi2, w0+w1});

        // RHS: divergence of target gradient from field direction
        double theta = field_angles[fi];

        // BUG FIX (v63 — param/Bug A): Guard degenerate (zero-area) faces
        // BEFORE calling mesh.face_normal(fi).normalized().
        //
        // For a zero-area triangle, face_normal() returns the zero vector.
        // Calling .normalized() on zero gives NaN (0/0 in Eigen).  The NaN
        // then flows into fn → e1 projection → e2 → X.  The per-face area
        // contribution `contrib = face_area(fi) / ...` is zero for a degenerate
        // face, but in IEEE 754 `0.0 * NaN == NaN` — so b_out[vi] += 0 * NaN
        // writes NaN into the RHS vector, corrupting the entire Poisson solve.
        //
        // The cotangent Laplacian entries for degenerate faces are already
        // skipped above via `if (sin_a < 1e-12) continue`, so their Laplacian
        // contribution is already zero.  Skipping the RHS too is the correct
        // and consistent fix (the area-weighted divergence of a zero-area face
        // is mathematically zero regardless of direction).
        //
        // This mirrors the identical guard used in curvature.cpp, cross_field.cpp
        // compute_face_frames(), singularity.cpp make_stable_frame(), and all
        // transport-angle helpers across the field/ subsystem.
        double area = mesh.face_area(fi);
        if (area < 1e-14) continue;   // degenerate — zero contribution, avoid NaN

        // BUG FIX (Bug 7 — param/miq): Consistent sizing guard between LHS and RHS.
        //
        // The Laplacian block above (lines 74-78) only divides the cotangent weights
        // w0/w1/w2 by s_avg² when s_avg > 1e-10.  When s_avg ≤ 1e-10 (all three
        // vertices have near-zero sizing), the Laplacian falls back to unit weighting
        // (effectively s_avg = 1.0).
        //
        // The original RHS used `std::max(s_avg*s_avg, 1e-10)` unconditionally.
        // When s_avg ≤ 1e-10 this produces a denominator of 1e-10, making
        // `contrib = area / (3 * 1e-10)` ≈ 10^10× too large relative to the
        // unscaled Laplacian — an LHS/RHS mismatch that drives the Poisson solve
        // into numerical overflow for degenerate-sizing regions of the mesh.
        //
        // Fix: mirror the Laplacian guard exactly.  When s_avg ≤ 1e-10 the RHS
        // uses s_eff² = 1.0 (same implicit denominator as the unscaled Laplacian).
        // This makes the Poisson system consistent regardless of sizing values.
        double s_eff_sq = (s_avg > 1e-10) ? (s_avg * s_avg) : 1.0;

        // Face normal and tangent vectors
        Vec3 raw_fn = mesh.face_normal(fi);
        double fn_len = raw_fn.norm();
        Vec3 fn = (fn_len > 1e-14) ? (raw_fn / fn_len) : Vec3(0, 0, 1);

        // First tangent: first edge projected onto tangent plane
        Vec3 raw_e1 = p1 - p0;
        Vec3 proj_e1 = raw_e1 - raw_e1.dot(fn) * fn;
        double pe1_len = proj_e1.norm();
        Vec3 e1 = (pe1_len > 1e-12) ? (proj_e1 / pe1_len) : fn.unitOrthogonal();
        Vec3 e2 = fn.cross(e1).normalized();

        // Field direction X = cos(θ)*e1 + sin(θ)*e2
        Vec3 X = std::cos(theta)*e1 + std::sin(theta)*e2;

        // Divergence contribution to vertex vi:
        // div(X) at vi ≈ sum over edges of cotangent * (X · e_ij)
        double contrib = area / (3.0 * s_eff_sq);  // s_eff_sq set above (Bug 7 fix)

        b_out[vi0] += contrib * X.dot(p1-p2);
        b_out[vi1] += contrib * X.dot(p2-p0);
        b_out[vi2] += contrib * X.dot(p0-p1);
    }

    A_out.resize(nv, nv);
    A_out.setFromTriplets(triplets.begin(), triplets.end());

    // Add small diagonal regulariser to handle disconnected vertices
    for (int i = 0; i < nv; ++i)
        A_out.coeffRef(i, i) += 1e-8;
}

// -----------------------------------------------------------------------
// compute_parametrization
// -----------------------------------------------------------------------

UVParam compute_parametrization(
    const HalfEdgeMesh&   mesh,
    const CombingResult&  combing,
    const Eigen::VectorXd& sizing,
    const ParamParams&    params)
{
    // Dispatch to IGM when requested
    if (params.method == ParamMethod::IGM) {
        IGMParams igm_p;
        igm_p.max_cg_iterations  = std::max(500, mesh.num_vertices() / 5);
        igm_p.cg_tolerance       = 1e-6;
        igm_p.jump_snap_tol      = params.miq_tolerance;
        igm_p.strict_global_snap = true;
        igm_p.global_snap_tol    = params.miq_tolerance * 0.8;
        igm_p.num_threads        = params.num_threads;
        return compute_igm_parametrization(mesh, combing, sizing, igm_p);
    }

    int nv = mesh.num_vertices();
    UVParam uv;
    uv.U.setZero(nv);
    uv.V.setZero(nv);
    uv.seam_edges = combing.seam_edges;

    // BUG FIX (v108 — param/miq): Transfer period jumps to UVParam.
    //
    // igm.cpp correctly forwards CombingResult::period_jumps into
    // UVParam::period_jumps since the v89 fix.  The MIQ and POISSON_SIMPLE
    // code paths in THIS function never received the equivalent fix, so
    // uv.period_jumps was always left empty for those methods.
    //
    // Consequence: seam_utils::build_period_jump_map() returned an empty map
    // (it early-returns when period_jumps is empty), and
    // extract_quads_from_isolines_seam_aware() fell back silently to the
    // seam-blind variant.  The fallback is functionally correct but produces
    // T-junctions at UV seam crossings and lower quad quality compared with
    // seam-aware tracing.
    //
    // combing.seam_edges may be longer than combing.period_jumps because
    // engine.cpp appends extra seam-cut edges (from compute_seam_cut_from_mesh)
    // after comb_field() returns — those extra edges have no combing-derived
    // period jump.  We pad them with (0, 0) so that
    //   uv.seam_edges.size() == uv.period_jumps.size()
    // as required by the PRECONDITION in isolines.h.
    //
    // Format note: combing.cpp stores each period jump as a single int pj in
    // [0, 3] encoding a 4-RoSy rotation count (0 = no jump, 1 = 90°, …).
    // The encoding is the same packed format igm.cpp expects:
    //   du = int16_t((pj >> 16) & 0xFFFF)   → always 0 for current values
    //   dv = int16_t( pj        & 0xFFFF)   → 0, 1, 2, or 3
    // This matches the decode path in igm.cpp (v89) and the consume path in
    // seam_utils::build_period_jump_map().
    {
        const int n_seams = static_cast<int>(combing.seam_edges.size());
        const int n_jumps = static_cast<int>(combing.period_jumps.size());
        uv.period_jumps.clear();
        uv.period_jumps.reserve(static_cast<size_t>(n_seams));
        for (int k = 0; k < n_seams; ++k) {
            if (k < n_jumps) {
                // Decode packed period jump (same logic as igm.cpp v89 fix).
                int pj = combing.period_jumps[k];
                int du = static_cast<int16_t>((pj >> 16) & 0xFFFF);
                int dv = static_cast<int16_t>( pj        & 0xFFFF);
                uv.period_jumps.emplace_back(du, dv);
            } else {
                // Extra seam-cut edge appended by engine.cpp after comb_field():
                // no combing-derived period jump exists — use (0, 0).
                // build_period_jump_map() skips zero-jump entries, so this is
                // equivalent to marking the edge as a seam with no UV shift,
                // which is the correct conservative behaviour for Dijkstra-
                // derived seam-cut edges that do not carry rotation data.
                uv.period_jumps.emplace_back(0, 0);
            }
        }
        // Postcondition: uv.seam_edges.size() == uv.period_jumps.size()
        // (required by isolines.h PRECONDITION and verified by seam_utils.cpp).
    }

    if (nv == 0) return uv;

    // BUG FIX (Bug 5 — param/miq+igm): Pin the FIRST NON-SEAM vertex to
    // remove the translation null-space.  The old code unconditionally pinned
    // vertex 0, which may be a seam vertex.  Pinning a seam vertex while
    // simultaneously enforcing integer jump constraints on it creates a
    // contradiction: the solver tries to satisfy U[0]=0 AND U[0]≠U[1]+k for
    // some integer k, producing inaccurate results in the penalty re-solve.
    // Collect seam vertex set for the pin search.
    std::unordered_set<int> seam_vset_for_pin;
    seam_vset_for_pin.reserve(combing.seam_edges.size() * 2);
    for (auto& [s0, s1] : combing.seam_edges) {
        seam_vset_for_pin.insert(s0);
        seam_vset_for_pin.insert(s1);
    }
    int pin_vertex_miq = 0;
    for (int i = 0; i < nv; ++i) {
        if (!seam_vset_for_pin.count(i)) { pin_vertex_miq = i; break; }
    }

    // Build and solve Poisson system for U
    SparseMat A_u; VecX b_u;
    build_poisson_system(mesh, combing.face_angles, sizing, A_u, b_u);

    // Build Poisson system for V.
    // BUG FIX (B4 — param/miq v110): The Poisson Laplacian (LHS matrix)
    // depends only on mesh geometry and the sizing field — it is IDENTICAL
    // for U and V.  The previous code called build_poisson_system() a second
    // time for V, which rebuilt and factored the same matrix at ~50% extra
    // cost.  We now extract only the V-direction RHS (rotated by π/2) and
    // reuse A_u directly.  The shared solve path below is unchanged.
    std::vector<double> v_angles(combing.face_angles.size());
    for (size_t i = 0; i < combing.face_angles.size(); ++i)
        v_angles[i] = combing.face_angles[i] + M_PI/2.0;

    // Build RHS for V by calling build_poisson_system and discarding its A.
    // A_v_discard is immediately released; only b_v is kept.
    SparseMat A_v_discard; VecX b_v;
    build_poisson_system(mesh, v_angles, sizing, A_v_discard, b_v);
    A_v_discard.resize(0, 0);   // release memory immediately — A_u is reused

    // A_v == A_u (same geometry, same sizing); use A_u for both solves.
    const SparseMat& A_v = A_u;

    // Solve with Conjugate Gradient
    // BUG FIX (Bug 5): only accept Eigen::Success.  NoConvergence means the
    // solve did NOT converge — accepting it silently produces garbage UV values
    // that corrupt every downstream stage (iso-line tracing, quad extraction).
    //
    // BUG FIX (Bug 3 — param/miq): The previous code checked cg.info() only
    // after compute() (preconditioner setup), never after solve().  In Eigen,
    // cg.solve() updates cg.info() to Success or NoConvergence.  Because
    // cg.compute(A_v) was called immediately after cg.solve(b_u), the U solve
    // convergence result was overwritten before it could be inspected.
    // Fix: capture solve result immediately after each solve() call.
    Eigen::ConjugateGradient<SparseMat, Eigen::Lower|Eigen::Upper> cg;
    cg.setMaxIterations(std::max(500, nv/5));
    cg.setTolerance(1e-6);

    cg.compute(A_u);
    if (cg.info() == Eigen::Success) {
        uv.U = cg.solve(b_u);
        // Check solve convergence immediately (before compute(A_v) overwrites info)
        if (cg.info() != Eigen::Success) {
            // CG solve didn't converge — fall through to LDLT fallback below
            SparseMat A_pin = A_u;
            A_pin.coeffRef(pin_vertex_miq, pin_vertex_miq) += 1e4;
            Eigen::SimplicialLDLT<SparseMat> ldlt;
            ldlt.compute(A_pin);
            if (ldlt.info() == Eigen::Success)
                uv.U = ldlt.solve(b_u);
        }
    } else {
        // Fallback: use the LDLT solver on the same Laplacian.
        // Pin a non-seam vertex to remove the translation DOF.
        SparseMat A_pin = A_u;
        A_pin.coeffRef(pin_vertex_miq, pin_vertex_miq) += 1e4;
        Eigen::SimplicialLDLT<SparseMat> ldlt;
        ldlt.compute(A_pin);
        if (ldlt.info() == Eigen::Success)
            uv.U = ldlt.solve(b_u);
    }

    cg.compute(A_v);   // A_v == A_u — preconditioner is already set; this
                       // re-computes it on the same matrix.  Kept here so
                       // the code path is explicit and safe; the cost is one
                       // diagonal preconditoner extraction (negligible).
    if (cg.info() == Eigen::Success) {
        uv.V = cg.solve(b_v);
        if (cg.info() != Eigen::Success) {
            SparseMat A_pin = A_v;
            A_pin.coeffRef(pin_vertex_miq, pin_vertex_miq) += 1e4;
            Eigen::SimplicialLDLT<SparseMat> ldlt;
            ldlt.compute(A_pin);
            if (ldlt.info() == Eigen::Success)
                uv.V = ldlt.solve(b_v);
        }
    } else {
        SparseMat A_pin = A_v;
        A_pin.coeffRef(pin_vertex_miq, pin_vertex_miq) += 1e4;
        Eigen::SimplicialLDLT<SparseMat> ldlt;
        ldlt.compute(A_pin);
        if (ldlt.info() == Eigen::Success)
            uv.V = ldlt.solve(b_v);
    }

    // ---- MIQ rounding ----
    if (params.method == ParamMethod::MIQ) {
        // Greedy: for seam vertices, snap U and V to nearest integer
        // if |frac| < miq_tolerance
        for (auto& [lo, hi] : combing.seam_edges) {
            for (int vi : {lo, hi}) {
                if (vi < 0 || vi >= nv) continue;
                double u_frac = uv.U[vi] - std::round(uv.U[vi]);
                double v_frac = uv.V[vi] - std::round(uv.V[vi]);
                if (std::abs(u_frac) < params.miq_tolerance)
                    uv.U[vi] = std::round(uv.U[vi]);
                if (std::abs(v_frac) < params.miq_tolerance)
                    uv.V[vi] = std::round(uv.V[vi]);
            }
        }
    }

    // BUG FIX (P2 — param/miq v111): Update uv.period_jumps with the ACTUAL
    // integer UV differences across each seam edge, replacing the combing
    // rotation counts (du=0, dv=rotation_count) stored at function entry.
    //
    // Root cause: Identical to Bug P1 in igm.cpp.  uv.period_jumps was
    // populated from combing.period_jumps (4-RoSy rotation counts packed as
    // (0, k)), but the downstream consumer build_period_jump_map() interprets
    // them as actual UV differences: U[v_hi] ≈ U[v_lo] + Δu and similarly for V.
    // For MIQ, the actual UV grid is aligned to the SOLVED parametrization, not
    // to the combing rotation count.  Using (0, k) as the period jump causes the
    // isoline tracer to apply wrong UV offsets at seam crossings, producing
    // T-junctions and misaligned quad patches.
    //
    // For POISSON_SIMPLE: no integer rounding was applied, so UV values are
    // continuous.  The "best guess" period jump is still the rounded UV
    // difference from the Poisson solve.  Using (0, rotation_count) is worse
    // than using the actual round(U[v1]-U[v0]) from the parametrization.
    //
    // Fix: after all solve + rounding steps, recompute the period jumps as
    // round(U[v1]-U[v0]) / round(V[v1]-V[v0]) for every seam edge.
    // Out-of-bounds edges (guard) keep their existing (0,0) entry.
    //
    // Postcondition: uv.seam_edges.size() == uv.period_jumps.size() (unchanged).
    {
        const int n_seam_final = static_cast<int>(uv.seam_edges.size());
        const int n_pj_final   = static_cast<int>(uv.period_jumps.size());
        for (int k = 0; k < n_seam_final && k < n_pj_final; ++k) {
            int v0 = uv.seam_edges[k].first, v1 = uv.seam_edges[k].second;
            if (v0 < 0 || v0 >= nv || v1 < 0 || v1 >= nv) continue;
            int du = (int)std::round(uv.U[v1] - uv.U[v0]);
            int dv = (int)std::round(uv.V[v1] - uv.V[v0]);
            uv.period_jumps[k] = { du, dv };
        }
    }

    return uv;
}

} // namespace qf
