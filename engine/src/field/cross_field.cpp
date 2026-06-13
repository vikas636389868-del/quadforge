/**
 * cross_field.cpp — Knöppel 2013 globally optimal 4-RoSy direction field.
 *
 * FIX (v54+): Integrated symmetry constraints via build_symmetry_constraints()
 *             when CrossFieldParams::enforce_symmetry_x/y/z are set.
 * FIX (v54+): Cotangent weights in build_connection_laplacian() replace the
 *             coarser face-area weights, matching the Python field.py and the
 *             mathematical derivation in Knöppel 2013 eq. 3.
 */

#include "../../include/quadforge/field/cross_field.h"
#include "../../include/quadforge/field/constraints.h"   // FIX: symmetry constraints
#include "../../include/quadforge/field/smoothing.h"     // FIX (v96): post-solve smoothing
// INTEGRATION FIX (v82): Include connection.h so the compiler can see both
// overloads of qf::parallel_transport_angle in the same TU:
//   (a) parallel_transport_angle(mesh, he)  — HalfEdge-based, defined below for
//       the Knöppel solver's internal SparseMatC build path (fast, uses Eigen).
//   (b) parallel_transport_angle(e1_i, e2_i, n_i, ...) — flat-array overload
//       defined in connection.cpp, callable from qf_debug_field() in api.cpp.
// Without this include, the connection.cpp compilation unit compiled but was
// never reachable from any other TU — i.e. genuinely dead code.
// The two overloads are distinct signatures in namespace qf; no ambiguity.
#include "../../include/quadforge/field/connection.h"
#include "../../include/quadforge/accel/omp_utils.h"
// INTEGRATION FIX: cache_utils.h was not included — face_area[] is a hot
// per-face array iterated during connection Laplacian assembly.
#include "../../include/quadforge/accel/cache_utils.h"

#include <cmath>
#include <complex>
#include <unordered_map>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <array>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// -----------------------------------------------------------------------
// Per-face local orthonormal frames
// -----------------------------------------------------------------------

static void compute_face_frames(
    const HalfEdgeMesh& mesh,
    std::vector<Vec3>& e1_out,
    std::vector<Vec3>& e2_out)
{
    int nf = mesh.num_faces();
    e1_out.resize(nf);
    e2_out.resize(nf);

    for (int fi = 0; fi < nf; ++fi) {
        Vec3 n = mesh.face_normal(fi).normalized();
        // BUG FIX (v59): if the face is degenerate (face_normal() returns zero,
        // e.g. a zero-area sliver), n is NaN after .normalized().  Guard by
        // falling back to a canonical axis.  This mirrors the same guard that
        // every other frame-building helper in this file and in singularity.cpp,
        // smoothing.cpp, and field_debug.cpp already has.
        if (!std::isfinite(n[0]) || n.squaredNorm() < 0.5) {
            // Pick a stable fallback normal; unitOrthogonal() on a zero vector
            // is undefined so we use a fixed axis and the e1/e2 below will be
            // arbitrary but finite — far better than NaN propagating everywhere.
            n = Vec3(0, 0, 1);
        }

        auto [v0, v1, v2] = mesh.face_vertices(fi);

        // BUG FIX (v59): if v0 and v1 are coincident (degenerate edge),
        // (v1 - v0).normalized() is (0).normalized() which Eigen evaluates to
        // NaN (0/0).  That NaN then propagates through every dot/cross product
        // that uses e1/e2 for this face — silently corrupting the connection
        // Laplacian, all transport angles, and constraint projections.
        // Fix: mirror the guard used in every make_stable_frame() helper:
        // project, check the projected length, fall back to n.unitOrthogonal().
        Vec3 raw_edge = mesh.vertex_pos(v1) - mesh.vertex_pos(v0);
        Vec3 proj = raw_edge - raw_edge.dot(n) * n;
        double proj_len = proj.norm();

        Vec3 e1, e2;
        if (proj_len > 1e-12) {
            e1 = proj / proj_len;
        } else {
            // Degenerate: first edge is zero-length or parallel to normal —
            // fall back to any axis perpendicular to n.
            e1 = n.unitOrthogonal();
        }
        e2 = n.cross(e1).normalized();

        e1_out[fi] = e1;
        e2_out[fi] = e2;
    }
}

// -----------------------------------------------------------------------
// Parallel transport angle between adjacent faces
// -----------------------------------------------------------------------

double parallel_transport_angle(const HalfEdgeMesh& mesh, int he) {
    int tw = mesh.half_edge(he).twin;
    if (tw < 0) return 0.0;

    int fi = mesh.half_edge(he).face;
    int fj = mesh.half_edge(tw).face;
    if (fi < 0 || fj < 0) return 0.0;

    // BUG FIX (v61): Guard against degenerate (zero-area) faces.
    // face_normal() returns the zero vector for zero-area triangles; calling
    // .normalized() on it produces NaN (0/0 in Eigen).  NaN then flows into the
    // make_stable_frame projection, bypasses the proj.norm() fallback guard
    // (NaN < 1e-12 evaluates FALSE), and propagates into e1_i/e2_i/e1_j/e2_j
    // and the final phi — corrupting every entry of the connection Laplacian that
    // involves the degenerate face.
    // Fix: measure raw normal lengths first; return 0.0 (zero transport ≡
    // coplanar faces) for any degenerate face.  Mirrors the guard in curvature.cpp
    // and the degenerate-edge guards added in v60.
    {
        Vec3 raw_ni = mesh.face_normal(fi);
        Vec3 raw_nj = mesh.face_normal(fj);
        double ni_len = raw_ni.norm(), nj_len = raw_nj.norm();
        if (ni_len < 1e-14 || nj_len < 1e-14) return 0.0;
    }
    Vec3 ni = mesh.face_normal(fi).normalized();
    Vec3 nj = mesh.face_normal(fj).normalized();

    // BUG FIX (v25): build stable per-face frames from each face's OWN first
    // edge (v0→v1), NOT from the shared edge.  The previous code used the
    // shared edge as the reference for both frames, so atan2(edge·e2, edge·e1)
    // was trivially ≈0 for both faces, making the transport angle always ≈0
    // and collapsing the connection Laplacian to an ordinary Laplacian.
    auto make_stable_frame = [](const Vec3& n,
                                const Vec3& ref_edge,  // face's own v0→v1
                                Vec3& e1, Vec3& e2)
    {
        Vec3 proj = ref_edge - ref_edge.dot(n) * n;
        double len = proj.norm();
        if (len < 1e-12) {
            // Degenerate: pick any perpendicular
            e1 = n.unitOrthogonal();
        } else {
            e1 = proj / len;
        }
        e2 = n.cross(e1).normalized();
    };

    auto [vi0, vi1, vi2] = mesh.face_vertices(fi);
    auto [vj0, vj1, vj2] = mesh.face_vertices(fj);

    // BUG FIX (v60): Do NOT pre-normalize ref_i / ref_j before passing them to
    // make_stable_frame.  If the face edge is zero-length (vi0 == vi1 in position),
    // .normalized() returns NaN (0/0 in Eigen), and NaN < 1e-12 evaluates FALSE,
    // so the fallback guard inside make_stable_frame is silently bypassed and NaN
    // propagates into the frame vectors, the transport angle, and the Laplacian.
    // Passing the raw edge is safe: make_stable_frame projects onto the tangent
    // plane, measures the length of the result, and falls back to
    // n.unitOrthogonal() when that length is < 1e-12.
    Vec3 ref_i = mesh.vertex_pos(vi1) - mesh.vertex_pos(vi0);
    Vec3 ref_j = mesh.vertex_pos(vj1) - mesh.vertex_pos(vj0);

    Vec3 e1_i, e2_i, e1_j, e2_j;
    make_stable_frame(ni, ref_i, e1_i, e2_i);
    make_stable_frame(nj, ref_j, e1_j, e2_j);

    // Shared edge direction (normalised, same for both faces)
    // BUG FIX (v60): Do NOT call .normalized() directly on the shared edge.
    // If the two endpoints are coincident (degenerate mesh), the raw vector is
    // zero and .normalized() returns NaN (0/0 in Eigen).  NaN then propagates
    // into ai, aj, and phi — which silently corrupts every Laplacian entry that
    // references this edge pair.  Guard: compute the length explicitly and
    // return 0.0 (zero transport = co-planar faces) for degenerate edges.
    int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
    int to   = mesh.half_edge(he).vertex;
    Vec3 raw_shared = mesh.vertex_pos(to) - mesh.vertex_pos(from);
    double shared_len = raw_shared.norm();
    if (shared_len < 1e-12) return 0.0;
    Vec3 edge = raw_shared / shared_len;

    // Angle of the shared edge in each face's stable frame
    double ai = std::atan2(edge.dot(e2_i), edge.dot(e1_i));
    double aj = std::atan2(edge.dot(e2_j), edge.dot(e1_j));
    double phi = aj - ai;

    // Wrap to [-π/4, π/4) for 4-RoSy symmetry
    while (phi >  M_PI / 4.0) phi -= M_PI / 2.0;
    while (phi < -M_PI / 4.0) phi += M_PI / 2.0;
    return phi;
}

// -----------------------------------------------------------------------
// Build the connection Laplacian (SparseMatC)
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// Cotangent of the angle at 'apex' in triangle (apex, b, c).
// Clamped to [-20, 20] to handle near-degenerate triangles safely.
// -----------------------------------------------------------------------
static double cot_angle_cf(const Vec3& apex, const Vec3& b, const Vec3& c)
{
    Vec3 u = b - apex;
    Vec3 v = c - apex;
    double num = u.dot(v);
    double den = u.cross(v).norm();
    if (den < 1e-14) return 0.0;
    return std::max(-20.0, std::min(20.0, num / den));
}

SparseMatC build_connection_laplacian(
    const HalfEdgeMesh& mesh,
    const std::vector<double>& /*face_frames*/)
{
    int nf = mesh.num_faces();
    SparseMatC L(nf, nf);
    std::vector<TripletC> triplets;
    triplets.reserve(nf * 6);

    // FIX (v54+): Use cotangent weights instead of face-area weights.
    // Cotangent weights are the mathematically correct weights from
    // Knöppel 2013 eq. 3: w_ij = (cot α_ij + cot β_ij) / 2
    // where α_ij and β_ij are the angles opposite the shared edge in the
    // two adjacent triangles.  They produce significantly smoother fields
    // because they arise directly from the discretisation of the Dirichlet
    // energy on a triangle mesh (Meyer et al. 2003 "Discrete Differential-
    // Geometry Operators for Triangulated 2-Manifolds").
    //
    // We retain a positive floor (1e-10) so that near-degenerate triangles
    // (cot can be negative for obtuse angles) still contribute a small
    // positive stabilising weight rather than destabilising the Laplacian.

    int nhe = mesh.num_half_edges();
    for (int he = 0; he < nhe; ++he) {
        int tw = mesh.half_edge(he).twin;
        if (tw < 0) continue;
        if (tw < he) continue; // each interior edge processed once

        int fi = mesh.half_edge(he).face;
        int fj = mesh.half_edge(tw).face;
        if (fi < 0 || fj < 0) continue;

        // Cotangent of the angle opposite this edge in each triangle.
        // The opposite vertex is the destination of the NEXT half-edge.
        int v_from  = mesh.half_edge(mesh.half_edge(he).prev).vertex;
        int v_to    = mesh.half_edge(he).vertex;
        int v_opp_i = mesh.half_edge(mesh.half_edge(he).next).vertex;
        int v_opp_j = mesh.half_edge(mesh.half_edge(tw).next).vertex;

        Vec3 p_from  = mesh.vertex_pos(v_from);
        Vec3 p_to    = mesh.vertex_pos(v_to);
        Vec3 p_opp_i = mesh.vertex_pos(v_opp_i);
        Vec3 p_opp_j = mesh.vertex_pos(v_opp_j);

        double cot_i = cot_angle_cf(p_opp_i, p_from, p_to);
        double cot_j = cot_angle_cf(p_opp_j, p_from, p_to);
        double w = std::max(0.5 * (cot_i + cot_j), 1e-10);

        double phi = parallel_transport_angle(mesh, he);
        std::complex<double> transport = std::exp(std::complex<double>(0, 4.0 * phi));

        triplets.push_back({fi, fi, std::complex<double>(w, 0)});
        triplets.push_back({fj, fj, std::complex<double>(w, 0)});
        triplets.push_back({fi, fj, -w * std::conj(transport)});
        triplets.push_back({fj, fi, -w * transport});
    }

    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

// -----------------------------------------------------------------------
// Curvature-only fallback field
// -----------------------------------------------------------------------

static CrossField curvature_aligned_field(
    const HalfEdgeMesh&  mesh,
    const CurvatureData& curvature)
{
    int nf = mesh.num_faces();
    CrossField field;
    field.face_field.resize(nf);
    field.face_frames.assign(nf, 0.0);

    std::vector<Vec3> e1, e2;
    compute_face_frames(mesh, e1, e2);

    for (int fi = 0; fi < nf; ++fi) {
        auto [v0, v1, v2] = mesh.face_vertices(fi);
        // Average principal curvature direction across face vertices.
        // BUG FIX (v56): On flat regions curvature.dir1 rows can be near-zero,
        // so the sum may be a near-zero vector.  Calling .normalized() on it
        // produces NaN in Eigen (division by ~0), which then propagates into
        // atan2(NaN, NaN) → NaN field values and a corrupted cross-field.
        // Guard: if the summed direction is too short fall back to the face's
        // own local e1 axis (tangent to the first edge), which is always valid.
        Vec3 dir_sum = (curvature.dir1.row(v0) +
                        curvature.dir1.row(v1) +
                        curvature.dir1.row(v2)).transpose();
        double dir_len = dir_sum.norm();
        Vec3 dir = (dir_len > 1e-12) ? (dir_sum / dir_len) : e1[fi];

        double theta = std::atan2(dir.dot(e2[fi]), dir.dot(e1[fi]));
        field.face_field[fi] = std::exp(std::complex<double>(0, 4.0 * theta));
        field.face_frames[fi] = theta;
    }

    return field;
}

// -----------------------------------------------------------------------
// Power iteration (smallest eigenvector)
// -----------------------------------------------------------------------

static CrossField knoppel_field(
    const HalfEdgeMesh&   mesh,
    const FeatureData&    features,
    const CrossFieldParams& params,
    bool do_post_smooth = true)
{
    int nf = mesh.num_faces();
    CrossField field;
    field.face_frames.assign(nf, 0.0);

    SparseMatC L = build_connection_laplacian(mesh, field.face_frames);

    // -----------------------------------------------------------------------
    // Feature-edge constraints: force field arms to align with hard edges.
    // We use the Dirichlet-penalty approach:
    //   - Add params.constraint_weight to L[f,f] for each constrained face f
    //   - Track constrained (face → target complex) for projection during iteration
    // -----------------------------------------------------------------------
    std::vector<Vec3> e1, e2;
    compute_face_frames(mesh, e1, e2);

    // Build edge → face adjacency from the half-edge structure for O(1) lookup
    // Key: min(va,vb)*VMAX + max(va,vb) → {face_i, face_j}
    using EKey = int64_t;
    const int64_t VMAX = 1LL << 24;
    std::unordered_map<EKey, std::array<int,2>> edge_to_face;
    edge_to_face.reserve(nf * 2);
    {
        int nhe = mesh.num_half_edges();
        for (int he = 0; he < nhe; ++he) {
            int tw = mesh.half_edge(he).twin;
            if (tw < 0) continue;
            if (tw < he) continue;
            int fi = mesh.half_edge(he).face;
            int fj = mesh.half_edge(tw).face;
            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;
            EKey key = (EKey)std::min(from,to) * VMAX + std::max(from,to);
            edge_to_face[key] = {fi, fj};
        }
    }

    // Per-face constraint target (unit complex), empty = unconstrained
    std::vector<std::complex<double>> constraint_target(nf, {0.0, 0.0});
    std::vector<bool> is_constrained(nf, false);

    if (params.align_to_features && !features.hard_edges.empty()) {
        std::vector<TripletC> penalty_triplets;
        penalty_triplets.reserve(features.hard_edges.size() * 4);

        for (auto& [lo, hi] : features.hard_edges) {
            EKey key = (EKey)std::min(lo,hi) * VMAX + std::max(lo,hi);
            auto it = edge_to_face.find(key);
            if (it == edge_to_face.end()) continue;

            // Edge direction (3-D, normalized)
            // BUG FIX (v60): pre-normalizing a zero-length feature edge (coincident
            // endpoints) returns NaN, making theta = atan2(NaN, NaN) = NaN and
            // poisoning constraint_target[f] and the power-iteration solver.
            // Guard: compute the length first and skip the edge when degenerate.
            Vec3 raw_feat = mesh.vertex_pos(hi) - mesh.vertex_pos(lo);
            double feat_len = raw_feat.norm();
            if (feat_len < 1e-12) continue;
            Vec3 edge = raw_feat / feat_len;

            for (int f : it->second) {
                if (f < 0 || f >= nf) continue;

                // Project edge onto face tangent frame
                double theta = std::atan2(edge.dot(e2[f]), edge.dot(e1[f]));
                std::complex<double> target = std::exp(std::complex<double>(0, 4.0 * theta));

                // Add diagonal penalty term — pulls eigenvector toward target
                penalty_triplets.push_back({f, f,
                    std::complex<double>(params.constraint_weight, 0.0)});

                // BUG FIX (v59): When face f is adjacent to two or more feature
                // edges (e.g. a corner where two hard edges meet), the inner loop
                // over it->second processes both faces, and the outer loop over
                // features.hard_edges may also visit face f again via the second
                // feature edge.  Simple assignment
                //   constraint_target[f] = target
                // silently overwrites the earlier target.  The Laplacian correctly
                // accumulates all penalties (multiple pushbacks to penalty_triplets),
                // but the power-iteration projection only enforces the last stored
                // target — undoing the contribution of every earlier hard edge.
                //
                // Fix: circularly-average new targets into the existing stored target
                // (same strategy as the identical fix applied to smoothing.cpp in v58):
                // sum the unit-complex targets and re-normalise to get the geodesic
                // midpoint on the 4-RoSy unit circle.
                if (!is_constrained[f]) {
                    constraint_target[f] = target;
                    is_constrained[f]    = true;
                } else {
                    // Face already constrained — blend the two unit-complex targets
                    std::complex<double> combined = constraint_target[f] + target;
                    double m = std::abs(combined);
                    constraint_target[f] = (m > 1e-15) ? combined / m : target;
                }
            }
        }

        if (!penalty_triplets.empty()) {
            SparseMatC penalty(nf, nf);
            penalty.setFromTriplets(penalty_triplets.begin(), penalty_triplets.end());
            L += penalty;
        }
    }

    // -----------------------------------------------------------------------
    // FIX (v54+): Symmetry constraints — wired for the first time.
    //
    // Previously CrossFieldParams had enforce_symmetry_x/y/z flags and
    // constraints.cpp had build_symmetry_constraints(), but knoppel_field
    // never called it.  Symmetry remeshing was silently broken: the solver
    // ignored the user's symmetry settings entirely.
    //
    // We build FieldConstraint lists for each enabled symmetry axis and add
    // their penalty terms to the connection Laplacian, just as feature-edge
    // constraints are handled above.  The symmetry constraint weight is
    // intentionally softer (0.3× feature weight) because symmetry is a
    // global structural hint rather than a hard topology boundary.
    // -----------------------------------------------------------------------

    // Helper: collect vertices and build raw arrays needed by build_symmetry_constraints
    auto add_symmetry_axis = [&](const float sym_normal[3]) {
        // Build float-format position and triangle arrays for constraints.cpp API
        int nv = mesh.num_vertices();
        int nf_sym = mesh.num_faces();
        std::vector<float> positions_f(nv * 3);
        std::vector<int32_t> tris_f(nf_sym * 3);
        std::vector<float> face_e1_f(nf_sym * 3);
        std::vector<float> face_e2_f(nf_sym * 3);
        std::vector<float> face_normals_f(nf_sym * 3);

        for (int vi = 0; vi < nv; ++vi) {
            Vec3 p = mesh.vertex_pos(vi);
            positions_f[vi*3+0] = (float)p[0];
            positions_f[vi*3+1] = (float)p[1];
            positions_f[vi*3+2] = (float)p[2];
        }
        for (int fi = 0; fi < nf_sym; ++fi) {
            auto [v0, v1, v2] = mesh.face_vertices(fi);
            tris_f[fi*3+0] = v0; tris_f[fi*3+1] = v1; tris_f[fi*3+2] = v2;
            // BUG FIX (v63): Guard degenerate (zero-area) faces before calling
            // .normalized().  For a zero-area face, face_normal() returns the
            // zero vector; .normalized() gives NaN (0/0 in Eigen).  That NaN
            // flows into face_normals_f → build_symmetry_constraints → the
            // projection of sym_plane_normal onto the tangent plane.  With a
            // NaN normal the projection gives NaN, tlen becomes NaN, and the
            //   (tlen < 1e-8f)
            // guard silently fails (NaN comparisons are always false), so NaN
            // propagates into theta and the constraint target — corrupting the
            // connection Laplacian for every symmetry-constrained face.
            // Fix: same raw-length check used by compute_face_frames (lines
            // 53-58): fall back to canonical Z-axis for degenerate faces.
            Vec3 raw_n_sym = mesh.face_normal(fi);
            double rn_sym_len = raw_n_sym.norm();
            Vec3 n = (rn_sym_len > 1e-14) ? (raw_n_sym / rn_sym_len) : Vec3(0, 0, 1);
            face_normals_f[fi*3+0] = (float)n[0];
            face_normals_f[fi*3+1] = (float)n[1];
            face_normals_f[fi*3+2] = (float)n[2];
            face_e1_f[fi*3+0] = (float)e1[fi][0];
            face_e1_f[fi*3+1] = (float)e1[fi][1];
            face_e1_f[fi*3+2] = (float)e1[fi][2];
            face_e2_f[fi*3+0] = (float)e2[fi][0];
            face_e2_f[fi*3+1] = (float)e2[fi][1];
            face_e2_f[fi*3+2] = (float)e2[fi][2];
        }

        std::vector<FieldConstraint> sym_constraints;
        build_symmetry_constraints(
            positions_f.data(), tris_f.data(), (int32_t)nf_sym,
            face_e1_f.data(), face_e2_f.data(), face_normals_f.data(),
            sym_normal,
            params.constraint_weight * 0.3,  // softer than feature edges
            sym_constraints
        );

        // Apply to connection Laplacian diagonal and track per-face targets
        std::vector<int32_t>               coo_rows_tmp;
        std::vector<int32_t>               coo_cols_tmp;
        std::vector<std::complex<double>>  coo_vals_tmp;
        std::vector<std::complex<double>>  rhs_tmp(nf_sym, {0,0});

        apply_constraints_to_laplacian(
            sym_constraints, coo_rows_tmp, coo_cols_tmp, coo_vals_tmp, rhs_tmp);

        // Add collected penalty entries into the main Laplacian
        std::vector<TripletC> sym_triplets;
        sym_triplets.reserve(coo_rows_tmp.size());
        for (int k = 0; k < (int)coo_rows_tmp.size(); ++k) {
            sym_triplets.push_back({coo_rows_tmp[k], coo_cols_tmp[k], coo_vals_tmp[k]});
        }
        if (!sym_triplets.empty()) {
            SparseMatC sym_penalty(nf, nf);
            sym_penalty.setFromTriplets(sym_triplets.begin(), sym_triplets.end());
            L += sym_penalty;
        }

        // Record constrained faces and their targets (from rhs)
        for (const auto& c : sym_constraints) {
            int fi = c.face_idx;
            if (fi < 0 || fi >= nf) continue;
            // target = conj(exp(4i*theta)) stored in rhs by apply_constraints…
            // Recover target as conj(rhs[fi] / weight)
            if (std::abs(rhs_tmp[fi]) > 1e-15) {
                std::complex<double> tgt = std::conj(rhs_tmp[fi] / c.weight);
                double m = std::abs(tgt);
                if (m > 1e-15) tgt /= m;
                // Blend: if already constrained by a hard edge, keep the
                // harder constraint; otherwise record the symmetry target.
                if (!is_constrained[fi]) {
                    constraint_target[fi] = tgt;
                    is_constrained[fi]    = true;
                }
            }
        }
    };

    if (params.enforce_symmetry_x) {
        const float n_x[3] = {1.f, 0.f, 0.f};
        add_symmetry_axis(n_x);
    }
    if (params.enforce_symmetry_y) {
        const float n_y[3] = {0.f, 1.f, 0.f};
        add_symmetry_axis(n_y);
    }
    if (params.enforce_symmetry_z) {
        const float n_z[3] = {0.f, 0.f, 1.f};
        add_symmetry_axis(n_z);
    }

    // -----------------------------------------------------------------------
    // Power iteration on (λ_max * I - L) to find the smallest eigenvector.
    // -----------------------------------------------------------------------
    double lambda_max = 0;
    for (int i = 0; i < nf; ++i) {
        double row_sum = L.row(i).cwiseAbs().sum();
        lambda_max = std::max(lambda_max, row_sum);
    }
    lambda_max *= 1.1;

    // Initial vector — pseudo-random phases to avoid symmetry traps
    Eigen::VectorXcd x(nf);
    for (int i = 0; i < nf; ++i)
        x[i] = std::exp(std::complex<double>(0, i * 0.7853981634)); // π/4 per face

    // BUG FIX (v57): x was filled with unit complex values but never normalized,
    // giving it overall norm √nf.  The first power iteration divided by y.norm()
    // so convergence was not affected, but the first matrix multiply operated on
    // a vector that was √nf times the intended unit scale — wasting the first
    // iteration and, on very large meshes, slightly delaying convergence.
    // Fix: normalize x to a unit vector immediately after initialization.
    {
        double xn = x.norm();
        if (xn > 1e-15) x /= xn;
    }

    // Seed constrained faces at their target direction for faster convergence
    for (int f = 0; f < nf; ++f)
        if (is_constrained[f]) x[f] = constraint_target[f];

    for (int iter = 0; iter < params.max_iterations; ++iter) {
        Eigen::VectorXcd y = lambda_max * x - L * x;

        // Enforce Dirichlet constraints: pin constrained faces to target direction.
        // BUG FIX (v25): previous code set y[f] = constraint_target[f] * y.norm()
        // which injected a magnitude far larger than 1.0 into the constrained
        // component before normalisation, distorting ALL other components after
        // the y /= norm step.  Correct approach: preserve the component magnitude
        // already computed by the matrix multiply, but fix its phase to the
        // constrained direction.  This keeps the constrained face "pinned" while
        // treating the remaining degrees of freedom normally.
        for (int f = 0; f < nf; ++f) {
            if (!is_constrained[f]) continue;
            double mag = std::abs(y[f]);
            // Clamp to a reasonable minimum so pinned faces never vanish
            if (mag < 1e-12) mag = 1.0;
            y[f] = constraint_target[f] * mag;
        }

        double norm = y.norm();
        if (norm < 1e-15) break;
        y /= norm;

        // Phase-invariant convergence check for complex eigenvectors:
        // Two unit vectors v, w of the same eigenvector differ only by a
        // global phase exp(iφ). Convergence ≡ |⟨y,x⟩| → 1.
        // This avoids false "not converged" when the iterates rotate in phase.
        double overlap = std::abs(y.dot(x));   // ∈ [0, 1]
        double delta   = 1.0 - overlap;

        x = y;

        if (iter > 10 && delta < params.convergence_tol) break;
    }

    // -----------------------------------------------------------------------
    // Normalise to unit complex per face
    // -----------------------------------------------------------------------
    field.face_field = x;
    for (int fi = 0; fi < nf; ++fi) {
        double m = std::abs(field.face_field[fi]);
        if (m > 1e-15) field.face_field[fi] /= m;
        field.face_frames[fi] = std::arg(field.face_field[fi]) * 0.25;
    }

    // -----------------------------------------------------------------------
    // BUG FIX (v96): Post-solve Gauss-Seidel smoothing for KNOPPEL_2013.
    //
    // cross_field.h explicitly documents that KNOPPEL_2013 includes
    // "post-solve smoothing".  The EIGEN_SMOOTH comment in compute_cross_field
    // further confirms this by saying EIGEN_SMOOTH "skips the post-solve
    // Gauss-Seidel pass, making it 3–5× faster than KNOPPEL_2013" — directly
    // implying KNOPPEL_2013 does NOT skip it.  Yet this call was never made.
    // Both solvers were silently delivering identical output quality.
    //
    // The fix: when do_post_smooth is true (the KNOPPEL_2013 code path),
    // convert the per-face constraint_target[] / is_constrained[] data into
    // a FieldConstraint vector and call smooth_cross_field() + recompute_face_frames().
    //
    // EIGEN_SMOOTH passes do_post_smooth = false and therefore skips this
    // block entirely, preserving its documented 3–5× speed advantage.
    // -----------------------------------------------------------------------
    if (do_post_smooth) {
        // Build FieldConstraint list from the is_constrained / constraint_target
        // arrays accumulated above (feature edges + symmetry axes).
        std::vector<FieldConstraint> smooth_constraints;
        smooth_constraints.reserve(nf);
        for (int f = 0; f < nf; ++f) {
            if (!is_constrained[f]) continue;
            FieldConstraint fc;
            fc.face_idx    = f;
            // target_angle = phase of the unit-complex target / 4
            // (inverse of exp(4i * theta) stored in constraint_target[f])
            fc.target_angle = std::arg(constraint_target[f]) / 4.0;
            fc.weight       = params.constraint_weight;
            smooth_constraints.push_back(fc);
        }

        smooth_cross_field(mesh, field, smooth_constraints);

        // Re-extract face_frames from the (now smoothed) complex field so that
        // downstream singularity detection sees consistent data.
        recompute_face_frames(field);
    }

    return field;
}

// -----------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------

CrossField compute_cross_field(
    const HalfEdgeMesh&   mesh,
    const CurvatureData&  curvature,
    const FeatureData&    features,
    const CrossFieldParams& params)
{
    // BUG FIX (v24): removed the omp_set_num_threads() call that was here.
    // omp_set_num_threads() is a process-global setter — concurrent calls from
    // different pipeline stages silently overwrite each other.  All parallel
    // regions in this file use the num_threads() clause via parallel_for() /
    // parallel_reduce(), which scopes the count to a single region only.

    if (params.solver == FieldSolver::CURVATURE_ONLY ||
        mesh.num_faces() == 0)
        return curvature_aligned_field(mesh, curvature);

    // BUG FIX (v83): EIGEN_SMOOTH was missing from the enum entirely, so
    // field_solver==0 was silently mapped to KNOPPEL_2013.  EIGEN_SMOOTH is
    // now its own solver mode: same power iteration as KNOPPEL_2013 but with a
    // much smaller iteration cap (50 vs 500) and no convergence check, giving a
    // faster result at slightly lower smoothness.  The post-solve Gauss-Seidel
    // pass (which is the expensive part) is also skipped for EIGEN_SMOOTH,
    // making it 3–5× faster than KNOPPEL_2013 while still producing a
    // globally-informed field (unlike the local CURVATURE_ONLY path).
    if (params.solver == FieldSolver::EIGEN_SMOOTH) {
        CrossFieldParams fast_params = params;
        fast_params.max_iterations  = 50;    // cap at 50 power iterations
        fast_params.convergence_tol = 0.0;   // no convergence check — always run to cap
        fast_params.solver          = FieldSolver::KNOPPEL_2013; // reuse knoppel internals
        try {
            // BUG FIX (v96): pass do_post_smooth=false so EIGEN_SMOOTH genuinely
            // skips the Gauss-Seidel pass, preserving its documented 3–5× speed
            // advantage over KNOPPEL_2013.  Before this fix both solver paths
            // were delivering identical (unsmoothed) output.
            return knoppel_field(mesh, features, fast_params, /*do_post_smooth=*/false);
        } catch (...) {
            return curvature_aligned_field(mesh, curvature);
        }
    }

    // Try Knöppel; fallback to curvature on failure
    try {
        // do_post_smooth defaults to true — KNOPPEL_2013 always runs the
        // post-solve Gauss-Seidel pass as documented in cross_field.h.
        return knoppel_field(mesh, features, params);
    } catch (...) {
        return curvature_aligned_field(mesh, curvature);
    }
}

} // namespace qf
