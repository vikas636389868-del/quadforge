/**
 * smoothing.cpp — Post-solve 4-RoSy field smoothing.
 *
 * After the Knöppel 2013 eigensolver produces the cross-field, residual
 * noise can remain from:
 *   – degenerate triangles perturbing the connection Laplacian entries
 *   – hard constraint penalty terms creating local discontinuities at
 *     feature edge boundaries
 *   – partial convergence when the power iteration hits max_iterations
 *
 * This module runs a fast iterated Gauss-Seidel smoothing pass directly
 * on the complex per-face field representation.  Each iteration sweeps
 * all faces, replacing u_f with the parallel-transport-corrected weighted
 * average of its neighbours, then projecting back onto the unit circle.
 * Feature-constrained faces are blended back toward their target after
 * each iteration so constraints are never fully abandoned.
 *
 * The implementation is double-buffered so that within a single iteration
 * reads come from the previous iterate and writes go to the new one.
 * This makes the result independent of the face traversal order and
 * safe to parallelise with OpenMP.
 *
 * Complexity: O(iter × F) time, O(F) additional memory.
 */

#include "../../include/quadforge/field/smoothing.h"
#include "../../include/quadforge/accel/omp_utils.h"
#include "../../include/quadforge/accel/cache_utils.h"

#include <cmath>
#include <algorithm>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// -----------------------------------------------------------------------
// Cotangent weight computation
// -----------------------------------------------------------------------

/**
 * Cotangent of the angle at vertex `apex` in the triangle (apex, b, c).
 * cot θ = (b-apex)·(c-apex) / |(b-apex)×(c-apex)|
 * Clamped to avoid blow-up on degenerate triangles.
 */
static double cot_angle(const Vec3& apex, const Vec3& b, const Vec3& c)
{
    Vec3 u = b - apex;
    Vec3 v = c - apex;
    double num = u.dot(v);
    double den = u.cross(v).norm();
    if (den < 1e-14) return 0.0;
    // Clamp to [-10, 10] — extremely obtuse triangles still participate
    // but at a bounded weight so they don't destabilise the iteration.
    return std::max(-10.0, std::min(10.0, num / den));
}

std::vector<std::pair<int,double>> compute_cotangent_weights(
    const HalfEdgeMesh& mesh)
{
    int nhe = mesh.num_half_edges();
    std::vector<std::pair<int,double>> result;
    result.reserve(nhe / 2);

    for (int he = 0; he < nhe; ++he) {
        int tw = mesh.half_edge(he).twin;
        if (tw < 0)  continue;   // boundary — skip
        if (tw < he) continue;   // process each interior edge once

        int fi = mesh.half_edge(he).face;
        int fj = mesh.half_edge(tw).face;
        if (fi < 0 || fj < 0) continue;

        // Vertices of the two triangles sharing this edge.
        // he goes from v_prev → v_to ; the opposite vertex is v_opp.
        int v_from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
        int v_to   = mesh.half_edge(he).vertex;

        // Opposite vertex in face fi: the vertex NOT on this edge.
        // In a half-edge structure the third vertex is the destination
        // of the NEXT half-edge.
        int v_opp_i = mesh.half_edge(mesh.half_edge(he).next).vertex;
        int v_opp_j = mesh.half_edge(mesh.half_edge(tw).next).vertex;

        Vec3 p_from  = mesh.vertex_pos(v_from);
        Vec3 p_to    = mesh.vertex_pos(v_to);
        Vec3 p_opp_i = mesh.vertex_pos(v_opp_i);
        Vec3 p_opp_j = mesh.vertex_pos(v_opp_j);

        // Cotangent of the angle at the opposite vertex in each triangle.
        double cot_i = cot_angle(p_opp_i, p_from, p_to);
        double cot_j = cot_angle(p_opp_j, p_from, p_to);

        // Standard cotangent weight: (cot α + cot β) / 2
        double w = 0.5 * (cot_i + cot_j);
        // Ensure weight is positive (obtuse triangles can give negative cot)
        w = std::max(w, 1e-10);

        result.push_back({he, w});
    }
    return result;
}

// -----------------------------------------------------------------------
// Re-extract face_frames from the complex field
// -----------------------------------------------------------------------

void recompute_face_frames(CrossField& field)
{
    int nf = (int)field.face_field.size();
    field.face_frames.resize(nf);
    for (int f = 0; f < nf; ++f) {
        // face_field[f] = exp(4i * theta_f)
        // => theta_f = arg(face_field[f]) / 4
        field.face_frames[f] = std::arg(field.face_field[f]) * 0.25;
    }
}

// -----------------------------------------------------------------------
// Parallel transport angle between adjacent faces (identical to
// cross_field.cpp, duplicated here to keep smoothing.cpp self-contained
// and avoid header coupling with internal cross_field helpers)
// -----------------------------------------------------------------------

static double transport_angle_for_smoothing(
    const HalfEdgeMesh& mesh, int he)
{
    int tw = mesh.half_edge(he).twin;
    if (tw < 0) return 0.0;

    int fi = mesh.half_edge(he).face;
    int fj = mesh.half_edge(tw).face;
    if (fi < 0 || fj < 0) return 0.0;

    // BUG FIX (v61): Guard against degenerate (zero-area) faces before calling
    // .normalized().  For a zero-area face, face_normal() returns the zero vector;
    // calling .normalized() on it produces NaN (0/0 in Eigen).  That NaN flows
    // into the make_frame projection, bypasses the proj.norm() fallback guard
    // (NaN < 1e-12 is FALSE), and then enters e1/e2 and ultimately the transport
    // angle — silently corrupting the Gauss-Seidel accumulator for every face
    // pair that touches the degenerate face.
    // Fix: measure the raw normal length explicitly; return 0.0 (zero transport ≡
    // coplanar faces) for any degenerate face, matching the same guard already
    // present in curvature.cpp and the degenerate-edge guards added in v60.
    {
        Vec3 raw_ni = mesh.face_normal(fi);
        Vec3 raw_nj = mesh.face_normal(fj);
        double ni_len = raw_ni.norm(), nj_len = raw_nj.norm();
        if (ni_len < 1e-14 || nj_len < 1e-14) return 0.0;
    }
    Vec3 ni = mesh.face_normal(fi).normalized();
    Vec3 nj = mesh.face_normal(fj).normalized();

    // Build stable per-face frames from each face's own v0→v1 edge.
    auto make_frame = [](const Vec3& n, const Vec3& ref_edge,
                         Vec3& e1, Vec3& e2)
    {
        Vec3 proj = ref_edge - ref_edge.dot(n) * n;
        double len = proj.norm();
        e1 = (len > 1e-12) ? proj / len : n.unitOrthogonal();
        e2 = n.cross(e1).normalized();
    };

    auto [vi0, vi1, vi2] = mesh.face_vertices(fi);
    auto [vj0, vj1, vj2] = mesh.face_vertices(fj);

    // BUG FIX (v60): Pass raw edges to make_frame, not .normalized() ones.
    // Pre-normalizing a zero-length edge yields NaN; NaN < 1e-12 is FALSE so
    // the guard inside make_frame is bypassed, producing NaN transport angles
    // that corrupt the Gauss-Seidel accumulator for the affected face pair.
    Vec3 ref_i = mesh.vertex_pos(vi1) - mesh.vertex_pos(vi0);
    Vec3 ref_j = mesh.vertex_pos(vj1) - mesh.vertex_pos(vj0);

    Vec3 e1_i, e2_i, e1_j, e2_j;
    make_frame(ni, ref_i, e1_i, e2_i);
    make_frame(nj, ref_j, e1_j, e2_j);

    int v_from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
    int v_to   = mesh.half_edge(he).vertex;
    // BUG FIX (v60): Guard the shared-edge normalisation.  If v_from and v_to are
    // coincident (degenerate edge), .normalized() returns NaN; NaN then enters ai,
    // aj, phi, and the transport complex r_ij stored in neighbours[] — silently
    // corrupting the weighted accumulator for every Gauss-Seidel sweep that
    // touches the affected face pair.  Zero transport (co-planar fallback) is safe.
    Vec3 raw_edge = mesh.vertex_pos(v_to) - mesh.vertex_pos(v_from);
    double elen = raw_edge.norm();
    if (elen < 1e-12) return 0.0;
    Vec3 edge = raw_edge / elen;

    double ai = std::atan2(edge.dot(e2_i), edge.dot(e1_i));
    double aj = std::atan2(edge.dot(e2_j), edge.dot(e1_j));
    double phi = aj - ai;

    // Wrap to [-π/4, π/4) for 4-RoSy symmetry
    while (phi >  M_PI / 4.0) phi -= M_PI / 2.0;
    while (phi < -M_PI / 4.0) phi += M_PI / 2.0;
    return phi;
}

// -----------------------------------------------------------------------
// Main smoothing function
// -----------------------------------------------------------------------

double smooth_cross_field(
    const HalfEdgeMesh&              mesh,
    CrossField&                      field,
    const std::vector<FieldConstraint>& constraints,
    const FieldSmoothingParams&      params)
{
    int nf = mesh.num_faces();
    if (nf == 0 || params.iterations <= 0) return 0.0;

    int nt = resolve_thread_count(params.num_threads);

    // -----------------------------------------------------------------
    // Build per-half-edge smoothing weights (cotangent or area-based)
    // -----------------------------------------------------------------
    // We store per-face neighbour lists: for each face fi a list of
    // (face_j, transport_complex_r_ij, weight_w_ij).
    // This avoids repeated half-edge traversal inside the iteration loop.

    struct Neighbour {
        int    face_j;
        std::complex<double> r_ij;   // exp(4i * phi_ij)
        double weight;
    };

    std::vector<std::vector<Neighbour>> neighbours(nf);

    if (params.use_cotangent_weights) {
        // Cotangent path
        auto cot_weights = compute_cotangent_weights(mesh);
        for (auto& [he, w] : cot_weights) {
            int tw = mesh.half_edge(he).twin;
            int fi = mesh.half_edge(he).face;
            int fj = mesh.half_edge(tw).face;
            if (fi < 0 || fj < 0) continue;

            double phi = transport_angle_for_smoothing(mesh, he);
            std::complex<double> r_ij = std::exp(std::complex<double>(0, 4.0 * phi));

            neighbours[fi].push_back({fj,  r_ij, w});
            neighbours[fj].push_back({fi,  std::conj(r_ij), w});
        }
    } else {
        // Area-based path (fallback for very irregular meshes where
        // cotangent weights could be excessively negative)
        int nhe = mesh.num_half_edges();
        for (int he = 0; he < nhe; ++he) {
            int tw = mesh.half_edge(he).twin;
            if (tw < 0 || tw < he) continue;

            int fi = mesh.half_edge(he).face;
            int fj = mesh.half_edge(tw).face;
            if (fi < 0 || fj < 0) continue;

            auto [v0i, v1i, v2i] = mesh.face_vertices(fi);
            auto [v0j, v1j, v2j] = mesh.face_vertices(fj);

            Vec3 ab_i = mesh.vertex_pos(v1i) - mesh.vertex_pos(v0i);
            Vec3 ac_i = mesh.vertex_pos(v2i) - mesh.vertex_pos(v0i);
            Vec3 ab_j = mesh.vertex_pos(v1j) - mesh.vertex_pos(v0j);
            Vec3 ac_j = mesh.vertex_pos(v2j) - mesh.vertex_pos(v0j);

            double area_i = 0.5 * ab_i.cross(ac_i).norm();
            double area_j = 0.5 * ab_j.cross(ac_j).norm();
            double w = std::max(0.5 * (area_i + area_j), 1e-12);

            double phi = transport_angle_for_smoothing(mesh, he);
            std::complex<double> r_ij = std::exp(std::complex<double>(0, 4.0 * phi));

            neighbours[fi].push_back({fj,  r_ij, w});
            neighbours[fj].push_back({fi,  std::conj(r_ij), w});
        }
    }

    // -----------------------------------------------------------------
    // Build constraint look-up: face_idx → target complex
    // -----------------------------------------------------------------
    // BUG FIX (v58): The previous code used simple assignment:
    //   constraint_map[c.face_idx] = exp(4i * theta)
    // When two constraints (e.g. a feature edge AND a boundary edge)
    // targeted the same face, only the LAST one survived in the map —
    // the earlier target was silently overwritten.  This was inconsistent
    // with the connection Laplacian, which correctly accumulates ALL
    // constraints as additive diagonal penalties (so the field is pulled
    // toward a blend of both targets by the Laplacian), while the
    // smoothing pass only enforced the last one, under-constraining the
    // face and producing field discontinuities at multiply-constrained edges.
    //
    // Fix: when a face already has an entry, circularly-average the two
    // unit-complex targets by summing and normalising.  This gives the
    // geodesic midpoint of the two target angles on the 4-RoSy unit circle,
    // which is consistent with what the Laplacian penalty approximates when
    // two equal-weight constraints pull in different directions.
    std::unordered_map<int, std::complex<double>> constraint_map;
    constraint_map.reserve(constraints.size() * 2);
    for (const auto& c : constraints) {
        // target in 4-RoSy: exp(4i * theta)
        std::complex<double> tgt =
            std::exp(std::complex<double>(0, 4.0 * c.target_angle));
        auto [it, inserted] = constraint_map.emplace(c.face_idx, tgt);
        if (!inserted) {
            // Face already has a constraint — circularly-average the two targets
            std::complex<double> combined = it->second + tgt;
            double m = std::abs(combined);
            it->second = (m > 1e-15) ? combined / m : tgt; // fallback: keep new target
        }
    }

    // -----------------------------------------------------------------
    // Gauss-Seidel iteration (double-buffered)
    // -----------------------------------------------------------------
    // 'current' is the read buffer, 'next' is the write buffer.
    // Swap after each iteration.

    std::vector<std::complex<double>> current(field.face_field.begin(),
                                              field.face_field.end());
    std::vector<std::complex<double>> next(nf);

    // Ensure all input values are unit complex (normalise first)
    for (int f = 0; f < nf; ++f) {
        double m = std::abs(current[f]);
        if (m > 1e-15) current[f] /= m;
        else           current[f]  = std::complex<double>(1.0, 0.0);
    }

    double max_delta = 0.0;
    const double alpha = params.constraint_alpha;

    for (int iter = 0; iter < params.iterations; ++iter) {

        max_delta = 0.0;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(nt) \
                reduction(max : max_delta)
#endif
        for (int fi = 0; fi < nf; ++fi) {
            const auto& nbrs = neighbours[fi];

            if (nbrs.empty()) {
                next[fi] = current[fi];
                continue;
            }

            // Weighted sum of parallel-transported neighbour fields:
            //   sum_j  w_ij * conj(r_ij) * u_j
            // conj(r_ij) transports u_j from frame j back to frame i.
            std::complex<double> acc(0.0, 0.0);
            for (const auto& nb : nbrs) {
                acc += nb.weight * std::conj(nb.r_ij) * current[nb.face_j];
            }

            // Project onto unit circle
            double acc_m = std::abs(acc);
            std::complex<double> u_new;
            if (acc_m > 1e-15) {
                u_new = acc / acc_m;
            } else {
                // Degenerate — keep current value
                u_new = current[fi];
            }

            // Blend with constraint target if this face is constrained
            auto cit = constraint_map.find(fi);
            if (cit != constraint_map.end()) {
                // alpha fraction from target, (1-alpha) from smoothed result
                u_new = (1.0 - alpha) * u_new + alpha * cit->second;
                double m = std::abs(u_new);
                if (m > 1e-15) {
                    u_new /= m;
                } else {
                    // BUG FIX (v56): When the smoothed direction and the
                    // constraint target are nearly antipodal their blend is
                    // near-zero.  The previous code left u_new ≈ 0 which
                    // produces a denorm field value at a constrained face and
                    // later corrupts singularity detection.
                    // Correct fallback: use the constraint target directly
                    // (it is already unit-complex by construction).
                    u_new = cit->second;
                }
            }

            // Track convergence
            double delta = std::abs(u_new - current[fi]);
            if (delta > max_delta) max_delta = delta;

            next[fi] = u_new;
        }

        std::swap(current, next);

        // Early exit if converged
        if (max_delta < params.convergence_tol) break;
    }

    // Write back to the CrossField
    for (int f = 0; f < nf; ++f) {
        field.face_field[f] = current[f];
    }

    // Re-extract face_frames so singularity detection sees consistent data
    recompute_face_frames(field);

    return max_delta;
}

} // namespace qf
