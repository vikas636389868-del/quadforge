/**
 * field_debug.cpp — Cross-field visual debugging utilities.
 *
 * Roadmap Phase 2 deliverable:
 *   "Visual debugging output: export cross-field as per-face direction vectors
 *    for visualization in Blender (as a temporary mesh with short edges showing
 *    the field directions)."
 *
 * All four exported functions are designed to be called from Python via the
 * optional qf_debug_field() C API entry, which is gated by the user clicking
 * "Show Field Debug" in the Advanced panel.  None of these are on the critical
 * remesh path.
 */

#include "../../include/quadforge/field/field_debug.h"

#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/**
 * Build a stable per-face orthonormal frame (e1, e2) from the face's
 * v0→v1 edge projected onto the tangent plane.  Identical logic to
 * cross_field.cpp / singularity.cpp to ensure consistent frames.
 */
static void make_stable_frame(
    const HalfEdgeMesh& mesh, int fi,
    Vec3& e1, Vec3& e2)
{
    auto [v0, v1, v2] = mesh.face_vertices(fi);
    // BUG FIX (v61): Guard against degenerate (zero-area) faces.
    // face_normal() returns zero for zero-area triangles; calling .normalized()
    // produces NaN (0/0 in Eigen).  NaN then flows into the projection below,
    // makes proj.norm() NaN, and the (len > 1e-12) fallback guard is silently
    // bypassed (NaN comparisons are always false) — ultimately producing NaN
    // in every exported direction vector for the affected face.
    // Fix: measure the raw normal length; fall back to canonical Z-axis if
    // degenerate (same strategy used by compute_face_frames in cross_field.cpp).
    Vec3 raw_n = mesh.face_normal(fi);
    double n_len = raw_n.norm();
    Vec3 n = (n_len > 1e-14) ? (raw_n / n_len) : Vec3(0, 0, 1);
    // BUG FIX (v60): Pass the raw edge, not .normalized(). A pre-normalized zero
    // edge produces NaN that bypasses the (len > 1e-12) fallback below, sending
    // NaN into every exported direction vector for the affected face.
    Vec3 ref = mesh.vertex_pos(v1) - mesh.vertex_pos(v0);
    Vec3 proj = ref - ref.dot(n) * n;
    double len = proj.norm();
    e1 = (len > 1e-12) ? proj / len : n.unitOrthogonal();
    e2 = n.cross(e1).normalized();
}

/**
 * Convert the complex field value u_f = exp(4i * theta_f) into the two
 * 3-D arm directions (one cross arm and its 90° rotation) in world space.
 *
 * The field angle theta = arg(u_f) / 4.
 * First arm:  e1 * cos(theta) + e2 * sin(theta)
 * Second arm: e1 * cos(theta + π/2) + e2 * sin(theta + π/2)
 *           = -e1 * sin(theta) + e2 * cos(theta)
 */
static void field_to_world_dirs(
    const std::complex<double>& u_f,
    const Vec3& e1, const Vec3& e2,
    Vec3& dir_u, Vec3& dir_v)
{
    double theta = std::arg(u_f) * 0.25;
    double c = std::cos(theta), s = std::sin(theta);
    dir_u = e1 * c + e2 * s;
    dir_v = e1 * (-s) + e2 * c;
}

// -----------------------------------------------------------------------
// 1. export_cross_field_vectors
// -----------------------------------------------------------------------

std::vector<float> export_cross_field_vectors(
    const HalfEdgeMesh& mesh,
    const CrossField&   field,
    float               arm_scale)
{
    int nf = mesh.num_faces();
    std::vector<float> result(nf * 6, 0.f);

    for (int fi = 0; fi < nf; ++fi) {
        Vec3 e1, e2;
        make_stable_frame(mesh, fi, e1, e2);

        std::complex<double> u = (fi < (int)field.face_field.size())
                                  ? field.face_field[fi]
                                  : std::complex<double>(1, 0);

        Vec3 dir_u, dir_v;
        field_to_world_dirs(u, e1, e2, dir_u, dir_v);

        dir_u *= arm_scale;
        dir_v *= arm_scale;

        result[fi*6+0] = (float)dir_u[0];
        result[fi*6+1] = (float)dir_u[1];
        result[fi*6+2] = (float)dir_u[2];
        result[fi*6+3] = (float)dir_v[0];
        result[fi*6+4] = (float)dir_v[1];
        result[fi*6+5] = (float)dir_v[2];
    }
    return result;
}

// -----------------------------------------------------------------------
// 2. export_cross_field_mesh
// -----------------------------------------------------------------------

void export_cross_field_mesh(
    const HalfEdgeMesh&     mesh,
    const CrossField&       field,
    float                   arm_len,
    std::vector<float>&     out_verts,
    std::vector<int32_t>&   out_edges)
{
    int nf = mesh.num_faces();

    // Auto arm length: half of the average edge length, sampled from the mesh
    if (arm_len <= 0.f) {
        double total = 0.0; int cnt = 0;
        int nhe = mesh.num_half_edges();
        for (int he = 0; he < nhe && cnt < 3000; ++he) {
            int tw = mesh.half_edge(he).twin;
            if (tw < 0 || tw < he) continue;
            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;
            total += (mesh.vertex_pos(to) - mesh.vertex_pos(from)).norm();
            ++cnt;
        }
        arm_len = (cnt > 0)
                  ? (float)(total / cnt * 0.45)
                  : 0.02f;
    }

    // Each face produces 5 vertices and 4 edges:
    //   v0 = centroid
    //   v1 = centroid + arm_len * dir_u
    //   v2 = centroid - arm_len * dir_u
    //   v3 = centroid + arm_len * dir_v
    //   v4 = centroid - arm_len * dir_v
    //   edges: (v2,v1), (v4,v3)  — two sticks crossing at v0
    //
    // Using 5 verts per face keeps it compact and avoids a separate
    // "centroid" vertex shared between faces.

    out_verts.clear();  out_verts.reserve(nf * 5 * 3);
    out_edges.clear();  out_edges.reserve(nf * 4);

    for (int fi = 0; fi < nf; ++fi) {
        // Face centroid
        auto [v0, v1, v2] = mesh.face_vertices(fi);
        Vec3 c = (mesh.vertex_pos(v0) +
                  mesh.vertex_pos(v1) +
                  mesh.vertex_pos(v2)) / 3.0;

        // Stable local frame
        Vec3 e1, e2;
        make_stable_frame(mesh, fi, e1, e2);

        // Field direction
        std::complex<double> u = (fi < (int)field.face_field.size())
                                  ? field.face_field[fi]
                                  : std::complex<double>(1, 0);
        Vec3 dir_u, dir_v;
        field_to_world_dirs(u, e1, e2, dir_u, dir_v);

        // Five points: centroid + four arm tips
        Vec3 pts[5] = {
            c,
            c + arm_len * dir_u,
            c - arm_len * dir_u,
            c + arm_len * dir_v,
            c - arm_len * dir_v
        };

        int base = fi * 5;
        for (int k = 0; k < 5; ++k) {
            out_verts.push_back((float)pts[k][0]);
            out_verts.push_back((float)pts[k][1]);
            out_verts.push_back((float)pts[k][2]);
        }

        // Two edges: (tip_minus_u → tip_plus_u) and (tip_minus_v → tip_plus_v)
        // This draws a "+" cross centred on the centroid.
        out_edges.push_back(base + 2); out_edges.push_back(base + 1); // u-arm
        out_edges.push_back(base + 4); out_edges.push_back(base + 3); // v-arm
    }
}

// -----------------------------------------------------------------------
// 3. export_singularities_as_points
// -----------------------------------------------------------------------

void export_singularities_as_points(
    const std::vector<SingularityInfo>& sings,
    std::vector<float>&                 out_positions,
    std::vector<float>&                 out_indices)
{
    int n = (int)sings.size();
    out_positions.resize(n * 3);
    out_indices.resize(n);

    for (int i = 0; i < n; ++i) {
        out_positions[i*3+0] = (float)sings[i].position[0];
        out_positions[i*3+1] = (float)sings[i].position[1];
        out_positions[i*3+2] = (float)sings[i].position[2];
        out_indices[i]       = (float)sings[i].index;
    }
}

// -----------------------------------------------------------------------
// 4. export_field_quality_map
// -----------------------------------------------------------------------

std::vector<float> export_field_quality_map(
    const HalfEdgeMesh& mesh,
    const CrossField&   field)
{
    int nf = mesh.num_faces();
    std::vector<float> quality(nf, 1.0f);

    // For each face, accumulate the parallel-transported neighbour fields
    // and measure how well they agree (magnitude of the weighted average).
    // A magnitude near 1.0 means all neighbours point the same way (smooth).
    // A magnitude near 0.0 means they cancel out (near a singularity).
    //
    // We use uniform weights here (not cotangent) because the quality map
    // is purely for visual output — speed matters more than precision.

    int nhe = mesh.num_half_edges();
    std::vector<std::complex<double>> acc(nf, {0, 0});
    std::vector<double> weight_sum(nf, 0.0);

    for (int he = 0; he < nhe; ++he) {
        int tw = mesh.half_edge(he).twin;
        if (tw < 0) continue;
        // BUG FIX (v57): The previous code iterated ALL half-edges including both
        // he and its twin tw.  Because the loop body added the symmetric contribution
        // to BOTH fi and fj in a single iteration, each interior edge pair (he, tw)
        // was accumulated TWICE into each face — once when he was processed and again
        // when tw was processed.  The quality score was still numerically correct
        // (the factor-of-2 cancelled in acc / weight_sum), but the function did
        // exactly 2× the necessary work.  Fix: skip the twin direction with the
        // same guard used throughout the rest of the field code (tw < he), then
        // keep the symmetric accumulation so each edge still contributes to both
        // adjacent faces in a single pass.
        if (tw < he) continue;   // process each interior edge exactly once

        int fi = mesh.half_edge(he).face;
        int fj = mesh.half_edge(tw).face;
        if (fi < 0 || fj < 0) continue;

        // Build stable frames for both faces
        Vec3 e1_i, e2_i, e1_j, e2_j;
        make_stable_frame(mesh, fi, e1_i, e2_i);
        make_stable_frame(mesh, fj, e1_j, e2_j);

        int v_from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
        int v_to   = mesh.half_edge(he).vertex;
        // BUG FIX (v60): Guard the shared-edge normalisation.  A degenerate edge
        // (v_from == v_to in position) makes .normalized() return NaN, which
        // propagates into ai, aj, phi, r_ij, and then into acc[] — silently
        // corrupting the quality heat-map for all faces adjacent to that edge.
        // Skip: a zero-length edge cannot transport any field direction.
        Vec3 raw_edge = mesh.vertex_pos(v_to) - mesh.vertex_pos(v_from);
        double elen = raw_edge.norm();
        if (elen < 1e-12) continue;
        Vec3 edge = raw_edge / elen;

        double ai = std::atan2(edge.dot(e2_i), edge.dot(e1_i));
        double aj = std::atan2(edge.dot(e2_j), edge.dot(e1_j));
        double phi = aj - ai;
        // Wrap to 4-RoSy period
        while (phi >  M_PI / 4.0) phi -= M_PI / 2.0;
        while (phi < -M_PI / 4.0) phi += M_PI / 2.0;
        std::complex<double> r_ij = std::exp(std::complex<double>(0, 4.0 * phi));

        // Transport u_j into frame i and accumulate for face i
        std::complex<double> u_j = (fj < (int)field.face_field.size())
                                    ? field.face_field[fj]
                                    : std::complex<double>(1, 0);
        acc[fi]        += std::conj(r_ij) * u_j;
        weight_sum[fi] += 1.0;

        // Symmetric: transport u_i into frame j for face j
        std::complex<double> u_i = (fi < (int)field.face_field.size())
                                    ? field.face_field[fi]
                                    : std::complex<double>(1, 0);
        acc[fj]        += r_ij * u_i;
        weight_sum[fj] += 1.0;
    }

    for (int fi = 0; fi < nf; ++fi) {
        if (weight_sum[fi] < 1e-12) {
            quality[fi] = 1.0f; // isolated face — assume perfect
        } else {
            // Normalised coherence: |acc| / weight_sum, clipped to [0, 1]
            double q = std::abs(acc[fi]) / weight_sum[fi];
            quality[fi] = (float)std::min(1.0, std::max(0.0, q));
        }
    }
    return quality;
}

// -----------------------------------------------------------------------
// 5. cross_field_summary
// -----------------------------------------------------------------------

std::string cross_field_summary(
    const HalfEdgeMesh&                  mesh,
    const CrossField&                    field,
    const std::vector<SingularityInfo>&  sings)
{
    int nf = mesh.num_faces();

    // Count singularities by sign
    int pos_count = 0, neg_count = 0;
    for (const auto& s : sings) {
        if (s.index > 0) ++pos_count;
        else             ++neg_count;
    }

    // Compute quality map statistics
    auto qmap = export_field_quality_map(mesh, field);
    double q_sum  = 0.0;
    float  q_min  = 1.0f;
    for (float q : qmap) {
        q_sum += q;
        if (q < q_min) q_min = q;
    }
    double q_mean = (nf > 0) ? q_sum / nf : 1.0;

    // Estimate average field magnitude (should be ~1.0 for a normalised field)
    double mag_sum = 0.0;
    for (int fi = 0; fi < (int)field.face_field.size(); ++fi)
        mag_sum += std::abs(field.face_field[fi]);
    double mag_mean = (field.face_field.size() > 0)
                      ? mag_sum / field.face_field.size()
                      : 0.0;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "=== Cross-Field Summary ===\n";
    ss << "  Faces:               " << nf << "\n";
    ss << "  Singularities:       " << sings.size()
       << "  (+" << pos_count << " / -" << neg_count << ")\n";
    ss << "  Mean quality score:  " << q_mean << "\n";
    ss << "  Min  quality score:  " << q_min
       << (q_min < 0.3f ? "  *** low — check for degenerate triangles" : "") << "\n";
    ss << "  Mean field magnitude:" << mag_mean
       << (mag_mean < 0.98 ? "  *** < 1 — field may not be fully converged" : "") << "\n";
    ss << "  Expected singularities (Poincaré-Hopf): "
       // For a 4-RoSy field each singularity has index ±1/4.
       // Poincaré-Hopf: Σ index_i = χ(M).
       // Minimum singularity count (all same sign) = |χ| × 4.
       // BUG FIX (v56): was "/ 4" (integer division → 0 for typical meshes)
       // instead of "* 4". For a sphere (χ=2) the correct answer is 8, not 0.
       << std::abs(mesh.euler_characteristic()) * 4 << " (minimum, all same sign)\n";
    return ss.str();
}

} // namespace qf
