/**
 * metrics.cpp — Quality metrics computation for the output quad mesh.
 *
 * Implements Step 5.4 of the QuadForge post-processing stage:
 *   - Quad/tri face percentages
 *   - Vertex valence histogram and irregularity ratio
 *   - Interior angle statistics (min / max / mean)
 *   - Face aspect-ratio statistics (max_edge / min_edge)
 *   - Scaled Jacobian quality metric (min and mean over all quad corners)
 *   - Feature-edge alignment error (mean and max angle vs. nearest feature tangent)
 *   - Catmull-Clark subdivision readiness flag
 *
 * v28 additions:
 *   - Added extended overload compute_quality_metrics(..., HalfEdgeMesh*)
 *     that fully implements feature-alignment error by looking up original
 *     mesh vertex positions via HalfEdgeMesh::vertex_pos().  The base
 *     overload previously left this metric as a no-op placeholder (−1).
 *
 * All per-face and per-vertex loops run in parallel via parallel_reduce /
 * parallel_for where thread-safety allows.
 */

#include "../../include/quadforge/postprocess/metrics.h"
#include "../../include/quadforge/mesh/halfedge.h"
#include "../../include/quadforge/accel/omp_utils.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <array>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

// Return the 4 vertex positions of quad fi as Vec3.
inline std::array<Vec3, 4> quad_verts(const QuadMesh& qm, int fi) {
    const auto& q = qm.quads[fi];
    std::array<Vec3, 4> v;
    for (int k = 0; k < 4; ++k) {
        int vi = q[k];
        v[k] = { qm.vertices[vi][0], qm.vertices[vi][1], qm.vertices[vi][2] };
    }
    return v;
}

// Interior angle at corner k of quad (v0,v1,v2,v3), in degrees.
// The angle at corner k is between the two edges that share vertex k.
//
//   quad layout:  v[0] -- v[1]
//                  |        |
//                 v[3] -- v[2]
//
// At v[k]: edges go to v[(k+1)%4] and v[(k+3)%4].
inline double corner_angle_deg(const std::array<Vec3, 4>& v, int k) {
    Vec3 e1 = v[(k + 1) % 4] - v[k];
    Vec3 e2 = v[(k + 3) % 4] - v[k];
    double l1 = e1.norm();
    double l2 = e2.norm();
    if (l1 < 1e-15 || l2 < 1e-15) return 90.0; // degenerate
    double cosA = e1.dot(e2) / (l1 * l2);
    cosA = std::max(-1.0, std::min(1.0, cosA));
    return std::acos(cosA) * (180.0 / M_PI);
}

// Scaled Jacobian at corner k.
// J = [e1 | e2]  where e1, e2 are the two edge directions at corner k.
// SJ = det(J) / (|e1| * |e2|)  evaluated via the cross-product magnitude.
//
// For 3-D quads embedded in R³ we use the signed magnitude of the cross
// product projected onto the approximate face normal:
//   SJ = (e1 × e2) · n_hat  /  (|e1| * |e2|)
// where n_hat is the unit face normal (cross of the two diagonals).
inline double scaled_jacobian_corner(const std::array<Vec3, 4>& v,
                                      int k,
                                      const Vec3& face_normal) {
    Vec3 e1 = v[(k + 1) % 4] - v[k];
    Vec3 e2 = v[(k + 3) % 4] - v[k];
    double l1 = e1.norm();
    double l2 = e2.norm();
    if (l1 < 1e-15 || l2 < 1e-15) return 0.0;
    double sj = e1.cross(e2).dot(face_normal) / (l1 * l2);
    return std::max(-1.0, std::min(1.0, sj)); // clamp to [-1,1]
}

// Approximate face normal using the two diagonals of a quad (v0..v3).
inline Vec3 quad_face_normal(const std::array<Vec3, 4>& v) {
    Vec3 d1 = v[2] - v[0];
    Vec3 d2 = v[3] - v[1];
    Vec3 n  = d1.cross(d2);
    double l = n.norm();
    if (l < 1e-15) return Vec3(0, 0, 1); // degenerate fallback
    return n / l;
}

// Aspect ratio = max_edge_length / min_edge_length.
// For a degenerate quad (zero edge), returns a large sentinel.
inline double quad_aspect_ratio(const std::array<Vec3, 4>& v) {
    double min_l = std::numeric_limits<double>::max();
    double max_l = 0.0;
    for (int k = 0; k < 4; ++k) {
        double l = (v[(k + 1) % 4] - v[k]).norm();
        if (l < min_l) min_l = l;
        if (l > max_l) max_l = l;
    }
    if (min_l < 1e-15) return 1e6;
    return max_l / min_l;
}

// Detect boundary vertices: a vertex is on the boundary if it has at least
// one incident edge with no twin face (i.e., fewer incident faces than the
// manifold-interior count).
// For a QuadMesh we approximate: a vertex is interior if every adjacent edge
// appears in exactly two faces.  We do a simple half-edge count approach:
// count for each edge (vi, vj) how many faces contain it.  Boundary edge → 1.
std::vector<bool> compute_boundary_flags(const QuadMesh& qm) {
    int nv = (int)qm.vertices.size();
    // For each directed edge, count occurrences.  An undirected edge with
    // only one direction present is a boundary edge.
    std::unordered_map<int64_t, int> edge_count;
    edge_count.reserve(qm.quads.size() * 4 + qm.tris.size() * 3);

    auto key = [](int a, int b) -> int64_t {
        return ((int64_t)(a < b ? a : b) << 32) | (int64_t)(uint32_t)(a < b ? b : a);
    };

    for (const auto& q : qm.quads) {
        for (int k = 0; k < 4; ++k) {
            int va = q[k], vb = q[(k + 1) % 4];
            if (va >= 0 && vb >= 0 && va < nv && vb < nv)
                edge_count[key(va, vb)]++;
        }
    }
    for (const auto& t : qm.tris) {
        for (int k = 0; k < 3; ++k) {
            int va = t[k], vb = t[(k + 1) % 3];
            if (va >= 0 && vb >= 0 && va < nv && vb < nv)
                edge_count[key(va, vb)]++;
        }
    }

    // Boundary vertex = incident to at least one edge that appears only once.
    std::vector<bool> bnd(nv, false);
    for (const auto& q : qm.quads) {
        for (int k = 0; k < 4; ++k) {
            int va = q[k], vb = q[(k + 1) % 4];
            if (va >= 0 && vb >= 0 && va < nv && vb < nv) {
                if (edge_count[key(va, vb)] == 1) {
                    bnd[va] = true;
                    bnd[vb] = true;
                }
            }
        }
    }
    for (const auto& t : qm.tris) {
        for (int k = 0; k < 3; ++k) {
            int va = t[k], vb = t[(k + 1) % 3];
            if (va >= 0 && vb >= 0 && va < nv && vb < nv) {
                if (edge_count[key(va, vb)] == 1) {
                    bnd[va] = true;
                    bnd[vb] = true;
                }
            }
        }
    }
    return bnd;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// compute_quality_metrics
// ---------------------------------------------------------------------------

QualityMetrics compute_quality_metrics(
    const QuadMesh&    qm,
    const FeatureData* features,
    int32_t            target_quads,
    double             snap_distance,
    int                num_threads)
{
    QualityMetrics m;
    m.target_quad_count = target_quads;
    m.valence_histogram.assign(10, 0);

    int nv = (int)qm.vertices.size();
    int nq = (int)qm.quads.size();
    int nt = (int)qm.tris.size();

    if (nv == 0 || (nq + nt) == 0) return m;

    // -----------------------------------------------------------------------
    // 1. Face composition
    // -----------------------------------------------------------------------
    m.num_quad_faces  = nq;
    m.num_tri_faces   = nt;
    m.quad_percentage = (nq + nt > 0)
        ? float(nq) / float(nq + nt)
        : 0.f;

    if (target_quads > 0) {
        m.count_accuracy = std::abs(float(nq) - float(target_quads))
                           / float(target_quads);
    }

    // -----------------------------------------------------------------------
    // 2. Vertex valence histogram
    // -----------------------------------------------------------------------
    {
        std::vector<int> valence(nv, 0);

        for (const auto& q : qm.quads)
            for (int k = 0; k < 4; ++k)
                if (q[k] >= 0 && q[k] < nv) valence[q[k]]++;

        for (const auto& t : qm.tris)
            for (int k = 0; k < 3; ++k)
                if (t[k] >= 0 && t[k] < nv) valence[t[k]]++;

        // Boundary flags (boundary vertices have different "ideal" valence).
        std::vector<bool> bnd = compute_boundary_flags(qm);

        double sum_valence  = 0.0;
        int    count_int    = 0; // interior vertices
        int    num_irreg    = 0;

        for (int vi = 0; vi < nv; ++vi) {
            int v = valence[vi];
            int clamped = std::min(v, 9);
            m.valence_histogram[clamped]++;

            if (!bnd[vi]) {
                sum_valence += v;
                count_int++;
                if (v != 4) num_irreg++;
            }
        }

        m.avg_valence       = (count_int > 0)
                              ? float(sum_valence / count_int) : 0.f;
        m.num_irregular     = num_irreg;
        m.irregularity_ratio= (count_int > 0)
                              ? float(num_irreg) / float(count_int) : 0.f;
        m.subdiv_ready      = (nt == 0) && (m.irregularity_ratio < 0.05f);
    }

    // -----------------------------------------------------------------------
    // 3. Per-quad angle, aspect-ratio, and scaled-Jacobian statistics
    //    (All are embarrassingly parallel over faces.)
    // -----------------------------------------------------------------------
    if (nq > 0) {
        struct FaceAcc {
            double sum_angle{0.0};
            double min_angle{std::numeric_limits<double>::max()};
            double max_angle{-std::numeric_limits<double>::max()};
            double sum_ar{0.0};
            double max_ar{-std::numeric_limits<double>::max()}; // BUG-D FIX: correct max-reduction identity (-inf), not 0.0 (v115)
            double sum_sj{0.0};
            double min_sj{std::numeric_limits<double>::max()};
            int    count{0};
        };

        FaceAcc global = parallel_reduce<FaceAcc>(
            nq,
            FaceAcc{},
            [&](int fi, FaceAcc& acc) {
                auto v     = quad_verts(qm, fi);
                Vec3 nrm   = quad_face_normal(v);
                double ar  = quad_aspect_ratio(v);
                double sj_min_corner = std::numeric_limits<double>::max();
                double sj_sum = 0.0;

                for (int k = 0; k < 4; ++k) {
                    double ang = corner_angle_deg(v, k);
                    acc.sum_angle += ang;
                    if (ang < acc.min_angle) acc.min_angle = ang;
                    if (ang > acc.max_angle) acc.max_angle = ang;

                    double sj = scaled_jacobian_corner(v, k, nrm);
                    sj_sum += sj;
                    if (sj < sj_min_corner) sj_min_corner = sj;
                }

                acc.sum_ar += ar;
                if (ar > acc.max_ar) acc.max_ar = ar;

                acc.sum_sj += sj_sum / 4.0;  // per-face mean SJ
                if (sj_min_corner < acc.min_sj) acc.min_sj = sj_min_corner;
                acc.count++;
            },
            [](FaceAcc a, FaceAcc b) -> FaceAcc {
                FaceAcc c;
                c.sum_angle = a.sum_angle + b.sum_angle;
                c.min_angle = std::min(a.min_angle, b.min_angle);
                c.max_angle = std::max(a.max_angle, b.max_angle);
                c.sum_ar    = a.sum_ar    + b.sum_ar;
                c.max_ar    = std::max(a.max_ar, b.max_ar);
                c.sum_sj    = a.sum_sj    + b.sum_sj;
                c.min_sj    = std::min(a.min_sj,  b.min_sj);
                c.count     = a.count     + b.count;
                return c;
            },
            num_threads);

        if (global.count > 0) {
            m.mean_angle_deg     = float(global.sum_angle / (global.count * 4));
            m.min_angle_deg      = float(global.min_angle);
            m.max_angle_deg      = float(global.max_angle);
            m.mean_aspect_ratio  = float(global.sum_ar   / global.count);
            m.max_aspect_ratio   = float(global.max_ar);
            m.mean_scaled_jacobian = float(global.sum_sj / global.count);
            m.min_scaled_jacobian  = float(global.min_sj);
        }
    }

    // -----------------------------------------------------------------------
    // 4. Feature alignment error
    //    Requires original mesh vertex positions (FeatureData stores index
    //    pairs, not positions).  The base overload cannot satisfy this without
    //    the HalfEdgeMesh.  Leave mean/max at the −1 sentinel; callers that
    //    have the ref_mesh should use the extended overload below.
    // -----------------------------------------------------------------------
    // (No-op in base overload — fields stay at −1 as initialised.)

    return m;
}

// ---------------------------------------------------------------------------
// apply_metrics_to_mesh
// ---------------------------------------------------------------------------

void apply_metrics_to_mesh(QuadMesh& quad_mesh, const QualityMetrics& metrics) {
    quad_mesh.quad_percentage = metrics.quad_percentage;
    quad_mesh.avg_valence     = metrics.avg_valence;
}

// ---------------------------------------------------------------------------
// compute_quality_metrics — extended overload with HalfEdgeMesh
//
// Identical to the base overload but also computes feature alignment error
// using ref_mesh->vertex_pos() to look up the absolute positions of each
// feature edge endpoint stored as index pairs in FeatureData::hard_edges.
//
// Algorithm:
//   For each directed quad edge (vi → vj):
//     1. Compute its midpoint M and unit direction D = (vj - vi) / |vj - vi|.
//     2. Iterate over all feature edges (fa, fb) in FeatureData::hard_edges.
//        For each, project M onto the segment [A, B] = [ref_mesh.vertex_pos(fa),
//        ref_mesh.vertex_pos(fb)] and record the squared distance.
//     3. If the closest projection is within snap_distance, measure the acute
//        angle θ between D and the feature tangent T = (B - A) / |B - A|.
//        θ = acos(|D · T|)  (unsigned angle, since alignment is orientation-free)
//     4. Accumulate sum and max of θ over all such near-feature edges.
// ---------------------------------------------------------------------------

QualityMetrics compute_quality_metrics(
    const QuadMesh&      qm,
    const FeatureData*   features,
    const HalfEdgeMesh*  ref_mesh,
    int32_t              target_quads,
    double               snap_distance,
    int                  num_threads)
{
    // Run all base metrics first.
    QualityMetrics m = compute_quality_metrics(qm, features,
                                               target_quads, snap_distance,
                                               num_threads);

    // Feature alignment requires both features and ref_mesh.
    if (!features || features->hard_edges.empty() ||
        !ref_mesh  || snap_distance <= 0.0)
        return m;

    const int nv = (int)qm.vertices.size();
    const int ne = (int)features->hard_edges.size();
    if (nv == 0 || ne == 0) return m;

    // Pre-cache feature edge endpoints (positions from original mesh).
    struct FeatEdge { Vec3 a, b, tangent; };
    std::vector<FeatEdge> fedges;
    fedges.reserve(static_cast<size_t>(ne));
    for (const auto& [fa, fb] : features->hard_edges) {
        if (fa < 0 || fb < 0 ||
            fa >= ref_mesh->num_vertices() ||
            fb >= ref_mesh->num_vertices())
            continue;
        Vec3 a = ref_mesh->vertex_pos(fa);
        Vec3 b = ref_mesh->vertex_pos(fb);
        double len = (b - a).norm();
        if (len < 1e-15) continue;
        fedges.push_back({ a, b, (b - a) / len });
    }
    if (fedges.empty()) return m;

    const double snap_dist_sq = snap_distance * snap_distance;
    const int nfe = (int)fedges.size();

    // Per-quad-edge alignment accumulator.
    struct AlignAcc {
        double sum_angle{0.0};
        double max_angle{0.0};
        int    count{0};
    };

    AlignAcc global = parallel_reduce<AlignAcc>(
        (int)qm.quads.size(),
        AlignAcc{},
        [&](int fi, AlignAcc& acc) {
            const auto& q = qm.quads[fi];
            for (int k = 0; k < 4; ++k) {
                int vi = q[k];
                int vj = q[(k + 1) % 4];
                if (vi < 0 || vj < 0 || vi >= nv || vj >= nv) continue;

                Vec3 p0 = { qm.vertices[vi][0],
                            qm.vertices[vi][1],
                            qm.vertices[vi][2] };
                Vec3 p1 = { qm.vertices[vj][0],
                            qm.vertices[vj][1],
                            qm.vertices[vj][2] };
                Vec3 mid      = (p0 + p1) * 0.5;
                Vec3 edge_dir = p1 - p0;
                double elen   = edge_dir.norm();
                if (elen < 1e-15) continue;
                edge_dir /= elen;

                // Find the nearest feature edge to this quad edge's midpoint.
                double best_d2     = std::numeric_limits<double>::max();
                double best_angle  = 0.0;

                for (int ei = 0; ei < nfe; ++ei) {
                    const Vec3& a  = fedges[ei].a;
                    // BUG-FIX (v114): was `const Vec3& ab = fedges[ei].b - a`
                    // which bound a const-reference to a temporary (fedges[ei].b - a).
                    // C++ extends the lifetime of a temporary bound to a const ref,
                    // but this is fragile style. Changed to a named value copy.
                    Vec3 ab        = fedges[ei].b - a;
                    double len_sq  = ab.squaredNorm();
                    if (len_sq < 1e-20) continue;

                    double t     = std::clamp((mid - a).dot(ab) / len_sq, 0.0, 1.0);
                    Vec3   proj  = a + t * ab;
                    double d2    = (mid - proj).squaredNorm();

                    if (d2 < best_d2) {
                        best_d2 = d2;
                        // Acute angle between quad edge direction and feature tangent.
                        // Use |cos θ| = |D · T| so orientation doesn't matter.
                        double dot = std::abs(edge_dir.dot(fedges[ei].tangent));
                        dot        = std::min(dot, 1.0); // clamp for acos safety
                        best_angle = std::acos(dot) * (180.0 / M_PI);
                    }
                }

                if (best_d2 <= snap_dist_sq) {
                    acc.sum_angle += best_angle;
                    if (best_angle > acc.max_angle) acc.max_angle = best_angle;
                    acc.count++;
                }
            }
        },
        [](AlignAcc a, AlignAcc b) -> AlignAcc {
            return { a.sum_angle + b.sum_angle,
                     std::max(a.max_angle, b.max_angle),
                     a.count + b.count };
        },
        num_threads);

    if (global.count > 0) {
        m.mean_feature_alignment_error_deg =
            float(global.sum_angle / global.count);
        m.max_feature_alignment_error_deg  = float(global.max_angle);
    }

    return m;
}

} // namespace qf
