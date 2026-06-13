/**
 * snap.cpp — Snap quad mesh vertices to feature curves.
 *
 * Bug-fixes:
 *   - (previous) Added parallel_for over vertices (embarrassingly parallel —
 *     each vertex finds its own closest feature point independently).
 *   - (previous) Pre-cache all feature edge endpoint positions so the inner
 *     loop does not repeatedly call ref_mesh.vertex_pos() inside the parallel
 *     region.
 *   - (previous) Added early-exit guard when features or snap_distance are
 *     trivial.
 *   - (v77) Added num_threads parameter and forwarded it to parallel_for.
 *   - (v77) Implemented adaptive_snap mode: when enabled, the per-vertex snap
 *     threshold is computed as snap_distance_scale * local_avg_edge_length.
 *     This satisfies the roadmap requirement (Step 5.3) that the threshold is
 *     "proportional to the local edge length," making snapping
 *     resolution-independent across differently-sized quad meshes.
 *   - (v78) Added bounds check on lo/hi vertex indices before every call to
 *     ref_mesh.vertex_pos().  A stale or corrupt FeatureData can carry
 *     out-of-range indices; without the check this was undefined behaviour
 *     (out-of-bounds std::vector access).  Invalid edges are written as
 *     degenerate (a == b) so the inner loop's len_sq < 1e-20 guard skips
 *     them safely — same pattern metrics.cpp has always applied.
 *   - (v115) BUG-E: Invalid-edge fallback used Vec3(0.f, 0.f, 0.f) which passes float literals
 *     into the engine-wide double-precision Vec3 constructor.  Replaced with Vec3::Zero().
 *   - (v81) Fixed build_avg_edge_lengths: interior edges (shared by 2 quads)
 *     were counted twice while boundary edges (shared by 1 quad) were counted
 *     once, biasing the adaptive snap threshold for boundary vertices upward.
 *     A boundary vertex with 1 boundary edge (length a) and 2 interior edges
 *     (length b) previously produced avg=(a+4b)/5 instead of (a+2b)/3.
 *     Fix: insert each undirected edge key into an unordered_set<int64_t>
 *     before accumulating; skip the edge if already seen, so every unique
 *     edge contributes exactly once regardless of how many faces share it.
 *
 *     Algorithm for adaptive snap:
 *       1. Build a per-vertex average-edge-length table in O(E) time using
 *          one serial pass over all quad faces.  Each edge (vi, vj) contributes
 *          its Euclidean length to both vi's and vj's accumulators.
 *       2. In the parallel snap loop, substitute:
 *              snap_dist_sq  →  (scale * avg_edge_length[vi])²
 *          so every vertex has its own adaptive threshold.
 */

#include "../../include/quadforge/postprocess/snap.h"
#include "../../include/quadforge/accel/omp_utils.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <cstdint>

namespace qf {

// ---------------------------------------------------------------------------
// Build per-vertex average incident edge length table (O(E), serial).
//
// Returns a vector of length nv where entry [vi] = mean of all edge lengths
// incident to vertex vi.  Vertices with no incident edges get value 1.0 as
// a safe fallback (avoids zero threshold).
//
// BUG-FIX (v81): The previous implementation called add_edge once per
// DIRECTED edge in the face list, with no deduplication.  In a manifold
// quad mesh every INTERIOR undirected edge (vi, vj) appears in exactly two
// quad faces — once traversed as vi→vj and once as vj→vi — so it was counted
// TWICE.  BOUNDARY undirected edges appear in only one quad face and were
// counted ONCE.  For a boundary vertex with one boundary edge (length a) and
// two interior edges (length b each), this produced:
//
//   edge_cnt[vi] = 1*1 + 2*2 = 5
//   edge_sum[vi] = 1*a + 2*2*b = a + 4b
//   avg = (a + 4b) / 5          ← WRONG
//
// The correct result is: (a + b + b) / 3 = (a + 2b) / 3.
//
// Interior vertices are not affected: all of their incident edges appear in
// exactly two quads, so numerator and denominator are both doubled and the
// mean is preserved.  The bug only biased the adaptive snap threshold for
// BOUNDARY vertices — the exact vertices most likely to be snapped to
// feature curves — causing them to be snapped more aggressively than intended.
//
// Fix: insert each undirected edge key into an unordered_set<int64_t> before
// accumulating; skip the edge if it has already been seen.  This ensures
// every unique edge contributes exactly once to each endpoint's accumulator,
// regardless of how many faces share it.
// ---------------------------------------------------------------------------
static std::vector<double>
build_avg_edge_lengths(const QuadMesh& qm)
{
    const int nv = static_cast<int>(qm.vertices.size());
    std::vector<double> edge_sum(static_cast<size_t>(nv), 0.0);
    std::vector<int>    edge_cnt(static_cast<size_t>(nv), 0);

    // Canonical undirected-edge key: lo vertex index in high 32 bits, hi in low.
    auto edge_key = [](int a, int b) -> int64_t {
        if (a > b) { int t = a; a = b; b = t; }
        return (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
    };

    // seen: one entry per unique undirected edge.  Each edge is accumulated
    // exactly once regardless of how many faces share it.
    std::unordered_set<int64_t> seen;
    seen.reserve(qm.quads.size() * 4 + qm.tris.size() * 3);

    auto add_edge = [&](int vi, int vj) {
        if (vi < 0 || vj < 0 || vi >= nv || vj >= nv) return;
        // Skip if this undirected edge has already been accumulated.
        if (!seen.insert(edge_key(vi, vj)).second) return;
        const double dx = qm.vertices[vj][0] - qm.vertices[vi][0];
        const double dy = qm.vertices[vj][1] - qm.vertices[vi][1];
        const double dz = qm.vertices[vj][2] - qm.vertices[vi][2];
        const double len = std::sqrt(dx*dx + dy*dy + dz*dz);
        edge_sum[vi] += len;  edge_cnt[vi]++;
        edge_sum[vj] += len;  edge_cnt[vj]++;
    };

    for (const auto& q : qm.quads) {
        for (int k = 0; k < 4; ++k)
            add_edge(q[k], q[(k + 1) % 4]);
    }
    for (const auto& t : qm.tris) {
        for (int k = 0; k < 3; ++k)
            add_edge(t[k], t[(k + 1) % 3]);
    }

    std::vector<double> avg(static_cast<size_t>(nv));
    for (int vi = 0; vi < nv; ++vi)
        avg[vi] = (edge_cnt[vi] > 0)
                  ? (edge_sum[vi] / edge_cnt[vi])
                  : 1.0;   // fallback: isolated vertex — unit threshold
    return avg;
}

// ---------------------------------------------------------------------------
// snap_to_features
// ---------------------------------------------------------------------------

void snap_to_features(
    QuadMesh&           quad_mesh,
    const HalfEdgeMesh& ref_mesh,
    const FeatureData&  features,
    double              snap_distance,
    bool                adaptive_snap,
    int                 num_threads)
{
    if (features.hard_edges.empty() || snap_distance <= 0.0) return;

    const int nv = static_cast<int>(quad_mesh.vertices.size());
    if (nv == 0) return;

    // ------------------------------------------------------------------
    // Pre-cache feature edge endpoint positions to avoid repeated virtual
    // dispatch / hash-map lookups inside the hot parallel loop.
    // ------------------------------------------------------------------
    const int ne = static_cast<int>(features.hard_edges.size());
    struct FeatureEdge { Vec3 a, b; };
    std::vector<FeatureEdge> fedges(static_cast<size_t>(ne));
    // BUG-FIX (v78): lo and hi are vertex indices into ref_mesh.  A stale or
    // corrupt FeatureData can carry out-of-range indices; calling vertex_pos()
    // on them is undefined behaviour (out-of-bounds vector access).
    // metrics.cpp correctly guards the same pattern (lines ~394-400).
    // Fix: validate lo and hi before every call to vertex_pos().  Invalid
    // edges are written as degenerate (a == b == origin) so that the inner
    // parallel loop's len_sq < 1e-20 guard skips them safely.
    const int ref_nv = ref_mesh.num_vertices();
    for (int ei = 0; ei < ne; ++ei) {
        auto [lo, hi] = features.hard_edges[ei];
        if (lo < 0 || lo >= ref_nv || hi < 0 || hi >= ref_nv) {
            fedges[ei].a = fedges[ei].b = Vec3::Zero(); // BUG-E FIX (v115): was Vec3(0.f,0.f,0.f) — float literals in double Vec3 ctor
            continue;
        }
        fedges[ei].a = ref_mesh.vertex_pos(lo);
        fedges[ei].b = ref_mesh.vertex_pos(hi);
    }

    // ------------------------------------------------------------------
    // Adaptive mode: build per-vertex average edge length table.
    // Fixed mode:    compute global snap_dist_sq once.
    // ------------------------------------------------------------------
    std::vector<double> avg_edge_len;
    double              fixed_snap_dist_sq = 0.0;

    if (adaptive_snap) {
        avg_edge_len = build_avg_edge_lengths(quad_mesh);
    } else {
        fixed_snap_dist_sq = snap_distance * snap_distance;
    }

    // ------------------------------------------------------------------
    // Parallel snap: each vertex independently searches all feature edges.
    // v77: forwarded num_threads to parallel_for.
    // ------------------------------------------------------------------
    parallel_for(nv, [&](int vi) {
        Vec3 p = { quad_mesh.vertices[vi][0],
                   quad_mesh.vertices[vi][1],
                   quad_mesh.vertices[vi][2] };

        // Per-vertex threshold (adaptive) or global threshold (fixed).
        const double threshold_sq = adaptive_snap
            ? (snap_distance * avg_edge_len[vi])
              * (snap_distance * avg_edge_len[vi])
            : fixed_snap_dist_sq;

        double best_d2 = std::numeric_limits<double>::infinity();
        Vec3   best_pt = p;

        for (int ei = 0; ei < ne; ++ei) {
            const Vec3& a = fedges[ei].a;
            const Vec3& b = fedges[ei].b;
            Vec3  ab      = b - a;
            double len_sq = ab.squaredNorm();
            if (len_sq < 1e-20) continue;

            // Closest point on segment [a, b] to p.
            double t    = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
            Vec3   proj = a + t * ab;
            double d2   = (p - proj).squaredNorm();

            if (d2 < best_d2) {
                best_d2 = d2;
                best_pt = proj;
            }
        }

        if (best_d2 <= threshold_sq) {
            quad_mesh.vertices[vi][0] = best_pt[0];
            quad_mesh.vertices[vi][1] = best_pt[1];
            quad_mesh.vertices[vi][2] = best_pt[2];
        }
    }, num_threads);   // v77: explicit thread count (0 = auto)
}

} // namespace qf
