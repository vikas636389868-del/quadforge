/**
 * geodesic.cpp — Geodesic distance computation on triangle meshes.
 *
 * Implements the declarations in geodesic.h using a Dijkstra-based
 * graph-geodesic approximation on the half-edge adjacency structure.
 *
 * Edge weights: Euclidean 3-D distance between adjacent vertex positions,
 * raised to opts.length_exponent (1.0 by default = plain distance).
 *
 * The approximation is O((V + E) log V) per call with early-exit when
 * opts.max_dist > 0.  It uses std::priority_queue with std::greater for
 * the min-heap.
 *
 * See geodesic.h for full API documentation and design notes.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#include "../../include/quadforge/mesh/geodesic.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

namespace qf {

// =========================================================================
// Internal helpers
// =========================================================================

namespace {

using DistVec = std::vector<double>;

constexpr double kInf = std::numeric_limits<double>::max();

/**
 * Min-heap entry: (tentative distance, vertex index).
 * Using > for the comparator gives us a min-heap (smallest dist at top).
 */
struct PQEntry {
    double dist;
    int    vertex;
    bool operator>(const PQEntry& o) const { return dist > o.dist; }
};

/**
 * Core Dijkstra propagation.  Seeds must have their distances pre-set in dist[].
 * All other entries in dist[] must be kInf before calling.
 *
 * @param mesh            The half-edge mesh.
 * @param dist            Distance array (modified in-place).
 * @param pq              Priority queue (pre-populated with seed entries).
 * @param max_dist        Stop when smallest queue distance exceeds this.  0 = no cap.
 * @param edge_exp        Edge-length exponent (1.0 = plain distance).
 * @param cross_boundary  If false, skip relaxation across boundary edges (edges
 *                        whose twin == -1 in the half-edge structure).  This
 *                        matches the GeodesicOptions::cross_boundary semantics
 *                        declared in geodesic.h.
 *
 * BUG FIX (v106): The parameter was previously named `exp`, which shadows
 * the standard-library function std::exp (introduced via <cmath>).  While no
 * call to std::exp exists inside this function today, the shadowing is a
 * latent hazard: any future maintainer adding e.g. a heat-method weight
 * `std::exp(-t)` would silently call the double parameter instead of the
 * math function, producing a nonsensical result without a compiler warning in
 * all common build configurations.  Renamed to `edge_exp` to match the
 * intent expressed in the parameter documentation above.
 *
 * BUG FIX (v107): The GeodesicOptions::cross_boundary field was declared in
 * geodesic.h and documented as controlling whether propagation crosses mesh
 * boundary edges, but it was NEVER passed into or checked inside dijkstra().
 * vertex_neighbours() always traverses all adjacent vertices regardless of
 * edge type, so setting cross_boundary=false had zero effect — the function
 * always propagated across boundary edges.
 *
 * Fix: accept a cross_boundary parameter.  When false, before relaxing edge
 * (v→w), check mesh.is_boundary_edge(v, w).  If the edge is a boundary edge
 * (only one half-edge, i.e. twin==-1), skip relaxation.  The check is O(log E)
 * per edge via the half-edge structure's edge_map lookup — acceptable for the
 * sparse boundary cases where cross_boundary=false is meaningful.
 *
 * The default (cross_boundary=true) path incurs no overhead from this change.
 */
void dijkstra(
    const HalfEdgeMesh& mesh,
    DistVec&             dist,
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>>& pq,
    double               max_dist,
    double               edge_exp,
    bool                 cross_boundary = true)
{
    const bool use_cap = (max_dist > 0.0);

    while (!pq.empty()) {
        auto [d, v] = pq.top();
        pq.pop();

        // Skip stale entries (lazy deletion).
        if (d > dist[v]) continue;

        // Early exit: every remaining entry has dist ≥ d > max_dist.
        if (use_cap && d > max_dist) break;

        // Relax all edges from v.
        mesh.vertex_neighbours(v, [&](int w) {
            // BUG FIX (v107): honour cross_boundary=false by skipping
            // relaxation on edges that have no twin (boundary edges).
            if (!cross_boundary && mesh.is_boundary_edge(v, w)) return;

            Vec3 pv = mesh.vertex_pos(v);
            Vec3 pw = mesh.vertex_pos(w);
            double edge_len = (pw - pv).norm();
            if (edge_exp != 1.0) edge_len = std::pow(edge_len, edge_exp);

            double nd = d + edge_len;
            if (use_cap && nd > max_dist) return;  // prune
            if (nd < dist[w]) {
                dist[w] = nd;
                pq.push({nd, w});
            }
        });
    }
}

/**
 * Build a min-heap and distance vector pre-seeded from a seed list.
 */
std::pair<DistVec,
          std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>>>
make_seeded_state(int nv, const std::vector<int>& seeds)
{
    DistVec dist(nv, kInf);
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;

    for (int s : seeds) {
        if (s < 0 || s >= nv) continue;
        if (dist[s] > 0.0) {
            dist[s] = 0.0;
            pq.push({0.0, s});
        }
    }
    return {std::move(dist), std::move(pq)};
}

} // anonymous namespace

// =========================================================================
// compute_geodesic_distances — single seed set
// =========================================================================

std::vector<double> compute_geodesic_distances(
    const HalfEdgeMesh&     mesh,
    const std::vector<int>& seeds,
    const GeodesicOptions&  opts)
{
    const int nv = mesh.num_vertices();
    if (nv == 0 || seeds.empty()) return std::vector<double>(nv, kInf);

    auto [dist, pq] = make_seeded_state(nv, seeds);
    dijkstra(mesh, dist, pq, opts.max_dist, opts.length_exponent, opts.cross_boundary);
    return dist;
}

// =========================================================================
// compute_geodesic_distances — two seed sets
// =========================================================================

std::vector<double> compute_geodesic_distances(
    const HalfEdgeMesh&     mesh,
    const std::vector<int>& seeds_a,
    const std::vector<int>& seeds_b,
    const GeodesicOptions&  opts)
{
    const int nv = mesh.num_vertices();
    if (nv == 0) return std::vector<double>(nv, kInf);

    DistVec dist(nv, kInf);
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;

    auto seed_one = [&](int s) {
        if (s < 0 || s >= nv) return;
        if (dist[s] > 0.0) {
            dist[s] = 0.0;
            pq.push({0.0, s});
        }
    };
    for (int s : seeds_a) seed_one(s);
    for (int s : seeds_b) seed_one(s);

    dijkstra(mesh, dist, pq, opts.max_dist, opts.length_exponent, opts.cross_boundary);
    return dist;
}

// =========================================================================
// compute_geodesic_from_edge_set
// =========================================================================

std::vector<double> compute_geodesic_from_edge_set(
    const HalfEdgeMesh&    mesh,
    const EdgeSet&          feature_edges,
    const GeodesicOptions&  opts)
{
    const int nv = mesh.num_vertices();
    if (nv == 0 || feature_edges.empty())
        return std::vector<double>(nv, kInf);

    // Collect all unique vertices incident on any edge in the set.
    std::vector<int> seeds;
    seeds.reserve(feature_edges.size() * 2);
    for (const auto& [v0, v1] : feature_edges) {
        seeds.push_back(v0);
        seeds.push_back(v1);
    }
    // Deduplicate (seeds may be large; sort+unique is cache-friendly).
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());

    return compute_geodesic_distances(mesh, seeds, opts);
}

// =========================================================================
// compute_geodesic_from_features
// =========================================================================

std::vector<double> compute_geodesic_from_features(
    const HalfEdgeMesh&    mesh,
    const FeatureData&     features,
    const GeodesicOptions& opts)
{
    const int nv = mesh.num_vertices();
    if (nv == 0) return std::vector<double>(nv, kInf);

    // Merge all edge sets into a single seed vertex list.
    std::vector<int> seeds;
    seeds.reserve(
        (features.hard_edges.size() +
         features.boundary_edges.size() +
         features.uv_seam_edges.size()) * 2);

    auto push_edge_set = [&](const EdgeSet& es) {
        for (const auto& [v0, v1] : es) {
            seeds.push_back(v0);
            seeds.push_back(v1);
        }
    };
    push_edge_set(features.hard_edges);
    push_edge_set(features.boundary_edges);
    push_edge_set(features.uv_seam_edges);

    // Also add corner vertices as zero-distance seeds.
    for (int cv : features.corner_vertices) {
        seeds.push_back(cv);
    }

    // Deduplicate.
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());

    if (seeds.empty()) return std::vector<double>(nv, kInf);

    return compute_geodesic_distances(mesh, seeds, opts);
}

// =========================================================================
// compute_geodesic_from_boundary
// =========================================================================

std::vector<double> compute_geodesic_from_boundary(
    const HalfEdgeMesh&    mesh,
    const GeodesicOptions& opts)
{
    const int nv = mesh.num_vertices();
    if (nv == 0) return std::vector<double>(nv, kInf);

    // Collect boundary vertices.
    const auto& bnd_edges = mesh.boundary_edges();
    if (bnd_edges.empty()) {
        // Closed mesh — return all-infinity.
        return std::vector<double>(nv, kInf);
    }

    std::vector<int> seeds;
    seeds.reserve(bnd_edges.size() * 2);
    for (const auto& [v0, v1] : bnd_edges) {
        seeds.push_back(v0);
        seeds.push_back(v1);
    }

    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());

    return compute_geodesic_distances(mesh, seeds, opts);
}

// =========================================================================
// vertices_within_geodesic_radius
// =========================================================================

std::vector<int> vertices_within_geodesic_radius(
    const HalfEdgeMesh&    mesh,
    const std::vector<int>& seeds,
    double                  radius)
{
    GeodesicOptions opts;
    opts.max_dist = radius;
    const auto dist = compute_geodesic_distances(mesh, seeds, opts);

    std::vector<int> result;
    result.reserve(mesh.num_vertices() / 4);  // rough estimate
    for (int v = 0; v < static_cast<int>(dist.size()); ++v) {
        if (dist[v] <= radius) result.push_back(v);
    }
    // Result is already sorted (we iterate in ascending v order).
    return result;
}

// =========================================================================
// faces_within_geodesic_radius
// =========================================================================

std::vector<int> faces_within_geodesic_radius(
    const HalfEdgeMesh&    mesh,
    const std::vector<int>& seeds,
    double                  radius)
{
    GeodesicOptions opts;
    opts.max_dist = radius;
    const auto dist = compute_geodesic_distances(mesh, seeds, opts);

    const int nf = mesh.num_faces();
    std::vector<int> result;
    result.reserve(nf / 4);

    for (int fi = 0; fi < nf; ++fi) {
        const auto verts = mesh.face_vertices(fi);
        // Include face only if ALL its vertices are within radius.
        bool inside = true;
        for (int v : verts) {
            if (dist[v] > radius) { inside = false; break; }
        }
        if (inside) result.push_back(fi);
    }
    return result;
}

// =========================================================================
// normalise_geodesic_distances
// =========================================================================

void normalise_geodesic_distances(std::vector<double>& distances)
{
    if (distances.empty()) return;

    // Find maximum finite value.
    double max_val = 0.0;
    for (double d : distances) {
        if (d < kInf && d > max_val) max_val = d;
    }

    if (max_val < 1e-30) {
        // All distances are zero or infinite — map everything to 1.
        std::fill(distances.begin(), distances.end(), 1.0);
        return;
    }

    const double inv_max = 1.0 / max_val;
    for (double& d : distances) {
        d = (d >= kInf) ? 1.0 : d * inv_max;
    }
}

} // namespace qf
