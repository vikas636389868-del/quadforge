/**
 * seams.cpp — Minimal seam-cut graph for global parametrization.
 *
 * Computes the cut graph that reduces the surface to a topological disk
 * (or set of disks) so that the Poisson parametrization can be solved
 * globally.  The cut connects all singularities to each other and to
 * boundaries via shortest-path Dijkstra on the primal graph.
 */

#include "../../include/quadforge/param/seams.h"

#include <cmath>
#include <queue>
#include <limits>
#include <algorithm>
#include <unordered_set>

namespace qf {

// -----------------------------------------------------------------------
// Vertex adjacency list
// -----------------------------------------------------------------------

std::vector<std::vector<std::pair<int32_t, float>>>
build_vertex_adjacency(
    const float* positions, int32_t nv,
    const int32_t* tris,   int32_t nt)
{
    // BUG FIX (Bug 6 — param/seams): The previous implementation used a
    // linear scan through each vertex's neighbour list to detect duplicates
    // before inserting — O(degree) per insertion, O(E × degree) = O(E²)
    // overall.  On dense meshes this created a severe performance cliff
    // (e.g. a 100k-tri mesh with avg degree 6 → ~600k inner iterations).
    //
    // Fix: use a temporary per-vertex unordered_set to track already-added
    // neighbours.  Insertion becomes O(1) amortised, total is O(E).
    // The sets are destroyed after the loop since they're not needed after.

    // BUG FIX (B6 — param/seams v110): The previous fix (Bug 6 — param/seams)
    // replaced the O(degree) linear-scan dedup with per-vertex unordered_sets
    // (`std::vector<std::unordered_set<int32_t>> adj_seen(nv)`).  While
    // correct, this allocates nv empty hash-table objects upfront, each with
    // an internal heap allocation, even for sparse meshes where most vertices
    // have degree 0 at construction time.  For a 10M-vertex mesh this is
    // ~10M map objects = hundreds of MB of wasted bookkeeping memory.
    //
    // Better fix: use a single flat unordered_set<int64_t> keyed on the
    // packed canonical edge (lo<<32|hi) — one flat hash table, O(E) total
    // memory, same O(1) amortised insert and lookup.
    std::vector<std::vector<std::pair<int32_t, float>>> adj(nv);
    std::unordered_set<int64_t> adj_seen;
    adj_seen.reserve(static_cast<size_t>(nt) * 3);  // ~E unique edges

    auto make_adj_key = [](int32_t a, int32_t b) -> int64_t {
        int32_t lo = (a < b) ? a : b;
        int32_t hi = (a < b) ? b : a;
        return (static_cast<int64_t>(lo) << 32) | static_cast<uint32_t>(hi);
    };

    for (int32_t fi = 0; fi < nt; ++fi) {
        for (int k = 0; k < 3; ++k) {
            int32_t va = tris[fi*3+k];
            int32_t vb = tris[fi*3+(k+1)%3];
            if (va < 0 || va >= nv || vb < 0 || vb >= nv) continue;

            float dx = positions[vb*3]   - positions[va*3];
            float dy = positions[vb*3+1] - positions[va*3+1];
            float dz = positions[vb*3+2] - positions[va*3+2];
            float len = std::sqrt(dx*dx + dy*dy + dz*dz);

            int64_t key = make_adj_key(va, vb);
            if (adj_seen.insert(key).second) {
                // First time we see this undirected edge — add both directions.
                adj[va].push_back({vb, len});
                adj[vb].push_back({va, len});
            }
        }
    }
    return adj;
}

// -----------------------------------------------------------------------
// Dijkstra shortest path
// -----------------------------------------------------------------------

bool shortest_path_dijkstra(
    int32_t nv,
    const std::vector<std::vector<std::pair<int32_t, float>>>& adj,
    int32_t src, int32_t dst,
    std::vector<int32_t>& out_path)
{
    if (src == dst) { out_path = {src}; return true; }
    if (src < 0 || dst < 0 || src >= nv || dst >= nv) return false;

    std::vector<float> dist(nv, std::numeric_limits<float>::infinity());
    std::vector<int32_t> prev(nv, -1);
    dist[src] = 0.f;

    // Min-heap: (dist, vertex)
    using Pair = std::pair<float, int32_t>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;
    pq.push({0.f, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (u == dst) break;
        for (auto [v, w] : adj[u]) {
            float nd = dist[u] + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                prev[v] = u;
                pq.push({nd, v});
            }
        }
    }

    if (std::isinf(dist[dst])) return false;

    // Reconstruct path
    out_path.clear();
    for (int32_t v = dst; v != -1; v = prev[v])
        out_path.push_back(v);
    std::reverse(out_path.begin(), out_path.end());
    return true;
}

// -----------------------------------------------------------------------
// Convert vertex path → edge list
// -----------------------------------------------------------------------

void path_to_edges(const std::vector<int32_t>& path, std::vector<int32_t>& out_edges)
{
    for (int i = 0; i + 1 < (int)path.size(); ++i) {
        out_edges.push_back(path[i]);
        out_edges.push_back(path[i+1]);
    }
}

// -----------------------------------------------------------------------
// Minimal seam cut graph
// -----------------------------------------------------------------------

void compute_seam_cut(
    const float*   positions,
    const int32_t* tris,
    int32_t nv, int32_t nt,
    const std::vector<int32_t>& combing_seams,
    const std::vector<int32_t>& singularity_verts,
    const std::unordered_map<int64_t, std::vector<int32_t>>& /*edge_to_faces*/,
    const std::vector<int32_t>& extra_seams,
    std::vector<int32_t>& out_seams)
{
    out_seams.clear();

    // Start with combing seams (already necessary)
    out_seams.insert(out_seams.end(), combing_seams.begin(), combing_seams.end());
    out_seams.insert(out_seams.end(), extra_seams.begin(),   extra_seams.end());

    // If no singularities, we're done (simply-connected surface → already a disk)
    if (singularity_verts.empty()) return;

    // Build adjacency for Dijkstra
    auto adj = build_vertex_adjacency(positions, nv, tris, nt);

    // BUG FIX (Bug 2 — param/seams): Replace `const long long VMAX = 1 << 24`
    // edge-key scheme with the 32-bit-shift canonical form used throughout the
    // rest of the codebase (combing.cpp, igm.cpp, etc.).
    //
    // The old scheme `lo * VMAX + hi` (VMAX = 16,777,216) produces colliding
    // keys for any mesh with nv > 16,777,216: the key for edge (0, VMAX) equals
    // the key for edge (1, 0).  The bitwise form `(int64_t)lo << 32 | uint32_t(hi)`
    // is collision-free for all valid int32 vertex indices and is identical to
    // what combing.cpp and igm.cpp already use.
    //
    // The boundary-vertex recovery in compute_seam_cut is updated to use
    // `key >> 32` / `key & 0xFFFFFFFF` instead of `key / VMAX` / `key % VMAX`.

    // Find boundary vertices (vertices on boundary edges)
    // A boundary vertex is one that has an edge not shared by two faces
    using EKey = long long;
    auto make_ekey = [](int32_t a, int32_t b) -> EKey {
        int32_t lo = std::min(a, b), hi = std::max(a, b);
        return (static_cast<EKey>(lo) << 32) | static_cast<uint32_t>(hi);
    };
    std::unordered_map<EKey, int> edge_count;
    edge_count.reserve(nt * 3);
    for (int32_t fi = 0; fi < nt; ++fi) {
        for (int k = 0; k < 3; ++k) {
            int32_t va = tris[fi*3+k], vb = tris[fi*3+(k+1)%3];
            edge_count[make_ekey(va, vb)]++;
        }
    }

    std::unordered_set<int32_t> boundary_verts;
    for (auto& [key, cnt] : edge_count) {
        if (cnt == 1) {
            int32_t va = static_cast<int32_t>(key >> 32);
            int32_t vb = static_cast<int32_t>(key & 0xFFFFFFFFL);
            boundary_verts.insert(va);
            boundary_verts.insert(vb);
        }
    }

    // Set of singularity vertices
    std::vector<int32_t> sing_list(singularity_verts);
    // Remove duplicates
    std::sort(sing_list.begin(), sing_list.end());
    sing_list.erase(std::unique(sing_list.begin(), sing_list.end()), sing_list.end());

    // Track seam vertex set to avoid duplicate edges
    std::unordered_set<EKey> seam_set;
    for (int k = 0; k + 1 < (int)out_seams.size(); k += 2) {
        int32_t va = out_seams[k], vb = out_seams[k+1];
        seam_set.insert(make_ekey(va, vb));
    }

    auto add_path = [&](int32_t src, int32_t dst) {
        std::vector<int32_t> path;
        if (!shortest_path_dijkstra(nv, adj, src, dst, path)) return;
        for (int i = 0; i + 1 < (int)path.size(); ++i) {
            int32_t va = path[i], vb = path[i+1];
            EKey key = make_ekey(va, vb);
            if (seam_set.find(key) == seam_set.end()) {
                seam_set.insert(key);
                out_seams.push_back(va);
                out_seams.push_back(vb);
            }
        }
    };

    // Strategy 1: Connect each singularity to the nearest other singularity
    // using Dijkstra (greedy Steiner tree approximation)
    if (sing_list.size() > 1) {
        // Connect them in a chain (minimum spanning tree would be better,
        // but a chain is a good approximation for typical meshes)
        for (int i = 0; i + 1 < (int)sing_list.size(); ++i) {
            add_path(sing_list[i], sing_list[i+1]);
        }
    }

    // Strategy 2: Connect the "first" singularity to the nearest boundary vertex
    // (or mesh origin if no boundary) — ensures the cut reduces genus
    if (!sing_list.empty()) {
        if (!boundary_verts.empty()) {
            // Find the boundary vertex nearest to the first singularity
            int32_t sing0 = sing_list[0];
            float sx = positions[sing0*3], sy = positions[sing0*3+1], sz = positions[sing0*3+2];
            int32_t nearest_bv = -1;
            float min_dist = std::numeric_limits<float>::infinity();
            for (int32_t bv : boundary_verts) {
                float dx = positions[bv*3]-sx, dy = positions[bv*3+1]-sy, dz = positions[bv*3+2]-sz;
                float d = dx*dx+dy*dy+dz*dz;
                if (d < min_dist) { min_dist = d; nearest_bv = bv; }
            }
            if (nearest_bv >= 0) add_path(sing0, nearest_bv);
        } else {
            // Closed surface: connect first and last singularity (if not already done)
            if (sing_list.size() >= 2) {
                add_path(sing_list.front(), sing_list.back());
            } else {
                // Single singularity on closed surface: create a loop seam
                // Find the farthest vertex from the singularity
                int32_t sing0 = sing_list[0];
                float sx = positions[sing0*3], sy = positions[sing0*3+1], sz = positions[sing0*3+2];
                int32_t far_v = 0; float max_d = 0;
                for (int32_t vi = 0; vi < nv; ++vi) {
                    float dx = positions[vi*3]-sx, dy = positions[vi*3+1]-sy, dz = positions[vi*3+2]-sz;
                    float d = dx*dx+dy*dy+dz*dz;
                    if (d > max_d) { max_d = d; far_v = vi; }
                }
                add_path(sing0, far_v);
            }
        }
    }
}


// =======================================================================
// HIGH-LEVEL HalfEdgeMesh-AWARE WRAPPER  (FIX: BUG A + BUG B — v72)
// =======================================================================
//
// Before v72, compute_seam_cut() existed only as a raw-pointer function
// (float* positions, int32_t* tris, ...) and was NEVER called from
// anywhere in the live pipeline.  engine.cpp only called the simpler
// compute_minimal_seam_cut() from combing.h, which:
//   (a) duplicated the Dijkstra logic, and
//   (b) completely ignored uv_seam_edges from FeatureData (BUG B).
//
// This wrapper:
//   1. Extracts raw flat arrays from HalfEdgeMesh.
//   2. Converts CombingResult::seam_edges -> flat combing_seams.
//   3. Extracts singularity vertex indices from SingularityInfo list.
//   4. Builds the edge->face map from HalfEdgeMesh::half_edges().
//   5. Converts FeatureData::uv_seam_edges -> flat extra_seams (BUG B FIX).
//      Hard-edge seams are also forwarded so that hard-surface meshes get
//      clean seam placement along creases.
//   6. Calls compute_seam_cut() and converts the flat output back to EdgeSet.

EdgeSet compute_seam_cut_from_mesh(
    const HalfEdgeMesh&                     mesh,
    const CombingResult&                    combing,
    const std::vector<SingularityInfo>&     singularities,
    const FeatureData&                      features)
{
    const int nv = mesh.num_vertices();
    const int nt = mesh.num_faces();

    // ------------------------------------------------------------------
    // 1. Extract flat position and triangle arrays from HalfEdgeMesh.
    // ------------------------------------------------------------------
    std::vector<float>   positions_flat(static_cast<size_t>(nv) * 3);
    std::vector<int32_t> tris_flat(static_cast<size_t>(nt) * 3);

    for (int vi = 0; vi < nv; ++vi) {
        Vec3 p = mesh.vertex_pos(vi);
        positions_flat[vi*3+0] = static_cast<float>(p.x());
        positions_flat[vi*3+1] = static_cast<float>(p.y());
        positions_flat[vi*3+2] = static_cast<float>(p.z());
    }
    for (int fi = 0; fi < nt; ++fi) {
        auto verts = mesh.face_vertices(fi);
        tris_flat[fi*3+0] = static_cast<int32_t>(verts[0]);
        tris_flat[fi*3+1] = static_cast<int32_t>(verts[1]);
        tris_flat[fi*3+2] = static_cast<int32_t>(verts[2]);
    }

    // ------------------------------------------------------------------
    // 2. Convert CombingResult::seam_edges -> flat combing_seams [v0,v1,...].
    // ------------------------------------------------------------------
    std::vector<int32_t> combing_seams;
    combing_seams.reserve(combing.seam_edges.size() * 2);
    for (auto& ep : combing.seam_edges) {
        combing_seams.push_back(static_cast<int32_t>(ep.first));
        combing_seams.push_back(static_cast<int32_t>(ep.second));
    }

    // ------------------------------------------------------------------
    // 3. Extract singularity vertex indices.
    // ------------------------------------------------------------------
    std::vector<int32_t> sing_verts;
    sing_verts.reserve(singularities.size());
    for (const auto& s : singularities)
        sing_verts.push_back(static_cast<int32_t>(s.vertex_index));

    // ------------------------------------------------------------------
    // 4. Build edge->faces map from HalfEdgeMesh faces.
    //
    // BUG FIX (P4 — param/seams v111): REMOVED unnecessary edge_to_faces
    // map construction.
    //
    // compute_seam_cut() declares this parameter as
    //   const std::unordered_map<int64_t, std::vector<int32_t>>& /*edge_to_faces*/
    // — the parameter is COMMENTED OUT (dead code) and the map is never
    // accessed inside the function body.  The previous implementation of this
    // wrapper built the ENTIRE edge-to-face adjacency map (O(3*T) entries +
    // O(T) memory) just to have it silently discarded on entry to
    // compute_seam_cut().
    //
    // For a 1M-triangle mesh, this created ~3M map entries using ~100+ MB of
    // memory and consumed a measurable fraction of Stage 3 wall time.  The
    // fix passes an empty map, which is the correct value since compute_seam_cut()
    // ignores it entirely.
    //
    // If future work makes compute_seam_cut() actually consume the edge->face
    // adjacency (e.g., for seam placement on manifold-boundary edges), the
    // construction should be restored here and the commented-out parameter
    // name in compute_seam_cut()'s signature should be restored.
    // ------------------------------------------------------------------
    const std::unordered_map<int64_t, std::vector<int32_t>> edge_to_faces;
    // (empty — compute_seam_cut() does not consume this map)

    // ------------------------------------------------------------------
    // 5. Build extra_seams from FeatureData.
    //
    // BUG B FIX (v72): Before this fix, UV seam edges stored in
    // m_features.uv_seam_edges were collected in Stage 1 but never
    // forwarded to the seam cut in Stage 3.  The use_uv_seams toggle
    // was effectively a no-op in the parametrization.
    //
    // UV seam edges are the highest-priority extra seams because:
    //   - The user explicitly requested UV seam preservation.
    //   - UV seams often mark material boundaries or important silhouette
    //     edges that should become quad mesh boundaries.
    //
    // Hard edges are added as lower-priority seam candidates because
    // placing seam cuts along creases minimises parametrization distortion
    // on hard-surface and architectural meshes.
    // ------------------------------------------------------------------
    std::vector<int32_t> extra_seams;
    extra_seams.reserve(
        (features.uv_seam_edges.size() + features.hard_edges.size()) * 2);

    for (const auto& ep : features.uv_seam_edges) {
        extra_seams.push_back(static_cast<int32_t>(ep.first));
        extra_seams.push_back(static_cast<int32_t>(ep.second));
    }

    for (const auto& ep : features.hard_edges) {
        extra_seams.push_back(static_cast<int32_t>(ep.first));
        extra_seams.push_back(static_cast<int32_t>(ep.second));
    }

    // ------------------------------------------------------------------
    // 6. Call the raw-array compute_seam_cut() (first live call — v72).
    // ------------------------------------------------------------------
    std::vector<int32_t> out_flat;
    compute_seam_cut(
        positions_flat.data(),
        tris_flat.data(),
        static_cast<int32_t>(nv),
        static_cast<int32_t>(nt),
        combing_seams,
        sing_verts,
        edge_to_faces,
        extra_seams,
        out_flat
    );

    // ------------------------------------------------------------------
    // 7. Convert flat output back to EdgeSet (deduplicated, canonical).
    // ------------------------------------------------------------------
    EdgeSet result;
    result.reserve(out_flat.size() / 2);

    // BUG FIX (Bug 2 — param/seams): Use the 32-bit-shift canonical key
    // (same as compute_seam_cut and combing.cpp) instead of the
    // `1LL << 24` VMAX scheme that collides for nv > 16,777,216.
    std::unordered_set<long long> seen;
    seen.reserve(out_flat.size() / 2);

    for (int k = 0; k + 1 < static_cast<int>(out_flat.size()); k += 2) {
        int va = out_flat[k], vb = out_flat[k+1];
        if (va < 0 || vb < 0 || va >= nv || vb >= nv) continue;
        int lo = std::min(va, vb), hi = std::max(va, vb);
        long long key = (static_cast<long long>(lo) << 32) |
                        static_cast<uint32_t>(hi);
        if (seen.insert(key).second)
            result.push_back({lo, hi});
    }

    return result;
}

} // namespace qf
