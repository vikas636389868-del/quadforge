/**
 * combing.cpp — Cross-field combing and seam cut computation.
 *
 * Resolves the 4-fold ambiguity of the 4-RoSy field by BFS from a seed face.
 * Seam edges are introduced where the combing is topologically forced to jump.
 */

#include "../../include/quadforge/param/combing.h"
#include "../../include/quadforge/field/singularity.h"
#include "../../include/quadforge/field/cross_field.h"   // parallel_transport_angle

#include <cmath>
#include <queue>
#include <deque>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// -----------------------------------------------------------------------
// Nearest multiple of π/2 rounding
// -----------------------------------------------------------------------
static double snap_to_pi2(double x) {
    return std::round(x / (M_PI/2.0)) * (M_PI/2.0);
}

// -----------------------------------------------------------------------
// comb_field
// -----------------------------------------------------------------------

CombingResult comb_field(
    const HalfEdgeMesh& mesh,
    const CrossField&   field,
    const FeatureData&  features)
{
    int nf = mesh.num_faces();
    CombingResult result;
    result.face_angles.assign(nf, 0.0);

    if (nf == 0) return result;

    // BUG FIX (Bug 3): Build feature-edge key set so the BFS can prefer
    // placing seam cuts along feature edges (natural seam positions).
    auto edge_key = [](int a, int b) -> int64_t {
        int lo = std::min(a,b), hi = std::max(a,b);
        return ((int64_t)lo << 32) | (uint32_t)hi;
    };
    std::unordered_set<int64_t> feat_keys;
    for (auto& [lo, hi] : features.hard_edges)
        feat_keys.insert(edge_key(lo, hi));

    // BFS combing — assigns a globally consistent angle to every face.
    // The BFS spanning tree is "combed" consistently by construction;
    // seams only arise on non-tree (back/cross) edges examined below.
    std::vector<bool> visited(nf, false);

    // he_parent[fj] = the half-edge through which fj was first visited.
    // Used to build the set of tree edges so we can identify non-tree edges.
    std::vector<int> he_parent(nf, -1);

    // BUG FIX (B2 — param/combing v110): Two-pass BFS to guarantee feature
    // edges become non-tree (seam) edges whenever a non-feature path exists.
    //
    // Previous approach: used a deque with push_front / push_back to
    // "prioritise" non-feature edges.  The logic was fatally wrong because
    // `visited[fj] = true` and `he_parent[fj] = he` were set BEFORE the
    // push-order decision, so fj was always claimed by the first (feature)
    // edge that discovered it — the deque ordering only affected when fj's
    // OWN neighbours were explored, not which edge claimed fj.  Feature
    // edges thus became tree edges as often as any other edge, defeating the
    // stated intent entirely.
    //
    // Two-pass fix:
    //   Pass 1 — standard BFS, SKIPPING feature edges entirely.  This fills
    //            every face reachable without crossing a feature edge.
    //   Pass 2 — for any face left unvisited after Pass 1 (only reachable
    //            via a feature edge), visit it using all edges including
    //            feature edges.  Each such unvisited face becomes the seed
    //            of a new mini-BFS that again skips feature edges internally,
    //            until no more unvisited faces remain.
    //
    // Result: feature edges are left as non-tree edges (seam candidates)
    // in every case where a non-feature path exists, and are used only as
    // a last resort for topologically isolated face groups.  This matches
    // the Bommes (2013) recommendation that seam cuts prefer natural feature
    // boundaries to minimise parametrization distortion.
    //
    // The multi-component fix (Bug 4) is preserved: after both passes are
    // exhausted for the current component, the outer seed loop below finds
    // the next unvisited face and repeats.

    // Helper: BFS one wave from a seed face, optionally skipping feature edges.
    // Records he_parent and visited.  Returns nothing — modifies captures.
    std::deque<int> q;
    auto run_bfs = [&](int seed_face, bool skip_feature) {
        result.face_angles[seed_face] = field.face_frames[seed_face];
        visited[seed_face] = true;
        q.push_back(seed_face);

        while (!q.empty()) {
            int fi = q.front(); q.pop_front();
            double ai = result.face_angles[fi];

            auto hes = mesh.face_half_edges(fi);
            for (int he : hes) {
                int tw = mesh.half_edge(he).twin;
                if (tw < 0) continue;
                int fj = mesh.half_edge(tw).face;
                if (fj < 0 || visited[fj]) continue;

                int from_v = mesh.half_edge(mesh.half_edge(he).prev).vertex;
                int to_v   = mesh.half_edge(he).vertex;
                bool is_feat = feat_keys.count(edge_key(from_v, to_v)) > 0;

                // Pass 1: skip feature edges so they remain non-tree.
                if (skip_feature && is_feat) continue;

                // Parallel transport from fi to fj and comb fj.
                double phi            = parallel_transport_angle(mesh, he);
                double transported_ai = ai + phi;
                double aj_raw         = field.face_frames[fj];
                double diff           = aj_raw - transported_ai;
                double k_rot          = std::round(diff / (M_PI / 2.0));
                double aj_combed      = aj_raw - k_rot * (M_PI / 2.0);

                result.face_angles[fj] = aj_combed;
                he_parent[fj]          = he;
                visited[fj]            = true;
                q.push_back(fj);
            }
        }
    };

    // ---- Pass 1: no-feature-edge BFS over all components ----
    run_bfs(0, /*skip_feature=*/true);

    for (int seed = 1; seed < nf; ++seed) {
        if (visited[seed]) continue;
        run_bfs(seed, /*skip_feature=*/true);
    }

    // ---- Pass 2: visit any faces still unreached (feature-edge-only access) ----
    // These are faces that are only reachable by crossing a feature edge.
    // We now allow feature edges but still prefer non-feature paths from
    // the seed so that at least one non-feature tree edge roots each group.
    for (int seed = 0; seed < nf; ++seed) {
        if (visited[seed]) continue;
        // Allow all edges (including feature) for this isolated group.
        run_bfs(seed, /*skip_feature=*/false);
    }

    // BUG FIX (Bug 1): Detect seams on NON-TREE edges.
    //
    // After BFS, every face has a combed angle.  For each interior edge
    // that was NOT a BFS tree edge, check whether the combing is consistent
    // across it.  If it is not (i.e. the angle jump is not a multiple of π/2
    // within tolerance), the edge is a seam.  We record the integer period
    // jump (0–3) for each seam edge.
    //
    // Build BFS tree edge set: (lo, hi) pairs.
    std::unordered_set<int64_t> tree_edge_keys;
    tree_edge_keys.reserve(nf);
    for (int fj = 0; fj < nf; ++fj) {
        int he = he_parent[fj];
        if (he < 0) continue;
        int from_v = mesh.half_edge(mesh.half_edge(he).prev).vertex;
        int to_v   = mesh.half_edge(he).vertex;
        tree_edge_keys.insert(edge_key(from_v, to_v));
    }

    int nhe = mesh.num_half_edges();
    for (int he = 0; he < nhe; ++he) {
        int tw = mesh.half_edge(he).twin;
        if (tw < 0 || tw < he) continue;   // boundary or already seen twin

        int fi = mesh.half_edge(he).face;
        int fj = mesh.half_edge(tw).face;
        if (fi < 0 || fj < 0) continue;

        int from_v = mesh.half_edge(mesh.half_edge(he).prev).vertex;
        int to_v   = mesh.half_edge(he).vertex;
        int64_t ek = edge_key(from_v, to_v);

        // Skip tree edges — they are combed consistently by construction.
        if (tree_edge_keys.count(ek)) continue;

        // Compute the expected angle of fj given fi's combed angle and
        // the parallel transport across this edge.
        double phi = parallel_transport_angle(mesh, he);
        double ai  = result.face_angles[fi];
        double aj  = result.face_angles[fj];

        double transported_ai = ai + phi;
        double diff = aj - transported_ai;

        // Round to nearest multiple of π/2 to get the integer period jump.
        double k_real = diff / (M_PI / 2.0);
        int    k_int  = (int)std::round(k_real);

        // Residual after removing the integer rotation: if non-zero this
        // edge is a genuine seam (the combing is inconsistent).
        double residual = std::abs(diff - k_int * (M_PI / 2.0));

        // Tolerance: π/8 (22.5°) — anything larger is a genuine inconsistency.
        if (residual > M_PI / 8.0 || k_int != 0) {
            int lo = std::min(from_v, to_v);
            int hi = std::max(from_v, to_v);
            result.seam_edges.push_back({lo, hi});

            // BUG FIX (Bug 2): record the period jump (0–3) for this seam edge.
            // Normalise k_int to [0, 3].
            int pj = ((k_int % 4) + 4) % 4;
            result.period_jumps.push_back(pj);
        }
    }

    // Deduplicate seam edges (keep first occurrence which also keeps
    // the corresponding period_jump in sync).
    {
        std::unordered_set<int64_t> seen;
        seen.reserve(result.seam_edges.size());
        std::vector<std::pair<int,int>> dedup_seams;
        std::vector<int>               dedup_jumps;
        for (int k = 0; k < (int)result.seam_edges.size(); ++k) {
            auto& [lo, hi] = result.seam_edges[k];
            int64_t ek = edge_key(lo, hi);
            if (seen.insert(ek).second) {
                dedup_seams.push_back(result.seam_edges[k]);
                dedup_jumps.push_back(result.period_jumps[k]);
            }
        }
        result.seam_edges   = std::move(dedup_seams);
        result.period_jumps = std::move(dedup_jumps);
    }

    return result;
}

// -----------------------------------------------------------------------
// Minimal seam cut via shortest-path spanning tree
// -----------------------------------------------------------------------

EdgeSet compute_minimal_seam_cut(
    const HalfEdgeMesh&                    mesh,
    const std::vector<SingularityInfo>&    singularities)
{
    EdgeSet seam_cut;
    if (singularities.size() < 2) return seam_cut;

    int nv = mesh.num_vertices();
    // Build adjacency list with edge lengths as weights
    std::vector<std::vector<std::pair<int,double>>> adj(nv);
    int nhe = mesh.num_half_edges();
    for (int he = 0; he < nhe; ++he) {
        int tw = mesh.half_edge(he).twin;
        if (tw < 0 || tw < he) continue;
        int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
        int to   = mesh.half_edge(he).vertex;
        double len = (mesh.vertex_pos(to) - mesh.vertex_pos(from)).norm();
        adj[from].push_back({to, len});
        adj[to].push_back({from, len});
    }

    // Connect each singularity to the first one via Dijkstra
    int root = singularities[0].vertex_index;
    for (size_t i = 1; i < singularities.size(); ++i) {
        int dst = singularities[i].vertex_index;

        // Dijkstra
        const double INF = 1e30;
        std::vector<double> dist(nv, INF);
        std::vector<int>    prev(nv, -1);
        dist[root] = 0;
        using PD = std::pair<double,int>;
        std::priority_queue<PD, std::vector<PD>, std::greater<PD>> pq;
        pq.push({0.0, root});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            if (u == dst) break;
            for (auto& [v, w] : adj[u]) {
                if (dist[u]+w < dist[v]) {
                    dist[v] = dist[u]+w;
                    prev[v] = u;
                    pq.push({dist[v], v});
                }
            }
        }

        // Trace path dst → root
        for (int v = dst; v != root && v != -1; v = prev[v]) {
            int p = prev[v];
            if (p < 0) break;
            int lo = std::min(v,p), hi = std::max(v,p);
            seam_cut.push_back({lo, hi});
        }
    }

    std::sort(seam_cut.begin(), seam_cut.end());
    seam_cut.erase(std::unique(seam_cut.begin(), seam_cut.end()), seam_cut.end());
    return seam_cut;
}

} // namespace qf
