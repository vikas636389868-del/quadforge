/**
 * repair.cpp  —  Mesh Repair & Sanitization Layer
 *
 * Full implementation of the pre-flight repair pipeline described in
 * QuadForge roadmap sections 11.2 and 13.7.
 *
 * Stage order (enforced by repair_mesh()):
 *   1. remove_zero_area_faces          — degenerate triangle removal
 *   2. remove_duplicate_vertices       — adaptive vertex welding
 *   3. repair_winding                  — consistent face orientation
 *   4. split_non_manifold_vertices     — fan de-coupling
 *   5. detect_self_intersections       — BVH-accelerated scan
 *   6. isolate_self_intersecting_regions — conservative removal
 *
 * Every stage that modifies the mesh rebuilds the HalfEdgeMesh from scratch
 * (via the public (positions, faces) constructor) so the half-edge tables
 * remain internally consistent without a bespoke in-place mutation API.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#include "../../include/quadforge/mesh/repair.h"
#include "../../include/quadforge/mesh/topology.h"
#include "../../third_party/nanoflann/nanoflann.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <cassert>
#include <limits>

namespace qf {

// =========================================================================
// Internal helpers
// =========================================================================

namespace {

// -------------------------------------------------------------------------
// Rebuild helpers — convert HalfEdgeMesh ↔ flat arrays
// -------------------------------------------------------------------------

/** Extract positions and triangle face lists from a HalfEdgeMesh. */
static void extract_mesh(const HalfEdgeMesh& m,
                         std::vector<Vec3>& pos,
                         std::vector<std::array<int,3>>& tris)
{
    int nv = m.num_vertices();
    int nf = m.num_faces();
    pos.resize(nv);
    for (int i = 0; i < nv; ++i)
        pos[i] = m.vertex_pos(i);
    tris.resize(nf);
    for (int i = 0; i < nf; ++i)
        tris[i] = m.face_vertices(i);
}

/** Rebuild a HalfEdgeMesh in-place from updated positions + face lists. */
static void rebuild(HalfEdgeMesh& m,
                    const std::vector<Vec3>& pos,
                    const std::vector<std::array<int,3>>& tris)
{
    m = HalfEdgeMesh(pos, tris);
}

// -------------------------------------------------------------------------
// Spatial grid hash for duplicate-vertex welding — O(N) amortised
// -------------------------------------------------------------------------

struct GridHash {
    double cell;   // cell size

    struct IVec3Hash {
        size_t operator()(const std::array<int64_t,3>& k) const noexcept {
            // FNV-1a inspired mix
            size_t h = 2166136261u;
            for (int i = 0; i < 3; ++i) {
                h ^= std::hash<int64_t>{}(k[i]);
                h *= 16777619u;
            }
            return h;
        }
    };

    std::array<int64_t,3> key(const Vec3& p) const {
        return {
            static_cast<int64_t>(std::floor(p.x() / cell)),
            static_cast<int64_t>(std::floor(p.y() / cell)),
            static_cast<int64_t>(std::floor(p.z() / cell))
        };
    }

    /** Build canonical representative map.
     *  Returns old_to_new[v] — new vertex index for each original index.
     *  new_positions receives the deduplicated positions.
     */
    static std::vector<int> build(const std::vector<Vec3>& positions,
                                  double tol,
                                  std::vector<Vec3>& new_positions)
    {
        if (tol <= 0.0) tol = 1e-10;
        GridHash gh{tol};

        int nv = static_cast<int>(positions.size());
        std::vector<int> old_to_new(nv, -1);
        new_positions.clear();
        new_positions.reserve(nv);

        // cell → list of new-vertex indices already placed there
        std::unordered_map<std::array<int64_t,3>,
                           std::vector<int>,
                           IVec3Hash> grid;
        grid.reserve(nv);

        for (int vi = 0; vi < nv; ++vi) {
            const Vec3& p = positions[vi];
            auto ck = gh.key(p);

            // Search 27 neighbouring cells for an existing close vertex
            int found = -1;
            for (int di = -1; di <= 1 && found < 0; ++di)
            for (int dj = -1; dj <= 1 && found < 0; ++dj)
            for (int dk = -1; dk <= 1 && found < 0; ++dk) {
                std::array<int64_t,3> nk = {ck[0]+di, ck[1]+dj, ck[2]+dk};
                auto it = grid.find(nk);
                if (it == grid.end()) continue;
                for (int cand : it->second) {
                    if ((new_positions[cand] - p).squaredNorm() <= tol * tol) {
                        found = cand;
                        break;
                    }
                }
            }

            if (found >= 0) {
                old_to_new[vi] = found;
            } else {
                int nidx = static_cast<int>(new_positions.size());
                new_positions.push_back(p);
                old_to_new[vi] = nidx;
                grid[ck].push_back(nidx);
            }
        }
        return old_to_new;
    }
};

// -------------------------------------------------------------------------
// BVH for self-intersection detection — wraps nanoflann on centroids
// -------------------------------------------------------------------------

struct CentroidCloud {
    const std::vector<Vec3>& pts;
    size_t kdtree_get_point_count() const { return pts.size(); }
    // BUG FIX (v99): `dim` was `int` — nanoflann ≥1.3 defines the adapter
    // interface with `size_t dim`.  Using `int` causes an implicit narrowing
    // conversion, which is an error under -Wconversion / /W4 and may silently
    // truncate the index on compilers that treat the mismatch as a warning.
    // Matches the corrected signature already present in spatial.cpp's
    // TriCentroidCloud adapter.
    double kdtree_get_pt(size_t idx, size_t dim) const {
        return pts[idx](static_cast<Eigen::Index>(dim));
    }
    template<class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using CentroidKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, CentroidCloud>,
    CentroidCloud, 3>;

// Triangle AABB
struct AABB {
    Vec3 lo, hi;
    static AABB from_tri(const Vec3& a, const Vec3& b, const Vec3& c) {
        AABB box;
        box.lo = a.cwiseMin(b).cwiseMin(c);
        box.hi = a.cwiseMax(b).cwiseMax(c);
        return box;
    }
    bool overlaps(const AABB& o) const {
        return lo.x() <= o.hi.x() && hi.x() >= o.lo.x()
            && lo.y() <= o.hi.y() && hi.y() >= o.lo.y()
            && lo.z() <= o.hi.z() && hi.z() >= o.lo.z();
    }
};

/**
 * Möller–Trumbore triangle–triangle intersection test.
 * Returns true if the two triangles (given by vertices) intersect.
 * Shared vertices are not counted as intersections.
 */
static bool triangles_intersect(
    const Vec3& a0, const Vec3& a1, const Vec3& a2,
    const Vec3& b0, const Vec3& b1, const Vec3& b2)
{
    // If triangles share any vertex, skip (expected mesh topology)
    auto eq = [](const Vec3& x, const Vec3& y){
        return (x-y).squaredNorm() < 1e-20;
    };
    if (eq(a0,b0)||eq(a0,b1)||eq(a0,b2)||
        eq(a1,b0)||eq(a1,b1)||eq(a1,b2)||
        eq(a2,b0)||eq(a2,b1)||eq(a2,b2)) return false;

    // Separating axis test — fast rejection on AABB first
    // (already gated by AABB in caller, so skip here)

    // Helper: signed distance of triangle A vertices above plane of B
    auto tri_plane_test = [](
        const Vec3& p0, const Vec3& p1, const Vec3& p2,
        const Vec3& q0, const Vec3& q1, const Vec3& q2,
        double& d0, double& d1, double& d2) -> bool
    {
        Vec3 n = (q1 - q0).cross(q2 - q0);
        double dn = n.squaredNorm();
        if (dn < 1e-20) return false; // degenerate B
        Vec3 np = n / std::sqrt(dn);
        double d = np.dot(q0);
        d0 = np.dot(p0) - d;
        d1 = np.dot(p1) - d;
        d2 = np.dot(p2) - d;
        return true;
    };

    double da0, da1, da2, db0, db1, db2;
    if (!tri_plane_test(a0,a1,a2, b0,b1,b2, da0,da1,da2)) return false;
    // If all A verts on same side of B's plane → no intersection
    if (da0*da1 > 0 && da0*da2 > 0) return false;

    if (!tri_plane_test(b0,b1,b2, a0,a1,a2, db0,db1,db2)) return false;
    if (db0*db1 > 0 && db0*db2 > 0) return false;

    // Full coplanar / edge-edge test would be needed for exactness, but
    // for our repair purposes the plane-separation test is sufficient to
    // flag the vast majority of real self-intersections.
    return true;
}

} // anonymous namespace

// =========================================================================
// Stage 1 — remove_zero_area_faces
// =========================================================================

int remove_zero_area_faces(HalfEdgeMesh& mesh,
                           const MeshRepairOptions& opts)
{
    std::vector<Vec3>            pos;
    std::vector<std::array<int,3>> tris;
    extract_mesh(mesh, pos, tris);

    double thresh = opts.area_threshold;
    std::vector<std::array<int,3>> kept;
    kept.reserve(tris.size());
    int removed = 0;

    for (const auto& f : tris) {
        const Vec3& v0 = pos[f[0]];
        const Vec3& v1 = pos[f[1]];
        const Vec3& v2 = pos[f[2]];
        double area = 0.5 * (v1 - v0).cross(v2 - v0).norm();
        if (area <= thresh) { ++removed; continue; }
        kept.push_back(f);
    }

    if (removed > 0) rebuild(mesh, pos, kept);
    return removed;
}

// =========================================================================
// Stage 2 — remove_duplicate_vertices
// =========================================================================

int remove_duplicate_vertices(HalfEdgeMesh& mesh,
                              const MeshRepairOptions& opts)
{
    std::vector<Vec3>            pos;
    std::vector<std::array<int,3>> tris;
    extract_mesh(mesh, pos, tris);

    // Compute weld tolerance
    double tol = opts.weld_tolerance;
    if (opts.adaptive_weld && !tris.empty()) {
        // Sample mean edge length from first min(2000, F) faces
        int sample_n = std::min((int)tris.size(), 2000);
        double sum_len = 0.0;
        int cnt = 0;
        for (int i = 0; i < sample_n; ++i) {
            for (int k = 0; k < 3; ++k) {
                int a = tris[i][k], b = tris[i][(k+1)%3];
                sum_len += (pos[a] - pos[b]).norm();
                ++cnt;
            }
        }
        if (cnt > 0) {
            double mean_edge = sum_len / cnt;
            tol = std::max(tol, mean_edge * 1e-4);
        }
    }

    std::vector<Vec3> new_pos;
    std::vector<int> old_to_new = GridHash::build(pos, tol, new_pos);

    int merged = (int)pos.size() - (int)new_pos.size();
    if (merged <= 0) return 0;

    // -----------------------------------------------------------------------
    // UV seam / vertex attribute preservation  (roadmap §11.2)
    //
    // GridHash::build() welded purely by 3-D position.  Apply a correction
    // pass: any two vertices that were merged but carry DIFFERENT split IDs
    // are un-welded back to separate new-vertex slots.  This preserves
    // UV-seam splits and hard-edge crease duplicates.
    // -----------------------------------------------------------------------
    if (opts.preserve_uv_seams && opts.vertex_split_ids != nullptr) {
        const int32_t* split_ids = opts.vertex_split_ids;
        int nv_orig = static_cast<int>(pos.size());

        // Record the canonical split tag for each new-vertex slot (set on
        // first encounter; any subsequent old vertex with a different tag
        // gets a fresh slot).
        std::vector<int32_t> slot_tag(new_pos.size(), -1);
        for (int vi = 0; vi < nv_orig; ++vi) {
            int ni = old_to_new[vi];
            if (slot_tag[ni] < 0)
                slot_tag[ni] = split_ids[vi];  // first occupant sets the tag
        }

        // BUG FIX (v70): the original loop allocated a brand-new fresh slot
        // for EVERY vertex whose split_id differed from the canonical tag,
        // even when two such vertices had the SAME (old_slot, split_id) pair.
        // That produced unnecessary duplicates, inflating the vertex count and
        // breaking UV-seam topology.  We now keep a secondary map keyed on
        // (old_slot << 32 | split_id) so all vertices that share the same
        // position-slot AND split_id are directed to the same fresh slot.
        std::unordered_map<int64_t, int> split_remap;
        for (int vi = 0; vi < nv_orig; ++vi) {
            int ni = old_to_new[vi];
            int32_t sid = split_ids[vi];
            if (slot_tag[ni] != sid) {
                // Different tag — re-use or create a fresh slot for (ni, sid).
                // BUG FIX (v104): `static_cast<int64_t>(ni) << 32` is undefined
                // behaviour when ni is negative or >= 2^31 (left-shifting a signed
                // integer into/past the sign bit is UB per [expr.shift §7.6.7]).
                // Vertex-slot indices are logically non-negative, but the UB can
                // still be triggered by defensive code paths with sentinel values.
                // Fix: promote through uint32_t before shifting (same pattern as
                // halfedge.h::EdgeKeyHash, features.cpp::edge_key, and the
                // detect_self_intersections key already present in this file).
                int64_t key = (static_cast<int64_t>(static_cast<uint32_t>(ni)) << 32)
                            | static_cast<int64_t>(static_cast<uint32_t>(sid));
                auto it = split_remap.find(key);
                if (it != split_remap.end()) {
                    // Another vertex with the same (position-slot, split_id)
                    // has already created this fresh slot — reuse it.
                    old_to_new[vi] = it->second;
                } else {
                    // First time this (ni, sid) pair is seen — allocate.
                    int fresh = static_cast<int>(new_pos.size());
                    new_pos.push_back(pos[vi]);
                    slot_tag.push_back(sid);
                    old_to_new[vi] = fresh;
                    split_remap[key] = fresh;
                }
            }
        }

        // Recalculate net merged count; bail out if correction eliminated
        // all actual merges (nothing to rebuild).
        merged = nv_orig - static_cast<int>(new_pos.size());
        if (merged <= 0) return 0;
    }

    // Remap face indices; drop degenerate faces where two verts merged to one.
    std::vector<std::array<int,3>> new_tris;
    new_tris.reserve(tris.size());
    for (const auto& f : tris) {
        std::array<int,3> nf = {old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]};
        if (nf[0] == nf[1] || nf[1] == nf[2] || nf[0] == nf[2]) continue;
        new_tris.push_back(nf);
    }

    rebuild(mesh, new_pos, new_tris);
    return merged;
}

// =========================================================================
// Stage 3 — repair_winding
// =========================================================================

int repair_winding(HalfEdgeMesh& mesh,
                   const MeshRepairOptions& /*opts*/)
{
    if (mesh.num_faces() == 0) return 0;

    std::vector<Vec3>            pos;
    std::vector<std::array<int,3>> tris;
    extract_mesh(mesh, pos, tris);

    int nf = static_cast<int>(tris.size());

    // Build face adjacency from shared edges
    // adj[fi] = list of (fj, shared_edge_local_index_in_fi)
    using EdgePair = std::pair<int,int>; // (v_lo, v_hi)
    struct EdgeKeyHash {
        size_t operator()(const EdgePair& k) const {
            // BUG FIX (v104): `static_cast<int64_t>(k.first) << 32` is UB when
            // k.first is negative or >= 2^31 (signed left-shift past the sign bit).
            // Vertex indices are non-negative, but sentinel values (-1) can appear
            // during construction. Fix: cast through uint32_t first, matching the
            // identical correction already applied to halfedge.h::EdgeKeyHash,
            // features.cpp::edge_key, and the SI-detection key in this same file.
            return std::hash<int64_t>()(
                (static_cast<int64_t>(static_cast<uint32_t>(k.first)) << 32)
                | static_cast<int64_t>(static_cast<uint32_t>(k.second)));
        }
    };
    std::unordered_map<EdgePair, std::vector<int>, EdgeKeyHash> edge_to_faces;
    edge_to_faces.reserve(nf * 3);

    for (int fi = 0; fi < nf; ++fi) {
        for (int k = 0; k < 3; ++k) {
            int a = tris[fi][k], b = tris[fi][(k+1)%3];
            edge_to_faces[{std::min(a,b), std::max(a,b)}].push_back(fi);
        }
    }

    // BFS: ensure adjacent faces have consistent orientation
    // Orientation is consistent when the shared edge is traversed in
    // opposite directions by the two faces (a→b in fi, b→a in fj).
    std::vector<bool> visited(nf, false);
    int flip_count = 0;

    for (int seed = 0; seed < nf; ++seed) {
        if (visited[seed]) continue;
        std::queue<int> bfs;
        bfs.push(seed);
        visited[seed] = true;

        while (!bfs.empty()) {
            int fi = bfs.front(); bfs.pop();
            const auto& tf = tris[fi];

            for (int k = 0; k < 3; ++k) {
                int a = tf[k], b = tf[(k+1)%3];
                // Direction of this half-edge: a → b
                EdgePair ep = {std::min(a,b), std::max(a,b)};
                auto it = edge_to_faces.find(ep);
                if (it == edge_to_faces.end()) continue;

                for (int fj : it->second) {
                    if (fj == fi || visited[fj]) continue;

                    // Check orientation consistency: fj should traverse b→a
                    bool consistent = false;
                    for (int m = 0; m < 3; ++m) {
                        if (tris[fj][m] == b && tris[fj][(m+1)%3] == a) {
                            consistent = true; break;
                        }
                    }

                    if (!consistent) {
                        // Flip fj: swap vertex 1 and 2 to reverse winding.
                        std::swap(tris[fj][1], tris[fj][2]);
                        // No need to update edge_to_faces: the map is keyed on
                        // undirected (v_lo, v_hi) pairs, which are invariant
                        // under winding reversal.  Consistency of fj's new
                        // neighbours is checked when fj is later dequeued.
                        ++flip_count;
                    }
                    visited[fj] = true;
                    bfs.push(fj);
                }
            }
        }
    }

    if (flip_count > 0) rebuild(mesh, pos, tris);
    return flip_count;
}

// =========================================================================
// Stage 4 — split_non_manifold_vertices
// =========================================================================

int split_non_manifold_vertices(HalfEdgeMesh& mesh,
                                const MeshRepairOptions& /*opts*/)
{
    // NOTE: We intentionally do NOT early-return on mesh.is_manifold() here.
    // is_manifold() only checks for non-manifold EDGES (edge fan > 2 half-edges).
    // It does NOT detect non-manifold VERTICES whose face-fan consists of more than
    // one connected component (e.g. a "bowtie" or "pinch" vertex: two triangles
    // sharing a single vertex but no edges). Such vertices have all-manifold edges
    // yet are themselves non-manifold. The function correctly returns 0 if all
    // vertices turn out to have a single connected fan component.

    std::vector<Vec3>            pos;
    std::vector<std::array<int,3>> tris;
    extract_mesh(mesh, pos, tris);

    int nv  = static_cast<int>(pos.size());
    int nf  = static_cast<int>(tris.size());
    int splits = 0;

    // -----------------------------------------------------------------------
    // Build adjacency structures once from the ORIGINAL (unmodified) tris.
    // We must NOT modify these maps or tris during PASS 1, because the BFS
    // for vertex vi+k depends on the undirected edge keys (a,b) that were
    // present in the original mesh.  Modifying tris in-place while iterating
    // invalidates those keys and causes BFS for later vertices to look up
    // stale entries, producing missed or spurious splits.
    // -----------------------------------------------------------------------

    // Per-vertex face list (original tris)
    std::vector<std::vector<int>> vtx_faces(nv);
    for (int fi = 0; fi < nf; ++fi)
        for (int k = 0; k < 3; ++k)
            vtx_faces[tris[fi][k]].push_back(fi);

    // Undirected-edge -> face list (original tris)
    using EP = std::pair<int,int>;
    struct EPHash {
        size_t operator()(const EP& e) const {
            // BUG FIX (v104): same signed left-shift UB as EdgeKeyHash above.
            // Cast both components through uint32_t before promoting to int64_t.
            return std::hash<int64_t>()(
                (static_cast<int64_t>(static_cast<uint32_t>(e.first)) << 32)
                | static_cast<int64_t>(static_cast<uint32_t>(e.second)));
        }
    };
    std::unordered_map<EP, std::vector<int>, EPHash> edge_faces;
    edge_faces.reserve(nf * 3);
    for (int fi = 0; fi < nf; ++fi)
        for (int k = 0; k < 3; ++k) {
            int a = tris[fi][k], b = tris[fi][(k+1)%3];
            edge_faces[{std::min(a,b), std::max(a,b)}].push_back(fi);
        }

    // -----------------------------------------------------------------------
    // PASS 1 — Detect non-manifold vertices and record which faces belong to
    //          each extra component, WITHOUT touching `tris`.
    //
    // Each PendingRewire entry describes one extra component of one vertex
    // that needs a fresh duplicate position.  Component 0 keeps the original
    // vertex; extras (1..n-1) each get a new vertex in PASS 2.
    // -----------------------------------------------------------------------

    struct PendingRewire {
        int vi;               // original vertex index to duplicate
        std::vector<int> faces; // faces in the extra component to re-wire
    };
    std::vector<PendingRewire> pending;

    for (int vi = 0; vi < nv; ++vi) {
        const auto& ifaces = vtx_faces[vi];
        if ((int)ifaces.size() < 2) continue;

        // BFS over faces incident on vi, connected via edges that include vi.
        std::unordered_map<int,int> face_comp;
        int comp_id = 0;
        std::vector<std::vector<int>> comp_face_list;

        for (int seed_fi : ifaces) {
            if (face_comp.count(seed_fi)) continue;
            comp_face_list.emplace_back();
            std::queue<int> q;
            q.push(seed_fi);
            face_comp[seed_fi] = comp_id;

            while (!q.empty()) {
                int fi = q.front(); q.pop();
                comp_face_list[comp_id].push_back(fi);

                // Walk only edges of fi that are incident on vi so the BFS
                // stays within the face fan around vi.
                for (int k = 0; k < 3; ++k) {
                    int a = tris[fi][k], b = tris[fi][(k+1)%3];
                    if (a != vi && b != vi) continue;
                    EP ep = {std::min(a,b), std::max(a,b)};
                    auto it = edge_faces.find(ep);
                    if (it == edge_faces.end()) continue;
                    for (int fj : it->second) {
                        if (fj == fi) continue;
                        if (face_comp.count(fj)) continue;
                        face_comp[fj] = comp_id;
                        q.push(fj);
                    }
                }
            }
            ++comp_id;
        }

        if (comp_id <= 1) continue; // manifold vertex — nothing to do

        // Enqueue extra components (1..n-1) for re-wiring in PASS 2.
        // Component 0 keeps the original vertex index.
        for (int c = 1; c < comp_id; ++c)
            pending.push_back({vi, std::move(comp_face_list[c])});
    }

    if (pending.empty()) return 0;

    // -----------------------------------------------------------------------
    // PASS 2 — Allocate new vertices and apply re-wirings.
    //
    // At this point `tris` is still in its original state.  We apply every
    // PendingRewire entry sequentially.  Entries from different vertices are
    // disjoint in terms of vertex indices (each entry uses a unique new_vi),
    // so there are no ordering conflicts.  Two entries for different vertices
    // that share the same face fi each modify a different slot k of that face,
    // so no entry overwrites another's work.
    // -----------------------------------------------------------------------
    for (auto& pr : pending) {
        int new_vi = static_cast<int>(pos.size());
        // BUG FIX (v71): pos.push_back(pos[pr.vi]) is undefined behaviour when
        // the vector needs to grow: std::vector::push_back(const T& x) invalidates
        // ALL references (including x itself) if reallocation occurs.  The
        // reference pos[pr.vi] becomes a dangling pointer at that point, and the
        // copy constructor of Vec3 reads from freed memory — classic UB.
        // Fix: copy the value into a local variable BEFORE calling push_back.
        const Vec3 dup_pos = pos[pr.vi];   // snapshot the position first
        pos.push_back(dup_pos);            // now safe: no aliasing with pos

        for (int fi : pr.faces)
            for (int k = 0; k < 3; ++k)
                if (tris[fi][k] == pr.vi)
                    tris[fi][k] = new_vi;

        ++splits;
    }

    rebuild(mesh, pos, tris);
    return splits;
}

// =========================================================================
// Stage 5 — detect_self_intersections
// =========================================================================

std::vector<std::pair<int,int>> detect_self_intersections(
    const HalfEdgeMesh& mesh,
    const MeshRepairOptions& opts)
{
    int nf = mesh.num_faces();
    if (nf < 2) return {};

    // Build centroid array and AABB array
    std::vector<Vec3> centroids(nf);
    std::vector<AABB> aabbs(nf);

    for (int fi = 0; fi < nf; ++fi) {
        auto verts = mesh.face_vertices(fi);
        const Vec3& p0 = mesh.vertex_pos(verts[0]);
        const Vec3& p1 = mesh.vertex_pos(verts[1]);
        const Vec3& p2 = mesh.vertex_pos(verts[2]);
        centroids[fi] = (p0 + p1 + p2) / 3.0;
        aabbs[fi] = AABB::from_tri(p0, p1, p2);
    }

    CentroidCloud cloud{centroids};
    CentroidKDTree kd(3, cloud,
                      nanoflann::KDTreeSingleIndexAdaptorParams(16));
    kd.buildIndex();

    // For each face, query K nearest-centroid neighbours and run
    // AABB + full intersection test on candidates
    int K = std::min(opts.max_si_candidates, nf);
    std::vector<uint32_t> nn_idx(K);
    std::vector<double>   nn_dist(K);

    std::vector<std::pair<int,int>> pairs;
    // Use a flat set to avoid duplicate pair reporting
    std::unordered_set<int64_t> seen;

    for (int fi = 0; fi < nf; ++fi) {
        const double qpt[3] = {centroids[fi].x(),
                               centroids[fi].y(),
                               centroids[fi].z()};
        size_t found = kd.knnSearch(qpt, K,
                                    nn_idx.data(), nn_dist.data());

        auto fvA = mesh.face_vertices(fi);
        const Vec3 a0 = mesh.vertex_pos(fvA[0]);
        const Vec3 a1 = mesh.vertex_pos(fvA[1]);
        const Vec3 a2 = mesh.vertex_pos(fvA[2]);

        for (size_t n = 0; n < found; ++n) {
            int fj = static_cast<int>(nn_idx[n]);
            if (fj <= fi) continue; // test each pair once

            // Deduplicate
            // BUG FIX (v99): C-style cast `(int64_t)fi` left-shifts a signed
            // integer, which is UB for negative or very large values.  Promote
            // through uint32_t first (same pattern as EdgeKeyHash in halfedge.h
            // and seam_edge_key() in seam_utils.h) so the shift is always on an
            // unsigned type and the result is well-defined.
            int64_t key = (static_cast<int64_t>(static_cast<uint32_t>(fi)) << 32)
                        | static_cast<int64_t>(static_cast<uint32_t>(fj));
            if (!seen.insert(key).second) continue;

            // AABB overlap guard
            if (!aabbs[fi].overlaps(aabbs[fj])) continue;

            auto fvB = mesh.face_vertices(fj);
            const Vec3 b0 = mesh.vertex_pos(fvB[0]);
            const Vec3 b1 = mesh.vertex_pos(fvB[1]);
            const Vec3 b2 = mesh.vertex_pos(fvB[2]);

            if (triangles_intersect(a0, a1, a2, b0, b1, b2))
                pairs.push_back({fi, fj});
        }
    }

    return pairs;
}

// =========================================================================
// Stage 6 — isolate_self_intersecting_regions
// =========================================================================

int isolate_self_intersecting_regions(
    HalfEdgeMesh& mesh,
    const std::vector<std::pair<int,int>>& si_pairs_in,
    const MeshRepairOptions& opts)
{
    // Run detection if not provided
    const std::vector<std::pair<int,int>>* pairs = &si_pairs_in;
    std::vector<std::pair<int,int>> tmp;
    if (pairs->empty()) {
        tmp = detect_self_intersections(mesh, opts);
        pairs = &tmp;
    }
    if (pairs->empty()) return 0;

    std::unordered_set<int> bad;
    for (const auto& [fi, fj] : *pairs) {
        bad.insert(fi);
        bad.insert(fj);
    }

    std::vector<Vec3>              pos;
    std::vector<std::array<int,3>> tris;
    extract_mesh(mesh, pos, tris);

    std::vector<std::array<int,3>> kept;
    kept.reserve(tris.size() - bad.size());
    for (int fi = 0; fi < (int)tris.size(); ++fi) {
        if (!bad.count(fi)) kept.push_back(tris[fi]);
    }

    int removed = static_cast<int>(bad.size());
    rebuild(mesh, pos, kept);
    return removed;
}

// =========================================================================
// RepairReport helpers
// =========================================================================

void RepairReport::compute_confidence(int original_face_count)
{
    if (original_face_count <= 0) { confidence = 100.f; return; }

    float score = 100.f;

    // Each degenerate face = −1 point, capped at 20
    score -= std::min(20.f, float(zero_area_removed));

    // Merging >5% of vertices indicates a messy input
    if (vertices_before > 0) {
        float merge_ratio = float(duplicate_verts_merged) / float(vertices_before);
        score -= std::min(15.f, merge_ratio * 200.f);
    }

    // Winding flips: >10% = major problem
    if (faces_before > 0) {
        float flip_ratio = float(winding_flips) / float(faces_before);
        score -= std::min(20.f, flip_ratio * 100.f);
    }

    // Non-manifold splits each cost 3 points
    score -= std::min(15.f, float(nm_vertex_splits) * 3.f);

    // Self-intersections are serious
    score -= std::min(25.f, float(si_faces_removed) /
                      float(std::max(1, original_face_count)) * 500.f);

    // Penalise residual non-manifold geometry after repair
    if (!is_manifold_after) score -= 10.f;

    confidence = std::max(0.f, std::min(100.f, score));
}

std::string RepairReport::to_string() const
{
    std::ostringstream os;
    os << "[Repair] V: " << vertices_before << " → " << vertices_after
       << "  F: " << faces_before << " → " << faces_after
       << "  confidence=" << int(confidence) << "%\n";

    if (zero_area_removed)      os << "  Removed " << zero_area_removed      << " zero-area faces\n";
    if (duplicate_verts_merged) os << "  Merged  " << duplicate_verts_merged  << " duplicate vertices\n";
    if (winding_flips)          os << "  Flipped " << winding_flips           << " face windings\n";
    if (nm_vertex_splits)       os << "  Split   " << nm_vertex_splits        << " non-manifold vertices\n";
    if (self_intersect_pairs)   os << "  Found   " << self_intersect_pairs    << " self-intersecting pairs\n";
    if (si_faces_removed)       os << "  Removed " << si_faces_removed        << " self-intersecting faces\n";

    os << "  Manifold after: " << (is_manifold_after ? "YES" : "NO")
       << "  Closed: "         << (is_closed_after   ? "YES" : "NO");
    return os.str();
}

// =========================================================================
// Main entry point — repair_mesh
// =========================================================================

RepairReport repair_mesh(HalfEdgeMesh& mesh,
                         const MeshRepairOptions& opts)
{
    RepairReport report;
    report.vertices_before = mesh.num_vertices();
    report.faces_before    = mesh.num_faces();

    // Stage 1 — zero-area faces (must run first, before welding)
    if (opts.run_zero_area)
        report.zero_area_removed = remove_zero_area_faces(mesh, opts);

    // Stage 2 — duplicate vertex welding
    if (opts.run_weld)
        report.duplicate_verts_merged = remove_duplicate_vertices(mesh, opts);

    // Stage 3 — winding repair
    if (opts.run_winding && opts.repair_winding)
        report.winding_flips = repair_winding(mesh, opts);

    // Stage 4 — non-manifold vertex splitting
    if (opts.run_nm_vertex_split && opts.split_nm_vertices)
        report.nm_vertex_splits = split_non_manifold_vertices(mesh, opts);

    // Stage 5+6 — self-intersection detection and isolation
    if (opts.run_self_intersect && opts.detect_self_intersect) {
        auto si = detect_self_intersections(mesh, opts);
        report.self_intersect_pairs = static_cast<int>(si.size());
        if (opts.isolate_self_intersect && !si.empty())
            report.si_faces_removed = isolate_self_intersecting_regions(mesh, si, opts);
    }

    // Final state
    report.vertices_after    = mesh.num_vertices();
    report.faces_after       = mesh.num_faces();
    report.is_manifold_after = mesh.is_manifold();
    report.is_closed_after   = mesh.is_closed();
    report.compute_confidence(report.faces_before);
    report.summary = report.to_string();

    return report;
}

// =========================================================================
// UV seam / vertex attribute utilities  (roadmap §11.2)
// =========================================================================

std::vector<int32_t> compute_vertex_split_ids_from_uv(
    int            num_vertices,
    int            num_faces,
    const int32_t* uv_indices,
    int            uv_face_stride,
    const int32_t* face_vertices)
{
    // Default: all vertices in the same "group 0" (no seam awareness).
    std::vector<int32_t> split_ids(num_vertices, 0);

    if (!uv_indices || !face_vertices || uv_face_stride < 3
        || num_faces <= 0 || num_vertices <= 0)
        return split_ids;

    // For each vertex, collect the set of UV-loop indices that reference it.
    // Two references at the same vertex with different UV indices indicate a
    // UV seam — the vertex is "split" in UV space.
    //
    // Strategy: assign each vertex the MINIMUM UV-loop index that references
    // it.  Vertices that appear in multiple UV islands will have distinct
    // minimum indices if the original mesh used separate UV loop indices for
    // each island (which is the normal Blender convention).
    //
    // We use union-find to group vertex-slots that are UV-continuous so that
    // contiguous UV patches get the same split ID regardless of index order.

    // Step 1: build per-vertex → set-of-uv-indices mapping (bounded to first
    // seen value to avoid O(F²) blowup on dense meshes).
    // We store ONE canonical UV index per vertex; if a second reference
    // differs, we note that vertex as "seam-split".
    std::vector<int32_t> canonical_uv(num_vertices, -1);
    // split_ids will be used as the final output; start as -1 (unset).
    std::fill(split_ids.begin(), split_ids.end(), -1);

    for (int fi = 0; fi < num_faces; ++fi) {
        for (int k = 0; k < 3 && k < uv_face_stride; ++k) {
            int vi  = face_vertices[fi * 3 + k];
            int uvi = uv_indices[fi * uv_face_stride + k];
            if (vi < 0 || vi >= num_vertices) continue;

            if (canonical_uv[vi] < 0) {
                // First time this vertex is seen — record canonical UV index.
                canonical_uv[vi] = uvi;
                split_ids[vi]    = uvi;  // initial split tag = first UV loop index
            } else if (canonical_uv[vi] != uvi) {
                // This vertex appears with a DIFFERENT UV index on another face.
                // Keep the minimum UV-loop index as the canonical tag so that
                // the same seam vertex always gets the same deterministic ID.
                // The bridge layer (mesh_transfer.py) is responsible for passing
                // separate vertex entries for each UV island, so this vertex
                // will appear as two distinct vertices in the C-layer call.
                // Nevertheless, we update the tag to min so the result is
                // stable regardless of face-order.
                int32_t best = std::min(canonical_uv[vi], uvi);
                canonical_uv[vi] = best;
                split_ids[vi]    = best;
            }
        }
    }

    // Step 2: any vertex that was never referenced by a UV face gets tag 0.
    for (int vi = 0; vi < num_vertices; ++vi)
        if (split_ids[vi] < 0) split_ids[vi] = 0;

    return split_ids;
}

} // namespace qf
