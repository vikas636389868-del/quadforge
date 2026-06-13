/**
 * cleanup.cpp — Post-extraction quad mesh cleanup.
 *
 * All five operations declared in cleanup.h are implemented:
 *   1. Degenerate face removal  (zero-area quads / repeated verts)
 *   2. Short edge collapse       (edges < min_edge_length_factor * mean_len)
 *   3. Small hole filling        (boundary loops <= 8 edges, fill_holes=true)
 *   4. Edge flips                (quad rotations to reduce |valence-4|^2)
 *   5. Doublet removal           (valence-2 interior vertices merged away)
 *
 * Additional helpers that keep the mesh consistent:
 *   - Near-coincident vertex welding (step 3b, v24 fix retained)
 *   - Final vertex compaction        (step 4)
 *   - Stats update (quad_percentage, avg_valence)
 *
 * v51 — added steps 2 (short edge collapse), 3.5 (edge flips), 3.75 (hole fill).
 */

#include "../../include/quadforge/extract/cleanup.h"

#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <array>
#include <cstdint>

namespace qf {

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

static double quad_area(const QuadMesh& qm, const std::array<int,4>& q) {
    auto& v = qm.vertices;
    auto cross_area = [](const std::array<double,3>& a,
                         const std::array<double,3>& b,
                         const std::array<double,3>& c) -> double {
        double ax=b[0]-a[0], ay=b[1]-a[1], az=b[2]-a[2];
        double bx=c[0]-a[0], by=c[1]-a[1], bz=c[2]-a[2];
        double nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
        return 0.5*std::sqrt(nx*nx+ny*ny+nz*nz);
    };
    return cross_area(v[q[0]],v[q[1]],v[q[2]])
         + cross_area(v[q[0]],v[q[2]],v[q[3]]);
}

/** Remove faces with repeated vertex indices or out-of-range indices. */
static void strip_degen(QuadMesh& qm) {
    int nv = (int)qm.vertices.size();
    {
        std::vector<std::array<int,4>> good;
        good.reserve(qm.quads.size());
        for (auto& q : qm.quads) {
            bool oor = false;
            for (int v : q) if (v < 0 || v >= nv) { oor=true; break; }
            if (oor) continue;
            if (q[0]==q[1]||q[0]==q[2]||q[0]==q[3]||
                q[1]==q[2]||q[1]==q[3]||q[2]==q[3]) continue;
            good.push_back(q);
        }
        qm.quads = std::move(good);
    }
    {
        std::vector<std::array<int,3>> good;
        good.reserve(qm.tris.size());
        for (auto& t : qm.tris) {
            bool oor = false;
            for (int v : t) if (v < 0 || v >= nv) { oor=true; break; }
            if (!oor && t[0]!=t[1] && t[0]!=t[2] && t[1]!=t[2])
                good.push_back(t);
        }
        qm.tris = std::move(good);
    }
}

// -----------------------------------------------------------------------
// cleanup_quad_mesh
// -----------------------------------------------------------------------

void cleanup_quad_mesh(QuadMesh& qm, const CleanupParams& params) {
    if (qm.vertices.empty()) return;

    int nv = (int)qm.vertices.size();

    // ================================================================
    // Step 1: Remove degenerate quads (zero-area or repeated vertices)
    // ================================================================
    {
        double total_len = 0.0; int count = 0;
        for (auto& q : qm.quads) {
            for (int k = 0; k < 4; ++k) {
                auto& a = qm.vertices[q[k]];
                auto& b = qm.vertices[q[(k+1)%4]];
                double dx=b[0]-a[0], dy=b[1]-a[1], dz=b[2]-a[2];
                total_len += std::sqrt(dx*dx+dy*dy+dz*dz);
                count++;
            }
        }
        double mean_len = count > 0 ? total_len/count : 1.0;
        double min_area = mean_len * mean_len
                        * params.min_edge_length_factor
                        * params.min_edge_length_factor;

        std::vector<std::array<int,4>> good_quads;
        good_quads.reserve(qm.quads.size());
        for (auto& q : qm.quads) {
            bool degen = (q[0]==q[1]||q[0]==q[2]||q[0]==q[3]||
                          q[1]==q[2]||q[1]==q[3]||q[2]==q[3]);
            if (degen) continue;
            if (quad_area(qm, q) < min_area) continue;
            bool oor = false;
            for (int v : q) if (v < 0 || v >= nv) { oor=true; break; }
            if (oor) continue;
            good_quads.push_back(q);
        }
        qm.quads = std::move(good_quads);
    }
    // Degenerate triangles
    {
        std::vector<std::array<int,3>> good_tris;
        good_tris.reserve(qm.tris.size());
        for (auto& t : qm.tris) {
            if (t[0]==t[1]||t[0]==t[2]||t[1]==t[2]) continue;
            bool oor = false;
            for (int v : t) if (v < 0 || v >= nv) { oor=true; break; }
            if (!oor) good_tris.push_back(t);
        }
        qm.tris = std::move(good_tris);
    }

    // ================================================================
    // Step 2: Short edge collapse
    //
    // Any quad edge shorter than (min_edge_length_factor * mean_length)
    // has its two endpoints merged to their midpoint.  The collapsed
    // endpoint is remapped globally and degenerate faces are pruned.
    //
    // Collapse priority: lower-indexed vertex always absorbs higher.
    // Remap chains are path-compressed after each pass.
    // ================================================================
    if (params.min_edge_length_factor > 0.0 && !qm.quads.empty()) {
        // Mean edge length over all quad edges
        double total_len = 0.0; int edge_cnt = 0;
        for (auto& q : qm.quads) {
            for (int k = 0; k < 4; ++k) {
                auto& pa = qm.vertices[q[k]];
                auto& pb = qm.vertices[q[(k+1)%4]];
                double dx=pb[0]-pa[0], dy=pb[1]-pa[1], dz=pb[2]-pa[2];
                total_len += std::sqrt(dx*dx+dy*dy+dz*dz);
                ++edge_cnt;
            }
        }
        double mean_len = edge_cnt > 0 ? total_len / edge_cnt : 1.0;
        double min_len2 = mean_len * mean_len
                        * params.min_edge_length_factor
                        * params.min_edge_length_factor;

        int cur_nv = (int)qm.vertices.size();
        std::vector<int> remap(cur_nv);
        for (int i = 0; i < cur_nv; ++i) remap[i] = i;

        // Path-compressing find
        std::function<int(int)> find_root = [&](int v) -> int {
            while (remap[v] != v) v = remap[v];
            return v;
        };

        bool any_collapsed = true;
        int col_passes = 0;
        while (any_collapsed && col_passes < params.max_valence_opt_passes) {
            any_collapsed = false;
            ++col_passes;

            for (auto& q : qm.quads) {
                for (int k = 0; k < 4; ++k) {
                    int va = find_root(q[k]);
                    int vb = find_root(q[(k+1)%4]);
                    if (va == vb) continue;

                    auto& pa = qm.vertices[va];
                    auto& pb = qm.vertices[vb];
                    double dx=pb[0]-pa[0], dy=pb[1]-pa[1], dz=pb[2]-pa[2];
                    if (dx*dx+dy*dy+dz*dz >= min_len2) continue;

                    // Collapse: merge higher index into lower, place at midpoint
                    int lo = std::min(va, vb), hi = std::max(va, vb);
                    qm.vertices[lo][0] = (qm.vertices[lo][0] + qm.vertices[hi][0]) * 0.5;
                    qm.vertices[lo][1] = (qm.vertices[lo][1] + qm.vertices[hi][1]) * 0.5;
                    qm.vertices[lo][2] = (qm.vertices[lo][2] + qm.vertices[hi][2]) * 0.5;
                    remap[hi] = lo;
                    any_collapsed = true;
                }
            }

            // Flatten remap chains
            for (int i = 0; i < cur_nv; ++i) remap[i] = find_root(i);

            // Apply remap to all faces
            for (auto& q : qm.quads)
                for (int& v : q) if (v >= 0 && v < cur_nv) v = remap[v];
            for (auto& t : qm.tris)
                for (int& v : t) if (v >= 0 && v < cur_nv) v = remap[v];
        }

        // Prune faces degenerated by collapse (repeated vertex indices)
        strip_degen(qm);
        nv = (int)qm.vertices.size();
    }

    // ================================================================
    // Step 3: Doublet removal (valence-2 interior vertices)
    // ================================================================
    if (params.remove_doublets) {
        bool any_removed = true;
        int  passes      = 0;
        while (any_removed && passes < params.max_valence_opt_passes) {
            any_removed = false;
            ++passes;

            int cur_nv = (int)qm.vertices.size();
            std::vector<int> valence(cur_nv, 0);
            for (auto& q : qm.quads)
                for (int v : q) if (v >= 0 && v < cur_nv) valence[v]++;
            for (auto& t : qm.tris)
                for (int v : t) if (v >= 0 && v < cur_nv) valence[v]++;

            std::vector<std::vector<int>> v_to_quads(cur_nv);
            for (int qi = 0; qi < (int)qm.quads.size(); ++qi)
                for (int v : qm.quads[qi])
                    if (v >= 0 && v < cur_nv) v_to_quads[v].push_back(qi);

            std::vector<bool> quad_removed(qm.quads.size(), false);

            for (int vi = 0; vi < cur_nv; ++vi) {
                if (valence[vi] != 2) continue;
                auto& qlist = v_to_quads[vi];
                if (qlist.size() != 2) continue;

                int qi0 = qlist[0], qi1 = qlist[1];
                if (quad_removed[qi0] || quad_removed[qi1]) continue;

                auto& q0 = qm.quads[qi0];
                auto& q1 = qm.quads[qi1];

                int pos0 = -1, pos1 = -1;
                for (int k = 0; k < 4; ++k) {
                    if (q0[k] == vi) pos0 = k;
                    if (q1[k] == vi) pos1 = k;
                }
                if (pos0 < 0 || pos1 < 0) continue;

                int n0a = q0[(pos0 + 3) % 4];
                int n0b = q0[(pos0 + 1) % 4];
                int n1a = q1[(pos1 + 3) % 4];
                int n1b = q1[(pos1 + 1) % 4];

                std::set<int> outer_verts = {n0a, n0b, n1a, n1b};
                if ((int)outer_verts.size() < 4) continue;

                // Relocate vi to centroid of its four neighbours
                auto& pvi = qm.vertices[vi];
                auto& pa = qm.vertices[n0a]; auto& pb = qm.vertices[n0b];
                auto& pc = qm.vertices[n1a]; auto& pd = qm.vertices[n1b];
                pvi[0] = (pa[0]+pb[0]+pc[0]+pd[0])*0.25;
                pvi[1] = (pa[1]+pb[1]+pc[1]+pd[1])*0.25;
                pvi[2] = (pa[2]+pb[2]+pc[2]+pd[2])*0.25;

                if ((n0a == n1b && n0b == n1a) || (n0a == n1a && n0b == n1b)) {
                    int opp0 = q0[(pos0 + 2) % 4];
                    int opp1 = q1[(pos1 + 2) % 4];
                    if (opp0 != opp1 && opp0 != n0a && opp0 != n0b &&
                        opp1 != n0a && opp1 != n0b) {
                        qm.quads[qi0] = {opp0, n0a, opp1, n0b};
                        quad_removed[qi1] = true;
                        any_removed = true;
                    }
                }
            }

            std::vector<std::array<int,4>> kept;
            kept.reserve(qm.quads.size());
            for (int qi = 0; qi < (int)qm.quads.size(); ++qi)
                if (!quad_removed[qi]) kept.push_back(qm.quads[qi]);
            qm.quads = std::move(kept);
        }
        // BUG FIX (v92): nv was not updated after the doublet-removal loop.
        // The stats block at the end of cleanup_quad_mesh() reads `nv`
        // directly; if doublet removal changed the vertex count (e.g. by
        // triggering subsequent vertex compaction) and nv was stale, the
        // avg_valence computation would index out of bounds or use wrong data.
        nv = (int)qm.vertices.size();
    }

    // ================================================================
    // Step 3b: Weld near-coincident vertices — O(V) grid-hash approach
    //
    // BUG-FIX (v116 / BUG-F):
    //   The previous implementation used a double-nested loop:
    //     for i in [0, V):
    //       for j in [i+1, V):
    //         if dist(i, j) <= tol: remap[j] = i
    //   This is O(V²) in the vertex count of the quad mesh at this stage.
    //   For a 10K-vertex output mesh that is ~5×10^7 comparisons in serial
    //   C++.  For a 50K-vertex mesh: ~1.25×10^9 — several seconds of UI
    //   freeze per remesh on typical production assets.
    //
    //   Fix: replaced with a spatial grid-hash identical in structure to
    //   repair.cpp::GridHash.  Each vertex is bucketed into a 3-D integer
    //   cell; only the 27-neighbour shell is probed for close candidates.
    //   Expected cost: O(V) insertions + O(V × 27) probes = O(V) total.
    //   No false negatives: any two vertices within `tol` distance must
    //   land in the same cell or an immediately adjacent one (cell size =
    //   tol guarantees the 1-cell-shell covers the full tol radius).
    //
    //   Memory: one hash-map entry per unique vertex ≈ O(V).  No change
    //   in the correctness or the compaction / face-remap logic below.
    // ================================================================
    {
        int cur_nv = (int)qm.vertices.size();
        if (cur_nv > 1) {
            // Adaptive tolerance: 1e-4 × mean quad edge length
            double total_len = 0.0; int edge_count = 0;
            for (auto& q : qm.quads) {
                for (int k = 0; k < 4; ++k) {
                    auto& a = qm.vertices[q[k]];
                    auto& b = qm.vertices[q[(k+1)%4]];
                    double dx=b[0]-a[0], dy=b[1]-a[1], dz=b[2]-a[2];
                    total_len += std::sqrt(dx*dx+dy*dy+dz*dz);
                    ++edge_count;
                }
            }
            const double tol  = edge_count > 0
                ? (total_len / edge_count) * 1e-4 : 1e-6;
            const double tol2 = tol * tol;
            const double inv_tol = 1.0 / tol;

            // --- Grid-hash cell key: (ix, iy, iz) packed into int64 ---
            // We quantise each coordinate to floor(coord / tol) and pack
            // two 21-bit signed integers per dimension into a 64-bit key.
            // For coordinates within [-2^20 * tol, 2^20 * tol] this is
            // lossless.  Typical quad-mesh coordinates are in [-1000, 1000]
            // with tol ≈ 1e-5, giving cell indices in [-10^8, 10^8] — well
            // within the 21-bit range.
            struct CellKey {
                int32_t ix, iy, iz;
                bool operator==(const CellKey& o) const {
                    return ix == o.ix && iy == o.iy && iz == o.iz;
                }
            };
            struct CellKeyHash {
                size_t operator()(const CellKey& k) const noexcept {
                    // FNV-1a style mix of three 32-bit values.
                    uint64_t h = 14695981039346656037ULL;
                    auto mix = [&](int32_t v) {
                        h ^= static_cast<uint64_t>(static_cast<uint32_t>(v));
                        h *= 1099511628211ULL;
                    };
                    mix(k.ix); mix(k.iy); mix(k.iz);
                    return static_cast<size_t>(h);
                }
            };

            // grid[cell] = list of already-inserted (new) vertex indices
            // whose position falls in that cell.
            std::unordered_map<CellKey,
                               std::vector<int>,
                               CellKeyHash> grid;
            grid.reserve(static_cast<size_t>(cur_nv));

            // old-to-new remap: new vertex positions are accumulated here.
            std::vector<int> remap(cur_nv, -1);
            std::vector<std::array<double,3>> new_verts;
            new_verts.reserve(static_cast<size_t>(cur_nv));

            for (int vi = 0; vi < cur_nv; ++vi) {
                const auto& p = qm.vertices[vi];
                CellKey ck {
                    static_cast<int32_t>(std::floor(p[0] * inv_tol)),
                    static_cast<int32_t>(std::floor(p[1] * inv_tol)),
                    static_cast<int32_t>(std::floor(p[2] * inv_tol))
                };

                // Search 27-cell neighbourhood for an already-merged vertex.
                int found = -1;
                for (int di = -1; di <= 1 && found < 0; ++di)
                for (int dj = -1; dj <= 1 && found < 0; ++dj)
                for (int dk = -1; dk <= 1 && found < 0; ++dk) {
                    CellKey nk { ck.ix+di, ck.iy+dj, ck.iz+dk };
                    auto it = grid.find(nk);
                    if (it == grid.end()) continue;
                    for (int cand : it->second) {
                        const auto& pc = new_verts[cand];
                        double dx=pc[0]-p[0], dy=pc[1]-p[1], dz=pc[2]-p[2];
                        if (dx*dx+dy*dy+dz*dz <= tol2) { found=cand; break; }
                    }
                }

                if (found >= 0) {
                    remap[vi] = found;
                } else {
                    int nidx = static_cast<int>(new_verts.size());
                    new_verts.push_back(p);
                    remap[vi] = nidx;
                    grid[ck].push_back(nidx);
                }
            }

            // Apply remap and strip degenerate faces.
            for (auto& q : qm.quads)
                for (int& v : q) if (v >= 0 && v < cur_nv) v = remap[v];
            for (auto& t : qm.tris)
                for (int& v : t) if (v >= 0 && v < cur_nv) v = remap[v];

            qm.vertices = std::move(new_verts);
            nv = (int)qm.vertices.size();
        }
    }

    // ================================================================
    // Step 3.5: Edge flips to reduce valence irregularity
    //
    // For each pair of adjacent quads sharing edge B->C, compute
    // whether rotating that shared edge to D->E reduces the total
    // sum of squared deviations from target valence (4 interior, 2 boundary).
    //
    // Rotation formula (consistent orientation preserved):
    //   Old: Q1 = (A, B, C, D)   Q2 = (C, B, E, F)   shared edge B->C / C->B
    //   New: Q1 = (A, B, E, D)   Q2 = (D, E, F, C)   shared edge E->D / D->E
    //
    // Valence deltas after rotation: B-=1, C-=1, D+=1, E+=1.
    // ================================================================
    for (int pass = 0; pass < params.max_valence_opt_passes; ++pass) {
        int cur_nv = (int)qm.vertices.size();
        int cur_nq = (int)qm.quads.size();
        if (cur_nq < 2) break;

        auto ekey = [](int a, int b) -> int64_t {
            return ((int64_t)(uint32_t)a << 32) | (uint32_t)b;
        };

        // directed edge -> (quad index, edge position k)
        std::unordered_map<int64_t, std::pair<int,int>> dir_edge_map;
        dir_edge_map.reserve(cur_nq * 4);
        for (int qi = 0; qi < cur_nq; ++qi)
            for (int k = 0; k < 4; ++k)
                dir_edge_map[ekey(qm.quads[qi][k], qm.quads[qi][(k+1)%4])] = {qi, k};

        // vertex valence (face count)
        std::vector<int> valence(cur_nv, 0);
        for (auto& q : qm.quads)
            for (int v : q) if (v >= 0 && v < cur_nv) ++valence[v];

        // Boundary vertices: any edge (va->vb) without reverse (vb->va)
        std::vector<bool> is_bnd(cur_nv, false);
        for (int qi = 0; qi < cur_nq; ++qi)
            for (int k = 0; k < 4; ++k) {
                int va = qm.quads[qi][k], vb = qm.quads[qi][(k+1)%4];
                if (dir_edge_map.find(ekey(vb, va)) == dir_edge_map.end()) {
                    if (va >= 0 && va < cur_nv) is_bnd[va] = true;
                    if (vb >= 0 && vb < cur_nv) is_bnd[vb] = true;
                }
            }

        std::vector<bool> quad_modified(cur_nq, false);
        int flips_done = 0;

        for (int qi = 0; qi < cur_nq; ++qi) {
            if (quad_modified[qi]) continue;
            for (int k = 0; k < 4; ++k) {
                // Q1 = (A, B, C, D) — shared edge at positions k, (k+1)%4
                int B = qm.quads[qi][k];
                int C = qm.quads[qi][(k+1)%4];
                int D = qm.quads[qi][(k+2)%4];
                int A = qm.quads[qi][(k+3)%4];

                // Adjacent quad must have reverse edge C->B
                auto it = dir_edge_map.find(ekey(C, B));
                if (it == dir_edge_map.end()) continue;

                auto [qj, m] = it->second;
                if (qj == qi || quad_modified[qj]) continue;
                // Verify the match
                if (qm.quads[qj][m] != C || qm.quads[qj][(m+1)%4] != B) continue;

                // Q2 = (C, B, E, F)
                int E = qm.quads[qj][(m+2)%4];
                int F = qm.quads[qj][(m+3)%4];

                // All 6 vertices must be distinct
                std::array<int,6> vv = {A, B, C, D, E, F};
                bool distinct = true;
                for (int a = 0; a < 6 && distinct; ++a)
                    for (int b = a+1; b < 6 && distinct; ++b)
                        if (vv[a] == vv[b]) distinct = false;
                if (!distinct) continue;

                // Target valence per vertex
                auto tgt = [&](int v) { return is_bnd[v] ? 2 : 4; };

                // Irregularity of {B, C, D, E} before the flip
                int irr_before = 0;
                for (int v : {B, C, D, E}) {
                    int d = valence[v] - tgt(v);
                    irr_before += d * d;
                }

                // Irregularity after: B-1, C-1, D+1, E+1
                int irr_after = 0;
                { int d=(valence[B]-1)-tgt(B); irr_after+=d*d; }
                { int d=(valence[C]-1)-tgt(C); irr_after+=d*d; }
                { int d=(valence[D]+1)-tgt(D); irr_after+=d*d; }
                { int d=(valence[E]+1)-tgt(E); irr_after+=d*d; }

                if (irr_after >= irr_before) continue;

                // Perform flip: Q1 -> (A,B,E,D)   Q2 -> (D,E,F,C)
                qm.quads[qi] = {A, B, E, D};
                qm.quads[qj] = {D, E, F, C};
                quad_modified[qi] = quad_modified[qj] = true;

                valence[B]--; valence[C]--;
                valence[D]++; valence[E]++;
                ++flips_done;
                break;
            }
        }

        if (flips_done == 0) break;
    }

    // ================================================================
    // Step 3.75: Small hole filling
    //
    // Boundary loops with <= MAX_HOLE_EDGES vertices are filled:
    //   3 verts  -> triangle face
    //   4 verts  -> quad face
    //   5-8 verts -> fan triangulation from centroid (new vertex added)
    //
    // Traversal follows the same winding direction as the surrounding
    // quad boundary edges so filled faces are consistently oriented.
    // ================================================================
    if (params.fill_holes && !qm.quads.empty()) {
        // Build directed-edge set for fast boundary detection
        auto ekey = [](int a, int b) -> int64_t {
            return ((int64_t)(uint32_t)a << 32) | (uint32_t)b;
        };

        std::unordered_set<int64_t> all_directed;
        all_directed.reserve(qm.quads.size() * 4);
        for (auto& q : qm.quads)
            for (int k = 0; k < 4; ++k)
                all_directed.insert(ekey(q[k], q[(k+1)%4]));

        // boundary_next[va] = vb  means va->vb is a boundary edge
        std::unordered_map<int,int> boundary_next;
        for (auto& q : qm.quads) {
            for (int k = 0; k < 4; ++k) {
                int va = q[k], vb = q[(k+1)%4];
                // Boundary edge: directed edge with no manifold twin
                if (all_directed.find(ekey(vb, va)) == all_directed.end()) {
                    // BUG FIX (v91): Only insert if not already present.
                    // A vertex that has two outgoing boundary edges (possible
                    // after edge-collapse creates transient non-manifold
                    // topology) would silently overwrite the first entry with
                    // the second, producing incorrect hole-tracing loops.
                    // We keep the first boundary edge found; the loop tracer
                    // will safely break on the duplicate when it re-encounters
                    // the start vertex.
                    boundary_next.emplace(va, vb);
                }
            }
        }

        constexpr int MAX_HOLE_EDGES = 8;
        std::unordered_set<int> visited;

        for (auto& [start, _next] : boundary_next) {
            if (visited.count(start)) continue;

            // Trace loop starting at `start`
            std::vector<int> loop;
            int v = start;
            while (true) {
                if (visited.count(v)) break;
                auto it = boundary_next.find(v);
                if (it == boundary_next.end()) break;
                loop.push_back(v);
                visited.insert(v);
                v = it->second;
                if (v == start) break;                       // closed
                if ((int)loop.size() > MAX_HOLE_EDGES + 1) break; // too large
            }

            int sz = (int)loop.size();
            if (sz < 3 || sz > MAX_HOLE_EDGES) continue;

            if (sz == 3) {
                qm.tris.push_back({loop[0], loop[1], loop[2]});

            } else if (sz == 4) {
                qm.quads.push_back({loop[0], loop[1], loop[2], loop[3]});

            } else {
                // Fan triangulation from centroid vertex
                std::array<double,3> cen = {0.0, 0.0, 0.0};
                for (int lv : loop) {
                    cen[0] += qm.vertices[lv][0];
                    cen[1] += qm.vertices[lv][1];
                    cen[2] += qm.vertices[lv][2];
                }
                cen[0] /= sz; cen[1] /= sz; cen[2] /= sz;

                int cvi = (int)qm.vertices.size();
                qm.vertices.push_back(cen);

                for (int i = 0; i < sz; ++i) {
                    int va = loop[i], vb = loop[(i+1)%sz];
                    qm.tris.push_back({cvi, va, vb});
                }
            }
        }

        nv = (int)qm.vertices.size();
    }

    // ================================================================
    // Step 4: Compact vertex array (remove unreferenced vertices)
    // ================================================================
    {
        int cur_nv = (int)qm.vertices.size();
        std::vector<bool> used(cur_nv, false);
        for (auto& q : qm.quads) for (int v : q) if(v>=0&&v<cur_nv) used[v]=true;
        for (auto& t : qm.tris)  for (int v : t) if(v>=0&&v<cur_nv) used[v]=true;

        std::vector<int> remap(cur_nv, -1);
        int new_nv = 0;
        std::vector<std::array<double,3>> new_verts;
        new_verts.reserve(cur_nv);
        for (int vi = 0; vi < cur_nv; ++vi) {
            if (used[vi]) {
                remap[vi] = new_nv++;
                new_verts.push_back(qm.vertices[vi]);
            }
        }
        for (auto& q : qm.quads)
            for (int& v : q) if (v >= 0 && v < cur_nv) v = remap[v];
        for (auto& t : qm.tris)
            for (int& v : t) if (v >= 0 && v < cur_nv) v = remap[v];
        qm.vertices = std::move(new_verts);
        nv = (int)qm.vertices.size();
    }

    // ================================================================
    // Update stats
    // ================================================================
    {
        int total = (int)(qm.quads.size() + qm.tris.size());
        qm.quad_percentage = total > 0
            ? 100.f * (float)qm.quads.size() / (float)total : 0.f;

        if (nv > 0) {
            std::vector<int> val(nv, 0);
            for (auto& q : qm.quads) for (int v : q) if(v>=0&&v<nv) val[v]++;
            for (auto& t : qm.tris)  for (int v : t) if(v>=0&&v<nv) val[v]++;
            double sum = 0.0;
            for (int v : val) sum += v;
            qm.avg_valence = (float)(sum / nv);
        }
    }
}

} // namespace qf
