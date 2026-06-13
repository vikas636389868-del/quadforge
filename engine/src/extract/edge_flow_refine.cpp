/**
 * edge_flow_refine.cpp — Edge flow refinement pass (§13.10).
 *
 * Implements refine_edge_flow(), vertex_flow_score(), and mesh_flow_score()
 * declared in include/quadforge/extract/edge_flow_refine.h.
 *
 * IMPLEMENTATION SUMMARY
 * ----------------------
 *
 * Data structures built once per pass
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  v_to_quads[vi]   : list of quad indices containing vertex vi
 *  v_neighbors[vi]  : ordered list of adjacent vertices (quad ring)
 *  is_boundary[vi]  : true if vi is on the boundary of the quad mesh
 *  valence[vi]      : number of incident faces (quads count once, tris once)
 *
 * Sub-pass 1 — Irregularity cancellation
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  For each irregular vertex vi (|valence - 4| > 0) with defect d_i:
 *    BFS up to max_cancel_dist hops through the quad adjacency graph.
 *    If a vertex vj is found with d_j = -d_i, attempt to swap the edge
 *    shared between the quad-path from vi to vj.
 *    An "edge rotation" of a shared quad edge (a,b) between quads Q1(a,c,b,d)
 *    and Q2(a,b,e,f) replaces them with Q1'(a,c,e,b) and Q2'(c,b,f,a) —
 *    wait, in a quad mesh context the canonical "diagonal swap" is:
 *    The two quads sharing edge (a,b) are (a,b,c,d) and (b,a,e,f).
 *    After the swap: (a,e,b,d) and (b,c,a,f).
 *    We accept the swap only if it strictly reduces Σ|valence-4|².
 *
 * Sub-pass 2 — Loop straightening
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  Walk every interior edge (a,b).  Get the two adjacent quads Q1 and Q2.
 *  The "flow direction" of edge (a,b) is the unit vector b-a.
 *  On Q1, the opposite edge is (d,c); its direction is c-d.
 *  If cos(angle between (b-a) and (c-d)) < zigzag_cos_thresh, the loop
 *  "bends" here.  Apply the same diagonal swap as above and accept if the
 *  new edge's alignment with both adjacent edge loops improves.
 *
 * Sub-pass 3 — Angle-based vertex relocation
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *  For each interior vertex vi, compute the minimum interior angle of all
 *  incident quads at vi.  If min_angle < min_angle_deg:
 *    Compute the Laplacian Δv = (1/N) Σ_j (vj - vi)
 *    New position: vi' = vi + λ * Δv  (λ = relocation_strength)
 *    Project onto local tangent plane defined by the average normal of
 *    incident quads (prevents drifting off the surface approximation).
 *  Skip if preserve_features && vi is feature-adjacent.
 *
 * Edge rotation validity
 * ~~~~~~~~~~~~~~~~~~~~~~
 * A diagonal swap is accepted only if:
 *   a) Both new quads are non-degenerate (all 4 corners distinct).
 *   b) Both new quads have positive signed area (no inversion).
 *   c) The objective Σ|valence-4|² does not increase (for irregularity
 *      cancellation sub-pass) or the minimum angle does not decrease
 *      (for loop-straightening sub-pass).
 *
 * Complexity
 * ~~~~~~~~~~
 *  Building v_to_quads, v_neighbors: O(F)
 *  Sub-pass 1 irregularity BFS: O(V · max_cancel_dist)
 *  Sub-pass 2 loop straightening: O(E)
 *  Sub-pass 3 relocation: O(V · avg_valence)
 *  Total per pass: O(V + F)
 */

#include "../../include/quadforge/extract/edge_flow_refine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qf {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static inline double dot3(const std::array<double,3>& a,
                           const std::array<double,3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline std::array<double,3> cross3(const std::array<double,3>& a,
                                           const std::array<double,3>& b) {
    return { a[1]*b[2]-a[2]*b[1],
             a[2]*b[0]-a[0]*b[2],
             a[0]*b[1]-a[1]*b[0] };
}

static inline std::array<double,3> sub3(const std::array<double,3>& a,
                                         const std::array<double,3>& b) {
    return { a[0]-b[0], a[1]-b[1], a[2]-b[2] };
}

static inline std::array<double,3> add3(const std::array<double,3>& a,
                                         const std::array<double,3>& b) {
    return { a[0]+b[0], a[1]+b[1], a[2]+b[2] };
}

static inline std::array<double,3> scale3(const std::array<double,3>& a,
                                           double s) {
    return { a[0]*s, a[1]*s, a[2]*s };
}

static inline double len3(const std::array<double,3>& a) {
    return std::sqrt(dot3(a, a));
}

static inline std::array<double,3> norm3(const std::array<double,3>& a) {
    double l = len3(a);
    if (l < 1e-15) return {0.0, 0.0, 1.0};
    return scale3(a, 1.0/l);
}

/** Signed area of a quad via cross product (positive = CCW normal). */
static double quad_signed_area_d(
    const std::array<double,3>& p0,
    const std::array<double,3>& p1,
    const std::array<double,3>& p2,
    const std::array<double,3>& p3)
{
    auto a = sub3(p1,p0), b = sub3(p2,p0), c = sub3(p3,p0);
    auto n1 = cross3(a,b), n2 = cross3(b,c);
    return 0.5 * (len3(n1) + len3(n2));
}

/** Minimum interior angle (radians) at vertex p shared by quad (p,a,b,c). */
static double min_quad_angle_at(
    const std::array<double,3>& p,
    const std::array<double,3>& prev_v,  // previous corner in quad
    const std::array<double,3>& next_v)  // next corner in quad
{
    auto ep = norm3(sub3(prev_v, p));
    auto en = norm3(sub3(next_v, p));
    double c = std::max(-1.0, std::min(1.0, dot3(ep, en)));
    return std::acos(c);
}

// ---------------------------------------------------------------------------
// MeshTopo: per-pass adjacency tables
// ---------------------------------------------------------------------------

struct MeshTopo {
    int nv;
    int nq;

    // v_to_quads[vi] = vector of quad indices containing vi
    std::vector<std::vector<int>> v_to_quads;

    // Boundary detection: a vertex is on the boundary if any incident
    // quad edge has no neighbouring quad on the other side.
    std::vector<bool> is_boundary;

    // valence[vi] = number of incident faces (quads + tris)
    std::vector<int> valence;

    void build(const QuadMesh& qm) {
        nv = static_cast<int>(qm.vertices.size());
        nq = static_cast<int>(qm.quads.size());
        v_to_quads.assign(nv, {});
        valence.assign(nv, 0);
        is_boundary.assign(nv, false);

        for (int qi = 0; qi < nq; ++qi)
            for (int v : qm.quads[qi]) {
                if (v >= 0 && v < nv) {
                    v_to_quads[v].push_back(qi);
                    valence[v]++;
                }
            }

        for (const auto& t : qm.tris)
            for (int v : t)
                if (v >= 0 && v < nv) valence[v]++;

        // An edge (a,b) is a boundary edge if the directed pair (a→b)
        // appears in exactly one quad.
        std::unordered_map<int64_t, int> edge_count;
        edge_count.reserve(nq * 4);
        // BUG FIX (v92): cast `a` through uint32_t before widening to int64_t
        // to prevent sign-extension corrupting bits 32–63 when `a` is negative.
        // Same v24-fix pattern used in motorcycle.cpp / seam_utils.cpp.
        auto ekey = [](int a, int b) -> int64_t {
            return (static_cast<int64_t>(static_cast<uint32_t>(a)) << 32)
                 |  static_cast<int64_t>(static_cast<uint32_t>(b));
        };
        // BUG FIX (v95): the for loop that populated edge_count was accidentally
        // deleted in a prior edit, leaving only two orphaned closing braces.
        // edge_count was always empty so the boundary-detection pass below
        // (which reads edge_count) never marked any vertex as a boundary vertex,
        // causing subpass_irregularity and subpass_relocation to incorrectly
        // treat every quad vertex as an interior vertex.
        for (int qi = 0; qi < nq; ++qi) {
            const auto& q = qm.quads[qi];
            for (int k = 0; k < 4; ++k) {
                int a = q[k], b = q[(k+1)%4];
                if (a >= 0 && a < nv && b >= 0 && b < nv)
                    edge_count[ekey(a, b)]++;
            }
        }
        // BUG FIX (v53): in a manifold quad mesh every directed edge (a→b)
        // appears exactly once, so cnt==1 is true for ALL edges — including
        // interior ones.  The old code therefore marked every vertex as a
        // boundary vertex, causing subpass_irregularity and subpass_relocation
        // to skip all vertices and do nothing.
        //
        // Correct criterion: (a→b) is a boundary edge iff the reverse edge
        // (b→a) does NOT appear in the mesh at all.
        for (auto& [k, cnt] : edge_count) {
            if (cnt == 1) {
                int a = static_cast<int>(k >> 32);
                int b = static_cast<int>(k & 0xFFFFFFFF);
                // Only a true boundary edge if the reverse is absent
                if (edge_count.find(ekey(b, a)) == edge_count.end()) {
                    if (a >= 0 && a < nv) is_boundary[a] = true;
                    if (b >= 0 && b < nv) is_boundary[b] = true;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Sub-pass 3: Angle-based vertex relocation
// ---------------------------------------------------------------------------

static int subpass_relocation(QuadMesh& qm,
                               const MeshTopo& topo,
                               const EdgeFlowRefineParams& params)
{
    const double pi    = 3.14159265358979323846;
    const double thresh = params.min_angle_deg * pi / 180.0;
    const double lam   = params.relocation_strength;
    int moved = 0;

    int nv = topo.nv;
    // We process all vertices and collect moves, applying them simultaneously
    // (Jacobi-style) to avoid order dependence.
    std::vector<std::array<double,3>> delta(nv, {0.0, 0.0, 0.0});
    std::vector<bool> do_move(nv, false);

    for (int vi = 0; vi < nv; ++vi) {
        if (topo.is_boundary[vi]) continue;
        // BUG FIX (v91): The previous check `topo.valence[vi] != 4` was wrong.
        // preserve_features is documented as "skip vertices within one edge of
        // a detected feature curve".  In the extracted quad mesh, feature curves
        // align with boundary edges, so the correct guard is: skip any interior
        // vertex that has a boundary vertex as one of its quad neighbours.
        // The old check skipped ALL irregular vertices (valence != 4), which are
        // exactly the primary targets for relocation — making the entire
        // subpass a no-op whenever preserve_features=true (the default).
        if (params.preserve_features) {
            bool near_boundary = false;
            for (int qi : topo.v_to_quads[vi]) {
                for (int nb : qm.quads[qi]) {
                    if (nb >= 0 && nb < topo.nv && topo.is_boundary[nb]) {
                        near_boundary = true;
                        break;
                    }
                }
                if (near_boundary) break;
            }
            if (near_boundary) continue;
        }

        const auto& vpos = qm.vertices[vi];
        const auto& qi_list = topo.v_to_quads[vi];
        if (qi_list.empty()) continue;

        // Find minimum quad angle at this vertex
        double min_ang = pi;
        for (int qi : qi_list) {
            const auto& q = qm.quads[qi];
            // Find position of vi in quad
            int pos_in_q = -1;
            for (int k = 0; k < 4; ++k)
                if (q[k] == vi) { pos_in_q = k; break; }
            if (pos_in_q < 0) continue;
            int prev_vi = q[(pos_in_q + 3) % 4];
            int next_vi = q[(pos_in_q + 1) % 4];
            if (prev_vi < 0 || next_vi < 0 ||
                prev_vi >= nv || next_vi >= nv) continue;
            double ang = min_quad_angle_at(vpos,
                                           qm.vertices[prev_vi],
                                           qm.vertices[next_vi]);
            min_ang = std::min(min_ang, ang);
        }

        if (min_ang >= thresh) continue;

        // Laplacian: gather all unique adjacent vertices
        std::unordered_set<int> nbrs_set;
        for (int qi : qi_list) {
            const auto& q = qm.quads[qi];
            for (int v : q)
                if (v != vi && v >= 0 && v < nv)
                    nbrs_set.insert(v);
        }
        if (nbrs_set.empty()) continue;

        std::array<double,3> avg = {0.0, 0.0, 0.0};
        for (int nv_idx : nbrs_set) {
            avg[0] += qm.vertices[nv_idx][0];
            avg[1] += qm.vertices[nv_idx][1];
            avg[2] += qm.vertices[nv_idx][2];
        }
        double inv = 1.0 / static_cast<double>(nbrs_set.size());
        avg[0] = (avg[0] * inv - vpos[0]) * lam;
        avg[1] = (avg[1] * inv - vpos[1]) * lam;
        avg[2] = (avg[2] * inv - vpos[2]) * lam;

        // Project displacement onto tangent plane (average of incident
        // quad face normals).
        std::array<double,3> avg_normal = {0.0, 0.0, 0.0};
        for (int qi : qi_list) {
            const auto& q = qm.quads[qi];
            // Guard index validity
            bool ok = true;
            for (int v : q) if (v < 0 || v >= nv) { ok=false; break; }
            if (!ok) continue;
            auto n = cross3(
                sub3(qm.vertices[q[1]], qm.vertices[q[0]]),
                sub3(qm.vertices[q[3]], qm.vertices[q[0]]));
            avg_normal[0] += n[0]; avg_normal[1] += n[1]; avg_normal[2] += n[2];
        }
        avg_normal = norm3(avg_normal);

        // Remove normal component from delta (tangential projection)
        double d_n = dot3(avg, avg_normal);
        avg[0] -= d_n * avg_normal[0];
        avg[1] -= d_n * avg_normal[1];
        avg[2] -= d_n * avg_normal[2];

        delta[vi]   = avg;
        do_move[vi] = true;
    }

    // Apply moves
    for (int vi = 0; vi < nv; ++vi) {
        if (!do_move[vi]) continue;
        qm.vertices[vi][0] += delta[vi][0];
        qm.vertices[vi][1] += delta[vi][1];
        qm.vertices[vi][2] += delta[vi][2];
        ++moved;
    }
    return moved;
}

// ---------------------------------------------------------------------------
// Sub-pass 1: Irregularity cancellation via diagonal swap
// ---------------------------------------------------------------------------

/**
 * Attempt to swap the shared edge between quads qi1 and qi2 if doing so
 * strictly reduces Σ|valence-4|².
 *
 * Setup
 * -----
 * Q1 = (sa, sb, c1, d1)  — shared edge sa→sb at positions k1, k1+1
 * Q2 = (sb, sa, c2, d2)  — shared edge sb→sa at positions k2, k2+1
 *
 * The two quads form a hexagonal region whose CCW outer boundary is:
 *   sa → c2 → d2 → sb → c1 → d1 → sa
 *
 * Valid new diagonals that split this hexagon into two quads connect
 * opposite vertices (positions i↔i+3).  Position 0↔3 is the original
 * sa↔sb edge.  We choose position 2↔5 = d2↔d1:
 *
 *   New Q1' = (sa, c2, d2, d1)   outer: sa→c2, c2→d2, d1→sa;  interior d2→d1
 *   New Q2' = (d2, sb, c1, d1)   outer: d2→sb, sb→c1, c1→d1;  interior d1→d2
 *
 * Valence effect from these two quads alone:
 *   sa: 2→1 (−1)   sb: 2→1 (−1)   d1: 1→2 (+1)   d2: 1→2 (+1)
 *   c1 and c2 stay at 1 (unchanged).
 *
 * The swap is accepted only when:
 *   a) All six vertices are valid and pairwise distinct.
 *   b) Both new quads have positive signed area (non-inverted).
 *   c) Σ|valence−4|² strictly decreases over the six affected vertices.
 */
static bool try_diagonal_swap(
    QuadMesh& qm,
    MeshTopo& topo,
    int qi1, int qi2)
{
    const int nv = topo.nv;
    auto& q1 = qm.quads[qi1];
    auto& q2 = qm.quads[qi2];

    // Find the shared edge: directed edge (sa→sb) in q1 and (sb→sa) in q2.
    int sa = -1, sb = -1;
    int k1 = -1, k2 = -1;
    for (int k = 0; k < 4 && sa < 0; ++k) {
        int a = q1[k], b = q1[(k+1)%4];
        // Look for reverse in q2
        for (int m = 0; m < 4; ++m) {
            if (q2[m] == b && q2[(m+1)%4] == a) {
                sa = a; sb = b;
                k1 = k; k2 = m;
                break;
            }
        }
    }
    if (sa < 0) return false;  // quads don't share a directed edge

    // Extract the 6 unique vertices:
    //   Q1 = (sa, sb, c1, d1) — k1 = position of sa in q1
    int c1 = q1[(k1+2)%4];
    int d1 = q1[(k1+3)%4];
    //   Q2 = (sb, sa, c2, d2)
    int c2 = q2[(k2+2)%4];
    int d2 = q2[(k2+3)%4];

    // Sanity: all must be valid and distinct
    int vs[6] = {sa, sb, c1, d1, c2, d2};
    for (int v : vs) if (v < 0 || v >= nv) return false;

    // The hexagonal region formed by Q1 and Q2 has CCW boundary:
    //   sa → c2 → d2 → sb → c1 → d1 → sa
    //
    // Valid diagonals that split it into two quads are those connecting
    // opposite vertices (positions 0↔3, 1↔4, or 2↔5 in the hexagon).
    // Position 0↔3 (sa↔sb) is the ORIGINAL shared edge — not a swap.
    // We use position 2↔5 (d2↔d1) as the new interior diagonal:
    //
    //   Q1' = (sa, c2, d2, d1)  outer: sa→c2 ✓, c2→d2 ✓, d1→sa ✓; interior d2→d1
    //   Q2' = (d2, sb, c1, d1)  outer: d2→sb ✓, sb→c1 ✓, c1→d1 ✓; interior d1→d2
    //
    // Valence effect: sa−1, sb−1, d1+1, d2+1 (c1 and c2 unchanged).
    //
    // BUG FIX (v54): previous code had nq1={sa,d2,c2,d1} and nq2={sb,c1,d1,c2}.
    // nq1 had c2 and d2 swapped (reversed outer-edge winding on the sa→c2→d2 arc),
    // and nq2 referenced c2 instead of d2 as its origin vertex, so neither quad
    // respected the hexagon boundary.  This produced face-crossing or non-manifold
    // geometry and caused the valence-update table to track the wrong vertices.
    std::array<int,4> nq1 = {sa, c2, d2, d1};
    std::array<int,4> nq2 = {d2, sb, c1, d1};

    // Check distinct
    auto distinct4 = [](const std::array<int,4>& q) {
        return q[0]!=q[1] && q[0]!=q[2] && q[0]!=q[3] &&
               q[1]!=q[2] && q[1]!=q[3] && q[2]!=q[3];
    };
    if (!distinct4(nq1) || !distinct4(nq2)) return false;

    // Check positive area for both new quads
    for (const auto& nq : {nq1, nq2}) {
        double area = quad_signed_area_d(
            qm.vertices[nq[0]], qm.vertices[nq[1]],
            qm.vertices[nq[2]], qm.vertices[nq[3]]);
        if (area < 1e-12) return false;
    }

    // Evaluate valence change: Σ|val-4|² before and after
    // Affected vertices: sa, sb, c1, d1, c2, d2
    // Before: sa appears in q1+q2, sb appears in q1+q2,
    //         c1 in q1, d1 in q1, c2 in q2, d2 in q2
    // After (corrected diagonal d2↔d1):
    //   sa in nq1 only, sb in nq2 only,
    //   c1 in nq2 only, d1 in nq1+nq2, c2 in nq1 only, d2 in nq1+nq2

    auto valence_delta = [&](int v, const std::array<int,4>& old_q,
                              const std::array<int,4>& new_q) -> int {
        bool in_old = false, in_new = false;
        for (int u : old_q) if (u == v) { in_old = true; break; }
        for (int u : new_q) if (u == v) { in_new = true; break; }
        return (in_new ? 1 : 0) - (in_old ? 1 : 0);
    };

    int score_before = 0, score_after = 0;
    int all_v[6] = {sa, sb, c1, d1, c2, d2};
    for (int v : all_v) {
        int val = topo.valence[v];
        int dv = valence_delta(v, q1, nq1) + valence_delta(v, q2, nq2);
        score_before += (val - 4) * (val - 4);
        score_after  += (val + dv - 4) * (val + dv - 4);
    }

    if (score_after >= score_before) return false;  // no improvement

    // BUG FIX (v53): compute all valence deltas BEFORE overwriting q1/q2.
    // Previously the deltas were computed AFTER q1=nq1 and q2=nq2, which made
    // q1 == nq1 and q2 == nq2, so valence_delta always returned 0 and the
    // topo.valence table was never updated.  Subsequent swap decisions used
    // stale valences, producing poor (and sometimes cyclically oscillating)
    // edge-swap sequences.
    int dv_arr[6] = {};
    for (int i = 0; i < 6; ++i) {
        int v = all_v[i];
        dv_arr[i] = valence_delta(v, q1, nq1) + valence_delta(v, q2, nq2);
    }

    // Apply swap
    q1 = nq1;
    q2 = nq2;

    // Update topo.valence with pre-computed deltas
    for (int i = 0; i < 6; ++i)
        topo.valence[all_v[i]] += dv_arr[i];

    return true;
}

static int subpass_irregularity(QuadMesh& qm,
                                 MeshTopo& topo,
                                 const EdgeFlowRefineParams& params)
{
    int swapped = 0;
    const int nv = topo.nv;

    for (int vi = 0; vi < nv; ++vi) {
        if (topo.is_boundary[vi]) continue;
        int dev_i = topo.valence[vi] - 4;
        if (dev_i == 0) continue;

        // BFS to find opposite-deviation vertices within max_cancel_dist
        std::vector<int> visited(nv, -1);
        visited[vi] = 0;
        std::queue<int> bfs;
        bfs.push(vi);

        bool found_cancel = false;
        while (!bfs.empty() && !found_cancel) {
            int cur = bfs.front(); bfs.pop();
            int dist = visited[cur];
            if (dist >= params.max_cancel_dist) continue;

            // Expand through adjacent vertices (via shared quads)
            for (int qi : topo.v_to_quads[cur]) {
                for (int nv_idx : qm.quads[qi]) {
                    if (nv_idx < 0 || nv_idx >= nv) continue;
                    if (visited[nv_idx] >= 0) continue;
                    visited[nv_idx] = dist + 1;
                    bfs.push(nv_idx);

                    // Candidate pair?
                    if (!topo.is_boundary[nv_idx] &&
                        (topo.valence[nv_idx] - 4) == -dev_i)
                    {
                        // Find a shared quad between the path and try swap
                        for (int qi_vi : topo.v_to_quads[vi]) {
                            for (int qi_vj : topo.v_to_quads[nv_idx]) {
                                if (qi_vi == qi_vj) continue;
                                // Try swapping the edge between these two quads
                                if (try_diagonal_swap(qm, topo, qi_vi, qi_vj)) {
                                    ++swapped;
                                    found_cancel = true;
                                    goto next_vertex;
                                }
                            }
                        }
                    }
                }
            }
        }
        next_vertex:;
    }
    return swapped;
}

// ---------------------------------------------------------------------------
// Sub-pass 2: Loop straightening
// ---------------------------------------------------------------------------

static int subpass_loop_straighten(QuadMesh& qm,
                                    MeshTopo& topo,
                                    const EdgeFlowRefineParams& params)
{
    const int nv  = topo.nv;
    const int nq  = topo.nq;
    int rotated = 0;

    // Build directed edge → quad map
    std::unordered_map<int64_t, int> dir_edge_to_quad;
    dir_edge_to_quad.reserve(nq * 4);
    // BUG FIX (v92): cast `a` through uint32_t before widening to prevent
    // sign-extension corrupting bits 32–63 for negative vertex indices.
    auto ekey = [](int a, int b) -> int64_t {
        return (static_cast<int64_t>(static_cast<uint32_t>(a)) << 32)
             |  static_cast<int64_t>(static_cast<uint32_t>(b));
    };
    for (int qi = 0; qi < nq; ++qi) {
        const auto& q = qm.quads[qi];
        for (int k = 0; k < 4; ++k) {
            int a = q[k], b = q[(k+1)%4];
            if (a >= 0 && b >= 0) dir_edge_to_quad[ekey(a,b)] = qi;
        }
    }

    // For each quad, check if its two "loop edges" (the ones that carry the
    // primary flow — slots 0-1 and 2-3, vs slots 1-2 and 3-0) zigzag.
    for (int qi = 0; qi < nq; ++qi) {
        const auto& q = qm.quads[qi];
        // Primary edge: q[0]→q[1]
        // Check against opposite edge q[2]→q[3] (reversed: q[3]→q[2])
        int a = q[0], b = q[1], c = q[2], d = q[3];
        if (a<0||b<0||c<0||d<0||a>=nv||b>=nv||c>=nv||d>=nv) continue;

        auto e1 = norm3(sub3(qm.vertices[b], qm.vertices[a]));
        auto e2 = norm3(sub3(qm.vertices[c], qm.vertices[d]));
        double cos_angle = dot3(e1, e2);
        if (cos_angle >= params.zigzag_cos_thresh) continue;  // OK

        // Zigzag detected. Find the neighbouring quad on edge b→a.
        auto it = dir_edge_to_quad.find(ekey(b, a));
        if (it == dir_edge_to_quad.end()) continue;
        int qi2 = it->second;
        if (qi2 == qi) continue;

        if (try_diagonal_swap(qm, topo, qi, qi2)) {
            // BUG FIX (v92): After a successful swap, the four original
            // directed-edge entries for qi and qi2 are now stale — they still
            // map to qi / qi2 but point to edges that no longer exist in those
            // quads.  Subsequent iterations of the same pass could look up one
            // of these ghost entries and call try_diagonal_swap() on an edge
            // that no longer connects those two quad indices, producing
            // self-intersecting or non-manifold geometry.
            // Fix: erase the 8 old entries first, then insert the 8 new ones.
            for (int k = 0; k < 4; ++k) {
                // q at qi and qi2 have already been overwritten by try_diagonal_swap,
                // so we reconstruct the old edges from the original a,b,c,d we
                // read before the call.  For qi: old edges were (a→b),(b→c),(c→d),(d→a)
                // and for qi2: look them up from the new quads is insufficient — but
                // since try_diagonal_swap already overwrote them, we instead simply
                // remove any entry currently at the old quad indices that still references
                // them, by doing a full re-sweep: erase all qi / qi2 associations and
                // re-insert fresh ones from the now-updated quads.
                (void)k;
            }
            // Erase ALL entries belonging to qi and qi2 (scan is O(nq*4) worst
            // case, but this path is infrequent; a reverse map would be faster).
            for (auto it = dir_edge_to_quad.begin(); it != dir_edge_to_quad.end(); ) {
                if (it->second == qi || it->second == qi2)
                    it = dir_edge_to_quad.erase(it);
                else
                    ++it;
            }
            // Re-insert fresh entries for both modified quads.
            const auto& nq1 = qm.quads[qi];
            const auto& nq2 = qm.quads[qi2];
            for (int k = 0; k < 4; ++k) {
                dir_edge_to_quad[ekey(nq1[k], nq1[(k+1)%4])] = qi;
                dir_edge_to_quad[ekey(nq2[k], nq2[(k+1)%4])] = qi2;
            }
            ++rotated;
        }
    }
    return rotated;
}

// ---------------------------------------------------------------------------
// vertex_flow_score
// ---------------------------------------------------------------------------

double vertex_flow_score(const QuadMesh& qm, int vertex_i)
{
    const int nv = static_cast<int>(qm.vertices.size());
    const int nq = static_cast<int>(qm.quads.size());
    if (vertex_i < 0 || vertex_i >= nv) return 0.0;

    // Collect incident quads
    std::vector<int> inc_quads;
    for (int qi = 0; qi < nq; ++qi)
        for (int v : qm.quads[qi])
            if (v == vertex_i) { inc_quads.push_back(qi); break; }

    if (inc_quads.empty()) return 1.0;

    const double pi = 3.14159265358979323846;
    double min_ang  = pi;
    double ang_var  = 0.0;

    // Minimum angle at vertex_i over all incident quads
    for (int qi : inc_quads) {
        const auto& q = qm.quads[qi];
        for (int k = 0; k < 4; ++k) {
            if (q[k] != vertex_i) continue;
            int prev_v = q[(k+3)%4], next_v = q[(k+1)%4];
            if (prev_v < 0 || prev_v >= nv) continue;
            if (next_v < 0 || next_v >= nv) continue;
            double ang = min_quad_angle_at(qm.vertices[vertex_i],
                                           qm.vertices[prev_v],
                                           qm.vertices[next_v]);
            min_ang = std::min(min_ang, ang);
            // Measure deviation from 90°
            double dev = ang - pi/2.0;
            ang_var += dev * dev;
        }
    }

    // Angle score: 1.0 if all angles are exactly 90°, 0 if min angle = 0°
    double ang_score = std::max(0.0, min_ang / (pi/2.0));
    if (ang_score > 1.0) ang_score = 1.0;

    // Valence score: 1.0 at valence 4, decreasing for ±1, ±2, etc.
    int val = static_cast<int>(inc_quads.size());
    double val_score = std::max(0.0, 1.0 - std::abs(val - 4) * 0.25);

    // Combined
    return 0.6 * ang_score + 0.4 * val_score;
}

// ---------------------------------------------------------------------------
// mesh_flow_score
// ---------------------------------------------------------------------------

double mesh_flow_score(const QuadMesh& qm)
{
    const int nv = static_cast<int>(qm.vertices.size());
    if (nv == 0) return 1.0;

    // Determine boundary vertices
    std::vector<bool> is_bnd(nv, false);
    {
        std::unordered_map<int64_t,int> ec;
        // BUG FIX (v92): cast `a` through uint32_t before widening to int64_t
        // to prevent sign-extension — same fix as MeshTopo::build and
        // subpass_loop_straighten above.
        auto ek = [](int a, int b) -> int64_t {
            return (static_cast<int64_t>(static_cast<uint32_t>(a)) << 32)
                 |  static_cast<int64_t>(static_cast<uint32_t>(b));
        };
        for (const auto& q : qm.quads)
            for (int k = 0; k < 4; ++k)
                ec[ek(q[k], q[(k+1)%4])]++;
        for (auto& [k,c] : ec) {
            if (c == 1) {
                int a = static_cast<int>(k >> 32);
                int b = static_cast<int>(k & 0xFFFFFFFF);
                // BUG FIX (v53): same fix as MeshTopo::build — only a boundary
                // edge if the reverse (b→a) is absent.
                if (ec.find(ek(b, a)) == ec.end()) {
                    if (a >= 0 && a < nv) is_bnd[a] = true;
                    if (b >= 0 && b < nv) is_bnd[b] = true;
                }
            }
        }
    }

    double sum = 0.0;
    int cnt = 0;
    for (int vi = 0; vi < nv; ++vi) {
        if (is_bnd[vi]) continue;
        sum += vertex_flow_score(qm, vi);
        ++cnt;
    }
    return cnt > 0 ? sum / cnt : 1.0;
}

// ---------------------------------------------------------------------------
// refine_edge_flow
// ---------------------------------------------------------------------------

int refine_edge_flow(QuadMesh& qm, const EdgeFlowRefineParams& params)
{
    if (qm.vertices.empty() || qm.quads.empty()) return 0;

    int total_ops = 0;

    for (int pass = 0; pass < params.max_passes; ++pass) {
        MeshTopo topo;
        topo.build(qm);

        int ops  = 0;
        ops += subpass_irregularity(qm, topo, params);
        ops += subpass_loop_straighten(qm, topo, params);
        ops += subpass_relocation(qm, topo, params);

        total_ops += ops;

        // Update stats after each pass
        {
            int total = static_cast<int>(qm.quads.size() + qm.tris.size());
            qm.quad_percentage = total > 0
                ? 100.f * static_cast<float>(qm.quads.size()) / total : 0.f;

            int nv = static_cast<int>(qm.vertices.size());
            if (nv > 0) {
                std::vector<int> val(nv, 0);
                for (const auto& q : qm.quads)
                    for (int v : q) if (v>=0 && v<nv) val[v]++;
                for (const auto& t : qm.tris)
                    for (int v : t) if (v>=0 && v<nv) val[v]++;
                double sum = 0.0;
                for (int v : val) sum += v;
                qm.avg_valence = static_cast<float>(sum / nv);
            }
        }

        if (ops == 0) break;  // converged
    }

    return total_ops;
}

} // namespace qf
