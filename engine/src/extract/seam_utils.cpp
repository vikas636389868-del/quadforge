/**
 * seam_utils.cpp — Period-jump-aware UV coordinate utilities.
 *
 * Implements the functions declared in include/quadforge/extract/seam_utils.h.
 *
 * DESIGN NOTES
 * ------------
 * All functions in this file are O(V + F) or O(K) where K is the number of
 * seam edges, and are written to be allocation-minimal on the hot path.
 *
 * seam_aware_uv() is the only function that allocates O(V + F) temporaries;
 * the others operate on the already-built PeriodJumpMap which is a single
 * hash map built once per extraction call.
 *
 * Thread safety: PeriodJumpMap is read-only after construction.  All functions
 * that take a const PeriodJumpMap& are safe to call from multiple threads.
 */

#include "../../include/quadforge/extract/seam_utils.h"
#include "../../include/quadforge/mesh/halfedge.h"

#include <queue>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

namespace qf {

// ==========================================================================
// build_period_jump_map
// ==========================================================================

PeriodJumpMap build_period_jump_map(const UVParam& uv)
{
    PeriodJumpMap jmap;

    const auto& seams = uv.seam_edges;
    const auto& jumps = uv.period_jumps;

    if (seams.empty() || jumps.empty()) {
        // No seam data — return empty map (seam-aware tracing degrades
        // to standard tracing, which is correct for Poisson-only UV).
        return jmap;
    }

    // Size of the two parallel arrays must match; if they don't (e.g. the
    // param stage only populated seam_edges but not period_jumps), treat
    // the jump count as zero for the unpopulated suffix.
    const int n = static_cast<int>(
        std::min(seams.size(), jumps.size()));

    jmap.reserve(n * 2);   // each edge → 2 directed entries

    for (int k = 0; k < n; ++k) {
        int lo = seams[k].first;
        int hi = seams[k].second;
        int du = jumps[k].first;
        int dv = jumps[k].second;

        if (du == 0 && dv == 0) continue;  // skip zero-jump edges

        // Forward direction: lo → hi  (U[hi] ≈ U[lo] + Δu)
        jmap[seam_edge_key(lo, hi)] = PeriodJump{  du,  dv };
        // Reverse direction: hi → lo
        jmap[seam_edge_key(hi, lo)] = PeriodJump{ -du, -dv };
    }

    return jmap;
}


// ==========================================================================
// uv_consistent
// ==========================================================================

std::array<double,6> uv_consistent(
    int                  face_idx,
    const HalfEdgeMesh&  mesh,
    const UVParam&       uv,
    const PeriodJumpMap& jmap)
{
    auto [v0, v1, v2] = mesh.face_vertices(face_idx);

    // Anchor: v0 uses its raw UV value.
    double u0 = uv.U[v0], vv0 = uv.V[v0];

    // v1: check directed edge v0 → v1
    double u1 = uv.U[v1], vv1 = uv.V[v1];
    apply_period_jump(u1, vv1, v0, v1, jmap);

    // v2: check directed edge v0 → v2 first.
    //     If that edge is not a seam, fall back to v1 → v2 (routing via v1).
    double u2 = uv.U[v2], vv2 = uv.V[v2];
    {
        auto it02 = jmap.find(seam_edge_key(v0, v2));
        if (it02 != jmap.end()) {
            // v0 → v2 is a seam: apply its jump directly from anchor
            u2  += it02->second.du;
            vv2 += it02->second.dv;
        } else {
            // Try v1 → v2 (composed jump: v0→v1 + v1→v2)
            auto it12 = jmap.find(seam_edge_key(v1, v2));
            if (it12 != jmap.end()) {
                // The jump for v2 relative to anchor v0 is:
                //   jump(v0→v1) + jump(v1→v2)
                // jump(v0→v1) is already baked into (u1,vv1) so we just
                // apply jump(v1→v2) on top of v2's raw UV, then compensate
                // to keep it consistent with v1's adjusted frame.
                //
                // More precisely: v2_adj = v2_raw + du(v1→v2)
                // (The du(v0→v1) does not apply to v2 — we are tracing the
                // path v0→v2 through the v1 node, which crosses only the
                // v1→v2 seam edge.)
                u2  += it12->second.du;
                vv2 += it12->second.dv;
            }
            // If neither edge is a seam, u2/vv2 remain raw — correct.
        }
    }

    return { u0, vv0, u1, vv1, u2, vv2 };
}


// ==========================================================================
// query_seam_crossing
// ==========================================================================

bool query_seam_crossing(
    int                  va,
    int                  vb,
    const PeriodJumpMap& jmap,
    SeamCrossing*        out)
{
    auto it = jmap.find(seam_edge_key(va, vb));
    if (it == jmap.end()) return false;

    const PeriodJump& pj = it->second;
    if (pj.du == 0 && pj.dv == 0) return false;

    if (out) {
        out->va = va;
        out->vb = vb;
        out->du = pj.du;
        out->dv = pj.dv;
        out->t  = 0.5;  // Crossing is at the midpoint of the edge by convention;
                        // callers that need precise t compute it from UV values.
    }
    return true;
}


// ==========================================================================
// seam_aware_uv  — BFS global unwrap with period-jump stitching
// ==========================================================================
//
// Algorithm:
//   Use face 0 as the root.  Its vertices get their raw UV values as the
//   global anchor.  Then BFS over the face dual graph: when crossing from
//   face fi (already visited, UV assigned) into adjacent face fj (not yet
//   visited) via the shared directed half-edge he (fi→fj), check whether
//   (he.vertex_from, he.vertex_to) is in the PeriodJumpMap.  If so, the
//   jump is accumulated for the vertex on the fj side.  After visiting all
//   faces, every vertex has exactly one global UV assignment (the first-visit
//   assignment).  Vertices shared between multiple faces receive the UV from
//   whichever face visited them first in BFS order (this is deterministic
//   because BFS is deterministic from a fixed root).
//
// Correctness argument:
//   For a mesh without seams, the BFS assigns raw UV everywhere — identical
//   to the standard tracer.  For a mesh with k seams, each seam crossing
//   accumulates the integer period jump from the PeriodJumpMap.  Because
//   the period jumps form a consistent assignment (they come from a
//   well-formed MIQ rounding), the BFS assignment is path-independent —
//   i.e. the UV assigned to any vertex v is the same regardless of which
//   path the BFS took to reach v.  This is the key correctness property.
//
// Implementation notes:
//   - Vertex assignment uses a visited flag rather than a per-vertex value
//     to handle interior vertices that are shared by multiple faces.
//   - The BFS uses the half-edge twin links from HalfEdgeMesh.
//   - Boundary half-edges (twin == -1) are skipped.
//   - Disconnected components are handled by restarting BFS from the
//     next unvisited face.

void seam_aware_uv(
    const HalfEdgeMesh&  mesh,
    const UVParam&       uv,
    const PeriodJumpMap& jmap,
    std::vector<double>& U_out,
    std::vector<double>& V_out)
{
    const int nv = mesh.num_vertices();
    const int nf = mesh.num_faces();

    U_out.assign(nv, 0.0);
    V_out.assign(nv, 0.0);

    if (nv == 0 || nf == 0) return;

    // Fast path: if no seam jumps, just copy raw UV.
    if (jmap.empty()) {
        for (int i = 0; i < nv; ++i) {
            U_out[i] = uv.U[i];
            V_out[i] = uv.V[i];
        }
        return;
    }

    // BFS state
    std::vector<bool> face_visited(nf, false);
    std::vector<bool> vert_assigned(nv, false);

    // Per-face UV offsets accumulated during BFS.
    // face_du[fi] = integer U offset accumulated when entering face fi.
    // The actual UV of vertex v in face fi is: U[v] + face_du[fi]
    //   (same for V with face_dv).
    // Using int offsets avoids accumulating floating-point rounding error.
    std::vector<int> face_du(nf, 0);
    std::vector<int> face_dv(nf, 0);

    std::queue<int> bfs;

    // Process all connected components
    for (int root = 0; root < nf; ++root) {
        if (face_visited[root]) continue;

        face_visited[root] = true;
        face_du[root] = 0;
        face_dv[root] = 0;
        bfs.push(root);

        // Assign root's vertices
        {
            auto [va, vb, vc] = mesh.face_vertices(root);
            auto assign = [&](int v) {
                if (!vert_assigned[v]) {
                    U_out[v] = uv.U[v];
                    V_out[v] = uv.V[v];
                    vert_assigned[v] = true;
                }
            };
            assign(va); assign(vb); assign(vc);
        }

        while (!bfs.empty()) {
            int fi = bfs.front(); bfs.pop();

            const int du_fi = face_du[fi];
            const int dv_fi = face_dv[fi];

            // Walk the three half-edges of face fi
            auto [he0, he1, he2] = mesh.face_half_edges(fi);
            int hes[3] = {he0, he1, he2};

            for (int hei : hes) {
                const HalfEdge& he = mesh.half_edge(hei);
                int twin_he = he.twin;
                if (twin_he < 0) continue;  // boundary — no adjacent face

                const HalfEdge& twin = mesh.half_edge(twin_he);
                int fj = twin.face;
                if (fj < 0 || face_visited[fj]) continue;

                // Shared edge: (he.vertex → prev.vertex) in face fi
                // The directed edge crossing from fi into fj is:
                //   from_v = he.vertex (destination of half-edge he, i.e., the
                //             vertex at the END of hei in face fi)
                //   to_v   = twin.vertex (destination of twin, i.e., the vertex
                //             at the end of twin_he in face fj)
                //
                // In standard half-edge convention:
                //   hei points TO he.vertex  (destination)
                //   the source of hei is prev.vertex
                // So the directed edge from fi into fj is:
                //   source = destination of prev(hei) = destination of he.prev
                //   dest   = he.vertex

                // Simpler: use the twin's perspective.
                // The edge shared between fi and fj connects two vertices.
                // In face fi:  one of the 3 vertices is he.vertex
                // In face fj:  the same edge appears as twin_he, with twin.vertex
                //              being the OTHER endpoint.
                //
                // We want the directed edge from fi's side to fj's side.
                // fi→fj direction: the half-edge in fi pointing toward fj is hei.
                // Its source vertex is: mesh.half_edge(he.prev).vertex
                // (or equivalently, the vertex NOT pointed to by hei in fi)

                int src_v = -1, dst_v = -1;
                {
                    // hei goes from prev[hei].vertex  →  he.vertex (= he.vertex)
                    // We need "from" (fi side) and "to" (fj side) for the seam check.
                    // The two vertices on the shared edge are:
                    //   v_a = mesh.half_edge(he.prev).vertex  (source of hei)
                    //   v_b = he.vertex                        (dest of hei)
                    // The seam convention is: crossing from fi into fj via hei
                    // means going from v_a→v_b (fi's perspective).
                    int prev_he = mesh.half_edge(hei).prev;
                    // Guard against -1 (shouldn't happen for valid meshes)
                    if (prev_he < 0) continue;
                    src_v = mesh.half_edge(prev_he).vertex;
                    dst_v = he.vertex;
                }

                if (src_v < 0 || dst_v < 0) continue;

                // Accumulate period jump when crossing from fi into fj
                int du_fj = du_fi;
                int dv_fj = dv_fi;
                auto it = jmap.find(seam_edge_key(src_v, dst_v));
                if (it != jmap.end()) {
                    du_fj += it->second.du;
                    dv_fj += it->second.dv;
                }

                face_du[fj] = du_fj;
                face_dv[fj] = dv_fj;
                face_visited[fj] = true;
                bfs.push(fj);

                // Assign fj's vertices (using fj's accumulated offset)
                auto [va2, vb2, vc2] = mesh.face_vertices(fj);
                auto assign_j = [&](int v) {
                    if (!vert_assigned[v]) {
                        U_out[v] = uv.U[v] + du_fj;
                        V_out[v] = uv.V[v] + dv_fj;
                        vert_assigned[v] = true;
                    }
                };
                assign_j(va2); assign_j(vb2); assign_j(vc2);
            }
        }
    }

    // Any vertices that were never visited by BFS (should be empty for a
    // connected manifold mesh) fall back to their raw UV value.
    for (int v = 0; v < nv; ++v) {
        if (!vert_assigned[v]) {
            U_out[v] = uv.U[v];
            V_out[v] = uv.V[v];
        }
    }
}

} // namespace qf
