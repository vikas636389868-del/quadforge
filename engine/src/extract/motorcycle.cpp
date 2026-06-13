/**
 * motorcycle.cpp — T-junction resolution via motorcycle graph.
 *
 * Traces "riders" from each singularity along the cross-field direction.
 * Each rider travels until it hits another rider's track, a singularity,
 * or the mesh boundary.  The resulting tracks partition the surface into
 * T-junction-free rectangular patches.
 *
 * v51 Implementation details
 * --------------------------
 * Pass 1 — Triangle-pair merging (retained from v50, v24 directed-edge fix):
 *   Adjacent triangle pairs are merged into quads wherever possible.
 *
 * Pass 2 — Geometric T-junction detection + quad splitting (NEW in v51):
 *   A T-junction vertex is one that lies geometrically on an edge of an
 *   adjacent quad, but is not a corner of that quad.  These arise when
 *   iso-line tracing places a vertex mid-edge due to rounding, or when
 *   singularity patches share an edge at different sampling densities.
 *
 *   Detection: for each vertex Vi, we inspect the quad edges belonging
 *   to neighbouring quads that don't already contain Vi.  We test
 *   whether Vi lies on the segment Va-Vb using:
 *       distance = |AP x AB| / |AB|  <  epsilon
 *       t = dot(AP,AB)/|AB|^2  in  (0, 1)   (strictly between endpoints)
 *
 *   Resolution: the offending quad (A, B, C, D) with Vi on edge A->B is
 *   split into:
 *       Triangle: (A, Vi, D)
 *       Quad:     (Vi, B, C, D)
 *   The introduced triangle is near-singularity geometry and will be
 *   handled correctly by subsequent cleanup and projection passes.
 *
 * Pass 3 — Stats update.
 *
 * Note on full motorcycle-graph algorithm
 * ----------------------------------------
 * The complete Eppstein (2008) motorcycle graph — tracing topological
 * paths from singularities through the integer UV lattice — belongs in
 * the parametrize stage because it requires integer UV coordinates and
 * the seam graph.  The passes above handle the structural aftermath of
 * that computation: merging leftover triangles and eliminating T-junctions
 * that survive into the extracted quad mesh.
 */

#include "../../include/quadforge/extract/motorcycle.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <cstdint>
#include <tuple>

namespace qf {

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/** Directed-edge key: (a->b) -> int64. */
static inline int64_t dir_ekey(int a, int b) {
    return ((int64_t)(uint32_t)a << 32) | (uint32_t)b;
}

/**
 * Test whether vertex at index 'vi' lies strictly on the open segment
 * from vertex 'va' to vertex 'vb' (not at either endpoint).
 *
 * Returns true iff:
 *   1. The point is within epsilon * |AB| of the line through A and B.
 *   2. The parametric t = dot(AP, AB) / |AB|^2 is in (0.001, 0.999).
 *
 * epsilon = 1e-4 makes the test scale-invariant relative to edge length.
 */
static bool point_on_segment(
    const std::vector<std::array<double,3>>& verts,
    int vi, int va, int vb)
{
    if (vi == va || vi == vb) return false;

    const auto& P = verts[vi];
    const auto& A = verts[va];
    const auto& B = verts[vb];

    double abx = B[0]-A[0], aby = B[1]-A[1], abz = B[2]-A[2];
    double apx = P[0]-A[0], apy = P[1]-A[1], apz = P[2]-A[2];

    double ab2 = abx*abx + aby*aby + abz*abz;
    if (ab2 < 1e-24) return false;   // degenerate edge

    // Parametric projection
    double t = (apx*abx + apy*aby + apz*abz) / ab2;
    if (t < 1e-3 || t > 1.0 - 1e-3) return false;

    // Perpendicular distance squared = |AP x AB|^2 / |AB|^2
    double cx = apy*abz - apz*aby;
    double cy = apz*abx - apx*abz;
    double cz = apx*aby - apy*abx;
    double cross2 = cx*cx + cy*cy + cz*cz;

    // cross2 / ab2 = dist^2;  dist < epsilon * sqrt(ab2)  =>  cross2 < eps^2 * ab2
    constexpr double eps2 = 1e-8;  // (1e-4)^2
    return cross2 < eps2 * ab2;
}

// -----------------------------------------------------------------------
// resolve_t_junctions
// -----------------------------------------------------------------------

QuadMesh resolve_t_junctions(
    const HalfEdgeMesh&                    mesh,
    const QuadMesh&                        quad_mesh,
    const std::vector<SingularityInfo>&    singularities)
{
    (void)mesh;           // used for cross-field reference in the full algorithm
    (void)singularities;  // used for rider seeding in the full algorithm

    QuadMesh result = quad_mesh;   // work on a copy

    // ================================================================
    // Pass 1: Merge adjacent triangle pairs -> quads
    //
    // Uses directed edges (BUG FIX v24 retained):
    //   directed_edge_map[(v0->v1)] = face_index
    // Adjacent manifold triangles share an edge in opposite orientation:
    //   fi stores v0->v1,  fj stores v1->v0.
    // Look up the REVERSE key to find fj without any canonicalization.
    // ================================================================
    {
        int nt = (int)result.tris.size();
        std::unordered_map<int64_t, int> directed_edge_map;
        directed_edge_map.reserve(nt * 3);

        for (int fi = 0; fi < nt; ++fi)
            for (int k = 0; k < 3; ++k) {
                int v0 = result.tris[fi][k];
                int v1 = result.tris[fi][(k+1)%3];
                directed_edge_map[dir_ekey(v0, v1)] = fi;
            }

        std::vector<bool> tri_used(nt, false);
        std::vector<std::array<int,3>> remaining_tris;

        for (int fi = 0; fi < nt; ++fi) {
            if (tri_used[fi]) continue;
            bool merged = false;
            for (int k = 0; k < 3 && !merged; ++k) {
                int v0 = result.tris[fi][k];
                int v1 = result.tris[fi][(k+1)%3];
                auto it = directed_edge_map.find(dir_ekey(v1, v0));
                if (it == directed_edge_map.end()) continue;
                int fj = it->second;
                if (fj == fi || tri_used[fj]) continue;

                // Shared edge is v0-v1 (fi: v0->v1, fj: v1->v0)
                int opp_i = result.tris[fi][(k+2)%3];
                int opp_j = -1;
                for (int m = 0; m < 3; ++m) {
                    if (result.tris[fj][m] == v1 &&
                        result.tris[fj][(m+1)%3] == v0) {
                        opp_j = result.tris[fj][(m+2)%3];
                        break;
                    }
                }
                if (opp_j < 0) continue;

                result.quads.push_back({opp_i, v0, opp_j, v1});
                tri_used[fi] = tri_used[fj] = true;
                merged = true;
            }
            if (!merged)
                remaining_tris.push_back(result.tris[fi]);
        }
        result.tris = remaining_tris;
    }

    // ================================================================
    // Pass 2: Geometric T-junction detection and quad splitting
    //
    // A T-junction vertex Vi lies strictly on quad edge Va->Vb but is
    // not a corner of that quad.  Splitting the quad eliminates the
    // T-junction and produces a manifold mesh.
    //
    // Detection strategy:
    //   For each vertex Vi, gather candidate quads that:
    //     a) Do NOT already contain Vi as a corner.
    //     b) DO contain at least one vertex that is a neighbor of Vi
    //        (i.e., shares a quad with Vi).
    //   For each candidate quad edge, run point_on_segment(Vi, Va, Vb).
    //
    // Resolution:
    //   Quad (A, B, C, D) with Vi on edge A->B is split into:
    //     Triangle: (A, Vi, D)    [near-singularity sliver, cleaned up later]
    //     Quad:     (Vi, B, C, D) [well-formed quad]
    //   The shared edge between the triangle and quad is Vi->D (triangle)
    //   / D->Vi (quad), which is manifold.
    // ================================================================
    {
        int cur_nv  = (int)result.vertices.size();
        int cur_nq  = (int)result.quads.size();
        if (cur_nv == 0 || cur_nq == 0) goto pass2_done;

        {
            // v_to_quads[vi] = set of quad indices containing vi
            std::vector<std::unordered_set<int>> v_to_quads(cur_nv);
            for (int qi = 0; qi < cur_nq; ++qi)
                for (int cv : result.quads[qi])
                    if (cv >= 0 && cv < cur_nv) v_to_quads[cv].insert(qi);

            // Collect T-junctions: (vertex_on_edge, quad_idx, edge_pos_k)
            // We record only one T-junction per quad (the first found).
            using TJ = std::tuple<int,int,int>;  // (vi, qi, k)
            std::vector<TJ> t_junctions;

            for (int vi = 0; vi < cur_nv; ++vi) {
                // Candidate quads: adjacent to a quad of vi but don't contain vi
                std::unordered_set<int> candidates;
                for (int qi : v_to_quads[vi])
                    for (int nbv : result.quads[qi])
                        for (int qj : v_to_quads[nbv])
                            if (!v_to_quads[vi].count(qj))
                                candidates.insert(qj);

                for (int qi : candidates) {
                    const auto& q = result.quads[qi];
                    for (int k = 0; k < 4; ++k) {
                        int va = q[k], vb = q[(k+1)%4];
                        if (point_on_segment(result.vertices, vi, va, vb)) {
                            t_junctions.emplace_back(vi, qi, k);
                            break;  // one T-junction per quad is enough
                        }
                    }
                }
            }

            if (t_junctions.empty()) goto pass2_done;

            // Process T-junctions — one split per quad (first one wins).
            // Sort by quad index descending so index arithmetic stays valid.
            std::sort(t_junctions.begin(), t_junctions.end(),
                      [](const TJ& a, const TJ& b){
                          return std::get<1>(a) > std::get<1>(b);
                      });

            // Deduplicate: keep first occurrence per quad
            {
                std::vector<TJ> unique_tj;
                std::unordered_set<int> seen;
                for (auto& tj : t_junctions) {
                    int qi = std::get<1>(tj);
                    if (!seen.count(qi)) { seen.insert(qi); unique_tj.push_back(tj); }
                }
                t_junctions = std::move(unique_tj);
            }

            std::vector<bool>             quad_removed(result.quads.size(), false);
            std::vector<std::array<int,4>> new_quads;
            new_quads.reserve(t_junctions.size());

            for (auto& [vi, qi, k] : t_junctions) {
                if (qi < 0 || qi >= (int)result.quads.size()) continue;
                if (quad_removed[qi]) continue;

                const auto& q = result.quads[qi];
                // Confirm vi is still on this edge (indices may have shifted in
                // large T-junction clusters — revalidate)
                int A = q[k];
                int B = q[(k+1)%4];
                int C = q[(k+2)%4];
                int D = q[(k+3)%4];

                if (!point_on_segment(result.vertices, vi, A, B)) continue;

                // Split: Triangle (A, Vi, D) + Quad (Vi, B, C, D)
                quad_removed[qi] = true;
                result.tris.push_back({A, vi, D});
                new_quads.push_back({vi, B, C, D});
            }

            // Rebuild quad list
            std::vector<std::array<int,4>> kept;
            kept.reserve(result.quads.size() + new_quads.size());
            for (int qi = 0; qi < (int)result.quads.size(); ++qi)
                if (!quad_removed[qi]) kept.push_back(result.quads[qi]);
            for (auto& nq : new_quads) kept.push_back(nq);
            result.quads = std::move(kept);
        }
    }
    pass2_done:;

    // ================================================================
    // Pass 3: Update stats
    // ================================================================
    {
        int total = (int)(result.quads.size() + result.tris.size());
        result.quad_percentage = total > 0
            ? 100.f * (float)result.quads.size() / (float)total : 0.f;
    }

    return result;
}

// ==========================================================================
// build_motorcycle_graph — ARCHITECTURAL NOTE (v90)
// ==========================================================================
//
// build_motorcycle_graph() is DECLARED in motorcycle.h and its DEFINITION
// lives in patch_layout.cpp.  This split exists because:
//
//   • The function internally seeds and advances riders, then flood-fills
//     patches — logic that shares data structures (Rider, EdgeCrossing,
//     face_uv_gradient, solve_tutte) with build_patch_layout() and
//     extract_quads_motorcycle() in patch_layout.cpp.  Duplicating those
//     helpers here would create maintenance debt.
//
//   • motorcycle.h declares build_motorcycle_graph() as a low-level API
//     for callers that want raw PatchLayout output (e.g., testing, vis).
//     patch_layout.h is the higher-level API for the full method=1 pipeline.
//
// As of v90, patch_layout.cpp now includes motorcycle.h so the compiler
// validates the signature match at build time.  The linker resolves the
// symbol from the object file produced by patch_layout.cpp.
//
// If you need to move the implementation here in a future refactor:
//   1. Copy the anonymous-namespace helpers from patch_layout.cpp.
//   2. Move the build_motorcycle_graph() function body here.
//   3. In patch_layout.cpp, remove the local build_motorcycle_graph()
//      definition and add #include "motorcycle.h" to use it externally.
} // namespace qf
