/**
 * subdiv.cpp — Catmull-Clark compatibility and tri→quad merging.
 *
 * Bug-fix history:
 *   1–20: (see v78–v81 detailed comments in prior versions)
 *  21. (v94) PRECISION FIX — float→double across all three public functions.
 *      subdiv.h was updated in v94 to change all position parameters from
 *      float to double (matching Scalar=double / QuadMesh::vertices = double).
 *      This .cpp was not updated, producing three linker errors:
 *
 *        check_subdiv_compatibility(const float* positions, ...)
 *          → const double*
 *        convert_tris_to_quads(..., const std::vector<float>&)
 *          → const std::vector<double>&
 *        catmull_clark_subdivide(const std::vector<float>&, ...,
 *                                std::vector<float>&, ...)
 *          → const std::vector<double>& / std::vector<double>&
 *
 *      All internal arithmetic, accumulator arrays, literals (0.f→0.0,
 *      0.25f→0.25, etc.), and casts ((float)→(double)) have been updated.
 *      All prior algorithmic correctness fixes (#1–#20) are preserved.
 */

#include "../../include/quadforge/postprocess/subdiv.h"

#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>

namespace qf {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

inline int64_t edge_key(int32_t a, int32_t b) {
    if (a > b) { int32_t t = a; a = b; b = t; }
    return (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// check_subdiv_compatibility
//
// (v94) positions parameter type: const float* → const double*
// The pointer is accepted but not dereferenced (topology-only algorithm).
// ---------------------------------------------------------------------------
SubdivCompatibilityReport check_subdiv_compatibility(
    const double*  /*positions*/,
    const int32_t* quad_faces, int32_t nq,
    const int32_t* tri_faces,  int32_t nt,
    int32_t nv)
{
    SubdivCompatibilityReport r;
    r.valence_histogram.assign(9, 0);
    r.num_irregular_interior = 0;
    r.num_irregular_boundary = 0;
    r.irregular_ratio = 0.f;
    r.subdiv_ready = false;

    if (nv <= 0) return r;

    // (v80) null-pointer guards
    if (quad_faces == nullptr) nq = 0;
    if (tri_faces  == nullptr) nt = 0;

    // (v81) assign r.all_quads AFTER null-pointer guards
    r.all_quads = (nt == 0);

    // 1. Valence table from quad + tri faces
    std::vector<int32_t> valence(static_cast<size_t>(nv), 0);
    for (int32_t fi = 0; fi < nq; ++fi)
        for (int k = 0; k < 4; ++k) {
            int32_t v = quad_faces[fi * 4 + k];
            if (v >= 0 && v < nv) valence[v]++;
        }
    for (int32_t fi = 0; fi < nt; ++fi)
        for (int k = 0; k < 3; ++k) {
            int32_t v = tri_faces[fi * 3 + k];
            if (v >= 0 && v < nv) valence[v]++;
        }

    // 2. Boundary detection via directed edge half-edge set
    std::unordered_map<int64_t, int8_t> dir_edge_seen;
    dir_edge_seen.reserve(static_cast<size_t>(nq) * 4 + static_cast<size_t>(nt) * 3);

    auto add_dir_edge = [&](int32_t a, int32_t b) {
        if (a < 0 || b < 0 || a >= nv || b >= nv) return;
        int64_t k = (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
        dir_edge_seen[k] = 1;
    };
    for (int32_t fi = 0; fi < nq; ++fi)
        for (int k = 0; k < 4; ++k)
            add_dir_edge(quad_faces[fi*4+k], quad_faces[fi*4+(k+1)%4]);
    for (int32_t fi = 0; fi < nt; ++fi)
        for (int k = 0; k < 3; ++k)
            add_dir_edge(tri_faces[fi*3+k], tri_faces[fi*3+(k+1)%3]);

    std::vector<bool> is_boundary(static_cast<size_t>(nv), false);
    for (auto& [k64, _] : dir_edge_seen) {
        int32_t a = static_cast<int32_t>(k64 >> 32);
        int32_t b = static_cast<int32_t>(k64 & 0xFFFFFFFFLL);
        int64_t rev = (static_cast<int64_t>(b) << 32) | static_cast<uint32_t>(a);
        if (dir_edge_seen.find(rev) == dir_edge_seen.end()) {
            if (a >= 0 && a < nv) is_boundary[a] = true;
            if (b >= 0 && b < nv) is_boundary[b] = true;
        }
    }

    // 3. Classify vertices
    int32_t num_interior = 0;
    for (int32_t vi = 0; vi < nv; ++vi) {
        int32_t val = valence[vi];
        r.valence_histogram[std::min(val, (int32_t)8)]++;
        if (is_boundary[vi]) {
            if (val > 2) r.num_irregular_boundary++;  // (v78) was (val != 2)
        } else {
            num_interior++;
            if (val != 4) r.num_irregular_interior++;
        }
    }

    r.irregular_ratio = (num_interior > 0)
        ? static_cast<float>(r.num_irregular_interior) / static_cast<float>(num_interior)
        : 0.f;
    r.subdiv_ready = r.all_quads && (r.irregular_ratio < 0.05f);
    return r;
}

// ---------------------------------------------------------------------------
// convert_tris_to_quads
//
// (v94) positions type: const std::vector<float>& → const std::vector<double>&
// The vector is accepted but not dereferenced (topology-only algorithm).
// ---------------------------------------------------------------------------
void convert_tris_to_quads(
    std::vector<int32_t>& quad_faces,
    std::vector<int32_t>& tri_faces,
    const std::vector<double>& /*positions*/)
{
    if (tri_faces.size() < 6) return;

    int32_t nt = static_cast<int32_t>(tri_faces.size() / 3);
    std::unordered_map<int64_t, int32_t> edge_to_tri;
    edge_to_tri.reserve(static_cast<size_t>(nt) * 3);

    for (int32_t fi = 0; fi < nt; ++fi)
        for (int k = 0; k < 3; ++k) {
            int32_t v0 = tri_faces[fi*3+k], v1 = tri_faces[fi*3+(k+1)%3];
            edge_to_tri[edge_key(v0, v1)] = fi;
        }

    std::vector<bool>    used(static_cast<size_t>(nt), false);
    std::vector<int32_t> new_tris;

    for (int32_t fi = 0; fi < nt; ++fi) {
        if (used[fi]) continue;
        bool merged = false;

        for (int k = 0; k < 3 && !merged; ++k) {
            int32_t v0 = tri_faces[fi*3+k];
            int32_t v1 = tri_faces[fi*3+(k+1)%3];

            auto it = edge_to_tri.find(edge_key(v1, v0));
            if (it == edge_to_tri.end()) continue;
            int32_t fj = it->second;
            if (fj == fi || used[fj]) continue;

            int opp_k = -1;
            for (int m = 0; m < 3; ++m) {
                if (tri_faces[fj*3+m]       == v1 &&
                    tri_faces[fj*3+(m+1)%3] == v0) {
                    opp_k = m; break;
                }
            }
            if (opp_k < 0) continue;

            int32_t prev_i = tri_faces[fi*3+(k+2)%3];
            int32_t opp    = tri_faces[fj*3+(opp_k+2)%3];

            quad_faces.push_back(prev_i);
            quad_faces.push_back(v0);
            quad_faces.push_back(opp);
            quad_faces.push_back(v1);

            used[fi] = used[fj] = true;
            merged = true;
        }

        if (!merged) {
            new_tris.push_back(tri_faces[fi*3+0]);
            new_tris.push_back(tri_faces[fi*3+1]);
            new_tris.push_back(tri_faces[fi*3+2]);
        }
    }
    tri_faces = std::move(new_tris);
}

// ---------------------------------------------------------------------------
// catmull_clark_subdivide
//
// (v94) All position vectors changed from std::vector<float> to
// std::vector<double>.  All internal accumulators, literals, and casts
// updated accordingly.  All algorithmic correctness fixes (#4–#20) preserved.
//
// Vertex layout in out_positions:
//   [0..nv-1]         V-points   (updated original vertices)
//   [nv..nv+nf-1]     F-points   (face centroids)
//   [nv+nf..nv+nf+ne-1] E-points (edge midpoints, CC-corrected)
//
// Each input quad generates 4 child quads.
// ---------------------------------------------------------------------------
bool catmull_clark_subdivide(
    const std::vector<double>&  positions,
    const std::vector<int32_t>& quad_faces,
    std::vector<double>&        out_positions,
    std::vector<int32_t>&       out_quad_faces)
{
    int32_t nv = static_cast<int32_t>(positions.size() / 3);
    int32_t nf = static_cast<int32_t>(quad_faces.size() / 4);
    if (nf == 0 || nv == 0) return false;

    // ------------------------------------------------------------------
    // 1. F-points: centroid of each input face.
    // ------------------------------------------------------------------
    std::vector<double> fpt(static_cast<size_t>(nf) * 3, 0.0);
    for (int32_t fi = 0; fi < nf; ++fi) {
        int valid_count = 0;
        for (int k = 0; k < 4; ++k) {
            int32_t v = quad_faces[fi*4+k];
            if (v < 0 || v >= nv) continue;   // (v79) bounds guard
            fpt[fi*3+0] += positions[v*3+0];
            fpt[fi*3+1] += positions[v*3+1];
            fpt[fi*3+2] += positions[v*3+2];
            ++valid_count;
        }
        if (valid_count > 0) {
            const double inv = 1.0 / static_cast<double>(valid_count);
            fpt[fi*3+0] *= inv;
            fpt[fi*3+1] *= inv;
            fpt[fi*3+2] *= inv;
        }
    }

    // ------------------------------------------------------------------
    // 2. E-points: Catmull-Clark interior/boundary rule (v77 fix).
    //    Interior: E = (A+B+F1+F2)/4
    //    Boundary: E = (A+B)/2
    // ------------------------------------------------------------------
    struct EdgeEntry {
        int32_t idx;
        int32_t adj_faces[2];
        int32_t adj_count{0};
    };
    std::unordered_map<int64_t, EdgeEntry> edge_map;
    edge_map.reserve(static_cast<size_t>(nf) * 4);

    int32_t ne_count = 0;
    for (int32_t fi = 0; fi < nf; ++fi) {
        for (int k = 0; k < 4; ++k) {
            int32_t va = quad_faces[fi*4+k];
            int32_t vb = quad_faces[fi*4+(k+1)%4];
            if (va < 0 || va >= nv || vb < 0 || vb >= nv) continue;  // (v80)
            int64_t ek = edge_key(va, vb);
            auto it = edge_map.find(ek);
            if (it == edge_map.end()) {
                EdgeEntry e;
                e.idx = ne_count++;
                e.adj_faces[0] = fi;
                e.adj_count = 1;
                edge_map[ek] = e;
            } else {
                EdgeEntry& e = it->second;
                if (e.adj_count < 2) e.adj_faces[e.adj_count] = fi;
                e.adj_count++;
            }
        }
    }

    std::unordered_map<int64_t, int32_t> edge_index;
    edge_index.reserve(edge_map.size());
    for (const auto& [ek, entry] : edge_map)
        edge_index[ek] = entry.idx;

    std::vector<double> ept(static_cast<size_t>(ne_count) * 3, 0.0);
    for (const auto& [ek, entry] : edge_map) {
        int32_t va = static_cast<int32_t>(static_cast<uint64_t>(ek) >> 32);
        int32_t vb = static_cast<int32_t>(static_cast<uint64_t>(ek) & 0xFFFFFFFFULL);
        if (va < 0 || va >= nv || vb < 0 || vb >= nv) continue;  // (v80)

        const int32_t ei3 = entry.idx * 3;
        if (entry.adj_count == 2) {
            int32_t f1 = entry.adj_faces[0];
            int32_t f2 = entry.adj_faces[1];
            for (int c = 0; c < 3; ++c)
                ept[ei3+c] = (positions[va*3+c] + positions[vb*3+c]
                            + fpt[f1*3+c]       + fpt[f2*3+c]) * 0.25;
        } else {
            for (int c = 0; c < 3; ++c)
                ept[ei3+c] = (positions[va*3+c] + positions[vb*3+c]) * 0.5;
        }
    }

    int32_t ne = ne_count;

    // ------------------------------------------------------------------
    // 3. V-points: CC vertex update rule.
    //    Interior: V' = (n-3)/n·V + (1/n²)·ΣF + (2/n²)·ΣM
    //    Boundary: V' = (6/8)·V + (1/8)·(V_prev + V_next)
    // ------------------------------------------------------------------
    std::vector<double>  v_face_sum(static_cast<size_t>(nv)*3, 0.0);
    std::vector<double>  v_edge_sum(static_cast<size_t>(nv)*3, 0.0);
    std::vector<int32_t> v_n(static_cast<size_t>(nv), 0);

    for (int32_t fi = 0; fi < nf; ++fi) {
        for (int k = 0; k < 4; ++k) {
            int32_t vi = quad_faces[fi*4+k];
            int32_t vj = quad_faces[fi*4+(k+1)%4];
            if (vi < 0 || vi >= nv || vj < 0 || vj >= nv) continue;  // (v79)
            v_n[vi]++;
            v_face_sum[vi*3+0] += fpt[fi*3+0];
            v_face_sum[vi*3+1] += fpt[fi*3+1];
            v_face_sum[vi*3+2] += fpt[fi*3+2];
            double mx = (positions[vi*3+0] + positions[vj*3+0]) * 0.5;
            double my = (positions[vi*3+1] + positions[vj*3+1]) * 0.5;
            double mz = (positions[vi*3+2] + positions[vj*3+2]) * 0.5;
            v_edge_sum[vi*3+0] += mx;
            v_edge_sum[vi*3+1] += my;
            v_edge_sum[vi*3+2] += mz;
        }
    }

    // Detect boundary vertices (v78 fix) — directed-edge scan
    struct BndNbrs { int32_t prev{-1}, next{-1}; };
    std::vector<BndNbrs> bnd_nbrs(static_cast<size_t>(nv));
    {
        std::unordered_map<int64_t, int8_t> dir_edges;
        dir_edges.reserve(static_cast<size_t>(nf) * 4);
        for (int32_t fi = 0; fi < nf; ++fi)
            for (int k = 0; k < 4; ++k) {
                int32_t a = quad_faces[fi*4+k];
                int32_t b = quad_faces[fi*4+(k+1)%4];
                if (a < 0 || b < 0 || a >= nv || b >= nv) continue;
                int64_t fwd = (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
                dir_edges[fwd] = 1;
            }

        std::unordered_map<int32_t, int32_t> bnd_next_map;
        for (const auto& [fwd, _] : dir_edges) {
            int32_t a = static_cast<int32_t>(static_cast<uint64_t>(fwd) >> 32);
            int32_t b = static_cast<int32_t>(static_cast<uint64_t>(fwd) & 0xFFFFFFFFULL);
            int64_t rev = (static_cast<int64_t>(b) << 32) | static_cast<uint32_t>(a);
            if (dir_edges.find(rev) == dir_edges.end())
                bnd_next_map[a] = b;
        }
        for (const auto& [a, b] : bnd_next_map) {
            if (b >= 0 && b < nv) bnd_nbrs[b].prev = a;
            if (a >= 0 && a < nv) bnd_nbrs[a].next = b;
        }
    }

    std::vector<double> vpt(static_cast<size_t>(nv)*3);
    for (int32_t vi = 0; vi < nv; ++vi) {
        const BndNbrs& bn = bnd_nbrs[vi];

        // Boundary vertex with both neighbours: CC boundary rule
        if (bn.prev >= 0 && bn.next >= 0) {
            for (int c = 0; c < 3; ++c)
                vpt[vi*3+c] = 0.75  * positions[vi*3+c]
                            + 0.125 * positions[bn.prev*3+c]
                            + 0.125 * positions[bn.next*3+c];
            continue;
        }
        // Degenerate boundary (single boundary edge): pin
        if (bn.prev >= 0 || bn.next >= 0) {
            for (int c = 0; c < 3; ++c)
                vpt[vi*3+c] = positions[vi*3+c];
            continue;
        }

        // Interior vertex
        double n = static_cast<double>(v_n[vi]);
        // (v79) pin degenerate-valence vertices; CC interior rule undefined for n<3
        if (n < 3.0) {
            for (int c = 0; c < 3; ++c)
                vpt[vi*3+c] = positions[vi*3+c];
            continue;
        }
        double inv_n  = 1.0 / n;
        double w      = (n - 3.0) * inv_n;   // (v78) was (n-2)/n
        double inv_n2 = inv_n * inv_n;
        // V' = (n-3)/n·V + (1/n²)·ΣF + (2/n²)·ΣM
        for (int c = 0; c < 3; ++c)
            vpt[vi*3+c] = w          * positions[vi*3+c]
                        + inv_n2     * v_face_sum[vi*3+c]
                        + 2.0*inv_n2 * v_edge_sum[vi*3+c];
    }

    // ------------------------------------------------------------------
    // 4. Pack output positions: [V-points | F-points | E-points]
    // ------------------------------------------------------------------
    out_positions.clear();
    out_positions.reserve(static_cast<size_t>(nv + nf + ne) * 3);
    for (int32_t vi = 0; vi < nv; ++vi) {
        out_positions.push_back(vpt[vi*3+0]);
        out_positions.push_back(vpt[vi*3+1]);
        out_positions.push_back(vpt[vi*3+2]);
    }
    for (int32_t fi = 0; fi < nf; ++fi) {
        out_positions.push_back(fpt[fi*3+0]);
        out_positions.push_back(fpt[fi*3+1]);
        out_positions.push_back(fpt[fi*3+2]);
    }
    for (int32_t ei = 0; ei < ne; ++ei) {
        out_positions.push_back(ept[ei*3+0]);
        out_positions.push_back(ept[ei*3+1]);
        out_positions.push_back(ept[ei*3+2]);
    }

    // ------------------------------------------------------------------
    // 5. Generate 4 child quads per input face.
    //    Child k of face fi: [ vs[k], EP[k], FP, EP[(k+3)%4] ]
    // ------------------------------------------------------------------
    out_quad_faces.clear();
    out_quad_faces.reserve(static_cast<size_t>(nf) * 16);

    for (int32_t fi = 0; fi < nf; ++fi) {
        int32_t FP = nv + fi;
        int32_t vs[4], EP[4];

        // (v80) validate all four corners before emitting child quads
        bool face_valid = true;
        for (int k = 0; k < 4; ++k) {
            vs[k] = quad_faces[fi*4+k];
            if (vs[k] < 0 || vs[k] >= nv) { face_valid = false; break; }
        }
        if (!face_valid) continue;

        for (int k = 0; k < 4; ++k) {
            auto it = edge_index.find(edge_key(vs[k], vs[(k+1)%4]));
            if (it == edge_index.end()) { face_valid = false; break; }
            EP[k] = nv + nf + it->second;
        }
        if (!face_valid) continue;

        for (int k = 0; k < 4; ++k) {
            out_quad_faces.push_back(vs[k]);
            out_quad_faces.push_back(EP[k]);
            out_quad_faces.push_back(FP);
            out_quad_faces.push_back(EP[(k+3)%4]);
        }
    }

    return true;
}

} // namespace qf
