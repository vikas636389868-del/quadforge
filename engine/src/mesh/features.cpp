/**
 * features.cpp — Feature edge detection: dihedral angle, normals, materials, UV seams.
 *
 * Bug fixes / new implementations (v17 pass):
 *  1. use_normals   — detect edges where per-vertex normals diverge significantly,
 *                     approximating Blender's split-normal feature edges.
 *  2. use_uv_seams  — detect seam edges via per-loop UV indices (uv_indices[]).
 *  3. is_boundary   — correctly mark FeatureCurve::is_boundary for boundary chains.
 *  4. Loop detection — fixed all-loop traversal to avoid duplicate/partial curves.
 */

#include "../../include/quadforge/mesh/features.h"

#include <cmath>
#include <set>
#include <unordered_map>
#include <queue>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static std::set<std::pair<int,int>> make_edge_set(const EdgeSet& es) {
    return {es.begin(), es.end()};
}

// ---------------------------------------------------------------------------
// detect_features
// ---------------------------------------------------------------------------

FeatureData detect_features(
    const HalfEdgeMesh& mesh,
    double angle_threshold,
    bool use_normals,
    bool use_materials,
    bool use_uv_seams,
    const std::vector<int>* material_ids,
    const Eigen::MatrixXf*  /* uv_coords — reserved, unused; see features.h */,
    const float*            raw_vertex_normals,
    const int32_t*          uv_indices,
    int32_t                 uv_face_stride)
{
    FeatureData fd;
    double threshold_rad = angle_threshold * M_PI / 180.0;
    // use_normals: half the hard-edge angle is used as the vertex-normal divergence threshold
    double normal_threshold_rad = threshold_rad * 0.5;
    int nhe = mesh.num_half_edges();

    // -------------------------------------------------------------------
    // Pass 1: Boundary detection + dihedral-angle hard edges
    // -------------------------------------------------------------------
    for (int he = 0; he < nhe; ++he) {
        int tw = mesh.half_edge(he).twin;

        if (tw < 0) {
            // Boundary edge — record once (he only; twin = -1)
            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;
            fd.boundary_edges.push_back({std::min(from,to), std::max(from,to)});
            continue;
        }
        if (tw < he) continue; // process each undirected interior edge once

        double dihedral = mesh.dihedral_angle(he);
        if (dihedral > threshold_rad) {
            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;
            fd.hard_edges.push_back({std::min(from,to), std::max(from,to)});
        }
    }

    // -------------------------------------------------------------------
    // Pass 2: use_normals — vertex-normal divergence
    //
    // Per-vertex normals from Blender carry custom/split-normal information.
    // An edge where the two endpoint normals diverge by more than
    // (angle_threshold / 2) is treated as a feature edge, even when the
    // dihedral angle is small (smooth geometry with marked sharp edges).
    // -------------------------------------------------------------------
    if (use_normals && raw_vertex_normals) {
        for (int he = 0; he < nhe; ++he) {
            int tw = mesh.half_edge(he).twin;
            if (tw < he || tw < 0) continue;

            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;

            float nfx = raw_vertex_normals[from*3  ];
            float nfy = raw_vertex_normals[from*3+1];
            float nfz = raw_vertex_normals[from*3+2];
            float ntx = raw_vertex_normals[to  *3  ];
            float nty = raw_vertex_normals[to  *3+1];
            float ntz = raw_vertex_normals[to  *3+2];

            float lf = std::sqrt(nfx*nfx + nfy*nfy + nfz*nfz);
            float lt = std::sqrt(ntx*ntx + nty*nty + ntz*ntz);
            if (lf < 1e-8f || lt < 1e-8f) continue;

            float dot_val = (nfx*ntx + nfy*nty + nfz*ntz) / (lf * lt);
            dot_val = std::max(-1.f, std::min(1.f, dot_val));

            if (std::acos(static_cast<double>(dot_val)) > normal_threshold_rad) {
                int lo = std::min(from,to), hi = std::max(from,to);
                fd.hard_edges.push_back({lo, hi});
            }
        }
    }

    // -------------------------------------------------------------------
    // Pass 3: use_materials — material-boundary edges
    // -------------------------------------------------------------------
    if (use_materials && material_ids) {
        for (int he = 0; he < nhe; ++he) {
            int tw = mesh.half_edge(he).twin;
            if (tw < he || tw < 0) continue;

            int fi = mesh.half_edge(he).face;
            int fj = mesh.half_edge(tw).face;
            if (fi < 0 || fj < 0) continue;

            int nf = (int)material_ids->size();
            if (fi >= nf || fj >= nf) continue;

            if ((*material_ids)[fi] != (*material_ids)[fj]) {
                int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
                int to   = mesh.half_edge(he).vertex;
                fd.hard_edges.push_back({std::min(from,to), std::max(from,to)});
            }
        }
    }

    // -------------------------------------------------------------------
    // Pass 4: use_uv_seams — per-loop UV seam detection
    //
    // uv_indices is a flat array laid out as
    //   [face0_v0_uv, face0_v1_uv, face0_v2_uv,  face1_v0_uv, ...]
    // where face indices match the HalfEdgeMesh's triangulated faces.
    // A seam exists on edge (a,b) when vertex 'a' or 'b' references
    // different UV-loop indices from the two adjacent faces.
    // -------------------------------------------------------------------
    if (use_uv_seams && uv_indices && uv_face_stride >= 3) {
        for (int he = 0; he < nhe; ++he) {
            int tw = mesh.half_edge(he).twin;
            if (tw < he || tw < 0) continue;

            int fi = mesh.half_edge(he).face;
            int fj = mesh.half_edge(tw).face;
            if (fi < 0 || fj < 0) continue;

            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;

            auto fv_i = mesh.face_vertices(fi);
            auto fv_j = mesh.face_vertices(fj);

            // Locate each endpoint within its face's vertex list
            int from_slot_i = -1, to_slot_i = -1;
            int from_slot_j = -1, to_slot_j = -1;
            for (int k = 0; k < 3; ++k) {
                if (fv_i[k] == from) from_slot_i = k;
                if (fv_i[k] == to)   to_slot_i   = k;
                if (fv_j[k] == from) from_slot_j = k;
                if (fv_j[k] == to)   to_slot_j   = k;
            }
            if (from_slot_i < 0 || from_slot_j < 0) continue;
            if (to_slot_i   < 0 || to_slot_j   < 0) continue;

            int32_t uv_from_fi = uv_indices[fi * uv_face_stride + from_slot_i];
            int32_t uv_from_fj = uv_indices[fj * uv_face_stride + from_slot_j];
            int32_t uv_to_fi   = uv_indices[fi * uv_face_stride + to_slot_i  ];
            int32_t uv_to_fj   = uv_indices[fj * uv_face_stride + to_slot_j  ];

            if (uv_from_fi != uv_from_fj || uv_to_fi != uv_to_fj) {
                int lo = std::min(from,to), hi = std::max(from,to);
                fd.uv_seam_edges.push_back({lo, hi});
                fd.hard_edges.push_back({lo, hi});
            }
        }
    }

    // -------------------------------------------------------------------
    // Deduplicate all edge sets
    // -------------------------------------------------------------------
    auto dedup = [](EdgeSet& es) {
        std::sort(es.begin(), es.end());
        es.erase(std::unique(es.begin(), es.end()), es.end());
    };
    dedup(fd.hard_edges);
    dedup(fd.boundary_edges);
    dedup(fd.uv_seam_edges);

    // -------------------------------------------------------------------
    // Chain edges into feature curves (hard + boundary combined)
    // -------------------------------------------------------------------
    EdgeSet all_feature_edges = fd.hard_edges;
    all_feature_edges.insert(all_feature_edges.end(),
                              fd.boundary_edges.begin(), fd.boundary_edges.end());
    dedup(all_feature_edges);

    fd.curves = chain_feature_edges(all_feature_edges, mesh.num_vertices());

    // -------------------------------------------------------------------
    // Mark is_boundary on curves consisting entirely of boundary edges
    // -------------------------------------------------------------------
    if (!fd.boundary_edges.empty()) {
        auto bnd_set = make_edge_set(fd.boundary_edges);
        for (auto& curve : fd.curves) {
            bool all_bnd = true;
            for (int i = 0; i + 1 < (int)curve.vertices.size(); ++i) {
                int lo = std::min(curve.vertices[i], curve.vertices[i+1]);
                int hi = std::max(curve.vertices[i], curve.vertices[i+1]);
                if (!bnd_set.count({lo, hi})) { all_bnd = false; break; }
            }
            if (all_bnd) curve.is_boundary = true;
        }
    }

    // -------------------------------------------------------------------
    // Corner vertices: valence ≥ 3 in the combined feature graph
    // -------------------------------------------------------------------
    std::unordered_map<int,int> valence;
    for (auto& [lo, hi] : all_feature_edges) {
        valence[lo]++;
        valence[hi]++;
    }
    for (auto& [v, val] : valence)
        if (val >= 3) fd.corner_vertices.push_back(v);

    return fd;
}

// ---------------------------------------------------------------------------
// chain_feature_edges
// ---------------------------------------------------------------------------
std::vector<FeatureCurve> chain_feature_edges(const EdgeSet& edges, int num_vertices) {
    (void)num_vertices;
    if (edges.empty()) return {};

    // Build adjacency
    std::unordered_map<int, std::vector<int>> adj;
    for (auto& [lo, hi] : edges) {
        adj[lo].push_back(hi);
        adj[hi].push_back(lo);
    }

    std::unordered_map<int,int> valence;
    for (auto& [v, nbs] : adj) valence[v] = (int)nbs.size();

    // Edge-usage tracking
    std::unordered_map<int64_t, bool> used_edges;
    // BUG FIX (v103): The original lambda used `int64_t(std::min(a,b)) << 32`
    // which is undefined behaviour when std::min(a,b) is negative — left-shifting
    // a negative signed integer is UB in C++ [expr.shift §7.6.7].  Although
    // vertex indices should always be non-negative, the same defensive fix applied
    // to halfedge.h::EdgeKeyHash and repair.cpp::detect_self_intersections should
    // be applied here for consistency and to silence -Wshift-overflow warnings.
    // Fix: promote both components through uint32_t → uint64_t before shifting,
    // exactly matching the pattern in halfedge.h::EdgeKeyHash.
    auto edge_key = [](int a, int b) -> int64_t {
        return (static_cast<int64_t>(static_cast<uint32_t>(std::min(a,b))) << 32)
             | static_cast<int64_t>(static_cast<uint32_t>(std::max(a,b)));
    };

    std::vector<FeatureCurve> curves;

    // Helper: trace one chain starting at (start→nb)
    auto trace_chain = [&](int start, int nb) {
        FeatureCurve curve;
        curve.vertices.push_back(start);
        int cur = start, next = nb;
        int guard = 0;
        int max_guard = (int)edges.size() * 2 + 4;

        while (true) {
            if (++guard > max_guard) break; // safety
            int64_t cur_ek = edge_key(cur, next);
            if (used_edges.count(cur_ek)) break;
            used_edges[cur_ek] = true;
            curve.vertices.push_back(next);

            if (next == start) { curve.is_closed = true; break; }
            if (valence[next] != 2) break; // endpoint or corner — stop

            // Continue along the single unused neighbour
            int new_next = -1;
            for (int nn : adj[next]) {
                if (nn != cur && !used_edges.count(edge_key(next, nn))) {
                    new_next = nn; break;
                }
            }
            if (new_next < 0) break;
            cur = next; next = new_next;
        }

        if ((int)curve.vertices.size() >= 2)
            curves.push_back(std::move(curve));
    };

    // --- Phase 1: open chains — start from endpoints (val=1) and corners (val≥3) ---
    for (auto& [start, nbs] : adj) {
        int val = valence[start];
        if (val == 2) continue;
        for (int nb : nbs) {
            if (!used_edges.count(edge_key(start, nb)))
                trace_chain(start, nb);
        }
    }

    // --- Phase 2: closed loops — only valence-2 vertices remain ---
    // Track which vertices we've already used as a loop seed to avoid starting
    // from the middle of an already-traced loop.
    std::unordered_map<int,bool> loop_started;
    for (auto& [start, nbs] : adj) {
        if (valence[start] != 2) continue;
        if (loop_started.count(start)) continue;

        for (int nb : nbs) {
            if (!used_edges.count(edge_key(start, nb))) {
                loop_started[start] = true;
                trace_chain(start, nb);
                // Mark all vertices of the newly added loop as started
                if (!curves.empty()) {
                    for (int v : curves.back().vertices)
                        loop_started[v] = true;
                }
                break; // only need one direction from each loop seed
            }
        }
    }

    return curves;
}

} // namespace qf
