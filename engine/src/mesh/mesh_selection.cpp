/**
 * mesh_selection.cpp — Incremental / region-based remeshing primitives.
 *
 * Implements the declarations in mesh_selection.h.
 *
 * Implementation notes
 * --------------------
 * * extract_sub_mesh() performs a two-pass linear scan:
 *     Pass 1 — collect used original vertex indices and build orig_to_sub / sub_to_orig.
 *     Pass 2 — remap face indices and extract boundary vertices.
 *   Both passes are O(F_sel + V_sel).
 *
 * * grow_selection_by_rings() and shrink_selection_by_rings() use the
 *   vertex-adjacency in the HalfEdgeMesh to expand/contract the selection.
 *   Each ring costs O(V + F) in the worst case.
 *
 * * stitch_sub_mesh() builds a TriangleBVH over the original boundary
 *   vertices for O(B log B) snapping queries.  It then compacts the original
 *   unselected mesh and appends the snapped remeshed geometry.
 *
 * * All newly constructed meshes are built from flat vertex-position +
 *   triangle-index arrays via HalfEdgeMesh constructors so they inherit
 *   the full half-edge adjacency structure.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#include "../../include/quadforge/mesh/mesh_selection.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "../../include/quadforge/mesh/spatial.h"    // TriangleBVH
#include "../../third_party/nanoflann/nanoflann.hpp" // BUG FIX (v103): BVH point-cloud NN for stitch_sub_mesh

namespace qf {
    struct PointCloud {
        const std::vector<Vec3>& pts;

        size_t kdtree_get_point_count() const { return pts.size(); }

        double kdtree_get_pt(size_t idx, size_t dim) const {
            return pts[idx][static_cast<Eigen::Index>(dim)];
        }

        template<class BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
    };

// =========================================================================
// SelectionMask helpers
// =========================================================================

SelectionMask build_selection_mask(int n_faces, const std::vector<int>& face_indices)
{
    SelectionMask mask(n_faces, false);
    for (int fi : face_indices) {
        if (fi >= 0 && fi < n_faces) mask[fi] = true;
    }
    return mask;
}

SelectionMask invert_selection_mask(const SelectionMask& mask)
{
    SelectionMask inv(mask.size());
    for (int i = 0; i < static_cast<int>(mask.size()); ++i)
        inv[i] = !mask[i];
    return inv;
}

int count_selected_faces(const SelectionMask& mask)
{
    int count = 0;
    for (bool b : mask) if (b) ++count;
    return count;
}

// =========================================================================
// Selection expansion / contraction
// =========================================================================

SelectionMask grow_selection_by_rings(
    const HalfEdgeMesh& mesh,
    const SelectionMask& mask,
    int rings)
{
    if (rings <= 0) return mask;

    SelectionMask current = mask;
    const int nv = mesh.num_vertices();
    const int nf = mesh.num_faces();

    // Build vertex→faces adjacency from the half-edge structure.
    // We use the half-edge vertex_faces callback.
    for (int ring = 0; ring < rings; ++ring) {
        // Collect all vertices in currently-selected faces.
        std::vector<bool> vertex_in_selection(nv, false);
        for (int fi = 0; fi < nf; ++fi) {
            if (!current[fi]) continue;
            const auto verts = mesh.face_vertices(fi);
            for (int v : verts) vertex_in_selection[v] = true;
        }

        // Grow: mark any face that shares a vertex with the selection.
        SelectionMask next = current;
        for (int fi = 0; fi < nf; ++fi) {
            if (next[fi]) continue;  // already selected
            const auto verts = mesh.face_vertices(fi);
            for (int v : verts) {
                if (vertex_in_selection[v]) { next[fi] = true; break; }
            }
        }
        current = std::move(next);
    }
    return current;
}

SelectionMask shrink_selection_by_rings(
    const HalfEdgeMesh& mesh,
    const SelectionMask& mask,
    int rings)
{
    if (rings <= 0) return mask;

    SelectionMask current = mask;
    const int nv = mesh.num_vertices();
    const int nf = mesh.num_faces();

    for (int ring = 0; ring < rings; ++ring) {
        // Collect all vertices in currently-UNselected faces.
        std::vector<bool> vertex_near_boundary(nv, false);
        for (int fi = 0; fi < nf; ++fi) {
            if (current[fi]) continue;  // selected → not a boundary marker
            const auto verts = mesh.face_vertices(fi);
            for (int v : verts) vertex_near_boundary[v] = true;
        }

        // Shrink: deselect any currently-selected face that has a vertex
        // touching an unselected face.
        SelectionMask next = current;
        for (int fi = 0; fi < nf; ++fi) {
            if (!current[fi]) continue;  // already unselected
            const auto verts = mesh.face_vertices(fi);
            for (int v : verts) {
                if (vertex_near_boundary[v]) { next[fi] = false; break; }
            }
        }
        current = std::move(next);
    }
    return current;
}

// =========================================================================
// identify_boundary_vertices
// =========================================================================

std::vector<int> identify_boundary_vertices(
    const HalfEdgeMesh&  mesh,
    const SelectionMask& mask)
{
    assert(static_cast<int>(mask.size()) == mesh.num_faces());

    // For each vertex, track whether it touches selected and unselected faces.
    const int nv = mesh.num_vertices();
    std::vector<bool> touches_selected  (nv, false);
    std::vector<bool> touches_unselected(nv, false);

    for (int fi = 0; fi < mesh.num_faces(); ++fi) {
        bool sel = mask[fi];
        const auto verts = mesh.face_vertices(fi);
        for (int v : verts) {
            if (sel) touches_selected  [v] = true;
            else     touches_unselected[v] = true;
        }
    }

    std::vector<int> boundary;
    boundary.reserve(256);
    for (int v = 0; v < nv; ++v) {
        if (touches_selected[v] && touches_unselected[v])
            boundary.push_back(v);
    }
    // Already in ascending order.
    return boundary;
}

// =========================================================================
// extract_sub_mesh
// =========================================================================

SubMeshData extract_sub_mesh(
    const HalfEdgeMesh&       mesh,
    const SelectionMask&      mask,
    const float*              normals,
    const std::vector<Vec3>*  vertex_colors,
    const std::vector<int>*   material_ids)
{
    assert(static_cast<int>(mask.size()) == mesh.num_faces());
    const int nv = mesh.num_vertices();
    const int nf = mesh.num_faces();

    // -----------------------------------------------------------------------
    // Pass 1 — collect used original vertices and selected faces
    // -----------------------------------------------------------------------
    std::vector<int> orig_to_sub(nv, -1);
    std::vector<int> sub_to_orig;

    std::vector<int> selected_orig_faces;
    for (int fi = 0; fi < nf; ++fi) {
        if (!mask[fi]) continue;
        selected_orig_faces.push_back(fi);
        const auto verts = mesh.face_vertices(fi);
        for (int v : verts) {
            if (orig_to_sub[v] == -1) {
                orig_to_sub[v] = static_cast<int>(sub_to_orig.size());
                sub_to_orig.push_back(v);
            }
        }
    }

    const int n_sub_v = static_cast<int>(sub_to_orig.size());
    const int n_sub_f = static_cast<int>(selected_orig_faces.size());

    // -----------------------------------------------------------------------
    // Pass 2 — build sub-mesh positions and remapped face indices
    // -----------------------------------------------------------------------
    std::vector<Vec3> sub_positions(n_sub_v);
    for (int sv = 0; sv < n_sub_v; ++sv)
        sub_positions[sv] = mesh.vertex_pos(sub_to_orig[sv]);

    std::vector<std::array<int,3>> sub_tris(n_sub_f);
    std::vector<int> sub_face_to_orig(n_sub_f);
    std::vector<int> orig_face_to_sub(nf, -1);

    for (int sfi = 0; sfi < n_sub_f; ++sfi) {
        int orig_fi = selected_orig_faces[sfi];
        sub_face_to_orig[sfi]   = orig_fi;
        orig_face_to_sub[orig_fi] = sfi;

        const auto ov = mesh.face_vertices(orig_fi);
        sub_tris[sfi] = {orig_to_sub[ov[0]],
                         orig_to_sub[ov[1]],
                         orig_to_sub[ov[2]]};
    }

    // -----------------------------------------------------------------------
    // Pass 3 — identify boundary vertices
    // -----------------------------------------------------------------------
    std::vector<int> bnd_orig = identify_boundary_vertices(mesh, mask);

    // Remap boundary to sub-mesh indices.
    std::vector<int> bnd_sub;
    bnd_sub.reserve(bnd_orig.size());
    for (int ov : bnd_orig) {
        int sv = orig_to_sub[ov];
        if (sv != -1) bnd_sub.push_back(sv);
    }
    // bnd_sub may have duplicates if multiple orig boundary verts map to
    // the same sub vertex (shouldn't happen, but sort+unique to be safe).
    std::sort(bnd_sub.begin(), bnd_sub.end());
    bnd_sub.erase(std::unique(bnd_sub.begin(), bnd_sub.end()), bnd_sub.end());

    // -----------------------------------------------------------------------
    // Build boundary EdgeSet (edges of the sub-mesh that touch boundary verts
    // and separate selected from unselected regions in the original mesh).
    // -----------------------------------------------------------------------
    // An edge (sv0, sv1) is a boundary edge if BOTH vertices are on the
    // boundary AND that edge is shared by exactly one sub-mesh face
    // (i.e., the twin face is unselected or boundary in the original mesh).
    std::unordered_set<int> bnd_sub_set(bnd_sub.begin(), bnd_sub.end());
    EdgeSet boundary_edges;

    for (int sfi = 0; sfi < n_sub_f; ++sfi) {
        int orig_fi = sub_face_to_orig[sfi];
        const auto ohe = mesh.face_half_edges(orig_fi);
        for (int hei : ohe) {
            const HalfEdge& he = mesh.half_edge(hei);
            int twin = he.twin;
            // Check if this half-edge's twin belongs to an unselected face.
            bool twin_unselected = (twin == -1)  // boundary
                || !mask[mesh.half_edge(twin).face];
            if (!twin_unselected) continue;

            // Edge vertices in sub-mesh indices.
            int v_dst = orig_to_sub[he.vertex];
            int v_src = orig_to_sub[mesh.half_edge(he.prev).vertex];
            if (v_dst < 0 || v_src < 0) continue;

            int v_lo = std::min(v_src, v_dst);
            int v_hi = std::max(v_src, v_dst);
            boundary_edges.push_back({v_lo, v_hi});
        }
    }
    // Deduplicate boundary edges.
    std::sort(boundary_edges.begin(), boundary_edges.end());
    boundary_edges.erase(std::unique(boundary_edges.begin(), boundary_edges.end()),
                         boundary_edges.end());

    // -----------------------------------------------------------------------
    // Build the HalfEdgeMesh for the sub-mesh.
    // -----------------------------------------------------------------------
    HalfEdgeMesh sub_mesh(sub_positions, sub_tris);

    // -----------------------------------------------------------------------
    // Forward optional per-vertex attributes.
    // -----------------------------------------------------------------------
    SubMeshData result;

    if (normals) {
        result.sub_normals.reserve(n_sub_v);
        for (int sv = 0; sv < n_sub_v; ++sv) {
            int ov = sub_to_orig[sv];
            result.sub_normals.push_back({
                static_cast<double>(normals[ov*3+0]),
                static_cast<double>(normals[ov*3+1]),
                static_cast<double>(normals[ov*3+2])});
        }
    }

    if (vertex_colors && static_cast<int>(vertex_colors->size()) >= nv) {
        result.sub_vertex_colors.reserve(n_sub_v);
        for (int sv = 0; sv < n_sub_v; ++sv)
            result.sub_vertex_colors.push_back((*vertex_colors)[sub_to_orig[sv]]);
    }

    if (material_ids && static_cast<int>(material_ids->size()) >= nf) {
        result.sub_material_ids.reserve(n_sub_f);
        for (int sfi = 0; sfi < n_sub_f; ++sfi)
            result.sub_material_ids.push_back((*material_ids)[sub_face_to_orig[sfi]]);
    }

    // -----------------------------------------------------------------------
    // Populate result struct.
    // -----------------------------------------------------------------------
    result.sub_mesh             = std::move(sub_mesh);
    result.sub_to_orig          = std::move(sub_to_orig);
    result.orig_to_sub          = std::move(orig_to_sub);
    result.sub_face_to_orig     = std::move(sub_face_to_orig);
    result.orig_face_to_sub     = std::move(orig_face_to_sub);
    result.boundary_sub_vertices  = std::move(bnd_sub);
    result.boundary_orig_vertices = std::move(bnd_orig);
    result.boundary_edges         = std::move(boundary_edges);
    result.num_sub_vertices       = n_sub_v;
    result.num_sub_faces          = n_sub_f;
    result.num_boundary_verts     = static_cast<int>(result.boundary_sub_vertices.size());

    return result;
}

// =========================================================================
// build_boundary_feature_edges
// =========================================================================

EdgeSet build_boundary_feature_edges(const SubMeshData& sub_data)
{
    // The boundary_edges field already contains the selection-boundary edges
    // in sub-mesh index space.  Return a copy.
    return sub_data.boundary_edges;
}

// =========================================================================
// Utility — find_free_boundary_vertices
// =========================================================================

std::vector<int> find_free_boundary_vertices(
    int n_verts,
    const std::vector<std::vector<int>>& faces)
{
    // Count how many faces reference each edge.
    std::unordered_map<int64_t, int> edge_count;
    edge_count.reserve(faces.size() * 3);

    auto edge_key = [](int v0, int v1) -> int64_t {
        if (v0 > v1) std::swap(v0, v1);
        return (static_cast<int64_t>(static_cast<uint32_t>(v0)) << 32)
               | static_cast<int64_t>(static_cast<uint32_t>(v1));
    };

    for (const auto& f : faces) {
        const int n = static_cast<int>(f.size());
        for (int i = 0; i < n; ++i) {
            int64_t k = edge_key(f[i], f[(i+1) % n]);
            ++edge_count[k];
        }
    }

    std::unordered_set<int> bnd_set;
    bnd_set.reserve(64);
    for (const auto& [key, count] : edge_count) {
        if (count == 1) {
            // BUG FIX (v107): Cast via uint32_t first to avoid sign-bit
            // corruption when vertex indices approach 2^31.  A direct
            // static_cast<int> from a 64-bit value whose bit-31 is set
            // produces a negative int (implementation-defined in C++17,
            // UB in C++14 and earlier).  Using uint32_t intermediate is
            // safe and matches the edge_key() packing convention above.
            int v0 = static_cast<int>(static_cast<uint32_t>((key >> 32) & 0xFFFFFFFFu));
            int v1 = static_cast<int>(static_cast<uint32_t>( key        & 0xFFFFFFFFu));
            bnd_set.insert(v0);
            bnd_set.insert(v1);
        }
    }

    std::vector<int> result(bnd_set.begin(), bnd_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

// =========================================================================
// Utility — mean_edge_length
// =========================================================================

double mean_edge_length(
    const Eigen::MatrixXd&               verts,
    const std::vector<std::vector<int>>& faces)
{
    if (faces.empty() || verts.rows() == 0) return 0.0;

    double total_len = 0.0;
    int    count     = 0;

    std::unordered_map<int64_t, bool> seen;
    seen.reserve(faces.size() * 3);

    auto edge_key = [](int v0, int v1) -> int64_t {
        if (v0 > v1) std::swap(v0, v1);
        return (static_cast<int64_t>(static_cast<uint32_t>(v0)) << 32)
               | static_cast<int64_t>(static_cast<uint32_t>(v1));
    };

    const int nv = static_cast<int>(verts.rows());
    for (const auto& f : faces) {
        const int n = static_cast<int>(f.size());
        for (int i = 0; i < n; ++i) {
            int va = f[i], vb = f[(i+1) % n];
            if (va < 0 || va >= nv || vb < 0 || vb >= nv) continue;
            int64_t k = edge_key(va, vb);
            if (!seen[k]) {
                seen[k] = true;
                Vec3 pa = verts.row(va).transpose();
                Vec3 pb = verts.row(vb).transpose();
                total_len += (pb - pa).norm();
                ++count;
            }
        }
    }

    return (count > 0) ? total_len / count : 0.0;
}

// =========================================================================
// stitch_sub_mesh
// =========================================================================

StitchResult stitch_sub_mesh(
    const HalfEdgeMesh&                  original_mesh,
    const SelectionMask&                 mask,
    const SubMeshData&                   sub_data,
    const Eigen::MatrixXd&               remeshed_verts,
    const std::vector<std::vector<int>>& remeshed_faces,
    const StitchOptions&                 opts)
{
    StitchResult result;
    result.stitch_ok = false;

    const int n_rem_v = static_cast<int>(remeshed_verts.rows());
    const int orig_nv = original_mesh.num_vertices();
    const int orig_nf = original_mesh.num_faces();

    if (n_rem_v == 0 || remeshed_faces.empty()) {
        result.error_message = "Empty remeshed mesh; nothing to stitch.";
        return result;
    }

    // -----------------------------------------------------------------------
    // 1. Auto-compute snap distance
    // -----------------------------------------------------------------------
    double snap_dist = opts.snap_distance;
    if (snap_dist <= 0.0) {
        snap_dist = 2.0 * mean_edge_length(remeshed_verts, remeshed_faces);
        if (snap_dist < 1e-12) snap_dist = 1e-3;  // fallback if mesh is tiny
    }
    result.snap_distance_used = snap_dist;

    // -----------------------------------------------------------------------
    // 2. Find free boundary vertices of the remeshed sub-mesh
    // -----------------------------------------------------------------------
    std::vector<int> rem_boundary =
        find_free_boundary_vertices(n_rem_v, remeshed_faces);
    result.boundary_total = static_cast<int>(rem_boundary.size());

    // -----------------------------------------------------------------------
    // 3. Build nearest-neighbour structure over original boundary vertices
    // -----------------------------------------------------------------------
    // BUG FIX (v103): The previous implementation used a LINEAR SCAN O(N²)
    // for ALL boundary sizes.  The comment said "use linear scan for small
    // boundary counts (<512)" but there was no BVH fallback for larger counts.
    // For large meshes the quadratic scan dominated total remeshing time.
    //
    // Fix: build a nanoflann KD-tree over the boundary vertex positions for
    // O(log N) per query when |boundary| > kLinearScanThreshold, and fall back
    // to the linear scan for small counts to avoid KD-tree build overhead.

    const auto& bnd_orig = sub_data.boundary_orig_vertices;
    if (bnd_orig.empty() && result.boundary_total > 0) {
        result.error_message = "Remeshed mesh has boundary vertices but original "
                               "sub-data has no boundary info; cannot stitch.";
        return result;
    }

    // Cache boundary vertex positions once.
    const int n_bnd = static_cast<int>(bnd_orig.size());
    std::vector<Vec3> bnd_pos_cache(n_bnd);
    for (int bi = 0; bi < n_bnd; ++bi)
        bnd_pos_cache[bi] = original_mesh.vertex_pos(bnd_orig[bi]);

    // nanoflann adapter for a flat Vec3 array.
   
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud>,
        PointCloud, 3>;

    constexpr int kLinearScanThreshold = 512;
    std::unique_ptr<PointCloud> kd_cloud;
    std::unique_ptr<KDTree>     kd_tree;

    if (n_bnd > kLinearScanThreshold) {
        kd_cloud = std::make_unique<PointCloud>(PointCloud{bnd_pos_cache});
        kd_tree  = std::make_unique<KDTree>(
            3, *kd_cloud,
            nanoflann::KDTreeSingleIndexAdaptorParams(16));
        kd_tree->buildIndex();
    }

    auto find_nearest_bnd = [&](const Vec3& q) -> std::pair<int, double> {
        if (kd_tree) {
            // O(log N) KD-tree query.
            const double qpt[3] = { q.x(), q.y(), q.z() };
            uint32_t nn_idx;
            double   nn_dist_sq;
            nanoflann::KNNResultSet<double, uint32_t> rs(1);
            rs.init(&nn_idx, &nn_dist_sq);
            kd_tree->findNeighbors(rs, qpt, nanoflann::SearchParameters());
            return { static_cast<int>(nn_idx), std::sqrt(nn_dist_sq) };
        }
        // O(N) linear scan for small sets.
        double best_dist = std::numeric_limits<double>::max();
        int    best_idx  = -1;
        for (int bi = 0; bi < n_bnd; ++bi) {
            double d = (q - bnd_pos_cache[bi]).norm();
            if (d < best_dist) { best_dist = d; best_idx = bi; }
        }
        return { best_idx, best_dist };
    };

    // -----------------------------------------------------------------------
    // 4. Snap remeshed boundary vertices
    // -----------------------------------------------------------------------
    // snap_remap[rem_vi] = original vertex index if snapped, else -1.
    std::vector<int> snap_remap(n_rem_v, -1);
    Eigen::MatrixXd  snapped_verts = remeshed_verts;  // mutable copy

    int snapped_count = 0;
    for (int rem_vi : rem_boundary) {
        Vec3 q = snapped_verts.row(rem_vi).transpose();
        auto [bi, dist] = find_nearest_bnd(q);
        if (bi >= 0 && dist <= snap_dist) {
            Vec3 p = original_mesh.vertex_pos(bnd_orig[bi]);
            snapped_verts.row(rem_vi) = p.transpose();
            snap_remap[rem_vi] = bnd_orig[bi];
            ++snapped_count;
        }
    }
    result.boundary_snapped = snapped_count;

    // Validate snap ratio.
    if (result.boundary_total > 0) {
        double snap_fail_ratio =
            1.0 - static_cast<double>(snapped_count) / result.boundary_total;
        if (snap_fail_ratio > opts.max_snap_failure_ratio) {
            result.error_message = "Too many boundary vertices failed to snap ("
                + std::to_string(static_cast<int>(snap_fail_ratio * 100))
                + "% failure; max allowed "
                + std::to_string(static_cast<int>(opts.max_snap_failure_ratio * 100))
                + "%).";
            return result;
        }
    }

    // -----------------------------------------------------------------------
    // 5. Build the "kept" original mesh (unselected faces only)
    // -----------------------------------------------------------------------
    // Compact: remove selected faces and orphaned vertices.
    std::vector<bool> vert_used(orig_nv, false);
    for (int fi = 0; fi < orig_nf; ++fi) {
        if (mask[fi]) continue;  // selected → removed
        const auto v = original_mesh.face_vertices(fi);
        for (int vv : v) vert_used[vv] = true;
    }

    // Also mark original boundary verts as used if they will be re-used as
    // merge targets for snapped remeshed vertices.
    if (opts.merge_snapped_vertices) {
        for (int ov : bnd_orig) vert_used[ov] = true;
    }

    // Build old→new compact index map for original vertices.
    std::vector<int> old_to_new_orig(orig_nv, -1);
    std::vector<Vec3> kept_positions;
    kept_positions.reserve(orig_nv);
    for (int v = 0; v < orig_nv; ++v) {
        if (vert_used[v]) {
            old_to_new_orig[v] = static_cast<int>(kept_positions.size());
            kept_positions.push_back(original_mesh.vertex_pos(v));
        }
    }
    const int n_kept = static_cast<int>(kept_positions.size());

    // Remap unselected face indices.
    std::vector<std::array<int,3>> kept_tris;
    kept_tris.reserve(orig_nf);
    for (int fi = 0; fi < orig_nf; ++fi) {
        if (mask[fi]) continue;
        const auto ov = original_mesh.face_vertices(fi);
        kept_tris.push_back({old_to_new_orig[ov[0]],
                             old_to_new_orig[ov[1]],
                             old_to_new_orig[ov[2]]});
    }

    // -----------------------------------------------------------------------
    // 6. Build final_remap[rem_vi] = index in combined vertex array
    // -----------------------------------------------------------------------
    // Remeshed vertices:
    //   - If snapped and merge_snapped_vertices: use the kept original vert index.
    //   - Otherwise: append as new vertex (index = n_kept + offset).
    std::vector<int> final_remap(n_rem_v, -1);
    std::vector<Vec3> appended_positions;
    appended_positions.reserve(n_rem_v);

    for (int rv = 0; rv < n_rem_v; ++rv) {
        int orig_v = snap_remap[rv];
        if (opts.merge_snapped_vertices && orig_v >= 0) {
            int compact_idx = old_to_new_orig[orig_v];
            if (compact_idx >= 0) {
                final_remap[rv] = compact_idx;
                continue;
            }
        }
        // Append as new vertex.
        final_remap[rv] = n_kept + static_cast<int>(appended_positions.size());
        Vec3 p = snapped_verts.row(rv).transpose();
        appended_positions.push_back(p);
    }

    // -----------------------------------------------------------------------
    // 7. Assemble combined vertex and face lists
    // -----------------------------------------------------------------------
    // Vertices: kept_positions + appended_positions
    std::vector<Vec3> combined_positions;
    combined_positions.reserve(n_kept + appended_positions.size());
    for (const Vec3& p : kept_positions)   combined_positions.push_back(p);
    for (const Vec3& p : appended_positions) combined_positions.push_back(p);

    // Faces: kept triangles + remeshed faces (quads split into tris for HalfEdgeMesh).
    std::vector<std::array<int,3>> combined_tris = kept_tris;
    EdgeSet stitch_edges;

    for (const auto& rf : remeshed_faces) {
        const int fn = static_cast<int>(rf.size());
        if (fn < 3) continue;  // degenerate

        // Remap through final_remap.
        std::vector<int> mapped(fn);
        for (int i = 0; i < fn; ++i) {
            int rv = rf[i];
            if (rv < 0 || rv >= n_rem_v) { mapped[i] = -1; continue; }
            mapped[i] = final_remap[rv];
        }

        // Fan-triangulate if quad or polygon.
        for (int i = 1; i + 1 < fn; ++i) {
            if (mapped[0] < 0 || mapped[i] < 0 || mapped[i+1] < 0) continue;

            // BUG FIX (v106): When two or more remeshed boundary vertices snap
            // to the *same* original vertex (opts.merge_snapped_vertices is
            // true and two close-by remeshed verts land on the same original
            // boundary vert), their final_remap[] entries become identical.
            // The triangle {a, b, b} or {a, a, c} has zero area and a
            // degenerate half-edge structure that breaks topology queries
            // downstream.  Drop such triangles before they reach the
            // HalfEdgeMesh constructor.
            if (mapped[0] == mapped[i] ||
                mapped[i] == mapped[i+1] ||
                mapped[0] == mapped[i+1]) continue;

            combined_tris.push_back({mapped[0], mapped[i], mapped[i+1]});
        }

        // Track stitch boundary edges (edges where one vertex came from
        // the original mesh and one from the remeshed output, or both are
        // snapped).
        if (opts.preserve_stitch_boundary_as_features) {
            for (int i = 0; i < fn; ++i) {
                int va = mapped[i], vb = mapped[(i+1) % fn];
                if (va < 0 || vb < 0) continue;
                bool va_orig = (va < n_kept);
                bool vb_orig = (vb < n_kept);
                if (va_orig != vb_orig) {
                    stitch_edges.push_back({std::min(va,vb), std::max(va,vb)});
                }
            }
        }
    }

    // Deduplicate stitch edges.
    std::sort(stitch_edges.begin(), stitch_edges.end());
    stitch_edges.erase(std::unique(stitch_edges.begin(), stitch_edges.end()),
                       stitch_edges.end());

    // -----------------------------------------------------------------------
    // 8. Build final HalfEdgeMesh
    // -----------------------------------------------------------------------
    HalfEdgeMesh combined(combined_positions, combined_tris);

    result.combined_mesh  = std::move(combined);
    result.stitch_edges   = std::move(stitch_edges);
    result.stitch_ok      = true;
    return result;
}

} // namespace qf
