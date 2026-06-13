#pragma once
/* quadforge/mesh/features.h — Feature edge detection.
 *
 * Detects hard edges by dihedral angle, normal discontinuities,
 * material boundaries, and UV seams.  Chains detected edges into
 * connected feature curves for use as cross-field constraints.
 */

#ifndef QUADFORGE_MESH_FEATURES_H
#define QUADFORGE_MESH_FEATURES_H

#include <vector>
#include <set>
#include <utility>
#include <cstdint>
#include "../types.h"
#include "halfedge.h"

namespace qf {

/** A single feature curve: ordered chain of vertex indices. */
struct FeatureCurve {
    std::vector<int> vertices;  ///< Vertex indices in order
    bool is_closed{false};      ///< True if the curve forms a loop
    bool is_boundary{false};    ///< True if this is a mesh boundary curve
};

/** All detected feature information for a mesh. */
struct FeatureData {
    EdgeSet                    hard_edges;     ///< Set of (v_lo, v_hi) feature edges
    EdgeSet                    boundary_edges; ///< Mesh boundary edges
    EdgeSet                    uv_seam_edges;  ///< UV seam edges
    std::vector<FeatureCurve>  curves;         ///< Chained feature curves
    std::vector<int>           corner_vertices;///< Vertices where ≥3 feature curves meet
};

/**
 * Detect feature edges using all requested methods.
 *
 * @param mesh            The half-edge mesh
 * @param angle_threshold Dihedral angle threshold in degrees (e.g. 30.0)
 * @param use_normals     Detect split-normal discontinuities via vertex normals
 * @param use_materials   Add material boundary edges
 * @param use_uv_seams    Add UV seam edges (requires uv_indices)
 * @param material_ids    Per-face material index (needed if use_materials)
 * @param uv_coords       Reserved / unused — pass nullptr.  UV seam detection
 *                        operates exclusively on uv_indices (integer loop IDs)
 *                        and does NOT require the float UV coordinates.
 *                        The parameter is kept for ABI compatibility but will
 *                        be removed in a future API version.
 * @param raw_vertex_normals  Flat [N*3] per-vertex normals from Blender (needed if use_normals)
 * @param uv_indices      Per-triangle-loop UV index array [T*3] (needed if use_uv_seams)
 * @param uv_face_stride  Number of UV indices per triangulated face (always 3 after triangulation)
 */
FeatureData detect_features(
    const HalfEdgeMesh& mesh,
    double              angle_threshold,
    bool                use_normals,
    bool                use_materials,
    bool                use_uv_seams,
    const std::vector<int>*   material_ids      = nullptr,
    const Eigen::MatrixXf*    uv_coords         = nullptr,
    const float*              raw_vertex_normals = nullptr,
    const int32_t*            uv_indices         = nullptr,
    int32_t                   uv_face_stride     = 3
);

/**
 * Assemble detected edges into connected feature curves.
 * Corner vertices (valence ≥ 3 in the feature graph) break curves.
 */
std::vector<FeatureCurve> chain_feature_edges(
    const EdgeSet& edges,
    int            num_vertices
);

} // namespace qf

#endif // QUADFORGE_MESH_FEATURES_H
