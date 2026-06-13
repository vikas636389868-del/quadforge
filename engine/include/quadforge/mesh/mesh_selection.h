#pragma once
/**
 * quadforge/mesh/mesh_selection.h — Incremental / region-based remeshing support.
 *
 * Implements the C++ mesh-layer primitives required by the incremental
 * remeshing workflow described in Roadmap §11.8 and §13.8:
 *
 *   §11.8 — "Add region-based remeshing with stitched boundaries between
 *             preserved and regenerated areas."
 *   §13.8 — "Face selection remesh, boundary loop preservation, seam-aware
 *             stitching, local density matching at the border."
 *
 * This header mirrors the Python-layer API in engine/selection_remesh.py but
 * operates on C++ HalfEdgeMesh objects.  The Python layer calls the Python
 * module directly for its own needs; this header exists so that the C++ engine
 * pipeline can perform native incremental remeshing without a Python round-trip.
 *
 * Primary types
 * -------------
 * SelectionMask      — per-face boolean mask identifying the selected region.
 * SubMeshData        — extracted sub-mesh with bidirectional index maps.
 * StitchOptions      — parameters controlling boundary vertex merging.
 * StitchResult       — combined mesh after stitching, with snapping statistics.
 *
 * Primary functions
 * -----------------
 * extract_sub_mesh()        — extract selected faces into a new HalfEdgeMesh.
 * identify_boundary_vertices() — find vertices shared by selected + unselected.
 * build_boundary_feature_edges() — turn boundary loop into feature edge set.
 * stitch_sub_mesh()         — merge the remeshed sub-mesh back into the original.
 * build_selection_mask()    — convenience: bool mask from index list.
 * grow_selection_by_rings() — expand selection outward by N edge rings.
 * shrink_selection_by_rings() — contract selection inward by N edge rings.
 *
 * Stitching algorithm
 * -------------------
 * 1.  Build a BVH over the original boundary vertices.
 * 2.  For each remeshed boundary vertex, snap to the nearest original
 *     boundary vertex within snap_distance (defaults to 2× mean remeshed edge length).
 * 3.  Remove the selected faces from the original mesh.
 * 4.  Compact vertex indices (remove orphaned vertices).
 * 5.  Append snapped remeshed vertices (merged with any original boundary
 *     vertex that was snapped to).
 * 6.  Append remeshed faces with shifted indices.
 * 7.  Return the combined mesh as a new HalfEdgeMesh.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#ifndef QUADFORGE_MESH_MESH_SELECTION_H
#define QUADFORGE_MESH_MESH_SELECTION_H

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "../types.h"
#include "features.h"   // EdgeSet
#include "halfedge.h"
#include "spatial.h"    // TriangleBVH (for boundary snapping)

namespace qf {

// =========================================================================
// SelectionMask
// =========================================================================

/**
 * Per-face boolean mask.  True = face belongs to the selected region.
 *
 * The mask must have length == mesh.num_faces().  Faces with value true
 * form the region to be remeshed; faces with value false are preserved.
 */
using SelectionMask = std::vector<bool>;

/**
 * Build a SelectionMask from a sorted or unsorted list of face indices.
 *
 * @param n_faces        Total number of faces in the mesh.
 * @param face_indices   Indices of selected faces (may be unsorted, may repeat).
 * @return               SelectionMask of length n_faces.
 */
SelectionMask build_selection_mask(int n_faces, const std::vector<int>& face_indices);

/**
 * Invert a SelectionMask (selected → unselected and vice versa).
 */
SelectionMask invert_selection_mask(const SelectionMask& mask);

/**
 * Count the number of selected (true) faces in a mask.
 */
int count_selected_faces(const SelectionMask& mask);

// =========================================================================
// Selection expansion / contraction
// =========================================================================

/**
 * Grow the selection outward by N edge-ring iterations.
 *
 * Each iteration adds every face that shares at least one vertex with any
 * currently-selected face.  This is equivalent to growing by N vertex rings
 * of adjacency.
 *
 * Used to create a "safe margin" around the selected region so that the
 * field constraints at the boundary have enough context.
 *
 * @param mesh   The half-edge mesh.
 * @param mask   Input selection mask (not modified).
 * @param rings  Number of ring-grow iterations (0 = return mask unchanged).
 * @return       Grown selection mask.
 */
SelectionMask grow_selection_by_rings(
    const HalfEdgeMesh& mesh,
    const SelectionMask& mask,
    int rings);

/**
 * Shrink the selection inward by N edge-ring iterations.
 *
 * Each iteration removes every face that has at least one vertex shared
 * with an unselected face.  The result is always a subset of the input mask.
 *
 * Used to ensure a safe interior core for the remesher, excluding faces too
 * close to the selection boundary.
 *
 * @param mesh   The half-edge mesh.
 * @param mask   Input selection mask (not modified).
 * @param rings  Number of ring-shrink iterations.
 * @return       Shrunk selection mask (may be empty if selection is thin).
 */
SelectionMask shrink_selection_by_rings(
    const HalfEdgeMesh& mesh,
    const SelectionMask& mask,
    int rings);

// =========================================================================
// SubMeshData
// =========================================================================

/**
 * Result of extracting a selected face region into a standalone sub-mesh.
 *
 * Mirrors the Python SelectionData class (engine/selection_remesh.py).
 */
struct SubMeshData {
    SubMeshData() = default;

  
    // --- Sub-mesh geometry -------------------------------------------------
    HalfEdgeMesh sub_mesh;              ///< The extracted triangle sub-mesh.

    // --- Bidirectional index maps ------------------------------------------
    std::vector<int> sub_to_orig;       ///< sub_to_orig[sub_v] = original vertex index
    std::vector<int> orig_to_sub;       ///< orig_to_sub[orig_v] = sub vertex index (-1 if absent)

    std::vector<int> sub_face_to_orig;  ///< sub_face_to_orig[sub_fi] = original face index
    std::vector<int> orig_face_to_sub;  ///< orig_face_to_sub[orig_fi] = sub face index (-1 if absent)

    // --- Boundary identification -------------------------------------------
    std::vector<int> boundary_sub_vertices;  ///< Sub-mesh vertex indices on the selection boundary
    std::vector<int> boundary_orig_vertices; ///< Original vertex indices on the selection boundary
    EdgeSet          boundary_edges;         ///< Boundary loop as sorted edge pairs (sub-mesh indices)

    // --- Optional per-vertex attributes (forwarded from original mesh) -----
    std::vector<Vec3>  sub_normals;         ///< Per-vertex normals in sub-mesh space (may be empty)
    std::vector<Vec3>  sub_vertex_colors;   ///< Per-vertex RGB colors (may be empty)
    std::vector<int>   sub_material_ids;    ///< Per-face material IDs for sub-mesh (may be empty)

    // --- Statistics --------------------------------------------------------
    int num_sub_vertices   = 0;
    int num_sub_faces      = 0;
    int num_boundary_verts = 0;
};

/**
 * Extract the selected face region from a mesh into a standalone HalfEdgeMesh.
 *
 * The input mesh must be a triangle mesh (build from triangulate_input first).
 * The sub-mesh shares the same 3-D vertex positions but has its own index space.
 *
 * Pre-conditions checked with assertions:
 *   - mask.size() == mesh.num_faces()
 *   - At least 1 and fewer than num_faces faces are selected
 *     (fully-selected meshes should use the regular remesh path).
 *
 * @param mesh            The original half-edge mesh.
 * @param mask            SelectionMask (length must equal mesh.num_faces()).
 * @param normals         Optional per-vertex normals [mesh.num_vertices()*3] float.
 *                        nullptr = skip normal forwarding.
 * @param vertex_colors   Optional per-vertex RGB colors [mesh.num_vertices()] Vec3.
 *                        nullptr = skip color forwarding.
 * @param material_ids    Optional per-face material IDs [mesh.num_faces()].
 *                        nullptr = skip material forwarding.
 * @return                Populated SubMeshData.
 */
SubMeshData extract_sub_mesh(
    const HalfEdgeMesh&        mesh,
    const SelectionMask&       mask,
    const float*               normals       = nullptr,
    const std::vector<Vec3>*   vertex_colors = nullptr,
    const std::vector<int>*    material_ids  = nullptr);

/**
 * Identify the boundary vertices of a selection: vertices that appear in
 * both selected and unselected faces.
 *
 * @param mesh   The original mesh.
 * @param mask   SelectionMask.
 * @return       Sorted list of original vertex indices on the boundary.
 */
std::vector<int> identify_boundary_vertices(
    const HalfEdgeMesh&  mesh,
    const SelectionMask& mask);

/**
 * Build a feature EdgeSet from the boundary loop of a selection.
 *
 * The resulting EdgeSet can be passed directly to detect_features() /
 * field constraint builders so the cross-field aligns to the selection
 * boundary loop.
 *
 * @param sub_data   Result of extract_sub_mesh().
 * @return           EdgeSet of boundary edges in sub-mesh index space.
 */
EdgeSet build_boundary_feature_edges(const SubMeshData& sub_data);

// =========================================================================
// StitchOptions
// =========================================================================

/**
 * Options controlling how the remeshed sub-mesh is stitched back into the
 * original mesh.
 */
struct StitchOptions {
    /**
     * Maximum distance for snapping a remeshed boundary vertex to its
     * nearest original boundary vertex.
     *
     * 0.0 (default) = auto-compute as 2× the mean remeshed edge length.
     * Set to a positive value to override.
     */
    double snap_distance = 0.0;

    /**
     * If true (default), merge snapped boundary vertices so the resulting
     * mesh has no duplicate vertices at the stitch boundary.
     * If false, keep remeshed boundary vertices as separate points
     * (produces T-junctions but may be useful for debugging).
     */
    bool merge_snapped_vertices = true;

    /**
     * If true, add the stitch boundary edges to the output's feature edge set
     * so downstream smoothing does not round them away.
     * Default: true.
     */
    bool preserve_stitch_boundary_as_features = true;

    /**
     * If the snap fails for more than this fraction of boundary vertices,
     * the stitch is considered failed and stitch_sub_mesh() returns an error.
     * Range: [0, 1].  Default: 0.5.
     */
    double max_snap_failure_ratio = 0.5;
};

// =========================================================================
// StitchResult
// =========================================================================

/**
 * Result of stitching the remeshed sub-mesh back into the original mesh.
 */
struct StitchResult {
    HalfEdgeMesh combined_mesh;   ///< Combined original + remeshed mesh.
    EdgeSet      stitch_edges;    ///< Edges at the stitch boundary (original indices).

    // --- Statistics --------------------------------------------------------
    int    boundary_snapped     = 0;   ///< Boundary verts successfully snapped
    int    boundary_total       = 0;   ///< Total boundary verts in remeshed sub-mesh
    double snap_distance_used   = 0.0; ///< Actual snap distance used
    bool   stitch_ok            = false; ///< True if stitching succeeded

    // --- Error reporting ---------------------------------------------------
    std::string error_message;   ///< Non-empty if stitch_ok == false
};

/**
 * Stitch a remeshed sub-mesh back into the original mesh.
 *
 * The function:
 *   1. Builds a BVH over the original boundary vertices.
 *   2. Snaps each remeshed boundary vertex to the nearest original boundary
 *      vertex within snap_distance.
 *   3. Removes the selected faces from the original mesh.
 *   4. Compacts vertex indices.
 *   5. Appends remeshed vertices (merged where snapped).
 *   6. Appends remeshed faces with index-shifted references.
 *   7. Returns the combined HalfEdgeMesh.
 *
 * @param original_mesh  The original triangle mesh.
 * @param mask           SelectionMask used to create sub_data.
 * @param sub_data       SubMeshData from extract_sub_mesh().
 * @param remeshed_verts Output vertex positions from the remesher [V_rem, 3].
 * @param remeshed_faces Output face index lists from the remesher.
 *                       Faces may be quads (4 indices) or triangles (3 indices).
 * @param opts           Stitching options.
 * @return               StitchResult (check stitch_ok before using combined_mesh).
 */
StitchResult stitch_sub_mesh(
    const HalfEdgeMesh&                  original_mesh,
    const SelectionMask&                 mask,
    const SubMeshData&                   sub_data,
    const Eigen::MatrixXd&               remeshed_verts,
    const std::vector<std::vector<int>>& remeshed_faces,
    const StitchOptions&                 opts = {});

// =========================================================================
// Utility: boundary vertex detection in a standalone polygon soup
// =========================================================================

/**
 * Find the vertex indices on the free boundary of a polygon mesh
 * (vertices incident on edges referenced by exactly one face).
 *
 * @param n_verts      Total number of vertices.
 * @param faces        Face connectivity (each sub-vector is a polygon).
 * @return             Sorted vector of boundary vertex indices.
 */
std::vector<int> find_free_boundary_vertices(
    int n_verts,
    const std::vector<std::vector<int>>& faces);

/**
 * Compute the mean edge length of a polygon mesh given its vertex positions
 * and face connectivity.
 *
 * @param verts  Vertex positions [n_verts, 3].
 * @param faces  Face connectivity (each sub-vector is a polygon).
 * @return       Mean edge length across all unique edges.
 */
double mean_edge_length(
    const Eigen::MatrixXd&               verts,
    const std::vector<std::vector<int>>& faces);

} // namespace qf

#endif // QUADFORGE_MESH_MESH_SELECTION_H
