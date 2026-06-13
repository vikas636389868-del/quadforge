#pragma once
/**
 * quadforge/mesh/repair.h  —  Mesh Repair & Sanitization Layer
 *
 * Implements the pre-flight repair stage described in the QuadForge roadmap
 * sections 11.2 (Mesh Repair & Sanitization Layer) and 13.7 (Extreme Mesh
 * Handling System).
 *
 * The repair pass runs before any geometry processing.  It makes the input
 * mesh as clean as possible and returns a RepairReport that the Python UI
 * can display as a confidence score and warning list.
 *
 * Repair stages (applied in order):
 *   1. remove_zero_area_faces()        — degenerate triangle removal
 *   2. remove_duplicate_vertices()     — adaptive vertex welding
 *   3. repair_winding()                — consistent face orientation
 *   4. split_non_manifold_vertices()   — topological vertex de-coupling
 *   5. detect_self_intersections()     — intersection cataloging
 *   6. isolate_self_intersecting_regions() — intersection isolation
 *
 * All routines that modify the mesh return a rebuilt HalfEdgeMesh
 * (move-assigned in-place) so callers hold a clean, valid structure.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#ifndef QUADFORGE_MESH_REPAIR_H
#define QUADFORGE_MESH_REPAIR_H

#include <vector>
#include <string>
#include <utility>
#include <cstdint>

#include "../types.h"
#include "halfedge.h"

namespace qf {

// =========================================================================
// Configuration
// =========================================================================

/**
 * Options for the mesh repair pass.
 *
 * All tolerances are in the same unit as the mesh vertex coordinates
 * (usually Blender's metres).
 */
struct MeshRepairOptions {
    // --- Weld / duplicate-vertex removal ---
    double weld_tolerance          = 1e-5;   ///< Vertices closer than this are merged
    bool   adaptive_weld           = true;   ///< Scale tolerance by local avg edge len

    // --- Zero-area face removal ---
    double area_threshold          = 1e-12;  ///< Faces with area < this are removed

    // --- Winding repair ---
    bool   repair_winding          = true;   ///< Re-orient inconsistent windings

    // --- Non-manifold vertex splitting ---
    bool   split_nm_vertices       = true;   ///< Duplicate verts to make fan manifold

    // --- Self-intersection ---
    bool   detect_self_intersect   = true;   ///< Scan for triangle-triangle overlaps
    bool   isolate_self_intersect  = true;   ///< Remove self-intersecting triangles
    int    max_si_candidates       = 64;     ///< BVH expansion limit per query

    // --- UV seam / vertex attribute preservation (roadmap §11.2) ---
    //
    // When non-null, vertex_split_ids[v] is an integer "split tag" for vertex v.
    // Two candidate vertices at the same 3-D position are only welded together
    // if their split tags are EQUAL.  This prevents remove_duplicate_vertices()
    // from merging UV-seam-split vertices (same world position, different UVs)
    // or vertices belonging to different hard-edge crease groups.
    //
    // Compute these tags with compute_vertex_split_ids_from_uv() before calling
    // repair_mesh(), then pass the resulting vector's data() pointer here.
    //
    // Set to nullptr (default) to disable attribute-aware welding and weld
    // purely by position (legacy behaviour).
    bool           preserve_uv_seams  = true;   ///< Gate: only check split_ids when true
    const int32_t* vertex_split_ids   = nullptr; ///< [num_vertices] split tag per vertex

    // --- Stage enable flags ---
    bool   run_zero_area           = true;
    bool   run_weld                = true;
    bool   run_winding             = true;
    bool   run_nm_vertex_split     = true;
    bool   run_self_intersect      = true;
};

// =========================================================================
// Results
// =========================================================================

/**
 * Per-stage counts and overall confidence score returned after repair.
 * The Python UI uses these to generate user-readable warning messages.
 */
struct RepairReport {
    // --- Stage outcome counts ---
    int   zero_area_removed        = 0;  ///< Degenerate faces removed
    int   duplicate_verts_merged   = 0;  ///< Vertices welded together
    int   winding_flips            = 0;  ///< Faces whose winding was reversed
    int   nm_vertex_splits         = 0;  ///< Non-manifold vertices split
    int   self_intersect_pairs     = 0;  ///< Triangle pairs found to intersect
    int   si_faces_removed         = 0;  ///< Faces removed to isolate intersections

    // --- Overall state after repair ---
    int   vertices_before          = 0;
    int   faces_before             = 0;
    int   vertices_after           = 0;
    int   faces_after              = 0;
    bool  is_manifold_after        = false;
    bool  is_closed_after          = false;

    /**
     * Scalar confidence in [0, 100].
     *
     * 100 = pristine mesh, no repairs needed.
     * 0   = too damaged to produce a good remesh result.
     *
     * Computed by confidence_score().
     */
    float confidence               = 100.f;

    /** Plain-English summary for the UI tooltip / console. */
    std::string summary;

    /** Compute the confidence score from the stage counts. */
    void compute_confidence(int original_face_count);

    /** Build a human-readable summary string. */
    std::string to_string() const;
};

// =========================================================================
// Individual repair operations
// =========================================================================

/**
 * Remove triangles whose area is below opts.area_threshold.
 *
 * These degenerate faces cause division-by-zero in curvature and
 * connection-Laplacian assembly.  Must run first, before welding.
 *
 * @return Number of faces removed.
 */
int remove_zero_area_faces(HalfEdgeMesh& mesh,
                           const MeshRepairOptions& opts = {});

/**
 * Weld duplicate (or near-duplicate) vertices within opts.weld_tolerance.
 *
 * When opts.adaptive_weld is true the tolerance is scaled by a fraction
 * of the mean edge length computed from a sample of the mesh, so large
 * meshes in centimetres and small meshes in millimetres both work.
 *
 * Algorithm: spatial grid hashing O(N) amortised.
 *
 * @return Number of vertices merged (= original_count - new_count).
 */
int remove_duplicate_vertices(HalfEdgeMesh& mesh,
                              const MeshRepairOptions& opts = {});

/**
 * Make all face windings consistently oriented via BFS flood-fill.
 *
 * Starting from a seed face, adjacent faces are flipped if their normals
 * point away from the majority direction.  Works on open and closed meshes.
 * Does NOT guarantee outward-pointing normals — only consistency.
 *
 * @return Number of faces flipped.
 */
int repair_winding(HalfEdgeMesh& mesh,
                   const MeshRepairOptions& opts = {});

/**
 * Split non-manifold vertices into separate manifold copies.
 *
 * A non-manifold vertex is one where the incident face fan consists of
 * more than one connected component (i.e. the ring walk terminates before
 * visiting all incident faces).  For each extra component, a new vertex is
 * created at the same position and the component's faces are re-wired to it.
 *
 * This operation is conservative — it may produce isolated edge seams at
 * the split, but the resulting mesh is locally manifold.
 *
 * @return Number of vertices split (each split increases V by 1).
 */
int split_non_manifold_vertices(HalfEdgeMesh& mesh,
                                const MeshRepairOptions& opts = {});

/**
 * Detect triangle-triangle self-intersections using a BVH acceleration
 * structure (centroid k-d tree + AABB overlap + Möller–Trumbore test).
 *
 * @return List of intersecting face-index pairs (fi, fj) with fi < fj.
 */
std::vector<std::pair<int,int>> detect_self_intersections(
    const HalfEdgeMesh& mesh,
    const MeshRepairOptions& opts = {});

/**
 * Remove all triangles involved in detected self-intersections.
 *
 * This is a conservative isolation strategy — it may over-remove, but it
 * guarantees that the surviving mesh has no self-intersection among its
 * triangles.  Small holes left by removal are NOT filled here; the
 * downstream pipeline handles them.
 *
 * @param si_pairs  Output of detect_self_intersections(); if empty,
 *                  the function runs detection internally first.
 * @return Number of faces removed.
 */
int isolate_self_intersecting_regions(
    HalfEdgeMesh& mesh,
    const std::vector<std::pair<int,int>>& si_pairs = {},
    const MeshRepairOptions& opts = {});

// =========================================================================
// UV seam / vertex attribute utilities  (roadmap §11.2)
// =========================================================================

/**
 * Compute a per-vertex "split ID" array suitable for use as
 * MeshRepairOptions::vertex_split_ids.
 *
 * Two vertices that share the same 3-D position but belong to different UV
 * seam islands are given DIFFERENT split IDs, preventing them from being
 * welded together by remove_duplicate_vertices().
 *
 * Algorithm:
 *   For each UV loop (uv_indices[face * uv_face_stride + local_vertex]),
 *   assign the minimum UV-loop index seen at each vertex position as its
 *   canonical "split tag".  Vertices at the same position but with at least
 *   one differing UV index get distinct canonical tags.
 *
 * @param num_vertices      Number of mesh vertices.
 * @param num_faces         Number of triangulated faces.
 * @param uv_indices        Per-loop UV index array [num_faces * uv_face_stride].
 *                          May be null — if null, returns a uniform vector of 0s.
 * @param uv_face_stride    UV indices per face (always 3 for triangulated input).
 * @param face_vertices     [num_faces * 3] triangle vertex indices.
 * @return                  Vector of length num_vertices; use .data() as
 *                          MeshRepairOptions::vertex_split_ids.
 */
std::vector<int32_t> compute_vertex_split_ids_from_uv(
    int            num_vertices,
    int            num_faces,
    const int32_t* uv_indices,
    int            uv_face_stride,
    const int32_t* face_vertices);

// =========================================================================
// Main entry point
// =========================================================================

/**
 * Run the full repair pipeline in the correct order and return a report.
 *
 * Stages are gated by opts.run_* flags so callers can skip stages they
 * do not need.
 *
 * The mesh is repaired IN PLACE.  On return it is the best approximation
 * to a clean, manifold triangle mesh that this module can produce from
 * the input.
 *
 * @param mesh   Input/output mesh — rebuilt in-place after each stage.
 * @param opts   Repair options.
 * @return       Report with per-stage counts, confidence score, and summary.
 */
RepairReport repair_mesh(HalfEdgeMesh& mesh,
                         const MeshRepairOptions& opts = {});

} // namespace qf

#endif // QUADFORGE_MESH_REPAIR_H
