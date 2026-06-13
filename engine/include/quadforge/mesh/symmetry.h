#pragma once
/**
 * quadforge/mesh/symmetry.h — Reflective Symmetry Detection & Analysis
 *
 * Implements Stage 1, Step 1.4 of the QuadForge pipeline as described in
 * the roadmap (section 3.1):
 *
 *   "When any symmetry axis is enabled, analyse the mesh for reflective
 *    symmetry across the corresponding plane (in the object's local coordinate
 *    system).  The detection algorithm samples vertex pairs across the candidate
 *    plane, builds a correspondence map using a BVH nearest-neighbor query,
 *    and scores the symmetry quality.  If the score exceeds a threshold
 *    (>95% vertex correspondence within tolerance), the symmetry is enforced
 *    in subsequent stages by constraining the cross-field and parametrization
 *    to be symmetric."
 *
 * This module sits in the mesh/ layer because:
 *   - It operates on HalfEdgeMesh geometry and TriangleBVH queries.
 *   - Its output (SymmetryData) is consumed by field/constraints.cpp to
 *     build symmetry-plane field constraints, and by param/ to mirror UV
 *     coordinates across the symmetry plane.
 *   - It has no dependency on the field or param layers.
 *
 * Usage:
 *
 *   // Detect X-axis symmetry
 *   SymmetryDetectionOptions opts;
 *   SymmetryResult result = detect_reflective_symmetry(
 *       mesh, bvh, SymmetryAxis::X, opts);
 *
 *   if (result.detected) {
 *       // Use result.vertex_pairs for constraint building
 *       // Use result.score as a confidence measure
 *   }
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#ifndef QUADFORGE_MESH_SYMMETRY_H
#define QUADFORGE_MESH_SYMMETRY_H

#include <vector>
#include <utility>
#include <cstdint>
#include <array>

#include "../types.h"
#include "halfedge.h"
#include "spatial.h"   // TriangleBVH

namespace qf {

// =========================================================================
// Enumerations
// =========================================================================

/**
 * Which Cartesian symmetry plane to test.
 *
 * X → plane x=0 (left/right mirror, normal = [1,0,0])
 * Y → plane y=0 (front/back mirror, normal = [0,1,0])
 * Z → plane z=0 (top/bottom mirror, normal = [0,0,1])
 *
 * All planes pass through the origin in the mesh's local coordinate system,
 * matching QuadRemesher's SymLocal=1 convention.
 */
enum class SymmetryAxis : uint8_t {
    X = 0,
    Y = 1,
    Z = 2,
};

// =========================================================================
// Options
// =========================================================================

/**
 * Parameters controlling the symmetry detection pass.
 */
struct SymmetryDetectionOptions {
    /**
     * Maximum distance (in mesh units) between a vertex and its proposed
     * symmetric counterpart.  Vertices within this tolerance are accepted
     * as a valid pair.
     *
     * 0.0 = auto: computed as (0.5% of the bounding-box diagonal), which
     * is a reliable default for meshes at Blender's default scale.
     */
    double tolerance = 0.0;

    /**
     * Minimum fraction of vertices that must have a valid symmetric partner
     * for the mesh to be considered symmetric on that axis.
     *
     * Default 0.95 → 95% correspondence required (roadmap threshold).
     */
    double detection_threshold = 0.95;

    /**
     * Maximum number of vertices to sample when computing the symmetry
     * score.  0 = use all vertices.
     *
     * Sampling is useful for large meshes (>500K vertices) where testing
     * every vertex is expensive.  A uniform random sample of 10K–50K
     * vertices gives a score accurate to within ~1%.
     */
    int max_sample_vertices = 0;

    /**
     * If true, build the full vertex-pair correspondence map
     * (SymmetryResult::vertex_pairs).  This is needed by the parametrization
     * stage to mirror UV coordinates, but costs extra memory.
     *
     * If false, only the symmetry score is computed (faster).
     */
    bool build_correspondence = true;

    /**
     * If true, include vertices that lie ON the symmetry plane (distance to
     * plane < plane_tolerance) as self-paired entries in vertex_pairs:
     *   { v, v }
     * This marks the symmetry seam explicitly for the parametrization stage.
     */
    bool include_plane_vertices = true;

    /**
     * Distance threshold for classifying a vertex as lying on the symmetry
     * plane itself (rather than on one side).  0.0 = auto (1/10 of tolerance).
     */
    double plane_tolerance = 0.0;
};

// =========================================================================
// Results
// =========================================================================

/**
 * Output of detect_reflective_symmetry().
 */
struct SymmetryResult {
    /** True when score >= opts.detection_threshold. */
    bool detected = false;

    /**
     * Fraction of sampled vertices with a valid symmetric partner ∈ [0, 1].
     *
     * 1.0 = perfectly symmetric mesh (every vertex has an exact mirror).
     * 0.0 = completely asymmetric.
     */
    float score = 0.f;

    /**
     * Which axis was tested.  Populated even when detected == false.
     */
    SymmetryAxis axis = SymmetryAxis::X;

    /**
     * The actual tolerance used (may differ from opts.tolerance if auto).
     */
    double tolerance_used = 0.0;

    /**
     * Unit normal of the symmetry plane (pointing in the positive axis direction).
     * e.g. for SymmetryAxis::X → {1,0,0}.
     */
    std::array<double, 3> plane_normal{1.0, 0.0, 0.0};

    /**
     * Vertex correspondence pairs: (v_source, v_mirror).
     *
     * Only populated when opts.build_correspondence == true.
     *
     * Each entry (i, j) means vertex i reflects to vertex j across the
     * symmetry plane.  For vertices ON the symmetry plane: i == j.
     *
     * Ordering: each undirected pair appears exactly once; by convention
     * the vertex with the smaller index in the positive-half-space appears
     * first.
     */
    std::vector<std::pair<int, int>> vertex_pairs;

    /**
     * Per-vertex mirror index: mirror_map[v] = v' (the mirror vertex of v),
     * or -1 if vertex v has no valid mirror partner.
     *
     * Only populated when opts.build_correspondence == true.
     * Size = mesh.num_vertices().
     */
    std::vector<int> mirror_map;

    /**
     * Vertices that lie exactly ON the symmetry plane (distance < plane_tolerance).
     * Only populated when opts.include_plane_vertices == true.
     */
    std::vector<int> plane_vertices;
};

// =========================================================================
// Core detection function
// =========================================================================

/**
 * Detect and score reflective symmetry of a mesh on a given Cartesian axis.
 *
 * Algorithm (roadmap §3.1 Step 1.4):
 *   1. For each (sampled) vertex v, reflect its position across the axis
 *      plane to get v_reflected.
 *   2. Query the BVH for the nearest surface point to v_reflected.
 *   3. If the nearest vertex (found by querying the reference BVH and then
 *      looking up which mesh vertex is closest to the BVH hit point) is within
 *      opts.tolerance, accept the pair (v, v_mirror).
 *   4. score = accepted / total_sampled.
 *   5. If score >= opts.detection_threshold, set detected = true.
 *
 * The BVH is queried on the REFLECTED position — so we test whether the mesh
 * looks the same when flipped across the plane, which is the correct definition
 * of reflective symmetry.
 *
 * @param mesh   Input triangle mesh (must be valid, with correct positions)
 * @param bvh    BVH built from the same mesh (for closest-point queries)
 * @param axis   Which Cartesian symmetry axis to test
 * @param opts   Detection options
 * @return       Symmetry analysis result
 */
SymmetryResult detect_reflective_symmetry(
    const HalfEdgeMesh&              mesh,
    const TriangleBVH&               bvh,
    SymmetryAxis                     axis,
    const SymmetryDetectionOptions&  opts = {}
);

// =========================================================================
// Multi-axis convenience
// =========================================================================

/**
 * Run detect_reflective_symmetry for each axis in the provided set and
 * return one SymmetryResult per axis (in order X, Y, Z).
 *
 * This is the preferred entry point for engine.cpp, which tests all three
 * axes when any symmetry flag is set in QFParams.
 *
 * @param mesh         Input mesh
 * @param bvh          BVH for the same mesh
 * @param test_x/y/z   Which axes to test (axes not tested return score=0)
 * @param opts         Options (same for all axes)
 * @return             Array of 3 results [X, Y, Z]
 */
std::array<SymmetryResult, 3> detect_all_symmetry_axes(
    const HalfEdgeMesh&              mesh,
    const TriangleBVH&               bvh,
    bool                             test_x,
    bool                             test_y,
    bool                             test_z,
    const SymmetryDetectionOptions&  opts = {}
);

// =========================================================================
// Utility helpers
// =========================================================================

/**
 * Reflect a 3-D point across a Cartesian axis plane.
 *
 * SymmetryAxis::X → negate the X component (plane x=0).
 * SymmetryAxis::Y → negate the Y component (plane y=0).
 * SymmetryAxis::Z → negate the Z component (plane z=0).
 */
Vec3 reflect_point(const Vec3& p, SymmetryAxis axis);

/**
 * Return the signed distance from point p to the symmetry plane for `axis`.
 *
 * Positive = on the positive-half-space side.
 * Negative = on the negative-half-space side.
 * Zero     = on the plane.
 */
double signed_plane_distance(const Vec3& p, SymmetryAxis axis);

/**
 * Compute the bounding-box diagonal of a HalfEdgeMesh.
 *
 * Used to auto-scale the detection tolerance when opts.tolerance == 0.
 */
double bounding_box_diagonal(const HalfEdgeMesh& mesh);

/**
 * Build a flat float position array [V*3] for TriangleBVH construction
 * directly from a HalfEdgeMesh, avoiding a round-trip through Eigen.
 *
 * Caller can use this together with build_face_index_array() to construct
 * a TriangleBVH from a HalfEdgeMesh without going through the C API layer.
 */
std::vector<float> build_float_position_array(const HalfEdgeMesh& mesh);

/**
 * Return the unit plane normal for a SymmetryAxis.
 *
 * X → {1,0,0}, Y → {0,1,0}, Z → {0,0,1}.
 */
std::array<float, 3> axis_plane_normal(SymmetryAxis axis);

} // namespace qf

#endif // QUADFORGE_MESH_SYMMETRY_H
