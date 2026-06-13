#pragma once
/* quadforge/field/field_debug.h — Cross-field visual debugging utilities.
 *
 * Roadmap Phase 2 deliverable:
 *   "Visual debugging output: export cross-field as per-face direction vectors
 *    for visualization in Blender (as a temporary mesh with short edges showing
 *    the field directions)."
 *
 * This module provides two export functions:
 *
 * 1. export_cross_field_vectors()
 *    Produces a flat array of per-face direction vectors (two perpendicular
 *    directions per face, i.e. the four arms of each cross collapsed to two).
 *    Callers can build a Blender "custom_normals" or line-set object from these.
 *
 * 2. export_cross_field_mesh()
 *    Produces a debug stick-mesh: for every face centroid, four short line
 *    segments (the four arms of the cross) are emitted as a pair of triangles
 *    degenerated to sticks.  This can be imported into Blender directly as a
 *    mesh object for visual inspection.
 *
 * 3. export_singularities_as_points()
 *    Returns per-singularity positions and indices for use as Blender empties
 *    or vertex-coloured point cloud.
 *
 * 4. export_field_quality_map()
 *    Returns per-face field quality scores (0–1) for a heat-map overlay.
 *    Quality is 1.0 when all four neighbours agree (smooth), 0.0 when there
 *    is maximum disagreement (near singularity or degenerate region).
 *
 * None of these functions are on the critical remesh path.  They are always
 * compiled (no QUADFORGE_ENABLE_DEBUG_EXPORTS gate — the earlier claim was
 * incorrect; field_debug.cpp has no such ifdef and neither does this header).
 * They are called from Python through the optional C API entry point
 * qf_debug_field(), which is itself gated by the "Show Field Debug" UI flag.
 *
 * BUG FIX (v95): Corrected the erroneous comment that claimed these functions
 * were only compiled when QUADFORGE_ENABLE_DEBUG_EXPORTS is defined.  No such
 * preprocessor guard exists anywhere in field_debug.h or field_debug.cpp.
 * The old comment was misleading: a reader seeing it might add
 *   #ifdef QUADFORGE_ENABLE_DEBUG_EXPORTS ... #endif
 * around a call site, silently dropping the debug output in every build that
 * doesn't define that macro (i.e. all normal builds).
 */

#ifndef QUADFORGE_FIELD_DEBUG_H
#define QUADFORGE_FIELD_DEBUG_H

#include <vector>
#include <array>
#include <string>
#include "../types.h"
#include "../mesh/halfedge.h"
#include "cross_field.h"
#include "singularity.h"

namespace qf {

// -----------------------------------------------------------------------
// Per-face direction vector export
// -----------------------------------------------------------------------

/**
 * Export the cross-field as per-face direction pairs.
 *
 * For each face f the function emits two unit 3-D vectors:
 *   dirs[f*6 + 0..2] = first  arm direction (the "U" direction)
 *   dirs[f*6 + 3..5] = second arm direction (the "V" direction, = U rotated 90°)
 *
 * The vectors lie in the face's tangent plane and are expressed in world space.
 * Output length: nf * 6 floats.
 *
 * @param mesh      The triangle mesh (read-only)
 * @param field     The computed cross-field
 * @param arm_scale Scale factor for the arm vectors (default 1.0 = unit length)
 * @return          Flat float array of size nf * 6
 */
std::vector<float> export_cross_field_vectors(
    const HalfEdgeMesh& mesh,
    const CrossField&   field,
    float               arm_scale = 1.0f
);

// -----------------------------------------------------------------------
// Stick-mesh export
// -----------------------------------------------------------------------

/**
 * Debug stick-mesh: every face centroid sprouts four short line-sticks,
 * one per arm of the cross.  Returned as a flat vertex/edge list suitable
 * for constructing a Blender mesh via mesh.from_pydata().
 *
 * Layout:
 *   vertices: flat [nf * 5 * 3] floats — for each face:
 *               centroid (1 point) + 4 arm-tip points
 *               (centroid ± arm_len * dir for each of the two axes → 4 tips)
 *   edges:    flat [nf * 4] int32 — for each face, 2 edges (4 indices) forming
 *               a "+" cross: one edge connects -u tip to +u tip, another
 *               connects -v tip to +v tip.
 *
 *   NOTE: Buffer sizes corrected in v56 — previous header said [nf*8*3] verts
 *   and [nf*4*2] edges, both of which were wrong and would cause callers to
 *   over-allocate or read past the actual data.
 *
 * @param mesh          The triangle mesh
 * @param field         The computed cross-field
 * @param arm_len       Arm half-length in world units (default: 0 = auto,
 *                      set to avg_edge_length / 2)
 * @param out_verts     Output vertex positions [N*3]
 * @param out_edges     Output edge pairs [M*2]
 */
void export_cross_field_mesh(
    const HalfEdgeMesh&     mesh,
    const CrossField&       field,
    float                   arm_len,
    std::vector<float>&     out_verts,
    std::vector<int32_t>&   out_edges
);

// -----------------------------------------------------------------------
// Singularity point export
// -----------------------------------------------------------------------

/**
 * Export detected singularities as a point cloud.
 *
 * @param sings         Singularity list from detect_singularities()
 * @param out_positions Output positions [N*3]
 * @param out_indices   Output singularity indices (+0.25 or -0.25) [N]
 */
void export_singularities_as_points(
    const std::vector<SingularityInfo>& sings,
    std::vector<float>&                 out_positions,
    std::vector<float>&                 out_indices
);

// -----------------------------------------------------------------------
// Per-face quality heat-map
// -----------------------------------------------------------------------

/**
 * Compute a per-face field quality score in [0, 1].
 *
 * For each face, the score is based on the magnitude of the weighted average
 * of its parallel-transported neighbour fields:
 *   score(f) = |Σ_j  w_ij * conj(r_ij) * u_j| / Σ_j w_ij
 *
 * score = 1.0 means all neighbours agree perfectly (smooth region).
 * score ≈ 0.0 means the field is highly incoherent (near a singularity or
 * degenerate triangle cluster).
 *
 * Output: flat float array of size nf, values in [0, 1].
 */
std::vector<float> export_field_quality_map(
    const HalfEdgeMesh& mesh,
    const CrossField&   field
);

// -----------------------------------------------------------------------
// Text / diagnostic summary
// -----------------------------------------------------------------------

/**
 * Return a human-readable summary of the cross-field:
 *   - total faces
 *   - number of singularities (+/-)
 *   - mean field quality score
 *   - min field quality score (identifies worst region)
 *   - convergence residual (if available in field metadata)
 */
std::string cross_field_summary(
    const HalfEdgeMesh&                  mesh,
    const CrossField&                    field,
    const std::vector<SingularityInfo>&  sings
);

} // namespace qf

#endif // QUADFORGE_FIELD_DEBUG_H
