#pragma once
/* quadforge/extract/motorcycle.h — Motorcycle graph for T-junction resolution
 * and surface patch decomposition.
 *
 * This header exposes two levels of the motorcycle graph algorithm:
 *
 *  1.  resolve_t_junctions()     — Post-extraction cleanup (merge triangle
 *      pairs into quads; detect and split geometric T-junctions).  Used by
 *      ALL extraction methods (0, 1, 2) after their primary extraction step.
 *
 *  2.  build_motorcycle_graph()  — Low-level rider-tracing algorithm.  Seeds
 *      riders at singularities and boundaries, advances them across the mesh
 *      following the cross-field direction, and returns the raw track network
 *      as a PatchLayout.  Used internally by extract_quads_motorcycle()
 *      (extraction_method=1) and exposed here for testing and visualisation.
 *
 * References:
 *   Eppstein et al. (2008) "Motorcycle Graphs: Canonical Quad Mesh Partitioning"
 *   Bommes et al. (2013) "Integer-Grid Maps for Reliable Quad Meshing," §3
 */

#ifndef QUADFORGE_EXTRACT_MOTORCYCLE_H
#define QUADFORGE_EXTRACT_MOTORCYCLE_H

#include "../types.h"
#include "../mesh/halfedge.h"
#include "../field/singularity.h"
#include "../field/cross_field.h"
#include "patch_layout.h"

namespace qf {

/* ======================================================================
 * resolve_t_junctions — post-extraction cleanup (all methods)
 * ====================================================================== */

/**
 * Resolve T-junctions in a freshly extracted quad mesh.
 *
 * Pass 1: Merge adjacent triangle pairs into quads (directed-edge algorithm,
 *         preserves manifold orientation).
 * Pass 2: Geometric T-junction detection — tests each vertex against
 *         neighbouring quad edges via point_on_segment(), splits offending
 *         quads into a triangle + quad.
 * Pass 3: Update quad_percentage and avg_valence statistics.
 *
 * @param mesh          Original triangle mesh (field direction reference)
 * @param quad_mesh     Freshly extracted quad mesh (may contain tris)
 * @param singularities Singularity list.  Currently unused inside this
 *                      function (kept for API stability and future use).
 *                      BUG FIX (v91): made optional with a default empty
 *                      vector so call sites that have no singularity data
 *                      do not need to construct a dummy container.
 * @return              Cleaned quad mesh with T-junctions resolved
 */
QuadMesh resolve_t_junctions(
    const HalfEdgeMesh&                     mesh,
    const QuadMesh&                         quad_mesh,
    const std::vector<SingularityInfo>&     singularities = {}
);


/* ======================================================================
 * build_motorcycle_graph — low-level rider tracing
 * ====================================================================== */

/**
 * Seed and advance motorcycle riders, returning the raw track network as
 * a PatchLayout.
 *
 * This function is the core of extraction_method=1.  It does NOT produce
 * a quad mesh; that step is done by extract_quads_motorcycle() in
 * patch_layout.h which calls this function to obtain the decomposition and
 * then runs per-patch parametrization + isoline extraction.
 *
 * Algorithm:
 *  1. For each +¼ singularity: seed 1 rider.
 *     For each -¼ singularity: seed 3 riders (one per branch).
 *     For each boundary corner (valence-1 boundary vertex): seed 1 rider.
 *
 *  2. For each active rider, determine the "dominant direction" from the
 *     cross-field face_field at the current triangle: the field direction
 *     most aligned with the gradient of U (or V, depending on orientation).
 *
 *  3. Advance the rider across the next triangle edge in the dominant
 *     direction.  Append the crossed edge to the rider's track.
 *
 *  4. Halt the rider when it hits:
 *       a. Another rider's track (T-junction resolution).
 *       b. A mesh boundary.
 *       c. A singularity other than its source.
 *       d. Its own starting face (loop completion for closed-surface fields).
 *       e. max_track_steps exceeded (safety cap).
 *
 *  5. Convert the halted track network to PatchLayout vertices, edges,
 *     and faces.
 *
 * @param mesh              Triangle mesh
 * @param field             Computed 4-RoSy cross-field
 * @param uv                UV parametrization (for dominant direction test)
 * @param singularities     Detected and relocated singularities
 * @param max_track_steps   Safety cap per rider.  Pass -1 (the default) to
 *                          use the automatic limit of 10 × vertex count.
 *                          Pass a positive integer to override.  Zero or
 *                          values other than -1 that are ≤ 0 are treated as
 *                          -1 (auto).  Do NOT pass -1 expecting it to mean
 *                          "unlimited" — it means auto-computed from nv.
 * @param num_threads       Thread count for parallel rider advancement
 * @return                  PatchLayout (faces may be empty if no singularities)
 */
PatchLayout build_motorcycle_graph(
    const HalfEdgeMesh&                     mesh,
    const CrossField&                       field,
    const UVParam&                          uv,
    const std::vector<SingularityInfo>&     singularities,
    int                                     max_track_steps = -1,
    int                                     num_threads     = 0
);

} // namespace qf

#endif // QUADFORGE_EXTRACT_MOTORCYCLE_H
