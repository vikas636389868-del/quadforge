#pragma once
/* quadforge/extract/edge_flow_refine.h — Edge flow refinement pass.
 *
 * Roadmap reference: §13.10 "Edge Flow Refinement Pass"
 *
 * PURPOSE
 * -------
 * After the main extraction pipeline (iso-line tracing or dual contouring)
 * and topology cleanup, the resulting quad mesh is mathematically valid but
 * may contain:
 *
 *   • "Awkward turn" vertices — interior vertices where adjacent quad edges
 *     make a tight acute angle instead of flowing smoothly.
 *
 *   • "Unnecessary irregularity clusters" — groups of 3+ adjacent irregular
 *     (valence ≠ 4) vertices that could be resolved by a local edge rotation
 *     without changing the mesh topology.
 *
 *   • "Poor loop segments" — chains of quads where the edge flow deviates
 *     significantly from the dominant principal direction in that region,
 *     visible as zigzag or pinched edge loops.
 *
 * This pass performs three targeted local operations:
 *
 *   1. Loop straightening — detect zigzag edge sequences (consecutive quads
 *      whose shared edge alternates direction by > threshold) and reroute
 *      the loop by rotating the offending quad edges.
 *
 *   2. Irregularity cluster reduction — for each pair of adjacent irregular
 *      vertices of opposite sign (+1/−1 deviation from valence 4), attempt a
 *      "diagonal swap" to cancel the pair.
 *
 *   3. Angle-based vertex relocation — relocate interior vertices toward the
 *      average of their neighbours when the minimum quad angle at that vertex
 *      is below a threshold, using a single Laplacian step projected back
 *      onto the local tangent plane.
 *
 * The pass is applied after cleanup_quad_mesh() and before the
 * postprocess (smoothing / projection) stages, so the refined mesh still
 * gets the full Taubin smoothing + BVH reprojection treatment.
 *
 * REFERENCES
 * ----------
 * Bommes et al. (2013) "Integer-Grid Maps for Reliable Quad Meshing," §5.3
 *   (local optimization post-extraction).
 * Tarini et al. (2011) "Simple Quad Domains for Field Aligned Mesh
 *   Parametrization," ACM TOG — loop-tracing ideas.
 */

#ifndef QUADFORGE_EXTRACT_EDGE_FLOW_REFINE_H
#define QUADFORGE_EXTRACT_EDGE_FLOW_REFINE_H

#include "../types.h"

namespace qf {

/**
 * Parameters controlling the edge flow refinement pass.
 */
struct EdgeFlowRefineParams {
    /**
     * Minimum acceptable quad angle in degrees.
     * Interior vertices whose incident minimum angle falls below this
     * threshold are relocated via a single Laplacian step.
     * Default: 25.0°   (improves worst-case angle without aggressive smoothing)
     */
    double min_angle_deg = 25.0;

    /**
     * Irregularity cancellation: maximum geodesic distance (in edges) between
     * an opposite-sign irregular vertex pair for them to be considered
     * "adjacent" in the cancellation heuristic.
     * Default: 2   (only directly adjacent pairs and one-step-removed pairs)
     */
    int max_cancel_dist = 2;

    /**
     * Loop-straightening cosine threshold.
     * A quad edge sequence is "zigzag" if the cosine of the angle between
     * consecutive edge directions falls below this threshold.
     * Default: -0.3   (≈ 107° — sharper turns than this are targeted)
     */
    double zigzag_cos_thresh = -0.3;

    /**
     * Maximum number of refinement passes.  Each pass visits every vertex /
     * edge once.  Two passes usually converge for typical production meshes.
     * Default: 2
     */
    int max_passes = 2;

    /**
     * Laplacian relocation strength for angle-based vertex updates.
     * 1.0 = full move to neighbour average.  Smaller values are more
     * conservative.
     *
     * Valid range: [0.0, 0.5].
     *   • Values in (0.5, 1.0] cause oscillation (the vertex overshoots
     *     and is corrected back across multiple passes, degrading quality).
     *   • Values > 1.0 cause divergence (vertex moves past the neighbour
     *     average, growing the error each pass).
     *   • Values < 0.0 are not meaningful (negative relocation).
     * The implementation ASSERTS this range in debug builds.
     *
     * Default: 0.3
     */
    double relocation_strength = 0.3;

    /**
     * If true, skip operations on vertices that are within one edge of a
     * detected feature curve (to avoid destroying hard-edge alignment).
     * Default: true
     */
    bool preserve_features = true;
};

/**
 * Run the edge flow refinement pass on an extracted quad mesh.
 *
 * This function modifies the mesh in place.  It does not change topology
 * (no vertex insertion or deletion), only vertex positions and — in the
 * loop-straightening sub-pass — the rotation of shared quad edges (which
 * reconnects the same vertices in a different winding order).
 *
 * @param qm       Quad mesh to refine (modified in place)
 * @param params   Refinement parameters
 *
 * @return  Number of vertices relocated + edges rotated across all passes.
 *          Useful for logging / convergence reporting.
 */
int refine_edge_flow(QuadMesh& qm,
                     const EdgeFlowRefineParams& params = EdgeFlowRefineParams{});

/**
 * Score function: per-vertex edge flow quality.
 *
 * Returns a value in [0, 1] where 1.0 = perfect (all incident quads have
 * angles ≥ 90° and the four outgoing edge directions are evenly spaced at
 * 90° intervals).  Values < 0.5 indicate a vertex that would benefit from
 * relocation or topological modification.
 *
 * Exposed separately for use by the confidence score system (§13.9) and
 * the auto quality optimizer (§13.3).
 *
 * @param qm        Quad mesh
 * @param vertex_i  Vertex index to evaluate
 */
double vertex_flow_score(const QuadMesh& qm, int vertex_i);

/**
 * Compute the mean edge flow quality score over all interior vertices.
 * Interior = not on the boundary of the quad mesh.
 *
 * @param qm  Quad mesh to evaluate
 * @return    Mean of vertex_flow_score over all interior vertices, or 1.0
 *            if there are no interior vertices.
 */
double mesh_flow_score(const QuadMesh& qm);

} // namespace qf

#endif // QUADFORGE_EXTRACT_EDGE_FLOW_REFINE_H
