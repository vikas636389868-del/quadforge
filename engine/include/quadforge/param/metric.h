#pragma once
/* quadforge/param/metric.h — Adaptive sizing field.
 *
 * Computes per-vertex target edge length from:
 *   1. Base edge length from total area and target quad count
 *   2. Curvature-based modulation (high curvature → smaller quads)
 *   3. Vertex color density map (red = finer, green = coarser)
 *   4. Feature proximity boost
 */

#ifndef QUADFORGE_PARAM_METRIC_H
#define QUADFORGE_PARAM_METRIC_H

#include <vector>
#include "../types.h"
#include "../mesh/halfedge.h"
#include "../field/curvature.h"

namespace qf {

/**
 * Compute per-vertex target edge length (sizing field).
 *
 * @param mesh              Triangle mesh
 * @param curvature         Principal curvature data
 * @param target_quad_count Desired output quad count
 * @param adaptivity        0.0 = uniform, 1.0 = max curvature adaptation
 * @param vertex_colors     [V×3] RGB density map (NULL = no density map)
 * @param feature_edges     Feature edges (for proximity boost)
 */
Eigen::VectorXd compute_sizing_field(
    const HalfEdgeMesh&    mesh,
    const CurvatureData&   curvature,
    int                    target_quad_count,
    double                 adaptivity,
    const Eigen::MatrixXf* vertex_colors = nullptr,
    const EdgeSet*         feature_edges = nullptr
);

/** Base edge length for a uniform quad mesh of target_quad_count quads
 *  on a surface of total_area. */
double compute_base_edge_length(double total_area, int target_quad_count);

} // namespace qf

#endif // QUADFORGE_PARAM_METRIC_H
