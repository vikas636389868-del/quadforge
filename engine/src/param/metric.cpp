/**
 * metric.cpp — Adaptive sizing field for parametrization.
 *
 * v100 REFACTOR: The three modifier blocks (curvature, vertex colour, feature
 * proximity) that were previously duplicated inline here have been replaced
 * with calls to the canonical mesh-layer functions declared in
 * mesh/sizing_field.h:
 *
 *   apply_curvature_modifier()
 *   apply_vertex_color_modifier()
 *   apply_feature_proximity_modifier()
 *
 * Motivation:
 *   Before this change, param/metric.cpp contained its own private copies of
 *   all three modifier algorithms.  The mesh-layer equivalents in
 *   mesh/sizing_field.cpp were compiled but never called — completely dead
 *   code.  Any bug fix applied to one copy was not reflected in the other,
 *   and the two implementations had already drifted (the `else if` → `if`
 *   vertex-colour fix from v2 was present in metric.cpp but missing from
 *   sizing_field.cpp, where it has now been applied separately as v100-B1).
 *
 *   By delegating here, metric.cpp is now a thin adapter that:
 *     1. Computes L_base and initialises the std::vector<double> field.
 *     2. Calls curvature_magnitudes_from_data() to bridge CurvatureData →
 *        std::vector<double> (the new v100 bridge function).
 *     3. Delegates each modifier to the mesh-layer function.
 *     4. Converts the result back to Eigen::VectorXd for the engine API.
 *
 *   This ensures there is exactly ONE authoritative implementation of each
 *   modifier, all bugs and improvements applied in one place propagate
 *   everywhere, and the mesh-layer sizing code is live.
 *
 * NOTE: The engine-facing signature (Eigen::VectorXd return, CurvatureData +
 * Eigen::MatrixXf* parameters) is unchanged — no downstream callers need
 * modification.
 */

#include "../../include/quadforge/param/metric.h"
#include "../../include/quadforge/mesh/sizing_field.h"  // mesh-layer modifiers + bridge

#include <cmath>
#include <algorithm>

namespace qf {

// ---------------------------------------------------------------------------
// compute_base_edge_length — param-layer overload (takes pre-computed area)
// ---------------------------------------------------------------------------

double compute_base_edge_length(double total_area, int target_quad_count) {
    if (target_quad_count <= 0 || total_area <= 0) return 1.0;
    // Each quad occupies approximately L^2 of surface area.
    // Solving: F * L^2 = A  ->  L = sqrt(A / F)
    return std::sqrt(total_area / target_quad_count);
}

// ---------------------------------------------------------------------------
// compute_sizing_field — param-layer entry point
//
// Delegates all modifier logic to the canonical mesh-layer functions.
// Returns Eigen::VectorXd for compatibility with the existing engine API.
// ---------------------------------------------------------------------------

Eigen::VectorXd compute_sizing_field(
    const HalfEdgeMesh&    mesh,
    const CurvatureData&   curvature,
    int                    target_quad_count,
    double                 adaptivity,
    const Eigen::MatrixXf* vertex_colors,
    const EdgeSet*         feature_edges)
{
    int nv = mesh.num_vertices();

    // --- Base size ---------------------------------------------------------
    double total_area = mesh.total_area();
    double L_base     = compute_base_edge_length(total_area, target_quad_count);

    // Initialise all vertices to L_base.
    std::vector<double> field(static_cast<std::size_t>(nv), L_base);

    // Early-out: uniform field if no adaptivity requested.
    if (adaptivity <= 0.0) {
        Eigen::VectorXd out(nv);
        for (int i = 0; i < nv; ++i) out[i] = L_base;
        return out;
    }

    // --- Modifier 1: curvature adaptivity ----------------------------------
    // Bridge CurvatureData -> std::vector<double> (v100 fix B2).
    // curvature_magnitudes_from_data() returns an empty vector when
    // max_abs_curv is empty; apply_curvature_modifier() treats nullptr /
    // empty as "skip", so the early-return guard in that function handles it.
    std::vector<double> kappa_vec = curvature_magnitudes_from_data(curvature);
    apply_curvature_modifier(
        field,
        kappa_vec.empty() ? nullptr : &kappa_vec,
        std::clamp(adaptivity, 0.0, 1.0)
    );

    // Clamp after curvature modifier: [L_base*0.1, L_base*2.0].
    for (double& v : field)
        v = std::clamp(v, L_base * 0.1, L_base * 2.0);

    // --- Modifier 2: vertex-colour density ---------------------------------
    // Convert Eigen::MatrixXf (column-major [V*3]) to interleaved flat float*
    // expected by apply_vertex_color_modifier().
    // NOTE: Eigen stores a [V*3] matrix column-major:
    //   data() = [ R0..R_{V-1}, G0..G_{V-1}, B0..B_{V-1} ]
    // apply_vertex_color_modifier() expects row-major interleaved:
    //   [ R0,G0,B0, R1,G1,B1, ... ]
    // We must build the interleaved buffer manually.
    if (vertex_colors && vertex_colors->rows() >= nv) {
        static thread_local std::vector<float> rgb_flat;
        rgb_flat.resize(static_cast<std::size_t>(nv) * 3);
        for (int vi = 0; vi < nv; ++vi) {
            rgb_flat[vi * 3    ] = (*vertex_colors)(vi, 0);
            rgb_flat[vi * 3 + 1] = (*vertex_colors)(vi, 1);
            rgb_flat[vi * 3 + 2] = (*vertex_colors)(vi, 2);
        }
        // Delegates to the mesh-layer modifier (includes the v100-B1 `if`
        // fix for independent red/green channel handling).
        apply_vertex_color_modifier(field, rgb_flat.data(),
                                    static_cast<int32_t>(nv));

        // Re-clamp: colour modifier can push vertices below the curvature floor.
        for (double& v : field)
            v = std::clamp(v, L_base * 0.1, L_base * 2.0);
    }

    // --- Modifier 3: feature proximity boost -------------------------------
    // Wrap the EdgeSet into a FeatureData so apply_feature_proximity_modifier()
    // can read hard_edges.  boundary_edges is left empty; the mesh-layer
    // function combines both sets, so passing hard_edges only is correct when
    // the engine has not separately detected boundary edges at this call site.
    if (feature_edges && !feature_edges->empty()) {
        FeatureData fd;
        fd.hard_edges = *feature_edges;

        SizingFieldOptions sf_opts;
        sf_opts.target_quad_count               = target_quad_count;
        sf_opts.use_feature_proximity           = true;
        sf_opts.feature_proximity_radius_factor = 2.0;  // match mesh-layer default

        // The Dijkstra-BFS proximity modifier is more accurate than the simple
        // unique-vertex boost that was previously inlined here: it computes
        // true geodesic distances from feature curves rather than just applying
        // a flat 1.5x boost to direct neighbours.
        apply_feature_proximity_modifier(field, mesh, fd, L_base, sf_opts);

        // Re-clamp after proximity modifier.
        for (double& v : field)
            v = std::clamp(v, L_base * 0.1, L_base * 2.0);
    }

    // --- Convert std::vector<double> -> Eigen::VectorXd --------------------
    Eigen::VectorXd out(nv);
    for (int i = 0; i < nv; ++i)
        out[i] = field[static_cast<std::size_t>(i)];
    return out;
}

} // namespace qf