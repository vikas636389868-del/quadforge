#pragma once
/**
 * quadforge/mesh/sizing_field.h — Per-vertex target edge length (sizing field).
 *
 * Implements Stage 1, Step 1.5 of the QuadForge pipeline as described in the
 * roadmap (section 3.1):
 *
 *   L_base(v) = sqrt(total_surface_area / target_quad_count)
 *
 * followed by three multiplicative modifiers:
 *
 *   1. Curvature factor (optional — requires pre-computed per-vertex principal
 *      curvature magnitudes):
 *        L(v) = L_base / (1 + alpha * kappa(v) * L_base)
 *      where alpha = curvature_adaptivity ∈ [0, 1].
 *
 *   2. Vertex-colour density factor (optional — requires per-vertex RGB density
 *      map from Blender):
 *        density_multiplier =
 *          4^(2*red - 1)   when red > 0.5   (denser / smaller quads)
 *          0.25^(2*green - 1) when green > 0.5 (sparser / larger quads)
 *      matching the density-paint encoding used by QuadRemesher 1.3.
 *
 *   3. Feature proximity factor (optional — requires a FeatureData struct):
 *      Vertices whose geodesic distance to the nearest feature edge is below
 *      a threshold receive a density boost so the quad mesh has enough
 *      resolution to conform to detected creases and corners.
 *
 * All three modifiers are independent and compose multiplicatively.
 * Callers may pass nullptr for any optional input to skip that modifier.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#ifndef QUADFORGE_MESH_SIZING_FIELD_H
#define QUADFORGE_MESH_SIZING_FIELD_H

#include <vector>
#include <cstdint>

#include "../types.h"
#include "halfedge.h"
#include "features.h"         // FeatureData
#include "geodesic.h"         // GeodesicOptions — used by apply_feature_proximity_modifier
#include "../field/curvature.h" // CurvatureData — needed for bridge overload

namespace qf {

// =========================================================================
// Options
// =========================================================================

/**
 * Options controlling which sizing modifiers are active and their parameters.
 *
 * All modifier inputs (curvature_magnitudes, vertex_colors_rgb,
 * feature_data) may be nullptr to disable that modifier.
 */
struct SizingFieldOptions {
    // --- Base size ---
    int     target_quad_count    = 5000;  ///< Desired number of output quads
    double  min_edge_length      = 0.0;   ///< Hard lower bound (0 = auto from mesh)
    double  max_edge_length      = 0.0;   ///< Hard upper bound (0 = auto from mesh)

    // --- Curvature modifier ---
    double  curvature_adaptivity = 0.5;   ///< α ∈ [0,1]; 0 = uniform, 1 = max adaptive

    // --- Vertex-colour density modifier ---
    bool    use_vertex_colors    = false;
    /// Per-vertex RGB density map [nv*3] in [0,1] range.
    /// Red channel > 0.5  → multiply density (smaller quads).
    /// Green channel > 0.5 → divide density  (larger quads).
    const float*  vertex_colors_rgb = nullptr;

    // --- Feature proximity modifier ---
    bool    use_feature_proximity = true;
    /// Fraction of L_base below which a vertex is considered "on a feature".
    double  feature_proximity_radius_factor = 2.0;
};

// =========================================================================
// Functions
// =========================================================================

/**
 * Compute the base target edge length from total surface area.
 *
 *   L_base = sqrt(total_surface_area / target_quad_count)
 *
 * Returns 0 if the mesh is degenerate (zero area).
 */
double compute_base_edge_length(
    const HalfEdgeMesh& mesh,
    int                 target_quad_count
);

/**
 * Apply the curvature adaptivity modifier in-place.
 *
 * For each vertex v:
 *   L_new(v) = L_old(v) / (1 + alpha * kappa(v) * L_old(v))
 *
 * where kappa(v) is the maximum absolute principal curvature magnitude.
 *
 * @param field              [nv] sizing values to modify in-place
 * @param curvature_magnitudes [nv] max |principal curvature| per vertex
 *                           (output of field/curvature — pass nullptr to skip)
 * @param alpha              curvature_adaptivity ∈ [0, 1]
 */
void apply_curvature_modifier(
    std::vector<double>&       field,
    const std::vector<double>* curvature_magnitudes,
    double                     alpha
);

/**
 * Apply the vertex-colour density modifier in-place.
 *
 * Encoding (matches QuadRemesher 1.3 density-paint convention):
 *   red   > 0.5  →  density_mult = 4^(2*red - 1)   (range [1, 4])
 *   green > 0.5  →  density_mult = 0.25^(2*green - 1) (range [1, 0.25])
 *   otherwise   →  density_mult = 1.0
 *
 * L_new(v) = L_old(v) / density_mult(v)
 *   (higher density → smaller target edge length)
 *
 * @param field          [nv] sizing values to modify in-place
 * @param vertex_colors  [nv*3] RGB in [0,1], or nullptr to skip
 * @param nv             vertex count
 */
void apply_vertex_color_modifier(
    std::vector<double>&  field,
    const float*          vertex_colors,
    int32_t               nv
);

/**
 * Apply the feature-proximity density boost in-place.
 *
 * Vertices within (opts.feature_proximity_radius_factor * L_base) of any
 * feature edge receive a sizing boost so the quad mesh has sufficient
 * resolution to conform to the feature curve.
 *
 * The boost function is:
 *
 *   factor(d) = min_fraction + (1 - min_fraction) * sqrt(d / R)
 *   L_new(v)  = L_old(v) * factor(d)
 *
 * where d is the geodesic distance to the nearest feature edge, R is
 * the proximity radius, and min_fraction = 0.25.
 *
 *   At d = 0  (vertex on the feature): factor = 0.25 → L_new = 0.25 * L_old
 *             (maximum density boost — quads are 4× finer than baseline).
 *   At d = R  (edge of the proximity zone):  factor = 1.0 → no change.
 *   At d > R  (outside zone): modifier is not applied.
 *
 * BUG FIX (v105): The header previously documented the old abandoned formula
 *   L_new(v) = L_old(v) * (d / R)^0.5
 * which yields L_new = 0 at d = 0, causing quads to collapse to zero size
 * at feature vertices.  The implementation has always used the min_fraction
 * blend above (which clamps to 0.25 * L_old at d = 0) but the header was
 * never updated.  Corrected to match the actual implementation.
 *
 * @param field     [nv] sizing values to modify in-place
 * @param mesh      The half-edge mesh (for distance computation)
 * @param features  Detected features (hard_edges + boundary_edges used)
 * @param L_base    Base target edge length (for radius computation)
 * @param opts      Sizing field options
 */
void apply_feature_proximity_modifier(
    std::vector<double>&       field,
    const HalfEdgeMesh&        mesh,
    const FeatureData&         features,
    double                     L_base,
    const SizingFieldOptions&  opts
);

/**
 * Compute the complete per-vertex sizing field in one call.
 *
 * Applies all enabled modifiers in the order specified by the roadmap
 * (curvature → vertex-colour → feature proximity), then clamps to
 * [opts.min_edge_length, opts.max_edge_length].
 *
 * @param mesh                The half-edge mesh
 * @param curvature_magnitudes [nv] max |κ| per vertex (nullptr = skip)
 * @param opts                Sizing field options
 * @return                    [nv] target edge lengths in mesh units
 */
std::vector<double> compute_sizing_field(
    const HalfEdgeMesh&        mesh,
    const std::vector<double>* curvature_magnitudes,
    const FeatureData*         features,
    const SizingFieldOptions&  opts
);

// =========================================================================
// CurvatureData bridge
// =========================================================================

/**
 * Extract per-vertex max |principal curvature| from a CurvatureData struct
 * into a plain std::vector<double> suitable for apply_curvature_modifier()
 * and the curvature_magnitudes parameter of compute_sizing_field().
 *
 * Returns CurvatureData::max_abs_curv as a std::vector<double>.
 * If data.max_abs_curv is empty or has fewer than nv entries, returns an
 * empty vector (callers treat empty == nullptr → modifier disabled).
 *
 * BUG FIX (v100): Previously there was no bridge between the CurvatureData
 * struct (which stores max_abs_curv as Eigen::VectorXd) and the mesh-layer
 * sizing API (which takes const std::vector<double>*).  Any call path that
 * obtained a CurvatureData from compute_curvature() could not forward it to
 * apply_curvature_modifier() without manually extracting the Eigen data —
 * there was no canonical place to do this, so it was never done and the
 * curvature modifier was silently skipped for all engine paths.
 */
std::vector<double> curvature_magnitudes_from_data(const CurvatureData& data);

/**
 * Convenience overload: compute the complete sizing field from a CurvatureData
 * struct and an optional Eigen vertex-colour matrix.
 *
 * This is the preferred overload for engine.cpp and param/metric.cpp since
 * they already hold a CurvatureData after Stage 1 and an optional
 * Eigen::MatrixXf for vertex colours.
 *
 * Internally calls curvature_magnitudes_from_data() + compute_sizing_field().
 *
 * @param mesh          The half-edge mesh
 * @param curvature     Full CurvatureData from compute_curvature()
 * @param features      Detected features (nullptr = skip proximity modifier)
 * @param opts          Sizing field options (vertex_colors_rgb is ignored;
 *                      pass vertex_colors_mat instead)
 * @param vertex_colors_mat  [V×3] Eigen RGB matrix (nullptr = skip colour modifier)
 * @return              [nv] target edge lengths in mesh units
 */
std::vector<double> compute_sizing_field(
    const HalfEdgeMesh&        mesh,
    const CurvatureData&       curvature,
    const FeatureData*         features,
    const SizingFieldOptions&  opts,
    const Eigen::MatrixXf*     vertex_colors_mat = nullptr
);

} // namespace qf

#endif // QUADFORGE_MESH_SIZING_FIELD_H
