/**
 * sizing_field.cpp — Per-vertex target edge length (sizing field).
 *
 * Implements Stage 1, Step 1.5 of the QuadForge pipeline as described in
 * the roadmap (section 3.1).  See sizing_field.h for API documentation.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#include "../../include/quadforge/mesh/sizing_field.h"
#include "../../include/quadforge/mesh/geodesic.h"  // compute_geodesic_from_features

#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>

namespace qf {

// =========================================================================
// compute_base_edge_length
// =========================================================================

double compute_base_edge_length(
    const HalfEdgeMesh& mesh,
    int                 target_quad_count)
{
    if (target_quad_count <= 0) target_quad_count = 1;

    double total_area = mesh.total_area();
    if (total_area < 1e-30) return 0.0;  // degenerate mesh

    // Each quad covers approximately L_base² of surface area.
    // Solving: F * L² ≈ A  →  L = sqrt(A / F)
    return std::sqrt(total_area / static_cast<double>(target_quad_count));
}

// =========================================================================
// apply_curvature_modifier
// =========================================================================

void apply_curvature_modifier(
    std::vector<double>&       field,
    const std::vector<double>* curvature_magnitudes,
    double                     alpha)
{
    if (!curvature_magnitudes || alpha <= 0.0) return;  // modifier disabled

    int nv = static_cast<int>(field.size());
    if (nv == 0) return;
    if (static_cast<int>(curvature_magnitudes->size()) < nv) return;

    // Clamp alpha to [0, 1] defensively.
    alpha = std::max(0.0, std::min(1.0, alpha));

    for (int vi = 0; vi < nv; ++vi) {
        double L    = field[vi];
        double kapp = (*curvature_magnitudes)[vi];

        // Clamp negative curvature to zero (principal curvature magnitudes
        // should be non-negative, but floating-point artefacts can produce
        // tiny negative values near flat regions).
        if (kapp < 0.0) kapp = 0.0;

        // Guard against zero L (shouldn't happen, but be safe).
        if (L < 1e-30) continue;

        // L_new = L_old / (1 + alpha * kappa * L_old)
        // This formula is numerically stable: denominator ≥ 1.
        double denom = 1.0 + alpha * kapp * L;
        field[vi] = L / denom;
    }
}

// =========================================================================
// apply_vertex_color_modifier
// =========================================================================

void apply_vertex_color_modifier(
    std::vector<double>&  field,
    const float*          vertex_colors,
    int32_t               nv)
{
    if (!vertex_colors || nv <= 0) return;

    int field_nv = static_cast<int>(field.size());
    if (field_nv == 0) return;

    // Process min(nv, field_nv) vertices to avoid out-of-bounds access.
    int count = std::min(static_cast<int>(nv), field_nv);

    for (int vi = 0; vi < count; ++vi) {
        float r = vertex_colors[vi * 3    ];
        float g = vertex_colors[vi * 3 + 1];
        // Blue channel is unused (matches QuadRemesher convention).

        // Clamp channels to [0, 1] (Blender may produce out-of-range values
        // from HDR vertex paint or floating-point precision).
        r = std::max(0.f, std::min(1.f, r));
        g = std::max(0.f, std::min(1.f, g));

        double density_mult = 1.0;

        if (r > 0.5f) {
            // Red channel: increase density → density_mult = 4^(2*red - 1)
            // At red=0.5 → mult=1.0; at red=1.0 → mult=4.0.
            // Smaller target edge length → denser quads.
            density_mult = std::pow(4.0, 2.0 * static_cast<double>(r) - 1.0);
        }
        // BUG FIX (v100): was `else if (g > 0.5f)`.  When a vertex is painted
        // with BOTH red > 0.5 AND green > 0.5 (mixed Blender vertex paint), the
        // old `else if` silently discarded the green channel — the coarser-quad
        // instruction was completely dropped.  Both channels must accumulate into
        // `density_mult` as independent multiplications so the artist's intent is
        // always respected.  This matches the identical fix applied to
        // param/metric.cpp in v2 (Bug 2 — param/metric) but not yet backported
        // to this mesh-layer implementation.
        if (g > 0.5f) {
            // Green channel: decrease density → density_mult = 0.25^(2*green - 1)
            // At green=0.5 → mult=1.0; at green=1.0 → mult=0.25.
            // Larger target edge length → sparser quads.
            density_mult *= std::pow(0.25, 2.0 * static_cast<double>(g) - 1.0);
        }

        if (density_mult > 1e-10) {
            // L_new = L_old / density_mult
            // Higher density_mult → smaller L → more quads in that region.
            field[vi] /= density_mult;
        }
    }
}

// =========================================================================
// apply_feature_proximity_modifier
//
// Delegates geodesic distance computation to geodesic.h
// (compute_geodesic_from_features) rather than inlining its own Dijkstra.
// This removes ~80 lines of duplicated priority-queue code and ensures
// that any future improvement to the geodesic module (e.g. heat method)
// is automatically inherited here.
//
// The density-boost formula is unchanged from the original inline version:
//   L_new(v) = L_old(v) * (min_fraction + (1 - min_fraction) * sqrt(d/R))
// where d is the geodesic distance to the nearest feature edge/vertex and
// R = feature_proximity_radius_factor * L_base.
// =========================================================================

void apply_feature_proximity_modifier(
    std::vector<double>&       field,
    const HalfEdgeMesh&        mesh,
    const FeatureData&         features,
    double                     L_base,
    const SizingFieldOptions&  opts)
{
    if (!opts.use_feature_proximity) return;

    const int nv = mesh.num_vertices();
    if (nv == 0 || field.empty()) return;

    const double R = opts.feature_proximity_radius_factor * L_base;
    if (R < 1e-30) return;

    // --- Compute geodesic distances using the dedicated module --------------
    // max_dist = R: Dijkstra stops propagating beyond the proximity radius,
    // keeping the call O(V_near log V_near) rather than O(V log V).
    GeodesicOptions geo_opts;
    geo_opts.max_dist = R;

    const std::vector<double> dist =
        compute_geodesic_from_features(mesh, features, geo_opts);

    // --- Apply smooth density boost ----------------------------------------
    // At d = 0   → factor = min_fraction (maximum density increase)
    // At d = R   → factor = 1.0          (no change at the radius boundary)
    // At d > R   → dist[v] = kInf → skipped
    const double min_fraction = 0.25;
    const double kInf = std::numeric_limits<double>::max();

    const int fn = static_cast<int>(field.size());
    for (int vi = 0; vi < fn && vi < nv; ++vi) {
        const double d = dist[vi];
        // BUG FIX (v107): The original condition was `d >= R || d >= kInf`.
        // The second clause is entirely redundant: kInf is
        // std::numeric_limits<double>::max(), which is always >= any finite R,
        // so any d that satisfies `d >= kInf` also satisfies `d >= R`
        // (since R is a finite user-supplied radius).  The redundant clause
        // adds no logical coverage and obscures the intent.  Removed.
        if (d >= R) continue;

        const double t      = d / R;   // ∈ [0, 1)
        const double factor = min_fraction + (1.0 - min_fraction) * std::sqrt(t);
        field[vi] *= factor;
    }
}

// =========================================================================
// compute_sizing_field — main entry point
// =========================================================================

std::vector<double> compute_sizing_field(
    const HalfEdgeMesh&        mesh,
    const std::vector<double>* curvature_magnitudes,
    const FeatureData*         features,
    const SizingFieldOptions&  opts)
{
    int nv = mesh.num_vertices();
    if (nv <= 0) return {};

    // --- Base size ---------------------------------------------------------
    double L_base = compute_base_edge_length(mesh, opts.target_quad_count);
    if (L_base < 1e-30) {
        // Degenerate mesh — return uniform tiny field.
        return std::vector<double>(nv, 1e-6);
    }

    // Initialise all vertices to L_base.
    std::vector<double> field(nv, L_base);

    // --- Modifier 1: curvature adaptivity ----------------------------------
    apply_curvature_modifier(field, curvature_magnitudes,
                             opts.curvature_adaptivity);

    // --- Modifier 2: vertex-colour density ---------------------------------
    if (opts.use_vertex_colors && opts.vertex_colors_rgb) {
        apply_vertex_color_modifier(field,
                                    opts.vertex_colors_rgb,
                                    static_cast<int32_t>(nv));
    }

    // --- Modifier 3: feature proximity boost -------------------------------
    if (opts.use_feature_proximity && features) {
        apply_feature_proximity_modifier(field, mesh, *features, L_base, opts);
    }

    // --- Clamp to [min, max] edge length -----------------------------------
    double lo = opts.min_edge_length;
    double hi = opts.max_edge_length;

    // Auto-compute bounds if not provided (0 sentinel).
    if (lo <= 0.0) lo = L_base * 0.05;   // no face smaller than 5% of base
    if (hi <= 0.0) hi = L_base * 10.0;   // no face larger than 10× base

    for (double& v : field)
        v = std::max(lo, std::min(hi, v));

    return field;
}

// =========================================================================
// curvature_magnitudes_from_data — CurvatureData → std::vector<double> bridge
//
// BUG FIX (v100): Previously there was no way to feed a CurvatureData
// (whose max_abs_curv is stored as Eigen::VectorXd) into the mesh-layer
// sizing API (which takes const std::vector<double>*).  Every caller that
// held a CurvatureData from compute_curvature() had to manually extract and
// convert the Eigen data — with no canonical location to do this, all
// engine code paths silently skipped the curvature modifier entirely.
//
// This function is the canonical extraction point.  It is a thin wrapper;
// all numerical work happens in apply_curvature_modifier() as before.
// =========================================================================

std::vector<double> curvature_magnitudes_from_data(const CurvatureData& data)
{
    const Eigen::VectorXd& src = data.max_abs_curv;
    if (src.size() == 0) return {};

    std::vector<double> out(static_cast<std::size_t>(src.size()));
    for (Eigen::Index i = 0; i < src.size(); ++i) {
        // Clamp to [0, ∞) — max_abs_curv is logically non-negative, but
        // floating-point noise near flat regions can produce tiny negatives.
        double v = src[i];
        out[static_cast<std::size_t>(i)] = v < 0.0 ? 0.0 : v;
    }
    return out;
}

// =========================================================================
// compute_sizing_field — CurvatureData + Eigen::MatrixXf overload
//
// Convenience entry point for the engine pipeline, which already holds a
// CurvatureData from Stage 1 and optionally an Eigen::MatrixXf for vertex
// colours.  Internally this:
//   1. Converts CurvatureData → std::vector<double> via curvature_magnitudes_from_data()
//   2. Converts Eigen::MatrixXf → flat float* (data() is guaranteed contiguous)
//   3. Forwards to compute_sizing_field(mesh, curvature_magnitudes*, features*, opts)
//
// The vertex_colors_mat parameter overrides opts.vertex_colors_rgb; if
// vertex_colors_mat is nullptr, opts.vertex_colors_rgb is used as a fallback
// (allowing the raw-float path to remain functional for tests).
// =========================================================================

std::vector<double> compute_sizing_field(
    const HalfEdgeMesh&       mesh,
    const CurvatureData&      curvature,
    const FeatureData*        features,
    const SizingFieldOptions& opts,
    const Eigen::MatrixXf*    vertex_colors_mat)
{
    // Extract curvature magnitudes — empty vector means modifier is disabled.
    std::vector<double> kappa_vec = curvature_magnitudes_from_data(curvature);
    const std::vector<double>* kappa_ptr =
        kappa_vec.empty() ? nullptr : &kappa_vec;

    // Build a modified options struct that points at the Eigen matrix data
    // if the caller supplied vertex_colors_mat.
    SizingFieldOptions eff_opts = opts;
    // BUG FIX (v106): The original guard checked only rows() >= num_vertices but
    // not cols() >= 3.  If the caller passes a matrix with fewer than 3 columns
    // (e.g. a greyscale [V×1] matrix), accessing column index 2 via
    // (*vertex_colors_mat)(vi, 2) is an out-of-bounds Eigen access that causes
    // undefined behaviour (assert in Debug, silent memory read in Release).
    // Fix: require both rows() >= nv AND cols() >= 3 before entering the block.
    if (vertex_colors_mat
        && vertex_colors_mat->rows() >= mesh.num_vertices()
        && vertex_colors_mat->cols() >= 3) {
        // Eigen::MatrixXf::data() returns a contiguous float* in column-major
        // order (Eigen default).  For a [V×3] RGB matrix the layout is
        //   [R0, R1, ..., R_{V-1},  G0, ...,  B0, ...]
        // which is NOT the interleaved [R0,G0,B0, R1,G1,B1,...] that
        // apply_vertex_color_modifier expects.
        //
        // We therefore manually build the interleaved flat array here so the
        // mesh-layer modifier receives data in the correct format.
        // (Alternatively the mesh could be re-arranged as row-major, but that
        // would change the apply_vertex_color_modifier contract.)
        static thread_local std::vector<float> rgb_flat;
        int nv = mesh.num_vertices();
        rgb_flat.resize(static_cast<std::size_t>(nv) * 3);
        for (int vi = 0; vi < nv; ++vi) {
            rgb_flat[vi * 3    ] = (*vertex_colors_mat)(vi, 0);
            rgb_flat[vi * 3 + 1] = (*vertex_colors_mat)(vi, 1);
            rgb_flat[vi * 3 + 2] = (*vertex_colors_mat)(vi, 2);
        }
        eff_opts.use_vertex_colors   = true;
        eff_opts.vertex_colors_rgb   = rgb_flat.data();
    }

    return compute_sizing_field(mesh, kappa_ptr, features, eff_opts);
}

} // namespace qf
