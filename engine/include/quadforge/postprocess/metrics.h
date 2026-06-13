#pragma once
/* quadforge/postprocess/metrics.h — Quality metrics for quad mesh output.
 *
 * Implements Step 5.4 of the QuadForge pipeline (Phase 5, Post-Processing):
 *
 *   "Compute and report: percentage of pure quad faces, histogram of vertex
 *    valences, distribution of face aspect ratios, minimum/maximum angles,
 *    scaled Jacobian quality metric, and comparison of the output face count
 *    to the target."
 *
 * All metrics are computed over the entire output QuadMesh.  Optionally
 * accepts FeatureData to also compute feature-alignment error.
 *
 * Bug-fixes:
 *   v79: Seven floating-point fields in QualityMetrics were initialised to
 *        0.f, which is ambiguous: 0° is a valid (degenerate) angle, 0 is
 *        a valid (degenerate) scaled Jacobian, and 0 is not a valid aspect
 *        ratio but was used as the default anyway.  Callers inspecting these
 *        fields on a mesh with no quad faces received 0.f and could not
 *        distinguish "not computed" from a real measurement.  Fixed:
 *          - min/max/mean_angle_deg    → −1.f  (valid range [0°, 360°])
 *          - mean/max_aspect_ratio     → −1.f  (valid range [1.0, ∞))
 *          - min/mean_scaled_jacobian  → −2.f  (valid range [−1, 1])
 *        The feature-alignment fields already used −1.f correctly and are
 *        unchanged.
 *
 * References:
 *   - Scaled Jacobian: Knupp (2000) "Achieving finite element mesh quality
 *     via optimization of the Jacobian matrix norm and associated quantities."
 *   - Quad quality survey: Shewchuk (2002) "What is a good linear finite
 *     element?"
 */

#ifndef QUADFORGE_POSTPROCESS_METRICS_H
#define QUADFORGE_POSTPROCESS_METRICS_H

#include "../types.h"
#include "../mesh/features.h"
#include "../mesh/halfedge.h"

#include <cstdint>
#include <vector>

namespace qf {

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

/**
 * Full quality report for one output QuadMesh.
 * All floating-point values are in a natural unit (fraction [0,1] or degrees).
 * Fields marked "−1 if not computed" are set when optional inputs are absent.
 */
struct QualityMetrics {

    // --- Face composition ---------------------------------------------------
    int32_t num_quad_faces{0};
    int32_t num_tri_faces{0};
    float   quad_percentage{0.f};   ///< num_quad / (num_quad + num_tri), range [0,1]

    // --- Valence statistics -------------------------------------------------
    /**
     * Vertex valence = number of quad/tri faces incident to that vertex.
     * Ideal for a pure quad mesh: 4 at every interior vertex.
     */
    float   avg_valence{0.f};
    float   irregularity_ratio{0.f}; ///< fraction of interior vertices with valence != 4
    int32_t num_irregular{0};        ///< count of irregular interior vertices

    /**
     * valence_histogram[k] = number of vertices with exactly k incident faces.
     * Index 0 = isolated vertex (degenerate), index ≥ 9 is clamped to index 9.
     * Size is always 10.
     */
    std::vector<int32_t> valence_histogram;

    // --- Angle statistics (degrees) -----------------------------------------
    /**
     * Interior angles of all quad faces.
     * A perfect square has all angles = 90°.
     * Negative scaled-Jacobian quads have at least one angle > 180°.
     *
     * BUG-FIX (v79): fields were initialised to 0.f, which is ambiguous:
     * callers could not distinguish "no quads present / not computed" from
     * "every quad has a degenerate 0° corner".  Changed to −1.f (a value
     * outside the valid range [0°, 360°]) to serve as an unambiguous
     * "not-computed" sentinel.  The compute path only writes these fields
     * when nq > 0, so a caller that receives −1.f knows the mesh has no
     * quad faces or that metrics were never run.
     */
    float min_angle_deg{-1.f};
    float max_angle_deg{-1.f};
    float mean_angle_deg{-1.f};

    // --- Aspect ratio -------------------------------------------------------
    /**
     * Per-quad aspect ratio = longest_edge / shortest_edge.
     * Perfect square = 1.0.  Values >> 1 indicate needle-like quads.
     *
     * BUG-FIX (v79): same sentinel ambiguity as angle fields.  Valid aspect
     * ratios are always ≥ 1.0, so −1.f is an unambiguous "not-computed"
     * sentinel.
     */
    float mean_aspect_ratio{-1.f};
    float max_aspect_ratio{-1.f};

    // --- Scaled Jacobian ----------------------------------------------------
    /**
     * Scaled Jacobian (SJ) of the bilinear map from the unit square to each
     * quad corner.  Range [−1, 1]:
     *   1.0  = perfect right-angle square
     *   0.0  = degenerate (at least one zero-area triangle)
     *  −1.0  = inverted / self-intersecting
     *
     * min_scaled_jacobian is the most critical value — any negative value
     * indicates an inverted element.
     *
     * BUG-FIX (v79): fields were initialised to 0.f, which is an ambiguous
     * sentinel: 0.0 is a *valid and meaningful* SJ value (degenerate quad
     * with a zero-area corner), so callers could not tell "not computed"
     * from "mesh contains a degenerate element".  Changed to −2.f, which
     * lies outside the mathematically valid SJ range [−1, 1] and serves as
     * an unambiguous "not-computed" marker.  The compute path only writes
     * these fields when nq > 0 and the parallel_reduce finds at least one
     * valid face.
     */
    float min_scaled_jacobian{-2.f};
    float mean_scaled_jacobian{-2.f};

    // --- Feature alignment error -------------------------------------------
    /**
     * For each quad edge whose midpoint is within snap_distance of a feature
     * curve, the angle between the edge direction and the nearest feature
     * tangent.  Ideal = 0° (edge perfectly aligned).
     *
     * Set to −1 if features == nullptr or no feature edges exist.
     */
    float mean_feature_alignment_error_deg{-1.f};
    float max_feature_alignment_error_deg{-1.f};

    // --- Target comparison -------------------------------------------------
    int32_t target_quad_count{0};    ///< User's requested count (0 = not set)
    float   count_accuracy{0.f};     ///< |actual − target| / target, 0 = perfect

    // --- Subdiv readiness --------------------------------------------------
    bool subdiv_ready{false};        ///< all_quads && irregularity_ratio < 0.05
};

// ---------------------------------------------------------------------------
// Main API
// ---------------------------------------------------------------------------

/**
 * Compute all quality metrics for a finished quad mesh.
 *
 * @param quad_mesh      Const reference to the output mesh.
 * @param features       Feature data from Stage 1 (pass nullptr to skip
 *                       feature-alignment computation).
 * @param target_quads   User-requested quad count; 0 = skip count-accuracy.
 * @param snap_distance  Max distance for considering an edge "near" a feature
 *                       curve when computing alignment error.  Ignored if
 *                       features == nullptr.
 * @param num_threads    0 = OpenMP auto-detect.
 * @return               Populated QualityMetrics struct.
 */
QualityMetrics compute_quality_metrics(
    const QuadMesh&    quad_mesh,
    const FeatureData* features      = nullptr,
    int32_t            target_quads  = 0,
    double             snap_distance = 0.0,
    int                num_threads   = 0
);

/**
 * Write the two fields that the QFResult struct exposes (quad_percentage
 * and avg_valence) back into the QuadMesh so that stage6_output() can
 * read them without a separate metrics call.
 */
void apply_metrics_to_mesh(QuadMesh&             quad_mesh,
                            const QualityMetrics& metrics);

/**
 * Extended overload that also computes feature-alignment error.
 *
 * The base overload cannot compute alignment error because FeatureData stores
 * vertex index pairs from the original mesh, not absolute positions.  This
 * overload accepts the original HalfEdgeMesh so it can call vertex_pos() to
 * retrieve the positions of each feature edge endpoint.
 *
 * All other metrics are identical to the base overload.
 *
 * BUG FIX (v94): ref_mesh was previously a required (no-default) parameter.
 * This created an overload-resolution ambiguity: calling
 *   compute_quality_metrics(qm, features_ptr, 0)
 * was ambiguous because the integer literal 0 could be implicitly converted
 * to either int32_t (target_quads in the first overload) or
 * const HalfEdgeMesh* null (ref_mesh in this overload).  The compiler
 * rejects ambiguous calls with a hard error, breaking any call site that
 * passed a zero target_quads explicitly.
 *
 * Fix: give ref_mesh a default of nullptr.  Now the second overload is only
 * selected when the caller explicitly passes a non-null HalfEdgeMesh pointer,
 * which is never an integer literal; 0 / nullptr always routes to the first
 * overload (target_quads path) as intended.
 *
 * @param ref_mesh       The original input HalfEdgeMesh (source of feature
 *                       vertex positions).  Pass nullptr to fall back to the
 *                       base overload behaviour (alignment stays −1).
 */
QualityMetrics compute_quality_metrics(
    const QuadMesh&      quad_mesh,
    const FeatureData*   features,
    const HalfEdgeMesh*  ref_mesh      = nullptr,
    int32_t              target_quads  = 0,
    double               snap_distance = 0.0,
    int                  num_threads   = 0
);

} // namespace qf

#endif // QUADFORGE_POSTPROCESS_METRICS_H
