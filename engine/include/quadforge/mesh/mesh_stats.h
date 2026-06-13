#pragma once
/**
 * quadforge/mesh/mesh_stats.h — Mesh statistical descriptors.
 *
 * Provides the per-mesh statistical summary used by:
 *   - The confidence scoring system (Roadmap §13.9).
 *   - The benchmark harness (Roadmap §13.4 / §11.7).
 *   - The adaptive pipeline switcher (Roadmap §13.5 — mesh classifier).
 *   - The auto quality optimizer (Roadmap §13.3).
 *
 * The key type is MeshStats, a flat struct of descriptors that can be
 * compared across runs, serialised to JSON, or fed into the Python-layer
 * quality gates and confidence scorer without going back to the C++ mesh.
 *
 * Design philosophy
 * -----------------
 * All descriptors are computed in a single O(V + F) pass over the mesh
 * (compute_mesh_stats).  Individual sub-routines (compute_edge_length_stats,
 * compute_angle_stats, etc.) are exposed separately so callers can request
 * only the statistics they need.
 *
 * Percentile computation uses a single call to std::nth_element on a local
 * copy of the raw values — O(N) per percentile — rather than a full sort.
 * This is appropriate for production use where N < 10M faces.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#ifndef QUADFORGE_MESH_MESH_STATS_H
#define QUADFORGE_MESH_MESH_STATS_H

#include <cstdint>
#include <string>
#include <vector>

#include "../types.h"
#include "halfedge.h"
#include "features.h"   // FeatureData (optional input)

namespace qf {

// =========================================================================
// Sub-structs
// =========================================================================

/**
 * Summary statistics for a 1-D scalar distribution.
 * Used for edge lengths, face areas, angles, and curvature.
 */
struct DistributionStats {
    double min     = 0.0;
    double max     = 0.0;
    double mean    = 0.0;
    double median  = 0.0;   // 50th percentile
    double p10     = 0.0;   // 10th percentile
    double p90     = 0.0;   // 90th percentile
    double std_dev = 0.0;
    int    count   = 0;

    /** Ratio of max/min.  ≥ 1.0; large values indicate high variation. */
    double range_ratio() const;
};

/**
 * Edge-length statistics derived from all undirected mesh edges.
 */
struct EdgeLengthStats {
    DistributionStats dist;
    double            mean_valence = 0.0;  ///< Mean number of edges per vertex

    /** True if edge-length variance is low (uniform mesh). */
    bool is_uniform(double threshold = 0.2) const;
};

/**
 * Face-angle statistics derived from all interior angles of all triangles.
 * All angles are in degrees.
 */
struct AngleStats {
    DistributionStats dist;
    double            min_interior    = 0.0;  ///< Absolute minimum angle (degrees)
    double            max_interior    = 0.0;  ///< Absolute maximum angle (degrees)
    double            mean_deviation  = 0.0;  ///< Mean |angle - 60°|
    int               obtuse_faces    = 0;    ///< Faces with any angle > 90°
    int               degenerate_faces= 0;   ///< Faces with any angle < 1°
};

/**
 * Face-area statistics.
 */
struct AreaStats {
    DistributionStats dist;
    double            total_area      = 0.0;
    double            mean_area       = 0.0;
    int               zero_area_faces = 0;   ///< Count of degenerate (zero-area) faces
};

/**
 * Mesh complexity indicators used by the adaptive pipeline classifier.
 */
struct ComplexityIndicators {
    // Topology
    int    num_vertices          = 0;
    int    num_faces             = 0;
    int    num_edges             = 0;
    int    genus                 = 0;
    int    num_boundary_loops    = 0;
    int    num_connected_components = 0;
    bool   is_manifold           = false;
    bool   is_closed             = false;
    int    non_manifold_edge_count = 0;

    // Feature density
    double feature_edge_ratio    = 0.0;  ///< |feature edges| / |total edges|
    double boundary_edge_ratio   = 0.0;  ///< |boundary edges| / |total edges|
    double corner_density        = 0.0;  ///< corner vertices / total vertices
    int    num_feature_curves    = 0;

    // Curvature-based complexity
    double mean_abs_curvature    = 0.0;  ///< Average |κ₁| + |κ₂|
    double max_abs_curvature     = 0.0;
    double curvature_range_ratio = 0.0;  ///< max / mean (scale of curvature variation)

    // Sizing
    double aspect_ratio_mean     = 0.0;  ///< Mean triangle aspect ratio
    double aspect_ratio_max      = 0.0;  ///< Worst triangle aspect ratio
};

/**
 * Classifier label produced by classify_mesh_type().
 */
enum class MeshCategory : uint8_t {
    Unknown      = 0,  ///< Cannot classify
    Organic      = 1,  ///< Smooth, high genus (characters, creatures)
    HardSurface  = 2,  ///< Flat regions with sharp feature edges (mechanical)
    CAD          = 3,  ///< Strict planarity, minimal curvature variation
    Scan         = 4,  ///< High vertex count, dense and noisy triangulation
    Architecture = 5,  ///< Mostly planar, axis-aligned, right-angle corners
    Damaged      = 6,  ///< Non-manifold, degenerate faces, self-intersections
};

/** Human-readable name for a MeshCategory. */
const char* mesh_category_name(MeshCategory cat);

// =========================================================================
// Full mesh statistics
// =========================================================================

/**
 * Complete statistical descriptor for a triangle mesh.
 *
 * Produced by compute_mesh_stats().  All fields are filled from a single
 * linear scan of the half-edge mesh structure plus the optional FeatureData.
 */
struct MeshStats {
    // --- Core sub-structs ---
    EdgeLengthStats   edge_lengths;
    AngleStats        angles;
    AreaStats         areas;
    ComplexityIndicators complexity;

    // --- Quality indicators (pre-computed for the confidence scorer) ---

    /** Fraction of faces with aspect ratio > 10 (poor quality). */
    double poor_quality_face_ratio = 0.0;

    /** Mean aspect ratio of all faces (target: close to 1.0 for equilateral). */
    double mean_aspect_ratio = 0.0;

    /**
     * Isotropy score ∈ [0, 1].
     * 1.0 = perfectly uniform sizing; 0.0 = extreme variation.
     * Computed as: 1 - std(edge_lengths) / mean(edge_lengths), clamped to [0,1].
     */
    double isotropy_score = 0.0;

    /**
     * Mesh classifier prediction.
     * Set by classify_mesh_type(); starts as MeshCategory::Unknown.
     */
    MeshCategory category = MeshCategory::Unknown;

    /**
     * Confidence that this category is correct (0–1).
     * Simple heuristic based on threshold distances in feature space.
     */
    double category_confidence = 0.0;

    // --- Serialisation ---

    /** Return a human-readable multi-line summary string. */
    std::string to_string() const;

    /** Return a compact single-line CSV header row. */
    static std::string csv_header();

    /** Return a compact single-line CSV data row matching csv_header(). */
    std::string to_csv_row() const;
};

// =========================================================================
// Computation functions
// =========================================================================

/**
 * Compute the full MeshStats from a half-edge mesh.
 *
 * Optional FeatureData enriches the complexity indicators
 * (feature edge ratios, curve counts, corner density).
 * Pass nullptr to skip feature-based statistics.
 *
 * Thread safety: this function is read-only on mesh and features; it is
 * safe to call from multiple threads on different meshes simultaneously.
 *
 * @param mesh      Input half-edge mesh (triangle mesh).
 * @param features  Optional pre-computed feature data.  nullptr = skip.
 * @param curvature_magnitudes  Optional per-vertex max |κ| array [nv].
 *                              nullptr = curvature stats default to 0.
 * @return          Populated MeshStats (category = Unknown until classify()
 *                  is called separately).
 */
MeshStats compute_mesh_stats(
    const HalfEdgeMesh& mesh,
    const FeatureData*  features            = nullptr,
    const std::vector<double>* curvature_magnitudes = nullptr);

/**
 * Compute edge-length statistics only.
 * Cheaper than the full compute_mesh_stats when only edge data is needed.
 */
EdgeLengthStats compute_edge_length_stats(const HalfEdgeMesh& mesh);

/**
 * Compute face-angle statistics only.
 * All returned angles are in degrees.
 */
AngleStats compute_angle_stats(const HalfEdgeMesh& mesh);

/**
 * Compute face-area statistics only.
 */
AreaStats compute_area_stats(const HalfEdgeMesh& mesh);

/**
 * Compute percentile values from a raw vector of scalars.
 *
 * Uses std::nth_element — O(N) per percentile.
 *
 * @param values      Scalar values (may be modified internally via copies).
 * @param percentiles Requested percentile levels ∈ [0, 100].
 * @return            One value per requested percentile, same order.
 */
std::vector<double> compute_percentiles(
    std::vector<double> values,          // intentional copy — nth_element mutates
    const std::vector<double>& percentiles);

// =========================================================================
// Mesh classifier
// =========================================================================

/**
 * Classify a mesh into one of the MeshCategory types.
 *
 * Uses a rule-based classifier operating on the complexity indicators
 * in the MeshStats struct.  The rules are thresholds tuned against the
 * benchmark dataset (Roadmap §13.4).
 *
 * Writes the result into stats.category and stats.category_confidence.
 * The function is idempotent: calling it twice on the same stats produces
 * the same result.
 *
 * @param stats  MeshStats produced by compute_mesh_stats().
 * @return       Predicted MeshCategory (also stored in stats.category).
 */
MeshCategory classify_mesh_type(MeshStats& stats);

// =========================================================================
// Benchmark helpers
// =========================================================================

/**
 * Compute a scalar "mesh difficulty" score ∈ [0, 1].
 *
 * Higher = harder for the remesher:
 *   0.0  — ideal uniform sphere-like mesh
 *   0.5  — typical production mesh with moderate feature complexity
 *   1.0  — severely damaged / degenerate / extremely high-curvature mesh
 *
 * Used by the benchmark harness to weight regressions and by the adaptive
 * pipeline to decide whether to run the full or simplified solver.
 *
 * @param stats  MeshStats from compute_mesh_stats().
 * @return       Difficulty score ∈ [0, 1].
 */
double compute_mesh_difficulty(const MeshStats& stats);

/**
 * Return true if the mesh is considered "safe" for the full global pipeline.
 *
 * "Safe" means: manifold, no degenerate faces, aspect ratios within
 * reasonable bounds, and curvature variation not extreme.
 *
 * If false, the engine should downgrade to the robust simplified solver.
 *
 * @param stats      Populated MeshStats.
 * @param reasons    If not nullptr, populated with plain-language reasons
 *                   for any unsafe flags found.
 * @return           true if safe for the full pipeline.
 */
bool mesh_is_safe_for_full_pipeline(
    const MeshStats&        stats,
    std::vector<std::string>* reasons = nullptr);

} // namespace qf

#endif // QUADFORGE_MESH_MESH_STATS_H
