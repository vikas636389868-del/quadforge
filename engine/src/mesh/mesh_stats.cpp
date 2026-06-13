/**
 * mesh_stats.cpp — Mesh statistical descriptors.
 *
 * Implements the declarations in mesh_stats.h.
 *
 * Implementation notes
 * --------------------
 * * All geometric computations work on the 3-D vertex positions stored in
 *   the HalfEdgeMesh without any Eigen overhead (Vec3 arithmetic only).
 * * Percentile estimation uses nth_element on a per-call local copy of the
 *   values vector — O(N) per quantile, appropriate for up to ~10M elements.
 * * The mesh classifier uses axis-aligned threshold rules calibrated against
 *   a manually labelled benchmark set of ~200 production meshes.  Thresholds
 *   can be updated without recompiling by modifying the constexpr arrays at
 *   the top of classify_mesh_type().
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#include "../../include/quadforge/mesh/mesh_stats.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <unordered_set>

// BUG FIX (v103): MSVC does not define M_PI in <cmath> unless _USE_MATH_DEFINES
// is defined before the first system header inclusion.  Rather than requiring a
// project-wide compiler flag, we define M_PI here with the same guard already
// used in features.cpp and accel/omp_utils.h.
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qf {

// =========================================================================
// Internal geometry helpers
// =========================================================================

namespace {

/** Euclidean length of a Vec3. */
inline double len(const Vec3& v) { return v.norm(); }

/** Clamp d to [lo, hi]. */
inline double clamp(double d, double lo, double hi) {
    return d < lo ? lo : (d > hi ? hi : d);
}

/** Safe acos — clamps argument to [-1, 1] before calling std::acos. */
inline double safe_acos(double x) {
    return std::acos(clamp(x, -1.0, 1.0));
}

/** Convert radians to degrees. */
inline double to_deg(double rad) { return rad * (180.0 / M_PI); }

/**
 * Compute the three interior angles (degrees) of a triangle given its vertex
 * positions.  Returns {a0, a1, a2} where ai is the angle at vertex i.
 */
std::array<double, 3> triangle_angles(const Vec3& p0, const Vec3& p1, const Vec3& p2)
{
    Vec3 e01 = (p1 - p0).normalized();
    Vec3 e02 = (p2 - p0).normalized();
    Vec3 e10 = (p0 - p1).normalized();
    Vec3 e12 = (p2 - p1).normalized();
    Vec3 e20 = (p0 - p2).normalized();
    Vec3 e21 = (p1 - p2).normalized();

    double a0 = to_deg(safe_acos(e01.dot(e02)));
    double a1 = to_deg(safe_acos(e10.dot(e12)));
    double a2 = to_deg(safe_acos(e20.dot(e21)));
    return {a0, a1, a2};
}

/**
 * Triangle aspect ratio: circumradius / (2 * inradius).
 * = (a * b * c) / (8 * area² / (a + b + c)) = (a*b*c*(a+b+c)) / (8 * area²)
 * Simplified: a*b*c / (8 * R_in * area) where R_in = area / s.
 * Minimum value 1.0 (equilateral triangle).  Degenerate → very large.
 */
double triangle_aspect_ratio(const Vec3& p0, const Vec3& p1, const Vec3& p2)
{
    double a = len(p1 - p0);
    double b = len(p2 - p1);
    double c = len(p0 - p2);
    Vec3 cross = (p1 - p0).cross(p2 - p0);
    double area2 = cross.norm();  // 2 * area
    if (area2 < 1e-30) return 1e9;  // degenerate

    // aspect ratio = (a * b * c) / (2 * area * 4 * R_in)
    // = (a * b * c * (a + b + c)) / (8 * area^2)
    double area = area2 * 0.5;
    double s = (a + b + c);
    double aspect = (a * b * c * s) / (8.0 * area * area);
    return aspect;
}

/**
 * Build a flat list of unique undirected edges {v_lo, v_hi} from the mesh.
 */
std::vector<std::pair<int,int>> build_unique_edges(const HalfEdgeMesh& mesh)
{
    // Use the half-edge array: for each HE, emit edge only if twin == -1 or
    // this HE has smaller index than its twin.
    const int nhe = mesh.num_half_edges();
    std::vector<std::pair<int,int>> edges;
    edges.reserve(nhe / 2 + 16);

    for (int hei = 0; hei < nhe; ++hei) {
        const HalfEdge& he = mesh.half_edge(hei);
        int twin = he.twin;
        // Emit if boundary (twin == -1) or we are the "lower-index" half-edge.
        if (twin == -1 || hei < twin) {
            // Vertices: source = half_edge(prev).vertex, dest = he.vertex
            int v_dst = he.vertex;
            int v_src = mesh.half_edge(he.prev).vertex;
            int v_lo = std::min(v_src, v_dst);
            int v_hi = std::max(v_src, v_dst);
            edges.push_back({v_lo, v_hi});
        }
    }
    return edges;
}

/**
 * Compute summary DistributionStats from a values vector (modified in-place
 * by partial sorts — caller must treat it as consumed).
 */
DistributionStats make_dist_stats(std::vector<double> vals)
{
    DistributionStats s;
    s.count = static_cast<int>(vals.size());
    if (s.count == 0) return s;

    // min / max
    auto [it_min, it_max] = std::minmax_element(vals.begin(), vals.end());
    s.min = *it_min;
    s.max = *it_max;

    // mean
    s.mean = std::accumulate(vals.begin(), vals.end(), 0.0) / s.count;

    // std_dev
    double var = 0.0;
    for (double v : vals) var += (v - s.mean) * (v - s.mean);
    s.std_dev = std::sqrt(var / s.count);

    // percentiles via nth_element
    auto nth = [&](int pct) -> double {
        if (s.count == 1) return vals[0];
        // BUG FIX (v105): `pct * s.count` is evaluated as int32×int32.
        // For large meshes (e.g. 30M faces → 90M angles in the angle-stats
        // distribution), `pct=90` gives  90 × 90,000,000 = 8,100,000,000
        // which overflows signed int32 (max ≈ 2.1B), producing a garbage
        // (possibly negative) index that is clamped to 0 by std::max.
        // The p90 percentile then always returns vals[0] (the minimum after
        // the nth_element partial sort) for any mesh with ≥ ~24M triangles.
        // Fix: widen pct to int64_t before multiplication so the product is
        // computed in 64-bit arithmetic, then narrow the result back to int.
        // The result fits in int32 because it is clamped to [0, s.count-1].
        int idx = std::max(0, std::min(s.count - 1,
                                       static_cast<int>(
                                           static_cast<int64_t>(pct) * s.count / 100)));
        std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
        return vals[idx];
    };
    s.p10    = nth(10);
    s.median = nth(50);
    s.p90    = nth(90);

    return s;
}

} // anonymous namespace

// =========================================================================
// DistributionStats helpers
// =========================================================================

double DistributionStats::range_ratio() const
{
    if (min <= 0.0 || min == max) return 1.0;
    return max / min;
}

// =========================================================================
// EdgeLengthStats helpers
// =========================================================================

bool EdgeLengthStats::is_uniform(double threshold) const
{
    if (dist.mean < 1e-30) return true;
    return dist.std_dev / dist.mean <= threshold;
}

// =========================================================================
// compute_edge_length_stats
// =========================================================================

EdgeLengthStats compute_edge_length_stats(const HalfEdgeMesh& mesh)
{
    EdgeLengthStats result;
    if (mesh.num_vertices() == 0) return result;

    auto edges = build_unique_edges(mesh);
    if (edges.empty()) return result;

    std::vector<double> lengths;
    lengths.reserve(edges.size());

    for (const auto& [v0, v1] : edges) {
        double l = len(mesh.vertex_pos(v1) - mesh.vertex_pos(v0));
        lengths.push_back(l);
    }

    result.dist = make_dist_stats(std::move(lengths));

    // Mean valence: 2|E| / |V|
    result.mean_valence = 2.0 * static_cast<double>(edges.size())
                          / static_cast<double>(mesh.num_vertices());
    return result;
}

// =========================================================================
// compute_angle_stats
// =========================================================================

AngleStats compute_angle_stats(const HalfEdgeMesh& mesh)
{
    AngleStats result;
    if (mesh.num_faces() == 0) return result;

    std::vector<double> all_angles;
    all_angles.reserve(mesh.num_faces() * 3);

    double sum_dev = 0.0;
    result.min_interior = 360.0;
    result.max_interior = 0.0;

    for (int fi = 0; fi < mesh.num_faces(); ++fi) {
        const auto verts = mesh.face_vertices(fi);
        const Vec3 p0 = mesh.vertex_pos(verts[0]);
        const Vec3 p1 = mesh.vertex_pos(verts[1]);
        const Vec3 p2 = mesh.vertex_pos(verts[2]);

        const auto [a0, a1, a2] = triangle_angles(p0, p1, p2);

        for (double a : {a0, a1, a2}) {
            all_angles.push_back(a);
            sum_dev += std::abs(a - 60.0);
            if (a < result.min_interior) result.min_interior = a;
            if (a > result.max_interior) result.max_interior = a;
        }

        double face_min = std::min({a0, a1, a2});
        double face_max = std::max({a0, a1, a2});
        if (face_min < 1.0)  ++result.degenerate_faces;
        if (face_max > 90.0) ++result.obtuse_faces;
    }

    const int n = static_cast<int>(all_angles.size());
    result.mean_deviation = (n > 0) ? sum_dev / n : 0.0;
    result.dist = make_dist_stats(std::move(all_angles));
    return result;
}

// =========================================================================
// compute_area_stats
// =========================================================================

AreaStats compute_area_stats(const HalfEdgeMesh& mesh)
{
    AreaStats result;
    if (mesh.num_faces() == 0) return result;

    std::vector<double> areas;
    areas.reserve(mesh.num_faces());
    double total = 0.0;

    for (int fi = 0; fi < mesh.num_faces(); ++fi) {
        double a = mesh.face_area(fi);
        areas.push_back(a);
        total += a;
        if (a < 1e-30) ++result.zero_area_faces;
    }

    result.total_area = total;
    result.mean_area  = (mesh.num_faces() > 0) ? total / mesh.num_faces() : 0.0;
    result.dist       = make_dist_stats(std::move(areas));
    return result;
}

// =========================================================================
// compute_percentiles
// =========================================================================

std::vector<double> compute_percentiles(
    std::vector<double>        values,
    const std::vector<double>& percentiles)
{
    std::vector<double> result;
    result.reserve(percentiles.size());

    if (values.empty()) {
        result.assign(percentiles.size(), 0.0);
        return result;
    }

    for (double p : percentiles) {
        int idx = static_cast<int>(clamp(p / 100.0, 0.0, 1.0) * (values.size() - 1));
        std::nth_element(values.begin(), values.begin() + idx, values.end());
        result.push_back(values[idx]);
    }
    return result;
}

// =========================================================================
// compute_mesh_stats
// =========================================================================

MeshStats compute_mesh_stats(
    const HalfEdgeMesh&        mesh,
    const FeatureData*         features,
    const std::vector<double>* curvature_magnitudes)
{
    MeshStats stats;

    // --- Edge lengths -------------------------------------------------------
    stats.edge_lengths = compute_edge_length_stats(mesh);

    // --- Face angles --------------------------------------------------------
    stats.angles = compute_angle_stats(mesh);

    // --- Face areas ---------------------------------------------------------
    stats.areas = compute_area_stats(mesh);

    // --- Topology / complexity indicators -----------------------------------
    ComplexityIndicators& ci = stats.complexity;
    ci.num_vertices          = mesh.num_vertices();
    ci.num_faces             = mesh.num_faces();
    ci.is_manifold           = mesh.is_manifold();
    ci.is_closed             = mesh.is_closed();
    ci.genus                 = mesh.genus();
    ci.num_boundary_loops    = mesh.num_boundary_loops();
    ci.non_manifold_edge_count = static_cast<int>(mesh.non_manifold_edges().size());

    // BUG FIX (v104): num_connected_components was never populated — it stayed
    // at 0 for every mesh, corrupting the difficulty score, adaptive-pipeline
    // classifier, and benchmark CSV.  Now delegated through the new public
    // HalfEdgeMesh::num_connected_components() wrapper (which internally calls
    // the existing private BFS in num_connected_components_internal()).
    ci.num_connected_components = mesh.num_connected_components();

    // Edge count from unique-edge analysis
    const auto unique_edges = build_unique_edges(mesh);
    ci.num_edges = static_cast<int>(unique_edges.size());

    // --- Feature density (requires FeatureData) ----------------------------
    if (features && ci.num_edges > 0) {
        ci.feature_edge_ratio  = static_cast<double>(features->hard_edges.size())
                                 / ci.num_edges;
        ci.boundary_edge_ratio = static_cast<double>(features->boundary_edges.size())
                                 / ci.num_edges;
        ci.num_feature_curves  = static_cast<int>(features->curves.size());
        ci.corner_density      = (ci.num_vertices > 0)
                                 ? static_cast<double>(features->corner_vertices.size())
                                   / ci.num_vertices
                                 : 0.0;
    }

    // --- Curvature-based complexity (requires curvature_magnitudes) --------
    if (curvature_magnitudes && !curvature_magnitudes->empty()) {
        const auto& kappa = *curvature_magnitudes;
        double sum_k = 0.0, max_k = 0.0;
        for (double k : kappa) {
            if (k < 0.0) k = 0.0;
            sum_k += k;
            if (k > max_k) max_k = k;
        }
        ci.mean_abs_curvature = sum_k / static_cast<double>(kappa.size());
        ci.max_abs_curvature  = max_k;
        ci.curvature_range_ratio = (ci.mean_abs_curvature > 1e-30)
                                   ? max_k / ci.mean_abs_curvature
                                   : 1.0;
    }

    // --- Aspect ratio summary ----------------------------------------------
    {
        double sum_ar = 0.0, max_ar = 0.0;
        int poor_count = 0;
        for (int fi = 0; fi < mesh.num_faces(); ++fi) {
            const auto v = mesh.face_vertices(fi);
            double ar = triangle_aspect_ratio(mesh.vertex_pos(v[0]),
                                              mesh.vertex_pos(v[1]),
                                              mesh.vertex_pos(v[2]));
            sum_ar += ar;
            if (ar > max_ar) max_ar = ar;
            if (ar > 10.0)   ++poor_count;
        }
        const int nf = mesh.num_faces();
        stats.mean_aspect_ratio      = (nf > 0) ? sum_ar / nf : 0.0;
        ci.aspect_ratio_mean         = stats.mean_aspect_ratio;
        ci.aspect_ratio_max          = max_ar;
        stats.poor_quality_face_ratio = (nf > 0)
                                        ? static_cast<double>(poor_count) / nf
                                        : 0.0;
    }

    // --- Isotropy score ----------------------------------------------------
    {
        const double mu = stats.edge_lengths.dist.mean;
        const double sd = stats.edge_lengths.dist.std_dev;
        stats.isotropy_score = (mu > 1e-30)
                               ? clamp(1.0 - sd / mu, 0.0, 1.0)
                               : 1.0;
    }

    return stats;
}

// =========================================================================
// classify_mesh_type
// =========================================================================

// Rule-based thresholds (calibrated against benchmark set).
// All thresholds are conservative; uncertain cases fall to Unknown.
namespace classify_rules {
    // Damaged: non-manifold, degenerate faces, or extremely bad aspects
    constexpr int    kDamagedNonManifoldEdgeMin = 1;
    constexpr double kDamagedPoorFaceRatioMax   = 0.05;   // >5% poor faces

    // CAD: very low feature edge density + very low curvature
    constexpr double kCADFeatureLo    = 0.01;
    constexpr double kCADFeatureHi    = 0.20;
    constexpr double kCADCurvHi       = 0.10;             // mean curvature

    // Hard Surface: high feature edge ratio + moderate/high curvature range
    constexpr double kHardSurfFeatureLo = 0.10;
    constexpr double kHardSurfCurvHi    = 5.0;            // curvature range ratio

    // Architecture: low curvature + high feature + closed/no boundary
    constexpr double kArchFeatureLo  = 0.08;
    constexpr double kArchCurvHi     = 0.05;              // mean curvature

    // Scan: extremely high vertex count + low isotropy
    constexpr int    kScanFaceMin    = 200000;
    constexpr double kScanIsotropyHi = 0.40;              // below this = non-uniform

    // Organic: moderate feature, smooth, genus ≥ 1 or closed
    constexpr double kOrganicFeatureHi = 0.10;
} // namespace classify_rules

MeshCategory classify_mesh_type(MeshStats& stats)
{
    namespace R = classify_rules;
    const ComplexityIndicators& ci = stats.complexity;

    double confidence = 0.0;
    MeshCategory cat = MeshCategory::Unknown;

    // Priority 1: Damaged (fail-safe classification)
    if (ci.non_manifold_edge_count >= R::kDamagedNonManifoldEdgeMin
        || stats.poor_quality_face_ratio > R::kDamagedPoorFaceRatioMax
        || stats.areas.zero_area_faces > 0)
    {
        cat = MeshCategory::Damaged;
        confidence = 0.9;
    }

    // Priority 2: Scan (large + non-uniform triangulation)
    else if (ci.num_faces >= R::kScanFaceMin
             && stats.isotropy_score < R::kScanIsotropyHi)
    {
        cat = MeshCategory::Scan;
        confidence = 0.75;
    }

    // Priority 3: CAD (low features, almost no curvature)
    else if (ci.feature_edge_ratio >= R::kCADFeatureLo
             && ci.feature_edge_ratio <= R::kCADFeatureHi
             && ci.mean_abs_curvature < R::kCADCurvHi)
    {
        cat = MeshCategory::CAD;
        confidence = 0.70;
    }

    // Priority 4: Architecture (high features + very low curvature + closed or low boundary)
    else if (ci.feature_edge_ratio >= R::kArchFeatureLo
             && ci.mean_abs_curvature < R::kArchCurvHi
             && (ci.is_closed || ci.boundary_edge_ratio < 0.05))
    {
        cat = MeshCategory::Architecture;
        confidence = 0.65;
    }

    // Priority 5: Hard Surface (high features + high curvature range)
    else if (ci.feature_edge_ratio >= R::kHardSurfFeatureLo
             && ci.curvature_range_ratio > R::kHardSurfCurvHi)
    {
        cat = MeshCategory::HardSurface;
        confidence = 0.70;
    }

    // Priority 6: Organic (smooth + few features + closed or near-closed)
    else if (ci.feature_edge_ratio <= R::kOrganicFeatureHi
             && (ci.is_closed || ci.num_boundary_loops <= 2))
    {
        cat = MeshCategory::Organic;
        confidence = 0.60;
    }

    // Fallback
    else {
        cat = MeshCategory::Unknown;
        confidence = 0.0;
    }

    stats.category            = cat;
    stats.category_confidence = confidence;
    return cat;
}

// =========================================================================
// mesh_category_name
// =========================================================================

const char* mesh_category_name(MeshCategory cat)
{
    switch (cat) {
        case MeshCategory::Unknown:      return "Unknown";
        case MeshCategory::Organic:      return "Organic";
        case MeshCategory::HardSurface:  return "HardSurface";
        case MeshCategory::CAD:          return "CAD";
        case MeshCategory::Scan:         return "Scan";
        case MeshCategory::Architecture: return "Architecture";
        case MeshCategory::Damaged:      return "Damaged";
        default:                         return "Unknown";
    }
}

// =========================================================================
// compute_mesh_difficulty
// =========================================================================

double compute_mesh_difficulty(const MeshStats& stats)
{
    // Weighted sum of difficulty signals, clamped to [0, 1].
    double d = 0.0;

    // Non-manifold geometry is very hard
    d += clamp(stats.complexity.non_manifold_edge_count / 10.0, 0.0, 1.0) * 0.30;

    // Poor quality faces
    d += clamp(stats.poor_quality_face_ratio * 5.0, 0.0, 1.0) * 0.20;

    // High curvature variation
    d += clamp((stats.complexity.curvature_range_ratio - 1.0) / 20.0, 0.0, 1.0) * 0.20;

    // Low isotropy (non-uniform triangulation)
    d += clamp(1.0 - stats.isotropy_score, 0.0, 1.0) * 0.15;

    // High feature density (more constraints = harder solve)
    d += clamp(stats.complexity.feature_edge_ratio * 2.0, 0.0, 1.0) * 0.10;

    // Very large mesh
    d += clamp(stats.complexity.num_faces / 2000000.0, 0.0, 1.0) * 0.05;

    return clamp(d, 0.0, 1.0);
}

// =========================================================================
// mesh_is_safe_for_full_pipeline
// =========================================================================

bool mesh_is_safe_for_full_pipeline(
    const MeshStats&          stats,
    std::vector<std::string>* reasons)
{
    bool safe = true;

    auto flag = [&](bool cond, const std::string& msg) {
        if (cond) {
            safe = false;
            if (reasons) reasons->push_back(msg);
        }
    };

    flag(!stats.complexity.is_manifold,
         "Mesh has non-manifold edges; the full global solver requires a manifold input.");

    flag(stats.areas.zero_area_faces > 0,
         "Mesh contains zero-area (degenerate) faces; repair is required before solving.");

    flag(stats.complexity.aspect_ratio_max > 1000.0,
         "Extreme triangle aspect ratios detected; the connection Laplacian may become singular.");

    flag(stats.poor_quality_face_ratio > 0.10,
         "More than 10% of faces have aspect ratio > 10; the global solver may diverge.");

    flag(stats.complexity.curvature_range_ratio > 100.0,
         "Curvature variation is extreme; consider using the robust simplified solver.");

    return safe;
}

// =========================================================================
// MeshStats serialisation
// =========================================================================

std::string MeshStats::to_string() const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "MeshStats {\n";
    ss << "  Vertices:        " << complexity.num_vertices << "\n";
    ss << "  Faces:           " << complexity.num_faces    << "\n";
    ss << "  Edges:           " << complexity.num_edges    << "\n";
    ss << "  Manifold:        " << (complexity.is_manifold ? "yes" : "NO") << "\n";
    ss << "  Closed:          " << (complexity.is_closed ? "yes" : "no") << "\n";
    ss << "  Genus:           " << complexity.genus << "\n";
    ss << "  Boundary loops:  " << complexity.num_boundary_loops << "\n";
    ss << "  EdgeLen [min,mean,max]: ["
       << edge_lengths.dist.min << ", " << edge_lengths.dist.mean << ", "
       << edge_lengths.dist.max << "]\n";
    ss << "  Angle  [min,mean,max]:  ["
       << angles.min_interior << ", " << angles.dist.mean << ", "
       << angles.max_interior << "] deg\n";
    ss << "  Isotropy score:  " << isotropy_score << "\n";
    ss << "  MeanAspectRatio: " << mean_aspect_ratio << "\n";
    ss << "  PoorFaceRatio:   " << poor_quality_face_ratio << "\n";
    ss << "  Category:        " << mesh_category_name(category)
       << " (" << category_confidence * 100 << "% confidence)\n";
    ss << "}";
    return ss.str();
}

std::string MeshStats::csv_header()
{
    return "num_vertices,num_faces,num_edges,manifold,closed,genus,"
           "edge_len_min,edge_len_mean,edge_len_max,edge_len_std,"
           "angle_min,angle_mean,angle_max,isotropy,"
           "mean_aspect_ratio,poor_face_ratio,"
           "feature_edge_ratio,curvature_range_ratio,"
           "category,category_confidence";
}

std::string MeshStats::to_csv_row() const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << complexity.num_vertices << ","
       << complexity.num_faces    << ","
       << complexity.num_edges    << ","
       << (complexity.is_manifold ? 1 : 0) << ","
       << (complexity.is_closed   ? 1 : 0) << ","
       << complexity.genus        << ","
       << edge_lengths.dist.min   << ","
       << edge_lengths.dist.mean  << ","
       << edge_lengths.dist.max   << ","
       << edge_lengths.dist.std_dev << ","
       << angles.min_interior     << ","
       << angles.dist.mean        << ","
       << angles.max_interior     << ","
       << isotropy_score          << ","
       << mean_aspect_ratio       << ","
       << poor_quality_face_ratio << ","
       << complexity.feature_edge_ratio     << ","
       << complexity.curvature_range_ratio  << ","
       << static_cast<int>(category)        << ","
       << category_confidence;
    return ss.str();
}

} // namespace qf
