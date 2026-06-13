#pragma once
/* quadforge/types.h — Internal C++ types used throughout the engine.
 *
 * Only included from C++ translation units.  Never included in api.h.
 */

#ifndef QUADFORGE_TYPES_H
#define QUADFORGE_TYPES_H

#include <cstdint>
#include <string>
#include <vector>
#include <complex>
#include <functional>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>

namespace qf {

// -----------------------------------------------------------------------
// Scalar & vector typedefs
// -----------------------------------------------------------------------
using Scalar   = double;
using Vec2     = Eigen::Matrix<Scalar, 2, 1>;
using Vec3     = Eigen::Matrix<Scalar, 3, 1>;
using Vec4     = Eigen::Matrix<Scalar, 4, 1>;
using Mat3     = Eigen::Matrix<Scalar, 3, 3>;
using MatX     = Eigen::MatrixXd;
using VecX     = Eigen::VectorXd;
using VecXf    = Eigen::VectorXf;
using SparseMat = Eigen::SparseMatrix<Scalar>;
using SparseMatC = Eigen::SparseMatrix<std::complex<Scalar>>;
using Triplet  = Eigen::Triplet<Scalar>;
using TripletC = Eigen::Triplet<std::complex<Scalar>>;

// -----------------------------------------------------------------------
// Progress callback type (same semantics as QFProgressCallback in api.h)
// -----------------------------------------------------------------------
using ProgressFn = std::function<bool(int stage, float progress, const std::string& name)>;

// -----------------------------------------------------------------------
// QFEngine forward declarations (full definition in engine.h)
// -----------------------------------------------------------------------
class HalfEdgeMesh;
class TriangleBVH;

// -----------------------------------------------------------------------
// Pipeline intermediate data
// -----------------------------------------------------------------------

/** Feature edge set: sorted pairs (v_lo, v_hi). */
using EdgeSet = std::vector<std::pair<int,int>>;

/** Cross-field: per-face complex representation exp(4iθ). */
struct CrossField {
    Eigen::VectorXcd    face_field;    // F complex values
    std::vector<double> face_frames;   // F reference angles
    std::vector<int>    singularities; // vertex indices of singularities
    // BUG FIX (v97): was std::vector<int>, which truncated ±0.25 to 0 and
    // silently discarded every singularity index.  The 4-RoSy singularity
    // index is a fractional value: +0.25 for a +1/4 singularity (valence-3
    // vertex) and -0.25 for a -1/4 singularity (valence-5 vertex).
    // Storing these in an int vector is a silent data loss bug.
    // Changed to std::vector<double> to match SingularityInfo::index (double)
    // and the flat-array detect_singularities() output (std::vector<float>).
    std::vector<double> sing_indices;  // ±0.25 fractional index per singularity
};

/** UV parametrization per triangle mesh vertex. */
struct UVParam {
    Eigen::VectorXd  U;            ///< U values per vertex [nv]
    Eigen::VectorXd  V;            ///< V values per vertex [nv]
    EdgeSet          seam_edges;   ///< Seam-cut edge pairs (v_lo, v_hi)

    /**
     * Integer period jumps per seam edge (parallel array to seam_edges).
     *
     * period_jumps[k] = (Δu, Δv) for seam_edges[k] = (v_lo, v_hi), meaning:
     *
     *     U[v_hi] ≈ U[v_lo] + Δu
     *     V[v_hi] ≈ V[v_lo] + Δv
     *
     * Populated by the parametrization stage (igm.cpp / miq.cpp / combing.cpp)
     * and consumed by the extraction stage via seam_utils::build_period_jump_map().
     *
     * Added in v89: previously the period-jump data lived only in
     * CombingResult::period_jumps and was never transferred to UVParam,
     * leaving extract_quads_from_isolines() blind to seam crossings.
     */
    std::vector<std::pair<int,int>> period_jumps; ///< (Δu, Δv) per seam edge
};

/** Output quad mesh before Blender conversion. */
struct QuadMesh {
    std::vector<std::array<double,3>> vertices;
    std::vector<std::array<int,4>>    quads;
    std::vector<std::array<int,3>>    tris;     // near-singularity leftovers
    std::vector<int>                  material_ids;
    float quad_percentage{0.f};
    float avg_valence{0.f};
    float elapsed_s{0.f};
};

} // namespace qf

#endif // QUADFORGE_TYPES_H
