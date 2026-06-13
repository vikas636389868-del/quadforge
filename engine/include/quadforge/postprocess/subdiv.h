#pragma once
#ifndef QUADFORGE_POSTPROCESS_SUBDIV_H
#define QUADFORGE_POSTPROCESS_SUBDIV_H

#include <cstdint>
#include <vector>

namespace qf {

/**
 * Catmull-Clark subdivision compatibility utilities.
 *
 * Catmull-Clark subdivision requires:
 *   - All faces are quads (no triangles)
 *   - All interior vertices have valence 4 (irregular = not optimal)
 *   - All boundary vertices have valence 2 (on open meshes)
 *
 * This module checks if the output quad mesh is CC-subdiv compatible
 * and reports metrics that tell the user how "clean" the mesh is.
 */

struct SubdivCompatibilityReport {
    bool   all_quads;                // True if no triangle faces remain
    int32_t num_irregular_interior;  // Interior vertices with valence ≠ 4
    /**
     * Boundary vertices whose face-valence is > 2.
     *
     * NOTE (v78 bug-fix): the criterion is val > 2, NOT val ≠ 2.
     * Boundary vertices with face-valence 1 (natural mesh corners) and
     * face-valence 2 (regular boundary vertices) are both handled correctly
     * by the CC boundary rule V' = (6V + V_prev + V_next)/8 and are NOT
     * counted as irregular.  Only val > 2 (T-junctions, non-manifold
     * boundaries) represents a genuine subdivision problem.
     * The comment previously said "≠ 2" which was stale after the v78 fix.
     */
    int32_t num_irregular_boundary;
    float  irregular_ratio;          // irregular / total interior vertices
    bool   subdiv_ready;             // True if all_quads && irregular_ratio < 0.05

    // Valence histogram: valence_histogram[k] = number of vertices with valence k.
    // Index 0 = isolated vertex; index ≥ 8 is clamped to index 8.  Size = 9.
    std::vector<int32_t> valence_histogram; // indices 0..8
};

/**
 * Check Catmull-Clark compatibility of a quad mesh.
 *
 * BUG FIX (v94): positions changed from const float* to const double* to match
 * the engine-wide Scalar=double convention (types.h) and the v91 fix in
 * quad_mesh.h.  QuadMesh::vertices stores std::array<double,3>; callers that
 * build a flat positions array from QuadMesh::vertices were passing double data
 * through a float* pointer, causing silent precision truncation and potential
 * alignment faults on strict platforms.
 *
 * @param positions    [3*V]
 * @param quad_faces   [4*Fq] — all quad faces
 * @param tri_faces    [3*Ft] — remaining triangle faces
 * @param nv           Vertex count
 * @return             Compatibility report
 */
SubdivCompatibilityReport check_subdiv_compatibility(
    const double*  positions,
    const int32_t* quad_faces, int32_t nq,
    const int32_t* tri_faces,  int32_t nt,
    int32_t nv
);

/**
 * Attempt to convert triangle faces near singularities into quads
 * by merging adjacent triangle pairs.
 *
 * This is a last-pass cleanup that reduces the triangle count.
 * Should only be called after the motorcycle graph and cleanup stages.
 *
 * BUG FIX (v94): positions changed from const std::vector<float>& to
 * const std::vector<double>& to match the engine-wide Scalar=double
 * convention and the v91 fix in quad_mesh.h.  Callers pass positions
 * built from QuadMesh::vertices (std::array<double,3>); the previous
 * float signature caused an implicit narrowing that truncated position
 * data and produced incorrect merge decisions for vertices that are
 * close but not identical in double precision.
 *
 * @param quad_faces  [4*Fq] (appended to)
 * @param tri_faces   [3*Ft] (reduced)
 * @param positions   [3*V]
 */
void convert_tris_to_quads(
    std::vector<int32_t>& quad_faces,
    std::vector<int32_t>& tri_faces,
    const std::vector<double>& positions
);

/**
 * Apply one step of Catmull-Clark subdivision.
 * Useful for testing that the quad mesh is topologically valid.
 *
 * BUG FIX (v94): all position arrays changed from std::vector<float> to
 * std::vector<double> to match the engine-wide Scalar=double convention
 * and the v91 quad_mesh.h fix.  Passing QuadMesh::vertices (double) through
 * float vectors caused implicit double→float narrowing on input and
 * float→double widening on output, losing sub-millimetre precision and
 * producing visible artefacts after subdivision on meshes with tightly
 * packed vertices (e.g. high-curvature organic models).
 *
 * @param positions    [3*V] input
 * @param quad_faces   [4*F] input (tri_faces must be empty)
 * @param out_positions  [3*V'] output (4× more vertices)
 * @param out_quad_faces [4*F'] output (4× more quads)
 * @return True if subdivision succeeded
 */
bool catmull_clark_subdivide(
    const std::vector<double>&  positions,
    const std::vector<int32_t>& quad_faces,
    std::vector<double>&        out_positions,
    std::vector<int32_t>&       out_quad_faces
);

} // namespace qf

#endif // QUADFORGE_POSTPROCESS_SUBDIV_H
