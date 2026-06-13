#pragma once
#ifndef QUADFORGE_EXTRACT_QUAD_MESH_H
#define QUADFORGE_EXTRACT_QUAD_MESH_H

#include <cstdint>
#include <vector>
#include <unordered_map>

namespace qf {

/**
 * Quad mesh construction from UV iso-line intersections.
 *
 * After tracing integer U and V iso-lines on the triangle mesh,
 * this module constructs the actual quad faces by:
 *   1. Collecting all iso-line intersection points (U-line ∩ V-line)
 *   2. Grouping them into integer (U, V) grid cells
 *   3. Creating a quad face for each grid cell with four corner points
 *   4. Connecting adjacent quads via shared edges
 */

/**
 * A point where an integer U-line and integer V-line intersect.
 *
 * BUG FIX (v91): pos[] changed from float[3] to double[3] to match the
 * engine-wide Scalar=double convention (see types.h).  Using float caused
 * silent precision loss when positions were later stored in QuadMesh::vertices
 * (which is std::vector<std::array<double,3>>).
 */
struct IsoIntersection {
    int32_t u_int;     // Integer U value
    int32_t v_int;     // Integer V value
    double  pos[3];    // 3D position (double to match engine Scalar type)
    int32_t vertex_id; // Index in the output quad mesh vertex array
};

/**
 * Construct quad faces from iso-line intersection points.
 *
 * BUG FIX (v91): Removed stale @param entries for u_intersections and
 * v_intersections which referenced parameters that no longer exist in the
 * function signature.  The current API takes only all_intersections.
 *
 * BUG FIX (v91): out_positions changed from std::vector<float> to
 * std::vector<double> to match the engine-wide Scalar=double convention.
 *
 * @param all_intersections All UV grid intersection points
 * @param out_positions    Output: quad mesh vertex positions [3*Vq]
 * @param out_quad_faces   Output: flat quad face indices [4*Fq]
 * @param out_tri_faces    Output: remaining triangle faces [3*Ft]
 *                         (near singularities where grid cells are triangular)
 */
void construct_quad_mesh(
    const std::vector<IsoIntersection>& all_intersections,
    std::vector<double>&  out_positions,
    std::vector<int32_t>& out_quad_faces,
    std::vector<int32_t>& out_tri_faces
);

/**
 * Merge duplicate vertices in the quad mesh within a tolerance.
 *
 * BUG FIX (v91): positions changed from std::vector<float> to
 * std::vector<double> and tol from float to double.  Using float with
 * tol=1e-6f was unreliable on large meshes where positions ~100 units give
 * relative precision of only 1e-8 — below float's representable range.
 *
 * @param positions     [3*V] vertex positions (modified in-place)
 * @param quad_faces    [4*F] quad faces (vertex indices updated)
 * @param tri_faces     [3*T] triangle faces (vertex indices updated)
 * @param nv            Vertex count (updated to new count after merging)
 * @param tol           Merge tolerance (default: 1e-6)
 */
void weld_vertices(
    std::vector<double>&  positions,
    std::vector<int32_t>& quad_faces,
    std::vector<int32_t>& tri_faces,
    int32_t& nv,
    double tol = 1e-6
);

/**
 * Remove degenerate faces (zero area, repeated vertex indices).
 *
 * BUG FIX (v91): positions changed from const std::vector<float>& to
 * const std::vector<double>&; min_area from float to double.
 *
 * @param positions    [3*V]
 * @param quad_faces   [4*F] — faces with area < min_area are removed
 * @param tri_faces    [3*T] — same
 * @param min_area     Area threshold (default: 1e-12)
 */
void remove_degenerate_faces(
    const std::vector<double>& positions,
    std::vector<int32_t>& quad_faces,
    std::vector<int32_t>& tri_faces,
    double min_area = 1e-12
);

/**
 * Reindex vertices: remove unreferenced vertices, compact the array.
 *
 * BUG FIX (v91): positions changed from std::vector<float>& to
 * std::vector<double>&.
 *
 * @param positions    [3*V] — compacted in-place
 * @param quad_faces   [4*F] — indices updated
 * @param tri_faces    [3*T] — indices updated
 * @param nv           Updated vertex count
 */
void compact_vertices(
    std::vector<double>&  positions,
    std::vector<int32_t>& quad_faces,
    std::vector<int32_t>& tri_faces,
    int32_t& nv
);

/**
 * Compute signed area of a quad face (sum of two triangle areas).
 * Returns < 0 for inverted (flipped normal) faces.
 *
 * BUG FIX (v91): changed from float to double to match engine precision.
 */
double quad_signed_area(
    const double* p0, const double* p1,
    const double* p2, const double* p3
);

} // namespace qf

#endif // QUADFORGE_EXTRACT_QUAD_MESH_H
