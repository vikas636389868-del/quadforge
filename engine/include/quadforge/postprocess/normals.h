#pragma once
/* quadforge/postprocess/normals.h — Normal computation for the output quad mesh.
 *
 * Phase 5 roadmap deliverable:
 *   "Normal computation for the output mesh."
 *
 * Two levels of normal are computed:
 *
 *   1. Per-face normals (computed first):
 *      For quad faces, the normal is the cross product of the two diagonals
 *      divided by its length.  This is more robust than splitting the quad
 *      into two triangles and averaging, because it avoids visible normal
 *      seams caused by the arbitrary choice of diagonal split direction.
 *
 *        n = normalize( (v2-v0) × (v3-v1) )
 *
 *      For degenerate quads (zero-area diagonals), the normal defaults to
 *      (0,0,1) to avoid NaN propagation.
 *
 *      For leftover tri faces, the standard CCW triangle normal is used:
 *        n = normalize( (v1-v0) × (v2-v0) )
 *
 *   2. Per-vertex normals (computed from per-face normals):
 *      Each vertex normal is the weighted average of the face normals of all
 *      incident faces.  The weight for each face is the interior angle at
 *      that vertex (angle-weighted average, Thürmer & Wüthrich 1998), which
 *      produces smooth normals that correctly reflect sharp features without
 *      over-weighting large faces.
 *
 *      Boundary effect: leftover tri faces near singularities also contribute
 *      to vertex normals.  Excluding them previously caused visible shading
 *      discontinuities at the quad/tri boundary.
 *
 * Thread safety:
 *   Face normal computation is embarrassingly parallel (independent per face).
 *   Vertex normal accumulation must be serial (multiple faces write to the
 *   same vertex slot — race condition otherwise).  Vertex normal normalisation
 *   is embarrassingly parallel.
 *
 * Reference:
 *   Thürmer & Wüthrich (1998), "Computing vertex normals from polygonal
 *   facets." Journal of Graphics Tools, 3(1):43-46.
 */

#ifndef QUADFORGE_POSTPROCESS_NORMALS_H
#define QUADFORGE_POSTPROCESS_NORMALS_H

#include "../types.h"
#include <vector>

namespace qf {

/**
 * Compute per-vertex normals for a finished quad mesh.
 *
 * The returned vector has one entry per vertex in quad_mesh.vertices.
 * Each entry is a unit Vec3 normal (or (0,0,1) for isolated/degenerate
 * vertices).
 *
 * @param quad_mesh     Input mesh (read-only).
 * @param num_threads   0 = OpenMP auto-detect.
 * @return              Per-vertex normals, size == quad_mesh.vertices.size().
 */
std::vector<Vec3> compute_vertex_normals(
    const QuadMesh& quad_mesh,
    int             num_threads = 0
);

/**
 * Compute per-face normals for a finished quad mesh.
 *
 * @param quad_mesh     Input mesh (read-only).
 * @param num_threads   0 = OpenMP auto-detect.
 * @return              Per-quad normals (size == quad_mesh.quads.size())
 *                      followed by per-tri normals (size == quad_mesh.tris.size())
 *                      in a single flat vector.  Use indices [0, nq) for
 *                      quads and [nq, nq+nt) for tris.
 */
std::vector<Vec3> compute_face_normals(
    const QuadMesh& quad_mesh,
    int             num_threads = 0
);

/**
 * Pack per-vertex normals into a flat float array suitable for the C API.
 *
 * Writes 3*N floats into out_normals in (nx0,ny0,nz0, nx1,ny1,nz1, ...)
 * order.  out_normals must be pre-allocated to at least 3*N elements.
 *
 * @param vertex_normals    Result of compute_vertex_normals().
 * @param out_normals       Output flat float array, length 3*N.
 * @param num_threads       0 = OpenMP auto-detect.
 */
void pack_normals_to_float(
    const std::vector<Vec3>& vertex_normals,
    float*                   out_normals,
    int                      num_threads = 0
);

} // namespace qf

#endif // QUADFORGE_POSTPROCESS_NORMALS_H
