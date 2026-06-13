#pragma once
#ifndef QUADFORGE_MESH_MESH_IO_H
#define QUADFORGE_MESH_MESH_IO_H

// api.h MUST be included before types.h here so QFResult is defined
// before it is referenced by pack_result() below.
#include "../api.h"
#include "../types.h"
#include "halfedge.h"
#include <cstdint>
#include <vector>

namespace qf {

/**
 * Import/export between the public C API flat arrays and the internal
 * half-edge mesh representation.
 */

/**
 * Triangulate an input mesh that may contain quads/polygons.
 *
 * @param positions     Input vertex positions [N*3]
 * @param faces_flat    Input face indices (packed, variable-length rows)
 * @param face_sizes    Number of vertices per face [F]
 * @param nv            Vertex count
 * @param nf            Face count
 * @param out_tris      Output: flat triangle index array [3*T], caller owns
 * @param out_nt        Output: number of triangles
 * @param poly_to_tri   Output: mapping from original polygon index to first
 *                      triangle index (size nf), for material transfer
 */
void triangulate_input(
    const float*   positions,
    const int32_t* faces_flat,
    const int32_t* face_sizes,
    int32_t nv, int32_t nf,
    std::vector<int32_t>& out_tris,
    int32_t&              out_nt,
    std::vector<int32_t>& poly_to_tri
);

/**
 * Convert a flat float position array [N*3] into a std::vector<Vec3>.
 *
 * This is the canonical conversion used by qfinput_to_halfedge() and by
 * engine.cpp when passing positions into any internal mesh operation.
 * Provided here so callers don't duplicate the cast/loop.
 *
 * @param positions  Flat float array [nv * 3]
 * @param nv         Vertex count
 * @return           Vector of nv Vec3 positions
 */
std::vector<Vec3> build_vertex_positions(
    const float*  positions,
    int32_t       nv
);

/**
 * One-shot conversion: QFInputMesh → triangulated HalfEdgeMesh.
 *
 * Combines triangulate_input() + build_vertex_positions() + HalfEdgeMesh
 * construction into a single call, which is the pattern every pipeline stage
 * needs.  The caller is responsible for running repair_mesh() and
 * detect_features() on the returned mesh before further processing.
 *
 * @param input         Source C-API mesh struct
 * @param out_tris      Output: flat [3*T] triangle indices (for downstream
 *                      use by curvature / field assembly that needs raw arrays)
 * @param out_poly_to_tri  Output: per-original-polygon first-triangle index
 *                         (for material-ID transfer back to QFResult)
 * @return              Fully constructed HalfEdgeMesh ready for the pipeline
 *
 * Lifetime contract: the returned HalfEdgeMesh owns all its data.  The
 * out_tris and out_poly_to_tri arrays remain valid until the caller clears
 * them; they do NOT alias any data inside the returned mesh.
 */
HalfEdgeMesh qfinput_to_halfedge(
    const QFInputMesh&     input,
    std::vector<int32_t>&  out_tris,
    std::vector<int32_t>&  out_poly_to_tri
);

/**
 * Compute per-face normals for a triangulated mesh.
 *
 * @param positions    [N*3]
 * @param tris         [3*T] flat triangle indices
 * @param nt           Triangle count
 * @param out_normals  Output [3*T] per-face unit normals
 */
void compute_face_normals(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    std::vector<float>& out_normals
);

/**
 * Compute per-vertex normals as angle-weighted average of adjacent faces.
 *
 * @param positions    [N*3]
 * @param tris         [3*T]
 * @param nt, nv       counts
 * @param face_normals [3*T] already computed face normals
 * @param out_normals  Output [3*N] per-vertex normals
 */
void compute_vertex_normals(
    const float*   positions,
    const int32_t* tris,
    int32_t nt, int32_t nv,
    const float* face_normals,
    std::vector<float>& out_normals
);

/**
 * Pack a result quad mesh back into flat C arrays for QFResult.
 *
 * @param quad_pos      [3*Vq] quad mesh positions (float)
 * @param quad_faces    [4*Fq] quad mesh faces (all quads)
 * @param tri_leftover  [3*Ft] leftover tri faces (may be empty)
 * @param result        Output struct (arrays allocated with new[])
 */
void pack_result(
    const std::vector<float>&   quad_pos,
    const std::vector<int32_t>& quad_faces,
    const std::vector<int32_t>& tri_leftover,
    QFResult* result
);

} // namespace qf

#endif // QUADFORGE_MESH_MESH_IO_H
