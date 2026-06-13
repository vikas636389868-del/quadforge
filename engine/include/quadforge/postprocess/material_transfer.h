#pragma once
/* quadforge/postprocess/material_transfer.h — BVH-based material ID transfer.
 *
 * Phase 5 deliverable (Post-Processing, Phase 5 roadmap):
 *   "Material ID transfer from input faces to output faces (using spatial
 *    correspondence)."
 *
 * After quad extraction, the output QuadMesh has no material IDs — the
 * original per-face material assignments from Blender need to be propagated
 * to the output quads.  This module implements the transfer using a
 * nearest-triangle BVH query on each output quad's centroid.
 *
 * Algorithm:
 *   1. Build (or accept a pre-built) TriangleBVH over the input mesh.
 *   2. For each output quad face:
 *        a. Compute the centroid as the average of its four vertex positions.
 *        b. Query the BVH for the nearest input triangle.
 *        c. Look up the material ID via tri_to_original_face[hit.face_index],
 *           because the input may have been triangulated from polygons and the
 *           per-face material IDs refer to original polygon indices.
 *        d. Write the material ID into out_material_ids[qi].
 *   3. Leftover triangle faces (near singularities) are transferred via the
 *      same BVH query on their centroids.
 *
 * Cost: O((Fq + Ft) * log(Nt)) — negligible vs. the pipeline solve time.
 *
 * Thread safety: all output quad slots are independent, so the quad loop
 * runs in parallel with OpenMP.  The tri loop is typically very small
 * (< 1% of total faces) and runs serially to keep overhead low.
 */

#ifndef QUADFORGE_POSTPROCESS_MATERIAL_TRANSFER_H
#define QUADFORGE_POSTPROCESS_MATERIAL_TRANSFER_H

#include "../types.h"
#include "../mesh/spatial.h"

#include <cstdint>
#include <vector>

namespace qf {

/**
 * Transfer material IDs from an input triangle mesh to an output quad mesh.
 *
 * @param quad_mesh       Output quad mesh whose vertex positions are used to
 *                        compute quad/tri centroids.
 * @param bvh             TriangleBVH built over the input (triangulated) mesh.
 *                        Must be valid (bvh.is_valid() == true).
 * @param input_mat_ids   Per-triangle material ID array of size num_input_tris.
 *                        Entry [ti] is the material ID of input triangle ti.
 *                        Typically this is the material ID of the original
 *                        polygon from which triangle ti was produced by
 *                        triangulate_input().
 * @param num_input_tris  Length of input_mat_ids.
 * @param out_quad_mids   Output array for quad material IDs, size
 *                        quad_mesh.quads.size().  Caller must pre-allocate.
 *                        Written in parallel; entries for invalid quads are
 *                        left at 0.
 * @param out_tri_mids    Output array for leftover triangle material IDs, size
 *                        quad_mesh.tris.size().  May be empty if the mesh has
 *                        no leftover triangles.  Caller must pre-allocate.
 * @param num_threads     0 = OpenMP auto-detect.
 */
void transfer_material_ids(
    const QuadMesh&    quad_mesh,
    const TriangleBVH& bvh,
    const int32_t*     input_mat_ids,
    int32_t            num_input_tris,
    std::vector<int32_t>& out_quad_mids,
    std::vector<int32_t>& out_tri_mids,
    int                num_threads = 0
);

/**
 * Convenience overload that returns the two output vectors directly.
 *
 * @return pair.first  = material ID per quad face (size == quad_mesh.quads.size())
 *         pair.second = material ID per tri face  (size == quad_mesh.tris.size())
 *
 * Both vectors are zero-initialised; entries stay 0 if no material_ids are
 * available or if the BVH query finds no valid triangle.
 */
std::pair<std::vector<int32_t>, std::vector<int32_t>>
transfer_material_ids(
    const QuadMesh&    quad_mesh,
    const TriangleBVH& bvh,
    const int32_t*     input_mat_ids,
    int32_t            num_input_tris,
    int                num_threads = 0
);

} // namespace qf

#endif // QUADFORGE_POSTPROCESS_MATERIAL_TRANSFER_H
