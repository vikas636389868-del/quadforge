/**
 * material_transfer.cpp — BVH-based material ID transfer from input triangles
 * to output quad/tri faces.
 *
 * Phase 5 roadmap deliverable:
 *   "Material ID transfer from input faces to output faces (using spatial
 *    correspondence)."
 *
 * This is a self-contained postprocess module.  engine.cpp also contains an
 * inline version of this logic (v84 fix in stage6_output()) that operates
 * directly on internal engine state.  This standalone module exposes the same
 * algorithm as a reusable API callable by tests, the Python layer, or any
 * future pipeline stage that needs material transfer without access to engine
 * internals.
 *
 * Algorithm overview:
 *   For each output face (quad or leftover tri):
 *     1. Compute the face centroid by averaging the positions of its vertices.
 *     2. Query bvh.closest_point(centroid) to find the nearest input triangle.
 *     3. Copy input_mat_ids[hit.face_index] to the output material ID slot.
 *
 * Parallelism:
 *   The quad loop is embarrassingly parallel (each quad's centroid query is
 *   independent) and is run with parallel_for.  The leftover tri loop is
 *   typically tiny (< 1% of faces for a good remesh) and runs serially.
 *
 * Safety:
 *   - All vertex index accesses are bounds-checked before use.
 *   - BVH validity is checked before any query.
 *   - hit.face_index is validated against num_input_tris before lookup.
 *   - Output vectors are pre-zero-initialised so missing transfers leave 0.
 */

#include "../../include/quadforge/postprocess/material_transfer.h"
#include "../../include/quadforge/accel/omp_utils.h"

#include <algorithm>
#include <utility>

namespace qf {

// ---------------------------------------------------------------------------
// Internal: compute centroid of a quad face
// ---------------------------------------------------------------------------
static Vec3 quad_centroid(const QuadMesh& qm, int fi)
{
    Vec3 c{0.0, 0.0, 0.0};
    const int nv = static_cast<int>(qm.vertices.size());
    int count = 0;
    for (int k = 0; k < 4; ++k) {
        int vi = qm.quads[fi][k];
        if (vi >= 0 && vi < nv) {
            c[0] += qm.vertices[vi][0];
            c[1] += qm.vertices[vi][1];
            c[2] += qm.vertices[vi][2];
            ++count;
        }
    }
    if (count > 0) {
        double inv = 1.0 / static_cast<double>(count);
        c *= inv;
    }
    return c;
}

// ---------------------------------------------------------------------------
// Internal: compute centroid of a tri face
// ---------------------------------------------------------------------------
static Vec3 tri_centroid(const QuadMesh& qm, int fi)
{
    Vec3 c{0.0, 0.0, 0.0};
    const int nv = static_cast<int>(qm.vertices.size());
    int count = 0;
    for (int k = 0; k < 3; ++k) {
        int vi = qm.tris[fi][k];
        if (vi >= 0 && vi < nv) {
            c[0] += qm.vertices[vi][0];
            c[1] += qm.vertices[vi][1];
            c[2] += qm.vertices[vi][2];
            ++count;
        }
    }
    if (count > 0) {
        double inv = 1.0 / static_cast<double>(count);
        c *= inv;
    }
    return c;
}

// ---------------------------------------------------------------------------
// transfer_material_ids — primary overload (pre-allocated output vectors)
// ---------------------------------------------------------------------------
void transfer_material_ids(
    const QuadMesh&    qm,
    const TriangleBVH& bvh,
    const int32_t*     input_mat_ids,
    int32_t            num_input_tris,
    std::vector<int32_t>& out_quad_mids,
    std::vector<int32_t>& out_tri_mids,
    int                num_threads)
{
    // Early-out guards
    if (!bvh.is_valid() || input_mat_ids == nullptr || num_input_tris <= 0) {
        // Leave output vectors in their current state (caller pre-zero-inits).
        return;
    }

    const int nqf = static_cast<int>(qm.quads.size());
    const int ntf = static_cast<int>(qm.tris.size());

    // Ensure output vectors are large enough; zero-init any new slots.
    out_quad_mids.assign(static_cast<size_t>(nqf), 0);
    out_tri_mids.assign(static_cast<size_t>(ntf),  0);

    if (nqf == 0 && ntf == 0) return;

    // ------------------------------------------------------------------
    // Quad faces — parallel BVH query
    // ------------------------------------------------------------------
    parallel_for(nqf, [&](int qi) {
        Vec3 centroid = quad_centroid(qm, qi);
        auto hit = bvh.closest_point(centroid);
        if (hit.face_index >= 0 && hit.face_index < num_input_tris)
            out_quad_mids[qi] = input_mat_ids[hit.face_index];
        // else: slot stays 0 (safe default)
    }, num_threads);

    // ------------------------------------------------------------------
    // Leftover tri faces — serial (typically very few)
    // ------------------------------------------------------------------
    for (int ti = 0; ti < ntf; ++ti) {
        Vec3 centroid = tri_centroid(qm, ti);
        auto hit = bvh.closest_point(centroid);
        if (hit.face_index >= 0 && hit.face_index < num_input_tris)
            out_tri_mids[ti] = input_mat_ids[hit.face_index];
    }
}

// ---------------------------------------------------------------------------
// transfer_material_ids — convenience overload (returns vectors)
// ---------------------------------------------------------------------------
std::pair<std::vector<int32_t>, std::vector<int32_t>>
transfer_material_ids(
    const QuadMesh&    qm,
    const TriangleBVH& bvh,
    const int32_t*     input_mat_ids,
    int32_t            num_input_tris,
    int                num_threads)
{
    std::pair<std::vector<int32_t>, std::vector<int32_t>> result;
    transfer_material_ids(qm, bvh, input_mat_ids, num_input_tris,
                          result.first, result.second, num_threads);
    return result;
}

} // namespace qf
