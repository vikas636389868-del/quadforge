#pragma once
/* quadforge/field/singularity.h — Singularity detection and optimisation.
 *
 * For a 4-RoSy field, singularities are vertices where the holonomy
 * (total field rotation around the vertex) is not a multiple of π/2.
 * Index +1/4 → valence-3 vertex; index -1/4 → valence-5 vertex.
 */

#ifndef QUADFORGE_FIELD_SINGULARITY_H
#define QUADFORGE_FIELD_SINGULARITY_H

#include <vector>
#include <utility>
#include <unordered_map>
#include "../types.h"
#include "../mesh/halfedge.h"
// BUG FIX (v57): singularity.h declared CrossField in its function signatures
// but never included the header that defines it.  Any translation unit that
// included singularity.h without also including cross_field.h first would
// fail to compile with "unknown type CrossField".  Fixed by adding the missing
// include here so the header is self-contained.
#include "cross_field.h"
// BUG FIX (v94): The flat-array overload of detect_singularities() (added
// in v94) takes the edge_to_faces map whose type
// (std::unordered_map<int64_t, std::vector<int32_t>>) is returned by
// build_edge_to_faces() declared in constraints.h.  test_crossfield.cpp
// calls build_edge_to_faces() and passes the result to detect_singularities()
// after including only connection.h / cross_field.h / singularity.h — so
// constraints.h must be reachable through one of those three.  Including it
// here is the natural choice because the flat-array overload lives here.
#include "constraints.h"

namespace qf {

struct SingularityInfo {
    int    vertex_index;
    double index;       ///< +0.25 or -0.25
    Vec3   position;
};

/**
 * Detect singularities by computing holonomy around each interior vertex.
 * HalfEdgeMesh-based overload (preferred for pipeline use).
 */
std::vector<SingularityInfo> detect_singularities(
    const HalfEdgeMesh& mesh,
    const CrossField&   field
);

/**
 * Flat-array overload of detect_singularities().
 *
 * Detects 4-RoSy field singularities directly from raw triangle arrays
 * without requiring a fully constructed HalfEdgeMesh.  Used by unit tests
 * and tools that work with the lower-level connection Laplacian API.
 *
 * Algorithm (discrete holonomy, Knöppel 2013 §2):
 *   For each interior vertex v, traverse its one-ring of faces in order.
 *   Accumulate per-edge holonomy contribution:
 *     Δ_k = (face_angles[f_{k+1}] - face_angles[f_k]) - φ_{f_k→f_{k+1}}
 *   where φ_{ij} is the parallel transport angle computed from the tangent
 *   frames e1, e2.  Δ_k is rounded to the nearest multiple of π/2 (4-RoSy
 *   branch cut) before summing.  If the total holonomy H ≈ ±π/2, the vertex
 *   is a ±1/4 singularity.
 *
 * @param positions      [3*N] vertex XYZ
 * @param tris           [3*T] triangle vertex indices
 * @param nt             Triangle count T
 * @param nv             Vertex count N
 * @param face_angles    [T] per-face field angle θ_f = arg(u_f) / 4   (radians)
 * @param e1             [3*T] per-face local X-axis (from compute_face_frames_cpp)
 * @param e2             [3*T] per-face local Y-axis (from compute_face_frames_cpp)
 * @param face_normals   [3*T] per-face unit normals (used to guard degenerate faces)
 * @param edge_to_faces  Edge-to-face adjacency map from build_edge_to_faces()
 * @param out_sing_vertices  Output: vertex indices of singularities
 * @param out_sing_indices   Output: singularity index (+0.25 or -0.25) per entry
 *
 * BUG FIX (v94): this overload was entirely absent — test_crossfield.cpp
 * called it but no header declared it, causing a compilation failure.
 */
void detect_singularities(
    const float*   positions,
    const int32_t* tris,
    int32_t        nt,
    int32_t        nv,
    const float*   face_angles,
    const float*   e1,
    const float*   e2,
    const float*   face_normals,
    const std::unordered_map<int64_t, std::vector<int32_t>>& edge_to_faces,
    std::vector<int32_t>& out_sing_vertices,
    std::vector<float>&   out_sing_indices
);

/**
 * Optimise singularity placement:
 *   1. Cancel unnecessary +/- pairs (reduces irregular vertices)
 *   2. Relocate remaining singularities to feature corners
 *   3. Enforce symmetric placement if required
 *
 * @return Updated field with improved singularity positions.
 */
CrossField optimise_singularities(
    const HalfEdgeMesh&             mesh,
    const CrossField&               field,
    const std::vector<int>&         corner_vertices,
    bool                            enforce_symmetry = false
);

} // namespace qf

#endif // QUADFORGE_FIELD_SINGULARITY_H
