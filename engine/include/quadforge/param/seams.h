#pragma once
#ifndef QUADFORGE_PARAM_SEAMS_H
#define QUADFORGE_PARAM_SEAMS_H

#include <cstdint>
#include <vector>
#include <unordered_set>
#include <unordered_map>

// High-level wrappers require these forward types
#include "../types.h"           // EdgeSet, Vec3
#include "../mesh/halfedge.h"   // HalfEdgeMesh
#include "../mesh/features.h"   // FeatureData
#include "../field/singularity.h" // SingularityInfo
#include "combing.h"            // CombingResult

namespace qf {

/**
 * Seam-cut graph computation for global parametrization.
 *
 * After combing the 4-RoSy field, the surface is cut into a
 * topological disk by introducing seam edges.  Seams come from:
 *   1. Field combing inconsistencies (where the combing direction jumps)
 *   2. Singularity-connecting paths (Dijkstra on dual graph)
 *   3. User-specified UV seam edges
 *
 * The resulting cut graph must:
 *   - Connect all singularities to each other and to boundaries
 *   - Reduce the genus of the surface to 0 (so Poisson can solve globally)
 *   - Be as short as possible (to minimise parametrization distortion)
 */

/**
 * Compute the minimal seam cut graph.
 *
 * @param positions       [3*N] vertex positions
 * @param tris            [3*T]
 * @param nv, nt          Counts
 * @param combing_seams   Edges where the combing direction is inconsistent
 *                        (flat pairs [v0,v1,...])
 * @param singularity_verts  Vertices that are field singularities
 * @param edge_to_faces   Edge adjacency map
 * @param extra_seams     Additional seam edges (UV seams, etc.)
 * @param out_seams       Output: flat edge pairs [v0,v1,...] forming the
 *                        cut graph (may include combing_seams)
 */
void compute_seam_cut(
    const float*   positions,
    const int32_t* tris,
    int32_t nv, int32_t nt,
    const std::vector<int32_t>& combing_seams,
    const std::vector<int32_t>& singularity_verts,
    const std::unordered_map<int64_t, std::vector<int32_t>>& edge_to_faces,
    const std::vector<int32_t>& extra_seams,
    std::vector<int32_t>& out_seams
);

/**
 * Find the shortest path between two vertices on the mesh surface
 * (Dijkstra on primal graph, edge weight = Euclidean length).
 *
 * @param positions  [3*N]
 * @param nv         Vertex count
 * @param adj        Adjacency list: adj[v] = list of (neighbour, edge_len)
 * @param src, dst   Source and destination vertex indices
 * @param out_path   Output: vertex path from src to dst (inclusive)
 * @return           True if a path was found
 */
bool shortest_path_dijkstra(
    int32_t nv,
    const std::vector<std::vector<std::pair<int32_t, float>>>& adj,
    int32_t src, int32_t dst,
    std::vector<int32_t>& out_path
);

/**
 * Build vertex adjacency list from triangle mesh.
 */
std::vector<std::vector<std::pair<int32_t, float>>>
build_vertex_adjacency(
    const float* positions, int32_t nv,
    const int32_t* tris, int32_t nt
);

/**
 * Convert a vertex path to a flat edge list [v0,v1, v1,v2, ...].
 */
void path_to_edges(const std::vector<int32_t>& path,
                   std::vector<int32_t>& out_edges);

// -----------------------------------------------------------------------
// High-level HalfEdgeMesh-aware entry point  (FIX: BUG A + BUG B)
// -----------------------------------------------------------------------

/**
 * Compute the full seam-cut graph from HalfEdgeMesh + pipeline context.
 *
 * This is the primary entry point for engine.cpp's stage3_parametrize().
 * It replaces the legacy compute_minimal_seam_cut() call in combing.h and
 * properly wires in:
 *   - combing inconsistency seams (from CombingResult::seam_edges)
 *   - singularity-connecting Dijkstra paths (from SingularityInfo list)
 *   - UV seam edges (from FeatureData::uv_seam_edges, previously DROPPED)
 *   - hard-edge–derived seams (from FeatureData::hard_edges)
 *
 * BUG A FIX: this overload makes the compute_seam_cut() raw-array function
 *   reachable from the live pipeline path (it was dead code before v72).
 *
 * BUG B FIX: UV seam edges are now forwarded as extra_seams so that
 *   use_uv_seams=1 actually influences the parametrization seam cut.
 *
 * @param mesh          Triangle mesh (half-edge structure)
 * @param combing       Combed cross-field result (provides initial seam edges)
 * @param singularities Detected field singularities
 * @param features      Feature data including uv_seam_edges and hard_edges
 * @return              EdgeSet of all seam edges (deduplicated)
 */
EdgeSet compute_seam_cut_from_mesh(
    const HalfEdgeMesh&                     mesh,
    const CombingResult&                    combing,
    const std::vector<SingularityInfo>&     singularities,
    const FeatureData&                      features
);

} // namespace qf

#endif // QUADFORGE_PARAM_SEAMS_H
