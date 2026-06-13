#pragma once
/* quadforge/mesh/topology.h — Topology analysis and manifold repair. */

#ifndef QUADFORGE_MESH_TOPOLOGY_H
#define QUADFORGE_MESH_TOPOLOGY_H

#include <vector>
#include <string>
#include "../types.h"
#include "halfedge.h"

namespace qf {

struct TopologyReport {
    int  num_vertices;
    int  num_edges;
    int  num_faces;
    int  euler_characteristic;
    int  genus;
    int  num_boundary_loops;
    int  num_connected_components;
    bool is_manifold;
    bool is_closed;
    int  num_non_manifold_edges;
    int  num_non_manifold_vertices;
    std::vector<std::pair<int,int>> non_manifold_edges;

    std::string to_string() const;
};

/** Analyse a half-edge mesh and return a topology report. */
TopologyReport analyse_topology(const HalfEdgeMesh& mesh);

/** Attempt to fix non-manifold geometry (edge splitting, face removal).
 *  Returns the number of issues fixed. */
int repair_non_manifold(HalfEdgeMesh& mesh);

/** Detect connected components; return per-vertex component id. */
std::vector<int> connected_components(const HalfEdgeMesh& mesh);

} // namespace qf

#endif // QUADFORGE_MESH_TOPOLOGY_H
