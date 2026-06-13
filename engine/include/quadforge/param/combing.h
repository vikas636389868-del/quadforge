#pragma once
/* quadforge/param/combing.h — Cross-field combing.
 *
 * Resolves the 4-fold ambiguity of a 4-RoSy field by choosing a globally
 * consistent orientation via BFS.  Introduces seam cuts where the combing
 * is necessarily inconsistent (topology requires it).
 */

#ifndef QUADFORGE_PARAM_COMBING_H
#define QUADFORGE_PARAM_COMBING_H

#include <vector>
#include "../types.h"
#include "../mesh/halfedge.h"
#include "../mesh/features.h"
#include "../field/singularity.h"

namespace qf {

/** Result of the combing process. */
struct CombingResult {
    std::vector<double>         face_angles; ///< Per-face combed angle
    EdgeSet                     seam_edges;  ///< Seam-cut edges
    std::vector<int>            period_jumps;///< Integer period jump per seam edge
};

/**
 * Comb the 4-RoSy field to a globally consistent orientation.
 *
 * The seam cut graph connects all singularities and boundary loops.
 * Seam edges carry an integer period jump (0, 1, 2, or 3).
 *
 * @param mesh      Triangle mesh
 * @param field     Input 4-RoSy cross-field
 * @param features  Feature edges (used to guide seam placement)
 */
CombingResult comb_field(
    const HalfEdgeMesh& mesh,
    const CrossField&   field,
    const FeatureData&  features
);

/**
 * Compute the minimal seam-cut graph connecting all singularities.
 * Uses shortest-path tree on the dual graph weighted by edge lengths.
 */
EdgeSet compute_minimal_seam_cut(
    const HalfEdgeMesh&             mesh,
    const std::vector<SingularityInfo>& singularities
);

} // namespace qf

#endif // QUADFORGE_PARAM_COMBING_H
