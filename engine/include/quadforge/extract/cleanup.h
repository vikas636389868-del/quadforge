#pragma once
/* quadforge/extract/cleanup.h — Post-extraction topology cleanup.
 *
 * Operations:
 *   - Degenerate face removal (zero-area quads)
 *   - Short edge collapse (below minimum length threshold)
 *   - Small hole filling
 *   - Edge flips to reduce valence irregularity
 *   - Doublet removal
 */

#ifndef QUADFORGE_EXTRACT_CLEANUP_H
#define QUADFORGE_EXTRACT_CLEANUP_H

#include "../types.h"

namespace qf {

struct CleanupParams {
    double min_edge_length_factor = 0.05; ///< Fraction of mean edge length
    int    max_valence_opt_passes = 3;
    bool   fill_holes             = true;
    bool   remove_doublets        = true;
};

/**
 * Clean up a raw quad mesh from extraction.
 *
 * @param quad_mesh   Input quad mesh (modified in place)
 * @param params      Cleanup configuration
 */
void cleanup_quad_mesh(QuadMesh& quad_mesh,
                       const CleanupParams& params = CleanupParams{});

} // namespace qf

#endif // QUADFORGE_EXTRACT_CLEANUP_H
