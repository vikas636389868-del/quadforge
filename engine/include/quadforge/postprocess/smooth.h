#pragma once
/* quadforge/postprocess/smooth.h — Taubin λ/μ bilaplacian smoothing.
 *
 * Applies shrink-free Laplacian smoothing (Taubin 1995) to the quad mesh.
 * Each iteration is embarrassingly parallel — all vertex updates are
 * independent within a single pass.
 *
 * Reference: G. Taubin, "A signal processing approach to fair surface
 * design," SIGGRAPH 1995.
 */

#ifndef QUADFORGE_POSTPROCESS_SMOOTH_H
#define QUADFORGE_POSTPROCESS_SMOOTH_H

#include "../types.h"

namespace qf {

struct SmoothParams {
    int    iterations    = 10;
    double lambda        =  0.5;   ///< Positive step (shrinks)
    double mu            = -0.53;  ///< Negative step (counter-shrinks)
    int    num_threads   = 0;
    /**
     * When true (default), boundary vertices — those incident to at least one
     * edge that appears in only one face — are pinned and not displaced.
     * This prevents Taubin smoothing from pulling open-mesh boundaries
     * inward and avoids geometry drift at the seam between remeshed and
     * preserved regions.
     */
    bool   lock_boundary = true;
};

/**
 * Taubin λ/μ smoothing.
 *
 * @param quad_mesh   Modified in place.
 */
void taubin_smooth(QuadMesh& quad_mesh,
                   const SmoothParams& params = SmoothParams{});

} // namespace qf

#endif // QUADFORGE_POSTPROCESS_SMOOTH_H
