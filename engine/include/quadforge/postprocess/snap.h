#pragma once
/* quadforge/postprocess/snap.h — Feature curve snapping.
 *
 * Snaps vertices and edges near detected feature curves onto those curves,
 * preserving sharp creases and corners that Taubin smoothing would blur.
 *
 * v77 fixes:
 *   - Added num_threads parameter (consistent with all other postprocess APIs).
 *   - Added adaptive_snap flag: when true, the effective snap radius at each
 *     vertex is snap_distance_scale * local_avg_edge_length.  This matches the
 *     roadmap requirement that "snapping uses a distance threshold proportional
 *     to the local edge length."  The old fixed-distance mode is kept as the
 *     default (adaptive_snap = false) for backward compatibility.
 */

#ifndef QUADFORGE_POSTPROCESS_SNAP_H
#define QUADFORGE_POSTPROCESS_SNAP_H

#include "../types.h"
#include "../mesh/halfedge.h"
#include "../mesh/features.h"

namespace qf {

/**
 * Snap quad mesh vertices near feature curves onto those curves.
 *
 * @param quad_mesh          Modified in place.
 * @param ref_mesh           Original triangle mesh (source of feature geometry)
 * @param features           Detected feature data from the input mesh
 * @param snap_distance      When adaptive_snap is false: maximum snap distance
 *                           in absolute world units.
 *                           When adaptive_snap is true: scale factor applied to
 *                           the per-vertex average incident edge length to derive
 *                           a local snap threshold (e.g. 0.5 = snap if within
 *                           half the local edge length of a feature curve).
 * @param adaptive_snap      When true, compute per-vertex snap threshold as
 *                           snap_distance * local_avg_edge_length.  This makes
 *                           snapping resolution-independent and proportional to
 *                           the local mesh density (roadmap Step 5.3 requirement).
 * @param num_threads        0 = OpenMP auto-detect.
 */
void snap_to_features(
    QuadMesh&           quad_mesh,
    const HalfEdgeMesh& ref_mesh,
    const FeatureData&  features,
    double              snap_distance,
    bool                adaptive_snap = false,
    int                 num_threads   = 0
);

} // namespace qf

#endif // QUADFORGE_POSTPROCESS_SNAP_H
