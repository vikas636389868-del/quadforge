#pragma once
/* quadforge/extract/isolines.h — Integer UV iso-curve tracing on the mesh.
 *
 * Traces all integer U-lines and V-lines across the triangle mesh, then
 * constructs quad faces from the resulting grid cells.
 *
 * Reference: Ebke et al. (2013), "QEx: Robust Quad Mesh Extraction."
 *
 * Two variants are provided:
 *
 *   extract_quads_from_isolines()        — standard (raw UV, no seam handling)
 *   extract_quads_from_isolines_seam()   — seam-aware (applies period jumps
 *                                          before tracing, see seam_utils.h)
 *
 * For meshes produced by the Poisson-only parametrization (no MIQ), the two
 * variants are equivalent.  For MIQ-rounded UV with seam cuts, the seam-aware
 * variant must be used to obtain a correct iso-grid.
 */

#ifndef QUADFORGE_EXTRACT_ISOLINES_H
#define QUADFORGE_EXTRACT_ISOLINES_H

#include "../types.h"
#include "../mesh/halfedge.h"
#include "seam_utils.h"

namespace qf {

/**
 * Trace integer iso-lines and extract a quad mesh.
 *
 * Standard variant: uses raw UV coordinates from uv.U / uv.V without
 * applying any period jumps.  Correct for Poisson-only UV.  For MIQ
 * UV with seam cuts, use extract_quads_from_isolines_seam() instead.
 *
 * @param mesh         Input triangle mesh
 * @param uv           UV parametrization with per-vertex U, V coordinates
 * @param num_threads  Thread count (0 = OpenMP auto)
 */
QuadMesh extract_quads_from_isolines(
    const HalfEdgeMesh& mesh,
    const UVParam&      uv,
    int                 num_threads = 0
);

/**
 * Seam-aware variant: builds a PeriodJumpMap from uv.period_jumps and
 * calls seam_aware_uv() to produce globally consistent U/V values before
 * tracing.  Eliminates broken iso-lines and duplicate vertices near seams.
 *
 * Falls back silently to extract_quads_from_isolines() if uv.period_jumps
 * is empty (e.g., Poisson-only param).
 *
 * PRECONDITION: uv.seam_edges.size() == uv.period_jumps.size().
 * These are parallel arrays — seam_edges[k] carries the period jump
 * stored in period_jumps[k].  If uv.seam_edges is non-empty but
 * uv.period_jumps is empty (e.g., period jumps were not transferred from
 * CombingResult to UVParam), this function silently falls back to the
 * non-seam-aware tracer, producing broken iso-lines and missing/duplicate
 * quads near every seam — WITHOUT any error or warning.  The caller is
 * responsible for ensuring period_jumps is populated before this call
 * whenever seam_edges is non-empty.
 *
 * @param mesh         Input triangle mesh
 * @param uv           UV parametrization (must have seam_edges + period_jumps)
 * @param num_threads  Thread count (0 = OpenMP auto)
 */
QuadMesh extract_quads_from_isolines_seam(
    const HalfEdgeMesh& mesh,
    const UVParam&      uv,
    int                 num_threads = 0
);

} // namespace qf

#endif // QUADFORGE_EXTRACT_ISOLINES_H
