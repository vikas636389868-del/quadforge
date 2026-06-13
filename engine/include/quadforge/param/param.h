#pragma once
/* quadforge/param/param.h — Convenience umbrella header for the param subsystem.
 *
 * Includes all public param/ headers in dependency order:
 *
 *   metric.h    — Adaptive sizing field (no dependencies within param/)
 *   combing.h   — Cross-field combing + seam cuts (depends on field/)
 *   seams.h     — Full seam-cut graph computation (depends on combing.h)
 *   miq.h       — Mixed-Integer Quadrangulation (depends on combing.h)
 *   igm.h       — Integer Grid Maps (depends on miq.h, combing.h)
 *   poisson.h   — Poisson UV parametrization (standalone)
 *
 * Added in v72.  Previously absent — each translation unit had to include
 * individual headers.  The umbrella is provided for convenience and for
 * test files that exercise the full param pipeline.
 *
 * NOTE: engine.cpp continues to include individual headers (unchanged) for
 * clarity about which symbols each stage actually uses.
 */

#ifndef QUADFORGE_PARAM_PARAM_H
#define QUADFORGE_PARAM_PARAM_H

#include "metric.h"   // compute_sizing_field, compute_base_edge_length
#include "combing.h"  // comb_field, compute_minimal_seam_cut, CombingResult
#include "seams.h"    // compute_seam_cut, compute_seam_cut_from_mesh
#include "miq.h"      // compute_parametrization, ParamMethod, ParamParams
#include "igm.h"      // compute_igm_parametrization, IGMParams
#include "poisson.h"  // build_cotangent_laplacian, solve_poisson_parametrization

#endif // QUADFORGE_PARAM_PARAM_H
