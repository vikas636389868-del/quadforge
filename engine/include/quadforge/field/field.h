#pragma once
/* quadforge/field/field.h — Convenience umbrella header for the field subsystem.
 *
 * Includes all public field/ headers in dependency order so that translation
 * units (tests, tools, external integrations) that need the full field API
 * can use a single include instead of enumerating individual headers.
 *
 * Dependency order within field/:
 *
 *   curvature.h     — Principal curvature computation (no field/ deps)
 *   connection.h    — Connection Laplacian + compute_face_frames_cpp (no field/ deps)
 *   constraints.h   — Feature/boundary/symmetry constraints (no field/ deps)
 *   cross_field.h   — 4-RoSy field solver (depends on curvature.h)
 *   singularity.h   — Singularity detection / optimisation
 *                     (depends on cross_field.h, constraints.h)
 *   smoothing.h     — Post-solve Gauss-Seidel smoothing
 *                     (depends on cross_field.h, constraints.h)
 *   field_debug.h   — Debug export utilities (depends on cross_field.h,
 *                     singularity.h; always compiled — no compile-time gate.
 *                     BUG FIX v96: the previous comment incorrectly stated
 *                     this header was "gated by QUADFORGE_ENABLE_DEBUG_EXPORTS".
 *                     That macro does not exist anywhere in the codebase.
 *                     field_debug.h is unconditionally included here and by
 *                     any consumer that includes field.h.)
 *
 * Added in v94.  Previously absent — only param/ had an umbrella header
 * (param.h, added in v72).  The absence forced every consumer (test files,
 * engine.cpp, api.cpp) to list individual field/ headers, and led directly
 * to test_crossfield.cpp missing constraints.h and therefore failing to
 * compile when build_edge_to_faces() was called.
 *
 * NOTE: engine.cpp and api.cpp continue to include individual headers for
 * clarity about which symbols each stage actually uses.  This umbrella is
 * primarily for test files and one-off tools.
 */

#ifndef QUADFORGE_FIELD_FIELD_H
#define QUADFORGE_FIELD_FIELD_H

// Level 0: no field/ dependencies
#include "curvature.h"    // CurvatureData, compute_curvature()
#include "connection.h"   // ConnectionLaplacian, compute_face_frames_cpp(),
                          // assemble_connection_laplacian(),
                          // solve_connection_laplacian_eigenvector()
#include "constraints.h"  // FieldConstraint, build_feature_edge_constraints(),
                          // build_boundary_constraints(),
                          // build_symmetry_constraints(),
                          // apply_constraints_to_laplacian(),
                          // build_edge_to_faces()

// Level 1: depends on curvature.h (via cross_field.h)
#include "cross_field.h"  // FieldSolver, CrossFieldParams, CrossField,
                          // compute_cross_field(),
                          // build_connection_laplacian(),
                          // parallel_transport_angle()

// Level 2: depends on cross_field.h and/or constraints.h
#include "singularity.h"  // SingularityInfo, detect_singularities() (both overloads),
                          // optimise_singularities()
#include "smoothing.h"    // FieldSmoothingParams, smooth_cross_field(),
                          // compute_cotangent_weights(), recompute_face_frames()

// Level 3: debug exports (depends on cross_field.h + singularity.h)
#include "field_debug.h"  // export_cross_field_vectors(), export_cross_field_mesh(),
                          // export_singularities_as_points(),
                          // export_field_quality_map(), cross_field_summary()

#endif // QUADFORGE_FIELD_FIELD_H
