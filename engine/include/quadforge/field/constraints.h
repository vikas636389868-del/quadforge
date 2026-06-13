#pragma once
#ifndef QUADFORGE_FIELD_CONSTRAINTS_H
#define QUADFORGE_FIELD_CONSTRAINTS_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <complex>

namespace qf {  // unified namespace — was incorrectly "quadforge"

/**
 * Field constraints — hard and soft constraints on the 4-RoSy field.
 *
 * Constraints come from three sources:
 *   1. Feature edges  → field must align with edge direction
 *   2. Boundary edges → field must be perpendicular to boundary
 *   3. Symmetry       → field must be symmetric across a plane
 *
 * Constraints are integrated into the connection Laplacian as
 * penalty terms: each constrained face gets a large diagonal entry
 * pinning its field direction.
 */

struct FieldConstraint {
    int32_t face_idx;          // Face index that has this constraint
    double  target_angle;      // Target angle (in local frame) in radians
    double  weight;            // Penalty weight (higher = harder constraint)
};

/**
 * Extract feature-edge constraints from hard edges.
 *
 * For each hard edge (v0, v1), the two adjacent faces get constraints
 * that force one arm of their cross to align with the edge direction.
 *
 * @param positions      [3*N]
 * @param tris           [3*T]
 * @param nt             Triangle count
 * @param face_e1        [3*T] local X-axis per face
 * @param face_e2        [3*T] local Y-axis per face
 * @param face_normals   [3*T]
 * @param feature_edges  Flat edge pairs [v0,v1,...]
 * @param edge_to_faces  Map from sorted(v0,v1) → [face_i, face_j]
 * @param weight         Constraint penalty weight
 * @param out_constraints Output: list of FieldConstraint
 */
void build_feature_edge_constraints(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    const float* face_e1,
    const float* face_e2,
    const float* face_normals,
    const std::vector<int32_t>& feature_edges,
    const std::unordered_map<int64_t, std::vector<int32_t>>& edge_to_faces,
    double weight,
    std::vector<FieldConstraint>& out_constraints
);

/**
 * Extract boundary-edge constraints.
 *
 * Boundary faces get a constraint aligning the field with (or perpendicular
 * to) their boundary edge, depending on perpendicular_to_boundary.
 *
 * @param positions               [3*N]
 * @param tris                    [3*T]
 * @param nt                      Triangle count
 * @param face_e1                 [3*T] local X-axis per face
 * @param face_e2                 [3*T] local Y-axis per face
 * @param face_normals            [3*T]
 * @param boundary_edges          Flat boundary edge pairs [v0,v1,...] (boundary only)
 * @param perpendicular_to_boundary  True → field ⊥ boundary; False → field ∥ boundary
 * @param weight                  Constraint penalty weight
 * @param out_constraints         Output: list of FieldConstraint
 */
void build_boundary_constraints(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    const float* face_e1,
    const float* face_e2,
    const float* face_normals,
    const std::vector<int32_t>& boundary_edges,
    bool perpendicular_to_boundary,
    double weight,
    std::vector<FieldConstraint>& out_constraints
);

/**
 * Extract symmetry-plane constraints.
 *
 * Faces that lie on (or very near) a symmetry plane are constrained so that
 * one arm of the cross lies within the plane.
 *
 * @param positions        [3*N]
 * @param tris             [3*T]
 * @param nt               Triangle count
 * @param face_e1          [3*T] local X-axis per face
 * @param face_e2          [3*T] local Y-axis per face
 * @param face_normals     [3*T]
 * @param sym_plane_normal [3] unit normal of the symmetry plane (through origin)
 * @param weight           Constraint penalty weight (typically 0.5× feature weight)
 * @param out_constraints  Output: list of FieldConstraint
 */
void build_symmetry_constraints(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    const float* face_e1,
    const float* face_e2,
    const float* face_normals,
    const float* sym_plane_normal,
    double weight,
    std::vector<FieldConstraint>& out_constraints
);

/**
 * Apply constraints to the connection Laplacian in COO format.
 *
 * For each FieldConstraint, appends a diagonal penalty entry to the COO
 * triplet list:
 *   L[fi, fi] += weight
 * and adds the matching right-hand side contribution:
 *   rhs[fi]   += weight * exp(4i * target_angle)
 *
 * Derivation (Wirtinger calculus on penalty energy):
 *   Minimize  u^H L u  +  weight * |u_f - target_f|^2
 *   d/d(conj(u_f)) →  (L*u)_f  +  weight*(u_f - target_f) = 0
 *   →  (L_ff + weight) * u_f = weight * target_f
 *   →  rhs[fi] = weight * exp(4i * target_angle)   (NO conj on target)
 *
 * BUG FIX (v97): the previous formula wrote
 *   rhs[fi] += weight * conj(exp(4i * target_angle))
 * which equals weight * exp(-4i * target_angle) and mirrors every constraint
 * angle through zero.  All feature-alignment constraints would push the field
 * 90° in the wrong direction.  The conj() has been removed.
 *
 * Callers must size rhs to at least (max face_idx + 1) before calling.
 *
 * @param constraints   The constraints to apply
 * @param coo_rows      COO row indices (appended to in place)
 * @param coo_cols      COO column indices (appended to in place)
 * @param coo_vals      COO complex values (appended to in place)
 * @param rhs           Per-face RHS penalty vector (complex, size ≥ F)
 */
void apply_constraints_to_laplacian(
    const std::vector<FieldConstraint>& constraints,
    std::vector<int32_t>&               coo_rows,
    std::vector<int32_t>&               coo_cols,
    std::vector<std::complex<double>>&  coo_vals,
    std::vector<std::complex<double>>&  rhs
);

/**
 * Build the edge-to-faces adjacency map for fast constraint lookup.
 * Key = min(v0,v1) * VMAX + max(v0,v1)  where VMAX = (int64_t)1 << 24
 * This packs both vertex indices into a 64-bit integer: the lower 24 bits
 * hold max(v0,v1) and the upper bits hold min(v0,v1).  VMAX = 1<<24 means
 * vertex indices must be < 16,777,216 (16.7 M) — sufficient for all practical
 * meshes.  Note: 32-bit indices would OVERFLOW this encoding (a 32-bit value
 * shifted by 24 bits requires 56 bits, not 48).  The 24-bit limit is a design
 * choice, NOT a consequence of indices being 32-bit.
 * BUG FIX (v96): the previous comment claimed "vertex indices are 32-bit, so
 * 24 bits is the safe shift" — the logic was completely backwards.  32-bit
 * vertex indices require a LARGER slot (≥32 bits each), not a smaller one.
 * Value = list of face indices sharing that edge (1 for boundary, 2 interior).
 *
 * BUG FIX (v60): previous doc comment stated "min<<32|max" which does not match
 * the actual VMAX = 1<<24 used everywhere in this file and in cross_field.cpp,
 * connection.cpp, and singularity.cpp.  Any external caller computing keys
 * manually from the header doc would produce keys that never match.
 */
std::unordered_map<int64_t, std::vector<int32_t>>
build_edge_to_faces(const int32_t* tris, int32_t nt);

} // namespace qf  // was "quadforge" — unified to qf

#endif // QUADFORGE_FIELD_CONSTRAINTS_H
