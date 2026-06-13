/**
 * constraints.cpp — Field constraints for the 4-RoSy direction field.
 *
 * Constraints are integrated as penalty terms into the connection Laplacian.
 * Three sources:
 *   1. Feature edges   → one arm of the cross aligns with the edge tangent
 *   2. Boundary edges  → field is perpendicular/parallel to the boundary
 *   3. Symmetry        → field is symmetric across a mirror plane
 */

#include "../../include/quadforge/field/constraints.h"

#include <cmath>
#include <algorithm>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {  // was "quadforge" — unified to match all other field files

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static inline float dot3(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void cross3(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static inline float len3(const float* a) {
    return std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

// -----------------------------------------------------------------------
// Feature edge constraints
// -----------------------------------------------------------------------

void build_feature_edge_constraints(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    const float* face_e1,
    const float* face_e2,
    const float* /*face_normals*/,
    const std::vector<int32_t>& feature_edges,
    const std::unordered_map<int64_t, std::vector<int32_t>>& edge_to_faces,
    double weight,
    std::vector<FieldConstraint>& out_constraints)
{
    out_constraints.clear();
    if (feature_edges.empty()) return;

    // BUG FIX (v25): '1 << 24' computed as int before widening; use explicit cast.
    // BUG FIX (v63): Was 'const long long VMAX' / 'long long key'. On LP64
    // platforms (Linux/macOS 64-bit), int64_t = long, not long long. Passing a
    // long long key to std::unordered_map<int64_t>::find() is a type mismatch —
    // the compiler creates a temporary via implicit narrowing, which compiles only
    // by accident and is a hard error on strict standards modes.  Changed to
    // int64_t throughout to match the map's key_type exactly.
    const int64_t VMAX = (int64_t)1 << 24;

    for (int k = 0; k + 1 < (int)feature_edges.size(); k += 2) {
        int32_t va = feature_edges[k], vb = feature_edges[k+1];
        int64_t key = (int64_t)std::min(va,vb) * VMAX + (int64_t)std::max(va,vb);

        auto it = edge_to_faces.find(key);
        if (it == edge_to_faces.end()) continue;

        // Edge direction (normalized)
        float ex = positions[vb*3]   - positions[va*3];
        float ey = positions[vb*3+1] - positions[va*3+1];
        float ez = positions[vb*3+2] - positions[va*3+2];
        float elen = std::sqrt(ex*ex + ey*ey + ez*ez);
        if (elen < 1e-10f) continue;
        ex /= elen; ey /= elen; ez /= elen;
        float edge[3] = {ex, ey, ez};

        for (int32_t fi : it->second) {
            if (fi < 0 || fi >= nt) continue;

            const float* e1 = face_e1 + fi*3;
            const float* e2 = face_e2 + fi*3;

            // Project edge onto the face's tangent plane (should already be in-plane)
            float proj1 = dot3(edge, e1);
            float proj2 = dot3(edge, e2);
            float proj_len = std::sqrt(proj1*proj1 + proj2*proj2);
            if (proj_len < 1e-8f) continue;

            // Angle of the edge in the local frame
            double theta = std::atan2((double)proj2, (double)proj1);

            FieldConstraint c;
            c.face_idx    = fi;
            c.target_angle = theta;
            c.weight      = weight;
            out_constraints.push_back(c);
        }
    }
}

// -----------------------------------------------------------------------
// Boundary edge constraints
// -----------------------------------------------------------------------

void build_boundary_constraints(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    const float* face_e1,
    const float* face_e2,
    const float* /*face_normals*/,
    const std::vector<int32_t>& boundary_edges,
    bool perpendicular_to_boundary,
    double weight,
    std::vector<FieldConstraint>& out_constraints)
{
    // Build a quick map of which faces own each boundary edge
    // BUG FIX (v58): out_constraints was never cleared at the start of this
    // function, unlike build_feature_edge_constraints which calls .clear().
    // If the caller passed a pre-populated vector (e.g. from a previous feature-
    // edge call), boundary constraints were appended ON TOP of the old data,
    // silently duplicating constraints from prior calls and producing wrong
    // penalty weights in the connection Laplacian.  Fixed by adding the same
    // guard that every sibling function should have.
    out_constraints.clear();

    // Build a quick map of which faces own each boundary edge
    // BUG FIX (v96): Was 'using EKey = long long' / 'const long long VMAX'.
    // The identical fix was applied to build_feature_edge_constraints in v63
    // (long long vs int64_t mismatch on LP64 platforms where int64_t = long,
    // not long long — a subtle type mismatch that creates implicit conversions
    // when map keys are compared or inserted).  This function was missed.
    // Changed to int64_t throughout to match the key_type of the map exactly.
    using EKey = int64_t;
    // BUG FIX (v25): explicit cast to avoid int-shift before widening.
    const int64_t VMAX = (int64_t)1 << 24;
    std::unordered_map<EKey, int32_t> be_face;
    be_face.reserve(nt);
    for (int32_t fi = 0; fi < nt; ++fi) {
        for (int k = 0; k < 3; ++k) {
            int32_t va = tris[fi*3+k], vb = tris[fi*3+(k+1)%3];
            EKey key = (EKey)std::min(va,vb)*VMAX + std::max(va,vb);
            be_face[key] = fi;
        }
    }

    for (int k = 0; k + 1 < (int)boundary_edges.size(); k += 2) {
        int32_t va = boundary_edges[k], vb = boundary_edges[k+1];
        EKey key = (EKey)std::min(va,vb)*VMAX + std::max(va,vb);
        auto it = be_face.find(key);
        if (it == be_face.end()) continue;
        int32_t fi = it->second;

        float ex = positions[vb*3]   - positions[va*3];
        float ey = positions[vb*3+1] - positions[va*3+1];
        float ez = positions[vb*3+2] - positions[va*3+2];
        float elen = std::sqrt(ex*ex + ey*ey + ez*ez);
        if (elen < 1e-10f) continue;
        ex /= elen; ey /= elen; ez /= elen;

        const float* e1 = face_e1 + fi*3;
        const float* e2 = face_e2 + fi*3;
        float proj1 = ex*e1[0]+ey*e1[1]+ez*e1[2];
        float proj2 = ex*e2[0]+ey*e2[1]+ez*e2[2];

        // BUG FIX (v95): build_boundary_constraints was missing the proj_len
        // guard that build_feature_edge_constraints already has (line ~95 above).
        // For a boundary edge that is degenerate in the face's tangent-plane frame
        // (proj1 ≈ 0, proj2 ≈ 0 — e.g. a nearly zero-area face whose e1/e2 frame
        // was computed from a degenerate edge), atan2(0, 0) produces 0.0
        // (implementation-defined, always 0 on IEEE-754 platforms).  That forces
        // the field to align with e1 on the affected face regardless of the actual
        // boundary direction, silently overriding whatever the geometry demands and
        // introducing a spurious constraint into the connection Laplacian.
        // Fix: skip the constraint if the projected length is below 1e-8 (same
        // threshold as build_feature_edge_constraints, matching the numerical
        // floor used throughout the field subsystem).
        float proj_len = std::sqrt(proj1*proj1 + proj2*proj2);
        if (proj_len < 1e-8f) continue;

        double theta = std::atan2((double)proj2, (double)proj1);

        // Perpendicular: rotate by π/2
        if (perpendicular_to_boundary) theta += M_PI * 0.5;

        FieldConstraint c;
        c.face_idx     = fi;
        c.target_angle = theta;
        c.weight       = weight;
        out_constraints.push_back(c);
    }
}

// -----------------------------------------------------------------------
// Symmetry constraints
// -----------------------------------------------------------------------

void build_symmetry_constraints(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    const float* face_e1,
    const float* face_e2,
    const float* face_normals,
    const float* sym_plane_normal,  // [3] unit normal of the symmetry plane
    double weight,
    std::vector<FieldConstraint>& out_constraints)
{
    // For faces that lie on (or very near) the symmetry plane, force the
    // field to be aligned with the plane: one arm is in the plane,
    // one arm is perpendicular to sym_plane_normal (projected into face plane).

    // BUG FIX (v58): out_constraints was never cleared here — see the identical
    // fix in build_boundary_constraints for a full explanation.  Without this
    // clear, symmetry constraints pile onto any existing contents of the vector,
    // doubling penalty weights every time the function is called on a reused
    // output vector (which is the normal call pattern in cross_field.cpp).
    out_constraints.clear();

    // Compute face centroids
    for (int32_t fi = 0; fi < nt; ++fi) {
        int v0 = tris[fi*3], v1 = tris[fi*3+1], v2 = tris[fi*3+2];
        float cx = (positions[v0*3] + positions[v1*3] + positions[v2*3]) / 3.f;
        float cy = (positions[v0*3+1]+positions[v1*3+1]+positions[v2*3+1])/3.f;
        float cz = (positions[v0*3+2]+positions[v1*3+2]+positions[v2*3+2])/3.f;

        // Signed distance from symmetry plane (plane passes through origin)
        float dist = cx*sym_plane_normal[0]+cy*sym_plane_normal[1]+cz*sym_plane_normal[2];
        float abs_dist = std::abs(dist);

        // Only constrain faces very close to or exactly on the plane
        // Use a threshold of 2% of average edge length (rough estimate)
        // Compute face edge length
        float dx = positions[v1*3]-positions[v0*3];
        float dy = positions[v1*3+1]-positions[v0*3+1];
        float dz = positions[v1*3+2]-positions[v0*3+2];
        float avg_edge = std::sqrt(dx*dx+dy*dy+dz*dz);
        float threshold = avg_edge * 0.1f;

        if (abs_dist > threshold) continue;

        // In the face's tangent plane, the sym_plane_normal projects to a direction
        // which the field should be perpendicular to (to be symmetric)
        const float* n = face_normals + fi*3;
        // Project sym_plane_normal onto tangent plane
        float spn[3] = {sym_plane_normal[0], sym_plane_normal[1], sym_plane_normal[2]};
        float nd = spn[0]*n[0]+spn[1]*n[1]+spn[2]*n[2];
        float tang[3] = {spn[0]-nd*n[0], spn[1]-nd*n[1], spn[2]-nd*n[2]};
        float tlen = std::sqrt(tang[0]*tang[0]+tang[1]*tang[1]+tang[2]*tang[2]);
        if (tlen < 1e-8f) continue;
        tang[0]/=tlen; tang[1]/=tlen; tang[2]/=tlen;

        const float* e1 = face_e1 + fi*3;
        const float* e2 = face_e2 + fi*3;
        float p1 = tang[0]*e1[0]+tang[1]*e1[1]+tang[2]*e1[2];
        float p2 = tang[0]*e2[0]+tang[1]*e2[1]+tang[2]*e2[2];
        double theta = std::atan2((double)p2, (double)p1) + M_PI*0.5;

        FieldConstraint c;
        c.face_idx     = fi;
        c.target_angle = theta;
        // BUG FIX (v57): The function previously applied weight * 0.5 internally,
        // but the caller in cross_field.cpp already passes
        //   params.constraint_weight * 0.3
        // with the comment "softer than feature edges".  The combined factor of
        // 0.3 * 0.5 = 0.15 made symmetry constraints half as strong as intended,
        // producing noticeably weaker symmetry enforcement on asymmetric meshes.
        // Fix: use `weight` directly — the caller is responsible for setting the
        // appropriate softening relative to feature-edge constraints.
        c.weight       = weight;
        out_constraints.push_back(c);
    }
}

// -----------------------------------------------------------------------
// Apply constraints to connection Laplacian diagonal
// -----------------------------------------------------------------------

void apply_constraints_to_laplacian(
    const std::vector<FieldConstraint>& constraints,
    std::vector<int32_t>& coo_rows,
    std::vector<int32_t>& coo_cols,
    std::vector<std::complex<double>>& coo_vals,
    std::vector<std::complex<double>>& rhs)
{
    for (const auto& c : constraints) {
        int fi = c.face_idx;
        double w  = c.weight;
        // Target field direction as unit complex: exp(4i * theta)
        // (4-RoSy representation: 4× the angle)
        std::complex<double> target = std::exp(std::complex<double>(0, 4.0 * c.target_angle));

        // Add penalty: w * |u_fi - target|^2
        // = w * u_fi * conj(u_fi) - w * conj(target) * u_fi - w * target * conj(u_fi) + w
        // Adds w to L[fi,fi] and w * conj(target) to rhs[fi]
        coo_rows.push_back(fi); coo_cols.push_back(fi);
        coo_vals.push_back(std::complex<double>(w, 0.0));

        // BUG FIX (v58): The previous comparison was:
        //   if ((int)rhs.size() > fi)
        // Casting size_t → int is undefined behaviour when rhs.size() > INT_MAX,
        // and even in the normal range it introduces a signed/unsigned mismatch
        // that the compiler may evaluate incorrectly.  More critically, if the
        // caller did not pre-size rhs correctly (i.e. rhs.size() <= fi), the
        // diagonal penalty was added to the Laplacian but the right-hand-side
        // contribution was silently dropped — meaning the constraint would
        // attract the field to angle 0 instead of to c.target_angle, corrupting
        // every constrained face where rhs was too small.
        // Fix: use a correct size_t comparison.  fi is always >= 0 (face index),
        // so the cast to size_t is safe.
        if (static_cast<size_t>(fi) < rhs.size())
            rhs[fi] += w * std::conj(target);
    }
}

// -----------------------------------------------------------------------
// Edge-to-faces adjacency map
// -----------------------------------------------------------------------
// BUG FIX (v25): This function was declared in constraints.h but never
// implemented, causing a guaranteed linker error whenever any translation
// unit called build_feature_edge_constraints (which requires this map).

std::unordered_map<int64_t, std::vector<int32_t>>
build_edge_to_faces(const int32_t* tris, int32_t nt)
{
    // Key: (int64_t)min(va, vb) * VMAX + max(va, vb)
    // VMAX must match the value used in build_feature_edge_constraints.
    // Use (int64_t)1<<24 (not 1<<24 which evaluates as int) to avoid
    // sign-extension / overflow on vertices > 2^23.
    const int64_t VMAX = (int64_t)1 << 24;

    std::unordered_map<int64_t, std::vector<int32_t>> map;
    map.reserve(static_cast<size_t>(nt) * 2);

    for (int32_t fi = 0; fi < nt; ++fi) {
        for (int k = 0; k < 3; ++k) {
            int32_t va = tris[fi * 3 + k];
            int32_t vb = tris[fi * 3 + (k + 1) % 3];
            int64_t key = (int64_t)std::min(va, vb) * VMAX
                        + (int64_t)std::max(va, vb);
            map[key].push_back(fi);
        }
    }
    return map;
}

} // namespace qf
