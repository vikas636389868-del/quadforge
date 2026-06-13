/**
 * normals.cpp — Per-vertex and per-face normal computation for the output
 * quad mesh.
 *
 * Phase 5 roadmap deliverable:
 *   "Normal computation for the output mesh."
 *
 * Two functions are implemented:
 *
 *   compute_face_normals()   — per-quad diagonal cross-product, per-tri CCW
 *   compute_vertex_normals() — angle-weighted average over incident faces
 *
 * Design notes:
 *   Face normals for quads use the diagonal cross-product method:
 *     n = normalize( (v2-v0) × (v3-v1) )
 *
 *   Vertex normals use angle-weighting (Thürmer & Wüthrich 1998).
 *
 * Threading model:
 *   Phase 1 (face normals)   — parallel_for over faces (independent)
 *   Phase 2 (accumulation)   — SERIAL (multiple faces → same vertex)
 *   Phase 3 (normalisation)  — parallel_for over vertices (independent)
 *
 * BUG-FIX (v115 / BUG-C):
 *   compute_face_normals() stored Vec3(0,0,1) as the fallback normal for
 *   degenerate quads/tris whose diagonal (or CCW) cross-product was below
 *   the 1e-15 length threshold.  The vertex-normal accumulation phase then
 *   weighted this (0,0,1) fallback by the interior angle at each corner of
 *   the degenerate face — always 90° (the same fallback in quad_corner_angle
 *   / tri_corner_angle).  The net effect: vertices of a degenerate face
 *   silently received a +Z bias proportional to 90°, pulling shading normals
 *   toward (0,0,1) near singularities and producing visible shading artefacts
 *   on meshes whose faces lie in the global X/Y plane.
 *
 *   Fix: compute_face_normals_with_validity() fills a parallel vector<bool>
 *   marking each face as valid (non-degenerate) or invalid.  The accumulation
 *   loop in compute_vertex_normals() skips invalid faces entirely — they
 *   contribute neither their fallback normal nor their fallback angle weight.
 *   The public compute_face_normals() API is unchanged; it still returns
 *   (0,0,1) for degenerate faces for backward compatibility with callers
 *   that only need face normals and don't call compute_vertex_normals().
 */

#include "../../include/quadforge/postprocess/normals.h"
#include "../../include/quadforge/accel/omp_utils.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

// Interior angle at vertex k of a quad (v0,v1,v2,v3) in radians.
inline double quad_corner_angle(const std::array<Vec3,4>& v, int k)
{
    Vec3 e1 = v[(k+1)%4] - v[k];
    Vec3 e2 = v[(k+3)%4] - v[k];
    double l1 = e1.norm(), l2 = e2.norm();
    if (l1 < 1e-15 || l2 < 1e-15) return M_PI * 0.5;
    double c = e1.dot(e2) / (l1 * l2);
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c);
}

// Interior angle at vertex k of a triangle (v0,v1,v2) in radians.
inline double tri_corner_angle(const std::array<Vec3,3>& v, int k)
{
    Vec3 e1 = v[(k+1)%3] - v[k];
    Vec3 e2 = v[(k+2)%3] - v[k];
    double l1 = e1.norm(), l2 = e2.norm();
    if (l1 < 1e-15 || l2 < 1e-15) return M_PI / 3.0;
    double c = e1.dot(e2) / (l1 * l2);
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Internal: compute face normals AND validity flags in one parallel pass.
//
// BUG-C FIX: separates the "which faces are geometrically valid" decision
// from the normal value, so the accumulation phase can skip degenerate faces
// rather than blending in their (0,0,1) fallback.
// ---------------------------------------------------------------------------
static void compute_face_normals_with_validity(
    const QuadMesh&       qm,
    std::vector<Vec3>&    fnormals,
    std::vector<bool>&    fvalid,
    int                   num_threads)
{
    const int nv  = static_cast<int>(qm.vertices.size());
    const int nqf = static_cast<int>(qm.quads.size());
    const int ntf = static_cast<int>(qm.tris.size());
    const int total = nqf + ntf;

    fnormals.assign(static_cast<size_t>(total), Vec3(0.0, 0.0, 1.0));
    fvalid.assign(  static_cast<size_t>(total), false);

    // Phase 1a: quad face normals — diagonal cross-product method
    parallel_for(nqf, [&](int fi) {
        bool ok = true;
        std::array<Vec3,4> v;
        for (int k = 0; k < 4; ++k) {
            int vi = qm.quads[fi][k];
            if (vi < 0 || vi >= nv) { ok = false; break; }
            v[k] = { qm.vertices[vi][0], qm.vertices[vi][1], qm.vertices[vi][2] };
        }
        if (!ok) return;  // fvalid[fi] stays false; fnormals[fi] = (0,0,1) fallback

        Vec3 d1 = v[2] - v[0];
        Vec3 d2 = v[3] - v[1];
        Vec3 n  = d1.cross(d2);
        double len = n.norm();
        if (len > 1e-15) {
            fnormals[fi] = n / len;
            fvalid[fi]   = true;  // BUG-C FIX: only mark valid when normal is real
        }
        // else: degenerate quad — keep (0,0,1) fallback, fvalid[fi] = false
        //       compute_vertex_normals() will skip this face entirely.
    }, num_threads);

    // Phase 1b: tri face normals — standard CCW
    parallel_for(ntf, [&](int ti) {
        bool ok = true;
        std::array<Vec3,3> v;
        for (int k = 0; k < 3; ++k) {
            int vi = qm.tris[ti][k];
            if (vi < 0 || vi >= nv) { ok = false; break; }
            v[k] = { qm.vertices[vi][0], qm.vertices[vi][1], qm.vertices[vi][2] };
        }
        if (!ok) return;

        Vec3 n = (v[1]-v[0]).cross(v[2]-v[0]);
        double len = n.norm();
        if (len > 1e-15) {
            fnormals[nqf + ti] = n / len;
            fvalid[nqf + ti]   = true;  // BUG-C FIX: mark valid only when real normal
        }
    }, num_threads);
}

// ---------------------------------------------------------------------------
// compute_face_normals (public API — unchanged behaviour, backward-compatible)
// ---------------------------------------------------------------------------
std::vector<Vec3> compute_face_normals(
    const QuadMesh& qm,
    int             num_threads)
{
    std::vector<Vec3> fnormals;
    std::vector<bool> fvalid;
    compute_face_normals_with_validity(qm, fnormals, fvalid, num_threads);
    return fnormals;  // Returns (0,0,1) for degenerate faces (unchanged API)
}

// ---------------------------------------------------------------------------
// compute_vertex_normals
// ---------------------------------------------------------------------------
std::vector<Vec3> compute_vertex_normals(
    const QuadMesh& qm,
    int             num_threads)
{
    const int nv  = static_cast<int>(qm.vertices.size());
    const int nqf = static_cast<int>(qm.quads.size());
    const int ntf = static_cast<int>(qm.tris.size());

    std::vector<Vec3> vn(static_cast<size_t>(nv), Vec3(0.0, 0.0, 0.0));

    if (nv == 0) return vn;

    // Compute face normals AND validity flags.
    // BUG-C FIX: use the internal variant that also returns fvalid[].
    std::vector<Vec3> fn;
    std::vector<bool> fvalid;
    compute_face_normals_with_validity(qm, fn, fvalid, num_threads);

    // ------------------------------------------------------------------
    // Phase 2: SERIAL accumulation — angle-weighted sum of face normals.
    //
    // BUG-C FIX: skip any face where fvalid[fi] == false.  Such faces have
    // a (0,0,1) placeholder normal and 90° fallback corner angles.  Including
    // them biases vertex normals toward (0,0,1) near singularities.
    // With the validity check, degenerate faces contribute nothing at all.
    // ------------------------------------------------------------------

    // Quad faces
    for (int fi = 0; fi < nqf; ++fi) {
        if (!fvalid[fi]) continue;  // BUG-C FIX: skip degenerate quads

        bool ok = true;
        std::array<Vec3, 4> v;
        for (int k = 0; k < 4; ++k) {
            int vi = qm.quads[fi][k];
            if (vi < 0 || vi >= nv) { ok = false; break; }
            v[k] = { qm.vertices[vi][0], qm.vertices[vi][1], qm.vertices[vi][2] };
        }
        if (!ok) continue;

        const Vec3& n = fn[fi];
        for (int k = 0; k < 4; ++k) {
            int vi = qm.quads[fi][k];
            double angle = quad_corner_angle(v, k);
            vn[vi] += angle * n;
        }
    }

    // Leftover tri faces
    for (int ti = 0; ti < ntf; ++ti) {
        if (!fvalid[nqf + ti]) continue;  // BUG-C FIX: skip degenerate tris

        bool ok = true;
        std::array<Vec3, 3> v;
        for (int k = 0; k < 3; ++k) {
            int vi = qm.tris[ti][k];
            if (vi < 0 || vi >= nv) { ok = false; break; }
            v[k] = { qm.vertices[vi][0], qm.vertices[vi][1], qm.vertices[vi][2] };
        }
        if (!ok) continue;

        const Vec3& n = fn[nqf + ti];
        for (int k = 0; k < 3; ++k) {
            int vi = qm.tris[ti][k];
            double angle = tri_corner_angle(v, k);
            vn[vi] += angle * n;
        }
    }

    // ------------------------------------------------------------------
    // Phase 3: PARALLEL normalisation.
    // Vertices with no valid incident face (all degenerate or isolated)
    // retain (0,0,0) accumulator → normalised to (0,0,1) safe fallback.
    // ------------------------------------------------------------------
    parallel_for(nv, [&](int vi) {
        double len = vn[vi].norm();
        if (len > 1e-15) {
            vn[vi] /= len;
        } else {
            vn[vi] = Vec3(0.0, 0.0, 1.0);
        }
    }, num_threads);

    return vn;
}

// ---------------------------------------------------------------------------
// pack_normals_to_float
// ---------------------------------------------------------------------------
void pack_normals_to_float(
    const std::vector<Vec3>& vertex_normals,
    float*                   out_normals,
    int                      num_threads)
{
    if (!out_normals) return;
    const int nv = static_cast<int>(vertex_normals.size());
    parallel_for(nv, [&](int vi) {
        out_normals[vi*3+0] = static_cast<float>(vertex_normals[vi][0]);
        out_normals[vi*3+1] = static_cast<float>(vertex_normals[vi][1]);
        out_normals[vi*3+2] = static_cast<float>(vertex_normals[vi][2]);
    }, num_threads);
}

} // namespace qf
