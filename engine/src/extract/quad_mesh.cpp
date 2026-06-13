/**
 * quad_mesh.cpp — Quad mesh construction from UV iso-line intersections.
 *
 * Implements the public API declared in include/quadforge/extract/quad_mesh.h.
 * These functions convert the raw IsoIntersection list (UV grid corner points)
 * produced by the iso-line tracer into a clean, compact quad mesh with all
 * degenerate geometry removed.
 *
 * Pipeline:
 *   construct_quad_mesh()    - Group intersections into grid cells → quad faces
 *   weld_vertices()          - Merge spatially coincident vertices
 *   remove_degenerate_faces() - Drop zero-area / repeated-index faces
 *   compact_vertices()       - Remove unreferenced vertices, reindex
 *   quad_signed_area()       - Per-face area utility (also used by callers)
 */

#include "../../include/quadforge/extract/quad_mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace qf {

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/** Pack (u_int, v_int) into a 64-bit key, offsetting to handle negatives. */
static inline int64_t cell_key(int32_t u, int32_t v) {
    // Shift so that [-32768, 32767] maps to [0, 65535], fits in 16 bits each.
    uint32_t uu = static_cast<uint32_t>(u + 32768);
    uint32_t vv = static_cast<uint32_t>(v + 32768);
    return (static_cast<int64_t>(uu) << 32) | static_cast<int64_t>(vv);
}

/** Squared distance between two 3-D double positions. */
static inline double dist2(const double* a, const double* b) {
    double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return dx*dx + dy*dy + dz*dz;
}

/** Cross product of two 3-D vectors stored as double[3]. */
static inline void cross3(const double* a, const double* b, double* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/** Length of a 3-D double vector. */
static inline double len3(const double* v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// -----------------------------------------------------------------------
// construct_quad_mesh
// -----------------------------------------------------------------------

void construct_quad_mesh(
    const std::vector<IsoIntersection>& all_intersections,
    std::vector<double>&  out_positions,
    std::vector<int32_t>& out_quad_faces,
    std::vector<int32_t>& out_tri_faces)
{
    out_positions.clear();
    out_quad_faces.clear();
    out_tri_faces.clear();

    if (all_intersections.empty()) return;

    // ---- Step 1: Copy vertex positions from intersections ----
    // The IsoIntersection array already has vertex_id fields assigned by the
    // caller (iso-line tracer). We just need to build the position array and
    // respect those IDs.

    // Find the maximum vertex_id to size the output correctly.
    int32_t max_vid = -1;
    for (const auto& p : all_intersections)
        if (p.vertex_id > max_vid) max_vid = p.vertex_id;

    if (max_vid < 0) return;

    out_positions.assign(static_cast<size_t>(max_vid + 1) * 3, 0.0);
    for (const auto& p : all_intersections) {
        if (p.vertex_id < 0) continue;
        size_t base = static_cast<size_t>(p.vertex_id) * 3;
        out_positions[base + 0] = p.pos[0];
        out_positions[base + 1] = p.pos[1];
        out_positions[base + 2] = p.pos[2];
    }

    // ---- Step 2: Group intersections by integer (U, V) cell ----
    // Each grid cell (u_int, v_int) is a unit square in UV space.
    // Its four corners are:
    //   corner 0: (u_int,   v_int  )
    //   corner 1: (u_int+1, v_int  )
    //   corner 2: (u_int+1, v_int+1)
    //   corner 3: (u_int,   v_int+1)
    //
    // An intersection point at integer (U=u_int, V=v_int) contributes to
    // up to four cells as a corner. We only need each corner once.

    // Map from cell_key → array of 4 corner vertex IDs (-1 = not yet filled)
    struct Cell {
        int32_t corners[4] = {-1, -1, -1, -1};
    };
    std::unordered_map<int64_t, Cell> cells;
    cells.reserve(all_intersections.size());

    for (const auto& p : all_intersections) {
        // This point is at integer (p.u_int, p.v_int).
        // It serves as corner 0 of cell (p.u_int,   p.v_int  )
        //                  corner 1 of cell (p.u_int-1, p.v_int  )
        //                  corner 2 of cell (p.u_int-1, p.v_int-1)
        //                  corner 3 of cell (p.u_int,   p.v_int-1)

        struct CellContrib { int32_t cu, cv; int slot; };
        CellContrib contribs[4] = {
            { p.u_int,     p.v_int,     0 },
            { p.u_int - 1, p.v_int,     1 },
            { p.u_int - 1, p.v_int - 1, 2 },
            { p.u_int,     p.v_int - 1, 3 },
        };

        for (const auto& c : contribs) {
            int64_t k = cell_key(c.cu, c.cv);
            auto& cell = cells[k];
            // Only assign if not yet set (first writer wins; duplicates are
            // welded later by weld_vertices if needed)
            if (cell.corners[c.slot] < 0)
                cell.corners[c.slot] = p.vertex_id;
        }
    }

    // ---- Step 3: Emit quad faces for fully-populated cells ----
    // Reserve generously to avoid reallocations.
    out_quad_faces.reserve(cells.size() * 4);

    for (const auto& [k, cell] : cells) {
        // Check all four corners are filled.
        bool full = (cell.corners[0] >= 0 && cell.corners[1] >= 0 &&
                     cell.corners[2] >= 0 && cell.corners[3] >= 0);
        if (!full) {
            // Partially-filled cells appear near boundaries and singularities.
            // Count filled corners.
            int filled = 0;
            for (int s = 0; s < 4; ++s) if (cell.corners[s] >= 0) ++filled;
            if (filled == 3) {
                // Emit a triangle for three-corner cells (singularity patches).
                int32_t tri[3];
                int ti = 0;
                for (int s = 0; s < 4 && ti < 3; ++s)
                    if (cell.corners[s] >= 0) tri[ti++] = cell.corners[s];
                // Basic degenerate check
                if (tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]) {
                    out_tri_faces.push_back(tri[0]);
                    out_tri_faces.push_back(tri[1]);
                    out_tri_faces.push_back(tri[2]);
                }
            }
            continue;
        }

        // Check for degenerate quad (any two corners the same vertex).
        const int32_t* c = cell.corners;
        bool degen = (c[0]==c[1] || c[0]==c[2] || c[0]==c[3] ||
                      c[1]==c[2] || c[1]==c[3] || c[2]==c[3]);
        if (degen) continue;

        // Bounds check against the position array.
        int32_t nv = static_cast<int32_t>(out_positions.size() / 3);
        bool oob = (c[0]>=nv || c[1]>=nv || c[2]>=nv || c[3]>=nv);
        if (oob) continue;

        out_quad_faces.push_back(c[0]);
        out_quad_faces.push_back(c[1]);
        out_quad_faces.push_back(c[2]);
        out_quad_faces.push_back(c[3]);
    }
}

// -----------------------------------------------------------------------
// weld_vertices
// -----------------------------------------------------------------------

void weld_vertices(
    std::vector<double>&  positions,
    std::vector<int32_t>& quad_faces,
    std::vector<int32_t>& tri_faces,
    int32_t& nv,
    double tol)
{
    if (positions.empty()) { nv = 0; return; }

    nv = static_cast<int32_t>(positions.size() / 3);
    if (nv == 0) return;

    // Build a remap array: for each vertex, find the lowest-index vertex
    // within tolerance and remap to it. This is an O(V²) algorithm in the
    // worst case; for typical quad mesh sizes (< 200K vertices) it is fast
    // enough. A grid-accelerated version can be substituted for very large
    // meshes.
    double tol2 = tol * tol;
    std::vector<int32_t> remap(static_cast<size_t>(nv));
    for (int32_t i = 0; i < nv; ++i) remap[i] = i;

    for (int32_t i = 0; i < nv; ++i) {
        if (remap[i] != i) continue; // already merged into an earlier vertex
        const double* pi = positions.data() + i * 3;
        for (int32_t j = i + 1; j < nv; ++j) {
            if (remap[j] != j) continue;
            const double* pj = positions.data() + j * 3;
            if (dist2(pi, pj) <= tol2) {
                remap[j] = i; // merge j → i
            }
        }
    }

    // Build compact vertex list and update remap to new indices.
    std::vector<double>  new_pos;
    std::vector<int32_t> new_idx(static_cast<size_t>(nv), -1);
    int32_t new_nv = 0;
    for (int32_t i = 0; i < nv; ++i) {
        int32_t root = remap[i];
        if (new_idx[root] < 0) {
            new_idx[root] = new_nv++;
            new_pos.push_back(positions[root * 3 + 0]);
            new_pos.push_back(positions[root * 3 + 1]);
            new_pos.push_back(positions[root * 3 + 2]);
        }
        remap[i] = new_idx[root];
    }

    // Remap quad face indices.
    for (auto& idx : quad_faces) {
        if (idx >= 0 && idx < nv) idx = remap[idx];
    }
    // Remap triangle face indices.
    for (auto& idx : tri_faces) {
        if (idx >= 0 && idx < nv) idx = remap[idx];
    }

    positions = std::move(new_pos);
    nv = new_nv;
}

// -----------------------------------------------------------------------
// remove_degenerate_faces
// -----------------------------------------------------------------------

void remove_degenerate_faces(
    const std::vector<double>& positions,
    std::vector<int32_t>& quad_faces,
    std::vector<int32_t>& tri_faces,
    double min_area)
{
    int32_t nv = static_cast<int32_t>(positions.size() / 3);

    // ---- Quads ----
    {
        std::vector<int32_t> clean;
        clean.reserve(quad_faces.size());
        size_t nq = quad_faces.size() / 4;
        for (size_t fi = 0; fi < nq; ++fi) {
            const int32_t* q = quad_faces.data() + fi * 4;
            // Bounds check
            bool oob = (q[0]<0||q[0]>=nv || q[1]<0||q[1]>=nv ||
                        q[2]<0||q[2]>=nv || q[3]<0||q[3]>=nv);
            if (oob) continue;
            // Repeated vertex check
            bool degen = (q[0]==q[1]||q[0]==q[2]||q[0]==q[3]||
                          q[1]==q[2]||q[1]==q[3]||q[2]==q[3]);
            if (degen) continue;
            // Area check — split quad into two triangles along diagonal p0-p2.
            // BUG FIX (v53): previous code used (p0,p1,p2) and (p0,p1,p3),
            // sharing edge p0-p1 rather than the diagonal p0-p2.  For convex
            // quads the areas happened to sum correctly, but for non-convex or
            // nearly-degenerate quads this gave the wrong area, keeping quads
            // that should have been removed.  Correct split: (p0,p1,p2) and
            // (p0,p2,p3).
            const double* p0 = positions.data() + q[0]*3;
            const double* p1 = positions.data() + q[1]*3;
            const double* p2 = positions.data() + q[2]*3;
            const double* p3 = positions.data() + q[3]*3;
            double a[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};  // p1-p0
            double b[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};  // p2-p0
            double c[3] = {p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]};  // p3-p2
            double n1[3], n2[3];
            cross3(a, b, n1);   // tri(p0,p1,p2)
            cross3(b, c, n2);   // tri(p0,p2,p3): (p2-p0) × (p3-p2)
            double area = 0.5 * (len3(n1) + len3(n2));
            if (area < min_area) continue;
            clean.push_back(q[0]); clean.push_back(q[1]);
            clean.push_back(q[2]); clean.push_back(q[3]);
        }
        quad_faces = std::move(clean);
    }

    // ---- Triangles ----
    {
        std::vector<int32_t> clean;
        clean.reserve(tri_faces.size());
        size_t nt = tri_faces.size() / 3;
        for (size_t fi = 0; fi < nt; ++fi) {
            const int32_t* t = tri_faces.data() + fi * 3;
            bool oob = (t[0]<0||t[0]>=nv || t[1]<0||t[1]>=nv || t[2]<0||t[2]>=nv);
            if (oob) continue;
            bool degen = (t[0]==t[1] || t[0]==t[2] || t[1]==t[2]);
            if (degen) continue;
            const double* p0 = positions.data() + t[0]*3;
            const double* p1 = positions.data() + t[1]*3;
            const double* p2 = positions.data() + t[2]*3;
            double a[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
            double b[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
            double n[3];
            cross3(a, b, n);
            double area = 0.5 * len3(n);
            if (area < min_area) continue;
            clean.push_back(t[0]); clean.push_back(t[1]); clean.push_back(t[2]);
        }
        tri_faces = std::move(clean);
    }
}

// -----------------------------------------------------------------------
// compact_vertices
// -----------------------------------------------------------------------

void compact_vertices(
    std::vector<double>&  positions,
    std::vector<int32_t>& quad_faces,
    std::vector<int32_t>& tri_faces,
    int32_t& nv)
{
    nv = static_cast<int32_t>(positions.size() / 3);
    if (nv == 0) return;

    // Mark which vertices are actually referenced.
    std::vector<bool> used(static_cast<size_t>(nv), false);
    for (int32_t idx : quad_faces) if (idx >= 0 && idx < nv) used[idx] = true;
    for (int32_t idx : tri_faces)  if (idx >= 0 && idx < nv) used[idx] = true;

    // Build old→new index remap.
    std::vector<int32_t> remap(static_cast<size_t>(nv), -1);
    int32_t new_nv = 0;
    std::vector<double> new_pos;
    new_pos.reserve(positions.size());
    for (int32_t i = 0; i < nv; ++i) {
        if (!used[i]) continue;
        remap[i] = new_nv++;
        new_pos.push_back(positions[i * 3 + 0]);
        new_pos.push_back(positions[i * 3 + 1]);
        new_pos.push_back(positions[i * 3 + 2]);
    }

    // Remap faces.
    // BUG FIX (v53): if remap[idx] == -1, the vertex was unreferenced before
    // this call.  Clamping to 0 silently produces degenerate faces pointing
    // at vertex 0.  Instead, mark the index as -1 so the caller's degenerate
    // face check can discard the face cleanly.
    for (auto& idx : quad_faces) {
        if (idx >= 0 && idx < nv) idx = remap[idx];
        // If idx is still -1 here it means the original index was out of
        // range or the vertex was unreferenced — leave as -1 for downstream
        // degenerate-face removal.
    }
    for (auto& idx : tri_faces) {
        if (idx >= 0 && idx < nv) idx = remap[idx];
    }

    positions = std::move(new_pos);
    nv = new_nv;
}

// -----------------------------------------------------------------------
// quad_signed_area
// -----------------------------------------------------------------------

double quad_signed_area(
    const double* p0, const double* p1,
    const double* p2, const double* p3)
{
    // Split the quad into two triangles: (p0,p1,p2) and (p0,p2,p3).
    // Use the cross-product to get the signed normal magnitude.
    // "Signed" here means the sign of the Z-component of the sum normal
    // (positive if the face normal points roughly +Z, negative if inverted).
    // For 3-D quads in arbitrary orientation, we return the signed magnitude
    // based on whether the face is convex / non-inverted.

    double a0 = p1[0]-p0[0], a1 = p1[1]-p0[1], a2 = p1[2]-p0[2];
    double b0 = p2[0]-p0[0], b1 = p2[1]-p0[1], b2 = p2[2]-p0[2];
    double c0 = p3[0]-p0[0], c1 = p3[1]-p0[1], c2 = p3[2]-p0[2];

    // Normal of triangle 1: a × b
    double n1[3];
    n1[0] = a1*b2 - a2*b1;
    n1[1] = a2*b0 - a0*b2;
    n1[2] = a0*b1 - a1*b0;

    // Normal of triangle 2: b × c  (using the diagonal b = p2-p0)
    double n2[3];
    n2[0] = b1*c2 - b2*c1;
    n2[1] = b2*c0 - b0*c2;
    n2[2] = b0*c1 - b1*c0;

    double area1 = 0.5 * len3(n1);
    double area2 = 0.5 * len3(n2);

    // Determine sign: normals should be parallel for a non-inverted quad.
    // If n1 and n2 point in opposite hemispheres, the quad is self-intersecting.
    double dot = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
    double sign = (dot >= 0.0) ? 1.0 : -1.0;

    return sign * (area1 + area2);
}

} // namespace qf
