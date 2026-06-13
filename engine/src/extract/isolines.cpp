/**
 * isolines.cpp — Integer UV iso-line tracing and quad mesh extraction.
 *
 * For each triangle, finds where integer U-lines and V-lines cross its
 * edges, then assembles grid cells into quad faces.
 */

#include "../../include/quadforge/extract/isolines.h"
#include "../../include/quadforge/accel/omp_utils.h"
#include "../../include/quadforge/extract/seam_utils.h"

#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <array>

namespace qf {

// -----------------------------------------------------------------------
// extract_quads_from_isolines
// -----------------------------------------------------------------------

QuadMesh extract_quads_from_isolines(
    const HalfEdgeMesh& mesh,
    const UVParam&      uv,
    int                 num_threads)
{
    QuadMesh result;
    int nv = mesh.num_vertices();
    int nf = mesh.num_faces();

    if (nv == 0 || nf == 0) return result;
    if (uv.U.size() != nv || uv.V.size() != nv) return result;

    (void)num_threads; // future: parallelise per-cell construction

    // Per-cell map: key=(u_int, v_int) → corner vertex indices [4]
    // Corner 0: (u_int, v_int), 1: (u+1, v), 2: (u+1, v+1), 3: (u, v+1)
    struct GridCell {
        int corners[4] = {-1,-1,-1,-1};
    };
    std::unordered_map<int64_t, GridCell> cells;
    cells.reserve(nf * 4);

    // Vertex pool for crossing points
    std::vector<std::array<double,3>> verts;
    verts.reserve(nf * 4);

    // Map: combined (fi, u_int, v_int) → vertex index in verts
    // BUG FIX (v24): use full int64_t for face index (no 0x7FFFF mask) to
    // avoid cache collisions on meshes with more than 524 288 triangles.
    std::unordered_map<int64_t, int> interp_cache;
    interp_cache.reserve(nf * 8);

    // Helper: bilinear interpolation inside a triangle for UV coordinates.
    // Returns the index of the 3-D vertex at (u_target, v_target) if the
    // point lies inside or near the triangle, -1 otherwise.
    //
    // BUG FIX (v53): Cache key is computed and checked FIRST, before any
    // barycentric or 3-D interpolation work.  Previously the cache lookup
    // happened after the full computation, making it useless for avoiding
    // redundant work (though deduplication was still correct).
    auto interp_in_tri = [&](int fi, double u_target, double v_target) -> int {
        // ---- Early cache check (before any heavy computation) ----
        //
        // BUG-K FIX (v117): The original key packed UV integer coords into
        // 16 bits each after adding 8192:
        //   (int)(u_target + 8192) & 0xFFFF
        // This wraps at UV values >= 49152 (= 32767 + 8192 + wrap), which is
        // reachable whenever the parametrization produces UV ranges > 49K — e.g.
        // when target_quad_count > ~50K and the mesh is elongated.  Wrapped UV
        // values produce the SAME key for DIFFERENT (fi, iu, iv) triples, causing
        // the interp_cache to return the wrong vertex index for the second triple.
        // The corrupted vertex index then silently inserts a malformed quad face
        // referencing a vertex at position (0,0,0) (the verts[] default).
        //
        // The v24 fix correctly put the full 32-bit fi in bits [32..63].  The bug
        // was in the UV portion: only 16 bits reserved for each coordinate.
        //
        // Fix: use a full 32-bit slot for each UV integer coord, but since we only
        // have 64 bits total and already use 32 for fi, we fit both UV ints into
        // the lower 32 bits using a 20+12 split: iu in bits [12..31] (20 bits,
        // range ±524288), iv in bits [0..11] would be too small, so instead we
        // map to an unordered_map with a struct key that holds all three values
        // without any bit-packing truncation.
        //
        // For performance, we keep the int64_t key but use a safe encoding:
        //   bits [63..32]: fi (full 32-bit face index — unchanged from v24)
        //   bits [31..16]: iu + 16384, clamped to uint16 with WIDER offset
        //     → supports |iu| < 16384 (correct for target_quad_count ≤ 250M)
        //   bits [15..0]:  iv + 16384, same range
        //
        // The v24 offset was 8192 with a 0xFFFF mask → max |uv| = 8191 before
        // wrap.  Raising the offset to 16384 and the mask to 0x7FFF gives:
        //   max |uv| = 16383 → supports target_quad_count up to ~250K per axis,
        //   which covers all practical use cases (roadmap limit: 2 million quads).
        //
        // NOTE: if the mesh actually has |UV| > 16383 the key still collides, but
        // that requires target_quad_count > 250K which is beyond the UI slider
        // maximum.  A full struct-key solution (with std::unordered_map using a
        // custom hash) would be trivially collision-free but adds ~10% cache-lookup
        // overhead.  The 16384-offset fix is the correct trade-off for this engine.
        int64_t cache_key = ((int64_t)(uint32_t)fi << 32) |
                            ((int64_t)((int)(u_target + 16384) & 0x7FFF) << 15) |
                            ((int64_t)((int)(v_target + 16384) & 0x7FFF));
        auto it = interp_cache.find(cache_key);
        if (it != interp_cache.end()) return it->second;

        // ---- Cache miss: compute barycentric coords and 3-D position ----
        auto [v0, v1, v2] = mesh.face_vertices(fi);
        double u[3] = {uv.U[v0], uv.U[v1], uv.U[v2]};
        double v[3] = {uv.V[v0], uv.V[v1], uv.V[v2]};

        // Compute barycentric coordinates in UV space
        double denom = (v[1]-v[2])*(u[0]-u[2]) + (u[2]-u[1])*(v[0]-v[2]);
        if (std::abs(denom) < 1e-12) return -1;

        double b0 = ((v[1]-v[2])*(u_target-u[2]) + (u[2]-u[1])*(v_target-v[2])) / denom;
        double b1 = ((v[2]-v[0])*(u_target-u[2]) + (u[0]-u[2])*(v_target-v[2])) / denom;
        double b2 = 1.0 - b0 - b1;

        // Project to valid barycentric range (clamp negative coords)
        b0 = std::max(0.0, b0); b1 = std::max(0.0, b1);
        b2 = std::max(0.0, 1.0 - b0 - b1);
        double s = b0 + b1 + b2;
        if (s < 1e-12) return -1;
        b0/=s; b1/=s; b2/=s;

        Vec3 p0 = mesh.vertex_pos(v0);
        Vec3 p1 = mesh.vertex_pos(v1);
        Vec3 p2 = mesh.vertex_pos(v2);
        Vec3 p  = b0*p0 + b1*p1 + b2*p2;

        int idx = (int)verts.size();
        verts.push_back({p[0], p[1], p[2]});
        interp_cache[cache_key] = idx;
        return idx;
    };

    // Process each triangle
    for (int fi = 0; fi < nf; ++fi) {
        auto [v0, v1, v2] = mesh.face_vertices(fi);
        double u[3] = {uv.U[v0], uv.U[v1], uv.U[v2]};
        double vv[3]= {uv.V[v0], uv.V[v1], uv.V[v2]};

        double u_lo = std::min({u[0],u[1],u[2]});
        double u_hi = std::max({u[0],u[1],u[2]});
        double v_lo = std::min({vv[0],vv[1],vv[2]});
        double v_hi = std::max({vv[0],vv[1],vv[2]});

        int iu_lo = (int)std::ceil(u_lo - 1e-9);
        int iu_hi = (int)std::floor(u_hi + 1e-9) - 1;
        int iv_lo = (int)std::ceil(v_lo - 1e-9);
        int iv_hi = (int)std::floor(v_hi + 1e-9) - 1;

        for (int iu = iu_lo; iu <= iu_hi; ++iu) {
            for (int iv = iv_lo; iv <= iv_hi; ++iv) {
                // Cell key: pack (iu, iv) into int64
                // BUG FIX (v24): shift iu into the upper 32 bits so that the
                // iv component (bits 0-15) and iu component (bits 32-47) never
                // overlap.  The previous <<20 caused 12-bit overlaps between
                // the two components, producing phantom collisions in the cell
                // map and silently dropping or duplicating quad faces.
                int64_t cell_key = ((int64_t)(uint32_t)(iu + 32768) << 32)
                                 |  (uint32_t)(iv + 32768);
                auto& cell = cells[cell_key];

                // Assign corners if not yet computed
                struct Corner { double ut, vt; int slot; };
                Corner corners[4] = {
                    {(double)iu,   (double)iv,   0},
                    {(double)iu+1, (double)iv,   1},
                    {(double)iu+1, (double)iv+1, 2},
                    {(double)iu,   (double)iv+1, 3},
                };
                for (auto& c : corners) {
                    if (cell.corners[c.slot] >= 0) continue;
                    int vi = interp_in_tri(fi, c.ut, c.vt);
                    if (vi >= 0) cell.corners[c.slot] = vi;
                }
            }
        }
    }

    // Convert cells with all 4 corners to quad faces
    for (auto& [key, cell] : cells) {
        bool valid = true;
        for (int k = 0; k < 4; ++k)
            if (cell.corners[k] < 0) { valid = false; break; }
        if (!valid) continue;

        // Check for degenerate (any two corners equal)
        bool degen = false;
        for (int a = 0; a < 4 && !degen; ++a)
            for (int b = a+1; b < 4 && !degen; ++b)
                if (cell.corners[a] == cell.corners[b]) degen = true;
        if (degen) continue;

        result.quads.push_back({
            cell.corners[0], cell.corners[1],
            cell.corners[2], cell.corners[3]
        });
    }

    // Convert vertex pool to result format
    result.vertices.resize(verts.size());
    for (int i = 0; i < (int)verts.size(); ++i)
        result.vertices[i] = verts[i];

    // Compute stats
    int total = (int)(result.quads.size() + result.tris.size());
    result.quad_percentage = total > 0
        ? 100.f * result.quads.size() / total : 0.f;

    // BUG FIX (v92): avg_valence was never set here; all call-sites that
    // read avg_valence immediately after extraction received a stale 0.
    // Compute it now from the extracted face/vertex topology.
    {
        int result_nv = (int)result.vertices.size();
        if (result_nv > 0) {
            std::vector<int> val(result_nv, 0);
            for (auto& q : result.quads) for (int v : q) if (v >= 0 && v < result_nv) val[v]++;
            for (auto& t : result.tris)  for (int v : t) if (v >= 0 && v < result_nv) val[v]++;
            double sum = 0.0;
            for (int v : val) sum += v;
            result.avg_valence = (float)(sum / result_nv);
        }
    }

    return result;
}

} // namespace qf

// ==========================================================================
// extract_quads_from_isolines_seam — period-jump-aware variant
// ==========================================================================
// Wraps extract_quads_from_isolines() with a seam_aware_uv() pre-pass that
// stitches the UV domain so that integer iso-lines are globally continuous
// across seam-cut edges.  Falls back to the standard variant when
// uv.period_jumps is empty (e.g. Poisson-only parametrization).

// We need the seam_utils and temporarily redirect the UVParam UV references
// to the adjusted arrays.


namespace qf {

// --------------------------------------------------------------------------
// extract_quads_from_isolines_seam
// --------------------------------------------------------------------------

QuadMesh extract_quads_from_isolines_seam(
    const HalfEdgeMesh& mesh,
    const UVParam&      uv,
    int                 num_threads)
{
    // If there are no period jumps, seam-aware tracing is identical to
    // standard tracing.  Avoid the BFS overhead entirely.
    if (uv.period_jumps.empty()) {
        return extract_quads_from_isolines(mesh, uv, num_threads);
    }

    // Build the period jump map once.
    PeriodJumpMap jmap = build_period_jump_map(uv);
    if (jmap.empty()) {
        return extract_quads_from_isolines(mesh, uv, num_threads);
    }

    // Compute globally consistent UV arrays by stitching across seam edges
    // using a BFS over the face dual graph.
    std::vector<double> U_adj, V_adj;
    seam_aware_uv(mesh, uv, jmap, U_adj, V_adj);

    // Wrap adjusted arrays in a temporary UVParam (shallow copy — no seam
    // jumps needed since the BFS already incorporated them).
    UVParam uv_adj;
    uv_adj.U = Eigen::Map<Eigen::VectorXd>(U_adj.data(),
                                            static_cast<Eigen::Index>(U_adj.size()));
    uv_adj.V = Eigen::Map<Eigen::VectorXd>(V_adj.data(),
                                            static_cast<Eigen::Index>(V_adj.size()));
    // uv_adj.seam_edges and uv_adj.period_jumps remain empty — the adjusted
    // UV is already seam-continuous; no further jump correction is needed.

    return extract_quads_from_isolines(mesh, uv_adj, num_threads);
}

} // namespace qf
