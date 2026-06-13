#pragma once
/* quadforge/extract/seam_utils.h — Period-jump-aware UV utilities for
 * isoline tracing across seam-cut edges.
 *
 * PROBLEM
 * -------
 * The MIQ parametrization produces UV coordinates that are only piecewise
 * continuous: across each seam-cut edge the U and/or V values jump by a
 * non-zero integer amount (the "period jump").  The combing stage stores
 * these period jumps in CombingResult::period_jumps, and the seam edges
 * themselves in UVParam::seam_edges (and UVParam::period_jumps once that
 * field was added to UVParam in v89).
 *
 * Without seam awareness, extract_quads_from_isolines() treats the UV
 * domain as globally continuous.  When an integer iso-line crosses a seam
 * edge, the per-vertex U or V value on one side of the edge appears shifted
 * by the period jump, so the isoline interpolation places the crossing point
 * at the wrong position — or misses it entirely.  The result is broken
 * iso-lines, missing quads, or duplicate quad vertices near every seam.
 *
 * FIX STRATEGY
 * ------------
 * This header exposes three utilities:
 *
 *  1.  PeriodJumpMap — a fast hash map from (v_lo, v_hi) → (Δu_int, Δv_int)
 *      built once from UVParam before tracing begins.
 *
 *  2.  uv_consistent() — given a face and one of its vertices chosen as the
 *      "anchor" (its raw UV is used as-is), returns the UV of every other
 *      vertex in the face after applying the period jumps for any seam edges
 *      that separate it from the anchor.  This makes the three UV values
 *      within a single triangle mutually consistent (no jumps inside the
 *      triangle) while still reflecting the correct integer grid coordinates
 *      for iso-line tracing.
 *
 *  3.  find_seam_crossings() — given a directed edge (va → vb), returns the
 *      SeamCrossing record if the edge is a seam-cut edge (period jump ≠ 0).
 *      Used by the parallel iso-tracer to split chains at seam boundaries.
 *
 * REFERENCES
 * ----------
 * Bommes et al. (2009) "Mixed-Integer Quadrangulation," §4 (period jumps).
 * Ebke   et al. (2013) "QEx: Robust Quad Mesh Extraction," §3.1.
 */

#ifndef QUADFORGE_EXTRACT_SEAM_UTILS_H
#define QUADFORGE_EXTRACT_SEAM_UTILS_H

#include "../types.h"
#include "../mesh/halfedge.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <array>
#include <vector>

namespace qf {

/* ======================================================================
 * PeriodJumpMap
 * ====================================================================== */

/**
 * Per-seam-edge integer period jump (Δu, Δv).
 *
 * Convention: for a directed edge (v_lo → v_hi) the period jump is
 * (Δu, Δv) such that:
 *
 *     U[v_hi] ≈ U[v_lo] + Δu    (integer Δu)
 *     V[v_hi] ≈ V[v_lo] + Δv    (integer Δv)
 *
 * The reverse direction (v_hi → v_lo) uses the negated jump (−Δu, −Δv).
 * Both directions are stored in the map for O(1) two-directional lookup.
 */
struct PeriodJump {
    int du{0};  ///< Integer U period jump
    int dv{0};  ///< Integer V period jump
};

/**
 * Compact hash map: directed_edge_key(v_lo, v_hi) → PeriodJump.
 *
 * Build with build_period_jump_map() below.  Once built, it is read-only
 * and safe to share across threads.
 */
using PeriodJumpMap = std::unordered_map<int64_t, PeriodJump>;

/**
 * Pack a directed edge (a → b) into a 64-bit key.
 *
 * The same packing as used in motorcycle.cpp / isolines.cpp so that keys
 * are interoperable between modules.  'a' goes into the upper 32 bits.
 */
inline int64_t seam_edge_key(int a, int b) noexcept {
    return (static_cast<int64_t>(static_cast<uint32_t>(a)) << 32)
         |  static_cast<int64_t>(static_cast<uint32_t>(b));
}

/**
 * Build a PeriodJumpMap from a UVParam that has been populated with
 * seam_edges and period_jumps (set by the parametrization stage).
 *
 * Both directed orientations of each seam edge are inserted.
 *
 * If uv.period_jumps is empty (e.g., Poisson-only parametrization with no
 * MIQ rounding), the returned map is also empty — seam-aware tracing
 * degrades gracefully to standard tracing.
 *
 * @param uv  UV parametrization from the param stage
 * @return    Populated PeriodJumpMap (empty if no seam data)
 */
PeriodJumpMap build_period_jump_map(const UVParam& uv);


/* ======================================================================
 * uv_consistent — intra-face UV consistency
 * ====================================================================== */

/**
 * Return UV coordinates for all three vertices of triangle face_idx that
 * are mutually consistent: no period jumps exist between any two of them.
 *
 * Algorithm:
 *   1. Use vertex v0 (first vertex of the face) as the anchor — its raw
 *      UV is used unchanged.
 *   2. For v1: if the directed edge v0 → v1 is in the PeriodJumpMap, add
 *      the jump (Δu, Δv) to raw UV[v1].  Otherwise use raw UV[v1].
 *   3. For v2: first try the direct edge v0 → v2; if found, add
 *      jump(v0→v2) to raw UV[v2].  Otherwise accumulate through v1:
 *
 *        total_jump = jump(v0→v1)    [0 if v0→v1 not a seam]
 *                   + jump(v1→v2)    [0 if v1→v2 not a seam]
 *        adjusted UV[v2] = raw UV[v2] + total_jump
 *
 *      Both components MUST be included — omitting jump(v0→v1) produces
 *      a result that is only consistent relative to v1, not to the anchor
 *      v0, breaking the iso-line tracing for the triangle.
 *
 * The returned array is in vertex order: [u0,v0, u1,v1, u2,v2].
 *
 * @param face_idx   Triangle index in the HalfEdgeMesh
 * @param mesh       Half-edge mesh (for vertex ring query)
 * @param uv         UV parametrization
 * @param jmap       Period jump map from build_period_jump_map()
 * @return           [u0,v0, u1,v1, u2,v2] consistent within the face
 */
std::array<double,6> uv_consistent(
    int                    face_idx,
    const HalfEdgeMesh&    mesh,
    const UVParam&         uv,
    const PeriodJumpMap&   jmap);


/* ======================================================================
 * SeamCrossing — metadata for a single seam-edge crossing
 * ====================================================================== */

/**
 * Describes a point where an integer iso-line crosses a seam-cut edge.
 *
 * During isoline tracing, when the tracer steps from triangle T across
 * edge (va → vb) into the adjacent triangle, it must check whether that
 * edge is a seam.  If so, the period jump must be applied to the UV
 * coordinates of the destination triangle's vertices before continuing.
 */
struct SeamCrossing {
    int    va;      ///< Origin  vertex of the seam edge (mesh index)
    int    vb;      ///< Dest    vertex of the seam edge (mesh index)
    int    du;      ///< Integer U jump when crossing va → vb
    int    dv;      ///< Integer V jump when crossing va → vb
    double t;       ///< Parametric position of the crossing along va→vb  [0,1]
};

/**
 * If the directed edge (va → vb) is a seam-cut edge with a non-zero period
 * jump, fills *out and returns true.  Otherwise returns false.
 *
 * NOTE (v91): out->t is always set to 0.5 as a placeholder.  This field
 * represents the parametric crossing position along va→vb but requires UV
 * values that are not available in this function.  Callers that need the
 * precise t MUST compute it themselves from the UV coordinates of va and vb
 * after this call returns true.  Do NOT use out->t directly without recomputing.
 *
 * @param va    Origin vertex index
 * @param vb    Destination vertex index
 * @param jmap  Period jump map
 * @param out   Output SeamCrossing (written only on true return)
 * @return      true iff the edge is a seam with du≠0 or dv≠0
 */
bool query_seam_crossing(
    int                   va,
    int                   vb,
    const PeriodJumpMap&  jmap,
    SeamCrossing*         out);


/* ======================================================================
 * apply_period_jump — convenience for the iso-tracer inner loop
 * ====================================================================== */

/**
 * Apply the period jump for the directed edge (va → vb) to a UV
 * coordinate pair.  If the edge is not in the jump map, (u, v) is
 * returned unchanged.
 *
 * Inline — hot path inside the triangle loop of extract_quads_from_isolines.
 *
 * @param u     U coordinate to adjust (in/out)
 * @param v     V coordinate to adjust (in/out)
 * @param va    Edge origin vertex index
 * @param vb    Edge dest   vertex index
 * @param jmap  Period jump map
 */
inline void apply_period_jump(double& u, double& v,
                               int va, int vb,
                               const PeriodJumpMap& jmap) noexcept
{
    auto it = jmap.find(seam_edge_key(va, vb));
    if (it != jmap.end()) {
        u += it->second.du;
        v += it->second.dv;
    }
}


/* ======================================================================
 * seam_aware_uv — full per-face UV with all jumps resolved
 * ====================================================================== */

/**
 * For every face in the mesh, compute consistent UV coordinates with all
 * period jumps applied relative to an arbitrary global anchor.
 *
 * This is the "global unwrap with jump stitching" operation used by the
 * seam-aware variant of extract_quads_from_isolines().  It walks the
 * dual graph of the mesh in BFS order, propagating period jumps across
 * seam edges, so that the integer iso-grid is globally consistent and
 * no iso-line is "split" by a seam.
 *
 * The result is a [V] pair of adjusted U, V values.  The values may be
 * large (e.g., U = 47.0) because they reflect the global integer lattice
 * position, not a [0,1] normalised UV.
 *
 * Complexity: O(V + F) amortised (single BFS over faces).
 *
 * @param mesh   Triangle mesh
 * @param uv     Raw UV from the parametrization stage
 * @param jmap   Period jump map
 * @param U_out  Output: adjusted U values per vertex  [V]
 * @param V_out  Output: adjusted V values per vertex  [V]
 */
void seam_aware_uv(
    const HalfEdgeMesh&  mesh,
    const UVParam&       uv,
    const PeriodJumpMap& jmap,
    std::vector<double>& U_out,
    std::vector<double>& V_out);

} // namespace qf

#endif // QUADFORGE_EXTRACT_SEAM_UTILS_H
