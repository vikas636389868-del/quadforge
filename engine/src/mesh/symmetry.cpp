/**
 * symmetry.cpp — Reflective Symmetry Detection & Analysis
 *
 * Implements Stage 1, Step 1.4 of the QuadForge pipeline (roadmap §3.1).
 * See mesh/symmetry.h for full API documentation.
 *
 * Algorithm summary:
 *   For each sampled vertex v, reflect its position across the symmetry plane
 *   to obtain v_reflected.  Query the BVH for the surface point nearest to
 *   v_reflected.  Walk from that surface point to the nearest VERTEX using
 *   triangle barycentric coordinates.  If the vertex is within tolerance,
 *   record (v, v_mirror) as a valid symmetric pair.
 *
 *   Score = fraction of sampled vertices with valid mirror partners.
 *   Detected when score >= opts.detection_threshold (roadmap: 95%).
 *
 * Performance:
 *   - O(V log F) for V vertex BVH queries (each is O(log F)).
 *   - Can be reduced to O(S log F) by sampling S < V vertices.
 *   - Vertex-pair building is O(V) after the BVH phase.
 *
 * Copyright (c) 2026 QuadForge Contributors.  MIT licence (engine).
 */

#include "../../include/quadforge/mesh/symmetry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <cassert>

namespace qf {

// =========================================================================
// Internal helpers
// =========================================================================

namespace {

/**
 * Given a ClosestPointResult from TriangleBVH and the mesh, find the mesh
 * vertex that is closest to the BVH hit point.
 *
 * Strategy: inspect the three vertices of the hit triangle and return the
 * one with the smallest 3-D distance to bvh_point.  This is exact (no extra
 * BVH query needed) and O(1) per call.
 *
 * Returns -1 if face_index is invalid.
 */
int closest_vertex_on_face(
    const HalfEdgeMesh&       mesh,
    int                       face_index,
    const Vec3&               bvh_point)
{
    if (face_index < 0 || face_index >= mesh.num_faces()) return -1;

    auto verts = mesh.face_vertices(face_index);
    int   best_v   = -1;
    double best_d2 = std::numeric_limits<double>::max();

    for (int vi : verts) {
        Vec3   vp = mesh.vertex_pos(vi);
        double d2 = (vp - bvh_point).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            best_v  = vi;
        }
    }
    return best_v;
}

/**
 * Resolve the auto-tolerance when opts.tolerance == 0.
 *
 * Uses 0.5% of the bounding-box diagonal as the default, clamped to
 * a minimum of 1e-6 to handle degenerate meshes.
 */
double resolve_tolerance(const HalfEdgeMesh& mesh, double user_tolerance)
{
    if (user_tolerance > 0.0) return user_tolerance;

    double diag = bounding_box_diagonal(mesh);
    double t    = diag * 0.005;   // 0.5% of diagonal
    return (t > 1e-6) ? t : 1e-6;
}

/**
 * Resolve the plane tolerance (for classifying vertices AS ON the plane).
 *
 * Defaults to 1/10 of the main tolerance.
 */
double resolve_plane_tolerance(double main_tol, double user_plane_tol)
{
    if (user_plane_tol > 0.0) return user_plane_tol;
    return main_tol * 0.1;
}

/**
 * Build a uniform random sample of vertex indices from [0, nv).
 *
 * When max_sample == 0 or max_sample >= nv, returns ALL indices in order.
 * Otherwise returns a subset of exactly max_sample unique indices.
 *
 * Uses a fixed seed (42) so results are deterministic across runs —
 * important for regression tests.
 */
std::vector<int> build_sample(int nv, int max_sample)
{
    std::vector<int> all(nv);
    std::iota(all.begin(), all.end(), 0);

    if (max_sample <= 0 || max_sample >= nv) return all;

    // Fisher-Yates partial shuffle — O(max_sample)
    std::mt19937 rng(42u);
    for (int i = 0; i < max_sample; ++i) {
        std::uniform_int_distribution<int> dist(i, nv - 1);
        int j = dist(rng);
        std::swap(all[i], all[j]);
    }
    all.resize(max_sample);
    return all;
}

} // anonymous namespace

// =========================================================================
// Utility functions (public API)
// =========================================================================

Vec3 reflect_point(const Vec3& p, SymmetryAxis axis)
{
    Vec3 r = p;
    switch (axis) {
        case SymmetryAxis::X: r.x() = -r.x(); break;
        case SymmetryAxis::Y: r.y() = -r.y(); break;
        case SymmetryAxis::Z: r.z() = -r.z(); break;
    }
    return r;
}

double signed_plane_distance(const Vec3& p, SymmetryAxis axis)
{
    switch (axis) {
        case SymmetryAxis::X: return p.x();
        case SymmetryAxis::Y: return p.y();
        case SymmetryAxis::Z: return p.z();
    }
    return 0.0; // unreachable
}

double bounding_box_diagonal(const HalfEdgeMesh& mesh)
{
    if (mesh.num_vertices() == 0) return 0.0;

    Vec3 lo = mesh.vertex_pos(0);
    Vec3 hi = lo;

    for (int vi = 0; vi < mesh.num_vertices(); ++vi) {
        Vec3 p = mesh.vertex_pos(vi);
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }

    return (hi - lo).norm();
}

std::vector<float> build_float_position_array(const HalfEdgeMesh& mesh)
{
    int nv = mesh.num_vertices();
    std::vector<float> out(static_cast<std::size_t>(nv) * 3);
    for (int vi = 0; vi < nv; ++vi) {
        Vec3 p = mesh.vertex_pos(vi);
        out[vi * 3    ] = static_cast<float>(p.x());
        out[vi * 3 + 1] = static_cast<float>(p.y());
        out[vi * 3 + 2] = static_cast<float>(p.z());
    }
    return out;
}

std::array<float, 3> axis_plane_normal(SymmetryAxis axis)
{
    switch (axis) {
        case SymmetryAxis::X: return {1.f, 0.f, 0.f};
        case SymmetryAxis::Y: return {0.f, 1.f, 0.f};
        case SymmetryAxis::Z: return {0.f, 0.f, 1.f};
    }
    return {1.f, 0.f, 0.f}; // unreachable
}

// =========================================================================
// detect_reflective_symmetry
// =========================================================================

SymmetryResult detect_reflective_symmetry(
    const HalfEdgeMesh&              mesh,
    const TriangleBVH&               bvh,
    SymmetryAxis                     axis,
    const SymmetryDetectionOptions&  opts)
{
    SymmetryResult result;
    result.axis = axis;

    // Set plane normal
    auto pn = axis_plane_normal(axis);
    result.plane_normal = { pn[0], pn[1], pn[2] };

    int nv = mesh.num_vertices();
    if (nv == 0 || !bvh.is_valid()) {
        result.score    = 0.f;
        result.detected = false;
        return result;
    }

    // --- Resolve tolerances -----------------------------------------------
    const double tol        = resolve_tolerance(mesh, opts.tolerance);
    const double plane_tol  = resolve_plane_tolerance(tol, opts.plane_tolerance);
    result.tolerance_used   = tol;
    const double tol_sq     = tol * tol;

    // --- Build sample -------------------------------------------------------
    const std::vector<int> sample = build_sample(nv, opts.max_sample_vertices);
    const int S = static_cast<int>(sample.size());

    // --- Allocate correspondence maps ---------------------------------------
    //   mirror_map[v] = -1 (no partner) or v' (mirror vertex index).
    //   Sized to full nv so random-access by vertex index is O(1).
    std::vector<int> mirror_map(nv, -1);
    std::vector<int> plane_verts;

    // --- Main BVH query loop ------------------------------------------------
    int matched = 0;

    for (int si = 0; si < S; ++si) {
        int vi = sample[si];
        Vec3 pos = mesh.vertex_pos(vi);

        // Classify: is this vertex ON the plane?
        double d = signed_plane_distance(pos, axis);

        if (std::abs(d) <= plane_tol) {
            // Vertex lies on the symmetry plane: self-paired
            mirror_map[vi] = vi;
            ++matched;
            if (opts.include_plane_vertices) {
                plane_verts.push_back(vi);
            }
            continue;
        }

        // Reflect across the symmetry plane
        Vec3 reflected = reflect_point(pos, axis);

        // Find nearest surface point to the reflected position
        auto hit = bvh.closest_point(reflected);
        if (hit.face_index < 0) continue;

        // Find the mesh vertex closest to the BVH hit point
        int v_mirror = closest_vertex_on_face(mesh, hit.face_index, hit.point);
        if (v_mirror < 0) continue;

        // Accept if the mirror vertex is within tolerance of the
        // reflected position (not of the BVH surface point, which
        // could be anywhere on the triangle face).
        Vec3   mirror_pos = mesh.vertex_pos(v_mirror);
        double dist_sq    = (mirror_pos - reflected).squaredNorm();
        if (dist_sq > tol_sq) continue;

        // Sanity check: the mirror vertex should be on the OPPOSITE side
        // of the plane (or on it).  If the mesh has geometry very close
        // to the plane on both sides, BVH might find a vertex on the
        // SAME side — discard those.
        double d_mirror = signed_plane_distance(mirror_pos, axis);
        // d > 0 → positive half; mirror should have d_mirror <= plane_tol
        // d < 0 → negative half; mirror should have d_mirror >= -plane_tol
        bool same_side = (d > 0.0 && d_mirror > plane_tol) ||
                         (d < 0.0 && d_mirror < -plane_tol);
        if (same_side) continue;

        mirror_map[vi] = v_mirror;
        // BUG FIX (v104): The inverse mapping mirror_map[v_mirror] = vi was
        // never written.  The vertex_pairs construction loop below emits (vi,vj)
        // only when vi < vj AND mirror_map[vi] == vj.  When sampling is active
        // (max_sample_vertices > 0) and only the higher-index vertex vj was
        // sampled (not vi), mirror_map[vi] remains -1 and the pair is silently
        // dropped — the parametrisation stage then misses those symmetry
        // constraints.
        //
        // Fix: after recording the forward mapping, also record the inverse so
        // the vertex_pairs loop can find every pair regardless of which end was
        // directly sampled.  We guard with `< 0` so a later direct sample of
        // v_mirror (if included in the sample set) can overwrite with the
        // exact BVH-computed partner rather than the heuristic inverse.
        if (mirror_map[v_mirror] < 0)
            mirror_map[v_mirror] = vi;
        ++matched;
    }

    // --- Compute score -------------------------------------------------------
    result.score = (S > 0) ? (static_cast<float>(matched) / static_cast<float>(S))
                           : 0.f;
    result.detected = (result.score >= static_cast<float>(opts.detection_threshold));

    // --- Build correspondence pairs -----------------------------------------
    if (opts.build_correspondence) {
        result.mirror_map = std::move(mirror_map);
        result.plane_vertices = std::move(plane_verts);

        // Build undirected vertex_pairs: emit each undirected pair once.
        // Convention: emit (i, j) with i <= j (by index).
        //
        // BUG FIX (v106): The original `else if (vi < vj)` guard silently
        // dropped mixed plane/cross pairs.  Consider vi_p (a plane vertex,
        // mirror_map[vi_p] = vi_p) and vj_c (a cross vertex,
        // mirror_map[vj_c] = vi_p) where vi_p < vj_c:
        //
        //   * Iterating vi_p: vj = vi_p → emits self-pair (vi_p, vi_p) only.
        //   * Iterating vj_c: vj = vi_p, and vi_p < vj_c so the old
        //     `vi < vj` condition evaluates (vj_c < vi_p) → false → skipped.
        //
        // The cross pair (vi_p, vj_c) was never emitted.  The parametrisation
        // stage then missed symmetry constraints for all vertices whose mirror
        // partner lies on the plane.
        //
        // Fix: emit using the canonical ordered form min(vi,vj) / max(vi,vj),
        // and track already-emitted pairs with a lightweight visited set so
        // we still emit each undirected pair exactly once.  Self-pairs
        // (vi == vj) are always emitted directly.
        result.vertex_pairs.reserve(static_cast<std::size_t>(matched));
        std::vector<bool> emitted(nv, false);

        for (int vi = 0; vi < nv; ++vi) {
            int vj = result.mirror_map[vi];
            if (vj < 0) continue;

            if (vj == vi) {
                // Plane vertex — self-pair, always emitted once.
                result.vertex_pairs.push_back({vi, vi});
                continue;
            }

            // Cross pair: emit via canonical (lo, hi) ordering.
            int lo = std::min(vi, vj);
            int hi = std::max(vi, vj);

            // `emitted[lo]` is set when we process the lower-index vertex
            // of this pair for the first time.  On the second encounter
            // (when iterating the higher-index vertex), emitted[lo] is true
            // and we skip to avoid duplicates.
            if (!emitted[lo]) {
                emitted[lo] = true;
                result.vertex_pairs.push_back({lo, hi});
            }
        }
    }

    return result;
}

// =========================================================================
// detect_all_symmetry_axes
// =========================================================================

std::array<SymmetryResult, 3> detect_all_symmetry_axes(
    const HalfEdgeMesh&              mesh,
    const TriangleBVH&               bvh,
    bool                             test_x,
    bool                             test_y,
    bool                             test_z,
    const SymmetryDetectionOptions&  opts)
{
    std::array<SymmetryResult, 3> results;

    // Initialise all three with axis labels so callers can always read .axis
    results[0].axis = SymmetryAxis::X;
    results[1].axis = SymmetryAxis::Y;
    results[2].axis = SymmetryAxis::Z;

    auto pnX = axis_plane_normal(SymmetryAxis::X);
    auto pnY = axis_plane_normal(SymmetryAxis::Y);
    auto pnZ = axis_plane_normal(SymmetryAxis::Z);
    results[0].plane_normal = { pnX[0], pnX[1], pnX[2] };
    results[1].plane_normal = { pnY[0], pnY[1], pnY[2] };
    results[2].plane_normal = { pnZ[0], pnZ[1], pnZ[2] };

    if (test_x)
        results[0] = detect_reflective_symmetry(mesh, bvh, SymmetryAxis::X, opts);
    if (test_y)
        results[1] = detect_reflective_symmetry(mesh, bvh, SymmetryAxis::Y, opts);
    if (test_z)
        results[2] = detect_reflective_symmetry(mesh, bvh, SymmetryAxis::Z, opts);

    return results;
}

} // namespace qf
