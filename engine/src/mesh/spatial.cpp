/**
 * spatial.cpp — BVH spatial index over a triangle mesh.
 *
 * Uses the vendored nanoflann single-header KD-tree (see
 * engine/third_party/nanoflann/nanoflann.hpp, upstream master,
 * NANOFLANN_VERSION 0x190) over triangle centroids.  All spatial queries
 * run in O(log N) expected time.
 *
 * Closest-point strategy
 * ─────────────────────
 * 1. Find K = min(max(16, nf/100+1), nf) nearest CENTROIDS using knnSearch.
 * 2. For each candidate triangle compute the exact closest point on the triangle.
 * 3. Expand K if the best candidate's distance is not yet guaranteed optimal
 *    (i.e. closest_centroid_dist - max_tri_half_diag > best_dist).
 * 4. Return the overall minimum.
 *
 * This gives O(K log N) per query, where K is small (~16) for typical
 * closest-point queries far from triangle edges.
 */

#include "../../include/quadforge/mesh/spatial.h"
#include "../../third_party/nanoflann/nanoflann.hpp"

#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <array>
#include <unordered_set>

namespace qf {

// -------------------------------------------------------------------------
// closest_point_on_triangle — exact, from Ericson "Real-Time Collision Detection"
// -------------------------------------------------------------------------
static Vec3 closest_point_on_triangle(
    const Vec3& p,
    const Vec3& v0, const Vec3& v1, const Vec3& v2)
{
    Vec3 ab = v1 - v0, ac = v2 - v0, ap = p - v0;

    double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return v0;

    Vec3 bp = p - v1;
    double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return v1;

    Vec3 cp = p - v2;
    double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) return v2;

    double vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        double v = d1 / (d1 - d3);
        return v0 + v * ab;
    }
    double vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        double w = d2 / (d2 - d6);
        return v0 + w * ac;
    }
    double va = d3*d6 - d5*d4;
    if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
        double w = (d4-d3) / ((d4-d3)+(d5-d6));
        return v1 + w * (v2 - v1);
    }
    double denom = 1.0 / (va + vb + vc);
    double bary_v = vb * denom, bary_w = vc * denom;
    return v0 + ab * bary_v + ac * bary_w;
}

// -------------------------------------------------------------------------
// moeller_trumbore — classic ray / triangle test.
//
// On hit, writes the ray parameter t to *out_t and returns true.  Matches
// the epsilon values the previous linear-scan ray_intersect used so the
// accelerated path stays behaviourally identical.
// -------------------------------------------------------------------------
static bool moeller_trumbore(
    const Vec3& origin, const Vec3& dir,
    const Vec3& v0, const Vec3& v1, const Vec3& v2,
    double* out_t)
{
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 h  = dir.cross(e2);
    double a = e1.dot(h);
    if (std::abs(a) < 1e-15) return false;          // ray parallel to triangle
    double f = 1.0 / a;
    Vec3 s  = origin - v0;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;
    Vec3 qv = s.cross(e1);
    double v = f * dir.dot(qv);
    if (v < 0.0 || u + v > 1.0) return false;
    double t = f * e2.dot(qv);
    if (t <= 1e-10) return false;                   // behind origin / too close
    *out_t = t;
    return true;
}

// -------------------------------------------------------------------------
// TriCentroidCloud — nanoflann dataset adapter over triangle centroids.
// The kdtree_get_pt signature matches nanoflann's canonical adapter form
// (size_t idx, size_t dim) as documented in nanoflann.hpp.  Using size_t for
// `dim` avoids the implicit size_t→int narrowing conversion that older
// adapters exhibited under -Wconversion / /W4.
struct TriCentroidCloud {
    using coord_t = double;
    const std::vector<Vec3>& pts;
    explicit TriCentroidCloud(const std::vector<Vec3>& p) : pts(p) {}
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(size_t idx, size_t dim) const {
        return pts[idx][static_cast<Eigen::Index>(dim)];
    }
    template<class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Adaptor<double, TriCentroidCloud>,
    TriCentroidCloud, 3 /*dim*/>;

// -------------------------------------------------------------------------
// Impl
// -------------------------------------------------------------------------
struct TriangleBVH::Impl {
    std::vector<Vec3>               centroids;   // one centroid per triangle
    std::vector<std::array<Vec3,3>> triangles;   // triangle vertex positions
    double                          max_half_diag{0.0}; // max edge/2 over all tris
    int                             nf{0};

    // Mesh axis-aligned bounding box — used by ray_intersect for ray clipping.
    // Filled during build(); sentinel values produce an empty box when nf==0.
    Vec3 bbox_min{ std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()};
    Vec3 bbox_max{-std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()};

    // KD-tree — must be declared AFTER centroids (member init order)
    TriCentroidCloud                cloud{centroids};
    std::unique_ptr<KDIndex>        kd;

    void build(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
        nf = F.rows();
        centroids.resize(nf);
        triangles.resize(nf);
        max_half_diag = 0.0;

        for (int fi = 0; fi < nf; ++fi) {
            Vec3 v0 = V.row(F(fi,0)).transpose();
            Vec3 v1 = V.row(F(fi,1)).transpose();
            Vec3 v2 = V.row(F(fi,2)).transpose();
            triangles[fi] = {v0, v1, v2};
            centroids[fi] = (v0 + v1 + v2) / 3.0;

            // Track the largest triangle "radius" for guaranteed-correctness
            // expansion of the kNN search radius.
            double d0 = (v1-v0).norm();
            double d1 = (v2-v0).norm();
            double d2 = (v2-v1).norm();
            max_half_diag = std::max(max_half_diag,
                                     std::max({d0,d1,d2}) * 0.5);

            // Grow the mesh AABB (used by ray_intersect for ray clipping).
            bbox_min = bbox_min.cwiseMin(v0).cwiseMin(v1).cwiseMin(v2);
            bbox_max = bbox_max.cwiseMax(v0).cwiseMax(v1).cwiseMax(v2);
        }

        if (nf > 0) {
            kd = std::make_unique<KDIndex>(
                3, cloud,
                nanoflann::KDTreeSingleIndexAdaptorParams(/*leaf_max_size=*/10));
            kd->buildIndex();
        }
    }
};

// -------------------------------------------------------------------------
// Construction / destruction
// -------------------------------------------------------------------------
TriangleBVH::TriangleBVH(const Eigen::MatrixXd& vertices,
                           const Eigen::MatrixXi& faces)
    : impl_(std::make_unique<Impl>())
{
    impl_->build(vertices, faces);
    valid_ = (impl_->nf > 0);
}

TriangleBVH::~TriangleBVH() = default;

int TriangleBVH::num_triangles() const {
    return impl_ ? impl_->nf : 0;
}

// -------------------------------------------------------------------------
// closest_point
// -------------------------------------------------------------------------
ClosestPointResult TriangleBVH::closest_point(const Vec3& q) const {
    ClosestPointResult res;
    res.face_index = -1;
    res.distance   = std::numeric_limits<double>::infinity();

    if (!valid_ || !impl_ || impl_->nf == 0) return res;

    int nf = impl_->nf;

    // Choose initial K: at least 16, at most nf
    int K = std::min(std::max(16, nf / 100 + 1), nf);

    std::vector<uint32_t> indices(K);
    std::vector<double>   sq_dists(K);

    double qd[3] = {q[0], q[1], q[2]};
    impl_->kd->knnSearch(qd, static_cast<size_t>(K),
                          indices.data(), sq_dists.data());

    double best_dist2 = std::numeric_limits<double>::infinity();
    int    best_fi    = -1;
    Vec3   best_pt;

    for (int ki = 0; ki < K; ++ki) {
        int fi = static_cast<int>(indices[ki]);
        if (fi < 0 || fi >= nf) continue;
        auto& tri = impl_->triangles[fi];
        Vec3 cp = closest_point_on_triangle(q, tri[0], tri[1], tri[2]);
        double d2 = (cp - q).squaredNorm();
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_fi    = fi;
            best_pt    = cp;
        }
    }

    // Safety check: if the worst centroid distance minus max_half_diag is still
    // larger than our best distance, we may have missed a closer triangle.
    // In that case expand K to cover more candidates (bounded by nf).
    if (K < nf && best_fi >= 0) {
        double worst_centroid_dist = std::sqrt(sq_dists[K-1]);
        double guaranteed_miss_dist = worst_centroid_dist - impl_->max_half_diag;
        double best_dist = std::sqrt(best_dist2);
        if (guaranteed_miss_dist < best_dist) {
            // Need more candidates — fall back to full linear scan for safety
            for (int fi = 0; fi < nf; ++fi) {
                auto& tri = impl_->triangles[fi];
                Vec3 cp = closest_point_on_triangle(q, tri[0], tri[1], tri[2]);
                double d2 = (cp - q).squaredNorm();
                if (d2 < best_dist2) {
                    best_dist2 = d2; best_fi = fi; best_pt = cp;
                }
            }
        }
    }

    if (best_fi >= 0) {
        res.face_index = best_fi;
        res.distance   = std::sqrt(best_dist2);
        res.point      = best_pt;
    }
    return res;
}

// -------------------------------------------------------------------------
// points_within_radius
// -------------------------------------------------------------------------
std::vector<ClosestPointResult> TriangleBVH::points_within_radius(
    const Vec3& center, double radius) const
{
    std::vector<ClosestPointResult> results;
    if (!valid_ || !impl_) return results;

    // Search radius for centroids: add max_half_diag so we don't miss triangles
    // whose centroid is slightly outside the radius but whose surface is inside.
    double search_r = radius + impl_->max_half_diag;
    double search_r2 = search_r * search_r;

    double cd[3] = {center[0], center[1], center[2]};
    // nanoflann >=1.5 returns ResultItem<IndexType,DistanceType>, which is a
    // plain struct with public `first`/`second` members (not std::pair).
    // It is still compatible with C++17 aggregate structured bindings.
    std::vector<nanoflann::ResultItem<uint32_t, double>> pairs;
    impl_->kd->radiusSearch(cd, search_r2, pairs);

    for (auto& item : pairs) {
        int fi = static_cast<int>(item.first);
        (void)item.second;  // centroid squared distance (unused)
        auto& tri = impl_->triangles[fi];
        Vec3 cp = closest_point_on_triangle(center, tri[0], tri[1], tri[2]);
        double d = (cp - center).norm();
        if (d <= radius)
            results.push_back({cp, fi, d});
    }
    return results;
}

// -------------------------------------------------------------------------
// ray_intersect
//
// Strategy: AABB-clipped ray marching with nanoflann radius-based candidate
// collection.
//
//   1. Slab-test the ray against the mesh AABB to obtain a [t_enter, t_exit]
//      interval on the ray — anything outside this interval cannot hit any
//      triangle, so we never march there.
//   2. March along the clipped segment in steps of 2*max_half_diag.  At each
//      sample point p we call nanoflann::radiusSearch with radius 2*max_half_diag.
//      Any triangle whose centroid is within max_half_diag of the ray line is
//      within 2*max_half_diag of some sample (a sample is at most r along the
//      axis, at most r perpendicular, so distance ≤ r*sqrt(2) < 2r), so this
//      cover is conservative and no candidate is missed.
//   3. Unique candidate triangle indices are collected into an unordered_set
//      and then tested with Möller–Trumbore.  The smallest positive hit wins.
//
// Falls back to a linear scan when:
//   * the BVH is empty
//   * the ray direction is degenerate
//   * the ray entirely misses the mesh AABB
//   * max_half_diag is zero (degenerate mesh) — no meaningful step size
//   * the computed number of march steps exceeds a safety cap (pathological
//     mesh geometries where max_half_diag is tiny relative to mesh extent).
// -------------------------------------------------------------------------
RayHitResult TriangleBVH::ray_intersect(const Vec3& origin,
                                          const Vec3& dir) const
{
    RayHitResult res;
    if (!valid_ || !impl_ || impl_->nf == 0) return res;

    const int nf = impl_->nf;

    // Reject degenerate direction vectors up front.
    double dir_len = dir.norm();
    if (dir_len < 1e-30) return res;
    Vec3 d_hat = dir / dir_len;

    // ---- inline linear-scan helper (used by several fallback cases) ----
    auto linear_scan = [&]() -> RayHitResult {
        RayHitResult r;
        double best_t_local = std::numeric_limits<double>::infinity();
        for (int fi = 0; fi < nf; ++fi) {
            const auto& tri = impl_->triangles[fi];
            double t;
            if (!moeller_trumbore(origin, dir,
                                  tri[0], tri[1], tri[2], &t)) continue;
            if (t < best_t_local) {
                best_t_local = t;
                r.hit        = true;
                r.face_index = fi;
                r.t          = t;
                r.point      = origin + t * dir;
            }
        }
        return r;
    };

    // Degenerate mesh — fall back.
    if (impl_->max_half_diag <= 1e-30) return linear_scan();

    // ---- Slab test: clip the ray against the mesh AABB in d_hat units ----
    double t_enter = 0.0;
    double t_exit  = std::numeric_limits<double>::infinity();
    for (int ax = 0; ax < 3; ++ax) {
        double o = origin[ax];
        double v = d_hat[ax];
        double mn = impl_->bbox_min[ax];
        double mx = impl_->bbox_max[ax];
        if (std::abs(v) < 1e-30) {
            // Ray parallel to this slab — miss if origin is outside.
            if (o < mn || o > mx) return res;
        } else {
            double inv = 1.0 / v;
            double t1 = (mn - o) * inv;
            double t2 = (mx - o) * inv;
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > t_enter) t_enter = t1;
            if (t2 < t_exit)  t_exit  = t2;
            if (t_enter > t_exit) return res;       // clean miss
        }
    }
    if (t_exit <= 0.0) return res;                  // mesh is entirely behind origin
    if (t_enter < 0.0) t_enter = 0.0;               // clamp to forward ray

    // ---- March parameters ----
    const double r = impl_->max_half_diag;
    const double step = 2.0 * r;
    const double search_r2 = step * step;           // radius² for nanoflann

    // Cap the number of samples.  For pathological meshes (e.g. one huge
    // triangle and a swarm of micro-triangles) max_half_diag can be tiny
    // relative to the mesh extent, producing millions of march steps.  In
    // that case a linear scan is faster.
    double span = t_exit - t_enter;
    double nsteps_f = span / step + 2.0;
    constexpr int kMaxSteps = 8192;
    if (nsteps_f > static_cast<double>(kMaxSteps)) return linear_scan();
    int nsteps = static_cast<int>(nsteps_f);

    // ---- Collect candidate triangles along the ray ----
    // unordered_set keeps dedup cheap; the expected candidate count is small
    // (tens to low hundreds) for typical rays on typical meshes.
    std::unordered_set<int> candidates;
    candidates.reserve(64);

    std::vector<nanoflann::ResultItem<uint32_t, double>> hits;
    hits.reserve(32);

    for (int s = 0; s <= nsteps; ++s) {
        double t = t_enter + static_cast<double>(s) * step;
        if (t > t_exit + step) break;               // over-step guard
        Vec3 p = origin + t * d_hat;
        double pc[3] = {p[0], p[1], p[2]};
        hits.clear();
        impl_->kd->radiusSearch(pc, search_r2, hits);
        for (const auto& h : hits) {
            int fi = static_cast<int>(h.first);
            if (fi >= 0 && fi < nf) candidates.insert(fi);
        }
    }

    if (candidates.empty()) return res;             // cylinder contained no centroids

    // ---- Test candidates, pick smallest positive t ----
    double best_t = std::numeric_limits<double>::infinity();
    for (int fi : candidates) {
        const auto& tri = impl_->triangles[fi];
        double t;
        if (!moeller_trumbore(origin, dir,
                              tri[0], tri[1], tri[2], &t)) continue;
        if (t < best_t) {
            best_t         = t;
            res.hit        = true;
            res.face_index = fi;
            res.t          = t;
            res.point      = origin + t * dir;
        }
    }
    return res;
}

} // namespace qf
