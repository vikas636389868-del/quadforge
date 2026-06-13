#pragma once
/* quadforge/mesh/spatial.h — BVH spatial index for nearest-point queries.
 *
 * Wraps nanoflann for O(log N) closest-point and ray-intersection queries.
 */

#ifndef QUADFORGE_MESH_SPATIAL_H
#define QUADFORGE_MESH_SPATIAL_H

#include <cstdint>
#include <memory>
#include <vector>
#include "../types.h"

// nanoflann types are fully hidden behind the pImpl pattern (see spatial.cpp).
// No forward-declaration needed here.

namespace qf {

struct ClosestPointResult {
    Vec3   point;       ///< Closest point on the surface
    int    face_index;  ///< Triangle index (-1 = not found)
    double distance;    ///< Euclidean distance to closest point
};

struct RayHitResult {
    bool   hit{false};
    Vec3   point;
    int    face_index{-1};
    double t{0.0};  ///< Ray parameter
};

/**
 * Axis-aligned BVH over a triangle mesh for fast spatial queries.
 *
 * Build once after mesh construction; query many times.
 */
class TriangleBVH {
public:
    /**
     * Build the BVH from a triangle mesh.
     * @param vertices  [V, 3]
     * @param faces     [F, 3] triangle indices
     */
    TriangleBVH(const Eigen::MatrixXd& vertices,
                const Eigen::MatrixXi& faces);

    ~TriangleBVH();

    /**
     * Find the closest point on the mesh to query_point.
     * O(log F) expected.
     */
    ClosestPointResult closest_point(const Vec3& query_point) const;

    /**
     * Find all mesh points within radius of query_point.
     */
    std::vector<ClosestPointResult> points_within_radius(
        const Vec3& center, double radius) const;

    /**
     * Ray intersection: find first hit along ray (origin + t*dir).
     */
    RayHitResult ray_intersect(const Vec3& origin, const Vec3& dir) const;

    /** True if this BVH is valid and ready to query. */
    bool is_valid() const { return valid_; }

    int num_triangles() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool valid_{false};
};

} // namespace qf

#endif // QUADFORGE_MESH_SPATIAL_H
