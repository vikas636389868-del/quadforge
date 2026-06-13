/**
 * project.cpp — Project quad mesh vertices onto the input reference surface.
 *
 * v77 fix: iterative_smooth_and_project used integer division
 *   std::max(1, 10 / std::max(rounds, 1))
 * to spread ~10 total smooth iterations across `rounds` project cycles.
 * Integer division loses the fractional part: rounds=3 → 3 iters, rounds=4 → 2
 * iters, rounds=7 → 1 iter, giving 9, 8, or 7 total steps instead of 10.
 * Fixed with std::lround() so the formula rounds to the nearest integer,
 * keeping the cumulative iteration count close to 10 across all round values.
 */

#include "../../include/quadforge/postprocess/project.h"
#include "../../include/quadforge/postprocess/smooth.h"   // BUG-FIX: taubin_smooth() is called below
#include "../../include/quadforge/accel/omp_utils.h"

#include <algorithm>
#include <cmath>

namespace qf {

void project_to_surface(QuadMesh& quad_mesh,
                         const TriangleBVH& bvh,
                         int num_threads)
{
    if (!bvh.is_valid()) return;
    int nv = (int)quad_mesh.vertices.size();

    parallel_for(nv, [&](int vi) {
        Vec3 q = {quad_mesh.vertices[vi][0],
                  quad_mesh.vertices[vi][1],
                  quad_mesh.vertices[vi][2]};
        auto res = bvh.closest_point(q);
        if (res.face_index >= 0) {
            quad_mesh.vertices[vi][0] = res.point[0];
            quad_mesh.vertices[vi][1] = res.point[1];
            quad_mesh.vertices[vi][2] = res.point[2];
        }
    }, num_threads);
}

void iterative_smooth_and_project(
    QuadMesh&          quad_mesh,
    const TriangleBVH& bvh,
    int                rounds,
    double             lambda,
    double             mu,
    int                num_threads,
    int                smooth_iters_per_round)
{
    SmoothParams sp;
    // BUG FIX (v83 / Bug 4): smooth_iters_per_round is now forwarded from
    // stage5_postprocess() so the user's QFParams::smooth_iterations slider
    // is actually honoured.  When smooth_iters_per_round == 0 (old call
    // sites that don't pass the argument), fall back to the v77 formula:
    //   lround(10.0 / rounds)
    // which distributes ~10 total smooth steps evenly across rounds.
    // When smooth_iters_per_round > 0, use it directly — this is the path
    // taken by engine.cpp after the Bug 4 fix.
    if (smooth_iters_per_round > 0) {
        sp.iterations = smooth_iters_per_round;
    } else {
        // v77 backward-compatible formula (lround avoids the integer-truncation
        // bug that was present before v77 with plain integer division).
        sp.iterations = static_cast<int>(
            std::max(1L, std::lround(10.0 / std::max(rounds, 1))));
    }
    sp.lambda      = lambda;
    sp.mu          = mu;
    sp.num_threads = num_threads;

    for (int r = 0; r < rounds; ++r) {
        taubin_smooth(quad_mesh, sp);
        project_to_surface(quad_mesh, bvh, num_threads);
    }
}

} // namespace qf
