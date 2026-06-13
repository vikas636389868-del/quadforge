/**
 * omp_utils.cpp — OpenMP thread utilities for QuadForge.
 */

#include "../../include/quadforge/accel/omp_utils.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <thread>
#include <functional>

namespace qf {

// -----------------------------------------------------------------------
// resolve_thread_count
// -----------------------------------------------------------------------

int resolve_thread_count(int requested) {
    if (requested > 0) return requested;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    return (hw > 0) ? hw : 4;
#endif
}

// -----------------------------------------------------------------------
// parallel_for
//
// BUG FIX (v23): replaced omp_set_num_threads(nthreads) with the
// num_threads(nthreads) clause on the parallel region.
//
// omp_set_num_threads() mutates a process-global setting: any concurrent
// call from a different thread (e.g. two pipeline stages running
// simultaneously) could silently overwrite each other's thread count,
// leading to non-deterministic parallelism levels or OpenMP warnings.
//
// The num_threads() clause is LOCAL to a single parallel region and does
// not affect other regions — this is the correct way to control thread
// counts per call-site.
// -----------------------------------------------------------------------

void parallel_for(int count, std::function<void(int)> fn, int threads) {
    if (count <= 0) return;
    int nthreads = resolve_thread_count(threads);

#ifdef _OPENMP
    // BUG FIX (v49): replaced hardcoded chunk=64 with an adaptive formula.
    //
    // The previous schedule(dynamic, 64) caused thread starvation on any
    // loop with fewer than 64 * nthreads iterations (e.g. 10 feature-edge
    // chains on 8 threads: all work went to 1 thread, 7 threads idle).
    //
    // Adaptive formula:  chunk = max(8, count / (nthreads * 4))
    //   • `count / (nthreads * 4)` divides the range into 4 chunks per
    //     thread, giving each thread a reasonable initial slice while
    //     leaving headroom for dynamic load balancing.
    //   • The floor of 8 iterations amortises OpenMP scheduling overhead
    //     (task dispatch is ~200 ns on modern hardware, equivalent to ~8
    //     trivial loop iterations).
    //   • On large loops (>64 * nthreads) the formula converges to roughly
    //     the old chunk=64 behaviour.
    //
    // This mirrors the adaptive chunk already used in parallel_reduce
    // (omp_utils.h) for consistency.
    int chunk = std::max(1, std::max(count / (nthreads * 4), 8));
    #pragma omp parallel for schedule(dynamic, chunk) num_threads(nthreads)
    for (int i = 0; i < count; ++i) fn(i);
#else
    (void)nthreads;
    for (int i = 0; i < count; ++i) fn(i);
#endif
}

} // namespace qf
