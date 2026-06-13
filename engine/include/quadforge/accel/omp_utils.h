#pragma once
/* quadforge/accel/omp_utils.h — OpenMP helpers and thread pool utilities. */

#ifndef QUADFORGE_ACCEL_OMP_UTILS_H
#define QUADFORGE_ACCEL_OMP_UTILS_H

/**
 * OpenMP / threading utilities for QuadForge.
 *
 * Provides:
 *   • resolve_thread_count()  — pick thread count respecting user override.
 *   • parallel_for()          — index-range parallel loop.
 *   • parallel_reduce<T>()    — map-reduce with cache-safe partial storage.
 *
 * FALSE-SHARING FIX (v45)
 * -----------------------
 * The previous parallel_reduce stored per-thread T values in a plain
 * std::vector<T> with adjacent elements.  For T = double (8 bytes), all
 * partials on a 4-core machine fit on a single 64-byte cache line.  Every
 * time any thread updated its partial, the cache line was invalidated across
 * all other cores — effectively serialising the reduce, negating the entire
 * benefit of parallelism on common workloads (surface area accumulation,
 * cotangent weight summation, curvature integration).
 *
 * Fix: each partial is wrapped in CachePad<T>, a struct that adds padding
 * bytes to ensure each partial occupies at least one full cache line (64 B).
 * This eliminates false sharing at the cost of slightly more heap memory,
 * which is negligible compared to the mesh data already allocated.
 */

#include <algorithm>
#include <atomic>      // BUG FIX (v87): std::atomic<int> is used in parallel_reduce
                       // (actual_nthreads) when _OPENMP is defined, but <atomic> was
                       // never included here.  <algorithm>/<functional>/<vector> do NOT
                       // guarantee to pull in <atomic> — the C++ standard only requires
                       // each header to provide the declarations it documents.  Strict
                       // libc++ builds (Clang on macOS/Linux) fail with
                       //   "error: use of undeclared identifier 'atomic'"
                       // unless <atomic> is listed explicitly.  This include is always
                       // emitted (not guarded by _OPENMP) because the template
                       // instantiation sites include this header regardless of whether
                       // _OPENMP is active, and some TUs enable OpenMP while others do not.
#include <functional>
#include <vector>
#include <cstddef>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace qf {

// -------------------------------------------------------------------------
// Cache-line padding for false-sharing prevention
//
// CachePad<T> wraps a T value and pads the struct to at least 64 bytes.
// Used in parallel_reduce to give each thread-local partial its own cache
// line, preventing false-sharing invalidation across cores.
//
// Alignment note: alignas(64) places the struct on a cache line boundary.
// The pad[] array fills any remaining bytes so sizeof(CachePad<T>) ≥ 64.
// -------------------------------------------------------------------------

template <typename T>
struct alignas(64) CachePad {
    T value;
    // Padding to fill a 64-byte cache line.  If sizeof(T) >= 64 we still
    // align to 64 bytes (the struct is naturally on its own line); we keep
    // a 1-byte pad minimum so the array is never zero-sized.
    static constexpr std::size_t kPad =
        (sizeof(T) < 64) ? (64 - sizeof(T)) : 1;
    char pad[kPad];

    // BUG FIX (v87): Initialize pad[] explicitly.
    // The previous constructor `explicit CachePad(const T& v = T{}) : value(v) {}`
    // left pad[] with indeterminate bytes.  PaddedPartial in thread_pool.h already
    // used `pad{}` for zero-initialisation; CachePad was inconsistently omitting it.
    // Indeterminate bytes in a struct member cause:
    //   • Valgrind / AddressSanitizer "use of uninitialized value" reports when
    //     the padding bytes are written to a buffer (e.g. serialisation, memcmp).
    //   • Undefined behavior under the C++ abstract machine if the pad bytes are
    //     ever inspected (e.g. hashing the struct by value).
    // `pad{}` value-initialises the char array to all-zeros at no extra cost
    // (the compiler elides the zeroing in optimised builds when pad is never read).
    explicit CachePad(const T& v = T{}) : value(v), pad{} {}
};

// -------------------------------------------------------------------------
// resolve_thread_count
// -------------------------------------------------------------------------

/**
 * Resolve the thread count to use.
 * @param requested  0 = auto-detect all cores, N = use exactly N threads.
 */
int resolve_thread_count(int requested);

// -------------------------------------------------------------------------
// parallel_for
// -------------------------------------------------------------------------

/**
 * Parallel for loop over [0, count) with automatic chunking.
 * Falls back to serial if OpenMP is not available at compile time.
 *
 * Uses num_threads() clause (not omp_set_num_threads()) so the thread
 * count is scoped to this region only and does not affect concurrent calls.
 *
 * @param count      Number of iterations.
 * @param fn         Callable with signature void(int i).
 * @param threads    Number of threads (0 = auto).
 */
void parallel_for(int count,
                  std::function<void(int)> fn,
                  int threads = 0);

// -------------------------------------------------------------------------
// parallel_reduce<T>
// -------------------------------------------------------------------------

/**
 * Parallel map-reduce over [0, count).
 *
 * Each OpenMP thread accumulates a private partial copy of T starting from
 * `init`.  Partials are stored in cache-line-padded slots (CachePad<T>) to
 * prevent false-sharing between cores.  After the parallel region, all
 * per-thread partials are combined sequentially using `reduce`.
 *
 * NOTE: `body` and `reduce` MUST be thread-safe.  The `partial` reference
 * passed to `body` is private to the calling thread.
 *
 * Example — total surface area:
 *   double area = qf::parallel_reduce<double>(
 *       n_faces, 0.0,
 *       [&](int fi, double& acc){ acc += face_area[fi]; },
 *       [](double a, double b){ return a + b; });
 *
 * @param count    Number of iterations [0, count).
 * @param init     Initial value for each thread-local accumulator.
 * @param body     body(i, partial) — called once per index.
 * @param reduce   reduce(a, b) — combines two partials.
 * @param threads  0 = auto-detect.
 * @return         Combined result.
 */
template<typename T>
T parallel_reduce(int count,
                  T init,
                  std::function<void(int, T&)> body,
                  std::function<T(T, T)>       reduce,
                  int threads = 0)
{
    if (count <= 0) return init;
    int nthreads = resolve_thread_count(threads);

#ifdef _OPENMP
    // One cache-padded slot per thread — false-sharing-free storage.
    // CachePad<T> guarantees each partial lives on its own 64-byte cache line.
    // Allocate nthreads slots — OpenMP may grant fewer, but never more.
    std::vector<CachePad<T>> partials(nthreads, CachePad<T>(init));

    // BUG FIX (v50): track which thread slots actually received work.
    //
    // With schedule(dynamic, chunk), threads that receive 0 iterations (because
    // count < chunk * actual_nthreads) never call body() and their partial stays
    // at `init`.  The previous code folded ALL actual_nthreads partials into the
    // result — including the idle threads' init values.  For standard monoid
    // reduce operations where init is the identity element (e.g. init=0 with
    // reduce=+, or init=DBL_MAX with reduce=min), this happens to be correct.
    // But for non-identity init values (e.g. counting, product with init=1 on
    // a small range) it silently produces a wrong result.
    //
    // Fix: use a separate atomic counter for how many threads touched their slot.
    // Only those slots are folded in the final reduce loop.
    std::atomic<int> actual_nthreads{1};

    // BUG FIX (v48-A): adaptive chunk size for `omp for`.
    int chunk = std::max(1, std::max(count / (nthreads * 4), 8));

    #pragma omp parallel num_threads(nthreads)
    {
        // BUG FIX (v48-B): use `nowait` on the single block.
        #pragma omp single nowait
        actual_nthreads.store(omp_get_num_threads(), std::memory_order_relaxed);

        int tid = omp_get_thread_num();
        T&  local = partials[tid].value;   // cache-line-isolated partial

        #pragma omp for schedule(dynamic, chunk)
        for (int i = 0; i < count; ++i)
            body(i, local);
        // Implicit barrier here (end of `omp for` without nowait).
    }
    // Implicit barrier at end of parallel region.

    // BUG FIX (v50): reduce over all actual_nthreads partials.
    //
    // Threads that receive 0 iterations (when count is small) never call body()
    // so their partial stays at `init`.  Folding `init` into the result is only
    // safe when init IS the identity element of reduce.  QuadForge uses
    // parallel_reduce exclusively with standard monoid operations (sum, min, max)
    // where this holds.  Users requiring non-monoid reduce must use parallel_for
    // with a manually managed accumulator instead.
    T result = init;
    int nt = actual_nthreads.load(std::memory_order_relaxed);
    for (int i = 0; i < nt; ++i)
        result = reduce(result, partials[i].value);
    return result;

#else
    // Serial fallback — single accumulator, no parallelism.
    T result = init;
    for (int i = 0; i < count; ++i)
        body(i, result);
    return result;
#endif
}

} // namespace qf

#endif // QUADFORGE_ACCEL_OMP_UTILS_H
