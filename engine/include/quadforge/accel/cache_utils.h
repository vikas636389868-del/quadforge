#pragma once
#ifndef QUADFORGE_ACCEL_CACHE_UTILS_H
#define QUADFORGE_ACCEL_CACHE_UTILS_H

/**
 * quadforge/accel/cache_utils.h — Cache-aligned allocation and prefetch utilities.
 *
 * WHY THIS EXISTS
 * ---------------
 * QuadForge's hot paths process millions of doubles per remesh:
 *
 *   • SpMV (A*p) in the CG solver streams row_ptr[n+1], col_idx[nnz],
 *     vals[nnz], x[n] — all accessed repeatedly per CG iteration.
 *   • Taubin smoothing iterates over all V vertices 5–20 times.
 *   • Cotangent Laplacian assembly reads 6 floats per edge, 3 per face.
 *
 * Two optimisations directly improve memory throughput:
 *
 * 1. ALIGNMENT: SIMD aligned loads (_mm256_load_pd, vld1q_f64) are faster
 *    than unaligned (_mm256_loadu_pd) on crossing-cacheline access because
 *    the hardware never needs to perform a split-line load.  std::vector
 *    guarantees only alignof(T) (= 8 for double), not 64-byte alignment.
 *    AlignedVector<T> guarantees 64-byte alignment for all elements.
 *
 * 2. PREFETCH: Software prefetch hints tell the CPU to start fetching a
 *    future cache line while the current iteration is computing, hiding
 *    main-memory latency (~100 cycles on DDR5).  On irregular access
 *    patterns (SpMV column gather) where hardware prefetchers cannot predict
 *    the access stream, software hints provide 10–30% throughput improvement.
 *
 * DESIGN
 * ------
 *   • AlignedAllocator<T, Alignment>  — STL-compatible allocator.
 *   • aligned_vector<T>               — std::vector with 64-byte alignment.
 *   • aligned_alloc_f64(n)            — raw helper for C++ new/delete.
 *   • prefetch_r(ptr)                 — prefetch for read.
 *   • prefetch_rw(ptr)                — prefetch for read-write.
 *   • prefetch_r_stream(ptr)          — non-temporal prefetch (write-combining).
 *   • CACHE_LINE_SIZE                 — compile-time constant (64 bytes).
 *
 * USAGE EXAMPLE
 * -------------
 *   // Allocate an aligned CG working vector.
 *   qf::aligned_vector<double> r(n), p(n), Ap(n);
 *
 *   // SpMV with software prefetch:
 *   for (int row = 0; row < n; ++row) {
 *       // Prefetch the next row's values 8 rows ahead.
 *       if (row + 8 < n)
 *           qf::prefetch_r(vals + row_ptr[row + 8]);
 *       y[row] = spmv_row(...);
 *   }
 *
 * PLATFORM NOTES
 * --------------
 *   x86 / x86_64 — uses _mm_prefetch() from <xmmintrin.h>.
 *   AArch64       — uses __builtin_prefetch() which maps to PRFM.
 *   Other         — falls back to __builtin_prefetch() (compiler hint only).
 *
 * This header is intentionally header-only (no .cpp) because all functions
 * are trivially inlinable and the STL allocator must be a class template.
 */

#include <cstddef>
#include <cstdlib>
#include <cstring>   // BUG FIX (v50): std::memset used in aligned_alloc_f64 but
                     // <cstring> was never included.  On GCC/Clang this produces
                     // "error: 'memset' was not declared in this scope" unless
                     // another included header pulls it in transitively (not
                     // guaranteed by the C++ standard).  MSVC is more permissive
                     // and often found it via <memory>, hiding the bug there.
#include <memory>
#include <new>
#include <vector>
#include <stdexcept>

// ── Platform prefetch includes ────────────────────────────────────────────

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <xmmintrin.h>   // _mm_prefetch
#  define QF_PREFETCH_X86 1
#else
#  define QF_PREFETCH_X86 0
#endif

namespace qf {

// =========================================================================
// Compile-time constants
// =========================================================================

/** Cache line size in bytes.  Universally 64 bytes on x86-64 and AArch64. */
static constexpr std::size_t CACHE_LINE_SIZE = 64;

// =========================================================================
// AlignedAllocator<T, Alignment>
//
// An STL-compatible allocator that guarantees Alignment-byte-aligned storage.
// Alignment must be a power of two and ≥ alignof(T).
//
// Compatible with std::vector, std::deque, and any STL container that
// accepts a custom allocator.
// =========================================================================

template <typename T, std::size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
    static_assert((Alignment & (Alignment - 1)) == 0,
                  "AlignedAllocator: Alignment must be a power of two.");
    static_assert(Alignment >= alignof(T),
                  "AlignedAllocator: Alignment must be at least alignof(T).");

public:
    using value_type = T;
    using size_type  = std::size_t;
    using pointer    = T*;

    // Rebind support for STL internals.
    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;

    template <typename U>
    explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    // -----------------------------------------------------------------
    // allocate
    // -----------------------------------------------------------------
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        std::size_t bytes = n * sizeof(T);

        void* ptr = nullptr;

#if defined(_MSC_VER)
        ptr = _aligned_malloc(bytes, Alignment);
        if (!ptr) throw std::bad_alloc{};
#else
        // POSIX: aligned_alloc requires size to be a multiple of alignment.
        std::size_t aligned_bytes =
            (bytes + Alignment - 1) & ~(Alignment - 1);
        ptr = std::aligned_alloc(Alignment, aligned_bytes);
        if (!ptr) throw std::bad_alloc{};
#endif
        return static_cast<T*>(ptr);
    }

    // -----------------------------------------------------------------
    // deallocate
    // -----------------------------------------------------------------
    void deallocate(T* p, std::size_t /*n*/) noexcept {
        if (!p) return;
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    // Two allocators of the same type and alignment are interchangeable.
    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }
    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>& o) const noexcept {
        return !(*this == o);
    }
};

// =========================================================================
// aligned_vector<T>
//
// std::vector with 64-byte-aligned storage.
// Drop-in replacement for std::vector<T> in solver hot paths.
//
// Usage:
//   qf::aligned_vector<double> r(n, 0.0);   // n doubles, cache-aligned
//   qf::simd::axpy(n, alpha, p.data(), r.data());  // guaranteed aligned
// =========================================================================

template <typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

// =========================================================================
// Raw helpers for one-time allocations
// =========================================================================

/**
 * Allocate n cache-aligned doubles, zero-initialised.
 * Caller must free with aligned_free_f64().
 */
inline double* aligned_alloc_f64(std::size_t n) {
    if (n == 0) return nullptr;
    std::size_t bytes = ((n * sizeof(double) + CACHE_LINE_SIZE - 1)
                         / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, CACHE_LINE_SIZE);
#else
    ptr = std::aligned_alloc(CACHE_LINE_SIZE, bytes);
#endif
    if (!ptr) throw std::bad_alloc{};
    // BUG FIX (v48): replaced scalar zero-init loop with std::memset.
    // IEEE 754 double zero is all-bits-zero, identical to integer zero, so
    // memset(ptr, 0, bytes) correctly zeroes every double element.
    // memset is typically SIMD-accelerated in libc and ~10× faster than a
    // scalar loop for large allocations (e.g. 500K-vertex CG working vectors).
    std::memset(ptr, 0, bytes);
    return static_cast<double*>(ptr);
}

inline void aligned_free_f64(double* p) noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// =========================================================================
// Software prefetch hints
//
// These map to architecture-specific prefetch instructions.  They are
// hints only — the hardware may ignore them.  Use them 8–16 iterations
// ahead of the iteration that will use the data (roughly matching the
// memory latency / loop iteration time ratio for your target CPU).
//
// WHEN TO USE
// -----------
//   • SpMV outer loop: prefetch vals[] and col_idx[] for row+8.
//   • Vertex position update: prefetch vertex[ring[j+4]] in ring loops.
//   • BVH traversal: prefetch child nodes before descending.
//
// WHEN NOT TO USE
// ---------------
//   • Already sequential strided access — hardware prefetcher handles it.
//   • Very short loops (< ~16 iterations) — prefetch overhead exceeds gain.
//   • Inside OpenMP parallel regions on very small per-thread ranges.
// =========================================================================

/**
 * Prefetch a cache line for reading.
 * Locality hint 3 = "keep in L1 cache as long as possible" (T0).
 */
inline void prefetch_r(const void* ptr) noexcept {
#if QF_PREFETCH_X86
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#else
    __builtin_prefetch(ptr, /*rw=*/0, /*locality=*/3);
#endif
}

/**
 * Prefetch a cache line for reading into L2 (T1 hint).
 * Use when the data will be used a bit further ahead and L1 pressure is high.
 */
inline void prefetch_r_l2(const void* ptr) noexcept {
#if QF_PREFETCH_X86
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T1);
#else
    __builtin_prefetch(ptr, /*rw=*/0, /*locality=*/2);
#endif
}

/**
 * Prefetch a cache line for read-write.
 * Use for output arrays that will be both written and read back.
 */
inline void prefetch_rw(const void* ptr) noexcept {
#if QF_PREFETCH_X86
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#else
    __builtin_prefetch(ptr, /*rw=*/1, /*locality=*/3);
#endif
}

/**
 * Non-temporal prefetch hint (NTA).
 * Use for data that is read only once and should NOT pollute the cache
 * (e.g., reading BVH node data during a traversal that will not revisit it).
 */
inline void prefetch_nta(const void* ptr) noexcept {
#if QF_PREFETCH_X86
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_NTA);
#else
    __builtin_prefetch(ptr, /*rw=*/0, /*locality=*/0);
#endif
}

// =========================================================================
// Cache line round-up helper
// =========================================================================

/**
 * Round n_bytes up to the next multiple of CACHE_LINE_SIZE.
 * Useful when computing strides for cache-friendly 2D arrays.
 */
constexpr std::size_t round_up_cache_line(std::size_t n_bytes) noexcept {
    return (n_bytes + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
}

} // namespace qf

#endif // QUADFORGE_ACCEL_CACHE_UTILS_H
