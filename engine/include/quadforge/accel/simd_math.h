/* quadforge/accel/simd_math.h
 *
 * SIMD-accelerated math utilities for QuadForge inner loops.
 *
 * Why this exists:
 *   The roadmap targets 5–10× speedup over QuadRemesher.  OpenMP thread
 *   parallelism delivers 4–6× on an 8-core CPU; SIMD vectorisation adds
 *   another 2–4× on top for the memory-bandwidth-bound inner loops:
 *   dot products, SpMV, Laplacian accumulation, and vertex-position
 *   update passes.
 *
 * Design:
 *   - Mostly header-only inline functions for the hot paths.
 *   - Compile-time dispatch: the widest SIMD level available at build
 *     time is selected via preprocessor macros (__AVX2__, __SSE4_2__,
 *     __ARM_NEON).  Falls back to scalar on any platform.
 *   - Runtime safety: simd_math.cpp provides cpu_has_avx2() /
 *     cpu_has_sse42() CPUID queries so callers can verify the binary's
 *     compiled SIMD level is safe on the current CPU.  Call
 *     assert_simd_safety() at engine init to detect mismatches early.
 *   - On ARM64 (Apple M1/M2) the NEON path activates automatically
 *     when __ARM_NEON is defined (NEON is mandatory on AArch64).
 *   - Namespace qf::simd isolates these from the rest of qf::.
 *
 * Usage:
 *   double s = qf::simd::dot(a.data(), b.data(), n);
 *   qf::simd::axpy(n, alpha, x, y);  // y += alpha * x
 *   qf::simd::spmv_csr(row, col, val, n, x, y);
 */

#pragma once
#ifndef QUADFORGE_ACCEL_SIMD_MATH_H
#define QUADFORGE_ACCEL_SIMD_MATH_H

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

// ── Platform capability detection ────────────────────────────────────────

#if defined(__AVX2__)
#  include <immintrin.h>
#  define QF_SIMD_AVX2  1
#  define QF_SIMD_SSE42 0
#  define QF_SIMD_NEON  0
#elif defined(__SSE4_2__)
#  include <nmmintrin.h>
#  define QF_SIMD_AVX2  0
#  define QF_SIMD_SSE42 1
#  define QF_SIMD_NEON  0
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#  define QF_SIMD_AVX2  0
#  define QF_SIMD_SSE42 0
#  define QF_SIMD_NEON  1
#else
#  define QF_SIMD_AVX2  0
#  define QF_SIMD_SSE42 0
#  define QF_SIMD_NEON  0
#endif

namespace qf {
namespace simd {

// =========================================================================
// dot — double-precision inner product
//
// Computes  Σ a[i] * b[i]  for i in [0, n).
// Uses:  AVX2 (256-bit, 4 doubles/cycle) → SSE4.2 (128-bit) → scalar.
// =========================================================================

inline double dot(const double* __restrict a,
                  const double* __restrict b,
                  int n) noexcept
{
    double s = 0.0;

#if QF_SIMD_AVX2
    // --- AVX2 path: process 4 doubles per iteration ---
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        acc = _mm256_fmadd_pd(va, vb, acc);   // fused multiply-add
    }
    // Horizontal sum of acc (4 lanes)
    __m128d lo  = _mm256_castpd256_pd128(acc);
    __m128d hi  = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    s = _mm_cvtsd_f64(sum);
    // Scalar tail
    for (; i < n; ++i) s += a[i] * b[i];

#elif QF_SIMD_SSE42
    // --- SSE4.2 path: process 2 doubles per iteration ---
    __m128d acc = _mm_setzero_pd();
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        acc = _mm_add_pd(acc, _mm_mul_pd(va, vb));
    }
    __m128d sh = _mm_shuffle_pd(acc, acc, 1);
    acc = _mm_add_pd(acc, sh);
    s = _mm_cvtsd_f64(acc);
    for (; i < n; ++i) s += a[i] * b[i];

#elif QF_SIMD_NEON
    // --- NEON path (ARM64): process 2 doubles per iteration ---
    float64x2_t acc = vdupq_n_f64(0.0);
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        acc = vfmaq_f64(acc, va, vb);
    }
    s = vaddvq_f64(acc);
    for (; i < n; ++i) s += a[i] * b[i];

#else
    // --- Scalar fallback with manual unroll-4 ---
    int i = 0;
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i+0] * b[i+0];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    s = s0 + s1 + s2 + s3;
    for (; i < n; ++i) s += a[i] * b[i];
#endif

    return s;
}

// =========================================================================
// norm2 — squared L2 norm
// =========================================================================

inline double norm2(const double* __restrict a, int n) noexcept
{
    return dot(a, a, n);
}

// =========================================================================
// axpy — y += alpha * x   (BLAS level-1)
//
// Critical in PCG: ~2 calls per iteration, dominant on large systems.
// =========================================================================

inline void axpy(int n, double alpha,
                 const double* __restrict x,
                 double*       __restrict y) noexcept
{
#if QF_SIMD_AVX2
    __m256d va = _mm256_set1_pd(alpha);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(x + i);
        __m256d vy = _mm256_loadu_pd(y + i);
        _mm256_storeu_pd(y + i, _mm256_fmadd_pd(va, vx, vy));
    }
    for (; i < n; ++i) y[i] += alpha * x[i];

#elif QF_SIMD_SSE42
    __m128d va = _mm_set1_pd(alpha);
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d vx = _mm_loadu_pd(x + i);
        __m128d vy = _mm_loadu_pd(y + i);
        _mm_storeu_pd(y + i, _mm_add_pd(vy, _mm_mul_pd(va, vx)));
    }
    for (; i < n; ++i) y[i] += alpha * x[i];

#elif QF_SIMD_NEON
    float64x2_t va = vdupq_n_f64(alpha);
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t vx = vld1q_f64(x + i);
        float64x2_t vy = vld1q_f64(y + i);
        vst1q_f64(y + i, vfmaq_f64(vy, va, vx));
    }
    for (; i < n; ++i) y[i] += alpha * x[i];

#else
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
#endif
}

// =========================================================================
// scale — x *= alpha
// =========================================================================

inline void scale(int n, double alpha, double* __restrict x) noexcept
{
#if QF_SIMD_AVX2
    __m256d va = _mm256_set1_pd(alpha);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(x + i, _mm256_mul_pd(_mm256_loadu_pd(x + i), va));
    for (; i < n; ++i) x[i] *= alpha;

#elif QF_SIMD_SSE42
    __m128d va = _mm_set1_pd(alpha);
    int i = 0;
    for (; i + 2 <= n; i += 2)
        _mm_storeu_pd(x + i, _mm_mul_pd(_mm_loadu_pd(x + i), va));
    for (; i < n; ++i) x[i] *= alpha;

#elif QF_SIMD_NEON
    float64x2_t va = vdupq_n_f64(alpha);
    int i = 0;
    for (; i + 2 <= n; i += 2)
        vst1q_f64(x + i, vmulq_f64(vld1q_f64(x + i), va));
    for (; i < n; ++i) x[i] *= alpha;

#else
    for (int i = 0; i < n; ++i) x[i] *= alpha;
#endif
}

// =========================================================================
// spmv_csr — Sparse Matrix-Vector multiply (CSR format)
//
//   y = A * x
//
// The inner loop (column traversal per row) is vectorised where the
// row is long enough; short rows fall back to scalar.
// =========================================================================

inline void spmv_csr(const int32_t* __restrict row_ptr,
                     const int32_t* __restrict col_idx,
                     const double*  __restrict vals,
                     int n,
                     const double*  __restrict x,
                     double*        __restrict y) noexcept
{
    for (int row = 0; row < n; ++row) {
        int32_t start = row_ptr[row];
        int32_t end   = row_ptr[row + 1];
        int32_t len   = end - start;
        double  sum   = 0.0;

#if QF_SIMD_AVX2
        int k = 0;
        __m256d acc = _mm256_setzero_pd();
        for (; k + 4 <= len; k += 4) {
            // Indirect gather: x[col_idx[start+k .. start+k+3]]
            // Use _mm256_i32gather_pd for AVX2 gather
            __m128i vcols = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(col_idx + start + k));
            __m256d vx  = _mm256_i32gather_pd(x, vcols, 8);
            __m256d vv  = _mm256_loadu_pd(vals + start + k);
            acc = _mm256_fmadd_pd(vv, vx, acc);
        }
        // Horizontal sum
        __m128d lo  = _mm256_castpd256_pd128(acc);
        __m128d hi  = _mm256_extractf128_pd(acc, 1);
        __m128d s2  = _mm_add_pd(lo, hi);
        s2  = _mm_hadd_pd(s2, s2);
        sum = _mm_cvtsd_f64(s2);
        // Scalar tail
        for (; k < len; ++k)
            sum += vals[start + k] * x[col_idx[start + k]];
#elif QF_SIMD_SSE42
        // BUG 8 FIX (v46): SSE4.2 has no gather instruction, so the previous
        // code fell through to the pure scalar path and left the SSE FP units
        // idle.  Fix: load 2 consecutive vals[] into a 128-bit register with
        // _mm_loadu_pd (fast, contiguous), perform 2 scalar x[] fetches (the
        // unavoidable gather penalty), then _mm_mul_pd in XMM.
        // Net gain: half the number of scalar FP multiply instructions; the
        // vals[] memory traffic is fully vectorised; the SSE adder is used.
        {
            __m128d acc = _mm_setzero_pd();
            int32_t k   = 0;
            for (; k + 2 <= len; k += 2) {
                __m128d vv = _mm_loadu_pd(vals + start + k);
                __m128d vx = _mm_set_pd(
                    x[col_idx[start + k + 1]],
                    x[col_idx[start + k + 0]]);
                acc = _mm_add_pd(acc, _mm_mul_pd(vv, vx));
            }
            __m128d sh = _mm_shuffle_pd(acc, acc, 1);
            acc  = _mm_add_pd(acc, sh);
            sum  = _mm_cvtsd_f64(acc);
            for (; k < len; ++k)
                sum += vals[start + k] * x[col_idx[start + k]];
        }
#elif QF_SIMD_NEON
        // NEON path (AArch64) — 2-wide vectorised gather.
        // NEON has no hardware gather instruction (same as SSE4.2), but we
        // load two consecutive vals[] elements into a float64x2_t and
        // perform two independent x[] scalar fetches, then vfmaq_f64 to
        // accumulate.  This halves the number of scalar FMUL instructions
        // and uses NEON's 2-wide FP pipeline.
        //
        // BUG FIX (v49): This block was previously guarded by a bare #else,
        // meaning it was compiled on ANY platform that lacked AVX2 and SSE4.2
        // (e.g. scalar/RISC-V/WASM).  On those platforms arm_neon.h is not
        // included and the NEON types (float64x2_t, vdupq_n_f64, vld1q_f64,
        // vcombine_f64, vdup_n_f64, vfmaq_f64, vaddvq_f64) are undefined,
        // causing a compilation failure.  Changed to #elif QF_SIMD_NEON so
        // the code only compiles when __ARM_NEON is defined and arm_neon.h
        // has been included at the top of this header.
        {
            float64x2_t acc = vdupq_n_f64(0.0);
            int32_t k = 0;
            for (; k + 2 <= len; k += 2) {
                float64x2_t vv = vld1q_f64(vals + start + k);
                float64x2_t vx = vcombine_f64(
                    vdup_n_f64(x[col_idx[start + k + 0]]),
                    vdup_n_f64(x[col_idx[start + k + 1]]));
                acc = vfmaq_f64(acc, vv, vx);
            }
            sum = vaddvq_f64(acc);
            for (; k < len; ++k)
                sum += vals[start + k] * x[col_idx[start + k]];
        }
#else
        // Pure scalar fallback — no SIMD available (e.g. RISC-V, WASM,
        // generic ARM32 without NEON).  Compiler may still auto-vectorise
        // the vals[] contiguous load; the x[] gather cannot be vectorised.
        for (int32_t k = 0; k < len; ++k)
            sum += vals[start + k] * x[col_idx[start + k]];
#endif

        y[row] = sum;
    }
}

// =========================================================================
// vec3_dot / vec3_cross / vec3_norm — 3-component geometry helpers
//
// These are used in curvature computation and feature detection where
// very large numbers of (x,y,z) operations happen per face/edge.
// =========================================================================

struct Vec3d {
    double x, y, z;
};

inline double vec3_dot(Vec3d a, Vec3d b) noexcept {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline Vec3d vec3_cross(Vec3d a, Vec3d b) noexcept {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

inline double vec3_norm(Vec3d a) noexcept {
    return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

inline Vec3d vec3_normalize(Vec3d a) noexcept {
    double len = vec3_norm(a);
    if (len < 1e-15) return {0,0,0};
    double inv = 1.0 / len;
    return { a.x*inv, a.y*inv, a.z*inv };
}

inline Vec3d vec3_sub(Vec3d a, Vec3d b) noexcept {
    return { a.x-b.x, a.y-b.y, a.z-b.z };
}

inline Vec3d vec3_add(Vec3d a, Vec3d b) noexcept {
    return { a.x+b.x, a.y+b.y, a.z+b.z };
}

inline Vec3d vec3_scale(Vec3d a, double s) noexcept {
    return { a.x*s, a.y*s, a.z*s };
}

// =========================================================================
// Additional vec3 helpers added in v45
//
// These complete the set needed by quality_optimizer, edge_flow_refine,
// and smooth.cpp — previously those files had local duplicates.
// =========================================================================

/** Linear interpolation: (1-t)*a + t*b */
inline Vec3d vec3_lerp(Vec3d a, Vec3d b, double t) noexcept {
    double it = 1.0 - t;
    return { it*a.x + t*b.x,
             it*a.y + t*b.y,
             it*a.z + t*b.z };
}

/** Fused multiply-add: a + s * b  (avoids a temporary Vec3d) */
inline Vec3d vec3_madd(Vec3d a, double s, Vec3d b) noexcept {
    return { a.x + s*b.x,
             a.y + s*b.y,
             a.z + s*b.z };
}

/** Negate: -a */
inline Vec3d vec3_neg(Vec3d a) noexcept {
    return { -a.x, -a.y, -a.z };
}

/** Component-wise multiply (Hadamard product) */
inline Vec3d vec3_mul(Vec3d a, Vec3d b) noexcept {
    return { a.x*b.x, a.y*b.y, a.z*b.z };
}

/** Squared distance between two points */
inline double vec3_dist2(Vec3d a, Vec3d b) noexcept {
    Vec3d d = vec3_sub(a, b);
    return vec3_dot(d, d);
}

/** Euclidean distance between two points */
inline double vec3_dist(Vec3d a, Vec3d b) noexcept {
    return std::sqrt(vec3_dist2(a, b));
}

/**
 * Batch normalise an array of Vec3d in-place with SIMD.
 * Vectors with length < epsilon are set to {0,0,0}.
 * Used in bulk normal renormalisation after Taubin smoothing.
 */
inline void vec3_normalize_batch(Vec3d* vecs, int n,
                                  double epsilon = 1e-15) noexcept {
    // Scalar loop; the compiler auto-vectorises the independent divisions.
    // AVX2 rsqrt is single-precision only, so we keep double-precision here.
    for (int i = 0; i < n; ++i) {
        double len2 = vec3_dot(vecs[i], vecs[i]);
        if (len2 < epsilon * epsilon) {
            vecs[i] = {0.0, 0.0, 0.0};
        } else {
            double inv = 1.0 / std::sqrt(len2);
            vecs[i] = { vecs[i].x * inv, vecs[i].y * inv, vecs[i].z * inv };
        }
    }
}

// =========================================================================
// SIMD capability query (runtime)
// =========================================================================

/** Returns a string describing the SIMD level active at compile time. */
inline const char* simd_level() noexcept {
#if QF_SIMD_AVX2
    return "AVX2";
#elif QF_SIMD_SSE42
    return "SSE4.2";
#elif QF_SIMD_NEON
    return "NEON";
#else
    return "scalar";
#endif
}

// =========================================================================
// Runtime CPU feature detection (implemented in simd_math.cpp)
//
// These functions use CPUID on x86 and compile-time constants on ARM64.
// They are non-inline so the detection logic is compiled once (simd_math.cpp)
// and avoids code-size bloat from inlining a heavy CPUID sequence everywhere.
//
// Usage at engine init:
//   qf::simd::assert_simd_safety();   // logs a warning if compiled SIMD
//                                     // level exceeds runtime capability
// =========================================================================

/** Returns true if the current CPU supports AVX2 at runtime. */
bool cpu_has_avx2() noexcept;

/** Returns true if the current CPU supports SSE4.2 at runtime. */
bool cpu_has_sse42() noexcept;

/** Returns true if the current CPU supports NEON at runtime (ARM64 always true). */
bool cpu_has_neon() noexcept;

/**
 * Returns a human-readable string of the widest SIMD level the current
 * CPU supports at runtime: "AVX2", "SSE4.2", "NEON", or "scalar".
 * This may differ from simd_level() when a binary compiled with AVX2 is
 * running on an SSE4.2-only machine — in which case assert_simd_safety()
 * will emit a warning.
 */
const char* runtime_simd_level() noexcept;

/**
 * Safety check: logs a warning (via stderr) if the compile-time SIMD
 * level exceeds what the current CPU actually supports.
 *
 * Call this once during engine initialisation (qf_init()).  On a mismatch
 * it writes a warning but does NOT abort — the caller can decide whether
 * to continue (relying on the OS SIGILL handler) or disable the engine.
 *
 * On correctly targeted builds (e.g. ARM64 binary on ARM64 hardware) this
 * is a no-op.
 *
 * @return  true if safe (compiled level ≤ runtime level), false on mismatch.
 */
bool assert_simd_safety() noexcept;

} // namespace simd
} // namespace qf

#endif // QUADFORGE_ACCEL_SIMD_MATH_H
