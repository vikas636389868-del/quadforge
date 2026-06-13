#pragma once
#ifndef QUADFORGE_ACCEL_GPU_SOLVER_H
#define QUADFORGE_ACCEL_GPU_SOLVER_H

/**
 * quadforge/accel/gpu_solver.h — GPU-accelerated sparse linear solver interface.
 *
 * This is an OPTIONAL component. The CPU path (Preconditioned Conjugate
 * Gradient in gpu_solver.cpp) is the primary fallback solver. GPU is a
 * bonus for users with NVIDIA / Apple Silicon hardware.
 *
 * ARCHITECTURE — three tiers:
 *   Tier 1 (always available): CPU-fallback PCG with Jacobi preconditioner
 *          + OpenMP parallelism. Implemented in gpu_solver.cpp. Always
 *          compiled; always linked. gpu_available() returns false.
 *
 *   Tier 2 (QF_GPU_ENABLED=1 + CUDA): CUDA sparse CG via cuSPARSE.
 *          Replace gpu_available() / gpu_solve_csr() with CUDA TUs.
 *
 *   Tier 3 (QF_GPU_ENABLED=1 + Metal): Apple Metal compute shader.
 *          Replace gpu_available() / gpu_solve_csr() with Metal TUs.
 *
 * ODR FIX (v45): Previous versions used #if !QF_GPU_ENABLED inline stubs
 * in this header.  That pattern causes an ODR violation: every TU that
 * included this header got the stub definitions (returning false/-1),
 * while gpu_solver.cpp provided a DIFFERENT non-inline definition of
 * the same functions.  The C++ standard requires that all definitions of
 * an inline function are token-identical (§6.2 ODR); the stubs and the
 * real PCG implementation violate this rule.  In practice the linker
 * favoured the inline stubs, making the entire CPU-fallback PCG solver
 * dead code even though it was compiled.
 *
 * Fix: this header now contains DECLARATIONS ONLY.  gpu_solver.cpp always
 * provides the sole definitions (CPU-fallback PCG).  Adding CUDA/Metal
 * backends means replacing gpu_solver.cpp with a GPU-capable TU, not
 * editing this header.
 *
 * Usage:
 *   if (qf::gpu_available()) {
 *       qf::gpu_solve_csr(row_ptr, col_idx, vals, n, rhs, x, max_iter, tol);
 *   }
 *   // If gpu_available() returns false, gpu_solve_csr() still works —
 *   // it runs the CPU-fallback PCG (Jacobi-preconditioned CG + OpenMP).
 */

#include <cstdint>

namespace qf {

/**
 * Query GPU availability.
 *
 * Returns true only if a compiled-in GPU backend (CUDA / Metal) detects a
 * suitable device at runtime.  In the default CPU-fallback build this always
 * returns false — callers should still call gpu_solve_csr(), which silently
 * runs the CPU PCG path.
 */
bool gpu_available();

/**
 * Solve a sparse SPD linear system  A x = rhs.
 *
 * Matrix A is provided in CSR (Compressed Sparse Row) format.
 * The solve uses Jacobi-preconditioned Conjugate Gradient (CG).
 *
 * On a GPU build (QF_GPU_ENABLED=1) the CG runs on the device via
 * cuSPARSE / Metal.  On the default CPU build it runs with OpenMP-
 * parallelised SpMV and SIMD-accelerated vector kernels (axpy, dot, scale
 * from simd_math.h).
 *
 * @param row_ptr     [n+1]  CSR row-pointer array (host memory).
 * @param col_idx     [nnz]  CSR column-index array (host memory).
 * @param vals        [nnz]  CSR non-zero values    (host memory).
 * @param n           Number of rows / columns (system dimension).
 * @param rhs         [n]    Right-hand side vector.
 * @param x           [n]    Initial guess on entry; solution on exit.
 * @param max_iter    Maximum CG iterations before giving up.
 * @param tol         Relative residual tolerance (‖r‖₂ / ‖b‖₂ < tol).
 * @param num_threads OpenMP thread count for all internal parallel work.
 *                    0 = auto-detect from hardware_concurrency / OMP_NUM_THREADS.
 *                    Pass QFParams.num_threads here so the solver respects the
 *                    user-configured thread budget rather than using the global
 *                    OpenMP state, which concurrent pipeline stages may alter.
 * @return            Number of iterations taken.
 *                    0  = initial guess was already within tolerance.
 *                   -1  = failed (singular matrix, max_iter exceeded, bad args).
 */
int gpu_solve_csr(
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double*  vals,
    int32_t        n,
    const double*  rhs,
    double*        x,
    int            max_iter    = 500,
    double         tol         = 1e-8,
    int            num_threads = 0
);

/**
 * Release any persistent GPU-side buffers (cached allocations, streams).
 * Call once at engine shutdown (qf_cleanup() in api.cpp).
 * No-op in the CPU-fallback build.
 */
void gpu_cleanup();

} // namespace qf

#endif // QUADFORGE_ACCEL_GPU_SOLVER_H
