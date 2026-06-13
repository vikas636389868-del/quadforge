/**
 * simd_math.cpp — Runtime CPU feature detection for QuadForge SIMD paths.
 *
 * This translation unit provides the non-inline implementations declared
 * at the bottom of simd_math.h:
 *
 *   qf::simd::cpu_has_avx2()         — CPUID leaf 7 EBX bit 5
 *   qf::simd::cpu_has_sse42()        — CPUID leaf 1 ECX bit 20
 *   qf::simd::cpu_has_neon()         — always true on AArch64
 *   qf::simd::runtime_simd_level()   — widest safe level string
 *   qf::simd::assert_simd_safety()   — warn if compiled > runtime
 *
 * WHY THIS MATTERS FOR DISTRIBUTED PREBUILTS
 * -------------------------------------------
 * CMake builds the engine with -march=native (GCC/Clang) or /arch:AVX2
 * (MSVC) for maximum performance on the build machine.  A binary built
 * on an AVX2 machine and distributed to an SSE4.2-only machine will
 * crash with SIGILL the moment an AVX2 instruction executes.
 *
 * qf_init() (api.cpp) calls assert_simd_safety() to detect this mismatch
 * early, before any SIMD code runs, and logs a diagnostic message.
 *
 * Platform notes:
 *   x86 / x86_64  — use __cpuid / __cpuidex (MSVC) or __cpuid_count (GCC)
 *   AArch64        — NEON is mandatory per the architecture spec; always true
 *   Other          — no SIMD, always "scalar"
 */

#include "../../include/quadforge/accel/simd_math.h"

#include <cstdio>   // fprintf, stderr
#include <cstring>  // memset

// ── Platform includes ──────────────────────────────────────────────────────

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define QF_ARCH_X86 1
#  if defined(_MSC_VER)
#    include <intrin.h>     // __cpuid, __cpuidex
#  else
#    include <cpuid.h>      // __cpuid_count (GCC / Clang)
#  endif
#else
#  define QF_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  define QF_ARCH_ARM64 1
#else
#  define QF_ARCH_ARM64 0
#endif

// ── Internal helpers ───────────────────────────────────────────────────────

#if QF_ARCH_X86

namespace {

/**
 * Portable CPUID wrapper.
 * leaf / subleaf map to EAX / ECX on entry.
 * Outputs are written to regs[0..3] = EAX,EBX,ECX,EDX.
 */
static void run_cpuid(int leaf, int subleaf, int regs[4]) noexcept {
#if defined(_MSC_VER)
    __cpuidex(regs, leaf, subleaf);
#else
    __cpuid_count(leaf, subleaf,
                  regs[0], regs[1], regs[2], regs[3]);
#endif
}

/** Check CPUID max leaf so we don't read beyond what the CPU reports. */
static int cpuid_max_leaf() noexcept {
    int regs[4] = {};
    run_cpuid(0, 0, regs);
    return regs[0];  // EAX = max basic leaf
}

static bool detect_sse42_x86() noexcept {
    // CPUID leaf 1, ECX bit 20 = SSE4.2
    if (cpuid_max_leaf() < 1) return false;
    int regs[4] = {};
    run_cpuid(1, 0, regs);
    return (regs[2] >> 20) & 1;  // ECX bit 20
}

static bool detect_avx2_x86() noexcept {
    // First confirm OS saves/restores YMM registers (OSXSAVE + AVX bits)
    if (cpuid_max_leaf() < 1) return false;
    int regs1[4] = {};
    run_cpuid(1, 0, regs1);

    bool avx_supported = (regs1[2] >> 28) & 1;  // ECX bit 28 — AVX
    bool osxsave       = (regs1[2] >> 27) & 1;  // ECX bit 27 — OSXSAVE

    if (!avx_supported || !osxsave) return false;

    // Check XCR0 bits 1-2 (SSE + AVX state enabled by OS)
#if defined(_MSC_VER)
    uint64_t xcr0 = _xgetbv(0);
#else
    // FIX (v45): The "=A" constraint on x86-64 covers only RAX (not RDX),
    // so the high 32 bits of XCR0 were silently discarded.  Use separate
    // "=a" / "=d" constraints for EAX and EDX, then reassemble manually.
    uint32_t xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    uint64_t xcr0 = (static_cast<uint64_t>(xcr0_hi) << 32) | xcr0_lo;
#endif
    bool ymm_saved = ((xcr0 & 0x6) == 0x6);
    if (!ymm_saved) return false;

    // CPUID leaf 7, subleaf 0, EBX bit 5 = AVX2
    if (cpuid_max_leaf() < 7) return false;
    int regs7[4] = {};
    run_cpuid(7, 0, regs7);
    return (regs7[1] >> 5) & 1;  // EBX bit 5
}

}  // anonymous namespace

#endif  // QF_ARCH_X86

// ── Public API ─────────────────────────────────────────────────────────────

namespace qf {
namespace simd {

bool cpu_has_avx2() noexcept {
#if QF_ARCH_X86
    // Memoise: CPUID is cheap but called potentially at every remesh invocation.
    static const bool val = detect_avx2_x86();
    return val;
#else
    return false;
#endif
}

bool cpu_has_sse42() noexcept {
#if QF_ARCH_X86
    static const bool val = detect_sse42_x86();
    return val;
#else
    return false;
#endif
}

bool cpu_has_neon() noexcept {
#if QF_ARCH_ARM64
    // NEON (Advanced SIMD) is mandatory in the AArch64 base architecture.
    // There is no runtime check needed — it is always present.
    return true;
#else
    return false;
#endif
}

const char* runtime_simd_level() noexcept {
    if (cpu_has_avx2())  return "AVX2";
    if (cpu_has_sse42()) return "SSE4.2";
    if (cpu_has_neon())  return "NEON";
    return "scalar";
}

bool assert_simd_safety() noexcept {
    // Determine the SIMD level this binary was compiled for.
    const char* compiled = simd_level();   // inline from header
    const char* runtime  = runtime_simd_level();

    // Map level names to a numeric rank for comparison.
    auto rank = [](const char* level) -> int {
        if (level[0] == 'A')  return 3;  // AVX2
        if (level[0] == 'S')  return 2;  // SSE4.2
        if (level[0] == 'N')  return 1;  // NEON
        return 0;                         // scalar
    };

    int compiled_rank = rank(compiled);
    int runtime_rank  = rank(runtime);

    if (compiled_rank > runtime_rank) {
        // This binary uses SIMD instructions the current CPU does not support.
        // Running any of the hot paths will raise SIGILL.
        std::fprintf(stderr,
            "[QuadForge] WARNING: SIMD mismatch — binary compiled for %s "
            "but CPU only supports %s.  Use the scalar or %s prebuilt binary "
            "for this platform to avoid SIGILL crashes.\n",
            compiled, runtime, runtime);
        return false;
    }

    // All good — compiled level is within what the CPU supports.
    return true;
}

}  // namespace simd
}  // namespace qf
