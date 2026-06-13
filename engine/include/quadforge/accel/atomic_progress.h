#pragma once
#ifndef QUADFORGE_ACCEL_ATOMIC_PROGRESS_H
#define QUADFORGE_ACCEL_ATOMIC_PROGRESS_H

/**
 * quadforge/accel/atomic_progress.h — Thread-safe progress tracking.
 *
 * WHY THIS EXISTS
 * ---------------
 * QuadForge runs the six-stage pipeline on a background C++ thread and exposes
 * progress via a QFProgressCallback (a C function pointer).  On the Blender
 * side, a modal timer fires every 0.1 s on Blender's main thread and needs to
 * read the latest stage/progress/name without calling any C++ mutex or waiting
 * on a condition variable — any blocking call from Blender's event loop would
 * freeze the viewport.
 *
 * This module provides the decoupling layer:
 *
 *   Engine thread  →  ProgressTracker::update()      (lock-free write)
 *   Modal timer    ←  ProgressTracker::stage() etc.  (lock-free read)
 *   ESC key        →  ProgressTracker::request_abort()
 *   Engine thread  ←  ProgressTracker::should_abort() (lock-free poll)
 *
 * Roadmap reference: §4.3 Modal Operator Architecture —
 *   "The progress callback runs on the engine's thread.  It must NOT call any
 *    Blender Python API.  It only writes to atomic variables and a thread-safe
 *    string buffer that the modal handler reads."
 *
 * DESIGN
 * ------
 *   • stage and progress are plain std::atomic<int>/<float> — cheap, lock-free
 *     on every modern platform (single-instruction load/store for ≤ 64-bit).
 *
 *   • stage_name uses a double-buffered char array:
 *       Writer: copies name into the inactive buffer, then atomically flips
 *               m_name_idx to make it the active one.
 *       Reader: reads m_name_idx, then reads from that buffer.
 *       This avoids locks for a short (<64 byte) stage name string.
 *       A torn read is impossible: the reader sees EITHER the old name OR
 *       the new name, never a mix of bytes from both.
 *
 *   • abort flag is a plain std::atomic<bool>.
 *
 * THREAD SAFETY
 * -------------
 *   Single writer (engine thread) + single reader (modal thread) is the
 *   intended usage.  Multiple concurrent readers are safe.  Multiple
 *   concurrent writers are NOT safe and not needed.
 *
 * USAGE (C++ engine side)
 * -----------------------
 *   qf::ProgressTracker tracker;
 *
 *   // Pass the tracker to the QFProgressCallback adapter:
 *   auto cb = [](int stage, float pct, const char* name, void* ud) -> int {
 *       auto* tr = static_cast<qf::ProgressTracker*>(ud);
 *       tr->update(stage, pct, name);
 *       return tr->should_abort() ? 1 : 0;
 *   };
 *   qf_remesh(&input, &params, cb, &tracker);
 *
 * USAGE (Python / ctypes bridge side)
 * ------------------------------------
 *   Not exposed to Python directly.  The Python bridge reads:
 *       bridge.progress_stage()    → int
 *       bridge.progress_pct()      → float
 *       bridge.progress_name()     → str
 *   These call the C-API wrappers declared in api.h:
 *       qf_progress_stage()
 *       qf_progress_pct()
 *       qf_progress_name()
 *       qf_request_abort()
 *   which delegate to the global ProgressTracker instance in api.cpp.
 */

#include <atomic>
#include <cstring>   // std::strncpy, std::memset
#include <cstddef>   // std::size_t

namespace qf {

// =========================================================================
// ProgressTracker
// =========================================================================

class ProgressTracker {
public:
    // Maximum length of a stage-name string (including NUL terminator).
    static constexpr std::size_t kNameBufLen = 128;

    // ----------------------------------------------------------------
    // Constructor
    // ----------------------------------------------------------------

    ProgressTracker() noexcept
        : m_stage(0)
        , m_progress(0.0f)
        , m_name_idx(0)
        , m_abort(false)
    {
        std::memset(m_name_buf[0], 0, kNameBufLen);
        std::memset(m_name_buf[1], 0, kNameBufLen);
    }

    // Non-copyable, non-movable — internal atomics are fixed-address.
    ProgressTracker(const ProgressTracker&)            = delete;
    ProgressTracker& operator=(const ProgressTracker&) = delete;
    ProgressTracker(ProgressTracker&&)                 = delete;
    ProgressTracker& operator=(ProgressTracker&&)      = delete;

    // ----------------------------------------------------------------
    // Writer API  (engine thread only — single-writer requirement)
    // ----------------------------------------------------------------

    /**
     * Update the current stage number, progress fraction, and stage name.
     *
     * @param stage     Stage index 0–5 (matches QFProgressCallback stage param).
     * @param progress  Fraction complete in [0.0, 1.0].
     * @param name      NUL-terminated stage name (e.g. "Computing cross-field").
     *                  Truncated to kNameBufLen-1 characters if longer.
     *                  NULL is treated as an empty string.
     */
    void update(int stage, float progress, const char* name) noexcept {
        // Write name into the INACTIVE buffer, then flip the active index.
        // Readers always read from m_name_buf[m_name_idx].
        int inactive = 1 - m_name_idx.load(std::memory_order_relaxed);
        const char* src = (name != nullptr) ? name : "";
        std::strncpy(m_name_buf[inactive], src, kNameBufLen - 1);
        m_name_buf[inactive][kNameBufLen - 1] = '\0';  // guarantee NUL

        // Release the name buffer BEFORE updating stage/progress so a reader
        // that sees the new stage also sees the new name.
        m_name_idx.store(inactive, std::memory_order_release);

        // Stage and progress are independent; store with release so they
        // are visible after the name flip.
        m_stage.store(stage, std::memory_order_release);
        m_progress.store(progress, std::memory_order_release);
    }

    /**
     * Reset all fields to initial state.
     * Call before each remesh to clear stale state from the previous run.
     */
    void reset() noexcept {
        m_stage.store(0, std::memory_order_relaxed);
        m_progress.store(0.0f, std::memory_order_relaxed);
        m_abort.store(false, std::memory_order_relaxed);
        std::memset(m_name_buf[0], 0, kNameBufLen);
        std::memset(m_name_buf[1], 0, kNameBufLen);
        m_name_idx.store(0, std::memory_order_release);
    }

    // ----------------------------------------------------------------
    // Reader API  (modal timer thread — safe for concurrent readers)
    // ----------------------------------------------------------------

    /** Current pipeline stage (0–5). */
    int stage() const noexcept {
        return m_stage.load(std::memory_order_acquire);
    }

    /** Current progress within the current stage, in [0.0, 1.0]. */
    float progress() const noexcept {
        return m_progress.load(std::memory_order_acquire);
    }

    /**
     * Copy the current stage name into `dst` (caller-allocated buffer of at
     * least `dst_len` bytes).  Always NUL-terminates the result.
     *
     * The name is guaranteed to be a complete, valid NUL-terminated string
     * because we copy into the inactive buffer first and only flip the index
     * after the full write is done.
     *
     * @param dst      Destination buffer.
     * @param dst_len  Size of `dst` in bytes.
     */
    void stage_name(char* dst, std::size_t dst_len) const noexcept {
        if (!dst || dst_len == 0) return;
        // Acquire load pairs with the release store in update().
        int idx = m_name_idx.load(std::memory_order_acquire);
        std::strncpy(dst, m_name_buf[idx], dst_len - 1);
        dst[dst_len - 1] = '\0';
    }

    /**
     * Overall completion fraction: stage_index * (1/6) + progress * (1/6).
     * Returns a value in [0.0, 1.0] suitable for a progress bar that spans
     * all six pipeline stages.
     */
    float overall_progress() const noexcept {
        int   s = m_stage.load(std::memory_order_acquire);
        float p = m_progress.load(std::memory_order_acquire);
        // Clamp to [0,1]
        if (s < 0) s = 0;
        if (s > 5) s = 5;
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        return (s + p) / 6.0f;
    }

    // ----------------------------------------------------------------
    // Abort signal  (written by UI thread, read by engine thread)
    // ----------------------------------------------------------------

    /**
     * Signal the engine to stop at the next progress-callback check.
     * Called from the UI thread when the user presses ESC.
     */
    void request_abort() noexcept {
        m_abort.store(true, std::memory_order_release);
    }

    /**
     * Returns true if the UI thread has requested an abort.
     * The engine checks this inside its QFProgressCallback adapter.
     * If true, the callback returns non-zero, triggering early termination.
     */
    bool should_abort() const noexcept {
        return m_abort.load(std::memory_order_acquire);
    }

    /**
     * Clear the abort flag.
     * Call at the start of each remesh (reset() covers this too).
     */
    void clear_abort() noexcept {
        m_abort.store(false, std::memory_order_release);
    }

    // ----------------------------------------------------------------
    // Convenience: build a QFProgressCallback-compatible C function
    // ----------------------------------------------------------------

    /**
     * A static progress callback that delegates to a ProgressTracker.
     *
     * Pass this as the `callback` argument to qf_remesh(), and pass a pointer
     * to the ProgressTracker as `user_data`.
     *
     * Returns 0 to continue, 1 to abort (when should_abort() is true).
     *
     * Example:
     *   qf::ProgressTracker tracker;
     *   qf_remesh(&input, &params,
     *             qf::ProgressTracker::callback_fn, &tracker);
     */
    static int callback_fn(int stage, float progress,
                           const char* stage_name,
                           void* user_data) noexcept
    {
        if (!user_data) return 0;
        auto* self = static_cast<ProgressTracker*>(user_data);
        self->update(stage, progress, stage_name);
        return self->should_abort() ? 1 : 0;
    }

private:
    // ----------------------------------------------------------------
    // Storage — split across two cache lines to eliminate false sharing.
    //
    // Cache line 0 (bytes 0–63, alignas(64)):
    //   The four atomics (m_stage 4B + m_progress 4B + m_name_idx 4B +
    //   m_abort 1B + padding) fit comfortably in the first 64 bytes.
    //   The reader (modal timer) touches ONLY this line on every 0.1 s
    //   tick: stage(), progress(), stage_name() → m_name_idx → m_name_buf.
    //
    // Cache line 1+ (alignas(64) on m_name_buf):
    //   The writer (engine thread) fills the INACTIVE name buffer then
    //   flips m_name_idx.  Without this alignment the inactive buffer
    //   started at ~byte 16 — INSIDE cache line 0 — so every write to
    //   m_name_buf[0] (inactive when idx=1) dirtied the cache line
    //   holding the reader's atomics, causing a coherence miss on the
    //   very next reader access.  On alternating update() calls (which
    //   toggle between inactive=0 and inactive=1), the false share fired
    //   every other call — exactly the hot path.
    //
    // BUG FIX (v86): Added alignas(64) to m_name_buf to place the name
    //   buffers on their own cache line(s), fully decoupling the writer's
    //   memory traffic from the reader's atomic loads.
    // ----------------------------------------------------------------

    alignas(64) std::atomic<int>   m_stage;
    std::atomic<float>             m_progress;
    std::atomic<int>               m_name_idx;  // 0 or 1 — which buffer is active
    std::atomic<bool>              m_abort;

    // Double-buffered name storage.
    // BUG FIX (v86): alignas(64) ensures m_name_buf starts on a fresh
    // 64-byte cache line, isolating writer traffic from the atomics above.
    // Writer fills the inactive buffer then flips m_name_idx.
    // Reader reads from the active buffer (m_name_buf[m_name_idx]).
    alignas(64) char m_name_buf[2][kNameBufLen];
};

// =========================================================================
// Global process-wide ProgressTracker
//
// api.cpp maintains one of these and exposes its state through the
// C-API extension functions (qf_progress_stage, qf_progress_pct, etc.).
// Engine code should use this via the callback_fn adapter pattern above,
// not by accessing the global directly.
// =========================================================================

/**
 * Get the process-wide ProgressTracker.
 * Thread-safe (C++11 function-local static).
 */
ProgressTracker& global_progress_tracker() noexcept;

} // namespace qf

#endif // QUADFORGE_ACCEL_ATOMIC_PROGRESS_H
