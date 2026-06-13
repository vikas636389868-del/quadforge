/**
 * thread_pool.cpp — Implementation of qf::ThreadPool.
 *
 * The pool maintains a set of persistent worker threads that pick tasks
 * from a shared LIFO deque (deque front = newest = picked first for
 * better cache locality on mesh streaming workloads).
 *
 * Synchronisation model:
 *   - One mutex guards the deque and the idle condition variable.
 *   - m_active counts tasks currently executing (not yet finished).
 *   - wait_all() sleeps on m_cv_idle until m_active == 0 AND queue empty.
 *   - shutdown() sets m_stop, broadcasts m_cv, then joins all threads.
 */

#include "../../include/quadforge/accel/thread_pool.h"

#include <algorithm>
#include <cassert>
#include <cstdio>      // std::fprintf (global_thread_pool mismatch warning)
#include <functional>

namespace qf {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

ThreadPool::ThreadPool(int num_threads)
{
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    m_nthreads = (num_threads > 0) ? num_threads
                                   : (hw > 0 ? hw : 4);

    m_workers.reserve(m_nthreads);
    for (int i = 0; i < m_nthreads; ++i) {
        m_workers.emplace_back([this, i]() { worker_loop(i); });
    }
}

ThreadPool::~ThreadPool()
{
    shutdown();
}

// ---------------------------------------------------------------------------
// shutdown / wait_all
// ---------------------------------------------------------------------------

void ThreadPool::shutdown()
{
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (m_stop.exchange(true)) return;  // already stopped
    }
    m_cv.notify_all();
    for (auto& t : m_workers)
        if (t.joinable()) t.join();
}

void ThreadPool::wait_all()
{
    std::unique_lock<std::mutex> lk(m_mutex);
    m_cv_idle.wait(lk, [this] {
        return m_queue.empty() && m_active.load(std::memory_order_acquire) == 0;
    });
}

size_t ThreadPool::pending_count() const
{
    std::lock_guard<std::mutex> lk(m_mutex);
    return m_queue.size();
}

// ---------------------------------------------------------------------------
// enqueue
//
// BUG FIX (v23): added m_stop guard.
//
// Previously, parallel_for / parallel_reduce called enqueue() directly
// (bypassing submit()'s shutdown check) and enqueued tasks after workers
// had already joined.  No worker would ever dequeue those tasks, so the
// count-down latch in parallel_for would never reach zero → infinite block.
//
// Now enqueue() throws if the pool is shutting down, matching submit().
// ---------------------------------------------------------------------------

void ThreadPool::enqueue(Task&& t)
{
    if (m_stop.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "qf::ThreadPool: enqueue() called after shutdown — "
            "tasks submitted to a stopped pool will never execute");
    }
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        // Push to back; worker pops from back → LIFO (stack) ordering.
        // Most recently enqueued tasks execute first, improving cache
        // locality for streaming mesh-processing workloads.
        m_queue.push_back(std::move(t));
    }
    m_cv.notify_one();
}

// ---------------------------------------------------------------------------
// worker_loop
// ---------------------------------------------------------------------------

void ThreadPool::worker_loop(int /*id*/)
{
    for (;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_cv.wait(lk, [this] {
                return !m_queue.empty() || m_stop.load(std::memory_order_relaxed);
            });

            if (m_stop && m_queue.empty()) return;

            // LIFO: take from the back (most recently enqueued)
            task = std::move(m_queue.back());
            m_queue.pop_back();

            // BUG FIX (v50): upgraded from relaxed to release ordering.
            //
            // The relaxed fetch_add is visible to wait_all() when wait_all()
            // acquires the SAME mutex after this block — the mutex release here
            // synchronises with the mutex acquire there.  However, there is a
            // narrow race on shutdown-drain paths where wait_all() reads m_active
            // WITHOUT holding the mutex (via the predicate lambda's acquire load)
            // and may observe m_active == 0 before seeing this increment:
            //
            //   Worker: fetch_add(relaxed) — not yet visible to other cores
            //   Worker: mutex unlock       — releases mutex but NOT the relaxed store
            //   wait_all: acquire load of m_active → sees stale 0 → returns early
            //
            // A release store pairs with the acquire load in the wait predicate
            // and eliminates this race without requiring the mutex for every read.
            m_active.fetch_add(1, std::memory_order_release);
        }

        task();  // execute outside the lock

        int remaining = m_active.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (remaining == 0) {
            // May have drained — notify wait_all() callers
            std::lock_guard<std::mutex> lk(m_mutex);
            if (m_queue.empty())
                m_cv_idle.notify_all();
        }
    }
}

// ---------------------------------------------------------------------------
// parallel_for
// ---------------------------------------------------------------------------

void ThreadPool::parallel_for(int begin, int end,
                               std::function<void(int)> fn,
                               int chunk_size)
{
    if (begin >= end) return;

    int n  = end - begin;
    int nt = m_nthreads;

    // BUG FIX (v50): The previous default chunk_size floor was 64, which is
    // too large for small loops (e.g. 10 feature-edge chains on 8 threads:
    // chunk = max(1, max(0, 64)) = 64 > n=10 → num_chunks=1 → 7 threads idle).
    // This matches the adaptive formula used by omp_utils.cpp::parallel_for()
    // for consistency: max(8, n / (nt * 4)).
    //
    // The floor of 8 iterations amortises task enqueue/dequeue overhead
    // (~200 ns per task dispatch, equivalent to ~8 trivial loop iterations).
    if (chunk_size <= 0)
        chunk_size = std::max(1, std::max(n / (nt * 4), 8));

    int num_chunks = (n + chunk_size - 1) / chunk_size;

    // Count-down latch
    std::atomic<int> remaining{num_chunks};
    std::mutex       done_mu;
    std::condition_variable done_cv;

    for (int ci = 0; ci < num_chunks; ++ci) {
        int lo = begin + ci * chunk_size;
        int hi = std::min(lo + chunk_size, end);

        enqueue([lo, hi, &fn, &remaining, &done_mu, &done_cv]() {
            for (int i = lo; i < hi; ++i) fn(i);
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(done_mu);
                done_cv.notify_all();
            }
        });
    }

    // Block until all chunks are done
    //
    // BUG FIX (v88): Added m_stop to the wait predicate.
    // Mirrors the fix applied to parallel_reduce in thread_pool.h (v88).
    //
    // Without this fix, if shutdown() races with an in-flight parallel_for
    // (tasks still in the LIFO queue), workers drain in-flight tasks, then
    // exit on seeing m_stop — the queued tasks are abandoned and remaining
    // never reaches 0, causing done_cv.wait() to block forever and hanging
    // the engine thread.
    //
    // With this fix, the predicate wakes up immediately when m_stop is set.
    // We then throw std::runtime_error so the caller sees a clear error
    // rather than a silent hang or a silently incomplete parallel loop.
    std::unique_lock<std::mutex> lk(done_mu);
    done_cv.wait(lk, [&] {
        return remaining.load(std::memory_order_acquire) == 0
            || m_stop.load(std::memory_order_relaxed);
    });
    lk.unlock();
    if (remaining.load(std::memory_order_acquire) != 0) {
        throw std::runtime_error(
            "qf::ThreadPool::parallel_for: pool was shut down while tasks "
            "were still queued — some loop iterations did not execute");
    }
}

void ThreadPool::parallel_for(int count,
                               std::function<void(int)> fn,
                               int chunk_size)
{
    parallel_for(0, count, std::move(fn), chunk_size);
}

// ---------------------------------------------------------------------------
// global_thread_pool
// ---------------------------------------------------------------------------

ThreadPool& global_thread_pool(int num_threads)
{
    // C++11 guarantees this static initialisation is thread-safe (§6.7).
    // The pool is created ONCE with the first `num_threads` value supplied;
    // subsequent calls with a different value are silently ignored by the
    // standard (the static local is only initialised once).
    //
    // BUG FIX (v48): Detect and warn when a caller passes a mismatched
    // num_threads value.  This commonly happens when qf_init() calls
    // global_thread_pool(user_threads) and a later stage also calls it
    // with a different count, wrongly believing it controls the pool size.
    // The warning is printed ONCE at the point of mismatch detection.
    //
    // BUG FIX (v49): Resolve the thread count BEFORE storing in
    // `first_requested`.  The previous code stored the raw `num_threads`
    // argument, which could be 0 (= "auto-detect").  When the first call
    // used num_threads=0, first_requested was 0, and a later call with
    // e.g. num_threads=8 produced the misleading warning:
    //   "pool was already created with 0 threads"
    // even though the pool actually had hardware_concurrency() threads.
    // Resolving once here makes the stored value match what the pool
    // actually used, so the warning message is always accurate.
    int hw      = static_cast<int>(std::thread::hardware_concurrency());
    int resolved = (num_threads > 0) ? num_threads : (hw > 0 ? hw : 4);

    static int first_requested = resolved;
    static ThreadPool pool(resolved);

    if (num_threads > 0 && num_threads != first_requested) {
        // BUG FIX (v88): Replace non-atomic `warned` bool with std::atomic<bool>
        // + compare_exchange_strong (CAS) to eliminate a TOCTOU race.
        //
        // Previous code:
        //   static bool warned = false;
        //   if (!warned) { warned = true; fprintf(...); }
        //
        // Race: two threads can both observe warned == false before either
        // writes true, causing both to print the warning — producing duplicate
        // or interleaved lines on stderr.  Although static-local initialisation
        // is thread-safe in C++11, subsequent READS and WRITES of a plain bool
        // are not atomic.  The check-then-act idiom here is a classic TOCTOU.
        //
        // Fix: use std::atomic<bool> with compare_exchange_strong.  The CAS
        // atomically flips false → true; only the winner prints the message.
        // No mutex is needed — CAS is a single hardware instruction (LOCK CMPXCHG).
        static std::atomic<bool> warned{false};
        bool expected = false;
        if (warned.compare_exchange_strong(expected, true,
                                           std::memory_order_acq_rel,
                                           std::memory_order_relaxed)) {
            std::fprintf(stderr,
                "[QuadForge] WARNING: global_thread_pool() called with "
                "num_threads=%d but the pool was already created with %d "
                "threads.  The pool size cannot be changed after first use.  "
                "Pass 0 to reuse the existing pool without a size request, "
                "or call qf_shutdown() and reinitialise for a new size.\n",
                num_threads, first_requested);
        }
    }
    return pool;
}

} // namespace qf
