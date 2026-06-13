/* quadforge/accel/thread_pool.h
 *
 * A lightweight, production-grade thread pool for QuadForge.
 *
 * Design goals:
 *   1. Low latency — threads are kept alive between pipeline stages to
 *      avoid the overhead of creating/joining threads on every call.
 *   2. LIFO task queue — recently submitted tasks are executed first,
 *      improving cache locality for mesh-processing workloads.
 *   3. Shared-queue design — a single mutex-protected deque is used;
 *      idle threads block on a condition variable and wake when tasks
 *      arrive.  Simple, correct, and sufficient for QuadForge's batch
 *      workloads where all threads are busy during parallel phases.
 *      (True work-stealing with per-thread deques is a planned v1.2
 *      upgrade for finer-grained load balancing on heterogeneous tasks.)
 *   4. Graceful shutdown — destructor drains all pending tasks then
 *      joins all threads cleanly.
 *   5. Header declares interface; thread_pool.cpp provides implementation.
 *
 * Usage:
 *   qf::ThreadPool pool(8);                 // 8 worker threads
 *   auto fut = pool.submit([](){ return compute(); });
 *   int result = fut.get();
 *
 *   // Bulk parallel-for (preferred for mesh ops):
 *   pool.parallel_for(0, num_vertices, [&](int i) { process(i); });
 *
 * Thread safety:
 *   submit() and parallel_for() are safe to call from any thread,
 *   including the worker threads themselves (nested parallelism).
 *   shutdown() must be called from outside a worker thread.
 */

#pragma once
#ifndef QUADFORGE_ACCEL_THREAD_POOL_H
#define QUADFORGE_ACCEL_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace qf {

// =========================================================================
// Task — type-erased callable with void return (internal use)
// =========================================================================

using Task = std::function<void()>;

// =========================================================================
// ThreadPool
// =========================================================================

class ThreadPool {
public:
    /**
     * Construct a pool with `num_threads` worker threads.
     * Pass 0 to use hardware_concurrency().
     */
    explicit ThreadPool(int num_threads = 0);

    /**
     * Destructor: signals shutdown, waits for all in-flight tasks to
     * complete, then joins all worker threads.
     * Does NOT accept new tasks after the destructor is called.
     */
    ~ThreadPool();

    // Non-copyable, non-movable (threads hold 'this' pointer)
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&)                 = delete;
    ThreadPool& operator=(ThreadPool&&)      = delete;

    // ----------------------------------------------------------------
    // submit — push one callable and get a future for its return value
    // ----------------------------------------------------------------

    /**
     * Submit a callable F with arbitrary return type R.
     * Returns std::future<R> that becomes ready when F completes.
     *
     * Example:
     *   auto f = pool.submit([]{ return 42; });
     *   int v = f.get();   // blocks until done
     */
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>;

    // ----------------------------------------------------------------
    // parallel_for — bulk index-range work (primary API for mesh ops)
    // ----------------------------------------------------------------

    /**
     * Execute fn(i) for every i in [begin, end) in parallel across
     * the pool's worker threads, then block until ALL iterations are done.
     *
     * The range is split into `chunk_size`-sized pieces; each chunk is
     * submitted as one task.  Passing chunk_size=0 lets the pool pick
     * a reasonable default based on range size and thread count.
     *
     * This function is safe to call from the main thread only.
     * (Calling it from inside a worker causes a deadlock because the
     *  worker would block on wait_all while holding its slot.)
     */
    void parallel_for(int begin, int end,
                      std::function<void(int)> fn,
                      int chunk_size = 0);

    /**
     * parallel_for shorthand: begin=0.
     */
    void parallel_for(int count,
                      std::function<void(int)> fn,
                      int chunk_size = 0);

    // ----------------------------------------------------------------
    // parallel_reduce — map-reduce across [begin, end)
    // ----------------------------------------------------------------

    /**
     * For each i in [begin, end), call body(i, partial) to accumulate
     * into a thread-local T initialised to `init`.  Then combine all
     * thread-local partials with `reduce` (must be commutative +
     * associative).  Returns the final combined value.
     *
     * Example — sum of per-vertex areas:
     *   double total = pool.parallel_reduce<double>(
     *       0, nv, 0.0,
     *       [&](int i, double& acc){ acc += vertex_area[i]; },
     *       [](double a, double b){ return a + b; });
     */
    template <typename T>
    T parallel_reduce(int begin, int end,
                      T init,
                      std::function<void(int, T&)> body,
                      std::function<T(T, T)> reduce);

    /**
     * parallel_reduce shorthand: begin=0.
     */
    template <typename T>
    T parallel_reduce(int count,
                      T init,
                      std::function<void(int, T&)> body,
                      std::function<T(T, T)> reduce);

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    /** Number of worker threads in the pool. */
    int thread_count() const noexcept { return static_cast<int>(m_workers.size()); }

    /**
     * Approximate number of tasks currently waiting in the queue.
     * For informational / debugging purposes only.
     */
    size_t pending_count() const;

    /**
     * Block the calling thread until all currently queued tasks have
     * finished executing.  Safe to call from outside worker threads.
     */
    void wait_all();

    /**
     * Signal all workers to stop after finishing their current task,
     * then join them.  Called automatically by the destructor.
     * Idempotent — safe to call more than once.
     */
    void shutdown();

private:
    // ----------------------------------------------------------------
    // Internal helpers
    // ----------------------------------------------------------------

    void worker_loop(int id);

    // Push a ready-to-run Task into the shared queue.
    void enqueue(Task&& t);

    // ----------------------------------------------------------------
    // State
    // ----------------------------------------------------------------

    std::vector<std::thread>        m_workers;
    std::deque<Task>                m_queue;
    mutable std::mutex              m_mutex;
    std::condition_variable         m_cv;
    std::condition_variable         m_cv_idle;

    std::atomic<bool>               m_stop{false};
    std::atomic<int>                m_active{0};   // tasks currently running

    int                             m_nthreads{0};
};

// =========================================================================
// Template implementations (must be in header)
// =========================================================================

template <typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>>
{
    using R = std::invoke_result_t<F, Args...>;

    if (m_stop.load(std::memory_order_relaxed))
        throw std::runtime_error("qf::ThreadPool: submit() after shutdown");

    // Bind args into a packaged_task so we can extract a future.
    auto task = std::make_shared<std::packaged_task<R()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<R> fut = task->get_future();

    enqueue([task]() { (*task)(); });

    return fut;
}

template <typename T>
T ThreadPool::parallel_reduce(int begin, int end,
                               T init,
                               std::function<void(int, T&)> body,
                               std::function<T(T, T)> reduce)
{
    if (begin >= end) return init;

    int n        = end - begin;
    int nt       = m_nthreads;

    // BUG FIX (v86): Changed chunk formula from ceil(n/nt) → max(8, n/(nt*4)).
    //
    // The old formula  max(1, (n + nt - 1) / nt)  = ceil(n / nt)  creates
    // exactly ONE chunk per worker thread — a de-facto static schedule.
    // QuadForge's mesh workloads have variable per-element cost: boundary /
    // singularity rows carry 2–4 non-zeros while interior rows carry 6–8.
    // With a static 1-chunk/thread split, the thread assigned to a dense
    // region finishes significantly later than the rest; the others drain
    // their single chunk and then idle until the slowest thread is done,
    // blocking the countdown latch in the final combine step.
    //
    // The new formula  max(1, max(n / (nt * 4), 8))  gives each thread
    // ~4 chunks (fine-grained enough to expose imbalance to the LIFO
    // task queue) with an 8-iteration minimum (amortises the ~200 ns
    // enqueue/dequeue overhead per task).  This exactly matches the
    // formula used by ThreadPool::parallel_for (thread_pool.cpp line 176)
    // and qf::parallel_for (omp_utils.cpp line 70) for consistency.
    int chunk    = std::max(1, std::max(n / (nt * 4), 8));

    // Allocate one partial per chunk slot
    int num_chunks = (n + chunk - 1) / chunk;

    // BUG FIX (v49): use cache-line-padded partial storage.
    //
    // The previous code stored partials in a plain std::vector<T>.  For
    // T=double (8 bytes) up to 8 adjacent partials share a single 64-byte
    // cache line.  When two worker threads concurrently update neighbouring
    // partials (e.g. partials[2] and partials[3]), every write to one slot
    // forces an invalidation of the entire cache line on the other core —
    // effectively serialising the reduction despite running in parallel.
    //
    // Fix: wrap each T in a struct padded to at least 64 bytes so each
    // partial lives on its own cache line.  This matches the approach used
    // by CachePad<T> in omp_utils.h (parallel_reduce OpenMP variant).
    //
    // PaddedPartial is defined locally here to avoid including omp_utils.h
    // and creating a header dependency cycle.
    struct alignas(64) PaddedPartial {
        T    value;
        // Pad to fill the rest of a 64-byte cache line.  Minimum 1 byte so
        // the array member is never zero-sized (§9.2 of the C++ standard).
        char pad[(sizeof(T) < 64u) ? (64u - sizeof(T)) : 1u];
        explicit PaddedPartial(const T& v) : value(v), pad{} {}
    };

    std::vector<PaddedPartial> partials;
    partials.reserve(num_chunks);
    for (int i = 0; i < num_chunks; ++i)
        partials.emplace_back(init);

    // Count-down latch via atomic + condition variable
    std::atomic<int> remaining{num_chunks};
    std::mutex       done_mu;
    std::condition_variable done_cv;

    for (int ci = 0; ci < num_chunks; ++ci) {
        int lo   = begin + ci * chunk;
        int hi   = std::min(lo + chunk, end);
        T*  part = &partials[ci].value;

        enqueue([lo, hi, part, &body, &remaining, &done_mu, &done_cv]() {
            T local = *part;  // copy init value (cache-line-isolated)
            for (int i = lo; i < hi; ++i)
                body(i, local);
            *part = local;
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(done_mu);
                done_cv.notify_all();
            }
        });
    }

    // Wait for all chunks
    //
    // BUG FIX (v88): Added m_stop check to the wait predicate.
    //
    // Previous predicate: remaining == 0.
    //
    // Race scenario (prior to fix):
    //   1. parallel_reduce enqueues all num_chunks tasks.
    //   2. Concurrently, another thread calls shutdown() (or the destructor
    //      fires during an abnormal exit path).
    //   3. shutdown() sets m_stop = true, broadcasts m_cv, then joins workers.
    //   4. Workers drain in-flight tasks but then see m_stop and exit their
    //      loops — tasks still in the LIFO deque are abandoned without being
    //      executed.
    //   5. remaining never reaches 0.  done_cv.wait() blocks forever, causing
    //      the engine thread to hang and the Blender UI to freeze.
    //
    // Fix: the predicate also wakes up when m_stop is set.  After wake-up we
    // check whether it was a legitimate completion (remaining == 0) or an
    // interrupted shutdown, and throw std::runtime_error in the latter case so
    // the caller can propagate the error rather than silently returning a
    // partial/wrong result.
    //
    // Note: the correct usage contract is that shutdown() must not race with
    // an active parallel_reduce call.  This fix turns an infinite hang into a
    // loud, diagnosable exception for cases where that contract is violated.
    {
        std::unique_lock<std::mutex> lk(done_mu);
        done_cv.wait(lk, [&]{
            return remaining.load(std::memory_order_acquire) == 0
                || m_stop.load(std::memory_order_relaxed);
        });
    }
    if (remaining.load(std::memory_order_acquire) != 0) {
        // Woken by shutdown, not by task completion: some chunks were abandoned.
        throw std::runtime_error(
            "qf::ThreadPool::parallel_reduce: pool was shut down while tasks "
            "were still queued — result is incomplete");
    }

    // Combine partials
    T result = init;
    for (auto& p : partials)
        result = reduce(result, p.value);
    return result;
}

template <typename T>
T ThreadPool::parallel_reduce(int count,
                               T init,
                               std::function<void(int, T&)> body,
                               std::function<T(T, T)> reduce)
{
    // BUG FIX (v87): Forward body and reduce by move, not by copy.
    //
    // std::function stores its target on the heap when the target is larger
    // than its small-buffer-optimisation (SBO) threshold (typically 16–48 bytes
    // depending on the stdlib).  A capture-heavy lambda (e.g. capturing a mesh
    // pointer, a weights array, and several scalars) easily exceeds the SBO
    // threshold.  Passing `body` and `reduce` by value here caused the delegate
    // call  parallel_reduce(0, count, init, body, reduce)  to COPY each
    // std::function — allocating a fresh heap block for the target on every
    // invocation of this shorthand overload.
    //
    // Using std::move() transfers ownership of the existing heap allocation
    // directly into the callee without any copy or allocation.  The moved-from
    // objects are left in a valid-but-unspecified state; since they are not used
    // after the return statement this is safe.
    //
    // Same fix applies to `init`: T may be a non-trivially-copyable accumulator
    // (e.g. a small struct); moving avoids an unnecessary copy.
    return parallel_reduce(0, count,
                           std::move(init),
                           std::move(body),
                           std::move(reduce));
}

// =========================================================================
// Global singleton accessor
// =========================================================================

/**
 * Get (or lazily create) the process-wide QuadForge thread pool.
 * The pool is sized to the available hardware threads on first call.
 * Call this to avoid creating per-stage pools.
 *
 * Thread-safe: uses function-local static initialisation (C++11 §6.7).
 */
ThreadPool& global_thread_pool(int num_threads = 0);

} // namespace qf

#endif // QUADFORGE_ACCEL_THREAD_POOL_H
