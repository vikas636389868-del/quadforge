/**
 * atomic_progress.cpp — Implementation of qf::ProgressTracker support.
 *
 * Provides the definition of global_progress_tracker().
 *
 * All other ProgressTracker methods are inline in the header because they
 * are trivially small (one atomic load/store each) and benefit from
 * inlining at every call site.  This TU only exists to give
 * global_progress_tracker() a single definition point in the shared library.
 *
 * Design note: the global tracker is intentionally a Meyers singleton
 * (function-local static) rather than a file-scope global.  C++11 §6.7
 * guarantees thread-safe initialisation of function-local statics, so the
 * first concurrent call to global_progress_tracker() from any thread is
 * safe without additional locking.
 */

#include "../../include/quadforge/accel/atomic_progress.h"

namespace qf {

ProgressTracker& global_progress_tracker() noexcept {
    static ProgressTracker instance;
    return instance;
}

} // namespace qf
