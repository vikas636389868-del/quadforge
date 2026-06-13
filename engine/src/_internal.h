/**
 * _internal.h — Private cross-TU declarations for the QuadForge engine.
 *
 * This header is intentionally NOT in include/quadforge/ because it is
 * NOT part of the public API.  It is only included by engine.cpp and
 * api.cpp — the two translation units that must share the error-state
 * setter without going through the public C API.
 *
 * BUG FIX v29: Previously engine.cpp used a raw `extern void
 * _qf_set_global_error(...)` forward declaration.  That works at runtime
 * (both TUs are in the same shared library) but is architecturally
 * fragile: if the signature changes in api.cpp the compiler cannot warn
 * about the mismatch because the declaration and definition live in
 * different TUs with no shared header.  A single shared header eliminates
 * that silent failure mode.
 *
 * INTEGRATION FIX v82: Added qf_cache_debug_state() — called by engine.cpp
 * after stage2 to populate the thread-local debug cache in api.cpp so that
 * qf_debug_field() can access the field data without an extra remesh.
 */

#pragma once
#ifndef QUADFORGE_INTERNAL_H
#define QUADFORGE_INTERNAL_H

#include <string>
#include <memory>
#include <vector>

// Forward-declare internal types (full headers pulled in by engine.cpp/api.cpp)
namespace qf {
    class HalfEdgeMesh;
    struct CrossField;
    struct SingularityInfo;
}

/**
 * Write msg into the process-global error string returned by qf_last_error().
 * Defined in api.cpp; called from engine.cpp (and only there).
 * Must NOT be called from multiple threads concurrently.
 */
void _qf_set_global_error(const std::string& msg);

/**
 * Cache the mesh/field/singularities produced by stage2 so that
 * qf_debug_field() can serve field visualization without re-running the
 * pipeline.  State is stored in a thread_local in api.cpp.
 * Defined in api.cpp; called from engine.cpp after stage2.
 */
void qf_cache_debug_state(
    std::unique_ptr<qf::HalfEdgeMesh>&&  mesh,
    qf::CrossField&&                     field,
    std::vector<qf::SingularityInfo>&&   sings);

#endif // QUADFORGE_INTERNAL_H
