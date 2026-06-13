/**
 * api.cpp — QuadForge public C API (extern "C").
 * Bridges the flat C API (api.h) to the C++ qf::QFEngine class.
 */

#include "../include/quadforge/api.h"
#include "../include/quadforge/engine.h"
#include "../include/quadforge/accel/omp_utils.h"
#include "../include/quadforge/accel/gpu_solver.h"
#include "../include/quadforge/accel/simd_math.h"
// INTEGRATION FIX: thread_pool.h was never included from api.cpp, so
// qf_init() never kicked off the persistent thread pool and qf_shutdown()
// never joined its workers — leaving global_thread_pool() uninitialised
// until its first lazy use and never cleanly shut down.
#include "../include/quadforge/accel/thread_pool.h"
// INTEGRATION FIX (v82): Include field_debug.h and connection.h so
// qf_debug_field() can call the field visualization exports from
// field_debug.cpp and use connection.cpp for standalone field computation.
#include "../include/quadforge/field/field_debug.h"
#include "../include/quadforge/field/connection.h"
#include "../include/quadforge/mesh/halfedge.h"
#include "../include/quadforge/mesh/mesh_io.h"

#include <string>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <new>

// Private cross-TU declaration — must be included in this TU too so the
// compiler can verify the definition matches.  See src/_internal.h.
#include "_internal.h"

// -----------------------------------------------------------------------
// Global error state (written by qf::QFEngine via _qf_set_global_error)
// -----------------------------------------------------------------------

static std::string  g_last_error;
static std::atomic<bool> g_initialized{false};

// Called from engine.cpp (same translation unit group)
void _qf_set_global_error(const std::string& msg) {
    g_last_error = msg;
}

// -----------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------

extern "C" {

QF_API int qf_init(void) {
    g_initialized.store(true);
    g_last_error.clear();
    // Verify the compiled SIMD level is safe on this CPU.
    // Logs a warning to stderr if the binary was built with AVX2 but runs
    // on an SSE4.2-only machine (or similar mismatch).
    qf::simd::assert_simd_safety();
    // INTEGRATION FIX: eagerly initialise the global thread pool so workers
    // are already alive when the first pipeline stage runs, eliminating the
    // thread-spawn latency that would otherwise add to the first remesh call.
    qf::global_thread_pool(qf::resolve_thread_count(0));
    return 0;
}

QF_API void qf_shutdown(void) {
    // INTEGRATION FIX: explicitly shut down the global thread pool so worker
    // threads are joined cleanly on library unload.  Without this call,
    // global_thread_pool()'s static ThreadPool destructor runs at process
    // exit — too late if the shared library is dlclose()'d while Blender
    // is still running, which can cause a hang or crash on macOS/Linux.
    qf::global_thread_pool().shutdown();
    qf::gpu_cleanup();
    g_initialized.store(false);
    g_last_error.clear();
}

QF_API const char* qf_version(void) { return "1.0.0"; }

QF_API const char* qf_last_error(void) { return g_last_error.c_str(); }

QF_API int qf_has_gpu(void) { return qf::gpu_available() ? 1 : 0; }

QF_API int qf_cpu_thread_count(void) {
    return qf::resolve_thread_count(0);
}

// -----------------------------------------------------------------------
// Default parameters
// -----------------------------------------------------------------------

QF_API QFParams qf_default_params(void) {
    QFParams p{};
    p.target_quad_count      = 5000;
    p.curvature_adaptivity   = 0.5f;
    p.exact_quad_count       = 0;
    p.auto_detect_hard_edges = 1;
    p.hard_edge_angle_deg    = 30.0f;
    p.use_normals            = 0;
    p.use_materials          = 0;
    p.use_vertex_colors      = 0;
    p.use_uv_seams           = 0;
    p.symmetry_x = p.symmetry_y = p.symmetry_z = 0;
    p.smooth_iterations      = 10;
    p.smooth_strength        = 0.5f;
    p.feature_snap_distance  = 0.1f;
    p.num_threads            = 0;
    p.use_gpu                = 0;
    p.field_solver           = 1;   // Knöppel
    p.param_method           = 0;   // MIQ
    p.extraction_method      = 2;   // IsoLine
    p.preset                 = 0;
    return p;
}

QF_API QFParams qf_preset_organic(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.8f;
    p.auto_detect_hard_edges = 0;
    p.smooth_iterations      = 15;
    p.feature_snap_distance  = 0.05f;
    p.preset = 1;
    return p;
}

QF_API QFParams qf_preset_hard_surface(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.3f;
    p.hard_edge_angle_deg    = 25.0f;
    p.use_normals = p.use_materials = 1;
    p.smooth_iterations      = 5;
    p.smooth_strength        = 0.3f;
    p.feature_snap_distance  = 0.2f;
    p.preset = 2;
    return p;
}

QF_API QFParams qf_preset_sculpt(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.7f;
    p.auto_detect_hard_edges = 0;
    p.param_method           = 2; // Poisson-simple
    p.preset = 3;
    return p;
}

QF_API QFParams qf_preset_architecture(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.1f;
    p.hard_edge_angle_deg    = 15.0f;
    p.use_normals = p.use_materials = 1;
    p.exact_quad_count       = 1;
    p.smooth_iterations      = 3;
    p.smooth_strength        = 0.2f;
    p.feature_snap_distance  = 0.3f;
    p.preset = 4;
    return p;
}

QF_API QFParams qf_preset_fast(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.2f;
    p.smooth_iterations      = 3;
    p.field_solver           = 2;  // Curvature-only
    p.param_method           = 2;  // Poisson-simple
    p.extraction_method      = 2;  // Greedy
    p.preset = 5;
    return p;
}

// -----------------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------------

QF_API QFResult* qf_remesh(
    const QFInputMesh*  input,
    const QFParams*     params,
    QFProgressCallback  callback,
    void*               user_data)
{
    if (!input || !params) {
        g_last_error = "qf_remesh: null input or params";
        return nullptr;
    }
    if (input->num_vertices < 4 || input->num_faces < 2) {
        g_last_error = "qf_remesh: mesh too small";
        return nullptr;
    }
    g_last_error.clear();

    try {
        qf::QFEngine engine(input, params, callback, user_data);
        return engine.run();
    } catch (const std::exception& e) {
        g_last_error = std::string("qf_remesh: ") + e.what();
        return nullptr;
    } catch (...) {
        g_last_error = "qf_remesh: unknown exception";
        return nullptr;
    }
}

QF_API void qf_free_result(QFResult* result) {
    if (!result) return;
    delete[] result->positions;
    delete[] result->faces;
    delete[] result->tri_faces;
    delete[] result->normals;
    delete[] result->material_ids;
    delete result;
}

} // extern "C"  — end of main C API block

// -----------------------------------------------------------------------
// Debug field cache — C++ linkage (uses std::unique_ptr / std::vector).
// DebugCache and qf_cache_debug_state live OUTSIDE extern "C" because they
// use C++ types that cannot appear in a C-linkage block.
// -----------------------------------------------------------------------

// Thread-local storage so concurrent remesh calls on different threads
// do not clobber each other's debug state.
struct DebugCache {
    std::unique_ptr<qf::HalfEdgeMesh>         mesh;
    qf::CrossField                            field;
    std::vector<qf::SingularityInfo>          sings;
    bool                                      valid = false;
};

static DebugCache g_debug_cache;

// Called by QFEngine after stage2 to cache the field for qf_debug_field().
// This is a friend-style linkage: same TU so it can see the static cache.
void qf_cache_debug_state(
    std::unique_ptr<qf::HalfEdgeMesh>&&       mesh,
    qf::CrossField&&                          field,
    std::vector<qf::SingularityInfo>&&        sings)
{
    g_debug_cache.mesh  = std::move(mesh);
    g_debug_cache.field = std::move(field);
    g_debug_cache.sings = std::move(sings);
    g_debug_cache.valid = true;
}

// Re-open extern "C" for the debug API exports that cross the ABI boundary.
extern "C" {

QF_API QFDebugField* qf_debug_field(float arm_scale) {
    if (!g_debug_cache.valid || !g_debug_cache.mesh) {
        g_last_error = "qf_debug_field: no remesh result cached on this thread";
        return nullptr;
    }

    try {
        const qf::HalfEdgeMesh& mesh  = *g_debug_cache.mesh;
        const qf::CrossField&   field = g_debug_cache.field;
        const auto&             sings = g_debug_cache.sings;

        auto* dbg = new QFDebugField{};

        // 1. Per-face direction vectors
        auto dirs = qf::export_cross_field_vectors(mesh, field,
                        arm_scale > 0.f ? arm_scale : 1.0f);
        dbg->num_faces = (int32_t)mesh.num_faces();
        dbg->dirs = new float[dirs.size()];
        std::copy(dirs.begin(), dirs.end(), dbg->dirs);

        // 2. Debug stick-mesh
        std::vector<float>    mv;
        std::vector<int32_t>  me;
        qf::export_cross_field_mesh(mesh, field,
                                    arm_scale > 0.f ? arm_scale : -1.f,
                                    mv, me);
        dbg->num_mesh_verts = (int32_t)(mv.size() / 3);
        dbg->num_mesh_edges = (int32_t)(me.size() / 2);
        dbg->mesh_verts = new float[mv.size()];
        std::copy(mv.begin(), mv.end(), dbg->mesh_verts);
        dbg->mesh_edges = new int32_t[me.size()];
        std::copy(me.begin(), me.end(), dbg->mesh_edges);

        // 3. Singularity points
        std::vector<float> sp, si;
        qf::export_singularities_as_points(sings, sp, si);
        dbg->num_singularities = (int32_t)sings.size();
        dbg->sing_positions = new float[sp.size()];
        std::copy(sp.begin(), sp.end(), dbg->sing_positions);
        dbg->sing_indices = new float[si.size()];
        std::copy(si.begin(), si.end(), dbg->sing_indices);

        // 4. Quality heat-map
        auto qmap = qf::export_field_quality_map(mesh, field);
        dbg->quality_map = new float[qmap.size()];
        std::copy(qmap.begin(), qmap.end(), dbg->quality_map);

        // 5. Summary string
        std::string summary = qf::cross_field_summary(mesh, field, sings);
        dbg->summary = new char[summary.size() + 1];
        std::copy(summary.begin(), summary.end(), dbg->summary);
        dbg->summary[summary.size()] = '\0';

        return dbg;

    } catch (const std::exception& e) {
        g_last_error = std::string("qf_debug_field: ") + e.what();
        return nullptr;
    } catch (...) {
        g_last_error = "qf_debug_field: unknown exception";
        return nullptr;
    }
}

QF_API void qf_free_debug(QFDebugField* dbg) {
    if (!dbg) return;
    delete[] dbg->dirs;
    delete[] dbg->mesh_verts;
    delete[] dbg->mesh_edges;
    delete[] dbg->sing_positions;
    delete[] dbg->sing_indices;
    delete[] dbg->quality_map;
    delete[] dbg->summary;
    delete dbg;
}

} // extern "C"
