#pragma once
/* quadforge/api.h — Public C API
 *
 * This is the ONLY header that bridge.py's ctypes layer interacts with.
 * ABI-stable across compiler versions and Python versions.
 *
 * All data is passed as flat C arrays — no C++ types, no templates,
 * no classes cross the ABI boundary.
 */

#ifndef QUADFORGE_API_H
#define QUADFORGE_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
    #define QF_API extern "C" __declspec(dllexport)
#else
    #define QF_API extern "C" __attribute__((visibility("default")))
#endif



/* ======================================================================
 * Data Structures
 * ====================================================================== */

/**
 * Input mesh passed from Python (via ctypes) to the engine.
 * All arrays are owned by Python and must remain valid for the
 * duration of qf_remesh().
 */
typedef struct {
    float*   positions;       /* [N*3] x0,y0,z0, x1,y1,z1, ... */
    int32_t* faces;           /* [F*max_verts_per_face] face vertex indices */
    int32_t* face_sizes;      /* [F] vertices per face (3 or 4) */
    float*   normals;         /* [N*3] per-vertex normals (NULL = compute) */
    float*   vertex_colors;   /* [N*3] RGB density map (NULL = disabled) */
    int32_t* material_ids;    /* [F] per-face material index (NULL = all 0) */
    float*   uv_coords;       /* [UV_count*2] UV coordinates (NULL = none) */
    int32_t* uv_indices;      /* [F*max_verts] UV indices (NULL = none) */
    int32_t  num_vertices;
    int32_t  num_faces;
    int32_t  num_uv_coords;
} QFInputMesh;

/**
 * Remeshing parameters. Mirrors QFParams in bridge.py exactly.
 */
typedef struct {
    /* Target & Sizing */
    int32_t target_quad_count;      /* desired quad count (default: 5000) */
    float   curvature_adaptivity;   /* 0.0 = uniform, 1.0 = max adaptation */
    int32_t exact_quad_count;       /* 1 = binary search exact count */

    /* Feature Detection */
    int32_t auto_detect_hard_edges;
    float   hard_edge_angle_deg;    /* dihedral threshold (default: 30°) */
    int32_t use_normals;
    int32_t use_materials;
    int32_t use_vertex_colors;
    int32_t use_uv_seams;

    /* Symmetry */
    int32_t symmetry_x;
    int32_t symmetry_y;
    int32_t symmetry_z;

    /* Post-processing */
    int32_t smooth_iterations;      /* Taubin passes (default: 10) */
    float   smooth_strength;        /* λ (default: 0.5) */
    float   feature_snap_distance;

    /* Performance */
    int32_t num_threads;    /* 0 = auto */
    int32_t use_gpu;

    /* Algorithm selection */
    int32_t field_solver;       /* 0=EigenSmooth, 1=Knöppel2013, 2=CurvatureOnly */
    int32_t param_method;       /* 0=MIQ, 1=IGM, 2=PoissonSimple */
    int32_t extraction_method;  /* 0=IsoLine, 1=MotorcycleGraph, 2=DualContour */

    /* Preset (0=Custom, 1=Organic, 2=HardSurface, 3=Sculpt, 4=Architecture, 5=Fast) */
    int32_t preset;
} QFParams;

/**
 * Output mesh.  All arrays are heap-allocated by the engine.
 * MUST be freed with qf_free_result().
 */
typedef struct {
    float*   positions;      /* [N*3] output vertex positions */
    int32_t* faces;          /* [F*4]  quad face indices */
    int32_t* tri_faces;      /* [T*3]  leftover triangles (NULL if none) */
    float*   normals;        /* [N*3]  per-vertex normals */
    int32_t* material_ids;   /* [F]    transferred material IDs */
    int32_t  num_vertices;
    int32_t  num_quad_faces;
    int32_t  num_tri_faces;
    float    quad_percentage;
    float    avg_valence;
    float    elapsed_seconds;
} QFResult;

/**
 * Progress callback, called from the engine thread.
 *
 * stage:    0=Preprocess … 5=Output
 * progress: 0.0 → 1.0
 * Return 0 to continue, non-zero to abort.
 *
 * WARNING: Do NOT call Blender Python API from this callback — it is
 * invoked on a background thread.  Write to atomic variables only.
 */
typedef int (*QFProgressCallback)(
    int         stage,
    float       progress,
    const char* stage_name,
    void*       user_data
);


/* ======================================================================
 * Engine Lifecycle
 * ====================================================================== */

/** Initialize the engine (call once at add-on load). Returns 0 on success. */
QF_API int         qf_init(void);

/** Shutdown and free all resources (call once at add-on unload). */
QF_API void        qf_shutdown(void);

/** Returns "1.0.0" etc. */
QF_API const char* qf_version(void);

/** Returns the last error message (thread-local). */
QF_API const char* qf_last_error(void);

/** Returns 1 if a GPU solver is available on this machine. */
QF_API int         qf_has_gpu(void);

/** Returns the number of CPU threads detected. */
QF_API int         qf_cpu_thread_count(void);


/* ======================================================================
 * Main Entry Point
 * ====================================================================== */

/**
 * Run the full remeshing pipeline.
 *
 * Thread safety: can be called from a background thread.
 * The progress callback is called from the same thread as qf_remesh().
 *
 * Returns NULL on failure; call qf_last_error() for the reason.
 * On success, the caller MUST call qf_free_result(result).
 */
QF_API QFResult* qf_remesh(
    const QFInputMesh*    input,
    const QFParams*       params,
    QFProgressCallback    callback,
    void*                 user_data
);

/** Free the QFResult returned by qf_remesh(). */
QF_API void qf_free_result(QFResult* result);


/* ======================================================================
 * Parameter Helpers
 * ====================================================================== */

QF_API QFParams qf_default_params(void);
QF_API QFParams qf_preset_organic(void);
QF_API QFParams qf_preset_hard_surface(void);
QF_API QFParams qf_preset_sculpt(void);
QF_API QFParams qf_preset_architecture(void);
QF_API QFParams qf_preset_fast(void);


/* ======================================================================
 * Debug / Visualization API  (INTEGRATION FIX v82)
 *
 * These functions expose field_debug.cpp's exports through the C API so
 * bridge.py can call them from the "Show Field Debug" panel button.
 *
 * None of these are on the critical remesh path.  They require a prior
 * successful qf_remesh() call to have cached state (or they can operate
 * on raw mesh + pre-computed field data passed directly).
 * ====================================================================== */

/**
 * Cross-field debug output structure.
 * All arrays are heap-allocated by qf_debug_field(); free with qf_free_debug().
 */
typedef struct {
    /* Per-face direction vectors: [nf * 6] floats
     * Each face has two perpendicular unit vectors (U and V directions):
     *   dirs[f*6 + 0..2] = U direction (world-space)
     *   dirs[f*6 + 3..5] = V direction (world-space)
     */
    float*   dirs;
    int32_t  num_faces;

    /* Debug stick-mesh: [nm * 3] vertex positions and [ne * 2] edge indices */
    float*   mesh_verts;
    int32_t* mesh_edges;
    int32_t  num_mesh_verts;
    int32_t  num_mesh_edges;

    /* Singularity points: [ns * 3] positions and [ns] index values (+1/-1) */
    float*   sing_positions;
    float*   sing_indices;
    int32_t  num_singularities;

    /* Per-face field quality heat-map: [nf] floats in [0, 1] */
    float*   quality_map;

    /* Null-terminated summary string (human-readable statistics) */
    char*    summary;
} QFDebugField;

/**
 * Compute cross-field debug visualization data.
 *
 * Operates on the last mesh processed by qf_remesh() on this thread.
 * Returns NULL if no remesh has been run or if an error occurs.
 * @param arm_scale  Scale factor for debug stick arms (0 = auto).
 */
QF_API QFDebugField* qf_debug_field(float arm_scale);

/** Free a QFDebugField returned by qf_debug_field(). */
QF_API void qf_free_debug(QFDebugField* dbg);



#endif /* QUADFORGE_API_H */
