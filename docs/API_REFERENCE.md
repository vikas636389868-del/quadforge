# QuadForge C API Reference

The C API is the single interface between the Python add-on (`bridge.py`)
and the compiled C++ engine (`libquadforge`). It is defined in
`engine/include/quadforge/api.h` and is intentionally minimal and stable.

---

## Engine Lifecycle

### `int qf_init(void)`
Initialize the engine. Call once at add-on load time.
Returns 0 on success, non-zero on failure.

### `void qf_shutdown(void)`
Free all engine resources. Call once at add-on unload.

### `const char* qf_version(void)`
Returns the engine version string, e.g. `"4.0.0"`.

### `const char* qf_last_error(void)`
Returns the last error message (thread-local). Valid until the next
API call on the same thread.

### `int qf_has_gpu(void)`
Returns 1 if a CUDA GPU solver is available, 0 otherwise.

### `int qf_cpu_thread_count(void)`
Returns the number of logical CPU threads detected.

---

## Main Entry Point

### `QFResult* qf_remesh(const QFInputMesh* input, const QFParams* params, QFProgressCallback callback, void* user_data)`

Run the complete remeshing pipeline. Thread-safe — can be called from
a background thread.

**Parameters:**
- `input` — Input mesh data (owned by caller; must remain valid for the call duration)
- `params` — Remeshing parameters
- `callback` — Progress callback (called on the calling thread; may be NULL)
- `user_data` — Passed verbatim to the callback

**Returns:** `QFResult*` on success, `NULL` on failure.

**Important:** On success, you **must** call `qf_free_result()` on the
returned pointer when done.

### `void qf_free_result(QFResult* result)`
Free all memory allocated by `qf_remesh()`.

---

## Data Structures

### `QFInputMesh`

```c
typedef struct {
    float*   positions;      // [num_vertices * 3]  x0,y0,z0, x1,y1,z1, ...
    int32_t* faces;          // [num_faces * max_verts_per_face]
    int32_t* face_sizes;     // [num_faces]  vertices per face (3 or 4)
    float*   normals;        // [num_vertices * 3]  (NULL = compute internally)
    float*   vertex_colors;  // [num_vertices * 3]  RGB (NULL = disabled)
    int32_t* material_ids;   // [num_faces]         (NULL = all material 0)
    float*   uv_coords;      // [num_uv_coords * 2] (NULL = none)
    int32_t* uv_indices;     // [num_faces * max_verts] (NULL = none)
    int32_t  num_vertices;
    int32_t  num_faces;
    int32_t  num_uv_coords;
} QFInputMesh;
```

All arrays are owned by the caller. Pass NULL for optional fields.

### `QFParams`

```c
typedef struct {
    int32_t target_quad_count;      // default: 5000
    float   curvature_adaptivity;   // 0.0–1.0, default: 0.5
    int32_t exact_quad_count;       // 0 or 1
    int32_t auto_detect_hard_edges; // 0 or 1
    float   hard_edge_angle_deg;    // default: 30.0
    int32_t use_normals;            // 0 or 1
    int32_t use_materials;          // 0 or 1
    int32_t use_vertex_colors;      // 0 or 1
    int32_t use_uv_seams;           // 0 or 1
    int32_t symmetry_x;             // 0 or 1
    int32_t symmetry_y;
    int32_t symmetry_z;
    int32_t smooth_iterations;      // default: 10
    float   smooth_strength;        // default: 0.5
    float   feature_snap_distance;  // world units
    int32_t num_threads;            // 0 = auto
    int32_t use_gpu;
    int32_t field_solver;    // 0=Knöppel2013, 1=EigenSmooth, 2=CurvatureOnly
    int32_t param_method;    // 0=MIQ, 1=IGM, 2=Poisson-simple
    int32_t extraction_method; // 0=IsoLine, 1=MotorcycleGraph, 2=DualContour, 3=GreedyMerge
    int32_t preset;          // 0=Custom,1=Organic,2=HardSurface,3=Sculpt,
                             // 4=Architecture,5=Fast
} QFParams;
```

### `QFResult`

```c
typedef struct {
    float*   positions;      // [num_vertices * 3]
    int32_t* faces;          // [num_quad_faces * 4]  quad indices
    int32_t* tri_faces;      // [num_tri_faces * 3]   leftover tris (may be NULL)
    float*   normals;        // [num_vertices * 3]
    int32_t* material_ids;   // [num_quad_faces]
    int32_t  num_vertices;
    int32_t  num_quad_faces;
    int32_t  num_tri_faces;
    float    quad_percentage;   // 0–100
    float    avg_valence;       // ideal ≈ 4.0
    float    elapsed_seconds;
} QFResult;
```

All arrays are owned by the engine. Do NOT free them manually.
Call `qf_free_result()` to release the entire struct.

### Progress callback

```c
typedef int (*QFProgressCallback)(
    int         stage,       // 0=Preprocess … 5=Output
    float       progress,    // 0.0 – 1.0
    const char* stage_name,  // human-readable name
    void*       user_data    // passed from qf_remesh()
);
```

**Return 0** to continue. Return any non-zero value to **abort** the remesh.

⚠️ This callback is called on the engine's thread. Do **not** call any
Blender Python API from inside it.

---

## Preset Helpers

```c
QFParams qf_default_params(void);
QFParams qf_preset_organic(void);
QFParams qf_preset_hard_surface(void);
QFParams qf_preset_sculpt(void);
QFParams qf_preset_architecture(void);
QFParams qf_preset_fast(void);
```

Each function returns a fully-configured `QFParams` struct with
settings optimised for that use case.

---

## ctypes Usage Example (Python)

```python
import ctypes, numpy as np

lib = ctypes.CDLL("libquadforge.so")
lib.qf_init()
print(lib.qf_version().decode())

# Build input mesh
verts = np.asarray(mesh_verts, dtype=np.float32)
faces = np.asarray(mesh_faces, dtype=np.int32)
sizes = np.full(len(faces), 3, dtype=np.int32)

class QFInputMesh(ctypes.Structure):
    _fields_ = [
        ("positions",    ctypes.POINTER(ctypes.c_float)),
        ("faces",        ctypes.POINTER(ctypes.c_int32)),
        ("face_sizes",   ctypes.POINTER(ctypes.c_int32)),
        # ... (see bridge.py for full definition)
        ("num_vertices", ctypes.c_int32),
        ("num_faces",    ctypes.c_int32),
        ("num_uv_coords",ctypes.c_int32),
    ]

inp = QFInputMesh()
inp.positions  = verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
inp.faces      = faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
inp.face_sizes = sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
inp.num_vertices = len(verts)
inp.num_faces    = len(faces)

params = lib.qf_default_params()
result = lib.qf_remesh(ctypes.byref(inp), ctypes.byref(params), None, None)
if result:
    r = result.contents
    print(f"{r.num_vertices}V {r.num_quad_faces}Q  {r.quad_percentage:.1f}% quads")
    lib.qf_free_result(result)

lib.qf_shutdown()
```

In practice, use `QuadForge/bridge.py` which handles all of this
automatically.
