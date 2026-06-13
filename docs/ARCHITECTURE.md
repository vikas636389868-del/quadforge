# QuadForge — Architecture Reference

**Version:** 1.0 — March 2026  
**Author:** Reynold  
**License:** GPL v3 (add-on) + MIT (C++ engine)

---

## Overview

QuadForge is a two-tier system:

1. **Python Add-on Layer** (`QuadForge/`) — Blender-facing: UI panels, operators, properties, presets, and the bridge to the engine.
2. **C++17 Engine Library** (`QuadForge/engine/`) — platform-compiled shared library (`libquadforge.so` / `.dylib` / `.dll`) implementing a six-stage quad remeshing pipeline. Also ships a pure-Python fallback (`engine/pipeline.py`) for development and platforms without a compiled binary.

---

## Directory Structure

```
QuadForge/
├── __init__.py           bl_info, register/unregister, F8 reload
├── operators.py          QUADFORGE_OT_remesh (modal), reset, paint density
├── panels.py             All UI sub-panels
├── properties.py         QFSettingsPropertyGroup (every user param)
├── preferences.py        AddonPreferences (threads, debug)
├── bridge.py             ctypes FFI ↔ libquadforge + Python fallback
├── mesh_transfer.py      blender_mesh_to_arrays / arrays_to_blender_mesh
├── presets.py            Built-in presets: Organic, Hard Surface, …
├── callbacks.py          Property update callbacks
├── utils.py              Platform detection, Blender version compat
├── icons/                PNG icons (placeholder size in v4)
│
└── engine/
    ├── pipeline.py        Pure-Python 9-stage master pipeline
    ├── halfedge.py        Half-edge mesh data structure
    ├── field.py           4-RoSy cross-field solver (Python)
    ├── curvature.py       Discrete principal curvature (Rusinkiewicz)
    ├── features.py        Hard-edge / normal / material / UV-seam detection
    ├── extraction.py      Iso-line tracing + quad construction
    ├── motorcycle.py      Motorcycle graph T-junction resolution
    ├── smoothing.py       Taubin λ/μ bilaplacian smoothing
    ├── projection.py      BVH surface projection
    ├── snapping.py        Feature curve snapping
    ├── singularity.py     Singularity detection + relocation
    ├── symmetry.py        Symmetry detection + enforcement
    ├── combing.py         Cross-field combing + seam cuts
    ├── parametrize.py     Poisson UV + MIQ rounding (Python)
    ├── metrics.py         Quality metrics (quad %, valence, Jacobian)
    ├── ...                (see engine/__init__.py for full list)
    │
    ├── include/quadforge/ C++ public headers (api.h is the ABI contract)
    └── src/               C++ implementation files (one per header)
```

---

## The Six Pipeline Stages

### Stage 1 — Preprocessing & Feature Detection

**Input:** Raw Blender mesh (vertices, faces, normals, materials, UVs, vertex colours).

**Operations:**
- Fan-triangulate any polygon faces (quads, N-gons) → flat triangle array.
- Compute per-face normals (cross-product) and area-weighted per-vertex normals.
- Detect feature edges by dihedral angle threshold (and optionally: split normals, material boundaries, UV seams).
- Compute adaptive sizing field: `L(v) = L_base / (1 + α·κ(v)·L_base)` where `α` is `curvature_adaptivity`.
- If vertex colour density is enabled: modulate sizing by `4^(2·red-1)` / `0.25^(2·green-1)`.

**Output:** Triangulated mesh, feature edge set, per-vertex target edge length.

---

### Stage 2 — Cross-Field Computation (Knöppel 2013)

**Goal:** Compute a globally smooth 4-fold rotationally symmetric (4-RoSy) direction field on the surface.

**Operations:**
1. Compute per-face orthonormal local frames `(e1, e2, n)`.
2. Assemble the **connection Laplacian** — a complex-valued sparse matrix `L[fi,fi]` that encodes parallel transport angles between adjacent faces.
3. Solve for the **smallest eigenvector** of `L` (inverse power iteration). This is globally optimal: no smoother field exists for the given mesh.
4. Integrate feature-edge constraints via penalty terms (hard edges force field alignment; boundary edges force perpendicular alignment).
5. Detect **singularities** by computing the holonomy (total field rotation) around each vertex. Holonomy ≠ multiple of π/2 → singularity.

**Math:**  
Field `u_f = exp(4iθ_f)` per face. Smoothness energy:  
`E = Σ_{edges} |u_{f_i} - u_{f_j}·exp(4iφ_{ij})|²`  
Minimised by the smallest eigenvector of the connection Laplacian.

---

### Stage 3 — Global Parametrization

**Goal:** Assign `(U, V)` coordinates to every vertex such that integer iso-lines become quad edges.

**Operations:**
1. **Comb** the 4-RoSy field: assign a consistent orientation across the mesh by BFS. Where combing is inconsistent → introduce seam edges.
2. **Seam cut**: connect all singularities via shortest paths (Dijkstra) to make the surface topologically a disk.
3. **Poisson solve**: minimise `∫|∇U - X|² + |∇V - Y|² dA` (cotangent Laplacian, sparse CG).
4. **MIQ rounding** (optional): snap near-integer seam-crossing period-jump variables to integers for cleaner iso-line alignment.

**Solvers available:**
- `MIQ` (default): Mixed-integer rounding — cleanest edge loops.
- `POISSON`: Poisson-only — faster, looser alignment.

---

### Stage 4 — Quad Extraction

**Goal:** Turn the UV parametrization into actual quad faces.

**Operations:**
1. **Iso-line tracing**: for each triangle, find where integer U and V values cross its edges via linear interpolation.
2. **Quad construction**: intersections of U-lines × V-lines define quad cell corners; each cell becomes a quad face.
3. **T-junction resolution** (Motorcycle Graph): singularities create T-junctions; the motorcycle algorithm traces "riders" from singularities to clean them up.
4. **Topology cleanup**: merge duplicate vertices (within tolerance), remove zero-area faces, collapse short edges, compact vertex array.

---

### Stage 5 — Post-Processing

**Operations:**
1. **Taubin smoothing**: alternating `v += λ·L(v)` (shrink) and `v += μ·L(v)` (expand, μ<0) — volume-preserving. Default: λ=0.5, μ=−0.53, 10 iterations.
2. **Surface projection**: after smoothing moves vertices off-surface, BVH nearest-point query projects each vertex back. Runs 3 interleaved rounds with smoothing.
3. **Feature snapping**: vertices within `feature_snap_distance` of a feature curve are slid onto it.
4. **Symmetry enforcement**: if symmetry is enabled, vertex pairs across the symmetry plane are averaged.
5. **Material transfer**: each output face's material ID is determined by the input triangle nearest to its centroid.

---

### Stage 6 — Output

Pack vertex positions, quad face indices, leftover triangle indices, per-vertex normals, quality metrics, and material IDs into a `QFResult` struct and return to Python.

---

## The C API (ABI Contract)

The bridge between Python ctypes and C++ is defined in `engine/include/quadforge/api.h`. Key points:

- **All data as flat C arrays** — no C++ types cross the ABI boundary.
- **`qf_remesh(input, params, callback, user_data)`** is the single entry point.
- Progress callbacks are invoked from the engine thread — must NOT call Blender Python API.
- `qf_free_result(result)` frees all heap-allocated output arrays.
- `qf_version()`, `qf_has_gpu()`, `qf_cpu_thread_count()` provide introspection.

---

## The Python Fallback Pipeline

When no compiled library is found, `bridge.py` routes to `engine/pipeline.py` — a pure-Python implementation of all six stages using NumPy and SciPy. It is functionally equivalent but 5–10× slower on large meshes. This path is the primary development/debugging path because it requires no build step.

---

## Thread Safety

- **Main operator thread** (`execute()`, `_build_output()`): calls Blender Python API freely.
- **Worker thread** (`_worker_thread()`): calls only `engine/pipeline.py` and numpy — no `bpy` calls.
- Progress callback writes to `_RemeshState` (plain Python atomics, thread-safe via GIL).
- The modal timer reads `_RemeshState` on the main thread and updates the UI.
- Small meshes (< 2000 faces) run synchronously on the main thread to avoid thread overhead.

---

## Adding a New Algorithm Stage

1. Add the Python implementation in `engine/` (e.g., `engine/my_stage.py`).
2. Add corresponding C++ header in `engine/include/quadforge/my_stage/my_stage.h`.
3. Add implementation in `engine/src/my_stage/my_stage.cpp`.
4. Add the module name to `__init__.py`'s reload list and import section.
5. Call it from `pipeline.py` at the appropriate stage.
6. Add a corresponding call in `engine/src/engine.cpp` in the relevant stage method.
7. Write a `tests/test_my_stage.cpp` unit test.

---

## Platform Notes

| Platform | Binary | Compile flags |
|----------|--------|---------------|
| Linux x86_64 | `libquadforge.so` | `-shared -fPIC -O3 -fopenmp -std=c++17` |
| macOS ARM64 | `libquadforge.dylib` | `-dynamiclib -O3 -Xpreprocessor -fopenmp -mmacosx-version-min=13.0` |
| macOS x86_64 | `libquadforge.dylib` | same with `-arch x86_64` |
| Windows x64 | `quadforge.dll` | `/O2 /openmp /std:c++17` (MSVC 2022+) |

All dependencies (Eigen, nanoflann, parallel-hashmap) are header-only and vendored. SuiteSparse CHOLMOD is optional — the built-in CG solver handles meshes up to ~500K faces adequately.

---

## Known Limitations (v1.0)

- **No pre-compiled binaries** in the distribution ZIP — users must compile from source or rely on the Python fallback.
- GPU solver not yet implemented (placeholder interface in `accel/gpu_solver.h`).
- MIQ rounding is simplified (greedy snap, no full integer programme). Quality is good but not optimal for high-genus meshes.
- Iso-line tracing uses bilinear interpolation within triangles; true global iso-line chaining across triangle boundaries is scheduled for v1.1.
