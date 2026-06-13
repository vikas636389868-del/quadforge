"""QuadForge bridge layer.

Two-tier execution strategy:
  1. NATIVE path  — if libquadforge.so / quadforge.dll / libquadforge.dylib is
     present in the add-on's engine/prebuilt/ directory, load it via ctypes
     and call qf_remesh() directly.  This is the production fast path.
  2. PYTHON FALLBACK — if no compiled library is found (development / source
     builds), delegate to the pure-Python engine in engine/pipeline.py.
     Functionally equivalent, ~5-10x slower on large meshes.

The public API of this module is independent of which path is active.
"""
from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

try:
    import bpy
except ImportError:
    bpy = None  # Agar blender nahi mila, toh bpy ko None kar do

import numpy as np


# -----------------------------------------------------------------------
# Shared-library discovery
# -----------------------------------------------------------------------

def _find_native_library() -> Optional[str]:
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    system = platform.system()
    if system == "Windows":
        lib_name, plat_dir = "quadforge.dll", "win64"
    elif system == "Darwin":
        lib_name = "libquadforge.dylib"
        plat_dir = "macos_arm64" if platform.machine() == "arm64" else "macos_x64"
    else:
        lib_name, plat_dir = "libquadforge.so", "linux64"

    for path in [
        os.path.join(addon_dir, "engine", "prebuilt", plat_dir, lib_name),
        os.path.join(addon_dir, "prebuilt", plat_dir, lib_name),
        os.path.join(addon_dir, lib_name),
    ]:
        if os.path.isfile(path):
            return path
    return None


# -----------------------------------------------------------------------
# ctypes struct mirrors of api.h
# -----------------------------------------------------------------------

class _QFInputMeshC(ctypes.Structure):
    _fields_ = [
        ("positions",     ctypes.POINTER(ctypes.c_float)),
        ("faces",         ctypes.POINTER(ctypes.c_int32)),
        ("face_sizes",    ctypes.POINTER(ctypes.c_int32)),
        ("normals",       ctypes.POINTER(ctypes.c_float)),
        ("vertex_colors", ctypes.POINTER(ctypes.c_float)),
        ("material_ids",  ctypes.POINTER(ctypes.c_int32)),
        ("uv_coords",     ctypes.POINTER(ctypes.c_float)),
        ("uv_indices",    ctypes.POINTER(ctypes.c_int32)),
        ("num_vertices",  ctypes.c_int32),
        ("num_faces",     ctypes.c_int32),
        ("num_uv_coords", ctypes.c_int32),
    ]

class _QFParamsC(ctypes.Structure):
    _fields_ = [
        ("target_quad_count",      ctypes.c_int32),
        ("curvature_adaptivity",   ctypes.c_float),
        ("exact_quad_count",       ctypes.c_int32),
        ("auto_detect_hard_edges", ctypes.c_int32),
        ("hard_edge_angle_deg",    ctypes.c_float),
        ("use_normals",            ctypes.c_int32),
        ("use_materials",          ctypes.c_int32),
        ("use_vertex_colors",      ctypes.c_int32),
        ("use_uv_seams",           ctypes.c_int32),
        ("symmetry_x",             ctypes.c_int32),
        ("symmetry_y",             ctypes.c_int32),
        ("symmetry_z",             ctypes.c_int32),
        ("smooth_iterations",      ctypes.c_int32),
        ("smooth_strength",        ctypes.c_float),
        ("feature_snap_distance",  ctypes.c_float),
        ("num_threads",            ctypes.c_int32),
        ("use_gpu",                ctypes.c_int32),
        ("field_solver",           ctypes.c_int32),
        ("param_method",           ctypes.c_int32),
        ("extraction_method",      ctypes.c_int32),
        ("preset",                 ctypes.c_int32),
    ]

class _QFResultC(ctypes.Structure):
    _fields_ = [
        ("positions",       ctypes.POINTER(ctypes.c_float)),
        ("faces",           ctypes.POINTER(ctypes.c_int32)),
        ("tri_faces",       ctypes.POINTER(ctypes.c_int32)),
        ("normals",         ctypes.POINTER(ctypes.c_float)),
        ("material_ids",    ctypes.POINTER(ctypes.c_int32)),
        ("num_vertices",    ctypes.c_int32),
        ("num_quad_faces",  ctypes.c_int32),
        ("num_tri_faces",   ctypes.c_int32),
        ("quad_percentage", ctypes.c_float),
        ("avg_valence",     ctypes.c_float),
        ("elapsed_seconds", ctypes.c_float),
    ]

# INTEGRATION FIX (v82): ctypes mirror of QFDebugField (api.h).
# Required for qf_debug_field() / qf_free_debug() C API calls.
class _QFDebugFieldC(ctypes.Structure):
    _fields_ = [
        ("dirs",              ctypes.POINTER(ctypes.c_float)),   # [nf * 6]
        ("num_faces",         ctypes.c_int32),
        ("mesh_verts",        ctypes.POINTER(ctypes.c_float)),   # [nv * 3]
        ("mesh_edges",        ctypes.POINTER(ctypes.c_int32)),   # [ne * 2]
        ("num_mesh_verts",    ctypes.c_int32),
        ("num_mesh_edges",    ctypes.c_int32),
        ("sing_positions",    ctypes.POINTER(ctypes.c_float)),   # [ns * 3]
        ("sing_indices",      ctypes.POINTER(ctypes.c_float)),   # [ns]
        ("num_singularities", ctypes.c_int32),
        ("quality_map",       ctypes.POINTER(ctypes.c_float)),   # [nf]
        ("summary",           ctypes.c_char_p),
    ]

_QFProgressCB = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_char_p, ctypes.c_void_p
)

_FIELD_SOLVER_MAP = {
    "EIGENVECTOR": 0,
    "EIGEN_SMOOTH": 0,
    "KNOPPEL": 1,
    "CURVATURE": 2,
}

_PARAM_METHOD_MAP = {
    "MIQ": 0,
    "IGM": 1,
    "POISSON": 2,
}

_EXTRACT_MAP = {
    "ISO": 0,
    "MOTORCYCLE": 1,
    "DUAL_CONTOUR": 2,
    "DUAL": 2,
    "GREEDY": 0,
}

_PRESET_MAP = {
    "CUSTOM": 0,
    "ORGANIC": 1,
    "HARD_SURFACE": 2,
    "SCULPT": 3,
    "ARCHITECTURE": 4,
    "FAST": 5,
}


# -----------------------------------------------------------------------
# Native engine wrapper
# -----------------------------------------------------------------------

class _NativeEngine:
    def __init__(self, path: str):
        system = platform.system()

        if system == "Windows":
            # On Windows + Python 3.8+, ctypes.CDLL no longer implicitly adds the
            # DLL's containing directory to the DLL search path for its own
            # dependencies (e.g. vcomp140.dll / libomp.dll for OpenMP).
            # os.add_dll_directory() fixes this.
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(os.path.dirname(os.path.abspath(path)))
                except OSError:
                    pass  # Non-fatal; DLL may still load if deps are on PATH/System32
            self._lib = ctypes.CDLL(path)

        elif system == "Linux":
            # BUG FIX: Linux requires RTLD_GLOBAL so libgomp's thread-local
            # storage is shared between QuadForge and any other OpenMP library
            # that Blender has already loaded.  Without this flag, each loaded
            # .so gets its own copy of the OpenMP runtime which causes subtle
            # data-race bugs and spurious crashes on multi-threaded meshes.
            import ctypes.util as _cu
            # Pre-load libgomp so its symbols resolve before quadforge loads.
            for _gomp in [_cu.find_library("gomp"),
                          "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
                          "/usr/lib64/libgomp.so.1"]:
                if _gomp:
                    try:
                        ctypes.CDLL(_gomp, mode=ctypes.RTLD_GLOBAL)
                        break
                    except OSError:
                        pass
            self._lib = ctypes.CDLL(
                path,
                mode=ctypes.RTLD_GLOBAL,  # RTLD_GLOBAL is critical for libgomp symbol sharing
            )

        else:  # macOS
            # BUG FIX: On macOS the dylib may embed @rpath install names that
            # need the containing directory on DYLD_LIBRARY_PATH (or equivalent).
            # Pre-load any bundled libomp if available, then load the dylib.
            dylib_dir = os.path.dirname(os.path.abspath(path))
            # Try to pre-load homebrew libomp (arm64: /opt/homebrew, x86: /usr/local)
            for _libomp in [
                os.path.join(dylib_dir, "libomp.dylib"),
                "/opt/homebrew/opt/libomp/lib/libomp.dylib",
                "/usr/local/opt/libomp/lib/libomp.dylib",
            ]:
                if os.path.isfile(_libomp):
                    try:
                        ctypes.CDLL(_libomp)
                        break
                    except OSError:
                        pass
            self._lib = ctypes.CDLL(path)

        # Configure all restype/argtypes before any calls.
        self._setup()
        # qf_init() returns 0 on success, non-zero on failure (standard C convention).
        ret = self._lib.qf_init()
        if ret != 0:
            raise RuntimeError(f"qf_init() failed (returned {ret})")
        self._native_ready = True

    def _setup(self):
        """Configure restype and argtypes for all exported C API functions.

        This must be called before any API function is invoked.  Without explicit
        restype the default is c_int which is fine for most functions, but pointer-
        returning functions (qf_version, qf_last_error, qf_remesh) would be silently
        misinterpreted as integers — a guaranteed crash or data corruption on 64-bit.
        """
        L = self._lib

        # ---- Lifecycle ----
        L.qf_init.restype     = ctypes.c_int;      L.qf_init.argtypes     = []
        L.qf_shutdown.restype = None;               L.qf_shutdown.argtypes = []

        # ---- Info ----
        L.qf_version.restype          = ctypes.c_char_p;  L.qf_version.argtypes          = []
        L.qf_last_error.restype       = ctypes.c_char_p;  L.qf_last_error.argtypes       = []
        L.qf_has_gpu.restype          = ctypes.c_int;     L.qf_has_gpu.argtypes          = []
        L.qf_cpu_thread_count.restype = ctypes.c_int;     L.qf_cpu_thread_count.argtypes = []

        # ---- Remesh ----
        L.qf_remesh.restype   = ctypes.POINTER(_QFResultC)
        L.qf_remesh.argtypes  = [
            ctypes.POINTER(_QFInputMeshC),
            ctypes.POINTER(_QFParamsC),
            _QFProgressCB,
            ctypes.c_void_p,
        ]
        L.qf_free_result.restype  = None
        L.qf_free_result.argtypes = [ctypes.POINTER(_QFResultC)]

        # ---- Params helpers ----
        for _fn_name in (
            "qf_default_params",
            "qf_preset_organic",
            "qf_preset_hard_surface",
            "qf_preset_sculpt",
            "qf_preset_architecture",
            "qf_preset_fast",
        ):
            fn = getattr(L, _fn_name)
            fn.restype  = _QFParamsC
            fn.argtypes = []

        # ---- Debug field (INTEGRATION FIX v82) ----
        L.qf_debug_field.restype  = ctypes.POINTER(_QFDebugFieldC)
        L.qf_debug_field.argtypes = [ctypes.c_float]
        L.qf_free_debug.restype   = None
        L.qf_free_debug.argtypes  = [ctypes.POINTER(_QFDebugFieldC)]

    def version(self) -> str:
        return self._lib.qf_version().decode("utf-8", errors="replace")

    def remesh(self, input_mesh, params, progress_cb=None):
        if not getattr(self, "_native_ready", False):
            ret = self._lib.qf_init()
            if ret != 0:
                raise RuntimeError(f"qf_init() failed (returned {ret})")
            self._native_ready = True

        vf = np.ascontiguousarray(input_mesh.vertices, dtype=np.float32)

        flat_faces = []
        face_sizes = []

        for f in input_mesh.faces:
            flat_faces.extend(f)
            face_sizes.append(len(f))

        faces_arr = np.array(flat_faces, dtype=np.int32)
        face_sizes_arr = np.array(face_sizes, dtype=np.int32)

        _keep_alive = [vf, faces_arr, face_sizes_arr]

        cm = _QFInputMeshC()
        ctypes.memset(ctypes.byref(cm), 0, ctypes.sizeof(cm))

        cm.positions = vf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cm.faces = faces_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        cm.face_sizes = face_sizes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        cm.num_vertices = int(vf.shape[0])
        cm.num_faces = int(len(face_sizes))

        # continue params + call qf_remesh after this...

        if input_mesh.normals is not None:
            nf = np.ascontiguousarray(input_mesh.normals, dtype=np.float32)
            cm.normals = nf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _keep_alive.append(nf)

        if input_mesh.vertex_colors is not None:
            vc = np.ascontiguousarray(input_mesh.vertex_colors, dtype=np.float32)
            cm.vertex_colors = vc.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _keep_alive.append(vc)

        if input_mesh.material_ids is not None:
            mi = np.ascontiguousarray(input_mesh.material_ids, dtype=np.int32)
            cm.material_ids = mi.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            _keep_alive.append(mi)

        if input_mesh.uv_coords is not None:
            uv = np.ascontiguousarray(input_mesh.uv_coords, dtype=np.float32)
            cm.uv_coords = uv.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            cm.num_uv_coords = int(uv.shape[0])
            _keep_alive.append(uv)

        if input_mesh.uv_loop_indices is not None:
            ui = np.ascontiguousarray(input_mesh.uv_loop_indices, dtype=np.int32)
            cm.uv_indices = ui.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            _keep_alive.append(ui)

        cp = _QFParamsC()
        cp.target_quad_count = params.target_quad_count
        cp.curvature_adaptivity = params.curvature_adaptivity
        cp.exact_quad_count = int(params.exact_quad_count)
        cp.auto_detect_hard_edges = int(params.auto_detect_hard_edges)
        cp.hard_edge_angle_deg = params.hard_edge_angle_deg
        cp.use_normals = int(params.use_normals)
        cp.use_materials = int(params.use_materials)
        cp.use_vertex_colors = int(params.use_vertex_colors)
        cp.use_uv_seams = int(params.use_uv_seams)
        cp.symmetry_x = int(params.symmetry[0])
        cp.symmetry_y = int(params.symmetry[1])
        cp.symmetry_z = int(params.symmetry[2])
        cp.smooth_iterations = params.smooth_iterations
        cp.smooth_strength = params.smooth_strength
        cp.feature_snap_distance = params.feature_snap_distance
        cp.num_threads = 1
        cp.use_gpu = 0

        cp.field_solver = params.field_solver
        cp.param_method = params.param_method
        cp.extraction_method = params.extraction_method
        cp.preset = params.preset

        c_cb = None

        if progress_cb:
            _names = [
                "Preprocessing",
                "Cross-field",
                "Parametrization",
                "Extraction",
                "Post-processing",
                "Output",
            ]

            def _wrap(stage, prog, name_b, _ud):
                name = name_b.decode() if name_b else _names[min(stage, 5)]
                progress_cb(name, float(prog))
                return 0

            c_cb = _QFProgressCB(_wrap)
            self._last_cb = c_cb

        print("[QuadForge][native] calling qf_remesh with ctypes ABI")
        rp = self._lib.qf_remesh(
            ctypes.byref(cm),
            ctypes.byref(cp),
            c_cb,
            None,
        )
        print("[QuadForge][native] qf_remesh returned")

        if not rp:
            err = self._lib.qf_last_error()
            raise RuntimeError(
                f"qf_remesh failed: {err.decode() if err else 'unknown'}"
            )

        r = rp.contents

        nv = r.num_vertices
        nq = r.num_quad_faces
        nt = r.num_tri_faces
        print(f"[QuadForge][native] qf_remesh result nv={nv} nq={nq} nt={nt}")

        overts = np.ctypeslib.as_array(
            r.positions,
            shape=(nv * 3,),
        ).copy().reshape(-1, 3)

        qflat = np.ctypeslib.as_array(
            r.faces,
            shape=(nq * 4,),
        ).copy()

        ofaces = [
            list(qflat[i * 4:(i + 1) * 4])
            for i in range(nq)
        ]

        if r.tri_faces and nt > 0:
            tflat = np.ctypeslib.as_array(
                r.tri_faces,
                shape=(nt * 3,),
            ).copy()

            ofaces += [
                list(tflat[i * 3:(i + 1) * 3])
                for i in range(nt)
            ]

        if nv == 0 or len(ofaces) == 0:
            self._lib.qf_free_result(rp)
            raise RuntimeError("qf_remesh completed stage 5 but produced empty geometry")

        self._lib.qf_free_result(rp)
        return overts, ofaces


    def shutdown(self):
        try: self._lib.qf_shutdown()
        except Exception: pass

    def debug_field(self, arm_scale: float = 0.0) -> Optional[dict]:
        """Return cross-field debug visualization data (requires prior remesh).

        Returns a dict with keys:
          dirs          — np.ndarray [nf, 2, 3]: per-face U/V direction pairs
          mesh_verts    — np.ndarray [nv, 3]: debug stick-mesh vertices
          mesh_edges    — np.ndarray [ne, 2]: debug stick-mesh edge indices
          sing_positions— np.ndarray [ns, 3]: singularity world positions
          sing_indices  — np.ndarray [ns]:    singularity index (+1/-1)
          quality_map   — np.ndarray [nf]:    per-face quality in [0,1]
          summary       — str: human-readable field statistics
        Returns None if qf_debug_field() fails (no cached remesh).
        """
        dp = self._lib.qf_debug_field(ctypes.c_float(arm_scale))
        if not dp:
            return None
        d = dp.contents
        nf = d.num_faces
        result = {}
        if d.dirs and nf > 0:
            raw = np.ctypeslib.as_array(d.dirs, shape=(nf * 6,)).copy()
            result["dirs"] = raw.reshape(nf, 2, 3)
        else:
            result["dirs"] = np.zeros((nf, 2, 3), dtype=np.float32)

        nv = d.num_mesh_verts
        ne = d.num_mesh_edges
        if d.mesh_verts and nv > 0:
            result["mesh_verts"] = np.ctypeslib.as_array(d.mesh_verts, shape=(nv * 3,)).copy().reshape(nv, 3)
        else:
            result["mesh_verts"] = np.zeros((0, 3), dtype=np.float32)
        if d.mesh_edges and ne > 0:
            result["mesh_edges"] = np.ctypeslib.as_array(d.mesh_edges, shape=(ne * 2,)).copy().reshape(ne, 2)
        else:
            result["mesh_edges"] = np.zeros((0, 2), dtype=np.int32)

        ns = d.num_singularities
        if d.sing_positions and ns > 0:
            result["sing_positions"] = np.ctypeslib.as_array(d.sing_positions, shape=(ns * 3,)).copy().reshape(ns, 3)
            result["sing_indices"]   = np.ctypeslib.as_array(d.sing_indices,   shape=(ns,)).copy()
        else:
            result["sing_positions"] = np.zeros((0, 3), dtype=np.float32)
            result["sing_indices"]   = np.zeros((0,),   dtype=np.float32)

        if d.quality_map and nf > 0:
            result["quality_map"] = np.ctypeslib.as_array(d.quality_map, shape=(nf,)).copy()
        else:
            result["quality_map"] = np.ones(nf, dtype=np.float32)

        result["summary"] = d.summary.decode("utf-8", errors="replace") if d.summary else ""

        self._lib.qf_free_debug(dp)
        return result


# -----------------------------------------------------------------------
# Engine selection
# -----------------------------------------------------------------------

_native_engine: Optional[_NativeEngine] = None
_native_available: Optional[bool] = None


def _get_engine():
    global _native_engine, _native_available
    if _native_available is None:
        path = _find_native_library()
        if path:
            try:
                _native_engine = _NativeEngine(path)
                _native_available = True
                print(f"[QuadForge] Native engine: {path} v{_native_engine.version()}")
            except Exception as exc:
                print(f"[QuadForge] Native engine load failed ({exc}), using Python fallback.")
                _native_available = False
        else:
            _native_available = False
            print("[QuadForge] No compiled library found — using pure-Python engine.")
    return _native_engine if _native_available else None


def is_native_available() -> bool:
    _get_engine()
    return bool(_native_available)


# -----------------------------------------------------------------------
# Input / Params containers
# -----------------------------------------------------------------------

@dataclass
class QFInputMesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uv_coords: Optional[np.ndarray] = None
    uv_loop_indices: Optional[np.ndarray] = None
    vertex_colors: Optional[np.ndarray] = None
    material_ids: Optional[np.ndarray] = None
    original_faces: Optional[list] = None
    # UV seam edges: set of (v0, v1) tuples (v0 < v1) for edges marked as UV seams.
    # Populated by extract_mesh() when the mesh has an active UV layer.
    uv_seam_edges: Optional[set] = None
    # Per-face smooth shading flags: True = smooth, False = flat.
    face_smooth: Optional[np.ndarray] = None


@dataclass
class QFParams:
    target_quad_count: int = 5000
    curvature_adaptivity: float = 0.5
    exact_quad_count: bool = False
    auto_detect_hard_edges: bool = True
    hard_edge_angle_deg: float = 30.0
    use_normals: bool = False
    use_materials: bool = False
    use_vertex_colors: bool = False
    use_uv_seams: bool = False
    symmetry: Tuple[bool, bool, bool] = (False, False, False)
    smooth_iterations: int = 10
    smooth_strength: float = 0.5
    feature_snap_distance: float = 0.1
    num_threads: int = 0
    use_gpu: bool = False
    field_solver: int = 1
    param_method: int = 0
    extraction_method: int = 0
    preset: int = 0
    shade_smooth_output: bool = True
    keep_original: bool = True


# -----------------------------------------------------------------------
# Mesh extraction  (Blender → numpy)
# -----------------------------------------------------------------------

def extract_mesh(obj: bpy.types.Object) -> QFInputMesh:
    mesh = obj.data
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        raise ValueError("Mesh has no valid geometry")

    num_verts = len(mesh.vertices)
    verts_flat = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts_flat)
    vertices = verts_flat.reshape((-1, 3))

    normals = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("normal", normals)
    normals = normals.reshape((-1, 3))

    original_faces = []
    for poly in mesh.polygons:
        face_verts = [mesh.loops[li].vertex_index
                      for li in range(poly.loop_start, poly.loop_start + poly.loop_total)]
        original_faces.append(face_verts)

    num_polys = len(mesh.polygons)
    material_ids_poly = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", material_ids_poly)

    mesh.calc_loop_triangles()
    num_tris = len(mesh.loop_triangles)
    tri_flat = np.empty(num_tris * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_flat)

    # BUG FIX (v84): material_ids must be per-TRIANGLE, not per-polygon.
    # The C engine receives num_faces = num_tris (all faces are triangulated),
    # so material_ids must have exactly num_tris entries.  On any mesh that
    # has quad or n-gon faces, num_tris > num_polys and passing the
    # per-polygon array caused an out-of-bounds read in stage1_preprocess()
    # (mat_ids.assign(material_ids, material_ids + m_nf) where m_nf=num_tris)
    # and in stage6_output() (m_input->material_ids[orig_fi] where orig_fi
    # is a triangle index up to num_tris-1).
    #
    # Fix: fetch the originating polygon index for each loop_triangle via
    # loop_triangles.foreach_get("polygon_index", ...) and use it to index
    # into material_ids_poly, producing a per-triangle array of the correct
    # length.  This is safe and cheap — it's one extra foreach_get call.
    poly_indices = np.empty(num_tris, dtype=np.int32)
    mesh.loop_triangles.foreach_get("polygon_index", poly_indices)
    material_ids = material_ids_poly[poly_indices]   # shape (num_tris,)
    faces = tri_flat.reshape((-1, 3))

    uv_coords = None
    uv_loop_indices = None
    uv_seam_edges = None
    if mesh.uv_layers.active is not None:
        num_loops = len(mesh.loops)
        uv_flat = np.empty(num_loops * 2, dtype=np.float32)
        mesh.uv_layers.active.data.foreach_get("uv", uv_flat)
        uv_coords = uv_flat.reshape((-1, 2))

        # BUG FIX: Extract per-triangle-face-vertex UV indices so the C API
        # receives uv_indices (which vertex in uv_coords corresponds to each
        # triangle corner).  loop_triangles gives us the loop index per corner,
        # and since UV data is indexed by loop, we can use loops directly as
        # UV indices into the flat uv_coords array.
        mesh.calc_loop_triangles()
        num_tris_uv = len(mesh.loop_triangles)
        uv_idx_flat = np.empty(num_tris_uv * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", uv_idx_flat)
        uv_loop_indices = uv_idx_flat  # shape (F*3,) — loop == uv index

        # Extract UV seam edges — edges flagged as seams in the mesh.
        # These become hard constraints in the cross-field when use_uv_seams=True.
        seam_set: set = set()
        use_seams_flat = np.empty(len(mesh.edges), dtype=bool)
        mesh.edges.foreach_get("use_seam", use_seams_flat)
        edge_verts_flat = np.empty(len(mesh.edges) * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edge_verts_flat)
        edge_verts = edge_verts_flat.reshape((-1, 2))
        seam_indices = np.where(use_seams_flat)[0]
        for ei in seam_indices:
            v0, v1 = int(edge_verts[ei, 0]), int(edge_verts[ei, 1])
            seam_set.add((v0, v1) if v0 < v1 else (v1, v0))
        uv_seam_edges = seam_set if seam_set else None

    vertex_colors = None
    if mesh.color_attributes:
        ca = mesh.color_attributes.active_color
        if ca is not None:
            if ca.domain == 'POINT':
                cf = np.empty(len(ca.data) * 4, dtype=np.float32)
                ca.data.foreach_get("color", cf)
                vertex_colors = cf.reshape(-1, 4)[:num_verts, :3].copy()
            elif ca.domain == 'CORNER':
                # BUG-G FIX (v116): Previous code iterated over mesh.loops in a
                # Python for-loop (O(loops) pure-Python ≈ 300K iterations for a
                # dense mesh, ~300 ms of UI freeze per remesh).  Replaced with
                # fully-vectorized NumPy scatter-add using np.add.at(), which
                # runs in C and is ~100× faster for typical meshes.
                cf = np.empty(len(ca.data) * 4, dtype=np.float32)
                ca.data.foreach_get("color", cf)
                col = cf.reshape(-1, 4)[:, :3]      # (num_loops, 3) RGB only
                num_loops = len(mesh.loops)
                # Gather vertex indices for every loop in one foreach_get call.
                vi_flat = np.empty(num_loops, dtype=np.int32)
                mesh.loops.foreach_get("vertex_index", vi_flat)
                # Scatter-add: accumulate RGB values from all loops onto vertices.
                vertex_colors = np.zeros((num_verts, 3), dtype=np.float32)
                counts        = np.zeros(num_verts,      dtype=np.float32)
                np.add.at(vertex_colors, vi_flat, col[:num_loops])
                np.add.at(counts,        vi_flat, 1.0)
                # Normalise — safe divide (isolated vertices get colour 0).
                mask = counts > 0
                vertex_colors[mask] /= counts[mask, np.newaxis]

    # Per-polygon smooth-shading flag — used by output_enhance to transfer shading.
    face_smooth = np.empty(num_polys, dtype=bool)
    mesh.polygons.foreach_get("use_smooth", face_smooth)

    return QFInputMesh(vertices=vertices, faces=faces, normals=normals,
                       uv_coords=uv_coords, uv_loop_indices=uv_loop_indices,
                       vertex_colors=vertex_colors,
                       material_ids=material_ids, original_faces=original_faces,
                       uv_seam_edges=uv_seam_edges, face_smooth=face_smooth)


# -----------------------------------------------------------------------
# Parameter conversion
# -----------------------------------------------------------------------

def build_params(settings) -> QFParams:
    preset = _PRESET_MAP.get(
        str(getattr(settings, "preset", "CUSTOM")).upper(),
        0,
    )

    field_solver = _FIELD_SOLVER_MAP.get(
        str(getattr(settings, "field_solver", "KNOPPEL")).upper(),
        1,
    )

    param_method = _PARAM_METHOD_MAP.get(
        str(getattr(settings, "param_method", "MIQ")).upper(),
        0,
    )

    extraction_method = _EXTRACT_MAP.get(
        str(getattr(settings, "extraction_method", "ISO")).upper(),
        0,
    )

    return QFParams(
        target_quad_count      = int(settings.target_quad_count),
        curvature_adaptivity   = float(settings.curvature_adaptivity),
        exact_quad_count       = bool(settings.exact_quad_count),
        auto_detect_hard_edges = bool(settings.auto_detect_hard_edges),
        hard_edge_angle_deg    = float(settings.hard_edge_angle_deg),
        use_normals            = bool(settings.use_normals),
        use_materials          = bool(settings.use_materials),
        use_vertex_colors      = bool(settings.use_vertex_colors),
        use_uv_seams           = bool(settings.use_uv_seams),
        symmetry = (
            bool(settings.symmetry_x),
            bool(settings.symmetry_y),
            bool(settings.symmetry_z),
        ),
        smooth_iterations      = int(settings.smooth_iterations),
        smooth_strength        = float(settings.smooth_strength),
        feature_snap_distance  = float(settings.feature_snap_distance),
        num_threads            = int(settings.num_threads),
        use_gpu                = bool(settings.use_gpu),

        preset                 = preset,
        field_solver           = field_solver,
        param_method           = param_method,
        extraction_method      = extraction_method,

        shade_smooth_output    = bool(getattr(settings, "shade_smooth_output", True)),
        keep_original          = bool(getattr(settings, "keep_original", True)),
    )

# -----------------------------------------------------------------------
# Main remesh entry point (dispatches native → Python)
# -----------------------------------------------------------------------

def _run_python_pipeline(input_mesh: QFInputMesh, params: QFParams, progress_cb: Optional[Callable[[str, float], None]] = None):
    from .engine.pipeline import run_pipeline, run_pipeline_exact_count
    if params.exact_quad_count:
        return run_pipeline_exact_count(input_mesh, params, progress_cb=progress_cb)
    return run_pipeline(input_mesh, params, progress_cb=progress_cb)


def remesh(
    input_mesh: QFInputMesh,
    params: QFParams,
    progress_cb: Optional[Callable[[str, float], None]] = None,
):
    """Run QuadForge remesher. Native C++ if available, else Python fallback.

    Returns either:
      - A _PipelineResult object  (has .verts, .faces, .output_uvs, …)  — new path
      - A tuple (verts, faces)    — native engine path or legacy fallback
      - A tuple (verts, faces, actual_count) — exact_quad_count Python path

    Callers should use hasattr(result, 'verts') to detect the object path,
    then isinstance(result, tuple) for the tuple paths.
    """
    eng = _get_engine()
    print(
        f"[QuadForge] remesh() called: verts={len(input_mesh.vertices)} "
        f"faces={len(input_mesh.faces)} native={'yes' if eng is not None else 'no'}"
    )
    if eng is not None:
        try:
            return eng.remesh(input_mesh, params, progress_cb)
        except Exception as exc:
            message = str(exc)
            if (
                isinstance(exc, (RuntimeError, OSError, ValueError, TypeError))
                or "Stage 5" in message
                or "No output geometry" in message
                or "qf_remesh failed" in message
            ):
                print(
                    "[QuadForge] Native engine failure — falling back to pure-Python engine: "
                    + message
                )
                return _run_python_pipeline(input_mesh, params, progress_cb)
            raise
    return _run_python_pipeline(input_mesh, params, progress_cb)


# -----------------------------------------------------------------------
# Output mesh construction  (numpy → Blender)
# -----------------------------------------------------------------------

def build_output_mesh(
    name: str,
    vertices: np.ndarray,
    faces: list,
    normals: Optional[np.ndarray] = None,
    material_ids: Optional[np.ndarray] = None,
) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(name)
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("build_output_mesh() received invalid vertex array shape")

    num_verts = len(verts)
    num_faces = len(faces)
    if num_verts == 0 or num_faces == 0:
        raise ValueError("build_output_mesh() received empty geometry")

    face_lists = [list(f) for f in faces]
    loop_total = sum(len(f) for f in face_lists)

    # Validate face indices before building any Blender mesh data.
    for fi, face in enumerate(face_lists):
        if len(face) < 3:
            raise ValueError(f"build_output_mesh() received face {fi} with fewer than 3 vertices")
        for vi in face:
            if not isinstance(vi, int):
                raise ValueError(f"build_output_mesh() received non-integer vertex index in face {fi}")
            if vi < 0 or vi >= num_verts:
                raise ValueError(
                    f"build_output_mesh() received invalid vertex index {vi} in face {fi}"
                )

    if num_verts < 10_000:
        mesh.from_pydata(verts.tolist(), [], face_lists)
    else:
        mesh.vertices.add(num_verts)
        mesh.vertices.foreach_set("co", verts.ravel())
        mesh.loops.add(loop_total)
        mesh.polygons.add(num_faces)
        lv = np.empty(loop_total, dtype=np.int32)
        ls = np.empty(num_faces,  dtype=np.int32)
        lt = np.empty(num_faces,  dtype=np.int32)
        offset = 0
        for fi, face in enumerate(face_lists):
            n = len(face)
            ls[fi] = offset
            lt[fi] = n
            for i, vi in enumerate(face):
                lv[offset + i] = vi
            offset += n
        mesh.loops.foreach_set("vertex_index", lv)
        mesh.polygons.foreach_set("loop_start", ls)
        mesh.polygons.foreach_set("loop_total", lt)

    mesh.update()
    mesh.validate()
    # Note: vertex normals in Blender 4.x are computed automatically by
    # mesh.update() — foreach_set("normal", ...) is read-only and has no
    # effect. Normals passed in are ignored here; callers should use
    # mesh.normals_split_custom_set() if custom split normals are needed.
    return mesh
