#!/usr/bin/env python3
"""
verify_api.py — Load libquadforge.so and verify all 14 exported C API functions.

Run this after building to confirm the .so matches the api.h contract
expected by bridge.py.  Exits 0 on success, 1 on failure.

Usage:
    python3 QuadForge/engine/prebuilt/linux64/verify_api.py
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "libquadforge.so")

# -----------------------------------------------------------------------
# ctypes mirrors (must match bridge.py exactly)
# -----------------------------------------------------------------------

class QFInputMesh(ctypes.Structure):
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

class QFParams(ctypes.Structure):
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

class QFResult(ctypes.Structure):
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

QFProgressCB = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_char_p, ctypes.c_void_p
)


def _make_tetrahedron():
    """Build a minimal 4-vert, 4-triangle tetrahedron for smoke testing."""
    # 4 vertices
    pos = (ctypes.c_float * 12)(
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
        0.5, 0.5, 1.0,
    )
    # 4 triangular faces
    faces = (ctypes.c_int32 * 12)(
        0, 1, 2,
        0, 1, 3,
        1, 2, 3,
        0, 2, 3,
    )
    face_sizes = (ctypes.c_int32 * 4)(3, 3, 3, 3)

    mesh = QFInputMesh()
    mesh.positions = ctypes.cast(pos, ctypes.POINTER(ctypes.c_float))
    mesh.faces = ctypes.cast(faces, ctypes.POINTER(ctypes.c_int32))
    mesh.face_sizes = ctypes.cast(face_sizes, ctypes.POINTER(ctypes.c_int32))
    mesh.num_vertices = 4
    mesh.num_faces = 4
    mesh.num_uv_coords = 0
    return mesh, pos, faces, face_sizes  # keep refs alive


def main() -> int:
    passed = 0
    failed = 0

    def ok(name: str, detail: str = ""):
        nonlocal passed
        passed += 1
        print(f"  ✓ {name}" + (f"  ({detail})" if detail else ""))

    def fail(name: str, detail: str = ""):
        nonlocal failed
        failed += 1
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))

    # ---- Load library ----
    print(f"\n=== QuadForge linux64 API verification ===\n")
    print(f"Library: {LIB_PATH}")

    if not os.path.isfile(LIB_PATH):
        fail("file_exists", f"{LIB_PATH} not found")
        return 1

    try:
        lib = ctypes.CDLL(LIB_PATH)
        ok("load_library")
    except OSError as e:
        fail("load_library", str(e))
        return 1

    # ---- 1. qf_init ----
    try:
        lib.qf_init.restype = ctypes.c_int
        lib.qf_init.argtypes = []
        ret = lib.qf_init()
        if ret == 0:
            ok("qf_init", f"returned {ret}")
        else:
            fail("qf_init", f"returned {ret} (expected 0)")
    except Exception as e:
        fail("qf_init", str(e))

    # ---- 2. qf_version ----
    try:
        lib.qf_version.restype = ctypes.c_char_p
        lib.qf_version.argtypes = []
        ver = lib.qf_version()
        ver_str = ver.decode("utf-8", errors="replace") if ver else ""
        if ver_str:
            ok("qf_version", ver_str)
        else:
            fail("qf_version", "empty string")
    except Exception as e:
        fail("qf_version", str(e))

    # ---- 3. qf_last_error ----
    try:
        lib.qf_last_error.restype = ctypes.c_char_p
        lib.qf_last_error.argtypes = []
        err = lib.qf_last_error()
        err_str = err.decode("utf-8", errors="replace") if err else ""
        ok("qf_last_error", f"'{err_str}'" if err_str else "(empty — correct after init)")
    except Exception as e:
        fail("qf_last_error", str(e))

    # ---- 4. qf_has_gpu ----
    try:
        lib.qf_has_gpu.restype = ctypes.c_int
        lib.qf_has_gpu.argtypes = []
        gpu = lib.qf_has_gpu()
        ok("qf_has_gpu", f"{gpu}")
    except Exception as e:
        fail("qf_has_gpu", str(e))

    # ---- 5. qf_cpu_thread_count ----
    try:
        lib.qf_cpu_thread_count.restype = ctypes.c_int
        lib.qf_cpu_thread_count.argtypes = []
        tc = lib.qf_cpu_thread_count()
        if tc >= 1:
            ok("qf_cpu_thread_count", f"{tc} threads")
        else:
            fail("qf_cpu_thread_count", f"{tc} (expected >= 1)")
    except Exception as e:
        fail("qf_cpu_thread_count", str(e))

    # ---- 6-10. Preset functions ----
    preset_fns = [
        "qf_default_params",
        "qf_preset_organic",
        "qf_preset_hard_surface",
        "qf_preset_sculpt",
        "qf_preset_architecture",
        "qf_preset_fast",
    ]
    for fn_name in preset_fns:
        try:
            fn = getattr(lib, fn_name)
            fn.restype = QFParams
            fn.argtypes = []
            p = fn()
            if p.target_quad_count > 0:
                ok(fn_name, f"target={p.target_quad_count}, adapt={p.curvature_adaptivity:.2f}")
            else:
                fail(fn_name, f"target_quad_count={p.target_quad_count}")
        except Exception as e:
            fail(fn_name, str(e))

    # ---- 11. qf_remesh (smoke test with tetrahedron) ----
    try:
        lib.qf_remesh.restype = ctypes.POINTER(QFResult)
        lib.qf_remesh.argtypes = [
            ctypes.POINTER(QFInputMesh),
            ctypes.POINTER(QFParams),
            QFProgressCB,
            ctypes.c_void_p,
        ]

        mesh, _p, _f, _fs = _make_tetrahedron()
        params = lib.qf_default_params()
        params.target_quad_count = 10
        params.smooth_iterations = 2

        stages_seen = []
        def _progress(stage, prog, name, _ud):
            sname = name.decode() if name else f"stage{stage}"
            stages_seen.append((stage, round(prog, 2), sname))
            return 0
        cb = QFProgressCB(_progress)

        rp = lib.qf_remesh(ctypes.byref(mesh), ctypes.byref(params), cb, None)
        if rp:
            r = rp.contents
            detail = (f"verts={r.num_vertices}, quads={r.num_quad_faces}, "
                      f"tris={r.num_tri_faces}, quad%={r.quad_percentage:.1f}, "
                      f"valence={r.avg_valence:.2f}, time={r.elapsed_seconds:.4f}s")
            ok("qf_remesh", detail)

            # ---- 12. qf_free_result ----
            lib.qf_free_result.restype = None
            lib.qf_free_result.argtypes = [ctypes.POINTER(QFResult)]
            lib.qf_free_result(rp)
            ok("qf_free_result")
        else:
            err = lib.qf_last_error()
            err_s = err.decode() if err else "unknown"
            fail("qf_remesh", f"returned NULL — {err_s}")
            fail("qf_free_result", "skipped (no result)")

        if stages_seen:
            ok("progress_callback", f"{len(stages_seen)} callbacks, stages 0..{max(s for s,_,_ in stages_seen)}")
        else:
            fail("progress_callback", "no callbacks received")
    except Exception as e:
        fail("qf_remesh", str(e))

    # ---- 13. qf_shutdown ----
    try:
        lib.qf_shutdown.restype = None
        lib.qf_shutdown.argtypes = []
        lib.qf_shutdown()
        ok("qf_shutdown")
    except Exception as e:
        fail("qf_shutdown", str(e))

    # ---- Summary ----
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"  Results:  {passed}/{total} passed", end="")
    if failed:
        print(f",  {failed} FAILED")
    else:
        print("  — ALL OK")
    print(f"{'='*50}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
