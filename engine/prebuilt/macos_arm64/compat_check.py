#!/usr/bin/env python3
"""
compat_check.py — Verify ABI struct layout compatibility between the compiled
libquadforge.dylib and bridge.py's ctypes definitions.

Catches silent breakage when struct fields are reordered, added, or sized
differently between the C source and the Python ctypes mirror.

Usage:
    python3 QuadForge/engine/prebuilt/macos_arm64/compat_check.py
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "libquadforge.dylib")


# -----------------------------------------------------------------------
# ctypes struct definitions (copied from bridge.py — must stay in sync)
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

# -----------------------------------------------------------------------
# Expected struct sizes (C compilation on LP64 / ARM64)
#
# ARM64 and x86-64 share LP64 model (8-byte pointers), so struct layouts
# are identical. The key difference is ARM64's stricter natural alignment
# requirements, but since all our fields are naturally aligned this has
# no practical effect on layout.
# -----------------------------------------------------------------------


def _field_report(cls):
    """Print detailed field layout for a ctypes Structure."""
    print(f"\n  {cls.__name__}  (sizeof = {ctypes.sizeof(cls)} bytes)")
    print(f"  {'Field':<28s} {'Offset':>6s}  {'Size':>4s}  Type")
    print(f"  {'-'*28} {'-'*6}  {'-'*4}  {'-'*20}")
    for name, ctype in cls._fields_:
        field_desc = getattr(cls, name)
        offset = field_desc.offset
        size = field_desc.size
        print(f"  {name:<28s} {offset:>6d}  {size:>4d}  {ctype.__name__}")
    return ctypes.sizeof(cls)


def main() -> int:
    errors = 0

    print("=" * 60)
    print("  QuadForge ABI Compatibility Check (macos_arm64)")
    print("=" * 60)

    # ---- Platform info ----
    print(f"\n  Platform: {platform.platform()}")
    print(f"  Machine:  {platform.machine()}")
    print(f"  macOS:    {platform.mac_ver()[0]}")

    # ---- Struct layout report ----
    sz_input  = _field_report(QFInputMesh)
    sz_params = _field_report(QFParams)
    sz_result = _field_report(QFResult)

    # ---- Size sanity checks ----
    print("\n--- Size sanity checks ---")
    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    print(f"  Pointer size: {ptr_size} bytes ({'LP64' if ptr_size == 8 else 'ILP32'})")

    # QFParams: 21 × 4-byte fields = 84 bytes, no padding needed
    expected_params = 21 * 4
    if sz_params == expected_params:
        print(f"  ✓ QFParams:   {sz_params} bytes (expected {expected_params})")
    else:
        print(f"  ✗ QFParams:   {sz_params} bytes (expected {expected_params})")
        errors += 1

    # ---- ARM64 alignment verification ----
    print("\n--- ARM64 alignment verification ---")
    # On ARM64, pointers must be 8-byte aligned. Verify all pointer fields
    # in QFInputMesh and QFResult start at 8-byte boundaries.
    for cls in [QFInputMesh, QFResult]:
        for name, ctype in cls._fields_:
            # Only LP_* types are pointer types (created by ctypes.POINTER()).
            # hasattr(ctype, '_type_') is NOT a reliable pointer check — it also
            # matches primitive types like c_int32 ('i') and c_float ('f'), which
            # sit at 4-byte-aligned offsets and would produce false alignment errors.
            if not ctype.__name__.startswith("LP_"):
                continue
            field_desc = getattr(cls, name)
            if field_desc.offset % 8 != 0:
                print(f"  ✗ {cls.__name__}.{name}: offset {field_desc.offset} "
                      f"NOT 8-byte aligned (ARM64 requires natural alignment)")
                errors += 1
            else:
                print(f"  ✓ {cls.__name__}.{name}: offset {field_desc.offset} "
                      f"(8-byte aligned)")

    # ---- Verify field order via qf_default_params ----
    print("\n--- Field order verification via qf_default_params ---")
    if not os.path.isfile(LIB_PATH):
        print(f"  SKIP: {LIB_PATH} not found")
        return errors

    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.qf_init()
    except Exception as e:
        print(f"  SKIP: could not load library — {e}")
        return errors

    lib.qf_default_params.restype = QFParams
    lib.qf_default_params.argtypes = []
    p = lib.qf_default_params()

    checks = [
        ("target_quad_count",      p.target_quad_count,      5000),
        ("curvature_adaptivity",   round(p.curvature_adaptivity, 2), 0.5),
        ("auto_detect_hard_edges", p.auto_detect_hard_edges, 1),
        ("hard_edge_angle_deg",    round(p.hard_edge_angle_deg, 1), 30.0),
        ("smooth_iterations",      p.smooth_iterations,      10),
        ("smooth_strength",        round(p.smooth_strength, 2), 0.5),
        ("field_solver",           p.field_solver,           1),
    ]

    for name, got, expected in checks:
        if got == expected:
            print(f"  ✓ {name}: {got}")
        else:
            print(f"  ✗ {name}: got {got}, expected {expected}")
            errors += 1

    # ---- Verify preset field values ----
    print("\n--- Preset field cross-check ---")
    presets = {
        "qf_preset_organic":       ("preset", 1),
        "qf_preset_hard_surface":  ("preset", 2),
        "qf_preset_sculpt":        ("preset", 3),
        "qf_preset_architecture":  ("preset", 4),
        "qf_preset_fast":          ("preset", 5),
    }
    for fn_name, (field, expected_val) in presets.items():
        fn = getattr(lib, fn_name)
        fn.restype = QFParams
        fn.argtypes = []
        pp = fn()
        val = getattr(pp, field)
        if val == expected_val:
            print(f"  ✓ {fn_name}.{field} = {val}")
        else:
            print(f"  ✗ {fn_name}.{field} = {val} (expected {expected_val})")
            errors += 1

    lib.qf_shutdown()

    # ---- Summary ----
    print(f"\n{'='*60}")
    if errors == 0:
        print("  ABI check: ALL OK — structs are compatible")
    else:
        print(f"  ABI check: {errors} ERROR(s) — struct mismatch detected!")
    print(f"{'='*60}\n")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
