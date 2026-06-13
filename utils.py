"""QuadForge utilities.

Platform detection, Blender API compatibility layer, and miscellaneous helpers.
All Blender API calls that differ across versions are gated through this module.

Risk 4 Mitigation (from roadmap §VII): all Blender API calls that changed between
4.x and 5.x are wrapped here so the rest of the add-on never calls bpy directly
for version-sensitive operations.
"""

from __future__ import annotations

import os
import sys
import platform
from typing import Optional, Tuple, List, Any


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def get_platform() -> str:
    """Return the platform string: 'windows', 'linux', or 'macos'."""
    p = platform.system().lower()
    if p == 'darwin':
        return 'macos'
    return p


def get_arch() -> str:
    """Return CPU architecture: 'arm64' or 'x86_64'."""
    m = platform.machine().lower()
    if m in ('arm64', 'aarch64'):
        return 'arm64'
    return 'x86_64'


def get_cpu_count() -> int:
    """Return the number of available CPU cores."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


# ---------------------------------------------------------------------------
# Blender version helpers
# ---------------------------------------------------------------------------

def get_blender_version() -> Tuple[int, int, int]:
    """Return the current Blender version as a tuple (major, minor, patch)."""
    try:
        import bpy
        return tuple(bpy.app.version)  # type: ignore[return-value]
    except Exception:
        return (4, 2, 0)


def is_blender_5() -> bool:
    """True if running in Blender 5.x or later."""
    return get_blender_version()[0] >= 5


def is_blender_4_2_plus() -> bool:
    """True if running in Blender 4.2+ (LTS baseline for this addon)."""
    v = get_blender_version()
    return (v[0], v[1]) >= (4, 2)


def is_blender_4_3_plus() -> bool:
    """True if running in Blender 4.3+."""
    v = get_blender_version()
    return (v[0], v[1]) >= (4, 3)


def check_blender_version_compatible() -> Tuple[bool, str]:
    """Return (ok, message) — True if the running Blender version is supported."""
    v = get_blender_version()
    if (v[0], v[1]) < (4, 2):
        return False, (
            f"QuadForge requires Blender 4.2 or later. "
            f"Current: {v[0]}.{v[1]}.{v[2]}"
        )
    return True, f"Blender {v[0]}.{v[1]}.{v[2]} — supported"


# ---------------------------------------------------------------------------
# NumPy / SciPy availability
# ---------------------------------------------------------------------------

def check_numpy_available() -> Tuple[bool, str]:
    """Return (ok, message) — True if numpy is importable at required version."""
    try:
        import numpy as np
        version = tuple(int(x) for x in np.__version__.split('.')[:2])
        if version < (1, 21):
            return False, f"QuadForge needs NumPy >= 1.21 (found {np.__version__})"
        return True, f"NumPy {np.__version__} — OK"
    except ImportError:
        return False, "NumPy not found — QuadForge cannot run"


def check_scipy_available() -> Tuple[bool, str]:
    """Return (ok, message) — SciPy is optional (speeds up sparse solves)."""
    try:
        import scipy
        return True, f"SciPy {scipy.__version__} — optional solver available"
    except ImportError:
        return False, "SciPy not installed — using built-in sparse solver"


# ---------------------------------------------------------------------------
# Add-on path helpers
# ---------------------------------------------------------------------------

def addon_dir() -> str:
    """Return the directory containing this addon."""
    return os.path.dirname(os.path.abspath(__file__))


def engine_prebuilt_dir(plat: Optional[str] = None, arch: Optional[str] = None) -> str:
    """Return the platform-specific prebuilt library directory."""
    if plat is None:
        plat = get_platform()
    if arch is None:
        arch = get_arch()

    dir_map = {
        ('windows', 'x86_64'): 'win64',
        ('linux',   'x86_64'): 'linux64',
        ('macos',   'arm64'):  'macos_arm64',
        ('macos',   'x86_64'): 'macos_x64',
    }
    subdir = dir_map.get((plat, arch), 'linux64')
    return os.path.join(addon_dir(), 'engine', 'prebuilt', subdir)


def native_library_name() -> str:
    """Return the expected filename for the compiled engine library."""
    p = get_platform()
    if p == 'windows':
        return 'quadforge.dll'
    if p == 'macos':
        return 'libquadforge.dylib'
    return 'libquadforge.so'


def native_library_path() -> str:
    """Return the full expected path to the compiled engine library."""
    return os.path.join(engine_prebuilt_dir(), native_library_name())


def native_library_exists() -> bool:
    """Return True if the compiled engine library exists on disk."""
    return os.path.isfile(native_library_path())


# ---------------------------------------------------------------------------
# Blender mesh API compatibility shims — version-sensitive calls only
# ---------------------------------------------------------------------------

def calc_loop_triangles(mesh) -> None:
    """Call mesh.calc_loop_triangles() — stable across Blender 4.x and 5.x."""
    mesh.calc_loop_triangles()


def get_active_uv_layer(mesh):
    """Return the active UV layer or None — compatible across versions."""
    try:
        return mesh.uv_layers.active
    except Exception:
        return None


def get_active_color_attribute(mesh):
    """Return the active vertex color attribute or None.

    Blender 3.2+ uses color_attributes; Blender 4.x and 5.x use it exclusively.
    """
    try:
        if hasattr(mesh, 'color_attributes') and mesh.color_attributes:
            active = mesh.color_attributes.active_color
            if active is not None:
                return active
    except Exception:
        pass
    # Legacy fallback (Blender < 3.2)
    try:
        if mesh.vertex_colors.active:
            return mesh.vertex_colors.active
    except Exception:
        pass
    return None


def set_mesh_smooth_shading(mesh, smooth: bool) -> None:
    """Set all polygons to smooth or flat shading — compatible across versions."""
    for poly in mesh.polygons:
        poly.use_smooth = smooth
    mesh.update()


def calc_normals_split(mesh) -> None:
    """Compute split normals — API changed in Blender 4.1.

    In Blender < 4.1: mesh.calc_normals_split() is required.
    In Blender 4.1+:  the method was removed; loop normals are automatic.
    """
    v = get_blender_version()
    if (v[0], v[1]) < (4, 1):
        try:
            mesh.calc_normals_split()
        except AttributeError:
            pass


def get_loop_normals(mesh):
    """Return per-loop normals as an (L, 3) float32 array, or None."""
    try:
        import numpy as np
        normals = np.zeros(len(mesh.loops) * 3, dtype=np.float32)
        mesh.loops.foreach_get("normal", normals)
        return normals.reshape(-1, 3)
    except Exception:
        return None


def get_face_sets(mesh):
    """Return the sculpt face-sets attribute or None (Blender 4.x/5.x).

    face_maps was removed in Blender 4.0 and replaced by int attributes.
    """
    try:
        for name in (".sculpt_face_set", "face_sets", ".face_set"):
            attr = mesh.attributes.get(name)
            if attr is not None:
                return attr
    except Exception:
        pass
    return None


def has_face_maps(mesh) -> bool:
    """Return True if the mesh has legacy face maps (Blender < 4.0 only)."""
    v = get_blender_version()
    if (v[0], v[1]) >= (4, 0):
        return False
    try:
        return len(mesh.face_maps) > 0
    except AttributeError:
        return False


def get_mesh_uv_seam_edges(mesh) -> List[int]:
    """Return list of edge indices that are UV seams (stable across all versions)."""
    return [e.index for e in mesh.edges if e.use_seam]


def foreach_get_verts(mesh):
    """Extract vertex positions as (V, 3) float32 via foreach_get (fast path)."""
    import numpy as np
    verts = np.zeros(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    return verts.reshape(-1, 3)


def foreach_get_loop_verts(mesh):
    """Extract vertex index for each loop via foreach_get."""
    import numpy as np
    v = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", v)
    return v


def foreach_get_material_ids(mesh):
    """Extract per-polygon material index via foreach_get."""
    import numpy as np
    n = len(mesh.polygons)
    mats = np.zeros(n, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", mats)
    return mats


def foreach_set_verts(mesh, positions) -> None:
    """Write vertex positions from a (V, 3) float32 array via foreach_set.

    50-200× faster than Python-level per-vertex assignment on large meshes.
    """
    import numpy as np
    flat = np.asarray(positions, dtype=np.float32).ravel()
    mesh.vertices.foreach_set("co", flat)
    mesh.update()


def set_auto_smooth_angle(mesh, angle_rad: float) -> None:
    """Set auto smooth angle — API changed in Blender 4.1.

    Blender < 4.1: mesh.auto_smooth_angle + mesh.use_auto_smooth = True
    Blender 4.1+:  auto-smooth is a modifier; no mesh attribute.
    """
    v = get_blender_version()
    if (v[0], v[1]) < (4, 1):
        try:
            mesh.use_auto_smooth = True
            mesh.auto_smooth_angle = angle_rad
        except AttributeError:
            pass


def link_object_to_collection(context, obj) -> None:
    """Link an object to the active collection — compatible across versions."""
    try:
        context.collection.objects.link(obj)
    except Exception:
        try:
            context.scene.collection.objects.link(obj)
        except Exception:
            pass


def deselect_all_objects(context) -> None:
    """Deselect all objects — compatible across versions."""
    try:
        import bpy
        bpy.ops.object.select_all(action='DESELECT')
    except Exception:
        for obj in context.scene.objects:
            try:
                obj.select_set(False)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def print_environment_info() -> None:
    """Print a diagnostic summary to the Blender console on load."""
    v_ok, v_msg = check_blender_version_compatible()
    n_ok, n_msg = check_numpy_available()
    s_ok, s_msg = check_scipy_available()

    pfx = "[QuadForge]"
    print(f"{pfx} {'='*44}")
    print(f"{pfx} QuadForge v13.0.0 — Environment Diagnostics")
    print(f"{pfx}   Platform : {get_platform()} / {get_arch()}")
    print(f"{pfx}   CPU cores: {get_cpu_count()}")
    print(f"{pfx}   Blender  : {v_msg}  {'OK' if v_ok else 'UNSUPPORTED'}")
    print(f"{pfx}   NumPy    : {n_msg}  {'OK' if n_ok else 'MISSING'}")
    print(f"{pfx}   SciPy    : {s_msg}")
    lib_found = native_library_exists()
    lib_path  = native_library_path()
    print(f"{pfx}   Native lib: {'FOUND — fast path active' if lib_found else 'not found (Python fallback)'}")
    print(f"{pfx}   Path: {lib_path}")
    print(f"{pfx} {'='*44}")


def get_environment_report() -> dict:
    """Return a dict summarising the runtime environment for the check operator."""
    v_ok, v_msg = check_blender_version_compatible()
    n_ok, n_msg = check_numpy_available()
    s_ok, s_msg = check_scipy_available()
    lib_found   = native_library_exists()
    return {
        "blender_ok":  v_ok,
        "blender_msg": v_msg,
        "numpy_ok":    n_ok,
        "numpy_msg":   n_msg,
        "scipy_ok":    s_ok,
        "scipy_msg":   s_msg,
        "native_lib":  lib_found,
        "native_path": native_library_path(),
        "platform":    get_platform(),
        "arch":        get_arch(),
        "cpu_count":   get_cpu_count(),
    }
