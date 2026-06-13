"""
loader_win64.py — Windows-specific native library loader helper for QuadForge.

This module is imported by bridge.py's _find_native_library() on Windows to
handle edge cases around DLL search order, Visual C++ runtime availability,
and OpenMP (vcomp / libomp) detection.

It is NOT imported directly by user code.

Typical use (from bridge.py):
    from .engine.prebuilt.win64 import loader_win64
    path = loader_win64.resolve_library_path(__file__)

Windows DLL loading notes
--------------------------
ctypes.CDLL on Windows uses LoadLibraryW under the hood.  The DLL search order
(simplified, Safe DLL Search Mode ON — the Windows default) is:

  1. The directory that contains the application executable (.exe)
  2. The System32 directory
  3. The 16-bit system directory
  4. The Windows directory
  5. The current working directory
  6. Directories listed in the PATH environment variable

Blender's embedded Python inherits Blender's CWD and PATH, which may NOT
include the add-on's prebuilt/win64 directory.  We therefore use an absolute
path from resolve_library_path() to bypass the search entirely.

Additionally, Windows requires that every DLL listed in quadforge.dll's import
table already be loaded (or findable on PATH) at the time LoadLibraryW is
called.  The main runtime dependencies are:

  - VCRUNTIME140.dll   (Visual C++ 2015–2022 Redistributable — almost always present)
  - MSVCP140.dll       (C++ STL — same redistributable)
  - vcomp140.dll       (OpenMP runtime from Visual C++ Redistributable)
    *OR* libomp.dll    (LLVM/OpenMP runtime if built with Clang-cl)

If quadforge.dll was built with the /MT flag (static runtime linkage), the
VCRUNTIME / MSVCP requirement is eliminated.  The prebuilt DLL uses /MT so
only the OpenMP runtime needs to be present.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import subprocess
import sys


def _this_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# OpenMP / runtime detection
# ---------------------------------------------------------------------------

def _find_vcomp() -> str | None:
    """
    Return the path to the OpenMP DLL (vcomp140.dll or libomp.dll) if available.

    Blender ships its own Python but does not ship vcomp*.dll.  We check the
    System32 directory and the Visual C++ Redistributable install paths.
    """
    # MSVC OpenMP runtime candidates (Visual C++ 2015–2022 Redist)
    candidates = [
        # vcomp140 ships in both x64 and x86 Redistributable packages
        os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "vcomp140.dll"),
        os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "SysWOW64", "vcomp140.dll"),
        # LLVM libomp.dll — present if built with clang-cl + libomp
        os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "libomp.dll"),
    ]

    # Also probe PATH entries (covers conda environments, etc.)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        candidates.append(os.path.join(d, "vcomp140.dll"))
        candidates.append(os.path.join(d, "libomp.dll"))

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _check_openmp() -> bool:
    """Return True if an OpenMP runtime DLL can be found on this system."""
    return _find_vcomp() is not None


def _check_vcruntime() -> bool:
    """Return True if the Visual C++ runtime (VCRUNTIME140.dll) is findable."""
    sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in ("VCRUNTIME140.dll", "vcruntime140.dll"):
        if os.path.isfile(os.path.join(sys32, name)):
            return True
    # Also check PATH
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isfile(os.path.join(d, "VCRUNTIME140.dll")):
            return True
    return False


# ---------------------------------------------------------------------------
# Library path resolution
# ---------------------------------------------------------------------------

def resolve_library_path(hint_file: str | None = None) -> str | None:
    """
    Return the absolute path to quadforge.dll, or None if not found.

    Search order:
      1. QUADFORGE_NATIVE_LIB environment variable override (highest priority)
      2. Same directory as this file (canonical install location)
      3. hint_file's directory (caller's module location)
      4. Directories listed in PATH
    """
    lib_name = "quadforge.dll"

    # 1. Env-var override (highest priority)
    env_path = os.environ.get("QUADFORGE_NATIVE_LIB", "")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Same directory as this loader (canonical install)
    local = os.path.join(_this_dir(), lib_name)
    if os.path.isfile(local):
        return local

    # 3. hint_file directory (bridge.py's location → engine/prebuilt/win64/)
    if hint_file:
        candidate = os.path.join(os.path.dirname(os.path.abspath(hint_file)), lib_name)
        if os.path.isfile(candidate):
            return candidate

    # 4. PATH directories
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        candidate = os.path.join(d, lib_name)
        if os.path.isfile(candidate):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def load_library(path: str) -> ctypes.CDLL:
    """
    Load quadforge.dll at the given absolute path.

    Steps:
      1. Pre-load the OpenMP runtime DLL (vcomp140.dll / libomp.dll) so its
         symbols are available when quadforge.dll is mapped.
      2. Call ctypes.CDLL with the absolute path.

    On failure, raises RuntimeError (bridge.py catches this and falls back to
    the pure-Python engine).
    """
    # Warn about missing runtime — but attempt the load anyway; if the DLL
    # was built with /MT the runtime is not needed as a separate file.
    if not _check_vcruntime():
        print(
            "[QuadForge] WARNING: VCRUNTIME140.dll not found.\n"
            "            Install the Visual C++ Redistributable (2015–2022):\n"
            "            https://aka.ms/vs/17/release/vc_redist.x64.exe"
        )

    # Pre-load OpenMP runtime so its symbols are available.
    vcomp = _find_vcomp()
    if vcomp is None:
        print(
            "[QuadForge] WARNING: vcomp140.dll / libomp.dll not found.\n"
            "            OpenMP parallelism will be disabled (single-threaded).\n"
            "            Install the Visual C++ Redistributable to enable it:\n"
            "            https://aka.ms/vs/17/release/vc_redist.x64.exe"
        )
    else:
        try:
            ctypes.CDLL(vcomp)
        except OSError:
            pass  # Non-fatal — quadforge.dll may still load without OpenMP

    # Use os.add_dll_directory (Python 3.8+) to add the DLL's directory to the
    # DLL search path.  This is required on Python 3.8+ where the CWD is no
    # longer implicitly on the DLL search path for security reasons.
    dll_dir = os.path.dirname(os.path.abspath(path))
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(dll_dir)
        except OSError:
            pass

    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        raise RuntimeError(
            f"[QuadForge] Failed to load native library at {path!r}: {exc}\n"
            "Falling back to pure-Python engine.\n"
            "If this error persists, check:\n"
            "  1. quadforge.dll is present in engine/prebuilt/win64/\n"
            "  2. The Visual C++ Redistributable (2015-2022) is installed\n"
            "  3. No antivirus is blocking the DLL load"
        ) from exc

    return lib


# ---------------------------------------------------------------------------
# Version verification
# ---------------------------------------------------------------------------

def check_version(lib: ctypes.CDLL, min_version: str = "1.0.0") -> bool:
    """
    Verify the loaded library's version is >= min_version.
    Returns True if compatible, False otherwise.
    """
    try:
        lib.qf_version.restype = ctypes.c_char_p
        raw = lib.qf_version()
        ver_str = raw.decode("utf-8", errors="replace").split("-")[0]  # strip "-win64"
        parts = [int(x) for x in ver_str.split(".")]
        min_parts = [int(x) for x in min_version.split(".")]
        return parts >= min_parts
    except Exception:
        return False


# ---------------------------------------------------------------------------
# System information (for bug reports)
# ---------------------------------------------------------------------------

def system_info() -> dict:
    """Return a dict of system information useful for bug reports."""
    info: dict = {
        "platform":    platform.platform(),
        "machine":     platform.machine(),
        "python":      sys.version,
        "openmp":      _check_openmp(),
        "vcruntime":   _check_vcruntime(),
        "lib_path":    resolve_library_path(__file__),   # pass __file__ so hint-dir search runs
        "vcomp_path":  _find_vcomp(),
    }

    # Windows version
    try:
        info["windows_version"] = platform.version()
        info["windows_release"] = platform.release()
    except Exception:
        info["windows_version"] = "unknown"

    # Check if running under a virtual environment (conda / venv)
    info["in_venv"] = (
        os.environ.get("VIRTUAL_ENV") is not None
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
    )

    # Detect Blender's embedded Python
    try:
        import bpy  # type: ignore
        info["blender_version"] = ".".join(str(x) for x in bpy.app.version)
    except ImportError:
        info["blender_version"] = "N/A (not running inside Blender)"

    return info
