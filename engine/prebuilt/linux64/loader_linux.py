"""
loader_linux.py — Linux-specific native library loader helper for QuadForge.

This module is imported by bridge.py's _find_native_library() on Linux
to handle edge cases around LD_LIBRARY_PATH, RPATH, and libgomp availability.
It is NOT imported directly by user code.

Typical use (from bridge.py):
    from .engine.prebuilt.linux64 import loader_linux
    path = loader_linux.resolve_library_path(__file__)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys


def _this_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _check_openmp() -> bool:
    """Return True if libgomp is findable on this system."""
    try:
        gomp = ctypes.util.find_library("gomp")
        if gomp:
            ctypes.CDLL(gomp)
            return True
    except OSError:
        pass
    # Common paths on Ubuntu / Debian / Fedora
    for candidate in [
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
        "/usr/lib64/libgomp.so.1",
        "/lib/x86_64-linux-gnu/libgomp.so.1",
    ]:
        if os.path.isfile(candidate):
            try:
                ctypes.CDLL(candidate)
                return True
            except OSError:
                pass
    return False


def resolve_library_path(hint_file: str | None = None) -> str | None:
    """
    Return the absolute path to libquadforge.so, or None if not found.

    Search order:
      1. Same directory as this file (canonical install location)
      2. QUADFORGE_NATIVE_LIB env var override
      3. LD_LIBRARY_PATH directories
    """
    lib_name = "libquadforge.so"

    # 1. Env-var override (highest priority)
    env_path = os.environ.get("QUADFORGE_NATIVE_LIB", "")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Same directory as this loader
    local = os.path.join(_this_dir(), lib_name)
    if os.path.isfile(local):
        return local

    # 3. hint_file directory (caller's module location)
    if hint_file:
        candidate = os.path.join(os.path.dirname(os.path.abspath(hint_file)), lib_name)
        if os.path.isfile(candidate):
            return candidate

    # 4. LD_LIBRARY_PATH
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        candidate = os.path.join(d, lib_name)
        if os.path.isfile(candidate):
            return candidate

    return None


def load_library(path: str) -> ctypes.CDLL:
    """
    Load libquadforge.so with proper flags for Blender's embedded Python.

    RTLD_GLOBAL is required so that libgomp's thread-local state is shared
    between QuadForge and any other OpenMP library Blender may have loaded.
    """
    import ctypes.util

    # Pre-load libgomp so its symbols are available before libquadforge loads.
    if not _check_openmp():
        print("[QuadForge] WARNING: libgomp not found — OpenMP parallelism disabled.")
    else:
        try:
            gomp = ctypes.util.find_library("gomp")
            if gomp:
                ctypes.CDLL(gomp, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    try:
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError(
            f"[QuadForge] Failed to load native library at {path!r}: {exc}\n"
            "Falling back to pure-Python engine."
        ) from exc

    return lib


def check_version(lib: ctypes.CDLL, min_version: str = "1.0.0") -> bool:
    """
    Verify the loaded library's version is >= min_version.
    Returns True if compatible, False otherwise.
    """
    try:
        lib.qf_version.restype = ctypes.c_char_p
        raw = lib.qf_version()
        ver_str = raw.decode("utf-8", errors="replace").split("-")[0]  # strip "-linux64"
        parts = [int(x) for x in ver_str.split(".")]
        min_parts = [int(x) for x in min_version.split(".")]
        return parts >= min_parts
    except Exception:
        return False


def system_info() -> dict:
    """Return a dict of system information useful for bug reports."""
    return {
        "platform": platform.platform(),
        "machine":  platform.machine(),
        "python":   sys.version,
        "openmp":   _check_openmp(),
        "lib_path": resolve_library_path(),
    }
