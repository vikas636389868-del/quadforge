"""
loader_macos.py — macOS ARM64-specific native library loader helper for QuadForge.

This module is imported by bridge.py's _find_native_library() on macOS ARM64
to handle edge cases around DYLD_LIBRARY_PATH, @rpath, and libomp availability.
It is NOT imported directly by user code.

Typical use (from bridge.py):
    from .engine.prebuilt.macos_arm64 import loader_macos
    path = loader_macos.resolve_library_path(__file__)
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


def _check_openmp() -> bool:
    """Return True if libomp is findable on this system."""
    # macOS uses libomp (from LLVM / Homebrew) instead of libgomp
    try:
        omp = ctypes.util.find_library("omp")
        if omp:
            ctypes.CDLL(omp)
            return True
    except OSError:
        pass
    # Common Homebrew paths on Apple Silicon
    for candidate in [
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/opt/homebrew/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
        "/usr/local/lib/libomp.dylib",
    ]:
        if os.path.isfile(candidate):
            try:
                ctypes.CDLL(candidate)
                return True
            except OSError:
                pass
    return False


def _find_libomp() -> str | None:
    """Return the path to libomp.dylib if available."""
    omp = ctypes.util.find_library("omp")
    if omp:
        return omp
    for candidate in [
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/opt/homebrew/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
        "/usr/local/lib/libomp.dylib",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_library_path(hint_file: str | None = None) -> str | None:
    """
    Return the absolute path to libquadforge.dylib, or None if not found.

    Search order:
      1. QUADFORGE_NATIVE_LIB env var override (highest priority)
      2. Same directory as this file (canonical install location)
      3. hint_file directory (caller's module location)
      4. DYLD_LIBRARY_PATH directories
      5. DYLD_FALLBACK_LIBRARY_PATH directories
    """
    lib_name = "libquadforge.dylib"

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

    # 4. DYLD_LIBRARY_PATH
    for d in os.environ.get("DYLD_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        candidate = os.path.join(d, lib_name)
        if os.path.isfile(candidate):
            return candidate

    # 5. DYLD_FALLBACK_LIBRARY_PATH
    for d in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        candidate = os.path.join(d, lib_name)
        if os.path.isfile(candidate):
            return candidate

    return None


def load_library(path: str) -> ctypes.CDLL:
    """
    Load libquadforge.dylib with proper flags for Blender's embedded Python.

    On macOS ARM64, the library uses libomp for OpenMP parallelism.
    We pre-load libomp so its symbols are available before libquadforge loads.
    """
    # Pre-load libomp so its symbols are available before libquadforge loads.
    if not _check_openmp():
        print("[QuadForge] WARNING: libomp not found — OpenMP parallelism disabled.")
        print("[QuadForge]          Install with: brew install libomp")
    else:
        omp_path = _find_libomp()
        if omp_path:
            try:
                ctypes.CDLL(omp_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    try:
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError(
            f"[QuadForge] Failed to load native library at {path!r}: {exc}\n"
            "Falling back to pure-Python engine."
        ) from exc

    # Verify architecture matches — prevent loading x86_64 library on ARM64
    machine = platform.machine()
    if machine != "arm64":
        print(f"[QuadForge] WARNING: Running on {machine}, expected arm64.")
        if machine == "x86_64":
            # Either an Intel Mac, or an x64 Python running under Rosetta 2
            # on Apple Silicon. In either case, this arm64 dylib won't load.
            if _is_running_under_rosetta():
                print("[QuadForge]          Your Python is running under Rosetta 2 on "
                      "Apple Silicon.")
                print("[QuadForge]          Install a native arm64 Python (or use "
                      "Blender 3.5+ for Apple Silicon).")
            else:
                print("[QuadForge]          This appears to be an Intel Mac — use the "
                      "macos_x64 build instead.")
        else:
            print("[QuadForge]          Library may not function correctly.")

    return lib


def _is_running_under_rosetta() -> bool:
    """Return True if the current Python process is running under Rosetta 2."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "1"
    except Exception:
        return False


def check_version(lib: ctypes.CDLL, min_version: str = "1.0.0") -> bool:
    """
    Verify the loaded library's version is >= min_version.
    Returns True if compatible, False otherwise.
    """
    try:
        lib.qf_version.restype = ctypes.c_char_p
        raw = lib.qf_version()
        ver_str = raw.decode("utf-8", errors="replace").split("-")[0]  # strip "-macos_arm64"
        parts = [int(x) for x in ver_str.split(".")]
        min_parts = [int(x) for x in min_version.split(".")]
        return parts >= min_parts
    except Exception:
        return False


def system_info() -> dict:
    """Return a dict of system information useful for bug reports."""
    info = {
        "platform": platform.platform(),
        "machine":  platform.machine(),
        "python":   sys.version,
        "openmp":   _check_openmp(),
        "lib_path": resolve_library_path(),
    }

    # macOS-specific: get OS version
    try:
        info["macos_version"] = platform.mac_ver()[0]
    except Exception:
        info["macos_version"] = "unknown"

    # Check Xcode Command Line Tools
    try:
        result = subprocess.run(
            ["xcode-select", "-p"], capture_output=True, text=True, timeout=5
        )
        info["xcode_path"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        info["xcode_path"] = None

    # Rosetta 2: detects an x64 Python running on Apple Silicon. When True on
    # an arm64 host, the arm64 dylib will NOT load — the user needs a native
    # arm64 Python (or must use the macos_x64 build via the x64 code path).
    info["rosetta2"] = _is_running_under_rosetta()

    return info
