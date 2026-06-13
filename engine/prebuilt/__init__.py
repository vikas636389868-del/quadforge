"""
QuadForge prebuilt native libraries — platform dispatch package.

This package contains platform-specific compiled shared libraries and
their associated helper scripts for loading, verifying, and building
the QuadForge C++ engine.

Structure
---------
prebuilt/
    linux64/        libquadforge.so   (x86-64 Linux, glibc 2.31+)
    win64/          quadforge.dll     (x86-64 Windows 10/11, MSVC runtime)
    macos_arm64/    libquadforge.dylib (Apple Silicon M1/M2/M3/M4)
    macos_x64/      libquadforge.dylib (Intel Mac x86_64)
    quadforge_stub.py  C API documentation / pure-Python shim reference

Usage
-----
bridge.py resolves the platform library via os.path.isfile() — it does
NOT import this package directly.  The per-platform loader modules
(loader_linux.py, loader_macos.py, loader_win64.py) are helpers for
advanced use: direct loading with platform-specific flags, version
checks, and system diagnostics.

If no compiled library is found, bridge.py automatically falls back to
the pure-Python engine in engine/pipeline.py.
"""

import platform as _platform

def get_platform_dir() -> str:
    """Return the name of the platform-specific subdirectory for the current host."""
    system = _platform.system()
    if system == "Windows":
        return "win64"
    elif system == "Darwin":
        return "macos_arm64" if _platform.machine() == "arm64" else "macos_x64"
    else:
        return "linux64"


def get_library_name() -> str:
    """Return the shared-library filename for the current platform."""
    system = _platform.system()
    if system == "Windows":
        return "quadforge.dll"
    elif system == "Darwin":
        return "libquadforge.dylib"
    else:
        return "libquadforge.so"


__all__ = ["get_platform_dir", "get_library_name"]
