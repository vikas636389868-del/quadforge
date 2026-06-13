"""
quadforge_macos_arm64.pyi — PEP 561 type stub for loader_macos.py (macos_arm64 package).

Provides complete type information for the four public symbols re-exported
from the macos_arm64 __init__.py so that static analysers (mypy, Pylance,
Pyright, PyCharm) can fully type-check the macOS ARM64 code path in bridge.py.

Generated from loader_macos.py — keep in sync when adding new helpers.
"""

import ctypes
from typing import Optional

# ---------------------------------------------------------------------------
# resolve_library_path
# ---------------------------------------------------------------------------

def resolve_library_path(hint_file: Optional[str] = None) -> Optional[str]:
    """
    Return the absolute path to ``libquadforge.dylib``, or ``None`` if not found.

    Search order:
      1. ``QUADFORGE_NATIVE_LIB`` environment variable override.
      2. Same directory as this loader module (canonical install location).
      3. ``hint_file``'s directory.
      4. ``DYLD_LIBRARY_PATH`` directories.

    Args:
        hint_file: Optional path used as a directory hint (pass ``__file__``).

    Returns:
        Absolute path string if found, ``None`` otherwise.
    """
    ...


# ---------------------------------------------------------------------------
# load_library
# ---------------------------------------------------------------------------

def load_library(path: str) -> ctypes.CDLL:
    """
    Load ``libquadforge.dylib`` at the given absolute *path*.

    Side effects:
      * Pre-loads ``libomp.dylib`` from Homebrew (``/opt/homebrew``) if
        available, so OpenMP symbols are ready before the dylib is mapped.

    Args:
        path: Absolute path to ``libquadforge.dylib``.

    Returns:
        Loaded :class:`ctypes.CDLL` handle.

    Raises:
        RuntimeError: If loading fails (missing dylib, wrong arch, …).
    """
    ...


# ---------------------------------------------------------------------------
# check_version
# ---------------------------------------------------------------------------

def check_version(lib: ctypes.CDLL, min_version: str = "1.0.0") -> bool:
    """
    Verify ``lib.qf_version()`` is ``>= min_version``.

    Args:
        lib:         :class:`ctypes.CDLL` from :func:`load_library`.
        min_version: Minimum version string (default ``"1.0.0"``).

    Returns:
        ``True`` if compatible, ``False`` on mismatch or any exception.
    """
    ...


# ---------------------------------------------------------------------------
# system_info
# ---------------------------------------------------------------------------

def system_info() -> dict:
    """
    Return system information for bug reports.

    Keys: ``"platform"``, ``"machine"``, ``"python"``, ``"openmp"``,
    ``"lib_path"``, ``"blender_version"``.

    Returns:
        Dictionary with the keys described above.
    """
    ...
