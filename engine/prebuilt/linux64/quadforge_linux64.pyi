"""
quadforge_linux64.pyi — PEP 561 type stub for loader_linux.py (linux64 package).

Provides complete type information for the four public symbols re-exported
from the linux64 __init__.py so that static analysers (mypy, Pylance, Pyright,
PyCharm) can fully type-check the Linux code path in bridge.py.

Generated from loader_linux.py — keep in sync when adding new helpers.
"""

import ctypes
from typing import Optional

# ---------------------------------------------------------------------------
# resolve_library_path
# ---------------------------------------------------------------------------

def resolve_library_path(hint_file: Optional[str] = None) -> Optional[str]:
    """
    Return the absolute path to ``libquadforge.so``, or ``None`` if not found.

    Search order:
      1. ``QUADFORGE_NATIVE_LIB`` environment variable override.
      2. Same directory as this loader module (canonical install location).
      3. ``hint_file``'s directory (caller's module location — typically
         ``bridge.py``'s parent directory).
      4. Directories listed in the ``LD_LIBRARY_PATH`` environment variable.

    Args:
        hint_file: Optional path used as a directory hint (pass ``__file__``
                   from the calling module).

    Returns:
        Absolute path string if found, ``None`` otherwise.
    """
    ...


# ---------------------------------------------------------------------------
# load_library
# ---------------------------------------------------------------------------

def load_library(path: str) -> ctypes.CDLL:
    """
    Load ``libquadforge.so`` at the given absolute *path* and return the
    :class:`ctypes.CDLL` handle.

    Side effects:
      * Pre-loads ``libgomp.so`` (GNU OpenMP runtime) with ``RTLD_GLOBAL`` so
        its thread-local storage is shared between QuadForge and any other
        OpenMP library already loaded by Blender.
      * Opens ``libquadforge.so`` with ``RTLD_GLOBAL | RTLD_LAZY``.

    Args:
        path: Absolute path to ``libquadforge.so``.

    Returns:
        Loaded :class:`ctypes.CDLL` handle.

    Raises:
        RuntimeError: If :func:`ctypes.CDLL` raises :exc:`OSError`
                      (missing dependencies, wrong architecture, …).
    """
    ...


# ---------------------------------------------------------------------------
# check_version
# ---------------------------------------------------------------------------

def check_version(lib: ctypes.CDLL, min_version: str = "1.0.0") -> bool:
    """
    Verify that the loaded *lib*'s ``qf_version()`` string is ``>= min_version``.

    Strips any platform suffix (e.g. ``"-linux64"``) before comparison.

    Args:
        lib:         A :class:`ctypes.CDLL` handle returned by :func:`load_library`.
        min_version: Minimum acceptable version string (default ``"1.0.0"``).

    Returns:
        ``True`` if the library version is compatible, ``False`` otherwise.
        Returns ``False`` on any exception (e.g. symbol missing).
    """
    ...


# ---------------------------------------------------------------------------
# system_info
# ---------------------------------------------------------------------------

def system_info() -> dict:
    """
    Return a :class:`dict` of system information useful for bug reports.

    Keys (all present, values may be ``None`` or ``False`` when unavailable):

    ==================  =====================================================
    Key                 Description
    ==================  =====================================================
    ``"platform"``      :func:`platform.platform` string
    ``"machine"``       :func:`platform.machine` string (e.g. ``"x86_64"``)
    ``"python"``        :attr:`sys.version` string
    ``"openmp"``        ``True`` if ``libgomp`` is findable and loadable
    ``"lib_path"``      Absolute path to ``libquadforge.so``, or ``None``
    ==================  =====================================================

    Returns:
        Dictionary with the keys described above.
    """
    ...
