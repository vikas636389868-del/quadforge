"""
quadforge_win64.pyi — PEP 561 type stub for loader_win64.py (win64 package).

Provides complete type information for the four public symbols re-exported
from the win64 __init__.py so that static analysers (mypy, Pylance, Pyright,
PyCharm) can fully type-check the Windows code path in bridge.py.

Generated from loader_win64.py — keep in sync when adding new helpers.
"""

import ctypes
from typing import Optional

# ---------------------------------------------------------------------------
# resolve_library_path
# ---------------------------------------------------------------------------

def resolve_library_path(hint_file: Optional[str] = None) -> Optional[str]:
    """
    Return the absolute path to ``quadforge.dll``, or ``None`` if not found.

    Search order:
      1. ``QUADFORGE_NATIVE_LIB`` environment variable override.
      2. Same directory as this loader module (canonical install location).
      3. ``hint_file``'s directory (caller's module location — typically
         ``bridge.py``'s parent directory).
      4. Directories listed in the ``PATH`` environment variable.

    Args:
        hint_file: Optional path to use as a directory hint (pass ``__file__``
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
    Load ``quadforge.dll`` at the given absolute *path* and return the
    :class:`ctypes.CDLL` handle.

    Side effects:
      * Pre-loads ``vcomp140.dll`` / ``libomp.dll`` (OpenMP runtime) so its
        symbols are available when ``quadforge.dll`` is mapped.
      * Calls :func:`os.add_dll_directory` (Python 3.8+) to add the DLL's
        directory to the DLL search path.

    Args:
        path: Absolute path to ``quadforge.dll``.

    Returns:
        Loaded :class:`ctypes.CDLL` handle.

    Raises:
        RuntimeError: If :func:`ctypes.CDLL` raises :exc:`OSError`
                      (dependency missing, wrong architecture, antivirus, …).
    """
    ...


# ---------------------------------------------------------------------------
# check_version
# ---------------------------------------------------------------------------

def check_version(lib: ctypes.CDLL, min_version: str = "1.0.0") -> bool:
    """
    Verify that the loaded *lib*'s ``qf_version()`` string is ``>= min_version``.

    Calls ``lib.qf_version()`` and compares the dotted-integer prefix of the
    returned string against *min_version* component-by-component.

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

    Keys (all present, values may be ``None`` or ``"N/A"`` when unavailable):

    =====================  =====================================================
    Key                    Description
    =====================  =====================================================
    ``"platform"``         :func:`platform.platform` string
    ``"machine"``          :func:`platform.machine` string (e.g. ``"AMD64"``)
    ``"python"``           :attr:`sys.version` string
    ``"openmp"``           ``True`` if vcomp140.dll or libomp.dll is found
    ``"vcruntime"``        ``True`` if VCRUNTIME140.dll is found
    ``"lib_path"``         Absolute path to quadforge.dll, or ``None``
    ``"vcomp_path"``       Absolute path to OpenMP DLL, or ``None``
    ``"windows_version"``  :func:`platform.version` string
    ``"windows_release"``  :func:`platform.release` string
    ``"in_venv"``          ``True`` if running inside a venv or conda env
    ``"blender_version"``  Blender version string, or ``"N/A"``
    =====================  =====================================================

    Returns:
        Dictionary with the keys described above.
    """
    ...
