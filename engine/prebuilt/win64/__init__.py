"""
QuadForge prebuilt win64 package.

Provides:
  quadforge.dll         — compiled native engine (x86-64, OpenMP via MSVC/vcomp)
  loader_win64.py       — Windows-specific library loader helper
  check_deps.ps1        — PowerShell runtime dependency checker
  build.bat             — rebuild DLL from C source (MSVC cl.exe)
  verify_api.py         — API function verification tool
  compat_check.py       — ABI struct layout compatibility checker
  VERSION               — library version string
  CHECKSUMS             — SHA-256 for binary integrity
  quadforge_win64.c     — C source for rebuilding from scratch
  MISSING_BINARY.md     — instructions if quadforge.dll is absent
"""
from .loader_win64 import resolve_library_path, load_library, check_version, system_info

__all__ = ["resolve_library_path", "load_library", "check_version", "system_info"]
