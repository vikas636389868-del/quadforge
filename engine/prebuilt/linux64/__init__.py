"""
QuadForge prebuilt linux64 package.

Provides:
  libquadforge.so       — compiled native engine (x86-64, OpenMP)
  loader_linux.py       — Linux-specific library loader helper
  check_deps.sh         — runtime dependency checker
  build.sh              — rebuild .so from C source
  verify_api.py         — API function verification tool
  compat_check.py       — ABI struct layout compatibility checker
  VERSION               — library version string
  CHECKSUMS             — SHA-256 for binary integrity
  libquadforge_linux64.c — C source for rebuilding from scratch
"""
from .loader_linux import resolve_library_path, load_library, check_version, system_info

__all__ = ["resolve_library_path", "load_library", "check_version", "system_info"]
