"""
QuadForge prebuilt macos_arm64 package.

Provides:
  libquadforge.dylib       — compiled native engine (Apple Silicon, ARM64)
  loader_macos.py          — macOS-specific library loader helper
  check_deps.sh            — runtime dependency checker
  build.sh                 — rebuild .dylib from C source
  verify_api.py            — API function verification tool
  compat_check.py          — ABI struct layout compatibility checker
  VERSION                  — library version string
  CHECKSUMS                — SHA-256 for binary integrity
  libquadforge_macos_arm64.c — C source for rebuilding from scratch
"""
from .loader_macos import resolve_library_path, load_library, check_version, system_info

__all__ = ["resolve_library_path", "load_library", "check_version", "system_info"]
