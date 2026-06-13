#!/usr/bin/env python3
"""
install_deps.py — Linux runtime dependency checker for QuadForge.

Checks that all shared libraries required by libquadforge.so are
present and loadable on the current system.  Provides actionable
install instructions when a dependency is missing.

Usage:
    python3 QuadForge/engine/prebuilt/linux64/install_deps.py
    python3 QuadForge/engine/prebuilt/linux64/install_deps.py --fix

With --fix the script will attempt to install missing packages via
apt-get, dnf, or zypper (whichever is available), requiring sudo.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH   = os.path.join(SCRIPT_DIR, "libquadforge.so")

# ---------------------------------------------------------------------------
# Dependency table
# ---------------------------------------------------------------------------
# Each entry: (display_name, so_name_hint, apt_pkg, dnf_pkg, zypper_pkg, optional)
DEPS = [
    (
        "GNU C Library (glibc)",
        "c",
        "libc6",
        "glibc",
        "glibc",
        False,
    ),
    (
        "GNU C++ Standard Library (libstdc++)",
        "stdc++",
        "libstdc++6",
        "libstdc++",
        "libstdc++6",
        False,
    ),
    (
        "GCC OpenMP runtime (libgomp) — enables multi-threading",
        "gomp",
        "libgomp1",
        "libgomp",
        "libgomp1",
        True,   # Optional: single-threaded fallback if missing
    ),
    (
        "GNU math library (libm)",
        "m",
        "libc6",          # libm is part of glibc on modern distros
        "glibc",
        "glibc",
        False,
    ),
    (
        "POSIX threads (libpthread)",
        "pthread",
        "libc6",          # libpthread merged into glibc 2.34+
        "glibc",
        "glibc",
        False,
    ),
]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _find_so(hint: str) -> str | None:
    """Return the full path to a shared library by its hint name, or None.

    Strategy:
      1. Try ctypes.util.find_library (needs ldconfig; may fail in containers).
      2. Probe common filesystem paths.
      3. For glibc builtins (c, m, pthread) that are always present when Python
         is running, attempt to load the well-known versioned soname directly.
         On glibc 2.34+ libm and libpthread are merged into libc.so.6.
    """
    # Known versioned sonames for glibc built-ins — always present if Python runs
    ALWAYS_PRESENT = {
        "c":       ["/lib/x86_64-linux-gnu/libc.so.6", "/lib64/libc.so.6", "/usr/lib64/libc.so.6"],
        "m":       ["/lib/x86_64-linux-gnu/libm.so.6", "/lib64/libm.so.6",
                    "/lib/x86_64-linux-gnu/libc.so.6"],   # merged in glibc 2.34+
        "pthread": ["/lib/x86_64-linux-gnu/libpthread.so.0", "/lib64/libpthread.so.0",
                    "/lib/x86_64-linux-gnu/libc.so.6"],   # merged in glibc 2.34+
        "stdc++":  ["/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
                    "/usr/lib64/libstdc++.so.6",
                    "/lib/x86_64-linux-gnu/libstdc++.so.6"],
    }
    if hint in ALWAYS_PRESENT:
        for candidate in ALWAYS_PRESENT[hint]:
            if os.path.isfile(candidate):
                return candidate
        # glibc is always present if we got here — Python itself requires it.
        # Return a sentinel so the caller marks it found.
        return f"<built-in: lib{hint} merged into libc.so.6>"

    path = ctypes.util.find_library(hint)
    if path and os.path.isfile(path):
        return path

    # Common fallback paths on Ubuntu / Debian / Fedora / RHEL
    for candidate in [
        f"/usr/lib/x86_64-linux-gnu/lib{hint}.so.1",
        f"/usr/lib64/lib{hint}.so.1",
        f"/lib/x86_64-linux-gnu/lib{hint}.so.1",
        f"/lib64/lib{hint}.so.1",
        f"/usr/lib/lib{hint}.so.1",
        f"/usr/lib/x86_64-linux-gnu/lib{hint}.so",
        f"/usr/lib64/lib{hint}.so",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _try_load(path: str | None) -> bool:
    """Return True if the given path can be dlopen'd, or is a known built-in sentinel."""
    if not path:
        return False
    if path.startswith("<built-in:"):
        return True   # glibc built-ins: always present if Python is running
    try:
        ctypes.CDLL(path)
        return True
    except OSError:
        return False


def _detect_pkg_manager() -> str | None:
    for pm in ("apt-get", "dnf", "yum", "zypper", "pacman"):
        if shutil.which(pm):
            return pm
    return None


def _install_pkg(pkg_manager: str, pkg_name: str) -> bool:
    """Attempt to install a package using the detected package manager."""
    cmd_map = {
        "apt-get": ["sudo", "apt-get", "install", "-y", pkg_name],
        "dnf":     ["sudo", "dnf",     "install", "-y", pkg_name],
        "yum":     ["sudo", "yum",     "install", "-y", pkg_name],
        "zypper":  ["sudo", "zypper",  "install", "-y", pkg_name],
        "pacman":  ["sudo", "pacman",  "-S",  "--noconfirm", pkg_name],
    }
    cmd = cmd_map.get(pkg_manager)
    if not cmd:
        return False
    try:
        result = subprocess.run(cmd, check=True, timeout=120)
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def check_deps(fix: bool = False) -> int:
    """Check all dependencies. Returns 0 if all required deps present, 1 otherwise."""
    print("=" * 60)
    print("  QuadForge linux64 — Dependency Check")
    print("=" * 60)

    if not os.path.isfile(LIB_PATH):
        print(f"\n  ⚠  libquadforge.so not found at:\n     {LIB_PATH}")
        print("  Build it first:  bash build.sh\n")
        return 1

    pkg_manager = _detect_pkg_manager()
    errors   = 0
    warnings = 0

    print(f"\n  Library: {LIB_PATH}")
    print(f"  Package manager: {pkg_manager or 'not detected'}\n")

    # Determine distro-specific package name column
    pm_col = {"apt-get": 3, "dnf": 4, "yum": 4, "zypper": 5, "pacman": 5}.get(
        pkg_manager or "", 3
    )

    for display, hint, apt_pkg, dnf_pkg, zyp_pkg, optional in DEPS:
        pkg_name = [apt_pkg, dnf_pkg, dnf_pkg, zyp_pkg, zyp_pkg][pm_col - 3]
        path = _find_so(hint)
        ok   = _try_load(path)

        if ok:
            print(f"  ✓  {display}")
            print(f"        {path}")
        else:
            tag = "⚠  (optional)" if optional else "✗  (required)"
            print(f"\n  {tag}  {display}")
            print(f"        not found — install with: ", end="")
            if pkg_manager:
                print(f"sudo {pkg_manager} install {pkg_name}")
            else:
                print(f"apt-get install {apt_pkg}  |  dnf install {dnf_pkg}")

            if fix and pkg_manager and not optional:
                print(f"        → Attempting install…")
                if _install_pkg(pkg_manager, pkg_name):
                    print(f"        ✓ Installed successfully.")
                else:
                    print(f"        ✗ Install failed — please install manually.")
                    errors += 1
            elif optional:
                warnings += 1
                print()
            else:
                errors += 1
                print()

    # Try loading the actual library last
    print("\n  --- Loading libquadforge.so ---")
    try:
        lib = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)
        lib.qf_version.restype = ctypes.c_char_p
        ver = lib.qf_version().decode("utf-8", errors="replace")
        print(f"  ✓  Loaded OK — version {ver}")
        lib.qf_shutdown.restype = None
        lib.qf_shutdown()
    except OSError as exc:
        print(f"  ✗  Load failed: {exc}")
        errors += 1

    # Summary
    print(f"\n{'=' * 60}")
    if errors == 0 and warnings == 0:
        print("  All dependencies satisfied — QuadForge ready.")
    elif errors == 0:
        print(f"  {warnings} optional dependency missing (OpenMP disabled — single-threaded mode).")
    else:
        print(f"  {errors} required dependency missing — QuadForge will use Python fallback.")
    print("=" * 60)
    return 0 if errors == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check (and optionally install) QuadForge linux64 runtime dependencies."
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Attempt to install missing required packages via the system package manager (requires sudo)."
    )
    args = parser.parse_args()
    sys.exit(check_deps(fix=args.fix))


if __name__ == "__main__":
    main()
