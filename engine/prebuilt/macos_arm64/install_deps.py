#!/usr/bin/env python3
"""
install_deps.py — macOS ARM64 runtime dependency checker for QuadForge.

Checks that all shared libraries required by libquadforge.dylib are
present and loadable on an Apple Silicon (M1/M2/M3/M4) system.

Usage:
    python3 QuadForge/engine/prebuilt/macos_arm64/install_deps.py
    python3 QuadForge/engine/prebuilt/macos_arm64/install_deps.py --fix

With --fix the script will attempt to install missing packages via
Homebrew (brew install).  Xcode Command Line Tools are checked but
cannot be auto-installed (requires user interaction).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH   = os.path.join(SCRIPT_DIR, "libquadforge.dylib")

# Homebrew root for Apple Silicon is /opt/homebrew (Intel: /usr/local)
BREW_PREFIX = "/opt/homebrew" if platform.machine() == "arm64" else "/usr/local"


# ---------------------------------------------------------------------------
# Dependency definitions
# ---------------------------------------------------------------------------
# (display_name, candidate_paths, brew_formula, optional)
DEPS = [
    (
        "Xcode Command Line Tools (clang runtime)",
        [
            "/Library/Developer/CommandLineTools/usr/lib/libclang.dylib",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/"
            "XcodeDefault.xctoolchain/usr/lib/libclang.dylib",
        ],
        None,   # installed via: xcode-select --install
        False,
    ),
    (
        "libc++ (LLVM C++ Standard Library)",
        ["/usr/lib/libc++.1.dylib"],
        None,   # ships with macOS
        False,
    ),
    (
        "OpenMP runtime (libomp) — enables multi-threading",
        [
            os.path.join(BREW_PREFIX, "opt/libomp/lib/libomp.dylib"),
            "/usr/local/opt/libomp/lib/libomp.dylib",   # fallback Intel prefix
            "/opt/homebrew/lib/libomp.dylib",
            os.path.join(SCRIPT_DIR, "libomp.dylib"),   # bundled copy
        ],
        "libomp",
        True,   # Optional: engine falls back to single-threaded if missing
    ),
]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _try_load(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        ctypes.CDLL(path)
        return True
    except OSError:
        return False


def _find_dep(candidates: list[str]) -> str | None:
    for c in candidates:
        if _try_load(c):
            return c
    return None


def _brew_available() -> bool:
    return shutil.which("brew") is not None


def _brew_install(formula: str) -> bool:
    try:
        result = subprocess.run(
            ["brew", "install", formula],
            check=True, timeout=300,
            capture_output=False,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_deps(fix: bool = False) -> int:
    print("=" * 60)
    print("  QuadForge macos_arm64 — Dependency Check")
    print("=" * 60)

    if platform.machine() != "arm64":
        print(f"\n  ⚠  This check is for Apple Silicon (arm64).")
        print(f"     Detected machine: {platform.machine()}")
        print(f"     Use macos_x64/install_deps.py for Intel Macs.\n")

    if not os.path.isfile(LIB_PATH):
        print(f"\n  ⚠  libquadforge.dylib not found at:\n     {LIB_PATH}")
        print("  Build it first:  bash build.sh\n")
        return 1

    brew = _brew_available()
    errors   = 0
    warnings = 0

    print(f"\n  Library: {LIB_PATH}")
    print(f"  Homebrew: {'available' if brew else 'not found'}\n")

    for display, candidates, brew_formula, optional in DEPS:
        found = _find_dep(candidates)
        if found:
            print(f"  ✓  {display}")
            print(f"        {found}")
        else:
            tag = "⚠  (optional)" if optional else "✗  (required)"
            print(f"\n  {tag}  {display}")
            if brew_formula:
                print(f"        install with: brew install {brew_formula}")
                if fix and brew:
                    print("        → Attempting brew install…")
                    if _brew_install(brew_formula):
                        print("        ✓ Installed.")
                    else:
                        print("        ✗ Install failed — run manually: brew install", brew_formula)
                        if not optional:
                            errors += 1
                    print()
                    continue
            else:
                print("        install with: xcode-select --install")
            if optional:
                warnings += 1
            else:
                errors += 1
            print()

    # Attempt to load the dylib itself
    print("\n  --- Loading libquadforge.dylib ---")
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.qf_version.restype = ctypes.c_char_p
        ver = lib.qf_version().decode("utf-8", errors="replace")
        print(f"  ✓  Loaded OK — version {ver}")
        lib.qf_shutdown.restype = None
        lib.qf_shutdown()
    except OSError as exc:
        print(f"  ✗  Load failed: {exc}")
        print("     Run: bash check_deps.sh  for detailed symbol diagnostics")
        errors += 1

    # Summary
    print(f"\n{'=' * 60}")
    if errors == 0 and warnings == 0:
        print("  All dependencies satisfied — QuadForge ready.")
    elif errors == 0:
        print(f"  {warnings} optional dep missing — OpenMP disabled (single-threaded).")
        print("  Fix: brew install libomp")
    else:
        print(f"  {errors} required dependency missing.")
    print("=" * 60)
    return 0 if errors == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check (and optionally install) QuadForge macos_arm64 dependencies."
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Attempt to install missing packages via Homebrew."
    )
    args = parser.parse_args()
    sys.exit(check_deps(fix=args.fix))


if __name__ == "__main__":
    main()
