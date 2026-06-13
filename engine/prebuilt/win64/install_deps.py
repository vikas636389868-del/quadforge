#!/usr/bin/env python3
"""
install_deps.py — Python-based runtime dependency checker for quadforge.dll on Windows.

This is the Python equivalent of check_deps.ps1.  Use it when PowerShell's
execution policy prevents running .ps1 scripts, or when you prefer a Python
diagnostic output (e.g. from Blender's Python console).

Usage (from repo root):
    python QuadForge\\engine\\prebuilt\\win64\\install_deps.py
    # or
    python -m QuadForge.engine.prebuilt.win64.install_deps

Exit codes:
    0  All required dependencies found — quadforge.dll should load.
    1  One or more required dependencies missing — see output for details.
    2  quadforge.dll itself is absent (build or download it first).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import os
import platform
import struct
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH   = os.path.join(SCRIPT_DIR, "quadforge.dll")

# ---------------------------------------------------------------------------
# ANSI colour helpers (Windows 10+ supports VT sequences; fallback to plain)
# ---------------------------------------------------------------------------

_USE_COLOR = (
    sys.stdout.isatty()
    and platform.system() == "Windows"
    and int(platform.version().split(".")[0]) >= 10
)

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def ok(msg: str)   -> None: print("  " + _c("32", "✓") + "  " + msg)
def warn(msg: str) -> None: print("  " + _c("33", "!") + "  " + msg)
def fail(msg: str) -> None: print("  " + _c("31", "✗") + "  " + msg)
def hdr(msg: str)  -> None: print("\n" + _c("36", f"--- {msg} ---"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system32() -> str:
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")


def _path_dirs() -> list[str]:
    return [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]


def _find_on_path(*names: str) -> str | None:
    """Return the first path where any of *names* exists (System32 + PATH)."""
    search = [_system32()] + _path_dirs()
    for d in search:
        for name in names:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dll_machine_type(path: str) -> str | None:
    """Read the PE header MACHINE field to confirm x86-64."""
    try:
        with open(path, "rb") as f:
            dos = f.read(64)
            if dos[:2] != b"MZ":
                return None
            pe_off = struct.unpack_from("<I", dos, 60)[0]
            f.seek(pe_off)
            sig = f.read(4)
            if sig != b"PE\x00\x00":
                return None
            machine = struct.unpack_from("<H", f.read(20))[0]
        return {0x8664: "x64", 0x014C: "x86", 0xAA64: "ARM64"}.get(machine, f"0x{machine:04X}")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_dll_present() -> bool:
    hdr("quadforge.dll presence")
    if not os.path.isfile(DLL_PATH):
        fail(f"quadforge.dll not found in:\n       {SCRIPT_DIR}")
        print(
            "\n  To obtain quadforge.dll choose one of:\n"
            "   A) Build it:  run build.bat from a VS 2022 Developer Command Prompt\n"
            "   B) Download:  grab QuadForge_win64_vX.Y.Z.zip from the GitHub Releases page\n"
            "   See MISSING_BINARY.md for step-by-step instructions."
        )
        return False
    size_kb = os.path.getsize(DLL_PATH) / 1024
    ok(f"quadforge.dll found  ({size_kb:.0f} KB)")
    return True


def check_architecture() -> bool:
    hdr("Architecture")
    is64 = struct.calcsize("P") == 8
    if not is64:
        fail("Running 32-bit Python — quadforge.dll is 64-bit only!")
        print("       Install Blender 4.2+ which ships 64-bit Python.")
        return False
    ok("64-bit Python  ✓")

    if os.path.isfile(DLL_PATH):
        mtype = _dll_machine_type(DLL_PATH)
        if mtype == "x64":
            ok("quadforge.dll is x64  ✓")
        elif mtype is not None:
            fail(f"quadforge.dll machine type is {mtype} — expected x64")
            return False
    return True


def check_checksum() -> bool:
    hdr("SHA-256 checksum")
    if not os.path.isfile(DLL_PATH):
        warn("DLL absent — skipping checksum")
        return True  # Not a dep failure per se
    actual = _sha256(DLL_PATH)
    checksums_path = os.path.join(SCRIPT_DIR, "CHECKSUMS")
    if not os.path.isfile(checksums_path):
        warn("CHECKSUMS file not found — cannot verify integrity")
        return True
    reference: str | None = None
    with open(checksums_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "quadforge.dll":
                reference = parts[0].lower()
                break
    if reference is None:
        warn("No quadforge.dll entry in CHECKSUMS")
        return True
    if reference == "0" * 64:
        warn("CHECKSUMS contains placeholder — run build.bat to populate it")
        return True
    if actual == reference:
        ok(f"SHA-256 matches  ({actual[:16]}…)")
        return True
    fail(f"SHA-256 MISMATCH!\n"
         f"       Expected: {reference}\n"
         f"       Actual:   {actual}\n"
         "       The DLL may be corrupted or from a different build.")
    return False


def check_vcruntime() -> bool:
    hdr("Visual C++ Runtime (VCRUNTIME140.dll)")
    path = _find_on_path("VCRUNTIME140.dll", "vcruntime140.dll")
    if path:
        ok(f"VCRUNTIME140.dll found: {path}")
        return True
    # Also check registry
    try:
        import winreg  # type: ignore
        for hive in (winreg.HKEY_LOCAL_MACHINE,):
            for subkey in (
                r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            ):
                try:
                    with winreg.OpenKey(hive, subkey) as k:
                        ver = winreg.QueryValueEx(k, "Version")[0]
                        ok(f"VC++ Redistributable 2015-2022 installed: {ver}")
                        return True
                except FileNotFoundError:
                    pass
    except ImportError:
        pass  # winreg not available outside Windows
    warn(
        "VCRUNTIME140.dll not found.\n"
        "       If quadforge.dll was built with /MT (static runtime, the default),\n"
        "       this is NOT required — the runtime is baked into the DLL.\n"
        "       If you see 'module not found' errors, install the Visual C++\n"
        "       Redistributable from:\n"
        "       https://aka.ms/vs/17/release/vc_redist.x64.exe"
    )
    return True   # Non-fatal for /MT builds


def check_openmp() -> bool:
    hdr("OpenMP runtime (vcomp140.dll / libomp.dll)")
    path = _find_on_path("vcomp140.dll", "vcomp140d.dll", "libomp.dll")
    if path:
        ok(f"OpenMP runtime found: {path}")
        return True
    warn(
        "vcomp140.dll / libomp.dll not found on PATH or in System32.\n"
        "       quadforge.dll will still load but OpenMP parallelism will be\n"
        "       disabled (single-threaded fallback inside the DLL).\n"
        "       To enable multi-threading install the Visual C++ Redistributable:\n"
        "       https://aka.ms/vs/17/release/vc_redist.x64.exe"
    )
    return True   # Soft dependency — DLL loads without it (serial mode)


def check_load() -> bool:
    hdr("Python ctypes load test")
    if not os.path.isfile(DLL_PATH):
        warn("DLL absent — skipping load test")
        return True
    try:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(SCRIPT_DIR)
        lib = ctypes.CDLL(DLL_PATH)
        lib.qf_version.restype  = ctypes.c_char_p
        lib.qf_version.argtypes = []
        lib.qf_init.restype     = ctypes.c_int
        lib.qf_init.argtypes    = []
        lib.qf_shutdown.restype = None
        lib.qf_shutdown.argtypes = []
        if lib.qf_init() != 0:
            fail("qf_init() returned non-zero — engine initialisation failed")
            return False
        ver = lib.qf_version()
        ver_str = ver.decode("utf-8", errors="replace") if ver else "(none)"
        lib.qf_shutdown()
        ok(f"DLL loaded and initialised successfully  —  version: {ver_str}")
        return True
    except OSError as exc:
        fail(f"ctypes.CDLL failed: {exc}")
        print(
            "\n  Common causes:\n"
            "   • Missing Visual C++ Redistributable (VCRUNTIME140.dll)\n"
            "   • Missing OpenMP runtime (vcomp140.dll / libomp.dll)\n"
            "   • Antivirus quarantine — add the add-on folder to exclusions\n"
            "   • Architecture mismatch (32-bit Python + 64-bit DLL)\n"
            "  Run check_deps.ps1 for a detailed dependency walk."
        )
        return False


def check_blender_version() -> bool:
    hdr("Blender version (optional)")
    try:
        import bpy  # type: ignore
        ver = ".".join(str(x) for x in bpy.app.version)
        major, minor = bpy.app.version[:2]
        if (major, minor) >= (4, 2):
            ok(f"Blender {ver}  ✓  (4.2+ required)")
        else:
            warn(f"Blender {ver} — QuadForge requires 4.2 or later")
    except ImportError:
        ok("Not running inside Blender — skipping Blender version check")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  QuadForge win64 — Python dependency checker")
    print(f"  Platform: {platform.platform()}")
    print(f"  Python:   {sys.version.split()[0]}")
    print("=" * 60)

    dll_present = check_dll_present()
    if not dll_present:
        print(f"\n{'='*60}")
        print("  RESULT: quadforge.dll absent — build or download it first.")
        print(f"{'='*60}\n")
        return 2

    errors = 0
    if not check_architecture():  errors += 1
    if not check_checksum():      errors += 1
    check_vcruntime()   # warnings only
    check_openmp()      # warnings only
    if not check_load():          errors += 1
    check_blender_version()

    print(f"\n{'='*60}")
    if errors == 0:
        print("  RESULT: All checks passed — quadforge.dll should load correctly.")
    else:
        print(f"  RESULT: {errors} check(s) FAILED — see output above.")
    print(f"{'='*60}\n")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
