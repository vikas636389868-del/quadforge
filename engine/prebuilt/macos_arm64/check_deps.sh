#!/usr/bin/env bash
# check_deps.sh — Verify runtime dependencies for libquadforge.dylib on macOS ARM64.
# Run this if Blender reports "Failed to load native library".
#
# Usage:
#   bash QuadForge/engine/prebuilt/macos_arm64/check_deps.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="$SCRIPT_DIR/libquadforge.dylib"

echo "=== QuadForge macos_arm64 dependency check ==="
echo

if [[ ! -f "$LIB" ]]; then
    echo "ERROR: libquadforge.dylib not found in $SCRIPT_DIR"
    echo "       Run the build or download a pre-built binary."
    exit 1
fi

echo "Library: $LIB"
echo "Size:    $(du -h "$LIB" | cut -f1)"
echo

echo "--- Architecture ---"
file "$LIB"
lipo -archs "$LIB" 2>/dev/null || true
echo

echo "--- otool -L output (linked libraries) ---"
otool -L "$LIB" 2>&1 || true
echo

echo "--- Exported API symbols ---"
nm -gU "$LIB" 2>/dev/null | grep "_qf_" | awk '{print $3}' | sed 's/^_//' | sort || true
echo

echo "--- OpenMP (libomp) check ---"
LIBOMP_FOUND=0
for candidate in \
    "/opt/homebrew/opt/libomp/lib/libomp.dylib" \
    "/opt/homebrew/lib/libomp.dylib" \
    "/usr/local/opt/libomp/lib/libomp.dylib" \
    "/usr/local/lib/libomp.dylib"; do
    if [[ -f "$candidate" ]]; then
        echo "✓ libomp found: $candidate"
        LIBOMP_FOUND=1
        break
    fi
done
if [[ $LIBOMP_FOUND -eq 0 ]]; then
    echo "✗ libomp not found — install with: brew install libomp"
    echo "  (OpenMP parallelism will be disabled without libomp)"
fi

echo
echo "--- Minimum macOS version ---"
otool -l "$LIB" 2>/dev/null | grep -A2 "LC_BUILD_VERSION\|LC_VERSION_MIN_MACOSX" || echo "(unable to determine)"
echo

echo "--- Code signing ---"
codesign -dvv "$LIB" 2>&1 | head -5 || echo "(unsigned — may require ad-hoc signing for Gatekeeper)"
echo

echo "--- SHA256 checksum ---"
shasum -a 256 "$LIB" 2>/dev/null || true
echo
echo "Reference checksum (CHECKSUMS file):"
cat "$SCRIPT_DIR/CHECKSUMS" 2>/dev/null || echo "(CHECKSUMS file not found)"

echo
echo "--- Quick load test (Python) ---"
# NOTE: heredoc must be unquoted (PYEOF, not 'PYEOF') so that $LIB is expanded
# by bash before being passed to Python as the dylib path.
python3 - "$LIB" << PYEOF
import ctypes, sys, platform, os

lib_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not lib_path or not os.path.isfile(lib_path):
    print(f"✗ Library not found at: {lib_path!r}", file=sys.stderr)
    sys.exit(1)

try:
    if platform.machine() != "arm64":
        print(f"⚠ Running on {platform.machine()}, library is built for arm64")
    lib = ctypes.CDLL(lib_path)
    lib.qf_version.restype = ctypes.c_char_p
    lib.qf_init()
    ver = lib.qf_version().decode()
    lib.qf_shutdown()
    print(f"✓ Load OK — version: {ver}")
except Exception as e:
    print(f"✗ Load FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
