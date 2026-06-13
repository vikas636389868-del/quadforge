#!/usr/bin/env bash
# check_deps.sh — Verify runtime dependencies for libquadforge.so on Linux.
# Run this if Blender reports "Failed to load native library".
#
# Usage:
#   bash QuadForge/engine/prebuilt/linux64/check_deps.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="$SCRIPT_DIR/libquadforge.so"

echo "=== QuadForge linux64 dependency check ==="
echo

if [[ ! -f "$LIB" ]]; then
    echo "ERROR: libquadforge.so not found in $SCRIPT_DIR"
    echo "       Run the build or download a pre-built binary."
    exit 1
fi

echo "Library: $LIB"
echo "Size:    $(du -h "$LIB" | cut -f1)"
echo

echo "--- ldd output ---"
ldd "$LIB" 2>&1 || true
echo

echo "--- Exported API symbols ---"
nm -D "$LIB" 2>/dev/null | grep " T qf_" | awk '{print $3}' | sort || true
echo

echo "--- OpenMP check ---"
if ldconfig -p 2>/dev/null | grep -q libgomp; then
    echo "✓ libgomp found via ldconfig"
else
    echo "✗ libgomp not found — install with: sudo apt-get install libgomp1"
fi

echo
echo "--- SHA256 checksum ---"
sha256sum "$LIB" 2>/dev/null || shasum -a 256 "$LIB" 2>/dev/null || true
echo
echo "Reference checksum (CHECKSUMS file):"
cat "$SCRIPT_DIR/CHECKSUMS" 2>/dev/null || echo "(CHECKSUMS file not found)"

echo
echo "--- Quick load test (Python) ---"
python3 - << 'PYEOF'
import ctypes, sys
try:
    lib = ctypes.CDLL("$(echo $LIB)")
    lib.qf_version.restype = ctypes.c_char_p
    lib.qf_init()
    ver = lib.qf_version().decode()
    lib.qf_shutdown()
    print(f"✓ Load OK — version: {ver}")
except Exception as e:
    print(f"✗ Load FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
