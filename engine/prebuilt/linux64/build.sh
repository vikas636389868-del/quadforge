#!/usr/bin/env bash
# build.sh — Rebuild libquadforge.so from C source on Linux x86-64.
#
# Usage:
#   bash QuadForge/engine/prebuilt/linux64/build.sh          # default (Release)
#   bash QuadForge/engine/prebuilt/linux64/build.sh debug     # Debug build
#   bash QuadForge/engine/prebuilt/linux64/build.sh clean     # Remove artefacts
#
# Requirements:
#   - GCC 11+ (or Clang 14+)
#   - libgomp  (sudo apt-get install libgomp1)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/libquadforge_linux64.c"
OUT="$SCRIPT_DIR/libquadforge.so"

# ---- Handle "clean" ----
if [[ "${1:-}" == "clean" ]]; then
    rm -f "$OUT" "$SCRIPT_DIR"/*.o
    echo "Cleaned."
    exit 0
fi

# ---- Compiler detection ----
CC="${CC:-gcc}"
if ! command -v "$CC" &>/dev/null; then
    echo "ERROR: $CC not found.  Install with: sudo apt-get install gcc"
    exit 1
fi

# ---- Build mode ----
MODE="${1:-release}"
case "$MODE" in
    debug)
        CFLAGS="-g -O0 -DDEBUG -fsanitize=address,undefined"
        LDFLAGS="-fsanitize=address,undefined"
        echo "Building libquadforge.so [DEBUG + sanitizers]"
        ;;
    release|*)
        CFLAGS="-O3 -march=x86-64-v2 -DNDEBUG -flto"
        LDFLAGS="-flto"
        echo "Building libquadforge.so [RELEASE]"
        ;;
esac

# ---- Common flags ----
CFLAGS="$CFLAGS -std=c11 -shared -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter"

# ---- OpenMP (optional) ----
OMP_FLAGS=""
if $CC -fopenmp -E - < /dev/null &>/dev/null 2>&1; then
    OMP_FLAGS="-fopenmp"
    echo "  OpenMP: enabled"
else
    echo "  OpenMP: not available (single-threaded build)"
fi

# ---- Compile ----
set -x
$CC $CFLAGS $OMP_FLAGS -o "$OUT" "$SRC" -lm -lpthread $LDFLAGS ${OMP_FLAGS}
set +x

# ---- Post-build ----
strip --strip-unneeded "$OUT" 2>/dev/null || true
chmod +x "$OUT"

echo
echo "Built: $OUT  ($(du -h "$OUT" | cut -f1))"
echo

# ---- Update CHECKSUMS ----
CKSUM=$(sha256sum "$OUT" | awk '{print $1}')
cat > "$SCRIPT_DIR/CHECKSUMS" <<EOF
# QuadForge linux64 binary checksums
# Verify with: sha256sum -c CHECKSUMS
${CKSUM}  libquadforge.so
EOF
echo "Updated CHECKSUMS: $CKSUM"

# ---- Quick smoke test ----
echo
echo "--- Quick symbol check ---"
nm -D "$OUT" 2>/dev/null | grep " T qf_" | awk '{print "  ✓ " $3}' || true
echo
echo "Build complete."
