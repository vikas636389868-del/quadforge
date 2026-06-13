#!/usr/bin/env bash
# build.sh — Rebuild libquadforge.dylib from C source on macOS ARM64.
#
# Usage:
#   bash QuadForge/engine/prebuilt/macos_arm64/build.sh          # default (Release)
#   bash QuadForge/engine/prebuilt/macos_arm64/build.sh debug     # Debug build
#   bash QuadForge/engine/prebuilt/macos_arm64/build.sh clean     # Remove artefacts
#
# Requirements:
#   - Xcode Command Line Tools  (xcode-select --install)
#   - libomp  (brew install libomp)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/libquadforge_macos_arm64.c"
OUT="$SCRIPT_DIR/libquadforge.dylib"

# ---- Handle "clean" ----
if [[ "${1:-}" == "clean" ]]; then
    rm -f "$OUT" "$SCRIPT_DIR"/*.o
    echo "Cleaned."
    exit 0
fi

# ---- Compiler detection ----
CC="${CC:-clang}"
if ! command -v "$CC" &>/dev/null; then
    echo "ERROR: $CC not found.  Install Xcode Command Line Tools:"
    echo "       xcode-select --install"
    exit 1
fi

# ---- Detect Homebrew prefix (ARM64 vs x86_64) ----
if [[ -d "/opt/homebrew" ]]; then
    BREW_PREFIX="/opt/homebrew"
elif [[ -d "/usr/local/Homebrew" ]]; then
    BREW_PREFIX="/usr/local"
else
    BREW_PREFIX=""
fi

# ---- Build mode ----
MODE="${1:-release}"
case "$MODE" in
    debug)
        CFLAGS="-g -O0 -DDEBUG -fsanitize=address,undefined"
        LDFLAGS="-fsanitize=address,undefined"
        echo "Building libquadforge.dylib [DEBUG + sanitizers]"
        ;;
    release|*)
        CFLAGS="-O3 -DNDEBUG -flto"
        LDFLAGS="-flto"
        echo "Building libquadforge.dylib [RELEASE]"
        ;;
esac

# ---- Common flags ----
CFLAGS="$CFLAGS -std=c11 -arch arm64 -shared -fPIC -fvisibility=hidden"
CFLAGS="$CFLAGS -Wall -Wextra -Wno-unused-parameter"
CFLAGS="$CFLAGS -mmacosx-version-min=13.0"
CFLAGS="$CFLAGS -install_name @rpath/libquadforge.dylib"

# ---- OpenMP (optional, via Homebrew libomp) ----
OMP_CFLAGS=""
OMP_LDFLAGS=""
LIBOMP_DIR=""
if [[ -n "$BREW_PREFIX" ]] && [[ -d "$BREW_PREFIX/opt/libomp" ]]; then
    LIBOMP_DIR="$BREW_PREFIX/opt/libomp"
elif [[ -d "/opt/homebrew/opt/libomp" ]]; then
    LIBOMP_DIR="/opt/homebrew/opt/libomp"
elif [[ -d "/usr/local/opt/libomp" ]]; then
    LIBOMP_DIR="/usr/local/opt/libomp"
fi

if [[ -n "$LIBOMP_DIR" ]]; then
    OMP_CFLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_DIR}/include"
    OMP_LDFLAGS="-L${LIBOMP_DIR}/lib -lomp"
    echo "  OpenMP: enabled (${LIBOMP_DIR})"
else
    echo "  OpenMP: not available (single-threaded build)"
    echo "          Install with: brew install libomp"
fi

# ---- Compile ----
set -x
$CC $CFLAGS $OMP_CFLAGS -o "$OUT" "$SRC" -lm -lpthread $LDFLAGS $OMP_LDFLAGS
set +x

# ---- Post-build ----
# strip local symbols — skip for debug builds where ASAN/UBSAN metadata must
# be preserved (stripping a sanitized binary corrupts it).
if [[ "$MODE" != "debug" ]]; then
    strip -x "$OUT" 2>/dev/null || true
else
    echo "  (strip skipped for debug+sanitizer build)"
fi

echo
echo "Built: $OUT  ($(du -h "$OUT" | cut -f1))"
echo

# ---- Verify architecture ----
echo "--- Architecture check ---"
file "$OUT"
lipo -archs "$OUT" 2>/dev/null || true
echo

# ---- Update CHECKSUMS ----
CKSUM=$(shasum -a 256 "$OUT" | awk '{print $1}')
cat > "$SCRIPT_DIR/CHECKSUMS" <<EOF
# QuadForge macos_arm64 binary checksums
# Verify with: shasum -a 256 -c CHECKSUMS
${CKSUM}  libquadforge.dylib
EOF
echo "Updated CHECKSUMS: $CKSUM"

# ---- Quick smoke test ----
echo
echo "--- Quick symbol check ---"
nm -gU "$OUT" 2>/dev/null | grep "_qf_" | awk '{print "  ✓ " $3}' || true
echo
echo "Build complete."
