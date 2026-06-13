# libquadforge.dylib — Binary Not Included

`libquadforge.dylib` must be **compiled on a macOS x86_64 machine** (Intel Mac).
It cannot be cross-compiled from Linux, Windows, or Apple Silicon without Rosetta 2.

## How to Build It (One Command)

```bash
# From the repo root, on a macOS x86_64 (Intel) machine:
bash QuadForge/engine/prebuilt/macos_x64/build.sh
```

### Prerequisites

```bash
xcode-select --install          # Xcode Command Line Tools
brew install libomp             # LLVM OpenMP runtime (optional but strongly recommended)
```

> **Homebrew on Intel Macs** installs to `/usr/local` (not `/opt/homebrew`).

## What the Build Produces

| File | Size (approx.) | Purpose |
|------|----------------|---------| 
| `libquadforge.dylib` | 250–350 KB | The compiled native engine — loaded by `bridge.py` at runtime |

After building, verify everything works:

```bash
python3 QuadForge/engine/prebuilt/macos_x64/verify_api.py    # API smoke-test
python3 QuadForge/engine/prebuilt/macos_x64/compat_check.py  # ABI layout check
bash QuadForge/engine/prebuilt/macos_x64/check_deps.sh       # Runtime dependency check
```

## Runtime Fallback

If `libquadforge.dylib` is absent when Blender loads the add-on, `bridge.py`
automatically falls back to the **pure-Python pipeline** (`engine/pipeline.py`).
The add-on remains fully functional — remeshing works correctly on all meshes —
but runs approximately **5–10× slower** on large inputs.

## CI / Automated Builds

The GitHub Actions workflow (`.github/workflows/build.yml`) compiles the dylib
for every version tag and uploads it as a release artefact.
Download the artefact and extract `libquadforge.dylib` into this directory.

## Apple Silicon (M1/M2/M3/M4)

If you are on Apple Silicon, use the `macos_arm64` build instead.
The `macos_x64` dylib can run on Apple Silicon via **Rosetta 2**, but
performance will be lower than a native arm64 build.

## Bug Fixes Applied to This Source (v21 → v22)

The following bugs were found and fixed in `libquadforge_macos_x64.c` by
a code review conducted in April 2026. The source you are building from
incorporates all fixes.

| # | Severity | Location | Description | Fix |
|---|----------|----------|-------------|-----|
| 1 | **Critical** | `edge_map_add()` line ~251 | `assert(pool_used < pool_cap)` is a no-op in release builds (`-DNDEBUG`), causing a silent out-of-bounds write on pathological non-manifold meshes | Replaced with a bounds check that returns early instead of crashing |
| 2 | **Performance** | Symmetry enforcement in `do_remesh()` | Mirror-vertex search was O(n²) — on a 500K-vertex mesh this took ~30 s per symmetry pass | Replaced with an O(n) spatial hash (`VHash`); same results, ×10–100 faster |
| 3 | **Cleanup** | `#include <assert.h>` | Left over after removing the only `assert()` call | Removed |
| 4 | **Correctness** | `compat_check.py` size checks | `QFInputMesh` and `QFResult` struct sizes were printed but never validated against expected values — silent failures possible | Added explicit size checks with pass/fail output for all three structs |
