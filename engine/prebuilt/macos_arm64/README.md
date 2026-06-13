# QuadForge Prebuilt Library — macos_arm64

This directory contains the compiled native engine for **macOS ARM64** (Apple Silicon).

## Files

| File | Description |
|------|-------------|
| `libquadforge.dylib` | Compiled shared library — the native QuadForge engine |
| `libquadforge_macos_arm64.c` | Complete C source — rebuild the .dylib without the full CMake tree |
| `loader_macos.py` | macOS-specific loader helper (handles @rpath, libomp, DYLD_LIBRARY_PATH) |
| `check_deps.sh` | Runtime dependency checker — run if Blender fails to load the library |
| `build.sh` | One-command rebuild script (`bash build.sh` / `bash build.sh debug`) |
| `verify_api.py` | Loads the .dylib and verifies all 14 C API functions work correctly |
| `compat_check.py` | ABI struct layout checker — catches C ↔ bridge.py field mismatches |
| `VERSION` | Library version string |
| `CHECKSUMS` | SHA-256 checksum for binary integrity verification |

## Runtime Requirements

| Library | Version | Purpose |
|---------|---------|---------|
| `libSystem.B.dylib` | macOS 13.0+ | C runtime (always present) |
| `libm.dylib` | Standard math | Math functions (always present) |
| `libomp.dylib` | LLVM OpenMP runtime | Parallel execution (optional) |

Install missing deps:
```bash
# Install OpenMP runtime (recommended for multi-threaded performance)
brew install libomp

# Install Xcode Command Line Tools (required for building from source)
xcode-select --install
```

## Verify the Binary

```bash
bash check_deps.sh
```

## Build from Source

### Quick build (single command)

```bash
bash build.sh           # Release build
bash build.sh debug     # Debug build with sanitizers
bash build.sh clean     # Remove artefacts
```

### Manual build

```bash
# From repo root:
python build/scripts/setup_third_party.py
cmake -S build -B _build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0
cmake --build _build --target quadforge -- -j$(sysctl -n hw.ncpu)
cmake --install _build
```

### Direct clang compilation (no CMake)

```bash
clang -O3 -arch arm64 -shared -fPIC \
      -fvisibility=hidden -mmacosx-version-min=13.0 \
      -Xpreprocessor -fopenmp \
      -I$(brew --prefix libomp)/include \
      -L$(brew --prefix libomp)/lib -lomp \
      -install_name @rpath/libquadforge.dylib \
      -o libquadforge.dylib \
      libquadforge_macos_arm64.c -lm -lpthread
```

## Python Fallback

If `libquadforge.dylib` is absent or fails to load, `bridge.py` automatically
falls back to the pure-Python pipeline in `engine/pipeline.py`.  
Functionally identical, approximately **5–10× slower** on large meshes.

## API

All 14 exported functions match `include/quadforge/api.h` exactly:

```
qf_init()               → int        (0 = success)
qf_shutdown()           → void
qf_version()            → const char*
qf_last_error()         → const char*
qf_has_gpu()            → int        (always 0 for CPU build)
qf_cpu_thread_count()   → int
qf_remesh(...)          → QFResult*
qf_free_result(...)     → void
qf_default_params()     → QFParams
qf_preset_organic()     → QFParams
qf_preset_hard_surface()→ QFParams
qf_preset_sculpt()      → QFParams
qf_preset_architecture()→ QFParams
qf_preset_fast()        → QFParams
```

## macOS-specific Notes

- **Architecture:** This library targets `arm64` (Apple Silicon M1/M2/M3/M4).
  For Intel Macs, use the `macos_x64` build instead.
- **Minimum macOS:** 13.0 (Ventura). Set via `-mmacosx-version-min=13.0`.
- **OpenMP:** Uses Homebrew's `libomp` instead of GCC's `libgomp`.
  The library works without OpenMP but runs single-threaded.
- **Code signing:** The built dylib may need ad-hoc signing for Gatekeeper:
  `codesign -s - libquadforge.dylib`
- **@rpath:** The library uses `@rpath/libquadforge.dylib` as its install name.
  Blender's embedded Python handles rpath resolution automatically.
- **Pointer alignment:** ARM64 requires natural alignment for pointer fields.
  The struct layout is LP64-compatible (identical to x86-64).

## Pre-built Binaries (CI)

Download from the GitHub **Releases** page → select your platform artefact
and extract here. The CI workflow (`.github/workflows/build.yml`) builds
all platforms on every version tag.
