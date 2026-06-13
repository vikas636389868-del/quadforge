# QuadForge Prebuilt Library — macos_x64

This directory contains the compiled native engine for **macOS x86_64** (Intel Mac).

## Files

| File | Description |
|------|-------------|
| `libquadforge.dylib` | Compiled shared library — the native QuadForge engine |
| `libquadforge_macos_x64.c` | Complete C source — rebuild the .dylib without the full CMake tree |
| `loader_macos.py` | macOS-specific loader helper (handles @rpath, libomp, DYLD_LIBRARY_PATH) |
| `check_deps.sh` | Runtime dependency checker — run if Blender fails to load the library |
| `build.sh` | One-command rebuild script (`bash build.sh` / `bash build.sh debug`) |
| `verify_api.py` | Loads the .dylib and verifies all 14 C API functions work correctly |
| `compat_check.py` | ABI struct layout checker — catches C <-> bridge.py field mismatches |
| `VERSION` | Library version string |
| `CHECKSUMS` | SHA-256 checksum for binary integrity verification |

## Runtime Requirements

| Library | Version | Purpose |
|---------|---------|---------|
| `libSystem.B.dylib` | macOS 12.0+ | C runtime (always present) |
| `libm.dylib` | Standard math | Math functions (always present) |
| `libomp.dylib` | LLVM OpenMP runtime | Parallel execution (optional) |

Install missing deps:
```bash
brew install libomp       # OpenMP runtime (recommended)
xcode-select --install    # Xcode Command Line Tools (for building)
```

> **Note:** On Intel Macs, Homebrew installs to `/usr/local` (not `/opt/homebrew`).

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

### Direct clang compilation (no CMake)

```bash
clang -O3 -arch x86_64 -shared -fPIC \
      -fvisibility=hidden -mmacosx-version-min=12.0 \
      -Xpreprocessor -fopenmp \
      -I$(brew --prefix libomp)/include \
      -L$(brew --prefix libomp)/lib -lomp \
      -install_name @rpath/libquadforge.dylib \
      -o libquadforge.dylib \
      libquadforge_macos_x64.c -lm -lpthread
```

## Python Fallback

If `libquadforge.dylib` is absent, `bridge.py` automatically falls back to
the pure-Python pipeline in `engine/pipeline.py`. Approximately **5-10x slower**
on large meshes, but functionally identical.

## API

All 14 exported functions match `include/quadforge/api.h` exactly.

## macOS x86_64-specific Notes

- **Architecture:** Targets `x86_64` (Intel Mac). For Apple Silicon, use `macos_arm64`.
- **Minimum macOS:** 12.0 (Monterey). Blender 4.x requires macOS 12+ on Intel.
- **Homebrew prefix:** Intel Macs use `/usr/local`. Scripts probe both paths.
- **Code signing:** `codesign -s - libquadforge.dylib` if Gatekeeper blocks it.
- **@rpath:** `@rpath/libquadforge.dylib` — Blender handles rpath automatically.
