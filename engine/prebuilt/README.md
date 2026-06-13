# QuadForge — Prebuilt Native Libraries

This directory holds the platform-specific compiled shared libraries for the
QuadForge C++ engine. These files are **not** included in the source tree —
they must be built from source or copied from a CI artefact.

## Directory Layout

```
prebuilt/
  linux64/        libquadforge.so
  win64/          quadforge.dll
  macos_arm64/    libquadforge.dylib
  macos_x64/      libquadforge.dylib
```

## How to Build

### Prerequisites

| Tool | Min version |
|------|-------------|
| CMake | 3.20+ |
| GCC / Clang / MSVC | C++17 capable |
| OpenMP | 4.5+ (compiler runtime) |
| Third-party headers | Run `python build/scripts/setup_third_party.py` first |

### Quick build (Linux / macOS)

```bash
# 1. Fetch vendored C++ headers (Eigen, nanoflann, phmap)
python build/scripts/setup_third_party.py

# 2. Configure & build
cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release
cmake --build _build --target quadforge -- -j$(nproc)

# 3. Install into addon prebuilt dir
cmake --install _build --prefix .
```

The install step copies the shared library to the correct
`QuadForge/engine/prebuilt/{platform}/` subdirectory automatically.

### Windows (MSVC)

```bat
python build\scripts\setup_third_party.py
cmake -S . -B _build -G "Visual Studio 17 2022" -A x64
cmake --build _build --config Release --target quadforge
cmake --install _build
```

### Using without a native library

If no compiled library is present, `bridge.py` automatically falls back to
the pure-Python engine in `engine/pipeline.py`. The Python fallback is
functionally identical but approximately 5–10× slower on large meshes (no
OpenMP parallelism, no CHOLMOD sparse Cholesky).

## CI / Pre-built Artefacts

The GitHub Actions workflow (`.github/workflows/build.yml`) builds all three
platform libraries on every release tag push. Download the artefact for your
platform from the **Releases** page and unzip it here.
