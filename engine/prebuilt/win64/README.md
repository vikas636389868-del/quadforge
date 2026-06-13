# QuadForge Prebuilt Library — win64

Place the compiled shared library for **win64** here:

  quadforge.dll

## How to Build

```bash
# 1. Fetch vendored C++ headers
python build/scripts/setup_third_party.py

# 2. Configure and build
cmake -S build -B _build -DCMAKE_BUILD_TYPE=Release
cmake --build _build --target quadforge -- -j$(nproc)

# 3. Install into this directory
cmake --install _build
```

## Pre-built Binaries (CI)

Download from the GitHub **Releases** page → select your platform artefact
and extract here.  The CI workflow (`.github/workflows/build.yml`) builds
all platforms on every version tag.

## Python Fallback

If no compiled library is present, `bridge.py` automatically falls back to
the pure-Python pipeline in `engine/pipeline.py`.  Functionally identical,
approximately 5–10× slower on large meshes (no OpenMP, no sparse Cholesky).
