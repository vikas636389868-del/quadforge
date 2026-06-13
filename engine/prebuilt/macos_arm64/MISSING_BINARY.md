# libquadforge.dylib — Binary Not Included

`libquadforge.dylib` must be **compiled on a macOS ARM64 machine** (Apple Silicon M1/M2/M3/M4).
It cannot be cross-compiled from Linux or Windows.

## How to Build It (One Command)

```bash
# From the repo root, on a macOS ARM64 machine:
bash QuadForge/engine/prebuilt/macos_arm64/build.sh
```

### Prerequisites

```bash
xcode-select --install          # Xcode Command Line Tools
brew install libomp             # LLVM OpenMP runtime (optional but strongly recommended)
```

## What the Build Produces

| File | Size (approx.) | Purpose |
|------|----------------|---------|
| `libquadforge.dylib` | 250–350 KB | The compiled native engine — loaded by `bridge.py` at runtime |

After building, verify everything works:

```bash
python3 QuadForge/engine/prebuilt/macos_arm64/verify_api.py    # API smoke-test
python3 QuadForge/engine/prebuilt/macos_arm64/compat_check.py  # ABI layout check
bash QuadForge/engine/prebuilt/macos_arm64/check_deps.sh       # Runtime dependency check
```

## Runtime Fallback

If `libquadforge.dylib` is absent when Blender loads the add-on, `bridge.py`
automatically falls back to the **pure-Python pipeline** (`engine/pipeline.py`).
The add-on remains fully functional — remeshing works correctly on all meshes —
but runs approximately **5–10× slower** on large inputs.

## CI / Automated Builds

The dedicated GitHub Actions workflow (`.github/workflows/build-macos-arm64.yml`)
runs on GitHub's native `macos-14` Apple Silicon runners and compiles
`libquadforge.dylib` on every push to `main`/`develop` that touches this folder,
every pull request, and every version tag (`v*`).

- **Push / PR builds** upload the dylib as a workflow artefact
  (`libquadforge-macos_arm64`) you can download from the Actions tab.
- **Tag builds** automatically attach the dylib and its `CHECKSUMS` file to
  the corresponding GitHub Release.

Download the artefact and extract `libquadforge.dylib` into this directory.
