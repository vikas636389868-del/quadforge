# SuiteSparse / CHOLMOD — Vendored Public Headers

QuadForge uses **CHOLMOD 7.x** (part of SuiteSparse) for sparse Cholesky
factorisation in the parametrisation stage when meshes exceed ~500K faces.
CHOLMOD is significantly faster than Eigen's built-in `SimplicialLLT` on
very large systems because it uses AMD/COLAMD fill-reducing reordering
and supernodal factorisation.

## Status

**Headers:**        Vendored — SuiteSparse 7.12.2 public headers included.
**Pre-built libs:** **Not** vendored. The shared / static library
                    (`libcholmod.so`, `libcholmod.dylib`, `cholmod.lib`)
                    must be supplied at build time by a system install.
                    Library binaries are platform-specific and cannot be
                    practically vendored.

## What's in this directory

    include/suitesparse/
        SuiteSparse_config.h     (SuiteSparse 7.12.2)
        amd.h
        camd.h
        ccolamd.h
        cholmod.h
        colamd.h
    licenses/
        SuiteSparse-LICENSE.txt
        AMD-License.txt
        CAMD-License.txt
        CCOLAMD-License.txt
        CHOLMOD-License.txt
        COLAMD-License.txt
        SuiteSparse_config-README.txt

These are the six public headers CHOLMOD needs. `cholmod.h` is self-contained
(it only includes `SuiteSparse_config.h` plus standard C/C++ headers), so
`-I third_party/suitesparse/include` is sufficient for compilation.

## Licensing

SuiteSparse components used here ship under a mix of LGPL-2.1+ (CHOLMOD,
AMD, CAMD, CCOLAMD, COLAMD) and Apache-2.0 (SuiteSparse_config) licenses.
Full license texts are preserved in the `licenses/` sub-folder as required
for LGPL redistribution compliance. When distributing QuadForge, ship the
`licenses/` folder alongside the headers.

Upstream: https://github.com/DrTimothyAldenDavis/SuiteSparse

## How the build finds SuiteSparse

`build/cmake/FindSuiteSparse.cmake` searches for `cholmod.h` in this order:

    1. QuadForge vendored headers         (this directory)
    2. $SUITESPARSE_ROOT/include           (environment override)
    3. /usr/include/suitesparse            (Linux distros)
    4. /usr/local/include/suitesparse      (manual install)
    5. /opt/homebrew/include/suitesparse   (macOS / Homebrew arm64)
    6. C:/Program Files/SuiteSparse/include

The library is searched in the corresponding `lib` directories with names
`cholmod` / `CHOLMOD`, plus `amd`, `camd`, `ccolamd`, `colamd`, and
`suitesparseconfig`.

CHOLMOD support is **opt-in** via the CMake option:

    cmake -B build -DQF_ENABLE_CHOLMOD=ON

When enabled and found, CMake defines `QF_HAS_CHOLMOD=1` on the
`quadforge` target.

> **Status:** As of QuadForge v40.4 the Poisson solver in
> `src/param/poisson.cpp` branches on `QF_HAS_CHOLMOD` and uses CHOLMOD's
> supernodal Cholesky as the primary SPD solver for meshes with ≥500k
> vertices.  The `CholmodSPDSolver` wrapper lives in
> `src/param/cholmod_solver.{h,cpp}` and is compiled to an empty
> translation unit when `QF_HAS_CHOLMOD` is not defined, so the engine
> still builds cleanly on systems without SuiteSparse installed — it
> simply falls through to Eigen's `SparseLU` / `SimplicialLDLT` path.

## Installing CHOLMOD libraries on each platform

### Linux (Ubuntu / Debian)
```bash
sudo apt install libsuitesparse-dev
```

### macOS (Homebrew)
```bash
brew install suite-sparse
```

### Windows
Pre-built binaries:
    https://github.com/jlblancoc/suitesparse-metis-for-windows

Or via vcpkg:
```
vcpkg install suitesparse
```

### Build from source
```bash
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build --target install
```

## Re-populating / upgrading the vendored headers

To refresh the vendored headers from a newer SuiteSparse release:

```bash
python build/scripts/setup_third_party.py --suitesparse /path/to/SuiteSparse-X.Y.Z
```

The script copies `cholmod.h`, `SuiteSparse_config.h`, `amd.h`, `camd.h`,
`ccolamd.h`, `colamd.h` and the corresponding license files into this
directory. Internal headers (`*_internal.h`, `cholmod_template.h`,
`cholmod_types.h`) are intentionally **not** copied — they are for
SuiteSparse's own build and would pull in many more transitive
dependencies.

## Fallback behaviour

When SuiteSparse is **not found** at build time, QuadForge falls back to
Eigen's `SimplicialLLT` / `ConjugateGradient` (header-only, bundled in
`third_party/eigen/`). The fallback is fully functional — it is only
~2–3× slower than CHOLMOD on meshes >500K faces.

To explicitly disable SuiteSparse even when installed:

    cmake -B build -DQF_ENABLE_CHOLMOD=OFF
