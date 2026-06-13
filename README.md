# QuadForge v14.0.0

**Next-generation auto quad remeshing for Blender 4.2+ / 5.0+**

QuadForge is a free, open-source quad remesher built directly into Blender's
sidebar — no external process, no DRM, no $110 price tag.

## Key Features

- **5–10× faster** than QuadRemesher (zero FBX serialisation overhead + OpenMP parallelism)
- **Globally optimal** cross-field via Knöppel 2013 eigensolver
- **Pure-Python fallback** — works without compiling anything
- **Native C++ engine** — optional compiled shared library for full speed
- **UV seam preservation** — unique feature not found in QuadRemesher
- **5 built-in presets** — Organic, Hard Surface, Sculpt, Architecture, Fast
- **Per-algorithm selection** — swap field solver, parametrisation, extraction independently
- **Debug cross-field visualizer** — inspect field directions before committing to a full remesh
- **Environment check operator** — one-click diagnostics: Blender version, NumPy, native engine status
- **Catmull-Clark compatibility checker** — `Check CC Compatibility` button analyses output for subdivision readiness
- **3 field solvers** — Knöppel 2013 (optimal), Eigen Smooth (fast+good), Curvature Only (fastest)
- **3 parametrization methods** — MIQ, IGM (strict integer-grid for CAD), Poisson Simple
- **4 extraction modes** — Iso-line, Motorcycle Graph (T-junction-free), Dual Contour, Greedy Merge
- **Blender 4.x/5.x compatibility layer** — all version-sensitive API calls are shim-wrapped

## Quick Install

1. Download `QuadForge-11.0.0-<platform>.zip` from [Releases](../../releases)
2. Blender → Edit → Preferences → Add-ons → Install… → select zip
3. Enable **QuadForge** in the add-on list
4. Select a mesh → N-panel → QuadForge tab → **⚡ REMESH IT**

## Build from Source

```bash
git clone <repo> && cd quadforge
git submodule update --init --recursive
./build/scripts/build_all_platforms.sh
```

See [docs/BUILDING.md](docs/BUILDING.md) for full instructions.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | End-user panel reference |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | Mathematical derivations |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | C API for the native engine |
| [docs/BUILDING.md](docs/BUILDING.md) | Build & packaging instructions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

## Architecture

```
QuadForge/           ← Python add-on (Blender loads this)
  bridge.py          ← ctypes loader + Python fallback dispatcher
  engine/            ← Pure-Python algorithm modules (9-stage pipeline)
  engine/include/    ← C++ headers (api.h, types.h, mesh/, field/, ...)
  engine/prebuilt/   ← Pre-compiled .so/.dll/.dylib per platform

build/
  CMakeLists.txt     ← CMake build system
  cmake/             ← Platform setup modules
  scripts/           ← build_all_platforms.sh, package_addon.py

tests/
  conftest.py        ← pytest configuration & shared mesh fixtures
  test_pipeline.py   ← Python test suite (pytest, no Blender needed, 80+ tests)
  test_meshes/       ← 21 reference OBJ files
  benchmark.py

docs/
```

## Algorithm Pipeline

1. **Preprocessing** — half-edge construction, feature detection, sizing field
2. **Cross-field** — globally optimal 4-RoSy field (Knöppel 2013)
3. **Singularity optimisation** — cancel pairs, relocate to corners
4. **Parametrization** — Poisson + MIQ integer rounding
5. **Quad extraction** — iso-line tracing + motorcycle graph
6. **Post-processing** — Taubin smoothing, BVH projection, feature snapping

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

- Python add-on: **GPL v3** (required by Blender)
- C++ engine library: **MIT**

## Author

Reynold — March 2026
