# QuadForge Prebuilt Library — linux64

This directory contains the compiled native engine for **Linux x86-64**.

## Files

| File | Description |
|------|-------------|
| `libquadforge.so` | Compiled shared library — the native QuadForge engine |
| `libquadforge_linux64.c` | Complete C source — rebuild the .so without the full CMake tree |
| `loader_linux.py` | Linux-specific loader helper (handles RPATH, libgomp, LD_LIBRARY_PATH) |
| `check_deps.sh` | Runtime dependency checker — run if Blender fails to load the library |
| `build.sh` | One-command rebuild script (`bash build.sh` / `bash build.sh debug`) |
| `verify_api.py` | Loads the .so and verifies all 14 C API functions work correctly |
| `compat_check.py` | ABI struct layout checker — catches C ↔ bridge.py field mismatches |
| `VERSION` | Library version string |
| `CHECKSUMS` | SHA-256 checksum for binary integrity verification |

## Runtime Requirements

| Library | Version | Purpose |
|---------|---------|---------|
| `libc.so.6` | glibc 2.35+ (Ubuntu 22.04+) | C runtime |
| `libm.so.6` | Standard math | Math functions |
| `libgomp.so.1` | GCC OpenMP runtime | Parallel execution |

Install missing deps on Ubuntu/Debian:
```bash
sudo apt-get install libgomp1
```

## Verify the Binary

```bash
bash check_deps.sh
```

## Build from Source

```bash
# From repo root:
python build/scripts/setup_third_party.py
cmake -S build -B _build -DCMAKE_BUILD_TYPE=Release
cmake --build _build --target quadforge -- -j$(nproc)
cmake --install _build
```

## Python Fallback

If `libquadforge.so` is absent or fails to load, `bridge.py` automatically
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
