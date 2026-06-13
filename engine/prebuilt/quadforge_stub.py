"""QuadForge native library stub — Python fallback shim.

When no compiled libquadforge.so / .dll / .dylib is available, bridge.py
uses the Python pipeline (engine/pipeline.py) automatically.

This file exists only as documentation of the C API surface.
It is NOT loaded by bridge.py — it is here for reference only.

To build the real native library:
    python build/scripts/setup_third_party.py
    cmake -S build -B _build -DCMAKE_BUILD_TYPE=Release
    cmake --build _build --target quadforge -- -j$(nproc)
    cmake --install _build

C API (from include/quadforge/api.h):
    qf_init()                    -> int   (0 = ok, non-zero = error)
    qf_shutdown()                -> void
    qf_version()                 -> const char*
    qf_last_error()              -> const char*
    qf_has_gpu()                 -> int   (1 = GPU available)
    qf_cpu_thread_count()        -> int
    qf_remesh(input, params, cb, userdata) -> QFResult*
    qf_free_result(result)       -> void
    qf_default_params()          -> QFParams
    qf_preset_organic()          -> QFParams
    qf_preset_hard_surface()     -> QFParams
    qf_preset_sculpt()           -> QFParams
    qf_preset_architecture()     -> QFParams
    qf_preset_fast()             -> QFParams
"""

VERSION = "1.0.0 (stub — native library not compiled)"
