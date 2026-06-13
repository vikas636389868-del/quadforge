# quadforge.dll — Binary Not Included

`quadforge.dll` must be **compiled on a Windows x64 machine** using either
the Visual C++ Build Tools (MSVC) or the LLVM/Clang toolchain.
It cannot be cross-compiled from Linux or macOS.

---

## Build in One Step

**Option A — MSVC (recommended)**

Open a **Developer Command Prompt for VS 2022** (or later), navigate to
the repo root, then run:

```bat
QuadForge\engine\prebuilt\win64\build.bat
```

That script detects your compiler, compiles `quadforge_win64.c`, links
`quadforge.dll`, updates `CHECKSUMS`, and runs a quick Python load test.

**Option B — Clang-cl** (LLVM for Windows, also produces an MSVC-ABI DLL)

```bat
QuadForge\engine\prebuilt\win64\build.bat clang
```

**Option C — Debug build** (adds `/fsanitize=address` where supported)

```bat
QuadForge\engine\prebuilt\win64\build.bat debug
```

---

## Prerequisites

### MSVC path (Option A / C)

| Requirement | Download |
|-------------|----------|
| Visual Studio 2022 Build Tools | <https://aka.ms/vs/17/release/vs_BuildTools.exe> |
| "Desktop development with C++" workload | (select during install) |
| OpenMP runtime (included automatically) | Bundled in Build Tools |

After installing, launch **Developer Command Prompt for VS 2022** so that
`cl.exe` and `link.exe` are on `%PATH%`.

### Clang-cl path (Option B)

| Requirement | Download |
|-------------|----------|
| LLVM for Windows | <https://github.com/llvm/llvm-project/releases> → `LLVM-*-win64.exe` |
| Visual Studio Build Tools (for headers/libs) | Same as Option A |

### Python dependency check

After building, verify all runtime dependencies are present:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "QuadForge\engine\prebuilt\win64\check_deps.ps1"
```

---

## What the Build Produces

| File | Size (approx.) | Purpose |
|------|----------------|---------| 
| `quadforge.dll` | 200–400 KB | Native engine — loaded by `bridge.py` at runtime |
| `quadforge.lib` | < 5 KB | Import library (not needed at runtime) |
| `quadforge.exp` | < 5 KB | Export file (not needed at runtime) |

Only `quadforge.dll` needs to be present in this directory.

---

## Verifying the Build

After building, run the three verification tools:

```bat
REM 1. API smoke test — loads the DLL and calls all 14 exported functions
python QuadForge\engine\prebuilt\win64\verify_api.py

REM 2. ABI layout check — verifies struct sizes match bridge.py's ctypes mirrors
python QuadForge\engine\prebuilt\win64\compat_check.py

REM 3. Dependency checker — finds missing runtime DLLs
powershell -ExecutionPolicy Bypass -File QuadForge\engine\prebuilt\win64\check_deps.ps1
```

All three should exit with no errors before distributing the DLL.

---

## CI / Automated Builds

The GitHub Actions workflow (`.github/workflows/build.yml`) compiles
`quadforge.dll` for every version tag on a `windows-latest` runner and
uploads it as a release artefact. To use a CI-built binary:

1. Go to **Releases** on the GitHub repository page.
2. Download `QuadForge_win64_vX.Y.Z.zip`.
3. Extract `quadforge.dll` into **this directory**.
4. Verify the SHA-256 hash against `CHECKSUMS`.

---

## Runtime Fallback

If `quadforge.dll` is absent when Blender loads the add-on, `bridge.py`
automatically falls back to the **pure-Python pipeline** (`engine/pipeline.py`).
The add-on remains fully functional — remeshing works correctly on all meshes —
but runs approximately **5–10× slower** on large inputs because:

- No OpenMP thread-level parallelism (single-threaded)
- No sparse Cholesky solver (pure NumPy linear algebra)
- No optimised memory layout (Python object overhead)

The fallback is transparent to the user — no error is shown, and all presets
and settings work identically.

---

## Troubleshooting

### "The specified module could not be found"

This means Windows cannot find one of `quadforge.dll`'s dependencies.
Run `check_deps.ps1` to identify the missing DLL.  Most commonly this is:

- **VCRUNTIME140.dll** — install the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- **vcomp140.dll** — same installer; provides the OpenMP runtime
- **libomp.dll** — only needed for Clang-cl builds; install [LLVM](https://releases.llvm.org/)

### "Not a valid Win32 application"

Architecture mismatch — you have a 32-bit Python running in Blender but a
64-bit DLL (or vice versa).  Blender ships **64-bit Python** on all supported
Windows platforms since Blender 2.80.  Ensure you are using a modern Blender
(4.2+).

### "Access is denied" when loading

Antivirus software may quarantine unsigned DLLs.  Add the add-on directory
to your antivirus exclusion list, or sign the DLL with a code-signing
certificate.
