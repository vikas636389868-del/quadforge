@echo off
REM build.bat — Rebuild quadforge.dll from C source on Windows x64 (MSVC cl.exe).
REM
REM Usage:
REM   build.bat                 — Release build (default)
REM   build.bat debug           — Debug build with /fsanitize=address
REM   build.bat clang           — Release build via clang-cl (requires LLVM)
REM   build.bat clean           — Remove build artefacts
REM
REM Requirements:
REM   - Visual Studio 2022 Build Tools (cl.exe, link.exe)
REM     https://aka.ms/vs/17/release/vs_BuildTools.exe
REM   OR
REM   - LLVM/Clang for Windows (clang-cl.exe) when using "build.bat clang"
REM     https://github.com/llvm/llvm-project/releases
REM
REM   NOTE: Run from a "Developer Command Prompt for VS 2022" so that cl.exe
REM   and link.exe are on the PATH.  Alternatively, call vcvarsall.bat first:
REM     "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

setlocal EnableExtensions EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set SRC=%SCRIPT_DIR%quadforge_win64.c
set OUT=%SCRIPT_DIR%quadforge.dll
set OBJ=%SCRIPT_DIR%quadforge_win64.obj

REM ---- Handle "clean" ----
if /I "%~1"=="clean" (
    echo Cleaning build artefacts...
    if exist "%OUT%" del /f /q "%OUT%"
    if exist "%OBJ%" del /f /q "%OBJ%"
    echo Cleaned.
    exit /b 0
)

REM ---- Check source file ----
if not exist "%SRC%" (
    echo ERROR: Source file not found: %SRC%
    echo        Ensure quadforge_win64.c is in the same directory as this script.
    exit /b 1
)

REM ---- Determine build mode and compiler ----
set MODE=%~1
if "%MODE%"=="" set MODE=release

if /I "%MODE%"=="clang" (
    REM ---- Clang-cl build ----
    where clang-cl.exe >nul 2>&1
    if errorlevel 1 (
        echo ERROR: clang-cl.exe not found on PATH.
        echo        Install LLVM from https://github.com/llvm/llvm-project/releases
        exit /b 1
    )
    set CC=clang-cl.exe
    set CFLAGS=/O2 /DNDEBUG /MT /W3 /wd4244 /wd4267
    set LDFLAGS=/link /DLL /OUT:"%OUT%" /MACHINE:X64
    set OMP_FLAGS=-fopenmp=libomp
    echo Building quadforge.dll [CLANG-CL RELEASE]
    goto :do_clang_build
)

REM ---- MSVC cl.exe detection ----
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found on PATH.
    echo        Open a "Developer Command Prompt for VS 2022" or run:
    echo        vcvarsall.bat x64
    echo        from your Visual Studio installation.
    exit /b 1
)

REM ---- MSVC build modes ----
if /I "%MODE%"=="debug" (
    set CFLAGS=/Od /Zi /DDEBUG /MTd /W3 /wd4244 /wd4267 /RTC1
    set LDFLAGS=/link /DLL /OUT:"%OUT%" /MACHINE:X64 /DEBUG /PDB:"%SCRIPT_DIR%quadforge.pdb"
    echo Building quadforge.dll [MSVC DEBUG]
) else (
    set CFLAGS=/O2 /GL /DNDEBUG /MT /W3 /wd4244 /wd4267
    set LDFLAGS=/link /DLL /OUT:"%OUT%" /MACHINE:X64 /LTCG
    echo Building quadforge.dll [MSVC RELEASE]
)

REM ---- Check for OpenMP support (/openmp flag) ----
echo Checking OpenMP availability...
echo int main(){return 0;} > %TEMP%\qf_omp_test.c
cl.exe /nologo /openmp /c %TEMP%\qf_omp_test.c /Fo%TEMP%\qf_omp_test.obj >nul 2>&1
if errorlevel 1 (
    echo   OpenMP: not available (single-threaded build)
    set OMP_FLAGS=
) else (
    echo   OpenMP: enabled ^(/openmp flag^)
    set OMP_FLAGS=/openmp
)
del /f /q %TEMP%\qf_omp_test.c %TEMP%\qf_omp_test.obj >nul 2>&1

REM ---- Compile ----
echo.
echo Compiling: cl.exe %CFLAGS% %OMP_FLAGS% /c /Fo"%OBJ%" "%SRC%"
cl.exe /nologo %CFLAGS% %OMP_FLAGS% /c /Fo"%OBJ%" "%SRC%"
if errorlevel 1 (
    echo ERROR: Compilation failed.
    exit /b 1
)

REM ---- Link ----
echo.
echo Linking:   link.exe /DLL /OUT:"%OUT%" "%OBJ%"
REM NOTE: ucrt.lib (DLL CRT import lib) is intentionally NOT listed here.
REM The /MT flag causes cl.exe to statically link libcmt.lib + libucrt.lib
REM automatically. Explicitly adding ucrt.lib causes LNK4098 defaultlib conflict
REM and produces a mixed static/dynamic CRT binary that can crash on heap-cross-DLL.
link.exe /nologo /DLL /OUT:"%OUT%" /MACHINE:X64 /EXPORT:qf_init /EXPORT:qf_shutdown ^
    /EXPORT:qf_version /EXPORT:qf_last_error /EXPORT:qf_has_gpu ^
    /EXPORT:qf_cpu_thread_count /EXPORT:qf_remesh /EXPORT:qf_free_result ^
    /EXPORT:qf_default_params /EXPORT:qf_preset_organic /EXPORT:qf_preset_hard_surface ^
    /EXPORT:qf_preset_sculpt /EXPORT:qf_preset_architecture /EXPORT:qf_preset_fast ^
    "%OBJ%" kernel32.lib
if errorlevel 1 (
    echo ERROR: Linking failed.
    exit /b 1
)
goto :post_build

:do_clang_build
REM ---- Clang-cl compile + link (single command) ----
echo.
echo Compiling + linking: clang-cl %CFLAGS% %OMP_FLAGS% /LD /Fe"%OUT%" "%SRC%"
clang-cl.exe /nologo %CFLAGS% %OMP_FLAGS% /LD /Fe"%OUT%" "%SRC%"
if errorlevel 1 (
    echo ERROR: Clang-cl build failed.
    exit /b 1
)

:post_build
REM ---- Verify output ----
if not exist "%OUT%" (
    echo ERROR: Expected output not found: %OUT%
    exit /b 1
)

for /f "tokens=*" %%A in ('powershell -NoProfile -Command "'{0:N0} KB' -f ((Get-Item '%OUT%').Length / 1KB)"') do set SIZE=%%A
echo.
echo Built: %OUT%  (%SIZE%)

REM ---- Update CHECKSUMS ----
echo.
echo Updating CHECKSUMS...
for /f "tokens=*" %%H in ('powershell -NoProfile -Command "(Get-FileHash '%OUT%' -Algorithm SHA256).Hash.ToLower()"') do set HASH=%%H
(
echo # QuadForge win64 binary checksums
echo # Verify with ^(PowerShell^):  Get-FileHash quadforge.dll -Algorithm SHA256
echo # Verify with ^(cmd^):         certutil -hashfile quadforge.dll SHA256
echo.
echo %HASH%  quadforge.dll
) > "%SCRIPT_DIR%CHECKSUMS"
echo CHECKSUMS updated: %HASH%

REM ---- Quick symbol check ----
echo.
echo --- Exported symbols ---
where dumpbin.exe >nul 2>&1
if not errorlevel 1 (
    dumpbin.exe /exports "%OUT%" 2>nul | findstr "qf_"
) else (
    echo (dumpbin.exe not found — install Visual Studio Build Tools to see exports)
)

REM ---- Quick Python smoke test ----
echo.
echo --- Quick load test ---
where python.exe >nul 2>&1
if not errorlevel 1 (
    python.exe -c "import ctypes,os; os.add_dll_directory(r'%SCRIPT_DIR%') if hasattr(os,'add_dll_directory') else None; lib=ctypes.CDLL(r'%OUT%'); lib.qf_version.restype=ctypes.c_char_p; lib.qf_init(); ver=lib.qf_version().decode(); lib.qf_shutdown(); print('  OK  version:', ver)"
    if errorlevel 1 (
        echo   Load test FAILED — check DLL dependencies with check_deps.ps1
    )
) else (
    echo (python.exe not found on PATH — skipping load test)
)

echo.
echo Build complete.
endlocal
exit /b 0
