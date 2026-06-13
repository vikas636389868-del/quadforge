# check_deps.ps1 — Verify runtime dependencies for quadforge.dll on Windows.
# Run this if Blender reports "Failed to load native library".
#
# Usage (from PowerShell):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   & "QuadForge\engine\prebuilt\win64\check_deps.ps1"
#
# Or from Command Prompt:
#   powershell -ExecutionPolicy Bypass -File "QuadForge\engine\prebuilt\win64\check_deps.ps1"

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DllPath   = Join-Path $ScriptDir "quadforge.dll"

Write-Host "=== QuadForge win64 dependency check ===" -ForegroundColor Cyan
Write-Host ""

# ---- Check DLL presence ----
if (-not (Test-Path $DllPath)) {
    Write-Host "ERROR: quadforge.dll not found in:" -ForegroundColor Red
    Write-Host "       $ScriptDir"
    Write-Host ""
    Write-Host "Solution: build the DLL or download a pre-built binary."
    Write-Host "See MISSING_BINARY.md for detailed instructions."
    exit 1
}

$dllInfo = Get-Item $DllPath
Write-Host "Library: $DllPath"
Write-Host ("Size:    {0:N0} KB" -f ($dllInfo.Length / 1KB))
Write-Host ("Date:    {0}" -f $dllInfo.LastWriteTime)
Write-Host ""

# ---- Architecture check ----
Write-Host "--- Architecture ---"
$arch = (Get-Process -Id $PID).StartInfo
if ([System.Environment]::Is64BitProcess) {
    Write-Host "  [OK] Running 64-bit Python — correct for quadforge.dll (x64)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Running 32-bit Python — quadforge.dll is 64-bit only!" -ForegroundColor Yellow
    Write-Host "         Install 64-bit Python to use the native engine."
}
Write-Host ""

# ---- SHA-256 checksum ----
Write-Host "--- SHA-256 checksum ---"
$hash = (Get-FileHash -Path $DllPath -Algorithm SHA256).Hash.ToLower()
Write-Host "  DLL hash:  $hash"
$checksumsPath = Join-Path $ScriptDir "CHECKSUMS"
if (Test-Path $checksumsPath) {
    $reference = Get-Content $checksumsPath | Where-Object { $_ -match "quadforge\.dll" } |
                 ForEach-Object { ($_ -split "\s+")[0] }
    if ($reference) {
        if ($hash -eq $reference.ToLower()) {
            Write-Host "  Reference: $reference" -ForegroundColor Green
            Write-Host "  [OK] Checksum matches CHECKSUMS file" -ForegroundColor Green
        } else {
            Write-Host "  Reference: $reference" -ForegroundColor Red
            Write-Host "  [FAIL] Checksum mismatch — DLL may be corrupted!" -ForegroundColor Red
        }
    }
} else {
    Write-Host "  (CHECKSUMS file not found — skipping reference comparison)"
}
Write-Host ""

# ---- Visual C++ Redistributable ----
Write-Host "--- Visual C++ Runtime ---"
$vcRuntimeKeys = @(
    "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
)
$vcFound = $false
foreach ($key in $vcRuntimeKeys) {
    if (Test-Path $key) {
        $ver = (Get-ItemProperty $key -ErrorAction SilentlyContinue).Version
        Write-Host "  [OK] Visual C++ 2015-2022 Redistributable found: $ver" -ForegroundColor Green
        $vcFound = $true
        break
    }
}
if (-not $vcFound) {
    # Fall back to checking System32 directly
    $vcrt = Join-Path $env:SystemRoot "System32\VCRUNTIME140.dll"
    if (Test-Path $vcrt) {
        Write-Host "  [OK] VCRUNTIME140.dll found in System32" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Visual C++ Redistributable not detected." -ForegroundColor Yellow
        Write-Host "         Download: https://aka.ms/vs/17/release/vc_redist.x64.exe"
    }
}
Write-Host ""

# ---- OpenMP runtime ----
Write-Host "--- OpenMP Runtime ---"
$ompCandidates = @(
    (Join-Path $env:SystemRoot "System32\vcomp140.dll"),
    (Join-Path $env:SystemRoot "SysWOW64\vcomp140.dll"),
    (Join-Path $env:SystemRoot "System32\libomp.dll")
)
$ompFound = $false
foreach ($candidate in $ompCandidates) {
    if (Test-Path $candidate) {
        Write-Host "  [OK] OpenMP runtime found: $candidate" -ForegroundColor Green
        $ompFound = $true
        break
    }
}
# Also check PATH
if (-not $ompFound) {
    $pathDirs = $env:PATH -split ";"
    foreach ($dir in $pathDirs) {
        $vcomp = Join-Path $dir "vcomp140.dll"
        $libomp = Join-Path $dir "libomp.dll"
        if (Test-Path $vcomp) {
            Write-Host "  [OK] vcomp140.dll found on PATH: $vcomp" -ForegroundColor Green
            $ompFound = $true
            break
        }
        if (Test-Path $libomp) {
            Write-Host "  [OK] libomp.dll found on PATH: $libomp" -ForegroundColor Green
            $ompFound = $true
            break
        }
    }
}
if (-not $ompFound) {
    Write-Host "  [WARN] vcomp140.dll / libomp.dll not found." -ForegroundColor Yellow
    Write-Host "         OpenMP parallelism will be disabled (single-threaded)."
    Write-Host "         Install the Visual C++ Redistributable to enable OpenMP:"
    Write-Host "         https://aka.ms/vs/17/release/vc_redist.x64.exe"
}
Write-Host ""

# ---- DLL dependency analysis (if dumpbin is available) ----
Write-Host "--- DLL Import Table (dumpbin) ---"
$dumpbin = Get-Command "dumpbin.exe" -ErrorAction SilentlyContinue
if ($dumpbin) {
    Write-Host "  Running: dumpbin /dependents $DllPath"
    & dumpbin.exe /dependents $DllPath 2>&1 | Where-Object { $_ -match "\.dll" } |
        ForEach-Object { Write-Host "    $_" }
} else {
    Write-Host "  (dumpbin.exe not found — install Visual Studio Build Tools to enable)"
    Write-Host "  Alternative: run 'Dependencies.exe' (open-source DLL dependency walker)"
}
Write-Host ""

# ---- Quick Python load test ----
Write-Host "--- Quick load test (Python ctypes) ---"
$python = Get-Command "python.exe" -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command "python3.exe" -ErrorAction SilentlyContinue }

if ($python) {
    $pyScript = @"
import ctypes, sys, os
dll = r'$($DllPath.Replace('\','\\'))'
try:
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(r'$($ScriptDir.Replace('\','\\'))')
    lib = ctypes.CDLL(dll)
    lib.qf_version.restype = ctypes.c_char_p
    lib.qf_init()
    ver = lib.qf_version().decode('utf-8', 'replace')
    lib.qf_shutdown()
    print(f'[OK] Load succeeded — version: {ver}')
except Exception as e:
    print(f'[FAIL] Load failed: {e}', file=sys.stderr)
    sys.exit(1)
"@
    $result = & python.exe -c $pyScript 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $result" -ForegroundColor Green
    } else {
        Write-Host "  $result" -ForegroundColor Red
    }
} else {
    Write-Host "  (python.exe not found on PATH — skipping load test)"
}
Write-Host ""

Write-Host "=== Dependency check complete ===" -ForegroundColor Cyan
