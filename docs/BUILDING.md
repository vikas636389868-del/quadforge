# Building QuadForge

This guide explains how to build QuadForge from source on Windows, Linux, and macOS with CMake.

---

## Prerequisites

Install the following before building:

- CMake 3.20+
- Python 3.10–3.11
- Git
- A C++17-capable compiler and build tools
- Blender (optional, for integration testing)

### Windows
- Visual Studio 2022 with the Desktop development with C++ workload
- CMake for Windows

### Linux (Ubuntu 22.04+)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev libgomp1 libomp-dev
```

### macOS (13+)
```bash
xcode-select --install
brew install cmake libomp
```

---

## Windows Build

1. Clone the repository and fetch submodules:
```bash
git clone https://github.com/vikas636389868-del/quadforge.git
cd quadforge
git submodule update --init --recursive
```

2. Open a Developer Command Prompt for Visual Studio 2022.

3. Configure and build the native library:
```bat
mkdir build\_cmake_win && cd build\_cmake_win
cmake ..\.. -G "Visual Studio 17 2022" -A x64 ^
  -DQF_ENABLE_OPENMP=ON ^
  -DQF_ENABLE_CHOLMOD=OFF ^
  -DQF_NATIVE_MARCH=OFF
cmake --build . --config Release --target quadforge
```

4. Copy the built DLL into the add-on tree:
```bat
copy Release\quadforge.dll ..\..\QuadForge\engine\prebuilt\win64\quadforge.dll
```

5. Verify the binary:
```bat
python QuadForge\engine\prebuilt\win64\verify_api.py
python QuadForge\engine\prebuilt\win64\compat_check.py
```

### Windows troubleshooting
- If CMake complains about a missing Visual Studio generator, re-run from the Developer Command Prompt.
- If the DLL is not found at runtime, ensure the file is copied to the correct platform folder.

---

## Linux Build

1. Clone the repository and fetch submodules:
```bash
git clone https://github.com/vikas636389868-del/quadforge.git
cd quadforge
git submodule update --init --recursive
```

2. Configure and build:
```bash
mkdir -p build/_cmake_linux && cd build/_cmake_linux
cmake ../.. \
  -DCMAKE_BUILD_TYPE=Release \
  -DQF_ENABLE_OPENMP=ON \
  -DQF_ENABLE_CHOLMOD=OFF \
  -DQF_NATIVE_MARCH=OFF
make -j$(nproc) quadforge
```

3. Copy the shared library into the prebuilt folder:
```bash
cp libquadforge.so ../../QuadForge/engine/prebuilt/linux64/libquadforge.so
```

4. Verify the binary:
```bash
cd ../../QuadForge/engine/prebuilt/linux64
python verify_api.py
python compat_check.py
```

### Linux troubleshooting
- If OpenMP symbols are missing, install `libomp-dev`.
- If CMake cannot find the compiler, install `build-essential`.

---

## macOS Build

1. Clone the repository and fetch submodules:
```bash
git clone https://github.com/vikas636389868-del/quadforge.git
cd quadforge
git submodule update --init --recursive
```

2. Configure and build for Apple Silicon:
```bash
mkdir -p build/_cmake_macos_arm64 && cd build/_cmake_macos_arm64
cmake ../.. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DQF_ENABLE_OPENMP=ON \
  -DQF_ENABLE_CHOLMOD=OFF \
  -DQF_NATIVE_MARCH=OFF
make -j$(sysctl -n hw.logicalcpu) quadforge
```

3. Copy the built library:
```bash
cp libquadforge.dylib ../../QuadForge/engine/prebuilt/macos_arm64/libquadforge.dylib
install_name_tool -id @rpath/libquadforge.dylib ../../QuadForge/engine/prebuilt/macos_arm64/libquadforge.dylib
```

4. Verify the binary:
```bash
cd ../../QuadForge/engine/prebuilt/macos_arm64
python verify_api.py
python compat_check.py
```

For Intel Macs, change the architecture flag to `-DCMAKE_OSX_ARCHITECTURES=x86_64` and copy to the `macos_x64` folder.

### macOS troubleshooting
- If the library is blocked by Gatekeeper, run:
```bash
xattr -d com.apple.quarantine QuadForge/engine/prebuilt/macos_arm64/libquadforge.dylib
```
- If Blender cannot load it, confirm the build target matches your Mac architecture.

---

## Packaging the Add-on

After the native binaries are in place, package the add-on for Blender:

```bash
python build/scripts/package_addon.py --platform auto --version 1.0.0 --output dist/
```

The generated zip should contain a flat `QuadForge/` folder suitable for Blender's add-on installer.

---

## Installing in Blender

1. Open Blender.
2. Go to Edit → Preferences → Add-ons → Install (Blender 4.1 and earlier) or Edit → Preferences → Extensions → Install from Disk (Blender 4.2+).
3. Select the packaged `.zip` file.
4. Enable the **QuadForge** add-on.
5. The QuadForge panel appears in the 3D Viewport sidebar under the **QuadForge** tab.

---

## Running Tests

```bash
pip install pytest numpy scipy
pytest tests/ -v --ignore=tests/test_pipeline_e2e.cpp
```

---

## Notes

- QuadForge can run in pure-Python mode without compiling the native engine, but the C++ library is recommended for the full performance experience.
- The build scripts in the `build/` folder are the preferred way to produce release zips for each platform.

