# PlatformSetup.cmake — platform-specific compiler and linker flags

if(WIN32)
  # Windows: MSVC or clang-cl
  add_compile_definitions(
    _USE_MATH_DEFINES           # M_PI etc.
    NOMINMAX                    # don't let windows.h pollute min/max
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
  )

  if(MSVC)
    # Use /MD in Release, /MDd in Debug (dynamic CRT — matches Blender)
   set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
  endif()

elseif(APPLE)
  # macOS: target 13.0+ (Ventura) for universal binary capability
  set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Min macOS version" FORCE)

  # Build a universal binary only when explicitly requested
  if(DEFINED QF_UNIVERSAL AND QF_UNIVERSAL)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    message(STATUS "Building universal binary (arm64 + x86_64)")
  else()
    message(STATUS "Building for host architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  # Use libc++ (Clang's stdlib) — required for compatibility with Blender's Python
  add_compile_options(-stdlib=libc++)
  add_link_options(-stdlib=libc++)

  # Suppress common macOS SDK warnings
  add_compile_options(-Wno-deprecated-declarations)

else()
  # Linux: target Ubuntu 20.04 glibc (2.31) for maximum compatibility
  add_compile_options(-fPIC)

  # Prefer gold or lld linker for faster link times
  find_program(LLD_LINKER "lld")
  if(LLD_LINKER)
    add_link_options(-fuse-ld=lld)
    message(STATUS "Using lld linker")
  else()
    find_program(GOLD_LINKER "ld.gold")
    if(GOLD_LINKER)
      add_link_options(-fuse-ld=gold)
      message(STATUS "Using gold linker")
    endif()
  endif()
endif()
