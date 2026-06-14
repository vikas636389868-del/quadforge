cmake_minimum_required(VERSION 3.20)

# MUST be before project
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
cmake_policy(SET CMP0091 NEW)

project(QuadForge VERSION 14.0.0 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/build/cmake")

# -----------------------------------------------------------------------
# C++ Standard
# -----------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------
# CMake module path
# -----------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# include(${CMAKE_SOURCE_DIR}/PlatformSetup.cmake) # File missing, so commented out

# -----------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------
option(QF_ENABLE_OPENMP   "Enable OpenMP parallelism"              OFF)
option(QF_ENABLE_CHOLMOD  "Enable SuiteSparse/CHOLMOD (optional)"  OFF)
option(QF_ENABLE_CUDA     "Enable CUDA GPU solver (optional)"      OFF)
option(QF_BUILD_TESTS     "Build test suite"                       OFF)
option(QF_STATIC_RUNTIME  "Statically link C++ runtime"            OFF)
option(QF_NATIVE_MARCH    "Build with -march=native"               OFF)

# -----------------------------------------------------------------------
# Third-party dependencies
# -----------------------------------------------------------------------
# 1. Sabse pehle paths set karein
set(QF_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/engine/third_party")
set(EIGEN3_INCLUDE_DIR "${QF_THIRD_PARTY_DIR}/eigen")
set(NANOFLANN_INCLUDE_DIR "${QF_THIRD_PARTY_DIR}/nanoflann")
set(PHMAP_INCLUDE_DIR "${QF_THIRD_PARTY_DIR}/parallel_hashmap")

# 2. Ab target_include_directories mein in variables ka use karein
target_include_directories(quadforge SYSTEM PRIVATE 
    "${EIGEN3_INCLUDE_DIR}" 
    "${NANOFLANN_INCLUDE_DIR}" 
    "${PHMAP_INCLUDE_DIR}"
)

# -----------------------------------------------------------------------
# OpenMP & SuiteSparse
# -----------------------------------------------------------------------
if(QF_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

if(QF_ENABLE_CHOLMOD)
  find_package(SuiteSparse REQUIRED)
endif()

# -----------------------------------------------------------------------
# Source files & Target
# -----------------------------------------------------------------------
set(ENGINE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/engine/src")
set(ENGINE_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/engine/include")

file(GLOB_RECURSE QF_SOURCES "${ENGINE_SRC_DIR}/*.cpp")

add_library(quadforge SHARED ${QF_SOURCES})

#target_include_directories(quadforge PUBLIC "${ENGINE_INC_DIR}" PRIVATE "${EIGEN3_INCLUDE_DIR}" "${NANOFLANN_INCLUDE_DIR}" "${PHMAP_INCLUDE_DIR}")
target_include_directories(quadforge PUBLIC "${ENGINE_INC_DIR}")
target_include_directories(quadforge SYSTEM PRIVATE 
    "${EIGEN3_INCLUDE_DIR}" 
    "${NANOFLANN_INCLUDE_DIR}" 
    "${PHMAP_INCLUDE_DIR}"
)
set_target_properties(quadforge PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
)

if(WIN32)
  set_target_properties(quadforge PROPERTIES OUTPUT_NAME "quadforge" SUFFIX ".dll" PREFIX "")
elseif(APPLE)
  set_target_properties(quadforge PROPERTIES OUTPUT_NAME "libquadforge" SUFFIX ".dylib" PREFIX "" MACOSX_RPATH ON BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "@loader_path")
else()
  set_target_properties(quadforge PROPERTIES OUTPUT_NAME "libquadforge" SUFFIX ".so" PREFIX "")
endif()

if(QF_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(quadforge PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(quadforge PRIVATE QF_HAS_OPENMP=1)
endif()

if(WIN32)
  add_custom_command(TARGET quadforge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:quadforge>
            "${CMAKE_CURRENT_SOURCE_DIR}/engine/prebuilt/win64/quadforge.dll"
    COMMENT "Copying quadforge.dll into the Blender addon prebuilt directory"
  )
endif()

# -----------------------------------------------------------------------
# Install & Summary
# -----------------------------------------------------------------------
if(WIN32)
  set(QF_PLATFORM_DIR "win64")
elseif(APPLE)
  set(QF_PLATFORM_DIR "macos_arm64")
else()
  set(QF_PLATFORM_DIR "linux64")
endif()

install(TARGETS quadforge LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/engine/prebuilt/${QF_PLATFORM_DIR}" RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/engine/prebuilt/${QF_PLATFORM_DIR}")
