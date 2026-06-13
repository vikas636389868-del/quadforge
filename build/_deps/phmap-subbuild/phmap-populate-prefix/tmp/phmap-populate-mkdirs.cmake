# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-src")
  file(MAKE_DIRECTORY "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-build"
  "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-subbuild/phmap-populate-prefix"
  "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-subbuild/phmap-populate-prefix/tmp"
  "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-subbuild/phmap-populate-prefix/src/phmap-populate-stamp"
  "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-subbuild/phmap-populate-prefix/src"
  "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-subbuild/phmap-populate-prefix/src/phmap-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-subbuild/phmap-populate-prefix/src/phmap-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/77yad/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/quadforge/build/_deps/phmap-subbuild/phmap-populate-prefix/src/phmap-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
