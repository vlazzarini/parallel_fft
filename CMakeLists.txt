cmake_minimum_required(VERSION 3.13)
project(OpenCL_FFT)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -- Global optimisation flags --
# Release: -O3, LTO, architecture-native tuning
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto=auto" CACHE STRING "Release CXX flags")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -flto=auto" CACHE STRING "Release C flags")
# On Apple Silicon, -mcpu=apple-m1 (or -march=armv8.5-a) would be ideal,
# but we keep it generic for portability; users can set -march=native via env.
if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Build architectures")
endif()

set(CMAKE_VERBOSE_MAKEFILE 1)

add_subdirectory(opencl)

if(APPLE)
  add_subdirectory(metal)
endif()
