cmake_minimum_required(VERSION 3.13)
project(OpenCL_FFT)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_VERBOSE_MAKEFILE 1)

add_subdirectory(opencl)

if(APPLE)
  add_subdirectory(metal)
endif()
