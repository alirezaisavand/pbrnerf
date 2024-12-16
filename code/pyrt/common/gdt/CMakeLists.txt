if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(GPU_Development_Tools)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(gdt 
  cmake/configure_build_type.cmake
  cmake/configure_optix.cmake
  cmake/FindOptiX.cmake
  
  gdt/gdt.h
  gdt/math/LinearSpace.h
  gdt/math/AffineSpace.h
  
  gdt/gdt.cpp
  )

