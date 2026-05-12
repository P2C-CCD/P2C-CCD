cmake_minimum_required(VERSION 3.24)

project(P2CCCD VERSION 0.1.0 LANGUAGES CXX)

option(P2CCCD_ENABLE_CUDA "Enable CUDA sources." OFF)
option(P2CCCD_ENABLE_OPTIX "Enable optional OptiX RT candidate backend." OFF)
option(P2CCCD_ENABLE_PYTHON "Build Python bindings with pybind11." OFF)
option(P2CCCD_BUILD_TESTS "Build P2CCCD C++ tests." ON)
option(P2CCCD_BUILD_TOOLS "Build P2CCCD developer tools." ON)

if(NOT DEFINED P2CCCD_OPTIX_ROOT_INITIALIZED)
  set(_p2cccd_default_optix_root "")
  foreach(_p2cccd_optix_candidate "$ENV{P2CCCD_OPTIX_ROOT}" "$ENV{OPTIX_ROOT}")
    if(_p2cccd_optix_candidate AND EXISTS "${_p2cccd_optix_candidate}/include/optix.h")
      set(_p2cccd_default_optix_root "${_p2cccd_optix_candidate}")
      break()
    endif()
  endforeach()
  set(P2CCCD_OPTIX_ROOT "${_p2cccd_default_optix_root}" CACHE PATH "Path to the OptiX SDK root.")
  unset(_p2cccd_default_optix_root)
  unset(_p2cccd_optix_candidate)
  set(P2CCCD_OPTIX_ROOT_INITIALIZED ON CACHE INTERNAL "P2CCCD OptiX root initialized")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(P2CCCD_ENABLE_OPTIX AND NOT P2CCCD_ENABLE_CUDA)
  set(P2CCCD_ENABLE_CUDA ON CACHE BOOL "Enable CUDA sources." FORCE)
endif()

if(P2CCCD_ENABLE_CUDA OR P2CCCD_ENABLE_OPTIX)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
endif()

if(P2CCCD_ENABLE_OPTIX)
  find_path(P2CCCD_OPTIX_INCLUDE_DIR
    NAMES optix.h
    HINTS
      ${P2CCCD_OPTIX_ROOT}
    PATH_SUFFIXES
      include
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

if(P2CCCD_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(cpp)
