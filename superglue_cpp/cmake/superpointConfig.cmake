# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS 2.5)
   message(FATAL_ERROR "CMake >= 2.6.0 required")
endif()
cmake_policy(PUSH)
cmake_policy(VERSION 2.6...3.17)
#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Protect against multiple inclusion, which would fail when already imported targets are added once more.
set(_targetsDefined)
set(_targetsNotDefined)
set(_expectedTargets)
foreach(_expectedTarget superpoint)
  list(APPEND _expectedTargets ${_expectedTarget})
  if(NOT TARGET ${_expectedTarget})
    list(APPEND _targetsNotDefined ${_expectedTarget})
  endif()
  if(TARGET ${_expectedTarget})
    list(APPEND _targetsDefined ${_expectedTarget})
  endif()
endforeach()
if("${_targetsDefined}" STREQUAL "${_expectedTargets}")
  unset(_targetsDefined)
  unset(_targetsNotDefined)
  unset(_expectedTargets)
  set(CMAKE_IMPORT_FILE_VERSION)
  cmake_policy(POP)
  return()
endif()
if(NOT "${_targetsDefined}" STREQUAL "")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_targetsDefined}\nTargets not yet defined: ${_targetsNotDefined}\n")
endif()
unset(_targetsDefined)
unset(_targetsNotDefined)
unset(_expectedTargets)


# Create imported target superpoint
add_library(superpoint SHARED IMPORTED)

set_target_properties(superpoint PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "/home/zmy/project_ws/superglue_apply/superglue_cpp/;/home/zmy/project_ws/superglue_apply/superglue_cpp/include;/home/zmy/project_ws/superglue_apply/superglue_cpp/;/home/zmy/project_ws/superglue_apply/superglue_cpp/3rd/libtorch/include;/home/zmy/project_ws/superglue_apply/superglue_cpp/3rd/libtorch/include/torch/csrc/api/include;/home/zmy/project_ws/superglue_apply/superglue_cpp/;/usr/local/include;/usr/local/include/opencv;/home/zmy/project_ws/superglue_apply/superglue_cpp/;/home/zmy/project_ws/superglue_apply/superglue_cpp/3rd/spdlog/include"
  INTERFACE_LINK_LIBRARIES "/home/zmy/project_ws/superglue_apply/superglue_cpp/3rd/libtorch/lib/libtorch.so;/home/zmy/project_ws/superglue_apply/superglue_cpp/3rd/libtorch/lib/libc10.so;/usr/local/cuda-10.2/lib64/stubs/libcuda.so;/usr/local/cuda-10.2/lib64/libnvrtc.so;/usr/local/cuda-10.2/lib64/libnvToolsExt.so;/usr/local/cuda-10.2/lib64/libcudart.so;/home/zmy/project_ws/superglue_apply/superglue_cpp/3rd/libtorch/lib/libc10_cuda.so;opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab;opencv_viz;yaml-cpp"
)

# Import target "superpoint" for configuration "Debug"
set_property(TARGET superpoint APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(superpoint PROPERTIES
  IMPORTED_LOCATION_DEBUG "/home/zmy/project_ws/superglue_apply/superglue_cpp/build/lib/libsuperpoint.so"
  IMPORTED_SONAME_DEBUG "libsuperpoint.so"
  )

# This file does not depend on other imported targets which have
# been exported from the same project but in a separate export set.

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
