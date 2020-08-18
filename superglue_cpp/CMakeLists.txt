cmake_minimum_required(VERSION 3.9)
project(superglue)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#----- configuration----
set(CMAKE_BUILD_TYPE "Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
set(USE_OPENMP TRUE CACHE BOOL "use OpenMp")

#---- set Debug modle -----
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

#set c++ compiler
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "C++11 support: OK (-std=c++17)")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set workspace path
set(WORKSPACE_DIR ${PROJECT_SOURCE_DIR})
configure_file("${PROJECT_SOURCE_DIR}/include/config.h.in" "${PROJECT_SOURCE_DIR}/include/config.h")

list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/3rd/libtorch)

#----- find package ------
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_FOUND)
        set(CMAKE_CXX_FLAG "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OPENMP)
        message(STATUS "OpenMP: ENABLED")
    else()
        message(STATUS "OpenMP: DISABLED")
    endif()
else()
    message(STATUS "OpenMP: DISABLED")
endif()

include_directories(include
                    ${TORCH_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/3rd/spdlog/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
message(STATUS "Library out path:" ${PROJECT_BINARY_DIR}/lib)
add_library(superpoint SHARED
    src/superpoint.cpp
)
# set_target_properties(superpoint PROPERTIES
#                       OUTPUT_NAME superpoint
#                       LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

target_link_libraries(superpoint 
                INTERFACE
                ${TORCH_LIBRARIES}
                ${OpenCV_LIBS}
                yaml-cpp
                )

add_library(superglue SHARED
    src/superglue.cpp
)
target_link_libraries(superglue 
                     INTERFACE
                     ${TORCH_LIBRARIES}
                     ${OpenCV_LIBS}
                     yaml-cpp)

add_subdirectory(examples)