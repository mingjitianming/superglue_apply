cmake_minimum_required(VERSION 3.0)
project(superglue)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set Debug modle
set(CMAKE_BUILD_TYPE "Debug")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

#set c++ compiler
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
check_cxx_compiler_flag("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "C++11 support: OK (-std=c++17)")
elseif(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_STANDARD 14)
    message(STATUS "C++11 support: OK (-std=c++14)")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set workspace path
set(WORKSPACE_DIR ${PROJECT_SOURCE_DIR})
configure_file("${PROJECT_SOURCE_DIR}/include/config.h.in" "${PROJECT_SOURCE_DIR}/include/config.h")

list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/libtorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(include
                    ${TORCH_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/3rd/spdlog/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
message("library out path:" ${PROJECT_BINARY_DIR}/lib)
add_library(superpoint SHARED
    src/superpoint.cpp
)
# set_target_properties(superpoint PROPERTIES
#                       OUTPUT_NAME superpoint
#                       LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

target_link_libraries(superpoint 
                PUBLIC
                ${TORCH_LIBRARIES}
                ${OpenCV_LIBS}
                yaml-cpp
                )
add_subdirectory(examples)