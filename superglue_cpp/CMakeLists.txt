cmake_minimum_required(VERSION 3.0)
project(superglue)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/libtorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(include
                    ${TORCH_INCLUDE_DIRS}
                    ${${OpenCV_INCLUDE_DIRS}})

add_library(superpoint SHARED
    src/superpoint.cpp
)

target_link_libraries(superpoint 
                ${TORCH_INCLUDE_DIRS}
                ${OpenCV_LIBS}
                yaml-cpp
                )