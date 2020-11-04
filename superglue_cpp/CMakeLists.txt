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

# add_compile_options(-fPIC)

# set workspace path
set(WORKSPACE_DIR ${PROJECT_SOURCE_DIR})
configure_file("${PROJECT_SOURCE_DIR}/include/config.h.in" "${PROJECT_SOURCE_DIR}/include/config.h")

list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/3rd/libtorch)

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install)

#----- find package ------
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OPENMP)
        message(STATUS "OpenMP: ENABLED")
    else()
        message(STATUS "OpenMP: DISABLED")
    endif()
else()
    message(STATUS "OpenMP: DISABLED")
endif()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
message(STATUS "Library out path:" ${PROJECT_BINARY_DIR}/lib)
add_library(superpoint SHARED
    src/superpoint.cpp
)

target_include_directories(superpoint
                           PUBLIC
                           $<BUILD_INTERFACE: include>
                           $<BUILD_INTERFACE: ${TORCH_INCLUDE_DIRS}>
                           $<BUILD_INTERFACE: ${OpenCV_INCLUDE_DIRS}>
                           $<BUILD_INTERFACE: ${PROJECT_SOURCE_DIR}/3rd/spdlog/include>
                        #   $<INSTALL_INTERFACE:include>
                        #   $<INSTALL_INTERFACE:${OpenCV_INCLUDE_DIRS}>
                        #    $<INSTALL_INTERFACE:${TORCH_INCLUDE_DIRS}>
                        #    $<BUILD_INTERFACE: include/spdlog/include>
                           )

# set_target_properties(superpoint PROPERTIES
#                       PUBLIC_HEADER ${HEADERS}
#                     #   LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
#                       )

target_link_libraries(superpoint 
                PUBLIC
                ${TORCH_LIBRARIES}
                ${OpenCV_LIBS}
                yaml-cpp
                )
set_target_properties(superpoint
            PROPERTIES
                POSITION_INDEPENDENT_CODE 1
                SKIP_BUILD_RPATH OFF
                BUILD_WITH_INSTALL_RPATH OFF
                BUILD_RPATH "../../3rd/libtorch/lib"
                INSTALL_RPATH_USE_LINK_PATH ON     )
add_library(superglue SHARED
    src/superglue.cpp
)

target_include_directories(superglue
                           PUBLIC
                           $<BUILD_INTERFACE: include>
                           $<BUILD_INTERFACE: ${TORCH_INCLUDE_DIRS}>
                           $<BUILD_INTERFACE: ${OpenCV_INCLUDE_DIRS}>
                           $<BUILD_INTERFACE: ${PROJECT_SOURCE_DIR}/3rd/spdlog/include>
                        #    $<INSTALL_INTERFACE:include>
                           )


target_link_libraries(superglue 
                     INTERFACE
                     ${TORCH_LIBRARIES}
                     ${OpenCV_LIBS}
                     yaml-cpp)

set_target_properties(superglue
            PROPERTIES
                POSITION_INDEPENDENT_CODE 1
                SKIP_BUILD_RPATH OFF
                BUILD_WITH_INSTALL_RPATH OFF
                BUILD_RPATH "../../3rd/libtorch/lib"
                INSTALL_RPATH_USE_LINK_PATH ON     )                    

add_subdirectory(examples)

export(TARGETS superpoint superglue
       FILE  cmake/superglueConfig.cmake)


# set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
# export(PACKAGE superpoint)
# message(STATUS ${EXPORT_LIB_INC_DIR})

# install(TARGETS superpoint
#         EXPORT superpointTargets
#         LIBRARY DESTINATION lib
#         )
# file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/*.h)
# install(FILES ${HEADERS} DESTINATION include)

# install(EXPORT superpointTargets
#         FILE superpointTargets.cmake
#         DESTINATION lib/cmake/superpoint)   
