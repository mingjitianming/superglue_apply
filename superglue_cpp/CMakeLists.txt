cmake_minimum_required(VERSION 3.11)
project(superglue 
        LANGUAGES CXX
        VERSION 1.0.0)

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

# set output dir 
include(GNUInstallDirs)
# set build output dir
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
#set install output dir
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for header files")
set(INSTALL_CMAKEDIR share/cmake/${PROJECT_NAME} CACHE PATH "Installation directory for CMake files")

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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OPENMP)
        message(STATUS "OpenMP: ENABLED")
    else()
        message(STATUS "OpenMP: DISABLED")
    endif()
else()
    message(STATUS "OpenMP: DISABLED")
endif()

add_library(superpoint SHARED
    src/superpoint.cpp
)


# list(APPEND TORCH_INSTALL_INCLUDE_DIRS)
# foreach(dir ${TORCH_INCLUDE_DIRS})
#     file(RELATIVE_PATH _rel ${CMAKE_INSTALL_PREFIX}/lib ${dir})
#     list(APPEND TORCH_INSTALL_INCLUDE_DIRS ${_rel})
# endforeach()
# string(REPLACE " " ";" TORCH_INSTALL_INCLUDE_DIRSS ${TORCH_INSTALL_INCLUDE_DIRS})

file(RELATIVE_PATH _rel ${CMAKE_INSTALL_PREFIX}/${INSTALL_BINDIR} ${CMAKE_INSTALL_PREFIX})
set(_rpath "\$ORIGIN/${_rel}")
file(TO_NATIVE_PATH "${_rpath}/${INSTALL_LIBDIR}" super_RPATH)

target_include_directories(superpoint
                           PUBLIC
                           $<BUILD_INTERFACE: include>
                           $<BUILD_INTERFACE: ${TORCH_INCLUDE_DIRS}>
                           $<BUILD_INTERFACE: ${OpenCV_INCLUDE_DIRS}>
                           $<BUILD_INTERFACE: ${PROJECT_SOURCE_DIR}/3rd/spdlog/include>
                           $<INSTALL_INTERFACE:include>
                           $<INSTALL_INTERFACE:../3rd/spdlog/include>
                           )

target_link_libraries(superpoint 
                PUBLIC
                ${TORCH_LIBRARIES}
                ${OpenCV_LIBS}
                yaml-cpp
                )

set_target_properties(superpoint
            PROPERTIES
                POSITION_INDEPENDENT_CODE 1
                DEBUG_POSTFIX "_sd"
                RELEASE_POSTFIX "_s"
                # SKIP_BUILD_RPATH OFF
                # BUILD_WITH_INSTALL_RPATH OFF
                # BUILD_RPATH "\$ORIGIN/../../3rd/libtorch/lib"
                # INSTALL_RPATH_USE_LINK_PATH ON
                )
add_library(superglue SHARED
    src/superglue.cpp
)

target_include_directories(superglue
                           PUBLIC
                           $<BUILD_INTERFACE: include>
                           $<BUILD_INTERFACE: ${TORCH_INCLUDE_DIRS}>
                           $<BUILD_INTERFACE: ${OpenCV_INCLUDE_DIRS}>
                           $<BUILD_INTERFACE: ${PROJECT_SOURCE_DIR}/3rd/spdlog/include>
                           $<INSTALL_INTERFACE:include>
                           $<INSTALL_INTERFACE:../3rd/spdlog/include>
                           )


target_link_libraries(superglue 
                     INTERFACE
                     ${TORCH_LIBRARIES}
                     ${OpenCV_LIBS}
                     yaml-cpp)

set_target_properties(superglue
            PROPERTIES
                POSITION_INDEPENDENT_CODE 1
                DEBUG_POSTFIX "_sd"
                RELEASE_POSTFIX "_s"
                # SKIP_BUILD_RPATH OFF
                # BUILD_WITH_INSTALL_RPATH OFF
                # BUILD_RPATH "\$ORIGIN/../../3rd/libtorch/lib;"
                # INSTALL_RPATH_USE_LINK_PATH ON 
                )                    

add_subdirectory(examples)

# export(TARGETS superpoint superglue
#        FILE  cmake/superglueConfig.cmake)


# set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
# export(PACKAGE superpoint)
# message(STATUS ${EXPORT_LIB_INC_DIR})

install(TARGETS superpoint superglue
        EXPORT superglueTargets
        LIBRARY DESTINATION ${INSTALL_LIBDIR} COMPONENT lib
        ARCHIVE DESTINATION ${INSTALL_LIBDIR} COMPONENT lib
        RUNTIME DESTINATION ${INSTALL_BINDIR} COMPONENT bin
        )
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/*.h)
install(FILES ${HEADERS} DESTINATION include)

install(EXPORT superglueTargets
        NAMESPACE "superglue::"
        DESTINATION ${INSTALL_CMAKEDIR}
        COMPONENT dev)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${INSTALL_CMAKEDIR}
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${INSTALL_CMAKEDIR}
)


#################################################################
## using cmake
#
#set(superglue_DIR "path of superglueConfig.cmake")
#find_package(superglue)
#
#################################################################


