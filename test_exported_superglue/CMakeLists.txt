cmake_minimum_required(VERSION 3.9)
project(test_superglue)

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


set(superglue_DIR "/home/zmy/project_ws/superglue_apply/superglue_cpp/install/share/cmake/superglue")

find_package(superglue REQUIRED)


add_executable(${PROJECT_NAME}
                src/test.cpp)
target_link_libraries(${PROJECT_NAME}
                    PUBLIC
                    superglue::superpoint
                    )





