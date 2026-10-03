cmake_minimum_required(VERSION 3.31)
project(ESIMD VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

include(ESIMDTargetSetup)

option(ESIMD_UNITY_BUILD "Compile ESIMD as a CMake unity build" OFF)
option(ESIMD_BUILD_TESTS "Build the ESIMD tests and benchmark" OFF)

add_subdirectory(Lib/ESIMD)

if (PROJECT_IS_TOP_LEVEL OR ESIMD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif ()
