cmake_minimum_required(VERSION 3.16)
project(2D_IPC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

# ── Core library ──────────────────────────────────────────────────────────────
add_library(ipc_core
    src/physics.cpp
    src/chain.cpp
    src/visualization.cpp
    src/broad_phase/bvh.cpp
    src/step_filter/ccd.cpp
    src/step_filter/trust_region.cpp
    src/initial_guess/trivial.cpp
    src/initial_guess/affine.cpp
    src/initial_guess/ccd.cpp
    src/initial_guess/trust_region.cpp
    src/solver.cpp
    src/example.cpp
    src/initial_guess/initial_guess.cpp
)

target_include_directories(ipc_core PUBLIC src)

# Eigen Library
include(FetchContent)

set(EIGEN_BUILD_DOC OFF CACHE INTERNAL "")
set(EIGEN_BUILD_TESTING OFF CACHE INTERNAL "")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE INTERNAL "")
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_Declare(
    eigen3
    URL      https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3)
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
target_include_directories(Eigen3::Eigen SYSTEM INTERFACE ${eigen3_SOURCE_DIR})

target_link_libraries(ipc_core PUBLIC Eigen3::Eigen)

# ── Simulation executable ─────────────────────────────────────────────────────
add_executable(simulation src/simulation.cpp)
target_link_libraries(simulation PRIVATE ipc_core)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
