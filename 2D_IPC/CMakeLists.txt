cmake_minimum_required(VERSION 3.16)
project(2D_IPC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

enable_testing()

# ── Core library ──────────────────────────────────────────────────────────────
add_library(ipc_core
    physics.cpp
    spring_energy.cpp
    barrier_energy.cpp
    node_segment_distance.cpp
    ogc_trust_region.cpp
    parallel_helper.cpp
    chain.cpp
    restart.cpp
    visualization.cpp
    broad_phase/bvh.cpp
    ccd.cpp
    initial_guess/trivial.cpp
    initial_guess/affine.cpp
    initial_guess/ccd.cpp
    solver.cpp
    example.cpp
    initial_guess/initial_guess.cpp
    rigid_body.cpp
)

target_include_directories(ipc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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

if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    if(EXISTS /opt/homebrew/opt/libomp)
        set(_LIBOMP_PREFIX /opt/homebrew/opt/libomp)
    else()
        set(_LIBOMP_PREFIX /usr/local/opt/libomp)
    endif()
    set(OpenMP_omp_LIBRARY ${_LIBOMP_PREFIX}/lib/libomp.dylib)
    include_directories(${_LIBOMP_PREFIX}/include)
endif()
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ipc_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── Simulation executable ─────────────────────────────────────────────────────
add_executable(simulation simulation.cpp)
target_link_libraries(simulation PRIVATE ipc_core)

# ── Tests ─────────────────────────────────────────────────────────────────────
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(rigid_body_ccd_test tests/rigid_body_ccd_test.cpp)
target_link_libraries(rigid_body_ccd_test PRIVATE ipc_core GTest::gtest GTest::gtest_main)
gtest_discover_tests(rigid_body_ccd_test)

add_executable(bvh_box_leaf_test tests/bvh_box_leaf_test.cpp)
target_link_libraries(bvh_box_leaf_test PRIVATE ipc_core GTest::gtest GTest::gtest_main)
gtest_discover_tests(bvh_box_leaf_test)

add_executable(parallel_helper_test parallel_helper_test.cpp)
target_link_libraries(parallel_helper_test PRIVATE ipc_core GTest::gtest GTest::gtest_main)
gtest_discover_tests(parallel_helper_test)
