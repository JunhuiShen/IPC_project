cmake_minimum_required(VERSION 3.21)

project(IPC3D VERSION 0.1.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Project options
# -----------------------------------------------------------------------------

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel
    )
endif()

option(IPC_BUILD_SIMULATOR "Build the 3D simulator" ON)
option(IPC_BUILD_TOOLS "Build developer tools" ON)
option(IPC_ENABLE_IPO "Enable interprocedural optimization in Release builds" ON)

include(CTest)
include(FetchContent)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# Keep Eigen at 3.4.0. Eigen 5 has produced numerical corruption in this
# project with AppleClang/libc++ on ARM64 and has not yet been qualified.
FetchContent_Declare(
    eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _ipc_header_only
)

# Tight-Inclusion provides continuous collision detection for general motion.
FetchContent_Declare(
    tight_inclusion
    GIT_REPOSITORY https://github.com/Continuous-Collision-Detection/Tight-Inclusion.git
    GIT_TAG v1.0.6
    GIT_SHALLOW TRUE
)

# SOURCE_SUBDIR above deliberately names a directory without a CMake project,
# so Eigen is populated without adding its documentation and test targets.
FetchContent_MakeAvailable(eigen3)
add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
target_include_directories(Eigen3::Eigen SYSTEM INTERFACE
    "${eigen3_SOURCE_DIR}"
)

FetchContent_MakeAvailable(tight_inclusion)

# AppleClang needs Homebrew's libomp and a compiler-specific OpenMP flag. Feed
# those paths to FindOpenMP instead of adding a global include directory.
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    find_path(IPC_LIBOMP_INCLUDE_DIR omp.h
        HINTS
            /opt/homebrew/opt/libomp/include
            /usr/local/opt/libomp/include
    )
    find_library(IPC_LIBOMP_LIBRARY omp
        HINTS
            /opt/homebrew/opt/libomp/lib
            /usr/local/opt/libomp/lib
    )
    if(IPC_LIBOMP_INCLUDE_DIR AND IPC_LIBOMP_LIBRARY)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING
            "C++ compiler flags for OpenMP" FORCE
        )
        set(OpenMP_CXX_LIB_NAMES omp CACHE STRING
            "C++ OpenMP library names" FORCE
        )
        set(OpenMP_omp_LIBRARY "${IPC_LIBOMP_LIBRARY}" CACHE FILEPATH
            "Path to the OpenMP runtime" FORCE
        )
        set(OpenMP_CXX_INCLUDE_DIR "${IPC_LIBOMP_INCLUDE_DIR}" CACHE PATH
            "Path to OpenMP headers" FORCE
        )
    endif()
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

# -----------------------------------------------------------------------------
# Project helpers
# -----------------------------------------------------------------------------

set(IPC_IPO_SUPPORTED FALSE)
if(IPC_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPC_IPO_SUPPORTED OUTPUT IPC_IPO_ERROR)
    if(NOT IPC_IPO_SUPPORTED)
        message(STATUS "IPO unavailable: ${IPC_IPO_ERROR}")
    endif()
endif()

function(ipc_enable_ipo target)
    if(IPC_IPO_SUPPORTED)
        set_property(TARGET "${target}" PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
endfunction()

function(ipc_configure_library target)
    target_compile_features("${target}" PUBLIC cxx_std_17)
    set_target_properties("${target}" PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories("${target}" PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    )
    ipc_enable_ipo("${target}")
endfunction()

# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------

# Geometry, constitutive models, contact kernels, and output utilities.
set(IPC_CORE_SOURCES
    IPC_math.cpp
    barrier_energy.cpp
    bending_energy.cpp
    ccd.cpp
    corotated_energy.cpp
    node_triangle_distance.cpp
    ogc_trust_region.cpp
    quaternion_math.cpp
    rigid_body.cpp
    rigid_body_ipc.cpp
    sdf_penalty_energy.cpp
    segment_segment_distance.cpp
    visualization.cpp
)

add_library(corotated_energy_lib STATIC ${IPC_CORE_SOURCES})
add_library(IPC::core ALIAS corotated_energy_lib)
ipc_configure_library(corotated_energy_lib)
target_link_libraries(corotated_energy_lib
    PUBLIC
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
    PRIVATE
        tight_inclusion::tight_inclusion
)

# Mesh setup, broad phase, physics assembly, and nonlinear solvers. Keeping
# these sources in a library prevents every test from recompiling them.
set(IPC_SIMULATION_SOURCES
    broad_phase.cpp
    example.cpp
    make_shape.cpp
    parallel_helper.cpp
    physics.cpp
    solver.cpp
)

add_library(ipc STATIC ${IPC_SIMULATION_SOURCES})
add_library(IPC::ipc ALIAS ipc)
ipc_configure_library(ipc)
target_link_libraries(ipc
    PUBLIC
        IPC::core
        OpenMP::OpenMP_CXX
)

# -----------------------------------------------------------------------------
# Applications and tools
# -----------------------------------------------------------------------------

if(IPC_BUILD_SIMULATOR)
    add_executable(3D_sim simulation.cpp)
    target_link_libraries(3D_sim PRIVATE IPC::ipc)
    ipc_enable_ipo(3D_sim)
endif()

if(IPC_BUILD_TOOLS)
    add_executable(generate_golden generate_golden.cpp)
    target_compile_definitions(generate_golden PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(generate_golden PRIVATE IPC::ipc)
    ipc_enable_ipo(generate_golden)
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    include(GoogleTest)

    function(ipc_add_test target)
        cmake_parse_arguments(PARSE_ARGV 1 IPC_TEST "" "" "DEFINITIONS")

        add_executable("${target}" "${target}.cpp")
        target_link_libraries("${target}" PRIVATE IPC::ipc GTest::gtest_main)
        if(IPC_TEST_DEFINITIONS)
            target_compile_definitions("${target}" PRIVATE ${IPC_TEST_DEFINITIONS})
        endif()
        ipc_enable_ipo("${target}")
        gtest_discover_tests("${target}"
            DISCOVERY_MODE PRE_TEST
            DISCOVERY_TIMEOUT 60
        )
    endfunction()

    set(IPC_TEST_TARGETS
        barrier_energy_test
        bending_energy_test
        broad_phase_test
        ccd_test
        corotated_energy_test
        initial_guess_test
        ipc_math_test
        make_shape_test
        node_triangle_distance_test
        parallel_helper_test
        rigid_body_ipc_test
        sdf_penalty_energy_test
        segment_segment_distance_test
        total_energy_test
    )
    foreach(test_target IN LISTS IPC_TEST_TARGETS)
        ipc_add_test("${test_target}")
    endforeach()

    ipc_add_test(simulation_snapshot_test DEFINITIONS
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    ipc_add_test(restart_test DEFINITIONS
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    ipc_add_test(visualization_test DEFINITIONS
        VIS_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/state_io_test.cpp")
        ipc_add_test(state_io_test)
    endif()
endif()
