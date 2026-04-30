cmake_minimum_required(VERSION 3.21)

project(CorotatedEnergy LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

include(FetchContent)

# Pin Eigen to 3.4.0. Homebrew shipped Eigen 5.0.x in early 2026 and that
# version triggered NaN corruption through std::function<Vec3(int)> boundaries
# in the experimental solver. Until the codebase is audited for Eigen 5
# compatibility, fetch 3.4.0 directly so every build uses the validated version.
set(EIGEN_BUILD_DOC OFF CACHE INTERNAL "")
set(EIGEN_BUILD_TESTING OFF CACHE INTERNAL "")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE INTERNAL "")
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_Declare(
    eigen3
    URL      https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
)
# Use GetProperties + Populate so we don't pull Eigen's CMakeLists into the
# project (it adds heaps of test/check targets). We just need the headers.
FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3)
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
target_include_directories(Eigen3::Eigen SYSTEM INTERFACE ${eigen3_SOURCE_DIR})

find_package(GTest REQUIRED)

# Tight-Inclusion CCD (https://github.com/Continuous-Collision-Detection/Tight-Inclusion)
# powers our general CCD (multi-vertex motion). Pulled in via FetchContent so the
# build is self-contained.
FetchContent_Declare(
    tight_inclusion
    GIT_REPOSITORY https://github.com/Continuous-Collision-Detection/Tight-Inclusion.git
    GIT_TAG        v1.0.6
)
FetchContent_MakeAvailable(tight_inclusion)
if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    # Support both Apple Silicon (/opt/homebrew) and Intel (/usr/local) Homebrew paths
    if(EXISTS /opt/homebrew/opt/libomp)
        set(_LIBOMP_PREFIX /opt/homebrew/opt/libomp)
    else()
        set(_LIBOMP_PREFIX /usr/local/opt/libomp)
    endif()
    set(OpenMP_omp_LIBRARY ${_LIBOMP_PREFIX}/lib/libomp.dylib)
    include_directories(${_LIBOMP_PREFIX}/include)
endif()
find_package(OpenMP REQUIRED)

# GPU mesh mirror structs.
# On machines without CUDA the stub provides CPU-heap equivalents of
# DeviceBuffer / upload / download so the rest of the code compiles unchanged.
# On a real CUDA machine: replace gpu_mesh_stub.cpp with gpu_mesh.cu and
# add CUDA to the LANGUAGES line above.
add_library(gpu_mesh_lib STATIC
    GPU_Sim/gpu_mesh_stub.cpp
    GPU_Sim/gpu_ccd_stub.cpp
    GPU_Sim/gpu_physics_stub.cpp
    GPU_Sim/gpu_solver_stub.cpp
    GPU_Sim/gpu_solver_bridge.cpp
)
target_link_libraries(gpu_mesh_lib PUBLIC Eigen3::Eigen corotated_energy_lib OpenMP::OpenMP_CXX)
target_include_directories(gpu_mesh_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/GPU_Sim
)

add_library(corotated_energy_lib
    corotated_energy.cpp
    bending_energy.cpp
    IPC_math.cpp
    node_triangle_distance.cpp
    segment_segment_distance.cpp
    barrier_energy.cpp
    sdf_penalty_energy.cpp
    ccd.cpp
    trust_region.cpp
)

target_link_libraries(corotated_energy_lib
    PUBLIC Eigen3::Eigen
    PRIVATE tight_inclusion::tight_inclusion
)

target_include_directories(corotated_energy_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(corotated_energy_test
    corotated_energy_test.cpp
)

target_link_libraries(corotated_energy_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(corotated_energy_test)

add_executable(node_triangle_distance_test
    node_triangle_distance_test.cpp
)

target_link_libraries(node_triangle_distance_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(node_triangle_distance_test)

add_executable(segment_segment_distance_test
    segment_segment_distance_test.cpp
)

target_link_libraries(segment_segment_distance_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(segment_segment_distance_test)

add_executable(barrier_energy_test
    barrier_energy_test.cpp
)

target_link_libraries(barrier_energy_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(barrier_energy_test)

add_executable(bending_energy_test
    bending_energy_test.cpp
)

target_link_libraries(bending_energy_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(bending_energy_test)

add_executable(sdf_penalty_energy_test
    sdf_penalty_energy_test.cpp
)

target_link_libraries(sdf_penalty_energy_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(sdf_penalty_energy_test)

add_executable(gpu_physics_test
    gpu_physics_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_link_libraries(gpu_physics_test
    PRIVATE
    gpu_mesh_lib
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(gpu_physics_test)

add_executable(gpu_solver_test
    gpu_solver_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_link_libraries(gpu_solver_test
    PRIVATE
    gpu_mesh_lib
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(gpu_solver_test)

add_executable(ccd_test
    ccd_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_link_libraries(ccd_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(ccd_test)

# Long-running randomized linear-vs-TICCD parity sweep. Local-only:
# `ccd_stress_test.cpp` is gitignored, so this target is built only when the
# source happens to be present on disk. Not registered with
# `gtest_discover_tests` — keeps `ctest` fast even when the source IS present.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ccd_stress_test.cpp")
    add_executable(ccd_stress_test
        ccd_stress_test.cpp
        make_shape.cpp
        physics.cpp
        solver.cpp
        parallel_helper.cpp
        broad_phase.cpp
    )

    target_link_libraries(ccd_stress_test
        PRIVATE
        corotated_energy_lib
        GTest::gtest
        GTest::gtest_main
    )
endif()

add_executable(trust_region_test
    trust_region_test.cpp
)

target_link_libraries(trust_region_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(trust_region_test)

add_executable(broad_phase_test
    broad_phase_test.cpp
    broad_phase.cpp
    make_shape.cpp
    visualization.cpp
)

target_link_libraries(broad_phase_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(broad_phase_test)

add_executable(3D_sim
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    visualization.cpp
    broad_phase.cpp
    simulation.cpp
    example.cpp
)

target_link_libraries(3D_sim
    PRIVATE
    gpu_mesh_lib
    corotated_energy_lib
    OpenMP::OpenMP_CXX
)

add_executable(make_shape_test
    make_shape_test.cpp
    make_shape.cpp
    physics.cpp
)

target_link_libraries(make_shape_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(make_shape_test)

add_executable(simulation_snapshot_test
    simulation_snapshot_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_compile_definitions(simulation_snapshot_test
    PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(simulation_snapshot_test
    PRIVATE
    gpu_mesh_lib
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(simulation_snapshot_test)

add_executable(generate_golden
    generate_golden.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_compile_definitions(generate_golden
    PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(generate_golden
    PRIVATE
    gpu_mesh_lib
    corotated_energy_lib
    OpenMP::OpenMP_CXX
)

add_executable(restart_test
    restart_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_compile_definitions(restart_test
    PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(restart_test
    PRIVATE
    gpu_mesh_lib
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(restart_test)

add_executable(parallel_serial_consistency_test
    parallel_serial_consistency_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_link_libraries(parallel_serial_consistency_test
    PRIVATE
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(parallel_serial_consistency_test)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/visualization_test.cpp")
    add_executable(visualization_test
        visualization_test.cpp
        make_shape.cpp
        physics.cpp
        broad_phase.cpp
        visualization.cpp
    )

    target_compile_definitions(visualization_test
        PRIVATE VIS_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(visualization_test
        PRIVATE
        corotated_energy_lib
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(visualization_test)
endif()

add_executable(parallel_helper_test
    parallel_helper_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_link_libraries(parallel_helper_test
    PRIVATE
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(parallel_helper_test)

add_executable(total_energy_test
    total_energy_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    visualization.cpp
    broad_phase.cpp
)

target_link_libraries(total_energy_test
    PRIVATE
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(total_energy_test)

add_executable(ipc_math_test
    ipc_math_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_link_libraries(ipc_math_test
    PRIVATE
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(ipc_math_test)
