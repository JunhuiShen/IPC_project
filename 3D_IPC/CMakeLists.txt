cmake_minimum_required(VERSION 3.10)

project(CorotatedEnergy LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
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

add_library(corotated_energy_lib
    corotated_energy.cpp
    IPC_math.cpp
    node_triangle_distance.cpp
    segment_segment_distance.cpp
    barrier_energy.cpp
)

target_link_libraries(corotated_energy_lib
    PUBLIC Eigen3::Eigen
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
)
add_test(NAME corotated_energy_test COMMAND corotated_energy_test)

add_executable(node_triangle_distance_test
    node_triangle_distance_test.cpp
)

target_link_libraries(node_triangle_distance_test
    PRIVATE
    corotated_energy_lib
)
add_test(NAME node_triangle_distance_test COMMAND node_triangle_distance_test)

add_executable(segment_segment_distance_test
    segment_segment_distance_test.cpp
)

target_link_libraries(segment_segment_distance_test
    PRIVATE
    corotated_energy_lib
)
add_test(NAME segment_segment_distance_test COMMAND segment_segment_distance_test)

add_executable(barrier_energy_test
    barrier_energy_test.cpp
)

target_link_libraries(barrier_energy_test
    PRIVATE
    corotated_energy_lib
)
add_test(NAME barrier_energy_test COMMAND barrier_energy_test)

add_executable(broad_phase_test
    broad_phase_test.cpp
    broad_phase.cpp
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
    visualization.cpp
    broad_phase.cpp
    simulation.cpp
)

target_link_libraries(3D_sim
    PRIVATE
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
    broad_phase.cpp
)

target_compile_definitions(simulation_snapshot_test
    PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(simulation_snapshot_test
    PRIVATE
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
    broad_phase.cpp
)

target_compile_definitions(generate_golden
    PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(generate_golden
    PRIVATE
    corotated_energy_lib
    OpenMP::OpenMP_CXX
)

add_executable(restart_test
    restart_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    broad_phase.cpp
)

target_compile_definitions(restart_test
    PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(restart_test
    PRIVATE
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
    broad_phase.cpp
)

target_compile_definitions(parallel_serial_consistency_test
    PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(parallel_serial_consistency_test
    PRIVATE
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(parallel_serial_consistency_test)

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

add_executable(total_energy_test
    total_energy_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    visualization.cpp
    broad_phase.cpp
)

target_link_libraries(total_energy_test
    PRIVATE
    corotated_energy_lib
    OpenMP::OpenMP_CXX
)
add_test(NAME total_energy_test COMMAND total_energy_test)
