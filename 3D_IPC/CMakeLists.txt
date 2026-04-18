cmake_minimum_required(VERSION 3.18)

project(CorotatedEnergy LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
# GTest was built without a Release config (only NOCONFIG); patch the
# Release IMPORTED_LOCATION so target resolution doesn't yield NOTFOUND.
get_target_property(_gtest_lib     GTest::gtest      IMPORTED_LOCATION_NOCONFIG)
get_target_property(_gtest_main_lib GTest::gtest_main IMPORTED_LOCATION_NOCONFIG)
if(_gtest_lib AND NOT _gtest_lib STREQUAL "_gtest_lib-NOTFOUND")
    set_target_properties(GTest::gtest PROPERTIES
        IMPORTED_LOCATION_RELEASE "${_gtest_lib}"
    )
endif()
if(_gtest_main_lib AND NOT _gtest_main_lib STREQUAL "_gtest_main_lib-NOTFOUND")
    set_target_properties(GTest::gtest_main PROPERTIES
        IMPORTED_LOCATION_RELEASE "${_gtest_main_lib}"
    )
endif()
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

# Optional CUDA support. If nvcc is found, swap stubs for real .cu files.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Default to sm_89 (Ada) — nvcc 12.0 doesn't know Blackwell (sm_120),
    # and enable_language(CUDA) otherwise defaults to sm_52 which JIT-compiles
    # to newer GPUs unreliably (crashes in complex kernels). Override via
    # -DCMAKE_CUDA_ARCHITECTURES=... if needed.
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA archs" FORCE)
    set(GPU_MESH_SOURCES
        GPU_Sim/gpu_mesh.cu
        GPU_Sim/gpu_ccd_stub.cpp
        GPU_Sim/gpu_physics_stub.cpp
        GPU_Sim/gpu_solver.cu
        GPU_Sim/gpu_solver_bridge.cpp
    )
    find_package(CUDAToolkit REQUIRED)
    set(GPU_CUDA_FOUND TRUE)
    message(STATUS "CUDA found — building GPU kernels (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(GPU_MESH_SOURCES
        GPU_Sim/gpu_mesh_stub.cpp
        GPU_Sim/gpu_ccd_stub.cpp
        GPU_Sim/gpu_physics_stub.cpp
        GPU_Sim/gpu_solver_stub.cpp
        GPU_Sim/gpu_solver_bridge.cpp
    )
    set(GPU_CUDA_FOUND FALSE)
    message(STATUS "CUDA not found — using CPU stubs")
endif()

add_library(gpu_mesh_lib STATIC ${GPU_MESH_SOURCES})
if(GPU_CUDA_FOUND)
    target_compile_options(gpu_mesh_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_link_libraries(gpu_mesh_lib PUBLIC Eigen3::Eigen corotated_energy_lib OpenMP::OpenMP_CXX CUDA::cudart)
else()
    target_link_libraries(gpu_mesh_lib PUBLIC Eigen3::Eigen corotated_energy_lib OpenMP::OpenMP_CXX)
endif()
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

add_executable(gpu_ccd_test
    gpu_ccd_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    parallel_helper.cpp
    broad_phase.cpp
)

target_link_libraries(gpu_ccd_test
    PRIVATE
    gpu_mesh_lib
    corotated_energy_lib
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(gpu_ccd_test)

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
