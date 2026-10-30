cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(IPC_GPU LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(IPC_GPU_BUILD_DRIVER "Build the shared-scene GPU simulator" ON)
option(IPC_GPU_BUILD_VALIDATION "Create shared CPU reference validation targets" ${IPC_GPU_BUILD_DRIVER})
set(IPC_GPU_WITH_CPU_SUPPORT OFF)
if(IPC_GPU_BUILD_DRIVER OR IPC_GPU_BUILD_VALIDATION)
    set(IPC_GPU_WITH_CPU_SUPPORT ON)
endif()

# CUDA numerical libraries do not require the shared CPU source or dependencies.
# Boost remains exact192 whenever shared scene/IO or CPU references are enabled.
if(IPC_GPU_WITH_CPU_SUPPORT)
    set(IPC_BUILD_SIMULATOR OFF CACHE BOOL "" FORCE)
    set(IPC_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(IPC_ENABLE_IPO OFF CACHE BOOL "" FORCE)
    # Pin the host exact/reference dependency before configuring the shared CPU tree.
    find_package(Boost 1.92.0 EXACT CONFIG REQUIRED)
    include(CMakePushCheckState)
    include(CheckCXXSourceCompiles)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES Boost::headers)
    # Recheck even when an installation changes its headers in place.
    unset(IPC_GPU_BOOST_HEADERS_EXACT_1_92 CACHE)
    unset(IPC_GPU_BOOST_HEADERS_EXACT_1_92)
    check_cxx_source_compiles("
        #include <boost/version.hpp>
        #if !defined(BOOST_VERSION) || BOOST_VERSION != 109200
        #error GPU builds require exactly Boost 1.92.0 headers
        #endif
        int main() { return 0; }
    " IPC_GPU_BOOST_HEADERS_EXACT_1_92)
    cmake_pop_check_state()
    if(NOT IPC_GPU_BOOST_HEADERS_EXACT_1_92)
        message(FATAL_ERROR
            "GPU builds require exactly Boost 1.92.0 headers through Boost::headers. "
            "Select the matching installed package with Boost_DIR."
        )
    endif()

    set(IPC_GPU_BOOST_POLICY_1_92 TRUE)
    set(IPC_CPU_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "CPU source tree (read-only)")
    get_filename_component(IPC_CPU_SOURCE "${IPC_CPU_SOURCE}" ABSOLUTE)
    add_subdirectory("${IPC_CPU_SOURCE}" cpu EXCLUDE_FROM_ALL)
    unset(IPC_GPU_BOOST_POLICY_1_92)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Default ALL builds the current GPU simulator and its required libraries.
# Validation targets remain available through explicit targets.
if(IPC_GPU_WITH_CPU_SUPPORT)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GeneratedSources.cmake")
endif()
enable_testing()

add_library(ipc_gpu_components STATIC elastic.cu broadphase.cu)
target_include_directories(ipc_gpu_components PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_gpu_components PUBLIC CUDA::cudart CUDA::cusparse)
target_compile_options(ipc_gpu_components PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_library(ipc_gpu_solver STATIC solver.cu contact_pipeline.cu
    contact_cooperative.cu
    linear_pipeline.cu rigid_pipeline.cu
    adaptive_boxes.cu global_guess.cu level_schedule.cu
    dat.cu
    contact_compact_point.cu cooperative_compact_point_study.cu)
target_include_directories(ipc_gpu_solver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_gpu_solver PUBLIC ipc_gpu_components CUDA::cudart Threads::Threads)
target_compile_options(ipc_gpu_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
if(IPC_GPU_BUILD_DRIVER OR IPC_GPU_BUILD_VALIDATION)
    add_library(ipc_gpu_bridge STATIC bridge.cpp)
    target_include_directories(ipc_gpu_bridge PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(ipc_gpu_bridge PUBLIC IPC::core ipc_gpu_solver)
endif()
if(IPC_GPU_BUILD_DRIVER)
    # The public simulator is the one executable: the generated current GPU
    # driver over the current device solver library.
    ipc_gpu_generate_driver()
    add_executable(3D_sim_gpu "${CMAKE_CURRENT_BINARY_DIR}/simulation_gpu.cpp")
    target_link_libraries(3D_sim_gpu PRIVATE ipc_gpu_bridge IPC::ipc Threads::Threads)
    # Track the shared driver and its dispatch fragment.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${IPC_CPU_SOURCE}/simulation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/driver_step.inc")
endif()

# Keep validation setup separate from the untouched CPU base and GPU solver kernels.
if(IPC_GPU_BUILD_VALIDATION)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Validation.cmake")

    # Match the CPU oracle used by the production solver. Keep the GPU-header
    # caller uncontracted, but obtain CPU contact functions from IPC::core
    # with its normal compiler settings, including boundary-sensitive FMA.
    add_executable(gpu_contact_policy_host_test EXCLUDE_FROM_ALL
        contact_policy_host_test.cpp)
    target_link_libraries(gpu_contact_policy_host_test PRIVATE IPC::core Boost::headers)
    target_compile_options(gpu_contact_policy_host_test PRIVATE -ffp-contract=off)
    add_test(NAME gpu_contact_policy_host COMMAND gpu_contact_policy_host_test)
    set_tests_properties(gpu_contact_policy_host PROPERTIES LABELS "gpu_solver;host_math")

    add_custom_target(gpu_validation DEPENDS
        gpu_elastic_test gpu_elastic_worklist_test gpu_broadphase_test gpu_contact_test
        gpu_solve_test gpu_rigid_test gpu_ccd_test
        gpu_sdf_test gpu_rigid_ccd_test gpu_contact_pipeline_test
        gpu_linear_pipeline_test gpu_rigid_pipeline_test
        gpu_contact_rejection_test gpu_contact_queue_test
        gpu_contact_warp_reduce_test gpu_contact_worklist_test gpu_contact_compact_test
        gpu_box_certificate_test gpu_box_pruning_test
        gpu_contact_cooperative_test gpu_contact_schedule_test
        gpu_ratio_host_test gpu_ratio_cuda_test
        gpu_exact_pool_test gpu_adaptive_boxes_test gpu_global_guess_test
        gpu_solver_test gpu_dat_test gpu_dat_geometry_test
        gpu_rigid_pins_test gpu_cpu_parity_test
        gpu_original_contact_test gpu_original_ccd_test gpu_original_mode_test)
    add_dependencies(gpu_validation gpu_contact_policy_host_test)
    if(IPC_GPU_BUILD_DRIVER)
        add_dependencies(gpu_validation 3D_sim_gpu)
    endif()
endif()

# A public-API caller with no shared CPU headers, references or scene generator.
add_executable(gpu_core_smoke EXCLUDE_FROM_ALL core_smoke.cpp)
target_link_libraries(gpu_core_smoke PRIVATE ipc_gpu_solver)
target_compile_options(gpu_core_smoke PRIVATE -ffp-contract=off)
add_test(NAME gpu_core_smoke COMMAND gpu_core_smoke)
set_tests_properties(gpu_core_smoke PROPERTIES LABELS gpu_solver)
add_custom_target(gpu_core_validation DEPENDS gpu_core_smoke)
if(IPC_GPU_BUILD_VALIDATION)
    add_dependencies(gpu_validation gpu_core_smoke)
endif()
