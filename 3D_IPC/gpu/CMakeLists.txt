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
option(IPC_GPU_BUILD_DRIVER "Build the shared-scene resident simulator" ON)
option(IPC_GPU_BUILD_VALIDATION "Create shared CPU reference validation targets" ${IPC_GPU_BUILD_DRIVER})
option(IPC_GPU_BUILD_HYBRID "Create legacy hybrid and CPU control targets" ${IPC_GPU_BUILD_DRIVER})
set(IPC_GPU_WITH_CPU_SUPPORT OFF)
if(IPC_GPU_BUILD_DRIVER OR IPC_GPU_BUILD_VALIDATION OR IPC_GPU_BUILD_HYBRID)
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

# Default ALL builds the current resident simulator and its required libraries.
# Legacy, control, and validation targets remain available through explicit targets.
if(IPC_GPU_WITH_CPU_SUPPORT)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GeneratedSources.cmake")
endif()
enable_testing()

add_library(ipc_resident_components STATIC resident_elastic.cu resident_broadphase.cu)
target_include_directories(ipc_resident_components PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_components PUBLIC CUDA::cudart CUDA::cusparse)
target_compile_options(ipc_resident_components PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
# DAT is independently buildable: only shared CUDA elasticity/BVH components.
add_library(ipc_resident_dat STATIC resident_dat.cu resident_dat_solver.cu)
target_include_directories(ipc_resident_dat PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_dat PUBLIC ipc_resident_components CUDA::cudart Threads::Threads)
target_compile_options(ipc_resident_dat PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_library(ipc_resident_solver STATIC resident_solver.cu resident_contact_pipeline.cu
    resident_contact_cooperative.cu
    resident_linear_pipeline.cu resident_rigid_pipeline.cu
    resident_adaptive_boxes.cu resident_global_guess.cu resident_level_schedule.cu
    resident_contact_compact_point.cu cooperative_compact_point_study.cu)
target_include_directories(ipc_resident_solver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_solver PUBLIC ipc_resident_dat ipc_resident_components CUDA::cudart Threads::Threads)
target_compile_options(ipc_resident_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
if(IPC_GPU_BUILD_DRIVER OR IPC_GPU_BUILD_VALIDATION)
    add_library(ipc_resident_bridge STATIC resident_bridge.cpp)
    target_include_directories(ipc_resident_bridge PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(ipc_resident_bridge PUBLIC IPC::core ipc_resident_solver)
endif()
if(IPC_GPU_BUILD_DRIVER)
    ipc_gpu_generate_resident_driver()
    add_executable(3D_sim_gpu_resident "${CMAKE_CURRENT_BINARY_DIR}/simulation_resident.cpp")
    target_link_libraries(3D_sim_gpu_resident PRIVATE ipc_resident_bridge IPC::ipc Threads::Threads)
    # DAT's dedicated driver links no ordinary GPU CCD or rigid solver.
    add_library(ipc_resident_dat_bridge STATIC resident_bridge.cpp)
    target_include_directories(ipc_resident_dat_bridge PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(ipc_resident_dat_bridge PUBLIC IPC::core ipc_resident_dat)
    ipc_gpu_generate_dat_driver()
    add_executable(3D_sim_gpu_dat "${CMAKE_CURRENT_BINARY_DIR}/simulation_dat.cpp")
    target_link_libraries(3D_sim_gpu_dat PRIVATE ipc_resident_dat_bridge IPC::ipc)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/resident_driver_step.inc")
endif()

# Keep legacy setup separate from the untouched CPU base and resident kernels.
if(IPC_GPU_BUILD_HYBRID)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Hybrid.cmake")
endif()
if(IPC_GPU_BUILD_VALIDATION)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Validation.cmake")

    add_custom_target(gpu_resident_validation DEPENDS
        gpu_resident_elastic_test gpu_resident_elastic_worklist_test gpu_resident_broadphase_test gpu_resident_contact_test
        gpu_resident_solve_test gpu_resident_rigid_test gpu_resident_ccd_test
        gpu_resident_sdf_test gpu_resident_rigid_ccd_test gpu_resident_contact_pipeline_test
        gpu_resident_linear_pipeline_test gpu_resident_rigid_pipeline_test
        gpu_resident_contact_rejection_test gpu_resident_contact_queue_test
        gpu_resident_contact_warp_reduce_test gpu_resident_contact_worklist_test gpu_resident_contact_compact_test
        gpu_resident_box_certificate_test gpu_resident_box_pruning_test
        gpu_resident_contact_cooperative_test gpu_resident_contact_schedule_test
        gpu_resident_ratio_host_test gpu_resident_ratio_cuda_test
        gpu_resident_exact_pool_test gpu_resident_adaptive_boxes_test gpu_resident_global_guess_test
        gpu_resident_solver_test)
    if(IPC_GPU_BUILD_DRIVER)
        add_dependencies(gpu_resident_validation 3D_sim_gpu_resident 3D_sim_gpu_dat)
    endif()
endif()

# A public-API caller with no shared CPU headers, references or scene generator.
add_executable(gpu_resident_core_smoke EXCLUDE_FROM_ALL resident_core_smoke.cpp)
target_link_libraries(gpu_resident_core_smoke PRIVATE ipc_resident_solver)
target_compile_options(gpu_resident_core_smoke PRIVATE -ffp-contract=off)
add_test(NAME gpu_resident_core_smoke COMMAND gpu_resident_core_smoke)
add_custom_target(gpu_resident_core_validation DEPENDS gpu_resident_core_smoke)
if(IPC_GPU_BUILD_VALIDATION)
    add_dependencies(gpu_resident_validation gpu_resident_core_smoke)
endif()

# DAT also validates in the CPU-independent numerical-core configuration.
add_executable(gpu_resident_dat_test EXCLUDE_FROM_ALL resident_dat_test.cpp)
target_link_libraries(gpu_resident_dat_test PRIVATE ipc_resident_solver)
target_compile_options(gpu_resident_dat_test PRIVATE -ffp-contract=off)
if(IPC_GPU_WITH_CPU_SUPPORT)
    target_compile_definitions(gpu_resident_dat_test PRIVATE IPC_DAT_CPU_REFERENCE)
    target_link_libraries(gpu_resident_dat_test PRIVATE ipc_resident_bridge IPC::ipc)
endif()
add_test(NAME gpu_resident_dat COMMAND gpu_resident_dat_test)
add_dependencies(gpu_resident_core_validation gpu_resident_dat_test)
if(IPC_GPU_BUILD_VALIDATION)
    add_dependencies(gpu_resident_validation gpu_resident_dat_test)
endif()

# Standalone DAT caller does not link the ordinary solver or CPU reference code.
add_executable(gpu_resident_dat_component_test EXCLUDE_FROM_ALL resident_dat_test.cpp)
target_compile_definitions(gpu_resident_dat_component_test PRIVATE IPC_DAT_COMPONENT)
target_compile_options(gpu_resident_dat_component_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_resident_dat_component_test PRIVATE ipc_resident_dat)
add_test(NAME gpu_resident_dat_component COMMAND gpu_resident_dat_component_test)
add_dependencies(gpu_resident_core_validation gpu_resident_dat_component_test)
if(IPC_GPU_BUILD_VALIDATION)
    add_dependencies(gpu_resident_validation gpu_resident_dat_component_test)
endif()
if(IPC_GPU_WITH_CPU_SUPPORT)
    add_executable(gpu_resident_dat_reference_test EXCLUDE_FROM_ALL resident_dat_test.cpp)
    target_compile_definitions(gpu_resident_dat_reference_test PRIVATE IPC_DAT_COMPONENT IPC_DAT_CPU_REFERENCE)
    target_compile_options(gpu_resident_dat_reference_test PRIVATE -ffp-contract=off)
    target_link_libraries(gpu_resident_dat_reference_test PRIVATE ipc_resident_dat IPC::ipc)
    add_test(NAME gpu_resident_dat_reference COMMAND gpu_resident_dat_reference_test)
    if(IPC_GPU_BUILD_VALIDATION)
        add_dependencies(gpu_resident_validation gpu_resident_dat_reference_test)
    endif()
endif()
