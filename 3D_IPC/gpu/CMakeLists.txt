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
set(IPC_BUILD_SIMULATOR OFF CACHE BOOL "" FORCE)
set(IPC_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(IPC_ENABLE_IPO OFF CACHE BOOL "" FORCE)
set(IPC_CPU_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "CPU source tree (read-only)")
get_filename_component(IPC_CPU_SOURCE "${IPC_CPU_SOURCE}" ABSOLUTE)
add_subdirectory("${IPC_CPU_SOURCE}" cpu EXCLUDE_FROM_ALL)
find_package(Boost 1.70 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Ordinary ALL builds the current resident simulator and its required libraries.
# Legacy, control, and validation targets remain available through explicit targets.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GeneratedSources.cmake")
enable_testing()

add_library(ipc_resident_components STATIC resident_elastic.cu resident_broadphase.cu)
target_include_directories(ipc_resident_components PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_components PUBLIC CUDA::cudart CUDA::cusparse)
target_compile_options(ipc_resident_components PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_library(ipc_resident_solver STATIC resident_solver.cu resident_contact_pipeline.cu
    resident_contact_cooperative.cu
    resident_linear_pipeline.cu resident_rigid_pipeline.cu
    resident_adaptive_boxes.cu resident_global_guess.cu
    resident_contact_compact_point.cu cooperative_compact_point_study.cu)
target_include_directories(ipc_resident_solver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_solver PUBLIC ipc_resident_components CUDA::cudart)
target_compile_options(ipc_resident_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_library(ipc_resident_bridge STATIC resident_bridge.cpp)
target_include_directories(ipc_resident_bridge PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_bridge PUBLIC IPC::core ipc_resident_solver)
ipc_gpu_generate_resident_driver()
add_executable(3D_sim_gpu_resident "${CMAKE_CURRENT_BINARY_DIR}/simulation_resident.cpp")
target_link_libraries(3D_sim_gpu_resident PRIVATE ipc_resident_bridge IPC::ipc Threads::Threads)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/resident_driver_step.inc")
# Keep legacy setup separate from the untouched CPU base and resident kernels.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Hybrid.cmake")
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
    gpu_resident_solver_test 3D_sim_gpu_resident)
