cmake_minimum_required(VERSION 3.24)
project(IPC_GPU LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(IPC_GPU_BUILD_DRIVER "Build the upstream-structured GPU driver and native scene exporter" ON)
option(IPC_GPU_BUILD_VALIDATION "Register the GPU equation and integration tests" ON)
set(IPC_CPU_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Shared native scene source")
get_filename_component(IPC_CPU_SOURCE "${IPC_CPU_SOURCE}" ABSOLUTE)
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)

# The host executable exports original scene/rest data and authored motion only.
# All evolving physics belongs to the modified Newton/Warp GPU program.
set(IPC_BUILD_SIMULATOR OFF CACHE BOOL "" FORCE)
set(IPC_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(IPC_BUILD_NEWTON_EXPORTER OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(IPC_ENABLE_IPO OFF CACHE BOOL "" FORCE)
add_subdirectory("${IPC_CPU_SOURCE}" cpu EXCLUDE_FROM_ALL)
add_executable(3D_export_newton_scene export_scene.cpp)
target_link_libraries(3D_export_newton_scene PRIVATE IPC::ipc)

# Keep the public executable name while executing the actual Python GPU driver.
# This launcher does no numerical work and does not invoke the old CUDA solver.
set(_ipc_gpu_source "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(_ipc_project_root "${_ipc_gpu_source}/.." ABSOLUTE)
set(_ipc_exporter "$<TARGET_FILE:3D_export_newton_scene>")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Driver.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/_gpu_driver.in" @ONLY)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/3D_sim_gpu"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/_gpu_driver.in"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
if(IPC_GPU_BUILD_DRIVER)
    add_custom_target(3D_sim_gpu ALL DEPENDS 3D_export_newton_scene)
endif()

enable_testing()
if(IPC_GPU_BUILD_VALIDATION)
    add_test(NAME gpu_equations_and_structure
        COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s gpu/warp_tests -t . -p "test_*.py" -v)
    set_tests_properties(gpu_equations_and_structure PROPERTIES
        WORKING_DIRECTORY "${_ipc_project_root}"
        LABELS "gpu_solver"
        ENVIRONMENT "IPC_NEWTON_SCENE_EXPORTER=$<TARGET_FILE:3D_export_newton_scene>;WARP_CACHE_PATH=${CMAKE_CURRENT_BINARY_DIR}/warp_cache")
    # The exact upstream and GS-hook snapshots must never share one process.
    add_test(NAME gpu_dat_equations_and_structure
        COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s gpu/dat_tests -t . -p "test_*.py" -v)
    set_tests_properties(gpu_dat_equations_and_structure PROPERTIES
        WORKING_DIRECTORY "${_ipc_project_root}" LABELS "gpu_solver"
        ENVIRONMENT "WARP_CACHE_PATH=${CMAKE_CURRENT_BINARY_DIR}/warp_cache")
    add_test(NAME gpu_cuda_required COMMAND "${Python3_EXECUTABLE}" -m gpu.run --check)
    set_tests_properties(gpu_cuda_required PROPERTIES
        WORKING_DIRECTORY "${_ipc_project_root}" LABELS "gpu_solver"
        ENVIRONMENT "WARP_CACHE_PATH=${CMAKE_CURRENT_BINARY_DIR}/warp_cache")
    add_custom_target(gpu_validation DEPENDS 3D_export_newton_scene)
    add_custom_target(gpu_core_validation DEPENDS gpu_validation)
endif()
