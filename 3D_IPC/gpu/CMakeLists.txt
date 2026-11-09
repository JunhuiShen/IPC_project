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
option(IPC_GPU_FAST_DEV_BUILD
    "Reduce compiler memory and time for the linear CCD translation units" ON)
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

# Keep concurrent CUDA compilers from adding their peak working sets together.
# CPU support targets were created above and retain normal build parallelism.
if(CMAKE_GENERATOR MATCHES "Ninja" AND NOT DEFINED CMAKE_JOB_POOL_COMPILE)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS ipc_gpu_cuda_compile=1)
    set(CMAKE_JOB_POOL_COMPILE ipc_gpu_cuda_compile)
endif()

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
set(IPC_GPU_LINEAR_CCD_SOURCES
    linear_pipeline.cu
    linear_resolve_nodes.cu linear_resolve_direct.cu
    linear_resolve_original.cu linear_resolve_rigid.cu
    linear_resolve_rigid_original.cu)
add_library(ipc_gpu_solver STATIC solver.cu contact_pipeline.cu
    contact_cooperative.cu
    ${IPC_GPU_LINEAR_CCD_SOURCES} rigid_pipeline.cu
    adaptive_boxes.cu global_guess.cu level_schedule.cu
    dat.cu
    contact_compact_point.cu cooperative_compact_point_study.cu)
target_include_directories(ipc_gpu_solver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_gpu_solver PUBLIC ipc_gpu_components CUDA::cudart Threads::Threads)
target_compile_options(ipc_gpu_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
# Reduce compiler memory for the exact/linear CCD units while retaining
# --fmad=false. Other CUDA translation units keep their Release optimization.
if(IPC_GPU_FAST_DEV_BUILD)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        message(FATAL_ERROR "IPC_GPU_FAST_DEV_BUILD requires the NVIDIA CUDA compiler")
    endif()
    set_property(SOURCE ${IPC_GPU_LINEAR_CCD_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O1>")
    set_property(SOURCE linear_resolve_original.cu linear_resolve_rigid_original.cu
        APPEND PROPERTY COMPILE_DEFINITIONS
        IPC_GPU_OUTLINE_EXACT_COMPARE=1)
    message(STATUS "IPC GPU: ptxas -O1 for linear CCD units (lower compiler memory)")
endif()
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
        gpu_linear_pipeline_test gpu_node_ccd_batch_test gpu_rigid_pipeline_test
        gpu_node_csr_partition_native_test gpu_node_csr_partition_test
        gpu_node_cooperative_native_test gpu_node_controller_test
        gpu_rigid_proxy_parallel_test gpu_rigid_pipeline_proxy_trusted_test
        gpu_rigid_proxy_index_test gpu_rigid_proxy_values_test
        gpu_contact_rejection_test gpu_contact_queue_test
        gpu_contact_warp_reduce_test gpu_contact_worklist_test gpu_contact_compact_test
        gpu_box_certificate_test gpu_box_pruning_test
        gpu_contact_cooperative_test gpu_contact_schedule_test
        gpu_ratio_host_test gpu_ratio_cuda_test
        gpu_exact_pool_test gpu_adaptive_boxes_test gpu_global_guess_test
        gpu_solver_test gpu_dat_test gpu_dat_geometry_test
        gpu_rigid_dat_math_test gpu_directed_rounding_test gpu_original_rigid_math_test
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

if(IPC_GPU_BUILD_VALIDATION)
    # Maintained numerical, ownership and allocation-lifetime regression tests.
    # No production flags, default ENV presets, benchmark hooks or ALL targets.
    set(_ipc_opt_tests "${CMAKE_CURRENT_SOURCE_DIR}/optimization_tests")
    function(ipc_gpu_opt_compile target)
        target_include_directories(${target} BEFORE PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${_ipc_opt_tests}")
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>"
            "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>")
    endfunction()
    function(ipc_gpu_opt_test name target host_only)
        add_test(NAME ${name} COMMAND ${target})
        if(host_only)
            set_tests_properties(${name} PROPERTIES LABELS "gpu_solver;host_math")
        else()
            set_tests_properties(${name} PROPERTIES LABELS gpu_solver RUN_SERIAL TRUE)
        endif()
    endfunction()

    add_executable(gpu_two_rigid_queue_policy_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/two_rigid_color_queue_policy_test.cpp")
    ipc_gpu_opt_compile(gpu_two_rigid_queue_policy_test)
    ipc_gpu_opt_test(gpu_two_rigid_queue_policy gpu_two_rigid_queue_policy_test TRUE)
    add_executable(gpu_two_rigid_queue_lifetime_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/two_rigid_color_queue_native_test.cpp")
    ipc_gpu_opt_compile(gpu_two_rigid_queue_lifetime_test)
    target_link_libraries(gpu_two_rigid_queue_lifetime_test PRIVATE ipc_gpu_solver)
    target_link_options(gpu_two_rigid_queue_lifetime_test PRIVATE
        "-Wl,--wrap=cudaMalloc" "-Wl,--wrap=cudaFree")
    ipc_gpu_opt_test(gpu_two_rigid_queue_lifetime gpu_two_rigid_queue_lifetime_test FALSE)

    add_executable(gpu_two_rigid_dense_policy_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/two_rigid_dense_csr_host_test.cpp")
    ipc_gpu_opt_compile(gpu_two_rigid_dense_policy_test)
    ipc_gpu_opt_test(gpu_two_rigid_dense_policy gpu_two_rigid_dense_policy_test TRUE)
    # CUDA includes its host oracle; compiling both would duplicate main.
    add_executable(gpu_two_rigid_dense_device_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/two_rigid_dense_csr_cuda_test.cu")
    ipc_gpu_opt_compile(gpu_two_rigid_dense_device_test)
    target_link_libraries(gpu_two_rigid_dense_device_test PRIVATE CUDA::cudart)
    ipc_gpu_opt_test(gpu_two_rigid_dense_device gpu_two_rigid_dense_device_test FALSE)
    # Dense native textually includes the queue allocation/public-solver fixture.
    add_executable(gpu_two_rigid_dense_lifetime_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/two_rigid_dense_csr_native_test.cpp")
    ipc_gpu_opt_compile(gpu_two_rigid_dense_lifetime_test)
    target_link_libraries(gpu_two_rigid_dense_lifetime_test PRIVATE ipc_gpu_solver)
    target_link_options(gpu_two_rigid_dense_lifetime_test PRIVATE
        "-Wl,--wrap=cudaMalloc" "-Wl,--wrap=cudaFree")
    ipc_gpu_opt_test(gpu_two_rigid_dense_lifetime gpu_two_rigid_dense_lifetime_test FALSE)

    add_executable(gpu_original_storage_host_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/original_self_storage_test.cpp")
    ipc_gpu_opt_compile(gpu_original_storage_host_test)
    ipc_gpu_opt_test(gpu_original_storage_host gpu_original_storage_host_test TRUE)
    add_executable(gpu_original_storage_native_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/original_self_storage_test.cpp"
        "${_ipc_opt_tests}/original_self_storage_test.cu")
    ipc_gpu_opt_compile(gpu_original_storage_native_test)
    target_compile_definitions(gpu_original_storage_native_test PRIVATE IPC_ORIGINAL_SELF_STORAGE_CUDA_TEST=1)
    target_link_libraries(gpu_original_storage_native_test PRIVATE ipc_gpu_solver CUDA::cudart)
    ipc_gpu_opt_test(gpu_original_storage_native gpu_original_storage_native_test FALSE)

    add_executable(gpu_proxy_owner_host_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/rigid_proxy_owner_host_test.cpp")
    ipc_gpu_opt_compile(gpu_proxy_owner_host_test)
    target_link_libraries(gpu_proxy_owner_host_test PRIVATE Threads::Threads)
    ipc_gpu_opt_test(gpu_proxy_owner_host gpu_proxy_owner_host_test TRUE)
    # This two-TU host authority oracle intentionally links no production Solver.
    add_executable(gpu_proxy_scope_host_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/scope_host_test.cpp" "${_ipc_opt_tests}/scope_host_consumer.cpp")
    ipc_gpu_opt_compile(gpu_proxy_scope_host_test)
    target_link_libraries(gpu_proxy_scope_host_test PRIVATE Threads::Threads)
    ipc_gpu_opt_test(gpu_proxy_scope_host gpu_proxy_scope_host_test TRUE)
    add_test(NAME gpu_proxy_scope_access COMMAND "${CMAKE_COMMAND}"
        "-DCXX=${CMAKE_CXX_COMPILER}"
        "-DGPU_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DTEST_SOURCE_DIR=${_ipc_opt_tests}"
        "-DCUDA_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}"
        "-DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/optimization_scope_access"
        -P "${_ipc_opt_tests}/check_scope_access.cmake")
    set_tests_properties(gpu_proxy_scope_access PROPERTIES LABELS "gpu_solver;host_math")

    add_executable(gpu_proxy_public_functional_test EXCLUDE_FROM_ALL
        "${_ipc_opt_tests}/rigid_proxy_public_solver_functional_test.cpp")
    ipc_gpu_opt_compile(gpu_proxy_public_functional_test)
    target_link_libraries(gpu_proxy_public_functional_test PRIVATE ipc_gpu_solver)
    ipc_gpu_opt_test(gpu_proxy_public_functional gpu_proxy_public_functional_test FALSE)

    set(_ipc_opt_targets gpu_two_rigid_queue_policy_test gpu_two_rigid_queue_lifetime_test
        gpu_two_rigid_dense_policy_test gpu_two_rigid_dense_device_test gpu_two_rigid_dense_lifetime_test
        gpu_original_storage_host_test gpu_original_storage_native_test
        gpu_proxy_owner_host_test gpu_proxy_scope_host_test gpu_proxy_public_functional_test)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/node_cooperative_grid.h")
        add_executable(gpu_node_grid_policy_host_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/node_cooperative_grid_test.cpp")
        ipc_gpu_opt_compile(gpu_node_grid_policy_host_test)
        ipc_gpu_opt_test(gpu_node_grid_policy_host gpu_node_grid_policy_host_test TRUE)
        list(APPEND _ipc_opt_targets gpu_node_grid_policy_host_test)
    endif()
    if(EXISTS "${_ipc_opt_tests}/original_pt_translation_test.cu")
        add_executable(gpu_pt_translation_functional_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/original_pt_translation_host_test.cpp"
            "${_ipc_opt_tests}/original_pt_translation_test.cu")
        ipc_gpu_opt_compile(gpu_pt_translation_functional_test)
        target_compile_definitions(gpu_pt_translation_functional_test PRIVATE IPC_ORIGINAL_PT_TRANSLATION_CUDA_TEST=1)
        target_link_libraries(gpu_pt_translation_functional_test PRIVATE ipc_gpu_solver CUDA::cudart)
        ipc_gpu_opt_test(gpu_pt_translation_functional gpu_pt_translation_functional_test FALSE)
        list(APPEND _ipc_opt_targets gpu_pt_translation_functional_test)
    endif()
    add_custom_target(gpu_optimization_validation DEPENDS ${_ipc_opt_targets})
    add_dependencies(gpu_validation gpu_optimization_validation)

    # Register selected regressions after source/test files are frozen.
    # Presence selects registration, never qualification. No ENV presets.
    set(_ipc_selected_targets)
    function(ipc_gpu_selected_require)
        foreach(_relative IN LISTS ARGN)
            if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_relative}")
                message(FATAL_ERROR "selected maintained test dependency missing: ${_relative}")
            endif()
        endforeach()
    endfunction()
    function(ipc_gpu_selected_role target role)
        ipc_gpu_selected_require("optimization_tests/check_selected_link_role.cmake")
        set(_map "${CMAKE_CURRENT_BINARY_DIR}/${target}.selected-link.map")
        target_link_options(${target} PRIVATE "-Wl,-Map,${_map}")
        set_property(TARGET ${target} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" "-DEXECUTABLE=$<TARGET_FILE:${target}>"
                "-DLINK_MAP=${_map}" "-DNM=${CMAKE_NM}" "-DROLE=${role}"
                -P "${_ipc_opt_tests}/check_selected_link_role.cmake"
            VERBATIM)
    endfunction()

    if(EXISTS "${_ipc_opt_tests}/two_rigid_kind_packets_native_test.cpp")
        ipc_gpu_selected_require("optimization_tests/two_rigid_color_queue_native_test.cpp"
            "two_rigid_dense_csr_policy.h")
        add_executable(gpu_two_rigid_kind_packets_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/two_rigid_kind_packets_native_test.cpp")
        ipc_gpu_opt_compile(gpu_two_rigid_kind_packets_test)
        target_link_libraries(gpu_two_rigid_kind_packets_test PRIVATE ipc_gpu_solver)
        target_link_options(gpu_two_rigid_kind_packets_test PRIVATE
            "-Wl,--wrap=cudaMalloc" "-Wl,--wrap=cudaFree")
        ipc_gpu_selected_role(gpu_two_rigid_kind_packets_test PUBLIC)
        ipc_gpu_opt_test(gpu_two_rigid_kind_packets gpu_two_rigid_kind_packets_test FALSE)
        list(APPEND _ipc_selected_targets gpu_two_rigid_kind_packets_test)
    endif()

    if(EXISTS "${_ipc_opt_tests}/one_slot_policy_host_test.cpp")
        ipc_gpu_selected_require("rigid_one_slot_policy.h"
            "optimization_tests/one_slot_graph_facade_native_test.cpp"
            "optimization_tests/one_slot_public_native_test.cpp"
            "optimization_tests/two_rigid_color_queue_native_test.cpp")
        add_executable(gpu_rigid_one_slot_policy_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/one_slot_policy_host_test.cpp")
        ipc_gpu_opt_compile(gpu_rigid_one_slot_policy_test)
        ipc_gpu_opt_test(gpu_rigid_one_slot_policy gpu_rigid_one_slot_policy_test TRUE)
        add_executable(gpu_rigid_one_slot_public_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/one_slot_public_native_test.cpp")
        ipc_gpu_opt_compile(gpu_rigid_one_slot_public_test)
        target_link_libraries(gpu_rigid_one_slot_public_test PRIVATE ipc_gpu_solver)
        target_link_options(gpu_rigid_one_slot_public_test PRIVATE
            "-Wl,--wrap=cudaMalloc" "-Wl,--wrap=cudaFree")
        ipc_gpu_selected_role(gpu_rigid_one_slot_public_test PUBLIC)
        ipc_gpu_opt_test(gpu_rigid_one_slot_public gpu_rigid_one_slot_public_test FALSE)
        add_executable(gpu_rigid_one_slot_facade_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/one_slot_graph_facade_native_test.cpp")
        ipc_gpu_opt_compile(gpu_rigid_one_slot_facade_test)
        target_link_libraries(gpu_rigid_one_slot_facade_test PRIVATE ipc_gpu_solver)
        ipc_gpu_selected_role(gpu_rigid_one_slot_facade_test FACADE)
        ipc_gpu_opt_test(gpu_rigid_one_slot_facade gpu_rigid_one_slot_facade_test FALSE)
        list(APPEND _ipc_selected_targets gpu_rigid_one_slot_policy_test
            gpu_rigid_one_slot_public_test gpu_rigid_one_slot_facade_test)
    endif()

    if(EXISTS "${_ipc_opt_tests}/host_sparse_probe_test.cpp")
        ipc_gpu_selected_require("original_sparse_gather_policy.h" "original_sparse_gather_kernels.cuh"
            "optimization_tests/sparse_gather_native_test.cpp"
            "optimization_tests/sparse_fixture.inc"
            "optimization_tests/sparse_public_native_test.cpp"
            "optimization_tests/sparse_probe_device_test.cu")
        add_executable(gpu_original_sparse_policy_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/host_sparse_probe_test.cpp")
        ipc_gpu_opt_compile(gpu_original_sparse_policy_test)
        target_include_directories(gpu_original_sparse_policy_test SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        ipc_gpu_opt_test(gpu_original_sparse_policy gpu_original_sparse_policy_test TRUE)
        add_executable(gpu_original_sparse_geometry_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/sparse_probe_device_test.cu")
        ipc_gpu_opt_compile(gpu_original_sparse_geometry_test)
        target_link_libraries(gpu_original_sparse_geometry_test PRIVATE CUDA::cudart)
        ipc_gpu_selected_role(gpu_original_sparse_geometry_test STANDALONE)
        ipc_gpu_opt_test(gpu_original_sparse_geometry gpu_original_sparse_geometry_test FALSE)
        add_executable(gpu_original_sparse_facade_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/sparse_gather_native_test.cpp")
        ipc_gpu_opt_compile(gpu_original_sparse_facade_test)
        target_link_libraries(gpu_original_sparse_facade_test PRIVATE ipc_gpu_solver)
        ipc_gpu_selected_role(gpu_original_sparse_facade_test FACADE)
        ipc_gpu_opt_test(gpu_original_sparse_facade gpu_original_sparse_facade_test FALSE)
        add_executable(gpu_original_sparse_public_test EXCLUDE_FROM_ALL
            "${_ipc_opt_tests}/sparse_public_native_test.cpp")
        ipc_gpu_opt_compile(gpu_original_sparse_public_test)
        target_link_libraries(gpu_original_sparse_public_test PRIVATE ipc_gpu_solver)
        target_link_options(gpu_original_sparse_public_test PRIVATE
            "-Wl,--wrap=cudaMalloc" "-Wl,--wrap=cudaFree")
        ipc_gpu_selected_role(gpu_original_sparse_public_test PUBLIC)
        ipc_gpu_opt_test(gpu_original_sparse_public gpu_original_sparse_public_test FALSE)
        list(APPEND _ipc_selected_targets gpu_original_sparse_policy_test
            gpu_original_sparse_geometry_test gpu_original_sparse_facade_test gpu_original_sparse_public_test)
    endif()


    if(_ipc_selected_targets)
        add_dependencies(gpu_optimization_validation ${_ipc_selected_targets})
    endif()
endif()
