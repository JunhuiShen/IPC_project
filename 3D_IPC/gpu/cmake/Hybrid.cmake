# Legacy targets remain available explicitly, without joining ordinary ALL.

add_library(ipc_cuda STATIC EXCLUDE_FROM_ALL kernels.cu)
target_include_directories(ipc_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_cuda PUBLIC CUDA::cudart)
target_compile_options(ipc_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)

# Keep production files unchanged. Generate a checked overlay in THIS build
# directory only; fail if an upstream edit invalidates the insertion anchors.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${IPC_CPU_SOURCE}/solver.cpp" "${IPC_CPU_SOURCE}/simulation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/color_sweep.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/prepare_color.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/before_color.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/general_sweep.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/driver_guard.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/driver_stats.inc")
ipc_gpu_generate_hybrid_sources()

# Keep upstream IPC::ipc unchanged. Extract this overlay first for legacy callers;
# its complete solver object provides the public solver definitions, so the
# original CPU solver object is not extracted from the later CPU base archive.
add_library(ipc_gpu_hybrid_overlay STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/solver_gpu.cpp"
    general_bridge.cpp rigid_bridge.cpp)
target_include_directories(ipc_gpu_hybrid_overlay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_gpu_hybrid_overlay PUBLIC IPC::ipc ipc_cuda)
set_target_properties(ipc_gpu_hybrid_overlay PROPERTIES
    CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)

add_executable(3D_sim_gpu EXCLUDE_FROM_ALL "${CMAKE_CURRENT_BINARY_DIR}/simulation_gpu.cpp")
target_link_libraries(3D_sim_gpu PRIVATE ipc_gpu_hybrid_overlay IPC::ipc Threads::Threads)

add_executable(gpu_kernel_test EXCLUDE_FROM_ALL kernel_test.cpp)
target_link_libraries(gpu_kernel_test PRIVATE IPC::core ipc_cuda Threads::Threads)
add_executable(gpu_solid_kernel_test EXCLUDE_FROM_ALL solid_kernel_test.cpp)
target_link_libraries(gpu_solid_kernel_test PRIVATE IPC::core ipc_cuda Threads::Threads)
target_compile_options(gpu_solid_kernel_test PRIVATE -ffp-contract=off)
add_executable(gpu_rigid_kernel_test EXCLUDE_FROM_ALL rigid_kernel_test.cpp)
target_link_libraries(gpu_rigid_kernel_test PRIVATE ipc_gpu_hybrid_overlay IPC::ipc Threads::Threads)

# An otherwise-identical host build for integration comparisons (same flags,
# dependencies and core kernels, unlike a potentially older production binary).
# Preserve the explicit original CPU solver control and its historical target name.
add_library(ipc_cpu_control STATIC EXCLUDE_FROM_ALL "${IPC_CPU_SOURCE}/solver.cpp")
target_link_libraries(ipc_cpu_control PUBLIC IPC::ipc)
add_executable(3D_sim_cpu_control EXCLUDE_FROM_ALL "${IPC_CPU_SOURCE}/simulation.cpp")
target_link_libraries(3D_sim_cpu_control PRIVATE ipc_cpu_control Threads::Threads)
add_executable(gpu_contact_fixture EXCLUDE_FROM_ALL contact_fixture.cpp)
target_compile_definitions(gpu_contact_fixture PRIVATE IPC_GPU_FIXTURE=1)
target_link_libraries(gpu_contact_fixture PRIVATE ipc_gpu_hybrid_overlay IPC::ipc Threads::Threads)
add_executable(cpu_contact_fixture EXCLUDE_FROM_ALL contact_fixture.cpp)
target_link_libraries(cpu_contact_fixture PRIVATE ipc_cpu_control Threads::Threads)
add_executable(gpu_mixed_fixture EXCLUDE_FROM_ALL mixed_fixture.cpp)
target_compile_definitions(gpu_mixed_fixture PRIVATE IPC_GPU_FIXTURE=1)
target_link_libraries(gpu_mixed_fixture PRIVATE ipc_gpu_hybrid_overlay IPC::ipc Threads::Threads)
add_executable(cpu_mixed_fixture EXCLUDE_FROM_ALL mixed_fixture.cpp)
target_link_libraries(cpu_mixed_fixture PRIVATE ipc_cpu_control Threads::Threads)
add_test(NAME gpu_kernel_correctness COMMAND gpu_kernel_test)
add_test(NAME gpu_solid_correctness COMMAND gpu_solid_kernel_test)
add_test(NAME gpu_rigid_correctness COMMAND gpu_rigid_kernel_test)


add_custom_target(gpu_hybrid_validation DEPENDS
    3D_sim_gpu 3D_sim_cpu_control gpu_kernel_test gpu_solid_kernel_test
    gpu_rigid_kernel_test gpu_contact_fixture cpu_contact_fixture
    gpu_mixed_fixture cpu_mixed_fixture)
