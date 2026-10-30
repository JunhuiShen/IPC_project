# Explicit current-backend qualification targets and support kernels; none join
# ALL. Test names, inputs, precision settings and assertions are retained; only
# the retired naming changed.

# Every test registered in this file exercises the current device solver, so it
# carries the gpu_solver label. `ctest -L gpu_solver` stays away from the
# legacy hybrid tests that are only built on request.
macro(ipc_gpu_solver_test name)
    add_test(NAME ${name} COMMAND ${ARGN})
    set_tests_properties(${name} PROPERTIES LABELS gpu_solver)
endmacro()

add_executable(gpu_elastic_test EXCLUDE_FROM_ALL elastic_test.cpp)
target_compile_options(gpu_elastic_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_elastic_test PRIVATE ipc_gpu_components)
add_executable(gpu_broadphase_test EXCLUDE_FROM_ALL broadphase_test.cpp)
target_link_libraries(gpu_broadphase_test PRIVATE ipc_gpu_components)
add_library(ipc_gpu_contact_test_kernel STATIC EXCLUDE_FROM_ALL contact_test_cuda.cu)
target_link_libraries(ipc_gpu_contact_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_gpu_contact_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_contact_test EXCLUDE_FROM_ALL contact_test.cpp)
target_compile_definitions(gpu_contact_test PRIVATE IPC_GPU_CONTACT_CUDA_TEST=1)
target_link_libraries(gpu_contact_test PRIVATE IPC::core ipc_gpu_contact_test_kernel)
ipc_gpu_solver_test(gpu_elastic gpu_elastic_test)
add_executable(gpu_elastic_worklist_test EXCLUDE_FROM_ALL elastic_worklist_test.cpp)
target_compile_options(gpu_elastic_worklist_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_elastic_worklist_test PRIVATE ipc_gpu_components)
ipc_gpu_solver_test(gpu_elastic_worklist gpu_elastic_worklist_test)
ipc_gpu_solver_test(gpu_broadphase gpu_broadphase_test)
ipc_gpu_solver_test(gpu_warp_dispatch_degree gpu_broadphase_test --warp-dispatch degree)
ipc_gpu_solver_test(gpu_warp_dispatch_spatial gpu_broadphase_test --warp-dispatch spatial)
ipc_gpu_solver_test(gpu_broadphase_options_degree gpu_broadphase_test --broadphase-options degree)
ipc_gpu_solver_test(gpu_broadphase_options_spatial gpu_broadphase_test --broadphase-options spatial)
ipc_gpu_solver_test(gpu_contact gpu_contact_test)

# CPU-oracle parity for the optional simultaneous Jacobi/DAT path and the
# rigid material-point springs. Native tests invoke the production kernels.
add_executable(gpu_dat_test EXCLUDE_FROM_ALL dat_test.cpp)
target_compile_options(gpu_dat_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_dat_test PRIVATE IPC::ipc ipc_gpu_solver)
ipc_gpu_solver_test(gpu_dat gpu_dat_test)
add_executable(gpu_dat_geometry_test EXCLUDE_FROM_ALL dat_geometry_test.cpp)
target_compile_options(gpu_dat_geometry_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_dat_geometry_test PRIVATE IPC::ipc ipc_gpu_solver)
ipc_gpu_solver_test(gpu_dat_geometry gpu_dat_geometry_test)
add_executable(gpu_rigid_pins_test EXCLUDE_FROM_ALL
    rigid_pins_test.cpp rigid_pins_test.cu)
target_compile_options(gpu_rigid_pins_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
target_link_libraries(gpu_rigid_pins_test PRIVATE IPC::ipc ipc_gpu_solver)
ipc_gpu_solver_test(gpu_rigid_pins gpu_rigid_pins_test)
add_executable(gpu_cpu_parity_test EXCLUDE_FROM_ALL cpu_parity_test.cpp)
target_compile_options(gpu_cpu_parity_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_cpu_parity_test PRIVATE IPC::ipc ipc_gpu_bridge)
ipc_gpu_solver_test(gpu_cpu_parity gpu_cpu_parity_test)
add_executable(gpu_contact_rejection_test EXCLUDE_FROM_ALL
    contact_rejection_test.cpp contact_rejection_test_cuda.cu)
target_compile_definitions(gpu_contact_rejection_test PRIVATE IPC_GPU_CONTACT_REJECTION_CUDA_TEST=1)
target_include_directories(gpu_contact_rejection_test PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(gpu_contact_rejection_test PRIVATE CUDA::cudart)
target_compile_options(gpu_contact_rejection_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
ipc_gpu_solver_test(gpu_contact_rejection gpu_contact_rejection_test)
add_library(ipc_gpu_solve_test_kernel STATIC EXCLUDE_FROM_ALL solve_test_cuda.cu)
target_link_libraries(ipc_gpu_solve_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_gpu_solve_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_solve_test EXCLUDE_FROM_ALL solve_test.cpp)
target_link_libraries(gpu_solve_test PRIVATE IPC::core ipc_gpu_solve_test_kernel)
ipc_gpu_solver_test(gpu_solve gpu_solve_test)
add_library(ipc_gpu_rigid_test_kernel STATIC EXCLUDE_FROM_ALL rigid_test_cuda.cu)
target_link_libraries(ipc_gpu_rigid_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_gpu_rigid_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_rigid_test EXCLUDE_FROM_ALL rigid_test.cpp)
target_link_libraries(gpu_rigid_test PRIVATE IPC::core ipc_gpu_rigid_test_kernel)
ipc_gpu_solver_test(gpu_rigid gpu_rigid_test)
add_library(ipc_gpu_ccd_test_kernel STATIC EXCLUDE_FROM_ALL ccd_cuda_test.cu)
target_link_libraries(ipc_gpu_ccd_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_gpu_ccd_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_ccd_test EXCLUDE_FROM_ALL ccd_test.cpp)
target_compile_definitions(gpu_ccd_test PRIVATE IPC_GPU_CUDA_TEST=1 IPC_GPU_PRODUCTION_REFERENCE=1)
target_link_libraries(gpu_ccd_test PRIVATE IPC::core Boost::headers ipc_gpu_ccd_test_kernel)
ipc_gpu_solver_test(gpu_ccd gpu_ccd_test)
add_executable(gpu_exact_pool_test EXCLUDE_FROM_ALL exact_pool_test.cu)
target_link_libraries(gpu_exact_pool_test PRIVATE CUDA::cudart)
target_compile_options(gpu_exact_pool_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
ipc_gpu_solver_test(gpu_exact_pool gpu_exact_pool_test)
add_executable(gpu_ratio_host_test EXCLUDE_FROM_ALL ratio_differential_test.cpp)
target_compile_options(gpu_ratio_host_test PRIVATE -ffp-contract=off)
ipc_gpu_solver_test(gpu_ratio_host gpu_ratio_host_test)
add_executable(gpu_ratio_cuda_test EXCLUDE_FROM_ALL ratio_differential_test.cu)
target_link_libraries(gpu_ratio_cuda_test PRIVATE CUDA::cudart)
target_compile_options(gpu_ratio_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
ipc_gpu_solver_test(gpu_ratio_cuda gpu_ratio_cuda_test)
add_library(ipc_gpu_sdf_test_kernel STATIC EXCLUDE_FROM_ALL sdf_test_cuda.cu)
target_link_libraries(ipc_gpu_sdf_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_gpu_sdf_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_sdf_test EXCLUDE_FROM_ALL sdf_test.cpp)
target_compile_definitions(gpu_sdf_test PRIVATE IPC_GPU_SDF_CUDA_TEST=1)
target_link_libraries(gpu_sdf_test PRIVATE IPC::core ipc_gpu_sdf_test_kernel)
ipc_gpu_solver_test(gpu_sdf gpu_sdf_test)
add_library(ipc_gpu_rigid_ccd_test_kernel STATIC EXCLUDE_FROM_ALL rigid_ccd_test_cuda.cu)
target_link_libraries(ipc_gpu_rigid_ccd_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_gpu_rigid_ccd_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_rigid_ccd_test EXCLUDE_FROM_ALL rigid_ccd_test.cpp)
target_link_libraries(gpu_rigid_ccd_test PRIVATE IPC::ipc ipc_gpu_rigid_ccd_test_kernel)
ipc_gpu_solver_test(gpu_rigid_ccd gpu_rigid_ccd_test)

add_library(ipc_gpu_contact_pipeline_test_kernel STATIC EXCLUDE_FROM_ALL contact_pipeline_test_cuda.cu)
target_link_libraries(ipc_gpu_contact_pipeline_test_kernel PUBLIC ipc_gpu_solver CUDA::cudart)
target_compile_options(ipc_gpu_contact_pipeline_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_contact_pipeline_test EXCLUDE_FROM_ALL contact_pipeline_test.cpp)
target_link_libraries(gpu_contact_pipeline_test PRIVATE IPC::core ipc_gpu_contact_pipeline_test_kernel)
ipc_gpu_solver_test(gpu_contact_pipeline gpu_contact_pipeline_test)
add_executable(gpu_contact_queue_test EXCLUDE_FROM_ALL contact_queue_test.cu)
target_link_libraries(gpu_contact_queue_test PRIVATE ipc_gpu_solver CUDA::cudart)
target_compile_options(gpu_contact_queue_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
ipc_gpu_solver_test(gpu_contact_queue gpu_contact_queue_test)
add_executable(gpu_contact_warp_reduce_test EXCLUDE_FROM_ALL contact_warp_reduce_test.cu)
target_link_libraries(gpu_contact_warp_reduce_test PRIVATE CUDA::cudart)
target_compile_options(gpu_contact_warp_reduce_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
ipc_gpu_solver_test(gpu_contact_warp_reduce gpu_contact_warp_reduce_test)
add_executable(gpu_contact_worklist_test EXCLUDE_FROM_ALL contact_worklist_test.cu)
target_link_libraries(gpu_contact_worklist_test PRIVATE ipc_gpu_solver CUDA::cudart)
target_compile_options(gpu_contact_worklist_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
ipc_gpu_solver_test(gpu_contact_worklist gpu_contact_worklist_test)
# Exercise the opt-in sole-self ordinary evaluator against the fixture's full
# generic path. The fixture compares derivative bytes and exact/error routing.
add_test(NAME gpu_contact_selected_ordinary COMMAND gpu_contact_worklist_test)
set_tests_properties(gpu_contact_selected_ordinary PROPERTIES
    LABELS gpu_solver
    ENVIRONMENT "IPC_GPU_SELECTED_ORDINARY_GATHER=1;IPC_GPU_DEFER_CONTACT_INPUTS=1")
add_executable(gpu_contact_compact_test EXCLUDE_FROM_ALL contact_compact_test.cpp)
ipc_gpu_solver_test(gpu_contact_compact gpu_contact_compact_test)
add_executable(gpu_box_certificate_test EXCLUDE_FROM_ALL box_certificate_test.cpp box_certificate_test.cu)
target_compile_definitions(gpu_box_certificate_test PRIVATE IPC_GPU_BOX_CERTIFICATE_CUDA_TEST=1)
target_include_directories(gpu_box_certificate_test PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(gpu_box_certificate_test PRIVATE CUDA::cudart)
target_compile_options(gpu_box_certificate_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
ipc_gpu_solver_test(gpu_box_certificate gpu_box_certificate_test)
add_executable(gpu_box_pruning_test EXCLUDE_FROM_ALL box_pruning_test.cu)
target_link_libraries(gpu_box_pruning_test PRIVATE ipc_gpu_solver CUDA::cudart)
target_compile_options(gpu_box_pruning_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
ipc_gpu_solver_test(gpu_box_pruning gpu_box_pruning_test)
add_executable(gpu_contact_cooperative_test EXCLUDE_FROM_ALL contact_cooperative_test.cu)
target_link_libraries(gpu_contact_cooperative_test PRIVATE ipc_gpu_solver CUDA::cudart)
target_compile_options(gpu_contact_cooperative_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
ipc_gpu_solver_test(gpu_contact_cooperative gpu_contact_cooperative_test)
add_executable(gpu_contact_schedule_test EXCLUDE_FROM_ALL contact_schedule_test.cpp)
ipc_gpu_solver_test(gpu_contact_schedule gpu_contact_schedule_test)
add_executable(gpu_solver_test EXCLUDE_FROM_ALL solver_test.cpp)
target_link_libraries(gpu_solver_test PRIVATE IPC::ipc ipc_gpu_bridge)
ipc_gpu_solver_test(gpu_solver gpu_solver_test)
add_executable(gpu_adaptive_boxes_test EXCLUDE_FROM_ALL adaptive_boxes_test.cpp)
target_link_libraries(gpu_adaptive_boxes_test PRIVATE ipc_gpu_solver Eigen3::Eigen)
target_compile_options(gpu_adaptive_boxes_test PRIVATE -ffp-contract=off)
ipc_gpu_solver_test(gpu_adaptive_boxes gpu_adaptive_boxes_test)
add_executable(gpu_global_guess_test EXCLUDE_FROM_ALL global_guess_test.cpp)
target_compile_definitions(gpu_global_guess_test PRIVATE IPC_GLOBAL_GUESS_CUDA_TEST=1)
target_compile_options(gpu_global_guess_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_global_guess_test PRIVATE ipc_gpu_solver tight_inclusion::tight_inclusion Threads::Threads)
ipc_gpu_solver_test(gpu_global_guess gpu_global_guess_test)
add_executable(gpu_linear_pipeline_test EXCLUDE_FROM_ALL linear_pipeline_test.cpp)
target_link_libraries(gpu_linear_pipeline_test PRIVATE ipc_gpu_solver)
ipc_gpu_solver_test(gpu_linear_pipeline gpu_linear_pipeline_test)
add_executable(gpu_rigid_pipeline_test EXCLUDE_FROM_ALL rigid_pipeline_test.cpp)
target_link_libraries(gpu_rigid_pipeline_test PRIVATE ipc_gpu_solver IPC::ipc)
ipc_gpu_solver_test(gpu_rigid_pipeline gpu_rigid_pipeline_test)

# Independent qualified ORIGINAL (exact_computation_fallback == false)
# components. The component sources stay read-only; these targets only compile
# and run them with their existing macros/markers, --fmad=false and host
# contraction disabled.
add_library(ipc_gpu_original_contact_test_kernel STATIC EXCLUDE_FROM_ALL
    original_contact_test_cuda.cu)
target_link_libraries(ipc_gpu_original_contact_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_gpu_original_contact_test_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_original_contact_test EXCLUDE_FROM_ALL original_contact_test.cpp)
target_compile_definitions(gpu_original_contact_test PRIVATE IPC_GPU_ORIGINAL_CUDA_TEST=1)
target_compile_options(gpu_original_contact_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_original_contact_test PRIVATE
    IPC::ipc Boost::headers ipc_gpu_original_contact_test_kernel)
ipc_gpu_solver_test(gpu_original_contact gpu_original_contact_test)

add_library(ipc_gpu_original_ccd_test_kernel STATIC EXCLUDE_FROM_ALL
    original_ccd_test_cuda.cu)
target_link_libraries(ipc_gpu_original_ccd_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_gpu_original_ccd_test_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_original_ccd_test EXCLUDE_FROM_ALL original_ccd_test.cpp)
target_compile_definitions(gpu_original_ccd_test PRIVATE
    IPC_GPU_CUDA_TEST=1 IPC_GPU_PRODUCTION_REFERENCE=1)
target_compile_options(gpu_original_ccd_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_original_ccd_test PRIVATE
    IPC::ipc Boost::headers ipc_gpu_original_ccd_test_kernel)
ipc_gpu_solver_test(gpu_original_ccd gpu_original_ccd_test)

# Actual original-mode pipeline/solver integration (routing + CPU false oracle
# checks). Component tests alone do not prove mode routing.
add_executable(gpu_original_mode_test EXCLUDE_FROM_ALL
    original_mode_test.cpp original_mode_test.cu)
target_compile_options(gpu_original_mode_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
target_link_libraries(gpu_original_mode_test PRIVATE
    IPC::ipc ipc_gpu_bridge CUDA::cudart)
ipc_gpu_solver_test(gpu_original_mode gpu_original_mode_test)
