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
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(ipc_cuda STATIC kernels.cu)
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
function(replace_once variable old new)
    string(FIND "${${variable}}" "${old}" found)
    if(found LESS 0)
        message(FATAL_ERROR "GPU overlay anchor missing: ${old}")
    endif()
    string(LENGTH "${old}" length)
    math(EXPR after "${found}+${length}")
    string(SUBSTRING "${${variable}}" ${after} -1 suffix)
    string(FIND "${suffix}" "${old}" duplicate)
    if(NOT duplicate LESS 0)
        message(FATAL_ERROR "GPU overlay anchor is ambiguous: ${old}")
    endif()
    string(REPLACE "${old}" "${new}" result "${${variable}}")
    set(${variable} "${result}" PARENT_SCOPE)
endfunction()
file(READ "${IPC_CPU_SOURCE}/solver.cpp" solver_source)
set(start "SolverResult global_gauss_seidel_solver_basic_experimental_v2(")
set(finish "// Same per-vertex numerical updates as basic; schedule independent grid cells")
string(FIND "${solver_source}" "${start}" begin)
string(FIND "${solver_source}" "${finish}" end)
if(begin LESS 0 OR end LESS begin)
    message(FATAL_ERROR "Cannot locate basic-v2 solver for isolated GPU overlay")
endif()
math(EXPR length "${end}-${begin}")
string(SUBSTRING "${solver_source}" 0 ${begin} prefix)
string(SUBSTRING "${solver_source}" ${begin} ${length} body)
string(SUBSTRING "${solver_source}" ${end} -1 suffix)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/color_sweep.inc" sweep)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/prepare_color.inc" prepare_color)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/before_color.inc" before_color)
replace_once(body "    const auto prepare_color_derivatives =" "${prepare_color}\n    const auto prepare_color_derivatives =")
replace_once(body "        if (use_v2_simd) return;" "${before_color}")
# The unchanged CPU contact sweep's callbacks must consume the color outputs,
# never recompute an 8-wide CPU batch or rebind those shared views.
replace_once(body "        // Per-worker AoS results live through the following vertex updates."
    "        if (use_contact_sweep) return; // GPU color already prepared collectively\n        // Per-worker AoS results live through the following vertex updates.")
replace_once(body "        if (use_contact_sweep) {" "${sweep}")
set(gpu_solver_source "${prefix}${body}${suffix}")

# General-v2 keeps the original block/color schedule and CPU safety checks.
# Gather cloth + solid records once per color, and consume the cached CUDA
# derivatives from the unchanged cooperative contact batches.
set(start "SolverResult global_gauss_seidel_solver_general_experimental_v2(")
string(FIND "${gpu_solver_source}" "${start}" begin)
if(begin LESS 0)
    message(FATAL_ERROR "Cannot locate general-v2 solver for GPU overlay")
endif()
string(SUBSTRING "${gpu_solver_source}" 0 ${begin} prefix)
string(SUBSTRING "${gpu_solver_source}" ${begin} -1 body)
replace_once(body "    solver_detail::GeneralSimdBatches simd_batches;"
    "    solver_detail::GeneralSimdBatches simd_batches;\n    ipc_gpu::GeneralColorBridge gpu_general;")
replace_once(body "solver_detail::prepare_general_simd_batch(" "gpu_general.prepare_batch(")
set(sweep_begin "        if (params.use_parallel) {\n            const int sweeps = params.fixed_iters")
set(sweep_end "        result.iterations = iteration;")
string(FIND "${body}" "${sweep_begin}" begin)
string(FIND "${body}" "${sweep_end}" end)
if(begin LESS 0 OR end LESS begin)
    message(FATAL_ERROR "Cannot locate general-v2 sweep for GPU overlay")
endif()
string(SUBSTRING "${body}" 0 ${begin} before)
string(SUBSTRING "${body}" ${end} -1 after)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/general_sweep.inc" general_sweep)
set(gpu_solver_source "${prefix}${before}${general_sweep}\n${after}")

# Rigid COM and omega remain sequential. Each GPU contact gather sees the
# current proxy positions, including the immediately preceding COM update.
# Scope replacements to the two update functions, not the CPU residual path.
set(start "Vec3 compute_com_update(")
set(finish "} // namespace rb_solver")
string(FIND "${gpu_solver_source}" "${start}" begin)
if(begin LESS 0)
    message(FATAL_ERROR "Cannot locate rigid update functions for GPU overlay")
endif()
string(SUBSTRING "${gpu_solver_source}" ${begin} -1 remaining)
string(FIND "${remaining}" "${finish}" length)
if(length LESS 0)
    message(FATAL_ERROR "Cannot locate rigid update functions for GPU overlay")
endif()
math(EXPR end "${begin}+${length}")
string(SUBSTRING "${gpu_solver_source}" 0 ${begin} prefix)
string(SUBSTRING "${gpu_solver_source}" ${begin} ${length} body)
string(SUBSTRING "${gpu_solver_source}" ${end} -1 suffix)
string(REGEX MATCHALL "ipc_simd::rigid_contact_derivatives\\(" rigid_calls "${body}")
list(LENGTH rigid_calls rigid_call_count)
if(NOT rigid_call_count EQUAL 2)
    message(FATAL_ERROR "Expected two rigid contact update calls; found ${rigid_call_count}")
endif()
string(REPLACE "ipc_simd::rigid_contact_derivatives(" "ipc_gpu::rigid_contact_derivatives(" body "${body}")
set(com_original [=[        gradient = inertia_translation_gradient(x_com_new[rb], x_com_n, v_com_n, dt, ref_mesh.total_mass[rb]);
        gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);

        hessian = inertia_translation_hessian(ref_mesh.total_mass[rb]);]=])
set(com_gpu [=[        if (general_simd_v2) {
            const auto derivatives = ipc_gpu::rigid_translation_inertia_gravity(
                x_com_new[rb], x_com_n, v_com_n, dt, ref_mesh.total_mass[rb], params.gravity.y());
            gradient = derivatives.first;
            hessian = derivatives.second;
        } else {
]=])
replace_once(body "${com_original}" "${com_gpu}${com_original}\n        }")
set(omega_original "        const auto derivatives = inertia_rotation_gradient_hessian(omega_new[rb], q_n, omega_n, dt, I_hat, &kinematics, rotation_predictor);")
replace_once(body "${omega_original}" [=[        const auto derivatives = general_simd_v2
            ? ipc_gpu::rigid_rotation_inertia(omega_new[rb], q_n, omega_n, dt, I_hat, &kinematics, rotation_predictor)
            : inertia_rotation_gradient_hessian(omega_new[rb], q_n, omega_n, dt, I_hat, &kinematics, rotation_predictor);]=])
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/solver_gpu.cpp"
    "#include \"color_bridge.h\"\n#include \"general_bridge.h\"\n#include \"rigid_bridge.h\"\n${prefix}${body}${suffix}")
get_target_property(host_sources ipc SOURCES)
set(gpu_host_sources)
foreach(source IN LISTS host_sources)
    if(source STREQUAL "solver.cpp")
        list(APPEND gpu_host_sources "${CMAKE_CURRENT_BINARY_DIR}/solver_gpu.cpp")
    elseif(IS_ABSOLUTE "${source}")
        list(APPEND gpu_host_sources "${source}")
    else()
        list(APPEND gpu_host_sources "${IPC_CPU_SOURCE}/${source}")
    endif()
endforeach()
set_property(TARGET ipc PROPERTY SOURCES "${gpu_host_sources}")
target_sources(ipc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/general_bridge.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/rigid_bridge.cpp")
target_include_directories(ipc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc PUBLIC ipc_cuda)

file(READ "${IPC_CPU_SOURCE}/simulation.cpp" driver)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/driver_guard.inc" guard)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/driver_stats.inc" stats)
replace_once(driver "    const bool uses_general_solver = is_mixed || has_solid;" "${guard}\n    const bool uses_general_solver = is_mixed || has_solid;")
set(timing_anchor "    std::cout << \"Total solver time:     \"")
replace_once(driver "${timing_anchor}" "${stats}\n${timing_anchor}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/simulation_gpu.cpp" "#include \"kernels.h\"\n${driver}")
add_executable(3D_sim_gpu "${CMAKE_CURRENT_BINARY_DIR}/simulation_gpu.cpp")
target_link_libraries(3D_sim_gpu PRIVATE IPC::ipc Threads::Threads)

add_executable(gpu_kernel_test kernel_test.cpp)
target_link_libraries(gpu_kernel_test PRIVATE IPC::core ipc_cuda Threads::Threads)
add_executable(gpu_solid_kernel_test solid_kernel_test.cpp)
target_link_libraries(gpu_solid_kernel_test PRIVATE IPC::core ipc_cuda Threads::Threads)
target_compile_options(gpu_solid_kernel_test PRIVATE -ffp-contract=off)
add_executable(gpu_rigid_kernel_test rigid_kernel_test.cpp)
target_link_libraries(gpu_rigid_kernel_test PRIVATE IPC::ipc Threads::Threads)

# An otherwise-identical host build for integration comparisons (same flags,
# dependencies and core kernels, unlike a potentially older production binary).
# Link the original solver first; the overlay archive object is not extracted.
add_library(ipc_cpu_control STATIC "${IPC_CPU_SOURCE}/solver.cpp")
target_link_libraries(ipc_cpu_control PUBLIC IPC::ipc)
add_executable(3D_sim_cpu_control "${IPC_CPU_SOURCE}/simulation.cpp")
target_link_libraries(3D_sim_cpu_control PRIVATE ipc_cpu_control Threads::Threads)
add_executable(gpu_contact_fixture contact_fixture.cpp)
target_compile_definitions(gpu_contact_fixture PRIVATE IPC_GPU_FIXTURE=1)
target_link_libraries(gpu_contact_fixture PRIVATE IPC::ipc Threads::Threads)
add_executable(cpu_contact_fixture contact_fixture.cpp)
target_link_libraries(cpu_contact_fixture PRIVATE ipc_cpu_control Threads::Threads)
add_executable(gpu_mixed_fixture mixed_fixture.cpp)
target_compile_definitions(gpu_mixed_fixture PRIVATE IPC_GPU_FIXTURE=1)
target_link_libraries(gpu_mixed_fixture PRIVATE IPC::ipc Threads::Threads)
add_executable(cpu_mixed_fixture mixed_fixture.cpp)
target_link_libraries(cpu_mixed_fixture PRIVATE ipc_cpu_control Threads::Threads)
enable_testing()
add_test(NAME gpu_kernel_correctness COMMAND gpu_kernel_test)
add_test(NAME gpu_solid_correctness COMMAND gpu_solid_kernel_test)
add_test(NAME gpu_rigid_correctness COMMAND gpu_rigid_kernel_test)

# GPU-resident building blocks. These are deliberately separate from the
# hybrid executable; passing component tests does not certify a full solver.
add_library(ipc_resident_components STATIC resident_elastic.cu resident_broadphase.cu)
target_include_directories(ipc_resident_components PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_components PUBLIC CUDA::cudart)
target_compile_options(ipc_resident_components PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_resident_elastic_test resident_elastic_test.cpp)
target_compile_options(gpu_resident_elastic_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_resident_elastic_test PRIVATE ipc_resident_components)
add_executable(gpu_resident_broadphase_test resident_broadphase_test.cpp)
target_link_libraries(gpu_resident_broadphase_test PRIVATE ipc_resident_components)
add_library(ipc_resident_contact_test_kernel STATIC resident_contact_test_cuda.cu)
target_link_libraries(ipc_resident_contact_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_resident_contact_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_resident_contact_test resident_contact_test.cpp)
target_compile_definitions(gpu_resident_contact_test PRIVATE IPC_RESIDENT_CONTACT_CUDA_TEST=1)
target_link_libraries(gpu_resident_contact_test PRIVATE IPC::core ipc_resident_contact_test_kernel)
add_test(NAME gpu_resident_elastic COMMAND gpu_resident_elastic_test)
add_executable(gpu_resident_elastic_worklist_test resident_elastic_worklist_test.cpp)
target_compile_options(gpu_resident_elastic_worklist_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_resident_elastic_worklist_test PRIVATE ipc_resident_components)
add_test(NAME gpu_resident_elastic_worklist COMMAND gpu_resident_elastic_worklist_test)
add_test(NAME gpu_resident_broadphase COMMAND gpu_resident_broadphase_test)
add_test(NAME gpu_resident_contact COMMAND gpu_resident_contact_test)
add_executable(gpu_resident_contact_rejection_test
    resident_contact_rejection_test.cpp resident_contact_rejection_test_cuda.cu)
target_compile_definitions(gpu_resident_contact_rejection_test PRIVATE IPC_RESIDENT_CONTACT_REJECTION_CUDA_TEST=1)
target_include_directories(gpu_resident_contact_rejection_test PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(gpu_resident_contact_rejection_test PRIVATE CUDA::cudart)
target_compile_options(gpu_resident_contact_rejection_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
add_test(NAME gpu_resident_contact_rejection COMMAND gpu_resident_contact_rejection_test)
add_library(ipc_resident_solve_test_kernel STATIC resident_solve_test_cuda.cu)
target_link_libraries(ipc_resident_solve_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_resident_solve_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_resident_solve_test resident_solve_test.cpp)
target_link_libraries(gpu_resident_solve_test PRIVATE IPC::core ipc_resident_solve_test_kernel)
add_test(NAME gpu_resident_solve COMMAND gpu_resident_solve_test)
add_library(ipc_resident_rigid_test_kernel STATIC resident_rigid_test_cuda.cu)
target_link_libraries(ipc_resident_rigid_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_resident_rigid_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_resident_rigid_test resident_rigid_test.cpp)
target_link_libraries(gpu_resident_rigid_test PRIVATE IPC::core ipc_resident_rigid_test_kernel)
add_test(NAME gpu_resident_rigid COMMAND gpu_resident_rigid_test)
add_library(ipc_resident_ccd_test_kernel STATIC resident_ccd_cuda_test.cu)
target_link_libraries(ipc_resident_ccd_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_resident_ccd_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_resident_ccd_test resident_ccd_test.cpp)
target_compile_definitions(gpu_resident_ccd_test PRIVATE IPC_RESIDENT_CUDA_TEST=1 IPC_RESIDENT_PRODUCTION_REFERENCE=1)
target_link_libraries(gpu_resident_ccd_test PRIVATE IPC::core ipc_resident_ccd_test_kernel)
add_test(NAME gpu_resident_ccd COMMAND gpu_resident_ccd_test)
add_executable(gpu_resident_exact_pool_test resident_exact_pool_test.cu)
target_link_libraries(gpu_resident_exact_pool_test PRIVATE CUDA::cudart)
target_compile_options(gpu_resident_exact_pool_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_test(NAME gpu_resident_exact_pool COMMAND gpu_resident_exact_pool_test)
add_executable(gpu_resident_ratio_host_test resident_ratio_differential_test.cpp)
target_compile_options(gpu_resident_ratio_host_test PRIVATE -ffp-contract=off)
add_test(NAME gpu_resident_ratio_host COMMAND gpu_resident_ratio_host_test)
add_executable(gpu_resident_ratio_cuda_test resident_ratio_differential_test.cu)
target_link_libraries(gpu_resident_ratio_cuda_test PRIVATE CUDA::cudart)
target_compile_options(gpu_resident_ratio_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_test(NAME gpu_resident_ratio_cuda COMMAND gpu_resident_ratio_cuda_test)
add_library(ipc_resident_sdf_test_kernel STATIC resident_sdf_test_cuda.cu)
target_link_libraries(ipc_resident_sdf_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_resident_sdf_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_resident_sdf_test resident_sdf_test.cpp)
target_compile_definitions(gpu_resident_sdf_test PRIVATE IPC_RESIDENT_SDF_CUDA_TEST=1)
target_link_libraries(gpu_resident_sdf_test PRIVATE IPC::core ipc_resident_sdf_test_kernel)
add_test(NAME gpu_resident_sdf COMMAND gpu_resident_sdf_test)
add_library(ipc_resident_rigid_ccd_test_kernel STATIC resident_rigid_ccd_test_cuda.cu)
target_link_libraries(ipc_resident_rigid_ccd_test_kernel PUBLIC CUDA::cudart)
target_compile_options(ipc_resident_rigid_ccd_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_resident_rigid_ccd_test resident_rigid_ccd_test.cpp)
target_link_libraries(gpu_resident_rigid_ccd_test PRIVATE IPC::ipc ipc_resident_rigid_ccd_test_kernel)
add_test(NAME gpu_resident_rigid_ccd COMMAND gpu_resident_rigid_ccd_test)

add_library(ipc_resident_solver STATIC resident_solver.cu resident_contact_pipeline.cu
    resident_contact_cooperative.cu
    resident_linear_pipeline.cu resident_rigid_pipeline.cu
    resident_adaptive_boxes.cu resident_global_guess.cu)
target_include_directories(ipc_resident_solver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_solver PUBLIC ipc_resident_components CUDA::cudart)
target_compile_options(ipc_resident_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_library(ipc_resident_bridge STATIC resident_bridge.cpp)
target_include_directories(ipc_resident_bridge PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ipc_resident_bridge PUBLIC IPC::core ipc_resident_solver)
add_library(ipc_resident_contact_pipeline_test_kernel STATIC resident_contact_pipeline_test_cuda.cu)
target_link_libraries(ipc_resident_contact_pipeline_test_kernel PUBLIC ipc_resident_solver CUDA::cudart)
target_compile_options(ipc_resident_contact_pipeline_test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_executable(gpu_resident_contact_pipeline_test resident_contact_pipeline_test.cpp)
target_link_libraries(gpu_resident_contact_pipeline_test PRIVATE IPC::core ipc_resident_contact_pipeline_test_kernel)
add_test(NAME gpu_resident_contact_pipeline COMMAND gpu_resident_contact_pipeline_test)
add_executable(gpu_resident_contact_queue_test resident_contact_queue_test.cu)
target_link_libraries(gpu_resident_contact_queue_test PRIVATE ipc_resident_solver CUDA::cudart)
target_compile_options(gpu_resident_contact_queue_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_test(NAME gpu_resident_contact_queue COMMAND gpu_resident_contact_queue_test)
add_executable(gpu_resident_contact_warp_reduce_test resident_contact_warp_reduce_test.cu)
target_link_libraries(gpu_resident_contact_warp_reduce_test PRIVATE CUDA::cudart)
target_compile_options(gpu_resident_contact_warp_reduce_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_test(NAME gpu_resident_contact_warp_reduce COMMAND gpu_resident_contact_warp_reduce_test)
add_executable(gpu_resident_contact_worklist_test resident_contact_worklist_test.cu)
target_link_libraries(gpu_resident_contact_worklist_test PRIVATE ipc_resident_solver CUDA::cudart)
target_compile_options(gpu_resident_contact_worklist_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_test(NAME gpu_resident_contact_worklist COMMAND gpu_resident_contact_worklist_test)
add_executable(gpu_resident_contact_compact_test resident_contact_compact_test.cpp)
add_test(NAME gpu_resident_contact_compact COMMAND gpu_resident_contact_compact_test)
add_executable(gpu_resident_box_certificate_test resident_box_certificate_test.cpp resident_box_certificate_test.cu)
target_compile_definitions(gpu_resident_box_certificate_test PRIVATE IPC_RESIDENT_BOX_CERTIFICATE_CUDA_TEST=1)
target_include_directories(gpu_resident_box_certificate_test PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(gpu_resident_box_certificate_test PRIVATE CUDA::cudart)
target_compile_options(gpu_resident_box_certificate_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
add_test(NAME gpu_resident_box_certificate COMMAND gpu_resident_box_certificate_test)
add_executable(gpu_resident_box_pruning_test resident_box_pruning_test.cu)
target_link_libraries(gpu_resident_box_pruning_test PRIVATE ipc_resident_solver CUDA::cudart)
target_compile_options(gpu_resident_box_pruning_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_test(NAME gpu_resident_box_pruning COMMAND gpu_resident_box_pruning_test)
add_executable(gpu_resident_contact_cooperative_test resident_contact_cooperative_test.cu)
target_link_libraries(gpu_resident_contact_cooperative_test PRIVATE ipc_resident_solver CUDA::cudart)
target_compile_options(gpu_resident_contact_cooperative_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_test(NAME gpu_resident_contact_cooperative COMMAND gpu_resident_contact_cooperative_test)
add_executable(gpu_resident_contact_schedule_test resident_contact_schedule_test.cpp)
add_test(NAME gpu_resident_contact_schedule COMMAND gpu_resident_contact_schedule_test)
add_executable(gpu_resident_solver_test resident_solver_test.cpp)
target_link_libraries(gpu_resident_solver_test PRIVATE IPC::ipc ipc_resident_bridge)
add_test(NAME gpu_resident_solver COMMAND gpu_resident_solver_test)
add_executable(gpu_resident_adaptive_boxes_test resident_adaptive_boxes_test.cpp)
target_link_libraries(gpu_resident_adaptive_boxes_test PRIVATE ipc_resident_solver Eigen3::Eigen)
target_compile_options(gpu_resident_adaptive_boxes_test PRIVATE -ffp-contract=off)
add_test(NAME gpu_resident_adaptive_boxes COMMAND gpu_resident_adaptive_boxes_test)
add_executable(gpu_resident_global_guess_test resident_global_guess_test.cpp)
target_compile_definitions(gpu_resident_global_guess_test PRIVATE IPC_GLOBAL_GUESS_CUDA_TEST=1)
target_compile_options(gpu_resident_global_guess_test PRIVATE -ffp-contract=off)
target_link_libraries(gpu_resident_global_guess_test PRIVATE ipc_resident_solver tight_inclusion::tight_inclusion Threads::Threads)
add_test(NAME gpu_resident_global_guess COMMAND gpu_resident_global_guess_test)
add_executable(gpu_resident_linear_pipeline_test resident_linear_pipeline_test.cpp)
target_link_libraries(gpu_resident_linear_pipeline_test PRIVATE ipc_resident_solver)
add_test(NAME gpu_resident_linear_pipeline COMMAND gpu_resident_linear_pipeline_test)
add_executable(gpu_resident_rigid_pipeline_test resident_rigid_pipeline_test.cpp)
target_link_libraries(gpu_resident_rigid_pipeline_test PRIVATE ipc_resident_solver IPC::ipc)
add_test(NAME gpu_resident_rigid_pipeline COMMAND gpu_resident_rigid_pipeline_test)

# Independent resident driver: it never calls an advance_one_frame CPU path.
# Keep setup, prescribed boundaries, output, and checkpoints from the project.
file(READ "${IPC_CPU_SOURCE}/simulation.cpp" resident_driver)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/resident_driver_step.inc" resident_step)
set(resident_dispatch [=[        if (uses_general_solver) {
            result = advance_one_frame_general(
                state, ref_mesh, adj, pins, params, broad_phase,
                frame_index, pin_updater, substep_cb, outdir);
        } else if (has_rigid) {
            result = advance_one_frame_rb(
                state, ref_mesh, params, broad_phase, frame_index, substep_cb);
        } else {
            result = advance_one_frame(
                state, ref_mesh, adj, pins, params, broad_phase,
                frame_index, pin_updater, substep_cb, outdir);
        }]=])
replace_once(resident_driver "${resident_dispatch}" "${resident_step}")
replace_once(resident_driver "    using Clock = std::chrono::steady_clock;"
    "    ipc_gpu::resident::ResidentSolver resident_solver(ipc_gpu::resident::describe_scene(ref_mesh,state,pins,params));\n    using Clock = std::chrono::steady_clock;")
replace_once(resident_driver "    const std::string& outdir = args.outdir;"
    "    (void)ipc_gpu::resident::describe_step(params); // reject unported modes before touching output\n    std::cout << \"GPU-resident numerical solver: cloth, solid, rigid; exact fallback on device\\n\";\n    const std::string& outdir = args.outdir;")
replace_once(resident_driver "        if (fs::exists(outdir)) fs::remove_all(outdir);"
    "        if (fs::exists(outdir)) throw std::runtime_error(\"Resident GPU run requires a new output directory (or --restart_frame)\");")
string(FIND "${resident_driver}" "    if ((params.use_basic_experimental" resident_print_begin)
string(FIND "${resident_driver}" "    if (num_rigid_bodies > 0)" resident_print_end)
if(resident_print_begin LESS 0 OR resident_print_end LESS resident_print_begin)
    message(FATAL_ERROR "Resident driver startup anchor missing")
endif()
string(SUBSTRING "${resident_driver}" 0 ${resident_print_begin} resident_before)
string(SUBSTRING "${resident_driver}" ${resident_print_end} -1 resident_after)
set(resident_driver "${resident_before}${resident_after}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/simulation_resident.cpp" "#include \"resident_bridge.h\"\n${resident_driver}")
add_executable(3D_sim_gpu_resident "${CMAKE_CURRENT_BINARY_DIR}/simulation_resident.cpp")
target_link_libraries(3D_sim_gpu_resident PRIVATE ipc_resident_bridge IPC::ipc Threads::Threads)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/resident_driver_step.inc")
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
