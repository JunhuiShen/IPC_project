# Checked source overlays share these helpers; all outputs stay in the build tree.
include_guard(GLOBAL)

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

# Keep generated source timestamps stable on a no-change reconfigure.
function(ipc_gpu_write_generated_if_different output content)
    if(EXISTS "${output}")
        file(READ "${output}" existing_content)
        if("${existing_content}" STREQUAL "${content}")
            return()
        endif()
    endif()
    file(WRITE "${output}" "${content}")
endfunction()

# Canonical generated sources shared by CMake targets and check_overlay.cmake.
function(ipc_gpu_generate_hybrid_sources)
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
ipc_gpu_write_generated_if_different("${CMAKE_CURRENT_BINARY_DIR}/solver_gpu.cpp"
    "#include \"color_bridge.h\"\n#include \"general_bridge.h\"\n#include \"rigid_bridge.h\"\n${prefix}${body}${suffix}")
file(READ "${IPC_CPU_SOURCE}/simulation.cpp" driver)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/driver_guard.inc" guard)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/driver_stats.inc" stats)
replace_once(driver "    const bool uses_general_solver = is_mixed || has_solid;" "${guard}\n    const bool uses_general_solver = is_mixed || has_solid;")
set(timing_anchor "    std::cout << \"Total solver time:     \"")
replace_once(driver "${timing_anchor}" "${stats}\n${timing_anchor}")
ipc_gpu_write_generated_if_different("${CMAKE_CURRENT_BINARY_DIR}/simulation_gpu.cpp" "#include \"kernels.h\"\n${driver}")
endfunction()

function(ipc_gpu_generate_resident_driver)
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
# The resident solver builds its incidence and conflict graph on the device.
# These CPU workspaces served only the frame dispatch replaced above.
replace_once(resident_driver
    "    VertexTriangleMap adj = build_incident_triangle_map(ref_mesh.tris);\n" "")
replace_once(resident_driver "    BroadPhase broad_phase;\n" "")
replace_once(resident_driver "    using Clock = std::chrono::steady_clock;"
    "    ipc_gpu::resident::ResidentSolver resident_solver(ipc_gpu::resident::describe_scene(ref_mesh,state,pins,params));\n    using Clock = std::chrono::steady_clock;")
replace_once(resident_driver "    const std::string& outdir = args.outdir;"
    "    (void)ipc_gpu::resident::describe_step(params); // reject unported modes before touching output\n    std::cout << \"GPU-resident numerical solver\\n\";\n    const std::string& outdir = args.outdir;")
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
ipc_gpu_write_generated_if_different("${CMAKE_CURRENT_BINARY_DIR}/simulation_resident.cpp" "#include \"resident_bridge.h\"\n${resident_driver}")
endfunction()


function(ipc_gpu_generate_dat_driver)
file(READ "${CMAKE_CURRENT_BINARY_DIR}/simulation_resident.cpp" dat_driver)
replace_once(dat_driver "#include \"resident_bridge.h\"" "#include \"resident_bridge.h\"\n#include \"resident_dat.h\"")
replace_once(dat_driver "ipc_gpu::resident::ResidentSolver resident_solver(" "ipc_gpu::resident::ResidentDATSolver resident_solver(")
replace_once(dat_driver "    if (!args.parse(argc, argv)) return 1;" "    if (!args.parse(argc, argv)) return 1;\n    args.use_dat_solver = true;")
ipc_gpu_write_generated_if_different("${CMAKE_CURRENT_BINARY_DIR}/simulation_dat.cpp" "${dat_driver}")
endfunction()
