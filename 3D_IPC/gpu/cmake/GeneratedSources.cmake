# Checked source overlays share these helpers; all outputs stay in the build tree.
# The canonical output is simulation_gpu.cpp (the current GPU driver).
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

# Generated copies live in the build tree, where the GPU headers shadow same-named
# CPU headers: gpu/solver.h shadows the CPU solver.h and gpu/dat.h shadows DAT.h
# on case-insensitive filesystems. Every generated copy rewrites those quoted
# includes to the read-only CPU tree by absolute path; the CPU originals are
# never edited.
#
# Canonical generated source shared by CMake targets and check_overlay.cmake.

# simulation_gpu.cpp is the current GPU driver: the shared example driver with
# its frame dispatch replaced by the device solver, so it never calls an
# advance_one_frame CPU path. Setup, prescribed boundaries, output and
# checkpoints stay with the project.
function(ipc_gpu_generate_driver)
file(READ "${IPC_CPU_SOURCE}/simulation.cpp" gpu_driver)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/driver_step.inc" gpu_step)
set(gpu_dispatch [=[        if (uses_general_solver) {
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
replace_once(gpu_driver "${gpu_dispatch}" "${gpu_step}")
# The device solver builds its incidence and conflict graph on the GPU.
# These CPU workspaces served only the frame dispatch replaced above.
replace_once(gpu_driver
    "    VertexTriangleMap adj = build_incident_triangle_map(ref_mesh.tris);\n" "")
replace_once(gpu_driver "    BroadPhase broad_phase;\n" "")
replace_once(gpu_driver "    using Clock = std::chrono::steady_clock;"
    "    ipc_gpu::solver::GpuSolver gpu_solver(ipc_gpu::solver::describe_scene(ref_mesh,state,pins,params));\n    using Clock = std::chrono::steady_clock;")
replace_once(gpu_driver "    const std::string& outdir = args.outdir;"
    "    (void)ipc_gpu::solver::describe_step(params); // reject unported modes before touching output\n    if (params.use_jacobi_dat && !ref_mesh.rb_nodes.empty()) {\n        throw std::invalid_argument(\"GPU Jacobi DAT is not supported on rigid-body scenes\");\n    }\n    if (!params.exact_computation_fallback && !ref_mesh.rb_nodes.empty()) {\n        throw std::invalid_argument(\"GPU original/exact fallback mode is not supported on rigid-body scenes\");\n    }\n    std::cout << \"GPU device solver backend\\n\";\n    const std::string& outdir = args.outdir;")
replace_once(gpu_driver "        if (fs::exists(outdir)) fs::remove_all(outdir);"
    "        if (fs::exists(outdir)) throw std::runtime_error(\"GPU run requires a new output directory (or --restart_frame)\");")
string(FIND "${gpu_driver}" "    if (!params.use_jacobi_dat\n        && (params.use_basic_experimental" gpu_print_begin)
string(FIND "${gpu_driver}" "    if (num_rigid_bodies > 0)" gpu_print_end)
if(gpu_print_begin LESS 0 OR gpu_print_end LESS gpu_print_begin)
    message(FATAL_ERROR "GPU driver startup anchor missing")
endif()
string(SUBSTRING "${gpu_driver}" 0 ${gpu_print_begin} gpu_before)
string(SUBSTRING "${gpu_driver}" ${gpu_print_end} -1 gpu_after)
set(gpu_driver "${gpu_before}${gpu_after}")
replace_once(gpu_driver "#include \"solver.h\"" "#include \"${IPC_CPU_SOURCE}/solver.h\"")
ipc_gpu_write_generated_if_different("${CMAKE_CURRENT_BINARY_DIR}/simulation_gpu.cpp" "#include \"bridge.h\"\n${gpu_driver}")
endfunction()
