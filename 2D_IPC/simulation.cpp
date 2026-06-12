#include "broad_phase/bvh.h"
#include "chain.h"
#include "example.h"
#include "ipc_args.h"
#include "restart.h"
#include "simulation.h"
#include "visualization.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

static std::string checkpoint_path(const std::string& outdir, int frame) {
    std::ostringstream ss;
    ss << outdir << "/state_" << std::setw(4) << std::setfill('0') << frame << ".bin";
    return ss.str();
}

int main(int argc, char** argv) {
    IPCArgs args;
    if (!args.parse(argc, argv)) return 1;
    args.validate();

    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    const bool restarting = args.restart_frame >= 0;
    if (!restarting && fs::exists(args.outdir)) {
        fs::remove_all(args.outdir);
    }
    fs::create_directories(args.outdir);

    ExampleScene scene = build_example(args.get_example_type(), args.number_of_nodes, args.mass_density);
    std::vector<Chain> chains = std::move(scene.chains);
    std::vector<int> offsets = compute_node_offsets(chains);
    RefMesh ref_mesh = build_ref_mesh(chains, offsets);

    if (restarting) {
        if (!read_checkpoint(checkpoint_path(args.outdir, args.restart_frame), chains)) {
            return 1;
        }
    }

    SimParams2D params;
    params.frame_dt = args.dt;
    params.substeps = args.substeps;
    params.k_spring = args.k_spring;
    params.k_barrier = args.k_barrier;
    params.gravity = {args.gx, args.gy};
    params.dhat = args.dhat;
    params.tol_abs = args.tol_abs;
    params.max_global_iters = args.max_global_iters;
    params.eta = args.eta;
    params.use_ccd_step_policy = args.use_ccd_step_policy();
    params.write_substeps = args.write_substeps;
    params.initial_guess_type = args.get_initial_guess_type();

    const OutputFormat output_format = args.get_output_format();
    BVHBroadPhase broad_phase;

    Vec x_combined(2 * ref_mesh.num_positions, 0.0);
    build_x_combined_from_chains(x_combined, chains, offsets);

    std::cout << "Vertices: " << ref_mesh.num_positions
              << " | Segments: " << ref_mesh.edges.size() << "\n";

    if (!restarting) {
        export_frame(args.outdir, 0, x_combined, ref_mesh.edges, output_format);
        write_checkpoint(checkpoint_path(args.outdir, 0), chains);
    }

    double max_global_residual = 0.0;
    double total_solver_time = 0.0;
    int sum_global_iters_used = 0;
    int frames_advanced = 0;

    const int start_frame = restarting ? args.restart_frame + 1 : 1;
    for (int frame = start_frame; frame <= args.total_frames; ++frame) {
        auto substep_export = [&](int global_substep, const Vec& x_substep) {
            if (params.write_substeps) {
                export_substep_frame(args.outdir, global_substep + 1, x_substep, ref_mesh.edges, output_format);
            }
        };

        auto solver_start = clock::now();
        AdvanceResult2D result = advance_one_frame(
                chains, ref_mesh, offsets, params, broad_phase, frame, substep_export);
        auto solver_end = clock::now();
        std::chrono::duration<double> solver_elapsed = solver_end - solver_start;

        build_x_combined_from_chains(x_combined, chains, offsets);
        export_frame(args.outdir, frame, x_combined, ref_mesh.edges, output_format);
        write_checkpoint(checkpoint_path(args.outdir, frame), chains);

        max_global_residual = std::max(max_global_residual, result.max_final_residual);
        total_solver_time += solver_elapsed.count();
        sum_global_iters_used += result.total_iterations;
        frames_advanced += 1;

        std::cout << "Frame " << std::setw(4) << frame
                  << " | initial_residual=" << std::scientific << result.first_initial_residual
                  << " | final_residual="   << std::scientific << result.max_final_residual
                  << " | global_iters="     << std::setw(3) << result.total_iterations
                  << " | solver_time="      << solver_elapsed.count() << " s"
                  << '\n';
    }

    auto t_end = clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    const double avg_global_iters_used =
            (frames_advanced > 0) ? double(sum_global_iters_used) / frames_advanced : 0.0;
    const double avg_solver_time =
            (frames_advanced > 0) ? total_solver_time / frames_advanced : 0.0;

    std::cout << "\n===== Simulation Summary =====\n";
    std::cout << "max_global_residual = " << std::scientific << max_global_residual << "\n";
    std::cout << "avg_global_iters = " << std::fixed << avg_global_iters_used << "\n";
    std::cout << "total_sim_time = " << elapsed.count() << " seconds\n";
    std::cout << "total_solver_time = " << total_solver_time << " seconds\n";
    std::cout << "avg_solver_time = " << avg_solver_time << " seconds/frame\n";

    return 0;
}
