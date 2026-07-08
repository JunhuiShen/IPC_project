#include "broad_phase.h"
#include "example.h"
#include "ipc_args.h"
#include "physics.h"
#include "simulation.h"
#include "visualization.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <utility>

namespace fs = std::filesystem;

static void export_scene_frame(const std::string& outdir, int frame, const Vec& dynamic_x, const std::vector<std::pair<int, int>>& dynamic_edges, OutputFormat format) {
    export_frame(outdir, frame, dynamic_x, dynamic_edges, format);
}

static void export_scene_substep(const std::string& outdir, int substep, const Vec& dynamic_x, const std::vector<std::pair<int, int>>& dynamic_edges, OutputFormat format) {
    export_substep_frame(outdir, substep, dynamic_x, dynamic_edges, format);
}

static void export_static_colliders(const std::string& outdir, const Vec& static_x, const std::vector<std::pair<int, int>>& static_edges, OutputFormat format) {
    if (static_x.empty()) return;

    const fs::path filename = fs::path(outdir) / (format == OutputFormat::GEO ? "static_colliders.geo" : "static_colliders.obj");
    if (format == OutputFormat::GEO) {
        export_geo(filename.string(), static_x, static_edges);
    } else {
        export_obj(filename.string(), static_x, static_edges);
    }
}

int main(int argc, char** argv) {
    IPCArgs args;
    if (!args.parse(argc, argv)) {
        return 1;
    }

    ExampleType example_type = ExampleType::Example1;
    OutputFormat output_format = OutputFormat::GEO;
    InitialGuessType initial_guess_type = InitialGuessType::CCD;
    bool use_ccd_step_policy = true;
    try {
        args.validate();
        example_type = args.get_example_type();
        output_format = args.get_output_format();
        initial_guess_type = args.get_initial_guess_type();
        use_ccd_step_policy = args.use_ccd_step_policy();
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    using clock = std::chrono::high_resolution_clock;
    const auto start_time = clock::now();

    ExampleScene scene =
        build_example(
            example_type, args.number_of_nodes, args.density, args.thickness);
    DeformedState state = std::move(scene.state);
    RefMesh ref_mesh = std::move(scene.ref_mesh);
    std::vector<Pin> pins = std::move(scene.pins);
    Vec static_positions = std::move(scene.static_positions);
    std::vector<std::pair<int, int>> static_edges = std::move(scene.static_edges);

    const double min_edge_length =
        *std::min_element(ref_mesh.rest_lengths.begin(), ref_mesh.rest_lengths.end());
    const double d_hat_limit = 0.5 * min_edge_length;
    if (!(args.d_hat < d_hat_limit)) {
        std::cerr << "Error: d_hat (" << args.d_hat
                  << ") must be < 0.5 * minimum edge length (" << d_hat_limit
                  << ", minimum edge length = " << min_edge_length << ").\n";
        return 1;
    }

    const bool restarting = args.restart_frame >= 0;
    if (!restarting && fs::exists(args.outdir)) {
        fs::remove_all(args.outdir);
    }
    fs::create_directories(args.outdir);

    if (restarting && !deserialize_state(args.outdir, args.restart_frame, state)) {
        return 1;
    }

    SimParams2D params;
    params.frame_dt = args.dt;
    params.substeps = args.substeps;
    params.k_spring = args.k_spring;
    params.kpin = args.kpin;
    params.k_barrier = args.k_barrier;
    params.k_sdf = args.k_sdf;
    params.eps_sdf = args.eps_sdf;
    params.sdf_grounds = std::move(scene.sdf_grounds);
    params.sdf_circles = std::move(scene.sdf_circles);
    params.sdf_planes  = std::move(scene.sdf_planes);
    params.gravity = {args.gx, args.gy};
    params.d_hat = args.d_hat;
    params.tol_abs = args.tol_abs;
    params.tol_rel = args.tol_rel;
    params.max_substep_iters = args.max_substep_iters;
    params.eta = args.eta;
    params.use_parallel = args.use_parallel;
    params.fixed_iters = args.fixed_iters;
    params.node_box_min = args.node_box_min;
    params.node_box_max = args.node_box_max;
    params.theta_box_min = args.theta_box_min;
    params.theta_box_max = args.theta_box_max;
    params.node_box_update_count = args.node_box_update_count;
    params.use_ccd_step_policy = use_ccd_step_policy;
    params.initial_guess_type = initial_guess_type;

    BroadPhase broad_phase;

    std::cout << "Vertices: " << ref_mesh.num_positions
              << " | Segments: " << ref_mesh.edges.size()
              << " | Min edge length: " << min_edge_length
              << " | d_hat limit: " << d_hat_limit << "\n";

    export_static_colliders(args.outdir, static_positions, static_edges, output_format);

    if (!restarting) {
        export_scene_frame(args.outdir, 0, state.deformed_positions, ref_mesh.edges, output_format);
        serialize_state(args.outdir, 0, state);
    }

    double max_global_residual = 0.0;
    double max_rigid_residual = 0.0;
    bool saw_rigid_residual = false;
    double total_solver_time = 0.0;
    int sum_global_iters_used = 0;
    int frames_advanced = 0;

    const int start_frame = restarting ? args.restart_frame + 1 : 1;
    for (int frame = start_frame; frame <= args.num_frames; ++frame) {
        auto substep_export = [&](int global_substep, const Vec& x_substep) {
            if (args.write_substeps) {
                export_scene_substep(args.outdir, global_substep + 1, x_substep, ref_mesh.edges, output_format);
            }
        };

        const auto solver_start = clock::now();
        AdvanceResult2D result;
        if (ref_mesh.rb_nodes.empty()) {
            result = advance_one_frame(state, ref_mesh, pins, params, broad_phase, frame, substep_export);
        } else {
            result = advance_one_frame_rb(state, ref_mesh, pins, params, broad_phase, frame, substep_export);
        }
        const auto solver_end = clock::now();
        const std::chrono::duration<double> solver_elapsed = solver_end - solver_start;

        export_scene_frame(args.outdir, frame, state.deformed_positions, ref_mesh.edges, output_format);
        serialize_state(args.outdir, frame, state);

        max_global_residual = std::max(max_global_residual, result.max_final_residual);
        if (result.has_rigid_residual) {
            saw_rigid_residual = true;
            max_rigid_residual = std::max(max_rigid_residual, result.max_final_rigid_residual);
        }
        total_solver_time += solver_elapsed.count();
        sum_global_iters_used += result.total_iterations;
        frames_advanced += 1;

        std::cout << "Frame " << std::setw(4) << frame
                  << " | initial_residual=" << std::scientific
                  << result.first_initial_residual
                  << " | final_residual=" << std::scientific
                  << result.max_final_residual;
        if (result.has_rigid_residual) {
            std::cout << " | final_rb_residual=" << std::scientific
                      << result.max_final_rigid_residual;
        }
        std::cout << " | global_iters=" << std::setw(3) << result.total_iterations
                  << " | solver_time=" << solver_elapsed.count() << " s"
                  << '\n';
    }

    const auto end_time = clock::now();
    const std::chrono::duration<double> elapsed = end_time - start_time;

    const double avg_global_iters_used =
            frames_advanced > 0
                    ? static_cast<double>(sum_global_iters_used) / frames_advanced
                    : 0.0;
    const double avg_solver_time =
            frames_advanced > 0 ? total_solver_time / frames_advanced : 0.0;

    std::cout << "\n===== Simulation Summary =====\n";
    std::cout << "max_global_residual = " << std::scientific
              << max_global_residual << "\n";
    if (saw_rigid_residual) {
        std::cout << "max_rb_residual = " << std::scientific
                  << max_rigid_residual << "\n";
    }
    std::cout << "avg_global_iters = " << std::fixed << avg_global_iters_used << "\n";
    std::cout << "total_sim_time = " << elapsed.count() << " seconds\n";
    std::cout << "total_solver_time = " << total_solver_time << " seconds\n";
    std::cout << "avg_solver_time = " << avg_solver_time << " seconds/frame\n";

    return 0;
}
