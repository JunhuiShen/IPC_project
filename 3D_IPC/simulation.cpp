#include "example.h"
#include "make_shape.h"
#include "physics.h"
#include "simulation.h"
#include "solver.h"
#include "visualization.h"
#include "ipc_args.h"
#include "broad_phase.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fs = std::__fs::filesystem;

int main(int argc, char** argv) {
    IPCArgs3D args;
    if (!args.parse(argc, argv)) return 1;

    SimParams params = args.to_sim_params();
    const int num_frames = args.num_frames;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;

    // Pick a scene. Swap these lines to run a different example.
    // build_two_sheets_example(args, ref_mesh, state, X, pins);
    // build_cloth_stack_example_low_res(ref_mesh, state, X, pins);
    build_cloth_stack_example_high_res(ref_mesh, state, X, pins);

    ref_mesh.build_lumped_mass(params.density, params.thickness);
    VertexTriangleMap adj = build_incident_triangle_map(ref_mesh.tris);

    const auto color_groups = greedy_color(
            build_vertex_adjacency_map(ref_mesh.tris),
            static_cast<int>(state.deformed_positions.size()));

    BroadPhase broad_phase;

    std::cout << "num_frames = " << num_frames << "\n";
    std::cout << "d_hat = " << params.d_hat
              << (params.d_hat > 0.0 ? "  (barrier ON)" : "  (barrier OFF)")
              << "\n";
    std::cout << "Vertices:  " << state.deformed_positions.size() << "\n";
    std::cout << "Triangles: " << ref_mesh.tris.size() / 3 << "\n";

    const std::string& outdir = args.outdir;
    const ExportFormat fmt = args.to_export_format();

    if (params.restart_frame < 0) {
        if (fs::exists(outdir)) fs::remove_all(outdir);
        fs::create_directories(outdir);
        export_frame(outdir, 0, state.deformed_positions, ref_mesh.tris, fmt);
        serialize_state(outdir, 0, state);
    } else {
        if (!fs::exists(outdir)) {
            std::cerr << "Error: restart requested but output directory does not exist: " << outdir << "\n";
            return 1;
        }
        if (!deserialize_state(outdir, params.restart_frame, state)) {
            std::cerr << "Error: failed to load restart frame " << params.restart_frame << "\n";
            return 1;
        }
    }

    using Clock = std::chrono::steady_clock;
    auto sim_start = Clock::now();
    double total_solver_ms = 0.0;

    int start_frame = (params.restart_frame >= 0) ? (params.restart_frame + 1) : 1;

    for (int frame_index = start_frame; frame_index <= num_frames; ++frame_index) {
        auto solver_start = Clock::now();
        SolverResult result;

        result = advance_one_frame(state, ref_mesh, adj, pins, params, color_groups, broad_phase);

        auto solver_end = Clock::now();
        double solver_ms = std::chrono::duration<double, std::milli>(solver_end - solver_start).count();
        total_solver_ms += solver_ms;

        std::cout << "Frame " << std::setw(4) << frame_index
                  << " | initial_residual = " << std::scientific << result.initial_residual
                  << " | final_residual = "   << std::scientific << result.final_residual
                  << " | global_iters = "     << std::setw(3)    << result.iterations
                  << " | solver_time = "      << std::fixed << std::setprecision(3)
                  << solver_ms << " ms\n";

        export_frame(outdir, frame_index, state.deformed_positions, ref_mesh.tris, fmt);
        serialize_state(outdir, frame_index, state);
    }

    auto sim_end = Clock::now();
    double total_sim_ms = std::chrono::duration<double, std::milli>(sim_end - sim_start).count();

    std::cout << "\nSimulation finished.\n";
    std::cout << "Total simulation time: " << std::fixed << std::setprecision(3) << total_sim_ms << " ms\n";
    std::cout << "Total solver time:     " << total_solver_ms << " ms\n";
    std::cout << "Average solver time:   " << (total_solver_ms / std::max(1, num_frames - start_frame + 1)) << " ms/frame\n";

    return 0;
}
