#include "make_shape.h"
#include "physics.h"
#include "solver.h"
#include "visualization.h"
#include "ipc_args.h"

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

    clear_model(ref_mesh, state, X, pins);

    int base = build_square_mesh(ref_mesh, state, X, args.nx, args.ny, args.width, args.height, Vec3(0.2, -0.1, 0.3));

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    // pin top-left and top-right corners
    append_pin(pins, base + args.ny * (args.nx + 1), state.deformed_positions);
    append_pin(pins, base + args.ny * (args.nx + 1) + args.nx, state.deformed_positions);

    ref_mesh.build_lumped_mass(params.density, params.thickness);
    VertexAdjacencyMap adj = build_incident_triangle_map(ref_mesh.tris);

    std::cout << "Vertices: " << state.deformed_positions.size() << "\n";
    std::cout << "Triangles: " << ref_mesh.tris.size() << "\n";

    const std::string& outdir = args.outdir;
    if (fs::exists(outdir)) fs::remove_all(outdir);
    fs::create_directories(outdir);

    export_frame(outdir, 0, state.deformed_positions, ref_mesh.tris);


    using Clock = std::chrono::steady_clock;
    auto sim_start = Clock::now();

    double total_solver_ms = 0.0;

    for (int frame_index = 1; frame_index <= num_frames; ++frame_index) {
        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, params.dt);

        // Initial guess
        std::vector<Vec3> xnew = state.deformed_positions;

        auto solver_start = Clock::now();
        SolverResult result = global_gauss_seidel_solver(
                ref_mesh, adj, pins, params, xnew, xhat
        );
        auto solver_end = Clock::now();

        double solver_ms =
                std::chrono::duration<double, std::milli>(solver_end - solver_start).count();
        total_solver_ms += solver_ms;

        std::cout << "Frame " << std::setw(4) << frame_index
                  << " | initial_residual = " << std::scientific << result.initial_residual
                  << " | final_residual = " << std::scientific << result.final_residual
                  << " | global_iters = " << std::setw(3) << result.iterations
                  << " | solver_time = " << std::fixed << std::setprecision(3)
                  << solver_ms << " ms\n";

        update_velocity(state.velocities, xnew, state.deformed_positions, params.dt);
        state.deformed_positions = xnew;

        export_frame(outdir, frame_index, state.deformed_positions, ref_mesh.tris);
    }

    auto sim_end = Clock::now();
    double total_sim_ms = std::chrono::duration<double, std::milli>(sim_end - sim_start).count();

    std::cout << "\nSimulation finished.\n";
    std::cout << "Total simulation time: " << std::fixed << std::setprecision(3)
              << total_sim_ms << " ms\n";
    std::cout << "Total solver time:     " << total_solver_ms << " ms\n";
    std::cout << "Average solver time:   " << (total_solver_ms / num_frames) << " ms/frame\n";

    return 0;
}
