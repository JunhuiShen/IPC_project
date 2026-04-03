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

    // Two sheets, side-by-side in x, embedded in the xz plane by build_square_mesh().
    const int base_left = build_square_mesh(
            ref_mesh, state, X,
            args.nx, args.ny, args.width, args.height,
            Vec3(args.left_x, args.sheet_y, args.left_z)
    );

    const int base_right = build_square_mesh(
            ref_mesh, state, X,
            args.nx, args.ny, args.width, args.height,
            Vec3(args.right_x, args.sheet_y, args.right_z)
    );

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    // Tiny asymmetry on the right sheet so the evolution is not perfectly symmetric.
    state.deformed_positions[base_right + 0] += Vec3(-0.02, 0.00, -0.01);
    state.deformed_positions[base_right + args.nx] += Vec3(-0.02, 0.00, -0.01);

    // Pin inner-side top and bottom corners.
    const int left_bottom_right = base_left + args.nx;
    const int left_top_right    = base_left + args.ny * (args.nx + 1) + args.nx;
    const int right_bottom_left = base_right + 0;
    const int right_top_left    = base_right + args.ny * (args.nx + 1);

    append_pin(pins, left_top_right,    state.deformed_positions);
    append_pin(pins, left_bottom_right, state.deformed_positions);
    append_pin(pins, right_top_left,    state.deformed_positions);
    append_pin(pins, right_bottom_left, state.deformed_positions);

    ref_mesh.build_lumped_mass(params.density, params.thickness);
    VertexTriangleMap adj = build_incident_triangle_map(ref_mesh.tris);

    const auto color_groups = greedy_color(
            build_vertex_adjacency_map(ref_mesh.tris),
            static_cast<int>(state.deformed_positions.size()));

    // Current project-wide pair builder.
    const BarrierPairs barrier_pairs = build_barrier_pairs(ref_mesh);

    std::cout << "num_frames = " << num_frames << "\n";
    std::cout << "d_hat = " << params.d_hat
              << (params.d_hat > 0.0 ? "  (barrier ON)" : "  (barrier OFF)")
              << "\n";
    std::cout << "Vertices:  " << state.deformed_positions.size() << "\n";
    std::cout << "Triangles: " << ref_mesh.tris.size() / 3 << "\n";
    std::cout << "NT pairs:  " << barrier_pairs.nt.size() << "\n";
    std::cout << "SS pairs:  " << barrier_pairs.ss.size() << "\n";

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

        for (int sub = 0; sub < params.substeps; ++sub) {
            std::vector<Vec3> xhat;
            build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());

            std::vector<Vec3> xnew = state.deformed_positions;
            result = global_gauss_seidel_solver(
                    ref_mesh, adj, pins, params, xnew, xhat,
                    barrier_pairs.nt, barrier_pairs.ss, color_groups);

            update_velocity(state.velocities, xnew, state.deformed_positions, params.dt());
            state.deformed_positions = xnew;
        }

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