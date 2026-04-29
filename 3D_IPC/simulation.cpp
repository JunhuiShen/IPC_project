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
#include <limits>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    IPCArgs3D args;
    if (!args.parse(argc, argv)) return 1;

    SimParams params = args.to_sim_params();
    const int num_frames = args.num_frames;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    TwistSpec twist_spec;

    std::vector<Vec3> static_x;
    std::vector<int>  static_tris;

    if      (args.example == 1) build_two_sheets_example(args, ref_mesh, state, X, pins);
    else if (args.example == 2) build_cloth_stack_example_low_res(ref_mesh, state, X, pins);
    else if (args.example == 3) build_cloth_stack_example_high_res(ref_mesh, state, X, pins);
    else if (args.example == 4) build_cloth_cylinder_drop_example(args, ref_mesh, state, X, pins, params, static_x, static_tris);
    else if (args.example == 5) build_twisting_cloth_example(args, ref_mesh, state, X, pins, twist_spec);
    else if (args.example == 6) build_cloth_sphere_drop_example(args, ref_mesh, state, X, pins, params);
    else {
        std::cerr << "Unknown --example " << args.example << ". Valid values: 1, 2, 3, 4, 5, 6.\n";
        return 1;
    }

    ref_mesh.build_lumped_mass(params.density, params.thickness);
    VertexTriangleMap adj = build_incident_triangle_map(ref_mesh.tris);

    // Validate d_hat against the minimum edge length in the mesh.
    if (params.d_hat > 0.0) {
        double min_edge_len = std::numeric_limits<double>::max();
        const int nt = static_cast<int>(ref_mesh.tris.size()) / 3;
        for (int t = 0; t < nt; ++t) {
            const int v0 = ref_mesh.tris[3 * t + 0];
            const int v1 = ref_mesh.tris[3 * t + 1];
            const int v2 = ref_mesh.tris[3 * t + 2];
            min_edge_len = std::min(min_edge_len, (state.deformed_positions[v1] - state.deformed_positions[v0]).norm());
            min_edge_len = std::min(min_edge_len, (state.deformed_positions[v2] - state.deformed_positions[v1]).norm());
            min_edge_len = std::min(min_edge_len, (state.deformed_positions[v0] - state.deformed_positions[v2]).norm());
        }
        const double d_hat_limit = 0.5 * min_edge_len;
        if (params.d_hat > d_hat_limit) {
            std::cerr << "Error: d_hat (" << params.d_hat
                      << ") > 0.5 * minimum edge length (" << d_hat_limit
                      << ", min_edge_len = " << min_edge_len
                      << "). Barrier activation distance must not exceed half "
                         "the shortest edge in the mesh.\n";
            return 1;
        }
        std::cout << "min_edge_length = " << min_edge_len
                  << " (d_hat limit = " << d_hat_limit << ")\n";
    }

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
    const int restart_frame = args.restart_frame;

    if (restart_frame < 0) {
        if (fs::exists(outdir)) fs::remove_all(outdir);
        fs::create_directories(outdir);
        export_frame(outdir, 0, state.deformed_positions, ref_mesh.tris, fmt);
        serialize_state(outdir, 0, state);
        if (!static_x.empty()) {
            const char* ext = (fmt == ExportFormat::GEO) ? ".geo"
                            : (fmt == ExportFormat::PLY) ? ".ply"
                            : (fmt == ExportFormat::USD) ? ".usda" : ".obj";
            const std::string path = outdir + "/static_colliders" + ext;
            if      (fmt == ExportFormat::GEO) export_geo(path, static_x, static_tris);
            else if (fmt == ExportFormat::PLY) export_ply(path, static_x, static_tris);
            else if (fmt == ExportFormat::USD) export_usd(path, static_x, static_tris);
            else                               export_obj(path, static_x, static_tris);
        }
    } else {
        if (!fs::exists(outdir)) {
            std::cerr << "Error: restart requested but output directory does not exist: " << outdir << "\n";
            return 1;
        }
        if (!deserialize_state(outdir, restart_frame, state)) {
            std::cerr << "Error: failed to load restart frame " << restart_frame << "\n";
            return 1;
        }
    }

    using Clock = std::chrono::steady_clock;
    auto sim_start = Clock::now();
    double total_solver_ms = 0.0;

    int start_frame = (restart_frame >= 0) ? (restart_frame + 1) : 1;

    for (int frame_index = start_frame; frame_index <= num_frames; ++frame_index) {
        auto solver_start = Clock::now();
        SolverResult result;

        SubstepCallback substep_cb = nullptr;
        if (params.write_substeps) {
            substep_cb = [&](int global_sub, const std::vector<Vec3>& pos) {
                export_frame(outdir, global_sub + 1, pos, ref_mesh.tris, fmt, nullptr);
            };
        }

        result = advance_one_frame(
            state, ref_mesh, adj, pins, params, color_groups, broad_phase,
            (args.example == 5) ? &twist_spec : nullptr, frame_index,
            (args.example == 5) ? &update_twist_pins : nullptr, substep_cb, outdir);

        if (!result.converged) {
            std::cerr << "Error: solver failed to converge at frame " << frame_index
                      << " (final_residual = " << result.final_residual
                      << ", max_substep_iters = " << params.max_global_iters << ")\n";
            return 1;
        }

        auto solver_end = Clock::now();
        double solver_ms = std::chrono::duration<double, std::milli>(solver_end - solver_start).count();
        total_solver_ms += solver_ms;

        std::cout << "Frame " << std::setw(4) << frame_index
                  << " | initial_residual = " << std::scientific << result.initial_residual
                  << " | final_residual = "   << std::scientific << result.final_residual
                  << " | global_iters = "     << std::setw(3)    << result.iterations;
        if (params.use_parallel) {
            std::cout << " | colors = "  << std::setw(3) << result.last_num_colors;
        }
        if (params.ccd_check)
            std::cout << " | ccd_viol = " << std::setw(3) << result.ccd_violations;
        std::cout << " | solver_time = " << std::fixed << std::setprecision(3)
                  << solver_ms << " ms\n";
        if (params.ccd_check && result.ccd_violations > 0) {
            std::cout << "Self-intersection detected on frame " << frame_index << " — aborting.\n";
            std::exit(1);
        }

        if (!params.write_substeps)
            export_frame(outdir, frame_index, state.deformed_positions, ref_mesh.tris, fmt, nullptr);
        
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
