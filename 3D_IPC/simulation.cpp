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
#include <sstream>
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
    TwistSpec         twist_spec;
    CylinderTwistSpec cyl_twist_spec;

    std::vector<Vec3> static_x;
    std::vector<int>  static_tris;

    if      (args.example == 1) build_twisting_cloth_example(args, ref_mesh, state, X, pins, twist_spec);
    else if (args.example == 2) build_two_cylinder_twist_example(args, ref_mesh, state, X, pins, params, static_x, static_tris, cyl_twist_spec);
    else {
        std::cerr << "Unknown --example " << args.example << ". Valid values: 1, 2.\n";
        return 1;
    }

    PinTargetUpdater pin_updater = nullptr;
    if (args.example == 1) {
        pin_updater = [&twist_spec](std::vector<Pin>& p, double t) {
            update_twist_pins(p, twist_spec, t);
        };
    } else if (args.example == 2) {
        pin_updater = [&cyl_twist_spec](std::vector<Pin>& p, double t) {
            update_cylinder_twist_pins(p, cyl_twist_spec, t);
        };
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

    const char* collider_ext = (fmt == ExportFormat::GEO) ? ".geo"
                             : (fmt == ExportFormat::PLY) ? ".ply"
                             : (fmt == ExportFormat::USD) ? ".usda" : ".obj";

    auto write_collider_mesh = [&](const std::string& path) {
        if      (fmt == ExportFormat::GEO) export_geo(path, static_x, static_tris);
        else if (fmt == ExportFormat::PLY) export_ply(path, static_x, static_tris);
        else if (fmt == ExportFormat::USD) export_usd(path, static_x, static_tris);
        else                               export_obj(path, static_x, static_tris);
    };

    auto frame_collider_path = [&](int f) {
        std::ostringstream oss;
        oss << outdir << "/collider_"
            << std::setw(4) << std::setfill('0') << f << collider_ext;
        return oss.str();
    };

    // Example 2's cylinders rotate over time, so the collider mesh has to be
    // re-exported per frame. Other examples keep the one-time "static_colliders".
    const bool collider_is_dynamic = (args.example == 2);

    if (restart_frame < 0) {
        if (fs::exists(outdir)) fs::remove_all(outdir);
        fs::create_directories(outdir);
        export_frame(outdir, 0, state.deformed_positions, ref_mesh.tris, fmt);
        serialize_state(outdir, 0, state);
        if (!static_x.empty()) {
            if (collider_is_dynamic) {
                write_collider_mesh(frame_collider_path(0));
            } else {
                write_collider_mesh(outdir + "/static_colliders" + collider_ext);
            }
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
            state, ref_mesh, adj, pins, params, broad_phase,
            frame_index, pin_updater, substep_cb, outdir);

        if (!result.converged) {
            std::cerr << "Error: solver failed to converge at frame " << frame_index
                      << " (max_substep_iters = " << params.max_global_iters << ")\n";
            return 1;
        }

        auto solver_end = Clock::now();
        double solver_ms = std::chrono::duration<double, std::milli>(solver_end - solver_start).count();
        total_solver_ms += solver_ms;

        std::cout << "Frame " << std::setw(4) << frame_index
                  << " | global_iters = " << std::setw(3) << result.iterations
                  << " | solver_time = "  << std::fixed << std::setprecision(3)
                  << solver_ms << " ms\n";

        if (!params.write_substeps)
            export_frame(outdir, frame_index, state.deformed_positions, ref_mesh.tris, fmt, nullptr);

        if (collider_is_dynamic && !static_x.empty()) {
            // t = end of this frame, matching pin-target time used by the solver.
            const double t_frame = frame_index / params.fps;
            update_cylinder_visuals(static_x, cyl_twist_spec, t_frame);
            write_collider_mesh(frame_collider_path(frame_index));
        }

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
