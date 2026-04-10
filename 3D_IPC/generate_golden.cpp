#include "make_shape.h"
#include "physics.h"
#include "simulation.h"
#include "solver.h"
#include "broad_phase.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {
    SimParams params;
    params.fps             = 30.0;
    params.substeps        = 1;
    params.mu              = 10.0;
    params.lambda          = 10.0;
    params.density         = 1.0;
    params.thickness       = 0.1;
    params.kpin            = 1e7;
    params.gravity         = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 100;
    params.tol_abs         = 1e-6;
    params.step_weight     = 1.0;
    params.use_parallel    = false;

    RefMesh ref_mesh; DeformedState state;
    std::vector<Pin> pins; std::vector<Vec2> X;
    clear_model(ref_mesh, state, X, pins);
    int nx = 10, ny = 10;
    int base = build_square_mesh(ref_mesh, state, X, nx, ny, 2.0, 2.0, Vec3(0.2, -0.1, 0.3));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    append_pin(pins, base + ny * (nx + 1),      state.deformed_positions);
    append_pin(pins, base + ny * (nx + 1) + nx, state.deformed_positions);
    ref_mesh.build_lumped_mass(params.density, params.thickness);
    VertexTriangleMap adj = build_incident_triangle_map(ref_mesh.tris);

    BroadPhase broad_phase;
    const auto color_groups = greedy_color(build_vertex_adjacency_map(ref_mesh.tris),
                                           static_cast<int>(state.deformed_positions.size()));

    const std::string checkpoint_dir = std::string(GOLDEN_DIR) + "/frame_50_checkpoint";
    std::filesystem::create_directories(checkpoint_dir);

    std::ofstream out(std::string(GOLDEN_DIR) + "/golden_frames.txt");
    out << std::setprecision(15);

    for (int frame = 1; frame <= 100; ++frame) {
        advance_one_frame(state, ref_mesh, adj, pins, params, color_groups, broad_phase);

        if (frame == 50)
            serialize_state(checkpoint_dir, 50, state);

        out << "frame " << frame << "\n";
        for (int i = 0; i < static_cast<int>(state.deformed_positions.size()); ++i)
            out << i << " " << state.deformed_positions[i].x()
                     << " " << state.deformed_positions[i].y()
                     << " " << state.deformed_positions[i].z() << "\n";
    }

    std::cout << "Golden file written.\n";
    return 0;
}
