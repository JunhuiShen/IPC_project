#include "make_triangle.h"
#include "physics.h"
#include "solver.h"
#include "visualization.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fs = std::__fs::filesystem;

int main() {
    SimParams params;
    params.dt = 1.0 / 30.0;
    params.mu = 200.0;
    params.lambda = 200.0;
    params.density = 1000.0;
    params.thickness = 0.1;
    params.kpin = 1e5;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 100;
    params.tol_abs = 1e-6;
    params.step_weight = 1.0;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Pin> pins;

    clear_model(ref_mesh, state, pins);

    int tri0 = build_single_triangle(
            ref_mesh, state,
            Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.2, 1.0),
            Vec3(-1.5, 0.0, 0.2), Vec3(-0.5, 0.0, 0.4), Vec3(-1.3, 1.0, 0.3)
    );

    int tri1 = build_single_triangle(
            ref_mesh, state,
            Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.2, 1.0),
            Vec3( 0.5, 0.0, 1.0), Vec3( 1.5, 0.0, -0.5), Vec3( 0.7, 1.0, 0.0)
    );

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    append_pin(pins, tri0 + 0, state.deformed_positions);
    append_pin(pins, tri1 + 0, state.deformed_positions);

    LumpedMass lumped_mass = build_lumped_mass(ref_mesh, params.density, params.thickness);
    VertexAdjacency adj = build_vertex_adjacency(ref_mesh);

    std::cout << "Vertices:  " << state.deformed_positions.size() << "\n";
    std::cout << "Triangles: " << ref_mesh.tris.size() << "\n";

    std::string outdir = "frames_sim3d";
    if (fs::exists(outdir)) fs::remove_all(outdir);
    fs::create_directories(outdir);

    export_frame(outdir, 0, state.deformed_positions, ref_mesh.tris);

    const int num_frames = 100;

    for (int frame_index = 1; frame_index <= num_frames; ++frame_index) {
        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, params.dt);

        std::vector<Vec3> xnew = xhat;

        SolverResult result = global_gauss_seidel_solver(ref_mesh, lumped_mass, adj, pins, params, xnew, xhat);

        std::cout << "Frame " << std::setw(4) << frame_index
                  << " | initial_residual = " << std::scientific << result.initial_residual
                  << " | final_residual = " << std::scientific << result.final_residual
                  << " | global_iters = " << std::setw(3) << result.iterations << "\n";

        update_velocity(state.velocities, xnew, state.deformed_positions, params.dt);
        state.deformed_positions = xnew;

        export_frame(outdir, frame_index, state.deformed_positions, ref_mesh.tris);
    }

    return 0;
}
