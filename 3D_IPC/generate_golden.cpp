#include "make_shape.h"
#include "physics.h"
#include "solver.h"
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

    const std::vector<NodeTrianglePair>   nt_pairs;
    const std::vector<SegmentSegmentPair> ss_pairs;
    const auto color_groups = greedy_color(build_vertex_adjacency_map(ref_mesh.tris),
                                           static_cast<int>(state.deformed_positions.size()));

    std::ofstream out(std::string(GOLDEN_DIR) + "/golden_frames.txt");
    out << std::setprecision(15);

    for (int frame = 1; frame <= 100; ++frame) {
        for (int sub = 0; sub < params.substeps; ++sub) {
            std::vector<Vec3> xhat;
            build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());
            std::vector<Vec3> xnew = state.deformed_positions;
            global_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat,
                                       nt_pairs, ss_pairs, color_groups);
            update_velocity(state.velocities, xnew, state.deformed_positions, params.dt());
            state.deformed_positions = xnew;
        }

        out << "frame " << frame << "\n";
        for (int i = 0; i < static_cast<int>(state.deformed_positions.size()); ++i)
            out << i << " " << state.deformed_positions[i].x()
                     << " " << state.deformed_positions[i].y()
                     << " " << state.deformed_positions[i].z() << "\n";
    }

    std::cout << "Golden file written.\n";
    return 0;
}
