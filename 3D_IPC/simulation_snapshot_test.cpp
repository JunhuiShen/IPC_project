#include "make_shape.h"
#include "mesh_utils.h"
#include "physics.h"
#include "simulation.h"
#include "solver.h"
#include "broad_phase.h"
#include <gtest/gtest.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>
#include <cstring>

// Tolerance for position comparison
static constexpr double kTol = 1e-3;

// ---------------------------------------------------------------------------
// Parse golden file produced by dump_frames
// golden[frame][vertex] = Vec3
// ---------------------------------------------------------------------------
static std::map<int, std::vector<Vec3>> load_golden(const std::string& path) {
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "Cannot open golden file: " << path;

    std::map<int, std::vector<Vec3>> golden;
    int current_frame = -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("frame ", 0) == 0) {
            current_frame = std::stoi(line.substr(6));
        } else if (current_frame >= 0) {
            std::istringstream ss(line);
            int idx; double x, y, z;
            ss >> idx >> x >> y >> z;
            golden[current_frame].emplace_back(x, y, z);
        }
    }
    return golden;
}

// ---------------------------------------------------------------------------
// Same scene setup as simulation.cpp / dump_frames.cpp
// ---------------------------------------------------------------------------
static void build_scene(RefMesh& ref_mesh, DeformedState& state, std::vector<Pin>& pins, VertexTriangleMap& adj, SimParams& params, std::vector<Vec2> X) {
    params.fps          = 30.0;
    params.substeps     = 1;
    params.mu           = 10.0;
    params.lambda       = 10.0;
    params.density      = 1.0;
    params.thickness    = 0.1;
    params.kpin         = 1e7;
    params.gravity      = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 100;
    params.tol_abs      = 1e-6;
    params.use_parallel = false;
    params.fixed_iters  = true;
    params.node_box_min = 0.001;
    params.node_box_max = 0.01;

    clear_model(ref_mesh, state, X, pins);
    int nx = 10, ny = 10;
    int base = build_square_mesh(ref_mesh, state, X, nx, ny, 2.0, 2.0, Vec3(0.2, -0.1, 0.3));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    append_pin(pins, base + ny * (nx + 1),      state.deformed_positions);
    append_pin(pins, base + ny * (nx + 1) + nx, state.deformed_positions);
    ref_mesh.build_lumped_mass(params.density, params.thickness);
    adj         = build_incident_triangle_map(ref_mesh.tris);
}

// ---------------------------------------------------------------------------
// Snapshot test
// ---------------------------------------------------------------------------
TEST(SimulationSnapshot, First5FramesMatchGolden) {
const std::string golden_path = std::string(GOLDEN_DIR) + "/golden_frames.txt";
auto golden = load_golden(golden_path);
ASSERT_FALSE(golden.empty()) << "Golden file empty or missing";

RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
VertexTriangleMap adj; SimParams params = SimParams::zeros(); std::vector<Vec2> X;
build_scene(ref_mesh, state, pins, adj, params, X);

BroadPhase broad_phase;

for (int frame = 1; frame <= 100; ++frame) {
advance_one_frame(state, ref_mesh, adj, pins, params, broad_phase);

ASSERT_TRUE(golden.count(frame)) << "No golden data for frame " << frame;
const auto& expected = golden[frame];
ASSERT_EQ(state.deformed_positions.size(), expected.size());

for (int i = 0; i < (int)expected.size(); ++i) {
EXPECT_NEAR(state.deformed_positions[i].x(), expected[i].x(), kTol)
<< "frame=" << frame << " vertex=" << i << " x mismatch";
EXPECT_NEAR(state.deformed_positions[i].y(), expected[i].y(), kTol)
<< "frame=" << frame << " vertex=" << i << " y mismatch";
EXPECT_NEAR(state.deformed_positions[i].z(), expected[i].z(), kTol)
<< "frame=" << frame << " vertex=" << i << " z mismatch";
}
}
}

TEST(SimulationSnapshot, ColoredBasicSolverIsBitwiseEqualAcrossThreadCounts) {
    struct RestoreThreads {
        int count = omp_get_max_threads();
        ~RestoreThreads() { omp_set_num_threads(count); }
    } restore;
    RefMesh meshes[2];
    DeformedState states[2];
    std::vector<Pin> pins[2];
    VertexTriangleMap adjacency[2];
    BroadPhase broad_phases[2];
    SimParams parameters[2];
    for (int run = 0; run < 2; ++run) {
        std::vector<Vec2> material;
        parameters[run] = SimParams::zeros();
        build_scene(meshes[run], states[run], pins[run], adjacency[run], parameters[run], material);
        // Rebuild with two nearby sheets to exercise contacts, CCD and cache
        // rebuilds above the parallel thresholds.
        clear_model(meshes[run], states[run], material, pins[run]);
        build_square_mesh(meshes[run], states[run], material, 12, 12, 1, 1, Vec3::Zero());
        build_square_mesh(meshes[run], states[run], material, 12, 12, 1, 1, Vec3(0.007, 0.008, 0.003));
        states[run].velocities.assign(states[run].deformed_positions.size(), Vec3(0.01, -0.02, 0.03));
        meshes[run].build_lumped_mass(parameters[run].density, parameters[run].thickness);
        adjacency[run] = build_incident_triangle_map(meshes[run].tris);
        append_pin(pins[run], 0, states[run].deformed_positions);
        parameters[run].use_parallel = true;
        parameters[run].use_ccd = true;
        parameters[run].max_global_iters = 6;
        parameters[run].node_box_update_count = 2;
        parameters[run].d_hat = 0.01;
        parameters[run].k_barrier = 1.0;
        parameters[run].friction_coefficient = 0.2;
        parameters[run].friction_velocity_epsilon = 0.01;
        omp_set_num_threads(run == 0 ? 1 : 4);
        for (int frame = 0; frame < 3; ++frame)
            advance_one_frame(states[run], meshes[run], adjacency[run], pins[run], parameters[run], broad_phases[run]);
    }
    ASSERT_EQ(states[0].deformed_positions.size(), states[1].deformed_positions.size());
    EXPECT_FALSE(broad_phases[0].cache().nt_pairs.empty());
    for (std::size_t i = 0; i < states[0].deformed_positions.size(); ++i) {
        EXPECT_EQ(std::memcmp(states[0].deformed_positions[i].data(), states[1].deformed_positions[i].data(), 3 * sizeof(double)), 0) << i;
        EXPECT_EQ(std::memcmp(states[0].velocities[i].data(), states[1].velocities[i].data(), 3 * sizeof(double)), 0) << i;
    }
}
