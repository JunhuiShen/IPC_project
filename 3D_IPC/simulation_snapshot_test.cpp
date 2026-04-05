#include "make_shape.h"
#include "physics.h"
#include "solver.h"
#include <gtest/gtest.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Tolerance for position comparison
static constexpr double kTol = 1e-6;

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
    params.step_weight  = 1.0;
    params.use_parallel = false;

    clear_model(ref_mesh, state, X, pins);
    int nx = 10, ny = 10;
    int base = build_square_mesh(ref_mesh, state, X, nx, ny, 2.0, 2.0, Vec3(0.2, -0.1, 0.3));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    append_pin(pins, base + ny * (nx + 1),      state.deformed_positions);
    append_pin(pins, base + ny * (nx + 1) + nx, state.deformed_positions);
    ref_mesh.build_lumped_mass(params.density, params.thickness);
    adj         = build_incident_triangle_map(ref_mesh.tris);
}

static std::vector<std::vector<int>> build_color_groups(const RefMesh& ref_mesh, int num_vertices) {
    return greedy_color(build_vertex_adjacency_map(ref_mesh.tris), num_vertices);
}

// ---------------------------------------------------------------------------
// Snapshot test
// ---------------------------------------------------------------------------
TEST(SimulationSnapshot, First5FramesMatchGolden) {
const std::string golden_path = std::string(GOLDEN_DIR) + "/golden_frames.txt";
auto golden = load_golden(golden_path);
ASSERT_FALSE(golden.empty()) << "Golden file empty or missing";

RefMesh ref_mesh; DeformedState state; std::vector<Pin> pins;
VertexTriangleMap adj; SimParams params; std::vector<Vec2> X;
build_scene(ref_mesh, state, pins, adj, params, X);

// No barrier, no coloring — serial path
const std::vector<NodeTrianglePair>   nt_pairs;
const std::vector<SegmentSegmentPair> ss_pairs;
const std::vector<std::vector<int>>   color_groups;

for (int frame = 1; frame <= 100; ++frame) {
for (int sub = 0; sub < params.substeps; ++sub) {
std::vector<Vec3> xhat;
build_xhat(xhat, state.deformed_positions, state.velocities, params.dt());
std::vector<Vec3> xnew = state.deformed_positions;
global_gauss_seidel_solver(ref_mesh, adj, pins, params, xnew, xhat, nt_pairs, ss_pairs, color_groups);
update_velocity(state.velocities, xnew, state.deformed_positions, params.dt());
state.deformed_positions = xnew;
}

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