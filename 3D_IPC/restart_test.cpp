#include "make_shape.h"
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

static constexpr double kTol         = 1e-3;
static constexpr int    kRestartFrame = 50;
static constexpr int    kTotalFrames  = 100;

// ---------------------------------------------------------------------------
// Reuse the same golden loader as simulation_snapshot_test
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
// Same scene as simulation_snapshot_test
// ---------------------------------------------------------------------------
static void build_scene(RefMesh& ref_mesh, DeformedState& state,
                        std::vector<Pin>& pins, VertexTriangleMap& adj,
                        SimParams& params, std::vector<Vec2>& X) {
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

    clear_model(ref_mesh, state, X, pins);
    int nx = 10, ny = 10;
    int base = build_square_mesh(ref_mesh, state, X, nx, ny, 2.0, 2.0, Vec3(0.2, -0.1, 0.3));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    append_pin(pins, base + ny * (nx + 1),      state.deformed_positions);
    append_pin(pins, base + ny * (nx + 1) + nx, state.deformed_positions);
    ref_mesh.build_lumped_mass(params.density, params.thickness);
    adj = build_incident_triangle_map(ref_mesh.tris);
}

// ---------------------------------------------------------------------------
// Restart test
// ---------------------------------------------------------------------------
TEST(RestartTest, RestartFromFrame50MatchesGolden) {
    const std::string golden_path = std::string(GOLDEN_DIR) + "/golden_frames.txt";
    auto golden = load_golden(golden_path);
    ASSERT_FALSE(golden.empty()) << "Golden file empty or missing";

    const std::string checkpoint_dir = std::string(GOLDEN_DIR) + "/frame_50_checkpoint";

    // Load the pre-committed checkpoint and run frames kRestartFrame+1..kTotalFrames
    {
        RefMesh ref_mesh; DeformedState state;
        std::vector<Pin> pins; VertexTriangleMap adj;
        SimParams params = SimParams::zeros(); std::vector<Vec2> X;
        build_scene(ref_mesh, state, pins, adj, params, X);

        const auto color_groups = greedy_color(build_vertex_adjacency_map(ref_mesh.tris),
                                               static_cast<int>(state.deformed_positions.size()));

        ASSERT_TRUE(deserialize_state(checkpoint_dir, kRestartFrame, state))
            << "Failed to deserialize state at frame " << kRestartFrame;

        BroadPhase broad_phase;

        for (int f = kRestartFrame + 1; f <= kTotalFrames; ++f) {
            advance_one_frame(state, ref_mesh, adj, pins, params, color_groups, broad_phase);

            ASSERT_TRUE(golden.count(f)) << "No golden data for frame " << f;
            const auto& expected = golden[f];
            ASSERT_EQ(state.deformed_positions.size(), expected.size());

            for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
                EXPECT_NEAR(state.deformed_positions[i].x(), expected[i].x(), kTol)
                    << "frame=" << f << " vertex=" << i << " x mismatch";
                EXPECT_NEAR(state.deformed_positions[i].y(), expected[i].y(), kTol)
                    << "frame=" << f << " vertex=" << i << " y mismatch";
                EXPECT_NEAR(state.deformed_positions[i].z(), expected[i].z(), kTol)
                    << "frame=" << f << " vertex=" << i << " z mismatch";
            }
        }
    }

    // Note: frame_50_checkpoint is kept on disk for use by parallel_serial_consistency_test
}
