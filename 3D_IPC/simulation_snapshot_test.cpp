#include "make_shape.h"
#include "mesh_utils.h"
#include "physics.h"
#include "simulation.h"
#include "solver.h"
#include "broad_phase.h"
#include "parallel_helper.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <limits>
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
// Keep this fixture consistent with restart_test.cpp and generate_golden.cpp.
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
    // Break the symmetric fixture's sensitivity to one-ULP input changes.
    // Keep this offset identical in the generator and both trajectory tests.
    pins.front().target_position.y() += 1.0 / 32.0;
    ref_mesh.build_lumped_mass(params.density, params.thickness);
    adj         = build_incident_triangle_map(ref_mesh.tris);
}

// ---------------------------------------------------------------------------
// Snapshot test
// ---------------------------------------------------------------------------
static void expect_trajectory_matches_golden(bool perturb_initial_position) {
    const auto golden = load_golden(std::string(GOLDEN_DIR) + "/golden_frames.txt");
    ASSERT_EQ(golden.size(), 100u);

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Pin> pins;
    VertexTriangleMap adj;
    SimParams params = SimParams::zeros();
    std::vector<Vec2> X;
    build_scene(ref_mesh, state, pins, adj, params, X);
    if (perturb_initial_position) {
        state.deformed_positions.front().x() = std::nextafter(
            state.deformed_positions.front().x(),
            std::numeric_limits<double>::infinity());
    }
    BroadPhase broad_phase;

    for (int frame = 1; frame <= 100; ++frame) {
        SCOPED_TRACE(::testing::Message() << "frame=" << frame);
        const auto result = advance_one_frame(state, ref_mesh, adj, pins, params, broad_phase);
        ASSERT_TRUE(result.converged);
        ASSERT_TRUE(golden.count(frame));
        const auto& expected = golden.at(frame);
        ASSERT_EQ(state.deformed_positions.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            SCOPED_TRACE(::testing::Message() << "vertex=" << i);
            ASSERT_TRUE(state.deformed_positions[i].allFinite());
            EXPECT_NEAR(state.deformed_positions[i].x(), expected[i].x(), kTol);
            EXPECT_NEAR(state.deformed_positions[i].y(), expected[i].y(), kTol);
            EXPECT_NEAR(state.deformed_positions[i].z(), expected[i].z(), kTol);
        }
    }
}

TEST(SimulationSnapshot, All100FramesMatchGolden) {
    expect_trajectory_matches_golden(false);
}

TEST(SimulationSnapshot, OneUlpPerturbationStaysWithinGoldenTolerance) {
    // A snapshot reference must tolerate ordinary input rounding, rather than
    // encode the arbitrary branch selected by a perfectly symmetric scene.
    expect_trajectory_matches_golden(true);
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

TEST(SimulationSnapshot, CooperativeContactsAreBitwiseEqualToOneThread) {
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
        parameters[run].friction_coefficient = 0.0;
        parameters[run].friction_velocity_epsilon = 0.01;
        omp_set_num_threads(run == 0 ? 1 : 8);
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

TEST(SimulationSnapshot, ExperimentalFrictionContactsAreBitwiseEqualAcrossAvailableThreads) {
    struct RestoreThreads {
        int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreThreads() {
            omp_set_num_threads(count);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    // Use native 64-thread coverage on the benchmark server without making
    // the default laptop suite oversubscribe busy-waiting scheduler workers.
    const int parallel_threads = std::min(64, omp_get_num_procs());
    RefMesh meshes[2];
    DeformedState states[2];
    std::vector<Pin> pins[2];
    VertexTriangleMap adjacency[2];
    BroadPhase broad_phases[2];
    SimParams parameters[2];
    std::vector<DeformedState> frames[2];
    for (int run = 0; run < 2; ++run) {
        std::vector<Vec2> material;
        parameters[run] = SimParams::zeros();
        build_scene(meshes[run], states[run], pins[run], adjacency[run],
                    parameters[run], material);
        clear_model(meshes[run], states[run], material, pins[run]);
        // Dense, nearby sheets exercise narrow and broad color groups,
        // with active friction and CCD contacts throughout the trajectory.
        constexpr int subdivisions = 8;
        constexpr int sheet_vertices = (subdivisions + 1) * (subdivisions + 1);
        build_square_mesh(meshes[run], states[run], material,
            subdivisions, subdivisions, 0.25, 0.25, Vec3::Zero());
        build_square_mesh(meshes[run], states[run], material,
            subdivisions, subdivisions, 0.25, 0.25, Vec3(0.007, 0.008, 0.003));
        for (std::size_t node = 0; node < states[run].deformed_positions.size(); ++node) {
            Vec3& position = states[run].deformed_positions[node];
            position.y() += 0.003 * std::sin(8.0 * position.x())
                * std::sin(7.0 * position.z());
            const double direction = node < sheet_vertices ? 1.0 : -1.0;
            states[run].velocities.emplace_back(0.03 * direction, -0.02, 0.01);
        }
        meshes[run].build_lumped_mass(parameters[run].density, parameters[run].thickness);
        adjacency[run] = build_incident_triangle_map(meshes[run].tris);
        append_pin(pins[run], 0, states[run].deformed_positions);
        append_pin(pins[run], sheet_vertices, states[run].deformed_positions);
        auto& params = parameters[run];
        params.use_parallel = true;
        params.use_basic_experimental = true;
        params.fixed_iters = true;
        params.use_ccd = true;
        params.use_ticcd = false;
        params.substeps = 2;
        // Each substep must execute batches of 3 + 3 + 1 sweeps. The last
        // partial batch checks both rebuilding and final iteration accounting.
        params.max_global_iters = 7;
        params.node_box_update_count = 3;
        params.node_box_min = params.node_box_max = 0.05;
        params.kB = 0.0025;
        params.d_hat = 0.012;
        params.k_barrier = 1.0;
        params.friction_coefficient = 0.1;
        params.friction_velocity_epsilon = 0.01;
        omp_set_num_threads(run == 0 ? 1 : parallel_threads);
        for (int frame = 0; frame < 3; ++frame) {
            SCOPED_TRACE(::testing::Message() << "run=" << run << " frame=" << frame);
            const auto result = advance_one_frame(states[run], meshes[run],
                adjacency[run], pins[run], params, broad_phases[run]);
            ASSERT_TRUE(result.converged);
            ASSERT_EQ(result.iterations, 14);
            ASSERT_FALSE(result.has_residual);

            const auto& cache = broad_phases[run].cache();
            const auto& positions = states[run].deformed_positions;
            const int vertices = static_cast<int>(positions.size());
            ASSERT_EQ(vertices, 162);
            ASSERT_FALSE(cache.nt_pairs.empty());
            ASSERT_FALSE(cache.ss_pairs.empty());
            // Require geometrically active contacts, not only broad-phase
            // candidates, throughout the multiframe friction regression.
            ASSERT_TRUE(std::any_of(cache.nt_pairs.begin(), cache.nt_pairs.end(),
                [&](const NodeTrianglePair& pair) {
                    return node_triangle_distance(positions[pair.node],
                        positions[pair.tri_v[0]], positions[pair.tri_v[1]],
                        positions[pair.tri_v[2]]).distance < params.d_hat;
                }));
            ASSERT_TRUE(std::any_of(cache.ss_pairs.begin(), cache.ss_pairs.end(),
                [&](const SegmentSegmentPair& pair) {
                    return segment_segment_distance(positions[pair.v[0]],
                        positions[pair.v[1]], positions[pair.v[2]],
                        positions[pair.v[3]]).distance < params.d_hat;
                }));

            // Check the actual last-rebuild graph has many color barriers;
            // a collision-free fixture would miss the targeted scheduling path.
            const auto elastic = build_elastic_adj(meshes[run], adjacency[run], vertices);
            std::vector<std::vector<int>> contacts, combined, colors;
            build_contact_adj(cache, vertices, contacts);
            union_adjacency(elastic, contacts, combined);
            greedy_color_conflict_graph(combined, colors);
            ASSERT_GE(vertices, 128);
            ASSERT_GE(colors.size(), std::size_t(32));
            frames[run].push_back(states[run]);
        }
    }

    const int vertices = static_cast<int>(states[1].deformed_positions.size());
    for (int frame = 0; frame < 3; ++frame) {
        ASSERT_EQ(frames[0][frame].deformed_positions.size(),
                  frames[1][frame].deformed_positions.size());
        for (int node = 0; node < vertices; ++node) {
            SCOPED_TRACE(::testing::Message() << "frame=" << frame << " node=" << node);
            ASSERT_TRUE(frames[1][frame].deformed_positions[node].allFinite());
            ASSERT_TRUE(frames[1][frame].velocities[node].allFinite());
            EXPECT_EQ(std::memcmp(frames[0][frame].deformed_positions[node].data(),
                frames[1][frame].deformed_positions[node].data(), 3 * sizeof(double)), 0);
            EXPECT_EQ(std::memcmp(frames[0][frame].velocities[node].data(),
                frames[1][frame].velocities[node].data(), 3 * sizeof(double)), 0);
        }
    }
}
