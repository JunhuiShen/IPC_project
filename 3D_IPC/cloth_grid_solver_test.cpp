#include "broad_phase.h"
#include "grid_contact_scheduling.h"
#include "ipc_args.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "simulation.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct RestoreOpenMPSettings {
    const int threads = omp_get_max_threads();
    const int dynamic = omp_get_dynamic();
    ~RestoreOpenMPSettings() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

SimParams valid_grid_parameters() {
    SimParams params = SimParams::zeros();
    params.use_cloth_grid = true;
    params.cloth_grid_dx = 0.055;
    params.node_box_min = 0.001;
    params.node_box_max = 0.01;
    params.node_box_update_count = 2;
    params.max_global_iters = 4;
    return params;
}

struct ClothScene {
    RefMesh mesh;
    DeformedState state;
    std::vector<Pin> pins;
    VertexTriangleMap adjacency;
    BroadPhase broad_phase;
    SimParams params;
};

void build_contact_scene(ClothScene& scene, double friction) {
    scene.params = valid_grid_parameters();
    auto& params = scene.params;
    params.fps = 30.0;
    params.substeps = 2;
    params.mu = 20.0;
    params.lambda = 25.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.kB = 0.003;
    params.kpin = 1e6;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.fixed_iters = true;
    params.use_ccd = true;
    params.use_ccd_guess = true;
    params.use_ticcd = false;
    params.d_hat = 0.012;
    params.k_barrier = 1.0;
    params.friction_coefficient = friction;
    params.friction_velocity_epsilon = 0.01;

    constexpr int subdivisions = 8;
    constexpr int sheet_vertices = (subdivisions + 1) * (subdivisions + 1);
    std::vector<Vec2> material;
    build_square_mesh(scene.mesh, scene.state, material, subdivisions,
        subdivisions, 0.45, 0.45, Vec3(-0.31, -0.022, -0.23));
    build_square_mesh(scene.mesh, scene.state, material, subdivisions,
        subdivisions, 0.45, 0.45, Vec3(-0.303, -0.014, -0.227));

    // Preserve flat rest hinges, then bend and stretch both nearby sheets.
    // The stretched edges can connect cells of the same parity, so parity
    // alone cannot make the parallel batches safe. Negative coordinates and
    // multiple vertices per cell also exercise cell ownership.
    for (std::size_t node = 0; node < scene.state.deformed_positions.size(); ++node) {
        Vec3& position = scene.state.deformed_positions[node];
        position.x() *= 1.4;
        position.y() += 0.004 * std::sin(8.0 * position.x())
            * std::sin(7.0 * position.z());
        const double direction = node < sheet_vertices ? 1.0 : -1.0;
        scene.state.velocities.emplace_back(0.035 * direction, -0.02,
            0.025 * direction);
    }
    append_pin(scene.pins, 0, scene.state.deformed_positions);
    append_pin(scene.pins, subdivisions, scene.state.deformed_positions);
    append_pin(scene.pins, sheet_vertices, scene.state.deformed_positions);
    // A moved pin target makes the local pin term active on the first sweep.
    scene.pins.back().target_position.x() += 0.002;
    scene.mesh.build_lumped_mass(params.density, params.thickness);
    scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
}

bool parse_arguments(IPCArgs3D& args, std::vector<std::string> words) {
    std::vector<char*> argv;
    argv.reserve(words.size());
    for (auto& word : words)
        argv.push_back(word.data());
    return args.parse(static_cast<int>(argv.size()), argv.data());
}

struct TemporaryArgsFile {
    std::filesystem::path directory;
    std::filesystem::path path;

    TemporaryArgsFile() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory = std::filesystem::temp_directory_path()
            / ("ipc_cloth_grid_args_" + std::to_string(stamp));
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("cannot create temporary args directory");
        path = directory / "args.txt";
    }

    ~TemporaryArgsFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        std::filesystem::remove(directory, ignored);
    }
};

} // namespace

TEST(ClothGridSolver, ContactFramesMatchSerialGridAcrossThreadCounts) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    constexpr std::array<int, 4> thread_counts = {1, 1, 4, 8};
    constexpr std::array<double, 2> friction_values = {0.0, 0.2};
    constexpr int frames = 3;

    // All eight mesh objects and their buffers stay alive throughout the
    // test. Switching to a different mesh resets the basic solver's static
    // previous-displacement workspace, without contaminating a later run or
    // discarding the adaptive history between frames of the same run.
    std::array<ClothScene, 8> scenes;
    std::array<std::array<DeformedState, frames>, 8> snapshots;
    for (std::size_t variant = 0; variant < friction_values.size(); ++variant) {
        for (std::size_t run = 0; run < thread_counts.size(); ++run)
            build_contact_scene(scenes[4 * variant + run], friction_values[variant]);
    }

    for (std::size_t variant = 0; variant < friction_values.size(); ++variant) {
        SCOPED_TRACE(::testing::Message() << "friction=" << friction_values[variant]);
        for (std::size_t run = 0; run < thread_counts.size(); ++run) {
            SCOPED_TRACE(::testing::Message() << "run=" << run
                << " threads=" << thread_counts[run]);
            const std::size_t index = 4 * variant + run;
            auto& scene = scenes[index];
            scene.params.use_parallel = run != 0;
            omp_set_num_threads(thread_counts[run]);
            ASSERT_FALSE(scene.mesh.hinges.empty());
            ASSERT_GT(scene.state.deformed_positions.size(), 128u);
            const auto initial_positions = scene.state.deformed_positions;
            int substep_callbacks = 0;

            for (int frame = 1; frame <= frames; ++frame) {
                SCOPED_TRACE(::testing::Message() << "frame=" << frame);
                const SolverResult result = advance_one_frame(scene.state,
                    scene.mesh, scene.adjacency, scene.pins, scene.params,
                    scene.broad_phase, frame, nullptr,
                    [&](int global_substep, const std::vector<Vec3>& positions) {
                        EXPECT_EQ(global_substep, substep_callbacks++);
                        const auto& boxes = scene.broad_phase.cache().node_boxes;
                        ASSERT_EQ(boxes.size(), positions.size());
                        for (std::size_t node = 0; node < positions.size(); ++node) {
                            ASSERT_TRUE(positions[node].allFinite()) << "node=" << node;
                            ASSERT_TRUE(boxes[node].min.allFinite()) << "node=" << node;
                            ASSERT_TRUE(boxes[node].max.allFinite()) << "node=" << node;
                            EXPECT_TRUE((positions[node].array()
                                >= boxes[node].min.array() - 1e-12).all()) << "node=" << node;
                            EXPECT_TRUE((positions[node].array()
                                <= boxes[node].max.array() + 1e-12).all()) << "node=" << node;
                            EXPECT_LE(0.5 * boxes[node].extent().maxCoeff(),
                                scene.params.node_box_max + 1e-12) << "node=" << node;
                        }
                    });
                ASSERT_TRUE(result.converged);
                EXPECT_EQ(result.iterations,
                    scene.params.substeps * scene.params.max_global_iters);
                ASSERT_FALSE(scene.broad_phase.cache().nt_pairs.empty());
                snapshots[index][frame - 1] = scene.state;

                const auto& reference = snapshots[4 * variant][frame - 1];
                ASSERT_EQ(scene.state.deformed_positions.size(),
                    reference.deformed_positions.size());
                ASSERT_EQ(scene.state.velocities.size(), reference.velocities.size());
                for (std::size_t node = 0; node < scene.state.deformed_positions.size(); ++node) {
                    ASSERT_TRUE(scene.state.velocities[node].allFinite()) << "node=" << node;
                    EXPECT_EQ(std::memcmp(scene.state.deformed_positions[node].data(),
                        reference.deformed_positions[node].data(), 3 * sizeof(double)), 0)
                        << "position node=" << node;
                    EXPECT_EQ(std::memcmp(scene.state.velocities[node].data(),
                        reference.velocities[node].data(), 3 * sizeof(double)), 0)
                        << "velocity node=" << node;
                }
            }
            EXPECT_EQ(substep_callbacks, frames * scene.params.substeps);
            double displacement = 0.0;
            for (std::size_t node = 0; node < initial_positions.size(); ++node)
                displacement += (scene.state.deformed_positions[node]
                    - initial_positions[node]).squaredNorm();
            EXPECT_GT(displacement, 1e-10);
        }
    }
}

TEST(ClothGridSolver, DenseSingleCellMatchesSerialWithBothCcdBackendsAndClippingModes) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    for (const bool ccd : {false, true}) {
        for (const bool tight_inclusion : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "ccd=" << ccd
                << " tight_inclusion=" << tight_inclusion);
            std::array<ClothScene, 2> scenes;
            for (auto& scene : scenes) {
                build_contact_scene(scene, 0.0);
                scene.params.cloth_grid_dx = 2.0;
                scene.params.node_box_max = 0.05;
                scene.params.node_box_min = 0.05;
                scene.params.use_ccd = ccd;
                scene.params.use_ticcd = tight_inclusion;
                for (auto& point : scene.state.deformed_positions)
                    point += Vec3::Ones();
                for (auto& pin : scene.pins)
                    pin.target_position += Vec3::Ones();
            }
            for (int mode = 0; mode < 2; ++mode) {
                auto& scene = scenes[mode];
                scene.params.use_parallel = mode != 0;
                omp_set_num_threads(mode == 0 ? 1 : 4);
                ASSERT_TRUE(advance_one_frame(scene.state, scene.mesh,
                    scene.adjacency, scene.pins, scene.params,
                    scene.broad_phase, 1).converged);
            }
            const auto& reference = scenes[0].state;
            const auto& actual = scenes[1].state;
            for (std::size_t node = 0; node < actual.deformed_positions.size(); ++node) {
                ASSERT_TRUE(actual.deformed_positions[node].allFinite());
                ASSERT_TRUE(actual.velocities[node].allFinite());
                EXPECT_EQ(std::memcmp(actual.deformed_positions[node].data(),
                    reference.deformed_positions[node].data(), 3 * sizeof(double)), 0)
                    << "position node=" << node;
                EXPECT_EQ(std::memcmp(actual.velocities[node].data(),
                    reference.velocities[node].data(), 3 * sizeof(double)), 0)
                    << "velocity node=" << node;
            }
            // Confirm this integration fixture actually requests cooperation,
            // rather than only exercising the small-contact fallback.
            const auto& cache = scenes[1].broad_phase.cache();
            solver_detail::ClothGridSchedule grid;
            grid.build(actual.deformed_positions, cache.node_boxes, {}, 2.0);
            ASSERT_EQ(grid.cells.size(), 1u);
            solver_detail::ClothGridContactSweep sweep;
            sweep.prepare(grid, cache);
            EXPECT_FALSE(sweep.cooperative_cells.empty());
        }
    }
}

TEST(ClothGridParameters, RejectsInvalidEnabledConfiguration) {
    const SimParams valid = valid_grid_parameters();
    EXPECT_NO_THROW(valid.validate_cloth_grid_parameters());
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (const double dx : {0.0, -0.05, 0.01, 0.02, infinity, nan}) {
        SCOPED_TRACE(::testing::Message() << "dx=" << dx);
        SimParams params = valid;
        params.cloth_grid_dx = dx;
        EXPECT_THROW(params.validate_cloth_grid_parameters(), std::invalid_argument);
    }
    for (const double minimum : {-0.001, 0.0, 1e-10, 0.02, infinity, nan}) {
        SCOPED_TRACE(::testing::Message() << "minimum=" << minimum);
        SimParams params = valid;
        params.node_box_min = minimum;
        EXPECT_THROW(params.validate_cloth_grid_parameters(), std::invalid_argument);
    }
    for (const double maximum : {-0.01, 0.0, 0.0001, infinity, nan}) {
        SCOPED_TRACE(::testing::Message() << "maximum=" << maximum);
        SimParams params = valid;
        params.node_box_max = maximum;
        EXPECT_THROW(params.validate_cloth_grid_parameters(), std::invalid_argument);
    }
    for (const int count : {-1, 0}) {
        SimParams params = valid;
        params.node_box_update_count = count;
        EXPECT_THROW(params.validate_cloth_grid_parameters(), std::invalid_argument);
        params = valid;
        params.max_global_iters = count;
        EXPECT_THROW(params.validate_cloth_grid_parameters(), std::invalid_argument);
    }
    SimParams params = valid;
    params.use_ogc = true;
    EXPECT_THROW(params.validate_cloth_grid_parameters(), std::invalid_argument);
    params = valid;
    params.use_ogc_solver = true;
    EXPECT_THROW(params.validate_cloth_grid_parameters(), std::invalid_argument);

    // Opting out leaves the pre-existing non-grid parameter contract intact.
    params = SimParams::zeros();
    params.cloth_grid_dx = nan;
    EXPECT_NO_THROW(params.validate_cloth_grid_parameters());
}

TEST(ClothGridParameters, CliParsesAndSerializesGridSettings) {
    IPCArgs3D defaults;
    EXPECT_FALSE(defaults.to_sim_params().use_cloth_grid);
    EXPECT_DOUBLE_EQ(defaults.to_sim_params().cloth_grid_dx, 0.05);

    IPCArgs3D args;
    ASSERT_TRUE(parse_arguments(args, {"3D_sim", "--use_cloth_grid", "true",
        "--cloth_grid_dx", "0.064", "--node_box_max", "0.004",
        "--node_box_update_count", "3", "--use_parallel", "false"}));
    const SimParams params = args.to_sim_params();
    EXPECT_TRUE(params.use_cloth_grid);
    EXPECT_FALSE(params.use_parallel);
    EXPECT_DOUBLE_EQ(params.cloth_grid_dx, 0.064);
    EXPECT_DOUBLE_EQ(params.node_box_max, 0.004);
    EXPECT_EQ(params.node_box_update_count, 3);

    TemporaryArgsFile saved;
    args.serialize(saved.path.string());
    ASSERT_TRUE(std::filesystem::exists(saved.path));
    IPCArgs3D restored;
    ASSERT_TRUE(restored.deserialize(saved.path.string()));
    const SimParams round_trip = restored.to_sim_params();
    EXPECT_TRUE(round_trip.use_cloth_grid);
    EXPECT_FALSE(round_trip.use_parallel);
    EXPECT_DOUBLE_EQ(round_trip.cloth_grid_dx, params.cloth_grid_dx);
    EXPECT_DOUBLE_EQ(round_trip.node_box_max, params.node_box_max);
    EXPECT_EQ(round_trip.node_box_update_count, params.node_box_update_count);

    IPCArgs3D bare_flag;
    ASSERT_TRUE(parse_arguments(bare_flag, {"3D_sim", "--use_cloth_grid"}));
    EXPECT_TRUE(bare_flag.to_sim_params().use_cloth_grid);
    IPCArgs3D invalid;
    ASSERT_TRUE(parse_arguments(invalid,
        {"3D_sim", "--use_cloth_grid", "true", "--cloth_grid_dx", "0.02"}));
    EXPECT_THROW(invalid.to_sim_params(), std::invalid_argument);
}

TEST(ClothGridSolver, BasicEntryRejectsRigidAndVolumetricMeshes) {
    const SimParams params = valid_grid_parameters();
    const VertexTriangleMap adjacency;
    const std::vector<Pin> pins;
    const std::vector<Vec3> predictor(4, Vec3::Zero());
    const std::vector<Vec3> velocity(4, Vec3::Zero());
    std::vector<Vec3> positions = predictor;
    BroadPhase broad_phase;
    RefMesh rigid;
    rigid.rb_nodes = {{0, 1, 2, 3}};
    EXPECT_THROW(global_gauss_seidel_solver_basic(rigid, adjacency, pins,
        params, positions, predictor, velocity, broad_phase), std::invalid_argument);

    RefMesh solid;
    solid.tet_nodes = {0, 1, 2, 3};
    solid.tets = {0, 1, 2, 3};
    EXPECT_THROW(global_gauss_seidel_solver_basic(solid, adjacency, pins,
        params, positions, predictor, velocity, broad_phase), std::invalid_argument);
    // Reject tet connectivity even before a caller has built tet_nodes.
    solid.tet_nodes.clear();
    EXPECT_THROW(global_gauss_seidel_solver_basic(solid, adjacency, pins,
        params, positions, predictor, velocity, broad_phase), std::invalid_argument);
}
