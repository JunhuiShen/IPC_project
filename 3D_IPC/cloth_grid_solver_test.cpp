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
    // test. Switching to a different mesh resets the grid solver's static
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

TEST(ClothGridSolver, SingleCellSerialUpdatesMatchBasicForContactCcdFrictionAndResidualModes) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    constexpr std::size_t configuration_count = 16;
    // Retain every mesh and its buffers until all configurations have finished,
    // so each entry point resets its topology-dependent workspace per run.
    std::array<ClothScene, 2 * configuration_count> scenes;
    for (std::size_t configuration = 0; configuration < configuration_count;
         ++configuration) {
        const bool contact = (configuration & 1) != 0;
        const bool ccd = (configuration & 2) != 0;
        const double friction = (configuration & 4) != 0 ? 0.2 : 0.0;
        const bool fixed = (configuration & 8) != 0;
        for (std::size_t method = 0; method < 2; ++method) {
            auto& scene = scenes[2 * configuration + method];
            build_contact_scene(scene, friction);
            scene.params.use_cloth_grid = method != 0;
            scene.params.use_parallel = false;
            scene.params.cloth_grid_dx = 2.0;
            scene.params.use_ccd = ccd;
            scene.params.use_ccd_guess = false;
            scene.params.d_hat = contact ? 0.012 : 0.0;
            scene.params.k_barrier = contact ? 1.0 : 0.0;
            scene.params.fixed_iters = fixed;
            scene.params.max_global_iters = 2;
            scene.params.node_box_update_count = 1;
            // Residual mode must execute the same two sweeps rather than exit
            // at its initial state; only residual reporting differs here.
            scene.params.tol_abs = 0.0;
            scene.params.tol_rel = 0.0;
            for (auto& position : scene.state.deformed_positions)
                position += Vec3::Ones();
            for (auto& pin : scene.pins)
                pin.target_position += Vec3::Ones();
        }
    }

    for (std::size_t configuration = 0; configuration < configuration_count;
         ++configuration) {
        SCOPED_TRACE(::testing::Message()
            << "contact=" << ((configuration & 1) != 0)
            << " ccd=" << ((configuration & 2) != 0)
            << " friction=" << ((configuration & 4) != 0 ? 0.2 : 0.0)
            << " fixed=" << ((configuration & 8) != 0));
        std::array<SolverResult, 2> results;
        for (std::size_t method = 0; method < 2; ++method) {
            auto& scene = scenes[2 * configuration + method];
            const auto initial_positions = scene.state.deformed_positions;
            std::vector<Vec3> predictor;
            build_xhat(predictor, initial_positions, scene.state.velocities,
                scene.params.dt());
            const auto solver = method == 0
                ? global_gauss_seidel_solver_basic
                : global_gauss_seidel_solver_ambient_grid;
            results[method] = solver(scene.mesh, scene.adjacency, scene.pins,
                scene.params, scene.state.deformed_positions, predictor,
                scene.state.velocities, scene.broad_phase, "", &initial_positions);
            EXPECT_EQ(results[method].iterations, scene.params.max_global_iters);
            EXPECT_EQ(results[method].converged, scene.params.fixed_iters);
            EXPECT_EQ(results[method].has_residual, !scene.params.fixed_iters);
            // Compare the velocity implied by the solved positions even when
            // zero tolerances deliberately leave residual mode unconverged.
            update_velocity(scene.state.velocities, scene.state.deformed_positions,
                initial_positions, scene.params.dt());

            // One cell preserves natural vertex order exactly; this fixture
            // compares numerical kernels rather than different GS schedules.
            solver_detail::ClothGridSchedule grid;
            grid.build(scene.state.deformed_positions,
                scene.broad_phase.cache().node_boxes, {}, scene.params.cloth_grid_dx);
            ASSERT_EQ(grid.cells.size(), 1u);
            const auto& vertices = grid.cells.front().vertices;
            ASSERT_EQ(vertices.size(), scene.state.deformed_positions.size());
            for (std::size_t node = 0; node < vertices.size(); ++node)
                EXPECT_EQ(vertices[node], static_cast<int>(node));
        }

        const auto& reference = scenes[2 * configuration].state;
        const auto& actual = scenes[2 * configuration + 1].state;
        ASSERT_EQ(actual.deformed_positions.size(), reference.deformed_positions.size());
        ASSERT_EQ(actual.velocities.size(), reference.velocities.size());
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
        EXPECT_EQ(results[1].iterations, results[0].iterations);
        EXPECT_EQ(results[1].converged, results[0].converged);
        EXPECT_EQ(results[1].has_residual, results[0].has_residual);
        EXPECT_EQ(results[1].has_residual_components, results[0].has_residual_components);
        const auto residual_values = [](const SolverResult& result) {
            return std::array<double, 8>{result.initial_residual, result.final_residual,
                result.initial_cloth_residual, result.final_cloth_residual,
                result.initial_solid_residual, result.final_solid_residual,
                result.initial_rigid_residual, result.final_rigid_residual};
        };
        const auto expected_residuals = residual_values(results[0]);
        const auto actual_residuals = residual_values(results[1]);
        for (std::size_t component = 0; component < actual_residuals.size(); ++component) {
            ASSERT_TRUE(std::isfinite(actual_residuals[component]));
            EXPECT_EQ(std::memcmp(&actual_residuals[component],
                &expected_residuals[component], sizeof(double)), 0)
                << "residual component=" << component;
        }
    }
}

TEST(ClothGridSolver, FrameDispatchMatchesExplicitSolverForBothMethods) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);

    for (const bool grid : {false, true}) {
        for (const double friction : {0.0, 0.2}) {
            SCOPED_TRACE(::testing::Message() << "grid=" << grid
                << " friction=" << friction);
            std::array<ClothScene, 2> scenes;
            for (auto& scene : scenes) {
                build_contact_scene(scene, friction);
                scene.params.use_parallel = true;
                scene.params.use_cloth_grid = grid;
            }

            auto& dispatched = scenes[0];
            const SolverResult frame_result = advance_one_frame(dispatched.state,
                dispatched.mesh, dispatched.adjacency, dispatched.pins,
                dispatched.params, dispatched.broad_phase, 1);
            ASSERT_TRUE(frame_result.converged);

            auto& explicit_entry = scenes[1];
            // Direct callers select the method by its entry point, not by
            // the frame driver's flag. Also exercise forwarding the old
            // positions needed by contact friction on every substep.
            explicit_entry.params.use_cloth_grid = !grid;
            const double dt = explicit_entry.params.dt();
            SolverResult explicit_result;
            for (int substep = 0; substep < explicit_entry.params.substeps; ++substep) {
                std::vector<Vec3> predictor;
                build_xhat(predictor, explicit_entry.state.deformed_positions,
                    explicit_entry.state.velocities, dt);
                std::vector<Vec3> positions = ccd_initial_guess(
                    explicit_entry.state.deformed_positions, predictor,
                    explicit_entry.mesh, &explicit_entry.broad_phase);
                const auto solver = grid
                    ? global_gauss_seidel_solver_ambient_grid
                    : global_gauss_seidel_solver_basic;
                const SolverResult sub_result = solver(explicit_entry.mesh,
                    explicit_entry.adjacency, explicit_entry.pins,
                    explicit_entry.params, positions, predictor,
                    explicit_entry.state.velocities, explicit_entry.broad_phase,
                    "", &explicit_entry.state.deformed_positions);
                ASSERT_TRUE(sub_result.converged);
                accumulate_solver_result(explicit_result, sub_result, substep == 0);
                update_velocity(explicit_entry.state.velocities, positions,
                    explicit_entry.state.deformed_positions, dt);
                explicit_entry.state.deformed_positions = positions;
            }

            EXPECT_EQ(explicit_result.iterations, frame_result.iterations);
            EXPECT_EQ(explicit_result.converged, frame_result.converged);
            EXPECT_EQ(explicit_result.has_residual, frame_result.has_residual);
            EXPECT_DOUBLE_EQ(explicit_result.initial_residual, frame_result.initial_residual);
            EXPECT_DOUBLE_EQ(explicit_result.final_residual, frame_result.final_residual);
            ASSERT_EQ(explicit_entry.state.deformed_positions.size(),
                dispatched.state.deformed_positions.size());
            ASSERT_EQ(explicit_entry.state.velocities.size(),
                dispatched.state.velocities.size());
            for (std::size_t node = 0;
                 node < explicit_entry.state.deformed_positions.size(); ++node) {
                EXPECT_EQ(std::memcmp(explicit_entry.state.deformed_positions[node].data(),
                    dispatched.state.deformed_positions[node].data(), 3 * sizeof(double)), 0)
                    << "position node=" << node;
                EXPECT_EQ(std::memcmp(explicit_entry.state.velocities[node].data(),
                    dispatched.state.velocities[node].data(), 3 * sizeof(double)), 0)
                    << "velocity node=" << node;
            }
        }
    }
}

TEST(ClothGridSolver, InterleavingMethodsKeepsAdaptiveNodeBoxHistoryIndependent) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    // Keep both meshes and their buffers alive while testing both directions.
    std::array<ClothScene, 2> scenes;
    for (auto& scene : scenes) {
        build_contact_scene(scene, 0.0);
        scene.params.use_parallel = false;
        scene.params.max_global_iters = 1;
        scene.params.node_box_update_count = 1;
        scene.state.velocities.assign(scene.state.deformed_positions.size(),
            Vec3::Zero());
    }

    for (std::size_t direction = 0; direction < scenes.size(); ++direction) {
        const bool grid_first = direction != 0;
        SCOPED_TRACE(::testing::Message() << "grid_first=" << grid_first);
        auto& scene = scenes[direction];
        const auto first_solver = grid_first
            ? global_gauss_seidel_solver_ambient_grid
            : global_gauss_seidel_solver_basic;
        const auto other_solver = grid_first
            ? global_gauss_seidel_solver_basic
            : global_gauss_seidel_solver_ambient_grid;
        const auto initial_positions = scene.state.deformed_positions;
        auto positions = initial_positions;
        ASSERT_TRUE(first_solver(scene.mesh, scene.adjacency, scene.pins,
            scene.params, positions, initial_positions, scene.state.velocities,
            scene.broad_phase, "", nullptr).converged);
        const auto first_result = positions;
        std::vector<double> expected_radii(positions.size());
        bool has_adapted_radius = false;
        for (std::size_t node = 0; node < positions.size(); ++node) {
            expected_radii[node] = std::clamp(
                1.2 * (first_result[node] - initial_positions[node]).norm(),
                scene.params.node_box_min, scene.params.node_box_max);
            has_adapted_radius = has_adapted_radius
                || expected_radii[node] < scene.params.node_box_max;
        }
        ASSERT_TRUE(has_adapted_radius)
            << "fixture must distinguish retained history from maximum-sized boxes";

        // Advance the other method on this same mesh with a different target.
        // Its displacement history must not replace the first method's cache.
        auto other_predictor = first_result;
        for (auto& position : other_predictor)
            position += Vec3(0.006, 0.0, 0.0);
        ASSERT_TRUE(other_solver(scene.mesh, scene.adjacency, scene.pins,
            scene.params, positions, other_predictor, scene.state.velocities,
            scene.broad_phase, "", nullptr).converged);
        const auto other_result = positions;
        bool histories_differ = false;
        for (std::size_t node = 0; node < positions.size(); ++node) {
            const double other_radius = std::clamp(
                1.2 * (other_result[node] - first_result[node]).norm(),
                scene.params.node_box_min, scene.params.node_box_max);
            histories_differ = histories_differ
                || std::abs(other_radius - expected_radii[node]) > 1e-8;
        }
        ASSERT_TRUE(histories_differ)
            << "fixture must distinguish each solver's displacement history";

        ASSERT_TRUE(first_solver(scene.mesh, scene.adjacency, scene.pins,
            scene.params, positions, other_result, scene.state.velocities,
            scene.broad_phase, "", nullptr).converged);
        const auto& boxes = scene.broad_phase.cache().node_boxes;
        ASSERT_EQ(boxes.size(), expected_radii.size());
        for (std::size_t node = 0; node < boxes.size(); ++node) {
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_NEAR(0.5 * boxes[node].extent()[axis], expected_radii[node], 1e-12)
                    << "node=" << node << " axis=" << axis;
                EXPECT_NEAR(0.5 * (boxes[node].min[axis] + boxes[node].max[axis]),
                    other_result[node][axis], 1e-12)
                    << "node=" << node << " axis=" << axis;
            }
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

TEST(ClothGridSolver, AmbientGridEntryRejectsRigidAndVolumetricMeshes) {
    SimParams params = valid_grid_parameters();
    const VertexTriangleMap adjacency;
    const std::vector<Pin> pins;
    const std::vector<Vec3> predictor(4, Vec3::Zero());
    const std::vector<Vec3> velocity(4, Vec3::Zero());
    std::vector<Vec3> positions = predictor;
    BroadPhase broad_phase;
    RefMesh rigid;
    rigid.rb_nodes = {{0, 1, 2, 3}};
    RefMesh solid;
    solid.tet_nodes = {0, 1, 2, 3};
    solid.tets = {0, 1, 2, 3};
    RefMesh unclassified_solid;
    unclassified_solid.tets = solid.tets;
    for (const bool enabled : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "use_cloth_grid=" << enabled);
        params.use_cloth_grid = enabled;
        EXPECT_THROW(global_gauss_seidel_solver_ambient_grid(rigid, adjacency, pins,
            params, positions, predictor, velocity, broad_phase), std::invalid_argument);
        EXPECT_THROW(global_gauss_seidel_solver_ambient_grid(solid, adjacency, pins,
            params, positions, predictor, velocity, broad_phase), std::invalid_argument);
        // Reject tet connectivity even before a caller has built tet_nodes.
        EXPECT_THROW(global_gauss_seidel_solver_ambient_grid(unclassified_solid,
            adjacency, pins, params, positions, predictor, velocity, broad_phase),
            std::invalid_argument);
    }
}

TEST(ClothGridSolver, AmbientGridEntryValidatesParametersRegardlessOfDispatchFlag) {
    ClothScene scene;
    build_contact_scene(scene, 0.0);
    const auto solve = [&](const SimParams& params) {
        auto positions = scene.state.deformed_positions;
        return global_gauss_seidel_solver_ambient_grid(scene.mesh,
            scene.adjacency, scene.pins, params, positions,
            scene.state.deformed_positions, scene.state.velocities,
            scene.broad_phase);
    };
    for (const bool enabled : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "use_cloth_grid=" << enabled);
        SimParams params = scene.params;
        params.use_cloth_grid = enabled;
        params.cloth_grid_dx = 0.02;
        EXPECT_THROW(solve(params), std::invalid_argument);
        params = scene.params;
        params.use_cloth_grid = enabled;
        params.node_box_update_count = 0;
        EXPECT_THROW(solve(params), std::invalid_argument);
        params = scene.params;
        params.use_cloth_grid = enabled;
        params.use_ogc = true;
        EXPECT_THROW(solve(params), std::invalid_argument);
        params = scene.params;
        params.use_cloth_grid = enabled;
        params.use_ogc_solver = true;
        EXPECT_THROW(solve(params), std::invalid_argument);
    }
}

TEST(ClothGridSolver, FrameDispatchRejectsGridWithEitherOgcModeBeforeAdvancing) {
    ClothScene scene;
    build_contact_scene(scene, 0.0);
    const auto initial_positions = scene.state.deformed_positions;
    for (const bool ogc_solver : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "use_ogc_solver=" << ogc_solver);
        scene.params.use_ogc = !ogc_solver;
        scene.params.use_ogc_solver = ogc_solver;
        bool pin_updater_called = false;
        bool substep_callback_called = false;
        EXPECT_THROW(advance_one_frame(scene.state, scene.mesh,
            scene.adjacency, scene.pins, scene.params, scene.broad_phase, 1,
            [&](std::vector<Pin>&, double) { pin_updater_called = true; },
            [&](int, const std::vector<Vec3>&) { substep_callback_called = true; }),
            std::invalid_argument);
        EXPECT_FALSE(pin_updater_called);
        EXPECT_FALSE(substep_callback_called);
        ASSERT_EQ(scene.state.deformed_positions.size(), initial_positions.size());
        for (std::size_t node = 0; node < initial_positions.size(); ++node)
            EXPECT_EQ(std::memcmp(scene.state.deformed_positions[node].data(),
                initial_positions[node].data(), 3 * sizeof(double)), 0);
    }
}
