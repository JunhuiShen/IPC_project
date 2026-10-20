#include "initial_guess.h"
#include "broad_phase.h"
#include "ipc_args.h"
#include "mesh_utils.h"
#include "node_triangle_distance.h"
#include "parallel_helper.h"
#include "safe_step.h"
#include "segment_segment_distance.h"
#include "simulation.h"
#include "solid_ipc.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_vec_near(const Vec3& actual, const Vec3& expected, double tol) {
    EXPECT_NEAR(actual.x(), expected.x(), tol);
    EXPECT_NEAR(actual.y(), expected.y(), tol);
    EXPECT_NEAR(actual.z(), expected.z(), tol);
}

RefMesh ref_mesh_with_masses(std::initializer_list<double> masses) {
    RefMesh ref_mesh;
    ref_mesh.mass.assign(masses.begin(), masses.end());
    ref_mesh.num_positions = ref_mesh.mass.size();
    return ref_mesh;
}

SimParams base_params() {
    SimParams params = SimParams::zeros();
    params.fps = 2.0;
    params.substeps = 1;
    params.gravity = Vec3::Zero();
    return params;
}

struct RestoreInitialGuessOpenMP {
    int threads = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    ~RestoreInitialGuessOpenMP() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

class CollisionColoredCCDInitialGuess : public testing::Test {
protected:
    RestoreInitialGuessOpenMP restore_openmp;
    SimParams params = base_params();
    void SetUp() override {
        omp_set_dynamic(0);
        omp_set_num_threads(4);
        params.d_hat = 0.005;
        params.use_parallel = true;
    }
};

class CollisionColoredCCDInitialGuessNew : public CollisionColoredCCDInitialGuess {};

class CollisionColoredCCDInitialGuessDriver
    : public CollisionColoredCCDInitialGuess, public testing::WithParamInterface<bool> {
protected:
    void select_guess(SimParams& scene_params) const {
        scene_params.use_colored_ccd_guess = !GetParam();
        scene_params.use_colored_ccd_guess_level_based = GetParam();
    }

    std::vector<Vec3> expected_guess(const std::vector<Vec3>& x,
        const std::vector<Vec3>& displacement, const RefMesh& mesh,
        const SimParams& scene_params, int sweeps) const {
        return GetParam()
            ? collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, scene_params, sweeps)
            : collision_colored_ccd_initial_guess(
                x, displacement, mesh, scene_params, sweeps);
    }
};

void expect_positions_identical(
    const std::vector<Vec3>& actual, const std::vector<Vec3>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t vertex = 0; vertex < actual.size(); ++vertex) {
        SCOPED_TRACE(vertex);
        EXPECT_EQ(std::memcmp(actual[vertex].data(), expected[vertex].data(),
                             3 * sizeof(double)), 0);
    }
}

// The fraction's physical target is an algebraic oracle, not a promise about
// contraction or intermediate rounding. Keep integration/reference equality
// bitwise separately, and use only a few roundoff units for this formula.
void expect_fraction_target_near(const std::vector<Vec3>& actual,
    const std::vector<Vec3>& target, double& maximum_absolute_error) {
    ASSERT_EQ(actual.size(), target.size());
    for (std::size_t node = 0; node < actual.size(); ++node) {
        for (int axis = 0; axis < 3; ++axis) {
            const double value = actual[node][axis], expected = target[node][axis];
            const double error = std::abs(value - expected);
            maximum_absolute_error = std::max(maximum_absolute_error, error);
            const double scale = std::max({1.0, std::abs(value), std::abs(expected)});
            EXPECT_LE(error, 8.0 * std::numeric_limits<double>::epsilon() * scale)
                << "node=" << node << " axis=" << axis << std::setprecision(17)
                << " actual=" << value << " expected=" << expected << " absdiff=" << error
                << std::hexfloat << " actual_hex=" << value << " expected_hex=" << expected;
        }
    }
}

void expect_fraction_reference_identical(const std::vector<Vec3>& actual,
    const std::vector<Vec3>& reference) {
    ASSERT_EQ(actual.size(), reference.size());
    for (std::size_t node = 0; node < actual.size(); ++node) {
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_EQ(std::memcmp(actual[node].data() + axis,
                reference[node].data() + axis, sizeof(double)), 0)
                << "node=" << node << " axis=" << axis << std::setprecision(17)
                << " actual=" << actual[node][axis] << " expected=" << reference[node][axis]
                << " absdiff=" << std::abs(actual[node][axis] - reference[node][axis])
                << std::hexfloat << " actual_hex=" << actual[node][axis]
                << " expected_hex=" << reference[node][axis];
        }
    }
}

std::vector<Vec3> point_above_triangle() {
    return {Vec3(0, 0, 1), Vec3(-1, -1, 0),
            Vec3(1, -1, 0), Vec3(0, 1, 0)};
}

RefMesh point_triangle_mesh() {
    RefMesh mesh = ref_mesh_with_masses({1, 1, 1, 1});
    mesh.tris = {1, 2, 3};
    return mesh;
}

struct UnevenContactGuessScene {
    RefMesh mesh;
    std::vector<Vec3> x, displacement;

    explicit UnevenContactGuessScene(int copies = 32) {
        // Every moving point has its own component. Alternate one and two
        // crossed triangles so the points share a level/color but not a cost.
        for (int copy = 0; copy < copies; ++copy) {
            const int base = static_cast<int>(x.size());
            const Vec3 offset(12 * copy, 0, 0);
            for (const auto& point : point_above_triangle()) x.push_back(point + offset);
            mesh.tris.insert(mesh.tris.end(), {base + 1, base + 2, base + 3});
            displacement.insert(displacement.end(), {Vec3(0, 0, -2),
                Vec3::Zero(), Vec3::Zero(), Vec3::Zero()});
            if (copy % 2) {
                const int triangle = static_cast<int>(x.size());
                x.insert(x.end(), {Vec3(-1, -1, -0.5) + offset,
                    Vec3(1, -1, -0.5) + offset, Vec3(0, 1, -0.5) + offset});
                mesh.tris.insert(mesh.tris.end(), {triangle, triangle + 1, triangle + 2});
                displacement.insert(displacement.end(), 3, Vec3::Zero());
            }
        }
        mesh.num_positions = x.size();
        mesh.mass.assign(x.size(), 1.0);
    }
};

struct BfsGuessReference {
    std::vector<Vec3> first, result;
    std::vector<int> levels, seeds;
};

// Independent serial replay using only existing public contact/CCD helpers.
// Exact mode lets safe_step supply the gap guard without exposing predictor
// internals. A discarded legacy sweep supplies the movement and contact graph;
// real sweeps restart at x using levels computed once and never refreshed.
// retain_probe is only a negative-control path for regression tests.
BfsGuessReference replay_bfs_guess(const std::vector<Vec3>& x,
    const std::vector<Vec3>& displacement, const RefMesh& mesh,
    SimParams params, int sweeps, bool retain_probe = false) {
    params.use_parallel = false;
    params.exact_computation_fallback = true;
    BroadPhase scratch;
    BfsGuessReference reference;
    reference.first = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 1, &scratch);
    reference.result = retain_probe ? reference.first : x;
    const int count = static_cast<int>(x.size());
    const auto rigid = [&](int node) {
        return !mesh.node_to_rb.empty() && mesh.node_to_rb[node] >= 0;
    };
    std::vector<Vec3> targets = x;
    for (int node = 0; node < count; ++node)
        if (!rigid(node)) targets[node] += displacement[node];
    std::vector<std::vector<int>> lower_edges, colors, graph(count);
    build_contact_adj(scratch.cache(), count, lower_edges);
    greedy_color_conflict_graph(lower_edges, colors);
    for (int node = 0; node < count; ++node) {
        for (const int neighbor : lower_edges[node]) {
            graph[node].push_back(neighbor);
            graph[neighbor].push_back(node);
        }
    }
    reference.levels.assign(count, -1);
    std::vector<unsigned char> visited(count, 0);
    for (int start = 0; start < count; ++start) {
        if (visited[start]) continue;
        std::vector<int> component{start};
        visited[start] = 1;
        for (std::size_t head = 0; head < component.size(); ++head) {
            for (const int neighbor : graph[component[head]]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    component.push_back(neighbor);
                }
            }
        }
        double maximum = -1.0;
        for (const int node : component)
            if (!rigid(node)) maximum = std::max(maximum,
                (reference.first[node] - x[node]).stableNorm());
        std::vector<int> queue;
        for (const int node : component) {
            if (!rigid(node)
                && (reference.first[node] - x[node]).stableNorm() == maximum) {
                reference.seeds.push_back(node);
                reference.levels[node] = 0;
                queue.push_back(node);
            }
        }
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const int node = queue[head];
            for (const int neighbor : graph[node]) {
                if (reference.levels[neighbor] == -1) {
                    reference.levels[neighbor] = reference.levels[node] + 1;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    const int max_level = *std::max_element(reference.levels.begin(), reference.levels.end());
    for (int iteration = 0; iteration < sweeps; ++iteration) {
        bool moved = false, remaining = false;
        for (int level = 0; level <= max_level; ++level) {
            for (const auto& color : colors) {
                for (const int node : color) {
                    if (rigid(node) || reference.levels[node] != level) continue;
                    const Vec3 before = reference.result[node];
                    per_vertex_safe_step(scratch, reference.result, node, targets[node],
                        0.9, true, false, false, false, nullptr, true);
                    moved |= (reference.result[node].array() != before.array()).any();
                    remaining |= !((targets[node] - reference.result[node]).squaredNorm() < 1e-28);
                }
            }
        }
        if (!moved || !remaining) break;
    }
    return reference;
}

void expect_trace_consistent(const ColoredCCDGuessTrace& trace,
    const std::vector<Vec3>& initial, const std::vector<Vec3>& displacement,
    const std::vector<Vec3>& result, int cap, bool bfs) {
    using Stage = ColoredCCDGuessTraceStage;
    const int count = static_cast<int>(initial.size());
    ASSERT_EQ(trace.targets.size(), initial.size());
    ASSERT_EQ(trace.color_ids.size(), initial.size());
    ASSERT_EQ(trace.level_ids.size(), initial.size());
    ASSERT_EQ(trace.first_sweep_movement.size(), initial.size());
    EXPECT_EQ(trace.probe_completed, bfs && cap > 0 && !initial.empty());
    for (int node = 0; node < count; ++node)
        expect_positions_identical({trace.targets[node]}, {initial[node] + displacement[node]});
    ASSERT_FALSE(trace.steps.empty());
    const auto& first = trace.steps.front();
    EXPECT_EQ(first.stage, Stage::Initial);
    EXPECT_EQ(first.iteration, 0);
    EXPECT_EQ(first.active_level, -1);
    EXPECT_EQ(first.active_color, -1);
    EXPECT_TRUE(first.active_vertices.empty());
    expect_positions_identical(first.positions, initial);
    int completed = 0, levels_built = 0, previous_level = -1, previous_color = -1;
    std::vector<int> visits(count, 0);
    for (std::size_t index = 1; index < trace.steps.size(); ++index) {
        SCOPED_TRACE(index);
        const auto& step = trace.steps[index];
        const auto& previous = trace.steps[index - 1].positions;
        ASSERT_EQ(step.positions.size(), initial.size());
        for (const auto& position : step.positions) EXPECT_TRUE(position.allFinite());
        if (step.stage == Stage::ColorUpdate) {
            EXPECT_EQ(step.iteration, completed + 1);
            EXPECT_GE(step.active_color, 0);
            if (!bfs) {
                EXPECT_EQ(levels_built, 0);
                EXPECT_EQ(step.active_level, -1);
                EXPECT_GT(step.active_color, previous_color);
            } else {
                EXPECT_EQ(levels_built, 1);
                EXPECT_GE(step.active_level, 0);
                EXPECT_TRUE(step.active_level > previous_level
                    || (step.active_level == previous_level && step.active_color > previous_color));
            }
            previous_level = step.active_level;
            previous_color = step.active_color;
            std::vector<int> expected, active = step.active_vertices;
            for (int node = 0; node < count; ++node) {
                if (trace.color_ids[node] == step.active_color
                    && (step.active_level < 0 || trace.level_ids[node] == step.active_level))
                    expected.push_back(node);
            }
            std::sort(active.begin(), active.end());
            EXPECT_EQ(active, expected);
            for (int node = 0; node < count; ++node) {
                if (std::binary_search(active.begin(), active.end(), node)) ++visits[node];
                else expect_positions_identical({step.positions[node]}, {previous[node]});
            }
        } else {
            EXPECT_EQ(step.active_level, -1);
            EXPECT_EQ(step.active_color, -1);
            EXPECT_TRUE(step.active_vertices.empty());
            expect_positions_identical(step.positions, previous);
            if (step.stage == Stage::SweepEnd) {
                EXPECT_EQ(step.iteration, ++completed);
                for (const int visited : visits) EXPECT_EQ(visited, 1);
                std::fill(visits.begin(), visits.end(), 0);
                previous_level = previous_color = -1;
                if (completed == 1 && !trace.probe_completed) {
                    for (int node = 0; node < count; ++node)
                        EXPECT_DOUBLE_EQ(trace.first_sweep_movement[node],
                            (step.positions[node] - initial[node]).stableNorm());
                }
            } else {
                EXPECT_EQ(step.stage, Stage::LevelsBuilt);
                EXPECT_TRUE(bfs);
                EXPECT_EQ(++levels_built, 1);
                EXPECT_EQ(completed, 0);
                EXPECT_EQ(step.iteration, 0);
                expect_positions_identical(step.positions, initial);
                for (const int level : trace.level_ids) EXPECT_GE(level, 0);
            }
        }
    }
    EXPECT_EQ(trace.completed_iterations, completed);
    EXPECT_LE(completed, cap);
    for (const int visited : visits) EXPECT_EQ(visited, 0);
    if (!levels_built)
        for (const int level : trace.level_ids) EXPECT_EQ(level, -1);
    bool reached = true;
    for (int node = 0; node < count; ++node)
        reached &= (trace.targets[node] - result[node]).squaredNorm() < 1e-28;
    EXPECT_EQ(trace.reached_targets, reached);
    expect_positions_identical(trace.steps.back().positions, result);
}

void expect_traces_identical(const ColoredCCDGuessTrace& actual,
    const ColoredCCDGuessTrace& expected) {
    expect_positions_identical(actual.targets, expected.targets);
    EXPECT_EQ(actual.color_ids, expected.color_ids);
    EXPECT_EQ(actual.level_ids, expected.level_ids);
    EXPECT_EQ(actual.first_sweep_movement, expected.first_sweep_movement);
    EXPECT_EQ(actual.probe_completed, expected.probe_completed);
    EXPECT_EQ(actual.completed_iterations, expected.completed_iterations);
    EXPECT_EQ(actual.reached_targets, expected.reached_targets);
    EXPECT_EQ(actual.stalled, expected.stalled);
    ASSERT_EQ(actual.steps.size(), expected.steps.size());
    for (std::size_t index = 0; index < actual.steps.size(); ++index) {
        SCOPED_TRACE(index);
        const auto& a = actual.steps[index];
        const auto& e = expected.steps[index];
        EXPECT_EQ(a.stage, e.stage);
        EXPECT_EQ(a.iteration, e.iteration);
        EXPECT_EQ(a.active_level, e.active_level);
        EXPECT_EQ(a.active_color, e.active_color);
        EXPECT_EQ(a.active_vertices, e.active_vertices);
        expect_positions_identical(a.positions, e.positions);
    }
}

std::vector<Vec3> expect_exact_level_schedules_identical(
    const std::vector<Vec3>& x, const std::vector<Vec3>& displacement,
    const RefMesh& mesh, SimParams params, int cap, int expected_point_group = 0) {
    params.exact_computation_fallback = true;
    params.use_parallel = false;
    const auto reference = replay_bfs_guess(x, displacement, mesh, params, cap);
    ColoredCCDGuessTrace trace;
    const auto baseline = collision_colored_ccd_initial_guess_level_based(
        x, displacement, mesh, params, cap, nullptr, &trace);
    expect_positions_identical(baseline, reference.result);
    EXPECT_EQ(trace.level_ids, reference.levels);
    expect_trace_consistent(trace, x, displacement, baseline, cap, true);
    if (expected_point_group > 0) {
        const auto group = std::find_if(trace.steps.begin(), trace.steps.end(),
            [&](const ColoredCCDGuessTraceStep& step) {
                return step.stage == ColoredCCDGuessTraceStage::ColorUpdate
                    && step.iteration == 1 && step.active_level == 0
                    && step.active_color == trace.color_ids[0];
            });
        EXPECT_NE(group, trace.steps.end());
        if (group != trace.steps.end())
            EXPECT_EQ(group->active_vertices.size(), expected_point_group);
    }

    std::vector<Vec3> actual;
    BroadPhase scratch;
    for (const int workers : {0, 1, 4}) {
        SCOPED_TRACE(::testing::Message() << "workers=" << workers);
        params.use_parallel = workers != 0;
        omp_set_num_threads(std::max(1, workers));
        for (int repeat = 0; repeat < 2; ++repeat) {
            SCOPED_TRACE(repeat);
            // A trace keeps the reference schedule. Explicitly pass nullptr
            // here so exact mode exercises the production scheduling path.
            actual = collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, params, cap, &scratch, nullptr);
            expect_positions_identical(actual, baseline);
        }
    }
    return actual;
}

bool parse_initial_guess_arguments(IPCArgs3D& args, std::vector<std::string> words) {
    std::vector<char*> argv;
    for (auto& word : words) argv.push_back(word.data());
    return args.parse(static_cast<int>(argv.size()), argv.data());
}

struct InitialGuessClothScene {
    RefMesh mesh = point_triangle_mesh();
    DeformedState state;
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;
    BroadPhase broad_phase;
    SimParams params = base_params();

    InitialGuessClothScene() {
        state.deformed_positions = point_above_triangle();
        mesh.initialize(state.deformed_positions);
        adjacency = build_incident_triangle_map(mesh.tris);
        state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
        // With no solver iterations, advance_one_frame exposes exactly the
        // chosen initial guess and the standard velocity update afterward.
        params.fixed_iters = true;
        params.max_global_iters = 0;
        params.node_box_min = 0.001;
        params.node_box_max = 0.01;
        params.d_hat = 0.005;
        params.use_colored_ccd_guess = true;
    }

    SolverResult advance(SubstepCallback callback = nullptr,
                         PinTargetUpdater updater = nullptr) {
        return advance_one_frame(state, mesh, adjacency, pins, params,
            broad_phase, 1, updater, callback);
    }
};

struct InitialGuessSolidScene {
    RefMesh mesh;
    DeformedState state;
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;
    BroadPhase broad_phase;
    SimParams params = base_params();

    explicit InitialGuessSolidScene(bool with_rigid = false) {
        // Node 4 is strictly interior, and is not a collision surface node.
        create_solid({Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0),
                         Vec3(0, 0, 1), Vec3(0.2, 0.2, 0.2)},
            {4, 1, 2, 3, 0, 4, 2, 3, 0, 1, 4, 3, 0, 1, 2, 4},
            24.0, mesh, state);
        if (with_rigid) {
            mesh.tris.insert(mesh.tris.end(), {5, 6, 7});
            create_rigid_body({Vec3(8, -1, 0), Vec3(10, -1, 0), Vec3(9, 2, 0)},
                Vec3(0.5, 0, 0), Vec4(1, 0, 0, 0), Vec3::Zero(),
                3.0, mesh, state);
            sync_rigid_body_particles(mesh, state);
        }
        mesh.build_deformable_nodes();
        adjacency = build_incident_triangle_map(mesh.tris);
        params.fixed_iters = true;
        params.max_global_iters = 0;
        params.node_box_min = 0.001;
        params.node_box_max = 0.01;
        params.d_hat = 0.005;
        params.use_colored_ccd_guess = true;
    }

    SolverResult advance(SubstepCallback callback = nullptr,
                         PinTargetUpdater updater = nullptr) {
        return advance_one_frame_general(state, mesh, adjacency, pins, params,
            broad_phase, 1, updater, callback);
    }
};

} // namespace

INSTANTIATE_TEST_SUITE_P(OriginalAndLevelBased, CollisionColoredCCDInitialGuessDriver,
    testing::Bool());

TEST(ColoredCCDGuessParameters, DefaultsAndCliExposeAnOptInIterationCount) {
    const auto zero = SimParams::zeros();
    EXPECT_FALSE(zero.use_colored_ccd_guess);
    EXPECT_FALSE(zero.use_colored_ccd_guess_level_based);
    EXPECT_EQ(zero.colored_ccd_guess_iters, 10);
    IPCArgs3D defaults;
    EXPECT_FALSE(defaults.to_sim_params().use_colored_ccd_guess);
    EXPECT_FALSE(defaults.to_sim_params().use_colored_ccd_guess_level_based);
    EXPECT_EQ(defaults.to_sim_params().colored_ccd_guess_iters, 10);

    for (bool level_based : {false, true}) {
        const std::string option = level_based
            ? "--use_colored_ccd_guess_level_based" : "--use_colored_ccd_guess";
        SCOPED_TRACE(option);
        for (int sweeps : {0, 3}) {
            IPCArgs3D selected;
            std::vector<std::string> words = {"3D_sim", option};
            if (sweeps != 0) words.push_back("true");
            words.insert(words.end(), {"--colored_ccd_guess_iters", std::to_string(sweeps)});
            ASSERT_TRUE(parse_initial_guess_arguments(selected, words));
            const auto params = selected.to_sim_params();
            EXPECT_EQ(params.use_colored_ccd_guess, !level_based);
            EXPECT_EQ(params.use_colored_ccd_guess_level_based, level_based);
            EXPECT_EQ(params.colored_ccd_guess_iters, sweeps);
            EXPECT_TRUE(params.use_ccd_guess);
        }
        IPCArgs3D disabled;
        ASSERT_TRUE(parse_initial_guess_arguments(disabled, {"3D_sim", option, "false"}));
        EXPECT_FALSE(disabled.to_sim_params().use_colored_ccd_guess);
        EXPECT_FALSE(disabled.to_sim_params().use_colored_ccd_guess_level_based);

        IPCArgs3D invalid;
        ASSERT_TRUE(parse_initial_guess_arguments(invalid, {"3D_sim", option,
            "true", "--colored_ccd_guess_iters", "-1"}));
        EXPECT_THROW(invalid.to_sim_params(), std::invalid_argument);
    }

    IPCArgs3D conflicting;
    ASSERT_TRUE(parse_initial_guess_arguments(conflicting, {"3D_sim",
        "--use_colored_ccd_guess", "true", "--use_colored_ccd_guess_level_based", "true"}));
    EXPECT_THROW(conflicting.to_sim_params(), std::invalid_argument);
}

TEST(ColoredCCDGuessParameters, ValidatesEnabledSettingsWithoutChangingLegacyContract) {
    for (bool level_based : {false, true}) {
        SCOPED_TRACE(level_based);
        auto params = SimParams::zeros();
        params.use_colored_ccd_guess = !level_based;
        params.use_colored_ccd_guess_level_based = level_based;
        EXPECT_NO_THROW(params.validate_colored_ccd_guess_parameters());
        params.colored_ccd_guess_iters = 0;
        EXPECT_NO_THROW(params.validate_colored_ccd_guess_parameters());
        params.colored_ccd_guess_iters = -1;
        EXPECT_THROW(params.validate_colored_ccd_guess_parameters(), std::invalid_argument);
        params.colored_ccd_guess_iters = 2;
        for (double invalid_distance : {-0.001, std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::quiet_NaN()}) {
            params.d_hat = invalid_distance;
            EXPECT_THROW(params.validate_colored_ccd_guess_parameters(), std::invalid_argument);
        }
        params.use_colored_ccd_guess = false;
        params.use_colored_ccd_guess_level_based = false;
        params.colored_ccd_guess_iters = -1;
        EXPECT_NO_THROW(params.validate_colored_ccd_guess_parameters());
    }
    auto conflicting = SimParams::zeros();
    conflicting.use_colored_ccd_guess = true;
    conflicting.use_colored_ccd_guess_level_based = true;
    EXPECT_THROW(conflicting.validate_colored_ccd_guess_parameters(), std::invalid_argument);
}

TEST(ColoredCCDGuessParameters, FractionDefaultsAndCliAcceptUnitInterval) {
    EXPECT_DOUBLE_EQ(SimParams::zeros().colored_ccd_guess_fraction, 1.0);
    IPCArgs3D defaults;
    EXPECT_DOUBLE_EQ(defaults.to_sim_params().colored_ccd_guess_fraction, 1.0);
    for (const std::string option : {"--use_colored_ccd_guess",
             "--use_colored_ccd_guess_level_based"}) {
        for (const std::string fraction : {"0", "0.9", "1"}) {
            SCOPED_TRACE(option + " fraction=" + fraction);
            IPCArgs3D args;
            ASSERT_TRUE(parse_initial_guess_arguments(args, {"3D_sim", option, "true",
                "--colored_ccd_guess_fraction", fraction}));
            EXPECT_DOUBLE_EQ(args.to_sim_params().colored_ccd_guess_fraction,
                std::stod(fraction));
        }
    }
    IPCArgs3D disabled;
    ASSERT_TRUE(parse_initial_guess_arguments(disabled,
        {"3D_sim", "--colored_ccd_guess_fraction", "0.9"}));
    EXPECT_DOUBLE_EQ(disabled.to_sim_params().colored_ccd_guess_fraction, 0.9);
    EXPECT_FALSE(disabled.to_sim_params().use_colored_ccd_guess);
    EXPECT_FALSE(disabled.to_sim_params().use_colored_ccd_guess_level_based);
}

TEST(ColoredCCDGuessParameters, FractionRejectsInvalidValuesEvenWhenDisabled) {
    for (int mode : {0, 1, 2}) {
        auto params = SimParams::zeros();
        params.use_colored_ccd_guess = mode == 1;
        params.use_colored_ccd_guess_level_based = mode == 2;
        for (double invalid : {-0.01, std::nextafter(1.0, 2.0),
                 std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::quiet_NaN()}) {
            SCOPED_TRACE(::testing::Message() << "mode=" << mode << " fraction=" << invalid);
            params.colored_ccd_guess_fraction = invalid;
            EXPECT_THROW(params.validate_colored_ccd_guess_parameters(), std::invalid_argument);
        }
        for (double valid : {0.0, 0.9, 1.0}) {
            params.colored_ccd_guess_fraction = valid;
            EXPECT_NO_THROW(params.validate_colored_ccd_guess_parameters());
        }
    }
    for (const std::string invalid : {"-0.01", "1.01", "nan", "inf", "-inf"}) {
        IPCArgs3D args;
        SCOPED_TRACE(invalid);
        // A numeric parser may reject nonfinite literals before SimParams.
        // If it accepts them, conversion must still reject the value.
        if (parse_initial_guess_arguments(args,
                {"3D_sim", "--colored_ccd_guess_fraction", invalid}))
            EXPECT_THROW(args.to_sim_params(), std::invalid_argument);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, ParallelSettingPreservesResults) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    const std::vector<Vec3> displacement(x.size(), Vec3(0.25, 0, 0));
    params.use_parallel = false;
    const auto serial = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 2);
    params.use_parallel = true;
    const auto parallel = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 2);
    expect_positions_identical(parallel, serial);
    for (std::size_t vertex = 0; vertex < x.size(); ++vertex)
        expect_vec_near(serial[vertex], x[vertex] + displacement[vertex], 0.0);
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothInitialGuessUsesXhatWithoutAddingGravity) {
    InitialGuessClothScene scene;
    select_guess(scene.params);
    scene.params.gravity = Vec3(0, -9.81, 0);
    scene.params.use_verlet_guess = true;
    scene.params.use_ccd_guess = true;
    for (auto& velocity : scene.state.velocities) velocity = Vec3(0.25, 0, 0);
    std::vector<Vec3> expected;
    build_xhat(expected, scene.state.deformed_positions, scene.state.velocities,
        scene.params.dt());
    ASSERT_TRUE(scene.advance().converged);
    expect_positions_identical(scene.state.deformed_positions, expected);
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothFractionScalesTargetOnceAcrossSweepBudgets) {
    for (bool parallel : {false, true}) {
        for (int sweeps : {1, 2, 6}) {
            SCOPED_TRACE(::testing::Message() << "parallel=" << parallel << " sweeps=" << sweeps);
            InitialGuessClothScene scene;
            select_guess(scene.params);
            scene.params.use_parallel = parallel;
            scene.params.colored_ccd_guess_fraction = 0.9;
            scene.params.colored_ccd_guess_iters = sweeps;
            scene.params.gravity = Vec3(0, -9.81, 0);
            scene.state.velocities.assign(4, Vec3(0.25, 0.125, 0));
            const auto before = scene.state.deformed_positions;
            std::vector<Vec3> xhat, expected = before;
            build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
            for (std::size_t node = 0; node < before.size(); ++node) {
                Vec3 displacement = xhat[node] - before[node];
                displacement *= 0.9;
                expected[node] += displacement;
            }
            ASSERT_TRUE(scene.advance().converged);
            // No collision restricts this translation. More sweeps must not
            // multiply by 0.9 again or continue toward the unscaled xhat.
            expect_positions_identical(scene.state.deformed_positions, expected);
            EXPECT_GT((xhat[0] - expected[0]).norm(), 0.001);
            for (std::size_t node = 0; node < before.size(); ++node)
                expect_vec_near(scene.state.velocities[node],
                    (expected[node] - before[node]) / scene.params.dt(), 0.0);
        }
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothFractionZeroAndOnePreserveEndpoints) {
    for (double fraction : {0.0, 1.0}) {
        for (int sweeps : {0, 1, 6}) {
            SCOPED_TRACE(::testing::Message() << "fraction=" << fraction << " sweeps=" << sweeps);
            InitialGuessClothScene scene;
            select_guess(scene.params);
            scene.params.colored_ccd_guess_fraction = fraction;
            scene.params.colored_ccd_guess_iters = sweeps;
            scene.state.velocities[0] = Vec3(0, 0, -4);
            const auto before = scene.state.deformed_positions;
            std::vector<Vec3> xhat, displacement(before.size());
            build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
            for (std::size_t node = 0; node < before.size(); ++node)
                displacement[node] = xhat[node] - before[node];
            // Direct helpers retain their original contract: displacement is
            // already prepared by the integration callsite, not rescaled.
            auto default_params = scene.params;
            default_params.colored_ccd_guess_fraction = 1.0;
            const auto expected = fraction == 0.0 ? before
                : expected_guess(before, displacement, scene.mesh, default_params, sweeps);
            ASSERT_TRUE(scene.advance().converged);
            expect_positions_identical(scene.state.deformed_positions, expected);
        }
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothFractionPreservesFullInertialTargetForSolver) {
    for (bool v2 : {false, true}) {
        SCOPED_TRACE(v2);
        InitialGuessClothScene scene;
        select_guess(scene.params);
        scene.params.colored_ccd_guess_fraction = 0.9;
        scene.params.colored_ccd_guess_iters = 2;
        scene.params.max_global_iters = 1;
        scene.params.use_basic_experimental_v2 = v2;
        scene.params.use_simd = v2;
        scene.params.use_parallel = true;
        scene.state.velocities.assign(4, Vec3(0.004, 0, 0));
        const auto before = scene.state.deformed_positions;
        std::vector<Vec3> xhat, displacement(before.size());
        build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
        const auto original_xhat = xhat;
        for (std::size_t node = 0; node < before.size(); ++node) {
            displacement[node] = xhat[node] - before[node];
            displacement[node] *= 0.9;
        }
        BroadPhase reference_broad_phase;
        const auto predictor = GetParam() ? collision_colored_ccd_initial_guess_level_based
                                         : collision_colored_ccd_initial_guess;
        auto expected = predictor(before, displacement, scene.mesh, scene.params, 2,
            &reference_broad_phase, nullptr);
        const auto reduced_start = expected;
        const auto solver = v2 ? global_gauss_seidel_solver_basic_experimental_v2
                               : global_gauss_seidel_solver_basic;
        ASSERT_TRUE(solver(scene.mesh, scene.adjacency, scene.pins, scene.params,
            expected, xhat, scene.state.velocities, reference_broad_phase, "", &before).converged);
        // The inertial objective still pulls toward full xhat, not the reduced
        // predictor target. A driver that rescales xhat would miss this update.
        EXPECT_GT((expected[1] - reduced_start[1]).norm(), 1e-5);
        expect_positions_identical(xhat, original_xhat);
        ASSERT_TRUE(scene.advance().converged);
        expect_positions_identical(scene.state.deformed_positions, expected);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothFractionRebuildsEachLiveSubstepTarget) {
    InitialGuessClothScene scene;
    select_guess(scene.params);
    scene.params.substeps = 3;
    scene.params.colored_ccd_guess_fraction = 0.9;
    scene.params.colored_ccd_guess_iters = 6;
    scene.state.velocities.assign(4, Vec3(0.25, 0.125, 0));
    DeformedState expected = scene.state;
    int observed = 0;
    double maximum_target_error = 0.0;
    ASSERT_TRUE(scene.advance([&](int global_sub, const std::vector<Vec3>& actual) {
        EXPECT_EQ(global_sub, observed++);
        const auto previous = expected.deformed_positions;
        std::vector<Vec3> xhat, displacement(previous.size()), analytic_target = previous;
        build_xhat(xhat, previous, expected.velocities, scene.params.dt());
        for (std::size_t node = 0; node < previous.size(); ++node) {
            displacement[node] = xhat[node] - previous[node];
            displacement[node] *= 0.9;
            analytic_target[node] += displacement[node];
        }
        auto reference_params = scene.params;
        reference_params.colored_ccd_guess_fraction = 1.0;
        expected.deformed_positions = expected_guess(previous, displacement,
            scene.mesh, reference_params, scene.params.colored_ccd_guess_iters);
        update_velocity(expected.velocities, expected.deformed_positions, previous,
            scene.params.dt());
        expect_fraction_reference_identical(actual, expected.deformed_positions);
        expect_fraction_target_near(actual, analytic_target, maximum_target_error);
    }).converged);
    EXPECT_EQ(observed, 3);
    expect_fraction_reference_identical(scene.state.velocities, expected.velocities);
    std::ostringstream error_property;
    error_property << std::setprecision(17) << maximum_target_error;
    RecordProperty("max_analytic_target_abs_error", error_property.str());
}

TEST_P(CollisionColoredCCDInitialGuessDriver, FractionValidationPrecedesStateMutation) {
    for (double invalid : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN()}) {
        InitialGuessClothScene cloth;
        InitialGuessSolidScene mixed(true);
        select_guess(cloth.params);
        select_guess(mixed.params);
        cloth.params.colored_ccd_guess_fraction = invalid;
        mixed.params.colored_ccd_guess_fraction = invalid;
        const auto cloth_before = cloth.state.deformed_positions;
        const auto cloth_velocity = cloth.state.velocities;
        const auto mixed_before = mixed.state.deformed_positions;
        const auto mixed_velocity = mixed.state.velocities;
        const auto mixed_com = mixed.state.x_coms;
        int pin_updates = 0, callbacks = 0;
        const auto updater = [&](std::vector<Pin>&, double) { ++pin_updates; };
        const auto callback = [&](int, const std::vector<Vec3>&) { ++callbacks; };
        EXPECT_THROW(cloth.advance(callback, updater), std::invalid_argument);
        EXPECT_THROW(mixed.advance(callback, updater), std::invalid_argument);
        EXPECT_EQ(pin_updates, 0);
        EXPECT_EQ(callbacks, 0);
        expect_positions_identical(cloth.state.deformed_positions, cloth_before);
        expect_positions_identical(cloth.state.velocities, cloth_velocity);
        expect_positions_identical(mixed.state.deformed_positions, mixed_before);
        expect_positions_identical(mixed.state.velocities, mixed_velocity);
        expect_positions_identical(mixed.state.x_coms, mixed_com);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothInitialGuessUsesSelectedSweepsAndTakesPrecedence) {
    for (bool parallel : {false, true}) {
        for (int sweeps : {0, 1, 2, 3}) {
            SCOPED_TRACE(::testing::Message() << "parallel=" << parallel << " sweeps=" << sweeps);
            InitialGuessClothScene scene;
            select_guess(scene.params);
            scene.params.use_parallel = parallel;
            scene.params.colored_ccd_guess_iters = sweeps;
            // All legacy options remain on deliberately: the opt-in colored
            // guess must win, including the meaningful zero-sweep case.
            scene.params.use_ccd_guess = true;
            scene.params.use_verlet_guess = true;
            scene.params.use_translation_guess = true;
            scene.params.gravity = Vec3(0, -9.81, 0);
            scene.state.velocities[0] = Vec3(0, 0, -4);
            for (std::size_t i = 1; i < 4; ++i)
                scene.state.velocities[i] = Vec3(8, 0, 0);
            const auto before = scene.state.deformed_positions;
            std::vector<Vec3> xhat, displacement(before.size());
            build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
            for (std::size_t i = 0; i < before.size(); ++i)
                displacement[i] = xhat[i] - before[i];
            const auto expected = expected_guess(
                before, displacement, scene.mesh, scene.params, sweeps);
            if (sweeps == 1) {
                const auto original = collision_colored_ccd_initial_guess(
                    before, displacement, scene.mesh, scene.params, 1);
                const auto level_based = collision_colored_ccd_initial_guess_level_based(
                    before, displacement, scene.mesh, scene.params, 1);
                ASSERT_GT(original[0].z(), level_based[0].z() + 0.5);
            }
            ASSERT_TRUE(scene.advance().converged);
            expect_positions_identical(scene.state.deformed_positions, expected);
            for (std::size_t i = 0; i < before.size(); ++i)
                expect_vec_near(scene.state.velocities[i],
                    (expected[i] - before[i]) / scene.params.dt(), 0.0);
        }
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothInitialGuessRebuildsTargetsFromEachLiveSubstep) {
    InitialGuessClothScene scene;
    select_guess(scene.params);
    scene.params.substeps = 3;
    scene.params.colored_ccd_guess_iters = 1;
    scene.state.velocities[0] = Vec3(0, 0, -12);
    for (std::size_t i = 1; i < 4; ++i) scene.state.velocities[i] = Vec3(0, 0, -6);
    DeformedState expected = scene.state;
    int observed = 0;
    ASSERT_TRUE(scene.advance([&](int global_sub, const std::vector<Vec3>& actual) {
        EXPECT_EQ(global_sub, observed);
        ++observed;
        const auto previous = expected.deformed_positions;
        std::vector<Vec3> xhat, displacement(previous.size());
        build_xhat(xhat, previous, expected.velocities, scene.params.dt());
        for (std::size_t i = 0; i < previous.size(); ++i)
            displacement[i] = xhat[i] - previous[i];
        expected.deformed_positions = expected_guess(
            previous, displacement, scene.mesh, scene.params, 1);
        update_velocity(expected.velocities, expected.deformed_positions, previous,
            scene.params.dt());
        expect_positions_identical(actual, expected.deformed_positions);
    }).converged);
    EXPECT_EQ(observed, 3);
    expect_positions_identical(scene.state.velocities, expected.velocities);
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothSweepBudgetCountsOnlyCommittedSweeps) {
    for (int sweeps : {0, 1, 2, 3}) {
        SCOPED_TRACE(sweeps);
        InitialGuessClothScene scene;
        select_guess(scene.params);
        scene.params.colored_ccd_guess_iters = sweeps;
        scene.state.velocities[0] = Vec3(0, 0, -4);
        ASSERT_TRUE(scene.advance().converged);
        // The level-based trial pass must neither consume the budget nor
        // retain its motion: each committed sweep removes 90% of the gap.
        expect_vec_near(scene.state.deformed_positions[0],
            Vec3(0, 0, std::pow(0.1, sweeps)), 1e-12);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, DisablingClothOptionPreservesLegacyGuessSelection) {
    for (int mode : {0, 1, 2, 3}) {
        SCOPED_TRACE(mode);
        InitialGuessClothScene scene;
        scene.params.use_colored_ccd_guess = false;
        scene.params.colored_ccd_guess_iters = -1;
        scene.params.use_ccd_guess = mode == 1;
        scene.params.use_verlet_guess = mode == 2;
        scene.params.use_translation_guess = mode == 3;
        scene.params.gravity = Vec3(0, -9.81, 0);
        for (auto& velocity : scene.state.velocities) velocity = Vec3(0.25, 0, 0);
        const auto before = scene.state.deformed_positions;
        std::vector<Vec3> xhat;
        build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
        std::vector<Vec3> expected = before;
        if (mode == 1) expected = ccd_initial_guess(before, xhat, scene.mesh);
        if (mode == 2) expected = verlet_initial_guess(before, xhat, scene.mesh, scene.params);
        if (mode == 3) expected = translation_initial_guess(before, xhat,
            scene.mesh, scene.pins, scene.params);
        ASSERT_TRUE(scene.advance().converged);
        expect_positions_identical(scene.state.deformed_positions, expected);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, OgcRetainsPrecedenceOverTheClothOption) {
    for (bool solver_ogc : {false, true}) {
        SCOPED_TRACE(solver_ogc);
        InitialGuessClothScene scene;
        select_guess(scene.params);
        scene.params.use_ogc = !solver_ogc;
        scene.params.use_ogc_solver = solver_ogc;
        scene.state.velocities.assign(4, Vec3(0.25, 0, 0));
        const auto before = scene.state.deformed_positions;
        ASSERT_TRUE(scene.advance().converged);
        expect_positions_identical(scene.state.deformed_positions, before);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothDriverRejectsRigidAndTetMeshesBeforeMutation) {
    for (bool rigid : {false, true}) {
        InitialGuessClothScene scene;
        select_guess(scene.params);
        const auto before = scene.state.deformed_positions;
        if (rigid) scene.mesh.rb_nodes = {{1, 2, 3}};
        else scene.mesh.tets = {0, 1, 2, 3};
        EXPECT_THROW(scene.advance(), std::invalid_argument);
        expect_positions_identical(scene.state.deformed_positions, before);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothDriverValidatesBeforeMutation) {
    InitialGuessClothScene scene;
    select_guess(scene.params);
    scene.state.velocities.assign(4, Vec3(0.25, 0, 0));
    const auto before = scene.state.deformed_positions;
    const auto before_velocity = scene.state.velocities;
    int pin_updates = 0, callbacks = 0;
    const auto updater = [&](std::vector<Pin>&, double) { ++pin_updates; };
    const auto callback = [&](int, const std::vector<Vec3>&) { ++callbacks; };
    scene.params.colored_ccd_guess_iters = -1;
    EXPECT_THROW(scene.advance(callback, updater), std::invalid_argument);
    scene.params.colored_ccd_guess_iters = 1;
    scene.params.d_hat = -0.001;
    EXPECT_THROW(scene.advance(callback, updater), std::invalid_argument);
    scene.params.d_hat = 0.005;
    scene.params.use_colored_ccd_guess = true;
    scene.params.use_colored_ccd_guess_level_based = true;
    EXPECT_THROW(scene.advance(callback, updater), std::invalid_argument);
    EXPECT_EQ(pin_updates, 0);
    EXPECT_EQ(callbacks, 0);
    expect_positions_identical(scene.state.deformed_positions, before);
    expect_positions_identical(scene.state.velocities, before_velocity);
}

TEST_P(CollisionColoredCCDInitialGuessDriver, ClothOptionIsSharedByEveryClothSolverDispatch) {
    for (int variant : {0, 1, 2, 3}) {
        SCOPED_TRACE(variant);
        InitialGuessClothScene scene;
        select_guess(scene.params);
        scene.params.use_parallel = true;
        scene.params.max_global_iters = 1;
        scene.params.damping = 0.0;
        scene.params.mu = 10.0;
        scene.params.lambda = 10.0;
        scene.params.colored_ccd_guess_iters = 1;
        scene.params.use_basic_experimental = variant == 1;
        scene.params.use_basic_experimental_v2 = variant == 2;
        scene.params.use_simd = variant == 2;
        scene.params.use_cloth_grid = variant == 3;
        scene.state.velocities[0] = Vec3(0, 0, -4);
        const auto before = scene.state.deformed_positions;
        std::vector<Vec3> xhat, displacement(before.size());
        build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
        for (std::size_t i = 0; i < before.size(); ++i)
            displacement[i] = xhat[i] - before[i];
        const auto expected = expected_guess(
            before, displacement, scene.mesh, scene.params, 1);
        ASSERT_TRUE(scene.advance().converged);
        expect_positions_identical(scene.state.deformed_positions, expected);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, UnconstrainedSweepsUseOriginalTargets) {
    RefMesh mesh = ref_mesh_with_masses({1, 1, 1});
    const std::vector<Vec3> x = {
        Vec3(0, 0, 0), Vec3(4, -2, 8), Vec3(-3, 2, 1)};
    // Deliberately much longer than the solver's normal node-box size, with
    // stationary axes that must not be clipped by the box's safety inset.
    const std::vector<Vec3> displacement = {
        Vec3(2, 0, -4), Vec3(0, 3, 0), Vec3::Zero()};
    const auto original_x = x;
    const auto original_displacement = displacement;
    std::vector<Vec3> target = x;
    for (std::size_t i = 0; i < x.size(); ++i) target[i] += displacement[i];

    for (const int sweeps : {1, 2, 5}) {
        SCOPED_TRACE(sweeps);
        expect_positions_identical(
            collision_colored_ccd_initial_guess(x, displacement, mesh, params, sweeps),
            target);
    }
    expect_positions_identical(x, original_x);
    expect_positions_identical(displacement, original_displacement);
}

TEST_F(CollisionColoredCCDInitialGuess, SolidSurfaceAndInteriorNodesReachTheirTargets) {
    InitialGuessSolidScene scene;
    const auto x = scene.state.deformed_positions;
    std::vector<Vec3> displacement(x.size(), Vec3(0.01, 0.02, -0.01));
    ASSERT_EQ(scene.mesh.surface_nodes.size(), 4U);
    for (const int node : scene.mesh.surface_nodes) ASSERT_NE(node, 4);

    params.use_parallel = false;
    const auto serial = collision_colored_ccd_initial_guess(
        x, displacement, scene.mesh, params, 2);
    params.use_parallel = true;
    const auto parallel = collision_colored_ccd_initial_guess(
        x, displacement, scene.mesh, params, 2);
    expect_positions_identical(parallel, serial);
    for (std::size_t node = 0; node < x.size(); ++node)
        expect_vec_near(serial[node], x[node] + displacement[node], 0.0);

    // This helper is collision-only, not a tet-inversion limiter. In
    // particular an interior node must not be mistaken for a surface point
    // and clipped against its own solid's boundary triangles.
    displacement.assign(x.size(), Vec3::Zero());
    displacement[4] = Vec3(2, 0, 0);
    const auto interior = collision_colored_ccd_initial_guess(
        x, displacement, scene.mesh, params, 1);
    expect_vec_near(interior[4], x[4] + displacement[4], 0.0);
    for (int node = 0; node < 4; ++node)
        expect_vec_near(interior[node], x[node], 0.0);
}

TEST_F(CollisionColoredCCDInitialGuess, RigidSurfaceStaysFixedAndClipsSolidMotion) {
    RefMesh mesh;
    DeformedState state;
    create_solid({Vec3(0, 0, 1), Vec3(0.2, 0, 1),
                     Vec3(0, 0.2, 1), Vec3(0, 0, 1.2)},
        {0, 1, 2, 3}, 24.0, mesh, state);
    mesh.tris.insert(mesh.tris.end(), {4, 5, 6});
    create_rigid_body({Vec3(-2, -2, 0), Vec3(2, -2, 0), Vec3(0, 2, 0)},
        Vec3::Zero(), Vec4(1, 0, 0, 0), Vec3::Zero(), 3.0, mesh, state);
    mesh.build_deformable_nodes();
    const auto x = state.deformed_positions;
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    // Even an explicit nonzero input may not independently move a rigid
    // proxy. Its triangle must nevertheless remain a collision obstacle.
    for (int node = 4; node < 7; ++node)
        displacement[node] = Vec3(0.25, 0.5, 1.0);
    for (bool parallel : {false, true}) {
        params.use_parallel = parallel;
        for (const int sweeps : {2, 20}) {
            SCOPED_TRACE(sweeps);
            const auto result = collision_colored_ccd_initial_guess(
                x, displacement, mesh, params, sweeps);
            if (sweeps == 2) {
                expect_vec_near(result[0], Vec3(0, 0, 0.01), 1e-12);
            } else {
                EXPECT_GE(result[0].z(), 1e-8);
                EXPECT_LT(result[0].z(), 1e-7);
            }
            for (int node = 1; node < 7; ++node)
                expect_positions_identical({result[node]}, {x[node]});
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuess, RejectsAmbiguousRigidOwnership) {
    auto mesh = point_triangle_mesh();
    const auto x = point_above_triangle();
    const std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    mesh.node_to_rb = {-1};
    EXPECT_THROW(collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 1), std::invalid_argument);
    mesh.node_to_rb.clear();
    mesh.rb_nodes = {{1, 2, 3}};
    EXPECT_THROW(collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 1), std::invalid_argument);
}

TEST_P(CollisionColoredCCDInitialGuessDriver, GeneralInitialGuessUsesXhatForBothSolverVariants) {
    for (bool v2 : {false, true}) {
        for (bool mixed : {false, true}) {
            for (int sweeps : {0, 2}) {
                SCOPED_TRACE(::testing::Message() << "v2=" << v2
                    << " mixed=" << mixed << " sweeps=" << sweeps);
                InitialGuessSolidScene scene(mixed);
                select_guess(scene.params);
                scene.params.use_basic_experimental_v2 = v2;
                scene.params.use_simd = v2;
                scene.params.use_parallel = true;
                scene.params.use_ccd_guess = true;
                scene.params.use_verlet_guess = true;
                scene.params.use_translation_guess = true;
                scene.params.gravity = Vec3(0, -9.81, 0);
                scene.params.colored_ccd_guess_iters = sweeps;
                scene.state.velocities.assign(scene.mesh.num_positions, Vec3(0.02, 0.04, -0.02));
                const auto before = scene.state.deformed_positions;
                const auto before_com = scene.state.x_coms;
                const auto before_q = scene.state.orientations;
                std::vector<Vec3> expected = before;
                if (sweeps != 0) {
                    for (int node = 0; node < 5; ++node)
                        expected[node] += scene.params.dt() * scene.state.velocities[node];
                }
                ASSERT_TRUE(scene.advance().converged);
                expect_positions_identical(scene.state.deformed_positions, expected);
                expect_positions_identical(scene.state.x_coms, before_com);
                ASSERT_EQ(scene.state.orientations.size(), before_q.size());
                for (std::size_t rb = 0; rb < before_q.size(); ++rb)
                    EXPECT_EQ(std::memcmp(scene.state.orientations[rb].data(),
                        before_q[rb].data(), 4 * sizeof(double)), 0);
            }
        }
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, GeneralFractionScalesDeformablesAndKeepsRigidProxiesFixed) {
    double maximum_target_error = 0.0;
    for (bool v2 : {false, true}) {
        for (bool mixed : {false, true}) {
            for (double fraction : {0.0, 0.9, 1.0}) {
                for (int sweeps : {1, 4}) {
                    SCOPED_TRACE(::testing::Message() << "v2=" << v2 << " mixed=" << mixed
                        << " fraction=" << fraction << " sweeps=" << sweeps);
                    InitialGuessSolidScene scene(mixed);
                    select_guess(scene.params);
                    scene.params.use_basic_experimental_v2 = v2;
                    scene.params.use_simd = v2;
                    scene.params.use_parallel = true;
                    scene.params.colored_ccd_guess_fraction = fraction;
                    scene.params.colored_ccd_guess_iters = sweeps;
                    scene.state.velocities.assign(scene.mesh.num_positions,
                        Vec3(0.02, 0.04, -0.02));
                    const auto before = scene.state.deformed_positions;
                    const auto before_com = scene.state.x_coms;
                    const auto before_q = scene.state.orientations;
                    std::vector<Vec3> xhat, displacement(before.size(), Vec3::Zero());
                    std::vector<Vec3> analytic_target = before;
                    build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
                    for (int node = 0; node < scene.mesh.num_positions; ++node) {
                        if (scene.mesh.node_to_rb[node] >= 0) continue;
                        displacement[node] = xhat[node] - before[node];
                        if (fraction != 1.0) displacement[node] *= fraction;
                        analytic_target[node] += displacement[node];
                    }
                    auto reference_params = scene.params;
                    reference_params.colored_ccd_guess_fraction = 1.0;
                    const auto expected = expected_guess(before, displacement,
                        scene.mesh, reference_params, sweeps);
                    ASSERT_TRUE(scene.advance().converged);
                    expect_fraction_reference_identical(scene.state.deformed_positions, expected);
                    expect_fraction_target_near(scene.state.deformed_positions,
                        analytic_target, maximum_target_error);
                    if (fraction == 0.0 || fraction == 1.0)
                        expect_positions_identical(scene.state.deformed_positions, analytic_target);
                    expect_positions_identical(scene.state.x_coms, before_com);
                    ASSERT_EQ(scene.state.orientations.size(), before_q.size());
                    for (std::size_t rb = 0; rb < before_q.size(); ++rb)
                        EXPECT_EQ(std::memcmp(scene.state.orientations[rb].data(),
                            before_q[rb].data(), 4 * sizeof(double)), 0);
                }
            }
        }
    }
    std::ostringstream error_property;
    error_property << std::setprecision(17) << maximum_target_error;
    RecordProperty("max_analytic_target_abs_error", error_property.str());
}

TEST_F(CollisionColoredCCDInitialGuess, DisabledColoredFractionDoesNotAlterLegacyPredictors) {
    for (int mode : {1, 2, 3}) {
        InitialGuessClothScene scene;
        scene.params.use_colored_ccd_guess = false;
        scene.params.use_colored_ccd_guess_level_based = false;
        scene.params.colored_ccd_guess_fraction = 0.0;
        scene.params.use_ccd_guess = mode == 1;
        scene.params.use_verlet_guess = mode == 2;
        scene.params.use_translation_guess = mode == 3;
        scene.state.velocities.assign(4, Vec3(0.25, 0, 0));
        const auto before = scene.state.deformed_positions;
        std::vector<Vec3> xhat;
        build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
        std::vector<Vec3> expected;
        if (mode == 1) expected = ccd_initial_guess(before, xhat, scene.mesh);
        if (mode == 2) expected = verlet_initial_guess(before, xhat, scene.mesh, scene.params);
        if (mode == 3) expected = translation_initial_guess(before, xhat,
            scene.mesh, scene.pins, scene.params);
        ASSERT_TRUE(scene.advance().converged);
        expect_positions_identical(scene.state.deformed_positions, expected);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, GeneralInitialGuessRebuildsLiveSubstepTargets) {
    for (bool v2 : {false, true}) {
        InitialGuessSolidScene scene;
        select_guess(scene.params);
        scene.params.use_basic_experimental_v2 = v2;
        scene.params.use_simd = v2;
        scene.params.substeps = 3;
        scene.params.colored_ccd_guess_iters = 1;
        scene.state.velocities.assign(5, Vec3(0.02, 0.04, -0.02));
        DeformedState expected = scene.state;
        int observed = 0;
        ASSERT_TRUE(scene.advance([&](int global_sub, const std::vector<Vec3>& actual) {
            EXPECT_EQ(global_sub, observed++);
            const auto previous = expected.deformed_positions;
            std::vector<Vec3> xhat, displacement(previous.size());
            build_xhat(xhat, previous, expected.velocities, scene.params.dt());
            for (std::size_t node = 0; node < previous.size(); ++node)
                displacement[node] = xhat[node] - previous[node];
            expected.deformed_positions = expected_guess(
                previous, displacement, scene.mesh, scene.params, 1);
            update_velocity(expected.velocities, expected.deformed_positions,
                previous, scene.params.dt());
            expect_positions_identical(actual, expected.deformed_positions);
        }).converged);
        EXPECT_EQ(observed, 3);
        expect_positions_identical(scene.state.velocities, expected.velocities);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, GeneralDriverSelectsTheRequestedCollisionPredictor) {
    for (bool with_solid : {false, true}) {
        for (bool v2 : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "solid=" << with_solid << " v2=" << v2);
            InitialGuessClothScene scene;
            select_guess(scene.params);
            if (with_solid) {
                create_solid({Vec3(20, 0, 0), Vec3(21, 0, 0),
                                 Vec3(20, 1, 0), Vec3(20, 0, 1)},
                    {0, 1, 2, 3}, 24.0, scene.mesh, scene.state);
            }
            scene.mesh.node_to_rb.assign(scene.mesh.num_positions, -1);
            scene.mesh.build_deformable_nodes();
            scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
            scene.params.use_basic_experimental_v2 = v2;
            scene.params.use_simd = v2;
            scene.params.colored_ccd_guess_iters = 1;
            scene.state.velocities[0] = Vec3(0, 0, -4);
            for (int node = 1; node < 4; ++node)
                scene.state.velocities[node] = Vec3(8, 0, 0);
            const auto before = scene.state.deformed_positions;
            std::vector<Vec3> xhat, displacement(before.size());
            build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
            for (std::size_t node = 0; node < before.size(); ++node)
                displacement[node] = xhat[node] - before[node];
            const auto original = collision_colored_ccd_initial_guess(
                before, displacement, scene.mesh, scene.params, 1);
            const auto level_based = collision_colored_ccd_initial_guess_level_based(
                before, displacement, scene.mesh, scene.params, 1);
            ASSERT_GT(original[0].z(), level_based[0].z() + 0.5);
            ASSERT_TRUE(advance_one_frame_general(scene.state, scene.mesh,
                scene.adjacency, scene.pins, scene.params, scene.broad_phase).converged);
            expect_positions_identical(scene.state.deformed_positions,
                GetParam() ? level_based : original);
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuess, DisabledGeneralOptionPreservesLegacyStart) {
    for (bool v2 : {false, true}) {
        InitialGuessSolidScene scene(true);
        scene.params.use_basic_experimental_v2 = v2;
        scene.params.use_simd = v2;
        scene.params.use_colored_ccd_guess = false;
        scene.params.colored_ccd_guess_iters = -1;
        scene.params.use_ccd_guess = true;
        scene.params.use_verlet_guess = true;
        scene.params.use_translation_guess = true;
        scene.state.velocities.assign(scene.mesh.num_positions, Vec3(0.02, 0.04, 0));
        const auto before = scene.state.deformed_positions;
        ASSERT_TRUE(scene.advance().converged);
        expect_positions_identical(scene.state.deformed_positions, before);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, OgcRetainsPrecedenceInGeneralDriver) {
    for (bool solver_ogc : {false, true}) {
        InitialGuessSolidScene scene;
        select_guess(scene.params);
        scene.params.use_ogc = !solver_ogc;
        scene.params.use_ogc_solver = solver_ogc;
        scene.state.velocities.assign(5, Vec3(0.02, 0.04, 0));
        const auto before = scene.state.deformed_positions;
        ASSERT_TRUE(scene.advance().converged);
        expect_positions_identical(scene.state.deformed_positions, before);
    }
}

TEST_P(CollisionColoredCCDInitialGuessDriver, GeneralDriverValidatesBeforeMutation) {
    InitialGuessSolidScene scene(true);
    select_guess(scene.params);
    scene.state.velocities.assign(scene.mesh.num_positions, Vec3(0.02, 0.04, 0));
    const auto before = scene.state.deformed_positions;
    const auto before_velocity = scene.state.velocities;
    int pin_updates = 0, callbacks = 0;
    const auto updater = [&](std::vector<Pin>&, double) { ++pin_updates; };
    const auto callback = [&](int, const std::vector<Vec3>&) { ++callbacks; };
    scene.params.colored_ccd_guess_iters = -1;
    EXPECT_THROW(scene.advance(callback, updater), std::invalid_argument);
    scene.params.colored_ccd_guess_iters = 1;
    scene.params.d_hat = -0.001;
    EXPECT_THROW(scene.advance(callback, updater), std::invalid_argument);
    scene.params.d_hat = 0.005;
    scene.params.use_colored_ccd_guess = true;
    scene.params.use_colored_ccd_guess_level_based = true;
    EXPECT_THROW(scene.advance(callback, updater), std::invalid_argument);
    EXPECT_EQ(pin_updates, 0);
    EXPECT_EQ(callbacks, 0);
    expect_positions_identical(scene.state.deformed_positions, before);
    expect_positions_identical(scene.state.velocities, before_velocity);
}

TEST_F(CollisionColoredCCDInitialGuess, EmptyAndZeroSweepInputsAreUnchanged) {
    const RefMesh empty_mesh;
    EXPECT_TRUE(collision_colored_ccd_initial_guess({}, {}, empty_mesh, params, 0).empty());
    EXPECT_TRUE(collision_colored_ccd_initial_guess({}, {}, empty_mesh, params, 3).empty());

    const auto x = point_above_triangle();
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    expect_positions_identical(
        collision_colored_ccd_initial_guess(x, displacement, point_triangle_mesh(), params, 0), x);
}

TEST_F(CollisionColoredCCDInitialGuess, StationaryTriangleClipsEachRemainingAttempt) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);

    for (const int sweeps : {1, 2, 3}) {
        const auto result = collision_colored_ccd_initial_guess(x, displacement, mesh, params, sweeps);
        // Each retry consumes 90% of the distance to the stationary z=0
        // triangle, not 90% of a newly added displacement.
        expect_vec_near(result[0], Vec3(0, 0, std::pow(0.1, sweeps)), 1e-12);
        EXPECT_GT(result[0].z(), 0.0);
        for (std::size_t i = 1; i < x.size(); ++i)
            expect_vec_near(result[i], x[i], 0.0);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, RepeatedRetriesKeepASeparationAndCanMoveAway) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);

    const auto close = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 20);
    // The target remains behind the triangle. Extra sweeps must not reduce
    // the gap into CCD's initial-contact tolerance and lock this vertex.
    EXPECT_GE(close[0].z(), 1e-8);
    EXPECT_LT(close[0].z(), 1e-7);
    for (std::size_t i = 1; i < x.size(); ++i)
        expect_positions_identical({close[i]}, {x[i]});

    displacement[0] = Vec3(0, 0, 0.25) - close[0];
    const auto separated = collision_colored_ccd_initial_guess(
        close, displacement, mesh, params, 1);
    expect_positions_identical({separated[0]}, {close[0] + displacement[0]});
    EXPECT_GT(separated[0].z(), close[0].z());
    for (std::size_t i = 1; i < x.size(); ++i)
        expect_positions_identical({separated[i]}, {x[i]});
}

TEST_F(CollisionColoredCCDInitialGuess, NearMissTargetRespectsTheSurfaceGap) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    // There is no surface crossing and hence no time of impact along this
    // segment, but accepting this target would still violate the gap floor.
    displacement[0] = Vec3(0, 0, 5e-9) - x[0];
    const auto clipped = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 40);
    EXPECT_GE(clipped[0].z(), 1e-8);
    EXPECT_LT(clipped[0].z(), 1e-7);
    EXPECT_GT(clipped[0].z(), (x[0] + displacement[0]).z());

    // A reachable target outside the surface margin is not shortened by a
    // fixed offset from the target: it is accepted exactly, even near contact.
    displacement[0] = Vec3(0, 0, 2e-8) - x[0];
    const auto safe = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 20);
    expect_positions_identical({safe[0]}, {x[0] + displacement[0]});
}

TEST_F(CollisionColoredCCDInitialGuess, ExistingSmallGapCannotShrinkButCanImprove) {
    auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    x[0].z() = 5e-9;
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, 1e-9) - x[0];
    const auto inward = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 20);
    {
        SCOPED_TRACE("pre-existing 5e-9 gap, inward target");
        EXPECT_EQ(inward[0].z(), x[0].z());
        expect_positions_identical(inward, x);
    }

    displacement[0] = Vec3(0, 0, 1e-4) - x[0];
    const auto outward = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 1);
    {
        SCOPED_TRACE("pre-existing 5e-9 gap, outward target");
        EXPECT_EQ(outward[0].z(), (x[0] + displacement[0]).z());
        expect_positions_identical({outward[0]}, {x[0] + displacement[0]});
    }

    // This safeguard prevents creating a locked gap; it does not repair
    // inputs already inside CCD's initial-contact tolerance or permit those
    // gaps to shrink further. Escaping motion is left to the existing CCD.
    x[0].z() = 5e-11;
    displacement[0] = Vec3(0, 0, 1e-11) - x[0];
    const auto below_tolerance = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 20);
    {
        SCOPED_TRACE("pre-existing 5e-11 gap, inward target");
        EXPECT_EQ(below_tolerance[0].z(), x[0].z());
        expect_positions_identical(below_tolerance, x);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, ExistingSmallGapDoesNotRelaxAnotherPairsFloor) {
    RefMesh mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1, 1});
    mesh.tris = {1, 2, 3, 4, 5, 6};
    const std::vector<Vec3> x = {
        Vec3(0, 0, 5e-9), Vec3(-1, -1, 0), Vec3(1, -1, 0), Vec3(0, 1, 0),
        Vec3(-1, -1, 1e-7), Vec3(1, -1, 1e-7), Vec3(0, 1, 1e-7)};
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, 9.5e-8) - x[0];
    const auto result = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 20);
    // The bottom triangle starts only 5e-9 away, but the top triangle starts
    // clear. A single global minimum must not let the latter shrink to 5e-9.
    EXPECT_GT(result[0].z(), x[0].z());
    EXPECT_GE(1e-7 - result[0].z(), 1e-8);
    EXPECT_LT(result[0].z(), (x[0] + displacement[0]).z());
    for (std::size_t i = 1; i < x.size(); ++i)
        expect_positions_identical({result[i]}, {x[i]});
}

TEST_F(CollisionColoredCCDInitialGuess, ObtuseTriangleChecksAllBoundaryEdges) {
    const auto mesh = point_triangle_mesh();
    const std::vector<Vec3> x = {
        Vec3(8e-9, -1, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(-1, 1, 0)};
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(8e-9, -8e-9, 0) - x[0];
    const auto result = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 40);
    // This point lies outside an obtuse triangle. Its closest feature is the
    // interior of edge (1,2), even when two plane barycentrics are negative.
    // The vertex-1 distance exceeds 1e-8, but the actual edge distance does not.
    EXPECT_DOUBLE_EQ(result[0].x(), x[0].x());
    EXPECT_DOUBLE_EQ(result[0].z(), 0.0);
    EXPECT_GE(-result[0].y(), 1e-8);
    EXPECT_LT(-result[0].y(), 1e-7);
    for (std::size_t i = 1; i < x.size(); ++i)
        expect_positions_identical({result[i]}, {x[i]});
}

TEST_F(CollisionColoredCCDInitialGuess, MovingTriangleVertexKeepsTheSurfaceGap) {
    const auto mesh = point_triangle_mesh();
    const std::vector<Vec3> x = {
        Vec3(0, 0, 0), Vec3(0, 1, 1),
        Vec3(-1, -1, 0), Vec3(1, -1, 0)};
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[1] = Vec3(0, 0, -2);
    const auto result = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 20);
    // The moving vertex is a triangle role, not the point role, in the NT
    // pair. Its contact distance differs from its own height above z=0.
    const double gap = node_triangle_distance(
        result[0], result[1], result[2], result[3]).distance;
    EXPECT_GE(gap, 1e-8);
    EXPECT_LT(gap, 1e-7);
    EXPECT_GT(result[1].z(), 0.0);
    for (const int node : {0, 2, 3})
        expect_positions_identical({result[node]}, {x[node]});
}

TEST_F(CollisionColoredCCDInitialGuess, OriginalCCDStillPreservesTheColoredGuessGap) {
    params.use_ticcd = false;
    params.exact_computation_fallback = false;
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    const auto result = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 40);
    const double gap = node_triangle_distance(
        result[0], result[1], result[2], result[3]).distance;
    EXPECT_GE(gap, 1e-8);
    EXPECT_LT(gap, 1e-7);
    for (std::size_t node = 1; node < x.size(); ++node)
        expect_positions_identical({result[node]}, {x[node]});
}

TEST_F(CollisionColoredCCDInitialGuess, LaterColorClearsObstacleForNextSweep) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3(4, 0, 0));
    displacement[0] = Vec3(0, 0, -2);
    // The candidate clique is colored in ascending vertex order: the point
    // first hits the original triangle, whose vertices then move to the right.
    const auto once = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 1);
    expect_vec_near(once[0], Vec3(0, 0, 0.1), 1e-12);
    for (std::size_t i = 1; i < x.size(); ++i)
        expect_vec_near(once[i], x[i] + displacement[i], 1e-12);

    for (const int sweeps : {2, 4}) {
        const auto result = collision_colored_ccd_initial_guess(x, displacement, mesh, params, sweeps);
        for (std::size_t i = 0; i < x.size(); ++i)
            expect_vec_near(result[i], x[i] + displacement[i], 1e-12);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, StalledPointRetriesAfterLaterColorsClearItsObstacle) {
    auto x = point_above_triangle();
    x[0].z() = 5e-9;
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3(4, 0, 0));
    displacement[0] = Vec3(0, 0, -2);
    const auto first = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 1);
    expect_positions_identical({first[0]}, {x[0]});
    const auto retried = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 20);
    for (std::size_t i = 0; i < x.size(); ++i)
        expect_vec_near(retried[i], x[i] + displacement[i], 1e-12);
}

TEST_F(CollisionColoredCCDInitialGuess, BarrierRangeDoesNotAddSeparatedGuessCandidates) {
    auto mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1, 1});
    mesh.tris = {1, 2, 3, 4, 5, 6};
    auto x = point_above_triangle();
    x.insert(x.end(), {Vec3(10, -1, 0), Vec3(12, -1, 0), Vec3(11, 1, 0)});
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    for (int i = 1; i < 4; ++i) displacement[i] = Vec3(4, 0, 0);
    std::vector<Vec3> expected;
    std::size_t nt_count = 0, ss_count = 0;
    for (const double activation_distance : {0.0, 0.005, 25.0}) {
        SCOPED_TRACE(activation_distance);
        params.d_hat = activation_distance;
        BroadPhase scratch;
        const auto result = collision_colored_ccd_initial_guess(
            x, displacement, mesh, params, 2, &scratch);
        if (expected.empty()) {
            expected = result;
            nt_count = scratch.nt_pairs().size();
            ss_count = scratch.ss_pairs().size();
            EXPECT_GT(nt_count, 0U);
        } else {
            expect_positions_identical(result, expected);
            EXPECT_EQ(scratch.nt_pairs().size(), nt_count);
            EXPECT_EQ(scratch.ss_pairs().size(), ss_count);
        }
        expect_vec_near(result[0], x[0] + displacement[0], 1e-12);
        for (const auto& pair : scratch.nt_pairs()) {
            EXPECT_LT(pair.node, 4);
            for (int node : pair.tri_v) EXPECT_LT(node, 4);
        }
        for (const auto& pair : scratch.ss_pairs())
            for (int node : pair.v) EXPECT_LT(node, 4);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, ReusedBroadPhaseRebuildsChangingMotionAndContacts) {
    const auto mesh = point_triangle_mesh();
    BroadPhase scratch;
    for (const int threads : {1, 8, 64}) {
        omp_set_num_threads(threads);
        for (const double height : {1.0, 20.0, 5e-9, 1.0}) {
            auto x = point_above_triangle();
            x[0].z() = height;
            std::vector<Vec3> displacement(x.size(), Vec3::Zero());
            displacement[0] = Vec3(0, 0, -2 * height);
            // The later colors clear the obstacle on alternating calls.
            if (height == 1.0)
                for (int i = 1; i < 4; ++i) displacement[i] = Vec3(4, 0, 0);
            for (const int sweeps : {0, 1, 4, 20}) {
                SCOPED_TRACE(::testing::Message() << "threads=" << threads
                    << " height=" << height << " sweeps=" << sweeps);
                params.d_hat = height == 20.0 ? 0.0 : 0.005;
                const auto expected = collision_colored_ccd_initial_guess(
                    x, displacement, mesh, params, sweeps);
                // A solver/other guess can overwrite the shared cache between
                // calls. The colored guess must replace its boxes and pairs.
                scratch.build_ccd_candidates(x, displacement, mesh, 0.01,
                    /*retain_solver_data=*/false);
                expect_positions_identical(collision_colored_ccd_initial_guess(
                    x, displacement, mesh, params, sweeps, &scratch), expected);
            }
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuess, SeparationCertificateCoversTheWholeSweep) {
    const auto mesh = point_triangle_mesh();
    const std::vector<Vec3> x = {Vec3(.25, .25, 0), Vec3(0, 1, 0),
        Vec3(1, 0, 0), Vec3(1, 1, 0)};
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    BroadPhase scratch;
    // The point lies inside the triangle's AABB but outside its finite face.
    // This entire short motion remains separated along a diagonal direction.
    displacement[0] = Vec3(.01, .01, 0);
    const auto clear = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 1, &scratch);
    ASSERT_EQ(scratch.nt_pairs().size(), 1U);
    for (const auto& row : scratch.cache().vertex_nt) EXPECT_TRUE(row.empty());
    expect_vec_near(clear[0], x[0] + displacement[0], 1e-12);

    // The same starting separation cannot certify a longer segment that
    // crosses the face. Preserve the candidate and clip before its first hit.
    displacement[0] = Vec3(.5, .5, 0);
    const auto crossing = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 1, &scratch);
    ASSERT_EQ(scratch.nt_pairs().size(), 1U);
    EXPECT_FALSE(scratch.cache().vertex_nt[0].empty());
    EXPECT_LT(crossing[0].x(), .5);
    EXPECT_LT(crossing[0].y(), .5);
}

TEST_F(CollisionColoredCCDInitialGuess, CachedColoringRebuildsForChangedTopology) {
    RefMesh mesh;
    std::vector<Vec3> x, displacement;
    std::vector<int> triangles;
    for (int copy = 0; copy < 48; ++copy) {
        const int base = static_cast<int>(x.size());
        for (const auto& point : point_above_triangle())
            x.push_back(point + Vec3(12 * copy, 0, 0));
        triangles.insert(triangles.end(), {base + 1, base + 2, base + 3});
        displacement.insert(displacement.end(), {Vec3(0, 0, -2),
            Vec3::Zero(), Vec3::Zero(), Vec3::Zero()});
    }
    mesh.num_positions = x.size();
    mesh.mass.assign(x.size(), 1.0);
    for (const int threads : {1, 8, 64}) {
        omp_set_num_threads(threads);
        for (const bool obstacles : {true, false, true}) {
            mesh.tris = obstacles ? triangles : std::vector<int>{};
            const auto result = collision_colored_ccd_initial_guess(
                x, displacement, mesh, params, 1);
            for (int copy = 0; copy < 48; ++copy)
                EXPECT_NEAR(result[4 * copy].z(), obstacles ? .1 : -1.0, 1e-12);
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuess, EdgeEdgeCrossingClipsMovingEndpoint) {
    RefMesh mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1});
    mesh.tris = {0, 1, 4, 2, 3, 5};
    const std::vector<Vec3> x = {
        Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0.5, -1, 0),
        Vec3(0.5, 1, 0), Vec3(-2, 0, 2), Vec3(0.5, 0, -2)};
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    // At t=1/2, edge (0,1) hits the interior of edge (2,3).
    // No moving vertex crosses the other triangle's plane, so this exercises
    // the segment-segment candidates rather than a point-triangle crossing.
    const auto result = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 1);
    expect_vec_near(result[0], Vec3(0, 0, 0.1), 1e-12);
    for (std::size_t i = 1; i < x.size(); ++i)
        expect_vec_near(result[i], x[i], 0.0);
}

TEST_F(CollisionColoredCCDInitialGuess, RepeatedEdgeEdgeRetriesKeepTheSurfaceGap) {
    RefMesh mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1});
    mesh.tris = {0, 1, 4, 2, 3, 5};
    const std::vector<Vec3> x = {
        Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0.5, -1, 0),
        Vec3(0.5, 1, 0), Vec3(-2, 0, 2), Vec3(0.5, 0, -2)};
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    const auto result = collision_colored_ccd_initial_guess(
        x, displacement, mesh, params, 20);
    const double gap = segment_segment_distance(
        result[0], result[1], result[2], result[3]).distance;
    EXPECT_GE(gap, 1e-8);
    EXPECT_LT(gap, 1e-7);
    EXPECT_GT(result[0].z(), 0.0);
    for (std::size_t i = 1; i < x.size(); ++i)
        expect_positions_identical({result[i]}, {x[i]});
}

TEST_F(CollisionColoredCCDInitialGuess, SerialAndParallelSweepsAreBitwiseIdentical) {
    RefMesh mesh;
    std::vector<Vec3> x, displacement;
    constexpr int copies = 48;
    for (int copy = 0; copy < copies; ++copy) {
        const int offset = static_cast<int>(x.size());
        for (const Vec3& p : point_above_triangle()) x.push_back(p + Vec3(12 * copy, 0, 0));
        mesh.tris.insert(mesh.tris.end(), {offset + 1, offset + 2, offset + 3});
        displacement.push_back(Vec3(0, 0, -2));
        for (int i = 0; i < 3; ++i)
            displacement.push_back(copy % 2 ? Vec3(4, 0, 0) : Vec3::Zero());
    }
    mesh.num_positions = x.size();
    mesh.mass.assign(x.size(), 1.0);
    // Disjoint candidate cliques put 48 vertices in each of four colors;
    // 192 vertices also exercises the parallel graph-coloring path.
    for (const int sweeps : {3, 20}) {
        SCOPED_TRACE(sweeps);
        omp_set_num_threads(1);
        const auto serial = collision_colored_ccd_initial_guess(
            x, displacement, mesh, params, sweeps);
        omp_set_num_threads(4);
        params.use_parallel = false;
        expect_positions_identical(
            collision_colored_ccd_initial_guess(x, displacement, mesh, params, sweeps), serial);
        params.use_parallel = true;
        for (int repeat = 0; repeat < 3; ++repeat) {
            const auto parallel = collision_colored_ccd_initial_guess(
                x, displacement, mesh, params, sweeps);
            expect_positions_identical(parallel, serial);
        }
        for (int copy = 0; copy < copies; ++copy) {
            if (copy % 2 || sweeps == 3) {
                EXPECT_NEAR(serial[4 * copy].z(), copy % 2 ? -1.0 : 0.001, 1e-12);
            } else {
                EXPECT_GE(serial[4 * copy].z(), 1e-8);
                EXPECT_LT(serial[4 * copy].z(), 1e-7);
            }
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuess, FallingSquaresRetryOriginalGravityTargets) {
    // Both squares are free falling from rest for one implicit-Euler step,
    // hence dx = dt^2 * gravity. The lower square is smaller; neither is fixed.
    // Number the upper square first so colored partial updates encounter the
    // lower square before it has completed its matching downward movement.
    RefMesh mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1, 1, 1});
    mesh.tris = {0, 2, 1, 0, 3, 2, 4, 6, 5, 4, 7, 6};
    const double dt = 0.2;
    const Vec3 gravity(0, -9.81, 0);
    for (const double gap : {0.005, 0.02, 0.05}) {
        SCOPED_TRACE(gap);
        const double lower_height = 1.05 - gap;
        const std::vector<Vec3> x = {
            Vec3(-1.0, 1.05, -1.0), Vec3(1.0, 1.05, -1.0),
            Vec3(1.0, 1.05, 1.0), Vec3(-1.0, 1.05, 1.0),
            Vec3(-0.5, lower_height, -0.5), Vec3(0.5, lower_height, -0.5),
            Vec3(0.5, lower_height, 0.5), Vec3(-0.5, lower_height, 0.5)};
        const std::vector<Vec3> displacement(x.size(), dt * dt * gravity);
        const auto original_x = x;
        const auto original_displacement = displacement;
        std::vector<Vec3> fixed_targets = x;
        for (std::size_t vertex = 0; vertex < x.size(); ++vertex)
            fixed_targets[vertex] += displacement[vertex];

        // Restart each call from the same initial state so extra sweeps retry
        // the remaining displacement rather than applying gravity again.
        const auto first = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 1);
        const auto second = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 2);
        const auto final = collision_colored_ccd_initial_guess(x, displacement, mesh, params, 10);
        expect_positions_identical(x, original_x);
        expect_positions_identical(displacement, original_displacement);
        bool has_remaining_displacement_after_first_sweep = false;
        for (std::size_t vertex = 0; vertex < x.size(); ++vertex) {
            SCOPED_TRACE(vertex);
            has_remaining_displacement_after_first_sweep |=
                (fixed_targets[vertex] - first[vertex]).norm() > 0.0;
            Vec3 previous = x[vertex];
            for (const auto* positions : {&first, &second, &final}) {
                EXPECT_DOUBLE_EQ((*positions)[vertex].x(), x[vertex].x());
                EXPECT_DOUBLE_EQ((*positions)[vertex].z(), x[vertex].z());
                EXPECT_GE((*positions)[vertex].y(), fixed_targets[vertex].y());
                EXPECT_LE((*positions)[vertex].y(), previous.y());
                EXPECT_LE((fixed_targets[vertex] - (*positions)[vertex]).norm(),
                          (fixed_targets[vertex] - previous).norm());
                previous = (*positions)[vertex];
            }
            EXPECT_LT(first[vertex].y(), x[vertex].y());
            expect_vec_near(second[vertex], fixed_targets[vertex], 1e-12);
            expect_vec_near(final[vertex], fixed_targets[vertex], 1e-12);
            EXPECT_NEAR(final[vertex].y() - x[vertex].y(), -0.3924, 1e-12);
        }
        EXPECT_TRUE(has_remaining_displacement_after_first_sweep);
        expect_positions_identical(final,
            collision_colored_ccd_initial_guess(x, displacement, mesh, params, 10));
    }
}

TEST_F(CollisionColoredCCDInitialGuess, RejectsInvalidInputAndNonfiniteTargets) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    const std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    EXPECT_THROW(collision_colored_ccd_initial_guess(x, {}, mesh, params, 1), std::invalid_argument);
    EXPECT_THROW(collision_colored_ccd_initial_guess(x, displacement, mesh, params, -1), std::invalid_argument);

    auto bad_params = params;
    for (const double invalid_d_hat : {-0.005, std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::quiet_NaN()}) {
        bad_params.d_hat = invalid_d_hat;
        EXPECT_THROW(collision_colored_ccd_initial_guess(x, displacement, mesh, bad_params, 1),
                     std::invalid_argument);
    }

    auto bad_mesh = mesh;
    bad_mesh.tris.push_back(0);
    EXPECT_THROW(collision_colored_ccd_initial_guess(x, displacement, bad_mesh, params, 1), std::invalid_argument);
    bad_mesh = mesh;
    bad_mesh.tris[0] = -1;
    EXPECT_THROW(collision_colored_ccd_initial_guess(x, displacement, bad_mesh, params, 1), std::out_of_range);
    bad_mesh.tris[0] = static_cast<int>(x.size());
    EXPECT_THROW(collision_colored_ccd_initial_guess(x, displacement, bad_mesh, params, 1), std::out_of_range);
    bad_mesh.tris = {0, 0, 1};
    EXPECT_THROW(collision_colored_ccd_initial_guess(x, displacement, bad_mesh, params, 1), std::invalid_argument);

    auto bad_x = x;
    bad_x[0].x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(collision_colored_ccd_initial_guess(bad_x, displacement, mesh, params, 1), std::invalid_argument);
    auto bad_displacement = displacement;
    bad_displacement[0].z() = std::numeric_limits<double>::infinity();
    EXPECT_THROW(collision_colored_ccd_initial_guess(x, bad_displacement, mesh, params, 1), std::invalid_argument);
    bad_x = x;
    bad_x[0].x() = std::numeric_limits<double>::max();
    bad_displacement = displacement;
    bad_displacement[0].x() = std::numeric_limits<double>::max();
    EXPECT_THROW(collision_colored_ccd_initial_guess(bad_x, bad_displacement, mesh, params, 1), std::invalid_argument);
    // A finite start/target can still fail to admit finite padded bounds.
    EXPECT_THROW(collision_colored_ccd_initial_guess(bad_x, displacement, mesh, params, 1), std::invalid_argument);
}

TEST_F(CollisionColoredCCDInitialGuessNew, ZeroIterationsMatchLegacyBitwise) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3(0, 0, -1));
    displacement[0] = Vec3(0, 0, -2);
    for (const bool exact : {false, true}) {
        params.exact_computation_fallback = exact;
        for (const bool parallel : {false, true}) {
            params.use_parallel = parallel;
            for (const int sweeps : {0}) {
                SCOPED_TRACE(::testing::Message() << "exact=" << exact
                    << " parallel=" << parallel << " sweeps=" << sweeps);
                const auto expected = collision_colored_ccd_initial_guess(
                    x, displacement, mesh, params, sweeps);
                expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
                    x, displacement, mesh, params, sweeps), expected);
            }
        }
    }
    EXPECT_TRUE(collision_colored_ccd_initial_guess_level_based({}, {}, RefMesh{}, params, 0).empty());
    EXPECT_TRUE(collision_colored_ccd_initial_guess_level_based({}, {}, RefMesh{}, params, 1000000).empty());
}

TEST_F(CollisionColoredCCDInitialGuessNew, ProbePositionsAreDiscardedBeforeTheFirstRealSweep) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    params.exact_computation_fallback = true;
    for (const bool parallel : {false, true}) {
        params.use_parallel = parallel;
        for (const int cap : {1, 2, 3}) {
            SCOPED_TRACE(::testing::Message() << "parallel=" << parallel << " cap=" << cap);
            const auto reference = replay_bfs_guess(x, displacement, mesh, params, cap);
            const auto retained_probe = replay_bfs_guess(x, displacement, mesh, params, cap, true);
            ColoredCCDGuessTrace trace;
            const auto actual = collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, params, cap, nullptr, &trace);
            expect_positions_identical(actual, reference.result);
            EXPECT_GT(actual[0].z(), retained_probe.result[0].z() + 1e-5);
            EXPECT_EQ(trace.completed_iterations, cap);
            EXPECT_TRUE(trace.probe_completed);
            expect_trace_consistent(trace, x, displacement, actual, cap, true);
            for (std::size_t node = 0; node < x.size(); ++node)
                EXPECT_DOUBLE_EQ(trace.first_sweep_movement[node],
                    (reference.first[node] - x[node]).stableNorm());
            ASSERT_GE(trace.steps.size(), 2U);
            EXPECT_EQ(trace.steps[1].stage, ColoredCCDGuessTraceStage::LevelsBuilt);
            expect_positions_identical(trace.steps[1].positions, x);
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, OneIterationUsesLevelOrderAfterTheDiscardedProbe) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3(4, 0, 0));
    displacement[0] = Vec3(0, 0, -2);
    params.exact_computation_fallback = true;
    const auto reference = replay_bfs_guess(x, displacement, mesh, params, 1);
    EXPECT_EQ(reference.seeds, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(reference.levels, (std::vector<int>{1, 0, 0, 0}));
    EXPECT_GT(reference.first[0].z(), reference.result[0].z() + 0.5);
    for (const bool parallel : {false, true}) {
        params.use_parallel = parallel;
        ColoredCCDGuessTrace trace;
        const auto actual = collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, 1, nullptr, &trace);
        expect_positions_identical(actual, reference.result);
        expect_trace_consistent(trace, x, displacement, actual, 1, true);
        EXPECT_EQ(trace.completed_iterations, 1);
        EXPECT_TRUE(trace.reached_targets);
        for (std::size_t node = 0; node < x.size(); ++node)
            expect_vec_near(actual[node], x[node] + displacement[node], 1e-12);
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, FreeFlightKeepsFixedTargetsAndStopsWhenReached) {
    const auto mesh = ref_mesh_with_masses({1, 1, 1});
    const std::vector<Vec3> x = {Vec3(0, 0, 0), Vec3(4, -2, 8), Vec3(-3, 2, 1)};
    const std::vector<Vec3> displacement = {Vec3(2, 0, -4), Vec3(0, 3, 0), Vec3::Zero()};
    const auto original_x = x, original_displacement = displacement;
    std::vector<Vec3> targets = x;
    for (std::size_t node = 0; node < x.size(); ++node) targets[node] += displacement[node];
    params.gravity = Vec3(0, -9.81, 0); // The helper must not add gravity again.
    for (const int sweeps : {1, 2, 1000000}) {
        SCOPED_TRACE(sweeps);
        expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, sweeps), targets);
    }
    expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
        x, std::vector<Vec3>(x.size(), Vec3::Zero()), mesh, params, 1000000), x);
    expect_positions_identical(x, original_x);
    expect_positions_identical(displacement, original_displacement);
}

TEST_F(CollisionColoredCCDInitialGuessNew, AllRigidProbeHasNoScheduledUpdates) {
    // Level construction must not leave an empty group for the tracer to index.
    const auto x = point_above_triangle();
    auto mesh = point_triangle_mesh();
    mesh.node_to_rb = {0, 0, 0, 0};
    mesh.rb_nodes = {{0, 1, 2, 3}};
    const std::vector<Vec3> displacement(x.size(), Vec3(1, -2, 3));
    for (const bool parallel : {false, true}) {
        params.use_parallel = parallel;
        ColoredCCDGuessTrace trace;
        const auto actual = collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, 1000000, nullptr, &trace);
        expect_positions_identical(actual, x);
        expect_positions_identical(trace.targets, x);
        EXPECT_TRUE(trace.probe_completed);
        EXPECT_TRUE(trace.reached_targets);
        EXPECT_TRUE(trace.stalled);
        EXPECT_EQ(trace.completed_iterations, 1);
        ASSERT_EQ(trace.steps.size(), 3U);
        EXPECT_EQ(trace.steps[0].stage, ColoredCCDGuessTraceStage::Initial);
        EXPECT_EQ(trace.steps[1].stage, ColoredCCDGuessTraceStage::LevelsBuilt);
        EXPECT_EQ(trace.steps[2].stage, ColoredCCDGuessTraceStage::SweepEnd);
        for (const auto& step : trace.steps) {
            EXPECT_TRUE(step.active_vertices.empty());
            expect_positions_identical(step.positions, x);
        }
        for (const int level : trace.level_ids) EXPECT_EQ(level, 0);
        for (const double movement : trace.first_sweep_movement) EXPECT_EQ(movement, 0.0);
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, RetriesStationaryTriangleAndPreservesGap) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    for (const bool exact : {false, true}) {
        SCOPED_TRACE(exact);
        params.exact_computation_fallback = exact;
        displacement[0] = Vec3(0, 0, -2);
        for (const int sweeps : {1, 2, 3}) {
            const auto result = collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, params, sweeps);
            expect_vec_near(result[0], Vec3(0, 0, std::pow(0.1, sweeps)), 1e-12);
            for (int node = 1; node < 4; ++node)
                expect_positions_identical({result[node]}, {x[node]});
        }
        const auto close = collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, 40);
        EXPECT_GE(close[0].z(), 1e-8);
        EXPECT_LT(close[0].z(), 1e-7);
        displacement[0] = Vec3(0, 0, 0.25) - close[0];
        const auto away = collision_colored_ccd_initial_guess_level_based(
            close, displacement, mesh, params, 4);
        expect_positions_identical({away[0]}, {close[0] + displacement[0]});
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, WholeSweepStoppingStillRetriesAStalledEarlyColor) {
    auto x = point_above_triangle();
    x[0].z() = 5e-9;
    const auto mesh = point_triangle_mesh();
    params.exact_computation_fallback = false;
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
        x, displacement, mesh, params, 1000000), x);
    for (int node = 1; node < 4; ++node) displacement[node] = Vec3(4, 0, 0);
    const auto first = collision_colored_ccd_initial_guess_level_based(x, displacement, mesh, params, 1);
    // The probe stalls at the early-colored point, but moves the triangle.
    // Real sweeps start over and visit that newly discovered level-zero
    // triangle first, so even the first real sweep can free the point.
    for (int node = 0; node < 4; ++node)
        expect_vec_near(first[node], x[node] + displacement[node], 1e-12);
    const auto retried = collision_colored_ccd_initial_guess_level_based(x, displacement, mesh, params, 2);
    for (int node = 0; node < 4; ++node)
        expect_vec_near(retried[node], x[node] + displacement[node], 1e-12);
}

TEST_F(CollisionColoredCCDInitialGuessNew, EdgeEdgeRetriesRetainTheSameGapGuard) {
    auto mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1});
    mesh.tris = {0, 1, 4, 2, 3, 5};
    const std::vector<Vec3> x = {Vec3(0, 0, 1), Vec3(1, 0, 0),
        Vec3(0.5, -1, 0), Vec3(0.5, 1, 0), Vec3(-2, 0, 2), Vec3(0.5, 0, -2)};
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    for (const bool exact : {false, true}) {
        params.exact_computation_fallback = exact;
        const auto result = collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, 20);
        const double gap = segment_segment_distance(
            result[0], result[1], result[2], result[3]).distance;
        EXPECT_GE(gap, 1e-8);
        EXPECT_LT(gap, 1e-7);
        for (std::size_t node = 1; node < x.size(); ++node)
            expect_positions_identical({result[node]}, {x[node]});
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, ExactNearMissGroupsPreserveGapAndCanEscape) {
    params.exact_computation_fallback = true;
    for (const int copies : {31, 32, 33}) {
        SCOPED_TRACE(copies);
        RefMesh mesh;
        std::vector<Vec3> x, displacement;
        for (int copy = 0; copy < copies; ++copy) {
            const int base = static_cast<int>(x.size());
            const Vec3 offset(12 * copy, 0, 0);
            for (const auto& point : point_above_triangle()) x.push_back(point + offset);
            mesh.tris.insert(mesh.tris.end(), {base + 1, base + 2, base + 3});
            displacement.insert(displacement.end(), {Vec3(0, 0, 5e-9 - 1.0),
                Vec3::Zero(), Vec3::Zero(), Vec3::Zero()});
        }
        mesh.num_positions = x.size();
        mesh.mass.assign(x.size(), 1.0);
        // Every point shares a level/color. The target does not cross the
        // triangle, so exact endpoint backtracking must enforce the gap floor
        // on both sides of the small-group scheduling threshold.
        const auto close = expect_exact_level_schedules_identical(x, displacement, mesh, params, 40, copies);
        for (int copy = 0; copy < copies; ++copy) {
            SCOPED_TRACE(copy);
            const int base = 4 * copy;
            const double gap = node_triangle_distance(close[base], close[base + 1],
                close[base + 2], close[base + 3]).distance;
            EXPECT_GE(gap, 1e-8);
            EXPECT_LT(gap, 1e-7);
            EXPECT_GT(close[base].z(), (x[base] + displacement[base]).z());
            for (int node = 1; node < 4; ++node)
                expect_positions_identical({close[base + node]}, {x[base + node]});
            displacement[base] = Vec3(close[base].x(), close[base].y(), 2e-8) - close[base];
        }
        const auto escaped = expect_exact_level_schedules_identical(close, displacement, mesh, params, 1);
        for (std::size_t node = 0; node < x.size(); ++node)
            expect_positions_identical({escaped[node]}, {close[node] + displacement[node]});
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, ExactSubFloorLocksAndTinyEscapesMatchReference) {
    params.exact_computation_fallback = true;
    const auto mesh = point_triangle_mesh();
    for (const double gap : {5e-9, 5e-11, 5e-17}) {
        SCOPED_TRACE(gap);
        auto x = point_above_triangle();
        x[0].z() = gap;
        std::vector<Vec3> displacement(x.size(), Vec3::Zero());
        displacement[0] = Vec3(0, 0, 0.5 * gap) - x[0];
        const auto inward = expect_exact_level_schedules_identical(x, displacement, mesh, params, 20);
        expect_positions_identical(inward, x);

        // Exact CCD must permit representable separation even below both the
        // initial-contact and predictor convergence tolerances.
        displacement[0] = Vec3(0, 0, 2 * gap) - x[0];
        const auto outward = expect_exact_level_schedules_identical(x, displacement, mesh, params, 1);
        expect_positions_identical({outward[0]}, {x[0] + displacement[0]});
        EXPECT_GT(outward[0].z(), x[0].z());
        for (std::size_t node = 1; node < x.size(); ++node)
            expect_positions_identical({outward[node]}, {x[node]});
    }
    auto x = point_above_triangle();
    x[0].z() = 5e-9;
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, 1e-15);
    const auto tiny_escape = expect_exact_level_schedules_identical(x, displacement, mesh, params, 1);
    expect_positions_identical({tiny_escape[0]}, {x[0] + displacement[0]});
    EXPECT_GT(tiny_escape[0].z(), x[0].z());
}

TEST_F(CollisionColoredCCDInitialGuessNew, ExactEdgeGapSchedulingMatchesReferenceAndCanEscape) {
    params.exact_computation_fallback = true;
    auto mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1});
    mesh.tris = {0, 1, 4, 2, 3, 5};
    const std::vector<Vec3> x = {Vec3(0, 0, 1), Vec3(1, 0, 0),
        Vec3(0.5, -1, 0), Vec3(0.5, 1, 0), Vec3(-2, 0, 2), Vec3(0.5, 0, -2)};
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    const auto close = expect_exact_level_schedules_identical(x, displacement, mesh, params, 20);
    const double gap = segment_segment_distance(close[0], close[1], close[2], close[3]).distance;
    EXPECT_GE(gap, 1e-8);
    EXPECT_LT(gap, 1e-7);
    EXPECT_GT(close[0].z(), 0.0);
    for (std::size_t node = 1; node < x.size(); ++node)
        expect_positions_identical({close[node]}, {x[node]});
    displacement[0] = Vec3(0, 0, 0.25);
    const auto escaped = expect_exact_level_schedules_identical(close, displacement, mesh, params, 1);
    for (std::size_t node = 0; node < x.size(); ++node)
        expect_positions_identical({escaped[node]}, {close[node] + displacement[node]});
    EXPECT_GT(segment_segment_distance(escaped[0], escaped[1], escaped[2], escaped[3]).distance, gap);
}

TEST_F(CollisionColoredCCDInitialGuessNew, AllTiedRootsPrecedeEarlierColorsAndKeepRigidsFixed) {
    auto mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1, 1, 1, 1});
    mesh.tris = {3, 4, 5, 6, 7, 8};
    mesh.node_to_rb = {-1, -1, -1, -1, -1, -1, 0, 0, 0};
    mesh.rb_nodes = {{6, 7, 8}};
    // Each point is near a different triangle corner. The triangle moves
    // down first only after all earlier-colored points have been clipped.
    // Its three vertices tie for maximum actual first-sweep displacement.
    const std::vector<Vec3> x = {Vec3(-0.97, -0.98, 1.01), Vec3(0.97, -0.98, 1.01),
        Vec3(0, 0.96, 1.01), Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(0, 1, 1),
        Vec3(-4, -3, 0), Vec3(4, -3, 0), Vec3(0, 5, 0)};
    std::vector<Vec3> displacement(x.size(), Vec3(20, 0, 10));
    for (int node = 0; node < 3; ++node)
        displacement[node] = Vec3(0, 0, 0.095 - x[node].z());
    for (int node = 3; node < 6; ++node) displacement[node] = Vec3(0, 0, -2);
    for (const bool exact : {false, true}) {
        params.exact_computation_fallback = exact;
        params.use_parallel = false;
        const auto old_second = collision_colored_ccd_initial_guess(
            x, displacement, mesh, params, 2);
        const auto new_second = collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, 2);
        for (int node = 0; node < 3; ++node) {
            SCOPED_TRACE(::testing::Message() << "exact=" << exact << " node=" << node);
            EXPECT_GT(old_second[node].z(), 0.095 + 1e-3);
            expect_vec_near(new_second[node], x[node] + displacement[node], 1e-12);
        }
        for (int node = 6; node < 9; ++node)
            expect_positions_identical({new_second[node]}, {x[node]});
        params.use_parallel = true;
        expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, 2), new_second);
    }
    params.exact_computation_fallback = true;
    for (const int sweeps : {2, 3, 5}) {
        const auto reference = replay_bfs_guess(x, displacement, mesh, params, sweeps);
        EXPECT_EQ(reference.seeds, (std::vector<int>{3, 4, 5}));
        for (int node = 0; node < 3; ++node) EXPECT_EQ(reference.levels[node], 1);
        expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, sweeps), reference.result);
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, ThreeFallingLayersPropagateThroughMultipleLevels) {
    RefMesh mesh;
    std::vector<Vec3> x;
    // Swept boxes connect only neighboring layers. The bottom layer moves
    // farther, providing unambiguous seeds even if colors interleave layers.
    for (int layer = 0; layer < 3; ++layer) {
        const int base = static_cast<int>(x.size());
        const double z = 2.0 - layer;
        x.insert(x.end(), {Vec3(-1, -1, z), Vec3(1, -1, z), Vec3(0, 1, z)});
        mesh.tris.insert(mesh.tris.end(), {base, base + 1, base + 2});
    }
    mesh.num_positions = x.size();
    mesh.mass.assign(x.size(), 1.0);
    std::vector<Vec3> displacement(x.size(), Vec3(0, 0, -1.5));
    for (int vertex = 6; vertex < 9; ++vertex) displacement[vertex].z() = -2.0;
    params.exact_computation_fallback = true;
    const auto reference = replay_bfs_guess(x, displacement, mesh, params, 2);
    EXPECT_EQ(reference.seeds, (std::vector<int>{6, 7, 8}));
    for (int vertex = 0; vertex < 9; ++vertex)
        EXPECT_EQ(reference.levels[vertex], 2 - vertex / 3);
    const auto result = collision_colored_ccd_initial_guess_level_based(
        x, displacement, mesh, params, 2);
    expect_positions_identical(result, reference.result);
    for (int vertex = 0; vertex < 9; ++vertex)
        expect_vec_near(result[vertex], x[vertex] + displacement[vertex], 1e-12);
}

TEST_F(CollisionColoredCCDInitialGuessNew, DisconnectedComponentsKeepLocalSeedsAndParallelResults) {
    RefMesh mesh;
    std::vector<Vec3> x, displacement;
    constexpr int copies = 48;
    for (int copy = 0; copy < copies; ++copy) {
        const int base = static_cast<int>(x.size());
        for (const auto& point : point_above_triangle()) x.push_back(point + Vec3(12 * copy, 0, 0));
        mesh.tris.insert(mesh.tris.end(), {base + 1, base + 2, base + 3});
        displacement.push_back(Vec3(0, 0, -2));
        for (int node = 0; node < 3; ++node)
            displacement.push_back(copy % 2 ? Vec3(4, 0, 0) : Vec3::Zero());
    }
    mesh.num_positions = x.size();
    mesh.mass.assign(x.size(), 1.0);
    // Components alternate first-sweep maxima .9 and 4. Using only a global
    // maximum would leave the stationary-obstacle components without seeds.
    BroadPhase scratch;
    for (const bool exact : {false, true}) {
        params.exact_computation_fallback = exact;
        params.use_parallel = false;
        const auto serial = collision_colored_ccd_initial_guess_level_based(x, displacement, mesh, params, 3);
        for (int copy = 0; copy < copies; ++copy)
            EXPECT_NEAR(serial[4 * copy].z(), copy % 2 ? -1.0 : 0.001, 1e-12);
        params.use_parallel = true;
        for (const int threads : {1, 4}) {
            omp_set_num_threads(threads);
            // Replace the optional cache with incompatible CCD-only data to
            // ensure each invocation rebuilds contacts before its first sweep.
            scratch.build_ccd_candidates(x, displacement, mesh, 0.01, false);
            expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, params, 3, &scratch), serial);
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, SolidInteriorNodesAreNotCollisionSurfacePoints) {
    InitialGuessSolidScene scene;
    const auto x = scene.state.deformed_positions;
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[4] = Vec3(2, 0, 0);
    const auto result = collision_colored_ccd_initial_guess_level_based(
        x, displacement, scene.mesh, params, 5);
    expect_positions_identical({result[4]}, {x[4] + displacement[4]});
    for (int node = 0; node < 4; ++node)
        expect_positions_identical({result[node]}, {x[node]});
}

TEST_F(CollisionColoredCCDInitialGuessNew, PreservesInputValidationForZeroAndPositiveCaps) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    const std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    EXPECT_THROW(collision_colored_ccd_initial_guess_level_based(
        x, displacement, mesh, params, -1), std::invalid_argument);
    for (const int sweeps : {0, 2}) {
        EXPECT_THROW(collision_colored_ccd_initial_guess_level_based(
            x, {}, mesh, params, sweeps), std::invalid_argument);
        auto bad_displacement = displacement;
        bad_displacement[0].z() = std::numeric_limits<double>::infinity();
        EXPECT_THROW(collision_colored_ccd_initial_guess_level_based(
            x, bad_displacement, mesh, params, sweeps), std::invalid_argument);
        auto bad_mesh = mesh;
        bad_mesh.node_to_rb = {-1};
        EXPECT_THROW(collision_colored_ccd_initial_guess_level_based(
            x, displacement, bad_mesh, params, sweeps), std::invalid_argument);
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, TwoSquareTraceRecordsOnePhysicalStepWithoutChangingResults) {
    auto mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1, 1, 1});
    mesh.tris = {0, 2, 1, 0, 3, 2, 4, 6, 5, 4, 7, 6};
    // Historical visualization geometry: side-2 upper square at y=1.05,
    // side-1 lower square at y=1.00. These sweeps retry ONE gravity step.
    const std::vector<Vec3> x = {Vec3(-1, 1.05, -1), Vec3(1, 1.05, -1),
        Vec3(1, 1.05, 1), Vec3(-1, 1.05, 1), Vec3(-0.5, 1, -0.5),
        Vec3(0.5, 1, -0.5), Vec3(0.5, 1, 0.5), Vec3(-0.5, 1, 0.5)};
    params.fps = 5.0;
    params.substeps = 1;
    params.gravity = Vec3(0, -9.81, 0);
    const std::vector<Vec3> displacement(x.size(), params.dt2() * params.gravity);
    EXPECT_NEAR(displacement[0].y(), -0.3924, 1e-15);
    for (const bool exact : {false, true}) {
        SCOPED_TRACE(exact);
        params.exact_computation_fallback = exact;
        ColoredCCDGuessTrace serial_trace;
        for (const bool parallel : {false, true}) {
            params.use_parallel = parallel;
            const auto plain = collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, params, 10);
            BroadPhase scratch;
            ColoredCCDGuessTrace trace;
            const auto traced = collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, params, 10, &scratch, &trace);
            expect_positions_identical(traced, plain);
            expect_trace_consistent(trace, x, displacement, traced, 10, true);
            EXPECT_GE(trace.completed_iterations, 1);
            EXPECT_LT(trace.completed_iterations, 10);
            EXPECT_TRUE(trace.reached_targets);
            EXPECT_FALSE(trace.stalled);
            const auto first = collision_colored_ccd_initial_guess(
                x, displacement, mesh, params, 1);
            for (std::size_t node = 0; node < x.size(); ++node)
                EXPECT_DOUBLE_EQ(trace.first_sweep_movement[node],
                    (first[node] - x[node]).stableNorm());
            ASSERT_GE(trace.steps.size(), 2U);
            EXPECT_EQ(trace.steps[1].stage, ColoredCCDGuessTraceStage::LevelsBuilt);
            expect_positions_identical(trace.steps[1].positions, x);
            if (parallel) expect_traces_identical(trace, serial_trace);
            else serial_trace = trace;
        }
        // The original function's optional trace must remain observational too.
        ColoredCCDGuessTrace old_trace;
        const auto old_plain = collision_colored_ccd_initial_guess(
            x, displacement, mesh, params, 10);
        const auto old_traced = collision_colored_ccd_initial_guess(
            x, displacement, mesh, params, 10, nullptr, &old_trace);
        expect_positions_identical(old_traced, old_plain);
        expect_trace_consistent(old_trace, x, displacement, old_traced, 10, false);
        EXPECT_EQ(old_trace.color_ids, serial_trace.color_ids);
        if (exact) {
            const auto reference = replay_bfs_guess(x, displacement, mesh, params, 10);
            EXPECT_EQ(serial_trace.level_ids, reference.levels);
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, ThreeSquareTracePreservesTheNewUpperSheetAndFixedTargets) {
    auto mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    mesh.tris = {0, 2, 1, 0, 3, 2, 4, 6, 5, 4, 7, 6, 8, 10, 9, 8, 11, 10};
    // Preserve the historical square IDs and add a side-1 sheet above the
    // existing side-2 sheet, with the same 0.05 vertical gap on each side.
    const std::vector<Vec3> x = {Vec3(-1, 1.05, -1), Vec3(1, 1.05, -1),
        Vec3(1, 1.05, 1), Vec3(-1, 1.05, 1), Vec3(-0.5, 1, -0.5),
        Vec3(0.5, 1, -0.5), Vec3(0.5, 1, 0.5), Vec3(-0.5, 1, 0.5),
        Vec3(-0.5, 1.10, -0.5), Vec3(0.5, 1.10, -0.5),
        Vec3(0.5, 1.10, 0.5), Vec3(-0.5, 1.10, 0.5)};
    params.fps = 5.0;
    params.substeps = 1;
    params.gravity = Vec3(0, -9.81, 0);
    const std::vector<Vec3> displacement(x.size(), params.dt2() * params.gravity);
    const auto original_x = x;
    const auto original_displacement = displacement;
    for (const bool exact : {false, true}) {
        SCOPED_TRACE(exact);
        params.exact_computation_fallback = exact;
        ColoredCCDGuessTrace serial_trace;
        for (const bool parallel : {false, true}) {
            SCOPED_TRACE(parallel);
            params.use_parallel = parallel;
            const auto plain = collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, params, 10);
            BroadPhase scratch;
            ColoredCCDGuessTrace trace;
            const auto traced = collision_colored_ccd_initial_guess_level_based(
                x, displacement, mesh, params, 10, &scratch, &trace);
            expect_positions_identical(traced, plain);
            expect_positions_identical(x, original_x);
            expect_positions_identical(displacement, original_displacement);
            expect_trace_consistent(trace, x, displacement, traced, 10, true);
            EXPECT_TRUE(trace.reached_targets);
            EXPECT_FALSE(trace.stalled);
            EXPECT_LT(trace.completed_iterations, 10);
            for (std::size_t node = 0; node < x.size(); ++node)
                expect_vec_near(traced[node], x[node] + displacement[node], 1e-14);
            const auto first = collision_colored_ccd_initial_guess(
                x, displacement, mesh, params, 1);
            for (std::size_t node = 0; node < x.size(); ++node)
                EXPECT_DOUBLE_EQ(trace.first_sweep_movement[node],
                    (first[node] - x[node]).stableNorm());
            ASSERT_GE(trace.steps.size(), 2U);
            EXPECT_EQ(trace.steps[1].stage, ColoredCCDGuessTraceStage::LevelsBuilt);
            expect_positions_identical(trace.steps[1].positions, x);
            if (parallel) expect_traces_identical(trace, serial_trace);
            else serial_trace = trace;
        }
        if (exact) {
            const auto reference = replay_bfs_guess(x, displacement, mesh, params, 10);
            EXPECT_EQ(serial_trace.level_ids, reference.levels);
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, TraceSeparatesTheSameColorAcrossDifferentLevels) {
    auto mesh = ref_mesh_with_masses({1, 1, 1, 1, 1, 1, 1, 1});
    mesh.tris = {1, 2, 3, 5, 6, 7};
    auto x = point_above_triangle();
    for (const auto& point : point_above_triangle()) x.push_back(point + Vec3(12, 0, 0));
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = displacement[4] = Vec3(0, 0, -2);
    for (int node = 5; node < 8; ++node) displacement[node] = Vec3(4, 0, 0);
    ColoredCCDGuessTrace trace;
    const auto result = collision_colored_ccd_initial_guess_level_based(
        x, displacement, mesh, params, 3, nullptr, &trace);
    expect_trace_consistent(trace, x, displacement, result, 3, true);
    ASSERT_EQ(trace.color_ids[0], trace.color_ids[4]);
    EXPECT_EQ(trace.level_ids[0], 0);
    EXPECT_EQ(trace.level_ids[4], 1);
    bool saw_level_zero = false, saw_level_one = false;
    for (const auto& step : trace.steps) {
        if (step.stage != ColoredCCDGuessTraceStage::ColorUpdate || step.iteration != 1
            || step.active_color != trace.color_ids[0]) continue;
        if (step.active_level == 0) {
            saw_level_zero = true;
            EXPECT_EQ(step.active_vertices, (std::vector<int>{0}));
        } else if (step.active_level == 1) {
            saw_level_one = true;
            EXPECT_EQ(step.active_vertices, (std::vector<int>{4}));
        }
    }
    EXPECT_TRUE(saw_level_zero);
    EXPECT_TRUE(saw_level_one);
}

TEST_F(CollisionColoredCCDInitialGuessNew, SchedulingThresholdPreservesTracedAndUntracedResults) {
    struct Scene {
        std::vector<Vec3> x, displacement;
        RefMesh mesh;
        int expected_point_group = 0;
    };
    std::vector<Scene> scenes;
    scenes.push_back({point_above_triangle(),
        {Vec3(0, 0, -2), Vec3(4, 0, 0), Vec3(4, 0, 0), Vec3(4, 0, 0)},
        point_triangle_mesh()});
    // Exercise both sides of the small-group scheduling threshold, including
    // unequal contact costs and stable heavy-first ties within the point group.
    for (const int copies : {31, 32, 33}) {
        const UnevenContactGuessScene uneven(copies);
        scenes.push_back({uneven.x, uneven.displacement, uneven.mesh, copies});
    }
    auto rigid_mesh = point_triangle_mesh();
    rigid_mesh.node_to_rb = {-1, 0, 0, 0};
    rigid_mesh.rb_nodes = {{1, 2, 3}};
    scenes.push_back({point_above_triangle(),
        {Vec3(0, 0, -2), Vec3(1, -2, 3), Vec3(1, -2, 3), Vec3(1, -2, 3)},
        rigid_mesh});
    scenes.push_back({{Vec3(0, 0, 0), Vec3(5, 2, 1)},
        {Vec3(0, -0.5, 0), Vec3::Zero()}, ref_mesh_with_masses({1, 1})});

    for (const bool exact : {false, true}) {
        params.exact_computation_fallback = exact;
        for (const int cap : {0, 1, 6}) {
            for (std::size_t scene_id = 0; scene_id < scenes.size(); ++scene_id) {
                SCOPED_TRACE(::testing::Message() << "exact=" << exact
                    << " cap=" << cap << " scene=" << scene_id);
                const auto& scene = scenes[scene_id];
                params.use_parallel = false;
                ColoredCCDGuessTrace serial_trace;
                const auto baseline = collision_colored_ccd_initial_guess_level_based(
                    scene.x, scene.displacement, scene.mesh, params, cap, nullptr, &serial_trace);
                if (cap > 0 && scene.expected_point_group > 0) {
                    const auto point_group = std::find_if(serial_trace.steps.begin(),
                        serial_trace.steps.end(), [&](const ColoredCCDGuessTraceStep& step) {
                            return step.stage == ColoredCCDGuessTraceStage::ColorUpdate
                                && step.iteration == 1 && step.active_level == 0
                                && step.active_color == serial_trace.color_ids[0];
                        });
                    ASSERT_NE(point_group, serial_trace.steps.end());
                    EXPECT_EQ(point_group->active_vertices.size(), scene.expected_point_group);
                }
                if (cap == 1) {
                    // Stable level scattering must retain each trial color's
                    // original order, including exact-mode cost ties.
                    ColoredCCDGuessTrace trial_trace;
                    collision_colored_ccd_initial_guess(scene.x, scene.displacement,
                        scene.mesh, params, 1, nullptr, &trial_trace);
                    for (const auto& step : serial_trace.steps) {
                        if (step.stage != ColoredCCDGuessTraceStage::ColorUpdate) continue;
                        std::vector<int> expected;
                        for (const auto& trial_step : trial_trace.steps) {
                            if (trial_step.stage != ColoredCCDGuessTraceStage::ColorUpdate
                                || trial_step.active_color != step.active_color) continue;
                            for (const int vertex : trial_step.active_vertices)
                                if (serial_trace.level_ids[vertex] == step.active_level)
                                    expected.push_back(vertex);
                        }
                        EXPECT_EQ(step.active_vertices, expected);
                    }
                }
                // The production path does not request a trace. Keep this
                // comparison explicit so trace-only scheduling cannot mask it.
                expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
                    scene.x, scene.displacement, scene.mesh, params, cap), baseline);
                params.use_parallel = true;
                for (const int threads : {1, 4}) {
                    SCOPED_TRACE(threads);
                    omp_set_num_threads(threads);
                    ColoredCCDGuessTrace trace;
                    const auto traced = collision_colored_ccd_initial_guess_level_based(
                        scene.x, scene.displacement, scene.mesh, params, cap, nullptr, &trace);
                    expect_positions_identical(traced, baseline);
                    expect_traces_identical(trace, serial_trace);
                    BroadPhase scratch;
                    for (int repeat = 0; repeat < 2; ++repeat) {
                        SCOPED_TRACE(repeat);
                        expect_positions_identical(collision_colored_ccd_initial_guess_level_based(
                            scene.x, scene.displacement, scene.mesh, params, cap, &scratch, nullptr),
                            baseline);
                    }
                }
            }
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, AlternatingPredictorsRebuildSharedWorkspaceAndScratch) {
    struct Input {
        std::vector<Vec3> x, displacement;
        RefMesh mesh;
    };
    const UnevenContactGuessScene uneven;
    std::vector<Vec3> point_displacement(4, Vec3(4, 0, 0));
    point_displacement[0] = Vec3(0, 0, -2);
    std::vector<Input> inputs = {
        {point_above_triangle(), point_displacement, point_triangle_mesh()},
        {point_above_triangle(), point_displacement, ref_mesh_with_masses({1, 1, 1, 1})},
        {uneven.x, uneven.displacement, uneven.mesh},
        {{Vec3(0, 0, 0), Vec3(5, 2, 1)},
            {Vec3(0, -0.5, 0), Vec3(1, 0, 0)}, ref_mesh_with_masses({1, 1})},
        {{}, {}, RefMesh{}}
    };
    struct Request {
        int input;
        bool bfs;
        int cap;
        bool exact;
    };
    const std::vector<Request> requests = {
        {2, true, 3, true}, {0, false, 1, false},
        {0, true, 1, true}, {1, false, 3, true},
        {2, true, 1, false}, {3, false, 0, false},
        {0, true, 0, true}, {4, false, 0, false},
        {2, true, 3, true}, {1, false, 1, false},
        {4, true, 3, true}, {3, false, 3, true},
        {0, true, 3, false}, {2, false, 3, false},
        {1, true, 1, true}, {0, false, 3, true}
    };
    const auto run = [&](const Request& request, BroadPhase& scratch,
                         ColoredCCDGuessTrace& trace) {
        const auto& input = inputs[request.input];
        params.exact_computation_fallback = request.exact;
        const auto predictor = request.bfs ? collision_colored_ccd_initial_guess_level_based
                                          : collision_colored_ccd_initial_guess;
        return predictor(input.x, input.displacement, input.mesh,
            params, request.cap, &scratch, &trace);
    };
    // Save clean-cache baselines before interleaving calls. Computing each
    // baseline just before its reused call could hide stale shared scheduling.
    std::vector<std::vector<Vec3>> expected;
    std::vector<ColoredCCDGuessTrace> expected_traces(requests.size());
    params.use_parallel = false;
    for (std::size_t index = 0; index < requests.size(); ++index) {
        BroadPhase fresh;
        expected.push_back(run(requests[index], fresh, expected_traces[index]));
    }
    BroadPhase scratch;
    ColoredCCDGuessTrace trace;
    for (const bool parallel : {false, true}) {
        params.use_parallel = parallel;
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (std::size_t index = 0; index < requests.size(); ++index) {
                SCOPED_TRACE(::testing::Message() << "parallel=" << parallel
                    << " repeat=" << repeat << " request=" << index);
                const auto& request = requests[index];
                const auto& input = inputs[request.input];
                const auto actual = run(request, scratch, trace);
                expect_positions_identical(actual, expected[index]);
                expect_traces_identical(trace, expected_traces[index]);
                expect_trace_consistent(trace, input.x, input.displacement,
                    actual, request.cap, request.bfs);
            }
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, ExactLevelGroupsRetainHeavyFirstOrderAndIndependentResults) {
    const UnevenContactGuessScene scene;
    params.exact_computation_fallback = true;
    const auto reference = replay_bfs_guess(scene.x, scene.displacement, scene.mesh, params, 3);
    ColoredCCDGuessTrace serial_trace;
    for (const bool parallel : {false, true}) {
        SCOPED_TRACE(parallel);
        params.use_parallel = parallel;
        BroadPhase scratch;
        ColoredCCDGuessTrace trace;
        const auto actual = collision_colored_ccd_initial_guess_level_based(
            scene.x, scene.displacement, scene.mesh, params, 3, &scratch, &trace);
        expect_positions_identical(actual, reference.result);
        EXPECT_EQ(trace.level_ids, reference.levels);
        expect_trace_consistent(trace, scene.x, scene.displacement, actual, 3, true);
        if (parallel) expect_traces_identical(trace, serial_trace);
        else serial_trace = trace;

        const auto& cache = scratch.cache();
        const auto cost = [&](int vertex) {
            return cache.vertex_nt[vertex].size() + cache.vertex_ss[vertex].size();
        };
        bool saw_unequal_costs = false, saw_tied_costs = false;
        for (const auto& step : trace.steps) {
            if (step.stage != ColoredCCDGuessTraceStage::ColorUpdate) continue;
            SCOPED_TRACE(::testing::Message() << "iteration=" << step.iteration
                << " level=" << step.active_level << " color=" << step.active_color);
            for (std::size_t index = 1; index < step.active_vertices.size(); ++index) {
                const int previous = step.active_vertices[index - 1];
                const int current = step.active_vertices[index];
                EXPECT_GE(cost(previous), cost(current));
                if (cost(previous) == cost(current)) {
                    saw_tied_costs = true;
                    EXPECT_LT(previous, current);
                } else {
                    saw_unequal_costs = true;
                }
            }
        }
        EXPECT_TRUE(saw_unequal_costs);
        EXPECT_TRUE(saw_tied_costs);
    }
}

TEST_F(CollisionColoredCCDInitialGuessNew, ReusedTraceClearsScheduleAndEventsOnShortRuns) {
    const auto x = point_above_triangle();
    const auto mesh = point_triangle_mesh();
    std::vector<Vec3> displacement(x.size(), Vec3::Zero());
    displacement[0] = Vec3(0, 0, -2);
    ColoredCCDGuessTrace trace;
    collision_colored_ccd_initial_guess_level_based(x, displacement, mesh, params, 3, nullptr, &trace);
    ASSERT_GT(trace.steps.size(), 1U);
    ASSERT_TRUE(std::any_of(trace.level_ids.begin(), trace.level_ids.end(),
        [](int level) { return level >= 0; }));
    for (const int cap : {0, 1}) {
        const auto result = collision_colored_ccd_initial_guess_level_based(
            x, displacement, mesh, params, cap, nullptr, &trace);
        expect_trace_consistent(trace, x, displacement, result, cap, true);
        EXPECT_EQ(trace.completed_iterations, cap);
        EXPECT_FALSE(trace.reached_targets);
        EXPECT_FALSE(trace.stalled);
        if (cap == 0) {
            EXPECT_FALSE(trace.probe_completed);
            EXPECT_EQ(trace.steps.size(), 1U);
            for (const int level : trace.level_ids) EXPECT_EQ(level, -1);
            for (const int color : trace.color_ids) EXPECT_EQ(color, -1);
            for (const double movement : trace.first_sweep_movement) EXPECT_EQ(movement, 0.0);
        } else {
            EXPECT_TRUE(trace.probe_completed);
            EXPECT_EQ(trace.level_ids, (std::vector<int>{0, 1, 1, 1}));
        }
    }
    const auto free_mesh = ref_mesh_with_masses({1, 1});
    const std::vector<Vec3> free_x = {Vec3(0, 0, 0), Vec3(5, 2, 1)};
    std::vector<Vec3> free_displacement(2, Vec3(0, -0.5, 0));
    for (const bool stationary : {false, true}) {
        if (stationary) free_displacement.assign(2, Vec3::Zero());
        const auto result = collision_colored_ccd_initial_guess_level_based(
            free_x, free_displacement, free_mesh, params, 1000000, nullptr, &trace);
        expect_trace_consistent(trace, free_x, free_displacement, result, 1000000, true);
        EXPECT_EQ(trace.completed_iterations, 1);
        // A probe which already reaches every target (or starts stationary)
        // still has to reset and perform the real sweep from the input.
        EXPECT_EQ(trace.steps.size(), 4U); // initial, levels, color, sweep end
        EXPECT_TRUE(trace.probe_completed);
        EXPECT_TRUE(trace.reached_targets);
        EXPECT_EQ(trace.stalled, stationary);
        EXPECT_EQ(trace.level_ids, (std::vector<int>{0, 0}));
        expect_positions_identical(trace.steps[1].positions, free_x);
        for (const double movement : trace.first_sweep_movement)
            EXPECT_DOUBLE_EQ(movement, stationary ? 0.0 : 0.5);
    }
}

TEST(CCDInitialGuess, ReturnsTargetWhenNoCollisionCandidates) {
    RefMesh ref_mesh = ref_mesh_with_masses({1.0, 1.0});
    ref_mesh.tris.clear();

    std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
    };
    std::vector<Vec3> xhat = {
        Vec3(0.0, 1.0, 0.0),
        Vec3(1.0, 1.0, 0.0),
    };

    const std::vector<Vec3> guess = ccd_initial_guess(x, xhat, ref_mesh);

    ASSERT_EQ(guess.size(), xhat.size());
    for (int i = 0; i < static_cast<int>(xhat.size()); ++i) {
        expect_vec_near(guess[i], xhat[i], 1e-12);
    }
}

TEST(VerletInitialGuess, AddsGravityAndReturnsCollisionFreeTarget) {
    SimParams params = base_params();
    params.gravity = Vec3(0.0, -4.0, 2.0);

    RefMesh ref_mesh = ref_mesh_with_masses({1.0, 1.0});
    ref_mesh.tris.clear();

    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 2.0, 3.0),
    };
    const std::vector<Vec3> xhat = {
        Vec3(0.5, 1.0, -0.5),
        Vec3(1.5, 3.0, 2.5),
    };

    const std::vector<Vec3> guess = verlet_initial_guess(x, xhat, ref_mesh, params);
    const Vec3 dt2g = params.dt2() * params.gravity;

    ASSERT_EQ(guess.size(), xhat.size());
    for (int i = 0; i < static_cast<int>(xhat.size()); ++i) {
        expect_vec_near(guess[i], xhat[i] + dt2g, 1e-12);
    }
}

TEST(TranslationInitialGuess, MatchesMassWeightedInertiaAndGravityClosedForm) {
    SimParams params = base_params();
    params.gravity = Vec3(0.0, -4.0, 2.0);

    RefMesh ref_mesh = ref_mesh_with_masses({2.0, 1.0, 3.0});
    std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 2.0, 0.0),
        Vec3(-1.0, 0.5, 3.0),
    };
    std::vector<Vec3> xhat = {
        x[0] + Vec3(1.0, 0.0, 0.0),
        x[1] + Vec3(0.0, 2.0, 0.0),
        x[2] + Vec3(0.0, 0.0, -1.0),
    };

    const std::vector<Vec3> guess = translation_initial_guess(x, xhat, ref_mesh, {}, params);

    // Inertia gives (2, 2, -3) / 6 = (1/3, 1/3, -1/2).
    // With dt = 1/2, gravity contributes dt^2 * g = (0, -1, 1/2).
    const Vec3 expected_C(1.0 / 3.0, -2.0 / 3.0, 0.0);

    ASSERT_EQ(guess.size(), x.size());
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        expect_vec_near(guess[i], x[i] + expected_C, 1e-12);
    }
}

TEST(TranslationInitialGuess, IncludesPinSpringsInClosedFormTranslation) {
    SimParams params = base_params();
    params.kpin = 20.0;

    RefMesh ref_mesh = ref_mesh_with_masses({2.0, 3.0});
    std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
    };
    std::vector<Vec3> xhat = {
        x[0] + Vec3(2.0, 0.0, 0.0),
        x[1] + Vec3(2.0, 0.0, 0.0),
    };
    std::vector<Pin> pins = {
        Pin{1, x[1] + Vec3(0.0, 4.0, 0.0)},
    };

    const std::vector<Vec3> guess = translation_initial_guess(x, xhat, ref_mesh, pins, params);

    const Vec3 expected_C(1.0, 2.0, 0.0);
    ASSERT_EQ(guess.size(), x.size());
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        expect_vec_near(guess[i], x[i] + expected_C, 1e-12);
    }
}

TEST(TranslationInitialGuess, AppliesOneNewtonCorrectionForPlaneSDF) {
    SimParams params = SimParams::zeros();
    params.fps = 1.0;
    params.substeps = 1;
    params.k_sdf = 10.0;
    params.eps_sdf = 0.0;
    params.gravity = Vec3::Zero();
    params.sdf_planes.push_back({Vec3::Zero(), Vec3::UnitY()});

    RefMesh ref_mesh = ref_mesh_with_masses({1.0, 1.0, 1.0});
    std::vector<Vec3> x = {
        Vec3(0.0, -0.1, 0.0),
        Vec3(1.0, -0.3, 0.0),
        Vec3(2.0,  0.2, 0.0),
    };
    std::vector<Vec3> xhat = x;

    const std::vector<Vec3> guess = translation_initial_guess(x, xhat, ref_mesh, {}, params);

    // Two active vertices have total penetration 0.4. With dt = 1 and k = 10:
    // C_y = 10 * 0.4 / (3 + 10 * 2) = 4/23.
    const Vec3 expected_C(0.0, 4.0 / 23.0, 0.0);

    ASSERT_EQ(guess.size(), x.size());
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        expect_vec_near(guess[i], x[i] + expected_C, 1e-12);
    }
}
