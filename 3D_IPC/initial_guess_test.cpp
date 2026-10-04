#include "initial_guess.h"
#include "broad_phase.h"
#include "ipc_args.h"
#include "mesh_utils.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"
#include "simulation.h"
#include "solid_ipc.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <cmath>
#include <cstring>
#include <limits>
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

void expect_positions_identical(
    const std::vector<Vec3>& actual, const std::vector<Vec3>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t vertex = 0; vertex < actual.size(); ++vertex) {
        SCOPED_TRACE(vertex);
        EXPECT_EQ(std::memcmp(actual[vertex].data(), expected[vertex].data(),
                             3 * sizeof(double)), 0);
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

    SolverResult advance(SubstepCallback callback = nullptr) {
        return advance_one_frame(state, mesh, adjacency, pins, params,
            broad_phase, 1, nullptr, callback);
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

TEST(ColoredCCDGuessParameters, DefaultsAndCliExposeAnOptInIterationCount) {
    const auto zero = SimParams::zeros();
    EXPECT_FALSE(zero.use_colored_ccd_guess);
    EXPECT_EQ(zero.colored_ccd_guess_iters, 10);
    IPCArgs3D defaults;
    EXPECT_FALSE(defaults.to_sim_params().use_colored_ccd_guess);
    EXPECT_EQ(defaults.to_sim_params().colored_ccd_guess_iters, 10);

    IPCArgs3D selected;
    ASSERT_TRUE(parse_initial_guess_arguments(selected, {"3D_sim",
        "--use_colored_ccd_guess", "true", "--colored_ccd_guess_iters", "3"}));
    EXPECT_TRUE(selected.to_sim_params().use_colored_ccd_guess);
    EXPECT_EQ(selected.to_sim_params().colored_ccd_guess_iters, 3);
    EXPECT_TRUE(selected.to_sim_params().use_ccd_guess);

    IPCArgs3D zero_sweeps;
    ASSERT_TRUE(parse_initial_guess_arguments(zero_sweeps, {"3D_sim",
        "--use_colored_ccd_guess", "--colored_ccd_guess_iters", "0"}));
    EXPECT_TRUE(zero_sweeps.to_sim_params().use_colored_ccd_guess);
    EXPECT_EQ(zero_sweeps.to_sim_params().colored_ccd_guess_iters, 0);

    IPCArgs3D invalid;
    ASSERT_TRUE(parse_initial_guess_arguments(invalid, {"3D_sim",
        "--use_colored_ccd_guess", "true", "--colored_ccd_guess_iters", "-1"}));
    EXPECT_THROW(invalid.to_sim_params(), std::invalid_argument);
}

TEST(ColoredCCDGuessParameters, ValidatesEnabledSettingsWithoutChangingLegacyContract) {
    auto params = SimParams::zeros();
    params.use_colored_ccd_guess = true;
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
    params.colored_ccd_guess_iters = -1;
    EXPECT_NO_THROW(params.validate_colored_ccd_guess_parameters());
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

TEST_F(CollisionColoredCCDInitialGuess, ClothInitialGuessUsesXhatWithoutAddingGravity) {
    InitialGuessClothScene scene;
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

TEST_F(CollisionColoredCCDInitialGuess, ClothInitialGuessUsesSelectedSweepsAndTakesPrecedence) {
    for (bool parallel : {false, true}) {
        for (int sweeps : {0, 1, 2, 3}) {
            SCOPED_TRACE(::testing::Message() << "parallel=" << parallel << " sweeps=" << sweeps);
            InitialGuessClothScene scene;
            scene.params.use_parallel = parallel;
            scene.params.colored_ccd_guess_iters = sweeps;
            // Both old options remain on deliberately: the opt-in colored
            // guess must win, including the meaningful zero-sweep case.
            scene.params.use_ccd_guess = true;
            scene.params.use_verlet_guess = true;
            scene.params.gravity = Vec3(0, -9.81, 0);
            scene.state.velocities[0] = Vec3(0, 0, -4);
            for (std::size_t i = 1; i < 4; ++i)
                scene.state.velocities[i] = Vec3(0, 0, -2);
            const auto before = scene.state.deformed_positions;
            std::vector<Vec3> xhat, displacement(before.size());
            build_xhat(xhat, before, scene.state.velocities, scene.params.dt());
            for (std::size_t i = 0; i < before.size(); ++i)
                displacement[i] = xhat[i] - before[i];
            const auto expected = collision_colored_ccd_initial_guess(
                before, displacement, scene.mesh, scene.params, sweeps);
            ASSERT_TRUE(scene.advance().converged);
            expect_positions_identical(scene.state.deformed_positions, expected);
            for (std::size_t i = 0; i < before.size(); ++i)
                expect_vec_near(scene.state.velocities[i],
                    (expected[i] - before[i]) / scene.params.dt(), 0.0);
        }
    }
}

TEST_F(CollisionColoredCCDInitialGuess, ClothInitialGuessRebuildsTargetsFromEachLiveSubstep) {
    InitialGuessClothScene scene;
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
        expected.deformed_positions = collision_colored_ccd_initial_guess(
            previous, displacement, scene.mesh, scene.params, 1);
        update_velocity(expected.velocities, expected.deformed_positions, previous,
            scene.params.dt());
        expect_positions_identical(actual, expected.deformed_positions);
    }).converged);
    EXPECT_EQ(observed, 3);
    expect_positions_identical(scene.state.velocities, expected.velocities);
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

TEST_F(CollisionColoredCCDInitialGuess, OgcRetainsPrecedenceOverTheClothOption) {
    InitialGuessClothScene scene;
    scene.params.use_ogc = true;
    scene.state.velocities.assign(4, Vec3(0.25, 0, 0));
    const auto before = scene.state.deformed_positions;
    ASSERT_TRUE(scene.advance().converged);
    expect_positions_identical(scene.state.deformed_positions, before);
}

TEST_F(CollisionColoredCCDInitialGuess, ClothDriverRejectsRigidAndTetMeshesBeforeMutation) {
    for (bool rigid : {false, true}) {
        InitialGuessClothScene scene;
        const auto before = scene.state.deformed_positions;
        if (rigid) scene.mesh.rb_nodes = {{1, 2, 3}};
        else scene.mesh.tets = {0, 1, 2, 3};
        EXPECT_THROW(scene.advance(), std::invalid_argument);
        expect_positions_identical(scene.state.deformed_positions, before);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, ClothOptionIsSharedByEveryClothSolverDispatch) {
    for (int variant : {0, 1, 2, 3}) {
        SCOPED_TRACE(variant);
        InitialGuessClothScene scene;
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
        const auto expected = collision_colored_ccd_initial_guess(
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

TEST_F(CollisionColoredCCDInitialGuess, GeneralInitialGuessUsesXhatForBothSolverVariants) {
    for (bool v2 : {false, true}) {
        for (bool mixed : {false, true}) {
            for (int sweeps : {0, 2}) {
                SCOPED_TRACE(::testing::Message() << "v2=" << v2
                    << " mixed=" << mixed << " sweeps=" << sweeps);
                InitialGuessSolidScene scene(mixed);
                scene.params.use_basic_experimental_v2 = v2;
                scene.params.use_simd = v2;
                scene.params.use_parallel = true;
                scene.params.use_ccd_guess = true;
                scene.params.use_verlet_guess = true;
                scene.params.gravity = Vec3(0, -9.81, 0);
                scene.params.colored_ccd_guess_iters = sweeps;
                for (int node = 0; node < 5; ++node)
                    scene.state.velocities[node] = Vec3(0.02, 0.04, -0.02);
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

TEST_F(CollisionColoredCCDInitialGuess, GeneralInitialGuessRebuildsLiveSubstepTargets) {
    for (bool v2 : {false, true}) {
        InitialGuessSolidScene scene;
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
            expected.deformed_positions = collision_colored_ccd_initial_guess(
                previous, displacement, scene.mesh, scene.params, 1);
            update_velocity(expected.velocities, expected.deformed_positions,
                previous, scene.params.dt());
            expect_positions_identical(actual, expected.deformed_positions);
        }).converged);
        EXPECT_EQ(observed, 3);
        expect_positions_identical(scene.state.velocities, expected.velocities);
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

TEST_F(CollisionColoredCCDInitialGuess, OgcRetainsPrecedenceInGeneralDriver) {
    for (bool solver_ogc : {false, true}) {
        InitialGuessSolidScene scene;
        scene.params.use_ogc = !solver_ogc;
        scene.params.use_ogc_solver = solver_ogc;
        scene.state.velocities.assign(5, Vec3(0.02, 0.04, 0));
        const auto before = scene.state.deformed_positions;
        ASSERT_TRUE(scene.advance().converged);
        expect_positions_identical(scene.state.deformed_positions, before);
    }
}

TEST_F(CollisionColoredCCDInitialGuess, GeneralDriverValidatesBeforeMutation) {
    InitialGuessSolidScene scene(true);
    const auto before = scene.state.deformed_positions;
    const auto before_velocity = scene.state.velocities;
    int pin_updates = 0;
    const auto updater = [&](std::vector<Pin>&, double) { ++pin_updates; };
    scene.params.colored_ccd_guess_iters = -1;
    EXPECT_THROW(scene.advance(nullptr, updater), std::invalid_argument);
    scene.params.colored_ccd_guess_iters = 1;
    scene.params.d_hat = -0.001;
    EXPECT_THROW(scene.advance(nullptr, updater), std::invalid_argument);
    EXPECT_EQ(pin_updates, 0);
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
