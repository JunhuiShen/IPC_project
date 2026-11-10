#include "DAT.h"
#include "solver.h"
#include "ipc_args.h"
#include "simulation.h"
#include "broad_phase.h"
#include "ccd.h"
#include "mesh_utils.h"
#include "solid_ipc.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double tolerance = 2e-12;

struct Scene {
    RefMesh mesh;
    std::vector<Vec3> reference;
};

Scene point_triangle(double gap = 1.0) {
    Scene scene;
    scene.reference = {Vec3(0, 0, gap), Vec3(-1, -1, 0),
        Vec3(1, -1, 0), Vec3(0, 1, 0)};
    scene.mesh.tris = {1, 2, 3};
    scene.mesh.initialize(scene.reference);
    scene.mesh.mass.assign(scene.reference.size(), 1.0);
    scene.mesh.node_to_rb.assign(scene.reference.size(), -1);
    scene.mesh.build_deformable_nodes();
    return scene;
}

BroadPhase candidates(const Scene& scene, double radius) {
    std::vector<AABB> boxes;
    for (const Vec3& position : scene.reference)
        boxes.emplace_back(position - Vec3::Constant(radius),
            position + Vec3::Constant(radius));
    BroadPhase phase;
    phase.initialize(boxes, scene.mesh, 0.0,
        BroadPhase::InitializationMode::DeformableSolver);
    return phase;
}

dat::Options planar_options(double radius = 4.0) {
    dat::Options options;
    options.mode = dat::Mode::Planar;
    options.gamma = 0.9;
    options.query_radius = radius;
    // A deliberate, representable band larger than the float32-scale floor
    // makes the hand-calculated geometry below independent of roundoff policy.
    options.separation_epsilon = 0.001;
    options.minimum_plane_fraction = 0.05;
    return options;
}

void expect_vec(const Vec3& actual, const Vec3& expected,
    double precision = tolerance) {
    for (int axis = 0; axis < 3; ++axis)
        EXPECT_NEAR(actual[axis], expected[axis], precision);
}

std::array<Vec3, 4> first_pair(const std::vector<Vec3>& positions) {
    return {positions[0], positions[1], positions[2], positions[3]};
}

struct RestoreOpenMP {
    const int threads = omp_get_max_threads();
    const int dynamic = omp_get_dynamic();
    ~RestoreOpenMP() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

SolverResult solve_dat(const RefMesh& mesh, const SimParams& params,
    std::vector<Vec3>& positions, const std::vector<Vec3>& target,
    const std::vector<Vec3>& velocity, BroadPhase& phase) {
    DeformedState state;
    state.deformed_positions = positions;
    state.velocities = velocity;
    if (params.use_ogc)
        return global_gauss_seidel_solver_ogc(mesh, {}, {}, params,
            positions, target, velocity, "", &state.deformed_positions);
    std::vector<Vec3> centers, omega;
    std::vector<Vec4> orientations;
    return global_gauss_seidel_solver_dat(mesh, state, {}, {}, params,
        positions, target, centers, orientations, omega, phase);
}

} // namespace

TEST(DATCanonical, StandalonePlanarFilterReservesTheStationarySideClearance) {
    const auto scene = point_triangle();
    std::vector<Vec3> proposed(4, Vec3::Zero());
    proposed[0] = Vec3(0, 0, -0.99);
    const auto accepted = planar_DAT(0.9, 4.0, scene.mesh, proposed, scene.reference);
    // The moving vertex receives 95% of the unit gap; subtract its side's
    // numerical clearance, then relax the crossing step exactly once.
    expect_vec(accepted[0], Vec3(0, 0, -0.9 * (0.95 - 1e-6)));
    for (int node = 1; node < 4; ++node) expect_vec(accepted[node], Vec3::Zero(), 0.0);
}

TEST(DATCanonical, ModesAreOptInAndDedicatedSelectorsAreExclusive) {
    const auto zero = SimParams::zeros();
    EXPECT_FALSE(zero.use_dat);
    EXPECT_FALSE(zero.use_ogc);
    IPCArgs3D defaults;
    const auto default_params = defaults.to_sim_params();
    EXPECT_FALSE(default_params.use_dat);
    EXPECT_FALSE(default_params.use_ogc);
    for (const bool isotropic : {false, true}) {
        IPCArgs3D args;
        std::vector<std::string> words{"3D_sim", isotropic ? "--use_ogc" : "--use_dat", "true"};
        std::vector<char*> argv;
        for (auto& word : words) argv.push_back(word.data());
        ASSERT_TRUE(args.parse(static_cast<int>(argv.size()), argv.data()));
        const auto params = args.to_sim_params();
        EXPECT_EQ(params.use_dat, !isotropic);
        EXPECT_EQ(params.use_ogc, isotropic);
        for (int conflict = 0; conflict < 3; ++conflict) {
            auto invalid = params;
            if (conflict == 0) invalid.use_dat = invalid.use_ogc = true;
            if (conflict == 1) invalid.use_cloth_grid = true;
            if (conflict == 2) invalid.use_jacobi_level_based_ccd = true;
            EXPECT_THROW(invalid.validate_dat_parameters(), std::invalid_argument);
        }
    }

}

TEST(DATCanonical, RemovedOptionsAndImportedExampleAreRejected) {
    for (const auto& words : std::vector<std::vector<std::string>>{
            {"3D_sim", "--use_reference_dat", "true"},
            {"3D_sim", "--use_ogc_solver", "true"},
            {"3D_sim", "--use_verlet_guess", "true"},
            {"3D_sim", "--use_translation_guess", "true"},
            {"3D_sim", "--use_reference_ogc", "true"},
            {"3D_sim", "--no_output", "true"},
            {"3D_sim", "--benchmark_json", "timings.json"},
            {"3D_sim", "--benchmark_warmup_frames", "5"},
            {"3D_sim", "--example", "21"}}) {
        IPCArgs3D args;
        auto mutable_words = words;
        std::vector<char*> argv;
        for (auto& word : mutable_words) argv.push_back(word.data());
        testing::internal::CaptureStderr();
        testing::internal::CaptureStdout();
        const bool accepted = args.parse(static_cast<int>(argv.size()), argv.data());
        testing::internal::GetCapturedStdout();
        const auto diagnostic = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(accepted);
        EXPECT_FALSE(diagnostic.empty());
    }
}

TEST(DATCanonical, BothModesValidateCheckpointControls) {
    for (bool isotropic : {false, true}) {
        auto params = SimParams::zeros();
        params.use_dat = !isotropic;
        params.use_ogc = isotropic;
        params.dat_query_radius = 1.0;
        for (int invalid = 0; invalid < 6; ++invalid) {
            auto bad = params;
            if (invalid == 0) bad.dat_gamma = 1.0;
            if (invalid == 1) bad.dat_query_radius = std::numeric_limits<double>::quiet_NaN();
            if (invalid == 2) bad.dat_node_box_padding = 0.99;
            if (invalid == 3) bad.dat_collision_detection_interval = -1;
            if (invalid == 4) { bad.dat_query_radius = 0; bad.d_hat = bad.node_box_max = 0; }
            if (invalid == 5) { bad.dat_query_radius = std::numeric_limits<double>::max(); bad.dat_node_box_padding = 2; }
            EXPECT_THROW(bad.validate_dat_parameters(), std::invalid_argument);
        }
        params.dat_inertial_predictor = false;
        params.dat_collision_detection_interval = 5;
        EXPECT_NO_THROW(params.validate_dat_parameters());
    }
}

TEST(DATCanonical, ZeroIterationCallsFilterThePhysicalInertialPredictor) {
    RefMesh mesh;
    mesh.num_positions = 1;
    mesh.mass = {1.0};
    mesh.node_to_rb = {-1};
    mesh.build_deformable_nodes();
    DeformedState state;
    state.deformed_positions = {Vec3::Zero()};
    state.velocities = {Vec3::UnitX()};
    const std::vector<Vec3> target{Vec3::UnitX()};
    auto params = SimParams::zeros();
    params.fps = 1.0;
    params.substeps = 1;
    params.fixed_iters = true;
    params.max_global_iters = 0;
    params.dat_query_radius = 1.0;
    params.dat_gamma = 0.9;
    for (bool isotropic : {false, true}) {
        SCOPED_TRACE(isotropic);
        auto accepted = state.deformed_positions;
        std::vector<Vec3> centers, omega;
        std::vector<Vec4> orientations;
        BroadPhase phase;
        const auto result = isotropic
            ? global_gauss_seidel_solver_ogc(mesh, {}, {}, params, accepted,
                target, state.velocities, "", &state.deformed_positions)
            : global_gauss_seidel_solver_dat(mesh, state, {}, {}, params,
                accepted, target, centers, orientations, omega, phase);
        EXPECT_TRUE(result.converged);
        EXPECT_EQ(result.iterations, 0);
        expect_vec(accepted[0], Vec3(0.45, 0, 0));
        expect_vec(state.deformed_positions[0], Vec3::Zero(), 0.0);
        expect_vec(state.velocities[0], Vec3::UnitX(), 0.0);
        expect_vec(target[0], Vec3::UnitX(), 0.0);
    }
}

TEST(DATCanonical, IsotropicDriversRejectAllRigidMetadataBeforeMutation) {
    for (int marker = 0; marker < 9; ++marker) for (int driver = 0; driver < 3; ++driver) {
        SCOPED_TRACE(marker);
        SCOPED_TRACE(driver);
        RefMesh mesh;
        mesh.num_positions = 1;
        mesh.mass = {1.0};
        mesh.node_to_rb = {-1};
        DeformedState state;
        state.deformed_positions = {Vec3::Zero()};
        state.velocities = {Vec3::UnitX()};
        if (marker == 0) mesh.rb_nodes.resize(1);
        if (marker == 1) mesh.total_mass.resize(1);
        if (marker == 2) mesh.rb_update_modes.resize(1);
        if (marker == 3) mesh.ref_positions.resize(1);
        if (marker == 4) mesh.node_to_rb[0] = 0;
        if (marker == 5) state.x_coms = {Vec3::Zero()};
        if (marker == 6) state.v_coms = {Vec3::Zero()};
        if (marker == 7) state.orientations = {Vec4(1, 0, 0, 0)};
        if (marker == 8) state.omega = {Vec3::Zero()};
        auto params = SimParams::zeros();
        params.use_ogc = true;
        params.dat_query_radius = 1.0;
        params.fps = 1;
        params.substeps = 1;
        std::vector<Pin> pins{{0, Vec3(100, 0, 0), true}};
        BroadPhase phase;
        int updates = 0, callbacks = 0;
        const auto updater = [&](std::vector<Pin>&, double) { ++updates; };
        const auto callback = [&](int, const std::vector<Vec3>&) { ++callbacks; };
        if (driver == 0) EXPECT_THROW(advance_one_frame(state, mesh, {}, pins, params,
            phase, 1, updater, callback), std::invalid_argument);
        if (driver == 1) EXPECT_THROW(advance_one_frame_general(state, mesh, {}, pins, params,
            phase, 1, updater, callback), std::invalid_argument);
        if (driver == 2) EXPECT_THROW(advance_one_frame_rb(state, mesh, params,
            phase, 1, callback), std::invalid_argument);
        EXPECT_EQ(updates, 0);
        EXPECT_EQ(callbacks, 0);
        expect_vec(state.deformed_positions[0], Vec3::Zero(), 0.0);
        expect_vec(state.velocities[0], Vec3::UnitX(), 0.0);
    }
}

TEST(PlanarDAT, FeasibleApproachKeepsTheWholeStep) {
    const auto scene = point_triangle();
    const auto options = planar_options();
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    std::vector<Vec3> proposed(4, Vec3(0, 0, 0.48));
    proposed[0].z() = -0.48;
    const auto result = checkpoint.filter(proposed);

    // The midpoint plane is z=.5, its clearance band is [.499,.501].
    // Endpoints z=.48 and .52 are already strictly feasible: no gamma damping.
    for (std::size_t node = 0; node < proposed.size(); ++node)
        expect_vec(result.displacements[node], proposed[node]);
    EXPECT_GT(compare_contact_distance_exact(first_pair(result.positions), true, 0), 0);
}

TEST(PlanarDAT, OneSidedApproachReservesStationarySideClearance) {
    const auto scene = point_triangle();
    const auto options = planar_options();
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    std::vector<Vec3> proposed(4, Vec3::Zero());
    proposed[0] = Vec3(0, 0, -0.99);
    const auto result = checkpoint.filter(proposed);

    // Stationary triangle gets .05 of the unit gap. The vertex's allowable
    // normal distance is .95-.001; crossing backoff preserves 10% of that.
    expect_vec(result.displacements[0], Vec3(0, 0, -0.8541));
    for (int node = 1; node < 4; ++node)
        expect_vec(result.displacements[node], Vec3::Zero());
}

TEST(PlanarDAT, RawApproachRatiosAreUsedBeforeFinalMagnitudeCap) {
    const auto scene = point_triangle();
    const auto options = planar_options(1.2);
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    std::vector<Vec3> proposed(4, Vec3(0, 0, 1.0));
    proposed[0] = Vec3(0, 0, -10.0);
    const auto result = checkpoint.filter(proposed);

    // The raw 10:1 approaches allocate 10/11 and 1/11 of the unit gap.
    // Only the larger accepted displacement then exceeds the .54 norm cap.
    expect_vec(result.displacements[0], Vec3(0, 0, -0.54));
    for (int node = 1; node < 4; ++node)
        expect_vec(result.displacements[node],
            Vec3(0, 0, 0.9 * (1.0 / 11.0 - 0.001)));
    EXPECT_GT(compare_contact_distance_exact(first_pair(result.positions), true, 0), 0);
}

TEST(PlanarDAT, TangentialMotionIsPreservedUntilTheNeighborhoodCap) {
    const auto scene = point_triangle();
    const auto options = planar_options(1.2);
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    std::vector<Vec3> proposed(4, Vec3::Zero());
    proposed[0] = Vec3(0.2, 0.1, 0.0);
    auto result = checkpoint.filter(proposed);
    expect_vec(result.displacements[0], proposed[0]);

    proposed[0] = Vec3(3, 4, 0);
    result = checkpoint.filter(proposed);
    expect_vec(result.displacements[0], Vec3(0.324, 0.432, 0));
    EXPECT_NEAR(result.displacements[0].norm(), 0.54, tolerance);
}

TEST(PlanarDAT, AccumulatedBudgetRemainsAnchoredToCollisionCheckpoint) {
    Scene scene;
    scene.reference = {Vec3(10, 0, 0)};
    scene.mesh.num_positions = 1;
    const auto options = planar_options(1.0);
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);

    const auto first = checkpoint.filter({Vec3(0.3, 0, 0)});
    const auto second = checkpoint.filter({Vec3(0.6, 0, 0)});
    expect_vec(first.positions[0], Vec3(10.3, 0, 0));
    expect_vec(second.positions[0], Vec3(10.45, 0, 0));
    expect_vec(checkpoint.reference_positions()[0], scene.reference[0]);
    expect_vec(scene.reference[0], Vec3(10, 0, 0));
}

TEST(PlanarDAT, OmittedOutsideRadiusPairCannotCloseItsGap) {
    const auto scene = point_triangle(1.1);
    const auto options = planar_options(1.0);
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    std::vector<Vec3> proposed(4, Vec3(0, 0, 100));
    proposed[0] = Vec3(0, 0, -100);
    const auto result = checkpoint.filter(proposed);

    for (const Vec3& displacement : result.displacements)
        EXPECT_LE(displacement.norm(), 0.45 + tolerance);
    // The radius-only guarantee leaves at least 1.1-2*.45=.2 of clearance.
    EXPECT_GE(result.positions[0].z() - result.positions[1].z(), 0.2 - tolerance);
    EXPECT_GT(compare_contact_distance_exact(first_pair(result.positions), true, 0), 0);
}

TEST(PlanarDAT, IndependentContinuousCCDConfirmsTheFilteredTrajectory) {
    const auto scene = point_triangle();
    const auto options = planar_options(1.2);
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    std::vector<Vec3> proposed = {Vec3(0.05, 0.02, -4),
        Vec3(0.1, 0, 0.8), Vec3(-0.1, 0, 1), Vec3(0, -0.1, 1.2)};
    const auto result = checkpoint.filter(proposed);
    const auto& x = scene.reference;
    const auto& dx = result.displacements;
    const double toi = node_triangle_general_ccd(x[0], dx[0], x[1], dx[1],
        x[2], dx[2], x[3], dx[3]);
    EXPECT_GE(toi, 1.0);
    EXPECT_GT(compare_contact_distance_exact(first_pair(result.positions), true, 0), 0);
}

TEST(IsotropicDAT, VertexFacePairBoundsEveryIncidentCorner) {
    const auto scene = point_triangle(0.4);
    auto options = planar_options(1.0);
    options.mode = dat::Mode::Isotropic;
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    ASSERT_EQ(checkpoint.isotropic_bounds().size(), 4u);
    for (double bound : checkpoint.isotropic_bounds())
        EXPECT_NEAR(bound, 0.18, tolerance);
    const auto result = checkpoint.filter(std::vector<Vec3>(4, Vec3(0, 10, 0)));
    for (const Vec3& displacement : result.displacements)
        expect_vec(displacement, Vec3(0, 0.18, 0));
}

TEST(IsotropicDAT, EdgePairBoundsBothEndpointsWithoutUnrelatedVertices) {
    Scene scene;
    scene.reference = {Vec3(-10, 0, 0), Vec3(10, 0, 0), Vec3(0, -10, -10),
        Vec3(0, -10, 0.4), Vec3(0, 10, 0.4), Vec3(10, 0, 10.4)};
    scene.mesh.num_positions = scene.reference.size();
    scene.mesh.tris = {0, 1, 2, 3, 4, 5};
    // Explicitly isolate this candidate pair to test its four-vertex incidence.
    BroadPhase phase;
    phase.mutable_cache().ss_pairs.push_back({{0, 1, 3, 4}});
    auto options = planar_options(1.0);
    options.mode = dat::Mode::Isotropic;
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    const auto& bounds = checkpoint.isotropic_bounds();
    ASSERT_EQ(bounds.size(), 6u);
    for (int node : {0, 1, 3, 4}) EXPECT_NEAR(bounds[node], 0.18, tolerance);
    for (int node : {2, 5}) EXPECT_NEAR(bounds[node], 0.45, tolerance);
}

TEST(PlanarDAT, EdgeEdgeTruncationPassesIndependentContinuousCCD) {
    Scene scene;
    scene.reference = {Vec3(-10, 0, 0), Vec3(10, 0, 0), Vec3(0, -10, -10),
        Vec3(0, -10, 0.4), Vec3(0, 10, 0.4), Vec3(10, 0, 10.4)};
    scene.mesh.num_positions = scene.reference.size();
    scene.mesh.tris = {0, 1, 2, 3, 4, 5};
    const auto options = planar_options(1.0);
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    const std::vector<Vec3> proposed{Vec3(0, 0, 0.4), Vec3(0, 0, 0.4),
        Vec3::Zero(), Vec3(0, 0, -0.4), Vec3(0, 0, -0.4), Vec3::Zero()};
    const auto result = checkpoint.filter(proposed);
    for (int node : {0, 1})
        expect_vec(result.displacements[node], Vec3(0, 0, 0.1791));
    for (int node : {3, 4})
        expect_vec(result.displacements[node], Vec3(0, 0, -0.1791));
    const auto& x = scene.reference;
    const auto& dx = result.displacements;
    const double toi = segment_segment_general_ccd(x[0], dx[0], x[1], dx[1],
        x[3], dx[3], x[4], dx[4]);
    EXPECT_GE(toi, 1.0);
    const std::array<Vec3, 4> edges{result.positions[0], result.positions[1],
        result.positions[3], result.positions[4]};
    EXPECT_GT(compare_contact_distance_exact(edges, false, 0), 0);
}

TEST(DATValidation, RejectsTouchingGeometryAndInvalidMotion) {
    const auto scene = point_triangle(0.0);
    const auto options = planar_options();
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    EXPECT_THROW(checkpoint.reset(scene.mesh, scene.reference, phase, options), std::domain_error);

    const auto separated = point_triangle();
    const auto separated_phase = candidates(separated, options.query_radius);
    checkpoint.reset(separated.mesh, separated.reference, separated_phase, options);
    EXPECT_THROW(checkpoint.filter({Vec3::Zero()}), std::invalid_argument);
    std::vector<Vec3> invalid(4, Vec3::Zero());
    invalid[0].x() = std::numeric_limits<double>::infinity();
    EXPECT_THROW(checkpoint.filter(invalid), std::invalid_argument);
}

TEST(DATValidation, ReusedOutputAndThreadCountsPreserveAcceptedCoordinates) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    Scene scene;
    const auto unit = point_triangle();
    std::vector<Vec3> proposed;
    for (int copy = 0; copy < 128; ++copy) {
        const int offset = static_cast<int>(scene.reference.size());
        for (const Vec3& position : unit.reference)
            scene.reference.push_back(position + Vec3(4.0 * copy, 0, 0));
        for (int node : unit.mesh.tris) scene.mesh.tris.push_back(offset + node);
        proposed.insert(proposed.end(), {Vec3(0, 0, -10),
            Vec3(0, 0, 1), Vec3(0, 0, 1), Vec3(0, 0, 1)});
    }
    scene.mesh.num_positions = scene.reference.size();
    auto options = planar_options(1.2);
    options.use_parallel = true;
    const auto phase = candidates(scene, options.query_radius);
    dat::Checkpoint checkpoint;
    checkpoint.reset(scene.mesh, scene.reference, phase, options);
    omp_set_num_threads(1);
    const auto expected = checkpoint.filter(proposed);
    dat::Result reused;
    for (int threads : {1, 2, 4}) {
        omp_set_num_threads(threads);
        checkpoint.filter_into(proposed, reused);
        ASSERT_EQ(reused.positions.size(), expected.positions.size());
        for (std::size_t node = 0; node < expected.positions.size(); ++node)
            EXPECT_EQ(std::memcmp(reused.positions[node].data(),
                expected.positions[node].data(), 3 * sizeof(double)), 0);
    }
}

TEST(DATSolver, CheckpointIntervalBoundsTotalMotionAcrossIterations) {
    for (bool isotropic : {false, true}) {
        for (int interval : {0, 1, 5}) {
            SCOPED_TRACE(isotropic);
            SCOPED_TRACE(interval);
            RefMesh mesh;
            mesh.num_positions = 1;
            mesh.mass = {1.0};
            mesh.node_to_rb = {-1};
            mesh.build_deformable_nodes();
            auto params = SimParams::zeros();
            params.fps = 1.0;
            params.substeps = 1;
            params.fixed_iters = true;
            params.max_global_iters = 3;
            params.damping = 1.0;
            params.use_dat = !isotropic;
            params.use_ogc = isotropic;
            params.dat_gamma = 0.9;
            params.dat_query_radius = 1.0;
            params.dat_collision_detection_interval = interval;
            // Deliberately isolate iterative checkpoint budgeting from the
            // solver's independently tested inertial predictor.
            params.dat_inertial_predictor = false;
            std::vector<Vec3> positions{Vec3::Zero()};
            const std::vector<Vec3> target{Vec3(100, 0, 0)};
            const std::vector<Vec3> velocity{Vec3::Zero()};
            BroadPhase phase;
            const auto result = solve_dat(mesh, params, positions, target, velocity, phase);
            EXPECT_TRUE(result.converged);
            EXPECT_EQ(result.iterations, 3);
            expect_vec(positions[0], Vec3(interval == 1 ? 1.35 : 0.45, 0, 0));
        }
    }
}

TEST(DATSolver, DefaultInitializationFiltersThePhysicalInertialPredictor) {
    RefMesh mesh;
    mesh.num_positions = 1;
    mesh.mass = {1.0};
    mesh.node_to_rb = {-1};
    mesh.build_deformable_nodes();
    auto params = SimParams::zeros();
    params.fps = 1.0;
    params.substeps = 1;
    params.fixed_iters = true;
    params.max_global_iters = 0;
    params.use_dat = true;
    params.dat_gamma = 0.9;
    params.dat_query_radius = 1.0;
    params.dat_collision_detection_interval = 0;
    std::vector<Vec3> positions{Vec3::Zero()};
    const std::vector<Vec3> target{Vec3::UnitX()};
    const std::vector<Vec3> velocity{Vec3::UnitX()};
    BroadPhase phase;
    const auto result = solve_dat(mesh, params, positions, target, velocity, phase);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    expect_vec(positions[0], Vec3(0.45, 0, 0));
    expect_vec(target[0], Vec3::UnitX(), 0.0);
}

TEST(DATSolver, PrescribedSolidReactionsAreExcludedButFreeForcesRemainInResidual) {
    RefMesh mesh;
    DeformedState state;
    const std::vector<Vec3> rest{Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY(), Vec3::UnitZ()};
    create_solid(rest, {0, 1, 2, 3}, 24.0, mesh, state);
    mesh.build_deformable_nodes();
    const auto adjacency = build_incident_triangle_map(mesh.tris);
    std::vector<Pin> pins;
    for (int node = 0; node < 4; ++node) {
        Pin pin;
        pin.vertex_index = node;
        pin.target_position = rest[node];
        pin.prescribed = true;
        pins.push_back(pin);
    }
    auto params = SimParams::zeros();
    params.use_dat = true;
    params.fps = 10;
    params.substeps = 1;
    params.fixed_iters = false;
    params.max_global_iters = 3;
    params.dat_query_radius = 2;
    params.gravity = Vec3(0, -4, 0);
    params.solid_mu = 20;
    params.solid_lambda = 30;
    params.tol_abs = 1e-12;
    const auto local = compute_solid_local_gradient_and_pbgs_block_no_barrier(
        0, mesh, pins, params, rest, rest);
    ASSERT_GT(local.first.norm(), 1e-3)
        << "Gravity must still exert a physical reaction on the prescribed node";
    auto positions = rest;
    std::vector<Vec3> centers, omega;
    std::vector<Vec4> orientations;
    BroadPhase phase;
    const auto fixed = global_gauss_seidel_solver_dat(mesh, state, adjacency, pins,
        params, positions, rest, centers, orientations, omega, phase);
    EXPECT_TRUE(fixed.converged);
    EXPECT_EQ(fixed.iterations, 0);
    EXPECT_TRUE(fixed.has_residual_components);
    EXPECT_DOUBLE_EQ(fixed.final_solid_residual, 0);
    for (int node = 0; node < 4; ++node) expect_vec(positions[node], rest[node], 0.0);

    pins[0].prescribed = false;
    params.dat_inertial_predictor = false;
    params.max_global_iters = 0;
    const auto free = global_gauss_seidel_solver_dat(mesh, state, adjacency, pins,
        params, positions, rest, centers, orientations, omega, phase);
    EXPECT_FALSE(free.converged);
    EXPECT_GT(free.final_solid_residual, params.tol_abs);
}

TEST(DATSolver, ParallelProductionColorUsesThePreparedCallerWorkspace) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    RefMesh mesh;
    constexpr std::size_t count = 256;
    mesh.num_positions = count;
    mesh.mass.assign(count, 1.0);
    mesh.node_to_rb.assign(count, -1);
    mesh.build_deformable_nodes();
    std::vector<Vec3> initial, target, velocity(count, Vec3::Zero());
    for (std::size_t node = 0; node < count; ++node) {
        initial.emplace_back(4.0 * node, -2.0 * node, 0.5 * node);
        target.push_back(initial.back() + Vec3(0.1, 0, 0));
    }
    // With no elastic edges, all 256 free blocks share one color. This must
    // exercise the production OpenMP proposal loop's >=128-block threshold,
    // not only Checkpoint's independent parallel filtering loops.
    for (bool isotropic : {false, true}) {
        SCOPED_TRACE(isotropic);
        auto params = SimParams::zeros();
        params.fps = 1.0;
        params.substeps = 1;
        params.fixed_iters = true;
        params.max_global_iters = 1;
        params.damping = 1.0;
        params.dat_query_radius = 2.0;
        params.dat_inertial_predictor = false;
        params.use_dat = !isotropic;
        params.use_ogc = isotropic;
        auto serial = initial;
        BroadPhase serial_phase;
        params.use_parallel = false;
        omp_set_num_threads(1);
        const auto serial_result = solve_dat(mesh, params, serial, target, velocity, serial_phase);
        ASSERT_TRUE(serial_result.converged);
        for (std::size_t node = 0; node < count; ++node)
            EXPECT_EQ(std::memcmp(serial[node].data(), target[node].data(), 3 * sizeof(double)), 0);
        for (int threads : {4, 8}) {
            SCOPED_TRACE(threads);
            omp_set_num_threads(threads);
            params.use_parallel = true;
            auto parallel = initial;
            BroadPhase phase;
            const auto result = solve_dat(mesh, params, parallel, target, velocity, phase);
            ASSERT_TRUE(result.converged);
            EXPECT_EQ(result.iterations, 1);
            for (std::size_t node = 0; node < count; ++node) {
                EXPECT_EQ(std::memcmp(parallel[node].data(), serial[node].data(), 3 * sizeof(double)), 0);
                EXPECT_EQ(std::memcmp(parallel[node].data(), target[node].data(), 3 * sizeof(double)), 0);
            }
        }
    }
}

TEST(DATUpstreamParity, MatchesPinnedNewtonWarpCpuFixtures) {
    // Frozen numeric outputs from the credited upstream CPU kernels. Keeping
    // these values in the test makes the regression independent of data files
    // and of an installed Newton/Warp runtime.
    static constexpr char golden[] = R"DAT_GOLDEN(IPC_NEWTON_DAT_FILTER_PARITY_V1
revision 2134db9cd0354019061d36a2762f886d59e88306
warp 1.17.0
cases 12
case vt_safe_approach planar_vt 0.90000000000000002 4 9.9999999747524271e-07
reference 0 0 1 -1 -1 0 1 -1 0 0 1 0
proposal 0 0 -0.47999998927116394 0 0 0.47999998927116394 0 0 0.47999998927116394 0 0 0.47999998927116394
expected 0 0 0.51999998092651367 -1 -1 0.47999998927116394 1 -1 0.47999998927116394 0 1 0.47999998927116394
case vt_one_sided planar_vt 0.90000000000000002 4 9.9999999747524271e-07
reference 0 0 1 -1 -1 0 1 -1 0 0 1 0
proposal 0 0 -0.99000000953674316 0 0 0 0 0 0 0 0 0
expected 0 0 0.14500093460083008 -1 -1 0 1 -1 0 0 1 0
case vt_cap_after_raw_ratio planar_vt 0.90000000000000002 1.2 9.9999999747524271e-07
reference 0 0 1 -1 -1 0 1 -1 0 0 1 0
proposal 0 0 -10 0 0 1 0 0 1 0 0 1
expected 0 0 0.45999997854232788 -1 -1 0.081817284226417542 1 -1 0.081817284226417542 0 1 0.081817284226417542
case vt_tangential planar_vt 0.90000000000000002 1.2 9.9999999747524271e-07
reference 0 0 1 -1 -1 0 1 -1 0 0 1 0
proposal 0.20000000298023224 0.10000000149011612 0 0 0 0 0 0 0 0 0 0
expected 0.20000000298023224 0.10000000149011612 1 -1 -1 0 1 -1 0 0 1 0
case vt_separating planar_vt 0.90000000000000002 1.2 9.9999999747524271e-07
reference 0 0 1 -1 -1 0 1 -1 0 0 1 0
proposal 0 0 0.20000000298023224 0 0 -0.10000000149011612 0 0 -0.10000000149011612 0 0 -0.10000000149011612
expected 0 0 1.2000000476837158 -1 -1 -0.10000000149011612 1 -1 -0.10000000149011612 0 1 -0.10000000149011612
case ee_both_approach planar_ee 0.90000000000000002 1 9.9999999747524271e-07
reference -1 0 0 1 0 0 0 -1 0.40000000596046448 0 1 0.40000000596046448
proposal 0 0 0.40000000596046448 0 0 0.40000000596046448 0 0 -0.40000000596046448 0 0 -0.40000000596046448
expected -1 0 0.17999909818172455 1 0 0.17999909818172455 0 -1 0.22000090777873993 0 1 0.22000090777873993
case ee_cap_after_raw_ratio planar_ee 0.90000000000000002 1.2 9.9999999747524271e-07
reference -1 0 0 1 0 0 0 -1 1 0 1 1
proposal 0 0 10 0 0 10 0 0 -1 0 0 -1
expected -1 0 0.54000002145767212 1 0 0.54000002145767212 0 -1 0.91818267107009888 0 1 0.91818267107009888
case vt_isotropic_tangential isotropic_vt 0.90000000000000002 1 9.9999999999999995e-07
reference 0 0 0.40000000596046448 -1 -1 0 1 -1 0 0 1 0
proposal 3 4 0 0 1 0 1 0 0 0 -1 0
expected 0.1080000102519989 0.14400000870227814 0.40000000596046448 -1 -0.81999999284744263 0 1.1800000667572021 -1 0 0 0.81999999284744263 0
case vt_oblique_face planar_vt 0.90000000000000002 1.5 9.9999999747524271e-07
reference -0.076980039477348328 -0.25635328888893127 0.45156684517860413 -1 -1 0 1 -1 0.40000000596046448 0 1 -0.20000000298023224
proposal 0.21245008707046509 -0.17245009541511536 -0.9622504711151123 -0.076980039477348328 0.076980039477348328 0.38490018248558044 -0.13471506536006927 0.13471506536006927 0.67357534170150757 -0.096225045621395111 0.096225045621395111 0.48112523555755615
expected -0.031990807503461838 -0.29287198185920715 0.24779705703258514 -1.0285277366638184 -0.97147226333618164 0.14263857901096344 0.97147226333618164 -0.97147226333618164 0.54263871908187866 -0.028527740389108658 1.0285277366638184 -0.057361304759979248
case ee_oblique_edges planar_ee 0.90000000000000002 1.2 9.9999999747524271e-07
reference -1 -0 -0.20000000298023224 1 0 0.20000000298023224 -0.25873327255249023 -1.0176199674606323 0.19366636872291565 0.14126671850681305 0.98238003253936768 0.39366638660430908
proposal -0.11746654659509659 -0.035239964723587036 0.58733272552490234 -0.11746654659509659 -0.035239964723587036 0.58733272552490234 0.039155516773462296 0.011746655218303204 -0.19577758014202118 0.039155516773462296 0.011746655218303204 -0.19577758014202118
expected -1.039644718170166 -0.011893431656062603 -0.0017761439085006714 0.96035522222518921 -0.011893433518707752 0.398223876953125 -0.24551846086978912 -1.0136555433273315 0.12759232521057129 0.15448154509067535 0.98634445667266846 0.32759225368499756
case vt_obtuse_exterior planar_vt 0.90000000000000002 0.29999999999999999 4.76837158203125e-06
reference 0.5 -0.0099999997764825821 0.10000000149011612 0 0 0 1 0 0 -10 0.10000000149011612 0
proposal 0 0.019999999552965164 -0.20000000298023224 0 0 0 0 0 0 0 0 0
expected 0.5 -0.0014504268765449524 0.014504268765449524 0 0 0 1 0 0 -10 0.10000000149011612 0
case vt_degenerate_triangle planar_vt 0.90000000000000002 2 9.9999999747524271e-07
reference 0.5 1 0.20000000298023224 0 0 0 1 0 0 2 0 0
proposal -0 -1.9611613750457764 -0.3922322690486908 0 0.1961161345243454 0.039223227649927139 0 0.1961161345243454 0.039223227649927139 0 0.1961161345243454 0.039223227649927139
expected 0.5 0.18181914091110229 0.036363840103149414 0 0.081817299127578735 0.016363460570573807 1 0.081817299127578735 0.016363460570573807 2 0.081817299127578735 0.016363460570573807
)DAT_GOLDEN";
    std::istringstream input(golden);
    const auto token = [&](const std::string& expected) {
        std::string actual;
        if (!(input >> actual) || actual != expected)
            throw std::runtime_error("Invalid DAT parity fixture token: " + expected);
    };
    token("IPC_NEWTON_DAT_FILTER_PARITY_V1");
    token("revision");
    std::string revision;
    ASSERT_TRUE(bool(input >> revision));
    ASSERT_EQ(revision, "2134db9cd0354019061d36a2762f886d59e88306");
    token("warp");
    std::string warp_version;
    ASSERT_TRUE(bool(input >> warp_version));
    ASSERT_EQ(warp_version, "1.17.0");
    token("cases");
    int count;
    ASSERT_TRUE(bool(input >> count));
    ASSERT_EQ(count, 12);
    double maximum_error = 0.0;
    for (int index = 0; index < count; ++index) {
        token("case");
        std::string name, kind;
        auto options = planar_options();
        ASSERT_TRUE(bool(input >> name >> kind >> options.gamma
            >> options.query_radius >> options.separation_epsilon));
        SCOPED_TRACE(name);
        Scene scene;
        scene.reference.resize(4);
        std::vector<Vec3> proposed(4), expected(4);
        for (auto entry : {std::make_pair("reference", &scene.reference),
                std::make_pair("proposal", &proposed),
                std::make_pair("expected", &expected)}) {
            token(entry.first);
            for (Vec3& value : *entry.second)
                ASSERT_TRUE(bool(input >> value.x() >> value.y() >> value.z()));
        }
        scene.mesh.num_positions = 4;
        const bool edge_pair = kind == "planar_ee";
        options.mode = kind == "isotropic_vt"
            ? dat::Mode::Isotropic : dat::Mode::Planar;
        ASSERT_TRUE(kind == "planar_vt" || kind == "planar_ee" || kind == "isotropic_vt");
        BroadPhase phase;
        if (edge_pair) {
            phase.mutable_cache().ss_pairs.push_back({{0, 1, 2, 3}});
        } else {
            scene.mesh.tris = {1, 2, 3};
            phase.mutable_cache().nt_pairs.push_back({0, {1, 2, 3}});
        }
        dat::Checkpoint checkpoint;
        checkpoint.reset(scene.mesh, scene.reference, phase, options);
        const auto result = checkpoint.filter(proposed);
        ASSERT_EQ(result.positions.size(), expected.size());
        for (std::size_t node = 0; node < expected.size(); ++node) {
            const double error = (result.positions[node] - expected[node]).cwiseAbs().maxCoeff();
            maximum_error = std::max(maximum_error, error);
            // The oracle evaluates original Warp float32 operations; the port
            // evaluates binary64 and additionally certifies represented endpoints.
            EXPECT_LE(error, 2e-6) << "vertex " << node;
        }
        EXPECT_GT(compare_contact_distance_exact(first_pair(result.positions), !edge_pair, 0), 0);
    }
    std::string trailing;
    EXPECT_FALSE(bool(input >> trailing));
    std::ostringstream error_text;
    error_text.precision(17);
    error_text << maximum_error;
    RecordProperty("maximum_absolute_coordinate_error", error_text.str());
}

