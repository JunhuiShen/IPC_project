#include "DAT.h"
#include "mesh_utils.h"
#include "simulation.h"
#include "solver.h"
#include "solid_ipc.h"
#include "rigid_dat_trajectory.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

struct RestoreOpenMP {
    int threads = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    ~RestoreOpenMP() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

struct RigidDATScene {
    RefMesh mesh;
    DeformedState state;
    SimParams params = SimParams::zeros();
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;

    RigidDATScene() {
        params.fps = 10;
        params.substeps = 1;
        params.damping = 1;
        params.fixed_iters = true;
        params.max_global_iters = 1;
        params.dat_query_radius = 10;
        params.use_jacobi_dat = true;
        params.k_barrier = 0;
    }

    int add_body(const std::vector<Vec3>& points,
        const Vec3& velocity = Vec3::Zero(),
        const Vec3& spin = Vec3::Zero(),
        RigidBodyUpdateMode mode = RigidBodyUpdateMode::TranslationAndOrientation) {
        return create_rigid_body(points, velocity, Vec4(1, 0, 0, 0), spin,
            static_cast<double>(points.size()), mesh, state, mode);
    }

    void finish() {
        mesh.initialize(state.deformed_positions);
        mesh.build_deformable_nodes();
        adjacency = build_incident_triangle_map(mesh.tris);
    }
};

std::vector<Vec3> tetrahedron(const Vec3& center = Vec3::Zero()) {
    return {center + Vec3(.1, .1, .1), center + Vec3(.1, -.1, -.1),
        center + Vec3(-.1, .1, -.1), center + Vec3(-.1, -.1, .1)};
}

void expect_vectors(const std::vector<Vec3>& actual,
    const std::vector<Vec3>& expected, double tolerance = 2e-12) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(actual[i].allFinite()) << i;
        EXPECT_LE((actual[i] - expected[i]).norm(), tolerance) << i;
    }
}

void expect_rigidity(const RefMesh& mesh, const std::vector<Vec3>& positions) {
    for (std::size_t body = 0; body < mesh.rb_nodes.size(); ++body) {
        const auto& nodes = mesh.rb_nodes[body];
        const auto& reference = mesh.ref_positions[body];
        for (std::size_t i = 0; i < nodes.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
                EXPECT_NEAR((positions[nodes[i]] - positions[nodes[j]]).norm(),
                    (reference[i] - reference[j]).norm(), 3e-12)
                    << "body " << body << ", nodes " << i << ", " << j;
    }
}

struct SolveState {
    std::vector<Vec3> positions, target, centers, omega;
    std::vector<Vec4> orientation;

    explicit SolveState(const RigidDATScene& scene)
        : positions(scene.state.deformed_positions),
          centers(scene.state.x_coms),
          omega(scene.state.omega.size(), Vec3::Zero()),
          orientation(scene.state.orientations) {
        build_xhat(target, scene.state.deformed_positions,
            scene.state.velocities, scene.params.dt());
        for (std::size_t node = 0; node < target.size(); ++node)
            if (scene.mesh.node_to_rb[node] >= 0)
                target[node] = scene.state.deformed_positions[node];
    }
};

SolverResult solve(const RigidDATScene& scene, SolveState& iterate,
    BroadPhase& phase) {
    return global_jacobi_solver_dat_general(scene.mesh, scene.state,
        scene.adjacency, scene.pins, scene.params, iterate.positions,
        iterate.target, iterate.centers, iterate.orientation, iterate.omega,
        phase);
}

RigidDATScene approaching_scene(bool deformable_vertex) {
    RigidDATScene scene;
    if (deformable_vertex) {
        scene.state.deformed_positions = {Vec3(0, 0, .2)};
        scene.state.velocities = {Vec3(0, 0, -10)};
        scene.mesh.mass = {1};
        scene.mesh.node_to_rb = {-1};
    } else {
        scene.add_body({Vec3(0, 0, .2), Vec3(.1, 0, .3),
            Vec3(0, .1, .3), Vec3(-.1, -.1, .3)}, Vec3(0, 0, -10),
            Vec3::Zero(), RigidBodyUpdateMode::TranslationOnly);
    }
    const int first = static_cast<int>(scene.state.deformed_positions.size());
    scene.add_body({Vec3(-1, -1, 0), Vec3(2, -1, 0), Vec3(-1, 2, 0),
        Vec3(0, 0, -.5)}, Vec3(0, 0, 6), Vec3::Zero(),
        RigidBodyUpdateMode::TranslationOnly);
    scene.mesh.tris = {first, first + 1, first + 2};
    scene.finish();
    return scene;
}

} // namespace

TEST(RigidDATSolver, FreeTranslationMatchesTheDiscreteEquationInOneSweep) {
    RigidDATScene scene;
    const Vec3 velocity(.4, -.2, .3);
    scene.params.gravity = Vec3(0, -4, 0);
    scene.add_body(tetrahedron(Vec3(1, 2, 3)), velocity, Vec3::Zero(),
        RigidBodyUpdateMode::TranslationOnly);
    scene.finish();
    SolveState iterate(scene);
    const double dt = scene.params.dt();
    const Vec3 expected_displacement = dt * velocity
        + dt * dt * scene.params.gravity;
    BroadPhase phase;
    const auto result = solve(scene, iterate, phase);
    ASSERT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_LE((iterate.centers[0] - scene.state.x_coms[0]
        - expected_displacement).norm(), 2e-13);
    for (std::size_t node = 0; node < iterate.positions.size(); ++node)
        EXPECT_LE((iterate.positions[node] - scene.state.deformed_positions[node]
            - expected_displacement).norm(), 3e-13);
    EXPECT_TRUE(iterate.omega[0].isZero(0));
    EXPECT_LE((iterate.orientation[0] - scene.state.orientations[0]).norm(), 1e-15);
    expect_rigidity(scene.mesh, iterate.positions);
}

TEST(RigidDATSolver, PreviousPhysicalSpinDrivesTheZeroInitialAngularIterate) {
    RigidDATScene scene;
    scene.params.max_global_iters = 8;
    scene.params.damping = .5;
    scene.add_body(tetrahedron(), Vec3::Zero(), Vec3(0, 0, .4),
        RigidBodyUpdateMode::OrientationOnly);
    scene.finish();
    SolveState iterate(scene);
    ASSERT_TRUE(iterate.omega[0].isZero(0));
    BroadPhase phase;
    ASSERT_TRUE(solve(scene, iterate, phase).converged);
    EXPECT_GT(iterate.omega[0].z(), .1);
    EXPECT_GT(iterate.orientation[0][3], 1e-3);
    EXPECT_LE((iterate.centers[0] - scene.state.x_coms[0]).norm(), 1e-15);
    EXPECT_NEAR(iterate.orientation[0].norm(), 1, 2e-14);
    expect_rigidity(scene.mesh, iterate.positions);
}

TEST(RigidDATSolver, EveryUpdateModeKeepsItsDisabledCoordinatesFixed) {
    for (const auto mode : {RigidBodyUpdateMode::TranslationAndOrientation,
             RigidBodyUpdateMode::TranslationOnly,
             RigidBodyUpdateMode::OrientationOnly, RigidBodyUpdateMode::None}) {
        SCOPED_TRACE(static_cast<int>(mode));
        RigidDATScene scene;
        scene.params.max_global_iters = 5;
        scene.params.damping = .5;
        scene.add_body(tetrahedron(), Vec3(.3, .2, -.1), Vec3(0, 0, .4), mode);
        scene.finish();
        SolveState iterate(scene);
        BroadPhase phase;
        ASSERT_TRUE(solve(scene, iterate, phase).converged);
        if (updates_rigid_translation(mode))
            EXPECT_GT((iterate.centers[0] - scene.state.x_coms[0]).norm(), 1e-3);
        else
            EXPECT_TRUE(iterate.centers[0].isApprox(scene.state.x_coms[0], 0));
        if (updates_rigid_orientation(mode))
            EXPECT_GT((iterate.orientation[0] - scene.state.orientations[0]).norm(), 1e-3);
        else {
            EXPECT_TRUE(iterate.orientation[0].isApprox(scene.state.orientations[0], 0));
            EXPECT_TRUE(iterate.omega[0].isZero(0));
        }
        expect_rigidity(scene.mesh, iterate.positions);
    }
}

TEST(RigidDATSolver, TranslationAndRotationProposalsReadTheSameFrozenPose) {
    RigidDATScene scene;
    scene.params.kpin=20;
    scene.params.damping=.3;
    scene.add_body(tetrahedron());
    RigidPin pin;
    pin.body=0;pin.local_position=Vec3(.1,.1,.1);
    pin.initial_target=Vec3(.13,.2,.14);
    scene.params.rigid_pins={pin};
    scene.finish();
    auto orientation_only=scene;
    orientation_only.mesh.rb_update_modes[0]=RigidBodyUpdateMode::OrientationOnly;
    SolveState combined(scene),isolated(orientation_only);
    BroadPhase combined_phase,isolated_phase;
    ASSERT_TRUE(solve(scene,combined,combined_phase).converged);
    ASSERT_TRUE(solve(orientation_only,isolated,isolated_phase).converged);
    EXPECT_GT((combined.centers[0]-scene.state.x_coms[0]).norm(),1e-5);
    EXPECT_GT(combined.omega[0].norm(),1e-5);
    EXPECT_LE((combined.omega[0]-isolated.omega[0]).norm(),1e-13);
    EXPECT_LE((combined.orientation[0]-isolated.orientation[0]).norm(),1e-13);
}

TEST(RigidDATSolver, BothMovingRigidBodiesKeepAGapWithoutBarrierOrCCD) {
    auto scene = approaching_scene(false);
    ASSERT_DOUBLE_EQ(scene.params.k_barrier, 0);
    ASSERT_FALSE(scene.params.use_ccd);
    const int triangle_node = scene.mesh.tris[0];
    const auto before = scene.state.deformed_positions;
    SolveState iterate(scene);
    BroadPhase phase;
    ASSERT_TRUE(solve(scene, iterate, phase).converged);
    EXPECT_GT(iterate.positions[0].z() - iterate.positions[triangle_node].z(), 0);
    EXPECT_LT(iterate.positions[0].z(), before[0].z());
    EXPECT_GT(iterate.positions[triangle_node].z(), before[triangle_node].z());
    expect_rigidity(scene.mesh, iterate.positions);
}

TEST(RigidDATSolver, RepeatedNearContactFramesNeverPublishARoundedIntersection) {
    auto scene=approaching_scene(false);
    const int triangle_node=scene.mesh.tris[0];
    BroadPhase phase;
    for(int frame=1;frame<=24;++frame) {
        SCOPED_TRACE(frame);
        const auto before=scene.state;
        try {
            ASSERT_TRUE(advance_one_frame_rb(scene.state,scene.mesh,scene.params,phase,frame).converged);
        } catch(const std::domain_error&) {
            // An unresolved floating-point gap may stop explicitly, but the
            // caller's last published physical state must stay separated.
            expect_vectors(scene.state.deformed_positions,before.deformed_positions,0);
            break;
        }
        EXPECT_GT(scene.state.deformed_positions[0].z()
            -scene.state.deformed_positions[triangle_node].z(),0);
    }
}

TEST(RigidDATSolver, MovingRigidTriangleAndDeformableVertexShareTheGap) {
    auto scene = approaching_scene(true);
    const auto before = scene.state.deformed_positions;
    const int triangle_node = scene.mesh.tris[0];
    SolveState iterate(scene);
    BroadPhase phase;
    ASSERT_TRUE(solve(scene, iterate, phase).converged);
    EXPECT_GT(iterate.positions[0].z() - iterate.positions[triangle_node].z(), 0);
    EXPECT_LT(iterate.positions[0].z(), before[0].z());
    EXPECT_GT(iterate.positions[triangle_node].z(), before[triangle_node].z());
    expect_rigidity(scene.mesh, iterate.positions);
}

TEST(RigidDATSolver, InternalRigidSurfacePairsDoNotCreateContactConstraints) {
    RigidDATScene scene;
    scene.add_body(tetrahedron(), Vec3(.3, 0, 0), Vec3::Zero(),
        RigidBodyUpdateMode::TranslationOnly);
    scene.mesh.tris = {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2};
    scene.params.d_hat = .5;
    scene.params.k_barrier = 100;
    scene.finish();
    SolveState iterate(scene);
    BroadPhase phase;
    ASSERT_TRUE(solve(scene, iterate, phase).converged);
    EXPECT_NEAR(iterate.centers[0].x() - scene.state.x_coms[0].x(), .03, 2e-13);
    EXPECT_TRUE(phase.cache().nt_pairs.empty());
    EXPECT_TRUE(phase.cache().ss_pairs.empty());
    expect_rigidity(scene.mesh, iterate.positions);
}

TEST(RigidDATSolver, ZeroDampingReportsUnresolvedRigidForces) {
    RigidDATScene scene;
    scene.add_body(tetrahedron(), Vec3(.4, 0, 0), Vec3::Zero(),
        RigidBodyUpdateMode::TranslationOnly);
    scene.params.fixed_iters = false;
    scene.params.max_global_iters = 3;
    scene.params.damping = 0;
    scene.params.tol_abs = 1e-14;
    scene.finish();
    SolveState iterate(scene);
    BroadPhase phase;
    const auto result = solve(scene, iterate, phase);
    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.has_residual_components);
    EXPECT_EQ(result.iterations, 3);
    const double expected_force = scene.mesh.total_mass[0]
        * scene.params.dt() * scene.state.v_coms[0].norm();
    EXPECT_NEAR(result.initial_rigid_residual, expected_force, 2e-13);
    EXPECT_NEAR(result.final_rigid_residual, expected_force, 2e-13);
    EXPECT_NEAR(result.final_residual, result.final_rigid_residual, 2e-13);
    EXPECT_DOUBLE_EQ(result.final_cloth_residual, 0);
    expect_vectors(iterate.positions, scene.state.deformed_positions, 0);
}

TEST(RigidDATSolver, PureRigidFrameDriverUsesDATAndThePhysicalSubstepState) {
    RigidDATScene scene;
    scene.params.substeps = 2;
    scene.params.gravity = Vec3(0, -3, 0);
    // All predictor flags are ignored by the DAT path.
    scene.params.use_verlet_guess = true;
    scene.params.use_translation_guess = true;
    const Vec3 initial_velocity(.4, .2, -.1);
    scene.add_body(tetrahedron(Vec3(1, 2, 3)), initial_velocity, Vec3::Zero(),
        RigidBodyUpdateMode::TranslationOnly);
    scene.finish();
    const Vec3 center = scene.state.x_coms[0];
    const double dt = scene.params.dt();
    int callbacks = 0;
    BroadPhase phase;
    const auto result = advance_one_frame_rb(scene.state, scene.mesh,
        scene.params, phase, 1, [&](int index, const std::vector<Vec3>& positions) {
            EXPECT_EQ(index, callbacks++);
            expect_rigidity(scene.mesh, positions);
        });
    ASSERT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 2);
    EXPECT_EQ(callbacks, 2);
    EXPECT_LE((scene.state.x_coms[0] - center - 2 * dt * initial_velocity
        - 3 * dt * dt * scene.params.gravity).norm(), 3e-13);
    EXPECT_LE((scene.state.v_coms[0] - initial_velocity
        - 2 * dt * scene.params.gravity).norm(), 3e-12);
}

TEST(RigidDATSolver, MixedFrameDriverCommitsParticleAndGeneralizedState) {
    RigidDATScene scene;
    scene.state.deformed_positions = {Vec3(-3, 0, 0)};
    scene.state.velocities = {Vec3(.2, .3, 0)};
    scene.mesh.mass = {2};
    scene.mesh.node_to_rb = {-1};
    scene.add_body(tetrahedron(Vec3(3, 0, 0)), Vec3(-.3, .1, .2),
        Vec3::Zero(), RigidBodyUpdateMode::TranslationOnly);
    scene.params.gravity = Vec3(0, -2, 0);
    scene.finish();
    const auto before = scene.state;
    const double dt = scene.params.dt();
    BroadPhase phase;
    const auto result = advance_one_frame_general(scene.state, scene.mesh,
        scene.adjacency, scene.pins, scene.params, phase);
    ASSERT_TRUE(result.converged);
    EXPECT_LE((scene.state.deformed_positions[0] - before.deformed_positions[0]
        - dt * before.velocities[0] - dt * dt * scene.params.gravity).norm(), 3e-13);
    EXPECT_LE((scene.state.x_coms[0] - before.x_coms[0]
        - dt * before.v_coms[0] - dt * dt * scene.params.gravity).norm(), 3e-13);
    expect_rigidity(scene.mesh, scene.state.deformed_positions);
}

TEST(RigidDATSolver, PureRigidDriverAdvancesMotorPinTimeBeforeAssembly) {
    RigidDATScene scene;
    scene.add_body(tetrahedron(Vec3(1, 2, 0)), Vec3::Zero(), Vec3::Zero(),
        RigidBodyUpdateMode::TranslationOnly);
    scene.params.kpin = 8;
    const Vec3 original_center = scene.state.x_coms[0];
    RigidPin pin;
    pin.body = 0;
    pin.local_position = Vec3::Zero();
    pin.initial_target = original_center + Vec3::UnitX();
    pin.rotation_center = original_center;
    pin.angular_velocity_z = 2;
    scene.params.rigid_pins = {pin};
    scene.finish();
    const double dt = scene.params.dt();
    const double physical_target_time = 3 * dt;
    const Vec3 rotated_offset(std::cos(2 * physical_target_time),
        std::sin(2 * physical_target_time), 0);
    const double spring_weight = dt * dt * scene.params.kpin;
    const Vec3 expected_center = original_center
        + spring_weight / (scene.mesh.total_mass[0] + spring_weight) * rotated_offset;
    BroadPhase phase;
    ASSERT_TRUE(advance_one_frame_rb(scene.state, scene.mesh, scene.params,
        phase, 3).converged);
    EXPECT_LE((scene.state.x_coms[0] - expected_center).norm(), 3e-13);
    // The frame driver owns the per-substep copy; caller parameters are immutable.
    EXPECT_DOUBLE_EQ(scene.params.rigid_pin_time, 0);
    expect_rigidity(scene.mesh, scene.state.deformed_positions);
}

TEST(RigidDATSolver, FrictionUsesOnePhysicalHistoryAcrossSweepsAndThreads) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    RigidDATScene scene;
    scene.add_body({Vec3(0, 0, .1), Vec3(.1, 0, .15),
        Vec3(0, .1, .15), Vec3(-.1, -.1, .15)}, Vec3(1, 0, 0),
        Vec3::Zero(), RigidBodyUpdateMode::TranslationOnly);
    scene.add_body({Vec3(-2, -2, 0), Vec3(4, -2, 0), Vec3(-2, 4, 0)},
        Vec3::Zero(), Vec3::Zero(), RigidBodyUpdateMode::None);
    scene.mesh.tris = {4, 5, 6};
    scene.params.d_hat = .2;
    scene.params.k_barrier = 20;
    scene.params.friction_coefficient = .6;
    scene.params.damping = .3;
    scene.params.max_global_iters = 6;
    scene.finish();
    const auto history = scene.state;
    SolveState initial(scene);
    initial.centers[0].x() += .04;
    for (int node : scene.mesh.rb_nodes[0]) initial.positions[node].x() += .04;
    auto one_sweep_scene = scene;
    one_sweep_scene.params.max_global_iters = 1;
    auto reference = initial;
    BroadPhase reference_phase;
    for (int sweep = 0; sweep < 6; ++sweep)
        ASSERT_TRUE(solve(one_sweep_scene, reference, reference_phase).converged);
    for (int threads : {1, 4}) {
        SCOPED_TRACE(threads);
        omp_set_num_threads(threads);
        scene.params.use_parallel = threads > 1;
        auto actual = initial;
        BroadPhase phase;
        ASSERT_TRUE(solve(scene, actual, phase).converged);
        expect_vectors(actual.positions, reference.positions);
        expect_vectors(actual.centers, reference.centers);
        expect_vectors(actual.omega, reference.omega);
        expect_vectors(scene.state.deformed_positions, history.deformed_positions, 0);
        expect_vectors(scene.state.velocities, history.velocities, 0);
    }
    auto frictionless = scene;
    frictionless.params.friction_coefficient = 0;
    auto unconstrained_tangent = initial;
    BroadPhase frictionless_phase;
    ASSERT_TRUE(solve(frictionless, unconstrained_tangent, frictionless_phase).converged);
    EXPECT_LT(reference.centers[0].x(), unconstrained_tangent.centers[0].x() - 1e-6);
}

TEST(RigidDATSolver, MalformedOwnershipIsRejectedBeforeAnyCandidateIsPublished) {
    for (int corruption : {0, 1, 2}) {
        SCOPED_TRACE(corruption);
        RigidDATScene scene;
        scene.add_body(tetrahedron(), Vec3(.5, 0, 0));
        scene.finish();
        SolveState iterate(scene);
        const auto original = iterate;
        if (corruption == 0) scene.mesh.node_to_rb[0] = 4;
        if (corruption == 1) scene.mesh.rb_nodes[0][1] = scene.mesh.rb_nodes[0][0];
        if (corruption == 2) scene.mesh.node_to_rb[0] = -1;
        BroadPhase phase;
        EXPECT_THROW(solve(scene, iterate, phase), std::invalid_argument);
        expect_vectors(iterate.positions, original.positions, 0);
        expect_vectors(iterate.centers, original.centers, 0);
        expect_vectors(iterate.omega, original.omega, 0);
    }
}

TEST(RigidDATFilter, IntervalVerificationFindsCrossingsHiddenBetweenAllEightSamples) {
    RigidDATScene scene;
    scene.add_body({Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0),
        Vec3(0, -1, 0), Vec3(0, 0, .1), Vec3(0, 0, -.1)});
    scene.add_body({Vec3(.5, -3, -3), Vec3(.5, 3, -3), Vec3(.5, 0, 3)},
        Vec3::Zero(), Vec3::Zero(), RigidBodyUpdateMode::None);
    scene.mesh.tris = {6, 7, 8};
    scene.finish();
    const double pi = std::acos(-1.0);
    const std::vector<Vec3> zero_updates(scene.state.deformed_positions.size(), Vec3::Zero());
    const std::vector<Vec3> angular(2, Vec3::Zero()), translation(2, Vec3::Zero());
    auto delta_omega = angular;
    // Eight turns: every k/8 endpoint returns to the initial position.
    // A sample-only implementation would accept despite multiple crossings.
    delta_omega[0] = Vec3(0, 0, 16 * pi);
    const auto accepted = general_planar_DAT(.9, 100, scene.mesh, zero_updates,
        scene.state.deformed_positions, scene.state.x_coms, angular,
        scene.state.orientations, translation, delta_omega, 1.0);
    ASSERT_EQ(accepted.body_factors.size(), 2u);
    const double factor = accepted.body_factors[0];
    ASSERT_GT(factor, 0);
    ASSERT_LT(factor, 1.0 / 48.0);
    for (int sample = 0; sample <= 1000; ++sample) {
        const double t = factor * sample / 1000;
        for (std::size_t local = 0; local < scene.mesh.rb_nodes[0].size(); ++local) {
            const int node = scene.mesh.rb_nodes[0][local];
            const Vec3 position = world_space_position(scene.mesh.ref_positions[0][local],
                scene.state.x_coms[0], scene.state.orientations[0], t * delta_omega[0], 1.0);
            const double original_side = scene.state.deformed_positions[node].x() - .5;
            EXPECT_GT(original_side * (position.x() - .5), 0)
                << "sample " << sample << ", local node " << local;
        }
    }
}

TEST(RigidDATFilter, InitialPoseRoundoffCannotReverseTheIntendedHalfSpace) {
    RigidDATScene scene;
    constexpr double gap=1e-14;
    scene.add_body({Vec3(gap,0,0)});
    scene.add_body({Vec3(0,-1,-1),Vec3(0,1,-1),Vec3(0,0,1)},
        Vec3::Zero(),Vec3::Zero(),RigidBodyUpdateMode::TranslationOnly);
    scene.mesh.tris={1,2,3};
    scene.finish();
    auto centers=scene.state.x_coms;
    // This difference fits the public pose-consistency tolerance, but moves
    // the mathematical initial point to the opposite side of the query face.
    centers[0].x()=-gap;
    const std::vector<Vec3> updates(4,Vec3::Zero()),omega(2,Vec3::Zero());
    auto dc=omega,dw=omega;
    dc[1].x()=-.1;dw[0].z()=1;
    EXPECT_THROW(general_planar_DAT(.9,1,scene.mesh,updates,
        scene.state.deformed_positions,centers,omega,scene.state.orientations,dc,dw,1),
        std::domain_error);
}

TEST(RigidDATBounds, EnclosesTheProductionQuaternionPathAcrossChangingRotationAxes) {
    std::mt19937 generator(19217);
    std::uniform_real_distribution<double> unit(-1,1);
    for(int trial=0;trial<80;++trial) {
        SCOPED_TRACE(trial);
        const auto vector=[&](double scale)->Vec3{return Vec3(unit(generator),unit(generator),unit(generator))*scale;};
        const Vec3 center=vector(100),dc=vector(.5),omega=vector(8),dw=vector(15),reference=vector(2);
        const Vec4 q0=quaternion_from_angular_velocity(Vec4(1,0,0,0),vector(1),.5);
        rigid_dat::Trajectory path;
        for(int k=0;k<3;++k) {
            path.center[k]=center[k];path.delta_center[k]=dc[k];
            path.omega[k]=omega[k];path.delta_omega[k]=dw[k];path.reference[k]=reference[k];
        }
        for(int k=0;k<4;++k)path.q0[k]=q0[k];
        path.dt=.1;
        const double a=.2,b=.7;
        const auto box=rigid_dat::trajectory_bounds(path,a,b);
        for(int sample=0;sample<=50;++sample) {
            const double t=a+(b-a)*sample/50;
            const Vec3 position=world_space_position(reference,center+t*dc,q0,omega+t*dw,path.dt);
            for(int k=0;k<3;++k) {
                EXPECT_GE(position[k],box.lo[k]);
                EXPECT_LE(position[k],box.hi[k]);
            }
        }
        const double t=.371;
        const auto point_box=rigid_dat::trajectory_bounds(path,t,t);
        const Vec3 point=world_space_position(reference,center+t*dc,q0,omega+t*dw,path.dt);
        for(int k=0;k<3;++k) {
            EXPECT_GE(point[k],point_box.lo[k]);
            EXPECT_LE(point[k],point_box.hi[k]);
        }
    }
}

TEST(RigidDATBounds, PointIntervalsIncludeTheActualNonunitQuaternionAndSmallAngleBranches) {
    for(double angle:{0.0,1e-10,1.999999e-4,2.000001e-4,.7}) {
        SCOPED_TRACE(angle);
        rigid_dat::Trajectory path;
        path.q0={1.00000000002,0,0,0};
        path.reference={1,.3,-.4};path.omega={angle,0,0};
        const auto box=rigid_dat::trajectory_bounds(path,0,0);
        const Vec3 actual=world_space_position(Vec3(1,.3,-.4),Vec3::Zero(),
            Vec4(path.q0[0],0,0,0),Vec3(angle,0,0),1);
        for(int k=0;k<3;++k) {
            EXPECT_GE(actual[k],box.lo[k]);
            EXPECT_LE(actual[k],box.hi[k]);
        }
    }
}

TEST(RigidDATSolver, MixedSolidAndRigidGravityUsesIndependentPhysicalMasses) {
    RigidDATScene scene;
    scene.params.gravity=Vec3(0,-2,0);
    scene.add_body(tetrahedron(Vec3(3,0,0)),Vec3(.3,.2,0),Vec3::Zero(),
        RigidBodyUpdateMode::TranslationOnly);
    const int first=int(scene.state.deformed_positions.size());
    const Vec3 corner(-3,0,0);
    create_solid({corner,corner+Vec3::UnitX(),corner+Vec3::UnitY(),corner+Vec3::UnitZ()},
        {0,1,2,3},24,scene.mesh,scene.state);
    for(int v=first;v<first+4;++v)scene.state.velocities[v]=Vec3(-.2,.1,0);
    scene.finish();
    const auto previous=scene.state;
    SolveState iterate(scene);
    BroadPhase phase;
    ASSERT_TRUE(solve(scene,iterate,phase).converged);
    const double dt=scene.params.dt();
    for(int v=first;v<first+4;++v)
        EXPECT_LE((iterate.positions[v]-previous.deformed_positions[v]
            -dt*previous.velocities[v]-dt*dt*scene.params.gravity).norm(),3e-13);
    EXPECT_LE((iterate.centers[0]-previous.x_coms[0]-dt*previous.v_coms[0]
        -dt*dt*scene.params.gravity).norm(),3e-13);
    expect_rigidity(scene.mesh,iterate.positions);
}
