#include "DAT.h"
#include "jacobi_step.h"
#include "mesh_utils.h"
#include "simulation.h"
#include "solid_ipc.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct RestoreOpenMP {
    const int threads = omp_get_max_threads();
    const int dynamic = omp_get_dynamic();
    ~RestoreOpenMP() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

struct Scene {
    RefMesh mesh;
    VertexTriangleMap adj;
    std::vector<Pin> pins;
    SimParams params = SimParams::zeros();
    std::vector<Vec3> x, xhat, velocity;

    Scene(std::vector<Vec3> positions = {}, std::vector<int> triangles = {})
        : x(std::move(positions)), xhat(x), velocity(x.size(), Vec3::Zero()) {
        mesh.tris = std::move(triangles);
        mesh.initialize(x);
        mesh.mass.assign(x.size(), 1.0);
        mesh.node_to_rb.assign(x.size(), -1);
        mesh.build_deformable_nodes();
        adj = build_incident_triangle_map(mesh.tris);
        params.fps = 20.0;
        params.substeps = 1;
        params.max_global_iters = 1;
        params.fixed_iters = true;
        params.dat_query_radius = 1.0;
    }
};

Scene cloth_scene() {
    Scene s({Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)},
        {0, 1, 2, 1, 3, 2});
    s.params.mu = 30.0;
    s.params.lambda = 20.0;
    s.params.kB = 0.003;
    s.params.damping = 0.5;
    s.x[0] += Vec3(-0.1, 0.03, 0.02);
    s.x[3] += Vec3(0.06, -0.02, 0.04);
    return s;
}

void append_solid(Scene& s, const Vec3& offset) {
    DeformedState state;
    state.deformed_positions = s.x;
    state.velocities = s.velocity;
    const std::vector<Vec3> rest{offset, offset + Vec3::UnitX(),
        offset + Vec3::UnitY(), offset + Vec3::UnitZ()};
    const std::size_t first = s.x.size();
    create_solid(rest, {0, 1, 2, 3}, 24.0, s.mesh, state);
    s.x = state.deformed_positions;
    s.velocity = state.velocities;
    s.xhat.insert(s.xhat.end(), rest.begin(), rest.end());
    s.x[first + 1].x() += 0.08;
    s.params.solid_mu = 25.0;
    s.params.solid_lambda = 35.0;
    s.params.damping = 0.5;
    s.mesh.build_deformable_nodes();
    s.adj = build_incident_triangle_map(s.mesh.tris);
}

Scene scene_for_kind(int kind) {
    Scene s = kind == 1 ? Scene{} : cloth_scene();
    if (kind != 0) append_solid(s, kind == 1 ? Vec3::Zero() : Vec3(4, 0, 0));
    return s;
}

SolverResult solve(Scene& s, BroadPhase& phase,
    const std::vector<Vec3>* previous = nullptr) {
    return global_jacobi_solver_dat(s.mesh, s.adj, s.pins, s.params,
        s.x, s.xhat, s.velocity, phase, "", previous);
}

std::vector<Vec3> composed_sweeps(const Scene& s, int count,
    const std::vector<Vec3>* previous = nullptr) {
    auto positions = s.x;
    for (int iteration = 0; iteration < count; ++iteration) {
        const auto raw = jacobi_step(s.mesh, s.adj, s.pins, s.params,
            positions, s.xhat, s.velocity, previous);
        const auto safe = planar_DAT(s.params.dat_gamma,
            s.params.dat_query_radius, s.mesh, raw, positions,
            s.params.dat_node_box_padding);
        for (std::size_t node = 0; node < positions.size(); ++node)
            positions[node] += safe[node];
    }
    return positions;
}

void expect_positions(const std::vector<Vec3>& actual,
    const std::vector<Vec3>& expected, double tolerance = 1e-13) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t node = 0; node < actual.size(); ++node) {
        ASSERT_TRUE(actual[node].allFinite()) << node;
        EXPECT_LE((actual[node] - expected[node]).norm(), tolerance) << node;
    }
}

} // namespace

TEST(JacobiDATSolver, OneSweepMatchesFrozenCompositionForClothSolidAndMixed) {
    for (int kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        auto s = scene_for_kind(kind);
        const auto target = s.xhat;
        const auto expected = composed_sweeps(s, 1);
        BroadPhase phase;
        // An explicit entry point selects DAT even with its driver flag off.
        ASSERT_FALSE(s.params.use_jacobi_dat);
        const auto result = solve(s, phase);
        EXPECT_EQ(result.iterations, 1);
        EXPECT_TRUE(result.converged);
        expect_positions(s.x, expected);
        expect_positions(s.xhat, target, 0.0);
    }
}

TEST(JacobiDATSolver, RepeatedParallelSweepsKeepTheTargetAndFrictionHistoryFrozen) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    Scene contact({Vec3(0, 0, 0.1), Vec3(-1, -1, 0),
        Vec3(2, -1, 0), Vec3(-1, 2, 0)}, {1, 2, 3});
    contact.params.d_hat = 0.2;
    contact.params.k_barrier = 100.0;
    contact.params.friction_coefficient = 0.3;
    contact.params.damping = 0.5;
    contact.params.max_global_iters = 5;
    auto previous = contact.x;
    previous[0].x() -= 0.02;
    contact.velocity[0] = Vec3(1.0, 0.1, -0.1);
    build_xhat(contact.xhat, previous, contact.velocity, contact.params.dt());
    const auto target = contact.xhat;
    const auto expected = composed_sweeps(contact, 5, &previous);
    for (int threads : {1, 4}) {
        SCOPED_TRACE(threads);
        omp_set_num_threads(threads);
        auto s = contact;
        s.params.use_parallel = threads > 1;
        BroadPhase phase;
        const auto result = solve(s, phase, &previous);
        EXPECT_EQ(result.iterations, 5);
        EXPECT_TRUE(result.converged);
        expect_positions(s.x, expected);
        expect_positions(s.xhat, target, 0.0);
    }
}

TEST(JacobiDATSolver, SoftPinConvergesToTheInertialGravitySpringEquilibrium) {
    Scene s({Vec3::Zero()});
    s.mesh.mass[0] = 2.0;
    s.xhat[0] = Vec3(0.1, 0.2, 0.3);
    s.pins = {Pin{0, Vec3(1, 0, 0)}};
    s.params.gravity = Vec3(0, -4, 0);
    s.params.kpin = 8.0;
    s.params.fixed_iters = false;
    s.params.max_global_iters = 4;
    s.params.tol_abs = 1e-12;
    const double dt2 = s.params.dt2();
    const Vec3 expected = (2.0 * s.xhat[0] + 2.0 * dt2 * s.params.gravity
        + dt2 * s.params.kpin * s.pins[0].target_position)
        / (2.0 + dt2 * s.params.kpin);
    BroadPhase phase;
    const auto result = solve(s, phase);
    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.has_residual);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_GT(result.initial_residual, 0.0);
    EXPECT_LE(result.final_residual, s.params.tol_abs);
    EXPECT_LE((s.x[0] - expected).norm(), 1e-13);
    EXPECT_GT(s.x[0].norm(), 0.0);
    EXPECT_GT((s.x[0] - s.pins[0].target_position).norm(), 0.1);
}

TEST(JacobiDATSolver, DATPreservesTheGapWhenBothSidesWouldCrossWithoutBarrier) {
    Scene s({Vec3(0, 0, 0.2), Vec3(-1, -1, 0),
        Vec3(2, -1, 0), Vec3(-1, 2, 0)}, {1, 2, 3});
    s.xhat[0].z() = -0.8;
    for (int node = 1; node < 4; ++node) s.xhat[node].z() = 0.6;
    ASSERT_DOUBLE_EQ(s.params.d_hat, 0.0);
    ASSERT_FALSE(s.params.use_ccd);
    BroadPhase phase;
    solve(s, phase);
    const double gap = s.x[0].z() - s.x[1].z();
    EXPECT_NEAR(gap, (1.0 - s.params.dat_gamma) * 0.2, 1e-13);
    EXPECT_GT(gap, 0.0);
    EXPECT_LT(s.x[0].z(), 0.2);
    for (int node = 1; node < 4; ++node) {
        EXPECT_GT(s.x[node].z(), 0.0);
        EXPECT_DOUBLE_EQ(s.x[node].z(), s.x[1].z());
    }
}

TEST(JacobiDATSolver, EquilibriumReturnsWithoutASweepWithZeroTolerances) {
    Scene s({Vec3(1, 2, 3)});
    s.params.fixed_iters = false;
    s.params.max_global_iters = 4;
    BroadPhase phase;
    const auto result = solve(s, phase);
    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.has_residual);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_DOUBLE_EQ(result.initial_residual, 0.0);
    EXPECT_DOUBLE_EQ(result.final_residual, 0.0);
    expect_positions(s.x, s.xhat, 0.0);
}

TEST(JacobiDATSolver, FinalResidualIncludesContactsActivatedByTheLastSweep) {
    Scene s({Vec3(0, 0, 0.3), Vec3(-1, -1, 0),
        Vec3(2, -1, 0), Vec3(-1, 2, 0)}, {1, 2, 3});
    s.xhat[0].z() = 0.1;
    s.params.d_hat = 0.2;
    s.params.k_barrier = 100.0;
    s.params.fixed_iters = false;
    s.params.tol_abs = 1e-14;
    BroadPhase phase;
    const auto result = solve(s, phase);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_FALSE(result.converged);
    EXPECT_NEAR(s.x[0].z(), 0.1, 1e-13);
    BroadPhase reference_phase;
    std::vector<AABB> boxes;
    for (const auto& position : s.x) boxes.emplace_back(position, position);
    reference_phase.initialize(boxes, s.mesh, s.params.d_hat,
        BroadPhase::InitializationMode::DeformableSolver);
    s.adj.try_emplace(0); // the isolated contact point has no incident triangle
    const double reference_residual = compute_global_residual(s.mesh, s.adj,
        s.pins, s.params, s.x, s.xhat, reference_phase);
    ASSERT_GT(reference_residual, s.params.tol_abs);
    EXPECT_NEAR(result.initial_residual, 0.2, 1e-13);
    EXPECT_NEAR(result.final_residual, reference_residual, 1e-13);
}

TEST(JacobiDATSolver, ZeroDampingDoesNotHideClothOrSolidForceResiduals) {
    auto s = scene_for_kind(2);
    s.params.damping = 0.0;
    s.params.fixed_iters = false;
    s.params.max_global_iters = 3;
    s.params.tol_abs = 1e-14;
    const auto original = s.x;
    std::vector<bool> solid(s.x.size(), false);
    for (int node : s.mesh.tet_nodes) solid[node] = true;
    double cloth_residual = 0.0, solid_residual = 0.0;
    BroadPhase phase;
    for (std::size_t node = 0; node < s.x.size(); ++node) {
        const auto local = solid[node]
            ? compute_solid_local_gradient_and_block(static_cast<int>(node),
                s.mesh, s.pins, s.params, s.x, s.xhat, phase)
            : compute_local_gradient_and_hessian_no_barrier(static_cast<int>(node),
                s.mesh, s.adj, s.pins, s.params, s.x, s.xhat);
        const double residual = local.first.cwiseAbs().maxCoeff() / s.mesh.mass[node];
        double& component = solid[node] ? solid_residual : cloth_residual;
        component = std::max(component, residual);
    }
    ASSERT_GT(cloth_residual, s.params.tol_abs);
    ASSERT_GT(solid_residual, s.params.tol_abs);
    const auto result = solve(s, phase);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 3);
    EXPECT_TRUE(result.has_residual_components);
    EXPECT_NEAR(result.initial_cloth_residual, cloth_residual, 1e-13);
    EXPECT_NEAR(result.final_cloth_residual, cloth_residual, 1e-13);
    EXPECT_NEAR(result.initial_solid_residual, solid_residual, 1e-13);
    EXPECT_NEAR(result.final_solid_residual, solid_residual, 1e-13);
    EXPECT_NEAR(result.final_residual, cloth_residual + solid_residual, 1e-13);
    expect_positions(s.x, original, 0.0);
}

TEST(JacobiDATSolver, PublicDriversMatchManualSubstepsAndSkipLegacyGuesses) {
    for (int kind = 0; kind < 3; ++kind) for (bool colored_guess : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "kind=" << kind
            << " colored_guess=" << colored_guess);
        auto s = scene_for_kind(kind);
        s.params.use_jacobi_dat = true;
        s.params.substeps = 2;
        s.params.max_global_iters = 2;
        s.params.use_verlet_guess = true;
        s.params.use_colored_ccd_guess = colored_guess;
        s.params.colored_ccd_guess_iters = 1;
        for (std::size_t node = 0; node < s.x.size(); ++node)
            s.velocity[node] = Vec3(0.2 + 0.03 * node, -0.1, 0.04);
        DeformedState actual, expected;
        actual.deformed_positions = s.x;
        actual.velocities = s.velocity;
        expected = actual;
        BroadPhase manual_phase;
        for (int substep = 0; substep < s.params.substeps; ++substep) {
            s.x = expected.deformed_positions;
            s.velocity = expected.velocities;
            build_xhat(s.xhat, s.x, s.velocity, s.params.dt());
            ASSERT_TRUE(solve(s, manual_phase, &expected.deformed_positions).converged);
            update_velocity(expected.velocities, s.x, expected.deformed_positions,
                s.params.dt());
            expected.deformed_positions = s.x;
        }
        int callbacks = 0;
        const SubstepCallback callback = [&](int substep, const std::vector<Vec3>&) {
            EXPECT_EQ(substep, callbacks++);
        };
        BroadPhase phase;
        const auto result = kind == 0
            ? advance_one_frame(actual, s.mesh, s.adj, s.pins, s.params,
                phase, 1, nullptr, callback)
            : advance_one_frame_general(actual, s.mesh, s.adj, s.pins,
                s.params, phase, 1, nullptr, callback);
        EXPECT_TRUE(result.converged);
        EXPECT_EQ(result.iterations, 4);
        EXPECT_EQ(callbacks, 2);
        expect_positions(actual.deformed_positions, expected.deformed_positions);
        expect_positions(actual.velocities, expected.velocities, 1e-11);
    }
}

TEST(JacobiDATSolver, RigidBlocksAreRejectedBeforePositionsChange) {
    Scene s;
    DeformedState state;
    create_rigid_body({Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY(), Vec3::UnitZ()},
        Vec3::Zero(), Vec4(1, 0, 0, 0), Vec3::Zero(), 4.0, s.mesh, state);
    s.x = state.deformed_positions;
    s.xhat = s.x;
    s.velocity = state.velocities;
    s.params.use_jacobi_dat = true;
    const auto original = s.x;
    BroadPhase phase;
    EXPECT_THROW(solve(s, phase), std::invalid_argument);
    expect_positions(s.x, original, 0.0);
    EXPECT_THROW(advance_one_frame_general(state, s.mesh, s.adj, s.pins,
        s.params, phase), std::invalid_argument);
    expect_positions(state.deformed_positions, original, 0.0);
}
