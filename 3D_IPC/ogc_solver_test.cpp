#include "IPC_math.h"
#include "broad_phase.h"
#include "friction_energy.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "node_triangle_distance.h"
#include "ogc_contact_cache.h"
#include "physics.h"
#include "safe_step.h"
#include "segment_segment_distance.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

TEST(OGCSolver, FrozenContactSweepsAreBitwiseEquivalentAcrossThreadCounts) {
    struct RestoreOpenMP {
        const int threads = omp_get_max_threads();
        const int dynamic = omp_get_dynamic();
        ~RestoreOpenMP() {
            omp_set_num_threads(threads);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    omp_set_num_threads(1);

    for (const bool bumpy : {true, false}) {
        SCOPED_TRACE(::testing::Message() << "bumpy=" << bumpy);
        omp_set_num_threads(1);
        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> material;
        constexpr int subdivisions = 8;
        constexpr int sheet_vertices =
            (subdivisions + 1) * (subdivisions + 1);
        build_square_mesh(mesh, state, material, subdivisions, subdivisions,
            0.45, 0.45, Vec3(-0.23, 0.0, -0.24));
        build_square_mesh(mesh, state, material, subdivisions, subdivisions,
            0.45, 0.45, Vec3(-0.227, 0.008, -0.237));
        ASSERT_FALSE(mesh.hinges.empty());

        // Retain the flat rest data while activating membrane and bending terms.
        // Both sheets also slide against each other, so frozen normal and friction
        // contacts couple vertices from otherwise independent elastic components.
        std::vector<Vec3>& initial = state.deformed_positions;
        std::vector<Vec3> velocity(initial.size());
        for (std::size_t node = 0; node < initial.size(); ++node) {
            Vec3& position = initial[node];
            position.x() *= 1.06;
            if (bumpy)
                position.y() += 0.001 * std::sin(8.0 * position.x())
                    * std::sin(7.0 * position.z());
            const double direction = node < sheet_vertices ? 1.0 : -1.0;
            velocity[node] = Vec3(0.035 * direction, -0.01,
                0.025 * direction);
        }
        const VertexTriangleMap adjacency = build_incident_triangle_map(mesh.tris);
        std::vector<Pin> pins;
        append_pin(pins, 0, initial);
        append_pin(pins, subdivisions, initial);
        append_pin(pins, sheet_vertices, initial);
        pins.back().target_position.x() += 0.001;

        SimParams params = SimParams::zeros();
        params.fps = 60.0;
        params.substeps = 1;
        params.mu = 20.0;
        params.lambda = 25.0;
        params.density = 1.0;
        params.thickness = 0.1;
        params.kB = 0.003;
        params.kpin = 100.0;
        params.gravity = Vec3(0.0, -1.0, 0.0);
        params.d_hat = 0.012;
        params.k_barrier = 1.0;
        params.friction_velocity_epsilon = 0.01;
        params.max_global_iters = 5;
        params.fixed_iters = true;
        params.damping = 0.45;
        // A fixed box size isolates thread equivalence from the solver's retained
        // displacement history across calls on the same mesh.
        params.node_box_min = 0.02;
        params.node_box_max = 0.02;
        mesh.build_lumped_mass(params.density, params.thickness);

        std::vector<AABB> boxes;
        const Vec3 radius = Vec3::Constant(params.node_box_max + params.d_hat);
        for (const Vec3& position : initial)
            boxes.emplace_back(position - radius, position + radius);
        BroadPhase contacts;
        contacts.initialize(boxes, mesh, params.d_hat);
        int active_nt = 0;
        int active_ss = 0;
        std::size_t active_robust_ss = 0;
        for (const auto& pair : contacts.cache().nt_pairs) {
            const double distance = node_triangle_distance(initial[pair.node],
                initial[pair.tri_v[0]], initial[pair.tri_v[1]],
                initial[pair.tri_v[2]]).distance;
            if (distance > 0.0 && distance < params.d_hat)
                ++active_nt;
        }
        for (const auto& pair : contacts.cache().ss_pairs) {
            const auto distance = segment_segment_distance(initial[pair.v[0]],
                initial[pair.v[1]], initial[pair.v[2]],
                initial[pair.v[3]]);
            if (distance.distance > 0.0 && distance.distance < params.d_hat)
                ++active_ss;
            if (distance.robust && distance.distance < params.d_hat
                && segment_aabbs_within_distance(initial[pair.v[0]],
                    initial[pair.v[1]], initial[pair.v[2]], initial[pair.v[3]],
                    params.d_hat * params.d_hat))
                ++active_robust_ss;
        }
        ASSERT_GT(active_nt, 0);
        ASSERT_GT(active_ss, 0);
        if (!bumpy) {
            // Flat parallel edges must cross the production cache threshold so
            // these same bitwise comparisons exercise the shared-Hessian branch.
            ASSERT_GT(active_robust_ss, 8U);
            ASSERT_GE(active_robust_ss, contacts.ss_pairs().size() / 512);
        }

        std::vector<Vec3> previous = initial;
        std::vector<Vec3> xhat = initial;
        for (std::size_t node = 0; node < initial.size(); ++node) {
            previous[node] -= params.dt() * velocity[node];
            xhat[node] += params.dt() * velocity[node];
        }
        for (const bool exact : {false, true}) {
            params.exact_computation_fallback = exact;
            for (const double friction : {0.0, 0.45}) {
                params.friction_coefficient = friction;
                SCOPED_TRACE(::testing::Message()
                    << "exact=" << exact << " friction=" << friction);
                std::vector<Vec3> reference;
                for (const bool parallel : {false, true}) {
                    params.use_parallel = parallel;
                    for (const int threads : {1, 4}) {
                        SCOPED_TRACE(::testing::Message()
                            << "parallel=" << parallel << " threads=" << threads);
                        omp_set_num_threads(threads);
                        std::vector<Vec3> current = initial;
                        const SolverResult result = global_gauss_seidel_solver_ogc(
                            mesh, adjacency, pins, params, current, xhat, velocity,
                            "", &previous);
                        ASSERT_TRUE(result.converged);
                        ASSERT_EQ(result.iterations, params.max_global_iters);
                        double total_motion = 0.0;
                        for (std::size_t node = 0; node < current.size(); ++node) {
                            ASSERT_TRUE(current[node].allFinite()) << "node=" << node;
                            total_motion += (current[node] - initial[node]).squaredNorm();
                            if (!reference.empty()) {
                                EXPECT_EQ(std::memcmp(current[node].data(),
                                    reference[node].data(), 3 * sizeof(double)), 0)
                                    << "node=" << node;
                            }
                        }
                        EXPECT_GT(total_motion, 1.0e-12);
                        if (reference.empty())
                            reference = current;
                    }
                }
            }
        }
    }
}

TEST(DeformableFrictionSolver,
     BasicAndOgcMeshContactsMatchLegacyLocalAssembly) {
    RefMesh ref_mesh;
    ref_mesh.num_positions = 4;
    ref_mesh.tris = {1, 2, 3};
    ref_mesh.mass.assign(4, 1.0);

    const std::vector<Vec3> initial{
        Vec3(0.25, 0.25, 0.10),
        Vec3(0.00, 0.00, 0.00),
        Vec3(1.00, 0.00, 0.00),
        Vec3(0.00, 1.00, 0.00)};
    std::vector<Vec3> previous = initial;
    previous[0] -= Vec3(0.035, -0.018, 0.0);
    const std::vector<Vec3> velocities(4, Vec3::Zero());
    VertexTriangleMap adjacency;
    for (int node = 0; node < 4; ++node)
        adjacency[node] = {};

    SimParams params = SimParams::zeros();
    params.fps = 20.0;
    params.substeps = 1;
    params.d_hat = 0.30;
    params.k_barrier = 29.0;
    params.friction_coefficient = 0.53;
    params.friction_velocity_epsilon = 0.08;
    params.max_global_iters = 1;
    params.fixed_iters = true;
    params.damping = 0.13;
    params.node_box_min = 2.0;
    params.node_box_max = 2.0;
    params.node_box_update_count = 1;
    params.use_ccd = false;
    params.use_parallel = false;

    solver_detail::OGCFrozenContactOperands operands;
    operands.resize(1, 0);
    NodeTrianglePair operand_pair{};
    operand_pair.node = 0;
    operand_pair.tri_v[0] = 1;
    operand_pair.tri_v[1] = 2;
    operand_pair.tri_v[2] = 3;
    const auto distance = node_triangle_distance(
        initial[0], initial[1], initial[2], initial[3]);
    operands.prepare_node_triangle(0, operand_pair, initial, distance, params.d_hat);
    ASSERT_EQ(operands.nt_aabb_active[0], 1);
    EXPECT_EQ(operands.size_bytes(), 17U);
    for (int role = 0; role < 4; ++role) {
        const auto original = node_triangle_barrier_self_gradient_and_hessian(
            initial[0], initial[1], initial[2], initial[3], params.d_hat,
            role, 1.0e-12, &distance);
        const auto cached = node_triangle_barrier_self_gradient_and_hessian(
            initial[0], initial[1], initial[2], initial[3], params.d_hat,
            role, 1.0e-12, &distance, &operands.nt_b_prime[0],
            &operands.nt_b_double_prime[0]);
        EXPECT_EQ(std::memcmp(original.first.data(), cached.first.data(),
            3 * sizeof(double)), 0);
        EXPECT_EQ(std::memcmp(original.second.data(), cached.second.data(),
            9 * sizeof(double)), 0);
    }
    // Refreshing an accepted but barrier-inactive pair must retain its AABB
    // flag, because the consumer still performs the original zero additions.
    std::vector<Vec3> changed = initial;
    changed[0] = Vec3(0.8, 0.8, 0.1);
    const auto outside_face = node_triangle_distance(
        changed[0], changed[1], changed[2], changed[3]);
    ASSERT_GT(outside_face.distance, params.d_hat);
    operands.prepare_node_triangle(0, operand_pair, changed, outside_face, params.d_hat);
    EXPECT_EQ(operands.nt_aabb_active[0], 1);
    EXPECT_DOUBLE_EQ(operands.nt_b_prime[0], 0.0);
    EXPECT_DOUBLE_EQ(operands.nt_b_double_prime[0], 0.0);
    // A rejected pair must not inspect singular scalar data from its supplied
    // distance record. An accepted zero-distance normal pair must still fail,
    // including when a zero normal stiffness would multiply its contribution.
    auto singular = distance;
    singular.distance = 0.0;
    changed[0].z() = 3.0;
    EXPECT_NO_THROW(operands.prepare_node_triangle(
        0, operand_pair, changed, singular, params.d_hat));
    EXPECT_EQ(operands.nt_aabb_active[0], 0);
    changed[0] = Vec3(0.25, 0.25, 0.0);
    EXPECT_THROW(operands.prepare_node_triangle(
        0, operand_pair, changed, singular, params.d_hat), std::runtime_error);
    const std::array<Vec3, 4> coincident{
        changed[0], changed[1], changed[2], changed[3]};
    EXPECT_NO_THROW(make_node_triangle_contact_evaluation(
        coincident, params.d_hat, 0.0, 1.0e-12, &singular));

    solver_detail::OGCFrozenFrictionOperands friction_operands;
    const auto prepare_friction = [&](const std::vector<Vec3>& positions,
        const NodeTriangleDistanceResult& dr, const SimParams& settings) {
        friction_operands.resize(1, 0);
        friction_operands.classify_node_triangle(0, operand_pair, positions,
            dr, settings.d_hat, settings.k_barrier);
        friction_operands.compact();
        friction_operands.prepare_node_triangle(0, operand_pair, positions,
            previous, dr, settings);
    };
    prepare_friction(initial, distance, params);
    ASSERT_EQ(friction_operands.nt_records.size(), 1U);
    const std::array<Vec3, 4> current_contact{
        initial[0], initial[1], initial[2], initial[3]};
    const std::array<Vec3, 4> prior_contact{
        previous[0], previous[1], previous[2], previous[3]};
    const auto evaluation = make_node_triangle_contact_evaluation(
        current_contact, params.d_hat, params.k_barrier, 1.0e-12, &distance);
    const auto frozen = make_node_triangle_frozen_friction_contact(
        current_contact, prior_contact, evaluation, params.dt(),
        params.friction_velocity_epsilon);
    for (int role = 0; role < 4; ++role) {
        const auto original_normal = node_triangle_barrier_self_gradient_and_hessian(
            initial[0], initial[1], initial[2], initial[3], role, evaluation);
        const auto cached_normal = node_triangle_barrier_self_gradient_and_hessian(
            initial[0], initial[1], initial[2], initial[3], role,
            friction_operands.node_triangle_evaluation(0));
        const auto original_friction = frozen_friction_role_gradient_and_hessian(
            frozen, role, params.friction_coefficient, params.dt2());
        const auto cached_friction =
            friction_operands.node_triangle_role_derivatives(0, role);
        EXPECT_EQ(std::memcmp(original_normal.first.data(), cached_normal.first.data(),
            3 * sizeof(double)), 0);
        EXPECT_EQ(std::memcmp(original_normal.second.data(), cached_normal.second.data(),
            9 * sizeof(double)), 0);
        EXPECT_EQ(std::memcmp(original_friction.first.data(), cached_friction.first.data(),
            3 * sizeof(double)), 0);
        EXPECT_EQ(std::memcmp(original_friction.second.data(), cached_friction.second.data(),
            9 * sizeof(double)), 0);
    }
    changed[0] = Vec3(0.8, 0.8, 0.1);
    prepare_friction(changed, outside_face, params);
    EXPECT_EQ(friction_operands.nt_aabb_active[0], 1);
    EXPECT_TRUE(friction_operands.nt_records.empty());
    EXPECT_FALSE(friction_operands.node_triangle_evaluation(0).active);
    EXPECT_TRUE(friction_operands.node_triangle_role_derivatives(0, 0).first.isZero());
    EXPECT_TRUE(friction_operands.node_triangle_role_derivatives(0, 0).second.isZero());
    EXPECT_THROW(friction_operands.node_triangle_role_derivatives(0, 4),
        std::invalid_argument);
    // Accepted inactive pairs still validate the coefficient once when their
    // relative derivatives are prepared.
    SimParams negative_friction = params;
    negative_friction.friction_coefficient = -0.1;
    EXPECT_THROW(prepare_friction(changed, outside_face, negative_friction),
        std::invalid_argument);
    SimParams zero_stiffness = params;
    zero_stiffness.k_barrier = 0.0;
    changed[0] = Vec3(0.25, 0.25, 0.0);
    EXPECT_NO_THROW(prepare_friction(changed, singular, zero_stiffness));
    EXPECT_TRUE(friction_operands.nt_records.empty());
    EXPECT_TRUE(friction_operands.node_triangle_role_derivatives(0, 0).first.isZero());
    SimParams negative_stiffness = params;
    negative_stiffness.k_barrier = -1.0;
    EXPECT_THROW(prepare_friction(initial, distance, negative_stiffness),
        std::invalid_argument);
    changed[0].z() = 3.0;
    EXPECT_NO_THROW(prepare_friction(changed, singular, negative_stiffness));
    EXPECT_EQ(friction_operands.nt_aabb_active[0], 0);

    const auto legacy_delta = [&](const std::vector<Vec3>& positions,
                                  const int role) -> Vec3 {
        const std::array<Vec3, 4> current{
            positions[0], positions[1], positions[2], positions[3]};
        const std::array<Vec3, 4> prior{
            previous[0], previous[1], previous[2], previous[3]};
        const auto normal =
            node_triangle_barrier_self_gradient_and_hessian(
                current[0], current[1], current[2], current[3],
                params.d_hat, role);
        const FrozenFrictionContact friction_contact =
            make_node_triangle_frozen_friction_contact(
                current, prior, params.d_hat, params.k_barrier,
                params.dt(), params.friction_velocity_epsilon);
        const auto friction =
            frozen_friction_role_gradient_and_hessian(
                friction_contact, role, params.friction_coefficient,
                params.dt2());
        const Vec3 gradient = params.dt2() * params.k_barrier
                * normal.first
            + friction.first;
        const Mat33 hessian = Mat33::Identity()
            + params.dt2() * params.k_barrier * normal.second
            + friction.second;
        return matrix3d_inverse(hessian) * gradient;
    };

    // Basic GS consumes the live position array, so reproduce the old local
    // kernel one role at a time in solver order.
    std::vector<Vec3> expected_basic = initial;
    for (int role = 0; role < 4; ++role) {
        expected_basic[static_cast<std::size_t>(role)] -=
            params.damping * legacy_delta(expected_basic, role);
    }
    std::vector<Vec3> actual_basic = initial;
    BroadPhase basic_broad_phase;
    const SolverResult basic_result = global_gauss_seidel_solver_basic(
        ref_mesh, adjacency, {}, params, actual_basic, initial, velocities,
        basic_broad_phase, "", &previous);
    ASSERT_TRUE(basic_result.converged);
    ASSERT_EQ(basic_broad_phase.cache().nt_pairs.size(), 1U);
    ASSERT_TRUE(basic_broad_phase.cache().ss_pairs.empty());
    for (int node = 0; node < 4; ++node) {
        EXPECT_TRUE(actual_basic[static_cast<std::size_t>(node)].isApprox(
            expected_basic[static_cast<std::size_t>(node)], 2.0e-14))
            << "basic node=" << node
            << " actual="
            << actual_basic[static_cast<std::size_t>(node)].transpose()
            << " expected="
            << expected_basic[static_cast<std::size_t>(node)].transpose();
    }

    // OGC evaluates every contact role from the same iteration-start
    // snapshot, then applies the same 0.4*d trust-region bound as production.
    const double frozen_distance = node_triangle_distance(
        initial[0], initial[1], initial[2], initial[3]).distance;
    const double trust_region_bound = 0.4 * frozen_distance;
    std::vector<Vec3> expected_ogc = initial;
    for (int role = 0; role < 4; ++role) {
        const Vec3 raw_step =
            -params.damping * legacy_delta(initial, role);
        const double step_norm = raw_step.norm();
        const double alpha = step_norm > 0.0
            ? std::min(1.0, trust_region_bound / step_norm)
            : 1.0;
        expected_ogc[static_cast<std::size_t>(role)] =
            initial[static_cast<std::size_t>(role)] + alpha * raw_step;
    }
    std::vector<Vec3> actual_ogc = initial;
    const SolverResult ogc_result = global_gauss_seidel_solver_ogc(
        ref_mesh, adjacency, {}, params, actual_ogc, initial, velocities,
        "", &previous);
    ASSERT_TRUE(ogc_result.converged);
    for (int node = 0; node < 4; ++node) {
        EXPECT_TRUE(actual_ogc[static_cast<std::size_t>(node)].isApprox(
            expected_ogc[static_cast<std::size_t>(node)], 2.0e-14))
            << "OGC node=" << node
            << " actual="
            << actual_ogc[static_cast<std::size_t>(node)].transpose()
            << " expected="
            << expected_ogc[static_cast<std::size_t>(node)].transpose();
    }
}

TEST(DeformableSDFFrictionSolver,
     BasicAndOgcReduceTangentialSlipOnPlane) {
    const auto run = [](const double friction_coefficient,
                        const bool use_ogc_solver) {
        RefMesh ref_mesh;
        ref_mesh.mass = {1.0};
        ref_mesh.num_positions = 1;
        const VertexTriangleMap adjacency = {{0, IncidentTriangles{}}};
        const std::vector<Vec3> previous = {
            Vec3(0.0, 0.05, 0.0)};
        const std::vector<Vec3> velocity = {Vec3::UnitX()};

        SimParams params = SimParams::zeros();
        params.fps = 10.0;
        params.substeps = 1;
        params.k_sdf = 100.0;
        params.eps_sdf = 0.1;
        params.sdf_planes = {
            PlaneSDF{Vec3::Zero(), Vec3::UnitY()}};
        params.friction_coefficient = friction_coefficient;
        params.friction_velocity_epsilon = 0.01;
        params.max_global_iters = 6;
        params.fixed_iters = true;
        params.damping = 0.5;
        params.node_box_min = 0.5;
        params.node_box_max = 0.5;
        params.node_box_update_count = 1;
        params.use_ccd = false;
        params.use_parallel = false;

        const std::vector<Vec3> xhat = {
            previous[0] + params.dt() * velocity[0]};
        std::vector<Vec3> current = previous;
        BroadPhase broad_phase;
        if (use_ogc_solver) {
            global_gauss_seidel_solver_ogc(
                ref_mesh, adjacency, {}, params, current, xhat, velocity,
                "", &previous);
        } else {
            global_gauss_seidel_solver_basic(
                ref_mesh, adjacency, {}, params, current, xhat, velocity,
                broad_phase, "", &previous);
        }
        return current[0];
    };

    for (const bool use_ogc_solver : {false, true}) {
        SCOPED_TRACE(use_ogc_solver ? "OGC" : "basic");
        const Vec3 frictionless = run(0.0, use_ogc_solver);
        const Vec3 frictional = run(0.6, use_ogc_solver);
        EXPECT_GT(frictionless.x(), 1.0e-3);
        EXPECT_GT(frictional.x(), 0.0);
        EXPECT_LT(frictional.x(), frictionless.x() - 1.0e-4);
        EXPECT_NEAR(frictional.y(), frictionless.y(), 1.0e-12);
        EXPECT_TRUE(frictional.allFinite());
    }
}

TEST(ExactComputationFallback, OGCTrustRadiusUsesTheSelectedDistancePolicy) {
    // The legacy absolute determinant threshold selects a boundary feature
    // for these short crossing edges; exact fallback resolves the interior gap.
    constexpr double length=1e-7, gap=1e-9;
    const std::vector<Vec3> initial{
        Vec3(-length/2,0,0),Vec3(length/2,0,0),
        Vec3(0,-length/2,gap),Vec3(0,length/2,gap)};
    BroadPhase phase;
    auto& cache=phase.mutable_cache();
    cache.node_boxes.assign(4,AABB(Vec3::Constant(-1),Vec3::Constant(1)));
    cache.vertex_nt.resize(4);
    cache.vertex_ss.resize(4);
    SegmentSegmentPair pair{};
    for(int i=0;i<4;++i)pair.v[i]=i;
    cache.ss_pairs.push_back(pair);
    cache.vertex_ss[0].push_back({0,0});
    for(bool fallback:{false,true}) {
        const auto distance = segment_segment_distance(initial[0], initial[1],
            initial[2], initial[3], 1.0e-12, fallback);
        solver_detail::OGCFrozenContactOperands operands;
        operands.resize(0, 1);
        operands.prepare_segment_segment(0, pair, initial, distance, 0.4);
        ASSERT_EQ(operands.ss_aabb_active[0], 1);
        solver_detail::OGCFrozenSegmentHessians hessians;
        hessians.resize(1);
        hessians.classify(0, distance, 0.4, operands.ss_aabb_active[0]);
        hessians.compact();
        hessians.prepare_segment_segment(0, pair, initial, distance, 0.4,
            operands.ss_b_prime[0], operands.ss_b_double_prime[0]);
        EXPECT_EQ(hessians.records.size(), distance.robust ? 1U : 0U);
        for (int role = 0; role < 4; ++role) {
            const auto original = segment_segment_barrier_self_gradient_and_hessian(
                initial[0], initial[1], initial[2], initial[3], 0.4, role,
                1.0e-12, &distance, nullptr, nullptr, fallback);
            const auto cached = segment_segment_barrier_self_gradient_and_hessian(
                initial[0], initial[1], initial[2], initial[3], 0.4, role,
                1.0e-12, &distance, &operands.ss_b_prime[0],
                &operands.ss_b_double_prime[0], fallback);
            EXPECT_EQ(std::memcmp(original.first.data(), cached.first.data(),
                3 * sizeof(double)), 0);
            EXPECT_EQ(std::memcmp(original.second.data(), cached.second.data(),
                9 * sizeof(double)), 0);
            const Mat33* shared_hessian = hessians.self_hessian(0, role);
            if (distance.robust) {
                ASSERT_NE(shared_hessian, nullptr);
                EXPECT_EQ(std::memcmp(original.second.data(), shared_hessian->data(),
                    9 * sizeof(double)), 0);
                const Vec3 gradient = segment_segment_barrier_gradient(
                    initial[0], initial[1], initial[2], initial[3], 0.4, role,
                    1.0e-12, &distance, &operands.ss_b_prime[0], fallback);
                EXPECT_EQ(std::memcmp(original.first.data(), gradient.data(),
                    3 * sizeof(double)), 0);
            } else EXPECT_EQ(shared_hessian, nullptr);
        }
        EXPECT_THROW(hessians.self_hessian(0, 4), std::invalid_argument);
        std::vector<Vec3> far = initial;
        far[2].z() = far[3].z() = 3.0;
        auto singular = distance;
        singular.distance = 0.0;
        EXPECT_NO_THROW(operands.prepare_segment_segment(
            0, pair, far, singular, 0.4));
        EXPECT_EQ(operands.ss_aabb_active[0], 0);
        EXPECT_DOUBLE_EQ(operands.ss_b_prime[0], 0.0);
        EXPECT_DOUBLE_EQ(operands.ss_b_double_prime[0], 0.0);

        SimParams friction_params = SimParams::zeros();
        friction_params.fps = 60.0;
        friction_params.substeps = 1;
        friction_params.d_hat = 0.4;
        friction_params.k_barrier = 2.0;
        friction_params.friction_coefficient = 0.45;
        friction_params.exact_computation_fallback = fallback;
        solver_detail::OGCFrozenFrictionOperands friction_operands;
        friction_operands.resize(0, 1);
        friction_operands.classify_segment_segment(
            0, pair, initial, distance, friction_params.d_hat,
            friction_params.k_barrier);
        friction_operands.compact();
        std::vector<Vec3> previous = initial;
        previous[0] -= Vec3(length / 5, length / 3, 0.0);
        friction_operands.prepare_segment_segment(
            0, pair, initial, previous, distance, friction_params);
        ASSERT_EQ(friction_operands.ss_records.size(), 1U);
        const std::array<Vec3, 4> current{initial[0], initial[1], initial[2], initial[3]};
        const std::array<Vec3, 4> prior{previous[0], previous[1], previous[2], previous[3]};
        const auto evaluation = make_segment_segment_contact_evaluation(current,
            friction_params.d_hat, friction_params.k_barrier, 1.0e-12,
            &distance, fallback);
        hessians.classify(0, distance, friction_params.d_hat,
            friction_operands.ss_aabb_active[0], evaluation.active);
        hessians.compact();
        hessians.prepare_segment_segment(0, pair, initial, distance,
            friction_params.d_hat, evaluation.b_prime, evaluation.b_double_prime);
        const auto frozen = make_segment_segment_frozen_friction_contact(
            current, prior, evaluation, friction_params.dt(),
            friction_params.friction_velocity_epsilon);
        for (int role = 0; role < 4; ++role) {
            const auto original = frozen_friction_role_gradient_and_hessian(
                frozen, role, 0.45, friction_params.dt2());
            const auto cached =
                friction_operands.segment_segment_role_derivatives(0, role);
            EXPECT_EQ(std::memcmp(original.first.data(), cached.first.data(),
                3 * sizeof(double)), 0);
            EXPECT_EQ(std::memcmp(original.second.data(), cached.second.data(),
                9 * sizeof(double)), 0);
            if (const Mat33* shared_hessian = hessians.self_hessian(0, role)) {
                const auto original_normal = segment_segment_barrier_self_gradient_and_hessian(
                    initial[0], initial[1], initial[2], initial[3], role, evaluation);
                EXPECT_EQ(std::memcmp(original_normal.second.data(), shared_hessian->data(),
                    9 * sizeof(double)), 0);
            }
        }
        // Friction's zero-stiffness normal evaluation is inactive, whereas
        // the normal-only path must not use stiffness to disable its kernel.
        const auto zero_stiffness = make_segment_segment_contact_evaluation(
            current, friction_params.d_hat, 0.0, 1.0e-12, &singular, fallback);
        hessians.classify(0, singular, friction_params.d_hat, true, zero_stiffness.active);
        hessians.compact();
        EXPECT_NO_THROW(hessians.prepare_segment_segment(0, pair, initial, singular,
            friction_params.d_hat, zero_stiffness.b_prime, zero_stiffness.b_double_prime));
        EXPECT_TRUE(hessians.records.empty());
        if (distance.robust) {
            hessians.classify(0, singular, friction_params.d_hat, true);
            hessians.compact();
            EXPECT_THROW(hessians.prepare_segment_segment(0, pair, initial, singular,
                friction_params.d_hat, evaluation.b_prime, evaluation.b_double_prime),
                std::runtime_error);
        }
        const double expected=.4*(fallback ? gap : std::hypot(length/2,gap));
        EXPECT_NEAR(compute_trust_region_bound_for_vertex(0,initial,phase,.4,fallback),expected,1e-20);
        for(bool ticcd:{false,true}) {
            auto x=initial;
            const Vec3 target=x[0]+Vec3(0,0,length);
            const double step=per_vertex_safe_step(phase,x,0,target,.9,false,ticcd,
                                                  true,false,nullptr,fallback);
            EXPECT_NEAR(step,expected/length,1e-13);
            EXPECT_NEAR(x[0].z(),expected,1e-20);
        }
    }

    // All nine active parameter features, parallel segments, and degeneracy
    // share the original per-role Hessian arithmetic. Repeated classification
    // also exercises sparse-record removal and restoration across threads.
    std::vector<std::array<Vec3, 4>> fixtures;
    for (double horizontal : {-1.0, .5, 2.0})
        for (double vertical : {-2.0, -.5, 1.0})
            fixtures.push_back({Vec3::Zero(), Vec3::UnitX(),
                Vec3(horizontal, vertical, .125), Vec3(horizontal, vertical+1, .125)});
    fixtures.push_back({Vec3::Zero(), Vec3::UnitX(), Vec3(0,.125,0), Vec3(1,.125,0)});
    fixtures.push_back({Vec3(.5,0,0), Vec3(.5,0,0), Vec3(0,.125,0), Vec3(1,.125,0)});
    fixtures.push_back({Vec3::Zero(), Vec3::Zero(), Vec3(0,.125,0), Vec3(0,.125,0)});
    struct RestoreThreads {
        int count = omp_get_max_threads();
        ~RestoreThreads() { omp_set_num_threads(count); }
    } restore;
    for (int threads : {1, 4}) {
        omp_set_num_threads(threads);
        solver_detail::OGCFrozenSegmentHessians shared;
        shared.resize(fixtures.size());
        std::vector<Vec3> positions(4*fixtures.size());
        std::vector<SegmentSegmentPair> pairs(fixtures.size());
        std::vector<SegmentSegmentDistanceResult> distances(fixtures.size());
        std::vector<double> first(fixtures.size()), second(fixtures.size());
        for (int refresh = 0; refresh < 3; ++refresh) {
            for (std::size_t contact = 0; contact < fixtures.size(); ++contact) {
                for (int role = 0; role < 4; ++role) {
                    pairs[contact].v[role] = static_cast<int>(4*contact + role);
                    positions[4*contact+role] = fixtures[contact][role];
                    if (role >= 2)
                        positions[4*contact+role].z() += refresh == 1 ? 10.0 : .001*refresh;
                }
                const auto* x = positions.data() + 4*contact;
                distances[contact] = segment_segment_distance(x[0], x[1], x[2], x[3], 8.0, true);
                ASSERT_TRUE(distances[contact].robust);
                first[contact] = scalar_barrier_gradient(distances[contact].distance, 4.0);
                second[contact] = scalar_barrier_hessian(distances[contact].distance, 4.0);
                shared.classify(contact, distances[contact], 4.0,
                    segment_aabbs_within_distance(x[0],x[1],x[2],x[3],16.0));
            }
            shared.compact();
            EXPECT_EQ(shared.records.size(), refresh == 1 ? 0U : fixtures.size());
            #pragma omp parallel for schedule(dynamic, 1)
            for (std::size_t contact = 0; contact < fixtures.size(); ++contact)
                shared.prepare_segment_segment(contact, pairs[contact], positions,
                    distances[contact], 4.0, first[contact], second[contact]);
            for (std::size_t contact = 0; contact < fixtures.size(); ++contact) {
                const auto* x = positions.data() + 4*contact;
                for (int role = 0; role < 4; ++role) {
                    SCOPED_TRACE(::testing::Message() << threads << ':' << refresh << ':' << contact << ':' << role);
                    const Mat33* actual = shared.self_hessian(contact, role);
                    if (refresh == 1) { EXPECT_EQ(actual, nullptr); continue; }
                    ASSERT_NE(actual, nullptr);
                    const Mat33 expected = segment_segment_barrier_self_hessian(
                        x[0],x[1],x[2],x[3],4.0,role,1.0e-12,&distances[contact],
                        &first[contact],&second[contact]);
                    EXPECT_EQ(std::memcmp(actual->data(), expected.data(), 9*sizeof(double)),0);
                }
            }
        }
    }
}

TEST(OGCSolver, TrustRegionSafeStepClampsToIncidentContactDistance) {
    NodeTrianglePair pair{};
    pair.node = 0;
    pair.tri_v[0] = 1;
    pair.tri_v[1] = 2;
    pair.tri_v[2] = 3;

    // Exercise the production OGC path with an incident pair initially 0.5
    // apart. Its trust radius is 0.4 * 0.5 = 0.2, so a 0.25 displacement is
    // scaled to 0.8 even though CCD clipping is disabled.
    std::vector<Vec3> x_ogc = {
        Vec3(0.0, 0.5, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    };
    const std::vector<Vec3> target_ogc = {
        Vec3(0.0, 0.25, 0.0),
        x_ogc[1], x_ogc[2], x_ogc[3],
    };

    BroadPhase ogc_bp;
    auto& ogc_cache = ogc_bp.mutable_cache();
    ogc_cache.node_boxes.assign(x_ogc.size(), AABB(Vec3::Constant(-2.0), Vec3::Constant(2.0)));
    ogc_cache.vertex_nt.resize(x_ogc.size());
    ogc_cache.vertex_ss.resize(x_ogc.size());
    ogc_cache.nt_pairs.push_back(pair);
    ogc_cache.vertex_nt[0].push_back({/*pair_index=*/0, /*dof=*/0});

    const double ogc_weight = per_vertex_safe_step(ogc_bp, x_ogc, 0, target_ogc[0], /*safety=*/0.9, /*clip_ccd=*/false, /*use_ticcd=*/false, /*use_ogc=*/true);

    EXPECT_DOUBLE_EQ(ogc_weight, 0.8);
    EXPECT_TRUE(x_ogc[0].isApprox(Vec3(0.0, 0.3, 0.0), 1.0e-12));

    // A barrier-inactive pair must still limit the production solver's trust
    // radius. The isolated node approaches a collision-only triangle; empty
    // elastic rows keep the expected update independent of material terms.
    RefMesh mesh;
    mesh.num_positions = 4;
    mesh.tris = {1, 2, 3};
    mesh.mass.assign(4, 1.0);
    const std::vector<Vec3> initial{
        Vec3(.25, .5, .25), Vec3(0, 0, 0),
        Vec3(1, 0, 0), Vec3(0, 0, 1)};
    VertexTriangleMap adjacency;
    for (int node = 0; node < 4; ++node) adjacency[node] = {};
    std::vector<Vec3> xhat = initial;
    xhat[0].y() = -.5;
    const std::vector<Vec3> velocity(4, Vec3::Zero());
    const std::vector<Pin> pins;
    SimParams params = SimParams::zeros();
    params.fps = 60.0;
    params.substeps = 1;
    params.fixed_iters = true;
    params.use_ccd = false;
    params.damping = 1.0;
    params.node_box_min = params.node_box_max = 1.0;
    params.d_hat = .01;
    params.k_barrier = 1.0;

    std::vector<AABB> boxes;
    const Vec3 radius = Vec3::Constant(params.node_box_max + params.d_hat);
    for (const Vec3& point : initial)
        boxes.emplace_back(point - radius, point + radius);
    BroadPhase search;
    search.initialize(boxes, mesh, params.d_hat,
        BroadPhase::InitializationMode::BulkRefittable);
    ASSERT_EQ(search.nt_pairs().size(), 1U);
    ASSERT_EQ(search.nt_pairs()[0].node, 0);
    const auto& candidate = search.nt_pairs()[0];
    const auto distance = node_triangle_distance(initial[candidate.node],
        initial[candidate.tri_v[0]], initial[candidate.tri_v[1]],
        initial[candidate.tri_v[2]]);
    ASSERT_DOUBLE_EQ(distance.distance, .5);
    solver_detail::OGCFrozenContactOperands operands;
    operands.resize(search.nt_pairs().size(), search.ss_pairs().size());
    operands.prepare_node_triangle(0, candidate, initial, distance, params.d_hat);
    ASSERT_EQ(operands.nt_aabb_active[0], 0);

    for (const double friction : {0.0, .45}) {
        params.friction_coefficient = friction;
        // Five sweeps require full incidence to return after each frozen AABB
        // compaction. Restarting at one sweep also checks the next solve call.
        for (const int iterations : {1, 5, 1}) {
            SCOPED_TRACE(::testing::Message()
                << "inactive contact: friction=" << friction
                << " iterations=" << iterations);
            params.max_global_iters = iterations;
            std::vector<Vec3> current = initial;
            const SolverResult result = global_gauss_seidel_solver_ogc(
                mesh, adjacency, pins, params, current, xhat, velocity,
                "", &initial);
            ASSERT_TRUE(result.converged);
            ASSERT_EQ(result.iterations, iterations);
            // Each sweep can travel only .4 times its current distance, so
            // y decreases by exactly the geometric factor .6 up to rounding.
            const double expected_y = .5 * std::pow(.6, iterations);
            ASSERT_GT(expected_y, params.d_hat);
            EXPECT_NEAR(current[0].y(), expected_y, 1e-14);
            EXPECT_DOUBLE_EQ(current[0].x(), initial[0].x());
            EXPECT_DOUBLE_EQ(current[0].z(), initial[0].z());
            for (std::size_t node = 0; node < current.size(); ++node) {
                ASSERT_TRUE(current[node].allFinite());
                if (node != 0)
                    EXPECT_EQ(std::memcmp(current[node].data(),
                        initial[node].data(), 3 * sizeof(double)), 0);
            }
        }
    }
}

} // namespace
