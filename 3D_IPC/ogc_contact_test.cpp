#include "ogc_contact.h"
#include "friction_energy.h"
#include "mesh_utils.h"
#include "parallel_helper.h"
#include "simulation.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace {

using Points = std::array<Vec3, 4>;

Points obtuse_triangle(const Vec3& point = Vec3(.5,-.01,.1)) {
    return {point, Vec3::Zero(), Vec3(1,0,0), Vec3(-10,.1,0)};
}

NodeTriangleDistanceResult cached_distance(const Points& x) {
    RefMesh mesh;
    mesh.num_positions = x.size();
    mesh.mass.assign(x.size(), 1.0);
    mesh.node_to_rb.assign(x.size(), -1);
    mesh.tris = {1,2,3};
    const std::vector<Vec3> positions(x.begin(), x.end());
    std::vector<AABB> boxes;
    for (const auto& point : positions) boxes.emplace_back(point, point);
    BroadPhase phase;
    phase.initialize(boxes, mesh, 2.0);
    ogc_detail::ContactCache contacts;
    contacts.evaluate(phase, positions, false, true);
    EXPECT_EQ(contacts.nt_distances.size(), 1);
    return contacts.nt_distances.at(0);
}

void expect_same_distance(const NodeTriangleDistanceResult& cached, const NodeTriangleDistanceResult& shared) {
    EXPECT_EQ(cached.region, shared.region);
    EXPECT_EQ(cached.closest_point, shared.closest_point);
    EXPECT_EQ(cached.distance, shared.distance);
    EXPECT_EQ(cached.phi, shared.phi);
    EXPECT_EQ(cached.normal, shared.normal);
    EXPECT_EQ(cached.tilde_x, shared.tilde_x);
    EXPECT_EQ(cached.barycentric_tilde_x, shared.barycentric_tilde_x);
}

} // namespace

TEST(OGCContact, OGCAndGSUseTheSameDistanceBehavior) {
    const auto x = obtuse_triangle();
    const auto shared = ::node_triangle_distance(x[0], x[1], x[2], x[3]);
    EXPECT_EQ(shared.region, NodeTriangleRegion::Edge12);
    EXPECT_LE((shared.closest_point - Vec3(.5,0,0)).norm(), 1e-13);
    EXPECT_NEAR(shared.distance, std::sqrt(.0101), 1e-14);
    expect_same_distance(cached_distance(x), shared);
}

TEST(OGCContact, CachedDistanceMatchesSharedAcrossVertexOrdersAndOrientations) {
    const auto initial = obtuse_triangle();
    const Mat33 rotation = Eigen::AngleAxisd(.73, Vec3(1,2,3).normalized()).toRotationMatrix();
    for (bool transformed : {false, true}) {
        const auto transform = [&](const Vec3& p) -> Vec3 { return transformed ? Vec3(rotation * p + Vec3(2,-3,4)) : p; };
        std::array<int, 3> order{1,2,3};
        do {
            const Points x{transform(initial[0]), transform(initial[order[0]]), transform(initial[order[1]]), transform(initial[order[2]])};
            const auto shared = ::node_triangle_distance(x[0], x[1], x[2], x[3]);
            const auto expected_edge = order[2] == 3 ? NodeTriangleRegion::Edge12
                : order[0] == 3 ? NodeTriangleRegion::Edge23 : NodeTriangleRegion::Edge31;
            EXPECT_EQ(shared.region, expected_edge);
            EXPECT_LE((shared.closest_point - transform(Vec3(.5,0,0))).norm(), 1e-12);
            EXPECT_NEAR(shared.distance, std::sqrt(.0101), 1e-12);
            expect_same_distance(cached_distance(x), shared);
        } while (std::next_permutation(order.begin(), order.end()));
    }
}

TEST(OGCContact, SharedVertexFeatureHasConsistentDerivatives) {
    const auto x = obtuse_triangle(Vec3(1.5,-.001,.1));
    const auto dr = cached_distance(x);
    ASSERT_EQ(dr.region, NodeTriangleRegion::Vertex2);
    EXPECT_TRUE(dr.closest_point.isApprox(x[2]));
    EXPECT_NEAR(dr.distance, std::sqrt(.260001), 1e-14);
    constexpr double step = 1e-6;
    for (int role = 0; role < 4; ++role) {
        const auto [g, H] = node_triangle_barrier_self_gradient_and_hessian(x[0], x[1], x[2], x[3], 2.0, role, 1e-12, &dr);
        for (int k = 0; k < 3; ++k) {
            auto plus = x, minus = x;
            plus[role][k] += step;
            minus[role][k] -= step;
            const auto dp = cached_distance(plus), dm = cached_distance(minus);
            const double difference = (scalar_barrier(dp.distance, 2.0) - scalar_barrier(dm.distance, 2.0)) / (2*step);
            EXPECT_NEAR(g[k], difference, 1e-7);
            const Vec3 gp = node_triangle_barrier_gradient(plus[0], plus[1], plus[2], plus[3], 2.0, role, 1e-12, &dp);
            const Vec3 gm = node_triangle_barrier_gradient(minus[0], minus[1], minus[2], minus[3], 2.0, role, 1e-12, &dm);
            EXPECT_LE((H.col(k) - (gp-gm)/(2*step)).norm(), 1e-6);
        }
    }
}

TEST(OGCContact, FaceAndDegenerateMetadataArePreserved) {
    const std::array<Points, 2> cases{Points{Vec3(.2,.3,.1), Vec3::Zero(), Vec3(1,0,0), Vec3(0,1,0)}, Points{Vec3(.5,1,.2), Vec3::Zero(), Vec3(1,0,0), Vec3(2,0,0)}};
    for (const auto& x : cases) {
        const auto original = ::node_triangle_distance(x[0], x[1], x[2], x[3]);
        expect_same_distance(cached_distance(x), original);
    }
}

namespace {

struct OgcSolverFixture {
    RefMesh mesh;
    SimParams params = SimParams::zeros();
    std::vector<Vec3> initial{Vec3(.5,-.01,.1), Vec3::Zero(), Vec3(1,0,0), Vec3(-10,.1,0)};
    std::vector<Vec3> target = initial;
    std::vector<Vec3> velocity = std::vector<Vec3>(initial.size(), Vec3::Zero());

    OgcSolverFixture() {
        mesh.num_positions = initial.size();
        mesh.mass.assign(initial.size(), 1.0);
        mesh.node_to_rb.assign(initial.size(), -1);
        mesh.tris = {1,2,3};
        params.fixed_iters = true;
        params.max_global_iters = 1;
        params.damping = 1.0;
        params.d_hat = 1.0;
        params.k_barrier = 0.0;
        params.node_box_min = params.node_box_max = 1.0;
    }

    std::vector<Vec3> solve() {
        auto x = initial;
        const auto result = global_gauss_seidel_solver_ogc(mesh, {}, {}, params, x, target, velocity, "", &initial);
        EXPECT_TRUE(result.converged);
        return x;
    }
};

} // namespace

TEST(OGCSolver, ObtuseTriangleUsesSharedDistanceForMotionBound) {
    OgcSolverFixture scene;
    const Vec3 proposal(0,.04,-.4);
    scene.target[0] += proposal;
    const auto x = scene.solve();
    const double bound = .4 * std::sqrt(.0101);
    EXPECT_LE((x[0] - (scene.initial[0] + (bound/proposal.norm())*proposal)).norm(), 1e-13);
    EXPECT_NEAR((x[0]-scene.initial[0]).norm(), bound, 1e-13);
    for (int i = 1; i < 4; ++i) EXPECT_EQ(x[i], scene.initial[i]);
}

TEST(OGCSolver, SharedDistanceControlsBarrierAndFrictionActivation) {
    for (double friction : {0.0,.3}) {
        OgcSolverFixture scene;
        scene.params.d_hat = .2;
        scene.params.k_barrier = 100;
        scene.params.friction_coefficient = friction;
        // The closest point is on edge12, so sqrt(.0101) is active at d_hat=.2.
        const auto active = scene.solve();
        const Vec3 nearest_edge_separation(0,-.01,.1);
        EXPECT_GT(nearest_edge_separation.dot(active[0]-scene.initial[0]), 0.0);
        scene.params.d_hat = .05;
        const auto inactive = scene.solve();
        for (std::size_t i = 0; i < inactive.size(); ++i) EXPECT_EQ(inactive[i], scene.initial[i]);
    }
}

TEST(OGCSolver, BarrierPairsDoNotAlterTheElasticColoring) {
    const std::vector<Vec3> rest{Vec3(0, 0, 0), Vec3(1, 0, 0),
        Vec3(0, 1, 0), Vec3(1, 1, 0), Vec3(0, 2, 0)};
    RefMesh mesh;
    mesh.tris = {0, 1, 2, 1, 3, 2, 2, 3, 4};
    mesh.initialize(rest);
    mesh.mass.assign(rest.size(), 1);
    mesh.node_to_rb.assign(rest.size(), -1);
    const auto adjacency = build_incident_triangle_map(mesh.tris);
    auto initial = rest;
    initial[0] += Vec3(-.05, .02, .01);
    initial[4] += Vec3(.08, -.03, .02);
    const std::vector<Vec3> velocity(rest.size(), Vec3::Zero());
    auto params = SimParams::zeros();
    params.fps = 20;
    params.substeps = 1;
    params.fixed_iters = true;
    params.max_global_iters = 2;
    params.damping = .5;
    params.mu = 30;
    params.lambda = 20;
    params.kB = .003;
    params.d_hat = 2;
    params.k_barrier = .1;
    params.node_box_min = params.node_box_max = 1;
    (void)params.dt2();
    const double padding = std::max(params.ogc_box_pad, params.d_hat);
    const auto phase_for = [&](const std::vector<Vec3>& positions) {
        std::vector<AABB> boxes;
        for (const auto& position : positions)
            boxes.emplace_back(position - Vec3::Constant(1 + padding),
                position + Vec3::Constant(1 + padding));
        BroadPhase phase;
        phase.initialize(boxes, mesh, padding);
        return phase;
    };
    const auto reference = [&](bool include_initial_contacts,
                               bool freeze_contact_per_iteration = false) {
        auto positions = initial;
        auto phase = phase_for(positions);
        auto graph = build_elastic_adj(mesh, adjacency, static_cast<int>(rest.size()));
        if (include_initial_contacts) {
            const auto clique = [&](const std::array<int, 4>& nodes) {
                for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) {
                    graph[nodes[a]].push_back(nodes[b]);
                    graph[nodes[b]].push_back(nodes[a]);
                }
            };
            for (const auto& pair : phase.cache().nt_pairs)
                clique({pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]});
            for (const auto& pair : phase.cache().ss_pairs)
                clique({pair.v[0], pair.v[1], pair.v[2], pair.v[3]});
        }
        for (auto& row : graph) {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        std::vector<std::vector<int>> colors;
        greedy_color_conflict_graph(graph, colors);
        EXPECT_EQ(colors.size(), include_initial_contacts ? 5u : 4u);
        // Numerical pairs refresh each iteration; this initial graph does not.
        for (int iteration = 0; iteration < params.max_global_iters; ++iteration) {
            if (iteration > 0) phase = phase_for(positions);
            const auto iteration_snapshot = positions;
            ogc_detail::ContactCache contacts;
            contacts.evaluate(phase, iteration_snapshot, false, params.exact_computation_fallback);
            std::vector<double> bounds(positions.size());
            for (std::size_t node = 0; node < positions.size(); ++node)
                bounds[node] = std::min(1.0, contacts.isotropic_motion_bound(phase, node, 1));
            for (const auto& color : colors) {
                const auto snapshot = freeze_contact_per_iteration ? iteration_snapshot : positions;
                if (!freeze_contact_per_iteration)
                    contacts.evaluate(phase, snapshot, false, params.exact_computation_fallback);
                std::vector<Vec3> steps(positions.size(), Vec3::Zero());
                for (int node : color) {
                    const Vec3 delta = -params.damping * ogc_detail::frozen_contact_vertex_delta(
                        node, mesh, adjacency, {}, params, rest, positions, snapshot,
                        phase, nullptr, nullptr, nullptr, &initial, contacts);
                    const double factor = delta.norm() > 0 ? std::min(1.0, bounds[node] / delta.norm()) : 1;
                    steps[node] = factor * delta;
                }
                for (int node : color) positions[node] += steps[node];
            }
        }
        return positions;
    };
    const auto expected = reference(false), with_contact_colors = reference(true);
    const auto iteration_contact = reference(false, true);
    double coloring_effect = 0;
    for (std::size_t node = 0; node < initial.size(); ++node)
        coloring_effect = std::max(coloring_effect, (expected[node] - with_contact_colors[node]).norm());
    ASSERT_GT(coloring_effect, 1e-8);
    double refresh_effect = 0;
    for (std::size_t node = 0; node < initial.size(); ++node)
        refresh_effect = std::max(refresh_effect, (expected[node] - iteration_contact[node]).norm());
    ASSERT_GT(refresh_effect, 1e-9);
    auto actual = initial;
    ASSERT_TRUE(global_gauss_seidel_solver_ogc(mesh, adjacency, {}, params,
        actual, rest, velocity, "", &initial).converged);
    for (std::size_t node = 0; node < actual.size(); ++node)
        EXPECT_LE((actual[node] - expected[node]).norm(), 1e-13) << node;
}

TEST(OGCSolver, FarFalsePositiveCannotMoveBeyondTheDetectionEnvelope) {
    OgcSolverFixture scene;
    scene.initial = {Vec3(90, 90, .001), Vec3::Zero(),
        Vec3(100, 0, 0), Vec3(0, 100, 0)};
    scene.target = scene.initial;
    scene.target[0] += Vec3(0, 2, 0);
    scene.params.node_box_min = scene.params.node_box_max = .01;
    scene.params.d_hat = .005;
    std::vector<AABB> boxes;
    for (const auto& position : scene.initial)
        boxes.emplace_back(position - Vec3::Constant(.015),
            position + Vec3::Constant(.015));
    BroadPhase phase;
    phase.initialize(boxes, scene.mesh, .005);
    ASSERT_EQ(phase.cache().nt_pairs.size(), 1u);
    ogc_detail::ContactCache contacts;
    contacts.evaluate(phase, scene.initial, false, true);
    ASSERT_GT(contacts.isotropic_motion_bound(phase, 0, .01), 1.0);

    const auto actual = scene.solve();
    EXPECT_LE((actual[0] - (scene.initial[0] + Vec3(0, .01, 0))).norm(), 1e-13);
    for (int node = 1; node < 4; ++node) EXPECT_EQ(actual[node], scene.initial[node]);
}

TEST(OGCSolver, ColorDistanceRefreshDoesNotResetTheIterationMotionBudget) {
    const std::vector<Vec3> initial{Vec3(0, 0, .1), Vec3(-1, -1, 0),
        Vec3(2, -1, 0), Vec3(-1, 2, 0)};
    RefMesh mesh;
    mesh.tris = {1, 2, 3};
    mesh.initialize(initial);
    mesh.mass.assign(initial.size(), 1);
    mesh.node_to_rb.assign(initial.size(), -1);
    const auto adjacency = build_incident_triangle_map(mesh.tris);
    auto target = initial;
    target[0].z() -= .06;
    target[2].z() += 1;
    auto params = SimParams::zeros();
    params.fixed_iters = true;
    params.max_global_iters = 1;
    params.damping = 1;
    params.d_hat = .2;
    // zeros() retains k_barrier=1. Disable its forces so only the inertial
    // targets and iteration-start distance budget determine these updates.
    params.k_barrier = 0;
    params.node_box_min = params.node_box_max = 1;
    auto actual = initial;
    ASSERT_TRUE(global_gauss_seidel_solver_ogc(mesh, adjacency, {}, params,
        actual, target, std::vector<Vec3>(initial.size(), Vec3::Zero()),
        "", &initial).converged);
    // The point moves in color zero, reducing the contact gap from .1 to .06.
    // The later triangle vertex still owns its original .4*.1 displacement
    // budget; a force-distance refresh must not replace it with .4*.06.
    EXPECT_NEAR(actual[0].z(), .06, 1e-13);
    EXPECT_NEAR(actual[2].z(), .04, 1e-13);
    EXPECT_EQ(actual[1], initial[1]);
    EXPECT_EQ(actual[3], initial[3]);
}

TEST(OGCSolver, AliasedFrictionHistoryMatchesAnImmutablePhysicalSnapshot) {
    RefMesh mesh;
    const std::vector<Vec3> initial{Vec3(0, 0, .1), Vec3(-1, -1, 0),
        Vec3(2, -1, 0), Vec3(-1, 2, 0)};
    mesh.tris = {1, 2, 3};
    mesh.initialize(initial);
    mesh.mass.assign(initial.size(), 1);
    mesh.node_to_rb.assign(initial.size(), -1);
    const auto adjacency = build_incident_triangle_map(mesh.tris);
    auto target = initial;
    target[0].x() += .05;
    const std::vector<Vec3> velocity(initial.size(), Vec3::Zero());
    auto params = SimParams::zeros();
    params.fps = 20;
    params.substeps = 1;
    params.fixed_iters = true;
    params.max_global_iters = 3;
    params.damping = .5;
    params.d_hat = .2;
    params.k_barrier = 100;
    params.friction_coefficient = .4;
    params.node_box_min = params.node_box_max = 1;
    auto immutable_history = initial, alias_history = initial;
    ASSERT_TRUE(global_gauss_seidel_solver_ogc(mesh, adjacency, {}, params,
        immutable_history, target, velocity, "", &initial).converged);
    ASSERT_GT((immutable_history[0] - initial[0]).norm(), 1e-6);
    ASSERT_TRUE(global_gauss_seidel_solver_ogc(mesh, adjacency, {}, params,
        alias_history, target, velocity, "", &alias_history).converged);
    for (std::size_t node = 0; node < initial.size(); ++node)
        EXPECT_LE((alias_history[node] - immutable_history[node]).norm(), 1e-13) << node;
}

TEST(OGCSolver, PublicDriverStartsAtPreviousPositionsAndRetainsPhysicalInertialVelocity) {
    struct RestoreOpenMP {
        int threads=omp_get_max_threads();
        ~RestoreOpenMP(){omp_set_num_threads(threads);}
    } restore;
    omp_set_num_threads(1);
    const std::vector<Vec3> rest{Vec3::Zero(),Vec3::UnitX(),Vec3::UnitY()};
    const Vec3 physical_velocity(1.2,-.7,.4);
    RefMesh mesh;
    mesh.tris={0,1,2};mesh.initialize(rest);
    mesh.mass.assign(rest.size(),1);mesh.node_to_rb.assign(rest.size(),-1);
    const auto adjacency=build_incident_triangle_map(mesh.tris);
    std::vector<Pin> pins;
    // Exercise each old predictor on its own, then a valid combination so its
    // precedence cannot hide a predictor which still overrides the OGC start.
    for(bool named_solver:{false,true})for(int predictor=0;predictor<6;++predictor) {
        SCOPED_TRACE(::testing::Message()<<"named_solver="<<named_solver<<" predictor="<<predictor);
        auto params=SimParams::zeros();
        params.fps=20;params.substeps=2;
        params.fixed_iters=true;params.max_global_iters=1;params.damping=.25;
        params.k_barrier=0;params.node_box_min=params.node_box_max=1;
        params.use_ogc=!named_solver;params.use_ogc_solver=named_solver;
        params.use_ccd_guess=predictor==0||predictor==5;
        params.use_verlet_guess=predictor==1||predictor==5;
        params.use_translation_guess=predictor==2||predictor==5;
        params.use_colored_ccd_guess=predictor==3||predictor==5;
        params.use_colored_ccd_guess_level_based=predictor==4;
        params.colored_ccd_guess_iters=1;
        DeformedState actual;
        actual.deformed_positions=rest;
        actual.velocities.assign(rest.size(),physical_velocity);
        auto expected=actual;
        std::vector<std::vector<Vec3>> substep_positions;
        BroadPhase manual_phase;
        const double dt=params.dt();
        for(int substep=0;substep<params.substeps;++substep) {
            // The zero-velocity guess means zero displacement from x_n; the
            // physical velocity must still appear in the separate xhat target.
            auto positions=expected.deformed_positions;
            std::vector<Vec3> target;
            build_xhat(target,expected.deformed_positions,expected.velocities,dt);
            ASSERT_GT((target[0]-positions[0]).norm(),0);
            const auto result=named_solver
                ? global_gauss_seidel_solver_ogc(mesh,adjacency,pins,params,
                    positions,target,expected.velocities,"",&expected.deformed_positions)
                : global_gauss_seidel_solver_basic(mesh,adjacency,pins,params,
                    positions,target,expected.velocities,manual_phase,"",&expected.deformed_positions);
            ASSERT_TRUE(result.converged);
            update_velocity(expected.velocities,positions,expected.deformed_positions,dt);
            expected.deformed_positions=positions;
            substep_positions.push_back(positions);
        }
        // With only inertia, each one-sweep solve takes one damped Newton step.
        // This oracle distinguishes preserving the real velocity from zeroing it.
        const Vec3 displacement=dt*physical_velocity*params.damping*(1+params.damping);
        for(std::size_t node=0;node<rest.size();++node) {
            EXPECT_LE((expected.deformed_positions[node]-rest[node]-displacement).norm(),1e-13);
            EXPECT_LE((expected.velocities[node]-params.damping*params.damping*physical_velocity).norm(),1e-12);
        }
        int callbacks=0;
        const SubstepCallback callback=[&](int substep,const std::vector<Vec3>& positions) {
            ASSERT_EQ(substep,callbacks++);
            ASSERT_EQ(positions.size(),rest.size());
            for(std::size_t node=0;node<positions.size();++node)
                EXPECT_LE((positions[node]-substep_positions[substep][node]).norm(),1e-13);
        };
        BroadPhase phase;
        const auto result=advance_one_frame(actual,mesh,adjacency,pins,params,
            phase,1,nullptr,callback);
        EXPECT_TRUE(result.converged);EXPECT_EQ(result.iterations,2);EXPECT_EQ(callbacks,2);
        for(std::size_t node=0;node<rest.size();++node) {
            EXPECT_LE((actual.deformed_positions[node]-expected.deformed_positions[node]).norm(),1e-13);
            EXPECT_LE((actual.velocities[node]-expected.velocities[node]).norm(),1e-12);
        }
    }
}
