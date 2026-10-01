#include "make_shape.h"
#include "general_simd_assembly.h"
#include "general_simd_contact.h"
#include "general_simd_scheduling.h"
#include "mesh_utils.h"
#include "simulation.h"
#include "solid_ipc.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
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

struct GeneralScene {
    RefMesh mesh;
    DeformedState state;
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;
    SimParams params = SimParams::zeros();
    BroadPhase broad_phase;
};

void build_scene(GeneralScene& scene, bool cloth = true, bool solid = true,
    bool rigid = true, RigidBodyUpdateMode mode =
        RigidBodyUpdateMode::TranslationAndOrientation) {
    auto& mesh = scene.mesh;
    auto& state = scene.state;
    auto& params = scene.params;
    params.fps = 30.0;
    params.substeps = 1;
    params.mu = 2.0;
    params.lambda = 3.0;
    params.kB = 1e-4;
    params.kpin = 1000.0;
    params.solid_mu = 2.0;
    params.solid_lambda = 3.0;
    params.gravity = Vec3(0.0, -0.1, 0.0);
    params.node_box_min = 0.05;
    params.node_box_max = 0.05;
    params.theta_box_min = 0.02;
    params.theta_box_max = 0.02;
    params.node_box_update_count = 2;
    params.max_global_iters = 4;
    params.fixed_iters = true;
    params.damping = 0.1;
    params.use_parallel = true;
    params.use_ccd = true;
    params.use_ticcd = false;
    params.use_ccd_guess = false;
    params.d_hat = 0.03;
    params.k_barrier = 1.0;
    params.friction_velocity_epsilon = 0.01;

    if (cloth) {
        std::vector<Vec2> material;
        build_square_mesh(mesh, state, material, 6, 6, 1.0, 1.0,
            Vec3(-0.5, 0.0, -0.5));
        mesh.build_lumped_mass(900.0, 0.001);
        state.velocities.assign(state.deformed_positions.size(),
            Vec3(-0.005, -0.003, 0.002));
        for (Vec3& x : state.deformed_positions) {
            x.x() *= 1.005;
            x.y() += 0.0005 * std::sin(4.0 * x.x())
                * std::cos(3.0 * x.z());
        }
        append_pin(scene.pins, 0, state.deformed_positions);
        append_pin(scene.pins, 6, state.deformed_positions);
    }
    if (solid) {
        // Four tets share an interior node, exercising several incident
        // elastic entries while keeping interior nodes out of contact/SDF.
        std::vector<Vec3> positions = {
            Vec3(-0.08, 0.006, -0.08), Vec3(0.08, 0.006, -0.08),
            Vec3(-0.08, 0.006, 0.08), Vec3(-0.08, 0.012, -0.08)};
        positions.push_back(0.25 * (positions[0] + positions[1]
            + positions[2] + positions[3]));
        const int first = static_cast<int>(state.deformed_positions.size());
        create_solid(positions,
            {4, 2, 1, 3, 0, 4, 1, 3, 0, 2, 4, 3, 0, 2, 1, 4},
            900.0, mesh, state);
        for (int node = first; node < static_cast<int>(state.velocities.size()); ++node)
            state.velocities[node] = Vec3(0.003, -0.002, -0.004);
        state.deformed_positions[first + 3] += Vec3(0.0002, 0.0001, -0.0001);
        state.deformed_positions[first + 4] += Vec3(-0.0001, 0.00005, 0.0001);
    }
    if (rigid) {
        const int rb = create_rigid_body(
            {Vec3(-0.4, 0.018, -0.4), Vec3(0.4, 0.018, -0.4),
             Vec3(0.0, 0.018, 0.4)},
            Vec3(0.01, -0.01, 0.02), Vec4(1.0, 0.0, 0.0, 0.0),
            Vec3(0.02, -0.01, 0.01), 1.0, mesh, state, mode);
        mesh.tris.insert(mesh.tris.end(), mesh.rb_nodes[rb].begin(),
            mesh.rb_nodes[rb].end());
        if (!cloth && !solid) {
            const int other = create_rigid_body(
                {Vec3(-0.2, 0.036, -0.2), Vec3(0.2, 0.036, -0.2),
                 Vec3(0.0, 0.036, 0.2)},
                Vec3(-0.01, -0.005, 0.003), Vec4(1.0, 0.0, 0.0, 0.0),
                Vec3(-0.01, 0.02, -0.01), 0.8, mesh, state, mode);
            mesh.tris.insert(mesh.tris.end(), mesh.rb_nodes[other].begin(),
                mesh.rb_nodes[other].end());
        }
    }
    mesh.node_to_rb.resize(state.deformed_positions.size(), -1);
    mesh.build_deformable_nodes();
    scene.adjacency = build_incident_triangle_map(mesh.tris);
}

// Call the public solver explicitly, then use the same state commit as the
// frame driver so comparisons include velocities and all rigid coordinates.
SolverResult solve_and_commit(GeneralScene& scene, bool simd) {
    auto& state = scene.state;
    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, scene.params.dt());
    for (std::size_t node = 0; node < xhat.size(); ++node)
        if (scene.mesh.node_to_rb[node] >= 0)
            xhat[node] = state.deformed_positions[node];
    auto positions = state.deformed_positions;
    auto centers = state.x_coms;
    auto orientations = state.orientations;
    std::vector<Vec3> omega(state.omega.size(), Vec3::Zero());
    const auto solve = simd ? global_gauss_seidel_solver_general_experimental_v2
                           : global_gauss_seidel_solver_basic_general;
    const SolverResult result = solve(scene.mesh, state, scene.adjacency,
        scene.pins, scene.params, positions, xhat, centers, orientations,
        omega, scene.broad_phase, "");
    // Nonconverged solves still expose their final iterates and residuals;
    // committing here lets the residual test compare those iterates as well.
    for (std::size_t node = 0; node < positions.size(); ++node) {
        if (scene.mesh.node_to_rb[node] >= 0) continue;
        state.velocities[node] = (positions[node] - state.deformed_positions[node])
            / scene.params.dt();
        state.deformed_positions[node] = positions[node];
    }
    update_velocity(state.v_coms, centers, state.x_coms, scene.params.dt());
    state.x_coms = centers;
    state.orientations = orientations;
    state.omega = omega;
    sync_rigid_body_particles(scene.mesh, state);
    return result;
}

SolverResult solve_basic_v2_and_commit(GeneralScene& scene) {
    auto& state = scene.state;
    std::vector<Vec3> predicted;
    build_xhat(predicted, state.deformed_positions, state.velocities, scene.params.dt());
    auto positions = state.deformed_positions;
    const auto result = global_gauss_seidel_solver_basic_experimental_v2(scene.mesh,
        scene.adjacency, scene.pins, scene.params, positions, predicted,
        state.velocities, scene.broad_phase, "", &state.deformed_positions);
    update_velocity(state.velocities, positions, state.deformed_positions, scene.params.dt());
    state.deformed_positions = positions;
    return result;
}

void enable_v2_flags(GeneralScene& scene) {
    scene.params.use_basic_experimental = true;
    scene.params.use_basic_experimental_v2 = true;
    scene.params.use_simd = true;
}

void build_two_cloth_layers(GeneralScene& scene) {
    build_scene(scene, true, false, false);
    std::vector<Vec2> material;
    for (int j = 0; j <= 6; ++j)
        for (int i = 0; i <= 6; ++i)
            material.emplace_back(i / 6.0, j / 6.0);
    build_square_mesh(scene.mesh, scene.state, material, 6, 6, 1.0, 1.0,
        Vec3(-0.5, 0.018, -0.5));
    scene.state.velocities.resize(scene.state.deformed_positions.size(),
        Vec3(0.01, -0.002, -0.003));
    scene.mesh.build_lumped_mass(900.0, 0.001);
    scene.mesh.node_to_rb.assign(scene.state.deformed_positions.size(), -1);
    scene.mesh.build_deformable_nodes();
    scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
    enable_v2_flags(scene);
}

struct SparseContacts {
    std::vector<Vec3> positions{
        Vec3(0, 0, 0.01), Vec3(-1, -1, 0), Vec3(1, -1, 0), Vec3(0, 1, 0),
        Vec3(10, -1, 0), Vec3(12, -1, 0), Vec3(11, 1, 0),
        Vec3(-1, -1, -0.8), Vec3(1, -1, 1.2), Vec3(0, 1, 0.2),
        Vec3(1, 0, 0.01), Vec3(0.4, -1, 0), Vec3(0.4, 1, 0),
        Vec3(10, 10, 10), Vec3(11, 10, 10)};
    std::vector<Vec3> previous = positions;
    std::vector<unsigned char> solid = std::vector<unsigned char>(positions.size(), 0);
    std::vector<unsigned char> surface = std::vector<unsigned char>(positions.size(), 1);
    BroadPhase::Cache cache;
    SimParams params = SimParams::zeros();

    SparseContacts() {
        previous[0].x() -= 0.003;
        params.d_hat = 0.1;
        params.k_barrier = 2.0;
        cache.vertex_nt.resize(positions.size());
        cache.vertex_ss.resize(positions.size());
        // Include AABB rejections, plane-only rejections whose bounding boxes
        // overlap, and active contacts. Sparse survivors cross tile boundaries.
        for (int i = 0; i < 96; ++i) {
            const int first = i % 3 == 0 ? 4 : (i % 3 == 1 ? 7 : 1);
            cache.nt_pairs.push_back({0, {first, first + 1, first + 2}});
            cache.vertex_nt[0].push_back({static_cast<std::size_t>(i), 0});
        }
        for (int i = 0; i < 65; ++i) {
            const int first = i % 3 == 0 ? 11 : 13;
            cache.ss_pairs.push_back({{0, 10, first, first + 1}});
            cache.vertex_ss[0].push_back({static_cast<std::size_t>(i), 0});
        }
    }
};

template <class Vector>
void expect_vectors_near(const std::vector<Vector>& actual,
    const std::vector<Vector>& expected, double tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(actual[i].allFinite()) << "entry=" << i;
        ASSERT_TRUE(expected[i].allFinite()) << "entry=" << i;
        EXPECT_LE((actual[i] - expected[i]).cwiseAbs().maxCoeff(),
            tolerance * (1.0 + expected[i].cwiseAbs().maxCoeff())) << "entry=" << i;
    }
}

void expect_states_near(const DeformedState& actual, const DeformedState& expected) {
    expect_vectors_near(actual.deformed_positions, expected.deformed_positions, 2e-9);
    expect_vectors_near(actual.velocities, expected.velocities, 2e-7);
    expect_vectors_near(actual.x_coms, expected.x_coms, 2e-9);
    expect_vectors_near(actual.orientations, expected.orientations, 2e-9);
    expect_vectors_near(actual.v_coms, expected.v_coms, 2e-7);
    expect_vectors_near(actual.omega, expected.omega, 2e-7);
}

template <class Vector>
void expect_vectors_bitwise_equal(const std::vector<Vector>& actual,
    const std::vector<Vector>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(0, std::memcmp(actual[i].data(), expected[i].data(),
            actual[i].size() * sizeof(double))) << "entry=" << i;
}

void expect_states_bitwise_equal(const DeformedState& actual,
    const DeformedState& expected) {
    expect_vectors_bitwise_equal(actual.deformed_positions, expected.deformed_positions);
    expect_vectors_bitwise_equal(actual.velocities, expected.velocities);
    expect_vectors_bitwise_equal(actual.x_coms, expected.x_coms);
    expect_vectors_bitwise_equal(actual.orientations, expected.orientations);
    expect_vectors_bitwise_equal(actual.v_coms, expected.v_coms);
    expect_vectors_bitwise_equal(actual.omega, expected.omega);
}

void expect_active_coupling(const GeneralScene& scene) {
    std::vector<int> kind(scene.state.deformed_positions.size(), 0);
    for (int node : scene.mesh.tet_nodes) kind[node] = 1;
    for (const auto& nodes : scene.mesh.rb_nodes)
        for (int node : nodes) kind[node] = 2;
    bool active[3][3]{};
    const auto& x = scene.state.deformed_positions;
    for (const auto& pair : scene.broad_phase.cache().nt_pairs) {
        const int a = kind[pair.node], b = kind[pair.tri_v[0]];
        if (a == b) continue;
        const auto distance = node_triangle_distance(x[pair.node],
            x[pair.tri_v[0]], x[pair.tri_v[1]], x[pair.tri_v[2]]);
        if (distance.distance < scene.params.d_hat)
            active[std::min(a, b)][std::max(a, b)] = true;
    }
    EXPECT_TRUE(active[0][1]) << "cloth-solid contact must be active";
    EXPECT_TRUE(active[0][2]) << "cloth-rigid contact must be active";
    EXPECT_TRUE(active[1][2]) << "solid-rigid contact must be active";
}

} // namespace

TEST(GeneralSIMDSolver, CoupledClothSolidRigidMatchesScalarWithAndWithoutFriction) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (bool parallel : {false, true}) {
        for (int contact_mode : {0, 1, 2}) {
            SCOPED_TRACE(::testing::Message() << "parallel=" << parallel
                << " contact_mode=" << contact_mode);
            std::array<GeneralScene, 2> scenes;
            for (auto& scene : scenes) {
                build_scene(scene);
                scene.params.use_parallel = parallel;
                scene.params.friction_coefficient = contact_mode == 2 ? 0.2 : 0.0;
                if (contact_mode == 0) {
                    scene.params.d_hat = 0.0;
                    scene.params.k_barrier = 0.0;
                    scene.params.use_ccd = false;
                }
            }
            for (int frame = 0; frame < 2; ++frame) {
                for (int method = 0; method < 2; ++method) {
                    const auto result = solve_and_commit(scenes[method], method == 1);
                    ASSERT_TRUE(result.converged);
                    EXPECT_EQ(result.iterations, scenes[method].params.max_global_iters);
                }
                expect_states_near(scenes[1].state, scenes[0].state);
            }
            if (contact_mode != 0) expect_active_coupling(scenes[1]);
        }
    }
}

TEST(GeneralSIMDSolver, ScalarAndV2PruneSeparatedMixedIncidenceAndRestoreForTiccd) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (bool simd : {false,true}) {
        GeneralScene scene;
        build_scene(scene);
        scene.params.node_box_min = scene.params.node_box_max = .002;
        scene.params.theta_box_min = scene.params.theta_box_max = .002;
        // Oblique faces produce overlapping AABBs for separated primitives.
        const Vec4 rotation = quaternion_from_angular_velocity(
            Vec4(1,0,0,0), Vec3(0,.61,0), 1.0);
        auto& state = scene.state;
        for (auto& x : state.deformed_positions) x = quaternion_rotate(rotation,x);
        for (auto& v : state.velocities) v = quaternion_rotate(rotation,v);
        for (auto& x : state.x_coms) x = quaternion_rotate(rotation,x);
        for (auto& v : state.v_coms) v = quaternion_rotate(rotation,v);
        for (auto& w : state.omega) w = quaternion_rotate(rotation,w);
        for (auto& q : state.orientations) q = quaternion_multiply(rotation,q);
        for (auto& pin : scene.pins) pin.target_position = quaternion_rotate(rotation,pin.target_position);
        for (bool ticcd : {false,true,false}) {
            SCOPED_TRACE(::testing::Message() << "simd=" << simd << " ticcd=" << ticcd);
            scene.params.use_ticcd = ticcd;
            ASSERT_TRUE(solve_and_commit(scene,simd).converged);
            const auto& actual = scene.broad_phase.cache();
            BroadPhase unpruned;
            unpruned.initialize_surface_nodes(actual.node_boxes,scene.mesh,scene.params.d_hat,
                BroadPhase::InitializationMode::GeneralSolver);
            const auto& full = unpruned.cache();
            ASSERT_EQ(actual.nt_pairs.size(),full.nt_pairs.size());
            ASSERT_EQ(actual.ss_pairs.size(),full.ss_pairs.size());
            std::size_t removed = 0;
            for (int node : scene.mesh.deformable_nodes) {
                removed += full.vertex_nt[node].size()-actual.vertex_nt[node].size();
                removed += full.vertex_ss[node].size()-actual.vertex_ss[node].size();
                for (const Vec3& direction : {Vec3(.004,0,0),Vec3(0,-.004,0),Vec3(0,0,.004)}) {
                    auto expected = state.deformed_positions, result = expected;
                    const Vec3 target = expected[node]+direction;
                    const double a = per_vertex_safe_step(unpruned,expected,node,target,.9,true,ticcd);
                    const double b = per_vertex_safe_step(scene.broad_phase,result,node,target,.9,true,ticcd);
                    EXPECT_DOUBLE_EQ(a,b);
                    EXPECT_TRUE((expected[node].array()==result[node].array()).all());
                }
            }
            if (ticcd) EXPECT_EQ(removed,0u);
            else EXPECT_GT(removed,0u);
            expect_active_coupling(scene);
        }
    }
}

TEST(GeneralSIMDSolver, ColoredContactHelpersPreserveEveryStateAcrossTeamSizes) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (double friction : {0.0, 0.2}) {
        std::array<GeneralScene, 3> scenes;
        for (int run = 0; run < 3; ++run) {
            SCOPED_TRACE(::testing::Message() << "friction=" << friction << " run=" << run);
            auto& scene = scenes[run];
            build_scene(scene);
            scene.params.friction_coefficient = friction;
            // One worker disables cooperative contacts without changing color
            // order; larger teams assign helpers to the contact-heavy body.
            omp_set_num_threads(1 << run);
            for (int frame = 0; frame < 2; ++frame)
                ASSERT_TRUE(solve_and_commit(scene, true).converged);
            EXPECT_GT(scene.broad_phase.cache().nt_pairs.size(), 32u);
            expect_active_coupling(scene);
            if (run != 0) expect_states_bitwise_equal(scene.state, scenes[0].state);
        }
    }
}

TEST(GeneralSIMDSolver, RigidUpdateModesKeepDisabledCoordinatesFixed) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (const auto mode : {RigidBodyUpdateMode::TranslationAndOrientation,
            RigidBodyUpdateMode::TranslationOnly, RigidBodyUpdateMode::OrientationOnly,
            RigidBodyUpdateMode::None}) {
        SCOPED_TRACE(static_cast<int>(mode));
        std::array<GeneralScene, 2> scenes;
        for (auto& scene : scenes) {
            build_scene(scene, true, true, true, mode);
            scene.params.friction_coefficient = 0.2;
        }
        const auto initial = scenes[1].state;
        // Reuse the workspace across calls as well as the four sweeps inside
        // each solve. Fixed-body placement and optional quaternion curvature
        // must remain correct after the first invocation has warmed caches.
        for (int frame = 0; frame < 2; ++frame) {
            SCOPED_TRACE(frame);
            ASSERT_TRUE(solve_and_commit(scenes[0], false).converged);
            ASSERT_TRUE(solve_and_commit(scenes[1], true).converged);
            expect_states_near(scenes[1].state, scenes[0].state);
            if (!updates_rigid_translation(mode)) {
                expect_vectors_bitwise_equal(scenes[1].state.x_coms, initial.x_coms);
                EXPECT_TRUE(scenes[1].state.v_coms[0].isZero(0.0));
            }
            if (!updates_rigid_orientation(mode)) {
                expect_vectors_bitwise_equal(scenes[1].state.orientations, initial.orientations);
                EXPECT_TRUE(scenes[1].state.omega[0].isZero(0.0));
            }
        }
    }
}

TEST(GeneralSIMDSolver, FixedProxyPlacementTracksPrescribedPoseAcrossRepeatedCalls) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (bool mixed : {false, true}) {
        for (int nodes : {3, 131}) {
            for (int threads : {1, 4}) {
                SCOPED_TRACE(::testing::Message() << "mixed=" << mixed
                    << " nodes=" << nodes << " threads=" << threads);
                omp_set_num_threads(threads);
                GeneralScene scene;
                auto& mesh = scene.mesh;
                auto& state = scene.state;
                auto& params = scene.params;
                if (mixed) {
                    state.deformed_positions = {
                        Vec3(-1, -1, 0), Vec3(3, -1, 0), Vec3(-1, 3, 0)};
                    state.velocities.assign(3, Vec3::Zero());
                    mesh.tris = {0, 1, 2};
                    mesh.initialize({Vec2(-1, -1), Vec2(3, -1), Vec2(-1, 3)},
                        state.deformed_positions);
                    mesh.mass.assign(3, 10.0);
                    mesh.node_to_rb.assign(3, -1);
                }
                std::vector<Vec3> points{Vec3(0, 0, 0.2)};
                for (int local = 1; local < nodes; ++local)
                    points.emplace_back(5 + 0.1 * local,
                        3 + 0.02 * (local % 7), 1 + 0.003 * local);
                const int rb = create_rigid_body(points, Vec3::Zero(),
                    Vec4(0.8, -0.2, 0.3, 0.4), Vec3::Zero(), nodes,
                    mesh, state, RigidBodyUpdateMode::None);
                mesh.build_deformable_nodes();
                scene.adjacency = build_incident_triangle_map(mesh.tris);
                params.fps = 10.0;
                params.substeps = 1;
                params.max_global_iters = 8;
                params.fixed_iters = true;
                params.damping = 0.25;
                params.d_hat = mixed ? 0.5 : 0.0;
                params.k_barrier = mixed ? 100.0 : 0.0;
                params.node_box_min = params.node_box_max = 1.0;
                params.theta_box_min = params.theta_box_max = 3.141592653589793;
                params.node_box_update_count = 3;
                params.use_parallel = true;

                for (int call = 0; call < 3; ++call) {
                    SCOPED_TRACE(call);
                    // Reuse the exact mesh/cache allocations with a changed
                    // externally prescribed fixed pose. Stale particle and
                    // caller candidate coordinates must not survive the call.
                    if (call != 0) {
                        state.x_coms[rb] += Vec3(0.013, -0.009, 0.011);
                        state.orientations[rb] = quaternion_normalize(
                            Vec4(0.8, -0.2 + 0.01 * call, 0.3, 0.4));
                    }
                    auto positions = state.deformed_positions;
                    for (int node : mesh.rb_nodes[rb])
                        positions[node] = Vec3(-7, 8, -9);
                    const auto predicted = state.deformed_positions;
                    std::vector<Vec3> centers(1, Vec3(9, 8, 7));
                    std::vector<Vec3> omega(1, Vec3(3, 2, 1));
                    std::vector<Vec4> orientations(1, Vec4(1, 0, 0, 0));
                    const auto result = global_gauss_seidel_solver_general_experimental_v2(
                        mesh, state, scene.adjacency, {}, params, positions,
                        predicted, centers, orientations, omega, scene.broad_phase);
                    ASSERT_TRUE(result.converged);
                    EXPECT_EQ(result.iterations, params.max_global_iters);
                    expect_vectors_bitwise_equal(centers, state.x_coms);
                    expect_vectors_bitwise_equal(orientations, state.orientations);
                    EXPECT_TRUE(omega[rb].isZero(0.0));
                    for (int local = 0; local < nodes; ++local) {
                        const Vec3 expected = world_space_position(
                            mesh.ref_positions[rb][local], state.x_coms[rb],
                            state.orientations[rb]);
                        EXPECT_EQ(0, std::memcmp(
                            positions[mesh.rb_nodes[rb][local]].data(),
                            expected.data(), 3 * sizeof(double))) << "local=" << local;
                    }
                    if (mixed && call == 0)
                        EXPECT_LT((positions[0].z() + positions[1].z()
                            + positions[2].z()) / 3.0, -1e-5);
                }
            }
        }
    }
}

TEST(GeneralSIMDSolver, PureClothSolidAndRigidScenesRetainTheirScalarResults) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (int type = 0; type < 3; ++type) {
        for (bool parallel : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << " parallel=" << parallel);
            std::array<GeneralScene, 2> scenes;
            for (auto& scene : scenes) {
                build_scene(scene, type == 0, type == 1, type == 2);
                scene.params.use_parallel = parallel;
                scene.params.d_hat = 0.0;
                scene.params.k_barrier = 0.0;
                scene.params.use_ccd = false;
            }
            ASSERT_TRUE(solve_and_commit(scenes[0], false).converged);
            ASSERT_TRUE(solve_and_commit(scenes[1], true).converged);
            expect_states_near(scenes[1].state, scenes[0].state);
        }
    }
}

TEST(GeneralSIMDSolver, PureRigidContactAndFrameDispatchMatchExplicitEntryPoints) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (double friction : {0.0, 0.2}) {
        SCOPED_TRACE(friction);
        std::array<GeneralScene, 3> scenes;
        for (auto& scene : scenes) {
            build_scene(scene, false, false, true);
            scene.params.friction_coefficient = friction;
        }
        auto& driven = scenes[2];
        driven.params.use_basic_experimental = true;
        driven.params.use_basic_experimental_v2 = true;
        driven.params.use_simd = true;
        for (int frame = 1; frame <= 2; ++frame) {
            ASSERT_TRUE(solve_and_commit(scenes[0], false).converged);
            ASSERT_TRUE(solve_and_commit(scenes[1], true).converged);
            ASSERT_TRUE(advance_one_frame_rb(driven.state, driven.mesh,
                driven.params, frame).converged);
            expect_states_near(scenes[1].state, scenes[0].state);
            expect_states_bitwise_equal(driven.state, scenes[1].state);
        }
        const auto& x = scenes[1].state.deformed_positions;
        const auto& lower = scenes[1].mesh.rb_nodes[0];
        const int upper = scenes[1].mesh.rb_nodes[1][0];
        const double distance = node_triangle_distance(x[upper], x[lower[0]],
            x[lower[1]], x[lower[2]]).distance;
        EXPECT_GT(distance, 0.0);
        EXPECT_LT(distance, scenes[1].params.d_hat);
    }
}

TEST(GeneralSIMDSolver, SdfAndFrictionContributionsRemainActiveForAllBlockTypes) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<GeneralScene, 2> scenes;
    for (auto& scene : scenes) {
        build_scene(scene);
        scene.params.friction_coefficient = 0.2;
        scene.params.k_sdf = 2.0;
        scene.params.eps_sdf = 0.03;
        scene.params.sdf_planes.push_back({Vec3(0.0, -0.003, 0.0), Vec3::UnitY()});
    }
    for (int frame = 0; frame < 2; ++frame) {
        ASSERT_TRUE(solve_and_commit(scenes[0], false).converged);
        ASSERT_TRUE(solve_and_commit(scenes[1], true).converged);
        expect_states_near(scenes[1].state, scenes[0].state);
    }
}

TEST(GeneralSIMDSolver, ResidualComponentsMatchScalarAfterUnconvergedSweeps) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<GeneralScene, 2> scenes;
    std::array<SolverResult, 2> results;
    for (int method = 0; method < 2; ++method) {
        auto& scene = scenes[method];
        build_scene(scene);
        scene.params.fixed_iters = false;
        scene.params.tol_abs = 1e-30;
        scene.params.friction_coefficient = 0.2;
        results[method] = solve_and_commit(scene, method == 1);
        EXPECT_FALSE(results[method].converged);
        EXPECT_TRUE(results[method].has_residual);
        EXPECT_TRUE(results[method].has_residual_components);
        EXPECT_EQ(results[method].iterations, scene.params.max_global_iters);
    }
    expect_states_near(scenes[1].state, scenes[0].state);
    for (auto member : {&SolverResult::initial_cloth_residual,
            &SolverResult::initial_solid_residual, &SolverResult::initial_rigid_residual,
            &SolverResult::final_cloth_residual, &SolverResult::final_solid_residual,
            &SolverResult::final_rigid_residual, &SolverResult::final_residual}) {
        EXPECT_TRUE(std::isfinite(results[1].*member));
        EXPECT_NEAR(results[1].*member, results[0].*member,
            2e-8 * (1.0 + std::abs(results[0].*member)));
    }
    EXPECT_DOUBLE_EQ(results[1].final_residual, results[1].final_cloth_residual
        + results[1].final_solid_residual + results[1].final_rigid_residual);
}

TEST(GeneralSIMDSolver, GeneralFrameDispatchMatchesExplicitV2EntryPoint) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<GeneralScene, 2> scenes;
    for (auto& scene : scenes) {
        build_scene(scene);
        scene.params.use_basic_experimental = true;
        scene.params.use_basic_experimental_v2 = true;
        scene.params.use_simd = true;
        scene.params.friction_coefficient = 0.2;
    }
    for (int frame = 1; frame <= 2; ++frame) {
        ASSERT_TRUE(solve_and_commit(scenes[0], true).converged);
        auto& scene = scenes[1];
        ASSERT_TRUE(advance_one_frame_general(scene.state, scene.mesh, scene.adjacency,
            scene.pins, scene.params, scene.broad_phase, frame).converged);
        expect_states_bitwise_equal(scenes[1].state, scenes[0].state);
    }
}

TEST(GeneralSIMDSolver, PureClothEntryMatchesBasicV2AcrossContactAndFrictionModes) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (int execution : {0, 1, 2}) {
        omp_set_num_threads(execution == 1 ? 1 : 4);
        for (int contact_mode : {0, 1, 2}) {
            SCOPED_TRACE(::testing::Message() << "execution=" << execution
                << " contact_mode=" << contact_mode);
            std::array<GeneralScene, 2> scenes;
            for (auto& scene : scenes) {
                build_two_cloth_layers(scene);
                scene.params.use_parallel = execution != 0;
                scene.params.friction_coefficient = contact_mode == 2 ? 0.2 : 0.0;
                if (contact_mode == 0) {
                    scene.params.d_hat = 0.0;
                    scene.params.k_barrier = 0.0;
                    scene.params.use_ccd = false;
                }
            }
            for (int frame = 0; frame < 2; ++frame) {
                const auto basic = solve_basic_v2_and_commit(scenes[0]);
                const auto general = solve_and_commit(scenes[1], true);
                ASSERT_TRUE(basic.converged);
                ASSERT_TRUE(general.converged);
                EXPECT_EQ(general.iterations, basic.iterations);
                expect_states_bitwise_equal(scenes[1].state, scenes[0].state);
            }
            if (contact_mode != 0) {
                bool active = false;
                const auto& scene = scenes[1];
                const auto& x = scene.state.deformed_positions;
                for (const auto& pair : scene.broad_phase.cache().nt_pairs)
                    if ((pair.node < 49) != (pair.tri_v[0] < 49)
                        && node_triangle_distance(x[pair.node], x[pair.tri_v[0]],
                            x[pair.tri_v[1]], x[pair.tri_v[2]]).distance < scene.params.d_hat)
                        active = true;
                EXPECT_TRUE(active) << "the two cloth layers must have active contact";
            }
        }
    }
}

TEST(GeneralSIMDSolver, MixedSceneClothSweepMatchesBasicV2WithoutCrossTypeCoupling) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (bool sdf_friction : {false, true}) {
        SCOPED_TRACE(sdf_friction);
        GeneralScene basic, general;
        build_scene(basic, true, false, false);
        build_scene(general);
        for (auto* scene : {&basic, &general}) {
            enable_v2_flags(*scene);
            scene->params.use_parallel = false;
            scene->params.max_global_iters = 1;
            scene->params.d_hat = 0.0;
            scene->params.k_barrier = 0.0;
            scene->params.use_ccd = false;
            if (sdf_friction) {
                scene->params.friction_coefficient = 0.2;
                scene->params.k_sdf = 2.0;
                scene->params.eps_sdf = 0.03;
                scene->params.sdf_planes.push_back({Vec3(0.0, -0.003, 0.0), Vec3::UnitY()});
            }
        }
        for (int node : general.mesh.tet_nodes)
            general.state.deformed_positions[node].x() += 5.0;
        general.state.x_coms[0].x() += 10.0;
        sync_rigid_body_particles(general.mesh, general.state);
        ASSERT_TRUE(solve_basic_v2_and_commit(basic).converged);
        ASSERT_TRUE(solve_and_commit(general, true).converged);
        const auto count = basic.state.deformed_positions.size();
        expect_vectors_near(std::vector<Vec3>(general.state.deformed_positions.begin(),
            general.state.deformed_positions.begin() + count), basic.state.deformed_positions, 1e-11);
        expect_vectors_near(std::vector<Vec3>(general.state.velocities.begin(),
            general.state.velocities.begin() + count), basic.state.velocities, 1e-9);
    }
}

TEST(GeneralSIMDSolver, FailedFirstColorRestoresParticlesAndRigidCoordinates) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (int threads : {1, 4}) {
        SCOPED_TRACE(threads);
        omp_set_num_threads(threads);
        GeneralScene scene;
        build_scene(scene);
        scene.params.d_hat = 0.0;
        scene.params.k_barrier = 0.0;
        scene.params.use_ccd = false;
        // Separate components so the first color contains cloth, solid, and
        // rigid blocks. A cloth failure must undo other blocks in that color.
        for (int node : scene.mesh.tet_nodes)
            scene.state.deformed_positions[node].x() += 5.0;
        scene.state.x_coms[0].x() += 10.0;
        sync_rigid_body_particles(scene.mesh, scene.state);
        std::vector<Mat22> healthy_rest = scene.mesh.Dm_inverse;
        for (auto& dm : scene.mesh.Dm_inverse)
            dm.setConstant(std::numeric_limits<double>::quiet_NaN());

        auto positions = scene.state.deformed_positions;
        auto centers = scene.state.x_coms;
        auto orientations = scene.state.orientations;
        std::vector<Vec3> omega(scene.state.omega.size(), Vec3::Zero());
        const auto initial_positions = positions;
        const auto initial_centers = centers;
        const auto initial_orientations = orientations;
        const auto initial_omega = omega;
        std::vector<Vec3> predicted;
        build_xhat(predicted, positions, scene.state.velocities, scene.params.dt());
        for (std::size_t node = 0; node < predicted.size(); ++node)
            if (scene.mesh.node_to_rb[node] >= 0) predicted[node] = positions[node];
        const auto solve = [&] {
            return global_gauss_seidel_solver_general_experimental_v2(scene.mesh,
                scene.state, scene.adjacency, scene.pins, scene.params, positions,
                predicted, centers, orientations, omega, scene.broad_phase);
        };
        EXPECT_THROW(solve(), std::runtime_error);
        expect_vectors_bitwise_equal(positions, initial_positions);
        expect_vectors_bitwise_equal(centers, initial_centers);
        expect_vectors_bitwise_equal(orientations, initial_orientations);
        expect_vectors_bitwise_equal(omega, initial_omega);
        // A fresh rest-data allocation also refreshes the general solver's
        // existing topology cache, isolating recovery from cache invalidation.
        scene.mesh.Dm_inverse.swap(healthy_rest);
        EXPECT_TRUE(solve().converged);
    }
}

TEST(GeneralSIMDSolver, InPlaceRestMatrixChangesRefreshWarmedShapeGradients) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    GeneralScene warmed, fresh;
    build_scene(warmed);
    build_scene(fresh);
    const DeformedState initial = warmed.state;
    ASSERT_TRUE(solve_and_commit(warmed, true).converged);
    warmed.state = initial;
    const Mat22* storage = warmed.mesh.Dm_inverse.data();
    for (auto* scene : {&warmed, &fresh}) {
        scene->mesh.Dm_inverse[0](0, 0) *= 1.08;
        scene->mesh.Dm_inverse[0](0, 1) += 0.4;
    }
    ASSERT_EQ(warmed.mesh.Dm_inverse.data(), storage);
    // Run the warmed mesh first: solving the fresh mesh beforehand would
    // replace the static workspace and accidentally hide stale gradients.
    ASSERT_TRUE(solve_and_commit(warmed, true).converged);
    ASSERT_TRUE(solve_and_commit(fresh, true).converged);
    expect_states_near(warmed.state, fresh.state);
}

TEST(GeneralSIMDScheduler, HelperFailureRestoresOnlyItsColorAndJoinsNestedTeams) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (int execution : {0, 1, 2}) {
        SCOPED_TRACE(execution);
        omp_set_num_threads(execution == 0 ? 1 : 4);
        solver_detail::GeneralSimdBatches batches;
        batches.prepare({{0}, {1, 2}, {3}}, 0,
            [](int) { return std::size_t(256); });
        std::array<int, 4> values{10, 20, 30, 40}, snapshots{};
        std::atomic<int> active_ranges{0};
        std::atomic<bool> body_updated{false}, helpers_used{false};
        std::exception_ptr failure;
        const auto process = [&](int item, bool cooperative) {
            const int block = batches.batches[item].blocks[0];
            ++values[block];
            if (block == 1) body_updated.store(true, std::memory_order_release);
            if (block != 2) return;
            // Force another block of this color to commit before the error.
            while (!body_updated.load(std::memory_order_acquire))
                solver_detail::contact_spin_hint();
            helpers_used.store(cooperative, std::memory_order_relaxed);
            solver_detail::ordered_contact_tasks(128, cooperative,
                [&](int contact) {
                    struct ActiveRange {
                        std::atomic<int>& active;
                        explicit ActiveRange(std::atomic<int>& value) : active(value) { ++active; }
                        ~ActiveRange() { --active; }
                    } range(active_ranges);
                    if (contact == 33) throw std::runtime_error("injected contact failure");
                    return contact;
                }, [](int) {});
        };
        const auto run = [&] {
            try {
                solver_detail::run_general_simd_batches(batches, 3, process,
                    [&](int item) {
                        const int block = batches.batches[item].blocks[0];
                        snapshots[block] = values[block];
                    }, [&](int item) {
                        const int block = batches.batches[item].blocks[0];
                        values[block] = snapshots[block];
                    });
            } catch (...) { failure = std::current_exception(); }
        };
        if (execution == 2) {
            #pragma omp parallel num_threads(2)
            {
                #pragma omp single
                { run(); }
            }
        } else run();
        ASSERT_TRUE(failure != nullptr);
        EXPECT_THROW(std::rethrow_exception(failure), std::runtime_error);
        EXPECT_EQ(values, (std::array<int, 4>{11, 20, 30, 40}));
        EXPECT_EQ(active_ranges.load(), 0);
        EXPECT_EQ(helpers_used.load(), execution == 1);
        EXPECT_EQ(solver_detail::active_contact_task_group, nullptr);
    }
}

TEST(GeneralSIMDScheduler, CheapAndUniformWholeBatchesUseStaticRoundRobinOwnership) {
    RestoreOpenMP restore;
    omp_set_num_threads(4);
    omp_set_dynamic(0);
    constexpr int count = 61, sweeps = 3;
    for (int cost_mode : {0, 1, 2}) {
        SCOPED_TRACE(cost_mode);
        std::vector<int> group(count);
        std::iota(group.begin(), group.end(), 0);
        solver_detail::GeneralSimdBatches batches;
        // Treat every block as a rigid singleton so ownership measures the
        // scheduler rather than a thread-count-dependent particle packing.
        batches.prepare({group}, 0, [&](int block) {
            return cost_mode == 0 ? std::size_t(0)
                : (cost_mode == 1 ? std::size_t(block % 32) : std::size_t(256));
        });
        std::array<std::atomic<int>, count> visits{};
        for (auto& visit : visits) visit.store(0);
        std::atomic<bool> wrong_worker{false}, unexpected_helpers{false};
        solver_detail::run_general_simd_batches(batches, sweeps,
            [&](int item, bool cooperative) {
                if (cooperative) unexpected_helpers.store(true);
                const int block = batches.batches[item].blocks[0];
                if (omp_get_thread_num() != block % omp_get_num_threads())
                    wrong_worker.store(true);
                visits[block].fetch_add(1);
            }, [](int) {}, [](int) {});
        EXPECT_FALSE(wrong_worker.load());
        EXPECT_FALSE(unexpected_helpers.load());
        for (int block = 0; block < count; ++block)
            EXPECT_EQ(visits[block].load(), sweeps) << "block=" << block;
    }
}

TEST(GeneralSIMDScheduler, UnevenWholeBatchesReserveHeavyFirstWaveForEveryWorker) {
    RestoreOpenMP restore;
    omp_set_num_threads(4);
    omp_set_dynamic(0);
    // Leave enough work after the first wave to exercise the adaptive cursor
    // chunk cap of eight, while keeping every item's cost below 512.
    constexpr int count = 193;
    std::vector<int> group(count);
    std::iota(group.begin(), group.end(), 0);
    solver_detail::GeneralSimdBatches batches;
    // The unsorted input has cheap work first; no cost reaches the helper
    // threshold. Adaptive whole-batch scheduling must reverse this order.
    batches.prepare({group}, 0,
        [](int block) { return std::size_t(block + 1); });
    std::array<int, 4> first_block{-1, -1, -1, -1};
    std::array<std::atomic<int>, count> visits{};
    for (auto& visit : visits) visit.store(0);
    std::atomic<int> actual_team{0};
    std::atomic<bool> unexpected_helpers{false};
    solver_detail::run_general_simd_batches(batches, 1,
        [&](int item, bool cooperative) {
            if (cooperative) unexpected_helpers.store(true);
            const int worker = omp_get_thread_num();
            const int block = batches.batches[item].blocks[0];
            if (first_block[worker] < 0) first_block[worker] = block;
            actual_team.store(omp_get_num_threads());
            visits[block].fetch_add(1);
        }, [](int) {}, [](int) {});
    ASSERT_EQ(actual_team.load(), 4);
    EXPECT_FALSE(unexpected_helpers.load());
    for (int worker = 0; worker < 4; ++worker)
        EXPECT_EQ(first_block[worker], count - 1 - worker) << "worker=" << worker;
    for (int block = 0; block < count; ++block)
        EXPECT_EQ(visits[block].load(), 1) << "block=" << block;
}

TEST(GeneralSIMDScheduler, StaticAndDynamicQueuesVisitEveryBlockOncePerColorAndSweep) {
    RestoreOpenMP restore;
    for (int threads : {1, 2, 4}) {
        for (int configuration : {0, 1, 2, 3}) {
            const int dynamic = configuration % 2;
            const bool uneven = configuration >= 2;
            SCOPED_TRACE(::testing::Message() << "threads=" << threads
                << " dynamic=" << dynamic << " uneven=" << uneven);
            omp_set_num_threads(threads);
            omp_set_dynamic(dynamic);
            constexpr int count = 61, sweeps = 3;
            std::vector<std::vector<int>> groups(3);
            for (int block = 0; block < count; ++block)
                groups[block < 17 ? 0 : (block < 56 ? 1 : 2)].push_back(block);
            solver_detail::GeneralSimdBatches batches;
            batches.prepare(groups, uneven ? 0 : count, [&](int block) {
                return uneven ? std::size_t(4 * (block + 1)) : std::size_t(0);
            });
            std::array<std::atomic<int>, count> visits{};
            for (auto& visit : visits) visit.store(0);
            std::atomic<bool> barrier_violation{false}, unexpected_helpers{false};
            const auto process = [&](int item, bool cooperative) {
                if (cooperative) unexpected_helpers.store(true);
                const auto& batch = batches.batches[item];
                for (std::size_t i = 0; i < batch.size; ++i) {
                    const int block = batch.blocks[i];
                    const int prior = visits[block].load();
                    const int color = block < 17 ? 0 : (block < 56 ? 1 : 2);
                    if (color > 0) {
                        for (int previous : groups[color - 1])
                            if (visits[previous].load() != prior + 1)
                                barrier_violation.store(true);
                    } else if (prior > 0) {
                        for (int previous : groups.back())
                            if (visits[previous].load() != prior)
                                barrier_violation.store(true);
                    }
                    visits[block].fetch_add(1);
                }
            };
            solver_detail::run_general_simd_batches(batches, sweeps, process,
                [](int) {}, [](int) {});
            EXPECT_FALSE(barrier_violation.load());
            EXPECT_FALSE(unexpected_helpers.load());
            for (int block = 0; block < count; ++block)
                EXPECT_EQ(visits[block].load(), sweeps) << "block=" << block;
        }
    }
}

TEST(GeneralSIMDScheduler, StaticAndDynamicQueueFailuresRollBackTheirColorBeforeReturning) {
    RestoreOpenMP restore;
    omp_set_num_threads(4);
    for (int configuration : {0, 1, 2, 3}) {
        const int dynamic = configuration % 2;
        const bool uneven = configuration >= 2;
        SCOPED_TRACE(::testing::Message() << "dynamic=" << dynamic << " uneven=" << uneven);
        omp_set_dynamic(dynamic);
        std::vector<std::vector<int>> groups(3, std::vector<int>(32));
        for (int color = 0; color < 3; ++color)
            std::iota(groups[color].begin(), groups[color].end(), 32 * color);
        solver_detail::GeneralSimdBatches batches;
        batches.prepare(groups, uneven ? 0 : 96, [&](int block) {
            return uneven ? std::size_t(8 * (block % 32 + 1)) : std::size_t(0);
        });
        std::array<int, 96> values{}, saved{};
        const auto snapshot = [&](int item, bool restore_values) {
            const auto& batch = batches.batches[item];
            for (std::size_t i = 0; i < batch.size; ++i) {
                const int block = batch.blocks[i];
                if (restore_values) values[block] = saved[block];
                else saved[block] = values[block];
            }
        };
        EXPECT_THROW(solver_detail::run_general_simd_batches(batches, 3,
            [&](int item, bool) {
                const auto& batch = batches.batches[item];
                for (std::size_t i = 0; i < batch.size; ++i) {
                    const int block = batch.blocks[i];
                    ++values[block];
                    if (block == 45) throw std::runtime_error("failed unsplit batch");
                }
            }, [&](int item) { snapshot(item, false); },
            [&](int item) { snapshot(item, true); }), std::runtime_error);
        for (int block = 0; block < 96; ++block)
            EXPECT_EQ(values[block], block < 32 ? 1 : 0) << "block=" << block;
    }
}

TEST(GeneralSIMDContacts, SparseTilesPreserveResultsAndOnlyCertifyAabbRejections) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (double friction : {0.0, 0.2}) {
        SparseContacts input;
        input.params.friction_coefficient = friction;
        solver_detail::GeneralSimdVertexSystem reference;
        reference.gradient = Vec3(0.25, -0.5, 0.75);
        reference.hessian = 0.5 * Mat33::Identity();
        solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
            input.params, input.positions, &input.previous, input.solid,
            input.surface, false, reference);
        for (bool cooperative : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "friction=" << friction
                << " cooperative=" << cooperative);
            solver_detail::GeneralSimdVertexSystem actual;
            safe_step_detail::VertexAabbRejections rejections;
            std::atomic<int> leader_calls{0};
            const std::function<void()> prepare_local = [&] {
                ++leader_calls;
                actual.gradient = Vec3(0.25, -0.5, 0.75);
                actual.hessian = 0.5 * Mat33::Identity();
            };
            const auto run = [&] {
                solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
                    input.params, input.positions, &input.previous, input.solid,
                    input.surface, cooperative, actual, &rejections, &prepare_local);
            };
            if (cooperative) {
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp single
                    { run(); }
                }
            } else run();
            EXPECT_EQ(leader_calls.load(), 1);
            EXPECT_EQ(0, std::memcmp(actual.gradient.data(), reference.gradient.data(), 3 * sizeof(double)));
            EXPECT_EQ(0, std::memcmp(actual.hessian.data(), reference.hessian.data(), 9 * sizeof(double)));
            EXPECT_DOUBLE_EQ(rejections.distance, input.params.d_hat);
            ASSERT_EQ(rejections.clear.size(), 161u);
            for (int i = 0; i < 96; ++i)
                EXPECT_EQ(rejections.clear[i], i % 3 == 0 ? 1 : 0) << "NT=" << i;
            for (int i = 0; i < 65; ++i)
                EXPECT_EQ(rejections.clear[96 + i], i % 3 == 0 ? 0 : 1) << "SS=" << i;
        }
    }
}

TEST(GeneralSIMDContacts, BarrierKernelsMatchScalarBitsForRotatedRoles) {
    std::array<ipc_simd::MeshContactInput, 32> inputs;
    std::array<ipc_simd::MeshContactOutput, 32> outputs;
    constexpr double d_hat = 0.1;
    for (bool segment : {false, true}) {
        for (int sample = 0; sample < 8; ++sample) {
            const Mat33 rotation = Eigen::AngleAxisd(0.17 * sample,
                Vec3(0.3, -0.2, 0.7).normalized()).toRotationMatrix();
            const std::array<Vec3, 4> base = segment
                ? std::array<Vec3, 4>{Vec3(-1, 0, 0), Vec3(1, 0.1, 0),
                    Vec3(0, -1, 0.007), Vec3(0, 1, 0.007)}
                : std::array<Vec3, 4>{Vec3(0.15, 0.2, 0.007), Vec3(0, 0, 0),
                    Vec3(1, 0, 0), Vec3(0, 1, 0)};
            for (int role = 0; role < 4; ++role) {
                auto& input = inputs[4 * sample + role];
                input.segment_segment = segment;
                input.role = role;
                for (int corner = 0; corner < 4; ++corner)
                    input.positions[corner] = rotation * base[corner]
                        + Vec3(0.031 * sample, -0.017 * sample, 0.011 * sample);
            }
        }
        ipc_simd::general_mesh_contact_derivatives_tile(inputs.data(), inputs.size(),
            d_hat, 2.0, 0.0, 1.0 / 600.0, 0.01, outputs.data());
        for (std::size_t entry = 0; entry < inputs.size(); ++entry) {
            SCOPED_TRACE(::testing::Message() << "segment=" << segment << " entry=" << entry);
            const auto& input = inputs[entry];
            const auto& x = input.positions;
            const auto expected = segment
                ? segment_segment_barrier_self_gradient_and_hessian(
                    x[0], x[1], x[2], x[3], d_hat, input.role)
                : node_triangle_barrier_self_gradient_and_hessian(
                    x[0], x[1], x[2], x[3], d_hat, input.role);
            const auto exact = [&](const auto& actual, const auto& reference,
                                   const char* field) {
                for (int component = 0; component < reference.size(); ++component) {
                    if (std::memcmp(actual.data() + component, reference.data() + component,
                            sizeof(double)) == 0) continue;
                    ADD_FAILURE() << field << '[' << component << "] actual="
                        << std::hexfloat << actual.data()[component]
                        << " expected=" << reference.data()[component];
                }
            };
            exact(outputs[entry].gradient, expected.first, "gradient");
            exact(outputs[entry].hessian, expected.second, "hessian");
        }
    }
}

TEST(GeneralSIMDContacts, LeaderFailureJoinsHelpersAndLeavesOutputUnaccumulated) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    SparseContacts input;
    for (bool cooperative : {false, true}) {
        solver_detail::GeneralSimdVertexSystem output;
        output.gradient = Vec3(1, 2, 3);
        output.hessian = Mat33::Identity();
        std::atomic<int> leader_calls{0};
        std::exception_ptr failure;
        const std::function<void()> fail = [&] {
            ++leader_calls;
            throw std::runtime_error("failed local preparation");
        };
        const auto run = [&] {
            try {
                solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
                    input.params, input.positions, &input.previous, input.solid,
                    input.surface, cooperative, output, nullptr, &fail);
            } catch (...) { failure = std::current_exception(); }
        };
        if (cooperative) {
            #pragma omp parallel num_threads(4)
            {
                #pragma omp single
                { run(); }
            }
        } else run();
        ASSERT_TRUE(failure != nullptr);
        EXPECT_THROW(std::rethrow_exception(failure), std::runtime_error);
        EXPECT_EQ(leader_calls.load(), 1);
        EXPECT_TRUE(output.gradient.isApprox(Vec3(1, 2, 3), 0.0));
        EXPECT_TRUE(output.hessian.isIdentity(0.0));
    }
}

TEST(GeneralSIMDContacts, FixedHelpersEvaluateContactsDuringLocalPreparationAndJoinOnFailure) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    SparseContacts input;
    solver_detail::GeneralSimdVertexSystem initial;
    initial.gradient = Vec3(0.25, -0.5, 0.75);
    initial.hessian = 0.5 * Mat33::Identity();
    auto reference = initial;
    solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
        input.params, input.positions, &input.previous, input.solid,
        input.surface, false, reference);

    for (bool fail_local : {false, true}) {
        SCOPED_TRACE(fail_local);
        solver_detail::GeneralSimdBatches batches;
        batches.prepare({{0}}, 0, [](int) { return std::size_t(256); });
        auto actual = initial;
        safe_step_detail::VertexAabbRejections rejections;
        std::atomic<int> leader_calls{0};
        bool helpers_completed_during_callback = false;
        bool unaccumulated_when_thrown = false;
        std::exception_ptr failure;
        const auto process = [&](int, bool cooperative) {
            auto* helpers = solver_detail::active_contact_task_group;
            if (!cooperative || helpers == nullptr || helpers->helpers == 0)
                throw std::runtime_error("test requires the fixed contact helper path");
            const std::function<void()> prepare_local = [&] {
                ++leader_calls;
                const auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(2);
                // dispatch publishes contacts before calling leader_work. The
                // leader cannot evaluate any ranges until this callback ends,
                // so completion here proves actual helper/leader overlap.
                while (helpers->remaining.load(std::memory_order_acquire) != 0) {
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("contact helpers did not finish during local preparation");
                    solver_detail::contact_spin_hint();
                }
                helpers_completed_during_callback = true;
                // The acquire above joins all helper writes, including masks.
                if (rejections.clear.size() != 161 || rejections.clear[0] != 1
                    || rejections.clear[1] != 0 || rejections.clear[2] != 0)
                    throw std::runtime_error("helper contact masks were not published before the join");
                if (fail_local) throw std::runtime_error("failed overlapped local preparation");
                actual.gradient = initial.gradient;
                actual.hessian = initial.hessian;
            };
            try {
                solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
                    input.params, input.positions, &input.previous, input.solid,
                    input.surface, cooperative, actual, &rejections, &prepare_local);
            } catch (...) {
                unaccumulated_when_thrown =
                    std::memcmp(actual.gradient.data(), initial.gradient.data(), 3 * sizeof(double)) == 0
                    && std::memcmp(actual.hessian.data(), initial.hessian.data(), 9 * sizeof(double)) == 0;
                throw;
            }
        };
        try {
            solver_detail::run_general_simd_batches(batches, 1, process,
                [](int) {}, [&](int) { actual = initial; });
        } catch (...) { failure = std::current_exception(); }
        EXPECT_EQ(leader_calls.load(), 1);
        EXPECT_TRUE(helpers_completed_during_callback);
        if (fail_local) {
            ASSERT_TRUE(failure != nullptr);
            EXPECT_THROW(std::rethrow_exception(failure), std::runtime_error);
            EXPECT_TRUE(unaccumulated_when_thrown);
            EXPECT_TRUE(actual.gradient.isApprox(initial.gradient, 0.0));
            EXPECT_TRUE(actual.hessian.isApprox(initial.hessian, 0.0));
        } else {
            EXPECT_TRUE(failure == nullptr);
            EXPECT_EQ(0, std::memcmp(actual.gradient.data(), reference.gradient.data(), 3 * sizeof(double)));
            EXPECT_EQ(0, std::memcmp(actual.hessian.data(), reference.hessian.data(), 9 * sizeof(double)));
        }
        EXPECT_EQ(solver_detail::active_contact_task_group, nullptr);
    }
}
