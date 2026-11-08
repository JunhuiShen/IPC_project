#include "DAT.h"
#include "jacobi_step.h"
#include "mesh_utils.h"
#include "parallel_helper.h"
#include "simulation.h"
#include "solid_ipc.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
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
    DeformedState state;
    state.velocities = s.velocity;
    state.deformed_positions = previous ? *previous : s.x;
    if (!previous && s.params.friction_coefficient > 0
        && s.xhat.size() == s.x.size() && s.velocity.size() == s.x.size()) {
        const double dt = s.params.dt();
        for (std::size_t node = 0; node < s.x.size(); ++node)
            state.deformed_positions[node] = s.xhat[node] - dt * s.velocity[node];
    }
    std::vector<Vec3> centers, omega;
    std::vector<Vec4> orientations;
    return global_gauss_seidel_solver_dat(s.mesh, state, s.adj, s.pins, s.params,
        s.x, s.xhat, centers, orientations, omega, phase);
}

// Build the elastic graph directly, independently of the solver workspace.
// In particular, the collision triangles of a tet are not a substitute for its
// six volume-element dependencies, and both hinge apices must be separated.
std::vector<std::vector<int>> structural_colors(const Scene& s) {
    std::vector<std::vector<int>> graph(s.x.size());
    const auto connect = [&](int a, int b) {
        if (a == b) return;
        graph[a].push_back(b);
        graph[b].push_back(a);
    };
    for (std::size_t i = 0; i < s.mesh.tris.size(); i += 3)
        for (int a = 0; a < 3; ++a) for (int b = a + 1; b < 3; ++b)
            connect(s.mesh.tris[i + a], s.mesh.tris[i + b]);
    for (const auto& hinge : s.mesh.hinges)
        for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b)
            connect(hinge.v[a], hinge.v[b]);
    for (std::size_t i = 0; i < s.mesh.tets.size(); i += 4)
        for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b)
            connect(s.mesh.tets[i + a], s.mesh.tets[i + b]);
    for (auto& row : graph) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    std::vector<std::vector<int>> colors;
    greedy_color_conflict_graph(graph, colors);
    return colors;
}

using LocalSystem = std::pair<Vec3, Mat33>;

std::vector<LocalSystem> mesh_contact_systems(const Scene& s,
    const std::vector<Vec3>& positions, const std::vector<Vec3>* previous) {
    std::vector<AABB> boxes;
    for (const auto& p : positions) boxes.emplace_back(p, p);
    BroadPhase phase;
    if (s.mesh.tets.empty())
        phase.initialize(boxes, s.mesh, s.params.d_hat,
            BroadPhase::InitializationMode::DeformableSolver);
    else
        phase.initialize_surface_nodes(boxes, s.mesh, s.params.d_hat,
            BroadPhase::InitializationMode::DeformableSolver);
    auto without_mesh = s.params;
    without_mesh.d_hat = 0;
    auto contact_mesh = s.mesh;
    // The solid scalar assembler also provides the shared mesh-contact law,
    // but requires an adjacency row even for a cloth node with no tetrahedra.
    contact_mesh.tet_adj.resize(positions.size());
    // Its default ownership masks intentionally discard cloth-only contacts.
    // For this shared-law oracle all retained pairs are contact-active; the
    // phase above already removed actual tet-interior collision queries.
    const std::vector<unsigned char> contact_nodes(positions.size(), 1);
    std::vector<LocalSystem> result(positions.size());
    for (int node = 0; node < static_cast<int>(positions.size()); ++node) {
        // Subtraction isolates only mesh barrier/friction, leaving SDF and
        // elastic terms in the live local evaluation below. This scalar oracle
        // intentionally does not use the new DAT contact snapshot helper.
        const auto full = compute_solid_local_gradient_and_block(node, contact_mesh,
            s.pins, s.params, positions, s.xhat, phase,
            &contact_nodes, &contact_nodes, nullptr, previous);
        const auto noncontact = compute_solid_local_gradient_and_block(node,
            contact_mesh, s.pins, without_mesh, positions, s.xhat, phase,
            &contact_nodes, &contact_nodes, nullptr, previous);
        result[node] = {full.first - noncontact.first,
            full.second - noncontact.second};
    }
    return result;
}

std::vector<Vec3> composed_sweeps(const Scene& s, int count,
    const std::vector<Vec3>* previous = nullptr, bool freeze_contact = false) {
    auto positions = s.x;
    // Structural colors are independent of contact candidates. Numerical
    // contact evaluations below refresh at each color's accepted positions.
    const auto colors = structural_colors(s);
    auto complete_adjacency = s.adj;
    for (std::size_t node = 0; node < positions.size(); ++node)
        complete_adjacency.try_emplace(static_cast<int>(node));
    std::vector<unsigned char> solid(positions.size(), 0);
    for (int node : s.mesh.tets) solid[node] = 1;
    auto without_mesh = s.params;
    without_mesh.d_hat = 0;
    std::vector<Vec3> reconstructed;
    if (s.params.friction_coefficient > 0 && !previous) {
        reconstructed.resize(positions.size());
        for (std::size_t node = 0; node < positions.size(); ++node)
            reconstructed[node] = s.xhat[node] - s.params.dt() * s.velocity[node];
        previous = &reconstructed;
    }
    for (int iteration = 0; iteration < count; ++iteration) {
        auto contact = mesh_contact_systems(s, positions, previous);
        for (const auto& color : colors) {
            if (!freeze_contact) contact = mesh_contact_systems(s, positions, previous);
            std::vector<Vec3> raw(positions.size(), Vec3::Zero());
            BroadPhase unused;
            for (int node : color) {
                auto local = solid[node]
                    ? compute_solid_local_gradient_and_block(node, s.mesh,
                        s.pins, without_mesh, positions, s.xhat, unused,
                        nullptr, nullptr, nullptr, previous)
                    : compute_local_gradient_and_hessian_no_barrier(node,
                        s.mesh, complete_adjacency, s.pins, without_mesh, positions, s.xhat,
                        nullptr, nullptr, nullptr, previous);
                local.first += contact[node].first;
                local.second += contact[node].second;
                raw[node] = -s.params.damping
                    * (matrix3d_inverse(local.second) * local.first);
            }
            // Rebuild discovery independently at the current accepted geometry.
            // This also checks that the solver's iteration-start pair superset
            // remains sufficient as earlier colors move.
            const auto safe = planar_DAT(s.params.dat_gamma,
                s.params.resolved_dat_query_radius(), s.mesh, raw, positions,
                s.params.dat_node_box_padding);
            for (std::size_t node = 0; node < positions.size(); ++node)
                positions[node] += safe[node];
        }
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

TEST(ColoredDATStructure, TetColorsDoNotDependOnSurfaceTrianglesAndTrackInPlaceEdits) {
    Scene s({Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY(), Vec3::UnitZ(),
        Vec3(1, 1, 1)});
    ASSERT_TRUE(s.mesh.tris.empty());
    s.mesh.tets = {0, 1, 2, 3};
    DATColoredStepWorkspace workspace;
    const auto vertex_colors = [&]() {
        std::vector<int> colors(s.x.size(), -1);
        for (std::size_t color = 0; color < workspace.colors().size(); ++color)
            for (int node : workspace.colors()[color]) {
                EXPECT_EQ(colors[node], -1) << "duplicate node " << node;
                colors[node] = static_cast<int>(color);
            }
        return colors;
    };
    const auto expect_tet_colors = [&](const std::vector<int>& colors) {
        for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b)
            EXPECT_NE(colors[s.mesh.tets[a]], colors[s.mesh.tets[b]]);
        for (int color : colors) EXPECT_GE(color, 0);
    };
    workspace.prepare(s.mesh, s.adj, s.pins, s.params, s.x.size());
    expect_tet_colors(vertex_colors());
    // Preserve the vector allocation and element count. A pointer/size-only
    // cache would leave vertices 0 and 4 with the same stale color here.
    const int* connectivity = s.mesh.tets.data();
    s.mesh.tets[3] = 4;
    ASSERT_EQ(s.mesh.tets.data(), connectivity);
    workspace.prepare(s.mesh, s.adj, s.pins, s.params, s.x.size());
    expect_tet_colors(vertex_colors());
}

TEST(ColoredDATStructure, HingeOppositeApicesHaveDifferentColors) {
    auto s = cloth_scene();
    ASSERT_EQ(s.mesh.hinges.size(), 1u);
    DATColoredStepWorkspace workspace;
    workspace.prepare(s.mesh, s.adj, s.pins, s.params, s.x.size());
    std::vector<int> colors(s.x.size(), -1);
    for (std::size_t color = 0; color < workspace.colors().size(); ++color)
        for (int node : workspace.colors()[color]) colors[node] = static_cast<int>(color);
    const auto& hinge = s.mesh.hinges[0];
    EXPECT_GE(colors[hinge.v[2]], 0);
    EXPECT_GE(colors[hinge.v[3]], 0);
    EXPECT_NE(colors[hinge.v[2]], colors[hinge.v[3]]);
}

TEST(ColoredDATStructure, ContactPairsDoNotChangeStructuralColorsAcrossSolves) {
    Scene s({Vec3(.2, .2, .1), Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY()},
        {1, 2, 3});
    s.params.dat_query_radius = .25;
    s.params.d_hat = .2;
    s.params.k_barrier = 10;
    const auto phase_for = [&](const std::vector<Vec3>& positions) {
        std::vector<AABB> boxes;
        for (const auto& position : positions)
            boxes.emplace_back(position - Vec3::Constant(.3),
                position + Vec3::Constant(.3));
        BroadPhase phase;
        phase.initialize(boxes, s.mesh, 0,
            BroadPhase::InitializationMode::DeformableSolver);
        return phase;
    };
    const auto color_ids = [&](const DATColoredStepWorkspace& workspace) {
        std::vector<int> ids(s.x.size(), -1);
        for (std::size_t color = 0; color < workspace.colors().size(); ++color)
            for (int node : workspace.colors()[color]) ids[node] = static_cast<int>(color);
        return ids;
    };
    DATColoredStepWorkspace workspace;
    workspace.prepare(s.mesh, s.adj, s.pins, s.params, s.x.size());
    const auto elastic_colors = color_ids(workspace);
    for (int a = 1; a < 4; ++a) for (int b = a + 1; b < 4; ++b)
        EXPECT_NE(elastic_colors[a], elastic_colors[b]);
    // The isolated point is contact-coupled to the triangle, but has no
    // elastic dependency and therefore shares a color with one of its corners.
    EXPECT_TRUE(elastic_colors[0] == elastic_colors[1]
        || elastic_colors[0] == elastic_colors[2] || elastic_colors[0] == elastic_colors[3]);
    const auto frozen_colors = workspace.colors();
    auto initial_phase = phase_for(s.x);
    ASSERT_FALSE(initial_phase.cache().nt_pairs.empty());
    workspace.freeze_contacts(s.params, s.x, nullptr, initial_phase);
    EXPECT_EQ(workspace.colors(), frozen_colors);

    auto moved = s.x;
    moved[0].z() = 5;
    auto refreshed_phase = phase_for(moved);
    ASSERT_TRUE(refreshed_phase.cache().nt_pairs.empty());
    workspace.freeze_contacts(s.params, moved, nullptr, refreshed_phase);
    EXPECT_EQ(workspace.colors(), frozen_colors)
        << "Numerical contact refresh must not alter elastic colors";

    workspace.prepare(s.mesh, s.adj, s.pins, s.params, s.x.size());
    workspace.freeze_contacts(s.params, moved, nullptr, refreshed_phase);
    EXPECT_EQ(color_ids(workspace), elastic_colors)
        << "A new solve with unchanged elastic topology must retain the colors";
    workspace.prepare(s.mesh, s.adj, s.pins, s.params, s.x.size());
    workspace.freeze_contacts(s.params, s.x, nullptr, initial_phase);
    EXPECT_EQ(color_ids(workspace), elastic_colors);
}

TEST(ColoredDATStructure, EdgeContactPairsDoNotAddElasticColorConflicts) {
    Scene s({Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(0, -2, 0),
        Vec3(0, -1, .1), Vec3(0, 1, .1), Vec3(2, 0, .1)},
        {0, 1, 2, 3, 4, 5});
    std::vector<AABB> boxes;
    for (const auto& position : s.x)
        boxes.emplace_back(position - Vec3::Constant(.25),
            position + Vec3::Constant(.25));
    BroadPhase phase;
    phase.initialize(boxes, s.mesh, 0,
        BroadPhase::InitializationMode::DeformableSolver);
    ASSERT_FALSE(phase.cache().ss_pairs.empty());
    DATColoredStepWorkspace workspace;
    workspace.prepare(s.mesh, s.adj, s.pins, s.params, s.x.size());
    std::vector<int> colors(s.x.size(), -1);
    for (std::size_t color = 0; color < workspace.colors().size(); ++color)
        for (int node : workspace.colors()[color]) colors[node] = static_cast<int>(color);
    ASSERT_EQ(workspace.colors().size(), 3u);
    for (int first : {0, 3})
        for (int a = 0; a < 3; ++a) for (int b = a + 1; b < 3; ++b)
            EXPECT_NE(colors[first + a], colors[first + b]);
    bool shared_contact_color = false;
    for (const auto& pair : phase.cache().ss_pairs)
        for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b)
            shared_contact_color |= colors[pair.v[a]] == colors[pair.v[b]];
    EXPECT_TRUE(shared_contact_color)
        << "Contact-only neighbors may share a color across independent triangles";
    const auto before_contact = workspace.colors();
    workspace.freeze_contacts(s.params, s.x, nullptr, phase);
    EXPECT_EQ(workspace.colors(), before_contact);
}

TEST(JacobiDATLegacyAPI, FrozenCompositionRemainsAvailableForClothSolidAndMixed) {
    for (int kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        auto s = scene_for_kind(kind);
        s.params.max_global_iters = 2;
        auto expected = s.x;
        for (int sweep = 0; sweep < s.params.max_global_iters; ++sweep) {
            const auto proposed = jacobi_step(s.mesh, s.adj, s.pins, s.params,
                expected, s.xhat, s.velocity);
            const auto accepted = planar_DAT(s.params.dat_gamma,
                s.params.resolved_dat_query_radius(), s.mesh, proposed, expected,
                s.params.dat_node_box_padding);
            for (std::size_t node = 0; node < expected.size(); ++node)
                expected[node] += accepted[node];
        }
        BroadPhase phase;
        const auto result = global_jacobi_solver_dat(s.mesh, s.adj, s.pins,
            s.params, s.x, s.xhat, s.velocity, phase);
        ASSERT_TRUE(result.converged);
        EXPECT_EQ(result.iterations, 2);
        expect_positions(s.x, expected);
    }
}

TEST(JacobiDATSolver, OneSweepMatchesColoredLiveElasticityForClothSolidAndMixed) {
    for (int kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        auto s = scene_for_kind(kind);
        const auto target = s.xhat;
        const auto expected = composed_sweeps(s, 1);
        const auto jacobi_proposal = jacobi_step(s.mesh, s.adj, s.pins,
            s.params, s.x, s.xhat, s.velocity);
        auto jacobi_positions = s.x;
        const auto jacobi_safe = planar_DAT(s.params.dat_gamma,
            s.params.dat_query_radius, s.mesh, jacobi_proposal, s.x,
            s.params.dat_node_box_padding);
        double cross_color_effect = 0;
        for (std::size_t node = 0; node < s.x.size(); ++node) {
            jacobi_positions[node] += jacobi_safe[node];
            cross_color_effect = std::max(cross_color_effect,
                (expected[node] - jacobi_positions[node]).norm());
        }
        ASSERT_GT(cross_color_effect, 1e-7);
        BroadPhase phase;
        // An explicit entry point selects DAT even with its driver flag off.
        ASSERT_FALSE(s.params.use_dat);
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
    const auto history = previous;
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
        expect_positions(previous, history, 0.0);
    }
}

TEST(JacobiDATSolver, LaterColorsRefreshContactGeometryAndFriction) {
    Scene s({Vec3(0, 0, .1), Vec3(-1, -1, 0),
        Vec3(2, -1, 0), Vec3(-1, 2, 0)}, {1, 2, 3});
    s.params.d_hat = .2;
    s.params.k_barrier = 100;
    s.params.friction_coefficient = .4;
    s.params.damping = .5;
    auto previous = s.x;
    previous[0].x() -= .03;
    s.xhat[0].x() += .02;
    const auto expected = composed_sweeps(s, 1, &previous);
    const auto iteration_contact = composed_sweeps(s, 1, &previous, true);
    double snapshot_effect = 0;
    for (std::size_t node = 0; node < s.x.size(); ++node)
        snapshot_effect = std::max(snapshot_effect,
            (expected[node] - iteration_contact[node]).norm());
    ASSERT_GT(snapshot_effect, 1e-6);
    BroadPhase phase;
    ASSERT_TRUE(solve(s, phase, &previous).converged);
    expect_positions(s.x, expected);
}

TEST(ColoredDATContacts, SelectedColorsMatchFullSnapshotsBitwiseAcrossThreadsAndFriction) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    std::vector<Vec3> initial;
    std::vector<int> triangles;
    for(int copy=0;copy<96;++copy) {
        const int first=int(initial.size());
        const Vec3 offset(12*copy,0,0);
        const std::vector<Vec3> component=copy%2
            ? std::vector<Vec3>{Vec3(-2,0,0),Vec3(2,0,0),Vec3(0,-2,-2),
                Vec3(0,-2,.1),Vec3(0,2,.1),Vec3(2,0,2.1)}
            : std::vector<Vec3>{Vec3(0,0,.1),Vec3(-1,-1,0),
                Vec3(2,-1,0),Vec3(-1,2,0)};
        for(const auto& point:component)initial.push_back(point+offset);
        const std::vector<int> faces=copy%2?std::vector<int>{0,1,2,3,4,5}
            :std::vector<int>{1,2,3};
        for(int node:faces)triangles.push_back(first+node);
    }
    Scene scene(initial,triangles);
    scene.params.d_hat=.3;scene.params.k_barrier=30;scene.params.damping=.2;
    std::vector<Vec3> history=initial;
    for(std::size_t node=0;node<initial.size();++node) {
        history[node].x()-=.02+.001*(node%3);
        scene.xhat[node]+=Vec3(.004*(node%3),-.003,.002);
    }
    const auto previous=history;
    std::vector<AABB> boxes;
    for(const auto& point:initial)
        boxes.emplace_back(point-Vec3::Constant(2),point+Vec3::Constant(2));
    BroadPhase phase;
    phase.initialize(boxes,scene.mesh,0,BroadPhase::InitializationMode::DeformableSolver);
    ASSERT_GT(phase.cache().nt_pairs.size(),128u);
    ASSERT_GT(phase.cache().ss_pairs.size(),128u);
    for(double friction:{0.,.4}) {
        SCOPED_TRACE(friction);
        scene.params.friction_coefficient=friction;
        std::vector<Vec3> serial_trace;
        for(int threads:{1,2,4,8}) {
            SCOPED_TRACE(threads);
            omp_set_num_threads(threads);
            scene.params.use_parallel=threads>1;
            DATColoredStepWorkspace full,selected;
            full.prepare(scene.mesh,scene.adj,scene.pins,scene.params,initial.size());
            selected.prepare(scene.mesh,scene.adj,scene.pins,scene.params,initial.size());
            ASSERT_EQ(full.colors(),selected.colors());
            ASSERT_FALSE(full.colors().empty());
            // The point and a triangle corner share an elastic color, so a
            // selected pair may have multiple roles and must be deduplicated.
            EXPECT_NE(std::find(full.colors()[0].begin(),full.colors()[0].end(),0),full.colors()[0].end());
            EXPECT_NE(std::find(full.colors()[0].begin(),full.colors()[0].end(),1),full.colors()[0].end());
            auto positions=initial;
            std::vector<Vec3> trace;
            for(int sweep=0;sweep<3;++sweep)for(const auto& color:selected.colors()) {
                full.freeze_contacts(scene.params,positions,&history,phase);
                selected.freeze_contacts(scene.params,positions,&history,phase,&color);
                std::vector<Vec3> proposals;
                for(int node:color) {
                    const auto reference=full.proposal(node,scene.mesh,scene.adj,scene.pins,
                        scene.params,positions,scene.xhat,&history);
                    const auto actual=selected.proposal(node,scene.mesh,scene.adj,scene.pins,
                        scene.params,positions,scene.xhat,&history);
                    ASSERT_TRUE(reference.allFinite());ASSERT_TRUE(actual.allFinite());
                    EXPECT_EQ(std::memcmp(actual.data(),reference.data(),3*sizeof(double)),0)<<node;
                    proposals.push_back(actual);trace.push_back(actual);
                }
                // Publish only after all same-color proposals have read their
                // immutable snapshot. Small steps keep this fixture separated.
                for(std::size_t index=0;index<color.size();++index) {
                    const double length=proposals[index].norm();
                    positions[color[index]]+=proposals[index]*std::min(1.,.001/std::max(length,1e-30));
                }
            }
            if(threads==1)serial_trace=trace;
            ASSERT_EQ(trace.size(),serial_trace.size());
            for(std::size_t index=0;index<trace.size();++index)
                EXPECT_EQ(std::memcmp(trace[index].data(),serial_trace[index].data(),3*sizeof(double)),0)<<index;
            expect_positions(history,previous,0);
        }
    }
}

TEST(ColoredDATContacts, SelectedSnapshotsClearDisappearingContactsAndRejectUnselectedParticles) {
    RefMesh mesh;DeformedState state;
    state.deformed_positions={Vec3(0,0,.1)};
    state.velocities={Vec3::Zero()};mesh.mass={1};mesh.node_to_rb={-1};
    create_rigid_body({Vec3(-1,-1,0),Vec3(2,-1,0),Vec3(-1,2,0)},
        Vec3::Zero(),Vec4(1,0,0,0),Vec3::Zero(),3,mesh,state,RigidBodyUpdateMode::None);
    mesh.tris={1,2,3};mesh.initialize(state.deformed_positions);
    const auto adjacency=build_incident_triangle_map(mesh.tris);
    const std::vector<Pin> pins;
    auto params=SimParams::zeros();
    params.fps=20;params.substeps=1;params.d_hat=.3;params.k_barrier=30;
    params.friction_coefficient=.4;params.damping=.2;
    auto history=state.deformed_positions;history[0].x()-=.02;
    const auto target=state.deformed_positions;
    const auto make_phase=[&](const std::vector<Vec3>& positions,double radius) {
        std::vector<AABB> boxes;
        for(const auto& point:positions)
            boxes.emplace_back(point-Vec3::Constant(radius),point+Vec3::Constant(radius));
        BroadPhase phase;
        phase.initialize(boxes,mesh,0,BroadPhase::InitializationMode::GeneralSolver);
        return phase;
    };
    auto candidates=make_phase(target,2);
    ASSERT_FALSE(candidates.cache().nt_pairs.empty());
    DATColoredStepWorkspace full,selected;
    full.prepare(mesh,adjacency,pins,params,target.size(),true);
    selected.prepare(mesh,adjacency,pins,params,target.size(),true);
    const std::vector<int> particle_and_body{0,0,int(target.size())};
    const std::vector<int> body_only{int(target.size())};
    const auto proposal=[&](DATColoredStepWorkspace& workspace,const std::vector<Vec3>& positions) {
        return workspace.proposal(0,mesh,adjacency,pins,params,positions,target,&history);
    };
    auto far=target;far[0].z()=2;
    auto no_candidates=make_phase(far,.01);
    ASSERT_TRUE(no_candidates.cache().nt_pairs.empty());
    const std::array<const std::vector<Vec3>*,5> snapshots{{&target,&far,&target,&far,&target}};
    for(std::size_t step=0;step<snapshots.size();++step) {
        const auto& positions=*snapshots[step];
        const auto& phase=step==3?no_candidates:candidates;
        full.freeze_contacts(params,positions,&history,phase);
        selected.freeze_contacts(params,positions,&history,phase,&particle_and_body);
        const auto expected=proposal(full,positions),actual=proposal(selected,positions);
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),3*sizeof(double)),0);
        selected.freeze_contacts(params,positions,&history,phase,&body_only);
        EXPECT_THROW((void)proposal(selected,positions),std::invalid_argument);
    }
    // An empty selection never exposes contact blocks left by a prior color.
    const std::vector<int> empty;
    selected.freeze_contacts(params,target,&history,candidates,&empty);
    EXPECT_THROW((void)proposal(selected,target),std::invalid_argument);
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
    const auto expected = composed_sweeps(s, 1);
    BroadPhase phase;
    solve(s, phase);
    expect_positions(s.x, expected);
    const Vec3 normal = (s.x[2] - s.x[1]).cross(s.x[3] - s.x[1]);
    EXPECT_GT(normal.dot(s.x[0] - s.x[1]), 0.0);
    EXPECT_LT(s.x[0].z(), 0.2);
    for (int node = 1; node < 4; ++node) {
        EXPECT_GT(s.x[node].z(), 0.0);
    }
}

TEST(JacobiDATSolver, ObtuseTriangleBarrierUsesTheCorrectedSharedDistance) {
    for (double d_hat : {.05, .2}) {
        Scene s({Vec3(.5, -.01, .1), Vec3::Zero(), Vec3(1, 0, 0), Vec3(-10, .1, 0)}, {1, 2, 3});
        const auto initial = s.x;
        s.params.d_hat = d_hat;
        s.params.k_barrier = .1;
        BroadPhase phase;
        solve(s, phase);
        // sqrt(.0101) activates the barrier at .2; the old distance .51 did not.
        if (d_hat == .2) EXPECT_GT(Vec3(0, -.01, .1).dot(s.x[0] - initial[0]), 0.0);
        else expect_positions(s.x, initial, 0.0);
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
        s.params.use_dat = true;
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

TEST(JacobiDATSolver, UnifiedEntryRejectsIncompleteRigidStateAndDriverAcceptsValidState) {
    Scene s;
    DeformedState state;
    create_rigid_body({Vec3::Zero(), Vec3::UnitX(), Vec3::UnitY(), Vec3::UnitZ()},
        Vec3::Zero(), Vec4(1, 0, 0, 0), Vec3::Zero(), 4.0, s.mesh, state);
    s.x = state.deformed_positions;
    s.xhat = s.x;
    s.velocity = state.velocities;
    s.params.use_dat = true;
    const auto original = s.x;
    BroadPhase phase;
    EXPECT_THROW(solve(s, phase), std::invalid_argument);
    expect_positions(s.x, original, 0.0);
    EXPECT_TRUE(advance_one_frame_general(state, s.mesh, s.adj, s.pins,
        s.params, phase).converged);
    expect_positions(state.deformed_positions, original, 0.0);
}

TEST(JacobiDATSolver, SharedQueryRejectsOverflowBeforeBuildingTheBVH) {
    Scene s({Vec3(std::numeric_limits<double>::max(),0,0)});
    s.params.dat_query_radius=1e308;s.params.dat_node_box_padding=1;
    s.params.damping=1;s.params.use_parallel=true;
    const auto initial=s.x;
    BroadPhase phase;
    EXPECT_THROW(solve(s,phase),std::invalid_argument);
    expect_positions(s.x,initial,0);
}
