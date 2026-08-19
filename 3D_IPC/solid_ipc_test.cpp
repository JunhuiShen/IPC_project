#include "solid_ipc.h"

#include "barrier_energy.h"
#include "broad_phase.h"
#include "mesh.h"
#include "mesh_utils.h"
#include "make_shape.h"
#include "physics.h"
#include "rigid_body_ipc.h"
#include "simulation.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

std::vector<Vec3> unit_tet_positions() {
    return {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    };
}

std::vector<Vec3> subdivided_tet_positions() {
    std::vector<Vec3> positions = unit_tet_positions();
    positions.push_back(Vec3(0.2, 0.2, 0.2));
    return positions;
}

std::vector<int> subdivided_tets() {
    return {
        4, 1, 2, 3,
        0, 4, 2, 3,
        0, 1, 4, 3,
        0, 1, 2, 4,
    };
}

std::vector<Vec3> translated(
    const std::vector<Vec3>& positions,
    const Vec3& translation) {
    std::vector<Vec3> result = positions;
    for (Vec3& position : result)
        position += translation;
    return result;
}

void expect_positions_equal(
    const std::vector<Vec3>& actual,
    const std::vector<Vec3>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(actual[i].isApprox(expected[i], 0.0))
            << "position " << i;
    }
}

void expect_zero_velocities(const std::vector<Vec3>& velocities) {
    for (std::size_t i = 0; i < velocities.size(); ++i) {
        EXPECT_TRUE(velocities[i].isZero(0.0)) << "velocity " << i;
    }
}

double mass_normalized_max_component(
    const RefMesh& ref_mesh,
    const std::vector<Vec3>& gradients) {
    double residual = 0.0;
    for (const int node : ref_mesh.tet_nodes) {
        Vec3 normalized = gradients[static_cast<std::size_t>(node)];
        const double mass = ref_mesh.mass[static_cast<std::size_t>(node)];
        if (mass > 0.0)
            normalized /= mass;
        residual = std::max(
            residual, normalized.cwiseAbs().maxCoeff());
    }
    return residual;
}

} // namespace

TEST(CreateSolid, AppendsSingleTetAndInitializesFoundation) {
    const std::vector<Vec3> positions = unit_tet_positions();
    RefMesh ref_mesh;
    DeformedState state;

    create_solid(positions, {0, 1, 2, 3}, 24.0, ref_mesh, state);

    expect_positions_equal(state.deformed_positions, positions);
    ASSERT_EQ(state.velocities.size(), positions.size());
    expect_zero_velocities(state.velocities);
    EXPECT_EQ(ref_mesh.num_positions, positions.size());
    EXPECT_EQ(ref_mesh.tets, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(ref_mesh.tet_nodes, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(ref_mesh.tris, (std::vector<int>{
        1, 2, 3,
        0, 3, 2,
        0, 1, 3,
        0, 2, 1,
    }));
    EXPECT_EQ(ref_mesh.surface_nodes, (std::vector<int>{1, 2, 3, 0}));

    ASSERT_EQ(ref_mesh.tet_rest_data.size(), 1U);
    const TetRestData& rest = ref_mesh.tet_rest_data[0];
    EXPECT_DOUBLE_EQ(rest.measure, 1.0 / 6.0);
    EXPECT_TRUE(rest.Dm_inverse.isApprox(Mat33::Identity(), 0.0));
    EXPECT_TRUE(
        rest.grad_N[0].isApprox(Vec3(-1.0, -1.0, -1.0), 0.0));
    EXPECT_TRUE(rest.grad_N[1].isApprox(Vec3::UnitX(), 0.0));
    EXPECT_TRUE(rest.grad_N[2].isApprox(Vec3::UnitY(), 0.0));
    EXPECT_TRUE(rest.grad_N[3].isApprox(Vec3::UnitZ(), 0.0));

    ASSERT_EQ(ref_mesh.tet_adj.size(), positions.size());
    for (int node = 0; node < 4; ++node) {
        ASSERT_EQ(ref_mesh.tet_adj[node].size(), 1U);
        EXPECT_EQ(ref_mesh.tet_adj[node][0], std::make_pair(0, node));
    }

    // A tet boundary is a collision/render surface, not a cloth element.
    EXPECT_TRUE(ref_mesh.Dm_inverse.empty());
    EXPECT_TRUE(ref_mesh.area.empty());
    EXPECT_TRUE(ref_mesh.hinges.empty());
    EXPECT_TRUE(ref_mesh.hinge_adj.empty());
    EXPECT_EQ(ref_mesh.mass, (std::vector<double>{1.0, 1.0, 1.0, 1.0}));
}

TEST(CreateSolid, AppendsAfterClothAndPreservesShellRestPrefix) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> material_positions;

    build_square_mesh(
        ref_mesh, state, material_positions,
        1, 1, 1.0, 1.0, Vec3::Zero());

    const std::vector<int> cloth_tris = ref_mesh.tris;
    const std::vector<Mat22> cloth_dm_inverse = ref_mesh.Dm_inverse;
    const std::vector<double> cloth_area = ref_mesh.area;
    const std::vector<Hinge> cloth_hinges = ref_mesh.hinges;

    EXPECT_NO_THROW(create_solid(
        translated(unit_tet_positions(), Vec3(0.0, 2.0, 0.0)),
        {0, 1, 2, 3}, 24.0, ref_mesh, state));

    ASSERT_GE(ref_mesh.tris.size(), cloth_tris.size());
    EXPECT_TRUE(std::equal(
        cloth_tris.begin(), cloth_tris.end(), ref_mesh.tris.begin()));
    ASSERT_EQ(ref_mesh.Dm_inverse.size(), cloth_dm_inverse.size());
    ASSERT_EQ(ref_mesh.area.size(), cloth_area.size());
    for (std::size_t triangle = 0;
         triangle < cloth_dm_inverse.size(); ++triangle) {
        EXPECT_TRUE(ref_mesh.Dm_inverse[triangle].isApprox(
            cloth_dm_inverse[triangle], 0.0));
        EXPECT_DOUBLE_EQ(ref_mesh.area[triangle], cloth_area[triangle]);
    }
    ASSERT_EQ(ref_mesh.hinges.size(), cloth_hinges.size());
    for (std::size_t hinge = 0; hinge < cloth_hinges.size(); ++hinge) {
        for (int local = 0; local < 4; ++local) {
            EXPECT_EQ(
                ref_mesh.hinges[hinge].v[local],
                cloth_hinges[hinge].v[local]);
        }
        EXPECT_DOUBLE_EQ(
            ref_mesh.hinges[hinge].bar_theta,
            cloth_hinges[hinge].bar_theta);
        EXPECT_DOUBLE_EQ(
            ref_mesh.hinges[hinge].c_e,
            cloth_hinges[hinge].c_e);
    }
    EXPECT_EQ(ref_mesh.num_positions, 8U);
    EXPECT_EQ(ref_mesh.tet_nodes, (std::vector<int>{4, 5, 6, 7}));
}

TEST(MixedLumpedMass, RebuildsOnlyClothAndPreservesSolidAndRigidMasses) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> material_positions;

    build_square_mesh(
        ref_mesh, state, material_positions,
        1, 1, 1.0, 1.0, Vec3::Zero());
    create_solid(
        translated(unit_tet_positions(), Vec3(0.0, 2.0, 0.0)),
        {0, 1, 2, 3}, 24.0, ref_mesh, state);

    const int rigid_base = static_cast<int>(state.deformed_positions.size());
    ref_mesh.tris.insert(
        ref_mesh.tris.end(),
        {rigid_base, rigid_base + 1, rigid_base + 2});
    create_rigid_body(
        {Vec3(3.0, 0.0, 0.0),
         Vec3(4.0, 0.0, 0.0),
         Vec3(3.0, 1.0, 0.0)},
        Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(),
        6.0, ref_mesh, state);

    const std::vector<double> masses_before = ref_mesh.mass;
    ASSERT_EQ(masses_before.size(), 11U);
    for (int cloth_node = 0; cloth_node < 4; ++cloth_node)
        ref_mesh.mass[static_cast<std::size_t>(cloth_node)] = 123.0;

    EXPECT_NO_THROW(ref_mesh.build_deformable_lumped_mass(12.0, 0.25));

    double cloth_mass = 0.0;
    for (int cloth_node = 0; cloth_node < 4; ++cloth_node) {
        EXPECT_GT(ref_mesh.mass[static_cast<std::size_t>(cloth_node)], 0.0);
        cloth_mass += ref_mesh.mass[static_cast<std::size_t>(cloth_node)];
    }
    EXPECT_NEAR(cloth_mass, 3.0, 1.0e-12);
    for (int solid_node = 4; solid_node < 8; ++solid_node) {
        EXPECT_DOUBLE_EQ(
            ref_mesh.mass[static_cast<std::size_t>(solid_node)],
            masses_before[static_cast<std::size_t>(solid_node)]);
    }
    for (int rigid_node = 8; rigid_node < 11; ++rigid_node) {
        EXPECT_DOUBLE_EQ(
            ref_mesh.mass[static_cast<std::size_t>(rigid_node)],
            masses_before[static_cast<std::size_t>(rigid_node)]);
    }
}

TEST(CreateSolid, AppendsDisconnectedSolidsAndRemapsLocalTopology) {
    const std::vector<Vec3> first_positions = unit_tet_positions();
    const std::vector<Vec3> second_positions = translated(
        subdivided_tet_positions(), Vec3(3.0, 0.0, 0.0));
    RefMesh ref_mesh;
    DeformedState state;

    create_solid(
        first_positions, {0, 1, 2, 3}, 1.0, ref_mesh, state);
    state.velocities[0] = Vec3(1.0, 2.0, 3.0);
    create_solid(
        second_positions, subdivided_tets(), 1.0, ref_mesh, state);

    std::vector<Vec3> expected_positions = first_positions;
    expected_positions.insert(
        expected_positions.end(), second_positions.begin(),
        second_positions.end());
    expect_positions_equal(state.deformed_positions, expected_positions);
    ASSERT_EQ(state.velocities.size(), expected_positions.size());
    EXPECT_TRUE(state.velocities[0].isApprox(Vec3(1.0, 2.0, 3.0), 0.0));
    for (std::size_t node = first_positions.size();
         node < state.velocities.size(); ++node) {
        EXPECT_TRUE(state.velocities[node].isZero(0.0))
            << "new velocity " << node;
    }

    const std::vector<int> expected_tets = {
        0, 1, 2, 3,
        8, 5, 6, 7,
        4, 8, 6, 7,
        4, 5, 8, 7,
        4, 5, 6, 8,
    };
    EXPECT_EQ(ref_mesh.num_positions, expected_positions.size());
    EXPECT_EQ(ref_mesh.tets, expected_tets);
    EXPECT_EQ(
        ref_mesh.tet_nodes,
        (std::vector<int>{0, 1, 2, 3, 8, 5, 6, 7, 4}));

    // Each solid's TGSL boundary is remapped and appended without reordering
    // collision/render triangles that were already registered.
    const std::vector<int> expected_tris = {
        1, 2, 3,
        0, 3, 2,
        0, 1, 3,
        0, 2, 1,
        5, 6, 7,
        4, 7, 6,
        4, 5, 7,
        4, 6, 5,
    };
    EXPECT_EQ(ref_mesh.tris, expected_tris);
    EXPECT_EQ(
        ref_mesh.surface_nodes,
        (std::vector<int>{1, 2, 3, 0, 5, 6, 7, 4}));

    // Local node 4 of the second solid is global node 8. It is volumetric,
    // incident on all four subdivided tets, but it is not on the boundary.
    EXPECT_EQ(std::count(ref_mesh.tet_nodes.begin(),
                         ref_mesh.tet_nodes.end(), 8),
              1);
    EXPECT_EQ(std::count(ref_mesh.surface_nodes.begin(),
                         ref_mesh.surface_nodes.end(), 8),
              0);
    ASSERT_EQ(ref_mesh.tet_adj.size(), expected_positions.size());
    EXPECT_EQ(ref_mesh.tet_adj[8].size(), 4U);
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), 5U);
}

TEST(CreateSolid, InvalidLocalTetIsTransactional) {
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        unit_tet_positions(), {0, 1, 2, 3}, 1.0, ref_mesh, state);

    const std::vector<Vec3> old_positions = state.deformed_positions;
    const std::vector<Vec3> old_velocities = state.velocities;
    const std::vector<int> old_tets = ref_mesh.tets;
    const std::vector<int> old_tris = ref_mesh.tris;
    const std::vector<int> old_tet_nodes = ref_mesh.tet_nodes;
    const std::vector<int> old_surface_nodes = ref_mesh.surface_nodes;
    const std::vector<double> old_mass = ref_mesh.mass;
    const auto old_tet_adj = ref_mesh.tet_adj;
    const std::size_t old_rest_data_size = ref_mesh.tet_rest_data.size();
    const std::size_t old_num_positions = ref_mesh.num_positions;

    EXPECT_THROW(
        create_solid(
            unit_tet_positions(), {0, 2, 1, 3}, 1.0, ref_mesh, state),
        std::invalid_argument);
    EXPECT_THROW(
        create_solid(
            unit_tet_positions(), {0, 1, 2, 3}, -1.0,
            ref_mesh, state),
        std::invalid_argument);
    EXPECT_THROW(
        create_solid(
            unit_tet_positions(), {0, 1, 2, 3},
            std::numeric_limits<double>::quiet_NaN(), ref_mesh, state),
        std::invalid_argument);

    expect_positions_equal(state.deformed_positions, old_positions);
    expect_positions_equal(state.velocities, old_velocities);
    EXPECT_EQ(ref_mesh.tets, old_tets);
    EXPECT_EQ(ref_mesh.tris, old_tris);
    EXPECT_EQ(ref_mesh.tet_nodes, old_tet_nodes);
    EXPECT_EQ(ref_mesh.surface_nodes, old_surface_nodes);
    EXPECT_EQ(ref_mesh.mass, old_mass);
    EXPECT_EQ(ref_mesh.tet_adj, old_tet_adj);
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), old_rest_data_size);
    EXPECT_EQ(ref_mesh.num_positions, old_num_positions);
}

TEST(CreateSolid, RigidAppendPreservesSolidIncidenceAndAppendOrder) {
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        unit_tet_positions(), {0, 1, 2, 3}, 24.0, ref_mesh, state);

    const int rb = create_rigid_body(
        {Vec3(3.0, 0.0, 0.0)},
        Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(), 1.0,
        ref_mesh, state);

    ASSERT_EQ(rb, 0);
    ASSERT_EQ(ref_mesh.tet_adj.size(), 5U);
    EXPECT_TRUE(ref_mesh.tet_adj[4].empty());
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), 1U);
    EXPECT_EQ(ref_mesh.tet_nodes, (std::vector<int>{0, 1, 2, 3}));

    create_solid(
        translated(unit_tet_positions(), Vec3(6.0, 0.0, 0.0)),
        {0, 1, 2, 3}, 48.0, ref_mesh, state);

    EXPECT_EQ(ref_mesh.num_positions, 9U);
    EXPECT_EQ(ref_mesh.tets, (std::vector<int>{
        0, 1, 2, 3,
        5, 6, 7, 8,
    }));
    ASSERT_EQ(ref_mesh.tet_adj.size(), 9U);
    EXPECT_TRUE(ref_mesh.tet_adj[4].empty());
    EXPECT_EQ(ref_mesh.node_to_rb[4], rb);
    EXPECT_EQ(ref_mesh.mass, (std::vector<double>{
        1.0, 1.0, 1.0, 1.0,
        1.0,
        2.0, 2.0, 2.0, 2.0,
    }));
}

TEST(CreateSolid, ClearModelClearsSolidFoundationAndState) {
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        subdivided_tet_positions(), subdivided_tets(), 1.0,
        ref_mesh, state);
    std::vector<Vec2> material_positions;
    std::vector<Pin> pins;

    clear_model(ref_mesh, state, material_positions, pins);

    EXPECT_TRUE(ref_mesh.tets.empty());
    EXPECT_TRUE(ref_mesh.tet_rest_data.empty());
    EXPECT_TRUE(ref_mesh.tet_adj.empty());
    EXPECT_TRUE(ref_mesh.tet_nodes.empty());
    EXPECT_TRUE(ref_mesh.surface_nodes.empty());
    EXPECT_TRUE(ref_mesh.tris.empty());
    EXPECT_TRUE(ref_mesh.mass.empty());
    EXPECT_EQ(ref_mesh.num_positions, 0U);
    EXPECT_TRUE(state.deformed_positions.empty());
    EXPECT_TRUE(state.velocities.empty());

    // A cleared RefMesh can immediately be reused for another solid.
    create_solid(
        unit_tet_positions(), {0, 1, 2, 3}, 1.0, ref_mesh, state);
    EXPECT_EQ(ref_mesh.num_positions, 4U);
    EXPECT_EQ(ref_mesh.tet_rest_data.size(), 1U);
}

TEST(SolidLumpedMass, SharedTetNodesAccumulateTgslContributions) {
    const std::vector<Vec3> positions = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 0.0, -1.0),
    };
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        positions,
        {0, 1, 2, 3, 0, 2, 1, 4},
        24.0, ref_mesh, state);

    const std::vector<double> expected = {2.0, 2.0, 2.0, 1.0, 1.0};
    ASSERT_EQ(ref_mesh.mass.size(), expected.size());
    for (std::size_t node = 0; node < expected.size(); ++node)
        EXPECT_NEAR(ref_mesh.mass[node], expected[node], 1.0e-12);
    EXPECT_NEAR(
        std::accumulate(ref_mesh.mass.begin(), ref_mesh.mass.end(), 0.0),
        8.0, 1.0e-12);
}

TEST(SolidLumpedMass, InteriorTetNodeReceivesVolumetricMass) {
    std::vector<Vec3> positions = unit_tet_positions();
    positions.push_back(Vec3(0.25, 0.25, 0.25));
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        positions, subdivided_tets(), 96.0, ref_mesh, state);

    const std::vector<double> expected = {3.0, 3.0, 3.0, 3.0, 4.0};
    ASSERT_EQ(ref_mesh.mass.size(), expected.size());
    for (std::size_t node = 0; node < expected.size(); ++node)
        EXPECT_NEAR(ref_mesh.mass[node], expected[node], 1.0e-12);
    EXPECT_EQ(
        std::count(ref_mesh.surface_nodes.begin(),
                   ref_mesh.surface_nodes.end(), 4),
        0);
}

TEST(SolidLumpedMass, ConstructsAcrossDisconnectedCreatedSolids) {
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        unit_tet_positions(), {0, 1, 2, 3}, 24.0, ref_mesh, state);

    std::vector<Vec3> large_positions = unit_tet_positions();
    for (Vec3& position : large_positions)
        position = 2.0 * position + Vec3(3.0, 0.0, 0.0);
    create_solid(
        large_positions, {0, 1, 2, 3}, 24.0, ref_mesh, state);

    const std::vector<double> expected = {
        1.0, 1.0, 1.0, 1.0,
        8.0, 8.0, 8.0, 8.0,
    };
    ASSERT_EQ(ref_mesh.mass.size(), expected.size());
    for (std::size_t node = 0; node < expected.size(); ++node)
        EXPECT_NEAR(ref_mesh.mass[node], expected[node], 1.0e-12);
    EXPECT_NEAR(
        std::accumulate(ref_mesh.mass.begin(), ref_mesh.mass.end(), 0.0),
        36.0, 1.0e-12);
}

TEST(SolidLumpedMass, NewSolidMassPreservesExistingSolidAndRigidMasses) {
    RefMesh ref_mesh;
    DeformedState state;
    create_rigid_body(
        {Vec3(-2.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0)},
        Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(), 10.0,
        ref_mesh, state);
    create_solid(
        translated(unit_tet_positions(), Vec3(2.0, 0.0, 0.0)),
        {0, 1, 2, 3}, 24.0, ref_mesh, state);
    EXPECT_DOUBLE_EQ(ref_mesh.mass[0], 5.0);
    EXPECT_DOUBLE_EQ(ref_mesh.mass[1], 5.0);
    for (int node = 2; node < 6; ++node)
        EXPECT_NEAR(ref_mesh.mass[node], 1.0, 1.0e-12);

    create_solid(
        translated(unit_tet_positions(), Vec3(5.0, 0.0, 0.0)),
        {0, 1, 2, 3}, 48.0, ref_mesh, state);

    EXPECT_DOUBLE_EQ(ref_mesh.mass[0], 5.0);
    EXPECT_DOUBLE_EQ(ref_mesh.mass[1], 5.0);
    for (int node = 2; node < 6; ++node)
        EXPECT_NEAR(ref_mesh.mass[node], 1.0, 1.0e-12);
    for (int node = 6; node < 10; ++node)
        EXPECT_NEAR(ref_mesh.mass[node], 2.0, 1.0e-12);
}

TEST(SolidIncrementalPotential, ExactInertiaGravityAndPinTerms) {
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        unit_tet_positions(), {0, 1, 2, 3}, 24.0, ref_mesh, state);

    std::vector<Vec3> x = unit_tet_positions();
    x[1] = Vec3(0.4, -0.3, 0.2);
    std::vector<Vec3> xhat = x;
    xhat[1] = Vec3(-0.1, 0.2, 0.05);
    const std::vector<Pin> pins = {
        Pin{1, Vec3(0.1, -0.1, 0.4)},
    };

    SimParams params = SimParams::zeros();
    params.fps = 2.0;
    params.substeps = 1;
    params.solid_mu = 0.0;
    params.solid_lambda = 0.0;
    params.gravity = Vec3(2.0, -4.0, 1.0);
    params.kpin = 8.0;

    EXPECT_NEAR(
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, pins, params, x, xhat),
        0.63125, 1.0e-13);

    const auto [gradient, block] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            1, ref_mesh, pins, params, x, xhat);
    EXPECT_TRUE(gradient.isApprox(Vec3(0.6, 0.1, -0.5), 1.0e-13));
    EXPECT_TRUE(block.isApprox(3.0 * Mat33::Identity(), 1.0e-13));
}

TEST(SolidIncrementalPotential,
     UsesSolidLameParametersIndependentlyOfClothParameters) {
    RefMesh ref_mesh;
    DeformedState state;
    const std::vector<Vec3> rest = unit_tet_positions();
    create_solid(rest, {0, 1, 2, 3}, 24.0, ref_mesh, state);

    std::vector<Vec3> x = rest;
    x[1].x() = 1.2;
    const std::vector<Vec3> xhat = x;

    SimParams params = SimParams::zeros();
    params.fps = 1.0;
    params.substeps = 1;
    params.mu = 1.0e6;
    params.lambda = 2.0e6;
    params.solid_mu = 2.3;
    params.solid_lambda = 5.7;

    const double baseline_energy =
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, {}, params, x, xhat);
    const auto [baseline_gradient, baseline_block] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            1, ref_mesh, {}, params, x, xhat);
    EXPECT_GT(baseline_energy, 0.0);

    params.mu = 7.0e8;
    params.lambda = 9.0e8;
    EXPECT_DOUBLE_EQ(
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, {}, params, x, xhat),
        baseline_energy);
    const auto [cloth_changed_gradient, cloth_changed_block] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            1, ref_mesh, {}, params, x, xhat);
    EXPECT_TRUE(cloth_changed_gradient.isApprox(
        baseline_gradient, 0.0));
    EXPECT_TRUE(cloth_changed_block.isApprox(baseline_block, 0.0));

    params.solid_mu *= 2.0;
    params.solid_lambda *= 3.0;
    EXPECT_NE(
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, {}, params, x, xhat),
        baseline_energy);
    const auto [solid_changed_gradient, solid_changed_block] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            1, ref_mesh, {}, params, x, xhat);
    EXPECT_FALSE(solid_changed_gradient.isApprox(
        baseline_gradient, 1.0e-14));
    EXPECT_FALSE(solid_changed_block.isApprox(
        baseline_block, 1.0e-14));
}

TEST(SolidIncrementalPotential,
     LocalAssemblyMatchesGlobalEnergyDifferenceOnMultipleTets) {
    const std::vector<Vec3> rest_positions = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 0.0, -1.0),
    };
    const std::vector<int> tets = {
        0, 1, 2, 3,
        0, 2, 1, 4,
    };
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(rest_positions, tets, 24.0, ref_mesh, state);

    std::vector<Vec3> x = rest_positions;
    x[0] += Vec3(0.03, -0.02, 0.01);
    x[1] += Vec3(0.08, 0.04, -0.03);
    x[2] += Vec3(-0.05, 0.06, 0.02);
    x[3] += Vec3(0.02, -0.04, 0.09);
    x[4] += Vec3(-0.03, 0.02, -0.08);

    std::vector<Vec3> xhat = x;
    for (std::size_t node = 0; node < xhat.size(); ++node) {
        const double index = static_cast<double>(node);
        xhat[node] += Vec3(
            0.01 * (index + 1.0),
            -0.007 * (index + 2.0),
            0.005 * (index + 3.0));
    }

    // The duplicate node-1 pins verify that local assembly accumulates every
    // pin term, matching the energy's loop over pins.
    const std::vector<Pin> pins = {
        Pin{1, x[1] + Vec3(0.10, -0.20, 0.05)},
        Pin{1, x[1] + Vec3(-0.03, 0.04, 0.02)},
        Pin{4, x[4] + Vec3(0.02, -0.01, 0.06)},
    };

    SimParams params = SimParams::zeros();
    params.fps = 4.0;
    params.substeps = 1;
    params.solid_mu = 2.3;
    params.solid_lambda = 5.7;
    params.gravity = Vec3(0.3, -1.1, 0.4);
    params.kpin = 3.4;
    const double dt2 = params.dt2();

    std::vector<Vec3> assembled_gradient(x.size(), Vec3::Zero());
    std::vector<Mat33> assembled_block(x.size(), Mat33::Zero());
    double assembled_energy = 0.0;
    for (const int node : ref_mesh.tet_nodes) {
        assembled_energy += 0.5 * ref_mesh.mass[node]
            * (x[node] - xhat[node]).squaredNorm();
        assembled_energy += dt2
            * (-ref_mesh.mass[node] * params.gravity.dot(x[node]));
        assembled_gradient[node] += ref_mesh.mass[node]
            * (x[node] - xhat[node]);
        assembled_gradient[node] +=
            dt2 * (-ref_mesh.mass[node] * params.gravity);
        assembled_block[node] += ref_mesh.mass[node] * Mat33::Identity();
    }
    for (const Pin& pin : pins) {
        const Vec3 displacement =
            x[pin.vertex_index] - pin.target_position;
        assembled_energy +=
            dt2 * 0.5 * params.kpin * displacement.squaredNorm();
        assembled_gradient[pin.vertex_index] +=
            dt2 * params.kpin * displacement;
        assembled_block[pin.vertex_index] +=
            dt2 * params.kpin * Mat33::Identity();
    }
    for (int element = 0; element < num_tets(ref_mesh); ++element) {
        const Mat33 F = ElementF(
            static_cast<std::size_t>(element), x,
            ref_mesh.tets, ref_mesh.tet_rest_data);
        CorotatedCache cache;
        cache.UpdateCache(F);
        assembled_energy += dt2 * EFEMElementInternalEnergy(
            cache, F, ref_mesh.tet_rest_data[element],
            params.solid_mu, params.solid_lambda);
        const std::array<Vec3, 4> element_gradient =
            EFEMElementEnergyGradient(
                cache, F, ref_mesh.tet_rest_data[element],
                params.solid_mu, params.solid_lambda);
        for (int local = 0; local < 4; ++local) {
            const int node = tet_vertex(ref_mesh, element, local);
            assembled_gradient[node] +=
                dt2 * element_gradient[static_cast<std::size_t>(local)];
            assembled_block[node] += dt2
                * PBGSElementNodeElasticityBlock(
                    cache, ref_mesh.tet_rest_data[element],
                    params.solid_mu, params.solid_lambda, local);
        }
    }

    EXPECT_NEAR(
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, pins, params, x, xhat),
        assembled_energy, 1.0e-13);

    constexpr double h = 1.0e-6;
    for (const int node : ref_mesh.tet_nodes) {
        const auto [local_gradient, local_block] =
            compute_solid_local_gradient_and_pbgs_block_no_barrier(
                node, ref_mesh, pins, params, x, xhat);
        EXPECT_TRUE(local_gradient.isApprox(
            assembled_gradient[node], 1.0e-12)) << "node=" << node;
        EXPECT_TRUE(local_block.isApprox(
            assembled_block[node], 1.0e-12)) << "node=" << node;

        for (int component = 0; component < 3; ++component) {
            std::vector<Vec3> plus = x;
            std::vector<Vec3> minus = x;
            plus[node][component] += h;
            minus[node][component] -= h;
            const double finite_difference =
                (compute_solid_incremental_potential_no_barrier(
                     ref_mesh, pins, params, plus, xhat)
                 - compute_solid_incremental_potential_no_barrier(
                     ref_mesh, pins, params, minus, xhat))
                / (2.0 * h);
            EXPECT_NEAR(
                local_gradient[component], finite_difference, 2.0e-8)
                << "node=" << node << " component=" << component;
        }
    }
}

TEST(SolidIncrementalPotential, PbgsBlockHasExactTgslRestFormula) {
    RefMesh ref_mesh;
    DeformedState state;
    const std::vector<Vec3> x = unit_tet_positions();
    create_solid(x, {0, 1, 2, 3}, 24.0, ref_mesh, state);
    const std::vector<Pin> pins = {Pin{1, x[1]}};

    SimParams params = SimParams::zeros();
    params.fps = 2.0;
    params.substeps = 1;
    params.solid_mu = 3.0;
    params.solid_lambda = 6.0;
    params.kpin = 8.0;

    const auto [node_one_gradient, node_one_block] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            1, ref_mesh, pins, params, x, x);
    EXPECT_TRUE(node_one_gradient.isZero(1.0e-12));
    Mat33 expected_node_one = Mat33::Zero();
    expected_node_one.diagonal() = Vec3(3.5, 3.25, 3.25);
    EXPECT_TRUE(node_one_block.isApprox(expected_node_one, 1.0e-13));

    const auto [node_zero_gradient, node_zero_block] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            0, ref_mesh, pins, params, x, x);
    EXPECT_TRUE(node_zero_gradient.isZero(1.0e-12));
    Mat33 expected_node_zero = Mat33::Constant(0.25);
    expected_node_zero.diagonal().setConstant(2.0);
    EXPECT_TRUE(node_zero_block.isApprox(expected_node_zero, 1.0e-13));
}

TEST(SolidIncrementalPotential, ExcludesRigidProxyInertia) {
    RefMesh ref_mesh;
    DeformedState state;
    create_rigid_body(
        {Vec3(-2.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0)},
        Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(), 10.0,
        ref_mesh, state);
    create_solid(
        translated(unit_tet_positions(), Vec3(2.0, 0.0, 0.0)),
        {0, 1, 2, 3}, 24.0, ref_mesh, state);

    const std::vector<Vec3> x = state.deformed_positions;
    std::vector<Vec3> xhat = x;
    xhat[0] += Vec3(10.0, -3.0, 2.0);
    xhat[1] += Vec3(-4.0, 5.0, 6.0);

    SimParams params = SimParams::zeros();
    params.fps = 2.0;
    params.substeps = 1;

    EXPECT_NEAR(
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, {}, params, x, xhat),
        0.0, 1.0e-14);

    const auto [gradient, block] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            2, ref_mesh, {}, params, x, xhat);
    EXPECT_TRUE(gradient.isZero(1.0e-14));
    EXPECT_TRUE(block.isApprox(Mat33::Identity(), 1.0e-14));
}

TEST(SolidResidual, MatchesIndependentlyAssembledMassNormalizedMaximum) {
    const std::vector<Vec3> rest_positions = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 0.0, -1.0),
    };
    const std::vector<int> tets = {
        0, 1, 2, 3,
        0, 2, 1, 4,
    };
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(rest_positions, tets, 24.0, ref_mesh, state);

    std::vector<Vec3> x = rest_positions;
    x[0] += Vec3(0.03, -0.02, 0.01);
    x[1] += Vec3(0.08, 0.04, -0.03);
    x[2] += Vec3(-0.05, 0.06, 0.02);
    x[3] += Vec3(0.02, -0.04, 0.09);
    x[4] += Vec3(-0.03, 0.02, -0.08);

    std::vector<Vec3> xhat = x;
    for (std::size_t node = 0; node < xhat.size(); ++node) {
        const double i = static_cast<double>(node + 1);
        xhat[node] += Vec3(0.012 * i, -0.009 * i, 0.006 * i);
    }
    const std::vector<Pin> pins = {
        Pin{1, x[1] + Vec3(0.10, -0.20, 0.05)},
        Pin{4, x[4] + Vec3(-0.04, 0.03, 0.08)},
    };

    SimParams params = SimParams::zeros();
    params.fps = 4.0;
    params.substeps = 1;
    params.solid_mu = 2.3;
    params.solid_lambda = 5.7;
    params.gravity = Vec3(0.3, -1.1, 0.4);
    params.kpin = 3.4;
    const double dt2 = params.dt2();

    // Assemble by elements, independently of the residual's per-node
    // incidence traversal.
    std::vector<Vec3> gradients(x.size(), Vec3::Zero());
    for (const int node : ref_mesh.tet_nodes) {
        const std::size_t index = static_cast<std::size_t>(node);
        gradients[index] += ref_mesh.mass[index] * (x[index] - xhat[index]);
        gradients[index] += dt2 * (-ref_mesh.mass[index] * params.gravity);
    }
    for (const Pin& pin : pins) {
        const std::size_t node = static_cast<std::size_t>(pin.vertex_index);
        gradients[node] += dt2 * params.kpin
            * (x[node] - pin.target_position);
    }
    for (int element = 0; element < num_tets(ref_mesh); ++element) {
        const Mat33 F = ElementF(
            static_cast<std::size_t>(element), x,
            ref_mesh.tets, ref_mesh.tet_rest_data);
        CorotatedCache cache;
        cache.UpdateCache(F);
        const std::array<Vec3, 4> element_gradient =
            EFEMElementEnergyGradient(
                cache, F, ref_mesh.tet_rest_data[element],
                params.solid_mu, params.solid_lambda);
        for (int local = 0; local < 4; ++local) {
            const int node = tet_vertex(ref_mesh, element, local);
            gradients[static_cast<std::size_t>(node)] +=
                dt2 * element_gradient[static_cast<std::size_t>(local)];
        }
    }

    const double expected =
        mass_normalized_max_component(ref_mesh, gradients);
    BroadPhase broad_phase;
    const double actual = compute_global_solid_residual(
        ref_mesh, pins, params, x, xhat, broad_phase);

    ASSERT_GT(expected, 0.0);
    EXPECT_NEAR(actual, expected, 1.0e-13 * (1.0 + expected));
}

TEST(SolidResidual, RestStateHasZeroResidual) {
    RefMesh ref_mesh;
    DeformedState state;
    const std::vector<Vec3> x = unit_tet_positions();
    create_solid(x, {0, 1, 2, 3}, 24.0, ref_mesh, state);

    SimParams params = SimParams::zeros();
    params.solid_mu = 3.0;
    params.solid_lambda = 6.0;
    BroadPhase broad_phase;

    EXPECT_NEAR(
        compute_global_solid_residual(
            ref_mesh, {}, params, x, x, broad_phase),
        0.0, 1.0e-13);
}

TEST(SolidResidual, IncludesInteriorTetNodes) {
    RefMesh ref_mesh;
    DeformedState state;
    create_rigid_body(
        {Vec3(-3.0, 0.0, 0.0), Vec3(-2.0, 0.0, 0.0)},
        Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(), 10.0,
        ref_mesh, state);
    create_solid(
        subdivided_tet_positions(), subdivided_tets(), 24.0,
        ref_mesh, state);
    constexpr int interior_node = 6;
    ASSERT_NE(
        std::find(
            ref_mesh.tet_nodes.begin(), ref_mesh.tet_nodes.end(),
            interior_node),
        ref_mesh.tet_nodes.end());
    ASSERT_EQ(
        std::find(
            ref_mesh.surface_nodes.begin(), ref_mesh.surface_nodes.end(),
            interior_node),
        ref_mesh.surface_nodes.end());

    const std::vector<Vec3>& x = state.deformed_positions;
    std::vector<Vec3> xhat = x;
    // A much larger rigid-proxy displacement must be ignored. Only the
    // interior tet node contributes to the solid residual.
    xhat[0] += Vec3(100.0, -200.0, 300.0);
    xhat[1] += Vec3(-400.0, 500.0, -600.0);
    xhat[interior_node] += Vec3(3.0, -7.0, 5.0);
    SimParams params = SimParams::zeros();
    BroadPhase broad_phase;

    EXPECT_NEAR(
        compute_global_solid_residual(
            ref_mesh, {}, params, x, xhat, broad_phase),
        7.0, 1.0e-13);
}

TEST(SolidResidual, LeavesZeroMassGradientUnnormalized) {
    RefMesh ref_mesh;
    DeformedState state;
    const std::vector<Vec3> x = unit_tet_positions();
    create_solid(x, {0, 1, 2, 3}, 0.0, ref_mesh, state);

    SimParams params = SimParams::zeros();
    params.fps = 2.0;
    params.kpin = 8.0;
    const std::vector<Pin> pins = {
        Pin{2, x[2] + Vec3(1.0, -2.0, 3.0)},
    };
    BroadPhase broad_phase;

    // dt^2*kpin*(x-target) = (-2, 4, -6). Cloth residual semantics
    // leave this gradient untouched when the nodal mass is zero.
    EXPECT_NEAR(
        compute_global_solid_residual(
            ref_mesh, pins, params, x, x, broad_phase),
        6.0, 1.0e-13);
}

namespace {

struct SolidBarrierFixture {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec3> x;
    std::vector<Vec3> xhat;
    std::vector<Pin> pins;
    SimParams params = SimParams::zeros();
    BroadPhase broad_phase;
};

void initialize_solid_barrier_fixture(SolidBarrierFixture& fixture) {
    // Two positively oriented, skew tetrahedra separated by a nonzero gap.
    // Their facing boundary features are within d_hat, but the solids do not
    // intersect and d_hat remains below half the minimum surface-edge length.
    const std::vector<Vec3> lower = {
        Vec3(0.00, 0.00, 0.00),
        Vec3(0.20, 2.00, 0.10),
        Vec3(2.00, 0.20, 0.00),
        Vec3(0.20, 0.20, -1.60),
    };
    const std::vector<Vec3> upper = {
        Vec3(0.45, 0.42, 0.24),
        Vec3(2.05, 0.60, 0.38),
        Vec3(0.65, 2.02, 0.31),
        Vec3(0.65, 0.58, 1.70),
    };
    create_solid(
        lower, {0, 1, 2, 3}, 24.0, fixture.ref_mesh, fixture.state);
    create_solid(
        upper, {0, 1, 2, 3}, 18.0, fixture.ref_mesh, fixture.state);

    fixture.x = fixture.state.deformed_positions;
    fixture.xhat = fixture.x;
    for (std::size_t node = 0; node < fixture.xhat.size(); ++node) {
        const double i = static_cast<double>(node + 1);
        fixture.xhat[node] += Vec3(0.002 * i, -0.001 * i, 0.0015 * i);
    }
    fixture.pins = {
        Pin{1, fixture.x[1] + Vec3(0.03, -0.02, 0.01)},
        Pin{6, fixture.x[6] + Vec3(-0.02, 0.01, 0.025)},
    };

    fixture.params.fps = 5.0;
    fixture.params.substeps = 2;
    fixture.params.solid_mu = 1.7;
    fixture.params.solid_lambda = 4.2;
    fixture.params.gravity = Vec3(0.2, -1.3, 0.15);
    fixture.params.kpin = 3.6;
    fixture.params.d_hat = 0.4;
    fixture.params.k_barrier = 7.3;

    fixture.broad_phase.initialize_surface_nodes(
        fixture.x, std::vector<Vec3>(fixture.x.size(), Vec3::Zero()),
        fixture.ref_mesh, fixture.params.dt(), fixture.params.d_hat);
}

double raw_candidate_barrier_energy(
    const std::vector<Vec3>& x,
    const BroadPhase::Cache& cache,
    const double d_hat,
    int& active_nt,
    int& active_ss) {
    double energy = 0.0;
    active_nt = 0;
    active_ss = 0;
    for (const NodeTrianglePair& pair : cache.nt_pairs) {
        const double pair_energy = node_triangle_barrier(
            x[static_cast<std::size_t>(pair.node)],
            x[static_cast<std::size_t>(pair.tri_v[0])],
            x[static_cast<std::size_t>(pair.tri_v[1])],
            x[static_cast<std::size_t>(pair.tri_v[2])], d_hat);
        energy += pair_energy;
        active_nt += pair_energy > 0.0 ? 1 : 0;
    }
    for (const SegmentSegmentPair& pair : cache.ss_pairs) {
        const double pair_energy = segment_segment_barrier(
            x[static_cast<std::size_t>(pair.v[0])],
            x[static_cast<std::size_t>(pair.v[1])],
            x[static_cast<std::size_t>(pair.v[2])],
            x[static_cast<std::size_t>(pair.v[3])], d_hat);
        energy += pair_energy;
        active_ss += pair_energy > 0.0 ? 1 : 0;
    }
    return energy;
}

} // namespace

TEST(SolidResidual, SuppliedMasksMatchInternallyBuiltMasks) {
    SolidBarrierFixture fixture;
    initialize_solid_barrier_fixture(fixture);

    std::vector<unsigned char> solid_node_mask(fixture.x.size(), 0);
    for (const int node : fixture.ref_mesh.tet_nodes)
        solid_node_mask[static_cast<std::size_t>(node)] = 1;
    std::vector<unsigned char> surface_node_mask(fixture.x.size(), 0);
    for (const int node : fixture.ref_mesh.surface_nodes)
        surface_node_mask[static_cast<std::size_t>(node)] = 1;
    const PinMap pin_map = build_pin_map(fixture.pins, static_cast<int>(fixture.x.size()));

    const double internally_built = compute_global_solid_residual(fixture.ref_mesh, fixture.pins, fixture.params, fixture.x, fixture.xhat, fixture.broad_phase, &pin_map);
    const double supplied = compute_global_solid_residual(fixture.ref_mesh, fixture.pins, fixture.params, fixture.x, fixture.xhat, fixture.broad_phase, &pin_map, &solid_node_mask, &surface_node_mask);

    EXPECT_DOUBLE_EQ(supplied, internally_built);
}

TEST(SolidResidual, FrozenWorkspaceIsBitwiseEquivalentForTetAndActiveContactGradients) {
    SolidBarrierFixture fixture;
    initialize_solid_barrier_fixture(fixture);
    fixture.params.use_parallel = false;

    std::vector<unsigned char> solid_node_mask(fixture.x.size(), 0);
    for (const int node : fixture.ref_mesh.tet_nodes) solid_node_mask[static_cast<std::size_t>(node)] = 1;
    std::vector<unsigned char> surface_node_mask(fixture.x.size(), 0);
    for (const int node : fixture.ref_mesh.surface_nodes) surface_node_mask[static_cast<std::size_t>(node)] = 1;
    const PinMap pin_map = build_pin_map(fixture.pins, static_cast<int>(fixture.x.size()));

    FrozenResidualWorkspace workspace;
    build_frozen_residual_workspace(fixture.ref_mesh, fixture.params, fixture.x, fixture.broad_phase, workspace);
    ASSERT_EQ(workspace.tet_gradients.size(), 2u);
    ASSERT_EQ(workspace.nt_gradients.size(), fixture.broad_phase.cache().nt_pairs.size());
    ASSERT_EQ(workspace.ss_gradients.size(), fixture.broad_phase.cache().ss_pairs.size());
    ASSERT_EQ(workspace.nt_barrier_active.size(), fixture.broad_phase.cache().nt_pairs.size());
    ASSERT_EQ(workspace.ss_barrier_active.size(), fixture.broad_phase.cache().ss_pairs.size());
    ASSERT_EQ(workspace.nt_gradient_cached.size(), fixture.broad_phase.cache().nt_pairs.size());
    ASSERT_EQ(workspace.ss_gradient_cached.size(), fixture.broad_phase.cache().ss_pairs.size());
    EXPECT_GT(std::count(workspace.nt_aabb_active.begin(), workspace.nt_aabb_active.end(), static_cast<unsigned char>(1)), 0);
    EXPECT_GT(std::count(workspace.ss_aabb_active.begin(), workspace.ss_aabb_active.end(), static_cast<unsigned char>(1)), 0);
    int active_nt = 0;
    int active_ss = 0;
    raw_candidate_barrier_energy(fixture.x, fixture.broad_phase.cache(), fixture.params.d_hat, active_nt, active_ss);
    ASSERT_GT(active_nt, 0);
    ASSERT_GT(active_ss, 0);

    for (std::size_t pair_index = 0; pair_index < fixture.broad_phase.cache().nt_pairs.size(); ++pair_index) {
        const NodeTrianglePair& pair = fixture.broad_phase.cache().nt_pairs[pair_index];
        const double distance = node_triangle_distance(fixture.x[static_cast<std::size_t>(pair.node)], fixture.x[static_cast<std::size_t>(pair.tri_v[0])], fixture.x[static_cast<std::size_t>(pair.tri_v[1])], fixture.x[static_cast<std::size_t>(pair.tri_v[2])]).distance;
        EXPECT_EQ(workspace.nt_barrier_active[pair_index], distance < fixture.params.d_hat ? 1 : 0);
        EXPECT_EQ(workspace.nt_gradient_cached[pair_index], workspace.nt_aabb_active[pair_index] != 0 && distance != 0.0 ? 1 : 0);
    }
    for (std::size_t pair_index = 0; pair_index < fixture.broad_phase.cache().ss_pairs.size(); ++pair_index) {
        const SegmentSegmentPair& pair = fixture.broad_phase.cache().ss_pairs[pair_index];
        const double distance = segment_segment_distance(fixture.x[static_cast<std::size_t>(pair.v[0])], fixture.x[static_cast<std::size_t>(pair.v[1])], fixture.x[static_cast<std::size_t>(pair.v[2])], fixture.x[static_cast<std::size_t>(pair.v[3])]).distance;
        EXPECT_EQ(workspace.ss_barrier_active[pair_index], distance < fixture.params.d_hat ? 1 : 0);
        EXPECT_EQ(workspace.ss_gradient_cached[pair_index], workspace.ss_aabb_active[pair_index] != 0 && distance != 0.0 ? 1 : 0);
    }

    for (int element_index = 0; element_index < num_tets(fixture.ref_mesh); ++element_index) {
        const std::size_t element = static_cast<std::size_t>(element_index);
        const Mat33 F = ElementF(element, fixture.x, fixture.ref_mesh.tets, fixture.ref_mesh.tet_rest_data);
        CorotatedCache cache;
        cache.UpdateCache(F, CorotatedCacheMode::Lean);
        for (int role = 0; role < 4; ++role) {
            const Vec3 expected = EFEMElementNodeEnergyGradient(cache, F, fixture.ref_mesh.tet_rest_data[element], fixture.params.solid_mu, fixture.params.solid_lambda, role);
            for (int component = 0; component < 3; ++component) EXPECT_EQ(workspace.tet_gradients[element][static_cast<std::size_t>(role)][component], expected[component]);
        }
    }
    for (std::size_t pair_index = 0; pair_index < fixture.broad_phase.cache().nt_pairs.size(); ++pair_index) {
        if (workspace.nt_aabb_active[pair_index] == 0) continue;
        const NodeTrianglePair& pair = fixture.broad_phase.cache().nt_pairs[pair_index];
        for (int role = 0; role < 4; ++role) {
            const Vec3 expected = node_triangle_barrier_gradient(fixture.x[static_cast<std::size_t>(pair.node)], fixture.x[static_cast<std::size_t>(pair.tri_v[0])], fixture.x[static_cast<std::size_t>(pair.tri_v[1])], fixture.x[static_cast<std::size_t>(pair.tri_v[2])], fixture.params.d_hat, role);
            for (int component = 0; component < 3; ++component) EXPECT_EQ(workspace.nt_gradients[pair_index][static_cast<std::size_t>(role)][component], expected[component]);
        }
    }
    for (std::size_t pair_index = 0; pair_index < fixture.broad_phase.cache().ss_pairs.size(); ++pair_index) {
        if (workspace.ss_aabb_active[pair_index] == 0) continue;
        const SegmentSegmentPair& pair = fixture.broad_phase.cache().ss_pairs[pair_index];
        for (int role = 0; role < 4; ++role) {
            const Vec3 expected = segment_segment_barrier_gradient(fixture.x[static_cast<std::size_t>(pair.v[0])], fixture.x[static_cast<std::size_t>(pair.v[1])], fixture.x[static_cast<std::size_t>(pair.v[2])], fixture.x[static_cast<std::size_t>(pair.v[3])], fixture.params.d_hat, role);
            for (int component = 0; component < 3; ++component) EXPECT_EQ(workspace.ss_gradients[pair_index][static_cast<std::size_t>(role)][component], expected[component]);
        }
    }

    const double uncached = compute_global_solid_residual(fixture.ref_mesh, fixture.pins, fixture.params, fixture.x, fixture.xhat, fixture.broad_phase, &pin_map, &solid_node_mask, &surface_node_mask);
    const double cached = compute_global_solid_residual(fixture.ref_mesh, fixture.pins, fixture.params, fixture.x, fixture.xhat, fixture.broad_phase, &pin_map, &solid_node_mask, &surface_node_mask, &workspace);
    EXPECT_EQ(cached, uncached);
}

TEST(SolidBarrierAssembly, CountsEachCandidateOnceAndAppliesDtSquaredScale) {
    SolidBarrierFixture fixture;
    initialize_solid_barrier_fixture(fixture);

    int active_nt = 0;
    int active_ss = 0;
    const double raw_energy = raw_candidate_barrier_energy(
        fixture.x, fixture.broad_phase.cache(), fixture.params.d_hat,
        active_nt, active_ss);
    ASSERT_GT(active_nt, 0);
    ASSERT_GT(active_ss, 0);

    const double expected = fixture.params.dt2()
        * fixture.params.k_barrier * raw_energy;
    const double actual = compute_solid_barrier_incremental_potential(
        fixture.ref_mesh, fixture.params, fixture.x, fixture.broad_phase);
    EXPECT_NEAR(actual, expected, 1.0e-13 * (1.0 + std::abs(expected)));
}

TEST(SolidBarrierAssembly,
     LocalGradientAndExactSelfHessianMatchCenteredDifferences) {
    SolidBarrierFixture fixture;
    initialize_solid_barrier_fixture(fixture);

    // The cache stays fixed for every perturbation. This differentiates one
    // smooth candidate energy rather than differentiating broad-phase rebuilds.
    constexpr double gradient_h = 1.0e-6;
    constexpr double hessian_h = 1.0e-5;
    int tested_nodes = 0;
    for (const int node : fixture.ref_mesh.tet_nodes) {
        const auto [gradient, self_hessian] =
            compute_solid_local_barrier_gradient_and_self_hessian(
                node, fixture.ref_mesh, fixture.params,
                fixture.x, fixture.broad_phase);

        const auto& cache = fixture.broad_phase.cache();
        if (cache.vertex_nt[static_cast<std::size_t>(node)].empty()
            && cache.vertex_ss[static_cast<std::size_t>(node)].empty()) {
            continue;
        }
        ++tested_nodes;

        for (int component = 0; component < 3; ++component) {
            std::vector<Vec3> plus = fixture.x;
            std::vector<Vec3> minus = fixture.x;
            plus[static_cast<std::size_t>(node)][component] += gradient_h;
            minus[static_cast<std::size_t>(node)][component] -= gradient_h;
            const double finite_difference =
                (compute_solid_barrier_incremental_potential(
                     fixture.ref_mesh, fixture.params,
                     plus, fixture.broad_phase)
                 - compute_solid_barrier_incremental_potential(
                     fixture.ref_mesh, fixture.params,
                     minus, fixture.broad_phase))
                / (2.0 * gradient_h);
            const double tolerance =
                2.0e-7 * (1.0 + std::abs(finite_difference));
            EXPECT_NEAR(gradient[component], finite_difference, tolerance)
                << "node=" << node << " component=" << component;
        }

        for (int column = 0; column < 3; ++column) {
            std::vector<Vec3> plus = fixture.x;
            std::vector<Vec3> minus = fixture.x;
            plus[static_cast<std::size_t>(node)][column] += hessian_h;
            minus[static_cast<std::size_t>(node)][column] -= hessian_h;
            const Vec3 plus_gradient =
                compute_solid_local_barrier_gradient_and_self_hessian(
                    node, fixture.ref_mesh, fixture.params,
                    plus, fixture.broad_phase)
                    .first;
            const Vec3 minus_gradient =
                compute_solid_local_barrier_gradient_and_self_hessian(
                    node, fixture.ref_mesh, fixture.params,
                    minus, fixture.broad_phase)
                    .first;
            const Vec3 finite_difference =
                (plus_gradient - minus_gradient) / (2.0 * hessian_h);
            for (int row = 0; row < 3; ++row) {
                const double tolerance =
                    2.0e-4 * (1.0 + std::abs(finite_difference[row]));
                EXPECT_NEAR(
                    self_hessian(row, column), finite_difference[row],
                    tolerance)
                    << "node=" << node << " row=" << row
                    << " column=" << column;
            }
        }
    }
    EXPECT_EQ(tested_nodes, static_cast<int>(fixture.ref_mesh.tet_nodes.size()));
}

TEST(SolidBarrierAssembly, DisabledBarrierReturnsExactZero) {
    SolidBarrierFixture fixture;
    initialize_solid_barrier_fixture(fixture);

    SimParams disabled_by_distance = SimParams::zeros();
    disabled_by_distance.fps = 5.0;
    disabled_by_distance.substeps = 2;
    disabled_by_distance.d_hat = 0.0;
    disabled_by_distance.k_barrier = fixture.params.k_barrier;
    EXPECT_DOUBLE_EQ(
        compute_solid_barrier_incremental_potential(
            fixture.ref_mesh, disabled_by_distance,
            fixture.x, fixture.broad_phase),
        0.0);
    const auto [distance_gradient, distance_hessian] =
        compute_solid_local_barrier_gradient_and_self_hessian(
            0, fixture.ref_mesh, disabled_by_distance,
            fixture.x, fixture.broad_phase);
    EXPECT_TRUE(distance_gradient.isZero(0.0));
    EXPECT_TRUE(distance_hessian.isZero(0.0));

    SimParams disabled_by_stiffness = SimParams::zeros();
    disabled_by_stiffness.fps = 5.0;
    disabled_by_stiffness.substeps = 2;
    disabled_by_stiffness.d_hat = fixture.params.d_hat;
    disabled_by_stiffness.k_barrier = 0.0;
    EXPECT_DOUBLE_EQ(
        compute_solid_barrier_incremental_potential(
            fixture.ref_mesh, disabled_by_stiffness,
            fixture.x, fixture.broad_phase),
        0.0);
    const auto [stiffness_gradient, stiffness_hessian] =
        compute_solid_local_barrier_gradient_and_self_hessian(
            0, fixture.ref_mesh, disabled_by_stiffness,
            fixture.x, fixture.broad_phase);
    EXPECT_TRUE(stiffness_gradient.isZero(0.0));
    EXPECT_TRUE(stiffness_hessian.isZero(0.0));
}

TEST(SolidBarrierAssembly, SurfaceInitializationAndAssemblyExcludeInteriorNode) {
    RefMesh ref_mesh;
    DeformedState state;
    const std::vector<Vec3> x = subdivided_tet_positions();
    create_solid(x, subdivided_tets(), 24.0, ref_mesh, state);

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.d_hat = 0.25;
    params.k_barrier = 5.0;
    const std::vector<Vec3> zero_velocity(x.size(), Vec3::Zero());

    BroadPhase generic_broad_phase;
    generic_broad_phase.initialize(
        x, zero_velocity, ref_mesh, params.dt(), params.d_hat);
    const auto& generic_cache = generic_broad_phase.cache();
    ASSERT_FALSE(generic_cache.vertex_nt[4].empty());
    EXPECT_TRUE(std::any_of(
        generic_cache.nt_pairs.begin(), generic_cache.nt_pairs.end(),
        [](const NodeTrianglePair& pair) { return pair.node == 4; }));

    BroadPhase surface_broad_phase;
    surface_broad_phase.initialize_surface_nodes(
        x, zero_velocity, ref_mesh, params.dt(), params.d_hat);
    const auto& surface_cache = surface_broad_phase.cache();
    EXPECT_TRUE(surface_cache.vertex_nt[4].empty());
    EXPECT_TRUE(surface_cache.vertex_ss[4].empty());
    EXPECT_TRUE(std::none_of(
        surface_cache.nt_pairs.begin(), surface_cache.nt_pairs.end(),
        [](const NodeTrianglePair& pair) { return pair.node == 4; }));

    // The solid assembly also filters an interior point defensively when it is
    // handed a generic (all-particle-query) cache.
    const auto [generic_gradient, generic_hessian] =
        compute_solid_local_barrier_gradient_and_self_hessian(
            4, ref_mesh, params, x, generic_broad_phase);
    EXPECT_TRUE(generic_gradient.isZero(0.0));
    EXPECT_TRUE(generic_hessian.isZero(0.0));
    const auto [surface_gradient, surface_hessian] =
        compute_solid_local_barrier_gradient_and_self_hessian(
            4, ref_mesh, params, x, surface_broad_phase);
    EXPECT_TRUE(surface_gradient.isZero(0.0));
    EXPECT_TRUE(surface_hessian.isZero(0.0));
    EXPECT_DOUBLE_EQ(
        compute_solid_barrier_incremental_potential(
            ref_mesh, params, x, generic_broad_phase),
        compute_solid_barrier_incremental_potential(
            ref_mesh, params, x, surface_broad_phase));
}

TEST(SolidBarrierAssembly, FullWrappersEqualNoBarrierPlusBarrier) {
    SolidBarrierFixture fixture;
    initialize_solid_barrier_fixture(fixture);

    const double no_barrier = compute_solid_incremental_potential_no_barrier(
        fixture.ref_mesh, fixture.pins, fixture.params,
        fixture.x, fixture.xhat);
    const double barrier = compute_solid_barrier_incremental_potential(
        fixture.ref_mesh, fixture.params, fixture.x, fixture.broad_phase);
    const double full = compute_solid_incremental_potential(
        fixture.ref_mesh, fixture.pins, fixture.params,
        fixture.x, fixture.xhat, fixture.broad_phase);
    EXPECT_NEAR(full, no_barrier + barrier,
                1.0e-13 * (1.0 + std::abs(full)));

    std::vector<unsigned char> solid_node_mask(fixture.x.size(), 0);
    std::vector<unsigned char> surface_node_mask(fixture.x.size(), 0);
    for (const int node : fixture.ref_mesh.tet_nodes)
        solid_node_mask[static_cast<std::size_t>(node)] = 1;
    for (const int node : fixture.ref_mesh.surface_nodes)
        surface_node_mask[static_cast<std::size_t>(node)] = 1;
    PinMap pin_map(fixture.x.size(), -1);
    for (int pin = 0; pin < static_cast<int>(fixture.pins.size()); ++pin)
        pin_map[fixture.pins[static_cast<std::size_t>(pin)].vertex_index] = pin;

    for (const int node : fixture.ref_mesh.tet_nodes) {
        const auto [no_barrier_gradient, no_barrier_block] = compute_solid_local_gradient_and_pbgs_block_no_barrier(node, fixture.ref_mesh, fixture.pins, fixture.params, fixture.x, fixture.xhat);
        const auto [barrier_gradient, barrier_hessian] = compute_solid_local_barrier_gradient_and_self_hessian(node, fixture.ref_mesh, fixture.params, fixture.x, fixture.broad_phase);
        const auto [full_gradient, full_block] = compute_solid_local_gradient_and_block(node, fixture.ref_mesh, fixture.pins, fixture.params, fixture.x, fixture.xhat, fixture.broad_phase);
        const auto [precomputed_gradient, precomputed_block] = compute_solid_local_gradient_and_block(node, fixture.ref_mesh, fixture.pins, fixture.params, fixture.x, fixture.xhat, fixture.broad_phase, &solid_node_mask, &surface_node_mask, &pin_map);

        EXPECT_TRUE(full_gradient.isApprox(
            no_barrier_gradient + barrier_gradient, 1.0e-13))
            << "node=" << node;
        EXPECT_TRUE(full_block.isApprox(
            no_barrier_block + barrier_hessian, 1.0e-13))
            << "node=" << node;
        EXPECT_TRUE(precomputed_gradient.isApprox(full_gradient, 1.0e-13))
            << "node=" << node;
        EXPECT_TRUE(precomputed_block.isApprox(full_block, 1.0e-13))
            << "node=" << node;
    }
}

TEST(SolidResidual, IncludesBarrierGradient) {
    SolidBarrierFixture fixture;
    initialize_solid_barrier_fixture(fixture);

    SimParams params = fixture.params;
    params.solid_mu = 0.0;
    params.solid_lambda = 0.0;
    params.gravity = Vec3::Zero();
    params.kpin = 0.0;

    // Scatter pair gradients directly, independently of the solid local
    // gradient wrapper used by the production residual.
    std::vector<Vec3> gradients(fixture.x.size(), Vec3::Zero());
    const double barrier_scale = params.dt2() * params.k_barrier;
    const BroadPhase::Cache& cache = fixture.broad_phase.cache();
    ASSERT_FALSE(cache.nt_pairs.empty());
    ASSERT_FALSE(cache.ss_pairs.empty());
    for (const NodeTrianglePair& pair : cache.nt_pairs) {
        const int nodes[4] = {
            pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
        for (int dof = 0; dof < 4; ++dof) {
            gradients[static_cast<std::size_t>(nodes[dof])] +=
                barrier_scale * node_triangle_barrier_gradient(
                    fixture.x[static_cast<std::size_t>(pair.node)],
                    fixture.x[static_cast<std::size_t>(pair.tri_v[0])],
                    fixture.x[static_cast<std::size_t>(pair.tri_v[1])],
                    fixture.x[static_cast<std::size_t>(pair.tri_v[2])],
                    params.d_hat, dof);
        }
    }
    for (const SegmentSegmentPair& pair : cache.ss_pairs) {
        for (int dof = 0; dof < 4; ++dof) {
            gradients[static_cast<std::size_t>(pair.v[dof])] +=
                barrier_scale * segment_segment_barrier_gradient(
                    fixture.x[static_cast<std::size_t>(pair.v[0])],
                    fixture.x[static_cast<std::size_t>(pair.v[1])],
                    fixture.x[static_cast<std::size_t>(pair.v[2])],
                    fixture.x[static_cast<std::size_t>(pair.v[3])],
                    params.d_hat, dof);
        }
    }

    const double expected =
        mass_normalized_max_component(fixture.ref_mesh, gradients);
    const double actual = compute_global_solid_residual(
        fixture.ref_mesh, {}, params, fixture.x, fixture.x,
        fixture.broad_phase);
    ASSERT_GT(expected, 0.0);
    EXPECT_NEAR(actual, expected, 1.0e-13 * (1.0 + expected));

    params.d_hat = 0.0;
    EXPECT_DOUBLE_EQ(
        compute_global_solid_residual(
            fixture.ref_mesh, {}, params, fixture.x, fixture.x,
            fixture.broad_phase),
        0.0);
}

TEST(SolidSdfAssembly,
     SurfaceEnergyGradientAndHessianMatchCenteredDifferences) {
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        unit_tet_positions(), {0, 1, 2, 3}, 0.0, ref_mesh, state);

    std::vector<Vec3> x = state.deformed_positions;
    x[0] = Vec3(0.0, -0.15, 0.0);
    x[1] = Vec3(1.0, 0.80, 0.0);
    x[2] = Vec3(0.0, 0.90, 0.0);
    x[3] = Vec3(0.0, 0.80, 1.0);
    const std::vector<Vec3> xhat = x;

    SimParams params = SimParams::zeros();
    params.fps = 4.0;
    params.substeps = 1;
    params.k_sdf = 7.5;
    params.eps_sdf = 0.4;
    params.sdf_planes.push_back({Vec3::Zero(), Vec3::UnitY()});

    const SDFEvaluation node_sdf = evaluate_sdf(params.sdf_planes[0], x[0]);
    const double expected_energy = params.dt2()
        * sdf_penalty_energy(node_sdf, params.k_sdf, params.eps_sdf);
    EXPECT_NEAR(
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, {}, params, x, xhat),
        expected_energy, 1.0e-14);

    const auto [gradient, hessian] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            0, ref_mesh, {}, params, x, xhat);
    EXPECT_TRUE(gradient.isApprox(
        params.dt2() * sdf_penalty_gradient(
            node_sdf, params.k_sdf, params.eps_sdf),
        1.0e-14));
    EXPECT_TRUE(hessian.isApprox(
        params.dt2() * sdf_penalty_hessian(
            node_sdf, params.k_sdf, params.eps_sdf,
            /*include_curvature=*/false),
        1.0e-14));

    constexpr double gradient_h = 1.0e-6;
    constexpr double hessian_h = 1.0e-5;
    for (int component = 0; component < 3; ++component) {
        std::vector<Vec3> plus = x;
        std::vector<Vec3> minus = x;
        plus[0][component] += gradient_h;
        minus[0][component] -= gradient_h;
        const double finite_difference =
            (compute_solid_incremental_potential_no_barrier(
                 ref_mesh, {}, params, plus, xhat)
             - compute_solid_incremental_potential_no_barrier(
                 ref_mesh, {}, params, minus, xhat))
            / (2.0 * gradient_h);
        EXPECT_NEAR(gradient[component], finite_difference, 1.0e-8)
            << "component=" << component;
    }

    for (int column = 0; column < 3; ++column) {
        std::vector<Vec3> plus = x;
        std::vector<Vec3> minus = x;
        plus[0][column] += hessian_h;
        minus[0][column] -= hessian_h;
        const Vec3 plus_gradient =
            compute_solid_local_gradient_and_pbgs_block_no_barrier(
                0, ref_mesh, {}, params, plus, xhat)
                .first;
        const Vec3 minus_gradient =
            compute_solid_local_gradient_and_pbgs_block_no_barrier(
                0, ref_mesh, {}, params, minus, xhat)
                .first;
        const Vec3 finite_difference =
            (plus_gradient - minus_gradient) / (2.0 * hessian_h);
        for (int row = 0; row < 3; ++row) {
            EXPECT_NEAR(
                hessian(row, column), finite_difference[row], 1.0e-8)
                << "row=" << row << " column=" << column;
        }
    }
}

TEST(SolidSdfAssembly, AppliesOnlyToTetBoundaryNodes) {
    RefMesh ref_mesh;
    DeformedState state;
    const std::vector<Vec3> x = subdivided_tet_positions();
    create_solid(x, subdivided_tets(), 0.0, ref_mesh, state);

    SimParams params = SimParams::zeros();
    params.fps = 5.0;
    params.substeps = 2;
    params.k_sdf = 9.0;
    params.eps_sdf = 0.0;

    // Only interior node 4 lies inside this sphere. It must not contribute to
    // the solid SDF energy or local system.
    params.sdf_spheres.push_back(
        {x[4] + Vec3(0.05, 0.0, 0.0), 0.15});
    EXPECT_DOUBLE_EQ(
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, {}, params, x, x),
        0.0);
    const auto [interior_gradient, interior_hessian] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            4, ref_mesh, {}, params, x, x);
    EXPECT_TRUE(interior_gradient.isZero(0.0));
    EXPECT_TRUE(interior_hessian.isZero(0.0));

    // Moving the same obstacle to boundary node 0 produces a nonzero term,
    // while the interior node remains excluded.
    params.sdf_spheres[0] = {x[0] + Vec3(0.05, 0.0, 0.0), 0.15};
    EXPECT_GT(
        compute_solid_incremental_potential_no_barrier(
            ref_mesh, {}, params, x, x),
        0.0);
    const auto [surface_gradient, surface_hessian] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            0, ref_mesh, {}, params, x, x);
    EXPECT_GT(surface_gradient.norm(), 0.0);
    EXPECT_GT(surface_hessian.norm(), 0.0);
    const auto [excluded_gradient, excluded_hessian] =
        compute_solid_local_gradient_and_pbgs_block_no_barrier(
            4, ref_mesh, {}, params, x, x);
    EXPECT_TRUE(excluded_gradient.isZero(0.0));
    EXPECT_TRUE(excluded_hessian.isZero(0.0));
}

TEST(SolidResidual, IncludesSurfaceSdfGradientOnly) {
    RefMesh ref_mesh;
    DeformedState state;
    const std::vector<Vec3> x = subdivided_tet_positions();
    create_solid(x, subdivided_tets(), 24.0, ref_mesh, state);

    SimParams params = SimParams::zeros();
    params.fps = 5.0;
    params.substeps = 2;
    params.k_sdf = 9.0;
    params.eps_sdf = 0.0;
    params.sdf_spheres.push_back(
        {x[0] + Vec3(0.05, 0.0, 0.0), 0.15});

    const SDFEvaluation sdf = evaluate_sdf(params.sdf_spheres[0], x[0]);
    const Vec3 gradient = params.dt2()
        * sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
    const double expected = gradient.cwiseAbs().maxCoeff()
        / ref_mesh.mass[0];
    BroadPhase broad_phase;
    EXPECT_NEAR(
        compute_global_solid_residual(
            ref_mesh, {}, params, x, x, broad_phase),
        expected, 1.0e-13 * (1.0 + expected));

    params.sdf_spheres[0] =
        {x[4] + Vec3(0.05, 0.0, 0.0), 0.15};
    EXPECT_NEAR(
        compute_global_solid_residual(
            ref_mesh, {}, params, x, x, broad_phase),
        0.0, 1.0e-13);
}

TEST(SolidGeneralSolver, AdvancesPureSolidWithoutRigidBodies) {
    RefMesh ref_mesh;
    DeformedState state;
    create_solid(
        unit_tet_positions(), {0, 1, 2, 3}, 900.0,
        ref_mesh, state);
    ref_mesh.build_deformable_nodes();

    const std::vector<Vec3> positions_n = state.deformed_positions;
    const std::vector<double> mass_before = ref_mesh.mass;

    // This is the route used by simulation.cpp for a scene containing tets
    // but no rigid bodies. With elasticity and contact disabled, one sweep
    // has the exact free-fall solution x^{n+1} = x^n + dt^2 g.
    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.max_global_iters = 1;
    params.fixed_iters = true;
    params.gravity = Vec3(0.0, -9.8, 0.0);
    params.node_box_min = 1.0;
    params.node_box_max = 1.0;

    // The general-path mass setup must retain density-derived tet masses;
    // only a leading cloth-triangle prefix would be rebuilt.
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);
    EXPECT_EQ(ref_mesh.mass, mass_before);

    const VertexTriangleMap adj =
        build_incident_triangle_map(ref_mesh.tris);
    std::vector<Pin> pins;
    BroadPhase broad_phase;
    const SolverResult result = advance_one_frame_general(
        state, ref_mesh, adj, pins, params, broad_phase);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_TRUE(state.x_coms.empty());
    EXPECT_TRUE(state.orientations.empty());
    const Vec3 displacement = params.dt2() * params.gravity;
    ASSERT_EQ(state.deformed_positions.size(), positions_n.size());
    for (std::size_t node = 0; node < positions_n.size(); ++node) {
        EXPECT_TRUE(state.deformed_positions[node].isApprox(
            positions_n[node] + displacement, 1.0e-12));
        EXPECT_TRUE(state.velocities[node].isApprox(
            params.dt() * params.gravity, 1.0e-12));
    }
}

TEST(SolidGeneralSolver,
     FullyPinnedRigidSurfaceRepelsSolidWithoutReceivingReactionMotion) {
    RefMesh ref_mesh;
    DeformedState state;
    const std::vector<Vec3> solid_positions = {
        Vec3(-0.3, -0.3, 0.2),
        Vec3( 0.3, -0.3, 0.2),
        Vec3(-0.3,  0.3, 0.2),
        Vec3(-0.3, -0.3, 0.8)};
    create_solid(
        solid_positions, {0, 1, 2, 3}, 24.0, ref_mesh, state);

    const std::vector<Vec3> rigid_positions = {
        Vec3(-2.0, -2.0, 0.0), Vec3( 2.0, -2.0, 0.0),
        Vec3( 2.0,  2.0, 0.0), Vec3(-2.0,  2.0, 0.0)};
    const int rb = create_rigid_body(
        rigid_positions, Vec3::Zero(),
        Vec4(0.8, -0.2, 0.3, 0.4), Vec3::Zero(), 40.0,
        ref_mesh, state, RigidBodyUpdateMode::None);
    const std::vector<int>& rigid_nodes = ref_mesh.rb_nodes[rb];
    ref_mesh.tris.insert(
        ref_mesh.tris.end(),
        {rigid_nodes[0], rigid_nodes[1], rigid_nodes[2],
         rigid_nodes[0], rigid_nodes[2], rigid_nodes[3]});
    ref_mesh.build_deformable_nodes();

    const std::vector<Vec3> rigid_positions_before = {
        state.deformed_positions[rigid_nodes[0]],
        state.deformed_positions[rigid_nodes[1]],
        state.deformed_positions[rigid_nodes[2]],
        state.deformed_positions[rigid_nodes[3]]};
    const Vec3 rigid_com_before = state.x_coms[rb];
    const Vec4 rigid_orientation_before = state.orientations[rb];

    SimParams params = SimParams::zeros();
    params.fps = 10.0;
    params.substeps = 1;
    params.max_global_iters = 1;
    params.fixed_iters = true;
    params.use_parallel = false;
    params.use_ccd = false;
    params.damping = 0.25;
    params.d_hat = 0.5;
    params.k_barrier = 100.0;
    params.solid_mu = 0.0;
    params.solid_lambda = 0.0;
    params.node_box_min = 1.0;
    params.node_box_max = 1.0;
    params.theta_box_min = M_PI;
    params.theta_box_max = M_PI;
    params.node_box_update_count = 1;

    const VertexTriangleMap adj =
        build_incident_triangle_map(ref_mesh.tris);
    std::vector<Pin> pins;
    BroadPhase broad_phase;
    const SolverResult result = advance_one_frame_general(
        state, ref_mesh, adj, pins, params, broad_phase);

    ASSERT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_GT(state.deformed_positions[0].z(), solid_positions[0].z());
    EXPECT_TRUE(state.x_coms[rb].isApprox(rigid_com_before, 0.0));
    EXPECT_TRUE(state.v_coms[rb].isZero(0.0));
    EXPECT_TRUE(state.orientations[rb].isApprox(
        rigid_orientation_before, 0.0));
    EXPECT_TRUE(state.omega[rb].isZero(0.0));
    for (int local = 0; local < 4; ++local) {
        const int node = rigid_nodes[local];
        EXPECT_TRUE(state.deformed_positions[node].isApprox(
            rigid_positions_before[local], 1.0e-14));
        EXPECT_TRUE(state.velocities[node].isZero(0.0));
    }
}
