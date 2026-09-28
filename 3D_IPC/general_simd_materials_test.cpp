#include "general_simd_assembly.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <vector>

namespace {

struct MaterialFixture {
    RefMesh mesh;
    std::vector<IncidentTriangles> incident;
    std::vector<ShapeGrads> shapes;
    std::vector<Vec3> positions, predicted;
    std::vector<Pin> pins;
    PinMap pin_map;
    std::vector<unsigned char> solid, surface;
    SimParams params;

    MaterialFixture() {
        positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
            Vec3(3, 0, 0), Vec3(4, 0, 0), Vec3(3, 1, 0), Vec3(3, 0, 1)};
        mesh.num_positions = positions.size();
        mesh.mass.assign(positions.size(), 1.0);
        mesh.node_to_rb.assign(positions.size(), -1);
        mesh.tris = {0, 1, 2, 0, 2, 3};
        mesh.compute_dm_inverse(positions);
        mesh.hinges.push_back(Hinge{{0, 2, 1, 3}, 0.0, 0.7});
        for (int role = 0; role < 4; ++role)
            mesh.hinge_adj[mesh.hinges[0].v[role]].emplace_back(0, role);
        incident.resize(positions.size());
        for (int triangle = 0; triangle < 2; ++triangle)
            for (int role = 0; role < 3; ++role)
                incident[mesh.tris[3 * triangle + role]].emplace_back(triangle, role);
        for (const auto& inverse : mesh.Dm_inverse)
            shapes.push_back(shape_function_gradients(inverse));
        mesh.tets = {4, 5, 6, 7};
        mesh.tet_nodes = mesh.surface_nodes = {4, 5, 6, 7};
        mesh.tet_rest_data = EFEMInitializeElasticMaterialState(positions, mesh.tets);
        mesh.tet_adj.resize(positions.size());
        for (int role = 0; role < 4; ++role) mesh.tet_adj[4 + role].emplace_back(0, role);
        solid.assign(positions.size(), 0); surface.assign(positions.size(), 0);
        for (int node : mesh.tet_nodes) solid[node] = surface[node] = 1;
        pin_map.assign(positions.size(), -1);
        positions[2].z() = 0.08;
        positions[5].x() += 0.13;
        positions[6].z() -= 0.04;
        predicted = positions;
        for (auto& x : predicted) x += Vec3(0.02, -0.01, 0.03);
        params.mu = 2.1; params.lambda = 3.2;
        params.solid_mu = 2.3; params.solid_lambda = 4.1;
        params.kB = 0.4; params.k_sdf = 0.0;
        params.friction_coefficient = 0.0;
        params.gravity = Vec3(0.0, -9.81, 0.0);
    }

    void evaluate(const int* nodes, std::size_t count,
        solver_detail::GeneralSimdVertexSystem* outputs,
        const solver_detail::GeneralSimdMaterials* materials) const {
        solver_detail::prepare_general_simd_batch(nodes, count, mesh, incident,
            shapes, pins, pin_map, params, positions, predicted, nullptr,
            solid, surface, outputs, materials);
    }
};

void expect_same(const solver_detail::GeneralSimdVertexSystem& a,
    const solver_detail::GeneralSimdVertexSystem& b) {
    EXPECT_EQ(std::memcmp(a.gradient.data(), b.gradient.data(), 3 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(a.hessian.data(), b.hessian.data(), 9 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(a.sdf_friction_gradient.data(), b.sdf_friction_gradient.data(), 3 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(a.sdf_friction_hessian.data(), b.sdf_friction_hessian.data(), 9 * sizeof(double)), 0);
}

} // namespace

TEST(GeneralSimdMaterials, RetainsIncidentOrderAndReusesUnchangedRecords) {
    MaterialFixture fixture;
    solver_detail::GeneralSimdMaterials materials;
    EXPECT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    EXPECT_EQ(materials.triangles.node_offsets, (std::vector<std::size_t>{0, 2, 3, 5, 6, 6, 6, 6, 6}));
    EXPECT_EQ(materials.tets.node_offsets, (std::vector<std::size_t>{0, 0, 0, 0, 0, 1, 2, 3, 4}));
    EXPECT_EQ(materials.hinges.node_offsets, (std::vector<std::size_t>{0, 1, 2, 3, 4, 4, 4, 4, 4}));
    EXPECT_EQ(materials.triangles.nodes[0], (std::array<int, 3>{0, 1, 2}));
    EXPECT_EQ(materials.triangles.nodes[1], (std::array<int, 3>{0, 2, 3}));
    const auto* retained = materials.triangles.dm_inverse.data();
    EXPECT_FALSE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    EXPECT_EQ(materials.triangles.dm_inverse.data(), retained);
    fixture.mesh.mass[0] = 2.0;
    fixture.positions[0].x() += 0.1;
    EXPECT_FALSE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    EXPECT_EQ(materials.triangles.dm_inverse.data(), retained);
}

TEST(GeneralSimdMaterials, InPlaceTopologyAdjacencyAndRestEditsInvalidate) {
    MaterialFixture fixture;
    solver_detail::GeneralSimdMaterials materials;
    ASSERT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    const auto rebuild = [&] {
        EXPECT_FALSE(materials.matches(fixture.mesh, fixture.incident, 8, true));
        EXPECT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
        EXPECT_TRUE(materials.matches(fixture.mesh, fixture.incident, 8, true));
    };
    fixture.mesh.Dm_inverse[0](0, 1) += 0.3; rebuild();
    EXPECT_TRUE(materials.triangles.shape_gradients[0].isApprox(
        shape_function_gradients(fixture.mesh.Dm_inverse[0])[0], 0.0));
    fixture.mesh.area[0] *= 1.2; rebuild();
    fixture.mesh.hinges[0].c_e *= 1.1; rebuild();
    fixture.mesh.hinges[0].bar_theta += 0.2; rebuild();
    std::swap(fixture.mesh.hinges[0].v[2], fixture.mesh.hinges[0].v[3]); rebuild();
    fixture.mesh.hinge_adj[0][0].second = 1; rebuild();
    std::swap(fixture.incident[0][0], fixture.incident[0][1]); rebuild();
    EXPECT_EQ(materials.triangles.nodes[0], (std::array<int, 3>{0, 2, 3}));
    std::swap(fixture.mesh.tris[1], fixture.mesh.tris[2]); rebuild();
    fixture.mesh.tet_rest_data[0].Dm_inverse(1, 2) += 0.1; rebuild();
    fixture.mesh.tet_rest_data[0].measure *= 1.2; rebuild();
    fixture.mesh.tet_rest_data[0].grad_N[0][2] += 0.4; rebuild();
    fixture.mesh.tet_adj[4][0].second = 1; rebuild();
    EXPECT_TRUE(materials.tets.shape_gradients[0].isApprox(fixture.mesh.tet_rest_data[0].grad_N[1], 0.0));
    std::swap(fixture.mesh.tets[1], fixture.mesh.tets[2]); rebuild();
    fixture.mesh.node_to_rb[0] = 0; rebuild();
    EXPECT_EQ(materials.triangles.node_offsets[1], 0u);
    EXPECT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, false));
    EXPECT_TRUE(materials.hinges.nodes.empty());
    EXPECT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    EXPECT_FALSE(materials.hinges.nodes.empty());
}

TEST(GeneralSimdMaterials, CachedAndLiveGatherAgreeForSingletonAndMixedBatches) {
    MaterialFixture fixture;
    solver_detail::GeneralSimdMaterials materials;
    ASSERT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    const std::array<int, 8> order{2, 5, 0, 7, 3, 4, 1, 6};
    for (int frame = 0; frame < 2; ++frame) {
        for (std::size_t width = 1; width <= ipc_simd::tile_width; ++width)
            for (std::size_t first = 0; first < order.size(); first += width) {
                const auto count = std::min(width, order.size() - first);
                std::array<solver_detail::GeneralSimdVertexSystem, ipc_simd::tile_width> cached, live;
                fixture.evaluate(order.data() + first, count, cached.data(), &materials);
                fixture.evaluate(order.data() + first, count, live.data(), nullptr);
                for (std::size_t i = 0; i < count; ++i) expect_same(cached[i], live[i]);
            }
        // Static cache remains valid while live positions and point data move.
        fixture.positions[2].z() += 0.03;
        fixture.positions[7].x() -= 0.07;
        fixture.mesh.mass[0] *= 1.4;
        EXPECT_FALSE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    }
}
