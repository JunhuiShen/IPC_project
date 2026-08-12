#include "make_shape.h"
#include "example.h"
#include "mesh_utils.h"
#include <gtest/gtest.h>

#include <array>

TEST(BuildIncidentTriangleMap, BasicExample) {
// [0,1,2, 1,2,5] -- two triangles
// New format: {tri_idx, local_node_index}
std::vector<int> indices = {0, 1, 2, 1, 2, 5};
auto map = build_incident_triangle_map(indices);

EXPECT_EQ(map[0], (std::vector<std::pair<int,int>>{{0, 0}}));
EXPECT_EQ(map[1], (std::vector<std::pair<int,int>>{{0, 1}, {1, 0}}));
EXPECT_EQ(map[2], (std::vector<std::pair<int,int>>{{0, 2}, {1, 1}}));
EXPECT_EQ(map[5], (std::vector<std::pair<int,int>>{{1, 2}}));
EXPECT_EQ(map.size(), 4u);
}
TEST(BuildIncidentTriangleMap, EmptyInput) {
std::vector<int> indices = {};
auto map = build_incident_triangle_map(indices);
EXPECT_TRUE(map.empty());
}

TEST(MixedExample, TenMixedRigidPolygonsAboveFourCornerPinnedCloth) {
    IPCArgs3D args;
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params = args.to_sim_params();

    build_ten_rigid_polygons_drop_on_pinned_cloth_example(
        args, ref_mesh, state, X, pins, params);

    constexpr int cloth_nx = 20;
    constexpr int cloth_nz = 12;
    constexpr int cloth_vertices =
        (cloth_nx + 1) * (cloth_nz + 1);
    constexpr int cloth_triangles = 2 * cloth_nx * cloth_nz;
    constexpr std::array<int, 10> polygon_sides = {
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    constexpr int rigid_vertices = 150;
    constexpr int rigid_triangles = 260;
    constexpr int total_vertices = cloth_vertices + rigid_vertices;
    constexpr int total_triangles = cloth_triangles + rigid_triangles;

    EXPECT_EQ(state.deformed_positions.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(state.velocities.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.num_positions,
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.node_to_rb.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.mass.size(),
              static_cast<std::size_t>(total_vertices));
    EXPECT_EQ(ref_mesh.tris.size(),
              static_cast<std::size_t>(3 * total_triangles));
    EXPECT_EQ(X.size(), static_cast<std::size_t>(cloth_vertices));
    EXPECT_EQ(ref_mesh.Dm_inverse.size(),
              static_cast<std::size_t>(cloth_triangles));
    EXPECT_EQ(ref_mesh.area.size(),
              static_cast<std::size_t>(cloth_triangles));
    ASSERT_EQ(pins.size(), 4u);
    EXPECT_EQ(pins[0].vertex_index, 0);
    EXPECT_EQ(pins[1].vertex_index, cloth_nx);
    EXPECT_EQ(pins[2].vertex_index,
              cloth_nz * (cloth_nx + 1));
    EXPECT_EQ(pins[3].vertex_index, cloth_vertices - 1);

    ASSERT_EQ(ref_mesh.rb_nodes.size(), polygon_sides.size());
    ASSERT_EQ(ref_mesh.ref_positions.size(), polygon_sides.size());
    ASSERT_EQ(ref_mesh.total_mass.size(), polygon_sides.size());
    ASSERT_EQ(ref_mesh.I_hat.size(), polygon_sides.size());
    ASSERT_EQ(state.x_coms.size(), polygon_sides.size());
    ASSERT_EQ(state.v_coms.size(), polygon_sides.size());
    ASSERT_EQ(state.orientations.size(), polygon_sides.size());
    ASSERT_EQ(state.omega.size(), polygon_sides.size());

    for (int node = 0; node < cloth_vertices; ++node)
        EXPECT_EQ(ref_mesh.node_to_rb[node], -1);

    for (int triangle = 0; triangle < cloth_triangles; ++triangle) {
        for (int corner = 0; corner < 3; ++corner) {
            const int node = ref_mesh.tris[3 * triangle + corner];
            EXPECT_EQ(ref_mesh.node_to_rb[node], -1);
        }
    }

    int triangle_cursor = cloth_triangles;
    for (std::size_t rb = 0; rb < polygon_sides.size(); ++rb) {
        const int side_count = polygon_sides[rb];
        const std::size_t expected_nodes =
            static_cast<std::size_t>(2 * side_count);
        ASSERT_EQ(ref_mesh.rb_nodes[rb].size(), expected_nodes);
        ASSERT_EQ(ref_mesh.ref_positions[rb].size(), expected_nodes);
        EXPECT_GT(ref_mesh.total_mass[rb], 0.0);
        EXPECT_NEAR(state.orientations[rb].norm(), 1.0, 1.0e-12);
        EXPECT_TRUE(state.v_coms[rb].isZero(1.0e-12));
        EXPECT_TRUE(state.omega[rb].isZero(1.0e-12));

        for (const int node : ref_mesh.rb_nodes[rb]) {
            EXPECT_EQ(ref_mesh.node_to_rb[node], static_cast<int>(rb));
            EXPECT_GT(ref_mesh.mass[node], 0.0);

            const Vec3& position = state.deformed_positions[node];
            EXPECT_GT(position.y(), 0.8 + params.d_hat);
            EXPECT_GE(position.x(), -1.0);
            EXPECT_LE(position.x(), 1.0);
            EXPECT_GE(position.z(), -0.6);
            EXPECT_LE(position.z(), 0.6);
        }

        const int body_triangles = 4 * side_count - 4;
        for (int local_triangle = 0;
             local_triangle < body_triangles;
             ++local_triangle, ++triangle_cursor) {
            for (int corner = 0; corner < 3; ++corner) {
                const int node =
                    ref_mesh.tris[3 * triangle_cursor + corner];
                EXPECT_EQ(
                    ref_mesh.node_to_rb[node], static_cast<int>(rb));
            }
        }
    }
    EXPECT_EQ(triangle_cursor, total_triangles);

    for (const Hinge& hinge : ref_mesh.hinges) {
        for (const int node : hinge.v) {
            EXPECT_GE(node, 0);
            EXPECT_LT(node, cloth_vertices);
        }
    }

    const std::vector<double> masses_before = ref_mesh.mass;
    ref_mesh.build_deformable_lumped_mass(
        params.density, params.thickness);

    double total_cloth_mass = 0.0;
    for (int node = 0; node < cloth_vertices; ++node) {
        EXPECT_GT(ref_mesh.mass[node], 0.0);
        total_cloth_mass += ref_mesh.mass[node];
    }
    EXPECT_NEAR(
        total_cloth_mass,
        params.density * params.thickness * 2.0 * 1.2,
        1.0e-12);
    for (const std::vector<int>& body_nodes : ref_mesh.rb_nodes) {
        for (const int node : body_nodes)
            EXPECT_DOUBLE_EQ(ref_mesh.mass[node], masses_before[node]);
    }
}

// ---------------------------------------------------------------------------
// build_sphere_mesh tests
// ---------------------------------------------------------------------------

TEST(BuildSphereMesh, VertexAndTriangleCounts) {
    // Level 2 icosphere: V = 10*4^2 + 2 = 162, F = 20*4^2 = 320.
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    const int subdiv = 2;
    const double radius = 0.5;
    const Vec3 center(0.0, 0.0, 0.0);

    const int base = build_sphere_mesh(ref_mesh, state, X, subdiv, radius, center);
    EXPECT_EQ(base, 0);

    const int expected_verts = 162;
    EXPECT_EQ(static_cast<int>(state.deformed_positions.size()), expected_verts);
    EXPECT_EQ(static_cast<int>(X.size()), expected_verts);

    const int expected_tris = 320;
    EXPECT_EQ(static_cast<int>(ref_mesh.tris.size()), 3 * expected_tris);
}

TEST(BuildSphereMesh, AllVerticesAtRadiusFromCenter) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    const double radius = 0.25;
    const Vec3 center(0.1, -0.2, 0.05);

    build_sphere_mesh(ref_mesh, state, X, /*subdiv=*/3, radius, center);

    // FP-precision tolerance: base icosahedron verts compute norm as s*sqrt(1+phi^2)
    // via a sqrt and a division, which leaves a few ULPs of error at the radius.
    // Subdivided midpoints are explicitly renormalized to exactly radius.
    // FP precision: base icosahedron verts compute norm via a sqrt and division
    // leaving a few ULPs; subdivided midpoints are explicitly renormalized.
    constexpr double kTol = 1e-10;
    for (const Vec3& p : state.deformed_positions)
        EXPECT_NEAR((p - center).norm(), radius, kTol);
}

TEST(BuildSphereMesh, BaseIcosahedron) {
    // subdiv = 0 is the base icosahedron: 12 vertices, 20 triangles.
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    build_sphere_mesh(ref_mesh, state, X, /*subdiv=*/0, /*radius=*/1.0, Vec3::Zero());

    EXPECT_EQ(static_cast<int>(state.deformed_positions.size()), 12);
    EXPECT_EQ(static_cast<int>(ref_mesh.tris.size()),            60);
}

TEST(BuildSphereMesh, ReferenceAreasNonDegenerate) {
    // Reference-space 2D triangle areas must be strictly positive so
    // ref_mesh.initialize(X) doesn't divide by zero when forming Dm_inverse.
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    build_sphere_mesh(ref_mesh, state, X, /*subdiv=*/2, /*radius=*/0.5, Vec3::Zero());

    const int nt = static_cast<int>(ref_mesh.tris.size()) / 3;
    for (int t = 0; t < nt; ++t) {
        const Vec2& a = X[ref_mesh.tris[3*t + 0]];
        const Vec2& b = X[ref_mesh.tris[3*t + 1]];
        const Vec2& c = X[ref_mesh.tris[3*t + 2]];
        const double area2 = std::abs((b.x()-a.x())*(c.y()-a.y()) - (b.y()-a.y())*(c.x()-a.x()));
        EXPECT_GT(area2, 1e-10) << "degenerate ref triangle " << t;
    }
}
