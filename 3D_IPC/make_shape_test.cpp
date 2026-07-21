#include "make_shape.h"
#include "mesh_utils.h"
#include <gtest/gtest.h>

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
