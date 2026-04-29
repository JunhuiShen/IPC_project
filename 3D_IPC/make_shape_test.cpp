#include "make_shape.h"
#include <algorithm>
#include <gtest/gtest.h>

// Helper: sort a neighbor list so comparisons are order-independent
static std::vector<int> sorted(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    return v;
}

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
  
// ---------------------------------------------------------------------------
// build_vertex_adjacency_map tests
// ---------------------------------------------------------------------------

TEST(BuildVertexAdjacencyMap, SingleTriangle) {
    // Triangle [0,1,2]: each vertex is adjacent to the other two
    std::vector<int> tris = {0, 1, 2};
    auto adj = build_vertex_adjacency_map(tris);

    EXPECT_EQ(adj.size(), 3u);
    EXPECT_EQ(sorted(adj[0]), (std::vector<int>{1, 2}));
    EXPECT_EQ(sorted(adj[1]), (std::vector<int>{0, 2}));
    EXPECT_EQ(sorted(adj[2]), (std::vector<int>{0, 1}));
}

TEST(BuildVertexAdjacencyMap, TwoTrianglesSharedEdge) {
    // Triangles [0,1,2] and [1,2,5] share edge 1-2
    // 0 -> {1,2}   1 -> {0,2,5}   2 -> {0,1,5}   5 -> {1,2}
    std::vector<int> tris = {0, 1, 2, 1, 2, 5};
    auto adj = build_vertex_adjacency_map(tris);

    EXPECT_EQ(adj.size(), 4u);
    EXPECT_EQ(sorted(adj[0]), (std::vector<int>{1, 2}));
    EXPECT_EQ(sorted(adj[1]), (std::vector<int>{0, 2, 5}));
    EXPECT_EQ(sorted(adj[2]), (std::vector<int>{0, 1, 5}));
    EXPECT_EQ(sorted(adj[5]), (std::vector<int>{1, 2}));
}

TEST(BuildVertexAdjacencyMap, NoSelfAdjacency) {
    // No vertex should list itself as a neighbor
    std::vector<int> tris = {0, 1, 2, 1, 2, 5};
    auto adj = build_vertex_adjacency_map(tris);

    for (auto& [v, neighbors] : adj)
        for (int n : neighbors)
            EXPECT_NE(n, v) << "vertex " << v << " lists itself as neighbor";
}

TEST(BuildVertexAdjacencyMap, EmptyInput) {
    auto adj = build_vertex_adjacency_map({});
    EXPECT_TRUE(adj.empty());
}

// ---------------------------------------------------------------------------
// greedy_color tests
// ---------------------------------------------------------------------------

TEST(GreedyColor, SingleTriangle) {
    // 3 vertices all adjacent to each other -> need 3 colors
    std::vector<int> tris = {0, 1, 2};
    auto adj = build_vertex_adjacency_map(tris);
    auto groups = greedy_color(adj, 3);

    EXPECT_EQ(groups.size(), 3u);

    // Every vertex appears exactly once
    std::vector<int> seen(3, 0);
    for (auto& g : groups)
        for (int v : g) seen[v]++;
    for (int v = 0; v < 3; ++v)
        EXPECT_EQ(seen[v], 1) << "vertex " << v << " not in exactly one group";
}

TEST(GreedyColor, NoTwoAdjacentSameColor) {
    // Property test: for any coloring, adjacent vertices must have different colors
    std::vector<int> tris = {0, 1, 2, 1, 2, 5, 2, 5, 6};
    auto adj = build_vertex_adjacency_map(tris);
    int nv = 7;
    auto groups = greedy_color(adj, nv);

    // Build color-per-vertex lookup
    std::vector<int> color(nv, -1);
    for (int c = 0; c < (int)groups.size(); ++c)
        for (int v : groups[c]) color[v] = c;

    for (auto& [v, neighbors] : adj)
        for (int n : neighbors)
            EXPECT_NE(color[v], color[n])
                << "vertices " << v << " and " << n << " are adjacent but share color " << color[v];
}

TEST(GreedyColor, AllVerticesCovered) {
    std::vector<int> tris = {0, 1, 2, 1, 2, 5};
    auto adj = build_vertex_adjacency_map(tris);
    int nv = 6;
    auto groups = greedy_color(adj, nv);

    std::vector<int> seen(nv, 0);
    for (auto& g : groups)
        for (int v : g) seen[v]++;
    for (int v = 0; v < nv; ++v)
        EXPECT_EQ(seen[v], 1) << "vertex " << v << " not in exactly one group";
}

TEST(GreedyColor, FewColorsForGrid) {
    // A 3x3 grid of quads (split into triangles) is planar -> at most 4 colors by 4-color theorem
    // Build a small 2x2 quad grid manually (8 triangles, 9 vertices)
    std::vector<int> tris = {
        0,1,4, 0,4,3,   // quad (0,1,4,3)
        1,2,5, 1,5,4,   // quad (1,2,5,4)
        3,4,7, 3,7,6,   // quad (3,4,7,6)
        4,5,8, 4,8,7,   // quad (4,5,8,7)
    };
    auto adj = build_vertex_adjacency_map(tris);
    auto groups = greedy_color(adj, 9);

    EXPECT_LE(groups.size(), 4u) << "greedy used more than 4 colors on a planar mesh";
}

// ---------------------------------------------------------------------------

TEST(BuildIncidentTriangleMap, EmptyInput) {
std::vector<int> indices = {};
auto map = build_incident_triangle_map(indices);
EXPECT_TRUE(map.empty());
}

TEST(BuildIncidentTriangleMap, SingleTriangle) {
std::vector<int> indices = {0, 1, 2};
auto map = build_incident_triangle_map(indices);

EXPECT_EQ(map[0], (std::vector<std::pair<int,int>>{{0, 0}}));
EXPECT_EQ(map[1], (std::vector<std::pair<int,int>>{{0, 1}}));
EXPECT_EQ(map[2], (std::vector<std::pair<int,int>>{{0, 2}}));
EXPECT_EQ(map.size(), 3u);
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