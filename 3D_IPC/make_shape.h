#pragma once
#include "physics.h"
#include <unordered_map>
#include <vector>

TriangleDef make_def_triangle(const std::vector<Vec3>& x, const RefMesh& ref_mesh, int tri_idx);

void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt);

void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt);

void clear_model(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, std::vector<Pin>& pins);

void append_pin(std::vector<Pin>& pins, int vertex_index, const std::vector<Vec3>& x);

int build_square_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int nx, int ny, double width, double height, const Vec3& origin);

// Triangulated cylinder whose long axis is +z, centered at `center`. `nu` is the
// circumferential subdivision count. Uses a brick-pattern triangulation where
// odd axial rings are rotated by half a circumferential step, so every
// triangle is near-equilateral. Axial row count is picked internally so the
// row height matches the equilateral height h = (2*pi*r/nu) * sqrt(3)/2.
// Callers must read the generated vertex count from the size of
// `state.deformed_positions` rather than assuming any formula.
int build_cylinder_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int nu, double radius, double length, const Vec3& center);

// Icosphere built by `subdiv` loop-subdivisions of a base icosahedron, with every
// midpoint normalized to `radius`. Vertex
// count: 10*4^subdiv + 2. Triangle count: 20*4^subdiv. Reference 2D coords use
// xz-projection; intended for use as a static pinned collider (elastic energy
// on sphere triangles is never evaluated because their vertices are all pinned).
int build_sphere_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int subdiv, double radius, const Vec3& center);

// Maps each vertex to {triangle_index, local_node_index} pairs.
// The local_node_index (0,1,2) is stored to avoid searching at call sites.
VertexTriangleMap build_incident_triangle_map(const std::vector<int>& indices);

// Maps each vertex to the set of other vertex indices it shares a triangle with.
// e.g. if vertex 0 appears in triangles with vertices 1,3,5,7 -> {0: [1,3,5,7]}
// Used for graph coloring.
std::unordered_map<int, std::vector<int>> build_vertex_adjacency_map(const std::vector<int>& tris);

// Greedy graph coloring. Returns groups[color] = list of vertex indices with that color.
// Vertices in the same group share no triangle and can be updated in parallel.
std::vector<std::vector<int>> greedy_color(
    const std::unordered_map<int, std::vector<int>>& adj,
    int num_vertices);