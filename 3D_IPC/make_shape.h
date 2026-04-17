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

// Triangulated cylinder whose long axis is +z, centered at `center`. `nu` and `nv` are the
// circumferential and axial subdivisions. The wrap column on the underside is omitted to avoid
// coincident seam vertices that would trip the IPC barrier.
int build_cylinder_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int nu, int nv, double radius, double length, const Vec3& center);

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