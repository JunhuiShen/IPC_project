#pragma once
#include "physics.h"
#include <unordered_map>
#include <vector>

TriangleRest make_rest_triangle(const std::vector<Vec2>& X, const Tri& tri);

TriangleDef make_def_triangle(const std::vector<Vec3>& x, const Tri& tri);

void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt);

void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt);

void clear_model(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, std::vector<Pin>& pins);

int build_single_triangle(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X,
                          const Vec2& X0, const Vec2& X1, const Vec2& X2,
                          const Vec3& x0, const Vec3& x1, const Vec3& x2);

void append_pin(std::vector<Pin>& pins, int vertex_index, const std::vector<Vec3>& x);

int build_square_mesh(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, int nx, int ny, double width, double height, const Vec3& origin);

// Maps each vertex index to all positions in the flat index buffer where it appears.
// e.g. for [0,1,2,1,2,5]: {0:[0], 1:[1,3], 2:[2,4], 5:[5]}
std::unordered_map<int, std::vector<int>> build_incident_triangle_map(const std::vector<int>& indices);