#pragma once
#include "physics.h"
#include <vector>

TriangleRest make_rest_triangle(const RefMesh& ref_mesh, const Tri& tri);

TriangleDef make_def_triangle(const std::vector<Vec3>& x, const Tri& tri);

void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt);

void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt);

void clear_model(RefMesh& ref_mesh, DeformedState& state, std::vector<Pin>& pins);

int build_single_triangle(RefMesh& ref_mesh, DeformedState& state, const Vec2& X0, const Vec2& X1, const Vec2& X2,
                          const Vec3& x0, const Vec3& x1, const Vec3& x2);

void append_pin(std::vector<Pin>& pins, int vertex_index, const std::vector<Vec3>& x);

int build_square_mesh(RefMesh& ref_mesh, DeformedState& state, int nx, int ny, double width, double height, const Vec3& origin);