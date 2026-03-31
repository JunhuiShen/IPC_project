#pragma once
#include "corotated_energy.h"
#include <array>

Mat33 matrix3d_inverse(const Mat33& H);

TriangleDef ZeroTriangleDef();

TriangleDef add_scale(const TriangleDef& a, const TriangleDef& b, double s);

Vec9 flatten_def(const TriangleDef& def);

Vec9 flatten_gradient(const std::array<Vec3, 3>& g);

double get_dof(const TriangleDef& def, int node, int comp);

void set_dof(TriangleDef& def, int node, int comp, double value);

double clamp_scalar(double v, double lo, double hi);

Vec3 segment_closest_point(const Vec3& x, const Vec3& a, const Vec3& b, double& t);

std::array<double, 3> triangle_plane_barycentric_coordinates(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1.0e-12);