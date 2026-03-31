#pragma once

#include <Eigen/Dense>
#include <array>

// Type aliases
using Vec2  = Eigen::Vector2d;
using Vec3  = Eigen::Vector3d;
using Vec9  = Eigen::Matrix<double, 9, 1>;
using Mat22 = Eigen::Matrix2d;
using Mat32 = Eigen::Matrix<double, 3, 2>;
using Mat33 = Eigen::Matrix3d;
using Mat66 = Eigen::Matrix<double, 6, 6>;
using Mat99 = Eigen::Matrix<double, 9, 9>;

// Triangle definitions
struct TriangleRest {
    Vec2 X[3];
};

struct TriangleDef {
    Vec3 x[3];
};

// Utility functions
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