#pragma once

#include <Eigen/Dense>
#include <array>

// Type aliases
using Vec2   = Eigen::Vector2d;
using Vec3   = Eigen::Vector3d;
using Vec6   = Eigen::Matrix<double, 6, 1>;
using Vec9   = Eigen::Matrix<double, 9, 1>;
using Vec12  = Eigen::Matrix<double, 12, 1>;
using Mat22  = Eigen::Matrix2d;
using Mat32  = Eigen::Matrix<double, 3, 2>;
using Mat33  = Eigen::Matrix3d;
using Mat39  = Eigen::Matrix<double, 3, 9>;
using Mat66  = Eigen::Matrix<double, 6, 6>;
using Mat312 = Eigen::Matrix<double, 3, 12>;
using Mat12  = Eigen::Matrix<double, 12, 12>;

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

double get_dof(const TriangleDef& def, int node, int comp);

void set_dof(TriangleDef& def, int node, int comp, double value);

double clamp_scalar(double v, double lo, double hi);

Vec3 segment_closest_point(const Vec3& x, const Vec3& a, const Vec3& b, double& t);

std::array<double, 3> triangle_plane_barycentric_coordinates(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1e-12);

bool nearly_zero(double value, double eps = 1e-12);

bool in_unit_interval(double t, double eps = 1e-12);

Vec3 point_at_linear_step(const Vec3& x, const Vec3& dx, double t);

bool point_in_triangle_on_plane(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1e-12);

bool segment_segment_parameters_if_not_parallel(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double& s, double& u, double eps = 1e-12);