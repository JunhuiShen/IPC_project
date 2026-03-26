#pragma once

#include <Eigen/Dense>
#include <array>

using Vec2  = Eigen::Vector2d;
using Vec3  = Eigen::Vector3d;
using Vec9  = Eigen::Matrix<double, 9, 1>;
using Mat22 = Eigen::Matrix2d;
using Mat32 = Eigen::Matrix<double, 3, 2>;
using Mat33 = Eigen::Matrix3d;
using Mat66 = Eigen::Matrix<double, 6, 6>;
using Mat99 = Eigen::Matrix<double, 9, 9>;

struct TriangleRest {
    Vec2 X[3];
};

struct TriangleDef {
    Vec3 x[3];
};

double corotated_energy(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);

std::array<Vec3, 3> corotated_node_gradient(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);

Mat99 corotated_node_hessian(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);