#include "corotated_energy.h"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <algorithm>

namespace {

    TriangleRest MakeRestTriangle() {
        TriangleRest rest;
        rest.X[0] = Vec2(0.0, 0.0);
        rest.X[1] = Vec2(1.2, 0.1);
        rest.X[2] = Vec2(0.2, 0.9);
        return rest;
    }

    TriangleDef MakeDeformedTriangle() {
        TriangleDef def;
        def.x[0] = Vec3(0.1, -0.2, 0.3);
        def.x[1] = Vec3(1.4, 0.2, -0.1);
        def.x[2] = Vec3(0.0, 1.0, 0.4);
        return def;
    }

    TriangleDef MakeDirection() {
        TriangleDef dx = ZeroTriangleDef();
        dx.x[0] = Vec3(0.3, -0.7, 0.2);
        dx.x[1] = Vec3(-0.4, 0.1, 0.5);
        dx.x[2] = Vec3(0.25, 0.6, -0.35);

        double norm = std::sqrt(dx.x[0].squaredNorm() +
                                dx.x[1].squaredNorm() +
                                dx.x[2].squaredNorm());
        for (auto& v : dx.x) v /= norm;
        return dx;
    }

} // namespace

TEST(CorotatedEnergyTest, EnergyIsFinite) {
TriangleRest rest = MakeRestTriangle();
TriangleDef def = MakeDeformedTriangle();

double E = corotated_energy(rest, def, 2.0, 5.0);
EXPECT_TRUE(std::isfinite(E));
}

TEST(CorotatedEnergyTest, GradientMatchesFiniteDifference) {
TriangleRest rest = MakeRestTriangle();
TriangleDef def = MakeDeformedTriangle();

const double mu = 2.0;
const double lambda = 5.0;
const double eps = 1e-6;

auto g = corotated_node_gradient(rest, def, mu, lambda);

double max_abs_err = 0.0;
double max_rel_err = 0.0;

for (int i = 0; i < 3; ++i) {
for (int c = 0; c < 3; ++c) {
TriangleDef def_plus = def;
TriangleDef def_minus = def;

set_dof(def_plus, i, c, get_dof(def_plus, i, c) + eps);
set_dof(def_minus, i, c, get_dof(def_minus, i, c) - eps);

double Eplus = corotated_energy(rest, def_plus, mu, lambda);
double Eminus = corotated_energy(rest, def_minus, mu, lambda);
double fd = (Eplus - Eminus) / (2.0 * eps);
double an = g[i](c);

double abs_err = std::abs(fd - an);
double denom = std::max(1.0, std::max(std::abs(fd), std::abs(an)));
double rel_err = abs_err / denom;

max_abs_err = std::max(max_abs_err, abs_err);
max_rel_err = std::max(max_rel_err, rel_err);
}
}

EXPECT_LT(max_abs_err, 1e-10);
EXPECT_LT(max_rel_err, 1e-10);
}

TEST(CorotatedEnergyTest, HessianMatchesFiniteDifference) {
TriangleRest rest = MakeRestTriangle();
TriangleDef def = MakeDeformedTriangle();

const double mu = 2.0;
const double lambda = 5.0;
const double eps = 1e-6;

Mat99 H = corotated_node_hessian(rest, def, mu, lambda);

double max_abs_err = 0.0;
double max_rel_err = 0.0;

for (int j = 0; j < 3; ++j) {
for (int c = 0; c < 3; ++c) {
TriangleDef def_plus = def;
TriangleDef def_minus = def;

set_dof(def_plus, j, c, get_dof(def_plus, j, c) + eps);
set_dof(def_minus, j, c, get_dof(def_minus, j, c) - eps);

auto g_plus = corotated_node_gradient(rest, def_plus, mu, lambda);
auto g_minus = corotated_node_gradient(rest, def_minus, mu, lambda);

for (int i = 0; i < 3; ++i) {
Vec3 fd = (g_plus[i] - g_minus[i]) / (2.0 * eps);
Vec3 an = H.block<3,1>(3 * i, 3 * j + c);
Vec3 err = fd - an;

double abs_err = err.cwiseAbs().maxCoeff();
double rel_err = abs_err / std::max(
        1.0,
        std::max(fd.cwiseAbs().maxCoeff(), an.cwiseAbs().maxCoeff())
);

max_abs_err = std::max(max_abs_err, abs_err);
max_rel_err = std::max(max_rel_err, rel_err);
}
}
}

EXPECT_LT(max_abs_err, 1e-10);
EXPECT_LT(max_rel_err, 1e-10);
}

TEST(CorotatedEnergyTest, HessianIsNearlySymmetric) {
TriangleRest rest = MakeRestTriangle();
TriangleDef def = MakeDeformedTriangle();

Mat99 H = corotated_node_hessian(rest, def, 2.0, 5.0);
double sym_err = (H - H.transpose()).cwiseAbs().maxCoeff();

EXPECT_LT(sym_err, 1e-8);
}

TEST(CorotatedEnergyTest, GradientDirectionalDerivativeConverges) {
TriangleRest rest = MakeRestTriangle();
TriangleDef def = MakeDeformedTriangle();
TriangleDef dx = MakeDirection();

const double mu = 2.0;
const double lambda = 5.0;

auto g = corotated_node_gradient(rest, def, mu, lambda);
Vec9 g_flat = flatten_gradient(g);
Vec9 dx_flat = flatten_def(dx);
double exact = g_flat.dot(dx_flat);

std::vector<double> eps_list = {1e-1, 5e-2, 2.5e-2, 1e-2, 5e-3};
std::vector<double> errors;

for (double eps : eps_list) {
TriangleDef def_plus = add_scale(def, dx, eps);
TriangleDef def_minus = add_scale(def, dx, -eps);

double Eplus = corotated_energy(rest, def_plus, mu, lambda);
double Eminus = corotated_energy(rest, def_minus, mu, lambda);

double fd = (Eplus - Eminus) / (2.0 * eps);
errors.push_back(std::abs(fd - exact));
}

for (size_t i = 1; i < errors.size(); ++i) {
EXPECT_LT(errors[i], errors[i - 1]);
}
}