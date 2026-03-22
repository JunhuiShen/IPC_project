#include "corotated_energy.h"
#include <gtest/gtest.h>

#include <cmath>
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

    EXPECT_LT(max_abs_err, 1e-5);
    EXPECT_LT(max_rel_err, 1e-5);
}

TEST(CorotatedEnergyTest, HessianIsNearlySymmetric) {
    TriangleRest rest = MakeRestTriangle();
    TriangleDef def = MakeDeformedTriangle();

    Mat99 H = corotated_node_hessian(rest, def, 2.0, 5.0);
    double sym_err = (H - H.transpose()).cwiseAbs().maxCoeff();

    EXPECT_LT(sym_err, 1e-8);
}
