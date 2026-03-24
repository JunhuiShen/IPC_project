#include "corotated_energy.h"
#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {

    constexpr double kMu = 2.0;
    constexpr double kLambda = 5.0;
    constexpr double kGradFdEps = 1e-6;
    constexpr double kHessFdEps = 1e-6;
    constexpr double kGradAbsTol = 1e-6;
    constexpr double kGradRelTol = 1e-6;
    constexpr double kHessAbsTol = 5e-5;
    constexpr double kHessRelTol = 5e-5;
    constexpr double kSymTol = 1e-8;
    constexpr double kRigidTol = 1e-10;

    TriangleRest MakeRestTriangle() {
        TriangleRest rest;
        rest.X[0] = Vec2(0.0, 0.0);
        rest.X[1] = Vec2(1.2, 0.1);
        rest.X[2] = Vec2(0.2, 0.9);
        return rest;
    }

    TriangleDef EmbedRestTriangle(const TriangleRest& rest) {
        TriangleDef def;
        for (int i = 0; i < 3; ++i) {
            def.x[i] = Vec3(rest.X[i](0), rest.X[i](1), 0.0);
        }
        return def;
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

    TriangleDef Translate(const TriangleDef& def, const Vec3& t) {
        TriangleDef out = def;
        for (auto& x : out.x) x += t;
        return out;
    }

    TriangleDef Rotate(const TriangleDef& def, const Eigen::AngleAxisd& aa) {
        TriangleDef out;
        const Mat33 R = aa.toRotationMatrix();
        for (int i = 0; i < 3; ++i) {
            out.x[i] = R * def.x[i];
        }
        return out;
    }

// ---- single generic helper (fixes your original bug cleanly)
    template <typename Derived>
    double MaxAbs(const Eigen::MatrixBase<Derived>& v) {
        return v.cwiseAbs().maxCoeff();
    }

} // namespace

TEST(CorotatedEnergyTest, EnergyIsFinite) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = MakeDeformedTriangle();

const double E = corotated_energy(rest, def, kMu, kLambda);
EXPECT_TRUE(std::isfinite(E));
}

TEST(CorotatedEnergyTest, RestStateHasZeroEnergyAndZeroGradient) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = EmbedRestTriangle(rest);

const double E = corotated_energy(rest, def, kMu, kLambda);
const auto g = corotated_node_gradient(rest, def, kMu, kLambda);

EXPECT_NEAR(E, 0.0, kRigidTol);
for (int i = 0; i < 3; ++i) {
EXPECT_LE(MaxAbs(g[i]), kRigidTol);
}
}

TEST(CorotatedEnergyTest, GradientMatchesFiniteDifference) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = MakeDeformedTriangle();

const auto g = corotated_node_gradient(rest, def, kMu, kLambda);

double max_abs_err = 0.0;
double max_rel_err = 0.0;

for (int i = 0; i < 3; ++i) {
for (int c = 0; c < 3; ++c) {
TriangleDef def_plus = def;
TriangleDef def_minus = def;

set_dof(def_plus, i, c, get_dof(def_plus, i, c) + kGradFdEps);
set_dof(def_minus, i, c, get_dof(def_minus, i, c) - kGradFdEps);

const double Eplus = corotated_energy(rest, def_plus, kMu, kLambda);
const double Eminus = corotated_energy(rest, def_minus, kMu, kLambda);
const double fd = (Eplus - Eminus) / (2.0 * kGradFdEps);
const double an = g[i](c);

const double abs_err = std::abs(fd - an);
const double denom = std::max(1.0, std::max(std::abs(fd), std::abs(an)));
const double rel_err = abs_err / denom;

max_abs_err = std::max(max_abs_err, abs_err);
max_rel_err = std::max(max_rel_err, rel_err);
}
}

EXPECT_LT(max_abs_err, kGradAbsTol);
EXPECT_LT(max_rel_err, kGradRelTol);
}

TEST(CorotatedEnergyTest, HessianMatchesFiniteDifference) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = MakeDeformedTriangle();

const Mat99 H = corotated_node_hessian(rest, def, kMu, kLambda);

double max_abs_err = 0.0;
double max_rel_err = 0.0;

for (int j = 0; j < 3; ++j) {
for (int c = 0; c < 3; ++c) {
TriangleDef def_plus = def;
TriangleDef def_minus = def;

set_dof(def_plus, j, c, get_dof(def_plus, j, c) + kHessFdEps);
set_dof(def_minus, j, c, get_dof(def_minus, j, c) - kHessFdEps);

const auto g_plus = corotated_node_gradient(rest, def_plus, kMu, kLambda);
const auto g_minus = corotated_node_gradient(rest, def_minus, kMu, kLambda);

for (int i = 0; i < 3; ++i) {
const Vec3 fd = (g_plus[i] - g_minus[i]) / (2.0 * kHessFdEps);
const Vec3 an = H.block<3, 1>(3 * i, 3 * j + c);
const Vec3 err = fd - an;

const double abs_err = MaxAbs(err);
const double rel_err = abs_err /
                       std::max(1.0,
                                std::max(MaxAbs(fd), MaxAbs(an)));

max_abs_err = std::max(max_abs_err, abs_err);
max_rel_err = std::max(max_rel_err, rel_err);
}
}
}

EXPECT_LT(max_abs_err, kHessAbsTol);
EXPECT_LT(max_rel_err, kHessRelTol);
}

TEST(CorotatedEnergyTest, HessianIsNearlySymmetric) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = MakeDeformedTriangle();

const Mat99 H = corotated_node_hessian(rest, def, kMu, kLambda);
EXPECT_LT(MaxAbs(H - H.transpose()), kSymTol);
}

TEST(CorotatedEnergyTest, EnergyIsTranslationInvariant) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = MakeDeformedTriangle();
const TriangleDef shifted = Translate(def, Vec3(2.5, -1.25, 0.75));

const double E0 = corotated_energy(rest, def, kMu, kLambda);
const double E1 = corotated_energy(rest, shifted, kMu, kLambda);

EXPECT_NEAR(E0, E1, kRigidTol);
}

TEST(CorotatedEnergyTest, EnergyIsRotationInvariantForRigidMotionFromRest) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def0 = EmbedRestTriangle(rest);
const TriangleDef def1 = Rotate(def0,
                                Eigen::AngleAxisd(0.7, Vec3(1.0, 2.0, -1.0).normalized()));

const double E = corotated_energy(rest, def1, kMu, kLambda);
const auto g = corotated_node_gradient(rest, def1, kMu, kLambda);

EXPECT_NEAR(E, 0.0, kRigidTol);
for (int i = 0; i < 3; ++i) {
EXPECT_LE(MaxAbs(g[i]), kRigidTol);
}
}

TEST(CorotatedEnergyTest, NodalGradientsSumToZero) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = MakeDeformedTriangle();

const auto g = corotated_node_gradient(rest, def, kMu, kLambda);
const Vec3 total = g[0] + g[1] + g[2];

EXPECT_LE(MaxAbs(total), 1e-10);
}

TEST(CorotatedEnergyTest, HessianHasTranslationNullModes) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = MakeDeformedTriangle();

const Mat99 H = corotated_node_hessian(rest, def, kMu, kLambda);

for (int axis = 0; axis < 3; ++axis) {
Vec9 t = Vec9::Zero();
for (int node = 0; node < 3; ++node) {
t(3 * node + axis) = 1.0;
}
EXPECT_LE(MaxAbs(H * t), 1e-8);
}
}

TEST(CorotatedEnergyTest, GradientDirectionalDerivativeShowsSecondOrderConvergence) {
const TriangleRest rest = MakeRestTriangle();
const TriangleDef def = MakeDeformedTriangle();
const TriangleDef dx = MakeDirection();

const auto g = corotated_node_gradient(rest, def, kMu, kLambda);
const double exact = flatten_gradient(g).dot(flatten_def(dx));

std::vector<double> eps_list = {8e-2, 4e-2, 2e-2, 1e-2};
std::vector<double> errors;

for (double eps : eps_list) {
const TriangleDef def_plus = add_scale(def, dx, eps);
const TriangleDef def_minus = add_scale(def, dx, -eps);

const double fd =
        (corotated_energy(rest, def_plus, kMu, kLambda) -
         corotated_energy(rest, def_minus, kMu, kLambda)) / (2.0 * eps);

errors.push_back(std::abs(fd - exact));
}

for (size_t i = 1; i < errors.size(); ++i) {
EXPECT_LT(errors[i], errors[i - 1]);
EXPECT_GT(errors[i - 1] / errors[i], 3.0);
}
}