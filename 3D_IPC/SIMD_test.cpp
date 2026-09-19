#include "SIMD.h"
#include "broad_phase.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "parallel_helper.h"
#include "simulation.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// Scalar-reference and independent derivative checks for the SIMD kernels.
namespace {

struct Inputs {
    RefMesh mesh;
    std::vector<Vec3> x;
    IncidentTriangles incident;
    std::vector<ShapeGrads> rest_shape_grads;

    void append(const Mat32& F, const Mat22& Dm, int role) {
        const int triangle = static_cast<int>(mesh.area.size());
        const int first = static_cast<int>(x.size());
        const Vec3 origin(0.37, -0.21, 0.13);
        x.push_back(origin);
        x.push_back(origin + F * Dm.col(0));
        x.push_back(origin + F * Dm.col(1));
        mesh.tris.insert(mesh.tris.end(), {first, first + 1, first + 2});
        mesh.Dm_inverse.push_back(Dm.inverse());
        mesh.area.push_back(0.5 * std::abs(Dm.determinant()));
        incident.emplace_back(triangle, role);
        rest_shape_grads.push_back(shape_function_gradients(mesh.Dm_inverse.back()));
    }
};

void reference(const Inputs& input, double mu, double lambda, double dt2,
               Vec3& g, Mat33& H) {
    for (const auto& [triangle, role] : input.incident) {
        const int* v = &input.mesh.tris[3 * triangle];
        Mat32 Ds;
        Ds.col(0) = input.x[v[1]] - input.x[v[0]];
        Ds.col(1) = input.x[v[2]] - input.x[v[0]];
        const Mat32 F = Ds * input.mesh.Dm_inverse[triangle];
        const auto cache = buildCorotatedCache(F);
        const Mat32 P = PCorotated32(cache, F, mu, lambda);
        Mat66 dPdF;
        dPdFCorotated32(cache, mu, lambda, dPdF);
        const auto gradients = shape_function_gradients(input.mesh.Dm_inverse[triangle]);
        g += dt2 * corotated_node_gradient(P, input.mesh.area[triangle], gradients, role);
        H += dt2 * corotated_node_hessian(dPdF, input.mesh.area[triangle], gradients, role);
    }
}

void compare(const Inputs& input, double mu, double lambda, double dt2,
             double tolerance = 2e-11) {
    const Vec3 seed_g(0.013, -0.032, 0.021);
    const Mat33 seed_H = (Mat33() << 2.1, 0.3, 0.2, 0.3, 1.8, -0.1, 0.2, -0.1, 1.2).finished();
    Vec3 expected_g = seed_g;
    Mat33 expected_H = seed_H;
    reference(input, mu, lambda, dt2, expected_g, expected_H);
    ASSERT_TRUE(expected_g.allFinite());
    ASSERT_TRUE(expected_H.allFinite());
    for (bool cached : {false, true}) {
        Vec3 g = seed_g;
        Mat33 H = seed_H;
        ipc_simd::accumulated_corotated_elasticity(input.mesh, input.x, input.incident,
            cached ? &input.rest_shape_grads : nullptr, mu, lambda, dt2, g, H);
        ASSERT_TRUE(g.allFinite());
        ASSERT_TRUE(H.allFinite());
        EXPECT_LE((g - expected_g).norm(), tolerance * (1.0 + expected_g.norm()))
            << "cached=" << cached << " count=" << input.incident.size();
        EXPECT_LE((H - expected_H).norm(), tolerance * (1.0 + expected_H.norm()))
            << "cached=" << cached << " count=" << input.incident.size();
    }
}

Mat33 rotation(double angle) {
    return Eigen::AngleAxisd(angle, Vec3(1.0, -2.0, 0.7).normalized()).toRotationMatrix();
}

} // namespace

TEST(SIMDKernel, EmptyIncidentListPreservesSeed) {
    Inputs input;
    compare(input, 123.0, 456.0, 0.37, 0.0);
    EXPECT_GE(ipc_simd::lane_width(), 1);
    EXPECT_NE(ipc_simd::backend_name(), nullptr);
}

TEST(SIMDKernel, RestAndRotatedRestMatchForAllVertexRoles) {
    for (int role = 0; role < 3; ++role) {
        Inputs input;
        for (int i = 0; i < 3 * ipc_simd::lane_width() + 1; ++i) {
            const Mat32 F = rotation(0.37 * i).leftCols<2>();
            const Mat22 Dm = (Mat22() << 1.1, 0.2, 0.1, 0.9).finished();
            input.append(F, Dm, role);
        }
        compare(input, 46000.0, 46000.0, 1.0 / 22500.0);
        Vec3 g = Vec3::Zero();
        Mat33 H = Mat33::Zero();
        ipc_simd::accumulated_corotated_elasticity(input.mesh, input.x, input.incident,
            &input.rest_shape_grads, 46000.0, 46000.0, 1.0 / 22500.0, g, H);
        EXPECT_LT(g.norm(), 1e-12);
    }
}

TEST(SIMDKernel, RandomStretchShearMaterialsAndBatchTailsMatchScalar) {
    std::mt19937_64 rng(572901);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    for (int sample = 0; sample < 500; ++sample) {
        Inputs input;
        const int count = 1 + sample % (4 * ipc_simd::lane_width() + 1);
        for (int i = 0; i < count; ++i) {
            Mat32 local;
            local << std::exp(1.5 * random(rng)), random(rng),
                     0.2 * random(rng), std::exp(1.5 * random(rng)),
                     0.3 * random(rng), 0.3 * random(rng);
            const Mat32 F = rotation(3.0 * random(rng)) * local;
            Mat22 Dm;
            Dm << 1.0 + 0.5 * random(rng), 0.2 * random(rng),
                  0.2 * random(rng), 1.0 + 0.5 * random(rng);
            input.append(F, Dm, (i + sample) % 3);
        }
        SCOPED_TRACE(sample);
        compare(input, sample % 7 == 0 ? 0.0 : 46000.0,
                sample % 5 == 0 ? 0.0 : 28000.0, 1.0 / 22500.0);
    }
}

TEST(SIMDKernel, NearDegenerateAndClampedLanesPreserveScalarBehavior) {
    // Test the clamp on both sides and mixed SIMD/scalar packets. Keep C
    // invertible because the existing reference itself is undefined at rank 1.
    for (double small : {2e-4, 2e-5, 2e-6, 1.01e-6, 0.99e-6, 2e-7}) {
        for (bool mixed : {false, true}) {
            Inputs input;
            for (int i = 0; i < 2 * ipc_simd::lane_width() + 1; ++i) {
                Mat32 F = Mat32::Zero();
                F(0, 0) = 1.3;
                F(1, 1) = mixed && i % 2 ? 0.9 : small;
                F = rotation(0.12 * i) * F;
                input.append(F, Mat22::Identity(), i % 3);
            }
            SCOPED_TRACE(small);
            SCOPED_TRACE(mixed);
            compare(input, 46000.0, 28000.0, 1.0 / 22500.0, 2e-10);
        }
    }
}

TEST(SIMDKernel, SkewAnisotropicInputsNearFallbackBoundary) {
    for (double small : {4e-3, 1.3e-3, 4e-4, 1.3e-4, 4e-5}) {
        for (int sample = 0; sample < 20; ++sample) {
            Inputs input;
            Mat22 stretch = Mat22::Zero();
            stretch(0, 0) = 1.0;
            stretch(1, 1) = small;
            const double angle = 0.047 + 0.12 * sample;
            Mat22 right;
            right << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
            const Mat32 F = rotation(0.3 * sample).leftCols<2>() * stretch * right;
            input.append(F, Mat22::Identity(), sample % 3);
            SCOPED_TRACE(small);
            SCOPED_TRACE(sample);
            compare(input, 46000.0, 28000.0, 1.0 / 22500.0, 2e-10);
        }
    }
}

TEST(SIMDKernel, ExactSelfHessianMatchesFiniteDifferencesOfGradient) {
    for (int role = 0; role < 3; ++role) {
        Inputs input;
        Mat32 F;
        F << 1.2, 0.3, -0.1, 0.8, 0.2, -0.3;
        Mat22 Dm;
        Dm << 1.1, 0.2, -0.1, 0.9;
        input.append(F, Dm, role);
        const auto evaluate = [&]() {
            Vec3 g = Vec3::Zero();
            Mat33 H = Mat33::Zero();
            ipc_simd::accumulated_corotated_elasticity(input.mesh, input.x, input.incident,
                &input.rest_shape_grads, 2.3, 4.7, 1.0, g, H);
            return std::make_pair(g, H);
        };
        const Mat33 H = evaluate().second;
        Mat33 numerical;
        const double step = 1e-5;
        for (int col = 0; col < 3; ++col) {
            const double old = input.x[role][col];
            input.x[role][col] = old + step;
            const Vec3 positive = evaluate().first;
            input.x[role][col] = old - step;
            const Vec3 negative = evaluate().first;
            input.x[role][col] = old;
            numerical.col(col) = (positive - negative) / (2.0 * step);
        }
        EXPECT_LE((H - numerical).norm(), 2e-8 * (1.0 + H.norm()));
        EXPECT_LE((H - H.transpose()).norm(), 1e-12 * (1.0 + H.norm()));
    }
}

namespace {

std::uint64_t ordered_double(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits & (std::uint64_t(1) << 63) ? ~bits : bits | (std::uint64_t(1) << 63);
}

void check_atan2(const std::vector<double>& y, const std::vector<double>& x) {
    ASSERT_EQ(y.size(), x.size());
    std::vector<double> actual(y.size(), 123.0);
    ipc_simd::atan2_batch(y.data(), x.data(), actual.data(), y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double expected = std::atan2(y[i], x[i]);
        SCOPED_TRACE(::testing::Message() << "i=" << i << " y=" << y[i] << " x=" << x[i]);
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual[i]));
        } else {
            ASSERT_TRUE(std::isfinite(actual[i]));
            const auto first = ordered_double(expected), second = ordered_double(actual[i]);
            const auto ulps = first >= second ? first - second : second - first;
            EXPECT_LE(ulps, 4u) << "expected=" << expected << " actual=" << actual[i];
            if (expected == 0.0) {
                EXPECT_EQ(actual[i], 0.0);
                EXPECT_EQ(std::signbit(actual[i]), std::signbit(expected));
            }
        }
    }
}

struct BendingInputs {
    RefMesh mesh;
    std::vector<Vec3> positions;
    std::vector<std::pair<int, int>> incident;

    void append(const HingeDef& def, int role, double c_e, double bar_theta) {
        Hinge hinge;
        for (int node = 0; node < 4; ++node) {
            hinge.v[node] = static_cast<int>(positions.size());
            positions.push_back(def.x[node]);
        }
        hinge.c_e = c_e;
        hinge.bar_theta = bar_theta;
        incident.emplace_back(static_cast<int>(mesh.hinges.size()), role);
        mesh.hinges.push_back(hinge);
    }

    HingeDef def(int index) const {
        HingeDef output;
        for (int node = 0; node < 4; ++node)
            output.x[node] = positions[mesh.hinges[index].v[node]];
        return output;
    }
};

HingeDef folded_hinge(double angle) {
    HingeDef def;
    def.x[0] = Vec3(0.13, -0.2, 0.3);
    def.x[1] = def.x[0] + Vec3(1.1, 0.0, 0.0);
    def.x[2] = def.x[0] + Vec3(0.24, 0.8, 0.0);
    def.x[3] = def.x[0] + Vec3(0.4, -0.7 * std::cos(angle), 0.7 * std::sin(angle));
    return def;
}

std::pair<Vec3, Mat33> evaluate_bending(const BendingInputs& input, double kB, double dt2,
    const Vec3& seed_g = Vec3::Zero(), const Mat33& seed_H = Mat33::Zero()) {
    Vec3 g = seed_g;
    Mat33 H = seed_H;
    ipc_simd::accumulate_bending(input.mesh, input.positions, input.incident, kB, dt2, g, H);
    return {g, H};
}

void compare_bending(const BendingInputs& input, double kB, double dt2) {
    const Vec3 seed_g(0.013, -0.032, 0.021);
    const Mat33 seed_H = (Mat33() << 2.1, 0.3, 0.2, 0.3, 1.8, -0.1, 0.2, -0.1, 1.2).finished();
    Vec3 reference_g = seed_g;
    Mat33 reference_H = seed_H;
    for (const auto& [index, role] : input.incident) {
        const auto& hinge = input.mesh.hinges[index];
        const auto [g, H] = bending_node_gradient_hessian_psd(
            input.def(index), kB, hinge.c_e, hinge.bar_theta, role);
        reference_g += dt2 * g;
        reference_H += dt2 * H;
    }
    const auto [g, H] = evaluate_bending(input, kB, dt2, seed_g, seed_H);
    ASSERT_TRUE(g.allFinite());
    ASSERT_TRUE(H.allFinite());
    EXPECT_LE((g - reference_g).norm(), 2e-11 * (1.0 + reference_g.norm()));
    EXPECT_LE((H - reference_H).norm(), 2e-11 * (1.0 + reference_H.norm()));
}

} // namespace

TEST(SIMDKernel, PackedAtan2DenseAnglesDynamicRangeAndTailsMatchLibm) {
    std::vector<double> y, x;
    const double pi = std::acos(-1.0);
    for (int i = 0; i <= 20000; ++i) {
        const double angle = -pi + 2.0 * pi * i / 20000.0;
        x.push_back(std::cos(angle));
        y.push_back(std::sin(angle));
    }
    std::mt19937_64 generator(187651);
    std::uniform_real_distribution<double> mantissa(-1.0, 1.0);
    std::uniform_int_distribution<int> exponent(-1000, 1000);
    for (int i = 0; i < 10000; ++i) {
        x.push_back(std::ldexp(mantissa(generator), exponent(generator)));
        y.push_back(std::ldexp(mantissa(generator), exponent(generator)));
    }
    check_atan2(y, x);
    for (int size = 0; size < 3 * ipc_simd::lane_width() + 1; ++size)
        check_atan2(std::vector<double>(y.begin(), y.begin() + size),
                    std::vector<double>(x.begin(), x.begin() + size));
}

TEST(SIMDKernel, PackedAtan2PreservesAxesSignedZeroAndSpecialValues) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double tiny = std::numeric_limits<double>::denorm_min();
    const std::array<double, 11> values = {0.0, -0.0, 1.0, -1.0, inf, -inf, nan,
        tiny, -tiny, std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
    std::vector<double> y, x;
    for (double ordinate : values)
        for (double abscissa : values) {
            y.push_back(ordinate);
            x.push_back(abscissa);
        }
    check_atan2(y, x);
}

TEST(SIMDKernel, PointEnergyTermsMatchQuadraticReferenceAcrossScalesAndPins) {
    std::mt19937_64 generator(627981);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    for (int sample = 0; sample < 400; ++sample) {
        SCOPED_TRACE(sample);
        const auto vector = [&]() { return Vec3(random(generator), random(generator), random(generator)); };
        const Vec3 x = 100.0 * vector(), xhat = 100.0 * vector();
        const Vec3 gravity = 20.0 * vector(), target = 100.0 * vector();
        const Vec3* pin = sample % 3 == 0 ? nullptr : &target;
        const double mass = sample % 11 == 0 ? 0.0 : std::pow(10.0, 8.0 * random(generator));
        const double kpin = sample % 7 == 0 ? 0.0 : std::pow(10.0, 9.0 * random(generator));
        const double dt2 = sample % 13 == 0 ? 0.0 : std::pow(10.0, -4.0 + 3.0 * random(generator));
        const Vec3 seed_g = vector();
        const Mat33 seed_H = (Mat33() << 1.1, .3, -.2, .4, 1.2, .7, -.1, .8, 1.4).finished();
        Vec3 expected_g = seed_g;
        Mat33 expected_H = seed_H;
        expected_g += mass * (x - xhat);
        expected_g += dt2 * (-mass * gravity);
        expected_H += mass * Mat33::Identity();
        if (pin != nullptr) {
            expected_g += dt2 * kpin * (x - *pin);
            expected_H += dt2 * kpin * Mat33::Identity();
        }
        Vec3 g = seed_g;
        Mat33 H = seed_H;
        ipc_simd::accumulate_point_terms(mass, x, xhat, gravity, pin, kpin, dt2, g, H);
        ASSERT_TRUE(g.allFinite());
        ASSERT_TRUE(H.allFinite());
        EXPECT_LE((g - expected_g).norm(), 2e-13 * (1.0 + expected_g.norm()));
        EXPECT_LE((H - expected_H).norm(), 2e-13 * (1.0 + expected_H.norm()));
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                if (row != col) EXPECT_EQ(H(row, col), seed_H(row, col));
    }
}

TEST(SIMDKernel, PointGradientAndHessianMatchIndependentEnergyDifferences) {
    const Vec3 x(.3, -.8, .4), xhat(-.2, .4, .7), gravity(.3, -9.81, -.6);
    const Vec3 target(.5, -.1, .2);
    constexpr double mass = .73, dt2 = .012, kpin = 23.0;
    for (bool pinned : {false, true}) {
        const Vec3* pin = pinned ? &target : nullptr;
        const auto evaluate = [&](const Vec3& position) {
            Vec3 g = Vec3::Zero();
            Mat33 H = Mat33::Zero();
            ipc_simd::accumulate_point_terms(mass, position, xhat, gravity, pin, kpin, dt2, g, H);
            return std::make_pair(g, H);
        };
        const auto energy = [&](const Vec3& position) {
            double value = .5 * mass * (position - xhat).squaredNorm()
                - dt2 * mass * gravity.dot(position);
            if (pinned) value += .5 * dt2 * kpin * (position - target).squaredNorm();
            return value;
        };
        const auto [g, H] = evaluate(x);
        Vec3 numerical_g;
        Mat33 numerical_H;
        constexpr double step = 1e-5;
        for (int axis = 0; axis < 3; ++axis) {
            Vec3 plus = x, minus = x;
            plus[axis] += step;
            minus[axis] -= step;
            numerical_g[axis] = (energy(plus) - energy(minus)) / (2.0 * step);
            numerical_H.col(axis) = (evaluate(plus).first - evaluate(minus).first) / (2.0 * step);
        }
        EXPECT_LE((g - numerical_g).norm(), 2e-10 * (1.0 + g.norm()));
        EXPECT_LE((H - numerical_H).norm(), 2e-10 * (1.0 + H.norm()));
    }
}

TEST(SIMDKernel, BendingRandomHingesAllRolesRestAnglesAndTailsMatchScalar) {
    std::mt19937_64 generator(76135);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    for (int sample = 0; sample < 400; ++sample) {
        BendingInputs input;
        const int count = sample % (4 * ipc_simd::lane_width() + 1);
        for (int index = 0; index < count; ++index) {
            HingeDef def = folded_hinge(2.8 * random(generator));
            const Mat33 transform = rotation(3.0 * random(generator))
                * Vec3(std::exp(random(generator)), std::exp(random(generator)),
                    std::exp(random(generator))).asDiagonal();
            for (Vec3& x : def.x) x = transform * x;
            input.append(def, (sample + index) % 4, std::exp(2.0 * random(generator)),
                3.0 * random(generator));
        }
        SCOPED_TRACE(sample);
        compare_bending(input, sample % 13 == 0 ? 0.0 : .009, 1.0 / 22500.0);
    }
}

TEST(SIMDKernel, BendingDegenerateAndThinLanesMatchScalarWithoutContaminatingNeighbors) {
    for (int role = 0; role < 4; ++role) {
        BendingInputs input;
        for (int index = 0; index < 4 * ipc_simd::lane_width() + 3; ++index) {
            HingeDef def = folded_hinge(.4);
            switch (index % 6) {
                case 0: def.x[1] = def.x[0]; break;
                case 1: def.x[2] = def.x[0]; break;
                case 2: def.x[3] = def.x[1]; break;
                case 3: def.x[2] = def.x[0] + .25 * (def.x[1] - def.x[0]); break;
                case 4: def.x[2] = def.x[0] + Vec3(.24, 1e-7, 0.0); break;
                default: break;
            }
            input.append(def, role, 1.7, -.31);
        }
        SCOPED_TRACE(role);
        compare_bending(input, .009, 1.0 / 22500.0);
        BendingInputs degenerate;
        HingeDef zero = folded_hinge(.4);
        zero.x[1] = zero.x[0];
        degenerate.append(zero, role, 1.7, -.31);
        const Vec3 seed_g(.1, -.2, .3);
        const Mat33 seed_H = Mat33::Identity();
        const auto [g, H] = evaluate_bending(degenerate, .009, .4, seed_g, seed_H);
        EXPECT_EQ((g - seed_g).norm(), 0.0);
        EXPECT_EQ((H - seed_H).norm(), 0.0);
    }
}

TEST(SIMDKernel, BendingNearAngleBranchPreservesUnwrappedRestAngleDifference) {
    const double pi = std::acos(-1.0);
    for (int role = 0; role < 4; ++role) {
        BendingInputs input;
        for (double angle : {-pi - 1e-8, -pi + 1e-8, -pi + 1e-12,
                              pi - 1e-12, pi - 1e-8, pi + 1e-8})
            for (double rest : {-3.0, 0.0, 3.0})
                input.append(folded_hinge(angle), role, 1.7, rest);
        compare_bending(input, .009, .03);
    }
}

TEST(SIMDKernel, BendingNonfiniteInputsPreserveReferencePropagation) {
    for (double nonfinite : {std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()}) {
        for (int changed_node = 0; changed_node < 4; ++changed_node) {
            for (int role = 0; role < 4; ++role) {
                BendingInputs input;
                HingeDef def = folded_hinge(.4);
                def.x[changed_node][1] = nonfinite;
                input.append(def, role, 1.7, -.31);
                const auto expected = bending_node_gradient_hessian_psd(def, .009, 1.7, -.31, role);
                const auto actual = evaluate_bending(input, .009, 1.0);
                const auto compare_value = [&](double value, double reference) {
                    if (std::isnan(reference)) EXPECT_TRUE(std::isnan(value));
                    else if (std::isinf(reference)) EXPECT_EQ(value, reference);
                    else EXPECT_NEAR(value, reference, 2e-11 * (1.0 + std::abs(reference)));
                };
                for (int row = 0; row < 3; ++row) {
                    compare_value(actual.first[row], expected.first[row]);
                    for (int col = 0; col < 3; ++col)
                        compare_value(actual.second(row, col), expected.second(row, col));
                }
            }
        }
    }
}

TEST(SIMDKernel, BendingGradientMatchesEnergyDifferencesAndGaussNewtonBlockIsPsd) {
    for (int role = 0; role < 4; ++role) {
        BendingInputs input;
        input.append(folded_hinge(.63), role, 1.7, -.27);
        constexpr double kB = .43, dt2 = .17, step = 1e-5;
        const auto [g, H] = evaluate_bending(input, kB, dt2);
        Vec3 numerical_g;
        const auto energy = [&]() { return dt2 * bending_energy(input.def(0), kB, 1.7, -.27); };
        const int vertex = input.mesh.hinges[0].v[role];
        for (int axis = 0; axis < 3; ++axis) {
            const double saved = input.positions[vertex][axis];
            input.positions[vertex][axis] = saved + step;
            const double positive = energy();
            input.positions[vertex][axis] = saved - step;
            const double negative = energy();
            input.positions[vertex][axis] = saved;
            numerical_g[axis] = (positive - negative) / (2.0 * step);
        }
        EXPECT_LE((g - numerical_g).norm(), 2e-8 * (1.0 + g.norm()));
        EXPECT_LE((H - H.transpose()).norm(), 1e-12 * (1.0 + H.norm()));
        Eigen::SelfAdjointEigenSolver<Mat33> eigen(H);
        ASSERT_EQ(eigen.info(), Eigen::Success);
        EXPECT_GE(eigen.eigenvalues()[0], -1e-12 * (1.0 + H.norm()));
        EXPECT_LE(std::abs(eigen.eigenvalues()[1]), 1e-12 * (1.0 + H.norm()));

        // At zero angle residual the existing Gauss-Newton block equals the
        // exact Hessian, allowing an independent derivative check for all roles.
        input.mesh.hinges[0].bar_theta = bending_theta(input.def(0));
        const auto at_rest = evaluate_bending(input, kB, dt2);
        EXPECT_LE(at_rest.first.norm(), 1e-12);
        Mat33 numerical_H;
        for (int axis = 0; axis < 3; ++axis) {
            const double saved = input.positions[vertex][axis];
            input.positions[vertex][axis] = saved + step;
            const Vec3 positive = evaluate_bending(input, kB, dt2).first;
            input.positions[vertex][axis] = saved - step;
            const Vec3 negative = evaluate_bending(input, kB, dt2).first;
            input.positions[vertex][axis] = saved;
            numerical_H.col(axis) = (positive - negative) / (2.0 * step);
        }
        EXPECT_LE((at_rest.second - numerical_H).norm(), 2e-8 * (1.0 + at_rest.second.norm()));
    }
}

// Collision-off solver integration, scheduling, and trajectory checks.
namespace {

struct RestoreOpenMPSettings {
    const int threads = omp_get_max_threads();
    const int dynamic = omp_get_dynamic();
    ~RestoreOpenMPSettings() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

struct ClothScene {
    RefMesh mesh;
    DeformedState state;
    std::vector<Pin> pins;
    VertexTriangleMap adjacency;
    BroadPhase broad_phase;
    SimParams params;
};

void build_scene(ClothScene& scene) {
    auto& params = scene.params;
    params = SimParams::zeros();
    params.fps = 30.0;
    params.substeps = 2;
    params.mu = 46.0;
    params.lambda = 46.0;
    params.density = 1000.0;
    params.thickness = 0.001;
    params.kB = 0.009;
    params.kpin = 1e9;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.node_box_min = 0.001;
    params.node_box_max = 0.01;
    params.node_box_update_count = 3;
    params.max_global_iters = 6;
    params.fixed_iters = true;
    params.use_basic_experimental = true;
    params.use_parallel = true;
    params.use_ccd = false;
    params.use_ccd_guess = false;
    params.d_hat = 0.0;
    params.k_barrier = 0.0;
    params.k_sdf = 0.0;
    params.friction_coefficient = 0.0;

    constexpr int nx = 13;
    constexpr int ny = 12;
    std::vector<Vec2> material;
    build_square_mesh(scene.mesh, scene.state, material, nx, ny,
        0.55, 0.45, Vec3(-0.27, 0.03, -0.22));
    // Construct flat rest hinges before applying both in-plane strain and
    // curvature. Boundary and interior nodes have different lane-tail sizes.
    for (Vec3& x : scene.state.deformed_positions) {
        x.x() *= 1.025;
        x.z() += 0.015 * x.x();
        x.y() += 0.006 * std::sin(6.0 * x.x()) * std::cos(7.0 * x.z());
        scene.state.velocities.emplace_back(
            0.012 * x.z(), -0.025, 0.01 * x.x());
    }
    append_pin(scene.pins, 0, scene.state.deformed_positions);
    append_pin(scene.pins, nx, scene.state.deformed_positions);
    scene.pins.front().target_position += Vec3(0.001, 0.002, -0.001);
    scene.pins.back().target_position += Vec3(-0.001, -0.001, 0.002);
    scene.mesh.build_lumped_mass(params.density, params.thickness);
    scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
}

void expect_state_near(const DeformedState& actual,
    const DeformedState& reference) {
    ASSERT_EQ(actual.deformed_positions.size(), reference.deformed_positions.size());
    ASSERT_EQ(actual.velocities.size(), reference.velocities.size());
    for (std::size_t node = 0; node < actual.deformed_positions.size(); ++node) {
        ASSERT_TRUE(actual.deformed_positions[node].allFinite()) << "node=" << node;
        ASSERT_TRUE(actual.velocities[node].allFinite()) << "node=" << node;
        EXPECT_LE((actual.deformed_positions[node]
            - reference.deformed_positions[node]).cwiseAbs().maxCoeff(), 1e-10)
            << "position node=" << node;
        EXPECT_LE((actual.velocities[node]
            - reference.velocities[node]).cwiseAbs().maxCoeff(), 1e-8)
            << "velocity node=" << node;
    }
}

void expect_state_bitwise_equal(const DeformedState& actual,
    const DeformedState& reference) {
    ASSERT_EQ(actual.deformed_positions.size(), reference.deformed_positions.size());
    ASSERT_EQ(actual.velocities.size(), reference.velocities.size());
    for (std::size_t node = 0; node < actual.deformed_positions.size(); ++node) {
        EXPECT_EQ(std::memcmp(actual.deformed_positions[node].data(),
            reference.deformed_positions[node].data(), 3 * sizeof(double)), 0)
            << "position node=" << node;
        EXPECT_EQ(std::memcmp(actual.velocities[node].data(),
            reference.velocities[node].data(), 3 * sizeof(double)), 0)
            << "velocity node=" << node;
    }
}

} // namespace

TEST(SIMDSolver, ActivationRequiresExperimentalCollisionOffCloth) {
    ClothScene scene;
    build_scene(scene);
    scene.params.use_simd = true;
    ASSERT_TRUE(physics_detail::collision_off_energy_simd_enabled(scene.mesh, scene.params));

    const auto enabled = scene.params;
    for (bool SimParams::*flag : {&SimParams::use_simd,
             &SimParams::use_basic_experimental}) {
        auto params = enabled;
        params.*flag = false;
        EXPECT_FALSE(physics_detail::collision_off_energy_simd_enabled(scene.mesh, params));
    }
    for (bool SimParams::*flag : {&SimParams::use_ccd,
             &SimParams::use_ccd_guess, &SimParams::use_cloth_grid,
             &SimParams::use_ogc, &SimParams::use_ogc_solver}) {
        auto params = enabled;
        params.*flag = true;
        EXPECT_FALSE(physics_detail::collision_off_energy_simd_enabled(scene.mesh, params));
    }
    for (double SimParams::*coefficient : {&SimParams::d_hat,
             &SimParams::k_barrier, &SimParams::k_sdf,
             &SimParams::friction_coefficient}) {
        auto params = enabled;
        params.*coefficient = 0.01;
        EXPECT_FALSE(physics_detail::collision_off_energy_simd_enabled(scene.mesh, params));
    }
    scene.mesh.tets = {0, 1, 2, 3};
    EXPECT_FALSE(physics_detail::collision_off_energy_simd_enabled(scene.mesh, enabled));
    scene.mesh.tets.clear();
    scene.mesh.rb_nodes = {{0, 1, 2}};
    EXPECT_FALSE(physics_detail::collision_off_energy_simd_enabled(scene.mesh, enabled));
}

TEST(SIMDSolver, AllFiveEnergyTermsMatchCompleteScalarLocalAssembly) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    ClothScene scene;
    build_scene(scene);
    scene.params.use_simd = true;
    scene.params.gravity = Vec3(.7, -9.81, -.4);
    // Nonzero rest angles exercise the bending energy residual independently
    // of the reference fixture's initially flat material configuration.
    for (std::size_t i = 0; i < scene.mesh.hinges.size(); ++i)
        scene.mesh.hinges[i].bar_theta = .08 * std::sin(.37 * i);
    for (std::size_t i = 0; i < scene.mesh.mass.size(); ++i)
        scene.mesh.mass[i] *= .5 + .05 * (i % 17);
    std::vector<Vec3> predictor;
    build_xhat(predictor, scene.state.deformed_positions, scene.state.velocities, scene.params.dt());
    PinMap pin_map(scene.state.deformed_positions.size(), -1);
    for (std::size_t index = 0; index < scene.pins.size(); ++index)
        pin_map[scene.pins[index].vertex_index] = static_cast<int>(index);
    std::vector<ShapeGrads> rest_grads;
    for (const auto& inverse : scene.mesh.Dm_inverse)
        rest_grads.push_back(shape_function_gradients(inverse));

    for (std::size_t node = 0; node < scene.state.deformed_positions.size(); ++node) {
        SCOPED_TRACE(::testing::Message() << "node=" << node);
        // The public checked assembly is the scalar reference even when the
        // experimental opt-in flag is set. Check both cached and uncached
        // adjacency/shape-gradient/pin lookup paths against the same inputs.
        const auto expected = compute_local_gradient_and_hessian_no_barrier(
            static_cast<int>(node), scene.mesh, scene.adjacency, scene.pins,
            scene.params, scene.state.deformed_positions, predictor,
            nullptr, nullptr, nullptr, nullptr);
        for (bool cached : {false, true}) {
            const auto actual = physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(
                static_cast<int>(node), scene.mesh, scene.adjacency, scene.pins,
                scene.params, scene.state.deformed_positions, predictor,
                cached ? &pin_map : nullptr,
                cached ? &scene.adjacency.at(static_cast<int>(node)) : nullptr,
                cached ? &rest_grads : nullptr, nullptr);
            ASSERT_TRUE(actual.first.allFinite());
            ASSERT_TRUE(actual.second.allFinite());
            EXPECT_LE((actual.first - expected.first).norm(),
                2e-11 * (1.0 + expected.first.norm())) << "cached=" << cached;
            EXPECT_LE((actual.second - expected.second).norm(),
                2e-11 * (1.0 + expected.second.norm())) << "cached=" << cached;
        }
    }
}

TEST(SIMDSolver, CollisionOffFixedFramesMatchScalarAndAreThreadDeterministic) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    constexpr int frames = 3;
    constexpr std::array<int, 4> thread_counts = {1, 1, 4, 4};
    // Cover per-sweep rebuilds and incomplete final batches at two intervals.
    constexpr std::array<int, 3> rebuild_intervals = {1, 3, 10};
    constexpr std::array<int, 3> iteration_limits = {6, 7, 13};
    // Keep each mesh identity and its allocation alive: the solver caches
    // adaptive node-box history by mesh identity across substeps and frames.
    std::array<std::array<ClothScene, 4>, rebuild_intervals.size()> scenes;
    for (auto& configuration : scenes)
        for (auto& scene : configuration)
            build_scene(scene);

    for (std::size_t configuration = 0; configuration < scenes.size(); ++configuration) {
        const int rebuild_interval = rebuild_intervals[configuration];
        std::array<std::array<DeformedState, frames>, 4> snapshots;
        for (std::size_t run = 0; run < thread_counts.size(); ++run) {
            SCOPED_TRACE(::testing::Message() << "rebuild=" << rebuild_interval
                << " run=" << run << " threads=" << thread_counts[run]);
            auto& scene = scenes[configuration][run];
            scene.params.use_simd = run != 0;
            scene.params.node_box_update_count = rebuild_interval;
            scene.params.max_global_iters = iteration_limits[configuration];
            omp_set_num_threads(thread_counts[run]);
            ASSERT_GT(scene.state.deformed_positions.size(), 128u);
            ASSERT_FALSE(scene.mesh.hinges.empty());
            std::vector<std::vector<int>> colors;
            greedy_color_conflict_graph(build_elastic_adj(scene.mesh,
                scene.adjacency, static_cast<int>(scene.state.deformed_positions.size())), colors);
            ASSERT_GT(colors.size(), 1u);
            const auto initial = scene.state.deformed_positions;

            for (int frame = 1; frame <= frames; ++frame) {
                SCOPED_TRACE(::testing::Message() << "frame=" << frame);
                int callbacks = 0;
                const auto result = advance_one_frame(scene.state, scene.mesh,
                    scene.adjacency, scene.pins, scene.params, scene.broad_phase,
                    frame, nullptr,
                    [&](int substep, const std::vector<Vec3>& positions) {
                        EXPECT_EQ(substep, (frame - 1) * scene.params.substeps + callbacks++);
                        const auto& cache = scene.broad_phase.cache();
                        EXPECT_TRUE(cache.nt_pairs.empty());
                        EXPECT_TRUE(cache.ss_pairs.empty());
                        ASSERT_EQ(cache.node_boxes.size(), positions.size());
                        for (std::size_t node = 0; node < positions.size(); ++node) {
                            EXPECT_TRUE((positions[node].array()
                                >= cache.node_boxes[node].min.array() - 1e-12).all());
                            EXPECT_TRUE((positions[node].array()
                                <= cache.node_boxes[node].max.array() + 1e-12).all());
                        }
                    });
                ASSERT_TRUE(result.converged);
                EXPECT_FALSE(result.has_residual);
                EXPECT_EQ(result.iterations,
                    scene.params.substeps * scene.params.max_global_iters);
                EXPECT_EQ(callbacks, scene.params.substeps);
                snapshots[run][frame - 1] = scene.state;
                if (run != 0)
                    expect_state_near(scene.state, snapshots[0][frame - 1]);
                if (run > 1)
                    expect_state_bitwise_equal(scene.state, snapshots[1][frame - 1]);
            }
            double movement = 0.0;
            for (std::size_t node = 0; node < initial.size(); ++node)
                movement += (scene.state.deformed_positions[node] - initial[node]).squaredNorm();
            EXPECT_GT(movement, 1e-8);
        }
    }
}

TEST(SIMDSolver, SerialVertexOrderMatchesScalarWithSimdEnabled) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    std::array<ClothScene, 2> scenes;
    for (auto& scene : scenes) {
        build_scene(scene);
        // The serial solver intentionally uses vertex order, while parallel
        // mode uses color order even at one thread. Compare like schedules.
        scene.params.use_parallel = false;
    }
    for (std::size_t run = 0; run < scenes.size(); ++run) {
        auto& scene = scenes[run];
        scene.params.use_simd = run != 0;
        const auto result = advance_one_frame(scene.state, scene.mesh,
            scene.adjacency, scene.pins, scene.params, scene.broad_phase);
        ASSERT_TRUE(result.converged);
        EXPECT_EQ(result.iterations,
            scene.params.substeps * scene.params.max_global_iters);
    }
    expect_state_near(scenes[1].state, scenes[0].state);
}

TEST(SIMDSolver, ResidualStoppingMatchesScalarAcrossThreadCounts) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    std::array<ClothScene, 3> scenes;
    std::array<SolverResult, 3> results;
    std::array<std::vector<Vec3>, 3> positions;
    for (auto& scene : scenes) {
        build_scene(scene);
        scene.params.fixed_iters = false;
        scene.params.tol_abs = 1e-9;
        scene.params.tol_rel = 0.1;
        scene.params.max_global_iters = 40;
        scene.params.node_box_update_count = 1;
        // Avoid a dominant initial pin residual that vanishes in one sweep;
        // require the elasticity/gravity residual to drive actual stopping.
        for (auto& pin : scene.pins)
            pin.target_position = scene.state.deformed_positions[pin.vertex_index];
    }
    for (std::size_t run = 0; run < scenes.size(); ++run) {
        SCOPED_TRACE(::testing::Message() << "run=" << run);
        auto& scene = scenes[run];
        scene.params.use_simd = run != 0;
        omp_set_num_threads(run == 2 ? 4 : 1);
        positions[run] = scene.state.deformed_positions;
        std::vector<Vec3> predictor;
        build_xhat(predictor, positions[run], scene.state.velocities, scene.params.dt());
        results[run] = global_gauss_seidel_solver_basic_experimental(
            scene.mesh, scene.adjacency, scene.pins, scene.params,
            positions[run], predictor, scene.state.velocities, scene.broad_phase,
            "", &scene.state.deformed_positions);
        const auto& result = results[run];
        ASSERT_TRUE(result.converged);
        ASSERT_TRUE(result.has_residual);
        EXPECT_GT(result.iterations, 1);
        EXPECT_LT(result.iterations, scene.params.max_global_iters);
        EXPECT_TRUE(std::isfinite(result.final_residual));
        EXPECT_LT(result.final_residual, scene.params.tol_rel * result.initial_residual);
        if (run == 0) continue;
        EXPECT_EQ(result.iterations, results[0].iterations);
        EXPECT_DOUBLE_EQ(result.initial_residual, results[0].initial_residual);
        EXPECT_NEAR(result.final_residual, results[0].final_residual,
            1e-10 * std::max(1.0, results[0].final_residual));
        for (std::size_t node = 0; node < positions[run].size(); ++node) {
            ASSERT_TRUE(positions[run][node].allFinite());
            EXPECT_LE((positions[run][node] - positions[0][node]).cwiseAbs().maxCoeff(), 1e-10)
                << "node=" << node;
            if (run == 2)
                EXPECT_EQ(std::memcmp(positions[run][node].data(),
                    positions[1][node].data(), 3 * sizeof(double)), 0) << "node=" << node;
        }
    }
}
