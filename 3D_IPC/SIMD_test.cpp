#include "SIMD.h"
#include "broad_phase.h"
#include "friction_energy.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "parallel_helper.h"
#include "physics.h"
#include "simulation.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
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

// Gather the fixture's incident records before exercising the production tile.
void accumulate_elasticity_tiles(const Inputs& input,
    const std::vector<ShapeGrads>* rest, double mu, double lambda, double dt2,
    Vec3& g, Mat33& H) {
    constexpr std::size_t width = ipc_simd::tile_width;
    std::array<Vec3, 3 * width> positions;
    std::array<Mat22, width> material;
    std::array<Vec2, width> shape;
    std::array<double, width> area;
    std::array<Vec3, width> gradients;
    std::array<Mat33, width> hessians;
    for (std::size_t begin = 0; begin < input.incident.size(); begin += width) {
        const auto count = std::min(width, input.incident.size() - begin);
        for (std::size_t e = 0; e < count; ++e) {
            const auto [triangle, role] = input.incident[begin + e];
            for (int corner = 0; corner < 3; ++corner)
                positions[3 * e + corner] = input.x[input.mesh.tris[3 * triangle + corner]];
            material[e] = input.mesh.Dm_inverse[triangle];
            area[e] = input.mesh.area[triangle];
            shape[e] = rest ? (*rest)[triangle][role]
                : shape_function_gradients(material[e])[role];
        }
        ipc_simd::corotated_derivatives_tile(positions.data(), material.data(), area.data(),
            shape.data(), count, mu, lambda, gradients.data(), hessians.data());
        for (std::size_t e = 0; e < count; ++e) {
            g += dt2 * gradients[e];
            H += dt2 * hessians[e];
        }
    }
}

void reference_point(const ipc_simd::PointInput& input, const Vec3& gravity,
    double kpin, double dt2, Vec3& g, Mat33& H) {
    g.setZero();
    H.setZero();
    for (int axis = 0; axis < 3; ++axis) {
        g[axis] += input.mass * (input.position[axis] - input.predicted_position[axis]);
        g[axis] += dt2 * (-input.mass * gravity[axis]);
        H(axis, axis) += input.mass;
        if (input.pin_target) {
            g[axis] += (dt2 * kpin) * (input.position[axis] - (*input.pin_target)[axis]);
            H(axis, axis) += dt2 * kpin;
        }
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
        accumulate_elasticity_tiles(input,
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

TEST(SIMDKernel, RestAndRotatedRestMatchForAllVertexRoles) {
    for (int role = 0; role < 3; ++role) {
        Inputs input;
        for (int i = 0; i < 3 * static_cast<int>(ipc_simd::tile_width) + 1; ++i) {
            const Mat32 F = rotation(0.37 * i).leftCols<2>();
            const Mat22 Dm = (Mat22() << 1.1, 0.2, 0.1, 0.9).finished();
            input.append(F, Dm, role);
        }
        compare(input, 46000.0, 46000.0, 1.0 / 22500.0);
        Vec3 g = Vec3::Zero();
        Mat33 H = Mat33::Zero();
        accumulate_elasticity_tiles(input,
            &input.rest_shape_grads, 46000.0, 46000.0, 1.0 / 22500.0, g, H);
        EXPECT_LT(g.norm(), 1e-12);
    }
}

TEST(SIMDKernel, RandomStretchShearMaterialsAndBatchTailsMatchScalar) {
    std::mt19937_64 rng(572901);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    for (int sample = 0; sample < 500; ++sample) {
        Inputs input;
        const int count = 1 + sample % (4 * static_cast<int>(ipc_simd::tile_width) + 1);
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
    // Test the clamp on both sides and mixed regular/clamped packets. Keep C
    // invertible because the existing reference itself is undefined at rank 1.
    for (double small : {2e-4, 2e-5, 2e-6, 1.01e-6, 0.99e-6, 2e-7}) {
        for (bool mixed : {false, true}) {
            Inputs input;
            for (int i = 0; i < 2 * static_cast<int>(ipc_simd::tile_width) + 1; ++i) {
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

TEST(SIMDKernel, SkewAnisotropicInputsMatchScalar) {
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
            accumulate_elasticity_tiles(input,
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
    constexpr std::size_t width = ipc_simd::tile_width;
    std::array<Vec3, 4 * width> positions;
    std::array<int, width> roles;
    std::array<double, width> coefficients, rest_angles;
    std::array<Vec3, width> gradients;
    std::array<Mat33, width> hessians;
    for (std::size_t begin = 0; begin < input.incident.size(); begin += width) {
        const auto count = std::min(width, input.incident.size() - begin);
        for (std::size_t e = 0; e < count; ++e) {
            const auto [index, role] = input.incident[begin + e];
            const auto& hinge = input.mesh.hinges[index];
            for (int corner = 0; corner < 4; ++corner)
                positions[4 * e + corner] = input.positions[hinge.v[corner]];
            roles[e] = role;
            coefficients[e] = hinge.c_e;
            rest_angles[e] = hinge.bar_theta;
        }
        ipc_simd::bending_derivatives_tile(positions.data(), roles.data(), coefficients.data(),
            rest_angles.data(), count, kB, gradients.data(), hessians.data());
        for (std::size_t e = 0; e < count; ++e) {
            g += dt2 * gradients[e];
            H += dt2 * hessians[e];
        }
    }
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
        const ipc_simd::PointInput input{mass, x, xhat,
            pin ? std::optional<Vec3>(*pin) : std::nullopt};
        Vec3 point_g;
        Mat33 point_H;
        ipc_simd::point_derivatives_tile(&input, 1, gravity, kpin, dt2, &point_g, &point_H);
        g += point_g;
        H += point_H;
        ASSERT_TRUE(g.allFinite());
        ASSERT_TRUE(H.allFinite());
        EXPECT_LE((g - expected_g).norm(), 2e-13 * (1.0 + expected_g.norm()));
        EXPECT_LE((H - expected_H).norm(), 2e-13 * (1.0 + expected_H.norm()));
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                if (row != col) EXPECT_EQ(H(row, col), seed_H(row, col));
    }
}

TEST(SIMDKernel, BendingRandomHingesAllRolesRestAnglesAndTailsMatchScalar) {
    std::mt19937_64 generator(76135);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    for (int sample = 0; sample < 400; ++sample) {
        BendingInputs input;
        const int count = sample % (4 * static_cast<int>(ipc_simd::tile_width) + 1);
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
        for (int index = 0; index < 4 * static_cast<int>(ipc_simd::tile_width) + 3; ++index) {
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

TEST(SIMDSolver, AllFiveEnergyTermsMatchCompleteScalarLocalAssembly) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    ClothScene scene;
    build_scene(scene);
    scene.params.use_simd = true;
    scene.params.use_basic_experimental = false;
    scene.params.use_basic_experimental_v2 = true;
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

TEST(SIMDSolver, NoncontactSimdRemainsEnabledWithContactSdfAndFriction) {
    auto params = SimParams::zeros();
    params.use_basic_experimental = true;
    params.use_basic_experimental_v2 = true;
    params.use_simd = true;
    params.d_hat = 0.02;
    params.k_barrier = 3.0;
    params.use_ccd = params.use_ccd_guess = true;
    params.k_sdf = 100.0;
    params.friction_coefficient = 0.4;
    EXPECT_TRUE(physics_detail::energy_simd_enabled(params));
    params.use_basic_experimental_v2 = false;
    EXPECT_FALSE(physics_detail::energy_simd_enabled(params));
    params.use_basic_experimental_v2 = true;
    params.use_simd = false;
    EXPECT_FALSE(physics_detail::energy_simd_enabled(params));
}

TEST(SIMDSolver, SimdEnergyAssemblyPreservesActiveSdfAndFrictionTerms) {
    ClothScene scene;
    build_scene(scene);
    auto& params = scene.params;
    params.use_basic_experimental_v2 = true;
    params.use_simd = true;
    params.d_hat = 0.02;
    params.k_barrier = 1.0;
    params.use_ccd = params.use_ccd_guess = true;
    params.k_sdf = 37.0;
    params.eps_sdf = 0.003;
    params.friction_coefficient = 0.4;
    params.sdf_planes.push_back(PlaneSDF{Vec3(0.0, 0.04, 0.0), Vec3::UnitY()});
    const auto& positions = scene.state.deformed_positions;
    auto previous = positions;
    for (auto& point : previous) point -= params.dt() * Vec3(0.12, -0.03, 0.08);
    std::vector<Vec3> predictor;
    build_xhat(predictor, positions, scene.state.velocities, params.dt());
    const auto pin_map = build_pin_map(scene.pins, static_cast<int>(positions.size()));
    double sdf_contribution = 0.0;
    for (std::size_t node = 0; node < positions.size(); node += 7) {
        const int vi = static_cast<int>(node);
        const auto expected = compute_local_gradient_and_hessian_no_barrier(
            vi, scene.mesh, scene.adjacency, scene.pins, params, positions, predictor,
            &pin_map, nullptr, nullptr, &previous);
        const auto actual = physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(
            vi, scene.mesh, scene.adjacency, scene.pins, params, positions, predictor,
            &pin_map, nullptr, nullptr, &previous);
        ASSERT_TRUE(actual.first.allFinite());
        ASSERT_TRUE(actual.second.allFinite());
        EXPECT_LE((actual.first - expected.first).norm(), 2e-11 * (1.0 + expected.first.norm()));
        EXPECT_LE((actual.second - expected.second).norm(), 2e-11 * (1.0 + expected.second.norm()));
        auto without_sdf = params;
        without_sdf.k_sdf = 0.0;
        const auto no_sdf = compute_local_gradient_and_hessian_no_barrier(
            vi, scene.mesh, scene.adjacency, scene.pins, without_sdf, positions, predictor,
            &pin_map, nullptr, nullptr, &previous);
        sdf_contribution += (expected.first - no_sdf.first).norm();
    }
    EXPECT_GT(sdf_contribution, 1e-8);
}

TEST(SIMDSolver, V1RemainsScalarRegardlessOfSimdOrV2Flags) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<ClothScene, 3> scenes;
    for (auto& scene : scenes) build_scene(scene);
    for (std::size_t run = 0; run < scenes.size(); ++run) {
        auto& scene = scenes[run];
        scene.params.use_simd = run != 0;
        scene.params.use_basic_experimental_v2 = run == 2;
        std::vector<Vec3> predictor;
        build_xhat(predictor, scene.state.deformed_positions,
            scene.state.velocities, scene.params.dt());
        ASSERT_TRUE(global_gauss_seidel_solver_basic_experimental(scene.mesh,
            scene.adjacency, scene.pins, scene.params, scene.state.deformed_positions,
            predictor, scene.state.velocities, scene.broad_phase).converged);
        if (run != 0) expect_state_bitwise_equal(scene.state, scenes[0].state);
    }
}

TEST(SIMDSolver, MixedClothRigidSceneMatchesScalarWithContactAndFriction) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<ClothScene, 2> scenes;
    for (std::size_t mode = 0; mode < scenes.size(); ++mode) {
        auto& scene = scenes[mode];
        build_scene(scene);
        append_rigid_polygon(6, scene.state, scene.mesh, Vec3(0.0, 0.08, 0.0),
            0.035, 1000.0, 0.015, Vec3(0.02, -0.1, 0.01));
        scene.mesh.build_deformable_nodes();
        scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
        scene.params.use_basic_experimental = true;
        scene.params.use_basic_experimental_v2 = mode == 1;
        scene.params.use_simd = mode == 1;
        scene.params.d_hat = 0.02;
        scene.params.k_barrier = 1.0;
        scene.params.use_ccd = true;
        scene.params.use_ticcd = false;
        scene.params.friction_coefficient = 0.2;
        scene.params.theta_box_min = 0.001;
        scene.params.theta_box_max = 0.02;
    }
    for (int frame = 1; frame <= 4; ++frame) {
        for (auto& scene : scenes)
            ASSERT_TRUE(advance_one_frame_general(scene.state, scene.mesh, scene.adjacency,
                scene.pins, scene.params, scene.broad_phase, frame).converged);
        expect_state_near(scenes[1].state, scenes[0].state);
        EXPECT_LE((scenes[1].state.x_coms[0] - scenes[0].state.x_coms[0]).norm(), 1e-10);
        EXPECT_LE((scenes[1].state.orientations[0] - scenes[0].state.orientations[0]).norm(), 1e-10);
        EXPECT_FALSE(scenes[1].broad_phase.cache().nt_pairs.empty());
    }
}

TEST(StoredMembraneAssembly, UsesWeightedBuffersInOrderWithoutReadingMembraneInputs) {
    ClothScene scene;
    build_scene(scene);
    scene.params.fps = 17.0;
    scene.params.substeps = 3;
    scene.params.kB = 0.0;
    scene.params.kpin = 0.0;
    scene.pins.clear();
    std::fill(scene.mesh.mass.begin(), scene.mesh.mass.end(), 0.0);
    // A stale adjacency map must not be consulted, and poisoned rest data
    // must not leak into assembly when weighted derivatives are supplied.
    scene.adjacency.clear();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (auto& inverse : scene.mesh.Dm_inverse) inverse.setConstant(nan);
    std::fill(scene.mesh.area.begin(), scene.mesh.area.end(), nan);
    const IncidentTriangles stale_incident{{0, 0}};
    const std::array<Vec3, 3> gradients{
        Vec3::Constant(1e16), Vec3::Constant(-1e16), Vec3(1.0, 2.0, 3.0)};
    const Mat33 final_hessian = (Mat33() <<
        2.0, 0.1, 0.2, 0.1, 3.0, 0.3, 0.2, 0.3, 4.0).finished();
    const std::array<Mat33, 3> hessians{
        Mat33::Constant(1e16), Mat33::Constant(-1e16), final_hessian};
    const physics_detail::MembraneDerivativeView stored{
        gradients.data(), hessians.data(), gradients.size()};

    for (bool simd : {false, true}) {
        scene.params.use_simd = simd;
        for (bool cached_incident : {false, true}) {
            std::pair<Vec3, Mat33> actual;
            ASSERT_NO_THROW(actual = physics_detail::
                compute_local_gradient_and_hessian_with_stored_membrane_unchecked(
                    0, scene.mesh, scene.adjacency, scene.pins, scene.params,
                    scene.state.deformed_positions, scene.state.deformed_positions,
                    nullptr, cached_incident ? &stale_incident : nullptr,
                    nullptr, nullptr, stored));
            // Cancellation makes a changed reduction order observable. The
            // supplied entries already include dt^2 and area: do not rescale.
            EXPECT_EQ((actual.first - gradients.back()).norm(), 0.0);
            EXPECT_EQ((actual.second - final_hessian).norm(), 0.0);
        }
    }
}

TEST(StoredMembraneAssembly, EmptyViewPreservesPointAndBendingTerms) {
    ClothScene scene;
    build_scene(scene);
    scene.params.use_simd = false;
    std::vector<Vec3> predictor;
    build_xhat(predictor, scene.state.deformed_positions,
        scene.state.velocities, scene.params.dt());
    auto no_membrane = scene.params;
    no_membrane.mu = no_membrane.lambda = 0.0;
    const auto expected = compute_local_gradient_and_hessian_no_barrier(
        0, scene.mesh, scene.adjacency, scene.pins, no_membrane,
        scene.state.deformed_positions, predictor);
    ASSERT_GT(expected.first.norm(), 0.0);
    ASSERT_GT(expected.second.norm(), 0.0);
    scene.adjacency.clear();
    for (bool simd : {false, true}) {
        scene.params.use_simd = simd;
        const auto actual = physics_detail::
            compute_local_gradient_and_hessian_with_stored_membrane_unchecked(
                0, scene.mesh, scene.adjacency, scene.pins, scene.params,
                scene.state.deformed_positions, predictor,
                nullptr, nullptr, nullptr, nullptr, {});
        EXPECT_LE((actual.first - expected.first).norm(),
            2e-11 * (1.0 + expected.first.norm()));
        EXPECT_LE((actual.second - expected.second).norm(),
            2e-11 * (1.0 + expected.second.norm()));
    }
}

TEST(StoredMembraneAssembly, ComputedTriangleEntriesMatchFullAssemblyForAllNodes) {
    ClothScene scene;
    build_scene(scene);
    scene.params.fps = 17.0;
    scene.params.substeps = 3;
    scene.params.gravity = Vec3(0.7, -9.81, -0.4);
    for (std::size_t i = 0; i < scene.mesh.hinges.size(); ++i)
        scene.mesh.hinges[i].bar_theta = 0.08 * std::sin(0.37 * i);
    std::vector<Vec3> predictor;
    build_xhat(predictor, scene.state.deformed_positions,
        scene.state.velocities, scene.params.dt());
    const PinMap pin_map = build_pin_map(scene.pins,
        static_cast<int>(scene.state.deformed_positions.size()));
    const VertexTriangleMap unused_adjacency;
    std::array<bool, 3> visited_roles{};
    for (std::size_t node = 0; node < scene.state.deformed_positions.size(); ++node) {
        SCOPED_TRACE(::testing::Message() << "node=" << node);
        const int vi = static_cast<int>(node);
        std::vector<Vec3> gradients;
        std::vector<Mat33> hessians;
        for (const auto& [triangle, role] : scene.adjacency.at(vi)) {
            visited_roles[role] = true;
            const int* corners = &scene.mesh.tris[3 * triangle];
            const auto& x = scene.state.deformed_positions;
            Mat32 Ds;
            Ds.col(0) = x[corners[1]] - x[corners[0]];
            Ds.col(1) = x[corners[2]] - x[corners[0]];
            const Mat32 F = Ds * scene.mesh.Dm_inverse[triangle];
            const auto cache = buildCorotatedCache(F);
            const auto gradN = shape_function_gradients(scene.mesh.Dm_inverse[triangle]);
            const Mat32 P = PCorotated32(cache, F, scene.params.mu, scene.params.lambda);
            Mat66 dPdF;
            dPdFCorotated32(cache, scene.params.mu, scene.params.lambda, dPdF);
            gradients.push_back(scene.params.dt2() * corotated_node_gradient(
                P, scene.mesh.area[triangle], gradN, role));
            hessians.push_back(scene.params.dt2() * corotated_node_hessian(
                dPdF, scene.mesh.area[triangle], gradN, role));
        }
        const auto expected = compute_local_gradient_and_hessian_no_barrier(
            vi, scene.mesh, scene.adjacency, scene.pins, scene.params,
            scene.state.deformed_positions, predictor);
        const physics_detail::MembraneDerivativeView stored{
            gradients.data(), hessians.data(), gradients.size()};
        for (bool simd : {false, true}) {
            scene.params.use_simd = simd;
            const auto actual = physics_detail::
                compute_local_gradient_and_hessian_with_stored_membrane_unchecked(
                    vi, scene.mesh, unused_adjacency, scene.pins, scene.params,
                    scene.state.deformed_positions, predictor,
                    &pin_map, nullptr, nullptr, nullptr, stored);
            EXPECT_LE((actual.first - expected.first).norm(),
                2e-11 * (1.0 + expected.first.norm()));
            EXPECT_LE((actual.second - expected.second).norm(),
                2e-11 * (1.0 + expected.second.norm()));
        }
    }
    for (bool visited : visited_roles) EXPECT_TRUE(visited);
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
            scene.params.use_basic_experimental = run == 0;
            scene.params.use_basic_experimental_v2 = run != 0;
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
        scene.params.use_basic_experimental = run == 0;
        scene.params.use_basic_experimental_v2 = run != 0;
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
        scene.params.use_basic_experimental = run == 0;
        scene.params.use_basic_experimental_v2 = run != 0;
        omp_set_num_threads(run == 2 ? 4 : 1);
        positions[run] = scene.state.deformed_positions;
        std::vector<Vec3> predictor;
        build_xhat(predictor, positions[run], scene.state.velocities, scene.params.dt());
        const auto solve = run == 0 ? global_gauss_seidel_solver_basic_experimental
            : global_gauss_seidel_solver_basic_experimental_v2;
        results[run] = solve(
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

TEST(SIMDV2Kernel, AoSTriangleTilesMatchScalarAccumulation) {
    for (int count = 0; count <= 3 * static_cast<int>(ipc_simd::tile_width) + 1; ++count) {
        Inputs input;
        for (int i = 0; i < count; ++i) {
            Mat32 F = rotation(0.17 * i).leftCols<2>();
            F.col(0) *= 0.8 + 0.03 * i;
            F.col(1) *= 1.2;
            input.append(F, (Mat22() << 1.1, 0.2, -0.1, 0.9).finished(), i % 3);
        }
        std::reverse(input.incident.begin(), input.incident.end());
        std::vector<Vec3> positions(3 * count), gradients(count + 1, Vec3::Constant(12345.0));
        std::vector<Mat33> hessians(count + 1, Mat33::Constant(12345.0));
        std::vector<Mat22> dm(count);
        std::vector<Vec2> q(count);
        std::vector<double> area(count);
        for (int i = 0; i < count; ++i) {
            const auto [triangle, role] = input.incident[i];
            for (int corner = 0; corner < 3; ++corner)
                positions[3*i+corner] = input.x[input.mesh.tris[3*triangle+corner]];
            dm[i] = input.mesh.Dm_inverse[triangle];
            q[i] = input.rest_shape_grads[triangle][role];
            area[i] = input.mesh.area[triangle];
        }
        Vec3 expected_g(0.13, -0.2, 0.03), actual_g = expected_g;
        Mat33 expected_h = 0.7 * Mat33::Identity(), actual_h = expected_h;
        constexpr double dt2 = 0.004;
        reference(input, 1.3, 2.7, dt2, expected_g, expected_h);
        for (std::size_t begin = 0; begin < static_cast<std::size_t>(count); begin += ipc_simd::tile_width) {
            const auto size = std::min(ipc_simd::tile_width, static_cast<std::size_t>(count) - begin);
            ipc_simd::corotated_derivatives_tile(positions.data()+3*begin, dm.data()+begin,
                area.data()+begin, q.data()+begin, size, 1.3, 2.7,
                gradients.data()+begin, hessians.data()+begin);
        }
        for (int e = 0; e < count; ++e) {
            actual_g += dt2 * gradients[e];
            actual_h += dt2 * hessians[e];
        }
        EXPECT_LE((actual_g - expected_g).norm(), 16.0 * std::numeric_limits<double>::epsilon() * (1.0 + expected_g.norm())) << count;
        EXPECT_LE((actual_h - expected_h).norm(), 16.0 * std::numeric_limits<double>::epsilon() * (1.0 + expected_h.norm())) << count;
        EXPECT_TRUE((gradients.back().array() == 12345.0).all());
        EXPECT_TRUE((hessians.back().array() == 12345.0).all());
    }
}

TEST(SIMDV2Kernel, AoSTriangleClampedInputsMatchScalarWithinRoundoff) {
    for (int count = 0; count <= 3 * static_cast<int>(ipc_simd::tile_width) + 1; ++count) {
        Inputs input;
        for (int i = 0; i < count; ++i) {
            Mat32 F = rotation(0.17 * i).leftCols<2>();
            F.col(0) *= 0.8 + 0.03 * i;
            F.col(1) *= i == 5 ? 1e-7 : 1.2;
            input.append(F, (Mat22() << 1.1, 0.2, -0.1, 0.9).finished(), i % 3);
        }
        std::reverse(input.incident.begin(), input.incident.end());
        std::vector<Vec3> positions(3 * count), gradients(count + 1, Vec3::Constant(12345.0));
        std::vector<Mat33> hessians(count + 1, Mat33::Constant(12345.0));
        std::vector<Mat22> dm(count);
        std::vector<Vec2> q(count);
        std::vector<double> area(count);
        for (int i = 0; i < count; ++i) {
            const auto [triangle, role] = input.incident[i];
            for (int corner = 0; corner < 3; ++corner)
                positions[3*i+corner] = input.x[input.mesh.tris[3*triangle+corner]];
            dm[i] = input.mesh.Dm_inverse[triangle];
            q[i] = input.rest_shape_grads[triangle][role];
            area[i] = input.mesh.area[triangle];
        }
        Vec3 expected_g(0.13, -0.2, 0.03), actual_g = expected_g;
        Mat33 expected_h = 0.7 * Mat33::Identity(), actual_h = expected_h;
        constexpr double dt2 = 0.004;
        reference(input, 1.3, 2.7, dt2, expected_g, expected_h);
        for (std::size_t begin = 0; begin < static_cast<std::size_t>(count); begin += ipc_simd::tile_width) {
            const auto size = std::min(ipc_simd::tile_width, static_cast<std::size_t>(count) - begin);
            ipc_simd::corotated_derivatives_tile(positions.data()+3*begin, dm.data()+begin,
                area.data()+begin, q.data()+begin, size, 1.3, 2.7,
                gradients.data()+begin, hessians.data()+begin);
        }
        for (int e = 0; e < count; ++e) {
            actual_g += dt2 * gradients[e];
            actual_h += dt2 * hessians[e];
        }
        EXPECT_LE((actual_g - expected_g).norm(), 16.0 * std::numeric_limits<double>::epsilon() * (1.0 + expected_g.norm())) << count;
        EXPECT_LE((actual_h - expected_h).norm(), 16.0 * std::numeric_limits<double>::epsilon() * (1.0 + expected_h.norm())) << count;
        EXPECT_TRUE((gradients.back().array() == 12345.0).all());
        EXPECT_TRUE((hessians.back().array() == 12345.0).all());
    }
}

TEST(SIMDV2Kernel, AoSHingeTilesMatchScalarAccumulation) {
    for (int count = 0; count <= 3 * static_cast<int>(ipc_simd::tile_width) + 1; ++count) {
        BendingInputs input;
        for (int i = 0; i < count; ++i)
            input.append(folded_hinge(0.08 * i - 0.7), i % 4, 0.8 + 0.02*i, -0.2);
        std::reverse(input.incident.begin(), input.incident.end());
        std::vector<Vec3> positions(4 * count), gradients(count + 1, Vec3::Constant(12345.0));
        std::vector<Mat33> hessians(count + 1, Mat33::Constant(12345.0));
        std::vector<int> roles(count);
        std::vector<double> ce(count), rest(count);
        for (int i = 0; i < count; ++i) {
            const auto [hinge, role] = input.incident[i];
            for (int corner = 0; corner < 4; ++corner)
                positions[4*i+corner] = input.positions[input.mesh.hinges[hinge].v[corner]];
            roles[i] = role;
            ce[i] = input.mesh.hinges[hinge].c_e;
            rest[i] = input.mesh.hinges[hinge].bar_theta;
        }
        Vec3 expected_g(0.13, -0.2, 0.03), actual_g = expected_g;
        Mat33 expected_h = 0.7 * Mat33::Identity(), actual_h = expected_h;
        constexpr double dt2 = 0.004, stiffness = 0.009;
        for (const auto& [hinge, role] : input.incident) {
            const auto contribution = bending_node_gradient_hessian_psd(input.def(hinge),
                stiffness, input.mesh.hinges[hinge].c_e, input.mesh.hinges[hinge].bar_theta, role);
            expected_g += dt2 * contribution.first;
            expected_h += dt2 * contribution.second;
        }
        for (std::size_t begin = 0; begin < static_cast<std::size_t>(count); begin += ipc_simd::tile_width) {
            const auto size = std::min(ipc_simd::tile_width, static_cast<std::size_t>(count) - begin);
            ipc_simd::bending_derivatives_tile(positions.data()+4*begin, roles.data()+begin,
                ce.data()+begin, rest.data()+begin, size, stiffness,
                gradients.data()+begin, hessians.data()+begin);
        }
        for (int e = 0; e < count; ++e) {
            actual_g += dt2 * gradients[e];
            actual_h += dt2 * hessians[e];
        }
        EXPECT_LE((actual_g - expected_g).norm(), 16.0 * std::numeric_limits<double>::epsilon() * (1.0 + expected_g.norm())) << count;
        EXPECT_LE((actual_h - expected_h).norm(), 16.0 * std::numeric_limits<double>::epsilon() * (1.0 + expected_h.norm())) << count;
        EXPECT_TRUE((gradients.back().array() == 12345.0).all());
        EXPECT_TRUE((hessians.back().array() == 12345.0).all());
    }
}

TEST(SIMDV2Solver, SimdFailureRestoresTheWholeColorAndJoinsWorkers) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    ClothScene scene;
    build_scene(scene);
    scene.params.use_basic_experimental = false;
    scene.params.use_basic_experimental_v2 = true;
    scene.params.use_simd = true;
    scene.params.use_parallel = true;
    scene.params.fixed_iters = true;
    scene.mesh.Dm_inverse[0].setConstant(std::numeric_limits<double>::quiet_NaN());
    const auto initial = scene.state.deformed_positions;
    std::vector<Vec3> predictor;
    build_xhat(predictor, initial, scene.state.velocities, scene.params.dt());
    EXPECT_THROW(global_gauss_seidel_solver_basic_experimental_v2(scene.mesh,
        scene.adjacency, scene.pins, scene.params, scene.state.deformed_positions,
        predictor, scene.state.velocities, scene.broad_phase), std::runtime_error);
    for (std::size_t i = 0; i < initial.size(); ++i)
        EXPECT_EQ(std::memcmp(initial[i].data(), scene.state.deformed_positions[i].data(),
            3*sizeof(double)), 0) << i;
}

TEST(SIMDV2Kernel, DegenerateHingesPreserveScalarZeroOutputsInMixedTiles) {
    constexpr std::size_t width = ipc_simd::tile_width;
    std::array<Vec3, 4 * width> positions;
    std::array<Vec3, width> gradients;
    std::array<Mat33, width> hessians;
    std::array<int, width> roles;
    std::array<double, width> coefficients, rest_angles;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t count = 1; count <= width; ++count) {
        for (std::size_t e = 0; e < count; ++e) {
            auto def = folded_hinge(0.3);
            const bool degenerate = e % 2 == 0;
            if (degenerate) def.x[3] = def.x[0];
            for (int corner = 0; corner < 4; ++corner)
                positions[4 * e + corner] = def.x[corner];
            roles[e] = static_cast<int>(e % 4);
            coefficients[e] = degenerate ? nan : 1.2;
            rest_angles[e] = degenerate ? nan : -0.1;
        }
        ipc_simd::bending_derivatives_tile(positions.data(), roles.data(), coefficients.data(),
            rest_angles.data(), count, 0.009, gradients.data(), hessians.data());
        for (std::size_t e = 0; e < count; ++e) {
            HingeDef def;
            for (int corner = 0; corner < 4; ++corner) def.x[corner] = positions[4 * e + corner];
            const auto expected = bending_node_gradient_hessian_psd(
                def, 0.009, coefficients[e], rest_angles[e], roles[e]);
            ASSERT_TRUE(gradients[e].allFinite());
            ASSERT_TRUE(hessians[e].allFinite());
            EXPECT_LE((gradients[e] - expected.first).norm(),
                16.0 * std::numeric_limits<double>::epsilon() * (1.0 + expected.first.norm()));
            EXPECT_LE((hessians[e] - expected.second).norm(),
                16.0 * std::numeric_limits<double>::epsilon() * (1.0 + expected.second.norm()));
        }
    }
}

TEST(SIMDV2Solver, WorkerBuffersTrackScalarResultsAcrossFramesAndTeamSizes) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    std::vector<DeformedState> reference;
    std::array<ClothScene, 3> scenes;
    for (auto& scene : scenes) {
        build_scene(scene);
        // Compare worker buffers independently of adaptive history from prior scenes.
        scene.params.node_box_min = scene.params.node_box_max = 0.01;
    }
    for (int run = 0; run < 3; ++run) {
        auto& scene = scenes[run];
        scene.params.use_basic_experimental = run == 0;
        scene.params.use_basic_experimental_v2 = run != 0;
        scene.params.use_simd = run != 0;
        omp_set_num_threads(run == 1 ? 1 : 4);
        for (int frame = 1; frame <= 48; ++frame) {
            scene.pins[0].target_position.y() = 0.03 + 0.02 * std::sin(0.07 * frame);
            scene.pins[1].target_position.z() = -0.22 + 0.015 * std::sin(0.11 * frame);
            ASSERT_TRUE(advance_one_frame(scene.state, scene.mesh, scene.adjacency,
                scene.pins, scene.params, scene.broad_phase, frame).converged);
            if (run == 0) reference.push_back(scene.state);
            else expect_state_near(scene.state, reference[frame - 1]);
        }
    }
}

TEST(SIMDV2Solver, CachedAoSLayoutRefreshesChangedHingePropertiesAndBendingMode) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<ClothScene, 2> scenes;
    for (int method = 0; method < 2; ++method) {
        auto& scene = scenes[method];
        build_scene(scene);
        scene.params.use_basic_experimental = method == 0;
        scene.params.use_basic_experimental_v2 = method == 1;
        scene.params.use_simd = method == 1;
        scene.params.use_parallel = true;
        scene.params.fixed_iters = true;
        scene.params.max_global_iters = 5;
        scene.params.node_box_update_count = 3;
    }
    for (int frame = 1; frame <= 4; ++frame) {
        for (auto& scene : scenes) {
            if (frame == 2) {
                scene.mesh.hinges[0].bar_theta += 0.17;
                scene.mesh.hinges[0].c_e *= 1.3;
            }
            if (frame == 3) scene.params.kB = 0.0;
            if (frame == 4) scene.params.kB = 0.009;
            ASSERT_TRUE(advance_one_frame(scene.state, scene.mesh, scene.adjacency,
                scene.pins, scene.params, scene.broad_phase, frame).converged);
        }
        expect_state_near(scenes[1].state, scenes[0].state);
    }
}

TEST(SIMDV2Solver, LaterColorFailurePreservesCompletedColorsAndJoinsWorkers) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    ClothScene scene;
    build_scene(scene);
    scene.params.use_basic_experimental = false;
    scene.params.use_basic_experimental_v2 = true;
    scene.params.use_simd = true;
    scene.params.use_parallel = true;
    scene.params.fixed_iters = true;
    scene.params.max_global_iters = 3;
    std::vector<std::vector<int>> groups;
    greedy_color_conflict_graph(build_elastic_adj(scene.mesh, scene.adjacency,
        static_cast<int>(scene.state.deformed_positions.size())), groups);
    std::vector<int> color(scene.state.deformed_positions.size(), -1);
    for (std::size_t c = 0; c < groups.size(); ++c)
        for (int node : groups[c]) color[node] = static_cast<int>(c);
    int failed_color = -1;
    for (std::size_t triangle = 0; triangle < scene.mesh.Dm_inverse.size(); ++triangle) {
        const int earliest = std::min({color[scene.mesh.tris[3*triangle]],
            color[scene.mesh.tris[3*triangle+1]], color[scene.mesh.tris[3*triangle+2]]});
        if (earliest > 0) {
            scene.mesh.Dm_inverse[triangle].setConstant(std::numeric_limits<double>::quiet_NaN());
            failed_color = earliest;
            break;
        }
    }
    ASSERT_GT(failed_color, 0);
    const auto initial = scene.state.deformed_positions;
    std::vector<Vec3> predictor;
    build_xhat(predictor, initial, scene.state.velocities, scene.params.dt());
    EXPECT_THROW(global_gauss_seidel_solver_basic_experimental_v2(scene.mesh,
        scene.adjacency, scene.pins, scene.params, scene.state.deformed_positions,
        predictor, scene.state.velocities, scene.broad_phase), std::runtime_error);
    double completed_movement = 0.0;
    for (std::size_t i = 0; i < initial.size(); ++i) {
        if (color[i] >= failed_color)
            EXPECT_EQ(std::memcmp(initial[i].data(), scene.state.deformed_positions[i].data(),
                3*sizeof(double)), 0) << i;
        else
            completed_movement += (initial[i] - scene.state.deformed_positions[i]).squaredNorm();
    }
    EXPECT_GT(completed_movement, 0.0);
}

TEST(SIMDContact, BarrierAndFrictionTilesCoverFeaturesRolesAndTails) {
    using Input = ipc_simd::MeshContactInput;
    std::vector<Input> cases;
    const auto append = [&](const std::array<Vec3,4>& x, bool ss) {
        for (int role=0;role<4;++role) {
            Input input; input.positions=x; input.previous_positions=x;
            input.role=role; input.segment_segment=ss;
            for(int node=0;node<4;++node)
                input.previous_positions[node]-=Vec3(.003*(node+1),.002*node,-.001*node*node);
            cases.push_back(input);
        }
    };
    for (const Vec3 point : {Vec3(.2,.3,.05),Vec3(.4,-.2,.06),Vec3(.6,.6,.06),
            Vec3(-.2,.4,.06),Vec3(-.2,-.3,.06),Vec3(1.2,-.1,.06),Vec3(-.1,1.2,.06),Vec3(.2,.3,3.0)})
        append({point,Vec3::Zero(),Vec3::UnitX(),Vec3::UnitY()},false);
    append({Vec3(.2,.2,.04),Vec3::Zero(),Vec3::UnitX(),2.0*Vec3::UnitX()},false);
    append({Vec3(.2,.2,.04),Vec3::Zero(),Vec3::Zero(),Vec3::Zero()},false);
    for(double x : {-.2,.4,1.2}) for(double y : {-.4,.2,-1.2})
        append({Vec3::Zero(),Vec3::UnitX(),Vec3(x,y,.06),Vec3(x,y+1,.06)},true);
    append({Vec3::Zero(),Vec3::UnitX(),Vec3(.2,.1,.03),Vec3(1.2,.1,.03)},true);
    append({Vec3::Zero(),Vec3::Zero(),Vec3(.2,.1,.03),Vec3(1.2,.1,.03)},true);
    constexpr double dhat=1.3,k=2.1,dt=.02,epsv=.3;
    for(double friction : {0.0,.3}) for(std::size_t count=1;count<=ipc_simd::contact_tile_width;++count) {
        for(std::size_t begin=0;begin<cases.size();begin+=count) {
            const auto size=std::min(count,cases.size()-begin);
            std::array<ipc_simd::MeshContactOutput,ipc_simd::contact_tile_width> result;
            std::array<unsigned char,ipc_simd::contact_tile_width> derivative_active;
            derivative_active.fill(255);
            ipc_simd::mesh_contact_derivatives_tile(cases.data()+begin,size,dhat,k,friction,dt,epsv,
                result.data(),derivative_active.data());
            for(std::size_t lane=0;lane<size;++lane) {
                const auto& input=cases[begin+lane];const auto& x=input.positions;
                SCOPED_TRACE(::testing::Message()<<"entry="<<begin+lane<<" count="<<count<<" friction="<<friction);
                std::pair<Vec3,Mat33> normal,friction_terms{Vec3::Zero(),Mat33::Zero()};
                if(input.segment_segment) {
                    auto evaluation=make_segment_segment_contact_evaluation(x,dhat,k);
                    normal=segment_segment_barrier_self_gradient_and_hessian(x[0],x[1],x[2],x[3],input.role,evaluation);
                    if(friction!=0.0) friction_terms=frozen_friction_role_gradient_and_hessian(
                        make_segment_segment_frozen_friction_contact(x,input.previous_positions,evaluation,dt,epsv),input.role,friction,dt*dt);
                } else {
                    auto evaluation=make_node_triangle_contact_evaluation(x,dhat,k);
                    normal=node_triangle_barrier_self_gradient_and_hessian(x[0],x[1],x[2],x[3],input.role,evaluation);
                    if(friction!=0.0) friction_terms=frozen_friction_role_gradient_and_hessian(
                        make_node_triangle_frozen_friction_contact(x,input.previous_positions,evaluation,dt,epsv),input.role,friction,dt*dt);
                }
                EXPECT_LE((result[lane].gradient-normal.first).norm(),2e-11*(1+normal.first.norm()));
                EXPECT_LE((result[lane].hessian-normal.second).norm(),2e-11*(1+normal.second.norm()));
                EXPECT_LE((result[lane].friction_gradient-friction_terms.first).norm(),2e-11*(1+friction_terms.first.norm()));
                EXPECT_LE((result[lane].friction_hessian-friction_terms.second).norm(),2e-11*(1+friction_terms.second.norm()));
                ASSERT_LE(derivative_active[lane],1);
                if (!derivative_active[lane]) {
                    EXPECT_TRUE((result[lane].gradient.array()==0.0).all());
                    EXPECT_TRUE((result[lane].hessian.array()==0.0).all());
                    EXPECT_TRUE((result[lane].friction_gradient.array()==0.0).all());
                    EXPECT_TRUE((result[lane].friction_hessian.array()==0.0).all());
                }
            }
        }
    }
}

TEST(SIMDContact, DerivativeActivitySkipsUnusedRolesWithoutChangingOutputsOrValidation) {
    std::array<ipc_simd::MeshContactInput,8> inputs;
    for (int entry = 0; entry < 8; ++entry) {
        auto& input = inputs[entry];
        input.positions = {Vec3(-.2,-.3,.08),Vec3::Zero(),Vec3::UnitX(),Vec3::UnitY()};
        if (entry >= 4) input.positions[0] = Vec3(.2,.3,2.0);
        input.previous_positions = input.positions;
        input.previous_positions[0] -= Vec3(.03,.01,0.0);
        input.role = entry % 4;
    }
    for (double friction : {0.0,.3}) {
        std::array<ipc_simd::MeshContactOutput,8> masked,unmasked;
        std::array<unsigned char,8> derivative_active;
        derivative_active.fill(255);
        ipc_simd::mesh_contact_derivatives_tile(inputs.data(),inputs.size(),1.0,2.0,friction,.02,.3,
            masked.data(),derivative_active.data());
        ipc_simd::mesh_contact_derivatives_tile(inputs.data(),inputs.size(),1.0,2.0,friction,.02,.3,unmasked.data());
        Vec3 all_g(.1,.2,.3),active_g=all_g;
        Mat33 all_H=Mat33::Identity(),active_H=all_H;
        for (std::size_t entry = 0; entry < inputs.size(); ++entry) {
            EXPECT_EQ(derivative_active[entry],entry < 2 ? 1 : 0) << entry;
            EXPECT_EQ((masked[entry].gradient-unmasked[entry].gradient).norm(),0.0);
            EXPECT_EQ((masked[entry].hessian-unmasked[entry].hessian).norm(),0.0);
            EXPECT_EQ((masked[entry].friction_gradient-unmasked[entry].friction_gradient).norm(),0.0);
            EXPECT_EQ((masked[entry].friction_hessian-unmasked[entry].friction_hessian).norm(),0.0);
            const Vec3 g=.0008*unmasked[entry].gradient+unmasked[entry].friction_gradient;
            const Mat33 H=.0008*unmasked[entry].hessian+unmasked[entry].friction_hessian;
            all_g+=g; all_H+=H;
            if (derivative_active[entry]) {active_g+=g;active_H+=H;}
        }
        EXPECT_EQ((active_g-all_g).norm(),0.0);
        EXPECT_EQ((active_H-all_H).norm(),0.0);
    }
    std::array<ipc_simd::MeshContactOutput,8> outputs;
    std::array<unsigned char,8> derivative_active;
    EXPECT_THROW(ipc_simd::mesh_contact_derivatives_tile(inputs.data(),inputs.size(),1.0,2.0,-.3,.02,.3,
        outputs.data(),derivative_active.data()),std::invalid_argument);
    EXPECT_THROW(ipc_simd::mesh_contact_derivatives_tile(inputs.data(),inputs.size(),1.0,2.0,-.3,.02,.3,
        outputs.data()),std::invalid_argument);
}

TEST(SIMDContact, RotatedBarrierDerivativesMatchScalar) {
    std::mt19937_64 generator(16204);
    std::uniform_real_distribution<double> coordinate(-.02, .02);
    constexpr double tolerance = 2e-11;
    for (int sample = 0; sample < 256; ++sample) {
        std::array<Vec3, 4> positions;
        for (auto& position : positions)
            for (int axis = 0; axis < 3; ++axis) position[axis] = coordinate(generator);
        std::array<ipc_simd::MeshContactInput, 8> inputs;
        std::array<ipc_simd::MeshContactOutput, 8> outputs;
        for (int entry = 0; entry < 8; ++entry) {
            inputs[entry].positions = positions;
            inputs[entry].previous_positions = positions;
            inputs[entry].role = entry % 4;
            inputs[entry].segment_segment = entry >= 4;
        }
        ipc_simd::mesh_contact_derivatives_tile(inputs.data(), inputs.size(),
            .1, 1.0, 0.0, .01, .01, outputs.data());
        for (int entry = 0; entry < 8; ++entry) {
            SCOPED_TRACE(::testing::Message() << "sample=" << sample << " entry=" << entry);
            const auto expected = entry < 4
                ? node_triangle_barrier_self_gradient_and_hessian(positions[0], positions[1],
                    positions[2], positions[3], .1, entry % 4)
                : segment_segment_barrier_self_gradient_and_hessian(positions[0], positions[1],
                    positions[2], positions[3], .1, entry % 4);
            EXPECT_LE((outputs[entry].gradient - expected.first).norm(),
                tolerance * (1.0 + expected.first.norm()));
            EXPECT_LE((outputs[entry].hessian - expected.second).norm(),
                tolerance * (1.0 + expected.second.norm()));
        }
    }
}

TEST(SIMDContact, SegmentContactsPreserveRegionsTiesDegeneracyAndTails) {
    std::vector<ipc_simd::MeshContactInput> inputs;
    const auto append = [&](const std::array<Vec3, 4>& positions) {
        for (int role = 0; role < 4; ++role) {
            ipc_simd::MeshContactInput input;
            input.positions = positions;
            input.previous_positions = positions;
            input.role = role;
            input.segment_segment = true;
            for (int node = 0; node < 4; ++node)
                input.previous_positions[node] -= Vec3(.001*node, -.002*node, .003);
            inputs.push_back(input);
        }
    };
    for (double x : {-.2, 0.0, 1e-15, .4, 1.0-1e-15, 1.0, 1.2})
        for (double y : {-.4, .2, -1.2}) {
            const std::array<Vec3, 4> positions{Vec3::Zero(), Vec3::UnitX(),
                Vec3(x,y,.06), Vec3(x,y+1.0,.06)};
            append(positions);
            append({positions[1],positions[0],positions[3],positions[2]});
        }
    // Parallel overlapping segments have tied boundary candidates. Retain the
    // first minimum, including when either segment's direction is reversed.
    append({Vec3::Zero(),Vec3::UnitX(),Vec3(.2,.1,.03),Vec3(1.2,.1,.03)});
    append({Vec3::UnitX(),Vec3::Zero(),Vec3(1.2,.1,.03),Vec3(.2,.1,.03)});
    append({Vec3::Zero(),Vec3::UnitX(),Vec3(.2,.1,.03),Vec3(1.2,.1+1e-13,.03)});
    append({Vec3::Zero(),Vec3::Zero(),Vec3(.2,.1,.03),Vec3(1.2,.1,.03)});
    append({Vec3::Zero(),Vec3::Zero(),Vec3(.2,.1,.03),Vec3(.2,.1,.03)});
    for (double factor : {1.0-1e-12, 1.0, 1.0+1e-12}) {
        const double length = 1e-6 * factor;
        append({Vec3::Zero(),Vec3(length,0,0),Vec3(.4*length,-.3*length,.1*length),
            Vec3(.4*length,.7*length,.1*length)});
    }
    for (double friction : {0.0, .3})
        for (std::size_t width : {std::size_t(1),std::size_t(7),std::size_t(8),
                std::size_t(9),std::size_t(31),ipc_simd::contact_tile_width})
            for (std::size_t begin = 0; begin < inputs.size(); begin += width) {
                const std::size_t count = std::min(width, inputs.size()-begin);
                std::array<ipc_simd::MeshContactOutput,ipc_simd::contact_tile_width> output;
                ipc_simd::mesh_contact_derivatives_tile(inputs.data()+begin,count,
                    1.3,2.1,friction,.02,.3,output.data());
                for (std::size_t lane = 0; lane < count; ++lane) {
                    const auto& input = inputs[begin+lane];
                    const auto& x = input.positions;
                    const auto evaluation = make_segment_segment_contact_evaluation(x,1.3,2.1);
                    const auto expected = segment_segment_barrier_self_gradient_and_hessian(
                        x[0],x[1],x[2],x[3],1.3,input.role);
                    SCOPED_TRACE(::testing::Message() << "entry=" << begin+lane << " width=" << width
                        << " region=" << to_string(evaluation.dr.region) << " friction=" << friction);
                    EXPECT_LE((output[lane].gradient-expected.first).norm(),2e-11*(1.0+expected.first.norm()));
                    EXPECT_LE((output[lane].hessian-expected.second).norm(),2e-11*(1.0+expected.second.norm()));
                    if (friction != 0.0) {
                        const auto frozen = make_segment_segment_frozen_friction_contact(
                            x,input.previous_positions,evaluation,.02,.3);
                        const auto expected_friction = frozen_friction_role_gradient_and_hessian(
                            frozen,input.role,friction,.02*.02);
                        EXPECT_LE((output[lane].friction_gradient-expected_friction.first).norm(),
                            2e-11*(1.0+expected_friction.first.norm()));
                        EXPECT_LE((output[lane].friction_hessian-expected_friction.second).norm(),
                            2e-11*(1.0+expected_friction.second.norm()));
                    }
                }
            }
}

TEST(SIMDContact, BarrierHessiansMatchEnergyFiniteDifferences) {
    std::vector<ipc_simd::MeshContactInput> inputs;
    const auto append = [&](const std::array<Vec3,4>& positions, bool segment) {
        for (int role = 0; role < 4; ++role) {
            ipc_simd::MeshContactInput input;
            input.positions = positions;
            input.previous_positions = positions;
            input.role = role;
            input.segment_segment = segment;
            inputs.push_back(input);
        }
    };
    append({Vec3(.2,.3,.08),Vec3::Zero(),Vec3::UnitX(),Vec3::UnitY()},false);
    append({Vec3(.4,-.2,.08),Vec3::Zero(),Vec3::UnitX(),Vec3::UnitY()},false);
    append({Vec3(-.2,-.3,.08),Vec3::Zero(),Vec3::UnitX(),Vec3::UnitY()},false);
    append({Vec3::Zero(),Vec3::UnitX(),Vec3(.4,-.3,.08),Vec3(.4,.7,.08)},true);
    append({Vec3::Zero(),Vec3::UnitX(),Vec3(-.2,-.3,.08),Vec3(-.2,.7,.08)},true);
    std::array<ipc_simd::MeshContactOutput,ipc_simd::contact_tile_width> output;
    ipc_simd::mesh_contact_derivatives_tile(inputs.data(),inputs.size(),1.3,1.0,0.0,.02,.3,output.data());
    constexpr double h = 1e-5;
    for (std::size_t entry = 0; entry < inputs.size(); ++entry) {
        const auto& input = inputs[entry];
        SCOPED_TRACE(::testing::Message() << "entry=" << entry);
        const auto energy = [&](const std::array<Vec3,4>& positions) {
            return input.segment_segment
                ? segment_segment_barrier(positions[0],positions[1],positions[2],positions[3],1.3)
                : node_triangle_barrier(positions[0],positions[1],positions[2],positions[3],1.3);
        };
        const double center = energy(input.positions);
        Mat33 finite_difference;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) {
                auto pp = input.positions, pm = input.positions;
                auto mp = input.positions, mm = input.positions;
                if (row == col) {
                    pp[input.role][row] += h;
                    mm[input.role][row] -= h;
                    finite_difference(row,col) = (energy(pp)-2.0*center+energy(mm))/(h*h);
                } else {
                    pp[input.role][row] += h; pp[input.role][col] += h;
                    pm[input.role][row] += h; pm[input.role][col] -= h;
                    mp[input.role][row] -= h; mp[input.role][col] += h;
                    mm[input.role][row] -= h; mm[input.role][col] -= h;
                    finite_difference(row,col) = (energy(pp)-energy(pm)-energy(mp)+energy(mm))/(4.0*h*h);
                }
            }
        EXPECT_LE((output[entry].hessian-finite_difference).norm(),
            2e-5*(1.0+finite_difference.norm()));
    }
}

TEST(SIMDContact, TinyGapInteriorContactsRetainScalarHessians) {
    const std::array<Vec3,4> positions{
        Vec3(.51516375049435637,-.10561811939691514,.010823124294003884),
        Vec3(.51542661145754476,-.092006671821605143,-.019860949688460236),
        Vec3(.49687351514057676,-.09560374221287278,-.0030195283661526105),
        Vec3(.52895529814256448,-.099869764004829761,-.0085986799104287668)};
    std::array<ipc_simd::MeshContactInput,4> inputs;
    std::array<ipc_simd::MeshContactOutput,4> outputs;
    for (int role = 0; role < 4; ++role) {
        inputs[role].positions = positions;
        inputs[role].previous_positions = positions;
        inputs[role].role = role;
        inputs[role].segment_segment = true;
    }
    ipc_simd::mesh_contact_derivatives_tile(inputs.data(),inputs.size(),.01,1.0,0.0,.02,.3,outputs.data());
    for (int role = 0; role < 4; ++role) {
        const auto expected = segment_segment_barrier_self_gradient_and_hessian(
            positions[0],positions[1],positions[2],positions[3],.01,role);
        EXPECT_LE((outputs[role].hessian-expected.second).norm(),2e-11*(1.0+expected.second.norm()));
    }
}

TEST(SIMDContact, SdfTilesPreservePiecewiseBoundariesAndCurvature) {
    std::vector<SDFEvaluation> data;
    for(double y : {-.03,0.0,.001,.002,.003}) {
        data.push_back(evaluate_sdf(PlaneSDF{Vec3::Zero(),Vec3::UnitY()},Vec3(.2,y,.3)));
        data.push_back(evaluate_sdf(SphereSDF{Vec3::Zero(),1.0},Vec3(1.0+y,0.0,0.0)));
        data.push_back(evaluate_sdf(CylinderSDF{Vec3::Zero(),Vec3::UnitZ(),1.0},Vec3(1.0+y,0.0,.3)));
        data.push_back(evaluate_sdf(SphereSDF{Vec3::Zero(),1.0},
            (1.0+y)*Vec3(1.0,2.0,3.0).normalized()));
        data.push_back(evaluate_sdf(CylinderSDF{Vec3::Zero(),Vec3::UnitZ(),1.0},
            Vec3((1.0+y)*.6,(1.0+y)*.8,.3)));
    }
    for(double eps : {0.0,.002}) for(bool curvature : {false,true})
        for(std::size_t count=1;count<=ipc_simd::tile_width;++count)
            for(std::size_t begin=0;begin<data.size();begin+=count) {
                const auto size=std::min(count,data.size()-begin);
                std::array<Vec3,ipc_simd::tile_width> g;
                std::array<Mat33,ipc_simd::tile_width> H;
                ipc_simd::sdf_derivatives_tile(data.data()+begin,size,37.0,eps,g.data(),H.data(),curvature);
                for(std::size_t lane=0;lane<size;++lane) {
                    const auto expected_g=sdf_penalty_gradient(data[begin+lane],37.0,eps);
                    const auto expected_H=sdf_penalty_hessian(data[begin+lane],37.0,eps,curvature);
                    EXPECT_LE((g[lane]-expected_g).norm(),2e-13*(1+expected_g.norm()));
                    EXPECT_LE((H[lane]-expected_H).norm(),2e-13*(1+expected_H.norm()));
                }
            }
}

TEST(SIMDContact, FrictionTilesCoverSlipBranchesInactiveDataAndValidation) {
    std::array<FrozenFrictionContact,ipc_simd::tile_width> data;
    std::array<int,ipc_simd::tile_width> roles;
    for(std::size_t e=0;e<data.size();++e) {
        auto& c=data[e];c.active=true;c.normal_force=1.2;c.eps_u=.02;
        c.weights={1.0,-.2,-.3,-.5};c.normal=Vec3::UnitZ();
        c.projector=Mat33::Identity()-c.normal*c.normal.transpose();
        c.tangential_displacement=Vec3(.005*e,0.0,0.0);roles[e]=e%4;
    }
    data.back().active=false;data.back().eps_u=std::numeric_limits<double>::quiet_NaN();
    data.back().projector.setConstant(std::numeric_limits<double>::quiet_NaN());
    for(std::size_t count=1;count<=data.size();++count) {
        std::array<Vec3,ipc_simd::tile_width> g;std::array<Mat33,ipc_simd::tile_width> H;
        ipc_simd::friction_derivatives_tile(data.data(),roles.data(),count,.3,.04,g.data(),H.data());
        for(std::size_t e=0;e<count;++e) {
            const auto expected=frozen_friction_role_gradient_and_hessian(data[e],roles[e],.3,.04);
            EXPECT_LE((g[e]-expected.first).norm(),2e-13*(1+expected.first.norm()));
            EXPECT_LE((H[e]-expected.second).norm(),2e-13*(1+expected.second.norm()));
        }
        EXPECT_THROW(ipc_simd::friction_derivatives_tile(data.data(),roles.data(),count,-.3,.04,g.data(),H.data()),std::invalid_argument);
    }
}

TEST(SIMDContact, ColoredSdfAndFrictionAssemblyTracksScalarAcrossFrames) {
    RestoreOpenMPSettings restore;
    omp_set_dynamic(0);omp_set_num_threads(4);
    std::array<ClothScene,2> scenes;
    for(std::size_t mode=0;mode<scenes.size();++mode) {
        auto& scene=scenes[mode];build_scene(scene);
        scene.params.use_basic_experimental_v2=mode==1;
        scene.params.use_simd=mode==1;
        scene.params.k_sdf=37.0;scene.params.eps_sdf=.003;
        scene.params.friction_coefficient=.2;
        scene.params.sdf_planes.push_back(PlaneSDF{Vec3(0,.04,0),Vec3::UnitY()});
    }
    for(int frame=1;frame<=12;++frame) {
        for(auto& scene:scenes)
            ASSERT_TRUE(advance_one_frame(scene.state,scene.mesh,scene.adjacency,scene.pins,
                scene.params,scene.broad_phase,frame).converged);
        expect_state_near(scenes[1].state,scenes[0].state);
    }
}

TEST(SIMDContact, MissingSdfObstaclesDoNotRequirePreviousPositions) {
    auto params=SimParams::zeros();params.k_sdf=37.0;params.eps_sdf=.003;
    params.friction_coefficient=.2;
    const Vec3 x(0,.03,0);
    physics_detail::SdfDerivatives result;
    EXPECT_NO_THROW(physics_detail::compute_sdf_derivatives_tile(params,&x,nullptr,1,&result));
    EXPECT_EQ(result.gradient.norm(),0.0);EXPECT_EQ(result.hessian.norm(),0.0);
    EXPECT_EQ(result.friction_gradient.norm(),0.0);EXPECT_EQ(result.friction_hessian.norm(),0.0);
}

TEST(SIMDContact, MovingSdfTilesPreserveFrictionAndLazyMotionValidation) {
    constexpr std::size_t count = 5;
    const Vec3 plane_normal = Vec3(1.0, 2.0, -1.0).normalized();
    const Vec3 translation(0.04, -0.03, -0.02);
    const Vec3 center(0.3, -0.2, 0.4);
    const Mat33 spin = Eigen::AngleAxisd(0.37,
        Vec3(0.2, -0.7, 0.5).normalized()).toRotationMatrix();
    for (bool sphere : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "sphere=" << sphere);
        SimParams params = SimParams::zeros();
        params.fps = 50.0;
        params.k_sdf = 37.0;
        params.eps_sdf = 0.1;
        params.friction_coefficient = 0.3;
        params.friction_velocity_epsilon = 0.5;
        if (sphere) {
            params.sdf_spheres.push_back(SphereSDF{center, 1.0});
            auto& motion = params.sdf_spheres[0].material_motion;
            motion.previous.translation = motion.current.translation = center;
            motion.current.rotation = spin;
        } else {
            params.sdf_planes.push_back(PlaneSDF{Vec3::Zero(), plane_normal});
            params.sdf_planes[0].material_motion.current.translation = translation;
        }
        std::array<Vec3, count> positions, previous;
        std::array<SDFEvaluation, count> evaluations;
        std::array<physics_detail::SdfDerivatives, count> outputs;
        const std::array<double, count> distances{{-0.02, 0.04, 0.07, 0.15, 0.01}};
        for (std::size_t e = 0; e < count; ++e) {
            if (sphere) {
                const Vec3 normal = Vec3(0.7 + 0.1 * e, 0.3, -0.4).normalized();
                positions[e] = center + (1.0 + distances[e]) * normal;
                evaluations[e] = evaluate_sdf(params.sdf_spheres[0], positions[e]);
                const Vec3 surface = evaluations[e].surface_point;
                const Vec3 old_surface = center + spin.transpose() * (surface - center);
                previous[e] = positions[e] - (surface - old_surface);
            } else {
                positions[e] = 3.0 * translation + distances[e] * plane_normal;
                evaluations[e] = evaluate_sdf(params.sdf_planes[0], positions[e]);
                previous[e] = positions[e] - translation;
            }
            if (e % 2 != 0) previous[e] = positions[e];
        }
        ASSERT_NO_THROW(physics_detail::compute_sdf_derivatives_tile(params,
            positions.data(), previous.data(), count, outputs.data()));
        for (std::size_t e = 0; e < count; ++e) {
            const Vec3 expected_g = sdf_penalty_gradient(evaluations[e],
                params.k_sdf, params.eps_sdf);
            const Mat33 expected_h = sdf_penalty_hessian(evaluations[e],
                params.k_sdf, params.eps_sdf, false);
            const auto frozen = make_sdf_frozen_friction_contact(positions[e],
                previous[e], evaluations[e], params.k_sdf, params.eps_sdf,
                params.dt(), params.friction_velocity_epsilon);
            const auto friction = frozen_friction_role_gradient_and_hessian(
                frozen, 0, params.friction_coefficient, params.dt2());
            EXPECT_LE((outputs[e].gradient - expected_g).norm(),
                2e-12 * (1.0 + expected_g.norm()));
            EXPECT_LE((outputs[e].hessian - expected_h).norm(),
                2e-12 * (1.0 + expected_h.norm()));
            EXPECT_LE((outputs[e].friction_gradient - friction.first).norm(),
                2e-12 * (1.0 + friction.first.norm()));
            EXPECT_LE((outputs[e].friction_hessian - friction.second).norm(),
                2e-12 * (1.0 + friction.second.norm()));
        }
        EXPECT_LT(outputs[0].friction_gradient.norm(), 1e-12);
        EXPECT_GT(outputs[1].friction_gradient.norm(), 1e-6);
        EXPECT_EQ(outputs[3].friction_gradient.norm(), 0.0);

        auto& motion = sphere ? params.sdf_spheres[0].material_motion
                              : params.sdf_planes[0].material_motion;
        motion.current.rotation(0, 0) += 0.5;
        params.friction_coefficient = 0.0;
        ASSERT_NO_THROW(physics_detail::compute_sdf_derivatives_tile(params,
            positions.data(), nullptr, count, outputs.data()));
        for (std::size_t e = 0; e < count; ++e) {
            const Vec3 expected = sdf_penalty_gradient(evaluations[e],
                params.k_sdf, params.eps_sdf);
            EXPECT_LE((outputs[e].gradient - expected).norm(),
                2e-12 * (1.0 + expected.norm()));
            EXPECT_EQ(outputs[e].friction_gradient.norm(), 0.0);
            EXPECT_EQ(outputs[e].friction_hessian.norm(), 0.0);
        }
        params.friction_coefficient = 0.3;
        EXPECT_THROW(physics_detail::compute_sdf_derivatives_tile(params,
            positions.data(), previous.data(), count, outputs.data()), std::invalid_argument);
    }
}

TEST(SIMDContact, ZeroStiffnessCoincidentContactsSkipUnusedFrictionGeometry) {
    constexpr std::size_t count = 8;
    std::array<ipc_simd::MeshContactInput, count> inputs;
    std::array<ipc_simd::MeshContactOutput, count> outputs;
    for (std::size_t e = 0; e < count; ++e) {
        auto& input = inputs[e];
        input.role = static_cast<int>(e % 4);
        input.segment_segment = e >= 4;
        input.positions = input.segment_segment
            ? std::array<Vec3, 4>{{Vec3(-1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
                  Vec3(0.0, -1.0, 0.0), Vec3(0.0, 1.0, 0.0)}}
            : std::array<Vec3, 4>{{Vec3(0.25, 0.25, 0.0), Vec3::Zero(),
                  Vec3::UnitX(), Vec3::UnitY()}};
        for (auto& previous : input.previous_positions)
            previous.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
    constexpr double d_hat = 0.1, dt = 0.02, eps_v = 0.5, friction = 0.3;
    ASSERT_NO_THROW(ipc_simd::mesh_contact_derivatives_tile(inputs.data(), count,
        d_hat, 0.0, friction, dt, eps_v, outputs.data()));
    for (std::size_t e = 0; e < count; ++e) {
        const auto& input = inputs[e];
        const auto& x = input.positions;
        FrozenFrictionContact frozen;
        std::pair<Vec3, Mat33> normal;
        if (input.segment_segment) {
            const auto evaluation = make_segment_segment_contact_evaluation(x, d_hat, 0.0);
            normal = segment_segment_barrier_self_gradient_and_hessian(
                x[0], x[1], x[2], x[3], input.role, evaluation);
            frozen = make_segment_segment_frozen_friction_contact(x,
                input.previous_positions, evaluation, dt, eps_v);
        } else {
            const auto evaluation = make_node_triangle_contact_evaluation(x, d_hat, 0.0);
            normal = node_triangle_barrier_self_gradient_and_hessian(
                x[0], x[1], x[2], x[3], input.role, evaluation);
            frozen = make_node_triangle_frozen_friction_contact(x,
                input.previous_positions, evaluation, dt, eps_v);
        }
        ASSERT_FALSE(frozen.active);
        const auto expected_friction = frozen_friction_role_gradient_and_hessian(
            frozen, input.role, friction, dt * dt);
        EXPECT_LE((outputs[e].gradient - normal.first).norm(), 1e-14);
        EXPECT_LE((outputs[e].hessian - normal.second).norm(), 1e-14);
        EXPECT_LE((outputs[e].friction_gradient - expected_friction.first).norm(), 1e-14);
        EXPECT_LE((outputs[e].friction_hessian - expected_friction.second).norm(), 1e-14);
    }
}

TEST(SIMDContact, InactiveFrictionIgnoresOverflowingUnusedScale) {
    FrozenFrictionContact contact;
    contact.weights[0] = 1.0;
    const int role = 0;
    const double large = std::numeric_limits<double>::max();
    Vec3 g;
    Mat33 H;
    EXPECT_NO_THROW(ipc_simd::friction_derivatives_tile(&contact, &role, 1, large, large, &g, &H));
    EXPECT_EQ(g.norm(), 0.0);
    EXPECT_EQ(H.norm(), 0.0);
}

TEST(SIMDPointTile, MixedPinsAndTailsMatchPointReference) {
    std::array<ipc_simd::PointInput,ipc_simd::tile_width> inputs;
    for (std::size_t e = 0; e < inputs.size(); ++e) {
        auto& input = inputs[e];
        input.mass = e % 3 == 0 ? 0.0 : .2 + .7*e;
        input.position = Vec3(-.3+.2*e,.4-.05*e,.6+.001*e);
        input.predicted_position = input.position + Vec3(.03*e,-.01*(e+1),.02);
        if (e % 2) input.pin_target = Vec3(.2,-.1+.02*e,.8);
    }
    const Vec3 gravity(-1.7,-9.81,3.2);
    constexpr double kpin = 37.0, dt2 = .002;
    for (std::size_t count = 0; count <= inputs.size(); ++count) {
        std::array<Vec3,ipc_simd::tile_width+1> g;
        std::array<Mat33,ipc_simd::tile_width+1> H;
        for (auto& value : g) value.setConstant(12345.0);
        for (auto& value : H) value.setConstant(12345.0);
        ipc_simd::point_derivatives_tile(inputs.data(),count,gravity,kpin,dt2,g.data(),H.data());
        for (std::size_t e = 0; e < count; ++e) {
            const auto& input = inputs[e];
            Vec3 expected_g = Vec3::Zero();
            Mat33 expected_H = Mat33::Zero();
            reference_point(input, gravity, kpin, dt2, expected_g, expected_H);
            EXPECT_LE((g[e]-expected_g).norm(),2e-13*(1.0+expected_g.norm()));
            EXPECT_LE((H[e]-expected_H).norm(),2e-13*(1.0+expected_H.norm()));
        }
        for (std::size_t e = count; e < g.size(); ++e) {
            EXPECT_TRUE((g[e].array()==12345.0).all());
            EXPECT_TRUE((H[e].array()==12345.0).all());
        }
    }
}

TEST(SIMDPointTile, IndependentVertexDerivativesMatchEnergyFiniteDifferences) {
    std::array<ipc_simd::PointInput,ipc_simd::tile_width> inputs;
    for (std::size_t e = 0; e < inputs.size(); ++e) {
        auto& input = inputs[e];
        input.mass = .3 + .4*e;
        input.position = Vec3(.1+.13*e,-.2+.01*e,.3-.05*e);
        input.predicted_position = Vec3(.3,-.1,.7);
        if (e % 2) input.pin_target = Vec3(-.4,.2,.1+.02*e);
    }
    const Vec3 gravity(.4,-9.81,-.7);
    constexpr double kpin = 23.0, dt2 = .003, h = 1e-4;
    std::array<Vec3,ipc_simd::tile_width> g;
    std::array<Mat33,ipc_simd::tile_width> H;
    ipc_simd::point_derivatives_tile(inputs.data(),inputs.size(),gravity,kpin,dt2,g.data(),H.data());
    for (std::size_t e = 0; e < inputs.size(); ++e) {
        const auto& input = inputs[e];
        const auto energy = [&](const Vec3& x) {
            double result = .5*input.mass*(x-input.predicted_position).squaredNorm()
                - dt2*input.mass*gravity.dot(x);
            if (input.pin_target) result += .5*dt2*kpin*(x-*input.pin_target).squaredNorm();
            return result;
        };
        Vec3 fd_g;
        Mat33 fd_H;
        const double center = energy(input.position);
        for (int row = 0; row < 3; ++row) {
            Vec3 plus = input.position, minus = input.position;
            plus[row] += h; minus[row] -= h;
            fd_g[row] = (energy(plus)-energy(minus))/(2.0*h);
            fd_H(row,row) = (energy(plus)-2.0*center+energy(minus))/(h*h);
            for (int col = row+1; col < 3; ++col) {
                Vec3 pp = plus, pm = plus, mp = minus, mm = minus;
                pp[col] += h; pm[col] -= h; mp[col] += h; mm[col] -= h;
                fd_H(row,col) = fd_H(col,row) = (energy(pp)-energy(pm)-energy(mp)+energy(mm))/(4.0*h*h);
            }
        }
        EXPECT_LE((g[e]-fd_g).norm(),2e-8*(1.0+fd_g.norm()));
        EXPECT_LE((H[e]-fd_H).norm(),2e-6*(1.0+fd_H.norm()));
    }
}

TEST(SIMDPointTile, PinMaskRetainsNonfiniteValueSemantics) {
    std::array<ipc_simd::PointInput,ipc_simd::tile_width> inputs;
    for (std::size_t e = 0; e < inputs.size(); ++e) {
        auto& input = inputs[e];
        input.mass = .2 + .3*e;
        input.position = Vec3(.1,.2,.3);
        input.predicted_position = Vec3(-.2,.1,.5);
        if (e % 2) input.pin_target = Vec3(.3,-.5,.4);
    }
    inputs[1].pin_target->x() = std::numeric_limits<double>::infinity();
    const Vec3 gravity(0.0,-9.81,0.0);
    for (double kpin : {0.0,12.0,std::numeric_limits<double>::quiet_NaN()}) {
        std::array<Vec3,ipc_simd::tile_width> g;
        std::array<Mat33,ipc_simd::tile_width> H;
        ipc_simd::point_derivatives_tile(inputs.data(),inputs.size(),gravity,kpin,.002,g.data(),H.data());
        for (std::size_t e = 0; e < inputs.size(); ++e) {
            const auto& input = inputs[e];
            Vec3 expected_g = Vec3::Zero();
            Mat33 expected_H = Mat33::Zero();
            reference_point(input, gravity, kpin, .002, expected_g, expected_H);
            for (int i = 0; i < 3; ++i) {
                if (std::isnan(expected_g[i])) EXPECT_TRUE(std::isnan(g[e][i]));
                else if (std::isinf(expected_g[i])) EXPECT_EQ(g[e][i],expected_g[i]);
                else EXPECT_NEAR(g[e][i],expected_g[i],16.0*std::numeric_limits<double>::epsilon()*(1.0+std::abs(expected_g[i])));
                for (int j = 0; j < 3; ++j) {
                    if (std::isnan(expected_H(i,j))) EXPECT_TRUE(std::isnan(H[e](i,j)));
                    else if (std::isinf(expected_H(i,j))) EXPECT_EQ(H[e](i,j),expected_H(i,j));
                    else EXPECT_NEAR(H[e](i,j),expected_H(i,j),16.0*std::numeric_limits<double>::epsilon()*(1.0+std::abs(expected_H(i,j))));
                }
            }
        }
    }
}

namespace simd_energy_test {

namespace {
constexpr double mu=1.3,lambda=2.7;
const Mat22 dm=(Mat22()<<1.1,.2,-.1,.9).finished();
const Mat22 dm_inv=dm.inverse();
constexpr double area=.7;

std::pair<Vec3,Mat33> scalar_elasticity(const Vec3* x,int role) {
    Mat32 ds;ds.col(0)=x[1]-x[0];ds.col(1)=x[2]-x[0];
    const Mat32 F=ds*dm_inv;
    const auto cache=buildCorotatedCache(F);
    const auto P=PCorotated32(cache,F,mu,lambda);
    Mat66 derivative;dPdFCorotated32(cache,mu,lambda,derivative);
    const auto shape=shape_function_gradients(dm_inv);
    return {corotated_node_gradient(P,area,shape,role),corotated_node_hessian(derivative,area,shape,role)};
}
template<class A,class B> void compare(const A& actual,const B& expected,double tolerance) {
    for(int i=0;i<expected.size();++i) {
        if(std::isnan(expected.data()[i]))EXPECT_TRUE(std::isnan(actual.data()[i]));
        else if(std::isinf(expected.data()[i]))EXPECT_EQ(actual.data()[i],expected.data()[i]);
        else EXPECT_NEAR(actual.data()[i],expected.data()[i],tolerance*(1+expected.norm()));
    }
}
HingeDef hinge(double theta) {
    HingeDef h;
    h.x[0]=Vec3(.13,-.2,.03);h.x[1]=h.x[0]+Vec3(1.1,.07,.03);
    h.x[2]=h.x[0]+Vec3(.3,1.1,.12);
    h.x[3]=h.x[0]+Vec3(.4,-std::cos(theta),std::sin(theta));
    return h;
}
}

TEST(SIMDEnergy, ElasticityMatchesScalarAcrossClampsAndTails) {
    std::mt19937_64 random(17713);std::uniform_real_distribution<double> noise(-.35,.35);
    for(int sample=0;sample<60;++sample)for(std::size_t count=1;count<=8;++count) {
        std::array<Vec3,24> positions;
        std::array<Mat22,8> materials;
        std::array<Vec2,8> shape;
        std::array<double,8> areas;
        std::array<Vec3,9> g;std::array<Mat33,9> H;
        g.back().setConstant(12345);H.back().setConstant(12345);
        for(std::size_t e=0;e<count;++e) {
            Mat32 F;for(int i=0;i<3;++i)for(int j=0;j<2;++j)F(i,j)=(i==j?1.0:0.0)+noise(random);
            if(sample%6==1 && e==count-1) {F.setZero();F(0,0)=1.0;F(1,1)=1e-7;}
            if(sample%6==2 && e==0) {F.setZero();F(0,0)=1.0;F(1,1)=1e-6;}
            if(sample%6==3 && e%2==0) {F.setZero();F(0,0)=1.0;F(0,1)=1.0;F(1,1)=1e-7;}
            if(sample%6==4) {F.setZero();F(0,0)=1.0;F(1,1)=1e-7;}
            if(sample%6==5 && e==count-1) F.setZero();
            const Vec3 origin(.13,-.2,.03);
            positions[3*e]=origin;positions[3*e+1]=origin+F*dm.col(0);positions[3*e+2]=origin+F*dm.col(1);
            materials[e]=dm_inv;shape[e]=shape_function_gradients(dm_inv)[e%3];areas[e]=area;
        }
        ipc_simd::corotated_derivatives_tile(positions.data(),materials.data(),areas.data(),shape.data(),count,mu,lambda,g.data(),H.data());
        for(std::size_t e=0;e<count;++e) {
            SCOPED_TRACE(::testing::Message()<<"sample="<<sample<<" count="<<count<<" entry="<<e);
            const auto expected=scalar_elasticity(positions.data()+3*e,e%3);
            compare(g[e],expected.first,2e-10);compare(H[e],expected.second,2e-10);
        }
        EXPECT_TRUE((g.back().array()==12345).all());EXPECT_TRUE((H.back().array()==12345).all());
    }
}

TEST(SIMDEnergy, ElasticityMatchesEnergyFiniteDifferences) {
    for(int sample=0;sample<12;++sample)for(int role=0;role<3;++role) {
        const Mat33 rotation=Eigen::AngleAxisd(.17*sample,Vec3(1,2,3).normalized()).toRotationMatrix();
        Mat32 F=rotation.leftCols<2>();F.col(0)*=.7+.08*sample;F.col(1)*=1.2;
        std::array<Vec3,3> x{Vec3(.13,-.2,.03),Vec3::Zero(),Vec3::Zero()};
        x[1]=x[0]+F*dm.col(0);x[2]=x[0]+F*dm.col(1);
        const Vec2 q=shape_function_gradients(dm_inv)[role];Vec3 g;Mat33 H;
        ipc_simd::corotated_derivatives_tile(x.data(),&dm_inv,&area,&q,1,mu,lambda,&g,&H);
        const auto energy=[&](const std::array<Vec3,3>& p){TriangleDef def;for(int i=0;i<3;++i)def.x[i]=p[i];return corotated_energy(area,dm_inv,def,mu,lambda);};
        constexpr double h=1e-6;Vec3 fd_g;Mat33 fd_H;
        for(int axis=0;axis<3;++axis) {
            auto plus=x,minus=x;plus[role][axis]+=h;minus[role][axis]-=h;
            fd_g[axis]=(energy(plus)-energy(minus))/(2*h);
            fd_H.col(axis)=(scalar_elasticity(plus.data(),role).first-scalar_elasticity(minus.data(),role).first)/(2*h);
        }
        EXPECT_LE((g-fd_g).norm(),2e-7*(1+fd_g.norm()));
        EXPECT_LE((H-fd_H).norm(),2e-7*(1+fd_H.norm()));
    }
}

TEST(SIMDEnergy, BendingMatchesScalarAcrossRolesAndDegeneracy) {
    for(std::size_t count=1;count<=8;++count)for(int sample=0;sample<24;++sample) {
        std::array<Vec3,32> positions;
        std::array<int,8> roles;std::array<double,8> coefficients,rest;
        std::array<Vec3,9> g;std::array<Mat33,9> H;
        g.back().setConstant(12345);H.back().setConstant(12345);
        for(std::size_t e=0;e<count;++e) {
            auto def=hinge(-2.7+.23*sample+.01*e);
            if(sample==23 && e%2==0)def.x[1]=def.x[0];
            for(int i=0;i<4;++i)positions[4*e+i]=def.x[i];
            roles[e]=e%4;coefficients[e]=.8+.03*e;rest[e]=-.2;
        }
        constexpr double stiffness=.009;
        ipc_simd::bending_derivatives_tile(positions.data(),roles.data(),coefficients.data(),rest.data(),count,stiffness,g.data(),H.data());
        for(std::size_t e=0;e<count;++e) {
            HingeDef def;for(int i=0;i<4;++i)def.x[i]=positions[4*e+i];
            SCOPED_TRACE(::testing::Message()<<"sample="<<sample<<" count="<<count<<" entry="<<e);
            const auto expected=bending_node_gradient_hessian_psd(def,stiffness,coefficients[e],rest[e],roles[e]);
            compare(g[e],expected.first,2e-11);compare(H[e],expected.second,2e-11);
            if(sample<20 && e==0) {
                constexpr double h=1e-6;Vec3 finite;
                for(int axis=0;axis<3;++axis) {auto plus=def,minus=def;plus.x[roles[e]][axis]+=h;minus.x[roles[e]][axis]-=h;
                    finite[axis]=(bending_energy(plus,stiffness,coefficients[e],rest[e])-bending_energy(minus,stiffness,coefficients[e],rest[e]))/(2*h);}
                EXPECT_LE((g[e]-finite).norm(),2e-7*(1+finite.norm()));
            }
        }
        EXPECT_TRUE((g.back().array()==12345).all());EXPECT_TRUE((H.back().array()==12345).all());
    }
}

TEST(SIMDEnergy, BendingHessianMatchesEnergyAtRest) {
    for(int sample=0;sample<8;++sample)for(int role=0;role<4;++role) {
        auto def=hinge(-1.4+.4*sample);const double rest=bending_theta(def),coefficient=.8,stiffness=.009;
        Vec3 g;Mat33 H;
        ipc_simd::bending_derivatives_tile(def.x,&role,&coefficient,&rest,1,stiffness,&g,&H);
        constexpr double h=1e-6;Mat33 finite;
        for(int axis=0;axis<3;++axis) {auto plus=def,minus=def;plus.x[role][axis]+=h;minus.x[role][axis]-=h;
            finite.col(axis)=(bending_node_gradient(plus,stiffness,coefficient,rest,role)-bending_node_gradient(minus,stiffness,coefficient,rest,role))/(2*h);}
        EXPECT_LE((H-finite).norm(),2e-7*(1+finite.norm()));
    }
}

TEST(SIMDEnergy, NonfiniteGeometryRetainsScalarBehavior) {
    std::array<Vec3,3> x{Vec3::Zero(),Vec3::UnitX(),Vec3::UnitY()};
    Mat22 invalid=dm_inv;invalid(0,0)=std::numeric_limits<double>::quiet_NaN();
    const Vec2 q(-1,-1);Vec3 g;Mat33 H;
    EXPECT_THROW(ipc_simd::corotated_derivatives_tile(x.data(),&invalid,&area,&q,1,mu,lambda,&g,&H),std::runtime_error);
    auto def=hinge(.3);def.x[2][1]=std::numeric_limits<double>::quiet_NaN();
    const int role=2;const double coefficient=.8,rest=-.2;
    const auto expected=bending_node_gradient_hessian_psd(def,.009,coefficient,rest,role);
    ipc_simd::bending_derivatives_tile(def.x,&role,&coefficient,&rest,1,.009,&g,&H);
    compare(g,expected.first,2e-11);compare(H,expected.second,2e-11);
}
} // namespace simd_energy_test
