#include "general_simd.h"
#include "barrier_energy.h"
#include "bending_energy.h"
#include "friction_energy.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "segment_segment_distance.h"
#include "simulation.h"
#include "solid_ipc.h"
#include "solver.h"
#include "volumetric_corotated_energy.h"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace general_simd_tests::solid {

namespace {

constexpr double kMu = 2.3;
constexpr double kLambda = 5.7;
constexpr double kTolerance = 64.0 * std::numeric_limits<double>::epsilon();

struct Records {
    std::vector<Vec3> positions;
    std::vector<Mat33> inverse;
    std::vector<double> measure;
    std::vector<Vec3> shape;
    std::vector<TetRestData> rest;
    std::vector<int> role;

    void append(const Mat33& F, int active, double scale = 1.0) {
        Mat33 Dm;
        Dm << 1.3, 0.1, -0.2, -0.1, 1.1, 0.2, 0.04, -0.13, 0.9;
        Dm *= scale;
        const Vec3 origin(0.31, -0.19, 0.23);
        const std::vector<Vec3> X = {
            origin, origin + Dm.col(0), origin + Dm.col(1), origin + Dm.col(2)};
        const auto material = EFEMInitializeElasticMaterialState(X, {0, 1, 2, 3})[0];
        positions.push_back(origin);
        for (int column = 0; column < 3; ++column)
            positions.push_back(origin + F * Dm.col(column));
        inverse.push_back(material.Dm_inverse);
        measure.push_back(material.measure);
        shape.push_back(material.grad_N[active]);
        rest.push_back(material);
        role.push_back(active);
    }

    std::pair<Vec3, Mat33> reference(std::size_t entry, double mu, double lambda) const {
        const std::vector<Vec3> x(positions.begin() + 4 * entry, positions.begin() + 4 * entry + 4);
        const Mat33 F = ElementF(0, x, {0, 1, 2, 3}, {rest[entry]});
        CorotatedCache cache;
        cache.UpdateCache(F, CorotatedCacheMode::Lean);
        return EFEMElementNodeGradientAndPBGSBlock(cache, F, rest[entry], mu, lambda, role[entry]);
    }

    void tile(std::size_t begin, std::size_t count, Vec3* g, Mat33* H,
        double mu = kMu, double lambda = kLambda) const {
        ipc_simd::solid_derivatives_tile(positions.data() + 4 * begin,
            inverse.data() + begin, measure.data() + begin, shape.data() + begin,
            count, mu, lambda, g, H);
    }
};

void expect_close(const Vec3& actual_g, const Mat33& actual_H,
    const Vec3& expected_g, const Mat33& expected_H) {
    ASSERT_TRUE(actual_g.allFinite());
    ASSERT_TRUE(actual_H.allFinite());
    EXPECT_LE((actual_g - expected_g).norm(), kTolerance * (1.0 + expected_g.norm()));
    EXPECT_LE((actual_H - expected_H).norm(), kTolerance * (1.0 + expected_H.norm()));
}

Records make_records() {
    Records records;
    const Mat33 rotation = Eigen::AngleAxisd(0.47, Vec3(0.3, -0.2, 1.0).normalized()).toRotationMatrix();
    for (int sample = 0; sample < 11; ++sample) {
        Mat33 deformation;
        deformation << 1.07 + 0.09 * sample, -0.13, 0.07,
            0.03, 0.89 - 0.01 * sample, -0.11,
            -0.06, 0.17, 1.13;
        for (int role = 0; role < 4; ++role)
            records.append(rotation * deformation, role, 0.3 + 0.13 * sample);
    }
    return records;
}

} // namespace

TEST(GeneralSimdSolid, EveryRoleAndTileTailMatchesScalarMaterial) {
    const Records records = make_records();
    const std::array<std::pair<double, double>, 4> material = {{{kMu, kLambda}, {0.0, kLambda}, {kMu, 0.0}, {0.0, 0.0}}};
    for (const auto& [mu, lambda] : material)
        for (std::size_t width = 1; width <= ipc_simd::tile_width; ++width)
            for (std::size_t begin = 0; begin < records.role.size(); begin += width) {
                const std::size_t count = std::min(width, records.role.size() - begin);
                std::array<Vec3, ipc_simd::tile_width + 1> g;
                std::array<Mat33, ipc_simd::tile_width + 1> H;
                for (auto& value : g) value.setConstant(123.0);
                for (auto& value : H) value.setConstant(456.0);
                records.tile(begin, count, g.data(), H.data(), mu, lambda);
                for (std::size_t e = 0; e < count; ++e) {
                    SCOPED_TRACE(::testing::Message() << "record=" << begin + e << " tile=" << width << " mu=" << mu << " lambda=" << lambda);
                    const auto expected = records.reference(begin + e, mu, lambda);
                    expect_close(g[e], H[e], expected.first, expected.second);
                }
                for (std::size_t e = count; e < g.size(); ++e) {
                    EXPECT_EQ((g[e].array() == 123.0).count(), 3);
                    EXPECT_EQ((H[e].array() == 456.0).count(), 9);
                }
            }
}

TEST(GeneralSimdSolid, EveryRoleAndTileTailIsBitwiseIdenticalToScalarMaterial) {
    const Records records = make_records();
    const std::array<std::pair<double, double>, 4> material = {{{kMu, kLambda},
        {0.0, kLambda}, {kMu, 0.0}, {0.0, 0.0}}};
    for (const auto& [mu, lambda] : material)
        for (std::size_t width = 1; width <= ipc_simd::tile_width; ++width)
            for (std::size_t begin = 0; begin < records.role.size(); begin += width) {
                const std::size_t count = std::min(width, records.role.size() - begin);
                std::array<Vec3, ipc_simd::tile_width> g;
                std::array<Mat33, ipc_simd::tile_width> H;
                records.tile(begin, count, g.data(), H.data(), mu, lambda);
                for (std::size_t e = 0; e < count; ++e) {
                    SCOPED_TRACE(::testing::Message() << "record=" << begin + e
                        << " tile=" << width << " mu=" << mu << " lambda=" << lambda);
                    const auto expected = records.reference(begin + e, mu, lambda);
                    ASSERT_EQ(std::memcmp(g[e].data(), expected.first.data(), 3 * sizeof(double)), 0);
                    ASSERT_EQ(std::memcmp(H[e].data(), expected.second.data(), 9 * sizeof(double)), 0);
                }
            }
}

TEST(GeneralSimdSolid, SingularInvertedAndRestStatesMatchScalar) {
    Records records;
    const std::array<Vec3, 5> singular_values = {
        Vec3(1.0, 1.0, 1.0), Vec3(-0.8, 1.1, 0.7), Vec3(0.0, 0.9, 1.2),
        Vec3(0.0, 0.0, 0.0), Vec3(1e-10, 0.8, 1.3)};
    for (const auto& values : singular_values)
        for (int role = 0; role < 4; ++role) records.append(values.asDiagonal(), role);
    for (std::size_t begin = 0; begin < records.role.size(); begin += ipc_simd::tile_width) {
        std::array<Vec3, ipc_simd::tile_width> g;
        std::array<Mat33, ipc_simd::tile_width> H;
        const std::size_t count = std::min(ipc_simd::tile_width, records.role.size() - begin);
        records.tile(begin, count, g.data(), H.data());
        for (std::size_t e = 0; e < count; ++e) {
            const auto expected = records.reference(begin + e, kMu, kLambda);
            expect_close(g[e], H[e], expected.first, expected.second);
            EXPECT_LE((H[e] - H[e].transpose()).norm(), kTolerance * (1.0 + H[e].norm()));
            Eigen::SelfAdjointEigenSolver<Mat33> eigen(H[e]);
            EXPECT_GE(eigen.eigenvalues().minCoeff(), -kTolerance * (1.0 + H[e].norm()));
        }
    }
}

TEST(GeneralSimdSolid, PacketBoundariesDoNotChangeOrderedAccumulation) {
    const Records records = make_records();
    Vec3 baseline_g = Vec3::Zero();
    Mat33 baseline_H = Mat33::Zero();
    constexpr double dt2 = 0.0004;
    for (std::size_t width = 1; width <= ipc_simd::tile_width; ++width) {
        Vec3 sum_g = Vec3::Zero();
        Mat33 sum_H = Mat33::Zero();
        for (std::size_t begin = 0; begin < records.role.size(); begin += width) {
            std::array<Vec3, ipc_simd::tile_width> g;
            std::array<Mat33, ipc_simd::tile_width> H;
            const std::size_t count = std::min(width, records.role.size() - begin);
            records.tile(begin, count, g.data(), H.data());
            for (std::size_t e = 0; e < count; ++e) {
                sum_g += dt2 * g[e];
                sum_H += dt2 * H[e];
            }
        }
        if (width == 1) {
            baseline_g = sum_g;
            baseline_H = sum_H;
        } else {
            EXPECT_EQ(std::memcmp(sum_g.data(), baseline_g.data(), sizeof(double) * 3), 0);
            EXPECT_EQ(std::memcmp(sum_H.data(), baseline_H.data(), sizeof(double) * 9), 0);
        }
    }
}

TEST(GeneralSimdSolid, MixedSignedSvdBranchesAndTailsMatchSingleEntryTilesBitwise) {
    Records records;
    const std::array<Vec3, 8> values = {Vec3(1, 1, 1), Vec3(-0.8, 1.1, 0.7),
        Vec3(0, 0.9, 1.2), Vec3::Zero(), Vec3(1e-12, 0.8, 1.3),
        Vec3(0, 0, 1), Vec3(1e-13, 2e-13, -1e-13), Vec3(1e25, 2e25, -1e25)};
    for (int sample = 0; sample < 37; ++sample) {
        const Mat33 rotation = Eigen::AngleAxisd(0.07 * sample,
            Vec3(0.3, -0.2, 1.0).normalized()).toRotationMatrix();
        records.append(rotation * values[sample % values.size()].asDiagonal(), sample % 4);
    }
    std::vector<Vec3> reference_g(records.role.size());
    std::vector<Mat33> reference_H(records.role.size());
    for (std::size_t i = 0; i < records.role.size(); ++i)
        records.tile(i, 1, &reference_g[i], &reference_H[i]);
    for (std::size_t width = 1; width <= ipc_simd::tile_width; ++width) {
        for (std::size_t first = 0; first < records.role.size(); first += width) {
            const std::size_t count = std::min(width, records.role.size() - first);
            std::array<Vec3, ipc_simd::tile_width> g;
            std::array<Mat33, ipc_simd::tile_width> H;
            records.tile(first, count, g.data(), H.data());
            for (std::size_t i = 0; i < count; ++i) {
                SCOPED_TRACE(::testing::Message() << "width=" << width << " record=" << first + i);
                ASSERT_EQ(std::memcmp(g[i].data(), reference_g[first + i].data(), 3 * sizeof(double)), 0);
                ASSERT_EQ(std::memcmp(H[i].data(), reference_H[first + i].data(), 9 * sizeof(double)), 0);
            }
        }
    }
}

TEST(GeneralSimdSolid, GradientMatchesFiniteDifferenceOfCorotatedEnergy) {
    Records records;
    Mat33 F;
    F << 1.17, 0.09, -0.07, -0.03, 0.93, 0.12, 0.08, -0.05, 1.09;
    for (int role = 0; role < 4; ++role) records.append(F, role);
    std::array<Vec3, 4> gradients;
    std::array<Mat33, 4> blocks;
    records.tile(0, 4, gradients.data(), blocks.data());
    const auto energy = [&](std::size_t entry) {
        const std::vector<Vec3> x(records.positions.begin() + 4 * entry, records.positions.begin() + 4 * entry + 4);
        const Mat33 deformation = ElementF(0, x, {0, 1, 2, 3}, {records.rest[entry]});
        CorotatedCache cache;
        cache.UpdateCache(deformation, CorotatedCacheMode::Lean);
        return EFEMElementInternalEnergy(cache, deformation, records.rest[entry], kMu, kLambda);
    };
    constexpr double step = 1e-6;
    for (int role = 0; role < 4; ++role)
        for (int axis = 0; axis < 3; ++axis) {
            double& coordinate = records.positions[4 * role + role][axis];
            const double original = coordinate;
            coordinate = original + step;
            const double plus = energy(role);
            coordinate = original - step;
            const double minus = energy(role);
            coordinate = original;
            EXPECT_NEAR(gradients[role][axis], (plus - minus) / (2.0 * step), 1e-8);
        }
}

TEST(GeneralSimdSolid, EmptyTileDoesNotAccessPointers) {
    ipc_simd::solid_derivatives_tile(nullptr, nullptr, nullptr, nullptr, 0,
        kMu, kLambda, nullptr, nullptr);
}

} // namespace general_simd_tests::solid

namespace general_simd_tests::rigid {

namespace {

constexpr double dt = 0.031, d_hat = 0.55, stiffness = 370.0;

ipc_simd::RigidContactOutput reference(const ipc_simd::RigidContactInput& input,
    RigidDerivativeMode mode, double friction) {
    ipc_simd::RigidContactOutput result;
    const auto& x = input.positions;
    FrozenFrictionContact contact;
    const Vec4 q(1.0, 0.0, 0.0, 0.0);
    const Vec3 omega = Vec3::Zero();
    if (input.segment_segment) {
        if (friction != 0.0) {
            const auto evaluation = make_segment_segment_contact_evaluation(x, d_hat, stiffness);
            result.barrier = segment_segment_barrier_rb(x[0], x[1], x[2], x[3],
                input.body_references, input.side, q, omega, dt, d_hat, mode,
                1e-12, input.kinematics, &evaluation.dr, &evaluation.b_prime, &evaluation.b_double_prime);
            contact = make_segment_segment_frozen_friction_contact(x, input.previous_positions, evaluation, dt, 0.1);
        } else result.barrier = segment_segment_barrier_rb(x[0], x[1], x[2], x[3],
            input.body_references, input.side, q, omega, dt, d_hat, mode, 1e-12, input.kinematics);
    } else {
        if (friction != 0.0) {
            const auto evaluation = make_node_triangle_contact_evaluation(x, d_hat, stiffness);
            result.barrier = node_triangle_barrier_rb(x[0], x[1], x[2], x[3],
                input.body_references, input.side, q, omega, dt, d_hat, mode,
                1e-12, input.kinematics, &evaluation.dr, &evaluation.b_prime, &evaluation.b_double_prime);
            contact = make_node_triangle_frozen_friction_contact(x, input.previous_positions, evaluation, dt, 0.1);
        } else result.barrier = node_triangle_barrier_rb(x[0], x[1], x[2], x[3],
            input.body_references, input.side, q, omega, dt, d_hat, mode, 1e-12, input.kinematics);
    }
    if (!contact.active) return result;
    const int first = input.side == RigidBarrierSide::FirstPrimitive ? 0 : (input.segment_segment ? 2 : 1);
    const int last = input.side == RigidBarrierSide::FirstPrimitive ? (input.segment_segment ? 1 : 0) : 3;
    const unsigned mask = input.body_role_mask ? input.body_role_mask : ((1u << (last + 1)) - (1u << first));
    double weight = 0.0;
    Mat33 J = Mat33::Zero();
    for (int role = 0; role < 4; ++role) {
        if (!(mask & (1u << role))) continue;
        weight += contact.weights[role];
        if (mode != RigidDerivativeMode::TranslationHessian && updates_rigid_orientation(input.update_mode))
            J += contact.weights[role] * dx_domega(input.body_references[role], *input.kinematics);
    }
    const Mat33 T = weight * Mat33::Identity();
    const auto [g, H] = frozen_friction_relative_gradient_and_hessian(contact, friction, dt * dt);
    const bool translation = updates_rigid_translation(input.update_mode);
    const bool orientation = updates_rigid_orientation(input.update_mode);
    if (translation && mode != RigidDerivativeMode::OrientationHessian)
        result.friction.translation_gradient = T.transpose() * g;
    if (orientation && mode != RigidDerivativeMode::TranslationHessian)
        result.friction.orientation_gradient = J.transpose() * g;
    if (translation && (mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::TranslationHessian))
        result.friction.translation_translation_hessian = T.transpose() * H * T;
    if (orientation && (mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::OrientationHessian))
        result.friction.orientation_orientation_hessian = J.transpose() * H * J;
    if (translation && orientation && mode == RigidDerivativeMode::Full)
        result.friction.translation_orientation_hessian = T.transpose() * H * J;
    return result;
}

void compare(const RigidEnergyDerivatives& actual, const RigidEnergyDerivatives& expected) {
    const auto check = [](const auto& a, const auto& b) {
        EXPECT_TRUE(a.allFinite());
        EXPECT_LE((a - b).norm(), 3e-12 * (1.0 + b.norm()));
    };
    check(actual.translation_gradient, expected.translation_gradient);
    check(actual.orientation_gradient, expected.orientation_gradient);
    check(actual.translation_translation_hessian, expected.translation_translation_hessian);
    check(actual.translation_orientation_hessian, expected.translation_orientation_hessian);
    check(actual.orientation_orientation_hessian, expected.orientation_orientation_hessian);
}

void add(RigidEnergyDerivatives& total, const RigidEnergyDerivatives& value) {
    total.translation_gradient += value.translation_gradient;
    total.orientation_gradient += value.orientation_gradient;
    total.translation_translation_hessian += value.translation_translation_hessian;
    total.translation_orientation_hessian += value.translation_orientation_hessian;
    total.orientation_orientation_hessian += value.orientation_orientation_hessian;
}

ipc_simd::RigidContactInput fixture(int index, const QuaternionOmegaKinematics& kinematics) {
    ipc_simd::RigidContactInput input;
    input.segment_segment = (index / 2) % 2 != 0;
    input.side = index % 2 == 0 ? RigidBarrierSide::FirstPrimitive : RigidBarrierSide::SecondPrimitive;
    input.kinematics = &kinematics;
    const int feature = (index / 4) % 6;
    if (!input.segment_segment) {
        input.positions = {Vec3(0.2, 0.25, 0.17), Vec3::Zero(), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
        if (feature == 1) input.positions[0] = Vec3(0.4, -0.13, 0.16);
        if (feature == 2) input.positions[0] = Vec3(-0.12, -0.11, 0.16);
        if (feature == 3) input.positions[0][2] *= -1.0;
        if (feature == 4) input.positions[3] = Vec3(0.6, 0.0, 0.0);
        if (feature == 5) input.positions[0][2] = 0.9;
    } else {
        input.positions = {Vec3(-0.7, 0.0, 0.0), Vec3(0.7, 0.0, 0.0), Vec3(0.0, -0.6, 0.17), Vec3(0.0, 0.6, 0.17)};
        if (feature == 1) { input.positions[2][1] = 0.12; input.positions[3][1] = 0.9; }
        if (feature == 2) { input.positions[2] = Vec3(0.9, 0.12, 0.17); input.positions[3] = Vec3(0.9, 0.9, 0.17); }
        if (feature == 3) { input.positions[2] = Vec3(-0.4, 0.12, 0.17); input.positions[3] = Vec3(0.5, 0.12, 0.17); }
        if (feature == 4) input.positions[1] = input.positions[0];
        if (feature == 5) { input.positions[2][2] = 0.9; input.positions[3][2] = 0.9; }
    }
    for (int i = 0; i < 4; ++i) {
        input.body_references[i] = input.positions[i] + Vec3(0.4, -0.7, 0.9);
        input.previous_positions[i] = input.positions[i] - (i + 1) * Vec3(0.001, -0.003, 0.002);
    }
    return input;
}

} // namespace

TEST(GeneralSIMDRigid, CoupledBarrierFrictionFeaturesSidesModesAndTailsMatchScalar) {
    const auto kinematics = quaternion_omega_kinematics(Vec4(0.8, 0.2, -0.4, 0.4).normalized(), Vec3(1.2, -0.9, 1.7), dt, true);
    std::array<ipc_simd::RigidContactInput, ipc_simd::contact_tile_width> inputs;
    std::array<ipc_simd::RigidContactOutput, ipc_simd::contact_tile_width> outputs;
    for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = fixture(static_cast<int>(i), kinematics);
    for (auto mode : {RigidDerivativeMode::Full, RigidDerivativeMode::Gradient,
             RigidDerivativeMode::TranslationHessian, RigidDerivativeMode::OrientationHessian})
        for (double friction : {0.0, 0.37})
            for (std::size_t count : {std::size_t(0), std::size_t(1), std::size_t(3), std::size_t(17), inputs.size()}) {
                ipc_simd::rigid_contact_derivatives_tile(inputs.data(), count, d_hat,
                    stiffness, friction, dt, 0.1, mode, outputs.data());
                for (std::size_t i = 0; i < count; ++i) {
                    SCOPED_TRACE(::testing::Message() << "entry=" << i << " count=" << count << " mode=" << int(mode) << " friction=" << friction);
                    const auto expected = reference(inputs[i], mode, friction);
                    compare(outputs[i].barrier, expected.barrier);
                    compare(outputs[i].friction, expected.friction);
                }
            }
}

TEST(GeneralSIMDRigid, RobustTinyGapUsesScalarGeometryForBothBodiesAndAllModes) {
    const std::array<Vec3,4> positions{
        Vec3(0.30643120172698846,1.2537041974945615,0.345215273017485),
        Vec3(0.2716303759805404,1.2435156840614272,0.3522099917587509),
        Vec3(0.3476899357030978,1.254750762137517,0.34482031348588066),
        Vec3(0.12785112730426101,1.247125469633767,0.34839148080369126)};
    ASSERT_TRUE(segment_segment_distance(positions[0], positions[1], positions[2], positions[3]).robust);
    const auto kinematics = quaternion_omega_kinematics(
        Vec4(0.8, 0.2, -0.4, 0.4).normalized(), Vec3(1.2, -0.9, 1.7), dt, true);
    std::array<ipc_simd::RigidContactInput,3> inputs;
    std::array<ipc_simd::RigidContactOutput,3> outputs;
    for (std::size_t entry = 0; entry < inputs.size(); ++entry) {
        inputs[entry] = fixture(static_cast<int>(entry) + 2, kinematics);
        if (entry == 2) continue; // Keep an ordinary contact in the tail packet.
        inputs[entry].positions = positions;
        for (int role = 0; role < 4; ++role) {
            inputs[entry].body_references[role] = positions[role] + Vec3(.4,-.7,.9);
            inputs[entry].previous_positions[role] = positions[role]
                - (role + 1) * Vec3(.001,-.003,.002);
        }
    }
    for (auto mode : {RigidDerivativeMode::Full, RigidDerivativeMode::Gradient,
             RigidDerivativeMode::TranslationHessian, RigidDerivativeMode::OrientationHessian})
        for (double friction : {0.0, .37}) {
            SCOPED_TRACE(::testing::Message() << "mode=" << int(mode) << " friction=" << friction);
            ipc_simd::rigid_contact_derivatives_tile(inputs.data(), inputs.size(), d_hat,
                stiffness, friction, dt, .1, mode, outputs.data());
            for (std::size_t entry = 0; entry < inputs.size(); ++entry) {
                SCOPED_TRACE(::testing::Message() << "entry=" << entry);
                const auto expected = reference(inputs[entry], mode, friction);
                compare(outputs[entry].barrier, expected.barrier);
                compare(outputs[entry].friction, expected.friction);
            }
        }
}

TEST(GeneralSIMDRigid, FrictionUsesSignedMultiRoleJacobianAndUpdateMode) {
    const auto kinematics = quaternion_omega_kinematics(Vec4(1, 0, 0, 0), Vec3(0.5, 1.1, -0.9), dt, true);
    std::array<ipc_simd::RigidContactInput, 4> inputs;
    std::array<ipc_simd::RigidContactOutput, 4> outputs;
    for (int i = 0; i < 4; ++i) {
        inputs[i] = fixture(1, kinematics);
        inputs[i].update_mode = static_cast<RigidBodyUpdateMode>(i);
    }
    // Include roles from both sides: their signed weights cancel in COM.
    inputs[0].body_role_mask = 15;
    ipc_simd::rigid_contact_derivatives_tile(inputs.data(), inputs.size(), d_hat,
        stiffness, 0.45, dt, 0.1, RigidDerivativeMode::Full, outputs.data());
    for (int i = 0; i < 4; ++i) {
        const auto expected = reference(inputs[i], RigidDerivativeMode::Full, 0.45);
        compare(outputs[i].barrier, expected.barrier);
        compare(outputs[i].friction, expected.friction);
    }
    EXPECT_LT(outputs[0].friction.translation_gradient.norm(), 1e-14);
    EXPECT_LT(outputs[0].friction.translation_translation_hessian.norm(), 1e-14);
    EXPECT_GT(outputs[1].friction.translation_translation_hessian.norm(), 1e-5);
    EXPECT_GT(outputs[2].friction.orientation_orientation_hessian.norm(), 1e-8);
}

TEST(GeneralSIMDRigid, TriangleOrientationRequiresCrossBlocksAndQuaternionCurvature) {
    const auto kinematics = quaternion_omega_kinematics(Vec4(1, 0, 0, 0), Vec3(3.0, 2.0, -1.0), dt, true);
    const auto input = fixture(1, kinematics);
    ipc_simd::RigidContactOutput output;
    ipc_simd::rigid_contact_derivatives_tile(&input, 1, d_hat, stiffness, 0.0,
        dt, 0.1, RigidDerivativeMode::OrientationHessian, &output);
    const auto expected = reference(input, RigidDerivativeMode::OrientationHessian, 0.0);
    compare(output.barrier, expected.barrier);
    const auto& x = input.positions;
    Mat33 diagonal_only = Mat33::Zero(), curvature = Mat33::Zero();
    for (int i = 1; i < 4; ++i) {
        const auto [g, H] = node_triangle_barrier_self_gradient_and_hessian(x[0], x[1], x[2], x[3], d_hat, i);
        const Mat33 J = dx_domega(input.body_references[i], kinematics);
        diagonal_only += J.transpose() * H * J;
        const auto second = d2x_domega2(input.body_references[i], kinematics);
        for (int c = 0; c < 3; ++c) curvature += g[c] * second[c];
    }
    EXPECT_GT(curvature.norm(), 1e-5);
    EXPECT_GT((output.barrier.orientation_orientation_hessian - diagonal_only - curvature).norm(), 1e-5);
}

TEST(GeneralSIMDRigid, OrderedAccumulatorMatchesTilesAndCooperativeHelpers) {
    RefMesh mesh;
    DeformedState state;
    state.orientations = {Vec4(1, 0, 0, 0)};
    const std::vector<Vec3> omega = {Vec3(0.4, -0.8, 0.6)};
    const auto kinematics = quaternion_omega_kinematics(state.orientations[0], omega[0], dt, true);
    const auto nt = fixture(1, kinematics), ss = fixture(2, kinematics);
    std::vector<Vec3> positions(nt.positions.begin(), nt.positions.end());
    positions.insert(positions.end(), ss.positions.begin(), ss.positions.end());
    state.deformed_positions.assign(nt.previous_positions.begin(), nt.previous_positions.end());
    state.deformed_positions.insert(state.deformed_positions.end(), ss.previous_positions.begin(), ss.previous_positions.end());
    mesh.node_to_rb = {-1, 0, 0, 0, 0, 0, -1, -1};
    mesh.rb_update_modes = {RigidBodyUpdateMode::TranslationAndOrientation};
    mesh.ref_positions = {{nt.body_references[1], nt.body_references[2], nt.body_references[3], ss.body_references[0], ss.body_references[1]}};
    const std::vector<int> local = {-1, 0, 1, 2, 3, 4, -1, -1};
    BroadPhase::Cache cache;
    cache.nt_pairs.resize(40);
    for (auto& pair : cache.nt_pairs) { pair.node = 0; pair.tri_v[0] = 1; pair.tri_v[1] = 2; pair.tri_v[2] = 3; }
    cache.ss_pairs.resize(37);
    for (auto& pair : cache.ss_pairs) for (int role = 0; role < 4; ++role) pair.v[role] = role + 4;
    std::vector<int> nt_indices(40), ss_indices(37);
    std::iota(nt_indices.begin(), nt_indices.end(), 0);
    std::iota(ss_indices.begin(), ss_indices.end(), 0);
    SimParams params;
    params.d_hat = d_hat;
    params.k_barrier = stiffness;
    params.friction_coefficient = 0.37;
    params.friction_velocity_epsilon = 0.1;
    for (double friction : {0.0, 0.37}) {
        params.friction_coefficient = friction;
    for (auto mode : {RigidDerivativeMode::Full, RigidDerivativeMode::Gradient,
             RigidDerivativeMode::TranslationHessian, RigidDerivativeMode::OrientationHessian}) {
        ipc_simd::RigidContactOutput expected;
        const auto nt_value = reference(nt, mode, params.friction_coefficient);
        const auto ss_value = reference(ss, mode, params.friction_coefficient);
        for (int i = 0; i < 40; ++i) { add(expected.barrier, nt_value.barrier); add(expected.friction, nt_value.friction); }
        for (int i = 0; i < 37; ++i) { add(expected.barrier, ss_value.barrier); add(expected.friction, ss_value.friction); }
        const auto serial = ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
            nt_indices, ss_indices, local, positions, omega, params, dt, mode, &kinematics);
        compare(serial.barrier, expected.barrier);
        compare(serial.friction, expected.friction);
        ipc_simd::RigidContactOutput parallel;
        int leader_calls = 0;
        const std::function<void()> leader = [&] { ++leader_calls; };
        #pragma omp parallel num_threads(4)
        {
            #pragma omp single
            parallel = ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
                nt_indices, ss_indices, local, positions, omega, params, dt, mode,
                &kinematics, true, &leader);
        }
        EXPECT_EQ(leader_calls, 1);
        compare(parallel.barrier, serial.barrier);
        compare(parallel.friction, serial.friction);
    }
    }
    // Repeated solves change both admission and derivative mode. Reusing
    // contact storage must not retain contributions from an earlier solve.
    for (int pass = 0; pass < 16; ++pass) {
        params.friction_coefficient = pass % 2 ? 0.37 : 0.0;
        const RigidDerivativeMode modes[] = {RigidDerivativeMode::Full, RigidDerivativeMode::Gradient,
            RigidDerivativeMode::TranslationHessian, RigidDerivativeMode::OrientationHessian};
        const auto mode = modes[(pass / 2) % 4];
        positions[0] = nt.positions[0] + (pass == 2 || pass == 3 ? Vec3(30, 0, 0) : Vec3::Zero());
        nt_indices.resize(pass % 3 == 0 ? 321 : pass % 3 == 1 ? 35 : 80);
        for (std::size_t i = 0; i < nt_indices.size(); ++i) nt_indices[i] = i % 40;
        const auto serial = ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
            nt_indices, ss_indices, local, positions, omega, params, dt, mode, &kinematics);
        ipc_simd::RigidContactOutput parallel;
        #pragma omp parallel num_threads(4)
        {
            #pragma omp master
            parallel = ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
                nt_indices, ss_indices, local, positions, omega, params, dt, mode,
                &kinematics, true);
        }
        compare(parallel.barrier, serial.barrier);
        compare(parallel.friction, serial.friction);
        for (int c = 0; c < 3; ++c) {
            EXPECT_EQ(parallel.barrier.translation_gradient[c], serial.barrier.translation_gradient[c]);
            EXPECT_EQ(parallel.barrier.orientation_gradient[c], serial.barrier.orientation_gradient[c]);
            EXPECT_EQ(parallel.friction.translation_gradient[c], serial.friction.translation_gradient[c]);
            EXPECT_EQ(parallel.friction.orientation_gradient[c], serial.friction.orientation_gradient[c]);
        }
        for (int c = 0; c < 9; ++c) {
            EXPECT_EQ(parallel.barrier.translation_translation_hessian.data()[c], serial.barrier.translation_translation_hessian.data()[c]);
            EXPECT_EQ(parallel.barrier.orientation_orientation_hessian.data()[c], serial.barrier.orientation_orientation_hessian.data()[c]);
            EXPECT_EQ(parallel.barrier.translation_orientation_hessian.data()[c], serial.barrier.translation_orientation_hessian.data()[c]);
            EXPECT_EQ(parallel.friction.translation_translation_hessian.data()[c], serial.friction.translation_translation_hessian.data()[c]);
            EXPECT_EQ(parallel.friction.orientation_orientation_hessian.data()[c], serial.friction.orientation_orientation_hessian.data()[c]);
            EXPECT_EQ(parallel.friction.translation_orientation_hessian.data()[c], serial.friction.translation_orientation_hessian.data()[c]);
        }
    }
    // Friction-free calls do not read previous positions, even if absent.
    params.friction_coefficient = 0.0;
    state.deformed_positions.clear();
    EXPECT_NO_THROW(ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
        nt_indices, ss_indices, local, positions, omega, params, dt,
        RigidDerivativeMode::TranslationHessian));
}

TEST(GeneralSIMDRigid, RejectedContactsSkipExactOverlapAndPreviousPositionAccess) {
    RefMesh mesh;
    mesh.node_to_rb = {0, 0, 0, 0};
    mesh.rb_update_modes = {RigidBodyUpdateMode::TranslationAndOrientation};
    DeformedState state;
    BroadPhase::Cache cache;
    cache.nt_pairs.resize(1);
    auto& pair = cache.nt_pairs[0];
    pair.node = 0; pair.tri_v[0] = 1; pair.tri_v[1] = 2; pair.tri_v[2] = 3;
    SimParams params;
    params.d_hat = d_hat; params.k_barrier = stiffness; params.friction_coefficient = 0.4;
    const std::vector<Vec3> positions(4, Vec3::Zero());
    // Same-body self pairs are rejected before exact distance/previous reads.
    EXPECT_NO_THROW(ipc_simd::rigid_contact_derivatives(0, mesh, state, cache,
        {0}, {}, {}, positions, {}, params, dt, RigidDerivativeMode::TranslationHessian));
    // An admitted exact intersection keeps the scalar exception contract.
    const auto kinematics = quaternion_omega_kinematics(Vec4(1, 0, 0, 0), Vec3::Zero(), dt, true);
    auto input = fixture(0, kinematics);
    input.positions[0][2] = 0.0;
    ipc_simd::RigidContactOutput output;
    EXPECT_THROW(ipc_simd::rigid_contact_derivatives_tile(&input, 1, d_hat,
        stiffness, 0.0, dt, 0.1, RigidDerivativeMode::Full, &output), std::runtime_error);
    // An admitted inactive contact still validates the frozen builder's
    // smoothing parameters, while unneeded previous coordinates may be NaN.
    input.positions[0][2] = 0.9;
    for (auto& p : input.previous_positions) p.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_NO_THROW(ipc_simd::rigid_contact_derivatives_tile(&input, 1, d_hat,
        stiffness, 0.4, dt, 0.1, RigidDerivativeMode::Full, &output));
    EXPECT_THROW(ipc_simd::rigid_contact_derivatives_tile(&input, 1, d_hat,
        stiffness, 0.4, dt, 0.0, RigidDerivativeMode::Full, &output), std::invalid_argument);
}

TEST(GeneralSIMDRigid, RepeatedNodePullbacksPreserveBitsAcrossTilesKeysAndKinematics) {
    std::array<QuaternionOmegaKinematics, 2> kinematics;
    std::array<ipc_simd::RigidContactInput, ipc_simd::contact_tile_width> inputs;
    std::array<ipc_simd::RigidContactOutput, ipc_simd::contact_tile_width> expected, actual;
    const auto equal = [](const RigidEnergyDerivatives& a, const RigidEnergyDerivatives& b) {
        const auto field = [](const auto& x, const auto& y) {
            EXPECT_TRUE(x.allFinite());
            EXPECT_EQ(std::memcmp(x.data(),y.data(),x.size()*sizeof(double)),0);
        };
        field(a.translation_gradient,b.translation_gradient);
        field(a.orientation_gradient,b.orientation_gradient);
        field(a.translation_translation_hessian,b.translation_translation_hessian);
        field(a.translation_orientation_hessian,b.translation_orientation_hessian);
        field(a.orientation_orientation_hessian,b.orientation_orientation_hessian);
    };
    for (int round = 0; round < 4; ++round) {
        // The same object addresses hold different kinematics on later calls.
        kinematics[0] = quaternion_omega_kinematics(Vec4(.8,.2,-.4,.4).normalized(),
            Vec3(1.2,-.9,1.7)+round*Vec3(.01,.02,-.03),dt,true);
        kinematics[1] = quaternion_omega_kinematics(Vec4(.3,-.2,.6,.7).normalized(),
            Vec3(-.2,.6,.5),dt,true);
        for (std::size_t e = 0; e < inputs.size(); ++e) {
            inputs[e] = fixture(static_cast<int>(e),kinematics[(e/4)%2]);
            for (int role = 0; role < 4; ++role) {
                const int node = (e+role)%8;
                inputs[e].body_references[role] = Vec3(node&1?1.:-1.,node&2?1.:-1.,node&4?1.:-1.);
                if (round == 1) inputs[e].body_references[role].x() =
                    std::nextafter(inputs[e].body_references[role].x(),2.);
                if (round == 2) inputs[e].body_references[role] += Vec3(e*.001,role*.003,e*.002);
            }
            if (round == 3) inputs[e].body_references[0].x() = e%2 ? -0. : 0.;
        }
        for (auto mode : {RigidDerivativeMode::Full,RigidDerivativeMode::Gradient,
                 RigidDerivativeMode::TranslationHessian,RigidDerivativeMode::OrientationHessian})
            for (double friction : {0.,.37}) {
                for (std::size_t e = 0; e < inputs.size(); ++e)
                    ipc_simd::rigid_contact_derivatives_tile(&inputs[e],1,d_hat,
                        stiffness,friction,dt,.1,mode,&expected[e]);
                ipc_simd::rigid_contact_derivatives_tile(inputs.data(),inputs.size(),d_hat,
                    stiffness,friction,dt,.1,mode,actual.data());
                for (std::size_t e = 0; e < inputs.size(); ++e) {
                    SCOPED_TRACE(::testing::Message()<<"round="<<round<<" entry="<<e<<" mode="<<int(mode));
                    equal(actual[e].barrier,expected[e].barrier);
                    equal(actual[e].friction,expected[e].friction);
                }
            }
    }
}

TEST(GeneralSIMDRigid, PointTranslationPacketsMatchMixedPacketsAndClearUnrequestedFields) {
    const auto kinematics=quaternion_omega_kinematics(Vec4(.8,.2,-.4,.4).normalized(),Vec3(1.2,-.9,1.7),dt,true);
    std::array<ipc_simd::RigidContactInput,ipc_simd::contact_tile_width> inputs;
    std::array<ipc_simd::RigidContactOutput,ipc_simd::contact_tile_width> expected,actual;
    const auto poison=[](RigidEnergyDerivatives& x){
        const double n=std::numeric_limits<double>::quiet_NaN();
        x.translation_gradient.setConstant(n);x.orientation_gradient.setConstant(n);
        x.translation_translation_hessian.setConstant(n);x.translation_orientation_hessian.setConstant(n);x.orientation_orientation_hessian.setConstant(n);
    };
    const auto equal=[](const RigidEnergyDerivatives& x,const RigidEnergyDerivatives& y){
        const auto field=[](const auto& a,const auto& b){EXPECT_TRUE(a.allFinite());EXPECT_EQ(std::memcmp(a.data(),b.data(),a.size()*sizeof(double)),0);};
        field(x.translation_gradient,y.translation_gradient);field(x.orientation_gradient,y.orientation_gradient);
        field(x.translation_translation_hessian,y.translation_translation_hessian);field(x.translation_orientation_hessian,y.translation_orientation_hessian);field(x.orientation_orientation_hessian,y.orientation_orientation_hessian);
    };
    for(int pattern:{0,1,2}) {
        for(std::size_t e=0;e<inputs.size();++e) {
            const bool segment=pattern==1 || (pattern==2 && e%2);
            inputs[e]=fixture(4*(e%6)+(segment?2:0)+(e%2),kinematics);
        }
        for(auto mode:{RigidDerivativeMode::Full,RigidDerivativeMode::TranslationHessian,RigidDerivativeMode::Gradient,RigidDerivativeMode::OrientationHessian,RigidDerivativeMode::TranslationHessian})
            for(double friction:{0.,.37})for(std::size_t count:{std::size_t(1),std::size_t(3),std::size_t(7),std::size_t(8),std::size_t(9),std::size_t(17),inputs.size()}) {
                for(std::size_t e=0;e<count;++e) {
                    poison(actual[e].barrier);poison(actual[e].friction);
                    ipc_simd::rigid_contact_derivatives_tile(&inputs[e],1,d_hat,stiffness,friction,dt,.1,mode,&expected[e]);
                }
                ipc_simd::rigid_contact_derivatives_tile(inputs.data(),count,d_hat,stiffness,friction,dt,.1,mode,actual.data());
                for(std::size_t e=0;e<count;++e) {
                    SCOPED_TRACE(::testing::Message()<<"pattern="<<pattern<<" mode="<<int(mode)<<" count="<<count<<" entry="<<e);
                    equal(actual[e].barrier,expected[e].barrier);equal(actual[e].friction,expected[e].friction);
                }
            }
    }
}

TEST(GeneralSIMDRigid, MultiTileRangesMatchFreshContactsWhenKinematicsAndReferencesChange) {
    RefMesh mesh;
    DeformedState state;
    mesh.node_to_rb = {-1, 0, 0, 0};
    mesh.rb_update_modes = {RigidBodyUpdateMode::TranslationAndOrientation};
    const std::vector<int> local = {-1, 0, 1, 2}, empty;
    BroadPhase::Cache cache;
    cache.nt_pairs.resize(1);
    cache.nt_pairs[0].node = 0;
    for (int i=0;i<3;++i) cache.nt_pairs[0].tri_v[i]=i+1;
    SimParams params;
    params.d_hat=d_hat;params.k_barrier=stiffness;params.friction_velocity_epsilon=.1;
    QuaternionOmegaKinematics kinematics;
    const auto equal=[](const RigidEnergyDerivatives& a,const RigidEnergyDerivatives& b){
        const auto check=[](const auto& x,const auto& y){EXPECT_TRUE(x.allFinite());EXPECT_EQ(std::memcmp(x.data(),y.data(),sizeof(double)*x.size()),0);};
        check(a.translation_gradient,b.translation_gradient);check(a.orientation_gradient,b.orientation_gradient);
        check(a.translation_translation_hessian,b.translation_translation_hessian);check(a.translation_orientation_hessian,b.translation_orientation_hessian);check(a.orientation_orientation_hessian,b.orientation_orientation_hessian);
    };
    for(int pass=0;pass<8;++pass) {
        state.orientations={Vec4(.8+.01*pass,.2,-.4,.4).normalized()};
        const std::vector<Vec3> omega={Vec3(.4+.03*pass,-.8,.6-.01*pass)};
        // Reassign the same object: an address alone must not survive a call.
        kinematics=quaternion_omega_kinematics(state.orientations[0],omega[0],dt,true);
        auto input=fixture(1,kinematics);
        input.positions[0][2]+=.003*pass;
        for(int i=1;i<4;++i)input.body_references[i][pass%3]+=.001*pass;
        mesh.ref_positions={{input.body_references[1],input.body_references[2],input.body_references[3]}};
        const std::vector<Vec3> positions(input.positions.begin(),input.positions.end());
        state.deformed_positions.assign(input.previous_positions.begin(),input.previous_positions.end());
        for(auto mode:{RigidDerivativeMode::Full,RigidDerivativeMode::Gradient,RigidDerivativeMode::OrientationHessian,RigidDerivativeMode::TranslationHessian})
            for(double friction:{0.,.37})for(int count:{31,32,33,65,257,1031}) {
                params.friction_coefficient=friction;
                std::vector<int> indices(count,0);
                ipc_simd::RigidContactOutput one,expected;
                ipc_simd::rigid_contact_derivatives_tile(&input,1,d_hat,stiffness,friction,dt,.1,mode,&one);
                for(int i=0;i<count;++i){add(expected.barrier,one.barrier);add(expected.friction,one.friction);}
                const auto actual=ipc_simd::rigid_contact_derivatives(0,mesh,state,cache,indices,empty,local,positions,omega,params,dt,mode,&kinematics);
                SCOPED_TRACE(::testing::Message()<<"pass="<<pass<<" count="<<count<<" mode="<<int(mode));
                equal(actual.barrier,expected.barrier);equal(actual.friction,expected.friction);
            }
    }
}

} // namespace general_simd_tests::rigid

namespace general_simd_tests::contact_bitwise {

namespace {

struct ContactFixture {
    const char* name;
    bool segment;
    std::array<Vec3, 4> positions;
};

std::vector<ContactFixture> fixtures() {
    std::vector<ContactFixture> result;
    const auto nt = [&](const char* name, const Vec3& point) {
        result.push_back({name, false,
            {point, Vec3::Zero(), Vec3(1, 0, 0), Vec3(0, 1, 0)}});
    };
    nt("nt-face", Vec3(0.2, 0.3, 0.17));
    nt("nt-face-negative", Vec3(0.2, 0.3, -0.17));
    nt("nt-edge12", Vec3(0.4, -0.12, 0.17));
    nt("nt-edge23", Vec3(0.6, 0.6, 0.17));
    nt("nt-edge31", Vec3(-0.12, 0.4, 0.17));
    nt("nt-vertex1", Vec3(-0.12, -0.11, 0.17));
    nt("nt-vertex2", Vec3(1.12, -0.11, 0.17));
    nt("nt-vertex3", Vec3(-0.12, 1.11, 0.17));
    result.push_back({"nt-degenerate-edge", false,
        {Vec3(0.4, 0.12, 0.17), Vec3::Zero(), Vec3(1, 0, 0), Vec3(0.6, 0, 0)}});
    result.push_back({"nt-degenerate-point", false,
        {Vec3(0.1, 0.12, 0.17), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()}});
    const auto ss = [&](const char* name, const Vec3& a, const Vec3& b) {
        result.push_back({name, true,
            {Vec3(-0.7, 0, 0), Vec3(0.7, 0, 0), a, b}});
    };
    ss("ss-interior", Vec3(0, -0.6, 0.17), Vec3(0, 0.6, 0.17));
    ss("ss-edge-s0", Vec3(-0.9, -0.6, 0.17), Vec3(-0.9, 0.6, 0.17));
    ss("ss-edge-s1", Vec3(0.9, -0.6, 0.17), Vec3(0.9, 0.6, 0.17));
    ss("ss-edge-t0", Vec3(0, 0.12, 0.17), Vec3(0, 0.9, 0.17));
    ss("ss-edge-t1", Vec3(0, -0.9, 0.17), Vec3(0, -0.12, 0.17));
    ss("ss-corner-s0t0", Vec3(-0.9, 0.12, 0.17), Vec3(-0.9, 0.9, 0.17));
    ss("ss-corner-s0t1", Vec3(-0.9, -0.9, 0.17), Vec3(-0.9, -0.12, 0.17));
    ss("ss-corner-s1t0", Vec3(0.9, 0.12, 0.17), Vec3(0.9, 0.9, 0.17));
    ss("ss-corner-s1t1", Vec3(0.9, -0.9, 0.17), Vec3(0.9, -0.12, 0.17));
    ss("ss-parallel", Vec3(-0.4, 0.12, 0.17), Vec3(0.5, 0.12, 0.17));
    result.push_back({"ss-degenerate-first", true,
        {Vec3::Zero(), Vec3::Zero(), Vec3(0, -0.6, 0.17), Vec3(0, 0.6, 0.17)}});
    result.push_back({"ss-two-points", true,
        {Vec3::Zero(), Vec3::Zero(), Vec3(0.1, 0.12, 0.17), Vec3(0.1, 0.12, 0.17)}});
    return result;
}

template <class Value>
void expect_bits(const Value& actual, const Value& expected, const char* field) {
    for (Eigen::Index i = 0; i < actual.size(); ++i) {
        if (std::memcmp(actual.data() + i, expected.data() + i, sizeof(double)) == 0)
            continue;
        ADD_FAILURE() << field << '[' << i << "] actual=" << std::hexfloat
            << actual.data()[i] << " expected=" << expected.data()[i];
        return;
    }
}

} // namespace

namespace {

struct CapturedInteriorContact {
    int pair, role;
    std::array<Vec3, 4> positions;
    Vec3 gradient;
    std::array<double, 9> hessian_column_major;
};

// Original general-solver captures from example 10, frame 28, substep 1,
// sweep 1: vertex 3255 (first three pairs), then vertex 5976. Hex literals
// preserve the actual input/output bits from the GCC 11.4 native server build.
std::array<CapturedInteriorContact, 4> captured_interior_contacts() {
    return {{
        {84528, 0,
            {Vec3(0x1.655f258fa1c5ap-8, 0x1.3068ebe350b17p+0, 0x1.3a870669a4232p-5),
             Vec3(0x1.3290effb08550p-6, 0x1.32436eab88c7ap+0, 0x1.a990780162676p-5),
             Vec3(0x1.c8d1a27c2e932p-8, 0x1.0235af24e0796p+0, 0x1.4c09c655584b4p-5),
             Vec3(0x1.c7602ddf6028dp-7, 0x1.3a72880e78a39p+0, 0x1.9b356b3edeab8p-5)},
            Vec3(-0x1.281ae1a4359c5p-17, -0x1.be726a945cc02p-24, 0x1.238029af2488cp-17),
            {0x1.c8b49fc6f7ee4p-6, 0x1.770d09590badfp-12, -0x1.cc870a7f003b8p-6,
             0x1.770d09590bae0p-12, 0x1.31ec94013dcabp-18, -0x1.797432fc921a4p-12,
             -0x1.cc870a7f003b8p-6, -0x1.797432fc921a4p-12, 0x1.d01ec273432ffp-6}},
        {84557, 0,
            {Vec3(0x1.655f258fa1c5ap-8, 0x1.3068ebe350b17p+0, 0x1.3a870669a4232p-5),
             Vec3(0x1.8b20f2ae1942cp-6, 0x1.2e49d70d2126ep+0, 0x1.01ed4f294a996p-4),
             Vec3(0x1.c8d1a27c2e932p-8, 0x1.0235af24e0796p+0, 0x1.4c09c655584b4p-5),
             Vec3(0x1.c7602ddf6028dp-7, 0x1.3a72880e78a39p+0, 0x1.9b356b3edeab8p-5)},
            Vec3(-0x1.d4e5599ad48eep-8, -0x1.bcaa84446ed13p-17, 0x1.63ccafe95665fp-8),
            {0x1.5cdd0aa3379c1p+3, 0x1.0cce1fe06c3afp-5, -0x1.11e6068f8ebfap+3,
             0x1.0cce1fe06c3afp-5, 0x1.60f5a278d4bc4p-14, -0x1.a0a557788d12ap-6,
             -0x1.11e6068f8ebfbp+3, -0x1.a0a557788d12bp-6, 0x1.ad99fef02314fp+2}},
        {118505, 1,
            {Vec3(0x1.11c60df884ca5p-6, 0x1.2a77033211af7p+0, 0x1.a6d2bb3ec1511p-5),
             Vec3(0x1.655f258fa1c5ap-8, 0x1.3068ebe350b17p+0, 0x1.3a870669a4232p-5),
             Vec3(0x1.c8d1a27c2e932p-8, 0x1.0235af24e0796p+0, 0x1.4c09c655584b4p-5),
             Vec3(0x1.c7602ddf6028dp-7, 0x1.3a72880e78a39p+0, 0x1.9b356b3edeab8p-5)},
            Vec3(-0x1.8f74ee3180619p-11, -0x1.197e896138298p-18, 0x1.5047abe7ca1f2p-11),
            {0x1.c3df89fd605bbp-2, 0x1.39765d892eaf7p-8, -0x1.b3309995bf53bp-2,
             0x1.39765d892eaf7p-8, 0x1.49974dfd22938p-15, -0x1.1b305cfe27413p-8,
             -0x1.b3309995bf53bp-2, -0x1.1b305cfe27413p-8, 0x1.9c7b0f1db400cp-2}},
        {196351, 1,
            {Vec3(-0x1.2c18c8c66e40fp-2, 0x1.1ec2150716936p+0, 0x1.197b3257a866ap-1),
             Vec3(-0x1.21292bdd29bdfp-2, 0x1.1c13333da5f52p+0, 0x1.1195624306c5cp-1),
             Vec3(-0x1.2db25f8c021c8p-2, 0x1.0500699cb96eep+0, 0x1.186a03f1729ddp-1),
             Vec3(-0x1.1d3796fab0645p-2, 0x1.3d0408c2ec085p+0, 0x1.10134540eba28p-1)},
            Vec3(-0x1.30636fda43117p-9, 0x1.983d46ef76965p-15, -0x1.ae2803eac5adap-10),
            {0x1.5c1223d715301p+1, -0x1.151aaf7251500p-4, 0x1.c7339dca641d8p+0,
             -0x1.151aaf7251500p-4, 0x1.ae3e9ed54d285p-10, -0x1.6eff6204829c2p-5,
             0x1.c7339dca641d8p+0, -0x1.6eff6204829c3p-5, 0x1.27b82f89539b8p+0}}
    }};
}

} // namespace

TEST(GeneralSimdContactBitwise, CapturedSolidInteriorContactsAndPacketTails) {
    constexpr double d_hat = 0x1.11ab7f760b865p-9;
    const auto scenes = captured_interior_contacts();
    std::array<ipc_simd::MeshContactInput, ipc_simd::contact_tile_width> inputs;
    std::array<ipc_simd::MeshContactOutput, ipc_simd::contact_tile_width> outputs;
    std::array<unsigned char, ipc_simd::contact_tile_width> active;
    for (const auto& scene : scenes) {
        const auto& x = scene.positions;
        ASSERT_EQ(segment_segment_distance(x[0], x[1], x[2], x[3]).region,
            SegmentSegmentRegion::Interior) << "pair=" << scene.pair;
    }
    // Rotate the records so every capture is exercised as a singleton, at
    // every packet lane, and at a partially filled final packet.
    for (std::size_t offset = 0; offset < scenes.size(); ++offset) {
        for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
            const auto& scene = scenes[(lane + offset) % scenes.size()];
            inputs[lane].positions = scene.positions;
            inputs[lane].role = scene.role;
            inputs[lane].segment_segment = true;
        }
        for (std::size_t count = 1; count <= inputs.size(); ++count) {
            for (auto& output : outputs) {
                output.gradient.setConstant(1234567.0);
                output.hessian.setConstant(1234567.0);
            }
            active.fill(255);
            ipc_simd::general_mesh_contact_derivatives_tile(inputs.data(), count,
                d_hat, 1000.0, 0.0, 1.0 / 600.0, 0.1, outputs.data(), active.data());
            for (std::size_t lane = 0; lane < count; ++lane) {
                const auto& scene = scenes[(lane + offset) % scenes.size()];
                const auto& x = scene.positions;
                SCOPED_TRACE(::testing::Message() << "pair=" << scene.pair
                    << " count=" << count << " lane=" << lane);
                const auto reference = segment_segment_barrier_self_gradient_and_hessian(
                    x[0], x[1], x[2], x[3], d_hat, scene.role);
                const Mat33 captured_hessian = Eigen::Map<const Mat33>(scene.hessian_column_major.data());
#if defined(__AVX512F__) && defined(__FMA__) && defined(__GNUC__) && !defined(__clang__)
                // Native GCC is the diagnosed contraction configuration.
                // Compare both the current scalar kernel and the original
                // captured bits so a common-mode compiler change cannot pass.
                expect_bits(outputs[lane].gradient, reference.first, "scalar_gradient");
                expect_bits(outputs[lane].hessian, reference.second, "scalar_hessian");
                expect_bits(outputs[lane].gradient, scene.gradient, "captured_gradient");
                expect_bits(outputs[lane].hessian, captured_hessian, "captured_hessian");
#else
                // Other compilers/ISAs retain numerical agreement, without
                // imposing GCC's captured contraction order on their scalar code.
                EXPECT_LE((outputs[lane].gradient - reference.first).norm(),
                    2e-11 * std::max(1e-12, reference.first.norm()));
                EXPECT_LE((outputs[lane].hessian - reference.second).norm(),
                    2e-11 * std::max(1e-12, reference.second.norm()));
                EXPECT_LE((outputs[lane].gradient - scene.gradient).norm(),
                    2e-11 * scene.gradient.norm());
                EXPECT_LE((outputs[lane].hessian - captured_hessian).norm(),
                    2e-11 * captured_hessian.norm());
#endif
                EXPECT_EQ(active[lane], 1);
                EXPECT_TRUE(outputs[lane].friction_gradient.isZero());
                EXPECT_TRUE(outputs[lane].friction_hessian.isZero());
            }
            for (std::size_t lane = count; lane < outputs.size(); ++lane) {
                EXPECT_TRUE((outputs[lane].gradient.array() == 1234567.0).all());
                EXPECT_TRUE((outputs[lane].hessian.array() == 1234567.0).all());
                EXPECT_EQ(active[lane], 255);
            }
        }
    }
}

} // namespace general_simd_tests::contact_bitwise

namespace general_simd_tests::solver {

namespace {

struct RestoreOpenMP {
    int threads = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    ~RestoreOpenMP() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

struct GeneralScene {
    RefMesh mesh;
    DeformedState state;
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;
    SimParams params = SimParams::zeros();
    BroadPhase broad_phase;
};

void build_scene(GeneralScene& scene, bool cloth = true, bool solid = true,
    bool rigid = true, RigidBodyUpdateMode mode =
        RigidBodyUpdateMode::TranslationAndOrientation) {
    auto& mesh = scene.mesh;
    auto& state = scene.state;
    auto& params = scene.params;
    params.fps = 30.0;
    params.substeps = 1;
    params.mu = 2.0;
    params.lambda = 3.0;
    params.kB = 1e-4;
    params.kpin = 1000.0;
    params.solid_mu = 2.0;
    params.solid_lambda = 3.0;
    params.gravity = Vec3(0.0, -0.1, 0.0);
    params.node_box_min = 0.05;
    params.node_box_max = 0.05;
    params.theta_box_min = 0.02;
    params.theta_box_max = 0.02;
    params.node_box_update_count = 2;
    params.max_global_iters = 4;
    params.fixed_iters = true;
    params.damping = 0.1;
    params.use_parallel = true;
    params.use_ccd = true;

    params.use_ccd_guess = false;
    params.d_hat = 0.03;
    params.k_barrier = 1.0;
    params.friction_velocity_epsilon = 0.01;

    if (cloth) {
        std::vector<Vec2> material;
        build_square_mesh(mesh, state, material, 6, 6, 1.0, 1.0,
            Vec3(-0.5, 0.0, -0.5));
        mesh.build_lumped_mass(900.0, 0.001);
        state.velocities.assign(state.deformed_positions.size(),
            Vec3(-0.005, -0.003, 0.002));
        for (Vec3& x : state.deformed_positions) {
            x.x() *= 1.005;
            x.y() += 0.0005 * std::sin(4.0 * x.x())
                * std::cos(3.0 * x.z());
        }
        append_pin(scene.pins, 0, state.deformed_positions);
        append_pin(scene.pins, 6, state.deformed_positions);
    }
    if (solid) {
        // Four tets share an interior node, exercising several incident
        // elastic entries while keeping interior nodes out of contact/SDF.
        std::vector<Vec3> positions = {
            Vec3(-0.08, 0.006, -0.08), Vec3(0.08, 0.006, -0.08),
            Vec3(-0.08, 0.006, 0.08), Vec3(-0.08, 0.012, -0.08)};
        positions.push_back(0.25 * (positions[0] + positions[1]
            + positions[2] + positions[3]));
        const int first = static_cast<int>(state.deformed_positions.size());
        create_solid(positions,
            {4, 2, 1, 3, 0, 4, 1, 3, 0, 2, 4, 3, 0, 2, 1, 4},
            900.0, mesh, state);
        for (int node = first; node < static_cast<int>(state.velocities.size()); ++node)
            state.velocities[node] = Vec3(0.003, -0.002, -0.004);
        state.deformed_positions[first + 3] += Vec3(0.0002, 0.0001, -0.0001);
        state.deformed_positions[first + 4] += Vec3(-0.0001, 0.00005, 0.0001);
    }
    if (rigid) {
        const int rb = create_rigid_body(
            {Vec3(-0.4, 0.018, -0.4), Vec3(0.4, 0.018, -0.4),
             Vec3(0.0, 0.018, 0.4)},
            Vec3(0.01, -0.01, 0.02), Vec4(1.0, 0.0, 0.0, 0.0),
            Vec3(0.02, -0.01, 0.01), 1.0, mesh, state, mode);
        mesh.tris.insert(mesh.tris.end(), mesh.rb_nodes[rb].begin(),
            mesh.rb_nodes[rb].end());
        if (!cloth && !solid) {
            const int other = create_rigid_body(
                {Vec3(-0.2, 0.036, -0.2), Vec3(0.2, 0.036, -0.2),
                 Vec3(0.0, 0.036, 0.2)},
                Vec3(-0.01, -0.005, 0.003), Vec4(1.0, 0.0, 0.0, 0.0),
                Vec3(-0.01, 0.02, -0.01), 0.8, mesh, state, mode);
            mesh.tris.insert(mesh.tris.end(), mesh.rb_nodes[other].begin(),
                mesh.rb_nodes[other].end());
        }
    }
    mesh.node_to_rb.resize(state.deformed_positions.size(), -1);
    mesh.build_deformable_nodes();
    scene.adjacency = build_incident_triangle_map(mesh.tris);
}

// Call the public solver explicitly, then use the same state commit as the
// frame driver so comparisons include velocities and all rigid coordinates.
SolverResult solve_and_commit(GeneralScene& scene, bool simd) {
    scene.params.use_simd = simd;
    auto& state = scene.state;
    std::vector<Vec3> xhat;
    build_xhat(xhat, state.deformed_positions, state.velocities, scene.params.dt());
    for (std::size_t node = 0; node < xhat.size(); ++node)
        if (scene.mesh.node_to_rb[node] >= 0)
            xhat[node] = state.deformed_positions[node];
    auto positions = state.deformed_positions;
    auto centers = state.x_coms;
    auto orientations = state.orientations;
    std::vector<Vec3> omega(state.omega.size(), Vec3::Zero());
    const auto solve = global_gauss_seidel_solver_general;
    const SolverResult result = solve(scene.mesh, state, scene.adjacency,
        scene.pins, scene.params, positions, xhat, centers, orientations,
        omega, scene.broad_phase, "");
    // Nonconverged solves still expose their final iterates and residuals;
    // committing here lets the residual test compare those iterates as well.
    for (std::size_t node = 0; node < positions.size(); ++node) {
        if (scene.mesh.node_to_rb[node] >= 0) continue;
        state.velocities[node] = (positions[node] - state.deformed_positions[node])
            / scene.params.dt();
        state.deformed_positions[node] = positions[node];
    }
    // Both assembly backends return the authoritative accepted proxy positions.
    commit_rigid_body_solver_state(scene.mesh, state, centers, orientations,
        omega, scene.params.dt(), &positions);
    return result;
}

SolverResult solve_cloth_and_commit(GeneralScene& scene) {
    auto& state = scene.state;
    std::vector<Vec3> predicted;
    build_xhat(predicted, state.deformed_positions, state.velocities, scene.params.dt());
    auto positions = state.deformed_positions;
    const auto result = global_gauss_seidel_solver_cloth(scene.mesh,
        scene.adjacency, scene.pins, scene.params, positions, predicted,
        state.velocities, scene.broad_phase, "", &state.deformed_positions);
    update_velocity(state.velocities, positions, state.deformed_positions, scene.params.dt());
    state.deformed_positions = positions;
    return result;
}

void enable_simd_backend(GeneralScene& scene) {
    scene.params.use_simd = true;
}

void build_two_cloth_layers(GeneralScene& scene) {
    build_scene(scene, true, false, false);
    std::vector<Vec2> material;
    for (int j = 0; j <= 6; ++j)
        for (int i = 0; i <= 6; ++i)
            material.emplace_back(i / 6.0, j / 6.0);
    build_square_mesh(scene.mesh, scene.state, material, 6, 6, 1.0, 1.0,
        Vec3(-0.5, 0.018, -0.5));
    scene.state.velocities.resize(scene.state.deformed_positions.size(),
        Vec3(0.01, -0.002, -0.003));
    scene.mesh.build_lumped_mass(900.0, 0.001);
    scene.mesh.node_to_rb.assign(scene.state.deformed_positions.size(), -1);
    scene.mesh.build_deformable_nodes();
    scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
    enable_simd_backend(scene);
}

struct SparseContacts {
    std::vector<Vec3> positions{
        Vec3(0, 0, 0.01), Vec3(-1, -1, 0), Vec3(1, -1, 0), Vec3(0, 1, 0),
        Vec3(10, -1, 0), Vec3(12, -1, 0), Vec3(11, 1, 0),
        Vec3(-1, -1, -0.8), Vec3(1, -1, 1.2), Vec3(0, 1, 0.2),
        Vec3(1, 0, 0.01), Vec3(0.4, -1, 0), Vec3(0.4, 1, 0),
        Vec3(10, 10, 10), Vec3(11, 10, 10)};
    std::vector<Vec3> previous = positions;
    std::vector<unsigned char> solid = std::vector<unsigned char>(positions.size(), 0);
    std::vector<unsigned char> surface = std::vector<unsigned char>(positions.size(), 1);
    BroadPhase::Cache cache;
    SimParams params = SimParams::zeros();

    SparseContacts() {
        previous[0].x() -= 0.003;
        params.d_hat = 0.1;
        params.k_barrier = 2.0;
        cache.vertex_nt.resize(positions.size());
        cache.vertex_ss.resize(positions.size());
        // Include AABB rejections, plane-only rejections whose bounding boxes
        // overlap, and active contacts. Sparse survivors cross tile boundaries.
        for (int i = 0; i < 96; ++i) {
            const int first = i % 3 == 0 ? 4 : (i % 3 == 1 ? 7 : 1);
            cache.nt_pairs.push_back({0, {first, first + 1, first + 2}});
            cache.vertex_nt[0].push_back({static_cast<std::size_t>(i), 0});
        }
        for (int i = 0; i < 65; ++i) {
            const int first = i % 3 == 0 ? 11 : 13;
            cache.ss_pairs.push_back({{0, 10, first, first + 1}});
            cache.vertex_ss[0].push_back({static_cast<std::size_t>(i), 0});
        }
    }
};

template <class Vector>
void expect_vectors_near(const std::vector<Vector>& actual,
    const std::vector<Vector>& expected, double tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(actual[i].allFinite()) << "entry=" << i;
        ASSERT_TRUE(expected[i].allFinite()) << "entry=" << i;
        EXPECT_LE((actual[i] - expected[i]).cwiseAbs().maxCoeff(),
            tolerance * (1.0 + expected[i].cwiseAbs().maxCoeff())) << "entry=" << i;
    }
}

void expect_states_near(const DeformedState& actual, const DeformedState& expected) {
    expect_vectors_near(actual.deformed_positions, expected.deformed_positions, 2e-9);
    expect_vectors_near(actual.velocities, expected.velocities, 2e-7);
    expect_vectors_near(actual.x_coms, expected.x_coms, 2e-9);
    expect_vectors_near(actual.orientations, expected.orientations, 2e-9);
    expect_vectors_near(actual.v_coms, expected.v_coms, 2e-7);
    expect_vectors_near(actual.omega, expected.omega, 2e-7);
}

template <class Vector>
void expect_vectors_bitwise_equal(const std::vector<Vector>& actual,
    const std::vector<Vector>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(0, std::memcmp(actual[i].data(), expected[i].data(),
            actual[i].size() * sizeof(double))) << "entry=" << i;
}

void expect_states_bitwise_equal(const DeformedState& actual,
    const DeformedState& expected) {
    expect_vectors_bitwise_equal(actual.deformed_positions, expected.deformed_positions);
    expect_vectors_bitwise_equal(actual.velocities, expected.velocities);
    expect_vectors_bitwise_equal(actual.x_coms, expected.x_coms);
    expect_vectors_bitwise_equal(actual.orientations, expected.orientations);
    expect_vectors_bitwise_equal(actual.v_coms, expected.v_coms);
    expect_vectors_bitwise_equal(actual.omega, expected.omega);
}

void expect_active_coupling(const GeneralScene& scene) {
    std::vector<int> kind(scene.state.deformed_positions.size(), 0);
    for (int node : scene.mesh.tet_nodes) kind[node] = 1;
    for (const auto& nodes : scene.mesh.rb_nodes)
        for (int node : nodes) kind[node] = 2;
    bool active[3][3]{};
    const auto& x = scene.state.deformed_positions;
    for (const auto& pair : scene.broad_phase.cache().nt_pairs) {
        const int a = kind[pair.node], b = kind[pair.tri_v[0]];
        if (a == b) continue;
        const auto distance = node_triangle_distance(x[pair.node],
            x[pair.tri_v[0]], x[pair.tri_v[1]], x[pair.tri_v[2]]);
        if (distance.distance < scene.params.d_hat)
            active[std::min(a, b)][std::max(a, b)] = true;
    }
    EXPECT_TRUE(active[0][1]) << "cloth-solid contact must be active";
    EXPECT_TRUE(active[0][2]) << "cloth-rigid contact must be active";
    EXPECT_TRUE(active[1][2]) << "solid-rigid contact must be active";
}

} // namespace

TEST(GeneralSIMDSolver, CoupledClothSolidRigidMatchesScalarWithAndWithoutFriction) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (bool parallel : {false, true}) {
        for (int contact_mode : {0, 1, 2}) {
            SCOPED_TRACE(::testing::Message() << "parallel=" << parallel
                << " contact_mode=" << contact_mode);
            std::array<GeneralScene, 2> scenes;
            for (auto& scene : scenes) {
                build_scene(scene);
                scene.params.use_parallel = parallel;
                scene.params.friction_coefficient = contact_mode == 2 ? 0.2 : 0.0;
                if (contact_mode == 0) {
                    scene.params.d_hat = 0.0;
                    scene.params.k_barrier = 0.0;
                    scene.params.use_ccd = false;
                }
            }
            for (int frame = 0; frame < 2; ++frame) {
                for (int method = 0; method < 2; ++method) {
                    const auto result = solve_and_commit(scenes[method], method == 1);
                    ASSERT_TRUE(result.converged);
                    EXPECT_EQ(result.iterations, scenes[method].params.max_global_iters);
                }
                expect_states_near(scenes[1].state, scenes[0].state);
            }
            if (contact_mode != 0) expect_active_coupling(scenes[1]);
        }
    }
}

TEST(GeneralSIMDSolver, ScalarAndSimdPruneSeparatedMixedIncidenceAndRestoreWithoutCcd) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (bool simd : {false,true}) {
        GeneralScene scene;
        build_scene(scene);
        scene.params.node_box_min = scene.params.node_box_max = .002;
        scene.params.theta_box_min = scene.params.theta_box_max = .002;
        // Oblique faces produce overlapping AABBs for separated primitives.
        const Vec4 rotation = quaternion_from_angular_velocity(
            Vec4(1,0,0,0), Vec3(0,.61,0), 1.0);
        auto& state = scene.state;
        for (auto& x : state.deformed_positions) x = quaternion_rotate(rotation,x);
        for (auto& v : state.velocities) v = quaternion_rotate(rotation,v);
        for (auto& x : state.x_coms) x = quaternion_rotate(rotation,x);
        for (auto& v : state.v_coms) v = quaternion_rotate(rotation,v);
        for (auto& w : state.omega) w = quaternion_rotate(rotation,w);
        for (auto& q : state.orientations) q = quaternion_multiply(rotation,q);
        for (auto& pin : scene.pins) pin.target_position = quaternion_rotate(rotation,pin.target_position);
        for (bool ccd : {true,false,true}) {
            SCOPED_TRACE(::testing::Message() << "simd=" << simd << " ccd=" << ccd);
            scene.params.use_ccd = ccd;
            ASSERT_TRUE(solve_and_commit(scene,simd).converged);
            const auto& actual = scene.broad_phase.cache();
            BroadPhase unpruned;
            unpruned.initialize_surface_nodes(actual.node_boxes,scene.mesh,scene.params.d_hat,
                BroadPhase::InitializationMode::GeneralSolver);
            const auto& full = unpruned.cache();
            ASSERT_EQ(actual.nt_pairs.size(),full.nt_pairs.size());
            ASSERT_EQ(actual.ss_pairs.size(),full.ss_pairs.size());
            std::size_t removed = 0;
            for (int node : scene.mesh.deformable_nodes) {
                removed += full.vertex_nt[node].size()-actual.vertex_nt[node].size();
                removed += full.vertex_ss[node].size()-actual.vertex_ss[node].size();
                for (const Vec3& direction : {Vec3(.004,0,0),Vec3(0,-.004,0),Vec3(0,0,.004)}) {
                    auto expected = state.deformed_positions, result = expected;
                    const Vec3 target = expected[node]+direction;
                    const double a = per_vertex_safe_step(unpruned, expected, node, target, .9, ccd);
                    const double b = per_vertex_safe_step(scene.broad_phase, result, node, target, .9, ccd);
                    EXPECT_DOUBLE_EQ(a,b);
                    EXPECT_TRUE((expected[node].array()==result[node].array()).all());
                }
            }
            if (!ccd) EXPECT_EQ(removed,0u);
            else EXPECT_GT(removed,0u);
            expect_active_coupling(scene);
        }
    }
}

TEST(GeneralSIMDSolver, ColoredContactHelpersPreserveEveryStateAcrossTeamSizes) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (double friction : {0.0, 0.2}) {
        std::array<GeneralScene, 3> scenes;
        for (int run = 0; run < 3; ++run) {
            SCOPED_TRACE(::testing::Message() << "friction=" << friction << " run=" << run);
            auto& scene = scenes[run];
            build_scene(scene);
            scene.params.friction_coefficient = friction;
            // One worker disables cooperative contacts without changing color
            // order; larger teams assign helpers to the contact-heavy body.
            omp_set_num_threads(1 << run);
            for (int frame = 0; frame < 2; ++frame)
                ASSERT_TRUE(solve_and_commit(scene, true).converged);
            EXPECT_GT(scene.broad_phase.cache().nt_pairs.size(), 32u);
            expect_active_coupling(scene);
            if (run != 0) expect_states_bitwise_equal(scene.state, scenes[0].state);
        }
    }
}

TEST(GeneralSIMDSolver, RigidUpdateModesKeepDisabledCoordinatesFixed) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (const auto mode : {RigidBodyUpdateMode::TranslationAndOrientation,
            RigidBodyUpdateMode::TranslationOnly, RigidBodyUpdateMode::OrientationOnly,
            RigidBodyUpdateMode::None}) {
        SCOPED_TRACE(static_cast<int>(mode));
        std::array<GeneralScene, 2> scenes;
        for (auto& scene : scenes) {
            build_scene(scene, true, true, true, mode);
            scene.params.friction_coefficient = 0.2;
        }
        const auto initial = scenes[1].state;
        // Reuse the workspace across calls as well as the four sweeps inside
        // each solve. Fixed-body placement and optional quaternion curvature
        // must remain correct after the first invocation has warmed caches.
        for (int frame = 0; frame < 2; ++frame) {
            SCOPED_TRACE(frame);
            ASSERT_TRUE(solve_and_commit(scenes[0], false).converged);
            ASSERT_TRUE(solve_and_commit(scenes[1], true).converged);
            expect_states_near(scenes[1].state, scenes[0].state);
            if (!updates_rigid_translation(mode)) {
                expect_vectors_bitwise_equal(scenes[1].state.x_coms, initial.x_coms);
                EXPECT_TRUE(scenes[1].state.v_coms[0].isZero(0.0));
            }
            if (!updates_rigid_orientation(mode)) {
                expect_vectors_bitwise_equal(scenes[1].state.orientations, initial.orientations);
                EXPECT_TRUE(scenes[1].state.omega[0].isZero(0.0));
            }
        }
    }
}

TEST(GeneralSIMDSolver, FixedProxyPlacementTracksPrescribedPoseAcrossRepeatedCalls) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (const bool simd : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "simd=" << simd);
        for (bool mixed : {false, true}) {
            for (int nodes : {3, 131}) {
                for (int threads : {1, 4}) {
                    SCOPED_TRACE(::testing::Message() << "mixed=" << mixed
                        << " nodes=" << nodes << " threads=" << threads);
                    omp_set_num_threads(threads);
                    GeneralScene scene;
                    auto& mesh = scene.mesh;
                    auto& state = scene.state;
                    auto& params = scene.params;
                    if (mixed) {
                        state.deformed_positions = {
                            Vec3(-1, -1, 0), Vec3(3, -1, 0), Vec3(-1, 3, 0)};
                        state.velocities.assign(3, Vec3::Zero());
                        mesh.tris = {0, 1, 2};
                        mesh.initialize({Vec2(-1, -1), Vec2(3, -1), Vec2(-1, 3)},
                            state.deformed_positions);
                        mesh.mass.assign(3, 10.0);
                        mesh.node_to_rb.assign(3, -1);
                    }
                    std::vector<Vec3> points{Vec3(0, 0, 0.2)};
                    for (int local = 1; local < nodes; ++local)
                        points.emplace_back(5 + 0.1 * local,
                            3 + 0.02 * (local % 7), 1 + 0.003 * local);
                    const int rb = create_rigid_body(points, Vec3::Zero(),
                        Vec4(0.8, -0.2, 0.3, 0.4), Vec3::Zero(), nodes,
                        mesh, state, RigidBodyUpdateMode::None);
                    mesh.build_deformable_nodes();
                    scene.adjacency = build_incident_triangle_map(mesh.tris);
                    params.fps = 10.0;
                    params.substeps = 1;
                    params.max_global_iters = 8;
                    params.fixed_iters = true;
                    params.damping = 0.25;
                    params.d_hat = mixed ? 0.5 : 0.0;
                    params.k_barrier = mixed ? 100.0 : 0.0;
                    params.node_box_min = params.node_box_max = 1.0;
                    params.theta_box_min = params.theta_box_max = 3.141592653589793;
                    params.node_box_update_count = 3;
                    params.use_parallel = true;
                    params.use_simd = simd;

                    for (int call = 0; call < 3; ++call) {
                        SCOPED_TRACE(call);
                        // Reuse the exact mesh/cache allocations with a changed
                        // externally prescribed fixed pose. Stale particle and
                        // caller candidate coordinates must not survive the call.
                        if (call != 0) {
                            state.x_coms[rb] += Vec3(0.013, -0.009, 0.011);
                            state.orientations[rb] = quaternion_normalize(
                                Vec4(0.8, -0.2 + 0.01 * call, 0.3, 0.4));
                        }
                        auto positions = state.deformed_positions;
                        for (int node : mesh.rb_nodes[rb])
                            positions[node] = Vec3(-7, 8, -9);
                        const auto predicted = state.deformed_positions;
                        std::vector<Vec3> centers(1, Vec3(9, 8, 7));
                        std::vector<Vec3> omega(1, Vec3(3, 2, 1));
                        std::vector<Vec4> orientations(1, Vec4(1, 0, 0, 0));
                        const auto result = global_gauss_seidel_solver_general(
                            mesh, state, scene.adjacency, {}, params, positions,
                            predicted, centers, orientations, omega, scene.broad_phase);
                        ASSERT_TRUE(result.converged);
                        EXPECT_EQ(result.iterations, params.max_global_iters);
                        expect_vectors_bitwise_equal(centers, state.x_coms);
                        expect_vectors_bitwise_equal(orientations, state.orientations);
                        EXPECT_TRUE(omega[rb].isZero(0.0));
                        for (int local = 0; local < nodes; ++local) {
                            const Vec3 expected = world_space_position(
                                mesh.ref_positions[rb][local], state.x_coms[rb],
                                state.orientations[rb]);
                            EXPECT_EQ(0, std::memcmp(
                                positions[mesh.rb_nodes[rb][local]].data(),
                                expected.data(), 3 * sizeof(double))) << "local=" << local;
                        }
                        if (mixed && call == 0)
                            EXPECT_LT((positions[0].z() + positions[1].z()
                                + positions[2].z()) / 3.0, -1e-5);
                    }
                }
            }
        }
    }
}

TEST(GeneralSIMDSolver, PureClothSolidAndRigidScenesRetainTheirScalarResults) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (int type = 0; type < 3; ++type) {
        for (bool parallel : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << " parallel=" << parallel);
            std::array<GeneralScene, 2> scenes;
            for (auto& scene : scenes) {
                build_scene(scene, type == 0, type == 1, type == 2);
                scene.params.use_parallel = parallel;
                scene.params.d_hat = 0.0;
                scene.params.k_barrier = 0.0;
                scene.params.use_ccd = false;
            }
            ASSERT_TRUE(solve_and_commit(scenes[0], false).converged);
            ASSERT_TRUE(solve_and_commit(scenes[1], true).converged);
            expect_states_near(scenes[1].state, scenes[0].state);
        }
    }
}

TEST(GeneralSIMDSolver, PureRigidContactAndFrameDispatchMatchExplicitEntryPoints) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (double friction : {0.0, 0.2}) {
        SCOPED_TRACE(friction);
        std::array<GeneralScene, 3> scenes;
        for (auto& scene : scenes) {
            build_scene(scene, false, false, true);
            scene.params.friction_coefficient = friction;
        }
        auto& driven = scenes[2];
        driven.params.use_simd = true;
        for (int frame = 1; frame <= 2; ++frame) {
            ASSERT_TRUE(solve_and_commit(scenes[0], false).converged);
            ASSERT_TRUE(solve_and_commit(scenes[1], true).converged);
            ASSERT_TRUE(advance_one_frame_rb(driven.state, driven.mesh,
                driven.params, frame).converged);
            expect_states_near(scenes[1].state, scenes[0].state);
            expect_states_bitwise_equal(driven.state, scenes[1].state);
        }
        const auto& x = scenes[1].state.deformed_positions;
        const auto& lower = scenes[1].mesh.rb_nodes[0];
        const int upper = scenes[1].mesh.rb_nodes[1][0];
        const double distance = node_triangle_distance(x[upper], x[lower[0]],
            x[lower[1]], x[lower[2]]).distance;
        EXPECT_GT(distance, 0.0);
        EXPECT_LT(distance, scenes[1].params.d_hat);
    }
}

TEST(GeneralSIMDSolver, SolverCommitPreservesAcceptedProxyAndFrictionHistoryBits) {
    GeneralScene scene;
    build_scene(scene, false, false, true);
    scene.params.friction_coefficient = 0.2;
    const auto previous_centers = scene.state.x_coms;
    auto centers = previous_centers;
    auto orientations = scene.state.orientations;
    auto angular_velocities = scene.state.omega;
    auto accepted = scene.state.deformed_positions;
    for (std::size_t rb = 0; rb < centers.size(); ++rb) {
        centers[rb] += Vec3(0.01, -0.02, 0.03);
        angular_velocities[rb] = Vec3(0.2, -0.4, 0.6);
        for (std::size_t local = 0; local < scene.mesh.rb_nodes[rb].size(); ++local)
            accepted[scene.mesh.rb_nodes[rb][local]] = world_space_position(
                scene.mesh.ref_positions[rb][local], centers[rb], orientations[rb]);
    }
    const int perturbed = scene.mesh.rb_nodes[0][0];
    const Vec3 canonical = accepted[perturbed];
    accepted[perturbed].x() = std::nextafter(canonical.x(),
        -std::numeric_limits<double>::infinity());
    ASSERT_NE(0, std::memcmp(accepted[perturbed].data(), canonical.data(),
        3 * sizeof(double)));

    commit_rigid_body_solver_state(scene.mesh, scene.state, centers, orientations,
        angular_velocities, scene.params.dt(), &accepted);

    // These exact positions, rather than a pose reconstruction, are the
    // physical previous positions used by friction at the next substep.
    expect_vectors_bitwise_equal(scene.state.deformed_positions, accepted);
    expect_vectors_bitwise_equal(scene.state.x_coms, centers);
    expect_vectors_bitwise_equal(scene.state.orientations, orientations);
    expect_vectors_bitwise_equal(scene.state.omega, angular_velocities);
    for (std::size_t rb = 0; rb < centers.size(); ++rb) {
        const Vec3 expected_com_velocity =
            (centers[rb] - previous_centers[rb]) / scene.params.dt();
        EXPECT_EQ(0, std::memcmp(scene.state.v_coms[rb].data(),
            expected_com_velocity.data(), 3 * sizeof(double)));
        for (int node : scene.mesh.rb_nodes[rb]) {
            const Vec3 expected_velocity = expected_com_velocity
                + angular_velocities[rb].cross(accepted[node] - centers[rb]);
            EXPECT_EQ(0, std::memcmp(scene.state.velocities[node].data(),
                expected_velocity.data(), 3 * sizeof(double))) << "node=" << node;
        }
    }
}

TEST(GeneralSIMDSolver, RigidFrameStorageReuseIsBitwiseEqualToFreshFrames) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    BroadPhase reusable;
    for (double friction : {0.0, 0.2}) {
        std::array<GeneralScene, 2> scenes;
        for (auto& scene : scenes) {
            build_scene(scene, false, false, true);
            scene.params.use_simd = true;
            scene.params.friction_coefficient = friction;
        }
        for (int frame = 1; frame <= 4; ++frame) {
            if (frame == 3) {
                for (auto& scene : scenes)
                    std::swap(scene.mesh.tris[1], scene.mesh.tris[2]);
            }
            SCOPED_TRACE(::testing::Message() << "friction=" << friction << " frame=" << frame);
            ASSERT_TRUE(advance_one_frame_rb(scenes[0].state, scenes[0].mesh,
                scenes[0].params, frame).converged);
            ASSERT_TRUE(advance_one_frame_rb(scenes[1].state, scenes[1].mesh,
                scenes[1].params, reusable, frame).converged);
            expect_states_bitwise_equal(scenes[1].state, scenes[0].state);
        }
    }
}

TEST(GeneralSIMDSolver, FreshRigidFrameEntryPreservesCallbackCopyCount) {
    struct Callback {
        int* copies;
        int* calls;
        Callback(int& copied, int& invoked) : copies(&copied), calls(&invoked) {}
        Callback(const Callback& other) : copies(other.copies), calls(other.calls) { ++*copies; }
        void operator()(int, const std::vector<Vec3>&) const { ++*calls; }
    };
    GeneralScene scene;
    build_scene(scene, false, false, true);
    scene.params.use_simd = true;
    scene.params.d_hat = 0.0;
    scene.params.use_ccd = false;
    int copies = 0, calls = 0;
    SubstepCallback callback = Callback(copies, calls);
    copies = 0;
    ASSERT_TRUE(advance_one_frame_rb(scene.state, scene.mesh, scene.params,
        1, callback).converged);
    // Preserve the original one copy into the public by-value argument.
    // The compatibility wrapper must transfer it to the implementation.
    EXPECT_EQ(copies, 1);
    EXPECT_EQ(calls, scene.params.substeps);
}

TEST(GeneralSIMDSolver, SdfAndFrictionContributionsRemainActiveForAllBlockTypes) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<GeneralScene, 2> scenes;
    for (auto& scene : scenes) {
        build_scene(scene);
        scene.params.friction_coefficient = 0.2;
        scene.params.k_sdf = 2.0;
        scene.params.eps_sdf = 0.03;
        scene.params.sdf_planes.push_back({Vec3(0.0, -0.003, 0.0), Vec3::UnitY()});
    }
    for (int frame = 0; frame < 2; ++frame) {
        ASSERT_TRUE(solve_and_commit(scenes[0], false).converged);
        ASSERT_TRUE(solve_and_commit(scenes[1], true).converged);
        expect_states_near(scenes[1].state, scenes[0].state);
    }
}

TEST(GeneralSIMDSolver, ResidualComponentsMatchScalarAfterUnconvergedSweeps) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<GeneralScene, 2> scenes;
    std::array<SolverResult, 2> results;
    for (int method = 0; method < 2; ++method) {
        auto& scene = scenes[method];
        build_scene(scene);
        scene.params.fixed_iters = false;
        scene.params.tol_abs = 1e-30;
        scene.params.friction_coefficient = 0.2;
        results[method] = solve_and_commit(scene, method == 1);
        EXPECT_FALSE(results[method].converged);
        EXPECT_TRUE(results[method].has_residual);
        EXPECT_TRUE(results[method].has_residual_components);
        EXPECT_EQ(results[method].iterations, scene.params.max_global_iters);
    }
    expect_states_near(scenes[1].state, scenes[0].state);
    for (auto member : {&SolverResult::initial_cloth_residual,
            &SolverResult::initial_solid_residual, &SolverResult::initial_rigid_residual,
            &SolverResult::final_cloth_residual, &SolverResult::final_solid_residual,
            &SolverResult::final_rigid_residual, &SolverResult::final_residual}) {
        EXPECT_TRUE(std::isfinite(results[1].*member));
        EXPECT_NEAR(results[1].*member, results[0].*member,
            2e-8 * (1.0 + std::abs(results[0].*member)));
    }
    EXPECT_DOUBLE_EQ(results[1].final_residual, results[1].final_cloth_residual
        + results[1].final_solid_residual + results[1].final_rigid_residual);
}

TEST(GeneralSIMDSolver, GeneralFrameDispatchMatchesExplicitEntryPoint) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::array<GeneralScene, 2> scenes;
    for (auto& scene : scenes) {
        build_scene(scene);
        scene.params.use_simd = true;
        scene.params.friction_coefficient = 0.2;
    }
    for (int frame = 1; frame <= 2; ++frame) {
        ASSERT_TRUE(solve_and_commit(scenes[0], true).converged);
        auto& scene = scenes[1];
        ASSERT_TRUE(advance_one_frame_general(scene.state, scene.mesh, scene.adjacency,
            scene.pins, scene.params, scene.broad_phase, frame).converged);
        expect_states_bitwise_equal(scenes[1].state, scenes[0].state);
    }
}

TEST(GeneralSIMDSolver, PureClothEntryMatchesClothSolverAcrossContactAndFrictionModes) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (int execution : {0, 1, 2}) {
        omp_set_num_threads(execution == 1 ? 1 : 4);
        for (int contact_mode : {0, 1, 2}) {
            SCOPED_TRACE(::testing::Message() << "execution=" << execution
                << " contact_mode=" << contact_mode);
            std::array<GeneralScene, 2> scenes;
            for (auto& scene : scenes) {
                build_two_cloth_layers(scene);
                scene.params.use_parallel = execution != 0;
                scene.params.friction_coefficient = contact_mode == 2 ? 0.2 : 0.0;
                if (contact_mode == 0) {
                    scene.params.d_hat = 0.0;
                    scene.params.k_barrier = 0.0;
                    scene.params.use_ccd = false;
                }
            }
            for (int frame = 0; frame < 2; ++frame) {
                const auto basic = solve_cloth_and_commit(scenes[0]);
                const auto general = solve_and_commit(scenes[1], true);
                ASSERT_TRUE(basic.converged);
                ASSERT_TRUE(general.converged);
                EXPECT_EQ(general.iterations, basic.iterations);
                expect_states_bitwise_equal(scenes[1].state, scenes[0].state);
            }
            if (contact_mode != 0) {
                bool active = false;
                const auto& scene = scenes[1];
                const auto& x = scene.state.deformed_positions;
                for (const auto& pair : scene.broad_phase.cache().nt_pairs)
                    if ((pair.node < 49) != (pair.tri_v[0] < 49)
                        && node_triangle_distance(x[pair.node], x[pair.tri_v[0]],
                            x[pair.tri_v[1]], x[pair.tri_v[2]]).distance < scene.params.d_hat)
                        active = true;
                EXPECT_TRUE(active) << "the two cloth layers must have active contact";
            }
        }
    }
}

TEST(GeneralSIMDSolver, MixedSceneClothSweepMatchesClothWithoutCrossTypeCoupling) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (bool sdf_friction : {false, true}) {
        SCOPED_TRACE(sdf_friction);
        GeneralScene basic, general;
        build_scene(basic, true, false, false);
        build_scene(general);
        for (auto* scene : {&basic, &general}) {
            enable_simd_backend(*scene);
            scene->params.use_parallel = false;
            scene->params.max_global_iters = 1;
            scene->params.d_hat = 0.0;
            scene->params.k_barrier = 0.0;
            scene->params.use_ccd = false;
            if (sdf_friction) {
                scene->params.friction_coefficient = 0.2;
                scene->params.k_sdf = 2.0;
                scene->params.eps_sdf = 0.03;
                scene->params.sdf_planes.push_back({Vec3(0.0, -0.003, 0.0), Vec3::UnitY()});
            }
        }
        for (int node : general.mesh.tet_nodes)
            general.state.deformed_positions[node].x() += 5.0;
        general.state.x_coms[0].x() += 10.0;
        sync_rigid_body_particles(general.mesh, general.state);
        ASSERT_TRUE(solve_cloth_and_commit(basic).converged);
        ASSERT_TRUE(solve_and_commit(general, true).converged);
        const auto count = basic.state.deformed_positions.size();
        expect_vectors_near(std::vector<Vec3>(general.state.deformed_positions.begin(),
            general.state.deformed_positions.begin() + count), basic.state.deformed_positions, 1e-11);
        expect_vectors_near(std::vector<Vec3>(general.state.velocities.begin(),
            general.state.velocities.begin() + count), basic.state.velocities, 1e-9);
    }
}

TEST(GeneralSIMDSolver, FailedFirstColorRestoresParticlesAndRigidCoordinates) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (const bool simd : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "simd=" << simd);
        for (int threads : {1, 4}) {
            SCOPED_TRACE(threads);
            omp_set_num_threads(threads);
            GeneralScene scene;
            build_scene(scene);
            scene.params.use_simd = simd;
            scene.params.d_hat = 0.0;
            scene.params.k_barrier = 0.0;
            scene.params.use_ccd = false;
            // Separate components so the first color contains cloth, solid, and
            // rigid blocks. A cloth failure must undo other blocks in that color.
            for (int node : scene.mesh.tet_nodes)
                scene.state.deformed_positions[node].x() += 5.0;
            scene.state.x_coms[0].x() += 10.0;
            sync_rigid_body_particles(scene.mesh, scene.state);
            std::vector<Mat22> healthy_rest = scene.mesh.Dm_inverse;
            for (auto& dm : scene.mesh.Dm_inverse)
                dm.setConstant(std::numeric_limits<double>::quiet_NaN());

            auto positions = scene.state.deformed_positions;
            auto centers = scene.state.x_coms;
            auto orientations = scene.state.orientations;
            std::vector<Vec3> omega(scene.state.omega.size(), Vec3::Zero());
            const auto initial_positions = positions;
            const auto initial_centers = centers;
            const auto initial_orientations = orientations;
            const auto initial_omega = omega;
            std::vector<Vec3> predicted;
            build_xhat(predicted, positions, scene.state.velocities, scene.params.dt());
            for (std::size_t node = 0; node < predicted.size(); ++node)
                if (scene.mesh.node_to_rb[node] >= 0) predicted[node] = positions[node];
            const auto solve = [&] {
                return global_gauss_seidel_solver_general(scene.mesh,
                    scene.state, scene.adjacency, scene.pins, scene.params, positions,
                    predicted, centers, orientations, omega, scene.broad_phase);
            };
            EXPECT_THROW(solve(), std::runtime_error);
            expect_vectors_bitwise_equal(positions, initial_positions);
            expect_vectors_bitwise_equal(centers, initial_centers);
            expect_vectors_bitwise_equal(orientations, initial_orientations);
            expect_vectors_bitwise_equal(omega, initial_omega);
            // A fresh rest-data allocation also refreshes the general solver's
            // existing topology cache, isolating recovery from cache invalidation.
            scene.mesh.Dm_inverse.swap(healthy_rest);
            EXPECT_TRUE(solve().converged);
        }
    }
}

TEST(GeneralSIMDSolver, InPlaceRestMatrixChangesRefreshWarmedShapeGradients) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    GeneralScene warmed, fresh;
    build_scene(warmed);
    build_scene(fresh);
    const DeformedState initial = warmed.state;
    ASSERT_TRUE(solve_and_commit(warmed, true).converged);
    warmed.state = initial;
    const Mat22* storage = warmed.mesh.Dm_inverse.data();
    for (auto* scene : {&warmed, &fresh}) {
        scene->mesh.Dm_inverse[0](0, 0) *= 1.08;
        scene->mesh.Dm_inverse[0](0, 1) += 0.4;
    }
    ASSERT_EQ(warmed.mesh.Dm_inverse.data(), storage);
    // Run the warmed mesh first: solving the fresh mesh beforehand would
    // replace the static workspace and accidentally hide stale gradients.
    ASSERT_TRUE(solve_and_commit(warmed, true).converged);
    ASSERT_TRUE(solve_and_commit(fresh, true).converged);
    expect_states_near(warmed.state, fresh.state);
}

TEST(GeneralSIMDScheduler, HelperFailureRestoresOnlyItsColorAndJoinsNestedTeams) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (int execution : {0, 1, 2}) {
        SCOPED_TRACE(execution);
        omp_set_num_threads(execution == 0 ? 1 : 4);
        solver_detail::GeneralSimdBatches batches;
        batches.prepare({{0}, {1, 2}, {3}}, 0,
            [](int) { return std::size_t(256); });
        std::array<int, 4> values{10, 20, 30, 40}, snapshots{};
        std::atomic<int> active_ranges{0};
        std::atomic<bool> body_updated{false}, helpers_used{false};
        std::exception_ptr failure;
        const auto process = [&](int item, bool cooperative) {
            const int block = batches.batches[item].blocks[0];
            ++values[block];
            if (block == 1) body_updated.store(true, std::memory_order_release);
            if (block != 2) return;
            // Force another block of this color to commit before the error.
            while (!body_updated.load(std::memory_order_acquire))
                solver_detail::contact_spin_hint();
            helpers_used.store(cooperative, std::memory_order_relaxed);
            solver_detail::ordered_contact_tasks(128, cooperative,
                [&](int contact) {
                    struct ActiveRange {
                        std::atomic<int>& active;
                        explicit ActiveRange(std::atomic<int>& value) : active(value) { ++active; }
                        ~ActiveRange() { --active; }
                    } range(active_ranges);
                    if (contact == 33) throw std::runtime_error("injected contact failure");
                    return contact;
                }, [](int) {});
        };
        const auto run = [&] {
            try {
                solver_detail::run_general_simd_batches(batches, 3, process,
                    [&](int item) {
                        const int block = batches.batches[item].blocks[0];
                        snapshots[block] = values[block];
                    }, [&](int item) {
                        const int block = batches.batches[item].blocks[0];
                        values[block] = snapshots[block];
                    });
            } catch (...) { failure = std::current_exception(); }
        };
        if (execution == 2) {
            #pragma omp parallel num_threads(2)
            {
                #pragma omp single
                { run(); }
            }
        } else run();
        ASSERT_TRUE(failure != nullptr);
        EXPECT_THROW(std::rethrow_exception(failure), std::runtime_error);
        EXPECT_EQ(values, (std::array<int, 4>{11, 20, 30, 40}));
        EXPECT_EQ(active_ranges.load(), 0);
        EXPECT_EQ(helpers_used.load(), execution == 1);
        EXPECT_EQ(solver_detail::active_contact_task_group, nullptr);
    }
}

TEST(GeneralSIMDScheduler, CheapAndUniformWholeBatchesUseStaticRoundRobinOwnership) {
    RestoreOpenMP restore;
    omp_set_num_threads(4);
    omp_set_dynamic(0);
    constexpr int count = 61, sweeps = 3;
    for (int cost_mode : {0, 1, 2}) {
        SCOPED_TRACE(cost_mode);
        std::vector<int> group(count);
        std::iota(group.begin(), group.end(), 0);
        solver_detail::GeneralSimdBatches batches;
        // Treat every block as a rigid singleton so ownership measures the
        // scheduler rather than a thread-count-dependent particle packing.
        batches.prepare({group}, 0, [&](int block) {
            return cost_mode == 0 ? std::size_t(0)
                : (cost_mode == 1 ? std::size_t(block % 32) : std::size_t(256));
        });
        std::array<std::atomic<int>, count> visits{};
        for (auto& visit : visits) visit.store(0);
        std::atomic<bool> wrong_worker{false}, unexpected_helpers{false};
        solver_detail::run_general_simd_batches(batches, sweeps,
            [&](int item, bool cooperative) {
                if (cooperative) unexpected_helpers.store(true);
                const int block = batches.batches[item].blocks[0];
                if (omp_get_thread_num() != block % omp_get_num_threads())
                    wrong_worker.store(true);
                visits[block].fetch_add(1);
            }, [](int) {}, [](int) {});
        EXPECT_FALSE(wrong_worker.load());
        EXPECT_FALSE(unexpected_helpers.load());
        for (int block = 0; block < count; ++block)
            EXPECT_EQ(visits[block].load(), sweeps) << "block=" << block;
    }
}

TEST(GeneralSIMDScheduler, UnevenWholeBatchesReserveHeavyFirstWaveForEveryWorker) {
    RestoreOpenMP restore;
    omp_set_num_threads(4);
    omp_set_dynamic(0);
    // Leave enough work after the first wave to exercise the adaptive cursor
    // chunk cap of eight, while keeping every item's cost below 512.
    constexpr int count = 193;
    std::vector<int> group(count);
    std::iota(group.begin(), group.end(), 0);
    solver_detail::GeneralSimdBatches batches;
    // The unsorted input has cheap work first; no cost reaches the helper
    // threshold. Adaptive whole-batch scheduling must reverse this order.
    batches.prepare({group}, 0,
        [](int block) { return std::size_t(block + 1); });
    std::array<int, 4> first_block{-1, -1, -1, -1};
    std::array<std::atomic<int>, count> visits{};
    for (auto& visit : visits) visit.store(0);
    std::atomic<int> actual_team{0};
    std::atomic<bool> unexpected_helpers{false};
    solver_detail::run_general_simd_batches(batches, 1,
        [&](int item, bool cooperative) {
            if (cooperative) unexpected_helpers.store(true);
            const int worker = omp_get_thread_num();
            const int block = batches.batches[item].blocks[0];
            if (first_block[worker] < 0) first_block[worker] = block;
            actual_team.store(omp_get_num_threads());
            visits[block].fetch_add(1);
        }, [](int) {}, [](int) {});
    ASSERT_EQ(actual_team.load(), 4);
    EXPECT_FALSE(unexpected_helpers.load());
    for (int worker = 0; worker < 4; ++worker)
        EXPECT_EQ(first_block[worker], count - 1 - worker) << "worker=" << worker;
    for (int block = 0; block < count; ++block)
        EXPECT_EQ(visits[block].load(), 1) << "block=" << block;
}

TEST(GeneralSIMDScheduler, StaticAndDynamicQueuesVisitEveryBlockOncePerColorAndSweep) {
    RestoreOpenMP restore;
    for (int threads : {1, 2, 4}) {
        for (int configuration : {0, 1, 2, 3}) {
            const int dynamic = configuration % 2;
            const bool uneven = configuration >= 2;
            SCOPED_TRACE(::testing::Message() << "threads=" << threads
                << " dynamic=" << dynamic << " uneven=" << uneven);
            omp_set_num_threads(threads);
            omp_set_dynamic(dynamic);
            constexpr int count = 61, sweeps = 3;
            std::vector<std::vector<int>> groups(3);
            for (int block = 0; block < count; ++block)
                groups[block < 17 ? 0 : (block < 56 ? 1 : 2)].push_back(block);
            solver_detail::GeneralSimdBatches batches;
            batches.prepare(groups, uneven ? 0 : count, [&](int block) {
                return uneven ? std::size_t(4 * (block + 1)) : std::size_t(0);
            });
            std::array<std::atomic<int>, count> visits{};
            for (auto& visit : visits) visit.store(0);
            std::atomic<bool> barrier_violation{false}, unexpected_helpers{false};
            const auto process = [&](int item, bool cooperative) {
                if (cooperative) unexpected_helpers.store(true);
                const auto& batch = batches.batches[item];
                for (std::size_t i = 0; i < batch.size; ++i) {
                    const int block = batch.blocks[i];
                    const int prior = visits[block].load();
                    const int color = block < 17 ? 0 : (block < 56 ? 1 : 2);
                    if (color > 0) {
                        for (int previous : groups[color - 1])
                            if (visits[previous].load() != prior + 1)
                                barrier_violation.store(true);
                    } else if (prior > 0) {
                        for (int previous : groups.back())
                            if (visits[previous].load() != prior)
                                barrier_violation.store(true);
                    }
                    visits[block].fetch_add(1);
                }
            };
            solver_detail::run_general_simd_batches(batches, sweeps, process,
                [](int) {}, [](int) {});
            EXPECT_FALSE(barrier_violation.load());
            EXPECT_FALSE(unexpected_helpers.load());
            for (int block = 0; block < count; ++block)
                EXPECT_EQ(visits[block].load(), sweeps) << "block=" << block;
        }
    }
}

TEST(GeneralSIMDScheduler, StaticAndDynamicQueueFailuresRollBackTheirColorBeforeReturning) {
    RestoreOpenMP restore;
    omp_set_num_threads(4);
    for (int configuration : {0, 1, 2, 3}) {
        const int dynamic = configuration % 2;
        const bool uneven = configuration >= 2;
        SCOPED_TRACE(::testing::Message() << "dynamic=" << dynamic << " uneven=" << uneven);
        omp_set_dynamic(dynamic);
        std::vector<std::vector<int>> groups(3, std::vector<int>(32));
        for (int color = 0; color < 3; ++color)
            std::iota(groups[color].begin(), groups[color].end(), 32 * color);
        solver_detail::GeneralSimdBatches batches;
        batches.prepare(groups, uneven ? 0 : 96, [&](int block) {
            return uneven ? std::size_t(8 * (block % 32 + 1)) : std::size_t(0);
        });
        std::array<int, 96> values{}, saved{};
        const auto snapshot = [&](int item, bool restore_values) {
            const auto& batch = batches.batches[item];
            for (std::size_t i = 0; i < batch.size; ++i) {
                const int block = batch.blocks[i];
                if (restore_values) values[block] = saved[block];
                else saved[block] = values[block];
            }
        };
        EXPECT_THROW(solver_detail::run_general_simd_batches(batches, 3,
            [&](int item, bool) {
                const auto& batch = batches.batches[item];
                for (std::size_t i = 0; i < batch.size; ++i) {
                    const int block = batch.blocks[i];
                    ++values[block];
                    if (block == 45) throw std::runtime_error("failed unsplit batch");
                }
            }, [&](int item) { snapshot(item, false); },
            [&](int item) { snapshot(item, true); }), std::runtime_error);
        for (int block = 0; block < 96; ++block)
            EXPECT_EQ(values[block], block < 32 ? 1 : 0) << "block=" << block;
    }
}

TEST(GeneralSIMDContacts, SparseTilesPreserveResultsAndOnlyCertifyAabbRejections) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    for (double friction : {0.0, 0.2}) {
        SparseContacts input;
        input.params.friction_coefficient = friction;
        solver_detail::GeneralSimdVertexSystem reference;
        reference.gradient = Vec3(0.25, -0.5, 0.75);
        reference.hessian = 0.5 * Mat33::Identity();
        solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
            input.params, input.positions, &input.previous, input.solid,
            input.surface, false, reference);
        for (bool cooperative : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "friction=" << friction
                << " cooperative=" << cooperative);
            solver_detail::GeneralSimdVertexSystem actual;
            safe_step_detail::VertexAabbRejections rejections;
            std::atomic<int> leader_calls{0};
            const std::function<void()> prepare_local = [&] {
                ++leader_calls;
                actual.gradient = Vec3(0.25, -0.5, 0.75);
                actual.hessian = 0.5 * Mat33::Identity();
            };
            const auto run = [&] {
                solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
                    input.params, input.positions, &input.previous, input.solid,
                    input.surface, cooperative, actual, &rejections, &prepare_local);
            };
            if (cooperative) {
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp single
                    { run(); }
                }
            } else run();
            EXPECT_EQ(leader_calls.load(), 1);
            EXPECT_EQ(0, std::memcmp(actual.gradient.data(), reference.gradient.data(), 3 * sizeof(double)));
            EXPECT_EQ(0, std::memcmp(actual.hessian.data(), reference.hessian.data(), 9 * sizeof(double)));
            EXPECT_DOUBLE_EQ(rejections.distance, input.params.d_hat);
            ASSERT_EQ(rejections.clear.size(), 161u);
            for (int i = 0; i < 96; ++i)
                EXPECT_EQ(rejections.clear[i], i % 3 == 0 ? 1 : 0) << "NT=" << i;
            for (int i = 0; i < 65; ++i)
                EXPECT_EQ(rejections.clear[96 + i], i % 3 == 0 ? 0 : 1) << "SS=" << i;
        }
    }
}

TEST(GeneralSIMDContacts, LeaderFailureJoinsHelpersAndLeavesOutputUnaccumulated) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    SparseContacts input;
    for (bool cooperative : {false, true}) {
        solver_detail::GeneralSimdVertexSystem output;
        output.gradient = Vec3(1, 2, 3);
        output.hessian = Mat33::Identity();
        std::atomic<int> leader_calls{0};
        std::exception_ptr failure;
        const std::function<void()> fail = [&] {
            ++leader_calls;
            throw std::runtime_error("failed local preparation");
        };
        const auto run = [&] {
            try {
                solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
                    input.params, input.positions, &input.previous, input.solid,
                    input.surface, cooperative, output, nullptr, &fail);
            } catch (...) { failure = std::current_exception(); }
        };
        if (cooperative) {
            #pragma omp parallel num_threads(4)
            {
                #pragma omp single
                { run(); }
            }
        } else run();
        ASSERT_TRUE(failure != nullptr);
        EXPECT_THROW(std::rethrow_exception(failure), std::runtime_error);
        EXPECT_EQ(leader_calls.load(), 1);
        EXPECT_TRUE(output.gradient.isApprox(Vec3(1, 2, 3), 0.0));
        EXPECT_TRUE(output.hessian.isIdentity(0.0));
    }
}

TEST(GeneralSIMDContacts, FixedHelpersEvaluateContactsDuringLocalPreparationAndJoinOnFailure) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    SparseContacts input;
    solver_detail::GeneralSimdVertexSystem initial;
    initial.gradient = Vec3(0.25, -0.5, 0.75);
    initial.hessian = 0.5 * Mat33::Identity();
    auto reference = initial;
    solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
        input.params, input.positions, &input.previous, input.solid,
        input.surface, false, reference);

    for (bool fail_local : {false, true}) {
        SCOPED_TRACE(fail_local);
        solver_detail::GeneralSimdBatches batches;
        batches.prepare({{0}}, 0, [](int) { return std::size_t(256); });
        auto actual = initial;
        safe_step_detail::VertexAabbRejections rejections;
        std::atomic<int> leader_calls{0};
        bool helpers_completed_during_callback = false;
        bool unaccumulated_when_thrown = false;
        std::exception_ptr failure;
        const auto process = [&](int, bool cooperative) {
            auto* helpers = solver_detail::active_contact_task_group;
            if (!cooperative || helpers == nullptr || helpers->helpers == 0)
                throw std::runtime_error("test requires the fixed contact helper path");
            const std::function<void()> prepare_local = [&] {
                ++leader_calls;
                const auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(2);
                // dispatch publishes contacts before calling leader_work. The
                // leader cannot evaluate any ranges until this callback ends,
                // so completion here proves actual helper/leader overlap.
                while (helpers->remaining.load(std::memory_order_acquire) != 0) {
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("contact helpers did not finish during local preparation");
                    solver_detail::contact_spin_hint();
                }
                helpers_completed_during_callback = true;
                // The acquire above joins all helper writes, including masks.
                if (rejections.clear.size() != 161 || rejections.clear[0] != 1
                    || rejections.clear[1] != 0 || rejections.clear[2] != 0)
                    throw std::runtime_error("helper contact masks were not published before the join");
                if (fail_local) throw std::runtime_error("failed overlapped local preparation");
                actual.gradient = initial.gradient;
                actual.hessian = initial.hessian;
            };
            try {
                solver_detail::accumulate_general_simd_contacts(0, false, input.cache,
                    input.params, input.positions, &input.previous, input.solid,
                    input.surface, cooperative, actual, &rejections, &prepare_local);
            } catch (...) {
                unaccumulated_when_thrown =
                    std::memcmp(actual.gradient.data(), initial.gradient.data(), 3 * sizeof(double)) == 0
                    && std::memcmp(actual.hessian.data(), initial.hessian.data(), 9 * sizeof(double)) == 0;
                throw;
            }
        };
        try {
            solver_detail::run_general_simd_batches(batches, 1, process,
                [](int) {}, [&](int) { actual = initial; });
        } catch (...) { failure = std::current_exception(); }
        EXPECT_EQ(leader_calls.load(), 1);
        EXPECT_TRUE(helpers_completed_during_callback);
        if (fail_local) {
            ASSERT_TRUE(failure != nullptr);
            EXPECT_THROW(std::rethrow_exception(failure), std::runtime_error);
            EXPECT_TRUE(unaccumulated_when_thrown);
            EXPECT_TRUE(actual.gradient.isApprox(initial.gradient, 0.0));
            EXPECT_TRUE(actual.hessian.isApprox(initial.hessian, 0.0));
        } else {
            EXPECT_TRUE(failure == nullptr);
            EXPECT_EQ(0, std::memcmp(actual.gradient.data(), reference.gradient.data(), 3 * sizeof(double)));
            EXPECT_EQ(0, std::memcmp(actual.hessian.data(), reference.hessian.data(), 9 * sizeof(double)));
        }
        EXPECT_EQ(solver_detail::active_contact_task_group, nullptr);
    }
}

} // namespace general_simd_tests::solver

namespace general_simd_tests::materials {

namespace {

struct MaterialFixture {
    RefMesh mesh;
    std::vector<IncidentTriangles> incident;
    std::vector<ShapeGrads> shapes;
    std::vector<Vec3> positions, predicted;
    std::vector<Pin> pins;
    PinMap pin_map;
    std::vector<unsigned char> solid, surface;
    SimParams params;

    MaterialFixture() {
        positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
            Vec3(3, 0, 0), Vec3(4, 0, 0), Vec3(3, 1, 0), Vec3(3, 0, 1)};
        mesh.num_positions = positions.size();
        mesh.mass.assign(positions.size(), 1.0);
        mesh.node_to_rb.assign(positions.size(), -1);
        mesh.tris = {0, 1, 2, 0, 2, 3};
        mesh.compute_dm_inverse(positions);
        mesh.hinges.push_back(Hinge{{0, 2, 1, 3}, 0.0, 0.7});
        for (int role = 0; role < 4; ++role)
            mesh.hinge_adj[mesh.hinges[0].v[role]].emplace_back(0, role);
        incident.resize(positions.size());
        for (int triangle = 0; triangle < 2; ++triangle)
            for (int role = 0; role < 3; ++role)
                incident[mesh.tris[3 * triangle + role]].emplace_back(triangle, role);
        for (const auto& inverse : mesh.Dm_inverse)
            shapes.push_back(shape_function_gradients(inverse));
        mesh.tets = {4, 5, 6, 7};
        mesh.tet_nodes = mesh.surface_nodes = {4, 5, 6, 7};
        mesh.tet_rest_data = EFEMInitializeElasticMaterialState(positions, mesh.tets);
        mesh.tet_adj.resize(positions.size());
        for (int role = 0; role < 4; ++role) mesh.tet_adj[4 + role].emplace_back(0, role);
        solid.assign(positions.size(), 0); surface.assign(positions.size(), 0);
        for (int node : mesh.tet_nodes) solid[node] = surface[node] = 1;
        pin_map.assign(positions.size(), -1);
        positions[2].z() = 0.08;
        positions[5].x() += 0.13;
        positions[6].z() -= 0.04;
        predicted = positions;
        for (auto& x : predicted) x += Vec3(0.02, -0.01, 0.03);
        params.mu = 2.1; params.lambda = 3.2;
        params.solid_mu = 2.3; params.solid_lambda = 4.1;
        params.kB = 0.4; params.k_sdf = 0.0;
        params.friction_coefficient = 0.0;
        params.gravity = Vec3(0.0, -9.81, 0.0);
    }

    void evaluate(const int* nodes, std::size_t count,
        solver_detail::GeneralSimdVertexSystem* outputs,
        const solver_detail::GeneralSimdMaterials* materials) const {
        solver_detail::prepare_general_simd_batch(nodes, count, mesh, incident,
            shapes, pins, pin_map, params, positions, predicted, nullptr,
            solid, surface, outputs, materials);
    }
};

void expect_same(const solver_detail::GeneralSimdVertexSystem& a,
    const solver_detail::GeneralSimdVertexSystem& b) {
    EXPECT_EQ(std::memcmp(a.gradient.data(), b.gradient.data(), 3 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(a.hessian.data(), b.hessian.data(), 9 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(a.sdf_friction_gradient.data(), b.sdf_friction_gradient.data(), 3 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(a.sdf_friction_hessian.data(), b.sdf_friction_hessian.data(), 9 * sizeof(double)), 0);
}

} // namespace

TEST(GeneralSimdMaterials, RetainsIncidentOrderAndReusesUnchangedRecords) {
    MaterialFixture fixture;
    solver_detail::GeneralSimdMaterials materials;
    EXPECT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    EXPECT_EQ(materials.triangles.node_offsets, (std::vector<std::size_t>{0, 2, 3, 5, 6, 6, 6, 6, 6}));
    EXPECT_EQ(materials.tets.node_offsets, (std::vector<std::size_t>{0, 0, 0, 0, 0, 1, 2, 3, 4}));
    EXPECT_EQ(materials.hinges.node_offsets, (std::vector<std::size_t>{0, 1, 2, 3, 4, 4, 4, 4, 4}));
    EXPECT_EQ(materials.triangles.nodes[0], (std::array<int, 3>{0, 1, 2}));
    EXPECT_EQ(materials.triangles.nodes[1], (std::array<int, 3>{0, 2, 3}));
    const auto* retained = materials.triangles.dm_inverse.data();
    EXPECT_FALSE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    EXPECT_EQ(materials.triangles.dm_inverse.data(), retained);
    fixture.mesh.mass[0] = 2.0;
    fixture.positions[0].x() += 0.1;
    EXPECT_FALSE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    EXPECT_EQ(materials.triangles.dm_inverse.data(), retained);
}

TEST(GeneralSimdMaterials, InPlaceTopologyAdjacencyAndRestEditsInvalidate) {
    MaterialFixture fixture;
    solver_detail::GeneralSimdMaterials materials;
    ASSERT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    const auto rebuild = [&] {
        EXPECT_FALSE(materials.matches(fixture.mesh, fixture.incident, 8, true));
        EXPECT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
        EXPECT_TRUE(materials.matches(fixture.mesh, fixture.incident, 8, true));
    };
    fixture.mesh.Dm_inverse[0](0, 1) += 0.3; rebuild();
    EXPECT_TRUE(materials.triangles.shape_gradients[0].isApprox(
        shape_function_gradients(fixture.mesh.Dm_inverse[0])[0], 0.0));
    fixture.mesh.area[0] *= 1.2; rebuild();
    fixture.mesh.hinges[0].c_e *= 1.1; rebuild();
    fixture.mesh.hinges[0].bar_theta += 0.2; rebuild();
    std::swap(fixture.mesh.hinges[0].v[2], fixture.mesh.hinges[0].v[3]); rebuild();
    fixture.mesh.hinge_adj[0][0].second = 1; rebuild();
    std::swap(fixture.incident[0][0], fixture.incident[0][1]); rebuild();
    EXPECT_EQ(materials.triangles.nodes[0], (std::array<int, 3>{0, 2, 3}));
    std::swap(fixture.mesh.tris[1], fixture.mesh.tris[2]); rebuild();
    fixture.mesh.tet_rest_data[0].Dm_inverse(1, 2) += 0.1; rebuild();
    fixture.mesh.tet_rest_data[0].measure *= 1.2; rebuild();
    fixture.mesh.tet_rest_data[0].grad_N[0][2] += 0.4; rebuild();
    fixture.mesh.tet_adj[4][0].second = 1; rebuild();
    EXPECT_TRUE(materials.tets.shape_gradients[0].isApprox(fixture.mesh.tet_rest_data[0].grad_N[1], 0.0));
    std::swap(fixture.mesh.tets[1], fixture.mesh.tets[2]); rebuild();
    fixture.mesh.node_to_rb[0] = 0; rebuild();
    EXPECT_EQ(materials.triangles.node_offsets[1], 0u);
    EXPECT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, false));
    EXPECT_TRUE(materials.hinges.nodes.empty());
    EXPECT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    EXPECT_FALSE(materials.hinges.nodes.empty());
}

TEST(GeneralSimdMaterials, CachedAndLiveGatherAgreeForSingletonAndMixedBatches) {
    MaterialFixture fixture;
    solver_detail::GeneralSimdMaterials materials;
    ASSERT_TRUE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    const std::array<int, 8> order{2, 5, 0, 7, 3, 4, 1, 6};
    for (int frame = 0; frame < 2; ++frame) {
        for (std::size_t width = 1; width <= ipc_simd::tile_width; ++width)
            for (std::size_t first = 0; first < order.size(); first += width) {
                const auto count = std::min(width, order.size() - first);
                std::array<solver_detail::GeneralSimdVertexSystem, ipc_simd::tile_width> cached, live;
                fixture.evaluate(order.data() + first, count, cached.data(), &materials);
                fixture.evaluate(order.data() + first, count, live.data(), nullptr);
                for (std::size_t i = 0; i < count; ++i) expect_same(cached[i], live[i]);
            }
        // Static cache remains valid while live positions and point data move.
        fixture.positions[2].z() += 0.03;
        fixture.positions[7].x() -= 0.07;
        fixture.mesh.mass[0] *= 1.4;
        EXPECT_FALSE(materials.prepare(fixture.mesh, fixture.incident, 8, true));
    }
}

TEST(GeneralSimdMaterials, ClothMembraneUsesSharedSimdKernelAndIncidentOrder) {
    MaterialFixture fixture;
    fixture.mesh = RefMesh{};
    fixture.params = SimParams::zeros();
    fixture.params.fps = 30.0;
    fixture.params.substeps = 5;
    fixture.params.mu = 2.1;
    fixture.params.lambda = 3.2;
    fixture.positions = {Vec3(0.13, -0.07, 0.1)};
    // The center's incident span crosses a full tile and ends with a tail.
    constexpr int triangles = static_cast<int>(ipc_simd::tile_width) + 3;
    for (int i = 0; i < triangles; ++i) {
        const double angle = 2.0 * std::acos(-1.0) * i / triangles;
        fixture.positions.emplace_back(0.13 + 1.1 * std::cos(angle),
            -0.07 + 0.9 * std::sin(angle), 0.1);
        fixture.mesh.tris.insert(fixture.mesh.tris.end(),
            {0, 1 + i, 1 + (i + 1) % triangles});
    }
    const auto node_count = fixture.positions.size();
    fixture.mesh.num_positions = node_count;
    fixture.mesh.mass.assign(node_count, 0.0);
    fixture.mesh.node_to_rb.assign(node_count, -1);
    fixture.mesh.compute_dm_inverse(fixture.positions);
    fixture.incident.assign(node_count, {});
    fixture.shapes.clear();
    for (int triangle = 0; triangle < triangles; ++triangle) {
        for (int role = 0; role < 3; ++role)
            fixture.incident[fixture.mesh.tris[3 * triangle + role]].emplace_back(triangle, role);
        fixture.shapes.push_back(shape_function_gradients(fixture.mesh.Dm_inverse[triangle]));
    }
    for (std::size_t node = 0; node < node_count; ++node) {
        auto& x = fixture.positions[node];
        x.x() *= 1.037;
        x.y() += 0.019 * std::sin(0.7 * node);
        x.z() += 0.08 * std::cos(0.4 * node);
    }
    fixture.predicted = fixture.positions;
    fixture.pins.clear();
    fixture.pin_map.assign(node_count, -1);
    fixture.solid.assign(node_count, 0);
    fixture.surface.assign(node_count, 0);
    // Zero mass/gravity, no pins/hinges/SDF/contact: only membrane remains.
    // Compare to the cloth SIMD backend; scalar arithmetic may round differently.
    const auto basic_membrane = [&](int node) {
        solver_detail::GeneralSimdVertexSystem result;
        for (const auto& [triangle, role] : fixture.incident[node]) {
            std::array<Vec3, 3> positions;
            for (int corner = 0; corner < 3; ++corner)
                positions[corner] = fixture.positions[fixture.mesh.tris[3 * triangle + corner]];
            Vec3 gradient;
            Mat33 hessian;
            ipc_simd::corotated_derivatives_tile(positions.data(),
                &fixture.mesh.Dm_inverse[triangle], &fixture.mesh.area[triangle],
                &fixture.shapes[triangle][role], 1, fixture.params.mu,
                fixture.params.lambda, &gradient, &hessian);
            result.gradient += fixture.params.dt2() * gradient;
            result.hessian += fixture.params.dt2() * hessian;
        }
        return result;
    };
    std::vector<int> order;
    for (std::size_t node = 0; node < node_count; ++node)
        order.push_back(static_cast<int>(node_count - 1 - node));
    for (int reversed = 0; reversed < 2; ++reversed) {
        SCOPED_TRACE(reversed);
        solver_detail::GeneralSimdMaterials materials;
        ASSERT_TRUE(materials.prepare(fixture.mesh, fixture.incident, node_count, false));
        for (std::size_t width = 1; width <= ipc_simd::tile_width; ++width) {
            SCOPED_TRACE(width);
            for (std::size_t first = 0; first < order.size(); first += width) {
                const auto count = std::min(width, order.size() - first);
                std::array<solver_detail::GeneralSimdVertexSystem, ipc_simd::tile_width> cached, live;
                fixture.evaluate(order.data() + first, count, cached.data(), &materials);
                fixture.evaluate(order.data() + first, count, live.data(), nullptr);
                for (std::size_t i = 0; i < count; ++i) {
                    SCOPED_TRACE(order[first + i]);
                    const auto expected = basic_membrane(order[first + i]);
                    ASSERT_TRUE(expected.gradient.allFinite());
                    ASSERT_TRUE(expected.hessian.allFinite());
                    expect_same(cached[i], expected);
                    expect_same(live[i], expected);
                }
            }
        }
        for (auto& incident : fixture.incident)
            std::reverse(incident.begin(), incident.end());
    }
}

} // namespace general_simd_tests::materials

namespace general_simd_tests::bitwise {

namespace {

std::uint64_t bits(double value) {
    std::uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

template<class Actual, class Expected>
void expect_bitwise(const Actual& actual, const Expected& expected,
    const char* quantity) {
    int mismatches = 0;
    std::ostringstream first;
    for (int i = 0; i < expected.size(); ++i) {
        if (bits(actual.data()[i]) == bits(expected.data()[i])) continue;
        if (mismatches++ == 0) {
            first << quantity << '[' << i << "] actual=" << std::setprecision(17)
                << actual.data()[i] << " reference=" << expected.data()[i]
                << " actual_bits=0x" << std::hex << bits(actual.data()[i])
                << " reference_bits=0x" << bits(expected.data()[i]);
        }
    }
    EXPECT_EQ(mismatches, 0) << first.str();
}

template<class Actual, class Expected>
void expect_numerically_equal(const Actual& actual, const Expected& expected,
    const char* quantity) {
    const double scale = std::max(1e-12, expected.norm());
    for (int component = 0; component < expected.size(); ++component) {
        SCOPED_TRACE(::testing::Message() << quantity << '[' << component << ']');
        if (std::isnan(expected.data()[component]))
            EXPECT_TRUE(std::isnan(actual.data()[component]));
        else if (std::isinf(expected.data()[component]))
            EXPECT_EQ(actual.data()[component], expected.data()[component]);
        else
            EXPECT_NEAR(actual.data()[component], expected.data()[component], 2e-11 * scale);
    }
}

struct ClothFixture {
    RefMesh mesh;
    VertexTriangleMap adjacency;
    std::vector<IncidentTriangles> incident;
    std::vector<ShapeGrads> shapes;
    std::vector<Vec3> positions, predicted;
    std::vector<Pin> pins;
    PinMap pin_map;
    std::vector<unsigned char> solid, surface;
    SimParams params = SimParams::zeros();

    ClothFixture() {
        const std::array<Vec3, 4> local = {Vec3(.13, -.2, .03),
            Vec3(1.23, -.13, .06), Vec3(.43, .9, .15), Vec3(.53, -1.1, -.2)};
        for (int object = 0; object < 2; ++object)
            for (const auto& value : local)
                positions.push_back(value + Vec3(3.1 * object, .01 * object, 0));
        mesh.num_positions = positions.size();
        mesh.tris = {0, 1, 2, 1, 0, 3, 4, 5, 6, 5, 4, 7};
        mesh.compute_dm_inverse(positions);
        mesh.mass.resize(8);
        for (int node = 0; node < 8; ++node) mesh.mass[node] = .13 + .027 * node;
        mesh.node_to_rb.assign(8, -1);
        mesh.hinges = {Hinge{{0, 1, 2, 3}, -.2, .7}, Hinge{{4, 5, 6, 7}, .13, 1.1}};
        for (int hinge = 0; hinge < 2; ++hinge)
            for (int role = 0; role < 4; ++role)
                mesh.hinge_adj[mesh.hinges[hinge].v[role]].emplace_back(hinge, role);
        incident.resize(8);
        for (int triangle = 0; triangle < 4; ++triangle) {
            shapes.push_back(shape_function_gradients(mesh.Dm_inverse[triangle]));
            for (int role = 0; role < 3; ++role)
                incident[mesh.tris[3 * triangle + role]].emplace_back(triangle, role);
        }
        for (int node = 0; node < 8; ++node) adjacency[node] = incident[node];
        predicted = positions;
        for (int node = 0; node < 8; ++node) {
            positions[node] += Vec3(.013 * (node + 1), -.017 * (node % 3), .031 * (node % 2));
            predicted[node] += Vec3(.021, -.012, .037);
        }
        pins.push_back(Pin{0, Vec3(.15, -.22, .01)});
        pins.push_back(Pin{5, Vec3(4.2, -.15, .02)});
        pin_map.assign(8, -1); pin_map[0] = 0; pin_map[5] = 1;
        solid.assign(8, 0); surface.assign(8, 1);
        params.fps = 30; params.substeps = 20;
        params.gravity = Vec3(.03, -9.81, -.017);
        params.mu = 115000 / 2.5; params.lambda = 115000 * .25 / (1.25 * .5);
        params.kB = .009; params.kpin = 1e9;
        params.use_simd = false;
    }

    void disable_point() {
        mesh.mass.assign(8, 0.0);
        pins.clear(); pin_map.assign(8, -1);
    }

    void disable_membrane() {
        for (int node = 0; node < 8; ++node) {
            incident[node].clear(); adjacency[node].clear();
        }
    }

    std::pair<Vec3, Mat33> reference_with_shared_bending(int node) const {
        SimParams without_bending = params;
        without_bending.kB = 0.0;
        auto result = compute_local_gradient_and_hessian_no_barrier(
            node, mesh, adjacency, pins, without_bending, positions, predicted,
            &pin_map, &incident[node], &shapes, nullptr);
        if (!(params.kB > 0.0)) return result;
        const auto found = mesh.hinge_adj.find(node);
        if (found == mesh.hinge_adj.end()) return result;
        // Point/membrane keep their exact scalar-reference contract. Bending
        // must instead match the same SIMD entry used by the cloth SIMD backend.
        for (const auto& [hinge, role] : found->second) {
            const auto& value = mesh.hinges[hinge];
            std::array<Vec3, 4> corners;
            for (int corner = 0; corner < 4; ++corner)
                corners[corner] = positions[value.v[corner]];
            Vec3 gradient;
            Mat33 hessian;
            ipc_simd::bending_derivatives_tile(corners.data(), &role, &value.c_e,
                &value.bar_theta, 1, params.kB, &gradient, &hessian);
            result.first += params.dt2() * gradient;
            result.second += params.dt2() * hessian;
        }
        return result;
    }

    void compare(bool cached = true) {
        const int nodes[] = {0, 1, 2, 3, 4, 5, 6, 7};
        solver_detail::GeneralSimdMaterials materials;
        materials.prepare(mesh, incident, 8, params.kB > 0.0);
        for (std::size_t count = 1; count <= 8; ++count) {
            std::array<solver_detail::GeneralSimdVertexSystem, 8> output;
            solver_detail::prepare_general_simd_batch(nodes, count, mesh, incident,
                shapes, pins, pin_map, params, positions, predicted, nullptr,
                solid, surface, output.data(), cached ? &materials : nullptr);
            for (std::size_t entry = 0; entry < count; ++entry) {
                SCOPED_TRACE(::testing::Message() << "batch_width=" << count << " node=" << entry);
                const auto reference = compute_local_gradient_and_hessian_no_barrier(
                    nodes[entry], mesh, adjacency, pins, params, positions, predicted,
                    &pin_map, &incident[entry], &shapes, nullptr);
                if (params.kB > 0.0) {
                    const auto shared_reference = reference_with_shared_bending(nodes[entry]);
                    expect_bitwise(output[entry].gradient, shared_reference.first, "shared_bending_gradient");
                    expect_bitwise(output[entry].hessian, shared_reference.second, "shared_bending_hessian");
                    expect_numerically_equal(output[entry].gradient, reference.first, "scalar_gradient");
                    expect_numerically_equal(output[entry].hessian, reference.second, "scalar_hessian");
                } else {
                    expect_bitwise(output[entry].gradient, reference.first, "gradient");
                    expect_bitwise(output[entry].hessian, reference.second, "hessian");
                }
            }
        }
    }
};

} // namespace

TEST(GeneralSimdBitwise, PointMatchesOriginalAssembly) {
    ClothFixture fixture;
    fixture.disable_membrane(); fixture.params.kB = 0;
    fixture.compare();
}

TEST(GeneralSimdBitwise, MixedSolidClothPointRoundingPinsAndBatchTails) {
    constexpr std::size_t width = ipc_simd::tile_width;
    static_assert(width == 8);
    RefMesh mesh;
    mesh.num_positions = width;
    mesh.node_to_rb.assign(width, -1);
    mesh.tet_adj.resize(width);
    mesh.mass.resize(width);
    std::vector<Vec3> positions(width), predicted(width);
    std::vector<IncidentTriangles> incident(width);
    const std::vector<ShapeGrads> shapes;
    std::vector<Pin> pins;
    PinMap pin_map(width, -1);
    std::vector<unsigned char> solid(width), surface(width, 1);
    SimParams params = SimParams::zeros();
    params.fps = 30; params.substeps = 20;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.kpin = 100000.0;
    for (std::size_t node = 0; node < width; ++node) {
        // Node zero retains the exact frame-28/substep-1/sweep-1 inputs
        // captured for solid vertex 1134 on the GCC native server build.
        mesh.mass[node] = 0x1.d865de9b0ba24p-8 + .001 * node;
        positions[node] = Vec3(-0x1.9c2762463df2bp-2,
            0x1.35198b5adc71cp+0, -0x1.fd7319a438896p-2);
        predicted[node] = Vec3(-0x1.9bf14a748594fp-2,
            0x1.357131181ab89p+0, -0x1.fd71b59147924p-2);
        predicted[node] += Vec3(.00001 * node, -.00003 * node, .00007 * node);
        if (node % 4 >= 2) {
            pin_map[node] = static_cast<int>(pins.size());
            pins.push_back(Pin{static_cast<int>(node),
                positions[node] + Vec3(.00013, -.00027, .00019)});
        }
    }
    // Permutation makes it impossible to accidentally use the batch lane as
    // the global node ID when selecting the cloth/solid arithmetic.
    const std::array<int, width> nodes{7, 0, 5, 2, 3, 4, 1, 6};
    for (int pattern = 0; pattern < 3; ++pattern) {
        mesh.tet_nodes.clear();
        for (std::size_t node = 0; node < width; ++node) {
            solid[node] = pattern == 1 || (pattern == 2 && node % 2 == 0);
            if (solid[node]) mesh.tet_nodes.push_back(static_cast<int>(node));
        }
        solver_detail::GeneralSimdMaterials materials;
        materials.prepare(mesh, incident, width, false);
        for (bool cached : {false, true}) {
            for (std::size_t count = 1; count <= width; ++count) {
                std::array<solver_detail::GeneralSimdVertexSystem, width> output;
                for (auto& value : output) {
                    value.gradient.setConstant(1234567.0);
                    value.hessian.setConstant(1234567.0);
                }
                solver_detail::prepare_general_simd_batch(nodes.data(), count,
                    mesh, incident, shapes, pins, pin_map, params, positions,
                    predicted, nullptr, solid, surface, output.data(),
                    cached ? &materials : nullptr);
                for (std::size_t lane = 0; lane < count; ++lane) {
                    const int node = nodes[lane];
                    SCOPED_TRACE(::testing::Message() << "pattern=" << pattern
                        << " cached=" << cached << " count=" << count
                        << " lane=" << lane << " node=" << node);
                    if (solid[node]) {
                        const auto expected = compute_solid_local_gradient_and_pbgs_block_no_barrier(
                            node, mesh, pins, params, positions, predicted, &surface, &pin_map);
#if defined(__AVX512F__) && defined(__FMA__) && defined(__GNUC__) && !defined(__clang__)
                        // Exactness is a contract for the diagnosed GCC native
                        // build, not a claim of cross-compiler reproducibility.
                        expect_bitwise(output[lane].gradient, expected.first, "solid_gradient");
                        expect_bitwise(output[lane].hessian, expected.second, "solid_hessian");
                        if (node == 0)
                            EXPECT_EQ(bits(output[lane].gradient.y()), UINT64_C(0xbee3ce1a61699710));
#else
                        expect_numerically_equal(output[lane].gradient, expected.first, "solid_gradient");
                        expect_numerically_equal(output[lane].hessian, expected.second, "solid_hessian");
#endif
                    } else {
                        ipc_simd::PointInput input;
                        input.mass = mesh.mass[node]; input.position = positions[node];
                        input.predicted_position = predicted[node];
                        if (pin_map[node] >= 0) input.pin_target = pins[pin_map[node]].target_position;
                        Vec3 gradient; Mat33 hessian;
                        ipc_simd::point_derivatives_tile(&input, 1, params.gravity,
                            params.kpin, params.dt2(), &gradient, &hessian);
                        // The solid fix must not change cloth SIMD arithmetic.
                        expect_bitwise(output[lane].gradient, gradient, "cloth_gradient");
                        expect_bitwise(output[lane].hessian, hessian, "cloth_hessian");
                    }
                }
                for (std::size_t lane = count; lane < width; ++lane) {
                    EXPECT_TRUE((output[lane].gradient.array() == 1234567.0).all());
                    EXPECT_TRUE((output[lane].hessian.array() == 1234567.0).all());
                }
            }
        }
    }
}

TEST(GeneralSimdBitwise, BendingMatchesSharedKernelAssembly) {
    ClothFixture fixture;
    fixture.disable_point(); fixture.disable_membrane();
    fixture.compare();
}

namespace {

void check_membrane_kernel(const ClothFixture& fixture, std::size_t count) {
    std::array<Vec3, 24> positions;
    std::array<Mat22, 8> inverse;
    std::array<Vec2, 8> shape;
    std::array<double, 8> areas;
    std::array<Vec3, 8> gradient;
    std::array<Mat33, 8> hessian;
    for (int e = 0; e < 8; ++e) {
        const int triangle = e / 2;
        for (int role = 0; role < 3; ++role)
            positions[3 * e + role] = fixture.positions[fixture.mesh.tris[3 * triangle + role]];
        inverse[e] = fixture.mesh.Dm_inverse[triangle];
        shape[e] = fixture.shapes[triangle][e % 3];
        areas[e] = fixture.mesh.area[triangle];
    }
    ipc_simd::general_corotated_derivatives_tile(positions.data(), inverse.data(), areas.data(),
        shape.data(), count, fixture.params.mu, fixture.params.lambda, gradient.data(), hessian.data());
    for (std::size_t e = 0; e < count; ++e) {
        SCOPED_TRACE(::testing::Message() << "entry=" << e);
        Mat32 ds;
        ds.col(0) = positions[3 * e + 1] - positions[3 * e];
        ds.col(1) = positions[3 * e + 2] - positions[3 * e];
        const Mat32 F = ds * inverse[e];
        const auto cache = buildCorotatedCache(F);
        const auto P = PCorotated32(cache, F, fixture.params.mu, fixture.params.lambda);
        Mat66 derivative;
        dPdFCorotated32(cache, fixture.params.mu, fixture.params.lambda, derivative);
        ShapeGrads shapes;
        shapes[0] = shape[e];
        expect_bitwise(gradient[e], corotated_node_gradient(P, areas[e], shapes, 0), "gradient");
        expect_bitwise(hessian[e], corotated_node_hessian(derivative, areas[e], shapes, 0), "hessian");
    }
}

void check_bending_kernel(const ClothFixture& fixture, std::size_t count) {
    std::array<Vec3, 32> positions;
    std::array<int, 8> roles;
    std::array<double, 8> coefficients, angles;
    std::array<Vec3, 8> gradient;
    std::array<Mat33, 8> hessian;
    for (int e = 0; e < 8; ++e) {
        const auto& hinge = fixture.mesh.hinges[e / 4];
        for (int role = 0; role < 4; ++role)
            positions[4 * e + role] = fixture.positions[hinge.v[role]];
        roles[e] = e % 4; coefficients[e] = hinge.c_e; angles[e] = hinge.bar_theta;
    }
    ipc_simd::bending_derivatives_tile(positions.data(), roles.data(), coefficients.data(),
        angles.data(), count, fixture.params.kB, gradient.data(), hessian.data());
    for (std::size_t e = 0; e < count; ++e) {
        SCOPED_TRACE(::testing::Message() << "entry=" << e);
        HingeDef def;
        for (int role = 0; role < 4; ++role) def.x[role] = positions[4 * e + role];
        const auto reference = bending_node_gradient_hessian_psd(
            def, fixture.params.kB, coefficients[e], angles[e], roles[e]);
        expect_numerically_equal(gradient[e], reference.first, "scalar_gradient");
        expect_numerically_equal(hessian[e], reference.second, "scalar_hessian");
        Vec3 single_gradient;
        Mat33 single_hessian;
        ipc_simd::bending_derivatives_tile(positions.data() + 4 * e,
            roles.data() + e, coefficients.data() + e, angles.data() + e,
            1, fixture.params.kB, &single_gradient, &single_hessian);
        expect_bitwise(gradient[e], single_gradient, "shared_kernel_gradient");
        expect_bitwise(hessian[e], single_hessian, "shared_kernel_hessian");
    }
}

void deform_fixture(ClothFixture& fixture, int sample) {
    for (std::size_t node = 0; node < fixture.positions.size(); ++node) {
        const double phase = static_cast<double>(sample * (node + 1));
        fixture.positions[node] += Vec3(.071 * std::sin(phase),
            .043 * std::sin(.7 * phase), .083 * std::sin(1.3 * phase));
    }
}

} // namespace

TEST(GeneralSimdBitwise, SharedBendingKernelRetainsScalarNumericalAgreement) {
    for (int sample = 0; sample < 16; ++sample) {
        ClothFixture fixture;
        deform_fixture(fixture, sample);
        for (std::size_t count = 1; count <= 8; ++count) {
            SCOPED_TRACE(::testing::Message() << "sample=" << sample << " count=" << count);
            check_bending_kernel(fixture, count);
        }
    }
}

TEST(GeneralSimdBitwise, BendingDegeneracyAndSignedZerosMatchReference) {
    ClothFixture fixture;
    const std::array<double, 4> stiffnesses = {0.0, -0.0, .009,
        std::numeric_limits<double>::quiet_NaN()};
    for (double stiffness : stiffnesses) {
        for (int role = 0; role < 4; ++role) {
            HingeDef def;
            for (int corner = 0; corner < 4; ++corner)
                def.x[corner] = fixture.positions[corner];
            def.x[1] = def.x[0];
            const double coefficient = std::numeric_limits<double>::quiet_NaN();
            const double rest = std::numeric_limits<double>::quiet_NaN();
            Vec3 gradient;
            Mat33 hessian;
            ipc_simd::bending_derivatives_tile(def.x, &role, &coefficient,
                &rest, 1, stiffness, &gradient, &hessian);
            const auto reference = bending_node_gradient_hessian_psd(
                def, stiffness, coefficient, rest, role);
            expect_bitwise(gradient, reference.first, "degenerate_gradient");
            expect_bitwise(hessian, reference.second, "degenerate_hessian");
        }
    }
    for (double stiffness : {0.0, -0.0}) {
        fixture.params.kB = stiffness;
        check_bending_kernel(fixture, 8);
    }
    // Empty tiles are valid even with null pointers.
    ipc_simd::bending_derivatives_tile(nullptr, nullptr, nullptr,
        nullptr, 0, .009, nullptr, nullptr);
    ipc_simd::general_corotated_derivatives_tile(nullptr, nullptr, nullptr,
        nullptr, 0, 2.0, 3.0, nullptr, nullptr);
}

TEST(GeneralSimdBitwise, BendingNonfiniteGeometryRetainsReferenceClassification) {
    ClothFixture fixture;
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity()}) {
        HingeDef def;
        for (int corner = 0; corner < 4; ++corner)
            def.x[corner] = fixture.positions[corner];
        def.x[2][1] = invalid;
        const double coefficient = .7, rest = -.2;
        for (int role = 0; role < 4; ++role) {
            Vec3 gradient;
            Mat33 hessian;
            ipc_simd::bending_derivatives_tile(def.x, &role, &coefficient,
                &rest, 1, .009, &gradient, &hessian);
            const auto reference = bending_node_gradient_hessian_psd(
                def, .009, coefficient, rest, role);
            const auto compare = [](const auto& actual, const auto& expected) {
                for (int component = 0; component < expected.size(); ++component) {
                    if (std::isnan(expected.data()[component]))
                        EXPECT_TRUE(std::isnan(actual.data()[component]));
                    else
                        EXPECT_EQ(bits(actual.data()[component]), bits(expected.data()[component]));
                }
            };
            compare(gradient, reference.first);
            compare(hessian, reference.second);
        }
    }
}

} // namespace general_simd_tests::bitwise
