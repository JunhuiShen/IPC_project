#include "general_simd_solid.h"
#include "general_simd_assembly.h"
#include "solid_ipc.h"
#include "volumetric_corotated_energy.h"

#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

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
