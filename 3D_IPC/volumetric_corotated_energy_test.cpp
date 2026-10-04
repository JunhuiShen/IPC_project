#include "volumetric_corotated_energy.h"
#include "batched_polar.h"
#include "third_party/tgsl/ImplicitQRSVD.h"

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr double kMu = 2.3;
constexpr double kLambda = 5.7;

void expect_double_bitwise_equal(const double actual, const double expected) {
    EXPECT_DOUBLE_EQ(actual, expected);
    EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(double)), 0);
}

void expect_vector_bitwise_equal(const Vec3& actual, const Vec3& expected) {
    for (int row = 0; row < 3; ++row)
        expect_double_bitwise_equal(actual[row], expected[row]);
}

void expect_matrix_bitwise_equal(const Mat33& actual, const Mat33& expected) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column)
            expect_double_bitwise_equal(actual(row, column), expected(row, column));
    }
}

template <typename Exception, typename Function>
void expect_exception_message(Function&& function, const char* expected) {
    bool caught = false;
    try {
        function();
    } catch (const Exception& error) {
        caught = true;
        EXPECT_STREQ(error.what(), expected);
    }
    EXPECT_TRUE(caught);
}

const std::vector<int>& single_tet_mesh() {
    static const std::vector<int> mesh = {0, 1, 2, 3};
    return mesh;
}

std::vector<Vec3> unit_tet_positions() {
    return {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    };
}

std::vector<Vec3> make_rest_positions() {
    return {
        Vec3(0.1, -0.2, 0.3),
        Vec3(1.3, -0.1, 0.4),
        Vec3(0.0, 1.1, 0.5),
        Vec3(-0.2, 0.2, 1.4),
    };
}

std::vector<Vec3> affine_transform(
    const std::vector<Vec3>& positions,
    const Mat33& A,
    const Vec3& translation) {
    std::vector<Vec3> transformed = positions;
    for (Vec3& position : transformed)
        position = A * position + translation;
    return transformed;
}

std::vector<Vec3> make_deformed_positions() {
    Mat33 A;
    A << 1.15, 0.12, -0.06,
        -0.08, 0.91, 0.10,
         0.04, -0.09, 1.08;
    return affine_transform(
        make_rest_positions(), A, Vec3(0.3, -0.4, 0.2));
}

struct ElementEvaluation {
    Mat33 F;
    CorotatedCache cache;
};

ElementEvaluation evaluate_element(
    const std::vector<Vec3>& x,
    const std::vector<TetRestData>& state) {
    ElementEvaluation evaluation;
    evaluation.F = ElementF(0, x, single_tet_mesh(), state);
    evaluation.cache.UpdateCache(evaluation.F);
    return evaluation;
}

double element_energy(
    const std::vector<Vec3>& x,
    const std::vector<TetRestData>& state) {
    const ElementEvaluation evaluation = evaluate_element(x, state);
    return EFEMElementInternalEnergy(
        evaluation.cache, evaluation.F, state[0], kMu, kLambda);
}

} // namespace

TEST(VolumetricCorotatedEnergy, ElementDsAndRestDataMatchTgsl) {
    const std::vector<Vec3> X = unit_tet_positions();
    const std::vector<TetRestData> state =
        EFEMInitializeElasticMaterialState(X, single_tet_mesh());
    ASSERT_EQ(state.size(), 1u);
    const TetRestData& element_state = state[0];

    EXPECT_TRUE(ElementDs(0, X, single_tet_mesh())
                    .isApprox(Mat33::Identity(), 0.0));
    EXPECT_DOUBLE_EQ(element_state.measure, 1.0 / 6.0);
    EXPECT_TRUE(element_state.Dm_inverse.isApprox(Mat33::Identity(), 0.0));
    EXPECT_TRUE(
        element_state.grad_N[0].isApprox(Vec3(-1.0, -1.0, -1.0), 0.0));
    EXPECT_TRUE(element_state.grad_N[1].isApprox(Vec3::UnitX(), 0.0));
    EXPECT_TRUE(element_state.grad_N[2].isApprox(Vec3::UnitY(), 0.0));
    EXPECT_TRUE(element_state.grad_N[3].isApprox(Vec3::UnitZ(), 0.0));
    EXPECT_TRUE(ElementF(0, X, single_tet_mesh(), state)
                    .isApprox(Mat33::Identity(), 0.0));
}

TEST(VolumetricCorotatedEnergy, RestStateHasZeroEnergyAndGradient) {
    const std::vector<Vec3> X = make_rest_positions();
    const std::vector<TetRestData> state =
        EFEMInitializeElasticMaterialState(X, single_tet_mesh());
    const ElementEvaluation evaluation = evaluate_element(X, state);

    EXPECT_NEAR(evaluation.cache.J_cache, 1.0, 1.0e-13);
    EXPECT_LT(
        (evaluation.cache.R_cache - Mat33::Identity()).norm(), 1.0e-13);
    EXPECT_NEAR(
        EFEMElementInternalEnergy(
            evaluation.cache, evaluation.F, state[0], kMu, kLambda),
        0.0, 1.0e-24);
    for (const Vec3& gradient : EFEMElementEnergyGradient(
             evaluation.cache, evaluation.F, state[0], kMu, kLambda)) {
        EXPECT_LT(gradient.norm(), 1.0e-12);
    }
}

TEST(VolumetricCorotatedEnergy, RigidRotationHasZeroEnergyAndGradient) {
    const std::vector<Vec3> X = make_rest_positions();
    const std::vector<TetRestData> state =
        EFEMInitializeElasticMaterialState(X, single_tet_mesh());
    const Mat33 rotation = Eigen::AngleAxisd(
        0.73, Vec3(1.0, -2.0, 0.5).normalized()).toRotationMatrix();
    const std::vector<Vec3> x =
        affine_transform(X, rotation, Vec3(-0.7, 1.2, 0.4));
    const ElementEvaluation evaluation = evaluate_element(x, state);

    EXPECT_NEAR(evaluation.cache.R_cache.determinant(), 1.0, 1.0e-13);
    EXPECT_LT((evaluation.cache.R_cache - rotation).norm(), 1.0e-12);
    EXPECT_NEAR(
        EFEMElementInternalEnergy(
            evaluation.cache, evaluation.F, state[0], kMu, kLambda),
        0.0, 1.0e-23);
    for (const Vec3& gradient : EFEMElementEnergyGradient(
             evaluation.cache, evaluation.F, state[0], kMu, kLambda)) {
        EXPECT_LT(gradient.norm(), 1.0e-11);
    }
}

TEST(VolumetricCorotatedEnergy, FirstPiolaMatchesEnergyDensityDifference) {
    Mat33 F;
    F << 1.15, 0.12, -0.06,
        -0.08, 0.91, 0.10,
         0.04, -0.09, 1.08;
    CorotatedCache cache;
    cache.UpdateCache(F);
    const Mat33 first_piola = cache.P(F, kMu, kLambda);

    constexpr double h = 1.0e-6;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            Mat33 plus = F;
            Mat33 minus = F;
            plus(row, column) += h;
            minus(row, column) -= h;

            CorotatedCache plus_cache;
            CorotatedCache minus_cache;
            plus_cache.UpdateCache(plus);
            minus_cache.UpdateCache(minus);
            const double finite_difference =
                (plus_cache.Psi(plus, kMu, kLambda)
                    - minus_cache.Psi(minus, kMu, kLambda))
                / (2.0 * h);
            EXPECT_NEAR(
                first_piola(row, column), finite_difference, 2.0e-8)
                << "row=" << row << " column=" << column;
        }
    }
}

TEST(VolumetricCorotatedEnergy, TetGradientMatchesEnergyDifference) {
    const std::vector<Vec3> X = make_rest_positions();
    const std::vector<TetRestData> state =
        EFEMInitializeElasticMaterialState(X, single_tet_mesh());
    const std::vector<Vec3> x = make_deformed_positions();
    const ElementEvaluation evaluation = evaluate_element(x, state);
    const std::array<Vec3, 4> gradient = EFEMElementEnergyGradient(
        evaluation.cache, evaluation.F, state[0], kMu, kLambda);

    constexpr double h = 1.0e-6;
    for (int node = 0; node < 4; ++node) {
        for (int component = 0; component < 3; ++component) {
            std::vector<Vec3> plus = x;
            std::vector<Vec3> minus = x;
            plus[static_cast<std::size_t>(node)][component] += h;
            minus[static_cast<std::size_t>(node)][component] -= h;
            const double finite_difference =
                (element_energy(plus, state) - element_energy(minus, state))
                / (2.0 * h);
            EXPECT_NEAR(
                gradient[static_cast<std::size_t>(node)][component],
                finite_difference, 2.0e-8)
                << "node=" << node << " component=" << component;
        }
    }

    EXPECT_LT(
        (gradient[0] + gradient[1] + gradient[2] + gradient[3]).norm(),
        1.0e-12);
}

TEST(VolumetricCorotatedEnergy, GradientSignIsOppositeTgslInternalForce) {
    const std::vector<Vec3> X = make_rest_positions();
    const std::vector<TetRestData> state =
        EFEMInitializeElasticMaterialState(X, single_tet_mesh());
    const ElementEvaluation evaluation =
        evaluate_element(make_deformed_positions(), state);
    const Mat33 first_piola =
        evaluation.cache.P(evaluation.F, kMu, kLambda);

    for (int node = 0; node < 4; ++node) {
        const Vec3 gradient = EFEMElementNodeEnergyGradient(
            evaluation.cache, evaluation.F, state[0],
            kMu, kLambda, node);
        const Vec3 tgsl_internal_force =
            -state[0].measure * first_piola * state[0].grad_N[node];
        EXPECT_TRUE(gradient.isApprox(-tgsl_internal_force, 1.0e-14));
    }
}

TEST(VolumetricCorotatedEnergy, PbgsNodeBlockMatchesTgslExpression) {
    const std::vector<Vec3> X = make_rest_positions();
    const std::vector<TetRestData> state =
        EFEMInitializeElasticMaterialState(X, single_tet_mesh());
    const ElementEvaluation evaluation =
        evaluate_element(make_deformed_positions(), state);

    for (int node = 0; node < 4; ++node) {
        const Vec3 u =
            evaluation.cache.JFinvT_cache * state[0].grad_N[node];
        const Mat33 expected = state[0].measure
            * (2.0 * kMu * state[0].grad_N[node].squaredNorm()
                   * Mat33::Identity()
               + kLambda * u * u.transpose());
        const Mat33 actual = PBGSElementNodeElasticityBlock(
            evaluation.cache, state[0], kMu, kLambda, node);

        EXPECT_TRUE(actual.isApprox(expected, 1.0e-14));
        EXPECT_TRUE(actual.isApprox(actual.transpose(), 1.0e-14));
        Eigen::SelfAdjointEigenSolver<Mat33> eigensolver(actual);
        ASSERT_EQ(eigensolver.info(), Eigen::Success);
        EXPECT_GE(eigensolver.eigenvalues().minCoeff(), -1.0e-12);
    }
}

TEST(VolumetricCorotatedEnergy, PolarFactorRemainsProperForInvertedF) {
    Mat33 F = Mat33::Identity();
    F(0, 0) = -0.8;
    F(1, 1) = 1.2;
    F(2, 2) = 0.9;

    CorotatedCache cache;
    cache.UpdateCache(F);
    EXPECT_TRUE((cache.R_cache.transpose() * cache.R_cache)
                    .isApprox(Mat33::Identity(), 1.0e-13));
    EXPECT_NEAR(cache.R_cache.determinant(), 1.0, 1.0e-13);
}

TEST(VolumetricCorotatedEnergy, CacheMatchesVendoredTgslPolarExactly) {
    Mat33 F;
    F << 1.15, 0.12, -0.06,
        -0.08, 0.91, 0.10,
         0.04, -0.09, 1.08;

    Mat33 tgsl_rotation;
    Mat33 tgsl_stretch;
    JIXIE::polarDecomposition(F, tgsl_rotation, tgsl_stretch);
    const Mat33 tgsl_Dinv = (tgsl_stretch.trace() * Mat33::Identity() - tgsl_stretch).inverse();

    CorotatedCache cache;
    cache.UpdateCache(F);

    expect_matrix_bitwise_equal(cache.R_cache, tgsl_rotation);
    expect_matrix_bitwise_equal(cache.Dinv_cache, tgsl_Dinv);
    expect_matrix_bitwise_equal(cache.JFinvT_cache, GradJ(F));
    expect_double_bitwise_equal(cache.J_cache, F.determinant());
}

TEST(VolumetricCorotatedEnergy, DefaultAndExplicitFullCacheModesMatchBitwise) {
    Mat33 F;
    F << 1.15, 0.12, -0.06, -0.08, 0.91, 0.10, 0.04, -0.09, 1.08;
    CorotatedCache default_full;
    CorotatedCache explicit_full;
    default_full.UpdateCache(F);
    explicit_full.UpdateCache(F, CorotatedCacheMode::Full);
    expect_matrix_bitwise_equal(default_full.JFinvT_cache, explicit_full.JFinvT_cache);
    expect_matrix_bitwise_equal(default_full.R_cache, explicit_full.R_cache);
    expect_matrix_bitwise_equal(default_full.Dinv_cache, explicit_full.Dinv_cache);
    expect_double_bitwise_equal(default_full.J_cache, explicit_full.J_cache);
}

TEST(VolumetricCorotatedEnergy, LeanCachePreservesUsedFieldsAndElementResultsBitwise) {
    Mat33 F;
    F << 1.15, 0.12, -0.06, -0.08, 0.91, 0.10, 0.04, -0.09, 1.08;
    Mat33 dinv_sentinel;
    dinv_sentinel << -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0;
    CorotatedCache full;
    CorotatedCache lean;
    full.UpdateCache(F);
    lean.Dinv_cache = dinv_sentinel;
    lean.UpdateCache(F, CorotatedCacheMode::Lean);
    expect_matrix_bitwise_equal(lean.JFinvT_cache, full.JFinvT_cache);
    expect_matrix_bitwise_equal(lean.R_cache, full.R_cache);
    expect_matrix_bitwise_equal(lean.Dinv_cache, dinv_sentinel);
    expect_double_bitwise_equal(lean.J_cache, full.J_cache);
    expect_double_bitwise_equal(lean.Psi(F, kMu, kLambda), full.Psi(F, kMu, kLambda));
    expect_matrix_bitwise_equal(lean.P(F, kMu, kLambda), full.P(F, kMu, kLambda));

    const std::vector<TetRestData> state = EFEMInitializeElasticMaterialState(unit_tet_positions(), single_tet_mesh());
    const Mat33 first_piola = lean.P(F, kMu, kLambda);
    for (int node = 0; node < 4; ++node) {
        const Vec3 full_gradient = EFEMElementNodeEnergyGradient(full, F, state[0], kMu, kLambda, node);
        const Vec3 cached_gradient = EFEMElementNodeEnergyGradient(lean, F, state[0], kMu, kLambda, node, &first_piola);
        const Mat33 full_block = PBGSElementNodeElasticityBlock(full, state[0], kMu, kLambda, node);
        const auto [lean_gradient, lean_block] = EFEMElementNodeGradientAndPBGSBlock(lean, F, state[0], kMu, kLambda, node);
        expect_vector_bitwise_equal(cached_gradient, full_gradient);
        expect_vector_bitwise_equal(lean_gradient, full_gradient);
        expect_matrix_bitwise_equal(lean_block, full_block);
    }
}

TEST(VolumetricCorotatedEnergy, BatchedPolarMatchesScalarAcrossDivergentRanksAndTails) {
    std::vector<Mat33> inputs{Mat33::Zero(), Mat33::Identity(), -Mat33::Identity()};
    Mat33 signed_zero = Mat33::Identity();
    signed_zero(0, 1) = -0.0;
    inputs.push_back(signed_zero);
    for (double scale : {1e-40, 1e-14, 1.0, 1e12, 1e24}) {
        for (int rank = 0; rank < 4; ++rank) {
            Mat33 m = Mat33::Zero();
            for (int i = 0; i < rank; ++i) m(i, i) = scale * (i == 2 ? -1.0 : 1.0);
            inputs.push_back(m);
        }
    }
    std::mt19937_64 random(0x4241544348504f4cULL);
    std::uniform_real_distribution<double> value(-2.0, 2.0);
    for (int i = 0; i < 8192; ++i) {
        Mat33 m;
        for (int j = 0; j < 9; ++j) m.data()[j] = value(random);
        if (i % 4 == 0) m.col(2) = m.col(0) + m.col(1);
        if (i % 7 == 0) m.row(1).setZero();
        if (i % 11 == 0) m *= 1e-13;
        inputs.push_back(m);
    }
    std::vector<Mat33> expected(inputs.size()), actual(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        CorotatedCache scalar;
        scalar.UpdateCache(inputs[i], CorotatedCacheMode::Lean);
        expected[i] = scalar.R_cache;
    }
    volumetric_detail::batched_signed_polar(nullptr, nullptr, 0);
    for (std::size_t stride : {1u, 3u, 4u, 7u, 8u, 13u, 32u, 65u}) {
        for (std::size_t i = 0; i < inputs.size(); i += stride)
            volumetric_detail::batched_signed_polar(inputs.data() + i, actual.data() + i,
                std::min(stride, inputs.size() - i));
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            SCOPED_TRACE(::testing::Message() << "stride=" << stride << " matrix=" << i);
            ASSERT_EQ(std::memcmp(actual[i].data(), expected[i].data(), 9 * sizeof(double)), 0);
        }
    }
    actual = inputs;
    volumetric_detail::batched_signed_polar(actual.data(), actual.data(), actual.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        ASSERT_EQ(std::memcmp(actual[i].data(), expected[i].data(), 9 * sizeof(double)), 0);
    inputs[3](0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(volumetric_detail::batched_signed_polar(inputs.data(), actual.data(), 8), std::invalid_argument);
}

TEST(VolumetricCorotatedEnergy, SoaPolarPreservesScalarBitsAndInactiveLanes) {
    std::mt19937_64 random(0x534f41504f4c4152ULL);
    std::uniform_real_distribution<double> value(-2.0, 2.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    volumetric_detail::signed_polar_soa_tile(nullptr, nullptr, 0);
    EXPECT_THROW(volumetric_detail::signed_polar_soa_tile(nullptr, nullptr, 9), std::invalid_argument);
    for (int sample = 0; sample < 64; ++sample) {
        std::array<Mat33, 8> matrices, expected;
        for (int lane = 0; lane < 8; ++lane) {
            auto& matrix = matrices[lane];
            for (int entry = 0; entry < 9; ++entry) matrix.data()[entry] = value(random);
            if (lane == 0) matrix.setZero();
            if (lane == 1) { matrix.setIdentity(); matrix(0, 1) = -0.0; }
            if (lane == 2) matrix.col(2) = matrix.col(0) + matrix.col(1);
            if (lane == 3) matrix.row(1).setZero();
            if (sample % 7 == 0 && lane == 5) matrix *= 1e-40;
            if (sample % 11 == 0 && lane == 6) matrix *= 1e24;
            CorotatedCache scalar;
            scalar.UpdateCache(matrix, CorotatedCacheMode::Lean);
            expected[lane] = scalar.R_cache;
        }
        for (std::size_t count = 0; count <= 8; ++count) {
            std::array<double, 72> inputs, output;
            inputs.fill(nan);
            output.fill(731.0);
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    for (std::size_t lane = 0; lane < count; ++lane)
                        inputs[8 * (3 * row + column) + lane] = matrices[lane](row, column);
            volumetric_detail::signed_polar_soa_tile(inputs.data(), output.data(), count);
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    for (std::size_t lane = 0; lane < 8; ++lane) {
                        SCOPED_TRACE(::testing::Message() << "sample=" << sample << " count=" << count << " lane=" << lane);
                        expect_double_bitwise_equal(output[8 * (3 * row + column) + lane],
                            lane < count ? expected[lane](row, column) : 731.0);
                    }
            // In-place tiles must gather before overwriting any active entry.
            auto in_place = inputs;
            volumetric_detail::signed_polar_soa_tile(in_place.data(), in_place.data(), count);
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    for (std::size_t lane = 0; lane < count; ++lane)
                        expect_double_bitwise_equal(in_place[8 * (3 * row + column) + lane], expected[lane](row, column));
        }
    }
    std::array<double, 72> invalid{};
    std::array<double, 72> output{};
    invalid[3] = nan;
    EXPECT_THROW(volumetric_detail::signed_polar_soa_tile(invalid.data(), output.data(), 8), std::invalid_argument);
    invalid[3] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(volumetric_detail::signed_polar_soa_tile(invalid.data(), output.data(), 8), std::invalid_argument);
}

TEST(VolumetricCorotatedEnergy, BatchedLeanCachePreservesAllFieldsAndLeavesInverseUntouched) {
    constexpr std::size_t size = 33;
    std::array<Mat33, size> inputs;
    std::array<CorotatedCache, size> expected, actual;
    for (std::size_t i = 0; i < size; ++i) {
        inputs[i] << 1.0 + 0.07 * i, -0.13, 0.07,
            0.03, 0.89 - 0.04 * i, -0.11, -0.06, 0.17, 1.13;
        if (i % 5 == 0) inputs[i].col(1).setZero();
        if (i % 7 == 0) inputs[i].setZero();
        expected[i].Dinv_cache.setConstant(-731.0);
        expected[i].UpdateCache(inputs[i], CorotatedCacheMode::Lean);
    }
    volumetric_detail::update_corotated_cache_batch(nullptr, nullptr, 0);
    for (std::size_t count = 0; count <= size; ++count) {
        for (auto& cache : actual) {
            cache.R_cache.setConstant(123.0);
            cache.JFinvT_cache.setConstant(456.0);
            cache.J_cache = 789.0;
            cache.Dinv_cache.setConstant(-731.0);
        }
        volumetric_detail::update_corotated_cache_batch(inputs.data(), actual.data(), count);
        for (std::size_t i = 0; i < size; ++i) {
            SCOPED_TRACE(::testing::Message() << "count=" << count << " lane=" << i);
            expect_matrix_bitwise_equal(actual[i].Dinv_cache, expected[i].Dinv_cache);
            if (i < count) {
                expect_matrix_bitwise_equal(actual[i].R_cache, expected[i].R_cache);
                expect_matrix_bitwise_equal(actual[i].JFinvT_cache, expected[i].JFinvT_cache);
                expect_double_bitwise_equal(actual[i].J_cache, expected[i].J_cache);
            } else {
                expect_matrix_bitwise_equal(actual[i].R_cache, Mat33::Constant(123.0));
                expect_matrix_bitwise_equal(actual[i].JFinvT_cache, Mat33::Constant(456.0));
                expect_double_bitwise_equal(actual[i].J_cache, 789.0);
            }
        }
    }
    inputs[3](0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(volumetric_detail::update_corotated_cache_batch(inputs.data(), actual.data(), 8), std::invalid_argument);
}

TEST(VolumetricCorotatedEnergy, RejectsNonPositiveRestMeasure) {
    std::vector<Vec3> inverted = unit_tet_positions();
    std::swap(inverted[1], inverted[2]);
    EXPECT_THROW(
        EFEMInitializeElasticMaterialState(inverted, single_tet_mesh()),
        std::invalid_argument);

    std::vector<Vec3> degenerate = unit_tet_positions();
    degenerate[3] = Vec3(0.25, 0.25, 0.0);
    EXPECT_THROW(
        EFEMInitializeElasticMaterialState(degenerate, single_tet_mesh()),
        std::invalid_argument);
}

TEST(VolumetricCorotatedEnergy, ElementAccessRejectsInvalidInput) {
    const std::vector<Vec3> X = unit_tet_positions();
    const std::vector<TetRestData> state =
        EFEMInitializeElasticMaterialState(X, single_tet_mesh());

    EXPECT_THROW(ElementDs(1, X, single_tet_mesh()), std::out_of_range);
    EXPECT_THROW(
        ElementF(1, X, single_tet_mesh(), state), std::out_of_range);
    EXPECT_THROW(
        ElementF(0, X, single_tet_mesh(), {}), std::invalid_argument);
    EXPECT_THROW(
        ElementDs(0, X, {0, 1, 2}), std::invalid_argument);
}

TEST(VolumetricCorotatedEnergy, ElementFPreservesValidationErrorsAndPrecedence) {
    const std::vector<Vec3> X = unit_tet_positions();
    const std::vector<TetRestData> state = EFEMInitializeElasticMaterialState(X, single_tet_mesh());
    expect_exception_message<std::invalid_argument>([&] { (void)ElementF(0, X, {0, 1, 2}, {}); }, "tet connectivity must contain four indices per element");
    expect_exception_message<std::out_of_range>([&] { (void)ElementF(1, X, single_tet_mesh(), {}); }, "tet element index is out of range");
    expect_exception_message<std::out_of_range>([&] { (void)ElementF(0, X, {0, 1, 2, 4}, state); }, "tet node index is out of range");
    expect_exception_message<std::invalid_argument>([&] { (void)ElementF(0, X, single_tet_mesh(), {}); }, "tet rest state must contain one record per element");
}
