#include "volumetric_corotated_energy.h"
#include "third_party/tgsl/ImplicitQRSVD.h"

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
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
