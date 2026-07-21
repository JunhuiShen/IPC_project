#include "initial_guess.h"

#include <gtest/gtest.h>
#include <vector>

namespace {

void expect_vec_near(const Vec3& actual, const Vec3& expected, double tol) {
    EXPECT_NEAR(actual.x(), expected.x(), tol);
    EXPECT_NEAR(actual.y(), expected.y(), tol);
    EXPECT_NEAR(actual.z(), expected.z(), tol);
}

RefMesh ref_mesh_with_masses(std::initializer_list<double> masses) {
    RefMesh ref_mesh;
    ref_mesh.mass.assign(masses.begin(), masses.end());
    ref_mesh.num_positions = ref_mesh.mass.size();
    return ref_mesh;
}

SimParams base_params() {
    SimParams params = SimParams::zeros();
    params.fps = 2.0;
    params.substeps = 1;
    params.gravity = Vec3::Zero();
    return params;
}

} // namespace

TEST(CCDInitialGuess, ReturnsTargetWhenNoCollisionCandidates) {
    RefMesh ref_mesh = ref_mesh_with_masses({1.0, 1.0});
    ref_mesh.tris.clear();

    std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
    };
    std::vector<Vec3> xhat = {
        Vec3(0.0, 1.0, 0.0),
        Vec3(1.0, 1.0, 0.0),
    };

    const std::vector<Vec3> guess = ccd_initial_guess(x, xhat, ref_mesh);

    ASSERT_EQ(guess.size(), xhat.size());
    for (int i = 0; i < static_cast<int>(xhat.size()); ++i) {
        expect_vec_near(guess[i], xhat[i], 1e-12);
    }
}

TEST(VerletInitialGuess, AddsGravityAndReturnsCollisionFreeTarget) {
    SimParams params = base_params();
    params.gravity = Vec3(0.0, -4.0, 2.0);

    RefMesh ref_mesh = ref_mesh_with_masses({1.0, 1.0});
    ref_mesh.tris.clear();

    const std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 2.0, 3.0),
    };
    const std::vector<Vec3> xhat = {
        Vec3(0.5, 1.0, -0.5),
        Vec3(1.5, 3.0, 2.5),
    };

    const std::vector<Vec3> guess = verlet_initial_guess(x, xhat, ref_mesh, params);
    const Vec3 dt2g = params.dt2() * params.gravity;

    ASSERT_EQ(guess.size(), xhat.size());
    for (int i = 0; i < static_cast<int>(xhat.size()); ++i) {
        expect_vec_near(guess[i], xhat[i] + dt2g, 1e-12);
    }
}

TEST(TranslationInitialGuess, MatchesMassWeightedInertiaAndGravityClosedForm) {
    SimParams params = base_params();
    params.gravity = Vec3(0.0, -4.0, 2.0);

    RefMesh ref_mesh = ref_mesh_with_masses({2.0, 1.0, 3.0});
    std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 2.0, 0.0),
        Vec3(-1.0, 0.5, 3.0),
    };
    std::vector<Vec3> xhat = {
        x[0] + Vec3(1.0, 0.0, 0.0),
        x[1] + Vec3(0.0, 2.0, 0.0),
        x[2] + Vec3(0.0, 0.0, -1.0),
    };

    const std::vector<Vec3> guess = translation_initial_guess(x, xhat, ref_mesh, {}, params);

    // Inertia gives (2, 2, -3) / 6 = (1/3, 1/3, -1/2).
    // With dt = 1/2, gravity contributes dt^2 * g = (0, -1, 1/2).
    const Vec3 expected_C(1.0 / 3.0, -2.0 / 3.0, 0.0);

    ASSERT_EQ(guess.size(), x.size());
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        expect_vec_near(guess[i], x[i] + expected_C, 1e-12);
    }
}

TEST(TranslationInitialGuess, IncludesPinSpringsInClosedFormTranslation) {
    SimParams params = base_params();
    params.kpin = 20.0;

    RefMesh ref_mesh = ref_mesh_with_masses({2.0, 3.0});
    std::vector<Vec3> x = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
    };
    std::vector<Vec3> xhat = {
        x[0] + Vec3(2.0, 0.0, 0.0),
        x[1] + Vec3(2.0, 0.0, 0.0),
    };
    std::vector<Pin> pins = {
        Pin{1, x[1] + Vec3(0.0, 4.0, 0.0)},
    };

    const std::vector<Vec3> guess = translation_initial_guess(x, xhat, ref_mesh, pins, params);

    const Vec3 expected_C(1.0, 2.0, 0.0);
    ASSERT_EQ(guess.size(), x.size());
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        expect_vec_near(guess[i], x[i] + expected_C, 1e-12);
    }
}

TEST(TranslationInitialGuess, AppliesOneNewtonCorrectionForPlaneSDF) {
    SimParams params = SimParams::zeros();
    params.fps = 1.0;
    params.substeps = 1;
    params.k_sdf = 10.0;
    params.eps_sdf = 0.0;
    params.gravity = Vec3::Zero();
    params.sdf_planes.push_back({Vec3::Zero(), Vec3::UnitY()});

    RefMesh ref_mesh = ref_mesh_with_masses({1.0, 1.0, 1.0});
    std::vector<Vec3> x = {
        Vec3(0.0, -0.1, 0.0),
        Vec3(1.0, -0.3, 0.0),
        Vec3(2.0,  0.2, 0.0),
    };
    std::vector<Vec3> xhat = x;

    const std::vector<Vec3> guess = translation_initial_guess(x, xhat, ref_mesh, {}, params);

    // Two active vertices have total penetration 0.4. With dt = 1 and k = 10:
    // C_y = 10 * 0.4 / (3 + 10 * 2) = 4/23.
    const Vec3 expected_C(0.0, 4.0 / 23.0, 0.0);

    ASSERT_EQ(guess.size(), x.size());
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        expect_vec_near(guess[i], x[i] + expected_C, 1e-12);
    }
}
