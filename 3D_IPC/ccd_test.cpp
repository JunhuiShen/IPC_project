#include "ccd.h"
#include "ipc_args.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "simulation.h"

#include <omp.h>
#include "safe_step.h"
#include "segment_segment_distance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <gtest/gtest.h>

// Shared TOI tolerance for the basic CCD assertions.
constexpr double kTol = 1.0e-6;

namespace {
const Vec3 ZERO_DX(0.0, 0.0, 0.0);
}  // namespace

TEST(CCDLinearRobustness, ThinTriangleInteriorHit) {
    for (double height : {2e-7, 1e-9, 1e-13}) {
        SCOPED_TRACE(height);
        const auto r = node_triangle_only_one_node_moves(
            Vec3(0.75, height * 0.25, 1), Vec3(0, 0, -2),
            Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX,
            Vec3(1, height, 0), ZERO_DX, 1e-12, false);
        ASSERT_TRUE(r.collision);
        EXPECT_NEAR(r.t, 0.5, height < 1e-10 ? 2e-5 : 1e-12);
    }
}

TEST(CCDLinearRobustness, LargeTranslationDoesNotRoundAwayMotion) {
    const Vec3 shift = Vec3::Constant(std::ldexp(1.0, 50));
    const auto r = node_triangle_only_one_node_moves(
        shift + Vec3(0.25, 0.25, 0.25), Vec3(0, 0, -0.75),
        shift, ZERO_DX, shift + Vec3(1, 0, 0), ZERO_DX,
        shift + Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0 / 3.0, 1e-12);
}

TEST(CCDLinearRobustness, ScaleDoesNotOverflowOrUnderflow) {
    for (int exponent : {-500, -250, 0, 250, 500}) {
        SCOPED_TRACE(exponent);
        const double s = std::ldexp(1.0, exponent);
        const auto nt = node_triangle_only_one_node_moves(
            s * Vec3(0.25, 0.25, 1), s * Vec3(0, 0, -2),
            Vec3::Zero(), ZERO_DX, s * Vec3(1, 0, 0), ZERO_DX,
            s * Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
        const auto ss = segment_segment_only_one_node_moves(
            s * Vec3(0, 0, 1), s * Vec3(0, 0, -2), s * Vec3(1, 0, 0),
            s * Vec3(0.5, -1, 0), s * Vec3(0.5, 1, 0), 1e-12, false);
        const auto translation = segment_segment_same_displacement_linear_ccd(
            s * Vec3(0, 0, 1), s * Vec3(0, 0, -2), s * Vec3(1, 0, 1),
            s * Vec3(0, 0, -2), s * Vec3(0.5, -1, 0), s * Vec3(0.5, 1, 0));
        for (const auto& r : {nt, ss, translation}) {
            EXPECT_TRUE(r.collision);
            // At sub-separation scales, TICCD regards the primitives as
            // already in contact. Preserve its conservative time envelope.
            EXPECT_NEAR(r.t, exponent < -35 ? 0.0 : 0.5, 1e-12);
        }
    }
}

TEST(CCDLinearRobustness, SmallInitialGapIsNotInitialContact) {
    const auto nt = node_triangle_only_one_node_moves(
        Vec3(0.25, 0.25, 5e-9), Vec3(0, 0, -1e-8),
        Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX,
        Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
    const auto ss = segment_segment_only_one_node_moves(
        Vec3(0, 0, 5e-9), Vec3(0, 0, -1e-8), Vec3(1, 0, 5e-9),
        Vec3(0.5, -1, 0), Vec3(0.5, 1, 0), 1e-12, false);
    ASSERT_TRUE(nt.collision);
    EXPECT_NEAR(nt.t, 0.5, 1e-12);
    ASSERT_TRUE(ss.collision);
    EXPECT_NEAR(ss.t, 1.0, 1e-12);
}

TEST(CCDLinearRobustness, DegeneratePrimitivesAgreeWithTICCD) {
    for (const Vec3& b : {Vec3(1, 0, 0), Vec3::Zero().eval()}) {
        const Vec3 p(0, 0, 1), dp(0, 0, -2), a = Vec3::Zero();
        const auto linear = node_triangle_only_one_node_moves(
            p, dp, a, ZERO_DX, b, ZERO_DX, a, ZERO_DX, 1e-12, false);
        const auto reference = node_triangle_only_one_node_moves(
            p, dp, a, ZERO_DX, b, ZERO_DX, a, ZERO_DX, 1e-12, true);
        ASSERT_TRUE(reference.collision);
        ASSERT_TRUE(linear.collision);
        EXPECT_NEAR(linear.t, reference.t, 2e-6);
    }
    const Vec3 p(0, 0, 1), dp(0, 0, -2), a = Vec3::Zero();
    const auto linear = segment_segment_only_one_node_moves(p, dp, Vec3(1, 0, 0), a, a, 1e-12, false);
    const auto reference = segment_segment_only_one_node_moves(p, dp, Vec3(1, 0, 0), a, a, 1e-12, true);
    ASSERT_TRUE(reference.collision);
    ASSERT_TRUE(linear.collision);
    EXPECT_NEAR(linear.t, reference.t, 2e-6);
    const auto translated = segment_segment_same_displacement_linear_ccd(p, dp, p, dp, a, Vec3(1, 0, 0));
    ASSERT_TRUE(translated.collision);
    EXPECT_NEAR(translated.t, 0.5, 2e-6);
}

TEST(CCDLinearRobustness, SeededTOIAgreementWithTICCD) {
    std::mt19937 random(0xCCD2026);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    for (int sample = 0; sample < 128; ++sample) {
        const Vec3 axis = Vec3(unit(random), unit(random), unit(random)).normalized();
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(3.0 * unit(random), axis).toRotationMatrix();
        const double scale = std::pow(10.0, sample % 3 - 1);
        const Vec3 offset = 100.0 * Vec3(unit(random), unit(random), unit(random));
        const double expected = 0.1 + 0.8 * (unit(random) + 1.0) / 2.0;
        const Vec3 motion = scale * rotation * Vec3(0.125, -0.25, 2.0);
        const auto transform = [&](const Vec3& p) -> Vec3 { return offset + scale * rotation * p; };
        const auto check = [&](const CCDResult& linear, const CCDResult& reference) {
            ASSERT_TRUE(reference.collision);
            ASSERT_TRUE(linear.collision);
            EXPECT_TRUE(std::isfinite(linear.t));
            EXPECT_NEAR(linear.t, expected, 2e-10);
            EXPECT_NEAR(linear.t, reference.t, 2e-5 / scale);
        };
        for (int role = 0; role < 4; ++role) {
            SCOPED_TRACE(::testing::Message() << "sample=" << sample << " NT role=" << role);
            std::array<Vec3, 4> x{{transform(Vec3(0.25, 0.25, 0)), transform(Vec3::Zero()),
                transform(Vec3(1, 0, 0)), transform(Vec3(0, 1, 0))}};
            std::array<Vec3, 4> dx{{ZERO_DX, ZERO_DX, ZERO_DX, ZERO_DX}};
            x[role] -= expected * motion;
            dx[role] = motion;
            check(node_triangle_only_one_node_moves(x[0], dx[0], x[1], dx[1], x[2], dx[2], x[3], dx[3], 1e-12, false),
                  node_triangle_only_one_node_moves(x[0], dx[0], x[1], dx[1], x[2], dx[2], x[3], dx[3], 1e-12, true));
        }
        for (int reversed = 0; reversed < 2; ++reversed) {
            SCOPED_TRACE(::testing::Message() << "sample=" << sample << " SS reversed=" << reversed);
            const Vec3 a = transform(Vec3(-1, 0, 0)), b = transform(Vec3(1, 0, 0));
            const Vec3 c = transform(Vec3(0, reversed ? 1 : -1, 0));
            const Vec3 d = transform(Vec3(0, reversed ? -1 : 1, 0));
            const Vec3 start = a - expected * motion;
            check(segment_segment_only_one_node_moves(start, motion, b, c, d, 1e-12, false),
                  segment_segment_only_one_node_moves(start, motion, b, c, d, 1e-12, true));
            const Vec3 end = b - expected * motion;
            check(segment_segment_same_displacement_linear_ccd(start, motion, end, motion, c, d),
                  CCDResult{true, segment_segment_general_ccd(start, motion, end, motion, c, ZERO_DX, d, ZERO_DX)});
        }
    }
}

TEST(CCDLinearRobustness, SeededSeparatedQueriesAgreeWithTICCD) {
    std::mt19937 random(0xCCD123);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    for (int sample = 0; sample < 128; ++sample) {
        SCOPED_TRACE(sample);
        const Vec3 axis = Vec3(unit(random), unit(random), unit(random)).normalized();
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(unit(random), axis).toRotationMatrix();
        const auto p = [&](const Vec3& x) -> Vec3 { return rotation * x; };
        for (bool reference : {false, true}) {
            const auto nt = node_triangle_only_one_node_moves(p(Vec3(2, 2, 1)), p(Vec3(0, 0, -2)),
                Vec3::Zero(), ZERO_DX, p(Vec3(1, 0, 0)), ZERO_DX, p(Vec3(0, 1, 0)), ZERO_DX, 1e-12, reference);
            EXPECT_FALSE(nt.collision);
            EXPECT_TRUE(std::isnan(nt.t));
            const auto ss = segment_segment_only_one_node_moves(p(Vec3(0, 0, 1)), p(Vec3(0, 0, -2)),
                p(Vec3(1, 0, 0)), p(Vec3(2, -1, 0)), p(Vec3(2, 1, 0)), 1e-12, reference);
            EXPECT_FALSE(ss.collision);
            EXPECT_TRUE(std::isnan(ss.t));
        }
    }
}

TEST(CCDLinearRobustness, NearParallelCrossingsAgreeWithTICCD) {
    for (double angle : {1e-4, 1e-7, 1e-10, 1e-13}) {
        SCOPED_TRACE(angle);
        const Vec3 a(-1, -angle, 1), b(1, angle, 0), c(-1, 0, 0), d(1, 0, 0), dx(0, 0, -2);
        const auto linear = segment_segment_only_one_node_moves(a, dx, b, c, d, 1e-12, false);
        const auto reference = segment_segment_only_one_node_moves(a, dx, b, c, d, 1e-12, true);
        ASSERT_TRUE(linear.collision);
        ASSERT_TRUE(reference.collision);
        if (angle > 1e-9) EXPECT_NEAR(linear.t, 0.5, 1e-12);
        EXPECT_NEAR(linear.t, reference.t, 2e-6);
    }
}

TEST(CCDLinearRobustness, ContactsNearTimeBoundaries) {
    for (double t : {0.0, 1e-12, 1.0 - 1e-12, 1.0}) {
        SCOPED_TRACE(t);
        const auto nt = node_triangle_only_one_node_moves(Vec3(0.25, 0.25, t), Vec3(0, 0, -1),
            Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
        const auto ss = segment_segment_only_one_node_moves(Vec3(0, 0, t), Vec3(0, 0, -1),
            Vec3(1, 0, 0), Vec3(0.5, -1, 0), Vec3(0.5, 1, 0), 1e-12, false);
        ASSERT_TRUE(nt.collision);
        ASSERT_TRUE(ss.collision);
        EXPECT_NEAR(nt.t, t, 2e-12);
        EXPECT_NEAR(ss.t, t, 2e-12);
    }
}

TEST(CCDLinearRobustness, RotatedThinTriangleBoundariesAgreeWithTICCD) {
    for (double width : {1e-5, 1e-8, 1e-11}) {
        for (double angle : {0.0, 0.37, 1.41}) {
            const Eigen::Matrix3d rotation = Eigen::AngleAxisd(angle, Vec3(1, 2, 3).normalized()).toRotationMatrix();
            for (const Vec3& contact : {Vec3(0.75, width * 0.25, 0), Vec3(0.5, 0, 0), Vec3(1, width, 0)}) {
                SCOPED_TRACE(::testing::Message() << "width=" << width << " angle=" << angle << " contact=" << contact.transpose());
                const Vec3 p = rotation * (contact + Vec3(0, 0, 0.7));
                const Vec3 dp = rotation * Vec3(0, 0, -2);
                const Vec3 a = Vec3::Zero(), b = rotation * Vec3(1, 0, 0), c = rotation * Vec3(1, width, 0);
                const auto linear = node_triangle_only_one_node_moves(p, dp, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, false);
                const auto reference = node_triangle_only_one_node_moves(p, dp, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, true);
                ASSERT_TRUE(reference.collision);
                ASSERT_TRUE(linear.collision);
                EXPECT_NEAR(linear.t, reference.t, 2e-5);
            }
        }
    }
}

TEST(CCDLinearRobustness, CoplanarMovingTriangleAndCollapseAgreeWithTICCD) {
    // The triangle changes orientation through a collapsed state at t=0.5.
    // The query point is first reached at t=0.625 (interior) or 0.75 (edge).
    for (double px : {-0.5, 0.0, 0.5}) {
        const Vec3 p(px, -0.25, 0), a(-1, 0, 0), b(1, 0, 0), c(0, -1, 0), dc(0, 2, 0);
        // Reverse the motion so the point is initially outside the triangle.
        const Vec3 start = c + dc;
        const auto linear = node_triangle_only_one_node_moves(p, ZERO_DX, a, ZERO_DX, b, ZERO_DX, start, -dc, 1e-12, false);
        const auto reference = node_triangle_only_one_node_moves(p, ZERO_DX, a, ZERO_DX, b, ZERO_DX, start, -dc, 1e-12, true);
        ASSERT_TRUE(linear.collision);
        ASSERT_TRUE(reference.collision);
        EXPECT_NEAR(linear.t, px == 0.0 ? 0.625 : 0.75, 1e-12);
        // Coplanar inclusion searches can exhaust the reference iteration
        // budget and return a wider conservative bracket.
        EXPECT_NEAR(linear.t, reference.t, 5e-5);
    }
    const auto linear = segment_segment_only_one_node_moves(Vec3(0, 0, 1), Vec3(0, 0, -2), Vec3::Zero(),
        Vec3(-1, 0, 0), Vec3(1, 0, 0), 1e-12, false);
    ASSERT_TRUE(linear.collision);
    EXPECT_DOUBLE_EQ(linear.t, 0.0); // The stationary endpoint is already in contact.
}

TEST(CCDLinearRobustness, CoplanarAndParallelNearMissesStaySeparated) {
    for (double gap : {1e-5, 1e-7, 1e-9}) {
        SCOPED_TRACE(gap);
        for (bool reference : {false, true}) {
            EXPECT_FALSE(node_triangle_only_one_node_moves(Vec3(0.25, 0.25, gap), Vec3(0.1, 0, 0),
                Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX, 1e-12, reference).collision);
            EXPECT_FALSE(segment_segment_only_one_node_moves(Vec3(0, gap, 0), Vec3(0.1, 0, 0),
                Vec3(1, gap, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), 1e-12, reference).collision);
        }
        EXPECT_FALSE(segment_segment_same_displacement_linear_ccd(Vec3(0, gap, 0), Vec3(0.1, 0, 0),
            Vec3(1, gap, 0), Vec3(0.1, 0, 0), Vec3::Zero(), Vec3(1, 0, 0)).collision);
    }
}

TEST(CCDLinearRobustness, NearbyClothMovingAwayDoesNotFreezeSafeStep) {
    // Tilted cloth patches have overlapping swept AABBs while separated.
    // The old 1e-8 geometric padding reported contact at t=0; TICCD correctly
    // permits the entire step. Exercise the production vertex update path.
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.6, Vec3(1, 0, 1).normalized()).toRotationMatrix();
    const Vec3 normal = rotation * Vec3::UnitY();
    const Vec3 fall(0, -0.001, 0);
    for (double scale : {100.0, 10000.0}) {
        const double gap = 5e-9 * scale;
        for (bool reference : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "scale=" << scale << " TICCD=" << reference);
            std::vector<Vec3> x = {scale * rotation * Vec3(0.25, 0, 0.25) - gap * normal,
                Vec3::Zero(), scale * rotation * Vec3(1, 0, 0), scale * rotation * Vec3(0, 0, 1)};
            BroadPhase bp;
            auto& cache = bp.mutable_cache();
            cache.vertex_nt.resize(4);
            cache.vertex_ss.resize(4);
            cache.node_boxes.assign(4, AABB(Vec3::Constant(-2 * scale), Vec3::Constant(2 * scale)));
            NodeTrianglePair pair{};
            pair.node = 0;
            for (int i = 0; i < 3; ++i) pair.tri_v[i] = i + 1;
            cache.nt_pairs.push_back(pair);
            cache.vertex_nt[0].push_back({0, 0});
            const Vec3 target = x[0] + fall;
            const double step = per_vertex_safe_step(bp, x, 0, target, 0.9, true, reference);
            EXPECT_DOUBLE_EQ(step, 1.0);
            EXPECT_TRUE(x[0].isApprox(target, 1e-14));
        }
    }
}

TEST(CCDLinearRobustness, SkinnyTriangleDoesNotFreezeAnExteriorNode) {
    // Cancellation in the old Gram barycentrics put this exterior point
    // inside the triangle at t=0, although TICCD reports no contact.
    const Vec3 p(5000, 0.0002 * 0.501, 0), dx(0, 0.0002, 0);
    for (bool reference : {false, true}) {
        const auto r = node_triangle_only_one_node_moves(p, dx, Vec3::Zero(), ZERO_DX,
            Vec3(10000, 0, 0), ZERO_DX, Vec3(10000, 0.0002, 0), ZERO_DX, 1e-12, reference);
        EXPECT_FALSE(r.collision);
        EXPECT_TRUE(std::isnan(r.t));
    }
}

TEST(CCDLinearRobustness, NearbyClothApproachingStillStopsBeforeContact) {
    const Vec3 a(0, 0, 0), b(1, 0, 0), c(0, 0, 1);
    const Vec3 p(0.25, 5e-9, 0.25), dx(0, -1e-8, 0);
    const auto r = node_triangle_only_one_node_moves(p, dx, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, false);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, 1e-12);
    EXPECT_GT((p + 0.9 * r.t * dx).y(), 0.0);
}

TEST(CCDLinearRobustness, ThinTriangleVertexHitSurvivesInterpolationRoundoff) {
    const std::array<Vec3, 3> corners{{Vec3::Zero(), Vec3(1, 0, 0), Vec3(1, 1e-8, 0)}};
    std::array<int, 3> permutation{{0, 1, 2}};
    do {
        const Vec3 a = corners[permutation[0]], b = corners[permutation[1]], c = corners[permutation[2]];
        const Vec3 p(1, -0.7, -0.7), dx(0, 1, 1);
        const auto linear = node_triangle_only_one_node_moves(p, dx, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, false);
        const auto reference = node_triangle_only_one_node_moves(p, dx, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, true);
        ASSERT_TRUE(reference.collision);
        ASSERT_TRUE(linear.collision);
        EXPECT_NEAR(linear.t, reference.t, 2e-5);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
}

TEST(CCDLinearRobustness, LongParallelEdgesDoNotLoseTransverseMotion) {
    const Vec3 a(0, 0, 1), b(1e170, 0, 1), c(0, 0, 0), d(1e170, 0, 0), dx(0, 0, -2);
    const auto r = segment_segment_same_displacement_linear_ccd(a, dx, b, dx, c, d);
    ASSERT_TRUE(r.collision);
    EXPECT_DOUBLE_EQ(r.t, 0.5);
}

TEST(CCDLinearRobustness, InitialMinimumSeparationAgreesWithTICCD) {
    for (double direction : {-1.0, 1.0}) {
        const Vec3 p(0.25, 0.25, 5e-11), dx(0, 0, direction * 1e-10);
        const auto linear = node_triangle_only_one_node_moves(p, dx, Vec3::Zero(), ZERO_DX,
            Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX, 1e-12, false);
        const auto reference = node_triangle_only_one_node_moves(p, dx, Vec3::Zero(), ZERO_DX,
            Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX, 1e-12, true);
        ASSERT_TRUE(reference.collision);
        ASSERT_TRUE(linear.collision);
        EXPECT_DOUBLE_EQ(linear.t, reference.t);
    }
}

TEST(CCDLinearRobustness, RootJustOutsideStepDoesNotCreateFalseEndpointContact) {
    for (bool after_step : {false, true}) {
        // The root is only 1e-13 outside [0,1], but that corresponds to a
        // resolvable 1e-4 spatial gap at this speed. In particular, clamping
        // the negative root to zero must not freeze separating motion.
        const double height = after_step ? 1e9 + 1e-4 : 1e-4;
        const Vec3 dx(0, 0, after_step ? -1e9 : 1e9);
        for (bool reference : {false, true}) {
            EXPECT_FALSE(node_triangle_only_one_node_moves(Vec3(0.25, 0.25, height), dx,
                Vec3::Zero(), ZERO_DX, Vec3(1, 0, 0), ZERO_DX, Vec3(0, 1, 0), ZERO_DX,
                1e-12, reference).collision);
            EXPECT_FALSE(segment_segment_only_one_node_moves(Vec3(0, 0, height), dx, Vec3(1, 0, 0),
                Vec3(0.5, -1, 0), Vec3(0.5, 1, 0), 1e-12, reference).collision);
        }
        EXPECT_FALSE(segment_segment_same_displacement_linear_ccd(Vec3(0, 0, height), dx,
            Vec3(1, 0, height), dx, Vec3(0.5, -1, 0), Vec3(0.5, 1, 0)).collision);
    }
}

TEST(CCDLinearRobustness, BarycentricPaddingDoesNotCreateInitialContact) {
    const Vec3 a = Vec3::Zero(), b(1e9, 0, 0), c(0, 1e9, 0);
    const Vec3 p(0.5e9, (0.5 + 5e-13) * 1e9, 0), dp(0, 1e9, 0);
    for (bool reference : {false, true}) {
        const auto r = node_triangle_only_one_node_moves(p, dp, a, ZERO_DX, b, ZERO_DX, c, ZERO_DX, 1e-12, reference);
        EXPECT_FALSE(r.collision);
        const auto ss = segment_segment_only_one_node_moves(Vec3(0, 0, 0), Vec3(-1, 0, 0),
            Vec3(1e9, 0, 0), Vec3(1e9 + 5e-4, 0, 0), Vec3(2e9, 0, 0), 1e-12, reference);
        EXPECT_FALSE(ss.collision);
    }
}

TEST(CCDLinearRobustness, FastTangentialMotionDoesNotHideTransverseSeparation) {
    const Vec3 a(0, 1e-4, 1), b(1e12, 1e-4, 1), c = Vec3::Zero(), d(1e12, 0, 0), dx(1e12, 0, -2);
    const auto linear = segment_segment_same_displacement_linear_ccd(a, dx, b, dx, c, d);
    EXPECT_FALSE(linear.collision);
    EXPECT_DOUBLE_EQ(segment_segment_general_ccd(a, dx, b, dx, c, ZERO_DX, d, ZERO_DX), 1.0);
}

TEST(CCDNodeTriangleSingleMovingNode, InteriorHit) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDNodeTriangleSingleMovingNode, ParallelNoCrossing) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

TEST(CCDNodeTriangleSingleMovingNode, CoplanarEntireStep) {
    const Vec3 x(0.25, 0.25, 0.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.0, kTol);
}

TEST(CCDNodeTriangleSingleMovingNode, CoplanarForEntireStepEndpointHit) {
    // The node remains in the triangle's plane and first reaches the
    // hypotenuse exactly at the end of the step.
    const Vec3 x(1.5, 0.25, 0.0);
    const Vec3 dx(-0.75, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0, 1.0e-12);
}

TEST(CCDNodeTriangleSingleMovingNode, CoplanarCloseDistinctRootsUseTrueEntry) {
    // The point first crosses the extension of edge (x1,x2) at t=0, then
    // enters the finite triangle through edge (x2,x3) 5.56e-10 later. These
    // are distinct events even though their normalized times are very close.
    const Vec3 x(1.0 + 5.0e-8, 0.0, 0.0);
    const Vec3 dx(-100.0, 10.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const double expected_t = 5.0e-8 / 90.0;

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);

    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, expected_t, 1.0e-15);
}

TEST(CCDNodeTriangleSingleMovingNode, CandidateOutsideStepInterval) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, 2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

// ===========================================================================
// Linear CCD with a single moving TRIANGLE VERTEX (any of the three corners).
// Same linear function as above; only the moving DOF differs.
//
// Setup uses triangle with vertices at (0,0,0), (2,0,0), (0,2,0). When the
// chosen corner rises in +z with dz=2, geometry was hand-derived so the
// moving plane sweeps across the static node x at exactly t=0.5.
// ===========================================================================

TEST(CCDNodeTriangleSingleMovingTriVertex, V0HitsStaticNode) {
    const Vec3 x(0.5, 0.5, 0.5);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 2.0);
    const Vec3 x2(2.0, 0.0, 0.0);
    const Vec3 x3(0.0, 2.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, dx1, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, V1HitsStaticNode) {
    const Vec3 x(1.0, 0.5, 0.5);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(2.0, 0.0, 0.0), dx2(0.0, 0.0, 2.0);
    const Vec3 x3(0.0, 2.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, ZERO_DX, x2, dx2, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, V2HitsStaticNode) {
    const Vec3 x(0.5, 1.0, 0.5);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(2.0, 0.0, 0.0);
    const Vec3 x3(0.0, 2.0, 0.0), dx3(0.0, 0.0, 2.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, ZERO_DX, x2, ZERO_DX, x3, dx3,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, MovingVertexCoincidesWithStaticNode) {
    // Vertex x1 moves up through the static node x; at t=0.5, x1 == x exactly.
    const Vec3 x(0.0, 0.0, 1.0);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, dx1, x2, ZERO_DX, x3, ZERO_DX,
        /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSingleMovingNode, InteriorHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSingleMovingNode, ParallelNoCrossing) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 1.0, 0.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSingleMovingNode, CoplanarEntireStep) {
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 dx1(0.0, 1.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.0, kTol);
}

TEST(CCDSegmentSegmentSingleMovingNode, CollinearForEntireStepEndpointHit) {
    // The moving endpoint extends the first segment until it first touches
    // the second segment exactly at the end of the step.
    const Vec3 x1(1.0, 0.0, 0.0);
    const Vec3 dx1(1.0, 0.0, 0.0);
    const Vec3 x2(0.0, 0.0, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x4(3.0, 0.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0, 1.0e-12);
}

TEST(CCDSegmentSegmentSingleMovingNode, CoplanarForEntireStepEndpointHit) {
    // The segments stay coplanar and non-collinear. The moving endpoint first
    // reaches the static segment exactly at the end of the step.
    const Vec3 x1(0.0, -1.0, 0.0);
    const Vec3 dx1(0.0, 1.0, 0.0);
    const Vec3 x2(1.0, -1.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(0.0, 1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0, 1.0e-12);
}

TEST(CCDSegmentSegmentSingleMovingNode, CoplanarCloseDistinctRootsUseTrueEntry) {
    // At t=0 the static endpoint x3 lies on the moving segment's supporting
    // line but is 5e-8 beyond the finite segment. The moving endpoint reaches
    // x3 at t=5e-10; the two events must not be merged.
    const Vec3 x1(5.0e-8, 0.0, 0.0);
    const Vec3 dx1(-100.0, 300.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(0.0, 4.0, 0.0);
    const double expected_t = 5.0e-10;

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);

    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, expected_t, 1.0e-15);
}

TEST(CCDSegmentSegmentSingleMovingNode, CandidateOutsideStepInterval) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, 2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSingleMovingNode, NearParallelHitAtNonSampledTime) {
    // At t=0.3 the moving segment is almost parallel to the static segment,
    // but crosses it at its midpoint. The TOI is not one of the historical
    // fixed probe times, so it must come from the affine coplanarity root.
    const Vec3 x1(0.0, -2.0e-7, -3.0e-7);
    const Vec3 dx1(0.0, 0.0, 1.0e-6);
    const Vec3 x2(1.0, 2.0e-7, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(1.0, 0.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.3, 1.0e-12);
}

TEST(CCDSegmentSegmentSingleMovingNode, NearParallelCoplanarityWithoutOverlap) {
    // The supporting lines become coplanar at t=0.3, but the finite segments
    // are disjoint. Coplanarity alone must not produce a collision.
    const Vec3 x1(0.0, -2.0e-7, -3.0e-7);
    const Vec3 dx1(0.0, 0.0, 1.0e-6);
    const Vec3 x2(1.0, 2.0e-7, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x4(3.0, 0.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
}

// ---------------------------------------------------------------------------
// Regression guard for an example-4 segment pair whose distance increases
// throughout the sampled motion. It should remain a no-collision case.
TEST(CCDSegmentSegmentSingleMovingNode, RepoExample4Frame9Sub6Iter5) {
    const Vec3 x1(-5.33363314325580457e-02, 1.57380202214873172e-01, 3.54253855534581164e-01);
    const Vec3 dx1( 3.14637561392388425e-03,-1.90288924652220982e-03, 5.87251965795404507e-03);
    const Vec3 x2(-9.03270283891810799e-02, 1.85085196018025616e-01, 3.53048101908432388e-01);
    const Vec3 x3(-7.56544663979064197e-02, 1.74344892610936164e-01, 3.42000252922963821e-01);
    const Vec3 x4(-6.78743242217503817e-02, 1.67293411642532724e-01, 3.60000372162717852e-01);

    double d_min = 1e9; double t_at_min = -1.0;
    double d_prev = -1.0;
    for (int i = 0; i <= 100; ++i) {
        double t = i / 100.0;
        Vec3 x1t = x1 + dx1 * t;
        auto d = segment_segment_distance(x1t, x2, x3, x4);
        if (d.distance < d_min) { d_min = d.distance; t_at_min = t; }
        if (i > 0) {
            EXPECT_GE(d.distance, d_prev - 1e-12) << "distance should not decrease at sample " << i;
        }
        d_prev = d.distance;
    }

    const CCDResult r = segment_segment_only_one_node_moves(
        x1, dx1, x2, x3, x4, /*eps=*/1.0e-12, /*use_ticcd=*/false);
    EXPECT_FALSE(r.collision);
    EXPECT_NEAR(d_min, 4.36963887902074195e-04, 1e-15);
    EXPECT_DOUBLE_EQ(t_at_min, 0.0);
}

TEST(CCDSegmentSegmentSameDisplacement, InteriorHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, CoplanarityWithoutFiniteOverlap) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(2.0, -1.0, 0.0);
    const Vec3 x4(2.0,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, SpatialEndpointHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(1.0, 0.0, 0.0);
    const Vec3 x4(1.0, 1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, StepEndpointHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, -1.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 1.0, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, CZeroDNonzero) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, CandidateOutsideStepInterval) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx(0.0, 0.0, 1.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, CoplanarSweep) {
    const Vec3 x1(0.0, -1.0, 0.0);
    const Vec3 dx(0.0, 2.0, 0.0);
    const Vec3 x2(1.0, -1.0, 0.0);
    const Vec3 x3(0.5, -0.25, 0.0);
    const Vec3 x4(0.5,  0.25, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.375, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, CoplanarSweepWithoutFiniteOverlap) {
    const Vec3 x1(-2.0, 2.0, 0.0);
    const Vec3 dx(3.0, 0.0, 0.0);
    const Vec3 x2(-1.0, 2.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(0.0, 1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, ParallelCollinearSweep) {
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 dx(2.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x4(3.0, 0.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, ParallelTransverseSweep) {
    const Vec3 x1(0.0, 1.0, 0.0);
    const Vec3 dx(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 1.0, 0.0);
    const Vec3 x3(0.5, 0.0, 0.0);
    const Vec3 x4(1.5, 0.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, ParallelTransverseSweepWithoutFiniteOverlap) {
    const Vec3 x1(0.0, 1.0, 0.0);
    const Vec3 dx(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 1.0, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x4(3.0, 0.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    EXPECT_FALSE(r.collision);
}

TEST(CCDSegmentSegmentSameDisplacement, SmallTransverseVelocityStillFindsContact) {
    // A transverse component small relative to the total displacement must
    // not be rounded to zero: the moving segment reaches x4 at t=0.6.
    const Vec3 x1(0.0, 5.4e-8, 0.0);
    const Vec3 dx(10.0, -9.0e-8, 0.0);
    const Vec3 x2(1.0, 5.4e-8, 0.0);
    const Vec3 x3(5.0, 0.0, 0.0);
    const Vec3 x4(6.0, 0.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.6, 1.0e-12);
}

TEST(CCDSegmentSegmentSameDisplacement, NearParallelCoplanarSweepUsesTrueEntry) {
    // The directions differ by only 1e-8 radians at this scale. Treating them
    // as exactly parallel delays the reported contact from x2/x4 at t=0.5 to
    // x1/x3 at t=1.
    const Vec3 x1(0.0, 2.0, 0.0);
    const Vec3 dx(0.0, -2.0, 0.0);
    const Vec3 x2(1.0e8, 2.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(1.0e8, 1.0, 0.0);

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSameDisplacement, TinyAngleStillUsesNonparallelRoot) {
    // Even an angle below eps has a well-defined nonparallel endpoint root.
    // Collapsing it to the parallel branch would delay contact until t=1.
    const Vec3 x1(0.0, 2.0, 0.0);
    const Vec3 dx(0.0, -2.0, 0.0);
    const Vec3 x2(1.0e8, 2.0, 0.0);
    const Vec3 x3(0.0, 0.0, 0.0);
    const Vec3 x4(1.0e8, 9.0e-5, 0.0);
    const double expected_t = (2.0 - 9.0e-5) / 2.0;

    const CCDResult r = segment_segment_same_displacement_linear_ccd(
        x1, dx, x2, dx, x3, x4);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(r.t, expected_t, 1.0e-12);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////  Rigid Body CCD Test /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
const Vec4 kIdentityQ(1.0, 0.0, 0.0, 0.0);

Vec4 AxisAngleQuat(const Vec3& axis, double angle) {
    const Vec3 normalized_axis = axis.normalized();
    const double sin_half_angle = std::sin(0.5 * angle);
    return Vec4(std::cos(0.5 * angle), sin_half_angle * normalized_axis[0], sin_half_angle * normalized_axis[1], sin_half_angle * normalized_axis[2]);
}
}  // namespace

TEST(SegmentSegmentRBRotationCCD, NoRotationReturnsFalse) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 2.0);
    const Vec3 x2(1.0, -0.5, 1.0);
    const Vec3 x3(2.0, 0.5, 1.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, kIdentityQ, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(SegmentSegmentRBRotationCCD, SegmentOnAxisReturnsFalse) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, 0.0, 1.0);
    const Vec3 x1(0.0, 0.0, 3.0);
    const Vec3 x2(1.0, 0.0, 2.0);
    const Vec3 x3(-1.0, 0.0, 2.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_FrustumHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 2.0);
    const Vec3 x2(1.0, -0.5, 1.0);
    const Vec3 x3(2.0, 0.5, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_PiercingAnnulusHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, 0.0, -1.0);
    const Vec3 x3(1.5, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseB2_SamePlaneHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, 0.0, 0.0);
    const Vec3 x3(3.0, 0.0, 0.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_SkewSegmentGeneralHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 2.0, 3.0);
    const Vec3 x2(-0.10131289848292738, 0.60967344361641129, 1.5);
    const Vec3 x3(-0.26524061172707147, 1.596145797425957, 1.5);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.4, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_SkewSegmentOppositeSidesHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(1.0, 0.0, -1.0);
    const Vec3 x1(0.0, 2.0, 2.0);
    const Vec3 x2(-0.10131289848292738, 0.60967344361641129, 0.5);
    const Vec3 x3(-0.26524061172707147, 1.596145797425957, 0.5);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.4, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_SkewSegmentOppositeXSidesHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.5, -1.0);
    const Vec3 x1(1.0, -0.3, 2.0);
    const Vec3 x2(0.23511410091698948, -0.32360679774997875, 0.5);
    const Vec3 x3(-0.35267115137548433, 0.48541019662496826, 0.5);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.4, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_ArbitraryAxisHit) {
    const Vec3 x_com(0.5, -0.3, 0.2);
    const Vec3 x0(1.5, -0.3, -0.8);
    const Vec3 x1(0.5, 1.7, 2.2);
    const Vec3 x2(1.2351048854627524, 0.30605919321519798, 0.85883592132205056);
    const Vec3 x3(0.48919814277551343, 0.9666188030347671, 0.94418305418971882);

    const Vec4 q_new = AxisAngleQuat(Vec3(1, 1, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.4, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseA_HourglassEndpointTouchHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(2.0, 0.0, -2.0);
    const Vec3 x1(-2.0, 0.0, 2.0);
    const Vec3 x2(0.0, 1.0, 1.0);
    const Vec3 x3(0.0, 1.0, -1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_TrueInnerRadiusHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.2, 0.0);
    const Vec3 x1(1.0, 0.2, 0.0);
    const Vec3 x2(0.5, 0.0, -1.0);
    const Vec3 x3(0.5, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), -M_PI / 4.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.52395952173781901, 1e-9);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_InsideTrueInnerRadiusNoCollision) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.2, 0.0);
    const Vec3 x1(1.0, 0.2, 0.0);
    const Vec3 x2(0.1, 0.0, -1.0);
    const Vec3 x3(0.1, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_SkewTrueInnerRadiusHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.2, 0.0);
    const Vec3 x1(1.0, 0.5, 0.0);
    const Vec3 x2(0.6, 0.0, -1.0);
    const Vec3 x3(0.6, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), -M_PI / 2.0);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.48624588465853014, 1e-9);
}

TEST(SegmentSegmentRBRotationCCD, CaseB1_SkewInsideTrueInnerRadiusNoCollision) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(-1.0, 0.2, 0.0);
    const Vec3 x1(1.0, 0.5, 0.0);
    const Vec3 x2(0.2, 0.0, -1.0);
    const Vec3 x3(0.2, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(SegmentSegmentRBRotationCCD, CaseB2_ParallelPlanesNoCollision) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, 0.0, 5.0);
    const Vec3 x3(3.0, 0.0, 5.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    EXPECT_FALSE(hit);
}

TEST(PointTriangleRBRotationCCD, CoplanarTriangleHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(3.0, 0.0, 0.0);
    const Vec3 x4(2.0, 1.0, 0.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = point_triangle_rb_rotation_ccd(
        x, x_com, q_new, kIdentityQ, x2, x3, x4, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-14);
}

TEST(PointTriangleRBRotationCCD, TiltedAxisAlignedTriangleHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 0.0, -1.0);
    const Vec3 x3(3.0, 0.0, -1.0);
    const Vec3 x4(2.0, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = point_triangle_rb_rotation_ccd(
        x, x_com, q_new, kIdentityQ, x2, x3, x4, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.5, 1e-14);
}

TEST(PointTriangleRBRotationCCD, SkewTiltedTriangleHit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, -0.5, -0.5);
    const Vec3 x3(2.7, 0.6, 0.4);
    const Vec3 x4(1.6, 0.7, 0.8);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), M_PI);

    double s = -1.0;
    const bool hit = point_triangle_rb_rotation_ccd(
        x, x_com, q_new, kIdentityQ, x2, x3, x4, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 0.51132265526877663, 1e-14);
}

TEST(PointTriangleRBRotationCCD, RotationBeyond180Hit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x(0.0, -2.0, 0.0);
    const Vec3 x2(1.0, 0.0, -1.0);
    const Vec3 x3(3.0, 0.0, -1.0);
    const Vec3 x4(2.0, 0.0, 1.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), 1.5 * M_PI);

    double s = -1.0;
    const bool hit = point_triangle_rb_rotation_ccd(
        x, x_com, q_new, kIdentityQ, x2, x3, x4, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 1.0 / 3.0, 1e-10);
}

TEST(SegmentSegmentRBRotationCCD, RotationBeyond180Hit) {
    const Vec3 x_com(0.0, 0.0, 0.0);
    const Vec3 x0(0.0, -1.0, 0.0);
    const Vec3 x1(0.0, -2.0, 0.0);
    const Vec3 x2(1.5, 0.0, 0.0);
    const Vec3 x3(3.0, 0.0, 0.0);

    const Vec4 q_new = AxisAngleQuat(Vec3(0, 0, 1), 1.5 * M_PI);

    double s = -1.0;
    const bool hit = segment_segment_rb_rotation_ccd(
        x0, x1, x_com, q_new, kIdentityQ, x2, x3, s);
    ASSERT_TRUE(hit);
    EXPECT_NEAR(s, 1.0 / 3.0, 1e-10);
}

TEST(CCDLinearRobustness, ExtremeAspectRatiosDoNotUnderflowIntoInitialContact) {
    const Vec3 zero = Vec3::Zero();
    for (double length : {1e165, 1e300}) {
        SCOPED_TRACE(length);
        const Vec3 p(0.25 * length, 0.25, 1), b(length, 0, 0), c(0, 1, 0);
        const auto away = node_triangle_only_one_node_moves(p, Vec3(0, 0, 1),
            zero, zero, b, zero, c, zero, 1e-12, false);
        EXPECT_FALSE(away.collision);
        const auto toward = node_triangle_only_one_node_moves(p, Vec3(0, 0, -2),
            zero, zero, b, zero, c, zero, 1e-12, false);
        ASSERT_TRUE(toward.collision);
        EXPECT_DOUBLE_EQ(toward.t, 0.5);
    }
    // Translating by a far corner must not merge the two distinct vertices
    // (0,0,0) and (1,0,0) before an ambiguous query is resolved.
    const auto thin = node_triangle_only_one_node_moves(Vec3(0.25, 0.25, 1), Vec3(0, 0, -2),
        zero, zero, Vec3(1, 0, 0), zero, Vec3(1e18, 1e18, 0), zero, 1e-12, false);
    ASSERT_TRUE(thin.collision);
    EXPECT_DOUBLE_EQ(thin.t, 0.5);
}

TEST(CCDLinearRobustness, SweptSeparationPreservesContactsAfterNormalizationLoss) {
    const Vec3 zero = Vec3::Zero(), down(0, 0, -2);
    const double tiny = std::ldexp(1.0, -55);
    const Vec3 a = zero, b(0.5, 1, 0), c(1, 0, 0);
    // Subtracting c.x() rounds tiny to the same local coordinate as zero.
    // A clear separation is safe to reject, but a real contact must survive.
    const auto miss = node_triangle_only_one_node_moves(Vec3(tiny, 2, 1), down,
        a, zero, b, zero, c, zero, 1e-12, false);
    EXPECT_FALSE(miss.collision);
    for (double y : {tiny, -5e-11}) {
        const auto hit = node_triangle_only_one_node_moves(Vec3(tiny, y, 1), down,
            a, zero, b, zero, c, zero, 1e-12, false);
        ASSERT_TRUE(hit.collision); // Includes the fixed boundary tolerance.
        EXPECT_DOUBLE_EQ(hit.t, 0.5);
    }
    const double large = std::numeric_limits<double>::max() * 0.75;
    for (double y : {-large, 0.0}) {
        const auto r = node_triangle_only_one_node_moves(Vec3(0, y, 1), down,
            Vec3(large, 0, 0), zero, Vec3(0, large, 0), zero,
            Vec3(-large, 0, 0), zero, 1e-12, false);
        EXPECT_EQ(r.collision, y == 0.0);
        if (r.collision) EXPECT_DOUBLE_EQ(r.t, 0.5);
    }
    // The physical tolerance must remain scaled even when a tiny displacement
    // requests exact arithmetic; all these primitives are initially within it.
    const double scale = std::ldexp(1.0, -800);
    const auto near = node_triangle_only_one_node_moves(scale * Vec3(0.25, 0.25, 1),
        Vec3(0, 0, std::numeric_limits<double>::denorm_min()), zero, zero,
        scale * Vec3(1, 0, 0), zero, scale * Vec3(0, 1, 0), zero, 1e-12, false);
    ASSERT_TRUE(near.collision);
    EXPECT_DOUBLE_EQ(near.t, 0.0);
}

TEST(CCDIntegration, TenSeparatedClothsFollowGravityAtTenFixedIterations) {
    struct RestoreOpenMPSettings {
        const int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreOpenMPSettings() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
    } restore;
    struct ClothScene {
        RefMesh mesh;
        DeformedState state;
        std::vector<Pin> pins;
        VertexTriangleMap adjacency;
        BroadPhase broad_phase;
        SimParams params;
    };
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    constexpr int frames = 15;
    constexpr int layers = 10;
    constexpr int subdivisions = 8;
    std::array<ClothScene, 3> scenes;
    for (int mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(mode); // Per-vertex CCD off, independent linear, TICCD.
        auto& scene = scenes[mode];
        IPCArgs3D args;
        args.substeps = 5;
        args.max_substep_iters = 10;
        args.fixed_iters = true;
        args.gy = -9.8;
        args.use_ccd = mode != 0;
        args.use_ticcd = mode == 2;
        scene.params = args.to_sim_params();
        const auto& params = scene.params;
        std::vector<Vec2> material;
        for (int layer = 0; layer < layers; ++layer) {
            build_square_mesh(scene.mesh, scene.state, material, subdivisions, subdivisions,
                1.0, 1.0, Vec3(-0.5, 0.8 + layer * 0.02, -0.5));
        }
        scene.state.velocities.assign(scene.state.deformed_positions.size(), Vec3::Zero());
        scene.mesh.node_to_rb.assign(scene.state.deformed_positions.size(), -1);
        scene.mesh.build_deformable_nodes();
        scene.mesh.build_lumped_mass(params.density, params.thickness);
        scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
        const auto initial = scene.state.deformed_positions;
        for (int frame = 1; frame <= frames; ++frame) {
            SCOPED_TRACE(frame);
            const auto result = advance_one_frame(scene.state, scene.mesh, scene.adjacency,
                scene.pins, params, scene.broad_phase, frame);
            ASSERT_TRUE(result.converged);
            EXPECT_EQ(result.iterations, 50);
            const int steps = frame * params.substeps;
            const double fall = 0.5 * steps * (steps + 1) * params.dt2() * params.gravity.y();
            const double velocity = steps * params.dt() * params.gravity.y();
            for (std::size_t i = 0; i < initial.size(); ++i) {
                // All internal forces vanish for a uniform translation. The
                // default stiffness and limited sweeps must preserve free fall.
                EXPECT_NEAR(scene.state.deformed_positions[i].y(), initial[i].y() + fall, 2e-6) << "node=" << i;
                EXPECT_NEAR(scene.state.velocities[i].y(), velocity, 2e-6) << "node=" << i;
            }
        }
        for (std::size_t i = 0; i < initial.size(); ++i) {
            EXPECT_LT((scene.state.deformed_positions[i] - scenes[0].state.deformed_positions[i]).norm(), 2e-6);
        }
    }
}
