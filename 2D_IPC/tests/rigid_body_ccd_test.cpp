#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include "ccd.h"
#include <random>

using namespace step_filter::ccd;

TEST(CCD, PointSegmentTranslation_Frame69EndpointHit) {
    Vec2 x1{0.02664057288132943, -3.0406644634341644};
    Vec2 dx1{-0.15774506361863982, 0.006790619555892387};
    Vec2 x2{-0.05169457285238759, -2.9854434893910375};
    Vec2 dx2{0.1431603686532615, 0.0022727067031325845};
    Vec2 x3{-0.02664057288132943, -3.0406644634341644};
    Vec2 dx3{0.15774506361863982, 0.006790619555892387};

    double t = -1.0;
    EXPECT_TRUE(point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t));
    EXPECT_NEAR(t, 0.16888371826160567, 1e-10);
    EXPECT_NEAR(safe_step(x1, dx1, x2, dx2, x3, dx3, 0.9),
                0.9 * 0.16888371826160567, 1e-10);
}

// # Run a single test
// ./build/tests/rigid_body_ccd_test --gtest_filter="CCD.PointSegmentRotation_HitsAt45Degrees"

// # Run all tests matching a pattern (wildcard)
// ./build/tests/rigid_body_ccd_test --gtest_filter="CCD.*"


TEST(CCD, PointSegmentRotation_HitsAt45Degrees) {
    // Point (1,0) rotates around origin from theta=0 to theta=pi/2.
    // It crosses the vertical segment at x=1/sqrt(2) exactly at theta=pi/4,
    // which is s=0.5 of the step.
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    Eigen::Vector2d x_com(0.0, 0.0);
    Eigen::Vector2d x(1.0, 0.0);
    double theta_n   = 0.0;
    double theta_new = M_PI / 2.0;

    Eigen::Vector2d x0(inv_sqrt2, 0.0);
    Eigen::Vector2d x1(inv_sqrt2, 1.0);

    double step = 1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);

    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 0.5, 1e-10);
}

TEST(CCD, PointSegmentRotation_MissesSegment) {
    // Same rotation but segment is behind the origin — arc never reaches it.
    Eigen::Vector2d x_com(0.0, 0.0);
    Eigen::Vector2d x(1.0, 0.0);
    double theta_n   = 0.0;
    double theta_new = M_PI / 2.0;

    Eigen::Vector2d x0(-2.0, -1.0);
    Eigen::Vector2d x1(-2.0,  1.0);

    double step = 1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);

    EXPECT_FALSE(hit);
}

TEST(CCD, PointSegmentRotation_HitsAtEndOfStep) {
    // Point (1,0) rotates to (0,1) exactly — segment is horizontal at y=1.
    // Collision happens at s=1.0 (end of step).
    Eigen::Vector2d x_com(0.0, 0.0);
    Eigen::Vector2d x(1.0, 0.0);
    double theta_n   = 0.0;
    double theta_new = M_PI / 2.0;

    Eigen::Vector2d x0(-1.0, 1.0);
    Eigen::Vector2d x1( 1.0, 1.0);

    double step = 1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);

    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 1.0, 1e-10);
}

// The following three tests share one geometry:
//   x_com = (0,-1), x = (1,-1), theta_n = 0
// r = (1,0): the point sits at "3 o'clock" relative to x_com.
// The circle of radius 1 around x_com is tangent to the segment
// x0=(-2,0)..x1=(2,0) at (0,0), the "12 o'clock" position.
// Reaching that tangent point requires a +90 degree (CCW) rotation.

TEST(CCD, RbRotation_TangentMiss) {
    // theta_new=pi/4 is only a 45-degree rotation — doesn't reach the
    // tangent point, so no collision.
    Eigen::Vector2d x_com(0.0, -1.0);
    Eigen::Vector2d x(1.0, -1.0);
    double theta_n   = 0.0;
    double theta_new = M_PI / 4.0;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_FALSE(hit);

    // No collision -> safe_step_rb_rotation reports the full step is safe.
    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 1.0, 1e-10);
}

TEST(CCD, RbRotation_TangentHitAtEndOfStep) {
    // theta_new=pi/2 is a 90-degree rotation — lands exactly on the
    // tangent point at the end of the step (s=1).
    Eigen::Vector2d x_com(0.0, -1.0);
    Eigen::Vector2d x(1.0, -1.0);
    double theta_n   = 0.0;
    double theta_new = M_PI / 2.0;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 1.0, 1e-10);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 0.9 * 1.0, 1e-10);
}

TEST(CCD, RbRotation_TangentHitAtMidpoint) {
    // theta_new=pi is a 180-degree rotation — sweeps through the tangent
    // point exactly halfway through the step (s=0.5).
    Eigen::Vector2d x_com(0.0, -1.0);
    Eigen::Vector2d x(1.0, -1.0);
    double theta_n   = 0.0;
    double theta_new = M_PI;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 0.5, 1e-10);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 0.9 * 0.5, 1e-10);
}

TEST(CCD, RbRotation_NonzeroThetaN) {
    // Same x, x_com, segment as the tests above, but theta_n != 0.
    // r = R(-theta_n)*dx = R(-pi/2)*(1,0) = (0,-1), a 90-degree rotation
    // (dtheta = pi - pi/2 = pi/2) sweeps this onto the tangent point (0,0)
    // at the end of the step (s=1) -- same dtheta, and thus same s, as
    // RbRotation_TangentHitAtEndOfStep (theta_n=0, theta_new=pi/2).
    // This exercises the R(-theta_n) transform: if it were dropped (r=dx),
    // this would instead come out as s=0.
    Eigen::Vector2d x_com(0.0, -1.0);
    Eigen::Vector2d x(1.0, -1.0);
    double theta_n   = M_PI / 2.0;
    double theta_new = M_PI;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 1.0, 1e-10);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 0.9 * 1.0, 1e-10);
}

TEST(CCD, RbRotation_NonzeroThetaN_Miss) {
    // Same x, x_com, segment, and theta_n=pi/2 as RbRotation_NonzeroThetaN,
    // but theta_new=3pi/4 so dtheta=pi/4 (same dtheta as
    // RbRotation_TangentMiss, theta_n=0/theta_new=pi/4) -- still a miss.
    Eigen::Vector2d x_com(0.0, -1.0);
    Eigen::Vector2d x(1.0, -1.0);
    double theta_n   = M_PI / 2.0;
    double theta_new = 3.0 * M_PI / 4.0;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_FALSE(hit);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 1.0, 1e-10);
}

TEST(CCD, RbRotation_NonzeroThetaN_HitAtMidpoint) {
    // Same x, x_com, segment, and theta_n=pi/2 as RbRotation_NonzeroThetaN,
    // but theta_new=3pi/2 so dtheta=pi (same dtheta as
    // RbRotation_TangentHitAtMidpoint, theta_n=0/theta_new=pi) -- sweeps
    // through the tangent point at s=0.5.
    Eigen::Vector2d x_com(0.0, -1.0);
    Eigen::Vector2d x(1.0, -1.0);
    double theta_n   = M_PI / 2.0;
    double theta_new = 3.0 * M_PI / 2.0;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 0.5, 1e-10);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 0.9 * 0.5, 1e-10);
}

// The following five tests share one geometry:
//   x_com = (0,0), x = (0,-1), theta_n = 0
// r = (0,-1): the point sits at "6 o'clock" relative to x_com.
// The circle of radius 1 around x_com (the origin) crosses the segment
// x0=(-2,0)..x1=(2,0) (the x-axis) at TWO points: (-1,0) ["9 o'clock",
// theta* = -pi/2, a CW rotation] and (1,0) ["3 o'clock", theta* = +pi/2,
// a CCW rotation]. For every theta_new tested below (all positive, i.e.
// CCW), s = theta*/theta_new is negative for theta* = -pi/2 and is
// rejected -- only the +pi/2 root at (1,0) is ever reached.

TEST(CCD, RbRotation_TwoCrossings_Miss) {
    // theta_new=pi/4 is only a 45-degree rotation -- doesn't reach the
    // +pi/2 crossing point, so no collision.
    Eigen::Vector2d x_com(0.0, 0.0);
    Eigen::Vector2d x(0.0, -1.0);
    double theta_n   = 0.0;
    double theta_new = M_PI / 4.0;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_FALSE(hit);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 1.0, 1e-10);
}

TEST(CCD, RbRotation_TwoCrossings_HitAtEndOfStep) {
    // theta_new=pi/2 is a 90-degree rotation -- lands exactly on the
    // crossing point (1,0) at the end of the step (s=1).
    Eigen::Vector2d x_com(0.0, 0.0);
    Eigen::Vector2d x(0.0, -1.0);
    double theta_n   = 0.0;
    double theta_new = M_PI / 2.0;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 1.0, 1e-10);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 0.9 * 1.0, 1e-10);
}

TEST(CCD, RbRotation_TwoCrossings_HitAtMidpoint) {
    // theta_new=pi is a 180-degree rotation -- sweeps through the
    // crossing point (1,0) halfway through the step (s=0.5).
    Eigen::Vector2d x_com(0.0, 0.0);
    Eigen::Vector2d x(0.0, -1.0);
    double theta_n   = 0.0;
    double theta_new = M_PI;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 0.5, 1e-10);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 0.9 * 0.5, 1e-10);
}

TEST(CCD, RbRotation_TwoCrossings_HitAtOneThird) {
    // theta_new=3pi/2 is a 270-degree rotation -- the +pi/2 crossing
    // point (1,0) is reached one third of the way through the step
    // (s = (pi/2)/(3pi/2) = 1/3).
    Eigen::Vector2d x_com(0.0, 0.0);
    Eigen::Vector2d x(0.0, -1.0);
    double theta_n   = 0.0;
    double theta_new = 3.0 * M_PI / 2.0;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 1.0 / 3.0, 1e-10);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 0.9 * (1.0 / 3.0), 1e-10);

}

TEST(CCD, RbRotation_TwoCrossings_HitAtQuarter) {
    // theta_new=2pi is a full 360-degree rotation -- the +pi/2 crossing
    // point (1,0) is reached one quarter of the way through the step
    // (s = (pi/2)/(2pi) = 0.25).
    Eigen::Vector2d x_com(0.0, 0.0);
    Eigen::Vector2d x(0.0, -1.0);
    double theta_n   = 0.0;
    double theta_new = 2.0 * M_PI;

    Eigen::Vector2d x0(-2.0, 0.0);
    Eigen::Vector2d x1( 2.0, 0.0);

    double step = -1.0;
    bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(step, 0.25, 1e-10);

    double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, 0.9);
    EXPECT_NEAR(omega, 0.9 * 0.25, 1e-10);

}

// Randomized test: x_com and radius are random, defining a circle
// centered at x_com. The particle x = x_com + radius*(1,0) starts at
// "3 o'clock" (theta_s=0). Let
//   Q(angle) = x_com + radius * (cos(angle), sin(angle))
// be the point reached by rotating x around x_com CCW by `angle`
// (Q(0) = x). The chord endpoints are x0 = Q(pi/3) and x1 = Q(pi - pi/3)
// = Q(2pi/3); since cos(pi-a) = -cos(a) and sin(pi-a) = sin(a), these
// share the same y-coordinate, so the chord between them is HORIZONTAL.
//
// Since x0, x1, and the particle's path Q(theta_s) all lie on the same
// circle, x0 and x1 are exactly the two points where that circle crosses
// the chord's line. With theta_n = 0 and theta_min = min(pi/3, 2pi/3) =
// pi/3, hit = (theta_new >= theta_min) and step = theta_min/theta_new --
// independent of x_com and radius:
//   theta_new = pi/6  <  theta_min -> miss
//   theta_new = pi/2  >= theta_min -> step = (pi/3)/(pi/2)  = 2/3
//   theta_new = 3pi/2 >= theta_min -> step = (pi/3)/(3pi/2) = 2/9
//   theta_new = 2pi   >= theta_min -> step = (pi/3)/(2pi)   = 1/6
//
// The chord is extended by `margin` past x0/x1 (along the same line) so
// the relevant crossing point is strictly interior to the segment
// (t_star in (0,1)) rather than landing exactly on a segment endpoint,
// which would be numerically fragile.
TEST(CCD, RbRotation_RandomHorizontalChord) {
    std::mt19937 rng(12345); // fixed seed -> deterministic test
    std::uniform_real_distribution<double> pos_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> radius_dist(0.5, 3.0);

    const double theta     = M_PI / 3.0;
    const double theta_b   = M_PI - theta; // 2pi/3
    const double theta_min = std::min(theta, theta_b); // pi/3
    const double theta_n   = 0.0;

    Eigen::Vector2d x_com(pos_dist(rng), pos_dist(rng));
    double radius = radius_dist(rng);

    Eigen::Vector2d x  = x_com + radius * Eigen::Vector2d(1.0, 0.0);
    Eigen::Vector2d x0 = x_com + radius * Eigen::Vector2d(std::cos(theta),   std::sin(theta));
    Eigen::Vector2d x1 = x_com + radius * Eigen::Vector2d(std::cos(theta_b), std::sin(theta_b));

    Eigen::Vector2d d_hat = (x1 - x0).normalized();
    double margin = radius;
    Eigen::Vector2d x0e = x0 - margin * d_hat;
    Eigen::Vector2d x1e = x1 + margin * d_hat;

    // theta_new = pi/6 < theta_min -> miss
    {
        double theta_new = M_PI / 6.0;
        double step = -1.0;
        bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0e, x1e, step);
        EXPECT_FALSE(hit);

        double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0e, x1e, 0.9);
        EXPECT_NEAR(omega, 1.0, 1e-9);
    }

    // theta_new = pi/2 >= theta_min -> hit at s = (pi/3)/(pi/2)
    {
        double theta_new = M_PI / 2.0;
        double step = -1.0;
        bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0e, x1e, step);
        EXPECT_TRUE(hit);

        double expect_step = theta_min / theta_new;
        EXPECT_NEAR(step, expect_step, 1e-9);

        double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0e, x1e, 0.9);
        EXPECT_NEAR(omega, 0.9 * expect_step, 1e-9);
    }

    // theta_new = 3pi/2 >= theta_min -> hit at s = (pi/3)/(3pi/2)
    {
        double theta_new = 3.0 * M_PI / 2.0;
        double step = -1.0;
        bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0e, x1e, step);
        EXPECT_TRUE(hit);

        double expect_step = theta_min / theta_new;
        EXPECT_NEAR(step, expect_step, 1e-9);

        double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0e, x1e, 0.9);
        EXPECT_NEAR(omega, 0.9 * expect_step, 1e-9);
    }

    // theta_new = 2pi >= theta_min -> hit at s = (pi/3)/(2pi)
    {
        double theta_new = 2.0 * M_PI;
        double step = -1.0;
        bool hit = point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0e, x1e, step);
        EXPECT_TRUE(hit);

        double expect_step = theta_min / theta_new;
        EXPECT_NEAR(step, expect_step, 1e-9);

        double omega = safe_step_rb_rotation(x, x_com, theta_n, theta_new, x0e, x1e, 0.9);
        EXPECT_NEAR(omega, 0.9 * expect_step, 1e-9);
    }
}
