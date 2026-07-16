#include "ccd.h"
#include "segment_segment_distance.h"

#include <cmath>
#include <gtest/gtest.h>

// Shared TOI tolerance for the basic CCD assertions.
constexpr double kTol = 1.0e-6;

namespace {
const Vec3 ZERO_DX(0.0, 0.0, 0.0);
}  // namespace

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
    return Rigid_Body::ALGEBRA::QuaternionFromVector(axis.normalized() * angle);
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
