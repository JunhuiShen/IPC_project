#include "ccd.h"
#include "segment_segment_distance.h"

#include <gtest/gtest.h>

namespace {

// TICCD's solving precision; the linear-CCD wrapper's TOI is accurate to
// roughly this bound.
constexpr double kTol = 1.0e-6;

void expect_no_flags(const CCDResult& r) {
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_FALSE(r.parallel_or_no_crossing);
}

}  // namespace

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
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
    expect_no_flags(r);
}

TEST(CCDNodeTriangleSingleMovingNode, ParallelNoCrossing) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX);
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_TRUE(r.parallel_or_no_crossing);
}

TEST(CCDNodeTriangleSingleMovingNode, CoplanarEntireStep) {
    const Vec3 x(0.25, 0.25, 0.0);
    const Vec3 dx(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.0, kTol);
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_FALSE(r.parallel_or_no_crossing);
}

TEST(CCDNodeTriangleSingleMovingNode, CandidateOutsideStepInterval) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, 2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX);
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_TRUE(r.parallel_or_no_crossing);
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
        x, ZERO_DX, x1, dx1, x2, ZERO_DX, x3, ZERO_DX);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
    expect_no_flags(r);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, V1HitsStaticNode) {
    const Vec3 x(1.0, 0.5, 0.5);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(2.0, 0.0, 0.0), dx2(0.0, 0.0, 2.0);
    const Vec3 x3(0.0, 2.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, ZERO_DX, x2, dx2, x3, ZERO_DX);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
    expect_no_flags(r);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, V2HitsStaticNode) {
    const Vec3 x(0.5, 1.0, 0.5);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(2.0, 0.0, 0.0);
    const Vec3 x3(0.0, 2.0, 0.0), dx3(0.0, 0.0, 2.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, ZERO_DX, x2, ZERO_DX, x3, dx3);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
    expect_no_flags(r);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, MovingVertexCoincidesWithStaticNode) {
    // Vertex x1 moves up through the static node x; at t=0.5, x1 == x exactly.
    const Vec3 x(0.0, 0.0, 1.0);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, dx1, x2, ZERO_DX, x3, ZERO_DX);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
}

TEST(CCDSegmentSegmentSingleMovingNode, InteriorHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(x1, dx1, x2, x3, x4);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
    expect_no_flags(r);
}

TEST(CCDSegmentSegmentSingleMovingNode, ParallelNoCrossing) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 1.0, 0.0);
    const Vec3 x2(1.0, 0.0, 1.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(x1, dx1, x2, x3, x4);
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_TRUE(r.parallel_or_no_crossing);
}

TEST(CCDSegmentSegmentSingleMovingNode, CoplanarEntireStep) {
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 dx1(0.0, 1.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(x1, dx1, x2, x3, x4);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_TRUE(r.collision);
    EXPECT_NEAR(r.t, 0.0, kTol);
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_FALSE(r.parallel_or_no_crossing);
}

TEST(CCDSegmentSegmentSingleMovingNode, CandidateOutsideStepInterval) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, 2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(x1, dx1, x2, x3, x4);
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_TRUE(r.parallel_or_no_crossing);
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

    const CCDResult r = segment_segment_only_one_node_moves(x1, dx1, x2, x3, x4);
    EXPECT_FALSE(r.collision);
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_TRUE(r.parallel_or_no_crossing);
    EXPECT_NEAR(d_min, 4.36963887902074195e-04, 1e-15);
    EXPECT_DOUBLE_EQ(t_at_min, 0.0);
}

TEST(GeneralCCDNodeTriangle, TightInclusionInteriorHit) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const double t = node_triangle_general_ccd(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX);

    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.5, 1e-6);
}

TEST(GeneralCCDNodeTriangle, TightInclusionNoCollision) {
    const Vec3 x(1.5, 1.5, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const double t = node_triangle_general_ccd(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX);

    EXPECT_DOUBLE_EQ(t, 1.0);
}

TEST(GeneralCCDSegmentSegment, TightInclusionInteriorHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const double t = segment_segment_general_ccd(
        x1, dx1, x2, ZERO_DX, x3, ZERO_DX, x4, ZERO_DX);

    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.5, 1e-6);
}

TEST(GeneralCCDSegmentSegment, TightInclusionNoCollision) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, 2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const double t = segment_segment_general_ccd(
        x1, dx1, x2, ZERO_DX, x3, ZERO_DX, x4, ZERO_DX);

    EXPECT_DOUBLE_EQ(t, 1.0);
}
