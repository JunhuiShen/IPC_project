#include "ccd.h"
#include "solver.h"
#include "make_shape.h"

#include <gtest/gtest.h>

namespace {

constexpr double kTol = 1.0e-12;

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

TEST(CCDNodeTriangleSingleMovingNode, PlaneCrossingButOutsideTriangle) {
    const Vec3 x(1.5, 1.5, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, ZERO_DX, x2, ZERO_DX, x3, ZERO_DX);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
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
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_TRUE(r.coplanar_entire_step);
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

TEST(CCDNodeTriangleSingleMovingTriVertex, OutsideTriangleAtCrossingTime) {
    // Plane crossing at t=0.5, but the static node projects outside the
    // (deformed) triangle, so no actual collision.
    const Vec3 x(2.0, 2.0, -1.0);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 2.0);
    const Vec3 x2(2.0, 0.0, 0.0);
    const Vec3 x3(0.0, 2.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, dx1, x2, ZERO_DX, x3, ZERO_DX);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_NEAR(r.t, 0.5, kTol);
    expect_no_flags(r);
}

TEST(CCDNodeTriangleSingleMovingTriVertex, MatchesGeneralCubicCCD) {
    // Cross-check: the linear single-vertex-moves form must agree with the
    // full cubic node_triangle_general_ccd whenever only one DOF moves.
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0), dx2(0.0, 0.0, 8.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult lin = node_triangle_only_one_node_moves(
        x, ZERO_DX, x1, ZERO_DX, x2, dx2, x3, ZERO_DX);
    const double cub = node_triangle_general_ccd(
        x, ZERO_DX, x1, ZERO_DX, x2, dx2, x3, ZERO_DX);
    ASSERT_TRUE(lin.has_candidate_time);
    EXPECT_NEAR(lin.t, cub, 1e-9);
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

TEST(CCDSegmentSegmentSingleMovingNode, CoplanarCandidateButNoIntersection) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(1.5, -1.0, 0.0);
    const Vec3 x4(1.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(x1, dx1, x2, x3, x4);
    EXPECT_TRUE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
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
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_TRUE(r.coplanar_entire_step);
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

// ===========================================================================
// General CCD tests (all vertices may move)
// ===========================================================================

TEST(GeneralCCDNodeTriangle, InteriorHitSingleNodeMoving) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0), dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0), dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_NEAR(t, 0.5, kTol);
}

TEST(GeneralCCDNodeTriangle, ConsistentWithSingleNodeVersion) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 zero(0.0, 0.0, 0.0);

    const CCDResult r = node_triangle_only_one_node_moves(
        x, dx, x1, zero, x2, zero, x3, zero);
    double t = node_triangle_general_ccd(x, dx, x1, zero, x2, zero, x3, zero);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(t, r.t, kTol);
}

TEST(GeneralCCDNodeTriangle, AllVerticesMoving) {
    const Vec3 x(0.25, 0.25, 1.0),  dx(0.0, 0.0, -1.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0,  0.5);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0,  0.5);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0,  0.5);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_GT(t, 0.0);
    EXPECT_LT(t, 1.0);

    Vec3 pt = x + dx * t;
    Vec3 a  = x1 + dx1 * t;
    Vec3 b  = x2 + dx2 * t;
    Vec3 c  = x3 + dx3 * t;
    Vec3 n  = (b - a).cross(c - a);
    EXPECT_NEAR(n.dot(pt - a), 0.0, 1e-8);
}

TEST(GeneralCCDNodeTriangle, NoCollision) {
    const Vec3 x(0.25, 0.25, 1.0),  dx(0.0, 0.0, 1.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_DOUBLE_EQ(t, 1.0);
}

TEST(GeneralCCDNodeTriangle, MissesTriangle) {
    const Vec3 x(5.0, 5.0, 1.0),  dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_DOUBLE_EQ(t, 1.0);
}

TEST(GeneralCCDNodeTriangle, AlreadyCollidingAtT0) {
    const Vec3 x(0.25, 0.25, 0.0),  dx(0.0, 0.0, 1.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_DOUBLE_EQ(t, 0.0);
}

TEST(GeneralCCDSegmentSegment, InteriorHitSingleNodeMoving) {
    const Vec3 x1(0.0, 0.0, 1.0),  dx1(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0), dx3(0.0, 0.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0), dx4(0.0, 0.0, 0.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_NEAR(t, 0.5, kTol);
}

TEST(GeneralCCDSegmentSegment, ConsistentWithSingleNodeVersion) {
    const Vec3 x1(0.0, 0.0, 1.0),  dx1(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);
    const Vec3 zero(0.0, 0.0, 0.0);

    const CCDResult r = segment_segment_only_one_node_moves(x1, dx1, x2, x3, x4);
    double t = segment_segment_general_ccd(x1, dx1, x2, zero, x3, zero, x4, zero);
    ASSERT_TRUE(r.collision);
    EXPECT_NEAR(t, r.t, kTol);
}

TEST(GeneralCCDSegmentSegment, AllVerticesMoving) {
    // Edge 1 along y-axis at z=0.5, drops. Edge 2 along x-axis at z=-0.5, rises.
    // They meet at t=0.5, z=0, crossing at origin.
    const Vec3 x1(0.0, -1.0, 0.5),  dx1(0.0, 0.0, -1.0);
    const Vec3 x2(0.0,  1.0, 0.5),  dx2(0.0, 0.0, -1.0);
    const Vec3 x3(-1.0, 0.0, -0.5), dx3(0.0, 0.0,  1.0);
    const Vec3 x4( 1.0, 0.0, -0.5), dx4(0.0, 0.0,  1.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_GT(t, 0.0);
    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.5, 1e-6);
}

TEST(GeneralCCDSegmentSegment, NoCollision) {
    const Vec3 x1(0.0, 0.0, 1.0),  dx1(0.0, 0.0, 1.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0), dx3(0.0, 0.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0), dx4(0.0, 0.0, 0.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_DOUBLE_EQ(t, 1.0);
}

TEST(GeneralCCDSegmentSegment, AlreadyCollidingAtT0) {
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 1.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0), dx3(0.0, 0.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0), dx4(0.0, 0.0, 0.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_DOUBLE_EQ(t, 0.0);
}

// ===========================================================================
// Degenerate case tests for cubic CCD
// ===========================================================================

// Coplanar entire step: point slides in the triangle plane, cubic coefficients all zero.
// The coplanar fallback (projected 2D) should detect the crossing.
TEST(GeneralCCDNodeTriangle, CoplanarPointSlidesIntoTriangle) {
    const Vec3 x(-0.5, 0.25, 0.0),  dx(2.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_GT(t, 0.0);
    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.25, 1e-8);
}

// Coplanar, point already inside at t=0.
TEST(GeneralCCDNodeTriangle, CoplanarAlreadyInside) {
    const Vec3 x(0.25, 0.25, 0.0),  dx(0.1, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_DOUBLE_EQ(t, 0.0);
}

// Coplanar but point moves parallel to triangle edge, never enters.
TEST(GeneralCCDNodeTriangle, CoplanarNoEntry) {
    const Vec3 x(2.0, 2.0, 0.0),  dx(1.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_DOUBLE_EQ(t, 1.0);
}

// Cubic degenerates to quadratic: triangle stays fixed, only the point moves.
// The cubic leading coefficient a = 0 because dp, dq are zero.
TEST(GeneralCCDNodeTriangle, CubicDegenToQuadratic) {
    const Vec3 x(0.25, 0.25, 0.5),  dx(0.0, 0.0, -1.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_NEAR(t, 0.5, kTol);
}

// Cubic degenerates to linear: point is stationary, one triangle vertex rises.
// Coplanarity polynomial: -0.5t + 0.25 = 0 => t = 0.5.
TEST(GeneralCCDNodeTriangle, CubicDegenToLinear) {
    const Vec3 x(0.25, 0.25, 0.25),  dx(0.0, 0.0, 0.0);
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0),  dx3(0.0, 0.0, 2.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_NEAR(t, 0.5, 1e-6);
}

// Collinear segments: both segments lie on the x-axis and slide into each other.
TEST(GeneralCCDSegmentSegment, CollinearOverlap) {
    const Vec3 x1(0.0, 0.0, 0.0),  dx1(1.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0),  dx2(1.0, 0.0, 0.0);
    const Vec3 x3(3.0, 0.0, 0.0),  dx3(-2.0, 0.0, 0.0);
    const Vec3 x4(4.0, 0.0, 0.0),  dx4(-2.0, 0.0, 0.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_GT(t, 0.0);
    EXPECT_LT(t, 1.0);
}

// Coplanar segments: edge 1 (vertical) sweeps right through fixed edge 2 (horizontal).
// Both in xy-plane. Edge 1 crosses edge 2 at t=0.5 when it reaches x=0.
TEST(GeneralCCDSegmentSegment, CoplanarSweptIntersection) {
    const Vec3 x1(-2.0, -0.5, 0.0), dx1(3.0, 0.0, 0.0);
    const Vec3 x2(-2.0,  0.5, 0.0), dx2(3.0, 0.0, 0.0);
    const Vec3 x3(-0.5,  0.0, 0.0), dx3(0.0, 0.0, 0.0);
    const Vec3 x4( 0.5,  0.0, 0.0), dx4(0.0, 0.0, 0.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_GT(t, 0.0);
    EXPECT_LT(t, 1.0);
}

// Cubic degenerates to quadratic: edge 1 drops uniformly, edge 2 contracts.
// Quadratic 0.5t^2 - 1.25t + 0.5 = 0, roots at t=0.5 and t=2.
TEST(GeneralCCDSegmentSegment, CubicDegenToQuadratic) {
    const Vec3 x1(0.0, 0.0, 0.5),   dx1(0.0, 0.0, -1.0);
    const Vec3 x2(0.0, 1.0, 0.5),   dx2(0.0, 0.0, -1.0);
    const Vec3 x3(-0.5, 0.5, 0.0),  dx3(0.25, 0.0, 0.0);
    const Vec3 x4( 0.5, 0.5, 0.0),  dx4(-0.25, 0.0, 0.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_NEAR(t, 0.5, 1e-6);
}

// Differential parity case against /Users/junhuishen/Downloads/ccd.cpp:
// reference returns t ~= 0.0376142, current implementation returns 1.0.
TEST(GeneralCCDParityWithDownloads, NodeTriangleMissedCollisionCase) {
    const Vec3 x(0.508426, -0.200533, -0.22161);
    const Vec3 dx(-0.495916, -0.843144, -0.63159);
    const Vec3 x1(0.993697, -0.700432, -0.471773);
    const Vec3 dx1(0.744377, 0.528337, -0.184252);
    const Vec3 x2(-0.372305, 0.541969, -0.606791);
    const Vec3 dx2(0.780543, 0.320786, -0.230345);
    const Vec3 x3(-0.5384, 0.499129, 0.861156);
    const Vec3 dx3(0.948062, 0.774797, -0.254482);

    const double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_NEAR(t, 0.0376142, 1e-6);
}

// Differential parity case against /Users/junhuishen/Downloads/ccd.cpp:
// reference returns t ~= 0.164204, current implementation returns 1.0.
TEST(GeneralCCDParityWithDownloads, SegmentSegmentMissedCollisionCase) {
    const Vec3 x1(0.823549, -0.305334, -0.385973);
    const Vec3 dx1(0.864594, -0.403605, 0.581959);
    const Vec3 x2(-0.24371, 0.307965, 0.258667);
    const Vec3 dx2(0.656026, -0.103142, -0.219862);
    const Vec3 x3(-0.282493, -0.855775, -0.02296);
    const Vec3 dx3(-0.9167, -0.266643, -0.61211);
    const Vec3 x4(0.569401, 0.523534, 0.153763);
    const Vec3 dx4(-0.210229, -0.287373, -0.215043);

    const double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_NEAR(t, 0.164204, 1e-6);
}

// ===========================================================================
// Numerical stress tests
// ===========================================================================

// Small but not tiny displacement: dx ~ 1e-4. Verifies CCD works at
// moderate scales. (At dx ~ 1e-8, the containment check cannot confirm
// coplanarity within eps2=1e-10, which is a known precision limit.)
TEST(CCDStress, SmallDisplacementNodeTriangle) {
    const Vec3 x(0.25, 0.25, 0.01);
    const Vec3 dx(0.0, 0.0, -0.02);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0), dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0), dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.5, 1e-4);
}

TEST(CCDStress, SmallDisplacementSegmentSegment) {
    const Vec3 x1(0.0, 0.0, 0.01),  dx1(0.0, 0.0, -0.02);
    const Vec3 x2(1.0, 0.0, 0.0),   dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0),  dx3(0.0, 0.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0),  dx4(0.0, 0.0, 0.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.5, 1e-4);
}

// Documents the precision floor: with dx ~ 1e-8, the geometric containment
// check (eps2=1e-10) cannot confirm coplanarity, so CCD returns 1.0.
// This is expected behavior, not a bug.
TEST(CCDStress, VerySmallDisplacementIsUndetectable) {
    const Vec3 x(0.25, 0.25, 0.01);
    const Vec3 dx(0.0, 0.0, -2e-8);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0), dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0), dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    // At this scale, CCD cannot confirm the collision. This documents the limit.
    EXPECT_DOUBLE_EQ(t, 1.0);
}

// ===========================================================================
// ccd_initial_guess tests
// ===========================================================================

TEST(CCDInitialGuess, CollisionFreeAndAdvancedTowardXhat) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params;
    params.fps = 30.0;
    params.substeps = 1;
    params.mu = 10.0;
    params.lambda = 10.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.d_hat = 0.01;

    clear_model(ref_mesh, state, X, pins);

    // Two sheets separated along y, pushed toward each other
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 2.0, 0.0));

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_lumped_mass(params.density, params.thickness);

    // xhat: push sheets toward each other along y
    std::vector<Vec3> xhat = state.deformed_positions;
    const int nv = static_cast<int>(xhat.size());
    const int half = nv / 2;
    for (int i = 0; i < half; ++i) xhat[i].y() += 1.5;
    for (int i = half; i < nv; ++i) xhat[i].y() -= 1.5;

    std::vector<Vec3> xnew = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh);

    ASSERT_EQ(xnew.size(), state.deformed_positions.size());

    // xnew should be different from x (advanced toward xhat)
    double total_displacement = 0.0;
    for (int i = 0; i < nv; ++i)
        total_displacement += (xnew[i] - state.deformed_positions[i]).norm();
    EXPECT_GT(total_displacement, 0.0) << "initial guess should advance toward xhat";

    // xnew should NOT have reached xhat fully (collision limits the step)
    double total_remaining = 0.0;
    for (int i = 0; i < nv; ++i)
        total_remaining += (xnew[i] - xhat[i]).norm();
    EXPECT_GT(total_remaining, 0.0) << "initial guess should be limited by CCD";

    // All positions should be finite
    for (int i = 0; i < nv; ++i) {
        EXPECT_TRUE(std::isfinite(xnew[i].x())) << "vertex " << i;
        EXPECT_TRUE(std::isfinite(xnew[i].y())) << "vertex " << i;
        EXPECT_TRUE(std::isfinite(xnew[i].z())) << "vertex " << i;
    }
}

TEST(CCDInitialGuess, NoCollisionTakesFullStep) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params;
    params.fps = 30.0;
    params.substeps = 1;
    params.mu = 10.0;
    params.lambda = 10.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.d_hat = 0.01;

    clear_model(ref_mesh, state, X, pins);

    // Single sheet, no collision possible
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_lumped_mass(params.density, params.thickness);

    // xhat: small displacement, no collision
    std::vector<Vec3> xhat = state.deformed_positions;
    for (auto& v : xhat) v += Vec3(0.01, 0.0, 0.0);

    std::vector<Vec3> xnew = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh);

    // Should reach xhat exactly (omega = 1.0, no collision)
    for (int i = 0; i < static_cast<int>(xnew.size()); ++i) {
        EXPECT_NEAR(xnew[i].x(), xhat[i].x(), 1e-12) << "vertex " << i;
        EXPECT_NEAR(xnew[i].y(), xhat[i].y(), 1e-12) << "vertex " << i;
        EXPECT_NEAR(xnew[i].z(), xhat[i].z(), 1e-12) << "vertex " << i;
    }
}

#ifdef _OPENMP
#include <omp.h>
TEST(CCDInitialGuess, ParallelMatchesSerial) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams params;
    params.fps = 30.0;
    params.substeps = 1;
    params.mu = 10.0;
    params.lambda = 10.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.d_hat = 0.01;

    clear_model(ref_mesh, state, X, pins);

    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 2.0, 0.0));

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_lumped_mass(params.density, params.thickness);

    std::vector<Vec3> xhat = state.deformed_positions;
    const int nv = static_cast<int>(xhat.size());
    const int half = nv / 2;
    for (int i = 0; i < half; ++i) xhat[i].y() += 1.5;
    for (int i = half; i < nv; ++i) xhat[i].y() -= 1.5;

    omp_set_num_threads(1);
    std::vector<Vec3> x_serial = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh);

    omp_set_num_threads(4);
    std::vector<Vec3> x_parallel = ccd_initial_guess(state.deformed_positions, xhat, ref_mesh);

    ASSERT_EQ(x_serial.size(), x_parallel.size());
    for (int i = 0; i < static_cast<int>(x_serial.size()); ++i) {
        EXPECT_NEAR(x_serial[i].x(), x_parallel[i].x(), 1e-12) << "vertex " << i;
        EXPECT_NEAR(x_serial[i].y(), x_parallel[i].y(), 1e-12) << "vertex " << i;
        EXPECT_NEAR(x_serial[i].z(), x_parallel[i].z(), 1e-12) << "vertex " << i;
    }
}
#endif

// Large coordinates: same collision geometry but shifted by 1e6.
// Cross products suffer cancellation with large offsets.
TEST(CCDStress, LargeCoordinateNodeTriangle) {
    const double offset = 1e6;
    const Vec3 x(offset + 0.25, offset + 0.25, offset + 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(offset, offset, offset), dx1(0.0, 0.0, 0.0);
    const Vec3 x2(offset + 1.0, offset, offset), dx2(0.0, 0.0, 0.0);
    const Vec3 x3(offset, offset + 1.0, offset), dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.5, 1e-4);
}

TEST(CCDStress, LargeCoordinateSegmentSegment) {
    const double offset = 1e6;
    const Vec3 x1(offset, offset, offset + 1.0),        dx1(0.0, 0.0, -2.0);
    const Vec3 x2(offset + 1.0, offset, offset),        dx2(0.0, 0.0, 0.0);
    const Vec3 x3(offset + 0.5, offset - 1.0, offset),  dx3(0.0, 0.0, 0.0);
    const Vec3 x4(offset + 0.5, offset + 1.0, offset),  dx4(0.0, 0.0, 0.0);

    double t = segment_segment_general_ccd(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.5, 1e-4);
}

// Near-grazing: point trajectory barely clips the triangle corner.
// Tests that the containment check doesn't reject valid edge-case hits.
TEST(CCDStress, NearGrazingNodeTriangle) {
    const Vec3 x(1e-6, 1e-6, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0), dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0), dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_LT(t, 1.0);
    EXPECT_NEAR(t, 0.5, 1e-4);
}

// Near-miss: point trajectory just barely outside the triangle.
// Must return 1.0 (no collision), not a false positive.
TEST(CCDStress, NearMissNodeTriangle) {
    const Vec3 x(-1e-4, -1e-4, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0), dx1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0), dx2(0.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0), dx3(0.0, 0.0, 0.0);

    double t = node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3);
    EXPECT_DOUBLE_EQ(t, 1.0);
}
