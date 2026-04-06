#include "ccd.h"

#include <gtest/gtest.h>

namespace {

constexpr double kTol = 1.0e-12;

void expect_no_flags(const CCDResult& r) {
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_FALSE(r.parallel_or_no_crossing);
}

}  // namespace

TEST(CCDNodeTriangleSingleMovingNode, InteriorHit) {
    const Vec3 x(0.25, 0.25, 1.0);
    const Vec3 dx(0.0, 0.0, -2.0);
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const CCDResult r = node_triangle_only_node_moves(x, dx, x1, x2, x3);
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

    const CCDResult r = node_triangle_only_node_moves(x, dx, x1, x2, x3);
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

    const CCDResult r = node_triangle_only_node_moves(x, dx, x1, x2, x3);
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

    const CCDResult r = node_triangle_only_node_moves(x, dx, x1, x2, x3);
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

    const CCDResult r = node_triangle_only_node_moves(x, dx, x1, x2, x3);
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_TRUE(r.parallel_or_no_crossing);
}

TEST(CCDSegmentSegmentSingleMovingNode, InteriorHit) {
    const Vec3 x1(0.0, 0.0, 1.0);
    const Vec3 dx1(0.0, 0.0, -2.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 0.0);
    const Vec3 x4(0.5,  1.0, 0.0);

    const CCDResult r = segment_segment_only_x1_moves(x1, dx1, x2, x3, x4);
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

    const CCDResult r = segment_segment_only_x1_moves(x1, dx1, x2, x3, x4);
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

    const CCDResult r = segment_segment_only_x1_moves(x1, dx1, x2, x3, x4);
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

    const CCDResult r = segment_segment_only_x1_moves(x1, dx1, x2, x3, x4);
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

    const CCDResult r = segment_segment_only_x1_moves(x1, dx1, x2, x3, x4);
    EXPECT_FALSE(r.has_candidate_time);
    EXPECT_FALSE(r.collision);
    EXPECT_FALSE(r.coplanar_entire_step);
    EXPECT_TRUE(r.parallel_or_no_crossing);
}
