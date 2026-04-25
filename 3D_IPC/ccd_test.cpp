#include "ccd.h"
#include "solver.h"
#include "make_shape.h"
#include "segment_segment_distance.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>

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

TEST(GeneralCCDSegmentSegment, RepoExample4Frame9Sub6SweepBoundaryEdge) {
    // Example-4 predictor sweep where the boundary edge primitive must clip
    // the step even though the neighboring node-triangle primitive is outside.
    const Vec3 x3588(-0.059539791821382186, 0.1540607437288348, 0.35529553899914945);
    const Vec3 dx3588(0.004847380237543629, 0.0043465291684171292, -2.5049316649261577e-05);
    const Vec3 x3589(-0.091034795625455009, 0.18520815040391844, 0.3532480684510913);
    const Vec3 dx3589(0.00013025060027307966, 0.00040432496563547193, -0.00010336345303363359);

    const Vec3 x3117(-0.075654645452310015, 0.17434507029226337, 0.34200013654345546);
    const Vec3 dx3117(9.2102319146358802e-08, -8.8077176663148293e-08, 1.2465886145562877e-07);
    const Vec3 x3148(-0.082705931560382612, 0.18212494614593469, 0.3600000809332693);
    const Vec3 dx3148(-2.5397006150895685e-08, 3.2212561784650262e-08, 4.4863568915420871e-08);
    const Vec3 x3149(-0.067874983350554322, 0.16729401010559197, 0.36000022549626293);
    const Vec3 dx3149(2.1052045717784296e-08, -1.4851148838479489e-08, 2.2516221354074162e-07);

    const std::array<double, 3> toi = {
        segment_segment_general_ccd(x3117, dx3117, x3148, dx3148, x3588, dx3588, x3589, dx3589),
        segment_segment_general_ccd(x3117, dx3117, x3149, dx3149, x3588, dx3588, x3589, dx3589),
        segment_segment_general_ccd(x3148, dx3148, x3149, dx3149, x3588, dx3588, x3589, dx3589),
    };

    EXPECT_DOUBLE_EQ(toi[0], 1.0);
    EXPECT_NEAR(toi[1], 0.86764166245631813, 1e-9);
    EXPECT_DOUBLE_EQ(toi[2], 1.0);
}

TEST(GeneralCCDNodeTriangle, RepoExample4Frame9Sub6SweepNodeTriangleOutside) {
    // Same captured sweep. The v3588 plane crossing is outside this cylinder
    // triangle for the predictor motion, so the boundary SS primitive above is
    // the decisive contact.
    const Vec3 x3588(-0.059539791821382186, 0.1540607437288348, 0.35529553899914945);
    const Vec3 dx3588(0.004847380237543629, 0.0043465291684171292, -2.5049316649261577e-05);
    const Vec3 x3117(-0.075654645452310015, 0.17434507029226337, 0.34200013654345546);
    const Vec3 dx3117(9.2102319146358802e-08, -8.8077176663148293e-08, 1.2465886145562877e-07);
    const Vec3 x3148(-0.082705931560382612, 0.18212494614593469, 0.3600000809332693);
    const Vec3 dx3148(-2.5397006150895685e-08, 3.2212561784650262e-08, 4.4863568915420871e-08);
    const Vec3 x3149(-0.067874983350554322, 0.16729401010559197, 0.36000022549626293);
    const Vec3 dx3149(2.1052045717784296e-08, -1.4851148838479489e-08, 2.2516221354074162e-07);

    const double toi = node_triangle_general_ccd(
        x3588, dx3588, x3117, dx3117, x3149, dx3149, x3148, dx3148);
    EXPECT_DOUBLE_EQ(toi, 1.0);
}

TEST(GeneralCCDClothCloth, RepoExample4Frame9Sub8FirstCrossing) {
    // Cloth-cloth example-4 sweep whose final edge-triangle crossing must be
    // represented by at least one legal VF/EE CCD event.
    const Vec3 x3346(0.08712833976681923, 0.17877421220858070, -0.32661607072106447);
    const Vec3 dx3346(0.00067734780013454, -0.00080126570631464, -0.00031187147184419);
    const Vec3 x3328(0.10968703434917194, 0.21990170027158606, -0.36743254758293364);
    const Vec3 dx3328(-0.00021517113302960, -0.00021279894679258, -0.00040805071333450);
    const Vec3 x3618(0.09703499570429237, 0.18800609993799147, -0.35201048636447191);
    const Vec3 dx3618(-0.00030715771853801, 0.00072942951647373, 0.00003814747329084);
    const Vec3 x3636(0.06721104360563963, 0.15255331981292530, -0.31062147477971258);
    const Vec3 dx3636(-0.00332165259637236, 0.00319812532310226, -0.00010903449705529);
    const Vec3 x3635(0.09601130899431572, 0.18606972009767381, -0.30860063125557996);
    const Vec3 dx3635(-0.00050502730280225, 0.00073262713672059, -0.00000312696759880);

    const std::array<double, 5> toi = {
        node_triangle_general_ccd(x3346, dx3346, x3618, dx3618, x3636, dx3636, x3635, dx3635),
        node_triangle_general_ccd(x3328, dx3328, x3618, dx3618, x3636, dx3636, x3635, dx3635),
        segment_segment_general_ccd(x3346, dx3346, x3328, dx3328, x3618, dx3618, x3636, dx3636),
        segment_segment_general_ccd(x3346, dx3346, x3328, dx3328, x3636, dx3636, x3635, dx3635),
        segment_segment_general_ccd(x3346, dx3346, x3328, dx3328, x3635, dx3635, x3618, dx3618),
    };

    const double toi_min = *std::min_element(toi.begin(), toi.end());
    EXPECT_LT(toi_min, 1.0);
}

TEST(GeneralCCDClothCloth, RepoExample4TenStackFrame6To7FirstCrossing) {
    // Coarser saved-frame sweep from the dense-stack example. At least one
    // legal VF/EE event must appear over the full-frame motion.
    const Vec3 x3341(0.0437676, 0.40053, -0.306259);
    const Vec3 x3358(0.0437671, 0.40053, -0.262509);
    const Vec3 x3613(0.0437701, 0.40653, -0.350001);
    const Vec3 x3631(0.0875185, 0.40653, -0.30625);
    const Vec3 x3630(0.0437685, 0.40653, -0.306251);

    const Vec3 y3341(0.0449519, 0.351985, -0.307173);
    const Vec3 y3358(0.0447381, 0.352262, -0.263718);
    const Vec3 y3613(0.0431897, 0.352311, -0.347623);
    const Vec3 y3631(0.0860174, 0.340867, -0.304771);
    const Vec3 y3630(0.0431711, 0.352483, -0.304094);

    const Vec3 dx3341 = y3341 - x3341;
    const Vec3 dx3358 = y3358 - x3358;
    const Vec3 dx3613 = y3613 - x3613;
    const Vec3 dx3631 = y3631 - x3631;
    const Vec3 dx3630 = y3630 - x3630;

    const std::array<double, 5> toi = {
        node_triangle_general_ccd(x3341, dx3341, x3613, dx3613, x3631, dx3631, x3630, dx3630),
        node_triangle_general_ccd(x3358, dx3358, x3613, dx3613, x3631, dx3631, x3630, dx3630),
        segment_segment_general_ccd(x3341, dx3341, x3358, dx3358, x3613, dx3613, x3631, dx3631),
        segment_segment_general_ccd(x3341, dx3341, x3358, dx3358, x3631, dx3631, x3630, dx3630),
        segment_segment_general_ccd(x3341, dx3341, x3358, dx3358, x3630, dx3630, x3613, dx3613),
    };

    const double toi_min = *std::min_element(toi.begin(), toi.end());
    EXPECT_LT(toi_min, 1.0);
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
    SimParams params = SimParams::zeros();
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
    SimParams params = SimParams::zeros();
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
    SimParams params = SimParams::zeros();
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
