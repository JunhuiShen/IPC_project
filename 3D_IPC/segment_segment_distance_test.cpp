#include "segment_segment_distance.h"

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace {

    constexpr double kTol = 1e-10;

    bool approx(double a, double b, double tol = kTol){
        return std::abs(a - b) <= tol;
    }

    bool approx_vec(const Vec3& a, const Vec3& b, double tol = kTol){
        return (a - b).norm() <= tol;
    }

    void print_result(const std::string& name, const SegmentSegmentDistanceResult& r){
        std::cout << name << "\n";
        std::cout << "  region     = " << to_string(r.region) << "\n";
        std::cout << "  s          = " << r.s << "\n";
        std::cout << "  t          = " << r.t << "\n";
        std::cout << "  distance   = " << r.distance << "\n";
        std::cout << "  closest_1  = " << r.closest_point_1.transpose() << "\n";
        std::cout << "  closest_2  = " << r.closest_point_2.transpose() << "\n";
    }

} // namespace

TEST(SegmentSegmentDistance, Interior){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 1.0);
    const Vec3 x4(0.5, 1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Interior case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Interior) << "Interior case: wrong region";
    EXPECT_NEAR(r.distance, 1.0, kTol) << "Interior case: wrong distance";
    EXPECT_NEAR(r.s, 0.5, kTol) << "Interior case: wrong s";
    EXPECT_NEAR(r.t, 0.5, kTol) << "Interior case: wrong t";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Interior case: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, 0.0, 1.0))) << "Interior case: wrong closest_2";
}

TEST(SegmentSegmentDistance, EdgeS0){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(-1.0, -1.0, 1.0);
    const Vec3 x4(-1.0, 1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Edge s=0 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Edge_s0) << "Edge s=0: wrong region";
    EXPECT_NEAR(r.s, 0.0, kTol) << "Edge s=0: wrong s";
    EXPECT_NEAR(r.t, 0.5, kTol) << "Edge s=0: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(2.0), kTol) << "Edge s=0: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.0, 0.0, 0.0))) << "Edge s=0: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(-1.0, 0.0, 1.0))) << "Edge s=0: wrong closest_2";
}

TEST(SegmentSegmentDistance, EdgeS1){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(2.0, -1.0, 1.0);
    const Vec3 x4(2.0, 1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Edge s=1 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Edge_s1) << "Edge s=1: wrong region";
    EXPECT_NEAR(r.s, 1.0, kTol) << "Edge s=1: wrong s";
    EXPECT_NEAR(r.t, 0.5, kTol) << "Edge s=1: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(2.0), kTol) << "Edge s=1: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(1.0, 0.0, 0.0))) << "Edge s=1: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(2.0, 0.0, 1.0))) << "Edge s=1: wrong closest_2";
}

TEST(SegmentSegmentDistance, EdgeT0){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, 1.0, 1.0);
    const Vec3 x4(0.5, 2.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Edge t=0 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Edge_t0) << "Edge t=0: wrong region";
    EXPECT_NEAR(r.s, 0.5, kTol) << "Edge t=0: wrong s";
    EXPECT_NEAR(r.t, 0.0, kTol) << "Edge t=0: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(2.0), kTol) << "Edge t=0: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Edge t=0: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, 1.0, 1.0))) << "Edge t=0: wrong closest_2";
}

TEST(SegmentSegmentDistance, EdgeT1){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -2.0, 1.0);
    const Vec3 x4(0.5, -1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Edge t=1 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Edge_t1) << "Edge t=1: wrong region";
    EXPECT_NEAR(r.s, 0.5, kTol) << "Edge t=1: wrong s";
    EXPECT_NEAR(r.t, 1.0, kTol) << "Edge t=1: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(2.0), kTol) << "Edge t=1: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Edge t=1: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, -1.0, 1.0))) << "Edge t=1: wrong closest_2";
}

TEST(SegmentSegmentDistance, CornerS0T0){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(-1.0, -1.0, 1.0);
    const Vec3 x4(-1.0, -2.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Corner s=0,t=0 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Corner_s0t0) << "Corner s0t0: wrong region";
    EXPECT_NEAR(r.s, 0.0, kTol) << "Corner s0t0: wrong s";
    EXPECT_NEAR(r.t, 0.0, kTol) << "Corner s0t0: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(3.0), kTol) << "Corner s0t0: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x1)) << "Corner s0t0: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x3)) << "Corner s0t0: wrong closest_2";
}

TEST(SegmentSegmentDistance, CornerS0T1){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(-2.0, -2.0, 1.0);
    const Vec3 x4(-1.0, -1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Corner s=0,t=1 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Corner_s0t1) << "Corner s0t1: wrong region";
    EXPECT_NEAR(r.s, 0.0, kTol) << "Corner s0t1: wrong s";
    EXPECT_NEAR(r.t, 1.0, kTol) << "Corner s0t1: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(3.0), kTol) << "Corner s0t1: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x1)) << "Corner s0t1: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x4)) << "Corner s0t1: wrong closest_2";
}

TEST(SegmentSegmentDistance, CornerS1T0){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(2.0, -1.0, 1.0);
    const Vec3 x4(2.0, -2.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Corner s=1,t=0 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Corner_s1t0) << "Corner s1t0: wrong region";
    EXPECT_NEAR(r.s, 1.0, kTol) << "Corner s1t0: wrong s";
    EXPECT_NEAR(r.t, 0.0, kTol) << "Corner s1t0: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(3.0), kTol) << "Corner s1t0: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x2)) << "Corner s1t0: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x3)) << "Corner s1t0: wrong closest_2";
}

TEST(SegmentSegmentDistance, CornerS1T1){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(3.0, -2.0, 1.0);
    const Vec3 x4(2.0, -1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Corner s=1,t=1 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Corner_s1t1) << "Corner s1t1: wrong region";
    EXPECT_NEAR(r.s, 1.0, kTol) << "Corner s1t1: wrong s";
    EXPECT_NEAR(r.t, 1.0, kTol) << "Corner s1t1: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(3.0), kTol) << "Corner s1t1: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x2)) << "Corner s1t1: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x4)) << "Corner s1t1: wrong closest_2";
}

TEST(SegmentSegmentDistance, Parallel){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(2.0, 0.0, 0.0);
    const Vec3 x3(0.5, 1.0, 0.0);
    const Vec3 x4(1.5, 1.0, 0.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Parallel case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::ParallelSegments) << "Parallel case: wrong region";
    EXPECT_NEAR(r.distance, 1.0, kTol) << "Parallel case: wrong distance";
}

TEST(SegmentSegmentDistance, ParallelNonOverlapping){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(3.0, 1.0, 0.0);
    const Vec3 x4(4.0, 1.0, 0.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Parallel non-overlapping case", r);

    EXPECT_NEAR(r.distance, std::sqrt(5.0), kTol) << "Parallel non-overlap: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x2)) << "Parallel non-overlap: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x3)) << "Parallel non-overlap: wrong closest_2";
}

TEST(SegmentSegmentDistance, DegenerateSegment){
    const Vec3 x1(0.5, 0.0, 0.0);
    const Vec3 x2(0.5, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x4(1.0, 1.0, 0.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Degenerate segment case", r);

    EXPECT_NEAR(r.distance, 1.0, kTol) << "Degenerate segment: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Degenerate segment: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, 1.0, 0.0))) << "Degenerate segment: wrong closest_2";
}

TEST(SegmentSegmentDistance, Symmetry){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.3, -0.5, 0.8);
    const Vec3 x4(0.7, 1.2, 0.8);

    const auto r1 = segment_segment_distance(x1, x2, x3, x4);
    const auto r2 = segment_segment_distance(x3, x4, x1, x2);

    print_result("Symmetry test (order 1)", r1);
    print_result("Symmetry test (order 2)", r2);

    EXPECT_NEAR(r1.distance, r2.distance, kTol) << "Symmetry: distances differ";
    EXPECT_TRUE(approx_vec(r1.closest_point_1, r2.closest_point_2)) << "Symmetry: closest points swapped";
    EXPECT_TRUE(approx_vec(r1.closest_point_2, r2.closest_point_1)) << "Symmetry: closest points swapped";
}

TEST(SegmentSegmentDistance, Touching){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, 0.0, 0.0);
    const Vec3 x4(0.5, 1.0, 0.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Touching case", r);

    EXPECT_NEAR(r.distance, 0.0, kTol) << "Touching: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Touching: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, 0.0, 0.0))) << "Touching: wrong closest_2";
}

TEST(SegmentSegmentDistance, NearParallelStress){
    std::cout << "--- Near-parallel stress ---\n";
    const Vec3 x1(0,0,0), x2(1,0,0);
    const Vec3 x3(0.1, 0.0, 0.3), x4(0.9, 1e-10, 0.3);

    auto r = segment_segment_distance(x1, x2, x3, x4);
    std::cout << "  distance = " << r.distance << "  s=" << r.s << "  t=" << r.t
              << "  region=" << to_string(r.region) << "\n";
    EXPECT_TRUE(std::isfinite(r.distance)) << "near-parallel distance should be finite";
    EXPECT_GT(r.distance, 0.0) << "near-parallel distance should be positive";
    EXPECT_LT(r.distance, 0.31) << "near-parallel distance should be ~0.3";
    EXPECT_GE(r.s, 0.0) << "s should be >= 0";
    EXPECT_LE(r.s, 1.0) << "s should be <= 1";
    EXPECT_GE(r.t, 0.0) << "t should be >= 0";
    EXPECT_LE(r.t, 1.0) << "t should be <= 1";
}

TEST(SegmentSegmentDistance, LargeCoordinatesStress){
    std::cout << "--- Large coordinates stress ---\n";
    const double off = 1e6;
    const Vec3 x1(off, off, off), x2(off+1, off, off);
    const Vec3 x3(off+0.5, off-1, off+0.5), x4(off+0.5, off+1, off+0.5);

    auto r = segment_segment_distance(x1, x2, x3, x4);
    std::cout << "  distance = " << r.distance << "  region=" << to_string(r.region) << "\n";
    EXPECT_TRUE(std::isfinite(r.distance)) << "large-coord distance should be finite";
    EXPECT_EQ(r.region, SegmentSegmentRegion::Interior) << "should be interior region";
    EXPECT_NEAR(r.distance, 0.5, 1e-6) << "large-coord distance should be ~0.5";
}

TEST(SegmentSegmentDistance, VeryShortSegmentStress){
    std::cout << "--- Very short segment stress ---\n";
    const Vec3 x1(0,0,0), x2(1,0,0);
    const Vec3 x3(0.5, 0.5, 0.0), x4(0.5, 0.5, 1e-10);

    auto r = segment_segment_distance(x1, x2, x3, x4);
    std::cout << "  distance = " << r.distance << "  region=" << to_string(r.region) << "\n";
    EXPECT_TRUE(std::isfinite(r.distance)) << "short segment distance should be finite";
    EXPECT_GT(r.distance, 0.0) << "short segment distance should be positive";
}

TEST(SegmentSegmentDistance, SavedExample20ContactPreservesSubUlpSeparation) {
    // The former world-space closest-point subtraction returned exactly zero
    // for this saved contact. These reference values were evaluated with exact
    // rationals from the binary64 input coordinates, not from a distance clamp.
    const std::array<Vec3, 4> x{{
        Vec3(0.30643120172698846, 1.2537041974945615, 0.345215273017485),
        Vec3(0.2716303759805404, 1.2435156840614272, 0.3522099917587509),
        Vec3(0.3476899357030978, 1.254750762137517, 0.34482031348588066),
        Vec3(0.12785112730426101, 1.247125469633767, 0.34839148080369126)}};
    constexpr double exact_gap = 2.81637760742210603e-17;
    const Vec3 exact_separation(
        -1.96611440957398385e-19, 1.63932584825574502e-17,
        2.29002336892925505e-17);
    const auto dr = segment_segment_distance(x[0], x[1], x[2], x[3]);
    ASSERT_EQ(dr.region, SegmentSegmentRegion::Interior);
    EXPECT_TRUE(dr.robust);
    ASSERT_GT(dr.distance, 0.0);
    EXPECT_NEAR(dr.distance, exact_gap, exact_gap * 2.0e-14);
    EXPECT_NEAR(dr.s, 0.0428138387495429762, 3.0e-16);
    EXPECT_NEAR(dr.t, 0.194454706288313905, 3.0e-16);
    EXPECT_LE((dr.separation - exact_separation).norm(), exact_gap * 2.0e-14);
    const Vec3 normal = dr.separation / dr.distance;
    EXPECT_TRUE(normal.allFinite());
    EXPECT_NEAR(normal.norm(), 1.0, 2.0e-14);
    EXPECT_NEAR(normal.dot(x[1] - x[0]), 0.0, 2.0e-16);
    EXPECT_NEAR(normal.dot(x[3] - x[2]), 0.0, 2.0e-16);

    const auto swapped = segment_segment_distance(x[2], x[3], x[0], x[1]);
    const auto reversed = segment_segment_distance(x[1], x[0], x[3], x[2]);
    EXPECT_NEAR(swapped.distance, exact_gap, exact_gap * 2.0e-14);
    EXPECT_NEAR(reversed.distance, exact_gap, exact_gap * 2.0e-14);
    EXPECT_LE((swapped.separation + dr.separation).norm(), exact_gap * 2.0e-14);
    EXPECT_LE((reversed.separation - dr.separation).norm(), exact_gap * 2.0e-14);
    EXPECT_NEAR(swapped.s, dr.t, 3.0e-16);
    EXPECT_NEAR(swapped.t, dr.s, 3.0e-16);
    EXPECT_NEAR(reversed.s, 1.0 - dr.s, 3.0e-16);
    EXPECT_NEAR(reversed.t, 1.0 - dr.t, 3.0e-16);
}

TEST(SegmentSegmentDistance, NearParallelInteriorIsNotReplacedByAnEndpoint) {
    // A*C-B*B rounds to zero here, but the represented edges are not parallel.
    const Vec3 x0(0.0, 0.0, 0.0), x1(1.0, 0.0, 0.0);
    const Vec3 x2(0.0, -5.0e-9, 0.125), x3(1.0, 5.0e-9, 0.125);
    const auto dr = segment_segment_distance(x0, x1, x2, x3);
    EXPECT_TRUE(dr.robust);
    EXPECT_EQ(dr.region, SegmentSegmentRegion::Interior);
    EXPECT_NEAR(dr.s, 0.5, 1.0e-14);
    EXPECT_NEAR(dr.t, 0.5, 1.0e-14);
    EXPECT_DOUBLE_EQ(dr.distance, 0.125);
    EXPECT_LE((dr.separation - Vec3(0.0, 0.0, -0.125)).norm(), 1.0e-15);
}

TEST(SegmentSegmentDistance, DegenerateEndpointRetainsObliqueSubUlpGap) {
    const double next = std::nextafter(1.5, 2.0);
    const Vec3 point(1.5, next, 1.5);
    const auto dr = segment_segment_distance(
        point, point, Vec3(1.0, 1.0, 1.0), Vec3(2.0, 2.0, 2.0));
    const double ulp = next - 1.5;
    const Vec3 expected(-ulp / 3.0, 2.0 * ulp / 3.0, -ulp / 3.0);
    EXPECT_TRUE(dr.robust);
    ASSERT_GT(dr.distance, 0.0);
    EXPECT_NEAR(dr.distance, ulp * std::sqrt(2.0 / 3.0), ulp * 2.0e-14);
    EXPECT_LE((dr.separation - expected).norm(), ulp * 2.0e-14);
    EXPECT_NEAR(dr.t, 0.5, 3.0e-16);
}

TEST(SegmentSegmentDistance, TranslatedEndpointAndTrueContactAreNotClamped) {
    const double offset = std::ldexp(1.0, 40);
    const double gap = std::nextafter(offset, INFINITY) - offset;
    const Vec3 origin(offset, offset, offset);
    const auto endpoint = segment_segment_distance(
        origin, origin + Vec3(1.0, 0.0, 0.0),
        origin + Vec3(-1.0, gap, 0.0), origin + Vec3(0.0, gap, 0.0));
    EXPECT_DOUBLE_EQ(endpoint.distance, gap);
    EXPECT_DOUBLE_EQ(endpoint.s, 0.0);
    EXPECT_DOUBLE_EQ(endpoint.t, 1.0);
    EXPECT_LE((endpoint.separation - Vec3(0.0, -gap, 0.0)).norm(), gap * 1.0e-14);

    for (const Vec3& translation : {Vec3::Zero().eval(), origin}) {
        const auto crossing = segment_segment_distance(
            translation, translation + Vec3(1.0, 0.0, 0.0),
            translation + Vec3(0.5, -1.0, 0.0),
            translation + Vec3(0.5, 1.0, 0.0));
        EXPECT_DOUBLE_EQ(crossing.distance, 0.0);
        EXPECT_TRUE(crossing.separation.isZero(0.0));
        const auto touching = segment_segment_distance(
            translation, translation + Vec3(1.0, 0.0, 0.0),
            translation + Vec3(1.0, 0.0, 0.0),
            translation + Vec3(2.0, 1.0, 0.0));
        EXPECT_DOUBLE_EQ(touching.distance, 0.0);
        EXPECT_TRUE(touching.separation.isZero(0.0));
    }
}

TEST(SegmentSegmentDistance, RobustEndpointWeightsPrecedeParameterRounding) {
    const double near_one = std::nextafter(1.0, 0.0);
    const double deficit = 1.0 - near_one;
    const Vec3 point(near_one, 2.0, 0.0);
    const auto dr = segment_segment_distance(
        Vec3::Zero(), Vec3(1.0, 2.0, 0.0), point, point);
    ASSERT_TRUE(dr.robust);
    // Exact s is 1-deficit/5: s rounds to one, but 1-s must remain nonzero
    // when constructing the contact weights for gradient/Hessian assembly.
    EXPECT_DOUBLE_EQ(dr.s, 1.0);
    ASSERT_GT(dr.weights[0], 0.0);
    EXPECT_NEAR(dr.weights[0], deficit / 5.0, deficit * 1.0e-15);
    EXPECT_DOUBLE_EQ(dr.weights[1], 1.0);
    EXPECT_DOUBLE_EQ(dr.weights[2], -1.0);
    EXPECT_DOUBLE_EQ(dr.weights[3], 0.0);
    EXPECT_NEAR(dr.distance, deficit * 2.0 / std::sqrt(5.0), deficit * 1.0e-14);
    const Vec3 expected(deficit * 4.0 / 5.0, -deficit * 2.0 / 5.0, 0.0);
    EXPECT_LE((dr.separation - expected).norm(), deficit * 1.0e-14);
}
