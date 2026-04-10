#include "segment_segment_distance.h"

#include <gtest/gtest.h>
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
