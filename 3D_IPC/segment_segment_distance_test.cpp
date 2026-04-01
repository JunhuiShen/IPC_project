#include "segment_segment_distance.h"

#include <cmath>
#include <cstdlib>
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

    void require(bool cond, const std::string& msg){
        if (!cond) {
            std::cerr << "TEST FAILED: " << msg << std::endl;
            std::exit(EXIT_FAILURE);
        }
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

    // ------------------------------------------------------------------
    // Case 1: Interior
    // Two skew segments whose closest points lie strictly inside both.
    // Segment 1: (0,0,0)--(1,0,0)  along the x-axis
    // Segment 2: (0.5,-1,1)--(0.5,1,1)  along the y-axis, offset by z=1
    // Closest pair: (0.5, 0, 0) and (0.5, 0, 1), distance = 1
    // ------------------------------------------------------------------
    void test_interior_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(0.5, -1.0, 1.0);
        const Vec3 x4(0.5, 1.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Interior case", r);

        require(r.region == SegmentSegmentRegion::Interior, "Interior case: wrong region");
        require(approx(r.distance, 1.0), "Interior case: wrong distance");
        require(approx(r.s, 0.5), "Interior case: wrong s");
        require(approx(r.t, 0.5), "Interior case: wrong t");
        require(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0)), "Interior case: wrong closest_1");
        require(approx_vec(r.closest_point_2, Vec3(0.5, 0.0, 1.0)), "Interior case: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Case 2: Edge s=0 (point x1 vs segment (x3,x4))
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (-1,-1,1)--(-1,1,1)
    // The closest point on seg1 is x1=(0,0,0), on seg2 it is (-1,0,1).
    // distance = sqrt(1 + 0 + 1) = sqrt(2)
    // ------------------------------------------------------------------
    void test_edge_s0_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(-1.0, -1.0, 1.0);
        const Vec3 x4(-1.0, 1.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Edge s=0 case", r);

        require(r.region == SegmentSegmentRegion::Edge_s0, "Edge s=0: wrong region");
        require(approx(r.s, 0.0), "Edge s=0: wrong s");
        require(approx(r.t, 0.5), "Edge s=0: wrong t");
        require(approx(r.distance, std::sqrt(2.0)), "Edge s=0: wrong distance");
        require(approx_vec(r.closest_point_1, Vec3(0.0, 0.0, 0.0)), "Edge s=0: wrong closest_1");
        require(approx_vec(r.closest_point_2, Vec3(-1.0, 0.0, 1.0)), "Edge s=0: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Case 3: Edge s=1 (point x2 vs segment (x3,x4))
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (2,-1,1)--(2,1,1)
    // Closest: x2=(1,0,0) to (2,0,1), distance = sqrt(2)
    // ------------------------------------------------------------------
    void test_edge_s1_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(2.0, -1.0, 1.0);
        const Vec3 x4(2.0, 1.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Edge s=1 case", r);

        require(r.region == SegmentSegmentRegion::Edge_s1, "Edge s=1: wrong region");
        require(approx(r.s, 1.0), "Edge s=1: wrong s");
        require(approx(r.t, 0.5), "Edge s=1: wrong t");
        require(approx(r.distance, std::sqrt(2.0)), "Edge s=1: wrong distance");
        require(approx_vec(r.closest_point_1, Vec3(1.0, 0.0, 0.0)), "Edge s=1: wrong closest_1");
        require(approx_vec(r.closest_point_2, Vec3(2.0, 0.0, 1.0)), "Edge s=1: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Case 4: Edge t=0 (point x3 vs segment (x1,x2))
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (0.5,1,1)--(0.5,2,1)
    // Closest: (0.5,0,0) to x3=(0.5,1,1), distance = sqrt(0+1+1) = sqrt(2)
    // ------------------------------------------------------------------
    void test_edge_t0_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(0.5, 1.0, 1.0);
        const Vec3 x4(0.5, 2.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Edge t=0 case", r);

        require(r.region == SegmentSegmentRegion::Edge_t0, "Edge t=0: wrong region");
        require(approx(r.s, 0.5), "Edge t=0: wrong s");
        require(approx(r.t, 0.0), "Edge t=0: wrong t");
        require(approx(r.distance, std::sqrt(2.0)), "Edge t=0: wrong distance");
        require(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0)), "Edge t=0: wrong closest_1");
        require(approx_vec(r.closest_point_2, Vec3(0.5, 1.0, 1.0)), "Edge t=0: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Case 5: Edge t=1 (point x4 vs segment (x1,x2))
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (0.5,-2,1)--(0.5,-1,1)
    // Closest: (0.5,0,0) to x4=(0.5,-1,1), distance = sqrt(0+1+1) = sqrt(2)
    // ------------------------------------------------------------------
    void test_edge_t1_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(0.5, -2.0, 1.0);
        const Vec3 x4(0.5, -1.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Edge t=1 case", r);

        require(r.region == SegmentSegmentRegion::Edge_t1, "Edge t=1: wrong region");
        require(approx(r.s, 0.5), "Edge t=1: wrong s");
        require(approx(r.t, 1.0), "Edge t=1: wrong t");
        require(approx(r.distance, std::sqrt(2.0)), "Edge t=1: wrong distance");
        require(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0)), "Edge t=1: wrong closest_1");
        require(approx_vec(r.closest_point_2, Vec3(0.5, -1.0, 1.0)), "Edge t=1: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Case 6: Corner (s,t)=(0,0), vertex x1 vs vertex x3
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (-1,-1,1)--(-1,-2,1)
    // Closest: x1=(0,0,0) to x3=(-1,-1,1), distance = sqrt(3)
    // ------------------------------------------------------------------
    void test_corner_s0t0_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(-1.0, -1.0, 1.0);
        const Vec3 x4(-1.0, -2.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Corner s=0,t=0 case", r);

        require(r.region == SegmentSegmentRegion::Corner_s0t0, "Corner s0t0: wrong region");
        require(approx(r.s, 0.0), "Corner s0t0: wrong s");
        require(approx(r.t, 0.0), "Corner s0t0: wrong t");
        require(approx(r.distance, std::sqrt(3.0)), "Corner s0t0: wrong distance");
        require(approx_vec(r.closest_point_1, x1), "Corner s0t0: wrong closest_1");
        require(approx_vec(r.closest_point_2, x3), "Corner s0t0: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Case 7: Corner (s,t)=(0,1), vertex x1 vs vertex x4
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (-2,-2,1)--(-1,-1,1)
    // Closest: x1=(0,0,0) to x4=(-1,-1,1), distance = sqrt(3)
    // ------------------------------------------------------------------
    void test_corner_s0t1_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(-2.0, -2.0, 1.0);
        const Vec3 x4(-1.0, -1.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Corner s=0,t=1 case", r);

        require(r.region == SegmentSegmentRegion::Corner_s0t1, "Corner s0t1: wrong region");
        require(approx(r.s, 0.0), "Corner s0t1: wrong s");
        require(approx(r.t, 1.0), "Corner s0t1: wrong t");
        require(approx(r.distance, std::sqrt(3.0)), "Corner s0t1: wrong distance");
        require(approx_vec(r.closest_point_1, x1), "Corner s0t1: wrong closest_1");
        require(approx_vec(r.closest_point_2, x4), "Corner s0t1: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Case 8: Corner (s,t)=(1,0), vertex x2 vs vertex x3
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (2,-1,1)--(2,-2,1)
    // Closest: x2=(1,0,0) to x3=(2,-1,1), distance = sqrt(3)
    // ------------------------------------------------------------------
    void test_corner_s1t0_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(2.0, -1.0, 1.0);
        const Vec3 x4(2.0, -2.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Corner s=1,t=0 case", r);

        require(r.region == SegmentSegmentRegion::Corner_s1t0, "Corner s1t0: wrong region");
        require(approx(r.s, 1.0), "Corner s1t0: wrong s");
        require(approx(r.t, 0.0), "Corner s1t0: wrong t");
        require(approx(r.distance, std::sqrt(3.0)), "Corner s1t0: wrong distance");
        require(approx_vec(r.closest_point_1, x2), "Corner s1t0: wrong closest_1");
        require(approx_vec(r.closest_point_2, x3), "Corner s1t0: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Case 9: Corner (s,t)=(1,1), vertex x2 vs vertex x4
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (3,-2,1)--(2,-1,1)
    // Closest: x2=(1,0,0) to x4=(2,-1,1), distance = sqrt(3)
    // ------------------------------------------------------------------
    void test_corner_s1t1_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(3.0, -2.0, 1.0);
        const Vec3 x4(2.0, -1.0, 1.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Corner s=1,t=1 case", r);

        require(r.region == SegmentSegmentRegion::Corner_s1t1, "Corner s1t1: wrong region");
        require(approx(r.s, 1.0), "Corner s1t1: wrong s");
        require(approx(r.t, 1.0), "Corner s1t1: wrong t");
        require(approx(r.distance, std::sqrt(3.0)), "Corner s1t1: wrong distance");
        require(approx_vec(r.closest_point_1, x2), "Corner s1t1: wrong closest_1");
        require(approx_vec(r.closest_point_2, x4), "Corner s1t1: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Parallel segments (Delta ~ 0)
    // Segment 1: (0,0,0)--(2,0,0)
    // Segment 2: (0.5,1,0)--(1.5,1,0)  same direction, offset by y=1
    // Closest pair is anywhere in the overlapping region; distance = 1
    // ------------------------------------------------------------------
    void test_parallel_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(2.0, 0.0, 0.0);
        const Vec3 x3(0.5, 1.0, 0.0);
        const Vec3 x4(1.5, 1.0, 0.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Parallel case", r);

        require(r.region == SegmentSegmentRegion::ParallelSegments, "Parallel case: wrong region");
        require(approx(r.distance, 1.0), "Parallel case: wrong distance");
    }

    // ------------------------------------------------------------------
    // Parallel non-overlapping segments
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (3,1,0)--(4,1,0)
    // Closest: x2=(1,0,0) to x3=(3,1,0), distance = sqrt(4+1) = sqrt(5)
    // ------------------------------------------------------------------
    void test_parallel_non_overlapping_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(3.0, 1.0, 0.0);
        const Vec3 x4(4.0, 1.0, 0.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Parallel non-overlapping case", r);

        require(approx(r.distance, std::sqrt(5.0)), "Parallel non-overlap: wrong distance");
        require(approx_vec(r.closest_point_1, x2), "Parallel non-overlap: wrong closest_1");
        require(approx_vec(r.closest_point_2, x3), "Parallel non-overlap: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Degenerate segment (zero-length segment 1)
    // Segment 1: (0.5, 0, 0)--(0.5, 0, 0) (a point)
    // Segment 2: (0, 1, 0)--(1, 1, 0)
    // Closest: (0.5, 0, 0) to (0.5, 1, 0), distance = 1
    // ------------------------------------------------------------------
    void test_degenerate_segment_case(){
        const Vec3 x1(0.5, 0.0, 0.0);
        const Vec3 x2(0.5, 0.0, 0.0);
        const Vec3 x3(0.0, 1.0, 0.0);
        const Vec3 x4(1.0, 1.0, 0.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Degenerate segment case", r);

        require(approx(r.distance, 1.0), "Degenerate segment: wrong distance");
        require(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0)), "Degenerate segment: wrong closest_1");
        require(approx_vec(r.closest_point_2, Vec3(0.5, 1.0, 0.0)), "Degenerate segment: wrong closest_2");
    }

    // ------------------------------------------------------------------
    // Symmetry test: swapping the two segments should give the same distance
    // ------------------------------------------------------------------
    void test_symmetry(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(0.3, -0.5, 0.8);
        const Vec3 x4(0.7, 1.2, 0.8);

        const auto r1 = segment_segment_distance(x1, x2, x3, x4);
        const auto r2 = segment_segment_distance(x3, x4, x1, x2);

        print_result("Symmetry test (order 1)", r1);
        print_result("Symmetry test (order 2)", r2);

        require(approx(r1.distance, r2.distance), "Symmetry: distances differ");
        require(approx_vec(r1.closest_point_1, r2.closest_point_2), "Symmetry: closest points swapped");
        require(approx_vec(r1.closest_point_2, r2.closest_point_1), "Symmetry: closest points swapped");
    }

    // ------------------------------------------------------------------
    // Touching segments (distance = 0)
    // Segment 1: (0,0,0)--(1,0,0)
    // Segment 2: (0.5,0,0)--(0.5,1,0)
    // They share the point (0.5, 0, 0), distance = 0
    // ------------------------------------------------------------------
    void test_touching_case(){
        const Vec3 x1(0.0, 0.0, 0.0);
        const Vec3 x2(1.0, 0.0, 0.0);
        const Vec3 x3(0.5, 0.0, 0.0);
        const Vec3 x4(0.5, 1.0, 0.0);

        const auto r = segment_segment_distance(x1, x2, x3, x4);
        print_result("Touching case", r);

        require(approx(r.distance, 0.0), "Touching: wrong distance");
        require(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0)), "Touching: wrong closest_1");
        require(approx_vec(r.closest_point_2, Vec3(0.5, 0.0, 0.0)), "Touching: wrong closest_2");
    }

} // namespace

int main(){
    test_interior_case();
    test_edge_s0_case();
    test_edge_s1_case();
    test_edge_t0_case();
    test_edge_t1_case();
    test_corner_s0t0_case();
    test_corner_s0t1_case();
    test_corner_s1t0_case();
    test_corner_s1t1_case();
    test_parallel_case();
    test_parallel_non_overlapping_case();
    test_degenerate_segment_case();
    test_symmetry();
    test_touching_case();

    std::cout << "\nAll segment-segment distance tests passed.\n";
    return 0;
}
