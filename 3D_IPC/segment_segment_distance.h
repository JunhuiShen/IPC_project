#pragma once

#include "IPC_math.h"
#include <array>
#include <string>

// Regions of the (s,t) parameter square [0,1]^2
enum class SegmentSegmentRegion{
    Interior, // 0 < s* < 1, 0 < t* < 1
    Edge_s0,  // s = 0, point x1 vs segment (x3,x4)
    Edge_s1, // s = 1, point x2 vs segment (x3,x4)
    Edge_t0,  // t = 0, point x3 vs segment (x1,x2)
    Edge_t1, // t = 1, point x4 vs segment (x1,x2)
    Corner_s0t0,  // (s,t) = (0,0), vertex x1 vs vertex x3
    Corner_s0t1, // (s,t) = (0,1), vertex x1 vs vertex x4
    Corner_s1t0, // (s,t) = (1,0), vertex x2 vs vertex x3
    Corner_s1t1, // (s,t) = (1,1), vertex x2 vs vertex x4
    ParallelSegments // segments are parallel or degenerate
};

std::string to_string(SegmentSegmentRegion region);

struct SegmentSegmentDistanceResult{
    Vec3 closest_point_1; // closest point on segment 1
    Vec3 closest_point_2; // closest point on segment 2
    double s; // parameter on segment 1: p(s) = (1-s)*x1 + s*x2
    double t; // parameter on segment 2: q(t) = (1-t)*x3 + t*x4
    double distance;
    SegmentSegmentRegion region;
    // Keep the local residual: subtracting the rounded world-space closest
    // points can erase a positive gap. This is the vector from segment 2 to 1.
    Vec3 separation = Vec3::Zero();
    bool robust = false; // exact finite-segment solve used before conversion
    // Exact selected feature, including parallel cases whose public region
    // retains the legacy ParallelSegments classification.
    SegmentSegmentRegion active_region = SegmentSegmentRegion::ParallelSegments;
    // Convert these before rounding s/t: an exact parameter just below 1 may
    // round to 1 even though the complementary endpoint weight is nonzero.
    std::array<double, 4> weights{{0.0, 0.0, 0.0, 0.0}};
};

// Disabling the exact fallback returns the original floating-point solve.
SegmentSegmentDistanceResult segment_segment_distance(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps = 1.0e-12, bool exact_computation_fallback = true);
