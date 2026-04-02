#include "segment_segment_distance.h"

#include <cmath>
#include <limits>

std::string to_string(SegmentSegmentRegion region){
    switch (region) {
        case SegmentSegmentRegion::Interior: return "Interior";
        case SegmentSegmentRegion::Edge_s0: return "Edge_s0";
        case SegmentSegmentRegion::Edge_s1: return "Edge_s1";
        case SegmentSegmentRegion::Edge_t0: return "Edge_t0";
        case SegmentSegmentRegion::Edge_t1: return "Edge_t1";
        case SegmentSegmentRegion::Corner_s0t0: return "Corner_s0t0";
        case SegmentSegmentRegion::Corner_s0t1:  return "Corner_s0t1";
        case SegmentSegmentRegion::Corner_s1t0:  return "Corner_s1t0";
        case SegmentSegmentRegion::Corner_s1t1: return "Corner_s1t1";
        case SegmentSegmentRegion::ParallelSegments: return "ParallelSegments";
        default: return "Unknown";
    }
}

// Given fixed s, find optimal t on [0,1] and return distance.
static double optimal_t_for_fixed_s(const Vec3& x1, const Vec3& a, const Vec3& x3, const Vec3& b, double s, double C, double& t_out){
    const Vec3 p = x1 + s * a;
    if (C <= 0.0) {
        t_out = 0.0;
    } else {
        t_out = clamp_scalar((p - x3).dot(b) / C, 0.0, 1.0);
    }
    const Vec3 q = x3 + t_out * b;
    return (p - q).norm();
}

// Given fixed t, find optimal s on [0,1] and return distance.
static double optimal_s_for_fixed_t(const Vec3& x1, const Vec3& a, const Vec3& x3, const Vec3& b, double t, double A, double& s_out){
    const Vec3 q = x3 + t * b;
    if (A <= 0.0) {
        s_out = 0.0;
    } else {
        s_out = clamp_scalar((q - x1).dot(a) / A, 0.0, 1.0);
    }
    const Vec3 p = x1 + s_out * a;
    return (p - q).norm();
}

// Classify an (s,t) pair into a region.
static SegmentSegmentRegion classify(double s, double t, double tol = 1e-14){
    const bool s_at_0 = (s <= tol);
    const bool s_at_1 = (s >= 1.0 - tol);
    const bool t_at_0 = (t <= tol);
    const bool t_at_1 = (t >= 1.0 - tol);

    if (s_at_0 && t_at_0) return SegmentSegmentRegion::Corner_s0t0;
    if (s_at_0 && t_at_1) return SegmentSegmentRegion::Corner_s0t1;
    if (s_at_1 && t_at_0) return SegmentSegmentRegion::Corner_s1t0;
    if (s_at_1 && t_at_1) return SegmentSegmentRegion::Corner_s1t1;

    if (s_at_0) return SegmentSegmentRegion::Edge_s0;
    if (s_at_1) return SegmentSegmentRegion::Edge_s1;
    if (t_at_0) return SegmentSegmentRegion::Edge_t0;
    if (t_at_1) return SegmentSegmentRegion::Edge_t1;

    return SegmentSegmentRegion::Interior;
}

SegmentSegmentDistanceResult segment_segment_distance(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps){
    SegmentSegmentDistanceResult out;
    out.closest_point_1 = Vec3::Zero();
    out.closest_point_2 = Vec3::Zero();
    out.s = 0.0;
    out.t = 0.0;
    out.distance = 0.0;
    out.region = SegmentSegmentRegion::ParallelSegments;

    // a = x2 - x1,  b = x4 - x3,  c = x1 - x3
    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const Vec3 c = x1 - x3;

    const double A = a.dot(a);  // ||a||^2
    const double B = a.dot(b);  // a . b
    const double C = b.dot(b);  // ||b||^2
    const double D = a.dot(c);  // a . c
    const double E = b.dot(c);  // b . c

    const double Delta = A * C - B * B;  // = ||a x b||^2

    // Non-parallel case: Delta > 0
    // Solve:  s* = (BE - CD) / Delta,  t* = (AE - BD) / Delta
    if (Delta > eps * eps) {
        const double s_unc = (B * E - C * D) / Delta;
        const double t_unc = (A * E - B * D) / Delta;

        // If the unconstrained solution lies inside [0,1]^2, we are done.
        if (s_unc >= 0.0 && s_unc <= 1.0 && t_unc >= 0.0 && t_unc <= 1.0) {
            out.s = s_unc;
            out.t = t_unc;
            out.closest_point_1 = x1 + out.s * a;
            out.closest_point_2 = x3 + out.t * b;
            out.distance = (out.closest_point_1 - out.closest_point_2).norm();
            out.region = SegmentSegmentRegion::Interior;
            return out;
        }

        // Unconstrained minimizer outside [0,1]^2. Check all four edges.
        double best_dist = std::numeric_limits<double>::max();
        double best_s = 0.0, best_t = 0.0;

        { // Edge s = 0
            double t_cand = 0.0;
            const double d = optimal_t_for_fixed_s(x1, a, x3, b, 0.0, C, t_cand);
            if (d < best_dist) { best_dist = d; best_s = 0.0; best_t = t_cand; }
        }
        { // Edge s = 1
            double t_cand = 0.0;
            const double d = optimal_t_for_fixed_s(x1, a, x3, b, 1.0, C, t_cand);
            if (d < best_dist) { best_dist = d; best_s = 1.0; best_t = t_cand; }
        }
        { // Edge t = 0
            double s_cand = 0.0;
            const double d = optimal_s_for_fixed_t(x1, a, x3, b, 0.0, A, s_cand);
            if (d < best_dist) { best_dist = d; best_s = s_cand; best_t = 0.0; }
        }
        { // Edge t = 1
            double s_cand = 0.0;
            const double d = optimal_s_for_fixed_t(x1, a, x3, b, 1.0, A, s_cand);
            if (d < best_dist) { best_dist = d; best_s = s_cand; best_t = 1.0; }
        }

        out.s = best_s;
        out.t = best_t;
        out.closest_point_1 = x1 + out.s * a;
        out.closest_point_2 = x3 + out.t * b;
        out.distance = best_dist;
        out.region = classify(out.s, out.t);
        return out;
    }

    // Parallel / degenerate case: Delta ～ 0. Check all four boundary edges.
    out.region = SegmentSegmentRegion::ParallelSegments;

    double best_dist = std::numeric_limits<double>::max();
    double best_s = 0.0, best_t = 0.0;

    { // Edge s = 0
        double t_cand = 0.0;
        const double d = optimal_t_for_fixed_s(x1, a, x3, b, 0.0, C, t_cand);
        if (d < best_dist) { best_dist = d; best_s = 0.0; best_t = t_cand; }
    }
    { // Edge s = 1
        double t_cand = 0.0;
        const double d = optimal_t_for_fixed_s(x1, a, x3, b, 1.0, C, t_cand);
        if (d < best_dist) { best_dist = d; best_s = 1.0; best_t = t_cand; }
    }
    { // Edge t = 0
        double s_cand = 0.0;
        const double d = optimal_s_for_fixed_t(x1, a, x3, b, 0.0, A, s_cand);
        if (d < best_dist) { best_dist = d; best_s = s_cand; best_t = 0.0; }
    }
    { // Edge t = 1
        double s_cand = 0.0;
        const double d = optimal_s_for_fixed_t(x1, a, x3, b, 1.0, A, s_cand);
        if (d < best_dist) { best_dist = d; best_s = s_cand; best_t = 1.0; }
    }

    out.s = best_s;
    out.t = best_t;
    out.closest_point_1 = x1 + out.s * a;
    out.closest_point_2 = x3 + out.t * b;
    out.distance = best_dist;

    return out;
}