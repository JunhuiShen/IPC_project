#include "segment_segment_distance.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
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
        t_out = std::clamp((p - x3).dot(b) / C, 0.0, 1.0);
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
        s_out = std::clamp((q - x1).dot(a) / A, 0.0, 1.0);
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

static SegmentSegmentDistanceResult segment_segment_distance_fast(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps){
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
            out.separation = out.closest_point_1 - out.closest_point_2;
            out.distance = out.separation.norm();
            out.region = SegmentSegmentRegion::Interior;
            out.active_region = out.region;
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
        out.separation = out.closest_point_1 - out.closest_point_2;
        out.distance = best_dist;
        out.region = classify(out.s, out.t);
        out.active_region = out.region;
        return out;
    }

    // Parallel / degenerate case: Delta ~ 0. Check all four boundary edges.
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
    out.separation = out.closest_point_1 - out.closest_point_2;
    out.distance = best_dist;

    return out;
}

using Exact = boost::multiprecision::cpp_rational;
using ExactVec3 = std::array<Exact, 3>;

static ExactVec3 exact_vec(const Vec3& x)
{
    // Conversion of each ORIGINAL binary64 coordinate is exact. Converting an
    // already rounded edge difference would permanently lose input bits.
    return {Exact(x[0]), Exact(x[1]), Exact(x[2])};
}

static ExactVec3 exact_subtract(const ExactVec3& x, const ExactVec3& y)
{
    return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

static Exact exact_dot(const ExactVec3& x, const ExactVec3& y)
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

static Exact exact_clamp_unit(const Exact& value)
{
    if (value < 0) return Exact(0);
    if (value > 1) return Exact(1);
    return value;
}

static SegmentSegmentRegion exact_region(const Exact& s, const Exact& t)
{
    if (s == 0 && t == 0) return SegmentSegmentRegion::Corner_s0t0;
    if (s == 0 && t == 1) return SegmentSegmentRegion::Corner_s0t1;
    if (s == 1 && t == 0) return SegmentSegmentRegion::Corner_s1t0;
    if (s == 1 && t == 1) return SegmentSegmentRegion::Corner_s1t1;
    if (s == 0) return SegmentSegmentRegion::Edge_s0;
    if (s == 1) return SegmentSegmentRegion::Edge_s1;
    if (t == 0) return SegmentSegmentRegion::Edge_t0;
    if (t == 1) return SegmentSegmentRegion::Edge_t1;
    return SegmentSegmentRegion::Interior;
}

static SegmentSegmentDistanceResult segment_segment_distance_exact(
    const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4)
{
    const ExactVec3 p = exact_vec(x1);
    const ExactVec3 q = exact_vec(x3);
    const ExactVec3 a = exact_subtract(exact_vec(x2), p);
    const ExactVec3 b = exact_subtract(exact_vec(x4), q);
    const ExactVec3 c = exact_subtract(p, q);
    const Exact A = exact_dot(a, a);
    const Exact B = exact_dot(a, b);
    const Exact C = exact_dot(b, b);
    const Exact D = exact_dot(a, c);
    const Exact E = exact_dot(b, c);
    const Exact determinant = A * C - B * B;

    Exact best_s = 0, best_t = 0, best_squared_distance = 0;
    ExactVec3 best_residual;
    bool have_best = false;
    const auto consider = [&](const Exact& s, const Exact& t) {
        const ExactVec3 residual = {
            c[0] + s * a[0] - t * b[0],
            c[1] + s * a[1] - t * b[1],
            c[2] + s * a[2] - t * b[2]};
        const Exact squared_distance = exact_dot(residual, residual);
        if (!have_best || squared_distance < best_squared_distance) {
            best_s = s;
            best_t = t;
            best_squared_distance = squared_distance;
            best_residual = residual;
            have_best = true;
        }
    };

    // A convex quadratic over [0,1]^2 attains its minimum at the feasible
    // unconstrained minimizer or on one of its four edges. All feasibility,
    // clamping, feature selection, and candidate comparisons stay exact.
    if (determinant > 0) {
        const Exact s = (B * E - C * D) / determinant;
        const Exact t = (A * E - B * D) / determinant;
        if (s >= 0 && s <= 1 && t >= 0 && t <= 1) consider(s, t);
    }
    consider(Exact(0), C > 0 ? exact_clamp_unit(E / C) : Exact(0));
    consider(Exact(1), C > 0 ? exact_clamp_unit((E + B) / C) : Exact(0));
    consider(A > 0 ? exact_clamp_unit(-D / A) : Exact(0), Exact(0));
    consider(A > 0 ? exact_clamp_unit((B - D) / A) : Exact(0), Exact(1));

    SegmentSegmentDistanceResult out;
    out.s = best_s.convert_to<double>();
    out.t = best_t.convert_to<double>();
    const std::array<Exact, 4> exact_weights = {
        Exact(1) - best_s, best_s, best_t - Exact(1), -best_t};
    for (int i = 0; i < 4; ++i) {
        out.weights[i] = exact_weights[i].convert_to<double>();
    }
    for (int i = 0; i < 3; ++i) {
        const Exact closest_1 = p[i] + best_s * a[i];
        const Exact closest_2 = q[i] + best_t * b[i];
        out.closest_point_1[i] = closest_1.convert_to<double>();
        out.closest_point_2[i] = closest_2.convert_to<double>();
        out.separation[i] = best_residual[i].convert_to<double>();
    }
    // Do not square a tiny residual or floor its norm. A true intersection
    // remains zero, while a representable positive gap remains positive.
    out.distance = out.separation.stableNorm();
    out.region = determinant == 0
        ? SegmentSegmentRegion::ParallelSegments
        : exact_region(best_s, best_t);
    out.active_region = exact_region(best_s, best_t);
    out.robust = true;
    return out;
}

SegmentSegmentDistanceResult segment_segment_distance(
    const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps,
    bool exact_computation_fallback)
{
    const auto out = segment_segment_distance_fast(x1, x2, x3, x4, eps);
    if (!exact_computation_fallback) return out;
    // Rational conversion requires finite inputs. Preserve the pre-existing
    // handling of invalid input rather than introducing conversion exceptions.
    if (!x1.allFinite() || !x2.allFinite() || !x3.allFinite() || !x4.allFinite()) {
        return out;
    }

    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const double A = a.squaredNorm();
    const double B = a.dot(b);
    const double C = b.squaredNorm();
    const double product = A * C;
    const double determinant = product - B * B;
    constexpr double roundoff = 64.0 * std::numeric_limits<double>::epsilon();
    const double coordinate_scale = std::max({
        x1.cwiseAbs().maxCoeff(), x2.cwiseAbs().maxCoeff(),
        x3.cwiseAbs().maxCoeff(), x4.cwiseAbs().maxCoeff()});

    // The absolute eps gate in the legacy solve can drop a valid interior
    // minimum even when the relative Gram condition is good (short edges).
    // The relative test also catches determinants lost through cancellation.
    const bool uncertain_parameters = !std::isfinite(determinant)
        || determinant <= roundoff * product || determinant <= eps * eps;
    const bool unresolved_gap = !std::isfinite(out.distance)
        || out.distance <= roundoff * coordinate_scale;
    if (uncertain_parameters || unresolved_gap) {
        return segment_segment_distance_exact(x1, x2, x3, x4);
    }
    return out;
}
