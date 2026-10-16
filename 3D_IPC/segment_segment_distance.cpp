#include "segment_segment_distance.h"
#include "segment_segment_distance_batch.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

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

// Denominator is strictly positive. Decide clipped endpoints before building
// a quotient, avoiding normalization of exact fractions that would be thrown
// away. Interior quotients are unchanged in scalar and cached-batch solves.
static Exact exact_clamped_positive_ratio(const Exact& numerator, const Exact& denominator)
{
    if (numerator <= 0) return Exact(0);
    if (numerator >= denominator) return Exact(1);
    return numerator / denominator;
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

static std::pair<Exact, Exact> exact_parallel_interval_parameters(
    const Exact& c, const Exact& u, const Exact& v)
{
    // Minimize the one-dimensional residual c+u*s-v*t for nonzero u,v.
    // Translate B's start to zero: its interval has endpoints 0,v,
    // and A's endpoints are c,c+u. If the intervals overlap, zero axial
    // residual is optimal. Visit endpoints in the original s0,s1,t0,t1
    // boundary order to retain the first exact minimizer, including closed
    // endpoint touches and reversed directions. The corresponding ratios
    // are exactly c/v,(c+u)/v,-c/u,(v-c)/u; all arithmetic remains rational.
    const Exact zero(0), end_a = c + u;
    const bool forward_a = u > 0, forward_b = v > 0;
    const Exact& min_a = forward_a ? c : end_a;
    const Exact& max_a = forward_a ? end_a : c;
    const Exact& min_b = forward_b ? zero : v;
    const Exact& max_b = forward_b ? v : zero;
    const auto in_b = [&](const Exact& point) { return point >= min_b && point <= max_b; };
    const auto in_a = [&](const Exact& point) { return point >= min_a && point <= max_a; };
    if (in_b(c)) return {Exact(0), c / v};
    if (in_b(end_a)) return {Exact(1), end_a / v};
    if (in_a(zero)) return {-c / u, Exact(0)};
    if (in_a(v)) return {(v - c) / u, Exact(1)};

    // With no shared endpoint the nonempty closed intervals are disjoint.
    // Their nearest endpoints are the unique minimizer, requiring no division.
    if (max_a < min_b)
        return {Exact(forward_a ? 1 : 0), Exact(forward_b ? 0 : 1)};
    return {Exact(forward_a ? 0 : 1), Exact(forward_b ? 1 : 0)};
}

enum class ExactResultMode { Scalar, CachedBatch };

static SegmentSegmentDistanceResult segment_segment_distance_exact_edges(
    const ExactVec3& p, const ExactVec3& a, const Exact& A,
    const ExactVec3& q, const ExactVec3& b, const Exact& C,
    int parallel_axis = -1, ExactResultMode mode = ExactResultMode::Scalar)
{
    const ExactVec3 c = exact_subtract(p, q);
    Exact determinant = 0;

    Exact best_s = 0, best_t = 0;
    bool have_best = false;
    const auto choose = [&](const Exact& s, const Exact& t) {
        best_s = s;
        best_t = t;
        have_best = true;
    };

    if (parallel_axis >= 0) {
        const auto parameters = exact_parallel_interval_parameters(
            c[parallel_axis], a[parallel_axis], b[parallel_axis]);
        choose(parameters.first, parameters.second);
    } else {
        const Exact B = exact_dot(a, b);
        determinant = A * C - B * B;
        const Exact D = exact_dot(a, c);
        if (mode == ExactResultMode::CachedBatch && determinant == 0 && A > 0 && C > 0) {
            // Exact Cauchy-Schwarz equality gives b=(B/A)*a and B!=0.
            // ||c+s*a-t*b||^2 = ||c_perp||^2 + (D+A*s-B*t)^2/A.
            // Hence intervals (D,A,B) give the identical first minimizer.
            // E=(B/A)*D and C=B^2/A make their four ratios identical to
            // the generic boundary ratios without computing E. Reversed
            // directions merely change B's sign; collapsed edges stay generic.
            const auto parameters = exact_parallel_interval_parameters(D, A, B);
            choose(parameters.first, parameters.second);
        } else {
            const Exact E = exact_dot(b, c);
            const auto clamped_ratio = [&](const Exact& numerator, const Exact& denominator) -> Exact {
                if (mode == ExactResultMode::CachedBatch)
                    return exact_clamped_positive_ratio(numerator, denominator);
                return exact_clamp_unit(numerator / denominator);
            };
            // q(s,t) = 1/2 ||c+s*a-t*b||^2 is convex. A feasible unconstrained
            // minimizer is already global; checking the boundaries cannot improve it.
            if (determinant > 0) {
                if (mode == ExactResultMode::CachedBatch) {
                    const Exact s_numerator = B * E - C * D;
                    // Multiplication by positive determinant preserves every
                    // feasibility comparison. Do not form either discarded
                    // quotient, or t's numerator when s is already infeasible.
                    if (s_numerator >= 0 && s_numerator <= determinant) {
                        const Exact t_numerator = A * E - B * D;
                        if (t_numerator >= 0 && t_numerator <= determinant)
                            choose(exact_clamped_positive_ratio(s_numerator, determinant),
                                exact_clamped_positive_ratio(t_numerator, determinant));
                    }
                } else {
                    const Exact s = (B * E - C * D) / determinant;
                    const Exact t = (A * E - B * D) / determinant;
                    if (s >= 0 && s <= 1 && t >= 0 && t <= 1) choose(s, t);
                }
            }
            // Each clamped boundary candidate already minimizes along that boundary.
            // The remaining exact KKT sign proves global optimality: dq/ds is
            // D+A*s-B*t, and dq/dt is C*t-E-B*s. Visit the original s0,s1,t0,t1
            // order and select the first global minimizer, preserving strict-< ties
            // even for parallel/zero-length segments. If A or C vanishes, its cross
            // products vanish too, so the same signs handle the degenerate case.
            if (mode == ExactResultMode::CachedBatch) {
                // For strict interior boundary ratios, multiply KKT by the
                // positive denominator before dividing. Failed candidates
                // stay dyadic polynomials instead of normalized quotients.
                // Clipped endpoints use their original signs and tie order.
                if (!have_best) {
                    if (C <= 0 || E <= 0) {
                        if (D >= 0) choose(Exact(0), Exact(0));
                    } else if (E >= C) {
                        if (D - B >= 0) choose(Exact(0), Exact(1));
                    } else if (D * C - B * E >= 0) {
                        choose(Exact(0), E / C);
                    }
                }
                if (!have_best) {
                    const Exact numerator = E + B;
                    const Exact offset = D + A;
                    if (C <= 0 || numerator <= 0) {
                        if (offset <= 0) choose(Exact(1), Exact(0));
                    } else if (numerator >= C) {
                        if (offset - B <= 0) choose(Exact(1), Exact(1));
                    } else if (offset * C - B * numerator <= 0) {
                        choose(Exact(1), numerator / C);
                    }
                }
                if (!have_best) {
                    const Exact numerator = -D;
                    if (A <= 0 || numerator <= 0) {
                        if (E <= 0) choose(Exact(0), Exact(0));
                    } else if (numerator >= A) {
                        if (E + B <= 0) choose(Exact(1), Exact(0));
                    } else if (E * A + B * numerator <= 0) {
                        choose(numerator / A, Exact(0));
                    }
                }
            } else {
                if (!have_best) {
                    const Exact t = C > 0 ? exact_clamp_unit(E / C) : Exact(0);
                    if (D - B * t >= 0) choose(Exact(0), t);
                }
                if (!have_best) {
                    const Exact t = C > 0 ? exact_clamp_unit((E + B) / C) : Exact(0);
                    if (D + A - B * t <= 0) choose(Exact(1), t);
                }
                if (!have_best) {
                    const Exact s = A > 0 ? exact_clamp_unit(-D / A) : Exact(0);
                    if (E + B * s <= 0) choose(s, Exact(0));
                }
            }
            // At least one boundary candidate satisfies KKT. If the first three
            // failed, t=1 is the remaining global minimizer (dq/dt <= 0 there).
            if (!have_best) {
                const Exact s = A > 0 ? clamped_ratio(B - D, A) : Exact(0);
                choose(s, Exact(1));
            }
        }
    }

    const ExactVec3 best_residual = {
        c[0] + best_s * a[0] - best_t * b[0],
        c[1] + best_s * a[1] - best_t * b[1],
        c[2] + best_s * a[2] - best_t * b[2]};

    SegmentSegmentDistanceResult out;
    out.s = best_s.convert_to<double>();
    out.t = best_t.convert_to<double>();
    if (mode == ExactResultMode::CachedBatch) {
        const Exact complement_s = Exact(1) - best_s;
        const Exact complement_t = best_t - Exact(1);
        out.weights[0] = complement_s.convert_to<double>();
        out.weights[1] = out.s;
        out.weights[2] = complement_t.convert_to<double>();
        // Exact zero converts to +0, while a tiny negative rational rounds to
        // -0. Reuse t's symmetric rounding without losing that distinction.
        out.weights[3] = best_t == 0 ? 0.0 : -out.t;
    } else {
        const std::array<Exact, 4> exact_weights = {
            Exact(1) - best_s, best_s, best_t - Exact(1), -best_t};
        for (int i = 0; i < 4; ++i) {
            out.weights[i] = exact_weights[i].convert_to<double>();
        }
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
    // Batch callers compute the norm once from their explicitly aligned copy.
    // Initialize the deferred field before returning/copying the whole record.
    out.distance = mode == ExactResultMode::Scalar ? out.separation.stableNorm() : 0.0;
    out.region = determinant == 0
        ? SegmentSegmentRegion::ParallelSegments
        : exact_region(best_s, best_t);
    out.active_region = exact_region(best_s, best_t);
    out.robust = true;
    return out;
}

static SegmentSegmentDistanceResult segment_segment_distance_exact(
    const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4)
{
    const ExactVec3 p = exact_vec(x1), q = exact_vec(x3);
    const ExactVec3 a = exact_subtract(exact_vec(x2), p);
    const ExactVec3 b = exact_subtract(exact_vec(x4), q);
    const Exact A = exact_dot(a, a), C = exact_dot(b, b);
    return segment_segment_distance_exact_edges(p, a, A, q, b, C);
}

static bool segment_segment_finite_inputs(
    const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4)
{
    // Rational conversion requires finite inputs. Preserve the pre-existing
    // handling of invalid input rather than introducing conversion exceptions.
    return x1.allFinite() && x2.allFinite() && x3.allFinite() && x4.allFinite();
}

static bool segment_segment_uncertain_parameters(
    const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps)
{
    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const double A = a.squaredNorm();
    const double B = a.dot(b);
    const double C = b.squaredNorm();
    const double product = A * C;
    const double determinant = product - B * B;
    constexpr double roundoff = 64.0 * std::numeric_limits<double>::epsilon();
    // The absolute eps gate in the legacy solve can drop a valid interior
    // minimum even when the relative Gram condition is good (short edges).
    // The relative test also catches determinants lost through cancellation.
    return !std::isfinite(determinant)
        || determinant <= roundoff * product || determinant <= eps * eps;
}

static bool segment_segment_unresolved_gap(
    const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
    const SegmentSegmentDistanceResult& out)
{
    constexpr double roundoff = 64.0 * std::numeric_limits<double>::epsilon();
    const double coordinate_scale = std::max({
        x1.cwiseAbs().maxCoeff(), x2.cwiseAbs().maxCoeff(),
        x3.cwiseAbs().maxCoeff(), x4.cwiseAbs().maxCoeff()});
    return !std::isfinite(out.distance)
        || out.distance <= roundoff * coordinate_scale;
}

SegmentSegmentDistanceResult segment_segment_distance(
    const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps,
    bool exact_computation_fallback)
{
    const auto out = segment_segment_distance_fast(x1, x2, x3, x4, eps);
    return exact_computation_fallback
        && segment_segment_finite_inputs(x1, x2, x3, x4)
        && (segment_segment_uncertain_parameters(x1, x2, x3, x4, eps)
            || segment_segment_unresolved_gap(x1, x2, x3, x4, out))
        ? segment_segment_distance_exact(x1, x2, x3, x4) : out;
}

struct SegmentSegmentDistanceBatchWorkspace::Impl {
    struct ExactEdge {
        ExactVec3 start, direction;
        Exact squared_length;
        int axis = -1;
    };
    const std::vector<Vec3>* positions = nullptr;
    const std::vector<std::array<int, 2>>* edges = nullptr;
    const std::vector<std::array<int, 2>>* pairs = nullptr;
    std::vector<SegmentSegmentDistanceResult>* outputs = nullptr;
    double eps = 1.0e-12;
    bool exact_fallback = true;
    std::vector<unsigned char> fallback_pairs;
    std::unique_ptr<std::atomic<unsigned char>[]> needed_edges;
    std::size_t needed_capacity = 0;
    std::vector<int> edge_to_exact;
    std::vector<std::size_t> used_edges;
    std::vector<ExactEdge> exact_edges;
    std::vector<unsigned char> needed_vertices;
    std::vector<int> vertex_to_exact;
    std::vector<std::size_t> used_vertices;
    std::vector<ExactVec3> exact_vertices;
    std::atomic<std::size_t> fallback_count{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;

    void capture_failure() noexcept {
        if (!failed.exchange(true, std::memory_order_relaxed))
            error = std::current_exception();
    }
};

SegmentSegmentDistanceBatchWorkspace::SegmentSegmentDistanceBatchWorkspace()
    : impl_(std::make_unique<Impl>()) {}
SegmentSegmentDistanceBatchWorkspace::~SegmentSegmentDistanceBatchWorkspace() = default;
SegmentSegmentDistanceBatchWorkspace::SegmentSegmentDistanceBatchWorkspace(
    SegmentSegmentDistanceBatchWorkspace&&) noexcept = default;
SegmentSegmentDistanceBatchWorkspace& SegmentSegmentDistanceBatchWorkspace::operator=(
    SegmentSegmentDistanceBatchWorkspace&&) noexcept = default;

void SegmentSegmentDistanceBatchWorkspace::prepare(
    const std::vector<Vec3>& positions,
    const std::vector<std::array<int, 2>>& edges,
    const std::vector<std::array<int, 2>>& pair_edges,
    double eps, bool exact_computation_fallback,
    std::vector<SegmentSegmentDistanceResult>& outputs)
{
    auto& w = *impl_;
    w.positions = &positions; w.edges = &edges; w.pairs = &pair_edges;
    w.outputs = &outputs; w.eps = eps; w.exact_fallback = exact_computation_fallback;
    w.fallback_count.store(0, std::memory_order_relaxed);
    w.failed.store(false, std::memory_order_relaxed);
    w.error = nullptr;
    w.used_edges.clear();
    w.used_vertices.clear();
    outputs.resize(pair_edges.size());
    if (!exact_computation_fallback) return;
    w.fallback_pairs.resize(pair_edges.size());
    w.edge_to_exact.resize(edges.size());
    w.needed_vertices.assign(positions.size(), 0);
    // Only currently marked/compacted vertices are referenced by used edges;
    // inactive retained map entries need no reset, just like edge_to_exact.
    w.vertex_to_exact.resize(positions.size());
    if (edges.size() > w.needed_capacity) {
        w.needed_edges = std::make_unique<std::atomic<unsigned char>[]>(edges.size());
        w.needed_capacity = edges.size();
    }
}

void SegmentSegmentDistanceBatchWorkspace::compute_collective()
{
    auto& w = *impl_;
    const auto& x = *w.positions;
    const auto& edges = *w.edges;
    const auto& pairs = *w.pairs;
    auto& out = *w.outputs;
    if (!w.exact_fallback) {
        #pragma omp for schedule(dynamic, 64)
        for (std::size_t pair = 0; pair < pairs.size(); ++pair) {
            const auto& a = edges[pairs[pair][0]];
            const auto& b = edges[pairs[pair][1]];
            out[pair] = segment_segment_distance_fast(x[a[0]], x[a[1]], x[b[0]], x[b[1]], w.eps);
        }
        return;
    }
    #pragma omp for schedule(static)
    for (std::size_t edge = 0; edge < edges.size(); ++edge)
        w.needed_edges[edge].store(0, std::memory_order_relaxed);

    std::size_t local_fallbacks = 0;
    #pragma omp for schedule(dynamic, 64) nowait
    for (std::size_t pair = 0; pair < pairs.size(); ++pair) {
        const auto& indices = pairs[pair];
        const auto& a = edges[indices[0]];
        const auto& b = edges[indices[1]];
        const bool finite = segment_segment_finite_inputs(x[a[0]], x[a[1]], x[b[0]], x[b[1]]);
        // The Gram/eps gate alone already requires an exact result for many
        // parallel grid edges. Avoid their otherwise discarded fast boundary
        // distances; exact solves overwrite these slots before any consumer.
        bool exact = finite && segment_segment_uncertain_parameters(
            x[a[0]], x[a[1]], x[b[0]], x[b[1]], w.eps);
        if (!exact) {
            out[pair] = segment_segment_distance_fast(x[a[0]], x[a[1]], x[b[0]], x[b[1]], w.eps);
            exact = finite && segment_segment_unresolved_gap(
                x[a[0]], x[a[1]], x[b[0]], x[b[1]], out[pair]);
        }
        w.fallback_pairs[pair] = exact;
        if (exact) {
            ++local_fallbacks;
            for (int edge : indices) {
                // After the first marks, readers leave the shared flag alone.
                if (!w.needed_edges[edge].load(std::memory_order_relaxed))
                    w.needed_edges[edge].store(1, std::memory_order_relaxed);
            }
        }
    }
    w.fallback_count.fetch_add(local_fallbacks, std::memory_order_relaxed);
    #pragma omp barrier
    #pragma omp single
    {
        try {
            if (w.fallback_count.load(std::memory_order_relaxed) != 0) {
                for (std::size_t edge = 0; edge < edges.size(); ++edge) {
                    if (w.needed_edges[edge].load(std::memory_order_relaxed)) {
                        w.edge_to_exact[edge] = static_cast<int>(w.used_edges.size());
                        w.used_edges.push_back(edge);
                        for (int vertex : edges[edge]) w.needed_vertices[vertex] = 1;
                    }
                }
                for (std::size_t vertex = 0; vertex < x.size(); ++vertex) {
                    if (w.needed_vertices[vertex]) {
                        w.vertex_to_exact[vertex] = static_cast<int>(w.used_vertices.size());
                        w.used_vertices.push_back(vertex);
                    }
                }
                w.exact_vertices.resize(w.used_vertices.size());
                w.exact_edges.resize(w.used_edges.size());
            }
        } catch (...) { w.capture_failure(); }
    }
    // The single's join publishes one uniform branch to the entire team.
    if (w.used_edges.empty()) return;
    // Adjacent used edges share source vertices. Convert each original binary64
    // coordinate once in this frozen snapshot, then join before edge reads.
    #pragma omp for schedule(static)
    for (std::size_t record = 0; record < w.used_vertices.size(); ++record) {
        if (w.failed.load(std::memory_order_relaxed)) continue;
        try {
            w.exact_vertices[record] = exact_vec(x[w.used_vertices[record]]);
        } catch (...) { w.capture_failure(); }
    }
    #pragma omp for schedule(static)
    for (std::size_t record = 0; record < w.used_edges.size(); ++record) {
        if (w.failed.load(std::memory_order_relaxed)) continue;
        try {
            auto& exact = w.exact_edges[record];
            const auto& edge = edges[w.used_edges[record]];
            exact.start = w.exact_vertices[w.vertex_to_exact[edge[0]]];
            exact.direction = exact_subtract(w.exact_vertices[w.vertex_to_exact[edge[1]]], exact.start);
            exact.squared_length = exact_dot(exact.direction, exact.direction);
            exact.axis = -1;
            for (int axis = 0; axis < 3; ++axis) {
                if (exact.direction[axis] == 0) continue;
                if (exact.axis >= 0) { exact.axis = -1; break; }
                exact.axis = axis;
            }
        } catch (...) { w.capture_failure(); }
    }
    #pragma omp for schedule(dynamic, 64)
    for (std::size_t pair = 0; pair < pairs.size(); ++pair) {
        if (!w.fallback_pairs[pair] || w.failed.load(std::memory_order_relaxed)) continue;
        try {
            const auto& a = w.exact_edges[w.edge_to_exact[pairs[pair][0]]];
            const auto& b = w.exact_edges[w.edge_to_exact[pairs[pair][1]]];
            out[pair] = segment_segment_distance_exact_edges(
                a.start, a.direction, a.squared_length,
                b.start, b.direction, b.squared_length,
                a.axis >= 0 && a.axis == b.axis ? a.axis : -1,
                ExactResultMode::CachedBatch);
            // Eigen's stableNorm splits at the first aligned coefficient, so
            // its rounding can depend on a Vec3's address. Legacy OGC evaluates
            // these returns in 16-byte-aligned storage. Match that convention
            // explicitly without changing other scalar callers' return layout.
            alignas(16) const Vec3 norm_separation = out[pair].separation;
            out[pair].distance = norm_separation.stableNorm();
        } catch (...) { w.capture_failure(); }
    }
}

bool SegmentSegmentDistanceBatchWorkspace::failed() const noexcept {
    return impl_->failed.load(std::memory_order_relaxed);
}

void SegmentSegmentDistanceBatchWorkspace::rethrow_if_failed() const {
    if (impl_->failed.load(std::memory_order_relaxed))
        std::rethrow_exception(impl_->error);
}

std::size_t SegmentSegmentDistanceBatchWorkspace::fallback_pair_count() const noexcept {
    return impl_->fallback_count.load(std::memory_order_relaxed);
}

std::size_t SegmentSegmentDistanceBatchWorkspace::prepared_edge_count() const noexcept {
    return impl_->used_edges.size();
}
