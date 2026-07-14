// Two CCD backends behind the public dispatchers in ccd.h:
//   - Linear   : closed-form, exact when one of the four vertices moves
//                over the step (Gauss-Seidel safe-step query).
//   - TICCD    : Tight-Inclusion CCD library [Wang et al. 2021] for the
//                general case where multiple vertices move.
// `node_triangle_general_ccd` / `segment_segment_general_ccd` are TICCD-only.
// `*_only_one_node_moves` route to one of the two via the `use_ticcd` arg.

#include "ccd.h"

#include <tight_inclusion/ccd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

// -----------------------------------------------------------------------------
// Internal helpers (linear backend + TICCD config + result translation)
// -----------------------------------------------------------------------------
namespace {

// Polynomial root finders (linear backend coplanar fallback)

void solve_linear_all(double c, double d, SmallRoots& roots, double eps) {
    double s = std::max(1.0, std::fabs(d));
    if (nearly_zero(c, eps * s)) return;
    add_root(roots, -d / c, eps);
}

void solve_quadratic_all(double a, double b, double c, SmallRoots& roots, double eps) {
    double s = max_abs_value_among_four_numbers(a, b, c, 0.0);
    if (nearly_zero(a, eps * s)) { solve_linear_all(b, c, roots, eps); return; }

    double D = b * b - 4.0 * a * c;
    const double coeff_scale = std::max({std::fabs(a), std::fabs(b), std::fabs(c)});
    double tol = eps * coeff_scale * coeff_scale * 16.0;
    if (D < -tol) return;
    if (std::fabs(D) <= tol) { add_root(roots, -b / (2.0 * a), eps); return; }

    double sqrtD = std::sqrt(std::max(0.0, D));
    double signb = (b >= 0.0) ? 1.0 : -1.0;
    double q = -0.5 * (b + signb * sqrtD);
    add_root(roots, q / a, eps);
    if (!nearly_zero(q, eps * s)) add_root(roots, c / q, eps);
}

// 2D projection helpers (coplanar fallback)

Vec2 project_drop_axis(const Vec3& v, int drop_axis) {
    if (drop_axis == 0) return Vec2(v.y(), v.z());
    if (drop_axis == 1) return Vec2(v.x(), v.z());
    return Vec2(v.x(), v.y());
}

int dominant_drop_axis(const Vec3& n0, const Vec3& n1) {
    Vec3 n(std::fabs(n0.x()) + std::fabs(n1.x()),
           std::fabs(n0.y()) + std::fabs(n1.y()),
           std::fabs(n0.z()) + std::fabs(n1.z()));
    if (n.x() >= n.y() && n.x() >= n.z()) return 0;
    if (n.y() >= n.x() && n.y() >= n.z()) return 1;
    return 2;
}

// Geometric containment tests at a fixed t

bool inside_triangle_3d_at(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                           const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3,
                           double t, double eps2) {
    Vec3 xt = x + dx * t;
    Vec3 a = x1 + dx1 * t, b = x2 + dx2 * t, c = x3 + dx3 * t;
    Vec3 e1 = b - a, e2 = c - a, w = xt - a;
    Vec3 n = e1.cross(e2);
    double n2 = n.squaredNorm();
    double d11 = e1.dot(e1), d12 = e1.dot(e2), d22 = e2.dot(e2);
    const double tri_scale = std::sqrt(d11) + std::sqrt(d22);
    const double eps_len = std::max(eps2, 1.0e-8 * tri_scale);
    if (n2 <= eps_len * eps_len) return false;
    if (std::fabs(n.dot(w)) > eps_len * std::sqrt(n2)) return false;

    double b1 = e1.dot(w), b2 = e2.dot(w);
    double det = d11 * d22 - d12 * d12;
    if (std::fabs(det) <= eps_len * std::max({1.0, d11, d22})) return false;

    double lam2 = (d22 * b1 - d12 * b2) / det;
    double lam3 = (d11 * b2 - d12 * b1) / det;
    double lam1 = 1.0 - lam2 - lam3;
    return lam1 >= -1e-8 && lam2 >= -1e-8 && lam3 >= -1e-8;
}

bool inside_segments_3d_at(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                           const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4,
                           double t, double eps2) {
    Vec3 a0 = x1 + dx1 * t, a1 = x2 + dx2 * t;
    Vec3 b0 = x3 + dx3 * t, b1 = x4 + dx4 * t;

    Vec3 u = a1 - a0, v = b1 - b0, w = b0 - a0;
    Vec3 n = u.cross(v);
    double n2 = n.squaredNorm();
    const double eps_len = std::max(eps2, 1.0e-8 * (u.norm() + v.norm()));

    if (n2 <= eps_len * eps_len) {
        if (w.cross(u).squaredNorm() > eps_len * eps_len * std::max(1.0, u.squaredNorm())) return false;

        Vec3 d0 = a1 - a0;
        double ax = std::fabs(d0.x()), ay = std::fabs(d0.y()), az = std::fabs(d0.z());
        int axis = (ax >= ay && ax >= az) ? 0 : (ay >= ax && ay >= az ? 1 : 2);
        auto comp1d = [&](const Vec3& p) { return axis == 0 ? p.x() : (axis == 1 ? p.y() : p.z()); };

        double A = comp1d(a0), B = comp1d(a1), C = comp1d(b0), D = comp1d(b1);
        if (A > B) std::swap(A, B);
        if (C > D) std::swap(C, D);
        return A <= D + 1e-8 && C <= B + 1e-8;
    }

    if (std::fabs(w.dot(n)) > eps_len * std::sqrt(n2)) return false;
    double s = w.cross(v).dot(n) / n2;
    double upar = w.cross(u).dot(n) / n2;
    Vec3 pa = a0 + u * s, pb = b0 + v * upar;
    if ((pa - pb).squaredNorm() > eps_len * eps_len) return false;
    return s >= -1e-8 && s <= 1.0 + 1e-8 && upar >= -1e-8 && upar <= 1.0 + 1e-8;
}

// 2D primitives for the coplanar / collinear fallback

bool point_in_triangle_projected(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c) {
    double e0 = cross_product_in_2d(b - a, p - a);
    double e1 = cross_product_in_2d(c - b, p - b);
    double e2 = cross_product_in_2d(a - c, p - c);
    return (e0 >= -1e-10 && e1 >= -1e-10 && e2 >= -1e-10) ||
           (e0 <=  1e-10 && e1 <=  1e-10 && e2 <=  1e-10);
}

// 2D orientation determinant linearized in t (one-moving-node makes the t^2
// term cross_2d(du, dv) vanish — it appears in both NT and SS coplanar paths).
// Shared by node_triangle_coplanar_interval and persistent_segment_segment_time.
void linear_orient_coeffs_2d(const Vec2& p0, const Vec2& dp,
                             const Vec2& q0, const Vec2& dq,
                             const Vec2& r0, const Vec2& dr, double out[2]) {
    Vec2 u0 = q0 - p0, du = dq - dp;
    Vec2 v0 = r0 - p0, dv = dr - dp;
    out[1] = cross_product_in_2d(du, v0) + cross_product_in_2d(u0, dv);
    out[0] = cross_product_in_2d(u0, v0);
}

double node_triangle_coplanar_interval(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                                       const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3,
                                       double eps) {
    Vec3 n0 = (x2 - x1).cross(x3 - x1);
    Vec3 n1 = (x2 + dx2 - x1 - dx1).cross(x3 + dx3 - x1 - dx1);
    int drop = dominant_drop_axis(n0, n1);

    Vec2 p0 = project_drop_axis(x , drop), dp = project_drop_axis(dx , drop);
    Vec2 a0 = project_drop_axis(x1, drop), da = project_drop_axis(dx1, drop);
    Vec2 b0 = project_drop_axis(x2, drop), db = project_drop_axis(dx2, drop);
    Vec2 c0 = project_drop_axis(x3, drop), dc = project_drop_axis(dx3, drop);

    if (point_in_triangle_projected(p0, a0, b0, c0)) return 0.0;

    SmallRoots roots;
    double k[2];
    linear_orient_coeffs_2d(a0, da, b0, db, p0, dp, k); solve_linear_all(k[1], k[0], roots, eps);
    linear_orient_coeffs_2d(b0, db, c0, dc, p0, dp, k); solve_linear_all(k[1], k[0], roots, eps);
    linear_orient_coeffs_2d(c0, dc, a0, da, p0, dp, k); solve_linear_all(k[1], k[0], roots, eps);

    std::sort(roots.begin(), roots.end());
    for (double t : roots) {
        Vec2 pt = p0 + dp * t, at = a0 + da * t, bt = b0 + db * t, ct = c0 + dc * t;
        if (point_in_triangle_projected(pt, at, bt, ct)) return t;
    }
    return 1.0;
}

double earliest_scalar_between(double p0, double dp, double a0, double da, double b0, double db, double eps) {
    auto h = [&](double t) {
        double ra = (p0 - a0) + t * (dp - da);
        double rb = (p0 - b0) + t * (dp - db);
        return ra * rb;
    };
    if (h(0.0) <= eps) return 0.0;

    double qa = (dp - da) * (dp - db);
    double qb = (p0 - a0) * (dp - db) + (p0 - b0) * (dp - da);
    double qc = (p0 - a0) * (p0 - b0);

    SmallRoots roots;
    solve_quadratic_all(qa, qb, qc, roots, eps);
    std::sort(roots.begin(), roots.end());
    for (double r : roots) {
        if (h(r) <= eps) return r;
        if (h(std::min(1.0, r + 1e-9)) <= eps) return r;
    }
    return 1.0;
}

bool segments_intersect_projected(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& d) {
    auto orient = [](const Vec2& p, const Vec2& q, const Vec2& r) {
        return cross_product_in_2d(q - p, r - p);
    };
    auto on_segment = [](const Vec2& p, const Vec2& q, const Vec2& r) {
        return q.x() >= std::min(p.x(), r.x()) - 1e-10 && q.x() <= std::max(p.x(), r.x()) + 1e-10 &&
               q.y() >= std::min(p.y(), r.y()) - 1e-10 && q.y() <= std::max(p.y(), r.y()) + 1e-10;
    };
    auto sgn = [](double v) { return (v > 1e-10) - (v < -1e-10); };

    double o1 = orient(a, b, c), o2 = orient(a, b, d);
    double o3 = orient(c, d, a), o4 = orient(c, d, b);
    int s1 = sgn(o1), s2 = sgn(o2), s3 = sgn(o3), s4 = sgn(o4);

    if (s1 * s2 < 0 && s3 * s4 < 0) return true;
    if (s1 == 0 && on_segment(a, c, b)) return true;
    if (s2 == 0 && on_segment(a, d, b)) return true;
    if (s3 == 0 && on_segment(c, a, d)) return true;
    if (s4 == 0 && on_segment(c, b, d)) return true;
    return false;
}

double collinear_segment_overlap_time(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                                      const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4,
                                      double eps) {
    Vec3 d0 = x2 - x1;
    Vec3 d1 = x2 + dx2 - x1 - dx1;
    double ax = std::fabs(d0.x()) + std::fabs(d1.x());
    double ay = std::fabs(d0.y()) + std::fabs(d1.y());
    double az = std::fabs(d0.z()) + std::fabs(d1.z());
    int axis = (ax >= ay && ax >= az) ? 0 : (ay >= ax && ay >= az ? 1 : 2);
    auto comp = [&](const Vec3& v) { return axis == 0 ? v.x() : (axis == 1 ? v.y() : v.z()); };

    double A0 = comp(x1), dA = comp(dx1);
    double B0 = comp(x2), dB = comp(dx2);
    double C0 = comp(x3), dC = comp(dx3);
    double D0 = comp(x4), dD = comp(dx4);

    double best = 1.0;
    best = std::min(best, earliest_scalar_between(A0, dA, C0, dC, D0, dD, eps));
    best = std::min(best, earliest_scalar_between(B0, dB, C0, dC, D0, dD, eps));
    best = std::min(best, earliest_scalar_between(C0, dC, A0, dA, B0, dB, eps));
    best = std::min(best, earliest_scalar_between(D0, dD, A0, dA, B0, dB, eps));
    return best;
}

double persistent_segment_segment_time(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                                       const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4,
                                       double eps) {
    Vec3 n0 = (x2 - x1).cross(x4 - x3);
    Vec3 n1 = (x2 + dx2 - x1 - dx1).cross(x4 + dx4 - x3 - dx3);
    if (n0.squaredNorm() <= eps * eps && n1.squaredNorm() <= eps * eps)
        return collinear_segment_overlap_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4, eps);
    int drop = dominant_drop_axis(n0, n1);

    Vec2 a0 = project_drop_axis(x1, drop), da = project_drop_axis(dx1, drop);
    Vec2 b0 = project_drop_axis(x2, drop), db = project_drop_axis(dx2, drop);
    Vec2 c0 = project_drop_axis(x3, drop), dc = project_drop_axis(dx3, drop);
    Vec2 d0 = project_drop_axis(x4, drop), dd = project_drop_axis(dx4, drop);

    if (segments_intersect_projected(a0, b0, c0, d0)) return 0.0;

    SmallRoots roots;
    double k[2];
    linear_orient_coeffs_2d(a0, da, b0, db, c0, dc, k); solve_linear_all(k[1], k[0], roots, eps);
    linear_orient_coeffs_2d(a0, da, b0, db, d0, dd, k); solve_linear_all(k[1], k[0], roots, eps);
    linear_orient_coeffs_2d(c0, dc, d0, dd, a0, da, k); solve_linear_all(k[1], k[0], roots, eps);
    linear_orient_coeffs_2d(c0, dc, d0, dd, b0, db, k); solve_linear_all(k[1], k[0], roots, eps);

    std::sort(roots.begin(), roots.end());
    for (double t : roots) {
        Vec2 at = a0 + da * t, bt = b0 + db * t, ct = c0 + dc * t, dt = d0 + dd * t;
        if (segments_intersect_projected(at, bt, ct, dt)) return t;
    }
    return collinear_segment_overlap_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4, eps);
}

// TICCD configuration (see Wang et al. 2021, "A Large-Scale Benchmark and an
// Inclusion-Based Algorithm for Continuous Collision Detection", and the
// upstream library at github.com/Continuous-Collision-Detection/Tight-Inclusion).
//
// TICCD is a conservative inclusion-based root finder for the CCD polynomial:
// it returns `false` only when there is provably no collision over [0, t_max].
// A `true` return is exact unless the closest distance between the moving
// primitives is below `tolerance + ms + err`, in which case it can be a false
// positive (which is the safe direction for collision response).
//
//   ms (kTiccdMinSeparation, 1e-10)
//     Minimum separation distance. A collision is *guaranteed* to be reported
//     whenever the pair gets closer than `ms`. Setting `ms = 0` requires
//     exact contact; any positive value pads the inclusion box.
//
//   tolerance (kTiccdTolerance, 1e-6)
//     Target solver precision: the maximum edge length of the bracketing
//     inclusion box at termination. 1e-6 matches the upstream reference
//     example. Tighter values cost more iterations.
//
//   max_itr (kTiccdMaxIter, 1e6)
//     Hard cap on inclusion-tree iterations. 1e6 matches upstream; -1 disables
//     the cap entirely.
//
//   no_zero_toi (kTiccdNoZeroToi)
//     When `ms > 0`, set true to keep refining instead of letting `toi` snap
//     to 0. We use the upstream default for parity with the reference.
//
//   ccd_method (kTiccdMethod, BREADTH_FIRST_SEARCH)
//     Inclusion-tree traversal order. Upstream-recommended default.
//
//   err (computed per-call, Array3::Constant(-1.0))
//     Per-axis numerical filter compensating for floating-point error in the
//     inclusion function. (-1, -1, -1) tells TICCD to compute it from the
//     query alone; for tighter filters call `ticcd::get_numerical_error()`
//     once with the simulation's scene AABB and reuse the result.
//
//   t_max (1.0, hard-coded at the call site)
//     Upper bound on the time interval to check. We always sweep the full
//     [0, 1] step.

constexpr double kTiccdMinSeparation = 1.0e-10;
constexpr double kTiccdTolerance     = 1.0e-6;
constexpr long   kTiccdMaxIter       = 1000000;
constexpr bool   kTiccdNoZeroToi     = ticcd::DEFAULT_NO_ZERO_TOI;
constexpr ticcd::CCDRootFindingMethod kTiccdMethod = ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH;

double clamp_toi(double toi) {
    if (!std::isfinite(toi)) return 1.0;
    if (toi < 0.0) return 0.0;
    if (toi > 1.0) return 1.0;
    return toi;
}

CCDResult ccd_result_from_toi(double toi) {
    CCDResult r;
    if (toi < 1.0) { r.collision = true; r.t = toi; }
    return r;
}

// -----------------------------------------------------------------------------
// Linear backend implementations.
// -----------------------------------------------------------------------------

CCDResult linear_node_triangle_impl(const Vec3& x,  const Vec3& dx,
                                    const Vec3& x1, const Vec3& dx1,
                                    const Vec3& x2, const Vec3& dx2,
                                    const Vec3& x3, const Vec3& dx3,
                                    double eps) {
    CCDResult result;
    constexpr double eps_inside = 1.0e-10;

    if (inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, 0.0, eps_inside)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }

    // f(t) = ((x2(t)-x1(t)) x (x3(t)-x1(t))) . (x(t)-x1(t))
    const Vec3 p0 = x2 - x1, dp = dx2 - dx1;
    const Vec3 q0 = x3 - x1, dq = dx3 - dx1;
    const Vec3 r0 = x  - x1, dr = dx  - dx1;
    const Vec3 p0xq0 = p0.cross(q0);
    const double d = p0xq0.dot(r0);
    const double c = dp.cross(q0).dot(r0) + p0.cross(dq).dot(r0) + p0xq0.dot(dr);

    if (nearly_zero(c, eps)) {
        if (nearly_zero(d, eps)) {
            const double t = node_triangle_coplanar_interval(x, dx, x1, dx1, x2, dx2, x3, dx3, eps);
            if (t < 1.0 && inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, t, eps_inside)) {
                result.collision = true;
                result.t = clamp_scalar(t, 0.0, 1.0);
            }
        }
        return result;
    }

    const double t = -d / c;
    if (!in_unit_interval(t, eps)) {
        return result;
    }
    const double candidate_t = clamp_scalar(t, 0.0, 1.0);
    if (inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, candidate_t, eps_inside)) {
        result.collision = true;
        result.t = candidate_t;
    }
    return result;
}

CCDResult linear_segment_segment_impl(const Vec3& x1, const Vec3& dx1,
                                      const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps) {
    CCDResult result;
    constexpr double eps_inside = 1.0e-10;
    const Vec3 zero = Vec3::Zero();

    if (inside_segments_3d_at(x1, dx1, x2, zero, x3, zero, x4, zero, 0.0, eps_inside)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }

    // Coplanarity condition: f(t) = ((x2-x1-t*dx1) x (x4-x3)) . (x3-x1-t*dx1)
    const Vec3 a = x2 - x1, b = x4 - x3, c0 = x3 - x1;
    const double d = (a.cross(b)).dot(c0);
    const double c = -(a.cross(b)).dot(dx1) - (dx1.cross(b)).dot(c0);

    if (nearly_zero(c, eps)) {
        if (nearly_zero(d, eps)) {
            const double t = persistent_segment_segment_time(x1, dx1, x2, zero, x3, zero, x4, zero, eps);
            if (t < 1.0 && inside_segments_3d_at(x1, dx1, x2, zero, x3, zero, x4, zero, t, eps_inside)) {
                result.collision = true;
                result.t = clamp_scalar(t, 0.0, 1.0);
            }
        }
        return result;
    }

    const double t = -d / c;
    const bool t_in_range = in_unit_interval(t, eps);
    if (t_in_range) {
        const double candidate_t = clamp_scalar(t, 0.0, 1.0);
        if (inside_segments_3d_at(x1, dx1, x2, zero, x3, zero, x4, zero, candidate_t, eps_inside)) {
            result.collision = true;
            result.t = candidate_t;
            return result;
        }
    }
    const double ab2   = (a.cross(b)).squaredNorm();
    const double scale = std::max(1.0, a.squaredNorm() * b.squaredNorm());
    const bool near_parallel = ab2 <= 1.0e-12 * scale;
    if (!t_in_range || near_parallel) {
        for (double ts : {0.0, 1.0, 0.25, 0.5, 0.75, 0.125, 0.375, 0.625, 0.875}) {
            if (inside_segments_3d_at(x1, dx1, x2, zero, x3, zero, x4, zero, ts, eps_inside)) {
                result.collision = true;
                result.t = ts;
                return result;
            }
        }
    }
    return result;
}

}  // namespace

// -----------------------------------------------------------------------------
// TICCD-backed general (all-vertices-may-move) entry points (public).
// -----------------------------------------------------------------------------

double node_triangle_general_ccd(const Vec3& x,  const Vec3& dx,
                                 const Vec3& x1, const Vec3& dx1,
                                 const Vec3& x2, const Vec3& dx2,
                                 const Vec3& x3, const Vec3& dx3) {
    const ticcd::Array3 err = ticcd::Array3::Constant(-1.0);
    double toi = std::numeric_limits<double>::infinity();
    double output_tolerance = kTiccdTolerance;
    const bool collision = ticcd::vertexFaceCCD(
        x,      x1,       x2,       x3,
        x + dx, x1 + dx1, x2 + dx2, x3 + dx3,
        err, kTiccdMinSeparation, toi, kTiccdTolerance, /*t_max=*/1.0,
        kTiccdMaxIter, output_tolerance, kTiccdNoZeroToi, kTiccdMethod);
    return collision ? clamp_toi(toi) : 1.0;
}

double segment_segment_general_ccd(const Vec3& x1, const Vec3& dx1,
                                   const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3,
                                   const Vec3& x4, const Vec3& dx4) {
    const ticcd::Array3 err = ticcd::Array3::Constant(-1.0);
    double toi = std::numeric_limits<double>::infinity();
    double output_tolerance = kTiccdTolerance;
    const bool collision = ticcd::edgeEdgeCCD(
        x1,       x2,       x3,       x4,
        x1 + dx1, x2 + dx2, x3 + dx3, x4 + dx4,
        err, kTiccdMinSeparation, toi, kTiccdTolerance, /*t_max=*/1.0,
        kTiccdMaxIter, output_tolerance, kTiccdNoZeroToi, kTiccdMethod);
    return collision ? clamp_toi(toi) : 1.0;
}

// -----------------------------------------------------------------------------
// Public one-moving-node dispatchers. `use_ticcd` selects the backend:
//   true  (default) -> TICCD library (conservative, robust)
//   false           -> closed-form linear (faster; exact for one moving node)
// -----------------------------------------------------------------------------

CCDResult node_triangle_only_one_node_moves(const Vec3& x,  const Vec3& dx,
                                            const Vec3& x1, const Vec3& dx1,
                                            const Vec3& x2, const Vec3& dx2,
                                            const Vec3& x3, const Vec3& dx3,
                                            double eps, bool use_ticcd) {
    if (use_ticcd) {
        return ccd_result_from_toi(
            node_triangle_general_ccd(x, dx, x1, dx1, x2, dx2, x3, dx3));
    }
    return linear_node_triangle_impl(x, dx, x1, dx1, x2, dx2, x3, dx3, eps);
}

CCDResult segment_segment_only_one_node_moves(const Vec3& x1, const Vec3& dx1,
                                              const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                              double eps, bool use_ticcd) {
    if (use_ticcd) {
        const Vec3 zero = Vec3::Zero();
        return ccd_result_from_toi(
            segment_segment_general_ccd(x1, dx1, x2, zero, x3, zero, x4, zero));
    }
    return linear_segment_segment_impl(x1, dx1, x2, x3, x4, eps);
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////  Rigid Body CCD  /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static bool point_segment_2d_rb_rotation(
    const Eigen::Vector2d& x, const Eigen::Vector2d& x_com, const double& theta_n,
    const double& theta_new, const Eigen::Vector2d& x0, const Eigen::Vector2d& x1,
    double& step) {
    step = 0.0;

    Eigen::Vector2d dx = x - x_com;
    double cos_n = std::cos(theta_n);
    double sin_n = std::sin(theta_n);

    Eigen::Vector2d r{cos_n * dx.x() + sin_n * dx.y(), -sin_n * dx.x() + cos_n * dx.y()};

    Eigen::Vector2d d = x1 - x0;
    double seg_len = d.norm();
    if (seg_len < 1e-14) {
        std::cerr << "Warning: degenerate segment in point_segment_2d_rotation(): seg_len = " << seg_len
                  << ", threshold = 1e-14\n";
        return false;
    }

    Eigen::Vector2d d_hat = d / seg_len;

    double A = r.x() * d_hat.y() - r.y() * d_hat.x();
    double B = -r.x() * d_hat.x() - r.y() * d_hat.y();
    double C = (x_com.x() - x0.x()) * d_hat.y() - (x_com.y() - x0.y()) * d_hat.x();

    double amplitude = std::sqrt(A * A + B * B);
    if (std::abs(C) > amplitude + 1e-14) return false;
    if (amplitude < 1e-14) {
        if (std::abs(C) > 1e-14) return false;
        const double t_star = (x - x0).dot(d) / (seg_len * seg_len);
        if (t_star < -1e-12 || t_star > 1.0 + 1e-12) return false;
        return true;
    }

    double phi = std::atan2(B, A);
    double arccos_val = std::acos(std::clamp(-C / amplitude, -1.0, 1.0)); // clamp to prevent numerical errors.

    double theta_candidates[2] = {phi + arccos_val, phi - arccos_val};

    double best_s = std::numeric_limits<double>::infinity();
    double dtheta = theta_new - theta_n;
    if (std::abs(dtheta) < 1e-14) {
        // std::cerr << "Warning: theta_new - theta_n < threshold\n";
        return false;
    }

    constexpr double two_pi = 2.0 * M_PI;

    auto consider_theta = [&](double theta_star) {
        double s = (theta_star - theta_n) / dtheta;
        if (s < -1e-12 || s > 1.0 + 1e-12) return;
        s = std::clamp(s, 0.0, 1.0);

        double theta_s = theta_star;
        Eigen::Vector2d x_s{
            std::cos(theta_s) * r.x() - std::sin(theta_s) * r.y(),
            std::sin(theta_s) * r.x() + std::cos(theta_s) * r.y()
        };
        x_s += x_com;

        double t_star = (x_s - x0).dot(d) / (seg_len * seg_len);
        if (t_star < -1e-12 || t_star > 1.0 + 1e-12) return;

        if (s < best_s) best_s = s;
    };

    for (double theta_base : theta_candidates) {
        double k = 0.0;
        if (dtheta > 0.0) {
            // Forward rotation: choose the first wrapped root at or after theta_n
            k = std::ceil((theta_n - theta_base - 1e-12) / two_pi);
        } else {
            // Backward rotation: choose the first wrapped root at or before theta_n
            k = std::floor((theta_n - theta_base + 1e-12) / two_pi);
        }
        consider_theta(theta_base + two_pi * k);
    }

    if (best_s == std::numeric_limits<double>::infinity()) return false;
    step = best_s;
    return true;
}

static void buildBasis(const Vec3& n_hat, Vec3& e1, Vec3& e2) {
    Vec3 ref = (std::abs(n_hat.x()) < 0.9) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    e1 = n_hat.cross(ref).normalized();
    e2 = n_hat.cross(e1);
}

static Vec2 project2D(const Vec3& p, const Vec3& x_com, const Vec3& e1, const Vec3& e2) {
    Vec3 dp = p - x_com;
    return Vec2(dp.dot(e1), dp.dot(e2));
}

bool segment_segment_rb_rotation_ccd(
    const Vec3& x0, const Vec3& x1, // moving segment world space posiitons
    const Vec3& x_com, // moving segment rigid body center of mass
    const Vec4& q_new, const Vec4& q_n, // moving segment rigid body orientations (quaternions)
    const Vec3& x2, const Vec3& x3, // stationary segment world space positions
    double& s)
{
    s = 0.0;
    constexpr double eps = 1e-10;

    // -------------------------
    // Step 1: Extract rotation axis from q_rel = q_new * q_n^(-1)
    // -------------------------
    Vec4 q_n_conj = Rigid_Body::ALGEBRA::ConjugateQuaternion(q_n);
    Vec4 q_rel    = Rigid_Body::ALGEBRA::QuaternionMultiply(q_new, q_n_conj);

    Vec3 v_rel(q_rel[1], q_rel[2], q_rel[3]);
    double v_rel_norm = v_rel.norm();
    if (v_rel_norm < eps) return false;

    Vec3 n_hat = v_rel / v_rel_norm;

    // -------------------------
    // Step 2: Decompose moving endpoints
    // -------------------------
    double h0 = (x0 - x_com).dot(n_hat);
    double h1 = (x1 - x_com).dot(n_hat);

    Vec3 x0_perp = (x0 - x_com) - h0 * n_hat;
    Vec3 x1_perp = (x1 - x_com) - h1 * n_hat;
    double r0    = x0_perp.norm();
    double r1    = x1_perp.norm();
    double r_max = std::max(r0, r1);

    if (r_max < eps) return false;

    // Signed dtheta from larger-radius endpoint
    const Vec3& ref_perp       = (r0 >= r1) ? x0_perp : x1_perp;
    Vec3        ref_perp_rotated = Rigid_Body::ALGEBRA::QuaternionRotate(q_rel, ref_perp);
    double dtheta = std::atan2(
        ref_perp.cross(ref_perp_rotated).dot(n_hat),
        ref_perp.dot(ref_perp_rotated));

    if (std::abs(dtheta) < eps) return false;

    Vec3 dx_perp = x1_perp - x0_perp;

    // -------------------------
    // Step 3: Decompose stationary segment
    // -------------------------
    Vec3   q2  = x2 - x_com;
    Vec3   q3  = x3 - x_com;
    Vec3   e   = q3 - q2;
    double h2  = q2.dot(n_hat);
    double en  = e.dot(n_hat);

    Vec3 q2_perp = q2 - h2 * n_hat;
    Vec3 e_perp  = e  - en * n_hat;

    // Build 2D basis
    Vec3 e1, e2;
    buildBasis(n_hat, e1, e2);

    // theta_n and theta_new in 2D frame
    Vec2   ref_2d       = Vec2(ref_perp.dot(e1), ref_perp.dot(e2));
    double theta_n_2d   = std::atan2(ref_2d.y(), ref_2d.x());
    double theta_new_2d = theta_n_2d + dtheta;

    bool   found  = false;
    double best_s = std::numeric_limits<double>::infinity();

    // -------------------------
    // Case A: General swept surface (h0 != h1)
    // -------------------------
    if (std::abs(h1 - h0) > eps) {

        double dh = h1 - h0;

        // For a stationary point at parameter u, the height is h2 + u*en
        // The corresponding t on the moving segment is t(u) = (h2 + u*en - h0) / (h1 - h0)
        // x_perp at t(u): x0_perp + t(u)*dx_perp = x0_perp' + u*beta*dx_perp
        // where x0_perp' = x0_perp + (h2-h0)/dh * dx_perp
        //       beta     = en / dh

        double beta      = en / dh;
        Vec3   x0_perp_p = x0_perp + ((h2 - h0) / dh) * dx_perp; // x_perp at t(0)

        // Quadratic: rho_stationary^2(u) - rho_moving^2(u) = 0
        // rho_stationary^2(u) = ||q2_perp + u*e_perp||^2
        // rho_moving^2(u)     = ||x0_perp' + u*beta*dx_perp||^2

        double A = e_perp.squaredNorm()   - beta * beta * dx_perp.squaredNorm();
        double B = 2.0 * (q2_perp.dot(e_perp) - beta * x0_perp_p.dot(dx_perp));
        double C = q2_perp.squaredNorm()  - x0_perp_p.squaredNorm();

        double disc = B * B - 4.0 * A * C;
        if (disc >= 0.0) {
            double sqrt_disc = std::sqrt(std::max(0.0, disc));

            for (int sign : {-1, 1}) {
                double u;
                if (std::abs(A) < eps) {
                    if (std::abs(B) < eps) {
                        if (std::abs(C) < eps) {
                            // infinite intersections: segment lies on swept surface
                            // only process once (skip second iteration)
                            if (sign == -1) continue;
                            u = 0.0; // take earliest contact
                        } 
                        else {
                            continue; // no intersection
                        }
                    } 
                    else {
                        u = -C / B;
                    }
                } 
                else {
                    u = (-B + sign * sqrt_disc) / (2.0 * A);
                }

                if (u < -eps || u > 1.0 + eps) continue;
                u = std::clamp(u, 0.0, 1.0);

                // corresponding t on moving segment
                double t = (h2 + u * en - h0) / dh;
                if (t < -eps || t > 1.0 + eps) continue;
                t = std::clamp(t, 0.0, 1.0);

                // contact point on moving segment (perpendicular part)
                Vec3 moving_perp = x0_perp + t * dx_perp;
                if (moving_perp.norm() < eps) continue; // on axis, degenerate

                // contact point on stationary segment (perpendicular part)
                Vec3 p_star      = (1.0 - u) * x2 + u * x3;
                Vec3 p_star_perp = (p_star - x_com) - ((p_star - x_com).dot(n_hat)) * n_hat;

                // recover signed angle: from moving_perp to p_star_perp
                double theta_ref  = std::atan2(moving_perp.dot(e2), moving_perp.dot(e1));
                double theta_star = std::atan2(p_star_perp.dot(e2), p_star_perp.dot(e1));
                double dangle     = theta_star - theta_ref;

                // wrap dangle consistent with sign of dtheta
                while (dtheta > 0 && dangle < -eps) dangle += 2.0 * M_PI;
                while (dtheta < 0 && dangle >  eps) dangle -= 2.0 * M_PI;

                double s_cand = dangle / dtheta;
                if (s_cand < -eps || s_cand > 1.0 + eps) continue;
                s_cand = std::clamp(s_cand, 0.0, 1.0);

                if (s_cand < best_s) { best_s = s_cand; found = true; }
            }
        }

    } 
    else {
        // -------------------------
        // Case B: Flat annulus (h0 == h1)
        // -------------------------

        // Find true inner radius
        double r_inner, r_outer;
        r_outer = std::max(r0, r1);

        double dx_perp_sq = dx_perp.squaredNorm();
        if (dx_perp_sq < eps) {
            // both endpoints at same perpendicular position
            r_inner = r0;
        } 
        else {
            double t_star = -x0_perp.dot(dx_perp) / dx_perp_sq;
            if (t_star > 0.0 && t_star < 1.0) {
                r_inner = (x0_perp + t_star * dx_perp).norm();
            } 
            else {
                r_inner = std::min(r0, r1);
            }
        }

        if (std::abs(en) > eps) {
            // Sub-case B1: stationary segment pierces annulus plane
            double u_star = (h0 - h2) / en;
            if (u_star >= -eps && u_star <= 1.0 + eps) {
                u_star = std::clamp(u_star, 0.0, 1.0);

                Vec3   p_star      = (1.0 - u_star) * x2 + u_star * x3;
                Vec3   p_star_perp = (p_star - x_com) - ((p_star - x_com).dot(n_hat)) * n_hat;
                double rho_star    = p_star_perp.norm();

                if (rho_star >= r_inner - eps && rho_star <= r_outer + eps) {
                    Vec2 p_star_2d = project2D(p_star, x_com, e1, e2);
                    Vec2 x0_2d     = project2D(x0,     x_com, e1, e2);
                    Vec2 x1_2d     = project2D(x1,     x_com, e1, e2);
                    Vec2 x_com_2d  = Vec2::Zero();

                    double s_cand = 0.0;
                    bool hit = point_segment_2d_rb_rotation(
                        p_star_2d, x_com_2d, 0.0, -dtheta, x0_2d, x1_2d, s_cand);

                    if (hit) { best_s = s_cand; found = true; }
                }
            }

        } 
        else if (std::abs(h0 - h2) < eps) {
            // Sub-case B2: both segments in same plane
            Vec2 x0_2d    = project2D(x0, x_com, e1, e2);
            Vec2 x1_2d    = project2D(x1, x_com, e1, e2);
            Vec2 x2_2d    = project2D(x2, x_com, e1, e2);
            Vec2 x3_2d    = project2D(x3, x_com, e1, e2);
            Vec2 x_com_2d = Vec2::Zero();

            struct Check { Vec2 x; double tn, tnew; Vec2 seg0, seg1; };
            std::vector<Check> checks = {
                {x0_2d, theta_n_2d, theta_new_2d, x2_2d, x3_2d},
                {x1_2d, theta_n_2d, theta_new_2d, x2_2d, x3_2d},
                {x2_2d, 0.0,       -dtheta,       x0_2d, x1_2d},
                {x3_2d, 0.0,       -dtheta,       x0_2d, x1_2d},
            };

            for (auto& c : checks) {
                double s_cand = 0.0;
                bool hit = point_segment_2d_rb_rotation(
                    c.x, x_com_2d, c.tn, c.tnew, c.seg0, c.seg1, s_cand);
                if (hit && s_cand < best_s) { best_s = s_cand; found = true; }
            }
        }
        // else: parallel planes -> no collision
    }

    if (!found) return false;
    s = best_s;
    return true;
}

bool point_triangle_rb_rotation_ccd(
    const Vec3& x,          // world position of particle at s=0
    const Vec3& x_com,      // center of mass
    const Vec4& q_new,      // quaternion at s=1
    const Vec4& q_n,        // quaternion at s=0
    const Vec3& x2,         // triangle vertex 0
    const Vec3& x3,         // triangle vertex 1
    const Vec3& x4,         // triangle vertex 2
    double& s)
{
    s = 0.0;
    constexpr double eps = 1e-10;

    // -------------------------
    // Step 0: Extract rotation axis from q_rel = q_new * q_n^(-1)
    // -------------------------
    Vec4 q_n_conj = Rigid_Body::ALGEBRA::ConjugateQuaternion(q_n);
    Vec4 q_rel    = Rigid_Body::ALGEBRA::QuaternionMultiply(q_new, q_n_conj);

    Vec3 v_rel(q_rel[1], q_rel[2], q_rel[3]);
    double v_rel_norm = v_rel.norm();
    if (v_rel_norm < eps) return false; // no rotation

    Vec3 n_hat = v_rel / v_rel_norm;

    // -------------------------
    // Step 1: Decompose particle offset (world space at s=0)
    // -------------------------
    Vec3   dx = x - x_com;
    double h  = dx.dot(n_hat);

    Vec3 r_parallel = h * n_hat;
    Vec3 r_perp     = dx - r_parallel;

    if (r_perp.norm() < eps) return false; // particle on rotation axis

    // Signed rotation angle
    Vec3 r_perp_rotated = Rigid_Body::ALGEBRA::QuaternionRotate(q_rel, r_perp);
    double dtheta = std::atan2(
        r_perp.cross(r_perp_rotated).dot(n_hat),
        r_perp.dot(r_perp_rotated));

    if (std::abs(dtheta) < eps) return false; // no rotation

    Vec3 n_cross_r_perp = n_hat.cross(r_perp);

    // -------------------------
    // Step 3: Triangle plane condition
    // -------------------------
    Vec3 e1 = x3 - x2;
    Vec3 e2 = x4 - x2;

    Vec3   n_tri      = e1.cross(e2);
    double n_tri_norm = n_tri.norm();
    if (n_tri_norm < eps) {
        std::cerr << "Warning: degenerate triangle detected in point_triangle_rb_rotation_ccd() when computing surface normal\n";
        return false; // degenerate triangle
    }

    Vec3 n_tri_hat = n_tri / n_tri_norm;

    double A = r_perp.dot(n_tri_hat);
    double B = n_cross_r_perp.dot(n_tri_hat);
    double C = (x_com + r_parallel - x2).dot(n_tri_hat);

    // Precompute barycentric system (triangle is stationary)
    double a11 = e1.dot(e1);
    double a12 = e1.dot(e2);
    double a22 = e2.dot(e2);
    double det = a11 * a22 - a12 * a12;
    if (std::abs(det) < eps){
        std::cerr << "Warning: degenerate triangle detected in point_triangle_rb_rotation_ccd() because of 0 determinant\n";
        return false; // degenerate triangle
    }

    // barycentric inside-check for an arbitrary point 
    auto inside_triangle = [&](const Vec3& p) -> bool {
        Vec3   r_tri = p - x2;
        double b1    = r_tri.dot(e1);
        double b2    = r_tri.dot(e2);

        double alpha   = ( b1 * a22 - b2 * a12) / det;
        double beta    = (-b1 * a12 + b2 * a11) / det;
        double lambda1 = 1.0 - alpha - beta;

        return (alpha >= -eps && beta >= -eps && lambda1 >= -eps);
    };

    double amplitude = std::sqrt(A * A + B * B);

    // -------------------------
    // Degenerate case: circle parallel to triangle plane
    // -------------------------
    if (amplitude < eps) {
        if (std::abs(C) > eps) return false; // constant nonzero distance, never touches

        // In-plane case: particle rotates within the triangle plane.

        // If the particle starts inside the triangle, contact at s = 0
        if (inside_triangle(x)) {
            s = 0.0;
            return true;
        }

        // Otherwise: 2D CCD against the three triangle edges
        Vec3 be1, be2;
        buildBasis(n_hat, be1, be2);

        Vec2 x_2d     = project2D(x,  x_com, be1, be2);
        Vec2 x2_2d    = project2D(x2, x_com, be1, be2);
        Vec2 x3_2d    = project2D(x3, x_com, be1, be2);
        Vec2 x4_2d    = project2D(x4, x_com, be1, be2);
        Vec2 x_com_2d = Vec2::Zero();

        // theta_n is a free gauge choice; only dtheta matters
        double theta_n_2d   = std::atan2(x_2d.y(), x_2d.x());
        double theta_new_2d = theta_n_2d + dtheta;

        double best_s_planar = std::numeric_limits<double>::infinity();

        const std::array<std::pair<Vec2, Vec2>, 3> edges = {{
            {x2_2d, x3_2d},
            {x3_2d, x4_2d},
            {x4_2d, x2_2d},
        }};

        for (const auto& edge : edges) {
            double s_cand = 0.0;
            bool hit = point_segment_2d_rb_rotation(
                x_2d, x_com_2d, theta_n_2d, theta_new_2d,
                edge.first, edge.second, s_cand);
            if (hit && s_cand < best_s_planar) best_s_planar = s_cand;
        }

        if (best_s_planar == std::numeric_limits<double>::infinity()) return false;
        s = best_s_planar;
        return true;
    }

    if (std::abs(C) > amplitude + eps) return false; // circle never reaches plane so no collision

    // -------------------------
    // Step 4: Solve A*cos(theta) + B*sin(theta) + C = 0, theta(s) = s*dtheta
    // -------------------------
    double phi        = std::atan2(B, A);
    double arccos_val = std::acos(std::clamp(-C / amplitude, -1.0, 1.0));

    double theta_candidates[2] = {phi + arccos_val, phi - arccos_val};

    constexpr double two_pi = 2.0 * M_PI;
    double best_s = std::numeric_limits<double>::infinity();

    auto consider_theta = [&](double theta_star) {
        // theta^n = 0 by construction (r_perp is world-space at s=0)
        double s_cand = theta_star / dtheta;
        if (s_cand < -eps || s_cand > 1.0 + eps) return;
        s_cand = std::clamp(s_cand, 0.0, 1.0);

        // Step 5: contact position via Rodrigues
        double theta_s = s_cand * dtheta;
        Vec3 x_s = x_com + r_parallel
                         + r_perp         * std::cos(theta_s)
                         + n_cross_r_perp * std::sin(theta_s);

        // Step 6: barycentric inside-triangle check
        if (!inside_triangle(x_s)) return;

        if (s_cand < best_s) best_s = s_cand;
    };

    for (double theta_base : theta_candidates) {
        // wrap to find first root at/after 0 (dtheta > 0) or at/before 0 (dtheta < 0)
        double k;
        if (dtheta > 0.0) {
            k = std::ceil((0.0 - theta_base - eps) / two_pi);
        } else {
            k = std::floor((0.0 - theta_base + eps) / two_pi);
        }
        consider_theta(theta_base + two_pi * k);
    }

    if (best_s == std::numeric_limits<double>::infinity()) return false;
    s = best_s;
    return true;
}