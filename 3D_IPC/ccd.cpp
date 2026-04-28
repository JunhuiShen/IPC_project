#include "ccd.h"

#include <tight_inclusion/ccd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace {
bool inside_triangle_3d_at(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                           const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3,
                           double t, double eps2);
bool inside_segments_3d_at(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                           const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4,
                           double t, double eps2);
double node_triangle_coplanar_interval(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                                       const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3,
                                       double eps);
double persistent_segment_segment_time(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                                       const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4,
                                       double eps);
}

CCDResult node_triangle_only_one_node_moves(const Vec3& x,  const Vec3& dx, const Vec3& x1, const Vec3& dx1, 
    const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3, double eps) {
    CCDResult result;
    constexpr double eps_inside = 1.0e-10;

    if (inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, 0.0, eps_inside)) {
        result.has_candidate_time = true;
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
            result.coplanar_entire_step = true;
            const double t = node_triangle_coplanar_interval(x, dx, x1, dx1, x2, dx2, x3, dx3, eps);
            if (t < 1.0 && inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, t, eps_inside)) {
                result.has_candidate_time = true;
                result.collision = true;
                result.t = clamp_scalar(t, 0.0, 1.0);
            }
        } else {
            result.parallel_or_no_crossing = true;
        }
        return result;
    }

    const double t = -d / c;
    if (!in_unit_interval(t, eps)) {
        result.parallel_or_no_crossing = true;
        return result;
    }

    result.has_candidate_time = true;
    result.t = clamp_scalar(t, 0.0, 1.0);

    result.collision = inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, result.t, eps_inside);
    return result;
}

CCDResult segment_segment_only_one_node_moves(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps) {
    CCDResult result;
    constexpr double eps_inside = 1.0e-10;

    if (inside_segments_3d_at(x1, dx1, x2, Vec3::Zero(), x3, Vec3::Zero(), x4, Vec3::Zero(), 0.0, eps_inside)) {
        result.has_candidate_time = true;
        result.collision = true;
        result.t = 0.0;
        return result;
    }

    // Coplanarity condition f(t) = dot((x2-x1-t*dx1) x (x4-x3), (x3-x1-t*dx1)) = 0
    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const Vec3 c0 = x3 - x1;

    const double d = (a.cross(b)).dot(c0);
    const double c = -(a.cross(b)).dot(dx1) - (dx1.cross(b)).dot(c0);

    if (nearly_zero(c, eps)) {
        if (nearly_zero(d, eps)) {
            result.coplanar_entire_step = true;
            const double t = persistent_segment_segment_time(
                x1, dx1, x2, Vec3::Zero(), x3, Vec3::Zero(), x4, Vec3::Zero(), eps);
            if (t < 1.0 && inside_segments_3d_at(
                    x1, dx1, x2, Vec3::Zero(), x3, Vec3::Zero(), x4, Vec3::Zero(), t, eps_inside)) {
                result.has_candidate_time = true;
                result.collision = true;
                result.t = clamp_scalar(t, 0.0, 1.0);
            }
        } else {
            result.parallel_or_no_crossing = true;
        }
        return result;
    }

    const double t = -d / c;
    if (!in_unit_interval(t, eps)) {
        result.parallel_or_no_crossing = true;
        return result;
    }

    result.has_candidate_time = true;
    result.t = clamp_scalar(t, 0.0, 1.0);
    result.collision = inside_segments_3d_at(
        x1, dx1, x2, Vec3::Zero(), x3, Vec3::Zero(), x4, Vec3::Zero(), result.t, eps_inside);
    return result;
}


// =============================================================================
// Linear CCD helpers
// =============================================================================
namespace {

// Polynomial solvers (linear CCD coplanar fallback only)

void solve_linear_all(double c, double d, SmallRoots& roots, double eps) {
    double s = std::max(1.0, std::fabs(d));
    if (nearly_zero(c, eps * s)) return;
    add_root(roots, -d / c, eps);
}

void solve_quadratic_all(double a, double b, double c, SmallRoots& roots, double eps) {
    double s = max_abs_value_among_four_numbers(a, b, c, 0.0);
    if (nearly_zero(a, eps * s)) {
        solve_linear_all(b, c, roots, eps);
        return;
    }

    double D = b * b - 4.0 * a * c;
    const double coeff_scale = std::max({std::fabs(a), std::fabs(b), std::fabs(c)});
    double tol = eps * coeff_scale * coeff_scale * 16.0;
    if (D < -tol) return;
    if (std::fabs(D) <= tol) {
        add_root(roots, -b / (2.0 * a), eps);
        return;
    }

    double sqrtD = std::sqrt(std::max(0.0, D));
    double signb = (b >= 0.0) ? 1.0 : -1.0;
    double q = -0.5 * (b + signb * sqrtD);
    add_root(roots, q / a, eps);
    if (!nearly_zero(q, eps * s)) add_root(roots, c / q, eps);
}

// Projection helpers (linear CCD coplanar fallback only)

Vec2 project_drop_axis(const Vec3& v, int drop_axis) {
    if (drop_axis == 0) return Vec2(v.y(), v.z());
    if (drop_axis == 1) return Vec2(v.x(), v.z());
    return Vec2(v.x(), v.y());
}

int dominant_drop_axis(const Vec3& n0, const Vec3& n1) {
    Vec3 n(std::fabs(n0.x()) + std::fabs(n1.x()), std::fabs(n0.y()) + std::fabs(n1.y()),  std::fabs(n0.z()) + std::fabs(n1.z()));
    if (n.x() >= n.y() && n.x() >= n.z()) return 0;
    if (n.y() >= n.x() && n.y() >= n.z()) return 1;
    return 2;
}

// Geometric containment tests (linear CCD only)

bool inside_triangle_3d_at(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                           const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3,
                           double t, double eps2) {
    Vec3 xt = x + dx * t;
    Vec3 a = x1 + dx1 * t;
    Vec3 b = x2 + dx2 * t;
    Vec3 c = x3 + dx3 * t;
    Vec3 e1 = b - a;
    Vec3 e2 = c - a;
    Vec3 w = xt - a;
    Vec3 n = e1.cross(e2);
    double n2 = n.squaredNorm();
    double d11 = e1.dot(e1);
    double d12 = e1.dot(e2);
    double d22 = e2.dot(e2);
    const double tri_scale = std::sqrt(d11) + std::sqrt(d22);
    const double eps_len = std::max(eps2, 1.0e-8 * tri_scale);
    if (n2 <= eps_len * eps_len) return false;
    if (std::fabs(n.dot(w)) > eps_len * std::sqrt(n2)) return false;

    double b1 = e1.dot(w);
    double b2 = e2.dot(w);
    double det = d11 * d22 - d12 * d12;
    if (std::fabs(det) <= eps_len * std::max({1.0, d11, d22})) return false;

    double lam2 = (d22 * b1 - d12 * b2) / det;
    double lam3 = (d11 * b2 - d12 * b1) / det;
    double lam1 = 1.0 - lam2 - lam3;
    return lam1 >= -1e-8 && lam2 >= -1e-8 && lam3 >= -1e-8;
}

bool inside_segments_3d_at(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                           const Vec3& x3, const Vec3& dx3,  const Vec3& x4, const Vec3& dx4,
                           double t, double eps2) {
    Vec3 a0 = x1 + dx1 * t, a1 = x2 + dx2 * t;
    Vec3 b0 = x3 + dx3 * t, b1 = x4 + dx4 * t;

    Vec3 u = a1 - a0;
    Vec3 v = b1 - b0;
    Vec3 w = b0 - a0;
    Vec3 n = u.cross(v);
    double n2 = n.squaredNorm();
    const double len_u = u.norm();
    const double len_v = v.norm();
    const double eps_len = std::max(eps2, 1.0e-8 * (len_u + len_v));

    if (n2 <= eps_len * eps_len) {
        Vec3 crossw = w.cross(u);
        if (crossw.squaredNorm() > eps_len * eps_len * std::max(1.0, u.squaredNorm())) return false;

        Vec3 d0 = a1 - a0;
        double ax = std::fabs(d0.x()), ay = std::fabs(d0.y()), az = std::fabs(d0.z());
        int axis = (ax >= ay && ax >= az) ? 0 : (ay >= ax && ay >= az ? 1 : 2);
        auto comp1d = [&](const Vec3& p) { return axis == 0 ? p.x() : (axis == 1 ? p.y() : p.z()); };

        double A = comp1d(a0), B = comp1d(a1);
        double C = comp1d(b0), D = comp1d(b1);
        if (A > B) std::swap(A, B);
        if (C > D) std::swap(C, D);
        return A <= D + 1e-8 && C <= B + 1e-8;
    }

    if (std::fabs(w.dot(n)) > eps_len * std::sqrt(n2)) return false;
    double s = w.cross(v).dot(n) / n2;
    double upar = w.cross(u).dot(n) / n2;
    Vec3 pa = a0 + u * s;
    Vec3 pb = b0 + v * upar;
    if ((pa - pb).squaredNorm() > eps_len * eps_len) return false;
    return s >= -1e-8 && s <= 1.0 + 1e-8 && upar >= -1e-8 && upar <= 1.0 + 1e-8;
}

// Degenerate-case handlers (linear CCD coplanar fallback only)

bool point_in_triangle_projected(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c) {
    double e0 = cross_product_in_2d(b - a, p - a);
    double e1 = cross_product_in_2d(c - b, p - b);
    double e2 = cross_product_in_2d(a - c, p - c);
    return (e0 >= -1e-10 && e1 >= -1e-10 && e2 >= -1e-10) || (e0 <=  1e-10 && e1 <=  1e-10 && e2 <=  1e-10);
}

double node_triangle_coplanar_interval(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                                       const Vec3& x2, const Vec3& dx2,  const Vec3& x3, const Vec3& dx3,
                                       double eps) {
    Vec3 n0 = (x2 - x1).cross(x3 - x1);
    Vec3 n1 = (x2 + dx2 - x1 - dx1).cross(x3 + dx3 - x1 - dx1);
    int drop = dominant_drop_axis(n0, n1);

    // Under one-moving-node, the t^2 term cross_2d(du, dv) is always 0
    // (one of du/dv is zero, or du == dv up to sign), so the boundary
    // equation reduces to linear in t.
    auto coeff_boundary_segment = [&](const Vec2& a0, const Vec2& da,  const Vec2& b0, const Vec2& db,
                                      const Vec2& p0, const Vec2& dp,  double out[2]) {
        Vec2 u0 = b0 - a0;
        Vec2 du = db - da;
        Vec2 v0 = p0 - a0;
        Vec2 dv = dp - da;
        out[1] = cross_product_in_2d(du, v0) + cross_product_in_2d(u0, dv);
        out[0] = cross_product_in_2d(u0, v0);
    };

    Vec2 p0 = project_drop_axis(x, drop), dp = project_drop_axis(dx, drop);
    Vec2 a0 = project_drop_axis(x1, drop), da = project_drop_axis(dx1, drop);
    Vec2 b0 = project_drop_axis(x2, drop), db = project_drop_axis(dx2, drop);
    Vec2 c0 = project_drop_axis(x3, drop), dc = project_drop_axis(dx3, drop);

    if (point_in_triangle_projected(p0, a0, b0, c0)) return 0.0;

    SmallRoots roots;
    double coeffs[2];
    coeff_boundary_segment(a0, da, b0, db, p0, dp, coeffs); solve_linear_all(coeffs[1], coeffs[0], roots, eps);
    coeff_boundary_segment(b0, db, c0, dc, p0, dp, coeffs); solve_linear_all(coeffs[1], coeffs[0], roots, eps);
    coeff_boundary_segment(c0, dc, a0, da, p0, dp, coeffs); solve_linear_all(coeffs[1], coeffs[0], roots, eps);

    std::sort(roots.begin(), roots.end());
    for (double t : roots) {
        Vec2 pt = p0 + dp * t;
        Vec2 at = a0 + da * t;
        Vec2 bt = b0 + db * t;
        Vec2 ct = c0 + dc * t;
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
        double t_after = std::min(1.0, r + 1e-9);
        if (h(t_after) <= eps) return r;
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

    double o1 = orient(a, b, c), o2 = orient(a, b, d);
    double o3 = orient(c, d, a), o4 = orient(c, d, b);

    auto sgn = [](double v) { return (v > 1e-10) - (v < -1e-10); };
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
    if (n0.squaredNorm() <= eps * eps && n1.squaredNorm() <= eps * eps) {
        return collinear_segment_overlap_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4, eps);
    }
    int drop = dominant_drop_axis(n0, n1);

    Vec2 a0 = project_drop_axis(x1, drop), da = project_drop_axis(dx1, drop);
    Vec2 b0 = project_drop_axis(x2, drop), db = project_drop_axis(dx2, drop);
    Vec2 c0 = project_drop_axis(x3, drop), dc = project_drop_axis(dx3, drop);
    Vec2 d0 = project_drop_axis(x4, drop), dd = project_drop_axis(dx4, drop);

    if (segments_intersect_projected(a0, b0, c0, d0)) return 0.0;

    // Same one-moving-node argument as in node_triangle_coplanar_interval:
    // the t^2 coefficient cross_2d(du, dv) always vanishes here, so the
    // 2D orientation equation is linear in t.
    auto orient_coeff = [](const Vec2& p0, const Vec2& dp,
                           const Vec2& q0, const Vec2& dq,
                           const Vec2& r0, const Vec2& dr,
                           double out[2]) {
        Vec2 u0 = q0 - p0;
        Vec2 du = dq - dp;
        Vec2 v0 = r0 - p0;
        Vec2 dv = dr - dp;
        out[1] = cross_product_in_2d(du, v0) + cross_product_in_2d(u0, dv);
        out[0] = cross_product_in_2d(u0, v0);
    };

    SmallRoots roots;
    double coeffs[2];
    orient_coeff(a0, da, b0, db, c0, dc, coeffs); solve_linear_all(coeffs[1], coeffs[0], roots, eps);
    orient_coeff(a0, da, b0, db, d0, dd, coeffs); solve_linear_all(coeffs[1], coeffs[0], roots, eps);
    orient_coeff(c0, dc, d0, dd, a0, da, coeffs); solve_linear_all(coeffs[1], coeffs[0], roots, eps);
    orient_coeff(c0, dc, d0, dd, b0, db, coeffs); solve_linear_all(coeffs[1], coeffs[0], roots, eps);

    std::sort(roots.begin(), roots.end());
    for (double t : roots) {
        Vec2 at = a0 + da * t, bt = b0 + db * t, ct = c0 + dc * t, dt2 = d0 + dd * t;
        if (segments_intersect_projected(at, bt, ct, dt2)) return t;
    }

    return collinear_segment_overlap_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4, eps);
}

} 

// General CCD: backed by Tight-Inclusion CCD [Wang et al. 2021].
// https://github.com/Continuous-Collision-Detection/Tight-Inclusion
//
// TICCD is a conservative inclusion-based root finder. Returning false is a
// guarantee of no collision in the step. Returning true may be a false
// positive, but only when the closest distance between the two primitives is
// at most (tolerance + ms + err).
//
// Parameters used here (matching TICCD's reference example, app/main.cpp):
//   ms          - minimum separation distance (>= 0). A collision is
//                 guaranteed to be reported if the pair gets closer than ms.
//   tolerance   - target solver precision (max edge length of the inclusion
//                 box). 1e-6 is the upstream-recommended value.
//   max_itr     - early-termination cap on inclusion-tree iterations. 1e6 is
//                 the upstream-recommended cap; -1 disables it.
//   err         - per-axis numerical filter compensating for floating-point
//                 error in the inclusion function. (-1,-1,-1) tells TICCD to
//                 compute it from the query alone; for tighter filters, call
//                 ticcd::get_numerical_error() once with the simulation's
//                 scene AABB and reuse the result.
//   t_max       - upper bound of the time interval to check (in [0, 1]).
//   ccd_method  - inclusion-tree traversal; default/recommended
//                 BREADTH_FIRST_SEARCH.
//   no_zero_toi - when ms > 0, set true to keep refining instead of letting
//                 toi snap to 0; default false.
namespace {

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

}  // namespace

double node_triangle_general_ccd(const Vec3& x,  const Vec3& dx,
                                 const Vec3& x1, const Vec3& dx1,
                                 const Vec3& x2, const Vec3& dx2,
                                 const Vec3& x3, const Vec3& dx3) {
    const ticcd::Array3 err  = ticcd::Array3::Constant(-1.0);
    double  toi  = std::numeric_limits<double>::infinity();
    double  output_tolerance = kTiccdTolerance;

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
    const ticcd::Array3 err  = ticcd::Array3::Constant(-1.0);
    double  toi  = std::numeric_limits<double>::infinity();
    double  output_tolerance = kTiccdTolerance;

    const bool collision = ticcd::edgeEdgeCCD(
        x1,       x2,       x3,       x4,
        x1 + dx1, x2 + dx2, x3 + dx3, x4 + dx4,
        err, kTiccdMinSeparation, toi, kTiccdTolerance, /*t_max=*/1.0,
        kTiccdMaxIter, output_tolerance, kTiccdNoZeroToi, kTiccdMethod);

    return collision ? clamp_toi(toi) : 1.0;
}
