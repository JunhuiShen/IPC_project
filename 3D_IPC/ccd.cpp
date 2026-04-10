#include "ccd.h"

#include <algorithm>
#include <array>

CCDResult node_triangle_only_one_node_moves(const Vec3& x,  const Vec3& dx, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps) {
    CCDResult result;

    const Vec3 n = (x2 - x1).cross(x3 - x1);
    const double d = n.dot(x - x1);
    const double c = n.dot(dx);

    if (nearly_zero(c, eps)) {
        if (nearly_zero(d, eps)) {
            result.coplanar_entire_step = true;
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

    const Vec3 p = point_at_linear_step(x, dx, result.t);
    result.collision = point_in_triangle_on_plane(p, x1, x2, x3, eps);
    return result;
}

CCDResult segment_segment_only_one_node_moves(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps) {
    CCDResult result;

    // Coplanarity condition f(t) = dot((x2-x1-t*dx1) x (x4-x3), (x3-x1-t*dx1)) = 0
    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const Vec3 c0 = x3 - x1;

    const double d = (a.cross(b)).dot(c0);
    const double c = -(a.cross(b)).dot(dx1) - (dx1.cross(b)).dot(c0);

    if (nearly_zero(c, eps)) {
        if (nearly_zero(d, eps)) {
            result.coplanar_entire_step = true;
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

    const Vec3 x1_star = point_at_linear_step(x1, dx1, result.t);
    double s = 0.0;
    double u = 0.0;
    if (!segment_segment_parameters_if_not_parallel(x1_star, x2, x3, x4, s, u, eps)) {
        return result;
    }

    result.collision = in_unit_interval(s, eps) && in_unit_interval(u, eps);
    return result;
}


// General CCD: all vertices may move, returns earliest TOI in [0,1].
namespace {

static constexpr double PI = 3.14159265358979323846;

//  Polynomial solvers

void solve_linear_all(double c, double d, SmallRoots& roots, double eps) {
    double s = std::max({1.0, std::fabs(c), std::fabs(d)});
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
    double tol = eps * s * s * 16.0;
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

void solve_cubic_all(double a, double b, double c, double d, SmallRoots& roots, double eps) {
    double s = max_abs_value_among_four_numbers(a, b, c, d);
    if (nearly_zero(a, eps * s)) {
        solve_quadratic_all(b, c, d, roots, eps);
        return;
    }

    double inv_a = 1.0 / a;
    double ba = b * inv_a;
    double ca = c * inv_a;
    double da = d * inv_a;
    double shift = -ba / 3.0;
    double p = ca - (ba * ba) / 3.0;
    double q = (2.0 / 27.0) * ba * ba * ba - (ba * ca) / 3.0 + da;
    double Delta = -4.0 * p * p * p - 27.0 * q * q;
    double dtol = eps * std::max({1.0, std::fabs(p * p * p), std::fabs(q * q)}) * 64.0;

    if (Delta > dtol) {
        double m = 2.0 * std::sqrt(std::max(0.0, -p / 3.0));
        if (nearly_zero(m, eps)) {
            add_root(roots, shift, eps);
        } else {
            double w = clamp_scalar((3.0 * q) / (p * m), -1.0, 1.0);
            double theta = std::acos(w) / 3.0;
            for (int k = 0; k < 3; ++k) {
                double sk = m * std::cos(theta - 2.0 * PI * k / 3.0);
                add_root(roots, sk + shift, eps);
            }
        }
    } else if (Delta < -dtol) {
        double Disc = q * q / 4.0 + p * p * p / 27.0;
        double sqrtD = std::sqrt(std::max(0.0, Disc));
        double C = std::cbrt(-q / 2.0 + sqrtD);
        double s0 = nearly_zero(C, eps) ? 0.0 : (C - p / (3.0 * C));
        add_root(roots, s0 + shift, eps);
    } else {
        if (nearly_zero(p, eps * std::max(1.0, std::fabs(p))) && nearly_zero(q, eps * std::max(1.0, std::fabs(q)))) {
            add_root(roots, shift, eps);
        } else if (!nearly_zero(p, eps * std::max(1.0, std::fabs(p)))) {
            add_root(roots,  3.0 * q / p + shift, eps);
            add_root(roots, -1.5 * q / p + shift, eps);
        } else {
            add_root(roots, shift, eps);
        }
    }
}

// Projection helpers

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

// Geometric containment tests

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
    if (n2 <= eps2 * eps2) return false;
    if (std::fabs(n.dot(w)) > eps2 * std::sqrt(n2)) return false;

    double d11 = e1.dot(e1);
    double d12 = e1.dot(e2);
    double d22 = e2.dot(e2);
    double b1 = e1.dot(w);
    double b2 = e2.dot(w);
    double det = d11 * d22 - d12 * d12;
    if (std::fabs(det) <= eps2 * std::max({1.0, d11, d22})) return false;

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

    if (n2 <= eps2 * eps2) {
        Vec3 crossw = w.cross(u);
        if (crossw.squaredNorm() > 1e-16) return false;

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

    if (std::fabs(w.dot(n)) > eps2 * std::sqrt(n2)) return false;
    double s = w.cross(v).dot(n) / n2;
    double upar = w.cross(u).dot(n) / n2;
    Vec3 pa = a0 + u * s;
    Vec3 pb = b0 + v * upar;
    if ((pa - pb).squaredNorm() > eps2 * eps2) return false;
    return s >= -1e-8 && s <= 1.0 + 1e-8 && upar >= -1e-8 && upar <= 1.0 + 1e-8;
}

// Degenerate-case handlers

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

    auto coeff_edge = [&](const Vec2& a0, const Vec2& da,  const Vec2& b0, const Vec2& db,
                          const Vec2& p0, const Vec2& dp,  double out[3]) {
        Vec2 u0 = b0 - a0;
        Vec2 du = db - da;
        Vec2 v0 = p0 - a0;
        Vec2 dv = dp - da;
        out[2] = cross_product_in_2d(du, dv);
        out[1] = cross_product_in_2d(du, v0) + cross_product_in_2d(u0, dv);
        out[0] = cross_product_in_2d(u0, v0);
    };

    Vec2 p0 = project_drop_axis(x, drop), dp = project_drop_axis(dx, drop);
    Vec2 a0 = project_drop_axis(x1, drop), da = project_drop_axis(dx1, drop);
    Vec2 b0 = project_drop_axis(x2, drop), db = project_drop_axis(dx2, drop);
    Vec2 c0 = project_drop_axis(x3, drop), dc = project_drop_axis(dx3, drop);

    if (point_in_triangle_projected(p0, a0, b0, c0)) return 0.0;

    SmallRoots roots;
    double coeffs[3];
    coeff_edge(a0, da, b0, db, p0, dp, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots, eps);
    coeff_edge(b0, db, c0, dc, p0, dp, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots, eps);
    coeff_edge(c0, dc, a0, da, p0, dp, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots, eps);

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

    auto orient_coeff = [](const Vec2& p0, const Vec2& dp,
                           const Vec2& q0, const Vec2& dq,
                           const Vec2& r0, const Vec2& dr,
                           double out[3]) {
        Vec2 u0 = q0 - p0;
        Vec2 du = dq - dp;
        Vec2 v0 = r0 - p0;
        Vec2 dv = dr - dp;
        out[2] = cross_product_in_2d(du, dv);
        out[1] = cross_product_in_2d(du, v0) + cross_product_in_2d(u0, dv);
        out[0] = cross_product_in_2d(u0, v0);
    };

    SmallRoots roots;
    double coeffs[3];
    orient_coeff(a0, da, b0, db, c0, dc, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots, eps);
    orient_coeff(a0, da, b0, db, d0, dd, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots, eps);
    orient_coeff(c0, dc, d0, dd, a0, da, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots, eps);
    orient_coeff(c0, dc, d0, dd, b0, db, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots, eps);

    std::sort(roots.begin(), roots.end());
    for (double t : roots) {
        Vec2 at = a0 + da * t, bt = b0 + db * t, ct = c0 + dc * t, dt2 = d0 + dd * t;
        if (segments_intersect_projected(at, bt, ct, dt2)) return t;
    }

    return collinear_segment_overlap_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4, eps);
}

} 

// General CCD functions
double node_triangle_general_ccd(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                                 const Vec3& x2, const Vec3& dx2,  const Vec3& x3, const Vec3& dx3,
                                 double eps1, double eps2) {
    if (inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, 0.0, eps2)) return 0.0;

    Vec3 p0 = x2 - x1, dp = dx2 - dx1;
    Vec3 q0 = x3 - x1, dq = dx3 - dx1;
    Vec3 r0 = x - x1, dr = dx - dx1;

    Vec3 p0xq0 = p0.cross(q0);
    Vec3 dpxq0 = dp.cross(q0);
    Vec3 p0xdq = p0.cross(dq);
    Vec3 dpxdq = dp.cross(dq);

    double d = p0xq0.dot(r0);
    double c = dpxq0.dot(r0) + p0xdq.dot(r0) + p0xq0.dot(dr);
    double b = dpxdq.dot(r0) + dpxq0.dot(dr) + p0xdq.dot(dr);
    double a = dpxdq.dot(dr);

    SmallRoots roots;
    solve_cubic_all(a, b, c, d, roots, eps1);
    std::sort(roots.begin(), roots.end());
    for (double t : roots) {
        if (inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, t, eps2)) return t;
    }

    double s = max_abs_value_among_four_numbers(a, b, c, d);
    if (nearly_zero(a, eps1 * s) && nearly_zero(b, eps1 * s) && nearly_zero(c, eps1 * s) && nearly_zero(d, eps1 * s)) {
        double t = node_triangle_coplanar_interval(x, dx, x1, dx1, x2, dx2, x3, dx3, eps1);
        if (t < 1.0 && inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, t, eps2)) return t;
    }

    return 1.0;
}

double segment_segment_general_ccd(const Vec3& x1, const Vec3& dx1,  const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4,
                                   double eps1, double eps2) {
    if (inside_segments_3d_at(x1, dx1, x2, dx2, x3, dx3, x4, dx4, 0.0, eps2)) return 0.0;

    Vec3 p0 = x2 - x1, dp = dx2 - dx1;
    Vec3 q0 = x4 - x3, dq = dx4 - dx3;
    Vec3 r0 = x3 - x1, dr = dx3 - dx1;

    Vec3 p0xq0 = p0.cross(q0);
    Vec3 dpxq0 = dp.cross(q0);
    Vec3 p0xdq = p0.cross(dq);
    Vec3 dpxdq = dp.cross(dq);

    double d = p0xq0.dot(r0);
    double c = dpxq0.dot(r0) + p0xdq.dot(r0) + p0xq0.dot(dr);
    double b = dpxdq.dot(r0) + dpxq0.dot(dr) + p0xdq.dot(dr);
    double a = dpxdq.dot(dr);

    SmallRoots roots;
    solve_cubic_all(a, b, c, d, roots, eps1);
    std::sort(roots.begin(), roots.end());
    for (double t : roots) {
        if (inside_segments_3d_at(x1, dx1, x2, dx2, x3, dx3, x4, dx4, t, eps2)) return t;
    }

    double s = max_abs_value_among_four_numbers(a, b, c, d);
    if (nearly_zero(a, eps1 * s) && nearly_zero(b, eps1 * s) && nearly_zero(c, eps1 * s) && nearly_zero(d, eps1 * s)) {
        double t = persistent_segment_segment_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4, eps1);
        if (t < 1.0 && inside_segments_3d_at(x1, dx1, x2, dx2, x3, dx3, x4, dx4, t, eps2)) return t;
    }

    return 1.0;
}
