#ifndef CCD_H
#define CCD_H

#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <limits>

namespace ccd {

    using Vec3 = Eigen::Vector3d;
    using Vec2 = Eigen::Vector2d;

    inline Vec2 make_vec2(double x, double y) {
        return Vec2(x, y);
    }

    inline Vec3 make_vec3(double x, double y, double z) {
        return Vec3(x, y, z);
    }

    static constexpr double ccd_infinity  = std::numeric_limits<double>::infinity();
    static constexpr double ccd_safety    = 0.9;
    static constexpr double ccd_coeff_eps = 1e-12;
    static constexpr double ccd_geom_eps  = 1e-10;

    inline double safe_step(double t_hit) {
        if (t_hit > 1.0) return 1.0;
        return ccd_safety * std::max(0.0, t_hit);
    }

    double node_segment_2d(const Vec2& x, const Vec2& dx,
                           const Vec2& x1, const Vec2& dx1,
                           const Vec2& x2, const Vec2& dx2);

    double node_triangle_3d(const Vec3& x, const Vec3& dx,
                            const Vec3& x1, const Vec3& dx1,
                            const Vec3& x2, const Vec3& dx2,
                            const Vec3& x3, const Vec3& dx3);

    double segment_segment_3d(const Vec3& x1, const Vec3& dx1,
                              const Vec3& x2, const Vec3& dx2,
                              const Vec3& x3, const Vec3& dx3,
                              const Vec3& x4, const Vec3& dx4);

} // namespace ccd

#endif

#include <array>
#include <vector>

namespace ccd {
    namespace {
        inline double cross2(const Vec2& a, const Vec2& b) {
            return a.x() * b.y() - a.y() * b.x();
        }

        static constexpr double PI = 3.14159265358979323846;

        inline double clamp01(double t) {
            return std::max(0.0, std::min(1.0, t));
        }

        inline bool in_unit_interval(double t, double eps = 1e-12) {
            return t >= -eps && t <= 1.0 + eps;
        }

        inline double filter_root(double t) {
            return in_unit_interval(t) ? clamp01(t) : ccd_infinity;
        }

        inline void add_root(std::vector<double>& roots, double t) {
            t = filter_root(t);
            if (!std::isfinite(t)) return;
            for (double r : roots) {
                if (std::fabs(r - t) <= 1e-9) return;
            }
            roots.push_back(t);
        }

        inline double scale4(double a, double b, double c, double d) {
            return std::max({1.0, std::fabs(a), std::fabs(b), std::fabs(c), std::fabs(d)});
        }

        inline bool near_zero(double v, double s = 1.0) {
            return std::fabs(v) <= ccd_coeff_eps * s;
        }

        void solve_linear_all(double c, double d, std::vector<double>& roots) {
            double s = std::max({1.0, std::fabs(c), std::fabs(d)});
            if (near_zero(c, s)) return;
            add_root(roots, -d / c);
        }

        void solve_quadratic_all(double a, double b, double c, std::vector<double>& roots) {
            double s = scale4(a, b, c, 0.0);
            if (near_zero(a, s)) {
                solve_linear_all(b, c, roots);
                return;
            }

            double D = b * b - 4.0 * a * c;
            double tol = ccd_coeff_eps * s * s * 16.0;
            if (D < -tol) return;
            if (std::fabs(D) <= tol) {
                add_root(roots, -b / (2.0 * a));
                return;
            }

            double sqrtD = std::sqrt(std::max(0.0, D));
            double signb = (b >= 0.0) ? 1.0 : -1.0;
            double q = -0.5 * (b + signb * sqrtD);
            add_root(roots, q / a);
            if (!near_zero(q, s)) add_root(roots, c / q);
        }

        void solve_cubic_all(double a, double b, double c, double d, std::vector<double>& roots) {
            double s = scale4(a, b, c, d);
            if (near_zero(a, s)) {
                solve_quadratic_all(b, c, d, roots);
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
            double dtol = ccd_coeff_eps * std::max({1.0, std::fabs(p * p * p), std::fabs(q * q)}) * 64.0;

            if (Delta > dtol) {
                double m = 2.0 * std::sqrt(std::max(0.0, -p / 3.0));
                if (near_zero(m)) {
                    add_root(roots, shift);
                } else {
                    double w = (3.0 * q) / (p * m);
                    w = std::max(-1.0, std::min(1.0, w));
                    double theta = std::acos(w) / 3.0;
                    for (int k = 0; k < 3; ++k) {
                        double sk = m * std::cos(theta - 2.0 * PI * k / 3.0);
                        add_root(roots, sk + shift);
                    }
                }
            } else if (Delta < -dtol) {
                double D = q * q / 4.0 + p * p * p / 27.0;
                double sqrtD = std::sqrt(std::max(0.0, D));
                double C = std::cbrt(-q / 2.0 + sqrtD);
                double s0 = near_zero(C) ? 0.0 : (C - p / (3.0 * C));
                add_root(roots, s0 + shift);
            } else {
                if (near_zero(p, std::max(1.0, std::fabs(p))) &&
                    near_zero(q, std::max(1.0, std::fabs(q)))) {
                    add_root(roots, shift);
                } else if (!near_zero(p, std::max(1.0, std::fabs(p)))) {
                    // Repeated-root case for depressed cubic:
                    // double root = -3q/(2p)
                    // simple root =  3q/p
                    add_root(roots,  3.0 * q / p + shift);
                    add_root(roots, -1.5 * q / p + shift);
                } else {
                    add_root(roots, shift);
                }
            }
        }

        Vec2 project_drop_axis(const Vec3& v, int drop_axis) {
            if (drop_axis == 0) return make_vec2(v.y(), v.z());
            if (drop_axis == 1) return make_vec2(v.x(), v.z());
            return make_vec2(v.x(), v.y());
        }

        int dominant_drop_axis(const Vec3& n0, const Vec3& n1) {
            Vec3 n = make_vec3(std::fabs(n0.x()) + std::fabs(n1.x()),
                               std::fabs(n0.y()) + std::fabs(n1.y()),
                               std::fabs(n0.z()) + std::fabs(n1.z()));
            if (n.x() >= n.y() && n.x() >= n.z()) return 0;
            if (n.y() >= n.x() && n.y() >= n.z()) return 1;
            return 2;
        }

        bool inside_segment_2d_at(const Vec2& x, const Vec2& dx,
                                  const Vec2& x1, const Vec2& dx1,
                                  const Vec2& x2, const Vec2& dx2,
                                  double t) {
            Vec2 xt = x + dx * t;
            Vec2 a = x1 + dx1 * t;
            Vec2 b = x2 + dx2 * t;
            Vec2 e = b - a;
            double len2 = e.dot(e);
            if (len2 <= ccd_geom_eps * ccd_geom_eps) {
                return (xt - a).dot(xt - a) <= ccd_geom_eps * ccd_geom_eps;
            }
            double area = cross2(e, xt - a);
            if (std::fabs(area) > ccd_geom_eps * std::sqrt(len2)) return false;
            double s = (xt - a).dot(e) / len2;
            return s >= -1e-8 && s <= 1.0 + 1e-8;
        }

        double earliest_scalar_between(double p0, double dp,
                                       double a0, double da,
                                       double b0, double db) {
            auto h = [&](double t) {
                double ra = (p0 - a0) + t * (dp - da);
                double rb = (p0 - b0) + t * (dp - db);
                return ra * rb;
            };

            if (h(0.0) <= ccd_geom_eps) return 0.0;

            double qa = (dp - da) * (dp - db);
            double qb = (p0 - a0) * (dp - db) + (p0 - b0) * (dp - da);
            double qc = (p0 - a0) * (p0 - b0);

            std::vector<double> roots;
            solve_quadratic_all(qa, qb, qc, roots);
            std::sort(roots.begin(), roots.end());
            for (double r : roots) {
                if (h(r) <= ccd_geom_eps) return r;
                double t_after = std::min(1.0, r + 1e-9);
                if (h(t_after) <= ccd_geom_eps) return r;
            }
            return ccd_infinity;
        }

        bool inside_triangle_3d_at(const Vec3& x, const Vec3& dx,
                                   const Vec3& x1, const Vec3& dx1,
                                   const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3,
                                   double t) {
            Vec3 xt = x + dx * t;
            Vec3 a = x1 + dx1 * t;
            Vec3 b = x2 + dx2 * t;
            Vec3 c = x3 + dx3 * t;
            Vec3 e1 = b - a;
            Vec3 e2 = c - a;
            Vec3 w = xt - a;
            Vec3 n = e1.cross(e2);
            double n2 = n.squaredNorm();
            if (n2 <= ccd_geom_eps * ccd_geom_eps) return false;
            if (std::fabs(n.dot(w)) > ccd_geom_eps * std::sqrt(n2)) return false;

            double d11 = e1.dot(e1);
            double d12 = e1.dot(e2);
            double d22 = e2.dot(e2);
            double b1 = e1.dot(w);
            double b2 = e2.dot(w);
            double det = d11 * d22 - d12 * d12;
            if (std::fabs(det) <= ccd_geom_eps * std::max({1.0, d11, d22})) return false;

            double lam2 = (d22 * b1 - d12 * b2) / det;
            double lam3 = (d11 * b2 - d12 * b1) / det;
            double lam1 = 1.0 - lam2 - lam3;
            return lam1 >= -1e-8 && lam2 >= -1e-8 && lam3 >= -1e-8;
        }

        bool point_in_triangle_projected(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c) {
            double e0 = cross2(b - a, p - a);
            double e1 = cross2(c - b, p - b);
            double e2 = cross2(a - c, p - c);
            return (e0 >= -1e-10 && e1 >= -1e-10 && e2 >= -1e-10) ||
                   (e0 <=  1e-10 && e1 <=  1e-10 && e2 <=  1e-10);
        }

        double node_triangle_coplanar_interval(const Vec3& x, const Vec3& dx,
                                               const Vec3& x1, const Vec3& dx1,
                                               const Vec3& x2, const Vec3& dx2,
                                               const Vec3& x3, const Vec3& dx3) {
            Vec3 n0 = (x2 - x1).cross(x3 - x1);
            Vec3 n1 = (x2 + dx2 - x1 - dx1).cross(x3 + dx3 - x1 - dx1);
            int drop = dominant_drop_axis(n0, n1);

            auto coeff_edge = [&](const Vec2& a0, const Vec2& da,
                                  const Vec2& b0, const Vec2& db,
                                  const Vec2& p0, const Vec2& dp,
                                  double out[3]) {
                Vec2 u0 = b0 - a0;
                Vec2 du = db - da;
                Vec2 v0 = p0 - a0;
                Vec2 dv = dp - da;
                out[2] = cross2(du, dv);
                out[1] = cross2(du, v0) + cross2(u0, dv);
                out[0] = cross2(u0, v0);
            };

            Vec2 p0 = project_drop_axis(x, drop), dp = project_drop_axis(dx, drop);
            Vec2 a0 = project_drop_axis(x1, drop), da = project_drop_axis(dx1, drop);
            Vec2 b0 = project_drop_axis(x2, drop), db = project_drop_axis(dx2, drop);
            Vec2 c0 = project_drop_axis(x3, drop), dc = project_drop_axis(dx3, drop);

            if (point_in_triangle_projected(p0, a0, b0, c0)) return 0.0;

            std::vector<double> roots;
            double coeffs[3];
            coeff_edge(a0, da, b0, db, p0, dp, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots);
            coeff_edge(b0, db, c0, dc, p0, dp, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots);
            coeff_edge(c0, dc, a0, da, p0, dp, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots);

            std::sort(roots.begin(), roots.end());
            for (double t : roots) {
                Vec2 pt = p0 + dp * t;
                Vec2 at = a0 + da * t;
                Vec2 bt = b0 + db * t;
                Vec2 ct = c0 + dc * t;
                if (point_in_triangle_projected(pt, at, bt, ct)) return t;
            }
            return ccd_infinity;
        }

        bool segments_intersect_projected(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& d) {
            auto orient = [](const Vec2& p, const Vec2& q, const Vec2& r) {
                return cross2(q - p, r - p);
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

        double collinear_segment_overlap_time(const Vec3& x1, const Vec3& dx1,
                                              const Vec3& x2, const Vec3& dx2,
                                              const Vec3& x3, const Vec3& dx3,
                                              const Vec3& x4, const Vec3& dx4) {
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

            double best = ccd_infinity;
            best = std::min(best, earliest_scalar_between(A0, dA, C0, dC, D0, dD));
            best = std::min(best, earliest_scalar_between(B0, dB, C0, dC, D0, dD));
            best = std::min(best, earliest_scalar_between(C0, dC, A0, dA, B0, dB));
            best = std::min(best, earliest_scalar_between(D0, dD, A0, dA, B0, dB));
            return best;
        }

        bool inside_segments_3d_at(const Vec3& x1, const Vec3& dx1,
                                   const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3,
                                   const Vec3& x4, const Vec3& dx4,
                                   double t) {
            Vec3 a0 = x1 + dx1 * t, a1 = x2 + dx2 * t;
            Vec3 b0 = x3 + dx3 * t, b1 = x4 + dx4 * t;

            Vec3 u = a1 - a0;
            Vec3 v = b1 - b0;
            Vec3 w = b0 - a0;
            Vec3 n = u.cross(v);
            double n2 = n.squaredNorm();

            if (n2 <= ccd_geom_eps * ccd_geom_eps) {
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

            if (std::fabs(w.dot(n)) > ccd_geom_eps * std::sqrt(n2)) return false;
            double s = w.cross(v).dot(n) / n2;
            double upar = w.cross(u).dot(n) / n2;
            Vec3 pa = a0 + u * s;
            Vec3 pb = b0 + v * upar;
            if ((pa - pb).squaredNorm() > ccd_geom_eps * ccd_geom_eps) return false;
            return s >= -1e-8 && s <= 1.0 + 1e-8 && upar >= -1e-8 && upar <= 1.0 + 1e-8;
        }

        double persistent_segment_segment_time(const Vec3& x1, const Vec3& dx1,
                                               const Vec3& x2, const Vec3& dx2,
                                               const Vec3& x3, const Vec3& dx3,
                                               const Vec3& x4, const Vec3& dx4) {
            Vec3 n0 = (x2 - x1).cross(x4 - x3);
            Vec3 n1 = (x2 + dx2 - x1 - dx1).cross(x4 + dx4 - x3 - dx3);
            if (n0.squaredNorm() <= ccd_geom_eps * ccd_geom_eps && n1.squaredNorm() <= ccd_geom_eps * ccd_geom_eps) {
                return collinear_segment_overlap_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
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
                out[2] = cross2(du, dv);
                out[1] = cross2(du, v0) + cross2(u0, dv);
                out[0] = cross2(u0, v0);
            };

            std::vector<double> roots;
            double coeffs[3];
            orient_coeff(a0, da, b0, db, c0, dc, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots);
            orient_coeff(a0, da, b0, db, d0, dd, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots);
            orient_coeff(c0, dc, d0, dd, a0, da, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots);
            orient_coeff(c0, dc, d0, dd, b0, db, coeffs); solve_quadratic_all(coeffs[2], coeffs[1], coeffs[0], roots);

            std::sort(roots.begin(), roots.end());
            for (double t : roots) {
                Vec2 at = a0 + da * t, bt = b0 + db * t, ct = c0 + dc * t, dt = d0 + dd * t;
                if (segments_intersect_projected(at, bt, ct, dt)) return t;
            }

            return collinear_segment_overlap_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
        }

    } // namespace

    double node_segment_2d(const Vec2& x, const Vec2& dx,
                           const Vec2& x1, const Vec2& dx1,
                           const Vec2& x2, const Vec2& dx2) {
        if (inside_segment_2d_at(x, dx, x1, dx1, x2, dx2, 0.0)) return 0.0;

        Vec2 dp = dx2 - dx1;
        Vec2 dr = dx - dx1;
        Vec2 p0 = x2 - x1;
        Vec2 r0 = x - x1;

        double a = cross2(dp, dr);
        double b = cross2(dp, r0) + cross2(p0, dr);
        double c = cross2(p0, r0);

        std::vector<double> roots;
        solve_quadratic_all(a, b, c, roots);
        std::sort(roots.begin(), roots.end());
        for (double t : roots) {
            if (inside_segment_2d_at(x, dx, x1, dx1, x2, dx2, t)) return t;
        }

        double s = std::max({1.0, std::fabs(a), std::fabs(b), std::fabs(c)});
        if (near_zero(a, s) && near_zero(b, s) && near_zero(c, s)) {
            Vec2 e0 = x2 - x1;
            Vec2 e1 = (x2 + dx2) - (x1 + dx1);
            double sx = std::fabs(e0.x()) + std::fabs(e1.x());
            double sy = std::fabs(e0.y()) + std::fabs(e1.y());
            double best = (sx >= sy)
                          ? earliest_scalar_between(x.x(), dx.x(), x1.x(), dx1.x(), x2.x(), dx2.x())
                          : earliest_scalar_between(x.y(), dx.y(), x1.y(), dx1.y(), x2.y(), dx2.y());
            if (std::isfinite(best) && inside_segment_2d_at(x, dx, x1, dx1, x2, dx2, best)) return best;
        }

        return ccd_infinity;
    }

    double node_triangle_3d(const Vec3& x, const Vec3& dx,
                            const Vec3& x1, const Vec3& dx1,
                            const Vec3& x2, const Vec3& dx2,
                            const Vec3& x3, const Vec3& dx3) {
        if (inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, 0.0)) return 0.0;

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

        std::vector<double> roots;
        solve_cubic_all(a, b, c, d, roots);
        std::sort(roots.begin(), roots.end());
        for (double t : roots) {
            if (inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, t)) return t;
        }

        double s = scale4(a, b, c, d);
        if (near_zero(a, s) && near_zero(b, s) && near_zero(c, s) && near_zero(d, s)) {
            double t = node_triangle_coplanar_interval(x, dx, x1, dx1, x2, dx2, x3, dx3);
            if (std::isfinite(t) && inside_triangle_3d_at(x, dx, x1, dx1, x2, dx2, x3, dx3, t)) return t;
        }

        return ccd_infinity;
    }

    double segment_segment_3d(const Vec3& x1, const Vec3& dx1,
                              const Vec3& x2, const Vec3& dx2,
                              const Vec3& x3, const Vec3& dx3,
                              const Vec3& x4, const Vec3& dx4) {
        if (inside_segments_3d_at(x1, dx1, x2, dx2, x3, dx3, x4, dx4, 0.0)) return 0.0;

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

        std::vector<double> roots;
        solve_cubic_all(a, b, c, d, roots);
        std::sort(roots.begin(), roots.end());
        for (double t : roots) {
            if (inside_segments_3d_at(x1, dx1, x2, dx2, x3, dx3, x4, dx4, t)) return t;
        }

        double s = scale4(a, b, c, d);
        if (near_zero(a, s) && near_zero(b, s) && near_zero(c, s) && near_zero(d, s)) {
            double t = persistent_segment_segment_time(x1, dx1, x2, dx2, x3, dx3, x4, dx4);
            if (std::isfinite(t) && inside_segments_3d_at(x1, dx1, x2, dx2, x3, dx3, x4, dx4, t)) return t;
        }

        return ccd_infinity;
    }

} // namespace ccd
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::__fs::filesystem;

// ============================================================
// Types
// ============================================================
using Vec2   = Eigen::Vector2d;
using Vec3   = Eigen::Vector3d;
using Vec6   = Eigen::Matrix<double, 6, 1>;
using Mat22  = Eigen::Matrix2d;
using Mat32  = Eigen::Matrix<double, 3, 2>;
using Mat33  = Eigen::Matrix3d;
using Mat39  = Eigen::Matrix<double, 3, 9>;
using Mat66  = Eigen::Matrix<double, 6, 6>;
using Mat312 = Eigen::Matrix<double, 3, 12>;

struct TriangleDef { Vec3 x[3]; };

struct Pin {
    int vertex_index{-1};
    Vec3 target_position{Vec3::Zero()};
};

struct SimParams {
    double fps{30.0};
    int    substeps{1};
    int    num_frames{60};

    double mu{10.0};
    double lambda{10.0};
    double density{1.0};
    double thickness{0.1};
    double kpin{1.0e7};
    Vec3   gravity{0.0, -9.81, 0.0};

    int    max_global_iters{80};
    double tol_abs{1.0e-6};
    double step_weight{0.25};
    double d_hat{0.0};

    double dt()  const { return 1.0 / (fps * static_cast<double>(substeps)); }
    double dt2() const { double h = dt(); return h * h; }
};

struct RefMesh {
    std::vector<Vec2> ref_positions;
    std::vector<int>  tris;      // flat triangle index buffer
    std::vector<Mat22> Dm_inv;
    std::vector<double> area;
    std::vector<double> mass;
    int num_positions{0};

    void initialize() {
        num_positions = static_cast<int>(ref_positions.size());
        const int nt = static_cast<int>(tris.size()) / 3;
        Dm_inv.resize(nt);
        area.resize(nt);
        for (int t = 0; t < nt; ++t) {
            const Vec2& X0 = ref_positions[tris[3 * t + 0]];
            const Vec2& X1 = ref_positions[tris[3 * t + 1]];
            const Vec2& X2 = ref_positions[tris[3 * t + 2]];
            Mat22 Dm;
            Dm.col(0) = X1 - X0;
            Dm.col(1) = X2 - X0;
            area[t] = 0.5 * std::abs(Dm.determinant());
            Dm_inv[t] = Dm.inverse();
        }
    }

    void build_lumped_mass(double density, double thickness) {
        mass.assign(num_positions, 0.0);
        const int nt = static_cast<int>(tris.size()) / 3;
        for (int t = 0; t < nt; ++t) {
            const double m = density * area[t] * thickness;
            const double mv = m / 3.0;
            for (int a = 0; a < 3; ++a) {
                mass[tris[3 * t + a]] += mv;
            }
        }
    }
};

struct DeformedState {
    std::vector<Vec3> x;
    std::vector<Vec3> v;
};

struct PatchInfo {
    int vertex_begin{0};
    int vertex_end{0};
    int tri_begin{0};
    int tri_end{0};
    int nx{0};
    int ny{0};
};

struct NodeTrianglePair {
    int node{-1};
    int tri_v[3]{-1, -1, -1};
};

struct SegmentSegmentPair {
    int v[4]{-1, -1, -1, -1};
};

using VertexTriangleMap = std::unordered_map<int, std::vector<std::pair<int, int>>>;

struct SolverResult {
    double initial_residual{0.0};
    double final_residual{0.0};
    int iterations{0};
};

// ============================================================
// Small helpers
// ============================================================
static inline int tri_vertex(const RefMesh& mesh, int tri, int local) {
    return mesh.tris[3 * tri + local];
}

static inline int num_tris(const RefMesh& mesh) {
    return static_cast<int>(mesh.tris.size()) / 3;
}

static inline double clamp_scalar(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

static Mat33 matrix3d_inverse(const Mat33& H) {
    const double det = H.determinant();
    if (std::abs(det) < 1e-12) {
        // Mild regularization instead of crashing.
        return (H + 1e-8 * Mat33::Identity()).inverse();
    }
    return H.inverse();
}

static TriangleDef make_def_triangle(const std::vector<Vec3>& x, const RefMesh& mesh, int tri_idx) {
    TriangleDef def;
    def.x[0] = x[tri_vertex(mesh, tri_idx, 0)];
    def.x[1] = x[tri_vertex(mesh, tri_idx, 1)];
    def.x[2] = x[tri_vertex(mesh, tri_idx, 2)];
    return def;
}

static void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt) {
    xhat.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) xhat[i] = x[i] + dt * v[i];
}

static void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt) {
    v.resize(xnew.size());
    for (size_t i = 0; i < xnew.size(); ++i) v[i] = (xnew[i] - xold[i]) / dt;
}

static Vec3 segment_closest_point(const Vec3& x, const Vec3& a, const Vec3& b, double& t) {
    const Vec3 ab = b - a;
    const double denom = ab.dot(ab);
    if (denom <= 0.0) {
        t = 0.0;
        return a;
    }
    t = clamp_scalar((x - a).dot(ab) / denom, 0.0, 1.0);
    return a + t * ab;
}

static std::array<double, 3> triangle_plane_barycentric_coordinates(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1.0e-12) {
    const Vec3 e1 = x2 - x1;
    const Vec3 e2 = x3 - x1;
    const Vec3 r  = x  - x1;

    const double a11 = e1.dot(e1);
    const double a12 = e1.dot(e2);
    const double a22 = e2.dot(e2);
    const double b1  = r.dot(e1);
    const double b2  = r.dot(e2);
    const double det = a11 * a22 - a12 * a12;

    if (std::abs(det) <= eps) return {{0.0, 0.0, 0.0}};

    const double alpha = ( b1 * a22 - b2 * a12) / det;
    const double beta  = (-b1 * a12 + b2 * a11) / det;
    return {{1.0 - alpha - beta, alpha, beta}};
}

static double safe_distance(double d) {
    return std::max(d, 1.0e-8);
}

// ============================================================
// Distance queries
// ============================================================
enum class NodeTriangleRegion {
    FaceInterior, Edge12, Edge23, Edge31, Vertex1, Vertex2, Vertex3, DegenerateTriangle
};

struct NodeTriangleDistanceResult {
    Vec3 closest_point{Vec3::Zero()};
    Vec3 tilde_x{Vec3::Zero()};
    Vec3 normal{Vec3::Zero()};
    std::array<double, 3> barycentric_tilde_x{{0.0, 0.0, 0.0}};
    double phi{0.0};
    double distance{0.0};
    NodeTriangleRegion region{NodeTriangleRegion::DegenerateTriangle};
};

static NodeTriangleDistanceResult node_triangle_distance(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1.0e-12) {
    NodeTriangleDistanceResult out;

    const Vec3 e12 = x2 - x1;
    const Vec3 e13 = x3 - x1;
    const Vec3 n_raw = e12.cross(e13);
    const double n_norm = n_raw.norm();

    if (n_norm <= eps) {
        double t12 = 0.0, t23 = 0.0, t31 = 0.0;
        const Vec3 q12 = segment_closest_point(x, x1, x2, t12);
        const Vec3 q23 = segment_closest_point(x, x2, x3, t23);
        const Vec3 q31 = segment_closest_point(x, x3, x1, t31);
        const double d12 = (x - q12).norm();
        const double d23 = (x - q23).norm();
        const double d31 = (x - q31).norm();
        if (d12 <= d23 && d12 <= d31) { out.closest_point = q12; out.distance = d12; }
        else if (d23 <= d12 && d23 <= d31) { out.closest_point = q23; out.distance = d23; }
        else { out.closest_point = q31; out.distance = d31; }
        out.tilde_x = out.closest_point;
        return out;
    }

    const Vec3 n = n_raw / n_norm;
    const double phi = n.dot(x - x1);
    const Vec3 tilde_x = x - phi * n;

    out.normal = n;
    out.phi = phi;
    out.tilde_x = tilde_x;
    out.barycentric_tilde_x = triangle_plane_barycentric_coordinates(tilde_x, x1, x2, x3, eps);

    const double l1 = out.barycentric_tilde_x[0];
    const double l2 = out.barycentric_tilde_x[1];
    const double l3 = out.barycentric_tilde_x[2];

    if (l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0) {
        out.closest_point = tilde_x;
        out.distance = std::abs(phi);
        out.region = NodeTriangleRegion::FaceInterior;
        return out;
    }
    if (l2 <= 0.0 && l3 <= 0.0) {
        out.closest_point = x1;
        out.distance = (x - x1).norm();
        out.region = NodeTriangleRegion::Vertex1;
        return out;
    }
    if (l3 <= 0.0 && l1 <= 0.0) {
        out.closest_point = x2;
        out.distance = (x - x2).norm();
        out.region = NodeTriangleRegion::Vertex2;
        return out;
    }
    if (l1 <= 0.0 && l2 <= 0.0) {
        out.closest_point = x3;
        out.distance = (x - x3).norm();
        out.region = NodeTriangleRegion::Vertex3;
        return out;
    }
    if (l3 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x1, x2, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge12;
        return out;
    }
    if (l1 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x2, x3, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge23;
        return out;
    }
    if (l2 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x3, x1, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge31;
        return out;
    }

    return out;
}

enum class SegmentSegmentRegion {
    Interior, Edge_s0, Edge_s1, Edge_t0, Edge_t1,
    Corner_s0t0, Corner_s0t1, Corner_s1t0, Corner_s1t1, ParallelSegments
};

struct SegmentSegmentDistanceResult {
    Vec3 closest_point_1{Vec3::Zero()};
    Vec3 closest_point_2{Vec3::Zero()};
    double s{0.0};
    double t{0.0};
    double distance{0.0};
    SegmentSegmentRegion region{SegmentSegmentRegion::ParallelSegments};
};

static double optimal_t_for_fixed_s(const Vec3& x1, const Vec3& a,
                                    const Vec3& x3, const Vec3& b,
                                    double s, double C, double& t_out) {
    const Vec3 p = x1 + s * a;
    t_out = (C <= 0.0) ? 0.0 : clamp_scalar((p - x3).dot(b) / C, 0.0, 1.0);
    const Vec3 q = x3 + t_out * b;
    return (p - q).norm();
}

static double optimal_s_for_fixed_t(const Vec3& x1, const Vec3& a,
                                    const Vec3& x3, const Vec3& b,
                                    double t, double A, double& s_out) {
    const Vec3 q = x3 + t * b;
    s_out = (A <= 0.0) ? 0.0 : clamp_scalar((q - x1).dot(a) / A, 0.0, 1.0);
    const Vec3 p = x1 + s_out * a;
    return (p - q).norm();
}

static SegmentSegmentRegion classify_region(double s, double t, double tol = 1.0e-14) {
    const bool s0 = (s <= tol);
    const bool s1 = (s >= 1.0 - tol);
    const bool t0 = (t <= tol);
    const bool t1 = (t >= 1.0 - tol);

    if (s0 && t0) return SegmentSegmentRegion::Corner_s0t0;
    if (s0 && t1) return SegmentSegmentRegion::Corner_s0t1;
    if (s1 && t0) return SegmentSegmentRegion::Corner_s1t0;
    if (s1 && t1) return SegmentSegmentRegion::Corner_s1t1;
    if (s0) return SegmentSegmentRegion::Edge_s0;
    if (s1) return SegmentSegmentRegion::Edge_s1;
    if (t0) return SegmentSegmentRegion::Edge_t0;
    if (t1) return SegmentSegmentRegion::Edge_t1;
    return SegmentSegmentRegion::Interior;
}

static SegmentSegmentDistanceResult segment_segment_distance(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps = 1.0e-12) {
    SegmentSegmentDistanceResult out;

    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const Vec3 c = x1 - x3;

    const double A = a.dot(a);
    const double B = a.dot(b);
    const double C = b.dot(b);
    const double D = a.dot(c);
    const double E = b.dot(c);
    const double Delta = A * C - B * B;

    if (Delta > eps * eps) {
        const double s_unc = (B * E - C * D) / Delta;
        const double t_unc = (A * E - B * D) / Delta;
        if (s_unc >= 0.0 && s_unc <= 1.0 && t_unc >= 0.0 && t_unc <= 1.0) {
            out.s = s_unc;
            out.t = t_unc;
            out.closest_point_1 = x1 + out.s * a;
            out.closest_point_2 = x3 + out.t * b;
            out.distance = (out.closest_point_1 - out.closest_point_2).norm();
            out.region = SegmentSegmentRegion::Interior;
            return out;
        }
    }

    double best_dist = 1.0e300;
    double best_s = 0.0, best_t = 0.0;

    {
        double t = 0.0;
        const double d = optimal_t_for_fixed_s(x1, a, x3, b, 0.0, C, t);
        if (d < best_dist) { best_dist = d; best_s = 0.0; best_t = t; }
    }
    {
        double t = 0.0;
        const double d = optimal_t_for_fixed_s(x1, a, x3, b, 1.0, C, t);
        if (d < best_dist) { best_dist = d; best_s = 1.0; best_t = t; }
    }
    {
        double s = 0.0;
        const double d = optimal_s_for_fixed_t(x1, a, x3, b, 0.0, A, s);
        if (d < best_dist) { best_dist = d; best_s = s; best_t = 0.0; }
    }
    {
        double s = 0.0;
        const double d = optimal_s_for_fixed_t(x1, a, x3, b, 1.0, A, s);
        if (d < best_dist) { best_dist = d; best_s = s; best_t = 1.0; }
    }

    out.s = best_s;
    out.t = best_t;
    out.closest_point_1 = x1 + out.s * a;
    out.closest_point_2 = x3 + out.t * b;
    out.distance = best_dist;
    out.region = (Delta > eps * eps) ? classify_region(best_s, best_t) : SegmentSegmentRegion::ParallelSegments;
    return out;
}

// ============================================================
// Corotated elasticity
// ============================================================
struct CorotatedCache32 {
    Mat22 S{Mat22::Zero()};
    Mat22 SInv{Mat22::Zero()};
    Mat32 R{Mat32::Zero()};
    Mat22 FTFinv{Mat22::Zero()};
    Mat32 FFTFInv{Mat32::Zero()};
    Eigen::Matrix3d FFTFInvFT{Eigen::Matrix3d::Zero()};
    double J{0.0};
    double traceS{0.0};
};

static inline int flatF(int a, int b) { return 2 * a + b; }

static Mat32 Ds(const TriangleDef& tri) {
    Mat32 out;
    out.col(0) = tri.x[1] - tri.x[0];
    out.col(1) = tri.x[2] - tri.x[0];
    return out;
}

static CorotatedCache32 buildCorotatedCache(const Mat32& F) {
    CorotatedCache32 c;
    const Mat22 C = F.transpose() * F;
    Eigen::SelfAdjointEigenSolver<Mat22> es(C);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigen decomposition failed.");

    Mat22 U = es.eigenvectors();
    Eigen::Vector2d evals = es.eigenvalues();
    evals(0) = std::max(evals(0), 1.0e-12);
    evals(1) = std::max(evals(1), 1.0e-12);

    Mat22 Sdiag = Mat22::Zero();
    Sdiag(0, 0) = std::sqrt(evals(0));
    Sdiag(1, 1) = std::sqrt(evals(1));

    c.S = U * Sdiag * U.transpose();
    c.SInv = c.S.inverse();
    c.R = F * c.SInv;
    c.J = Sdiag(0, 0) * Sdiag(1, 1);
    c.traceS = c.S.trace();
    c.FTFinv = C.inverse();
    c.FFTFInv = F * c.FTFinv;
    c.FFTFInvFT = c.FFTFInv * F.transpose();
    return c;
}

static double PsiCorotated32(const CorotatedCache32& c, const Mat32& F, double mu, double lambda) {
    return mu * (F - c.R).squaredNorm()
           + 0.5 * lambda * (c.J - 1.0) * (c.J - 1.0);
}

static Mat32 PCorotated32(const CorotatedCache32& c, const Mat32& F, double mu, double lambda) {
    return 2.0 * mu * (F - c.R)
           + lambda * (c.J - 1.0) * c.J * c.FFTFInv;
}

static void dPdFCorotated32(const CorotatedCache32& c, double mu, double lambda, Mat66& dPdF) {
    const Mat22& SInv      = c.SInv;
    const Mat32& R         = c.R;
    const Mat22& FTFinv    = c.FTFinv;
    const Mat32& FFTFInv   = c.FFTFInv;
    const Eigen::Matrix3d& FFTFInvFT = c.FFTFInvFT;
    const double J         = c.J;
    const double traceS    = c.traceS;

    const Eigen::Matrix3d RRT = R * R.transpose();

    Mat32 Re;
    Re(0,0) =  R(0,1); Re(0,1) = -R(0,0);
    Re(1,0) =  R(1,1); Re(1,1) = -R(1,0);
    Re(2,0) =  R(2,1); Re(2,1) = -R(2,0);

    Vec6 dcdF;
    dcdF << -R(0,1) / traceS,  R(0,0) / traceS,
            -R(1,1) / traceS,  R(1,0) / traceS,
            -R(2,1) / traceS,  R(2,0) / traceS;

    static constexpr int idx[12] = {0,0, 0,1, 1,0, 1,1, 2,0, 2,1};

    Mat66 dRdF = Mat66::Zero();
    for (int c1 = 0; c1 < 6; ++c1) {
        for (int c2 = 0; c2 < 6; ++c2) {
            const int m = idx[2 * c1], n = idx[2 * c1 + 1];
            const int i = idx[2 * c2], j = idx[2 * c2 + 1];
            double v = 0.0;
            if (m == i) v += SInv(j, n);
            v -= RRT(m, i) * SInv(j, n);
            v -= dcdF(c2) * Re(m, n);
            dRdF(c1, c2) = v;
        }
    }

    dPdF.setZero();
    for (int c1 = 0; c1 < 6; ++c1) {
        for (int c2 = 0; c2 < 6; ++c2) {
            const int m = idx[2 * c1], n = idx[2 * c1 + 1];
            const int i = idx[2 * c2], j = idx[2 * c2 + 1];
            double v = 0.0;
            if (m == i) v += lambda * (J - 1.0) * J * FTFinv(j, n);
            v -= lambda * (J - 1.0) * J *
                 (FFTFInv(m, j) * FFTFInv(i, n) + FFTFInvFT(m, i) * FTFinv(j, n));
            v += 0.5 * lambda * (2.0 * J - 1.0) * J *
                 (FFTFInv(i, j) * FFTFInv(m, n) + FFTFInv(i, j) * FFTFInv(m, n));
            dPdF(c1, c2) = v;
        }
    }

    dPdF += 2.0 * mu * (Mat66::Identity() - dRdF);
}

static double corotated_energy(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda) {
    const Mat32 F = Ds(def) * Dm_inv;
    const CorotatedCache32 c = buildCorotatedCache(F);
    return ref_area * PsiCorotated32(c, F, mu, lambda);
}

using ShapeGrads = std::array<Vec2, 3>;

static ShapeGrads shape_function_gradients(const Mat22& Dm_inv) {
    ShapeGrads grads;
    grads[1] = Dm_inv.row(0).transpose();
    grads[2] = Dm_inv.row(1).transpose();
    grads[0] = -grads[1] - grads[2];
    return grads;
}

static Vec3 corotated_node_gradient(const Mat32& P, double ref_area, const ShapeGrads& gradN, int node) {
    Vec3 g = Vec3::Zero();
    for (int gamma = 0; gamma < 3; ++gamma) {
        double value = 0.0;
        for (int beta = 0; beta < 2; ++beta) value += P(gamma, beta) * gradN[node](beta);
        g(gamma) = ref_area * value;
    }
    return g;
}

static Mat39 corotated_node_hessian(const Mat66& dPdF, double ref_area, const ShapeGrads& gradN, int node) {
    Mat39 H = Mat39::Zero();
    for (int j = 0; j < 3; ++j) {
        for (int gamma = 0; gamma < 3; ++gamma) {
            for (int delta = 0; delta < 3; ++delta) {
                double value = 0.0;
                for (int beta = 0; beta < 2; ++beta) {
                    for (int eta = 0; eta < 2; ++eta) {
                        value += dPdF(flatF(gamma, beta), flatF(delta, eta))
                                 * gradN[node](beta) * gradN[j](eta);
                    }
                }
                H(gamma, 3 * j + delta) = ref_area * value;
            }
        }
    }
    return H;
}

// ============================================================
// Barrier
// ============================================================
static double scalar_barrier(double delta, double d_hat) {
    if (d_hat <= 0.0) return 0.0;
    const double d = safe_distance(delta);
    if (d >= d_hat) return 0.0;
    const double s = d - d_hat;
    return -(s * s) * std::log(d / d_hat);
}

static double scalar_barrier_gradient(double delta, double d_hat) {
    if (d_hat <= 0.0) return 0.0;
    const double d = safe_distance(delta);
    if (d >= d_hat) return 0.0;
    const double s = d - d_hat;
    return -2.0 * s * std::log(d / d_hat) - (s * s) / d;
}

static double scalar_barrier_hessian(double delta, double d_hat) {
    if (d_hat <= 0.0) return 0.0;
    const double d = safe_distance(delta);
    if (d >= d_hat) return 0.0;
    const double ratio = d_hat / d;
    return ratio * ratio + 2.0 * ratio - 3.0 - 2.0 * std::log(d / d_hat);
}

static double segment_parameter_from_closest_point(const Vec3& q, const Vec3& a, const Vec3& b) {
    double denom = 0.0, numer = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double ab = b(k) - a(k);
        denom += ab * ab;
        numer += (q(k) - a(k)) * ab;
    }
    if (denom <= 0.0) return 0.0;
    return clamp_scalar(numer / denom, 0.0, 1.0);
}

static double levi_civita(int i, int j, int k) {
    return 0.5 * (i - j) * (j - k) * (k - i);
}

static double node_triangle_barrier(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps = 1.0e-12) {
    const auto dr = node_triangle_distance(x, x1, x2, x3, eps);
    return scalar_barrier(dr.distance, d_hat);
}

static Vec3 node_triangle_barrier_gradient(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12) {
    const auto dr = node_triangle_distance(x, x1, x2, x3, eps);
    const double delta = safe_distance(dr.distance);
    const double bp = scalar_barrier_gradient(delta, d_hat);

    Vec3 g = Vec3::Zero();
    if (bp == 0.0) return g;

    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = (x(k) - dr.closest_point(k)) / delta;

    double coeff[4] = {0.0, 0.0, 0.0, 0.0};
    double face_n[3] = {0.0, 0.0, 0.0};
    bool use_normal = false;

    switch (dr.region) {
        case NodeTriangleRegion::FaceInterior: {
            const double phi = dr.phi;
            const double sphi = (phi > 0.0) ? 1.0 : (phi < 0.0) ? -1.0 : 0.0;
            coeff[0] =  bp * sphi;
            coeff[1] = -bp * sphi * dr.barycentric_tilde_x[0];
            coeff[2] = -bp * sphi * dr.barycentric_tilde_x[1];
            coeff[3] = -bp * sphi * dr.barycentric_tilde_x[2];
            for (int k = 0; k < 3; ++k) face_n[k] = dr.normal(k);
            use_normal = true;
            break;
        }
        case NodeTriangleRegion::Edge12: {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x1, x2);
            coeff[0] =  bp; coeff[1] = -bp * (1.0 - t); coeff[2] = -bp * t; coeff[3] = 0.0;
            break;
        }
        case NodeTriangleRegion::Edge23: {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x2, x3);
            coeff[0] =  bp; coeff[1] =  0.0; coeff[2] = -bp * (1.0 - t); coeff[3] = -bp * t;
            break;
        }
        case NodeTriangleRegion::Edge31: {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x3, x1);
            coeff[0] =  bp; coeff[1] = -bp * t; coeff[2] =  0.0; coeff[3] = -bp * (1.0 - t);
            break;
        }
        case NodeTriangleRegion::Vertex1: coeff[0] =  bp; coeff[1] = -bp; coeff[2] =  0.0; coeff[3] =  0.0; break;
        case NodeTriangleRegion::Vertex2: coeff[0] =  bp; coeff[1] =  0.0; coeff[2] = -bp; coeff[3] =  0.0; break;
        case NodeTriangleRegion::Vertex3: coeff[0] =  bp; coeff[1] =  0.0; coeff[2] =  0.0; coeff[3] = -bp; break;
        case NodeTriangleRegion::DegenerateTriangle: {
            coeff[0] = bp;
            double d1 = (dr.closest_point - x1).norm();
            double d2 = (dr.closest_point - x2).norm();
            double d3 = (dr.closest_point - x3).norm();
            if (d1 <= d2 && d1 <= d3) coeff[1] = -bp;
            else if (d2 <= d3) coeff[2] = -bp;
            else coeff[3] = -bp;
            break;
        }
    }

    if (use_normal) {
        for (int k = 0; k < 3; ++k) g(k) = coeff[dof] * face_n[k];
    } else {
        for (int k = 0; k < 3; ++k) g(k) = coeff[dof] * u[k];
    }
    return g;
}

static Mat312 node_triangle_barrier_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12) {
    Mat312 H_row = Mat312::Zero();

    const auto dr = node_triangle_distance(x, x1, x2, x3, eps);
    const double delta = safe_distance(dr.distance);
    const double bp  = scalar_barrier_gradient(delta, d_hat);
    const double bpp = scalar_barrier_hessian(delta, d_hat);

    if (bp == 0.0 && bpp == 0.0) return H_row;

    const Vec3* Y[4] = {&x, &x1, &x2, &x3};
    const int p = dof;

    switch (dr.region) {
        case NodeTriangleRegion::Vertex1:
        case NodeTriangleRegion::Vertex2:
        case NodeTriangleRegion::Vertex3: {
            const int a_idx = (dr.region == NodeTriangleRegion::Vertex1) ? 1 :
                              (dr.region == NodeTriangleRegion::Vertex2) ? 2 : 3;
            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[0] = 1.0;
            sp[a_idx] = -1.0;
            if (sp[p] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = (x(k) - (*Y[a_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;
            for (int q = 0; q < 4; ++q) {
                if (sp[q] == 0.0) continue;
                const double sq = sp[p] * sp[q];
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;
                        H_row(k, 3 * q + l) = sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                    }
                }
            }
            break;
        }

        case NodeTriangleRegion::Edge12:
        case NodeTriangleRegion::Edge23:
        case NodeTriangleRegion::Edge31: {
            int a_idx, b_idx;
            if      (dr.region == NodeTriangleRegion::Edge12) { a_idx = 1; b_idx = 2; }
            else if (dr.region == NodeTriangleRegion::Edge23) { a_idx = 2; b_idx = 3; }
            else                                              { a_idx = 3; b_idx = 1; }

            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[0] = 1.0;
            omega[a_idx] = -1.0;
            epsilon[a_idx] = -1.0;
            epsilon[b_idx] = 1.0;

            const Vec3& xa = *Y[a_idx];
            const Vec3& xb = *Y[b_idx];

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) {
                e[i] = xb(i) - xa(i);
                w[i] = x(i)  - xa(i);
            }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) {
                alpha += w[i] * e[i];
                beta  += e[i] * e[i];
            }
            const double t = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = x(i) - (xa(i) + t * e[i]);
                u[i] = r[i] / delta;
            }

            double t_d[4][3];
            double r_d[4][3][3];
            double q_d[4][3][3];

            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    const double alpha_pk = omega[pp] * e[k] + epsilon[pp] * w[k];
                    const double beta_pk  = 2.0 * epsilon[pp] * e[k];
                    t_d[pp][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dpa = (pp == a_idx) ? 1.0 : 0.0;
                        const double dpx = (pp == 0) ? 1.0 : 0.0;
                        q_d[pp][k][i] = dpa * dik + t_d[pp][k] * e[i] + t * epsilon[pp] * dik;
                        r_d[pp][k][i] = dpx * dik - q_d[pp][k][i];
                    }
                }
            }

            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;

                        const double alpha_pk = omega[p] * e[k] + epsilon[p] * w[k];
                        const double alpha_ql = omega[q] * e[l] + epsilon[q] * w[l];
                        const double alpha_pkql = (omega[p] * epsilon[q] + omega[q] * epsilon[p]) * dkl;
                        const double beta_pk   = 2.0 * epsilon[p] * e[k];
                        const double beta_ql   = 2.0 * epsilon[q] * e[l];
                        const double beta_pkql = 2.0 * epsilon[p] * epsilon[q] * dkl;

                        const double t_pkql = alpha_pkql / beta
                                              - (alpha_pk * beta_ql + alpha_ql * beta_pk + alpha * beta_pkql) / (beta * beta)
                                              + 2.0 * alpha * beta_pk * beta_ql / (beta * beta * beta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i] * r_d[p][k][i];
                            ddelta_ql += u[i] * r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                const double dij = (i == j) ? 1.0 : 0.0;
                                proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[q][l][j];
                            }
                        }
                        proj_term /= delta;

                        double uq_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            const double dik = (i == k) ? 1.0 : 0.0;
                            const double dil = (i == l) ? 1.0 : 0.0;
                            const double q_ipkql = t_pkql * e[i]
                                                   + t_d[p][k] * epsilon[q] * dil
                                                   + t_d[q][l] * epsilon[p] * dik;
                            uq_term += u[i] * q_ipkql;
                        }

                        const double d2delta = proj_term - uq_term;
                        H_row(k, 3 * q + l) = bpp * ddelta_pk * ddelta_ql + bp * d2delta;
                    }
                }
            }
            break;
        }

        case NodeTriangleRegion::FaceInterior: {
            double sig_a[4] = { 0.0, -1.0,  1.0,  0.0};
            double sig_b[4] = { 0.0, -1.0,  0.0,  1.0};
            double sig_w[4] = { 1.0, -1.0,  0.0,  0.0};

            double a[3], b[3], w[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x2(i) - x1(i);
                b[i] = x3(i) - x1(i);
                w[i] = x(i)  - x1(i);
            }

            double N[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i)
                for (int m = 0; m < 3; ++m)
                    for (int n = 0; n < 3; ++n)
                        N[i] += levi_civita(i, m, n) * a[m] * b[n];

            double eta = 0.0;
            for (int i = 0; i < 3; ++i) eta += N[i] * N[i];
            eta = std::sqrt(eta);

            double n[3];
            for (int i = 0; i < 3; ++i) n[i] = N[i] / eta;

            double psi = 0.0;
            for (int i = 0; i < 3; ++i) psi += N[i] * w[i];
            const double phi = psi / eta;
            const double s_sign = (phi > 0.0) ? 1.0 : (phi < 0.0) ? -1.0 : 0.0;

            double Nd[4][3][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < 3; ++i) {
                        double val = 0.0;
                        for (int nn = 0; nn < 3; ++nn) val += sig_a[pp] * levi_civita(i, k, nn) * b[nn];
                        for (int m = 0; m < 3; ++m)   val += sig_b[pp] * levi_civita(i, m, k) * a[m];
                        Nd[pp][k][i] = val;
                    }
                }
            }

            double eta_d[4][3], psi_d[4][3], phi_d[4][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    double eta_pk = 0.0;
                    for (int i = 0; i < 3; ++i) eta_pk += n[i] * Nd[pp][k][i];
                    eta_d[pp][k] = eta_pk;

                    double psi_pk = 0.0;
                    for (int i = 0; i < 3; ++i) psi_pk += Nd[pp][k][i] * w[i];
                    psi_pk += sig_w[pp] * N[k];
                    psi_d[pp][k] = psi_pk;

                    phi_d[pp][k] = psi_pk / eta - psi * eta_pk / (eta * eta);
                }
            }

            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double coeff_N2 = sig_a[p] * sig_b[q] - sig_a[q] * sig_b[p];

                        double nN2 = 0.0;
                        for (int i = 0; i < 3; ++i) nN2 += n[i] * coeff_N2 * levi_civita(i, k, l);

                        double proj_NN = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                const double dij = (i == j) ? 1.0 : 0.0;
                                proj_NN += (dij - n[i] * n[j]) * Nd[p][k][i] * Nd[q][l][j];
                            }
                        }

                        const double eta_pkql = nN2 + proj_NN / eta;

                        double psi_pkql = 0.0;
                        for (int i = 0; i < 3; ++i) psi_pkql += coeff_N2 * levi_civita(i, k, l) * w[i];
                        psi_pkql += sig_w[q] * Nd[p][k][l];
                        psi_pkql += sig_w[p] * Nd[q][l][k];

                        const double phi_pkql =
                                psi_pkql / eta
                                - (psi_d[p][k] * eta_d[q][l] + psi_d[q][l] * eta_d[p][k] + psi * eta_pkql) / (eta * eta)
                                + 2.0 * psi * eta_d[p][k] * eta_d[q][l] / (eta * eta * eta);

                        H_row(k, 3 * q + l) = bpp * phi_d[p][k] * phi_d[q][l] + s_sign * bp * phi_pkql;
                    }
                }
            }
            break;
        }

        case NodeTriangleRegion::DegenerateTriangle:
            break;
    }

    return H_row;
}

static double segment_segment_barrier(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12) {
    const auto dr = segment_segment_distance(x1, x2, x3, x4, eps);
    return scalar_barrier(dr.distance, d_hat);
}

static Vec3 segment_segment_barrier_gradient(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12) {
    const auto dr = segment_segment_distance(x1, x2, x3, x4, eps);
    const double delta = safe_distance(dr.distance);
    const double bp = scalar_barrier_gradient(delta, d_hat);

    Vec3 g = Vec3::Zero();
    if (bp == 0.0) return g;

    const Vec3 r = dr.closest_point_1 - dr.closest_point_2;
    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = r(k) / delta;

    const double s = dr.s;
    const double t = dr.t;
    double mu[4] = {0.0, 0.0, 0.0, 0.0};

    switch (dr.region) {
        case SegmentSegmentRegion::Interior:     mu[0] =  bp * (1.0 - s); mu[1] =  bp * s;         mu[2] = -bp * (1.0 - t); mu[3] = -bp * t;         break;
        case SegmentSegmentRegion::Edge_s0:      mu[0] =  bp;             mu[1] =  0.0;            mu[2] = -bp * (1.0 - t); mu[3] = -bp * t;         break;
        case SegmentSegmentRegion::Edge_s1:      mu[0] =  0.0;            mu[1] =  bp;             mu[2] = -bp * (1.0 - t); mu[3] = -bp * t;         break;
        case SegmentSegmentRegion::Edge_t0:      mu[0] =  bp * (1.0 - s); mu[1] =  bp * s;         mu[2] = -bp;             mu[3] =  0.0;            break;
        case SegmentSegmentRegion::Edge_t1:      mu[0] =  bp * (1.0 - s); mu[1] =  bp * s;         mu[2] =  0.0;            mu[3] = -bp;             break;
        case SegmentSegmentRegion::Corner_s0t0:  mu[0] =  bp;             mu[1] =  0.0;            mu[2] = -bp;             mu[3] =  0.0;            break;
        case SegmentSegmentRegion::Corner_s0t1:  mu[0] =  bp;             mu[1] =  0.0;            mu[2] =  0.0;            mu[3] = -bp;             break;
        case SegmentSegmentRegion::Corner_s1t0:  mu[0] =  0.0;            mu[1] =  bp;             mu[2] = -bp;             mu[3] =  0.0;            break;
        case SegmentSegmentRegion::Corner_s1t1:  mu[0] =  0.0;            mu[1] =  bp;             mu[2] =  0.0;            mu[3] = -bp;             break;
        case SegmentSegmentRegion::ParallelSegments: {
            const double fallback[4] = {1.0 - s, s, -(1.0 - t), -t};
            for (int k = 0; k < 3; ++k) g(k) = bp * fallback[dof] * u[k];
            return g;
        }
    }

    for (int k = 0; k < 3; ++k) g(k) = mu[dof] * u[k];
    return g;
}

static Mat312 segment_segment_barrier_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12) {
    Mat312 H_row = Mat312::Zero();

    const auto dr = segment_segment_distance(x1, x2, x3, x4, eps);
    const double delta = safe_distance(dr.distance);
    const double bp  = scalar_barrier_gradient(delta, d_hat);
    const double bpp = scalar_barrier_hessian(delta, d_hat);
    if (bp == 0.0 && bpp == 0.0) return H_row;

    const Vec3* Y[4] = {&x1, &x2, &x3, &x4};
    const int p = dof;

    switch (dr.region) {
        case SegmentSegmentRegion::Corner_s0t0:
        case SegmentSegmentRegion::Corner_s0t1:
        case SegmentSegmentRegion::Corner_s1t0:
        case SegmentSegmentRegion::Corner_s1t1: {
            int a_idx, b_idx;
            if      (dr.region == SegmentSegmentRegion::Corner_s0t0) { a_idx = 0; b_idx = 2; }
            else if (dr.region == SegmentSegmentRegion::Corner_s0t1) { a_idx = 0; b_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Corner_s1t0) { a_idx = 1; b_idx = 2; }
            else                                                     { a_idx = 1; b_idx = 3; }

            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[a_idx] = 1.0;
            sp[b_idx] = -1.0;
            if (sp[p] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = ((*Y[a_idx])(k) - (*Y[b_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;
            for (int q = 0; q < 4; ++q) {
                if (sp[q] == 0.0) continue;
                const double sq = sp[p] * sp[q];
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;
                        H_row(k, 3 * q + l) = sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                    }
                }
            }
            break;
        }

        case SegmentSegmentRegion::Edge_s0:
        case SegmentSegmentRegion::Edge_s1:
        case SegmentSegmentRegion::Edge_t0:
        case SegmentSegmentRegion::Edge_t1: {
            int query_idx, ea_idx, eb_idx;
            if      (dr.region == SegmentSegmentRegion::Edge_s0) { query_idx = 0; ea_idx = 2; eb_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Edge_s1) { query_idx = 1; ea_idx = 2; eb_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Edge_t0) { query_idx = 2; ea_idx = 0; eb_idx = 1; }
            else                                                 { query_idx = 3; ea_idx = 0; eb_idx = 1; }

            const Vec3& xq  = *Y[query_idx];
            const Vec3& xea = *Y[ea_idx];
            const Vec3& xeb = *Y[eb_idx];

            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[query_idx] = 1.0;
            omega[ea_idx] = -1.0;
            epsilon[ea_idx] = -1.0;
            epsilon[eb_idx] = 1.0;

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) {
                e[i] = xeb(i) - xea(i);
                w[i] = xq(i)  - xea(i);
            }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) {
                alpha += w[i] * e[i];
                beta  += e[i] * e[i];
            }
            const double t_param = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = xq(i) - (xea(i) + t_param * e[i]);
                u[i] = r[i] / delta;
            }

            double t_d[4][3];
            double r_d[4][3][3];
            double q_d[4][3][3];

            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    const double alpha_pk = omega[pp] * e[k] + epsilon[pp] * w[k];
                    const double beta_pk  = 2.0 * epsilon[pp] * e[k];
                    t_d[pp][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);
                    for (int i = 0; i < 3; ++i) {
                        const double dik   = (i == k) ? 1.0 : 0.0;
                        const double dp_ea = (pp == ea_idx) ? 1.0 : 0.0;
                        const double dp_q  = (pp == query_idx) ? 1.0 : 0.0;
                        q_d[pp][k][i] = dp_ea * dik + t_d[pp][k] * e[i] + t_param * epsilon[pp] * dik;
                        r_d[pp][k][i] = dp_q * dik - q_d[pp][k][i];
                    }
                }
            }

            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;

                        const double alpha_pk = omega[p] * e[k] + epsilon[p] * w[k];
                        const double alpha_ql = omega[q] * e[l] + epsilon[q] * w[l];
                        const double alpha_pkql = (omega[p] * epsilon[q] + omega[q] * epsilon[p]) * dkl;
                        const double beta_pk = 2.0 * epsilon[p] * e[k];
                        const double beta_ql = 2.0 * epsilon[q] * e[l];
                        const double beta_pkql = 2.0 * epsilon[p] * epsilon[q] * dkl;

                        const double t_pkql = alpha_pkql / beta
                                              - (alpha_pk * beta_ql + alpha_ql * beta_pk + alpha * beta_pkql) / (beta * beta)
                                              + 2.0 * alpha * beta_pk * beta_ql / (beta * beta * beta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i] * r_d[p][k][i];
                            ddelta_ql += u[i] * r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                const double dij = (i == j) ? 1.0 : 0.0;
                                proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[q][l][j];
                            }
                        }
                        proj_term /= delta;

                        double uq_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            const double dik = (i == k) ? 1.0 : 0.0;
                            const double dil = (i == l) ? 1.0 : 0.0;
                            const double q_ipkql = t_pkql * e[i]
                                                   + t_d[p][k] * epsilon[q] * dil
                                                   + t_d[q][l] * epsilon[p] * dik;
                            uq_term += u[i] * q_ipkql;
                        }

                        H_row(k, 3 * q + l) = bpp * ddelta_pk * ddelta_ql + bp * (proj_term - uq_term);
                    }
                }
            }
            break;
        }

        case SegmentSegmentRegion::Interior: {
            double sig_a[4] = {-1.0,  1.0,  0.0,  0.0};
            double sig_b[4] = { 0.0,  0.0, -1.0,  1.0};
            double sig_c[4] = { 1.0,  0.0, -1.0,  0.0};

            double a[3], b[3], c[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x2(i) - x1(i);
                b[i] = x4(i) - x3(i);
                c[i] = x1(i) - x3(i);
            }

            double A = 0.0, B = 0.0, C = 0.0, D = 0.0, E = 0.0;
            for (int i = 0; i < 3; ++i) {
                A += a[i] * a[i];
                B += a[i] * b[i];
                C += b[i] * b[i];
                D += a[i] * c[i];
                E += b[i] * c[i];
            }

            const double Delta = A * C - B * B;
            const double nu    = B * E - C * D;
            const double zeta  = A * E - B * D;
            const double s_val = nu / Delta;
            const double t_val = zeta / Delta;

            double Ad[4][3], Bd[4][3], Cd[4][3], Dd[4][3], Ed[4][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    Ad[pp][k] = 2.0 * sig_a[pp] * a[k];
                    Bd[pp][k] = sig_a[pp] * b[k] + sig_b[pp] * a[k];
                    Cd[pp][k] = 2.0 * sig_b[pp] * b[k];
                    Dd[pp][k] = sig_a[pp] * c[k] + sig_c[pp] * a[k];
                    Ed[pp][k] = sig_b[pp] * c[k] + sig_c[pp] * b[k];
                }
            }

            double nu_d[4][3], zeta_d[4][3], Delta_d[4][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    nu_d[pp][k]    = Bd[pp][k] * E + B * Ed[pp][k] - Cd[pp][k] * D - C * Dd[pp][k];
                    zeta_d[pp][k]  = Ad[pp][k] * E + A * Ed[pp][k] - Bd[pp][k] * D - B * Dd[pp][k];
                    Delta_d[pp][k] = Ad[pp][k] * C + A * Cd[pp][k] - 2.0 * B * Bd[pp][k];
                }
            }

            double s_d[4][3], t_d[4][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    s_d[pp][k] = nu_d[pp][k] / Delta   - nu   * Delta_d[pp][k] / (Delta * Delta);
                    t_d[pp][k] = zeta_d[pp][k] / Delta - zeta * Delta_d[pp][k] / (Delta * Delta);
                }
            }

            double r_vec[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r_vec[i] = (x1(i) + s_val * a[i]) - (x3(i) + t_val * b[i]);
                u[i] = r_vec[i] / delta;
            }

            double p_d[4][3][3], q_d_arr[4][3][3], r_d[4][3][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dp0 = (pp == 0) ? 1.0 : 0.0;
                        const double dp2 = (pp == 2) ? 1.0 : 0.0;
                        p_d[pp][k][i]     = dp0 * dik + s_d[pp][k] * a[i] + s_val * sig_a[pp] * dik;
                        q_d_arr[pp][k][i] = dp2 * dik + t_d[pp][k] * b[i] + t_val * sig_b[pp] * dik;
                        r_d[pp][k][i]     = p_d[pp][k][i] - q_d_arr[pp][k][i];
                    }
                }
            }

            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;

                        const double A_pkql = 2.0 * sig_a[p] * sig_a[q] * dkl;
                        const double B_pkql = (sig_a[p] * sig_b[q] + sig_a[q] * sig_b[p]) * dkl;
                        const double C_pkql = 2.0 * sig_b[p] * sig_b[q] * dkl;
                        const double D_pkql = (sig_a[p] * sig_c[q] + sig_a[q] * sig_c[p]) * dkl;
                        const double E_pkql = (sig_b[p] * sig_c[q] + sig_b[q] * sig_c[p]) * dkl;

                        const double nu_pkql = B_pkql * E + Bd[p][k] * Ed[q][l] + Bd[q][l] * Ed[p][k] + B * E_pkql
                                               - C_pkql * D - Cd[p][k] * Dd[q][l] - Cd[q][l] * Dd[p][k] - C * D_pkql;

                        const double Delta_pkql = A_pkql * C + Ad[p][k] * Cd[q][l] + Ad[q][l] * Cd[p][k] + A * C_pkql
                                                  - 2.0 * (Bd[p][k] * Bd[q][l] + B * B_pkql);

                        const double zeta_pkql = A_pkql * E + Ad[p][k] * Ed[q][l] + Ad[q][l] * Ed[p][k] + A * E_pkql
                                                 - B_pkql * D - Bd[p][k] * Dd[q][l] - Bd[q][l] * Dd[p][k] - B * D_pkql;

                        const double s_pkql = nu_pkql / Delta
                                              - (nu_d[p][k] * Delta_d[q][l] + nu_d[q][l] * Delta_d[p][k] + nu * Delta_pkql) / (Delta * Delta)
                                              + 2.0 * nu * Delta_d[p][k] * Delta_d[q][l] / (Delta * Delta * Delta);

                        const double t_pkql = zeta_pkql / Delta
                                              - (zeta_d[p][k] * Delta_d[q][l] + zeta_d[q][l] * Delta_d[p][k] + zeta * Delta_pkql) / (Delta * Delta)
                                              + 2.0 * zeta * Delta_d[p][k] * Delta_d[q][l] / (Delta * Delta * Delta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i] * r_d[p][k][i];
                            ddelta_ql += u[i] * r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                const double dij = (i == j) ? 1.0 : 0.0;
                                proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[q][l][j];
                            }
                        }
                        proj_term /= delta;

                        double ur_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            const double dik = (i == k) ? 1.0 : 0.0;
                            const double dil = (i == l) ? 1.0 : 0.0;
                            const double p_ipkql = s_pkql * a[i] + s_d[p][k] * sig_a[q] * dil + s_d[q][l] * sig_a[p] * dik;
                            const double q_ipkql = t_pkql * b[i] + t_d[p][k] * sig_b[q] * dil + t_d[q][l] * sig_b[p] * dik;
                            ur_term += u[i] * (p_ipkql - q_ipkql);
                        }

                        H_row(k, 3 * q + l) = bpp * ddelta_pk * ddelta_ql + bp * (proj_term + ur_term);
                    }
                }
            }
            break;
        }

        case SegmentSegmentRegion::ParallelSegments:
            break;
    }

    return H_row;
}

static std::pair<Vec3, Mat312> node_triangle_barrier_gradient_and_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12) {
    return {
            node_triangle_barrier_gradient(x, x1, x2, x3, d_hat, dof, eps),
            node_triangle_barrier_hessian(x, x1, x2, x3, d_hat, dof, eps)
    };
}

static std::pair<Vec3, Mat312> segment_segment_barrier_gradient_and_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12) {
    return {
            segment_segment_barrier_gradient(x1, x2, x3, x4, d_hat, dof, eps),
            segment_segment_barrier_hessian(x1, x2, x3, x4, d_hat, dof, eps)
    };
}

// ============================================================
// Scene construction
// ============================================================
static void clear_scene(RefMesh& mesh, DeformedState& state, std::vector<Pin>& pins) {
    mesh.ref_positions.clear();
    mesh.tris.clear();
    mesh.Dm_inv.clear();
    mesh.area.clear();
    mesh.mass.clear();
    mesh.num_positions = 0;
    state.x.clear();
    state.v.clear();
    pins.clear();
}

static PatchInfo build_square_patch(
        RefMesh& mesh, DeformedState& state,
        int nx, int ny, double width, double height, const Vec3& origin) {
    PatchInfo patch;
    patch.vertex_begin = static_cast<int>(state.x.size());
    patch.tri_begin = num_tris(mesh);
    patch.nx = nx;
    patch.ny = ny;

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const double u = static_cast<double>(i) / nx;
            const double v = static_cast<double>(j) / ny;
            const double xr = u * width;
            const double yr = v * height;
            mesh.ref_positions.push_back(Vec2(xr, yr));
            state.x.push_back(origin + Vec3(xr, 0.0, yr));
        }
    }

    auto vertex_index = [base = patch.vertex_begin, nx](int i, int j) {
        return base + j * (nx + 1) + i;
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int v00 = vertex_index(i, j);
            const int v10 = vertex_index(i + 1, j);
            const int v01 = vertex_index(i, j + 1);
            const int v11 = vertex_index(i + 1, j + 1);

            mesh.tris.push_back(v00); mesh.tris.push_back(v10); mesh.tris.push_back(v11);
            mesh.tris.push_back(v00); mesh.tris.push_back(v11); mesh.tris.push_back(v01);
        }
    }

    patch.vertex_end = static_cast<int>(state.x.size());
    patch.tri_end = num_tris(mesh);
    return patch;
}

static void append_pin(std::vector<Pin>& pins, int vi, const std::vector<Vec3>& x) {
    pins.push_back(Pin{vi, x[vi]});
}

static int patch_corner_index(const PatchInfo& patch, int i, int j) {
    return patch.vertex_begin + j * (patch.nx + 1) + i;
}

static VertexTriangleMap build_incident_triangle_map(const std::vector<int>& tris) {
    VertexTriangleMap map;
    for (int idx = 0; idx < static_cast<int>(tris.size()); ++idx) {
        const int v = tris[idx];
        map[v].push_back({idx / 3, idx % 3});
    }
    return map;
}

static std::vector<std::array<int, 2>> extract_patch_edges(const RefMesh& mesh, const PatchInfo& patch) {
    std::set<std::pair<int, int>> edges;
    for (int t = patch.tri_begin; t < patch.tri_end; ++t) {
        for (int e = 0; e < 3; ++e) {
            int a = tri_vertex(mesh, t, e);
            int b = tri_vertex(mesh, t, (e + 1) % 3);
            if (a > b) std::swap(a, b);
            edges.insert({a, b});
        }
    }
    std::vector<std::array<int, 2>> out;
    out.reserve(edges.size());
    for (const auto& [a, b] : edges) out.push_back({a, b});
    return out;
}

static std::vector<std::array<int, 2>> extract_all_unique_edges(const RefMesh& mesh) {
    std::set<std::pair<int, int>> edges;
    for (int t = 0; t < num_tris(mesh); ++t) {
        for (int e = 0; e < 3; ++e) {
            int a = tri_vertex(mesh, t, e);
            int b = tri_vertex(mesh, t, (e + 1) % 3);
            if (a > b) std::swap(a, b);
            edges.insert({a, b});
        }
    }
    std::vector<std::array<int, 2>> out;
    out.reserve(edges.size());
    for (const auto& [a, b] : edges) out.push_back({a, b});
    return out;
}

struct SweptAABB3 {
    Vec3 min{Vec3::Zero()};
    Vec3 max{Vec3::Zero()};
};

static SweptAABB3 make_swept_node_box(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int node, double dt, double pad) {
    SweptAABB3 box;
    box.min = x[node].cwiseMin(x[node] + dt * v[node]);
    box.max = x[node].cwiseMax(x[node] + dt * v[node]);
    box.min.array() -= pad;
    box.max.array() += pad;
    return box;
}

static SweptAABB3 make_swept_triangle_box(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int a, int b, int c, double dt, double pad) {
    SweptAABB3 box;
    box.min = x[a].cwiseMin(x[a] + dt * v[a]).cwiseMin(x[b].cwiseMin(x[b] + dt * v[b])).cwiseMin(x[c].cwiseMin(x[c] + dt * v[c]));
    box.max = x[a].cwiseMax(x[a] + dt * v[a]).cwiseMax(x[b].cwiseMax(x[b] + dt * v[b])).cwiseMax(x[c].cwiseMax(x[c] + dt * v[c]));
    box.min.array() -= pad;
    box.max.array() += pad;
    return box;
}

static SweptAABB3 make_swept_edge_box(const std::vector<Vec3>& x, const std::vector<Vec3>& v, int a, int b, double dt, double pad) {
    SweptAABB3 box;
    box.min = x[a].cwiseMin(x[a] + dt * v[a]).cwiseMin(x[b].cwiseMin(x[b] + dt * v[b]));
    box.max = x[a].cwiseMax(x[a] + dt * v[a]).cwiseMax(x[b].cwiseMax(x[b] + dt * v[b]));
    box.min.array() -= pad;
    box.max.array() += pad;
    return box;
}

static bool swept_aabb_intersects(const SweptAABB3& a, const SweptAABB3& b) {
    return (a.min.array() <= b.max.array()).all() && (a.max.array() >= b.min.array()).all();
}

static Vec3 swept_aabb_extent(const SweptAABB3& b) {
    return b.max - b.min;
}

static Vec3 swept_aabb_centroid(const SweptAABB3& b) {
    return 0.5 * (b.min + b.max);
}

static SweptAABB3 merge_swept_aabb(const SweptAABB3& a, const SweptAABB3& b) {
    SweptAABB3 out;
    out.min = a.min.cwiseMin(b.min);
    out.max = a.max.cwiseMax(b.max);
    return out;
}

struct BVHNode3 {
    SweptAABB3 box;
    int left{-1};
    int right{-1};
    int leaf_primitive{-1};
    int parent{-1};
};

static int build_bvh3(const std::vector<SweptAABB3>& boxes, std::vector<BVHNode3>& out_nodes) {
    out_nodes.clear();
    if (boxes.empty()) return -1;

    std::vector<int> idx(boxes.size());
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) idx[i] = i;

    struct Task { int node_idx; int begin; int end; };
    std::vector<Task> stack;
    out_nodes.push_back(BVHNode3{});
    stack.push_back(Task{0, 0, static_cast<int>(idx.size())});

    while (!stack.empty()) {
        const Task task = stack.back();
        stack.pop_back();

        SweptAABB3 node_box = boxes[idx[task.begin]];
        for (int i = task.begin + 1; i < task.end; ++i) {
            node_box = merge_swept_aabb(node_box, boxes[idx[i]]);
        }
        out_nodes[task.node_idx].box = node_box;

        const int count = task.end - task.begin;
        if (count == 1) {
            out_nodes[task.node_idx].leaf_primitive = idx[task.begin];
            continue;
        }

        const Vec3 e = swept_aabb_extent(node_box);
        int axis = 0;
        if (e.y() > e.x() && e.y() >= e.z()) axis = 1;
        else if (e.z() > e.x() && e.z() >= e.y()) axis = 2;

        const int mid = task.begin + count / 2;
        std::nth_element(
                idx.begin() + task.begin, idx.begin() + mid, idx.begin() + task.end,
                [&](int a, int b) {
                    return swept_aabb_centroid(boxes[a])[axis] < swept_aabb_centroid(boxes[b])[axis];
                });

        const int left = static_cast<int>(out_nodes.size());
        out_nodes.push_back(BVHNode3{});
        const int right = static_cast<int>(out_nodes.size());
        out_nodes.push_back(BVHNode3{});

        out_nodes[task.node_idx].left = left;
        out_nodes[task.node_idx].right = right;
        out_nodes[left].parent = task.node_idx;
        out_nodes[right].parent = task.node_idx;

        stack.push_back(Task{right, mid, task.end});
        stack.push_back(Task{left, task.begin, mid});
    }

    return 0;
}

static void query_bvh3(const std::vector<BVHNode3>& nodes, int root, const SweptAABB3& box, std::vector<int>& hits) {
    hits.clear();
    if (root < 0) return;

    std::vector<int> stack;
    stack.push_back(root);
    while (!stack.empty()) {
        const int ni = stack.back();
        stack.pop_back();
        const BVHNode3& n = nodes[ni];
        if (!swept_aabb_intersects(n.box, box)) continue;
        if (n.leaf_primitive >= 0) {
            hits.push_back(n.leaf_primitive);
        } else {
            stack.push_back(n.left);
            stack.push_back(n.right);
        }
    }
}

static std::vector<int> build_leaf_node_map(const std::vector<BVHNode3>& nodes, int primitive_count) {
    std::vector<int> leaf_node_of_primitive(primitive_count, -1);
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        if (nodes[i].leaf_primitive >= 0) {
            leaf_node_of_primitive[nodes[i].leaf_primitive] = i;
        }
    }
    return leaf_node_of_primitive;
}

static void refit_bvh3_from_leaf(
        std::vector<BVHNode3>& nodes,
        const std::vector<SweptAABB3>& boxes,
        int leaf_node_idx) {
    if (leaf_node_idx < 0) return;
    int ni = leaf_node_idx;
    while (ni >= 0) {
        BVHNode3& n = nodes[ni];
        if (n.leaf_primitive >= 0) {
            n.box = boxes[n.leaf_primitive];
        } else {
            n.box = merge_swept_aabb(nodes[n.left].box, nodes[n.right].box);
        }
        ni = n.parent;
    }
}

struct BroadPhaseState {
    std::vector<std::array<int, 2>> edges;
    std::vector<std::vector<int>> incident_tris_by_vertex;
    std::vector<std::vector<int>> incident_edges_by_vertex;

    std::vector<SweptAABB3> node_boxes;
    std::vector<SweptAABB3> tri_boxes;
    std::vector<SweptAABB3> edge_boxes;

    std::vector<BVHNode3> tri_bvh;
    std::vector<BVHNode3> edge_bvh;
    int tri_root{-1};
    int edge_root{-1};
    std::vector<int> tri_leaf_node_of_primitive;
    std::vector<int> edge_leaf_node_of_primitive;

    std::vector<NodeTrianglePair> nt_pairs;
    std::vector<int> nt_pair_tri_indices;
    std::unordered_map<std::uint64_t, std::size_t> nt_pair_index;

    std::vector<SegmentSegmentPair> ss_pairs;
    std::vector<std::array<int, 2>> ss_pair_edge_indices;
    std::unordered_map<std::uint64_t, std::size_t> ss_pair_index;

    double dt{0.0};
    double d_hat{0.0};
    bool use_ccd_filter{false};
};

static std::uint64_t nt_pair_key(int node, int tri) {
    return (std::uint64_t(std::uint32_t(node)) << 32) | std::uint32_t(tri);
}

static std::uint64_t ss_pair_key(int e0, int e1) {
    if (e0 > e1) std::swap(e0, e1);
    return (std::uint64_t(std::uint32_t(e0)) << 32) | std::uint32_t(e1);
}

static bool edge_share_vertex(const std::array<int, 2>& a, const std::array<int, 2>& b) {
    return a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1];
}

static void remove_nt_pair_at(BroadPhaseState& bp, std::size_t idx) {
    const std::size_t last = bp.nt_pairs.size() - 1;
    const int removed_node = bp.nt_pairs[idx].node;
    const int removed_tri = bp.nt_pair_tri_indices[idx];
    bp.nt_pair_index.erase(nt_pair_key(removed_node, removed_tri));

    if (idx != last) {
        bp.nt_pairs[idx] = bp.nt_pairs[last];
        bp.nt_pair_tri_indices[idx] = bp.nt_pair_tri_indices[last];
        bp.nt_pair_index[nt_pair_key(bp.nt_pairs[idx].node, bp.nt_pair_tri_indices[idx])] = idx;
    }

    bp.nt_pairs.pop_back();
    bp.nt_pair_tri_indices.pop_back();
}

static void remove_ss_pair_at(BroadPhaseState& bp, std::size_t idx) {
    const std::size_t last = bp.ss_pairs.size() - 1;
    const int e0_removed = bp.ss_pair_edge_indices[idx][0];
    const int e1_removed = bp.ss_pair_edge_indices[idx][1];
    bp.ss_pair_index.erase(ss_pair_key(e0_removed, e1_removed));

    if (idx != last) {
        bp.ss_pairs[idx] = bp.ss_pairs[last];
        bp.ss_pair_edge_indices[idx] = bp.ss_pair_edge_indices[last];
        bp.ss_pair_index[ss_pair_key(bp.ss_pair_edge_indices[idx][0], bp.ss_pair_edge_indices[idx][1])] = idx;
    }

    bp.ss_pairs.pop_back();
    bp.ss_pair_edge_indices.pop_back();
}

static void remove_nt_pairs_for_node(BroadPhaseState& bp, int node) {
    std::size_t i = 0;
    while (i < bp.nt_pairs.size()) {
        if (bp.nt_pairs[i].node == node) {
            remove_nt_pair_at(bp, i);
        } else {
            ++i;
        }
    }
}

static void remove_nt_pairs_for_triangle(BroadPhaseState& bp, int tri_idx) {
    std::size_t i = 0;
    while (i < bp.nt_pairs.size()) {
        if (bp.nt_pair_tri_indices[i] == tri_idx) {
            remove_nt_pair_at(bp, i);
        } else {
            ++i;
        }
    }
}

static void remove_ss_pairs_for_edge(BroadPhaseState& bp, int edge_idx) {
    std::size_t i = 0;
    while (i < bp.ss_pairs.size()) {
        const auto& e = bp.ss_pair_edge_indices[i];
        if (e[0] == edge_idx || e[1] == edge_idx) {
            remove_ss_pair_at(bp, i);
        } else {
            ++i;
        }
    }
}

static bool passes_nt_ccd(
        const BroadPhaseState& bp,
        const RefMesh& mesh,
        const std::vector<Vec3>& x,
        const std::vector<Vec3>& v,
        int node,
        int tri_idx) {
    if (!bp.use_ccd_filter) return true;

    const int v0 = tri_vertex(mesh, tri_idx, 0);
    const int v1 = tri_vertex(mesh, tri_idx, 1);
    const int v2 = tri_vertex(mesh, tri_idx, 2);

    const double toi = ccd::node_triangle_3d(
            ccd::make_vec3(x[node].x(), x[node].y(), x[node].z()),
            ccd::make_vec3(bp.dt * v[node].x(), bp.dt * v[node].y(), bp.dt * v[node].z()),
            ccd::make_vec3(x[v0].x(), x[v0].y(), x[v0].z()),
            ccd::make_vec3(bp.dt * v[v0].x(), bp.dt * v[v0].y(), bp.dt * v[v0].z()),
            ccd::make_vec3(x[v1].x(), x[v1].y(), x[v1].z()),
            ccd::make_vec3(bp.dt * v[v1].x(), bp.dt * v[v1].y(), bp.dt * v[v1].z()),
            ccd::make_vec3(x[v2].x(), x[v2].y(), x[v2].z()),
            ccd::make_vec3(bp.dt * v[v2].x(), bp.dt * v[v2].y(), bp.dt * v[v2].z()));
    return std::isfinite(toi);
}

static bool passes_ss_ccd(
        const BroadPhaseState& bp,
        const std::vector<std::array<int, 2>>& edges,
        const std::vector<Vec3>& x,
        const std::vector<Vec3>& v,
        int e0,
        int e1) {
    if (!bp.use_ccd_filter) return true;

    const int a0 = edges[e0][0];
    const int a1 = edges[e0][1];
    const int b0 = edges[e1][0];
    const int b1 = edges[e1][1];

    const double toi = ccd::segment_segment_3d(
            ccd::make_vec3(x[a0].x(), x[a0].y(), x[a0].z()),
            ccd::make_vec3(bp.dt * v[a0].x(), bp.dt * v[a0].y(), bp.dt * v[a0].z()),
            ccd::make_vec3(x[a1].x(), x[a1].y(), x[a1].z()),
            ccd::make_vec3(bp.dt * v[a1].x(), bp.dt * v[a1].y(), bp.dt * v[a1].z()),
            ccd::make_vec3(x[b0].x(), x[b0].y(), x[b0].z()),
            ccd::make_vec3(bp.dt * v[b0].x(), bp.dt * v[b0].y(), bp.dt * v[b0].z()),
            ccd::make_vec3(x[b1].x(), x[b1].y(), x[b1].z()),
            ccd::make_vec3(bp.dt * v[b1].x(), bp.dt * v[b1].y(), bp.dt * v[b1].z()));
    return std::isfinite(toi);
}

static void add_nt_pair(
        BroadPhaseState& bp,
        const RefMesh& mesh,
        const std::vector<Vec3>& x,
        const std::vector<Vec3>& v,
        int node,
        int tri_idx) {
    const std::uint64_t key = nt_pair_key(node, tri_idx);
    if (bp.nt_pair_index.count(key)) return;
    if (!passes_nt_ccd(bp, mesh, x, v, node, tri_idx)) return;

    NodeTrianglePair p;
    p.node = node;
    p.tri_v[0] = tri_vertex(mesh, tri_idx, 0);
    p.tri_v[1] = tri_vertex(mesh, tri_idx, 1);
    p.tri_v[2] = tri_vertex(mesh, tri_idx, 2);

    const std::size_t idx = bp.nt_pairs.size();
    bp.nt_pairs.push_back(p);
    bp.nt_pair_tri_indices.push_back(tri_idx);
    bp.nt_pair_index[key] = idx;
}

static void add_ss_pair(
        BroadPhaseState& bp,
        const std::vector<Vec3>& x,
        const std::vector<Vec3>& v,
        int e0,
        int e1) {
    if (e0 > e1) std::swap(e0, e1);
    const std::uint64_t key = ss_pair_key(e0, e1);
    if (bp.ss_pair_index.count(key)) return;
    if (!passes_ss_ccd(bp, bp.edges, x, v, e0, e1)) return;

    const auto& a = bp.edges[e0];
    const auto& b = bp.edges[e1];
    SegmentSegmentPair p{{a[0], a[1], b[0], b[1]}};

    const std::size_t idx = bp.ss_pairs.size();
    bp.ss_pairs.push_back(p);
    bp.ss_pair_edge_indices.push_back({e0, e1});
    bp.ss_pair_index[key] = idx;
}

static void initialize_broad_phase_topology(BroadPhaseState& bp, const RefMesh& mesh) {
    bp.edges = extract_all_unique_edges(mesh);

    const int nv = static_cast<int>(mesh.num_positions);
    const int nt = num_tris(mesh);
    const int ne = static_cast<int>(bp.edges.size());

    bp.incident_tris_by_vertex.assign(nv, {});
    for (int t = 0; t < nt; ++t) {
        for (int l = 0; l < 3; ++l) {
            bp.incident_tris_by_vertex[tri_vertex(mesh, t, l)].push_back(t);
        }
    }

    bp.incident_edges_by_vertex.assign(nv, {});
    for (int e = 0; e < ne; ++e) {
        bp.incident_edges_by_vertex[bp.edges[e][0]].push_back(e);
        bp.incident_edges_by_vertex[bp.edges[e][1]].push_back(e);
    }
}

static void broad_phase_initialize(
        BroadPhaseState& bp,
        const RefMesh& mesh,
        const std::vector<Vec3>& x,
        const std::vector<Vec3>& v,
        double dt,
        double d_hat,
        bool use_ccd_filter) {
    if (bp.edges.empty()) {
        initialize_broad_phase_topology(bp, mesh);
    }

    bp.dt = dt;
    bp.d_hat = d_hat;
    bp.use_ccd_filter = use_ccd_filter;
    bp.nt_pairs.clear();
    bp.nt_pair_tri_indices.clear();
    bp.nt_pair_index.clear();
    bp.ss_pairs.clear();
    bp.ss_pair_edge_indices.clear();
    bp.ss_pair_index.clear();

    if (d_hat <= 0.0) return;

    const int nv = static_cast<int>(mesh.num_positions);
    const int nt = num_tris(mesh);
    const int ne = static_cast<int>(bp.edges.size());

    bp.node_boxes.resize(nv);
    bp.tri_boxes.resize(nt);
    bp.edge_boxes.resize(ne);

    for (int i = 0; i < nv; ++i) {
        bp.node_boxes[i] = make_swept_node_box(x, v, i, dt, d_hat);
    }
    for (int t = 0; t < nt; ++t) {
        const int a = tri_vertex(mesh, t, 0);
        const int b = tri_vertex(mesh, t, 1);
        const int c = tri_vertex(mesh, t, 2);
        bp.tri_boxes[t] = make_swept_triangle_box(x, v, a, b, c, dt, d_hat);
    }
    for (int e = 0; e < ne; ++e) {
        bp.edge_boxes[e] = make_swept_edge_box(x, v, bp.edges[e][0], bp.edges[e][1], dt, d_hat);
    }

    bp.tri_root = build_bvh3(bp.tri_boxes, bp.tri_bvh);
    bp.edge_root = build_bvh3(bp.edge_boxes, bp.edge_bvh);
    bp.tri_leaf_node_of_primitive = build_leaf_node_map(bp.tri_bvh, nt);
    bp.edge_leaf_node_of_primitive = build_leaf_node_map(bp.edge_bvh, ne);

    std::vector<int> hits;
    for (int node = 0; node < nv; ++node) {
        query_bvh3(bp.tri_bvh, bp.tri_root, bp.node_boxes[node], hits);
        for (int t : hits) {
            const int a = tri_vertex(mesh, t, 0);
            const int b = tri_vertex(mesh, t, 1);
            const int c = tri_vertex(mesh, t, 2);
            if (node == a || node == b || node == c) continue;
            add_nt_pair(bp, mesh, x, v, node, t);
        }
    }

    for (int e = 0; e < ne; ++e) {
        query_bvh3(bp.edge_bvh, bp.edge_root, bp.edge_boxes[e], hits);
        for (int f : hits) {
            if (f == e) continue;
            if (edge_share_vertex(bp.edges[e], bp.edges[f])) continue;
            add_ss_pair(bp, x, v, e, f);
        }
    }
}

static void broad_phase_refresh_after_vertex_move(
        BroadPhaseState& bp,
        const RefMesh& mesh,
        const std::vector<Vec3>& x,
        const std::vector<Vec3>& v,
        int moved_node) {
    if (bp.d_hat <= 0.0) return;

    const int nv = static_cast<int>(mesh.num_positions);
    bp.node_boxes[moved_node] = make_swept_node_box(x, v, moved_node, bp.dt, bp.d_hat);

    for (int tri_idx : bp.incident_tris_by_vertex[moved_node]) {
        const int a = tri_vertex(mesh, tri_idx, 0);
        const int b = tri_vertex(mesh, tri_idx, 1);
        const int c = tri_vertex(mesh, tri_idx, 2);
        bp.tri_boxes[tri_idx] = make_swept_triangle_box(x, v, a, b, c, bp.dt, bp.d_hat);
        refit_bvh3_from_leaf(bp.tri_bvh, bp.tri_boxes, bp.tri_leaf_node_of_primitive[tri_idx]);
    }

    for (int edge_idx : bp.incident_edges_by_vertex[moved_node]) {
        const int a = bp.edges[edge_idx][0];
        const int b = bp.edges[edge_idx][1];
        bp.edge_boxes[edge_idx] = make_swept_edge_box(x, v, a, b, bp.dt, bp.d_hat);
        refit_bvh3_from_leaf(bp.edge_bvh, bp.edge_boxes, bp.edge_leaf_node_of_primitive[edge_idx]);
    }

    remove_nt_pairs_for_node(bp, moved_node);
    std::vector<int> hits;
    query_bvh3(bp.tri_bvh, bp.tri_root, bp.node_boxes[moved_node], hits);
    for (int t : hits) {
        const int a = tri_vertex(mesh, t, 0);
        const int b = tri_vertex(mesh, t, 1);
        const int c = tri_vertex(mesh, t, 2);
        if (moved_node == a || moved_node == b || moved_node == c) continue;
        add_nt_pair(bp, mesh, x, v, moved_node, t);
    }

    for (int tri_idx : bp.incident_tris_by_vertex[moved_node]) {
        remove_nt_pairs_for_triangle(bp, tri_idx);
        const int a = tri_vertex(mesh, tri_idx, 0);
        const int b = tri_vertex(mesh, tri_idx, 1);
        const int c = tri_vertex(mesh, tri_idx, 2);
        for (int node = 0; node < nv; ++node) {
            if (node == a || node == b || node == c) continue;
            if (!swept_aabb_intersects(bp.node_boxes[node], bp.tri_boxes[tri_idx])) continue;
            add_nt_pair(bp, mesh, x, v, node, tri_idx);
        }
    }

    for (int edge_idx : bp.incident_edges_by_vertex[moved_node]) {
        remove_ss_pairs_for_edge(bp, edge_idx);
        query_bvh3(bp.edge_bvh, bp.edge_root, bp.edge_boxes[edge_idx], hits);
        for (int other_edge_idx : hits) {
            if (other_edge_idx == edge_idx) continue;
            if (edge_share_vertex(bp.edges[edge_idx], bp.edges[other_edge_idx])) continue;
            add_ss_pair(bp, x, v, edge_idx, other_edge_idx);
        }
    }
}

// ============================================================
// Solver / energy assembly
// ============================================================
static std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(
        int vi, const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat) {
    const double dt2 = params.dt2();
    Vec3 g = Vec3::Zero();
    Mat33 H = Mat33::Zero();

    g += mesh.mass[vi] * (x[vi] - xhat[vi]);
    g += dt2 * (-mesh.mass[vi] * params.gravity);
    H += mesh.mass[vi] * Mat33::Identity();

    for (const auto& pin : pins) {
        if (pin.vertex_index == vi) {
            g += dt2 * params.kpin * (x[vi] - pin.target_position);
            H += dt2 * params.kpin * Mat33::Identity();
        }
    }

    const auto it = adj.find(vi);
    if (it != adj.end()) {
        for (const auto& [tri_idx, local_node] : it->second) {
            const TriangleDef def = make_def_triangle(x, mesh, tri_idx);
            const Mat32 F = Ds(def) * mesh.Dm_inv[tri_idx];
            const double A = mesh.area[tri_idx];

            const CorotatedCache32 cache = buildCorotatedCache(F);
            const ShapeGrads gradN = shape_function_gradients(mesh.Dm_inv[tri_idx]);
            const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
            Mat66 dPdF;
            dPdFCorotated32(cache, params.mu, params.lambda, dPdF);

            g += dt2 * corotated_node_gradient(P, A, gradN, local_node);
            H += dt2 * corotated_node_hessian(dPdF, A, gradN, local_node).template block<3, 3>(0, 3 * local_node);
        }
    }

    return {g, H};
}

static Vec3 compute_local_gradient(
        int vi, const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
        const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    Vec3 g = compute_local_gradient_and_hessian_no_barrier(vi, mesh, adj, pins, params, x, xhat).first;
    if (params.d_hat <= 0.0) return g;
    const double dt2 = params.dt2();
    for (const auto& p : nt_pairs) {
        int dof = -1;
        if      (vi == p.node)      dof = 0;
        else if (vi == p.tri_v[0])  dof = 1;
        else if (vi == p.tri_v[1])  dof = 2;
        else if (vi == p.tri_v[2])  dof = 3;
        if (dof >= 0) g += dt2 * node_triangle_barrier_gradient(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
    }
    for (const auto& p : ss_pairs) {
        int dof = -1;
        if      (vi == p.v[0]) dof = 0;
        else if (vi == p.v[1]) dof = 1;
        else if (vi == p.v[2]) dof = 2;
        else if (vi == p.v[3]) dof = 3;
        if (dof >= 0) g += dt2 * segment_segment_barrier_gradient(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, dof);
    }
    return g;
}

static double compute_global_residual(
        const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
        const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    double r_inf = 0.0;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        const Vec3 g = compute_local_gradient(i, mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
        r_inf = std::max(r_inf, g.cwiseAbs().maxCoeff());
    }
    return r_inf;
}

static void update_one_vertex(
        int vi, const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, const std::vector<Vec3>& xhat, std::vector<Vec3>& x,
        const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    static bool printed = false;
    if (!printed) {
        std::cout << "update_one_vertex: barrier "
                  << (params.d_hat > 0.0 ? "ENABLED" : "DISABLED")
                  << ", d_hat = " << params.d_hat << "\n";
        printed = true;
    }

    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, mesh, adj, pins, params, x, xhat);

    if (params.d_hat > 0.0) {
        const double dt2 = params.dt2();
        for (const auto& p : nt_pairs) {
            int dof = -1;
            if      (vi == p.node)      dof = 0;
            else if (vi == p.tri_v[0])  dof = 1;
            else if (vi == p.tri_v[1])  dof = 2;
            else if (vi == p.tri_v[2])  dof = 3;
            if (dof < 0) continue;
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
            g += dt2 * bg;
            H += dt2 * bH.block<3, 3>(0, 3 * dof);
        }
        for (const auto& p : ss_pairs) {
            int dof = -1;
            if      (vi == p.v[0]) dof = 0;
            else if (vi == p.v[1]) dof = 1;
            else if (vi == p.v[2]) dof = 2;
            else if (vi == p.v[3]) dof = 3;
            if (dof < 0) continue;
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, dof);
            g += dt2 * bg;
            H += dt2 * bH.block<3, 3>(0, 3 * dof);
        }
    }

    // Mild diagonal regularization improves robustness when barrier terms stiffen up.
    H += 1.0e-8 * Mat33::Identity();
    const Vec3 dx = matrix3d_inverse(H) * g;
    x[vi] -= params.step_weight * dx;
}

static SolverResult global_gauss_seidel_solver(
        const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
        const std::vector<Vec3>& sweep_v, BroadPhaseState& broad_phase) {
    SolverResult result;
    result.initial_residual = compute_global_residual(
            mesh, adj, pins, params, xnew, xhat, broad_phase.nt_pairs, broad_phase.ss_pairs);
    result.final_residual = result.initial_residual;
    if (result.initial_residual < params.tol_abs) return result;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        for (int vi = 0; vi < static_cast<int>(xnew.size()); ++vi) {
            update_one_vertex(vi, mesh, adj, pins, params, xhat, xnew, broad_phase.nt_pairs, broad_phase.ss_pairs);
            broad_phase_refresh_after_vertex_move(broad_phase, mesh, xnew, sweep_v, vi);
        }
        result.final_residual = compute_global_residual(
                mesh, adj, pins, params, xnew, xhat, broad_phase.nt_pairs, broad_phase.ss_pairs);
        result.iterations = iter;
        if (result.final_residual < params.tol_abs) return result;
    }
    return result;
}

// ============================================================
// Output
// ============================================================
static void export_obj(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }
    for (const auto& p : x) out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    for (int t = 0; t < static_cast<int>(tris.size()); t += 3) {
        out << "f " << (tris[t] + 1) << ' ' << (tris[t + 1] + 1) << ' ' << (tris[t + 2] + 1) << '\n';
    }
}

static void export_frame(const std::string& outdir, int frame, const std::vector<Vec3>& x, const std::vector<int>& tris) {
    std::ostringstream ss;
    ss << outdir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    export_obj(ss.str(), x, tris);
}

// ============================================================
// Main experiment: two sheets, inter-sheet pairs only
// ============================================================
int main() {
    SimParams params;
    params.fps = 30.0;
    params.substeps = 1;
    params.num_frames = 60;
    params.mu = 10.0;
    params.lambda = 10.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.kpin = 1.0e7;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 500;
    params.tol_abs = 1.0e-6;
    params.step_weight = 1.0;

    // Toggle this to compare no-barrier vs barrier runs.
    params.d_hat = 0.10;

    std::cout << "num_frames = " << params.num_frames << "\n";
    std::cout << "d_hat = " << params.d_hat
              << (params.d_hat > 0.0 ? "  (barrier ON)" : "  (barrier OFF)")
              << "\n";

    RefMesh mesh;
    DeformedState state;
    std::vector<Pin> pins;
    clear_scene(mesh, state, pins);

    // Two separate sheets, side-by-side in x, with a small z offset.
    const PatchInfo left = build_square_patch(
            mesh, state, 10, 10, 1.0, 1.0,
            Vec3(-0.75, 0.20, 0.10)
    );

    const PatchInfo right = build_square_patch(
            mesh, state, 10, 10, 1.0, 1.0,
            Vec3(0.75, 0.20, 0.10)
    );

    state.v.assign(state.x.size(), Vec3::Zero());

    // Tiny asymmetry so they do not evolve perfectly symmetrically.
    state.x[right.vertex_begin + 0] += Vec3(-0.02, 0.00, 0.00);
    state.x[right.vertex_begin + 2] += Vec3(-0.02, 0.00, 0.00);

    // Pin the inner-side top and bottom corners.
    append_pin(pins, patch_corner_index(left,  left.nx, left.ny), state.x);   // left top-right
    append_pin(pins, patch_corner_index(left,  left.nx, 0),       state.x);   // left bottom-right

    append_pin(pins, patch_corner_index(right, 0,        right.ny), state.x); // right top-left
    append_pin(pins, patch_corner_index(right, 0,        0),        state.x); // right bottom-left

    mesh.initialize();
    mesh.build_lumped_mass(params.density, params.thickness);
    const VertexTriangleMap adj = build_incident_triangle_map(mesh.tris);

    // Broad phase over swept AABBs, initialized once per substep then
    // maintained incrementally during Gauss-Seidel vertex updates.
    BroadPhaseState broad_phase;
    const bool use_ccd_filter = true;
    broad_phase_initialize(broad_phase, mesh, state.x, state.v, params.dt(), params.d_hat, use_ccd_filter);

    std::cout << "Vertices:  " << state.x.size() << "\n";
    std::cout << "Triangles: " << num_tris(mesh) << "\n";
    std::cout << "NT pairs:  " << broad_phase.nt_pairs.size() << "\n";
    std::cout << "SS pairs:  " << broad_phase.ss_pairs.size() << "\n";
    std::cout << "CCD filter: " << (use_ccd_filter ? "ON" : "OFF") << "\n";

    const std::string outdir = (params.d_hat > 0.0) ? "frames_clean_barrier_on" : "frames_clean_barrier_off";
    if (fs::exists(outdir)) fs::remove_all(outdir);
    fs::create_directories(outdir);
    export_frame(outdir, 0, state.x, mesh.tris);

    using Clock = std::chrono::steady_clock;
    const auto sim_start = Clock::now();
    double total_solver_ms = 0.0;

    for (int frame = 1; frame <= params.num_frames; ++frame) {
        const auto solver_start = Clock::now();
        SolverResult result;

        for (int sub = 0; sub < params.substeps; ++sub) {
            std::vector<Vec3> xhat;
            build_xhat(xhat, state.x, state.v, params.dt());

            std::vector<Vec3> xnew = state.x;
            broad_phase_initialize(broad_phase, mesh, state.x, state.v, params.dt(), params.d_hat, use_ccd_filter);
            result = global_gauss_seidel_solver(mesh, adj, pins, params, xnew, xhat, state.v, broad_phase);
            update_velocity(state.v, xnew, state.x, params.dt());
            state.x = xnew;
        }

        const auto solver_end = Clock::now();
        const double solver_ms = std::chrono::duration<double, std::milli>(solver_end - solver_start).count();
        total_solver_ms += solver_ms;

        std::cout << "Frame " << std::setw(4) << frame
                  << " | initial_residual = " << std::scientific << result.initial_residual
                  << " | final_residual = "   << std::scientific << result.final_residual
                  << " | global_iters = "     << std::setw(3)    << result.iterations
                  << " | solver_time = "      << std::fixed << std::setprecision(3)
                  << solver_ms << " ms\n";

        export_frame(outdir, frame, state.x, mesh.tris);
    }

    const auto sim_end = Clock::now();
    const double total_sim_ms = std::chrono::duration<double, std::milli>(sim_end - sim_start).count();

    std::cout << "\nSimulation finished.\n";
    std::cout << "Total simulation time: " << std::fixed << std::setprecision(3) << total_sim_ms << " ms\n";
    std::cout << "Total solver time:     " << total_solver_ms << " ms\n";
    std::cout << "Average solver time:   " << (total_solver_ms / params.num_frames) << " ms/frame\n";
    std::cout << "Frames written to:     " << outdir << "\n";
    return 0;
}
