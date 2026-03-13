#include "physics.h"
#include <cmath>
#include <functional>

namespace physics {
    using namespace math;

    // ======================================================
    // Spring energy
    // ======================================================

    Vec2 local_spring_grad(int i, const Vec &x, double k, const std::vector<double> &L) {
        Vec2 g_i{0.0, 0.0};

        std::function<void(int, int, double)> contrib = [&](int a, int b, double Lref) {
            Vec2 xa = get_xi(x, a), xb = get_xi(x, b);
            double dx = xb.x - xa.x, dy = xb.y - xa.y;
            double ell = std::sqrt(dx * dx + dy * dy);
            if (ell < 1e-12) return;

            double coeff = k / Lref * (ell - Lref) / ell;
            double sgn = (i == b) ? +1.0 : -1.0;

            g_i.x += sgn * coeff * dx;
            g_i.y += sgn * coeff * dy;
        };

        int N = (int)L.size() + 1;
        if (i - 1 >= 0)     contrib(i - 1, i,     L[i - 1]);
        if (i + 1 <= N - 1) contrib(i,     i + 1, L[i]);

        return g_i;
    }

    Mat2 local_spring_hess(int i, const Vec &x, double k, const std::vector<double> &L) {
        Mat2 H_ii{0.0, 0.0, 0.0, 0.0};

        std::function<void(int, int, double)> contrib = [&](int a, int b, double Lref) {
            Vec2 xa = get_xi(x, a), xb = get_xi(x, b);
            double dx = xb.x - xa.x, dy = xb.y - xa.y;
            double ell = std::sqrt(dx * dx + dy * dy);
            if (ell < 1e-12) return;

            double coeff1 = k / Lref * (ell - Lref) / ell;
            double coeff2 = k / Lref * Lref / (ell * ell * ell);

            double Kxx = coeff1 + coeff2 * dx * dx;
            double Kyy = coeff1 + coeff2 * dy * dy;
            double Kxy = coeff2 * dx * dy;

            H_ii.a11 += Kxx;
            H_ii.a12 += Kxy;
            H_ii.a21 += Kxy;
            H_ii.a22 += Kyy;
        };

        int N = (int)L.size() + 1;
        if (i - 1 >= 0)     contrib(i - 1, i,     L[i - 1]);
        if (i + 1 <= N - 1) contrib(i,     i + 1, L[i]);

        return H_ii;
    }

    // ======================================================
    // Scalar barrier
    // ======================================================

    double barrier_energy(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -(d - dhat) * (d - dhat) * std::log(d / dhat);
    }

    double barrier_grad(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -2 * (d - dhat) * std::log(d / dhat) - (d - dhat) * (d - dhat) / d;
    }

    double barrier_hess(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -2 * std::log(d / dhat) - 4 * (d - dhat) / d + (d - dhat) * (d - dhat) / (d * d);
    }

    // ======================================================
    // Point-segment distance
    // ======================================================

    double node_segment_distance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1,
                                 double &t, Vec2 &p, Vec2 &r) {
        Vec2 seg = {xjp1.x - xj.x, xjp1.y - xj.y};
        double seg_len2 = seg.x * seg.x + seg.y * seg.y;

        if (seg_len2 < 1e-14) {
            t = 0.0;
            p = xj;
            r = {xi.x - p.x, xi.y - p.y};
            return std::sqrt(r.x * r.x + r.y * r.y);
        }

        Vec2 q  = {xi.x - xj.x, xi.y - xj.y};
        double dot = q.x * seg.x + q.y * seg.y;
        t = dot / seg_len2;
        t = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);

        p = {xj.x + t * seg.x, xj.y + t * seg.y};
        r = {xi.x - p.x, xi.y - p.y};
        return std::sqrt(r.x * r.x + r.y * r.y);
    }

    // ======================================================
    // Barrier gradient helpers (internal)
    // ======================================================

    static Vec2 get_v(const Vec &x, int node, int seg0) {
        return sub(get_xi(x, node), get_xi(x, seg0));
    }

    static double get_theta(const Vec2 &u, const Vec2 &v) {
        return u.x * v.x + u.y * v.y;
    }

    static Mat2x2x2 get_grad_jp_analytic(const Vec2 &u, const Mat2 &P, double L) {
        Mat2x2x2 gradP{};

        double P_il_x = P.a11, P_il_y = P.a21;
        gradP.m0.a11 = (1.0 / L) * (P_il_x * u.x + u.x * P_il_x);
        gradP.m0.a12 = (1.0 / L) * (P_il_x * u.y + u.x * P.a12);
        gradP.m0.a21 = (1.0 / L) * (P_il_y * u.x + u.y * P_il_x);
        gradP.m0.a22 = (1.0 / L) * (P_il_y * u.y + u.y * P.a12);

        double P_il_x_y = P.a12, P_il_y_y = P.a22;
        gradP.m1.a11 = (1.0 / L) * (P_il_x_y * u.x + u.x * P_il_x_y);
        gradP.m1.a12 = (1.0 / L) * (P_il_x_y * u.y + u.x * P_il_y_y);
        gradP.m1.a21 = (1.0 / L) * (P_il_y_y * u.x + u.y * P_il_x_y);
        gradP.m1.a22 = (1.0 / L) * (P_il_y_y * u.y + u.y * P_il_y_y);

        return {scale(gradP.m0, -1.0), scale(gradP.m1, -1.0)};
    }

    static Mat2x2x2 get_grad_j_common_analytic(double L, const Vec2 &u, const Vec2 &v,
                                                double theta, const Mat2 &P, const Vec2 &r) {
        Mat2 I = {1, 0, 0, 1};
        Mat2 T = add(add(scale(I, theta), outer(u, v)), scale(outer(u, u), -2 * theta));

        Mat2 dJ_dx = scale(T, u.x / (L * L));
        Mat2 dJ_dy = scale(T, u.y / (L * L));

        auto build_term_b = [&](double dtheta, const Vec2 &dv, const Vec2 &du) -> Mat2 {
            Mat2 dT = {0, 0, 0, 0};
            dT = add(dT, scale(I, dtheta));
            dT = add(dT, outer(du, v));
            dT = add(dT, outer(u, dv));
            dT = add(dT, scale(outer(u, u), -2.0 * dtheta));
            dT = add(dT, scale(outer(du, u), -2.0 * theta));
            dT = add(dT, scale(outer(u, du), -2.0 * theta));
            return dT;
        };

        dJ_dx = add(dJ_dx, scale(build_term_b(-r.x / L - u.x, {-1, 0}, {-P.a11 / L, -P.a21 / L}), 1.0 / L));
        dJ_dy = add(dJ_dy, scale(build_term_b(-r.y / L - u.y, {0, -1}, {-P.a12 / L, -P.a22 / L}), 1.0 / L));

        return {dJ_dx, dJ_dy};
    }

    static Mat2x2x2 get_grad_j_common_analytic_x2(double L, const Vec2 &u, const Vec2 &v,
                                                    double theta, const Mat2 &P, const Vec2 &r) {
        Mat2 I = {1, 0, 0, 1};
        Mat2 T = add(add(scale(I, theta), outer(u, v)), scale(outer(u, u), -2 * theta));

        Mat2 dJ_dx = scale(T, -u.x / (L * L));
        Mat2 dJ_dy = scale(T, -u.y / (L * L));

        auto build_term_b = [&](double dtheta, const Vec2 &du) -> Mat2 {
            Mat2 dT = {0, 0, 0, 0};
            dT = add(dT, scale(I, dtheta));
            dT = add(dT, outer(du, v));
            dT = add(dT, scale(outer(u, u), -2.0 * dtheta));
            dT = add(dT, scale(outer(du, u), -2.0 * theta));
            dT = add(dT, scale(outer(u, du), -2.0 * theta));
            return dT;
        };

        dJ_dx = add(dJ_dx, scale(build_term_b(r.x / L, {P.a11 / L, P.a21 / L}), 1.0 / L));
        dJ_dy = add(dJ_dy, scale(build_term_b(r.y / L, {P.a12 / L, P.a22 / L}), 1.0 / L));

        return {dJ_dx, dJ_dy};
    }

    // ======================================================
    // Barrier gradient
    // ======================================================

    Vec2 local_barrier_grad(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
        Vec2 xi = get_xi(x, node);
        Vec2 x1 = get_xi(x, seg0);
        Vec2 x2 = get_xi(x, seg1);

        Vec2 v = {xi.x - x1.x, xi.y - x1.y};

        double t;
        Vec2 p{}, r{};
        double d = node_segment_distance(xi, x1, x2, t, p, r);
        if (d >= dhat) return {0, 0};
        d = std::max(d, 1e-12);

        Vec2 s{x2.x - x1.x, x2.y - x1.y};
        double L = std::sqrt(s.x * s.x + s.y * s.y);
        if (L < 1e-12) return {0, 0};
        Vec2 u{s.x / L, s.y / L};

        Mat2 P{1 - u.x * u.x, -u.x * u.y, -u.x * u.y, 1 - u.y * u.y};

        Vec2 n{r.x / d, r.y / d};
        double bp = barrier_grad(d, dhat);
        Vec2 f{bp * n.x, bp * n.y};
        double u_dot_v = u.x * v.x + u.y * v.y;

        if (t <= 1e-6) {
            if (who == node) return {bp * n.x, bp * n.y};
            if (who == seg0) return {-bp * n.x, -bp * n.y};
            return {0, 0};
        }
        if (t >= 1.0 - 1e-6) {
            if (who == node) return {bp * n.x, bp * n.y};
            if (who == seg1) return {-bp * n.x, -bp * n.y};
            return {0, 0};
        }

        if (who == node) return matvec(P, f);

        Mat2 vuT = {v.x * u.x, v.x * u.y, v.y * u.x, v.y * u.y};
        Mat2 uuT = {u.x * u.x, u.x * u.y, u.y * u.x, u.y * u.y};
        Mat2 term_T_T = add(add(scale(Mat2{1, 0, 0, 1}, u_dot_v), vuT), scale(uuT, -2 * u_dot_v));
        Mat2 T_common_T = scale(term_T_T, 1.0 / L);

        if (who == seg0) {
            Mat2 J1_T = add(T_common_T, scale(P, -1.0));
            return matvec(J1_T, f);
        } else if (who == seg1) {
            Mat2 J2_T = scale(T_common_T, -1.0);
            return matvec(J2_T, f);
        }

        return {0, 0};
    }

    // ======================================================
    // Barrier Hessian
    // ======================================================

    Mat2 local_barrier_hess(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
        Vec2 xi = get_xi(x, node);
        Vec2 x1 = get_xi(x, seg0);
        Vec2 x2 = get_xi(x, seg1);

        double t;
        Vec2 p{}, r_vec{};
        double d = node_segment_distance(xi, x1, x2, t, p, r_vec);
        if (d >= dhat) return {0, 0, 0, 0};
        d = std::max(d, 1e-12);

        Vec2 n{r_vec.x / d, r_vec.y / d};
        double bp  = barrier_grad(d, dhat);
        double bpp = barrier_hess(d, dhat);

        Mat2 Hrr{bpp * n.x * n.x + (bp / d) * (1 - n.x * n.x), (bpp - bp / d) * n.x * n.y,
                 (bpp - bp / d) * n.x * n.y, bpp * n.y * n.y + (bp / d) * (1 - n.y * n.y)};

        if (t <= 1e-6) {
            if (who == node || who == seg0) return Hrr;
            return {0, 0, 0, 0};
        }
        if (t >= 1.0 - 1e-6) {
            if (who == node || who == seg1) return Hrr;
            return {0, 0, 0, 0};
        }

        Vec2 s{x2.x - x1.x, x2.y - x1.y};
        double L = std::sqrt(s.x * s.x + s.y * s.y);
        if (L < 1e-12) return {0, 0, 0, 0};
        Vec2 u{s.x / L, s.y / L};

        Mat2 P{1 - u.x * u.x, -u.x * u.y, -u.x * u.y, 1 - u.y * u.y};

        Vec2 v    = get_v(x, node, seg0);
        double theta = get_theta(u, v);
        Vec2 f{bp * n.x, bp * n.y};

        Mat2 I{1, 0, 0, 1};
        Mat2 uvT{u.x * v.x, u.x * v.y, u.y * v.x, u.y * v.y};
        Mat2 uuT{u.x * u.x, u.x * u.y, u.y * u.x, u.y * u.y};
        Mat2 term     = add(add(scale(I, theta), uvT), scale(uuT, -2 * theta));
        Mat2 T_common = scale(term, 1.0 / L);

        Mat2 Jx  = P;
        Mat2 Jx1 = add(T_common, scale(P, -1.0));
        Mat2 Jx2 = scale(T_common, -1.0);

        auto hessTerm = [&](const Mat2 &M) -> Mat2 {
            return matmul(Mat2{M.a11, M.a21, M.a12, M.a22}, matmul(Hrr, M));
        };

        Mat2 Hnode = hessTerm(Jx);
        Mat2 H1    = hessTerm(Jx1);
        Mat2 H2    = hessTerm(Jx2);

        Mat2x2x2 gradJC1 = get_grad_j_common_analytic(L, u, v, theta, P, r_vec);
        Mat2x2x2 gradJP1 = get_grad_jp_analytic(u, P, L);

        Mat2 dJ1_dx = add(gradJC1.m0, gradJP1.m0);
        Mat2 dJ1_dy = add(gradJC1.m1, gradJP1.m1);

        Vec2 col1_K1 = matvec(transpose(dJ1_dx), f);
        Vec2 col2_K1 = matvec(transpose(dJ1_dy), f);
        Mat2 K1 = {col1_K1.x, col2_K1.x, col1_K1.y, col2_K1.y};

        Mat2x2x2 gradJC2 = get_grad_j_common_analytic_x2(L, u, v, theta, P, r_vec);

        Mat2 dJ2_dx = scale(gradJC2.m0, -1.0);
        Mat2 dJ2_dy = scale(gradJC2.m1, -1.0);

        Vec2 col1_K2 = matvec(transpose(dJ2_dx), f);
        Vec2 col2_K2 = matvec(transpose(dJ2_dy), f);
        Mat2 K2 = {col1_K2.x, col2_K2.x, col1_K2.y, col2_K2.y};

        if      (who == node) return Hnode;
        else if (who == seg0) return add(H1, K1);
        else if (who == seg1) return add(H2, K2);
        return {0, 0, 0, 0};
    }

    // ======================================================
    // Incremental potential (no barrier)
    // ======================================================

    Vec2 local_grad_no_barrier(int i, const Vec &x, const Vec &xhat, const Vec &xpin,
                               const std::vector<double> &mass, const std::vector<double> &L,
                               double dt, double k, const Vec2 &g_accel) {
        Vec2 xi = get_xi(x, i), xhi = get_xi(xhat, i);
        Vec2 gi{0.0, 0.0};

        gi.x += mass[i] * (xi.x - xhi.x);
        gi.y += mass[i] * (xi.y - xhi.y);

        Vec2 gs = local_spring_grad(i, x, k, L);
        gi.x += dt * dt * gs.x;
        gi.y += dt * dt * gs.y;

        gi.x -= dt * dt * mass[i] * g_accel.x;
        gi.y -= dt * dt * mass[i] * g_accel.y;

        constexpr double k_pin = 5e6;
        if (i == 0) {
            Vec2 xpi = get_xi(xpin, i);
            gi.x += dt * dt * k_pin * (xi.x - xpi.x);
            gi.y += dt * dt * k_pin * (xi.y - xpi.y);
        }

        return gi;
    }

    Mat2 local_hess_no_barrier(int i, const Vec &x,
                               const std::vector<double> &mass, const std::vector<double> &L,
                               double dt, double k) {
        Mat2 H{mass[i], 0, 0, mass[i]};

        Mat2 Hs = local_spring_hess(i, x, k, L);
        H.a11 += dt * dt * Hs.a11;
        H.a12 += dt * dt * Hs.a12;
        H.a21 += dt * dt * Hs.a21;
        H.a22 += dt * dt * Hs.a22;

        constexpr double k_pin = 5e6;
        if (i == 0) {
            H.a11 += dt * dt * k_pin;
            H.a22 += dt * dt * k_pin;
        }

        return H;
    }

} // namespace physics
