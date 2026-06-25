#include "barrier_energy.h"
#include "node_segment_distance.h"
#include <algorithm>
#include <cmath>

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

namespace {
Vec2 get_v(const Vec& x, int node, int seg0) {
    return sub(get_xi(x, node), get_xi(x, seg0));
}

double get_theta(const Vec2& u, const Vec2& v) {
    return u.x * v.x + u.y * v.y;
}

Mat2x2x2 get_grad_jp_analytic(const Vec2& u, const Mat2& P, double L) {
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

Mat2x2x2 get_grad_j_common_analytic(double L, const Vec2& u, const Vec2& v,
                                    double theta, const Mat2& P, const Vec2& r) {
    Mat2 I = {1, 0, 0, 1};
    Mat2 T = add(add(scale(I, theta), outer(u, v)), scale(outer(u, u), -2 * theta));

    Mat2 dJ_dx = scale(T, u.x / (L * L));
    Mat2 dJ_dy = scale(T, u.y / (L * L));

    auto build_term_b = [&](double dtheta, const Vec2& dv, const Vec2& du) -> Mat2 {
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

Mat2x2x2 get_grad_j_common_analytic_x2(double L, const Vec2& u, const Vec2& v,
                                       double theta, const Mat2& P, const Vec2& r) {
    Mat2 I = {1, 0, 0, 1};
    Mat2 T = add(add(scale(I, theta), outer(u, v)), scale(outer(u, u), -2 * theta));

    Mat2 dJ_dx = scale(T, -u.x / (L * L));
    Mat2 dJ_dy = scale(T, -u.y / (L * L));

    auto build_term_b = [&](double dtheta, const Vec2& du) -> Mat2 {
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
}

double node_segment_barrier_energy(const Vec& x, int node, int seg0, int seg1, double dhat) {
    double t = 0.0;
    Vec2 p{}, r{};
    const double d = node_segment_distance(get_xi(x, node), get_xi(x, seg0), get_xi(x, seg1), t, p, r);
    return barrier_energy(d, dhat);
}

Vec2 local_barrier_grad(int who, const Vec& x, int node, int seg0, int seg1, double dhat) {
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

Mat2 local_barrier_hess(int who, const Vec& x, int node, int seg0, int seg1, double dhat) {
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

    Vec2 v = get_v(x, node, seg0);
    double theta = get_theta(u, v);
    Vec2 f{bp * n.x, bp * n.y};

    Mat2 I{1, 0, 0, 1};
    Mat2 uvT{u.x * v.x, u.x * v.y, u.y * v.x, u.y * v.y};
    Mat2 uuT{u.x * u.x, u.x * u.y, u.y * u.x, u.y * u.y};
    Mat2 term = add(add(scale(I, theta), uvT), scale(uuT, -2 * theta));
    Mat2 T_common = scale(term, 1.0 / L);

    Mat2 Jx = P;
    Mat2 Jx1 = add(T_common, scale(P, -1.0));
    Mat2 Jx2 = scale(T_common, -1.0);

    auto hess_term = [&](const Mat2& M) -> Mat2 {
        return matmul(Mat2{M.a11, M.a21, M.a12, M.a22}, matmul(Hrr, M));
    };

    Mat2 Hnode = hess_term(Jx);
    Mat2 H1 = hess_term(Jx1);
    Mat2 H2 = hess_term(Jx2);

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

RigidBarrierGradient local_barrier_grad_rb(const std::vector<int>& rb_nodes, const Vec& x, const Vec2& x_com, int node, int seg0, int seg1, double dhat) {
    RigidBarrierGradient result;

    for (int who : rb_nodes) {
        if (who != node && who != seg0 && who != seg1) continue;

        const Vec2 gx = local_barrier_grad(who, x, node, seg0, seg1, dhat);
        const Vec2 r = get_xi(x, who) - x_com;
        const Vec2 dx_dtheta{-r.y, r.x};
        
        // g_y += g_x
        result.translation = result.translation + gx;
        // g_theta += (dx/dtheta)^T g_x
        result.rotation += dot(dx_dtheta, gx);
    }

    return result;
}

RigidBarrierHessian local_barrier_hess_rb(const std::vector<int>& rb_nodes, const Vec& x, const Vec2& x_com, int node, int seg0, int seg1, double dhat) {
    RigidBarrierHessian result;

    for (int who : rb_nodes) {
        if (who != node && who != seg0 && who != seg1) continue;

        const Vec2 gx = local_barrier_grad(who, x, node, seg0, seg1, dhat);
        const Mat2 Hx = local_barrier_hess(who, x, node, seg0, seg1, dhat);
        const Vec2 r = get_xi(x, who) - x_com;
        const Vec2 dx_dtheta{-r.y, r.x};
        const Vec2 d2x_dtheta2{-r.x, -r.y};
        const Vec2 Hx_dx_dtheta = matvec(Hx, dx_dtheta);

        // H_yy += H_x
        result.translation_translation = add(result.translation_translation, Hx);
        // H_ytheta  += H_x * dx/dtheta
        result.translation_rotation = result.translation_rotation + Hx_dx_dtheta;
        // H_thetatheta += (dx/dtheta)^T H_x (dx/dtheta)  + g_x^T d2x/dtheta2
        result.rotation_rotation += dot(dx_dtheta, Hx_dx_dtheta) + dot(gx, d2x_dtheta2);
    }

    return result;
}
