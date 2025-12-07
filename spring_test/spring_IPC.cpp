#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <string>


// ======================================================
// Prelim structure and functions
// ======================================================

typedef std::vector<double> Vec;
namespace fs = std::__fs::filesystem;

// --- Core data structure ---
struct Vec2 {
    double x, y;
};

struct Mat2 {
    double a11, a12, a21, a22;
};

// rank-3 tensor, stored as two 2x2 matrices: [m0, m1]
struct Mat2x2x2 {
    Mat2 m0, m1;
};

// Global Vector Utilities
Vec2 getXi(const Vec &x, int i) {
    return {x[2 * i], x[2 * i + 1]};
}

void setXi(Vec &x, int i, const Vec2 &v) {
    x[2 * i] = v.x;
    x[2 * i + 1] = v.y;
}

namespace math {
    double dist2(const Vec &a, int i, int j) {
        double dx = a[2 * i] - a[2 * j];
        double dy = a[2 * i + 1] - a[2 * j + 1];
        return dx * dx + dy * dy;
    }

    double norm(const Vec &a, int i, int j) {
        return std::sqrt(dist2(a, i, j));
    }

    // --- Vec2 Operations ---

    // Add two vectors
    static inline Vec2 add(const Vec2 &a, const Vec2 &b) {
        return {a.x + b.x, a.y + b.y};
    }

    // Subtract two vectors
    static inline Vec2 sub(const Vec2 &a, const Vec2 &b) {
        return {a.x - b.x, a.y - b.y};
    }

    // Scale vector
    static inline Vec2 scale(const Vec2 &a, double s) {
        return {s * a.x, s * a.y};
    }

    // Dot product
    static inline double dot(const Vec2 &a, const Vec2 &b) {
        return a.x * b.x + a.y * b.y;
    }

    // 2D Cross product
    static inline double cross(const Vec2 &a, const Vec2 &b) {
        return a.x * b.y - a.y * b.x;
    }

    // Squared norm
    static inline double norm2(const Vec2 &a) {
        return a.x * a.x + a.y * a.y;
    }

    // Outer product (v * v^T)
    static inline Mat2 outer(const Vec2 &a, const Vec2 &b) {
        return {a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y};
    }

    // --- Mat2 Operations ---

    // Matrix-matrix multiplication
    static inline Mat2 mul(const Mat2 &A, const Mat2 &B) {
        return {
                A.a11 * B.a11 + A.a12 * B.a21, A.a11 * B.a12 + A.a12 * B.a22,
                A.a21 * B.a11 + A.a22 * B.a21, A.a21 * B.a12 + A.a22 * B.a22
        };
    }

    // Add two matrices
    static inline Mat2 add(const Mat2 &A, const Mat2 &B) {
        return {A.a11 + B.a11, A.a12 + B.a12, A.a21 + B.a21, A.a22 + B.a22};
    }

    // Scale matrix
    static inline Mat2 scale(const Mat2 &A, double s) {
        return {s * A.a11, s * A.a12, s * A.a21, s * A.a22};
    }

// Transpose
    static inline Mat2 transpose(const Mat2 &M) {
        return {M.a11, M.a21, M.a12, M.a22};
    }

// --- Mixed Operations ---

// Matrix-vector multiplication
    static inline Vec2 mul(const Mat2 &A, const Vec2 &v) {
        return {A.a11 * v.x + A.a12 * v.y, A.a21 * v.x + A.a22 * v.y};
    }
}

namespace physics {
    using namespace math;

    // ======================================================
    // Local Gradient of the spring energy
    // ======================================================

    // Analytic local spring gradient at node i
    Vec2 localSpringGrad(int i, const Vec &x, double k, const std::vector<double> &L) {
        Vec2 g_i{0.0, 0.0};

        std::function<void(int, int, double)> contrib = [&](int a, int b, double Lref) {
            Vec2 xa = getXi(x, a), xb = getXi(x, b);
            double dx = xb.x - xa.x, dy = xb.y - xa.y;
            double ell = std::sqrt(dx * dx + dy * dy);
            if (ell < 1e-12) return;

            double coeff = k / Lref * (ell - Lref) / ell;
            double sgn = (i == b) ? +1.0 : -1.0;

            g_i.x += sgn * coeff * dx;
            g_i.y += sgn * coeff * dy;
        };

        int N = (int) L.size() + 1;
        if (i - 1 >= 0) contrib(i - 1, i, L[i - 1]);
        if (i + 1 <= N - 1) contrib(i, i + 1, L[i]);

        return g_i;
    }

    // ======================================================
    // Local Hessian of the spring energy
    // ======================================================

    // Analytic local spring Hessian (2x2 block at node i)
    Mat2 localSpringHess(int i, const Vec &x, double k, const std::vector<double> &L) {
        Mat2 H_ii{0.0, 0.0, 0.0, 0.0};

        std::function<void(int, int, double)> contrib = [&](int a, int b, double Lref) {
            Vec2 xa = getXi(x, a), xb = getXi(x, b);
            double dx = xb.x - xa.x, dy = xb.y - xa.y;
            double ell = std::sqrt(dx * dx + dy * dy);
            if (ell < 1e-12) return;

            // Derivatives from d/dx of k/L * (ell - L) * (dx/ell, dy/ell)
            // Arrange as: coeff1 * I + coeff2 * [dx;dy][dx dy]
            double coeff1 = k / Lref * (ell - Lref) / ell;
            double coeff2 = k / Lref * (Lref) / (ell * ell * ell);

            // K_j = [Kxx, Kxy; Kxy; Kyy]
            double Kxx = coeff1 + coeff2 * dx * dx;
            double Kyy = coeff1 + coeff2 * dy * dy;
            double Kxy = coeff2 * dx * dy;

            H_ii.a11 += Kxx;
            H_ii.a12 += Kxy;
            H_ii.a21 += Kxy;
            H_ii.a22 += Kyy;
        };

        int N = (int) L.size() + 1;
        if (i - 1 >= 0) contrib(i - 1, i, L[i - 1]);
        if (i + 1 <= N - 1) contrib(i, i + 1, L[i]);

        return H_ii;
    }

    // ======================================================
    // Barrier energy
    // ======================================================
    struct BarrierPair {
        int node;   // i
        int seg0;   // j
        int seg1;   // j+1
    };

    // Scalar barrier energy
    double barrierEnergy(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -(d - dhat) * (d - dhat) * std::log(d / dhat);
    }

    // Scalar barrier energy gradient
    double barrierGrad(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -2 * (d - dhat) * std::log(d / dhat) - (d - dhat) * (d - dhat) / d;
    }

    // Scalar barrier energy hessian
    double barrierHess(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -2 * std::log(d / dhat) - 4 * (d - dhat) / d + (d - dhat) * (d - dhat) / (d * d);
    }

    // ======================================================
    // Compute the point–segment distance
    // ======================================================
    double nodeSegmentDistance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1, double &t, Vec2 &p, Vec2 &r) {
        // Segment direction
        Vec2 seg = {xjp1.x - xj.x, xjp1.y - xj.y};
        double seg_len2 = seg.x * seg.x + seg.y * seg.y;

        // Handle degenerate segment
        if (seg_len2 < 1e-14) {
            t = 0.0;
            p = xj;
            r = {xi.x - p.x, xi.y - p.y};
            return std::sqrt(r.x * r.x + r.y * r.y);
        }

        // Project point onto segment
        Vec2 q = {xi.x - xj.x, xi.y - xj.y};
        double dot = q.x * seg.x + q.y * seg.y;
        t = dot / seg_len2;

        // Clamp to segment
        t = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);

        // Closest point
        p = {xj.x + t * seg.x, xj.y + t * seg.y};

        // Vector and distance
        r = {xi.x - p.x, xi.y - p.y};
        return std::sqrt(r.x * r.x + r.y * r.y);
    }

    // ======================================================
    // Local barrier gradient for a node
    // ======================================================
    Vec2 localBarrierGrad(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
        Vec2 xi = getXi(x, node);
        Vec2 x1 = getXi(x, seg0);
        Vec2 x2 = getXi(x, seg1);

        // Define v = x - x1
        Vec2 v = {xi.x - x1.x, xi.y - x1.y};

        double t;
        Vec2 p{}, r{};
        double d = nodeSegmentDistance(xi, x1, x2, t, p, r);
        if (d >= dhat) return {0, 0};
        d = std::max(d, 1e-12);

        // Segment geometry
        Vec2 s{x2.x - x1.x, x2.y - x1.y};
        double L = std::sqrt(s.x * s.x + s.y * s.y);
        if (L < 1e-12) return {0, 0};
        Vec2 u{s.x / L, s.y / L};

        // Projector P = I - uu^T
        Mat2 P{1 - u.x * u.x, -u.x * u.y,
               -u.x * u.y, 1 - u.y * u.y};

        // Barrier quantities
        Vec2 n{r.x / d, r.y / d}; // unit normal
        double bp = barrierGrad(d, dhat);    //b'(d)
        Vec2 f{bp * n.x, bp * n.y}; // f = b'(d) * n
        double u_dot_v = u.x * v.x + u.y * v.y;

        // Handle endpoint branches
        if (t <= 1e-6) {
            // Closest to x1
            if (who == node) return {bp * n.x, bp * n.y}; // n = (xi-x1)/d
            if (who == seg0) return {-bp * n.x, -bp * n.y};
            return {0, 0};
        }
        if (t >= 1.0 - 1e-6) {
            // Closest to x2
            if (who == node) return {bp * n.x, bp * n.y}; // n = (xi-x2)/d
            if (who == seg1) return {-bp * n.x, -bp * n.y};
            return {0, 0};
        }

        // Interior branch
        if (who == node) {
            return mul(P, f);
        }

        // Build the term [ (u^T v)I + v u^T - 2 (u^T v) u u^T ] / L
        // This is T_common_T = (T/L)^T = J1^T + P
        Mat2 vuT = {v.x * u.x, v.x * u.y, v.y * u.x, v.y * u.y}; // outer(v, u) = v u^T
        Mat2 uuT = {u.x * u.x, u.x * u.y, u.y * u.x, u.y * u.y}; // outer(u, u)
        Mat2 term_T_T = add(add(scale(Mat2{1, 0, 0, 1}, u_dot_v), vuT), scale(uuT, -2 * u_dot_v)); // This is T^T
        Mat2 T_common_T = scale(term_T_T, 1.0 / L); // This is (T/L)^T

        if (who == seg0) {
            // g1 = J1^T f = ( (T/L)^T - P^T ) f
            Mat2 J1_T = add(T_common_T, scale(P, -1.0)); // P is symmetric
            return mul(J1_T, f); // Returns J1^T * f
        } else if (who == seg1) {
            // g2 = J2^T f = ( -(T/L) )^T f = - (T/L)^T f
            Mat2 J2_T = scale(T_common_T, -1.0);
            return mul(J2_T, f); // Returns J2^T * f
        }

        return {0, 0};
    }

    // ======================================================
    // Analytic derivatives for Tensor Term
    // ======================================================

    // Gets v = r_p = x - x1
    Vec2 getV(const Vec &x, int node, int seg0) {
        Vec2 xi = getXi(x, node);
        Vec2 x1 = getXi(x, seg0);
        return sub(xi, x1);
    }

    // Gets theta = u^T v
    double getTheta(const Vec2 &u, const Vec2 &v) {
        return u.x * v.x + u.y * v.y;
    }

    // The gradient of projection matrix P w.r.t. x1 (but this one return -grad(P))
    Mat2x2x2 getGradJP_analytic(const Vec2 &u, const Mat2 &P, double L) {
        Mat2x2x2 gradP{};
        // dP_ik / dx1_l = (1/L) * (P_il u_k + u_i P_kl)

        // dP/dx1_x (l=0)
        double P_il_x = P.a11, P_il_y = P.a21; // P_i0 (col 0 of P)
        gradP.m0.a11 = (1.0 / L) * (P_il_x * u.x + u.x * P_il_x);
        gradP.m0.a12 = (1.0 / L) * (P_il_x * u.y + u.x * P.a12);
        gradP.m0.a21 = (1.0 / L) * (P_il_y * u.x + u.y * P_il_x);
        gradP.m0.a22 = (1.0 / L) * (P_il_y * u.y + u.y * P.a12);

        // dP/dx1_y (l=1)
        double P_il_x_y = P.a12, P_il_y_y = P.a22; // P_i1 (col 1 of P)
        gradP.m1.a11 = (1.0 / L) * (P_il_x_y * u.x + u.x * P_il_x_y);
        gradP.m1.a12 = (1.0 / L) * (P_il_x_y * u.y + u.x * P_il_y_y);
        gradP.m1.a21 = (1.0 / L) * (P_il_y_y * u.x + u.y * P_il_x_y);
        gradP.m1.a22 = (1.0 / L) * (P_il_y_y * u.y + u.y * P_il_y_y);

        // return -grad(P)
        return {scale(gradP.m0, -1.0), scale(gradP.m1, -1.0)};
    }

    // grad(J_common) w.r.t. x1
    // This implements Term A + Term B from the note
    Mat2x2x2 getGradJCommon_analytic(double L, const Vec2 &u, const Vec2 &v, double theta, const Mat2 &P, const Vec2 &r) {
        Mat2 I = {1, 0, 0, 1};
        Mat2 T = add(add(scale(I, theta), outer(u, v)), scale(outer(u, u), -2 * theta)); // T = J_common * L

        // Term A: d(1/L)/dx1_l * T
        // d(1/L)/dx1_l = u_l / L^2
        Mat2 dJ_dx = scale(T, u.x / (L * L)); // (l=0)
        Mat2 dJ_dy = scale(T, u.y / (L * L)); // (l=1)

        // Term B: (1/L) * dT/dx1_l
        // dT/dx1_l = (d(theta)/dx1_l) * I + (d(u)/dx1_l)*v^T + u*(d(v)/dx1_l)^T
        //            - 2*(d(theta)/dx1_l)*uuT - 2*theta*(d(u)/dx1_l)*u^T - 2*theta*u*(d(u)/dx1_l)^T

        // Derivatives we need (w.r.t. x1_l)
        // dL/dx1_l = -u_l
        // d(1/L)/dx1_l = u_l / L^2
        // d(v)/dx1_l = -I_l (col l of -I)
        // d(u)/dx1_l = -P_l / L (col l of -P/L)
        // d(theta)/dx1_l = -r_l/L - u_l

        // Build Term B for l=0 (x-component)
        double dtheta_l_x = -r.x / L - u.x;  // d(theta)/dx1_x
        Vec2 dv_l_x = {-1, 0};  // d(v)/dx1_x
        Vec2 du_l_x = {-P.a11 / L, -P.a21 / L}; // d(u)/dx1_x (col 0 of -P/L)

        Mat2 dT_l = {0, 0, 0, 0};
        dT_l = add(dT_l, scale(I, dtheta_l_x));
        dT_l = add(dT_l, outer(du_l_x, v));
        dT_l = add(dT_l, outer(u, dv_l_x));
        dT_l = add(dT_l, scale(outer(u, u), -2.0 * dtheta_l_x));
        dT_l = add(dT_l, scale(outer(du_l_x, u), -2.0 * theta));
        dT_l = add(dT_l, scale(outer(u, du_l_x), -2.0 * theta));

        dJ_dx = add(dJ_dx, scale(dT_l, 1.0 / L)); // Add Term B (l=0)

        // Build Term B for l=1 (y-component)
        double dtheta_l_y = -r.y / L - u.y; // d(theta)/dx1_y
        Vec2 dv_l_y = {0, -1};  // d(v)/dx1_y
        Vec2 du_l_y = {-P.a12 / L, -P.a22 / L}; // d(u)/dx1_y (col 1 of -P/L)

        dT_l = {0, 0, 0, 0}; // Reset
        dT_l = add(dT_l, scale(I, dtheta_l_y));
        dT_l = add(dT_l, outer(du_l_y, v));
        dT_l = add(dT_l, outer(u, dv_l_y));
        dT_l = add(dT_l, scale(outer(u, u), -2.0 * dtheta_l_y));
        dT_l = add(dT_l, scale(outer(du_l_y, u), -2.0 * theta));
        dT_l = add(dT_l, scale(outer(u, du_l_y), -2.0 * theta));

        dJ_dy = add(dJ_dy, scale(dT_l, 1.0 / L)); // Add Term B (l=1)

        return {dJ_dx, dJ_dy};
    }

    // grad(J_common) w.r.t. x2
    Mat2x2x2 getGradJCommon_analytic_x2(double L, const Vec2 &u, const Vec2 &v, double theta, const Mat2 &P, const Vec2 &r) {
        Mat2 I = {1, 0, 0, 1};
        Mat2 T = add(add(scale(I, theta), outer(u, v)), scale(outer(u, u), -2 * theta)); // T = J_common * L

        // Term A: d(1/L)/dx2_l * T
        // d(1/L)/dx2_l = -u_l / L^2
        Mat2 dJ_dx = scale(T, -u.x / (L * L)); // (l=0)
        Mat2 dJ_dy = scale(T, -u.y / (L * L)); // (l=1)

        // Term B: (1/L) * dT/dx2_l
        // Derivatives we need (w.r.t. x2_l)
        // d(v)/dx2_l = 0
        // d(u)/dx2_l = P_l / L (col l of P/L)
        // d(theta)/dx2_l = (d(u)/dx2_l)^T * v = (P_l/L)^T * v = (1/L) * (P v)_l = r_l / L

        // Build Term B for l=0 (x-component)
        double dtheta_l_x = r.x / L; // d(theta)/dx2_x
        Vec2 du_l_x = {P.a11 / L, P.a21 / L}; // d(u)/dx2_x (col 0 of P/L)

        Mat2 dT_l = {0, 0, 0, 0};
        dT_l = add(dT_l, scale(I, dtheta_l_x));
        dT_l = add(dT_l, outer(du_l_x, v));
        dT_l = add(dT_l, scale(outer(u, u), -2.0 * dtheta_l_x));
        dT_l = add(dT_l, scale(outer(du_l_x, u), -2.0 * theta));
        dT_l = add(dT_l, scale(outer(u, du_l_x), -2.0 * theta));

        dJ_dx = add(dJ_dx, scale(dT_l, 1.0 / L)); // Add Term B (l=0)

        // Build Term B for l=1 (y-component)
        double dtheta_l_y = r.y / L; // d(theta)/dx2_y
        Vec2 du_l_y = {P.a12 / L, P.a22 / L}; // d(u)/dx2_y (col 1 of P/L)

        dT_l = {0, 0, 0, 0}; // Reset
        dT_l = add(dT_l, scale(I, dtheta_l_y));
        dT_l = add(dT_l, outer(du_l_y, v));
        dT_l = add(dT_l, scale(outer(u, u), -2.0 * dtheta_l_y));
        dT_l = add(dT_l, scale(outer(du_l_y, u), -2.0 * theta));
        dT_l = add(dT_l, scale(outer(u, du_l_y), -2.0 * theta));

        dJ_dy = add(dJ_dy, scale(dT_l, 1.0 / L)); // Add Term B

        return {dJ_dx, dJ_dy};
    }

    // ==============================
    // Local barrier hessian for a node
    // ==============================
    Mat2 localBarrierHess(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
        Vec2 xi = getXi(x, node);
        Vec2 x1 = getXi(x, seg0);
        Vec2 x2 = getXi(x, seg1);

        double t;
        Vec2 p{}, r_vec{}; // r_vec is the residual r = P(xi-x1)
        double d = nodeSegmentDistance(xi, x1, x2, t, p, r_vec);
        if (d >= dhat) return {0, 0, 0, 0};
        d = std::max(d, 1e-12);

        // Barrier quantities
        Vec2 n{r_vec.x / d, r_vec.y / d};
        double bp = barrierGrad(d, dhat);
        double bpp = barrierHess(d, dhat);

        // Inner Hessian w.r.t residual
        Mat2 Hrr{bpp * n.x * n.x + (bp / d) * (1 - n.x * n.x), (bpp - bp / d) * n.x * n.y,
                 (bpp - bp / d) * n.x * n.y, bpp * n.y * n.y + (bp / d) * (1 - n.y * n.y)
        };

        // --- Endpoint branch ---
        if (t <= 1e-6) {
            // Closest to x1
            if (who == node or who == seg0) return Hrr;
            else return {0, 0, 0, 0};
        }

        if (t >= 1.0 - 1e-6) {
            // Closest to x2
            if (who == node or who == seg1) return Hrr;
            else return {0, 0, 0, 0};
        }

        // --- Interior branch ---
        // Geometry
        Vec2 s{x2.x - x1.x, x2.y - x1.y};
        double L = std::sqrt(s.x * s.x + s.y * s.y);
        if (L < 1e-12) return {0, 0, 0, 0};
        Vec2 u{s.x / L, s.y / L};

        // Projection matrix
        Mat2 P{1 - u.x * u.x, -u.x * u.y,
               -u.x * u.y, 1 - u.y * u.y};

        // Shared quantities
        Vec2 v = getV(x, node, seg0);      // v = r_p = xi - x1
        double theta = getTheta(u, v);     // theta = u^T v
        Vec2 f{bp * n.x, bp * n.y};

        // Base term T_common = [(u^T v)I + u v^T - 2(u^T v)u u^T]/L
        Mat2 I{1, 0, 0, 1};
        Mat2 uvT{u.x * v.x, u.x * v.y, u.y * v.x, u.y * v.y}; // u v^T
        Mat2 uuT{u.x * u.x, u.x * u.y, u.y * u.x, u.y * u.y};
        Mat2 term = add(add(scale(I, theta), uvT), scale(uuT, -2 * theta));
        Mat2 T_common = scale(term, 1.0 / L);

        // Jacobians
        Mat2 Jx = P;
        Mat2 Jx1 = add(T_common, scale(P, -1.0));
        Mat2 Jx2 = scale(T_common, -1.0);

        // Helper: M^T * Hrr * M
        std::function<Mat2(const Mat2 &)> hessTerm = [&](const Mat2 &M) {
            return mul(Mat2{M.a11, M.a21, M.a12, M.a22}, mul(Hrr, M)); // M^T * Hrr * M
        };

        // Base Hessians (J^T Hrr J)
        Mat2 Hnode = hessTerm(Jx);
        Mat2 H1 = hessTerm(Jx1);
        Mat2 H2 = hessTerm(Jx2);

        // --- Tensor term calculation ---
        Mat2 K1{};
        Mat2 K2{};

        // K1 Term
        // K1 = [d(J1^T)/dx1] * f = [d(J_common^T)/dx1 + d(J_P^T)/dx1] * f
        // (K1)_il = sum_k [ d(J_common)_ki/dx1_l + d(J_P)_ki/dx1_l ] * f_k
        // Get analytic 3-tensors T_ikl = d(J_ik)/dx1_l
        Mat2x2x2 gradJC1 = getGradJCommon_analytic(L, u, v, theta, P, r_vec);
        Mat2x2x2 gradJP1 = getGradJP_analytic(u, P, L);

        // Get tensor slices for d(J1)/dx1_l = d(J_common)/dx1_l + d(J_P)/dx1_l
        Mat2 dJ1_dx = add(gradJC1.m0, gradJP1.m0);
        Mat2 dJ1_dy = add(gradJC1.m1, gradJP1.m1);

        // Contract: (K1)_il = sum_k d(J1)_ki/dx1_l * f_k
        // Col 1 (l=0): (K1)_i0 = sum_k d(J1)_ki/dx1_x * f_k = (dJ1_dx^T) * f
        Vec2 col1_K1 = mul(transpose(dJ1_dx), f);
        // Col 2 (l=1): (K1)_i1 = sum_k d(J1)_ki/dx1_y * f_k = (dJ1_dy^T) * f
        Vec2 col2_K1 = mul(transpose(dJ1_dy), f);

        K1 = {col1_K1.x, col2_K1.x, col1_K1.y, col2_K1.y};


        // K2 Term
        // K2 = [d(J2^T)/dx2] * f = [d(-J_common^T)/dx2] * f
        // (K2)_il = sum_k [ -d(J_common)_ki/dx2_l ] * f_k
        // Get analytic 3-tensor T_ikl = d(J_common_ik)/dx2_l
        Mat2x2x2 gradJC2 = getGradJCommon_analytic_x2(L, u, v, theta, P, r_vec);

        // Get tensor slices for d(J2)/dx2_l = -d(J_common)/dx2_l
        Mat2 dJ2_dx = scale(gradJC2.m0, -1.0);
        Mat2 dJ2_dy = scale(gradJC2.m1, -1.0);

        // Contract: (K2)_il = sum_k d(J2)_ki/dx2_l * f_k
        // Col 1 (l=0): (K2)_i0 = sum_k d(J2)_ki/dx2_x * f_k = (dJ2_dx^T) * f
        Vec2 col1_K2 = mul(transpose(dJ2_dx), f);
        // Col 2 (l=1): (K2)_i1 = sum_k d(J2)_ki/dx2_y * f_k = (dJ2_dy^T) * f
        Vec2 col2_K2 = mul(transpose(dJ2_dy), f);

        K2 = {col1_K2.x, col2_K2.x, col1_K2.y, col2_K2.y};

        // Return appropriate block
        if (who == node) return Hnode;
        else if (who == seg0) return add(H1, K1);
        else if (who == seg1) return add(H2, K2);
        else return {0, 0, 0, 0};
    }

    // ==============================
    // Local gradient of the function Psi
    // ==============================
    Vec2 PsiLocalGrad(int i, const Vec &x, const Vec &xhat, const std::vector<double> &mass,
                      const std::vector<double> &L, double dt, double k, const Vec2 &g_accel,
                      const std::vector<BarrierPair> &barriers, double dhat) {

        Vec2 xi = getXi(x, i), xhi = getXi(xhat, i);
        Vec2 gi{0.0, 0.0};

        // Mass term
        gi.x += mass[i] * (xi.x - xhi.x);
        gi.y += mass[i] * (xi.y - xhi.y);

        // Spring term
        Vec2 gs = localSpringGrad(i, x, k, L);
        gi.x += dt * dt * gs.x;
        gi.y += dt * dt * gs.y;

        // Gravity
        gi.x -= dt * dt * mass[i] * g_accel.x;
        gi.y -= dt * dt * mass[i] * g_accel.y;

        // Barrier forces
        for (const BarrierPair &c: barriers) {
            for (int who: {c.node, c.seg0, c.seg1}) {
                if (who != i) continue;
                Vec2 gb = localBarrierGrad(i, x, c.node, c.seg0, c.seg1, dhat);
                gi.x += dt * dt * gb.x;
                gi.y += dt * dt * gb.y;
            }
        }

        constexpr double k_pin = 5000000.0;

        if (i == 0) {
            gi.x += dt * dt * k_pin * (xi.x - xhi.x);
            gi.y += dt * dt * k_pin * (xi.y - xhi.y);
        }

        return gi;
    }

    // ==============================
    // Local hessian of the function Psi
    // ==============================
    Mat2 PsiLocalHess(int i, const Vec &x, const std::vector<double> &mass, const std::vector<double> &L, double dt,
                      double k, const std::vector<BarrierPair> &barriers,
                      double dhat) {

        Mat2 H{mass[i], 0, 0, mass[i]};

        // Spring term
        Mat2 Hs = localSpringHess(i, x, k, L);
        H.a11 += dt * dt * Hs.a11;
        H.a12 += dt * dt * Hs.a12;
        H.a21 += dt * dt * Hs.a21;
        H.a22 += dt * dt * Hs.a22;

        // Barrier term (local node-block only)
        for (const BarrierPair &c: barriers) {
            for (int who: {c.node, c.seg0, c.seg1}) {
                if (who != i) continue;
                Mat2 Hb = localBarrierHess(who, x, c.node, c.seg0, c.seg1, dhat);
                H.a11 += dt * dt * Hb.a11;
                H.a12 += dt * dt * Hb.a12;
                H.a21 += dt * dt * Hb.a21;
                H.a22 += dt * dt * Hb.a22;
            }
        }

        constexpr double k_pin = 5000000.0;

        if (i == 0) {
            H.a11 += dt * dt * k_pin;
            H.a22 += dt * dt * k_pin;
        }

        return H;
    }
}


namespace ccd {
    using namespace math;

    // ============================================================
    // Broad-phase AABB
    // ============================================================
    struct AABB2 {
        Vec2 min, max;
    };

    enum ObjectType {
        NODE, SEGMENT
    };

    struct Object {
        AABB2 box;        // Swept bounding box
        int id;           // Index (node or segment start)
        ObjectType type;  // Type
    };

    // y-overlap
    inline bool overlap_y(const AABB2 &A, const AABB2 &B) {
        return !(A.max.y < B.min.y || A.min.y > B.max.y);
    }

    // Broad-phase AABB
    template<typename Callback>
    void broad_phase_ccd(std::vector<Object> &objects, Callback report) {
        const int n = static_cast<int>(objects.size());

        // Sort by x_min
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return objects[a].box.min.x < objects[b].box.min.x;
        });

        // Make a list A
        std::vector<int> A;

        // For each Bi in sorted order
        for (int idx_i: order) {
            const AABB2 &Bi = objects[idx_i].box;

            // Remove boxes with whose xmax < Bi.xmin
            A.erase(std::remove_if(A.begin(), A.end(), [&](int idx_j) {
                const AABB2 &Bj = objects[idx_j].box;
                return Bj.max.x < Bi.min.x;
            }), A.end());

            // Test against remaining boxes in the list
            for (int idx_j: A) {
                const AABB2 &Bj = objects[idx_j].box;

                // Skip same-type pairs
                if (objects[idx_i].type == objects[idx_j].type)
                    continue;

                // Check overlaps in y-direction
                if (overlap_y(Bi, Bj)) {
                    report(objects[idx_i], objects[idx_j]);
                }
            }

            // Add Bi to the list
            A.push_back(idx_i);
        }
    }

    // =======================
    // Narrow-phase 2D point–segment CCD
    // =======================
    bool ccd_point_segment_2d(const Vec2 &x1, const Vec2 &dx1, const Vec2 &x2, const Vec2 &dx2,
                              const Vec2 &x3, const Vec2 &dx3, double &t_out, double eps = 1e-12) {

        // Compute coefficients of f(t) = a t^2 + b t + c
        Vec2 x21 = sub(x1, x2);
        Vec2 x32 = sub(x3, x2);
        Vec2 dx21 = sub(dx1, dx2);
        Vec2 dx32 = sub(dx3, dx2);

        double a = cross(dx32, dx21);
        double b = cross(dx32, x21) + cross(x32, dx21);
        double c = cross(x32, x21);

        double t_candidates[2];
        int num_roots = 0;

        // Degenerate case if a = 0
        if (std::fabs(a) < eps) {
            if (std::fabs(b) < eps) return false;
            double t = -c / b;
            if (t >= 0.0 and t <= 1.0)
                t_candidates[num_roots++] = t;
        } else {
            double D = b * b - 4.0 * a * c;
            if (D < 0.0) return false; // No real roots

            double sqrtD = std::sqrt(std::max(D, 0.0));
            double s = (b >= 0.0) ? 1.0 : -1.0;
            double q = -0.5 * (b + s * sqrtD);

            double t1 = q / a;
            double t2 = c / q;

            if (t1 >= 0.0 and t1 <= 1.0)
                t_candidates[num_roots++] = t1;
            if (t2 >= 0.0 and t2 <= 1.0)
                t_candidates[num_roots++] = t2;
        }

        if (num_roots == 0) return false;

        // Choose earliest valid collision time
        double t_star = t_candidates[0];
        if (num_roots == 2 and t_candidates[1] < t_star)
            t_star = t_candidates[1];

        // Inside-segment test
        Vec2 x1t = add(x1, scale(dx1, t_star));
        Vec2 x2t = add(x2, scale(dx2, t_star));
        Vec2 x3t = add(x3, scale(dx3, t_star));

        Vec2 seg = sub(x3t, x2t);
        Vec2 rel = sub(x1t, x2t);

        double seg_len2 = norm2(seg);
        if (seg_len2 < eps) return false; // Degenerate segment

        double s = dot(rel, seg) / seg_len2;
        if (s < 0.0 or s > 1.0) return false;

        // Valid collision
        t_out = t_star;
        return true;
    }

    double ccd_get_safe_step(const Vec2 &x1, const Vec2 &dx1, const Vec2 &x2, const Vec2 &dx2,
                             const Vec2 &x3, const Vec2 &dx3, double eta = 0.9) {

        double t_hit; // Variable to store the time of impact

        // Run the 2D CCD to find the exact time of impact.
        bool collision_found = ccd_point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t_hit);

        if (collision_found) {
            if (t_hit <= 1e-12) {
                // Already in collision, don't move at all
                return 0.0;
            }
            return eta * t_hit;
        } else {
            // No collision found in the [0, 1] interval
            return 1.0;
        }
    }
}

namespace solver {
    using namespace math;
    using namespace physics;
    using namespace ccd;

    // =====================================================
    //  Nonlinear Gauss–Seidel using local grad/hess
    // =====================================================

    // Inverse of any 2x2 matrix
    Mat2 matrix2d_inverse(const Mat2 &H) {
        double det = H.a11 * H.a22 - H.a12 * H.a21;
        if (std::abs(det) < 1e-12) {
            throw std::runtime_error("Singular matrix in inverse()");
        }
        double inv_det = 1.0 / det;
        Mat2 Hi{};
        Hi.a11 = H.a22 * inv_det;
        Hi.a12 = -H.a12 * inv_det;
        Hi.a21 = -H.a21 * inv_det;
        Hi.a22 = H.a11 * inv_det;
        return Hi;
    }

    // Compute local gradient + barrier
    Vec2 compute_local_gradient(int i, const Vec &x_local, const Vec &xhat_local, const std::vector<double> &mass_local,
                                const std::vector<double> &L_local, double dt, double k, const Vec2 &g_accel,
                                const std::vector<BarrierPair> &barriers_global,
                                double dhat, const Vec &x_global, int global_offset) {
        Vec2 gi = PsiLocalGrad(i, x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                               {}, dhat);

        // Add barrier contributions
        int who_global = global_offset + i;
        Vec2 gbar{0.0, 0.0};
        for (const BarrierPair &c: barriers_global) {
            if (c.node != who_global and c.seg0 != who_global and c.seg1 != who_global)
                continue;
            Vec2 gb = localBarrierGrad(who_global, x_global, c.node, c.seg0, c.seg1, dhat);
            gbar.x += gb.x;
            gbar.y += gb.y;
        }

        gi.x += dt * dt * gbar.x;
        gi.y += dt * dt * gbar.y;
        return gi;
    }

    // Compute local Hessian + barrier
    Mat2 compute_local_hessian(int i, const Vec &x_local, const std::vector<double> &mass_local,
                               const std::vector<double> &L_local, double dt, double k,
                               const std::vector<BarrierPair> &barriers_global,
                               double dhat, const Vec &x_global, int global_offset) {
        Mat2 Hi = PsiLocalHess(i, x_local, mass_local, L_local, dt, k, {}, dhat);
        int who_global = global_offset + i;

        for (const BarrierPair &c: barriers_global) {
            if (c.node != who_global and c.seg0 != who_global and c.seg1 != who_global)
                continue;
            Mat2 Hb = localBarrierHess(who_global, x_global, c.node, c.seg0, c.seg1, dhat);
            Hi.a11 += dt * dt * Hb.a11;
            Hi.a12 += dt * dt * Hb.a12;
            Hi.a21 += dt * dt * Hb.a21;
            Hi.a22 += dt * dt * Hb.a22;
        }

        return Hi;
    }

    double compute_safe_step(int who_global, const Vec2 &dx, const Vec &x_global,
                             const std::vector<BarrierPair> &barriers_global, double eta) {
        double omega_hat = 1.0;

        for (const BarrierPair &c: barriers_global) {

            Vec2 xi = getXi(x_global, c.node);
            Vec2 xj = getXi(x_global, c.seg0);
            Vec2 xk = getXi(x_global, c.seg1);

            // Zero increments for nodes
            Vec2 dxi = {0, 0}, dxj = {0, 0}, dxk = {0, 0};

            // Descent direction is dx, so full step is -dx
            Vec2 full_step = {-dx.x, -dx.y};

            if (who_global == c.node) {
                // Point moves, segment stays
                dxi = full_step;

            } else if (who_global == c.seg0) {
                // Segment endpoint j moves, point+k stays
                dxj = full_step;

            } else if (who_global == c.seg1) {
                // Segment endpoint k moves, point+j stays
                dxk = full_step;

            } else {
                continue;
            }

            double omega_c = ccd_get_safe_step(xi, dxi, xj, dxj, xk, dxk, eta);
            omega_hat = std::min(omega_hat, omega_c);
        }

        return omega_hat;
    }

    double compute_residual(const Vec &x_local, const Vec &xhat_local, const std::vector<double> &mass_local,
                            const std::vector<double> &L_local, double dt, double k, const Vec2 &g_accel,
                            const std::vector<BarrierPair> &barriers_global,
                            double dhat, const Vec &x_global, int global_offset) {
        const int N = (int) mass_local.size();
        double sum = 0.0;

        for (int i = 0; i < N; ++i) {
            Vec2 g = compute_local_gradient(i, x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                                            barriers_global, dhat, x_global, global_offset);
            sum += g.x * g.x + g.y * g.y;
        }

        return std::sqrt(sum);
    }

    std::pair<double, int>
    gauss_seidel_solver(Vec &x_local, const Vec &xhat_local, const std::vector<double> &mass_local,
                        const std::vector<double> &L_local, double dt, double k, const Vec2 &g_accel,
                        const std::vector<BarrierPair> &barriers_global,
                        double dhat, Vec &x_global, int global_offset, int max_iterations,
                        double tol_abs, double eta) {
        const int N_local = (int) mass_local.size();

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            for (int i = 0; i < N_local; ++i) {

                Vec2 gi = compute_local_gradient(i, x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                                                 barriers_global, dhat, x_global, global_offset);

                Mat2 Hi = compute_local_hessian(i, x_local, mass_local, L_local, dt, k,
                                                barriers_global,
                                                dhat, x_global, global_offset);

                Vec2 dx = mul(matrix2d_inverse(Hi), gi);

                int who_global = global_offset + i;
                double omega = compute_safe_step(who_global, dx, x_global, barriers_global, eta);

                Vec2 xi = getXi(x_local, i);
                xi.x -= omega * dx.x;
                xi.y -= omega * dx.y;
                setXi(x_local, i, xi);
                setXi(x_global, who_global, xi);
            }

            double rn = compute_residual(x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                                         barriers_global, dhat, x_global, global_offset);
            if (rn < tol_abs)
                return {rn, iteration + 1};
        }

        double rn = compute_residual(x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                                     barriers_global, dhat, x_global, global_offset);
        return {rn, max_iterations};
    }
}


namespace io {
    using namespace math;

    // =====================================================
    //  Export a Obj file
    // =====================================================
    void export_obj(const std::string &filename, const Vec &x, const std::vector<std::pair<int, int>> &edges) {
        std::ofstream out(filename);

        if (!out) {
            std::cerr << "Error: cannot write " << filename << "\n";
            return;
        }
        int N = (int) (x.size() / 2);

        for (int i = 0; i < N; ++i) {
            Vec2 xi = getXi(x, i);
            out << "v " << xi.x << " " << xi.y << " 0.0\n";
        }

        for (const std::pair<int, int> &e: edges)
            out << "l " << (e.first + 1) << " " << (e.second + 1) << "\n";
        out.close();
    }

    // Export function
    void export_frame(const std::string &outdir, int frame, const Vec &x_combined,
                      const std::vector<std::pair<int, int>> &edges_combined) {
        std::ostringstream ss;
        ss << outdir << "/frame_" << std::setw(4)
           << std::setfill('0') << frame << ".obj";
        export_obj(ss.str(), x_combined, edges_combined);
    }
}

namespace simulation {
    using namespace math;
    using namespace solver;
    using namespace io;

    // ==============================
    // Numerical Experiment
    // ==============================
    struct Chain {
        int N{}; // number of nodes
        Vec x; // positions (2*N)
        Vec v; // velocities (2*N)
        Vec xhat;  // predicted positions (2*N)
        std::vector<double> mass; // per-node masses
        std::vector<double> rest_lengths; // rest spring lengths
        std::vector<std::pair<int, int>> edges; // connectivity list
    };

    Chain make_chain(Vec2 start, Vec2 end, int N, double mass_value) {
        Chain c;
        c.N = N;
        c.x.resize(2 * N);
        c.v.assign(2 * N, 0.0);
        c.xhat.assign(2 * N, 0.0);
        c.mass.assign(N, mass_value);

        // Node positions
        for (int i = 0; i < N; ++i) {
            double t = (N == 1) ? 0.0 : double(i) / (N - 1);
            Vec2 xi{start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)};
            setXi(c.x, i, xi);
        }

        // Edges and rest lengths
        for (int i = 0; i < N - 1; ++i) {
            c.edges.emplace_back(i, i + 1);
            c.rest_lengths.push_back(norm(c.x, i, i + 1));
        }

        return c;
    }

    // Combine node positions from both chains
    void combine_positions(Vec &x_combined, const Vec &x_left, const Vec &x_right, int N_left, int N_right) {
        for (int i = 0; i < N_left; ++i)
            setXi(x_combined, i, getXi(x_left, i));
        for (int i = 0; i < N_right; ++i)
            setXi(x_combined, N_left + i, getXi(x_right, i));
    }

    // Node predictor and velocity update
    void predictor_step(Chain &c, double dt) {
        for (int i = 0; i < c.N; ++i) {
            Vec2 xi = getXi(c.x, i);
            Vec2 vi = getXi(c.v, i);
            setXi(c.xhat, i, {xi.x + dt * vi.x, xi.y + dt * vi.y});
        }
    }

    void update_velocity(Chain &c, const Vec &xnew, double dt) {
        for (int i = 0; i < c.N; ++i) {
            Vec2 xi_new = getXi(xnew, i);
            Vec2 xi_old = getXi(c.x, i);
            setXi(c.v, i, {(xi_new.x - xi_old.x) / dt, (xi_new.y - xi_old.y) / dt});
        }
        c.x = xnew;
    }

    // Build barrier pairs using the broad-phase AABB
    std::vector<BarrierPair> build_barrier_pairs(const Vec &x_combined, const Vec &v_combined, int N_left, int N_right, double dt) {
        std::vector<BarrierPair> barriers;
        const int total_nodes = N_left + N_right;
        std::vector<Object> objects;
        objects.reserve(total_nodes * 2);

        // Node AABBs
        double r = 0.05; // small radius per node

        for (int i = 0; i < total_nodes; ++i) {
            Vec2 x0 = getXi(x_combined, i);
            Vec2 v = getXi(v_combined, i);
            Vec2 x1 = {x0.x + dt * v.x, x0.y + dt * v.y};

            double min_x = std::min({x0.x - r, x1.x - r});
            double max_x = std::max({x0.x + r, x1.x + r});
            double min_y = std::min({x0.y - r, x1.y - r});
            double max_y = std::max({x0.y + r, x1.y + r});

            AABB2 node_box{{min_x, min_y},
                           {max_x, max_y}};
            objects.push_back({node_box, i, NODE});
        }

        // Segment AABBs
        for (int j = 0; j < total_nodes - 1; ++j) {
            bool is_left = (j < N_left - 1);
            bool is_right = (j >= N_left && j < N_left + N_right - 1);
            if (!is_left && !is_right) continue;

            Vec2 x0 = getXi(x_combined, j);
            Vec2 x1 = getXi(x_combined, j + 1);
            Vec2 v0 = getXi(v_combined, j);
            Vec2 v1 = getXi(v_combined, j + 1);

            double min_x = std::min({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x});
            double max_x = std::max({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x});
            double min_y = std::min({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y});
            double max_y = std::max({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y});

            AABB2 seg_box{{min_x, min_y},
                          {max_x, max_y}};
            objects.push_back({seg_box, j, SEGMENT});
        }

        // Overlap test
        broad_phase_ccd(objects, [&](const Object &A, const Object &B) {
            const Object *nodeObj = (A.type == NODE) ? &A : (B.type == NODE) ? &B : nullptr;
            const Object *segObj = (A.type == SEGMENT) ? &A : (B.type == SEGMENT) ? &B : nullptr;
            if (!nodeObj || !segObj) return;

            int node = nodeObj->id;
            int seg0 = segObj->id, seg1 = seg0 + 1;
            if (seg1 >= total_nodes) return;

            bool invalid = (node == seg0) || (node == seg1) || (seg0 < N_left - 1 && node < N_left) ||
                           (seg0 >= N_left && node >= N_left);
            if (!invalid)
                barriers.push_back({node, seg0, seg1});
        });

        return barriers;
    }

    // Main Simulation
    int main() {
        using clock = std::chrono::high_resolution_clock;
        std::chrono::time_point<clock> t_start = clock::now();   // start timer

        std::string outdir = "frames_spring_IPC";
        fs::create_directory(outdir);

        // Parameters
        double dt = 1.0 / 30.0;
        Vec2 g_accel = {0.0, -9.81};
        double k_spring = 20.0;
        int total_frame = 600;
        int max_iterations = 300;
        double tol_abs = 1e-8;
        double dhat = 0.1;
        double eta = 0.9;
        int number_of_segments = 11;

        // Create chains
        Chain left = make_chain({-1.0, 0.0}, {4.0, -5.0}, number_of_segments,0.05); // y = -x - 1
        Chain right = make_chain({-1.5, 0.5}, {3.5, 0.5}, number_of_segments,0.05); // y = 0.5

        // Combined geometry for OBJ export
        const int total_nodes = left.N + right.N;

        std::vector<std::pair<int, int>> edges_combined = left.edges;
        for (std::pair<int, int> &e: right.edges)
            edges_combined.emplace_back(e.first + left.N, e.second + left.N);

        // Initialize combined data
        Vec x_combined(2 * total_nodes, 0.0);
        Vec v_combined(2 * total_nodes, 0.0);

        combine_positions(x_combined, left.x, right.x, left.N, right.N);
        export_frame(outdir, 0, x_combined, edges_combined);

        // Main simulation loop
        for (int frame = 1; frame <= total_frame; ++frame) {

            Vec xnew_left, xnew_right;
            
            // Predictor step: xhat = x + dt * v
            predictor_step(left, dt);
            predictor_step(right, dt);

            // Combine positions for barrier queries
            combine_positions(x_combined, left.x, right.x, left.N, right.N);

            // Combine chain velocities into one vector
            for (int i = 0; i < left.N; ++i)
                setXi(v_combined, i, getXi(left.v, i));
            for (int i = 0; i < right.N; ++i)
                setXi(v_combined, left.N + i, getXi(right.v, i));

            // Use the broad-phase AABB to build barrier pairs
            std::vector<BarrierPair> barrier_pairs = build_barrier_pairs(x_combined, v_combined, left.N, right.N, dt);

            // Solve the left chain (global_offset = 0)
            xnew_left = left.x;
            auto [residual_left, iterations_left] = gauss_seidel_solver(xnew_left, left.xhat,
                                                                        left.mass, left.rest_lengths,
                                                                        dt, k_spring, g_accel,
                                                                        barrier_pairs, dhat, x_combined,
                                                                        0, max_iterations, tol_abs, eta);

            // Velocity update for the left chain
            update_velocity(left, xnew_left, dt);

            // Solve the right chain (global_offset = N+1)
            xnew_right = right.x;
            auto [residual_right, iterations_right] = gauss_seidel_solver(xnew_right, right.xhat,
                                                                          right.mass, right.rest_lengths,
                                                                          dt, k_spring, g_accel,
                                                                          barrier_pairs, dhat, x_combined,
                                                                          left.N, max_iterations, tol_abs, eta);

            // Velocity update for the right chain
            update_velocity(right, xnew_right, dt);

            // Export frame
            combine_positions(x_combined, left.x, right.x, left.N, right.N);
            export_frame(outdir, frame, x_combined, edges_combined);

            // Print info
            std::cout << "Frame " << std::setw(4) << frame
                      << " | residualL=" << std::scientific << residual_left
                      << " | residualR=" << std::scientific << residual_right
                      << " | iterationsL=" << std::setw(3) << iterations_left
                      << " | iterationsR=" << std::setw(3) << iterations_right
                      << '\n';
        }

        // Timing
        std::chrono::time_point<clock> t_end = clock::now();
        std::chrono::duration<double> elapsed = t_end - t_start;
        std::cout << "Total runtime: " << elapsed.count() << " seconds\n";

        return 0;
    }
}

int main() {
    return simulation::main();
}
