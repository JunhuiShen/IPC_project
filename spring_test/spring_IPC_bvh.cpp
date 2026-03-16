#include "bvh_box_leaf.h"
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
#include <functional>
#include <unordered_map>
#include <cstdint>

// ======================================================
// Prelim structure and functions
// ======================================================
using namespace std;
typedef std::vector<double> Vec;
namespace fs = std::__fs::filesystem;

// --- Core data structure ---
// Vec2 is provided by bvh_box_leaf.h

struct Mat2 {
    double a11, a12, a21, a22;
};

// rank-3 tensor, stored as two 2x2 matrices: [m0, m1]
struct Mat2x2x2 {
    Mat2 m0, m1;
};

// Get positions of the nodes
Vec2 get_xi(const Vec &x, int i) {
    return {x[2 * i], x[2 * i + 1]};
}

// Set positions of the nodes
void set_xi(Vec &x, int i, const Vec2 &v) {
    x[2 * i] = v.x;
    x[2 * i + 1] = v.y;
}

// Core math and linear algebra
namespace math {

    double node_distance(const Vec &a, int i, int j) {
        double dx = a[2 * i] - a[2 * j];
        double dy = a[2 * i + 1] - a[2 * j + 1];
        return std::sqrt(dx * dx + dy * dy);
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

    // Squared 2-norm
    static inline double norm2(const Vec2 &a) {
        return dot(a,a);
    }

    // 2-norm
    static inline double norm(const Vec2 &a)  {
        return std::sqrt(norm2(a));
    }

    // Outer product (v * v^T)
    static inline Mat2 outer(const Vec2 &a, const Vec2 &b) {
        return {a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y};
    }

    // --- Mat2 Operations ---

    // Matrix-matrix multiplication
    static inline Mat2 matmul(const Mat2 &A, const Mat2 &B) {
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
    static inline Vec2 matvec(const Mat2 &A, const Vec2 &v) {
        return {A.a11 * v.x + A.a12 * v.y, A.a21 * v.x + A.a22 * v.y};
    }
}

// ======================================================
// The force models of the spring and the barrier IPC potential and its analytic gradients and hessians
// ======================================================
namespace physics {
    using namespace math;

    // --- Local Gradient of the spring energy ---

    // Analytic local spring gradient at node i
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

        int N = (int) L.size() + 1;
        if (i - 1 >= 0) contrib(i - 1, i, L[i - 1]);
        if (i + 1 <= N - 1) contrib(i, i + 1, L[i]);

        return g_i;
    }

    // --- Local Hessian of the spring energy ---

    // Analytic local spring Hessian (2x2 block at node i)
    Mat2 local_spring_hess(int i, const Vec &x, double k, const std::vector<double> &L) {
        Mat2 H_ii{0.0, 0.0, 0.0, 0.0};

        std::function<void(int, int, double)> contrib = [&](int a, int b, double Lref) {
            Vec2 xa = get_xi(x, a), xb = get_xi(x, b);
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


    struct NodeSegmentPair {
        int node;   // i
        int seg0;   // j
        int seg1;   // j+1
    };

    // Scalar barrier energy
    double barrier_energy(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -(d - dhat) * (d - dhat) * std::log(d / dhat);
    }

    // Scalar barrier energy gradient
    double barrier_grad(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -2 * (d - dhat) * std::log(d / dhat) - (d - dhat) * (d - dhat) / d;
    }

    // Scalar barrier energy hessian
    double barrier_hess(double d, double dhat) {
        if (d >= dhat) return 0.0;
        return -2 * std::log(d / dhat) - 4 * (d - dhat) / d + (d - dhat) * (d - dhat) / (d * d);
    }

    // Compute the point–segment distance
    double node_segment_distance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1, double &t, Vec2 &p, Vec2 &r) {
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

    // Local barrier gradient for a node
    Vec2 local_barrier_grad(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
        Vec2 xi = get_xi(x, node);
        Vec2 x1 = get_xi(x, seg0);
        Vec2 x2 = get_xi(x, seg1);

        // Define v = x - x1
        Vec2 v = {xi.x - x1.x, xi.y - x1.y};

        double t;
        Vec2 p{}, r{};
        double d = node_segment_distance(xi, x1, x2, t, p, r);
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
        double bp = barrier_grad(d, dhat);  //b'(d)
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
            return matvec(P, f);
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
            return matvec(J1_T, f); // Returns J1^T * f
        } else if (who == seg1) {
            // g2 = J2^T f = ( -(T/L) )^T f = - (T/L)^T f
            Mat2 J2_T = scale(T_common_T, -1.0);
            return matvec(J2_T, f); // Returns J2^T * f
        }

        return {0, 0};
    }

    // --- Analytic derivatives for Tensor Term ---

    // Gets v = r_p = x - x1
    Vec2 get_v(const Vec &x, int node, int seg0) {
        Vec2 xi = get_xi(x, node);
        Vec2 x1 = get_xi(x, seg0);
        return sub(xi, x1);
    }

    // Gets theta = u^T v
    double get_theta(const Vec2 &u, const Vec2 &v) {
        return u.x * v.x + u.y * v.y;
    }

    // The gradient of projection matrix P w.r.t. x1 (but this one return -grad(P))
    Mat2x2x2 get_grad_jp_analytic(const Vec2 &u, const Mat2 &P, double L) {
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
    Mat2x2x2 get_grad_j_common_analytic(double L, const Vec2 &u, const Vec2 &v, double theta, const Mat2 &P, const Vec2 &r) {
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
    Mat2x2x2 get_grad_j_common_analytic_x2(double L, const Vec2 &u, const Vec2 &v, double theta, const Mat2 &P, const Vec2 &r) {
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

    // Local barrier hessian for a node
    Mat2 local_barrier_hess(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
        Vec2 xi = get_xi(x, node);
        Vec2 x1 = get_xi(x, seg0);
        Vec2 x2 = get_xi(x, seg1);

        double t;
        Vec2 p{}, r_vec{}; // r_vec is the residual r = P(xi-x1)
        double d = node_segment_distance(xi, x1, x2, t, p, r_vec);
        if (d >= dhat) return {0, 0, 0, 0};
        d = std::max(d, 1e-12);

        // Barrier quantities
        Vec2 n{r_vec.x / d, r_vec.y / d};
        double bp = barrier_grad(d, dhat);
        double bpp = barrier_hess(d, dhat);

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
        Vec2 v = get_v(x, node, seg0);      // v = r_p = xi - x1
        double theta = get_theta(u, v);     // theta = u^T v
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
            return matmul(Mat2{M.a11, M.a21, M.a12, M.a22}, matmul(Hrr, M)); // M^T * Hrr * M
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
        Mat2x2x2 gradJC1 = get_grad_j_common_analytic(L, u, v, theta, P, r_vec);
        Mat2x2x2 gradJP1 = get_grad_jp_analytic(u, P, L);

        // Get tensor slices for d(J1)/dx1_l = d(J_common)/dx1_l + d(J_P)/dx1_l
        Mat2 dJ1_dx = add(gradJC1.m0, gradJP1.m0);
        Mat2 dJ1_dy = add(gradJC1.m1, gradJP1.m1);

        // Contract: (K1)_il = sum_k d(J1)_ki/dx1_l * f_k
        // Col 1 (l=0): (K1)_i0 = sum_k d(J1)_ki/dx1_x * f_k = (dJ1_dx^T) * f
        Vec2 col1_K1 = matvec(transpose(dJ1_dx), f);
        // Col 2 (l=1): (K1)_i1 = sum_k d(J1)_ki/dx1_y * f_k = (dJ1_dy^T) * f
        Vec2 col2_K1 = matvec(transpose(dJ1_dy), f);

        K1 = {col1_K1.x, col2_K1.x, col1_K1.y, col2_K1.y};

        // K2 Term
        // K2 = [d(J2^T)/dx2] * f = [d(-J_common^T)/dx2] * f
        // (K2)_il = sum_k [ -d(J_common)_ki/dx2_l ] * f_k
        // Get analytic 3-tensor T_ikl = d(J_common_ik)/dx2_l
        Mat2x2x2 gradJC2 = get_grad_j_common_analytic_x2(L, u, v, theta, P, r_vec);

        // Get tensor slices for d(J2)/dx2_l = -d(J_common)/dx2_l
        Mat2 dJ2_dx = scale(gradJC2.m0, -1.0);
        Mat2 dJ2_dy = scale(gradJC2.m1, -1.0);

        // Contract: (K2)_il = sum_k d(J2)_ki/dx2_l * f_k
        // Col 1 (l=0): (K2)_i0 = sum_k d(J2)_ki/dx2_x * f_k = (dJ2_dx^T) * f
        Vec2 col1_K2 = matvec(transpose(dJ2_dx), f);
        // Col 2 (l=1): (K2)_i1 = sum_k d(J2)_ki/dx2_y * f_k = (dJ2_dy^T) * f
        Vec2 col2_K2 = matvec(transpose(dJ2_dy), f);

        K2 = {col1_K2.x, col2_K2.x, col1_K2.y, col2_K2.y};

        // Return appropriate block
        if (who == node) return Hnode;
        else if (who == seg0) return add(H1, K1);
        else if (who == seg1) return add(H2, K2);
        else return {0, 0, 0, 0};
    }

    // Per-node gradient of the non-barrier part of the incremental potential
    // This includes inertia, spring forces, gravity, and pin constraint
    // Barrier gradients are intentionally excluded here and are added separately in the solver using the barrier pairs
    Vec2 local_grad_no_barrier(int i, const Vec &x, const Vec &xhat, const Vec &xpin,
                               const std::vector<double> &mass, const std::vector<double> &L,
                               const std::vector<char> &is_pinned,
                               double dt, double k, const Vec2 &g_accel) {

        Vec2 xi = get_xi(x, i), xhi = get_xi(xhat, i);
        Vec2 gi{0.0, 0.0};

        // Inertia term
        gi.x += mass[i] * (xi.x - xhi.x);
        gi.y += mass[i] * (xi.y - xhi.y);

        // Spring term
        Vec2 gs = local_spring_grad(i, x, k, L);
        gi.x += dt * dt * gs.x;
        gi.y += dt * dt * gs.y;

        // Gravity
        gi.x -= dt * dt * mass[i] * g_accel.x;
        gi.y -= dt * dt * mass[i] * g_accel.y;

        // Large penalty stiffness for the pin constraint
        constexpr double k_pin = 5e6;

        if (is_pinned[i]) {
            Vec2 xpi = get_xi(xpin, i);
            gi.x += dt * dt * k_pin * (xi.x - xpi.x);
            gi.y += dt * dt * k_pin * (xi.y - xpi.y);
        }

        return gi;
    }

    // Per-node Hessian of the non-barrier part of the incremental potential.
    // Includes the block-local contributions from inertia, spring energy, and the pin constraint
    // Gravity contributes no Hessian
    // Barrier Hessians are excluded here and are added separately in the solver using the barrier pairs.
    Mat2 local_hess_no_barrier(int i, const Vec &x, const std::vector<double> &mass,
                               const std::vector<double> &L, const std::vector<char> &is_pinned,
                               double dt, double k) {

        // Inertia contribution
        Mat2 H{mass[i], 0, 0, mass[i]};

        // Spring contribution
        Mat2 Hs = local_spring_hess(i, x, k, L);
        H.a11 += dt * dt * Hs.a11;
        H.a12 += dt * dt * Hs.a12;
        H.a21 += dt * dt * Hs.a21;
        H.a22 += dt * dt * Hs.a22;

        // Soft pin constraint
        constexpr double k_pin = 5e6;

        if (is_pinned[i]) {
            H.a11 += dt * dt * k_pin;
            H.a22 += dt * dt * k_pin;
        }

        return H;
    }
}

// ======================================================
// Collision-free step filtering
// ======================================================
namespace collision_filtering {

    // ======================================================
    // Broad-phase: BVH
    // ======================================================
    namespace aabb {
        using namespace math;
        using namespace physics;

        // ------------------------------------------------------
        // Geometric helpers
        // ------------------------------------------------------

        // Check whether j can be the first node of a valid segment (j, j+1)
        inline bool is_valid_segment_start(int j, const std::vector<char>& segment_valid) {
            return (j >= 0 and j < (int)segment_valid.size() and segment_valid[j]);
        }

        // Check whether the node is an endpoint of the segment
        inline bool is_invalid_node_segment_pair(int node, int seg0, int seg1) {
            return (node == seg0) or (node == seg1);
        }

        // Build a hash key for a node-segment pair
        inline std::uint64_t pair_key(int node, int seg0) {
            return (std::uint64_t(std::uint32_t(node)) << 32) | std::uint32_t(seg0);
        }

        // Build one swept node AABB
        inline AABB build_node_box(const Vec& x_combined, const Vec& v_combined, int i, double dt, double pad) {
            Vec2 x0 = get_xi(x_combined, i);
            Vec2 v  = get_xi(v_combined, i);
            Vec2 x1{x0.x + dt * v.x, x0.y + dt * v.y};

            double min_x = std::min(x0.x, x1.x) - pad;
            double max_x = std::max(x0.x, x1.x) + pad;
            double min_y = std::min(x0.y, x1.y) - pad;
            double max_y = std::max(x0.y, x1.y) + pad;

            return AABB(Vec2(min_x, min_y), Vec2(max_x, max_y));
        }

        // Build one swept segment AABB
        inline AABB build_segment_box(const Vec& x_combined, const Vec& v_combined, int seg0, double dt, double pad) {
            Vec2 x0 = get_xi(x_combined, seg0);
            Vec2 x1 = get_xi(x_combined, seg0 + 1);
            Vec2 v0 = get_xi(v_combined, seg0);
            Vec2 v1 = get_xi(v_combined, seg0 + 1);

            double min_x = std::min({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x}) - pad;
            double max_x = std::max({x0.x, x1.x, x0.x + dt * v0.x, x1.x + dt * v1.x}) + pad;
            double min_y = std::min({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y}) - pad;
            double max_y = std::max({x0.y, x1.y, x0.y + dt * v0.y, x1.y + dt * v1.y}) + pad;

            return AABB(Vec2(min_x, min_y), Vec2(max_x, max_y));
        }

        // ------------------------------------------------------
        // BVH-based broad-phase cache
        //
        // A BVH is built over all valid segment AABBs at initialization.
        // To find node-segment candidate pairs, each node’s AABB is queried
        // against the segment BVH (O(log n + k) per query).
        //
        // On incremental update after one node moves:
        //   1. Recompute the moved node’s AABB.
        //   2. Recompute the adjacent segment AABBs that actually exist.
        //   3. Refit the segment BVH bottom-up (O(n) but cheap in practice).
        //   4. Remove stale pairs for the moved node and its adjacent segments.
        //   5. Re-query the BVH for the moved node’s new AABB.
        //   6. Brute-force scan nodes for each updated segment (O(n) over nodes,
        //      acceptable since only a small number of segments are updated per step).
        // ------------------------------------------------------

        struct BroadPhaseCache {
            // Per-node swept AABBs
            std::vector<AABB> node_boxes;

            // Per-segment swept AABBs (only valid segment indices are populated)
            std::vector<AABB> segment_boxes;
            std::vector<char> segment_valid;

            // BVH over valid segments.
            // seg_leaf_to_seg0[k] = segment start index for BVH leaf k (0..n_valid-1).
            // seg0_to_leaf[seg0]  = BVH leaf index for segment seg0 (-1 if invalid).
            // seg_bvh_boxes[k]    = AABB of BVH leaf k (kept in sync with segment_boxes).
            std::vector<int>     seg_leaf_to_seg0;
            std::vector<int>     seg0_to_leaf;
            std::vector<AABB>    seg_bvh_boxes;
            std::vector<BVHNode> seg_bvh_nodes;
            int                  seg_bvh_root = -1;

            // Current node-segment candidate pairs
            std::vector<NodeSegmentPair> pairs;
            std::unordered_map<std::uint64_t, std::size_t> pair_index;
        };

        // ------------------------------------------------------
        // Pair cache helpers
        // ------------------------------------------------------

        inline void clear_pairs(BroadPhaseCache& cache) {
            cache.pairs.clear();
            cache.pair_index.clear();
        }

        inline void add_pair(BroadPhaseCache& cache, int node, int seg0) {
            std::uint64_t key = pair_key(node, seg0);
            if (cache.pair_index.find(key) != cache.pair_index.end()) return;
            std::size_t idx = cache.pairs.size();
            cache.pairs.push_back({node, seg0, seg0 + 1});
            cache.pair_index[key] = idx;
        }

        inline void erase_pair_at(BroadPhaseCache& cache, std::size_t idx) {
            const std::size_t last = cache.pairs.size() - 1;
            NodeSegmentPair victim = cache.pairs[idx];
            std::uint64_t victim_key = pair_key(victim.node, victim.seg0);

            if (idx != last) {
                cache.pairs[idx] = cache.pairs[last];
                NodeSegmentPair moved = cache.pairs[idx];
                cache.pair_index[pair_key(moved.node, moved.seg0)] = idx;
            }

            cache.pairs.pop_back();
            cache.pair_index.erase(victim_key);
        }

        inline void remove_pairs_touching_node(BroadPhaseCache& cache, int node) {
            for (std::size_t i = cache.pairs.size(); i > 0; --i) {
                if (cache.pairs[i - 1].node == node)
                    erase_pair_at(cache, i - 1);
            }
        }

        inline void remove_pairs_touching_segment(BroadPhaseCache& cache, int seg0) {
            for (std::size_t i = cache.pairs.size(); i > 0; --i) {
                if (cache.pairs[i - 1].seg0 == seg0)
                    erase_pair_at(cache, i - 1);
            }
        }

        // ------------------------------------------------------
        // BVH queries
        // ------------------------------------------------------

        // Query the segment BVH with one node box, add valid node-segment pairs.
        inline void query_bvh_for_node(BroadPhaseCache& cache, int node) {
            if (cache.seg_bvh_root < 0) return;
            const int total_nodes = static_cast<int>(cache.node_boxes.size());

            std::vector<int> hits;
            query_bvh(cache.seg_bvh_nodes, cache.seg_bvh_root, cache.node_boxes[node], hits);

            for (int leaf_k : hits) {
                int seg0 = cache.seg_leaf_to_seg0[leaf_k];
                int seg1 = seg0 + 1;
                if (seg1 >= total_nodes) continue;
                if (is_invalid_node_segment_pair(node, seg0, seg1)) continue;
                add_pair(cache, node, seg0);
            }
        }

        // Brute-force: find all nodes whose AABB overlaps a segment’s AABB.
        // Used when a segment box changes.
        inline void scan_nodes_for_segment(BroadPhaseCache& cache, int seg0) {
            if (!is_valid_segment_start(seg0, cache.segment_valid)) return;

            const int total_nodes = static_cast<int>(cache.node_boxes.size());
            const int seg1 = seg0 + 1;
            const AABB& sb = cache.segment_boxes[seg0];

            for (int node = 0; node < total_nodes; ++node) {
                if (is_invalid_node_segment_pair(node, seg0, seg1)) continue;
                if (aabb_intersects(cache.node_boxes[node], sb))
                    add_pair(cache, node, seg0);
            }
        }

        // ------------------------------------------------------
        // Initialization
        // ------------------------------------------------------

        // Initialize the BVH broad-phase cache: build swept node/segment AABB
        inline BroadPhaseCache initialize_cache(const Vec& x_combined, const Vec& v_combined,
                                                const std::vector<char>& segment_valid,
                                                double dt, double node_pad, double segment_pad) {
            BroadPhaseCache cache;
            const int total_nodes = static_cast<int>(x_combined.size() / 2);
            const int nseg = std::max(0, total_nodes - 1);

            cache.node_boxes.resize(total_nodes);
            cache.segment_boxes.resize(nseg);
            cache.segment_valid = segment_valid;
            cache.seg0_to_leaf.assign(nseg, -1);

            for (int i = 0; i < total_nodes; ++i)
                cache.node_boxes[i] = build_node_box(x_combined, v_combined, i, dt, node_pad);

            for (int j = 0; j < nseg; ++j) {
                if (!is_valid_segment_start(j, cache.segment_valid)) continue;
                cache.segment_boxes[j] = build_segment_box(x_combined, v_combined, j, dt, segment_pad);
                int leaf_k = static_cast<int>(cache.seg_leaf_to_seg0.size());
                cache.seg0_to_leaf[j] = leaf_k;
                cache.seg_leaf_to_seg0.push_back(j);
                cache.seg_bvh_boxes.push_back(cache.segment_boxes[j]);
            }

            cache.seg_bvh_root = build_bvh(cache.seg_bvh_boxes, cache.seg_bvh_nodes);

            clear_pairs(cache);
            for (int i = 0; i < total_nodes; ++i)
                query_bvh_for_node(cache, i);

            return cache;
        }

        // ------------------------------------------------------
        // Incremental update after one node moved
        // ------------------------------------------------------

        // Refresh the BVH and candidate pairs after one node’s position changed.
        inline void refresh_pairs_for_moved_node(BroadPhaseCache& cache, const Vec& x_combined, const Vec& v_combined,
                                                 int moved_node, double dt, double node_pad, double segment_pad) {
            const int total_nodes = static_cast<int>(cache.node_boxes.size());
            if (moved_node < 0 or moved_node >= total_nodes) return;

            cache.node_boxes[moved_node] = build_node_box(x_combined, v_combined, moved_node, dt, node_pad);

            const int left_seg  = moved_node - 1;
            const int right_seg = moved_node;

            auto update_seg = [&](int seg0) {
                if (!is_valid_segment_start(seg0, cache.segment_valid)) return;
                cache.segment_boxes[seg0] = build_segment_box(x_combined, v_combined, seg0, dt, segment_pad);
                int leaf_k = cache.seg0_to_leaf[seg0];
                if (leaf_k >= 0)
                    cache.seg_bvh_boxes[leaf_k] = cache.segment_boxes[seg0];
            };

            update_seg(left_seg);
            update_seg(right_seg);

            if (cache.seg_bvh_root >= 0)
                refit_bvh(cache.seg_bvh_nodes, cache.seg_bvh_boxes);

            remove_pairs_touching_node(cache, moved_node);
            if (is_valid_segment_start(left_seg, cache.segment_valid))
                remove_pairs_touching_segment(cache, left_seg);
            if (is_valid_segment_start(right_seg, cache.segment_valid))
                remove_pairs_touching_segment(cache, right_seg);

            query_bvh_for_node(cache, moved_node);
            scan_nodes_for_segment(cache, left_seg);
            scan_nodes_for_segment(cache, right_seg);
        }

        // ------------------------------------------------------
        // Convenience wrapper for a one-shot full rebuild
        // ------------------------------------------------------

        // Generic helper to rebuild node–segment candidate pairs using swept AABBs with user-specified padding.
        inline std::vector<NodeSegmentPair> build_aabb_candidates(const Vec& x_combined, const Vec& v_combined,
                                                                  const std::vector<char>& segment_valid,
                                                                  double dt, double node_pad, double segment_pad) {
            BroadPhaseCache tmp = initialize_cache(x_combined, v_combined, segment_valid, dt, node_pad, segment_pad);
            return tmp.pairs;
        }

        // Initialize the persistent barrier active-set cache to detect node–segment pairs within barrier distance dhat
        inline BroadPhaseCache initialize_barrier_cache(const Vec& x_combined, const Vec& v_combined,
                                                        const std::vector<char>& segment_valid,
                                                        double dt, double dhat) {
            return initialize_cache(x_combined, v_combined, segment_valid, dt, /*node_pad=*/dhat, /*segment_pad=*/0.0);
        }

        // One-shot CCD candidate build for collision filtering
        inline std::vector<NodeSegmentPair> build_ccd_candidates(const Vec& x_combined, const Vec& v_combined,
                                                                 const std::vector<char>& segment_valid,
                                                                 double dt) {
            return build_aabb_candidates(x_combined, v_combined, segment_valid, dt, /*node_pad=*/0.0, /*segment_pad=*/0.0);
        }

        // One-shot trust-region candidate build for collision filtering
        inline std::vector<NodeSegmentPair> build_trust_region_candidates(const Vec& x_combined, const Vec& v_combined,
                                                                          const std::vector<char>& segment_valid,
                                                                          double dt, double motion_pad) {
            return build_aabb_candidates(x_combined, v_combined, segment_valid, dt, /*node_pad=*/motion_pad, /*segment_pad=*/0.0);
        }
    }

    // ======================================================
    // Narrow-phase: Exact CCD
    // ======================================================
    namespace ccd {
        using namespace math;

        // Point–segment continuous collision detection
        bool ccd_point_segment_2d(const Vec2 &x1, const Vec2 &dx1, const Vec2 &x2, const Vec2 &dx2,
                                  const Vec2 &x3, const Vec2 &dx3, double &t_out, double eps = 1e-12) {

            // Compute the quadratic coefficients of f(t) = a t^2 + b t + c
            Vec2 x21 = sub(x1, x2);
            Vec2 x32 = sub(x3, x2);
            Vec2 dx21 = sub(dx1, dx2);
            Vec2 dx32 = sub(dx3, dx2);

            double a = cross(dx32, dx21);
            double b = cross(dx32, x21) + cross(x32, dx21);
            double c = cross(x32, x21);

            double t_candidates[2];
            int num_roots = 0;

            // Degenerate linear case if a = 0
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

            double s_param = dot(rel, seg) / seg_len2;
            if (s_param < 0.0 or s_param > 1.0) return false;

            // Valid collision
            t_out = t_star;
            return true;
        }

        // CCD-based safe step computation
        double ccd_get_safe_step(const Vec2 &x1, const Vec2 &dx1, const Vec2 &x2, const Vec2 &dx2,
                                 const Vec2 &x3, const Vec2 &dx3, double eta = 0.9) {

            double t_hit;
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

    // ======================================================
    // Trust-region method
    // ======================================================
    namespace trust_region {
        using namespace math;
        using namespace physics;

        static inline double trust_region_weight(const Vec2& xi, const Vec2& dxi,
                                                 const Vec2& xj, const Vec2& dxj,
                                                 const Vec2& xk, const Vec2& dxk,
                                                 double eta = 0.4) {
            double s;
            Vec2 p{}, r{};
            double d0 = physics::node_segment_distance(xi, xj, xk, s, p, r);

            constexpr double eps = 1e-12;
            d0 = std::max(d0, eps);

            const double M = norm(dxi) + norm(dxj) + norm(dxk);

            if (M <= eps)
                return 1.0;

            const double w = eta * d0 / M;
            return std::max(0.0, std::min(1.0, w));
        }
    }

}

// ======================================================
// Gauss-Seidel solver
// ======================================================
namespace solver {
    using namespace math;
    using namespace physics;
    using namespace collision_filtering::aabb;
    using namespace collision_filtering::ccd;

    // Select how Newton updates are filtered to maintain collision-free iterates
    enum class StepPolicy {
        CCD,
        TrustRegion
    };

    // ------------------------------------------------------
    // CCD statistics
    // ------------------------------------------------------
    // ccd_total_tests: number of actual narrow-phase CCD evaluations performed
    // ccd_total_collisions: number of those CCD evaluations that detected a collision
    inline std::uint64_t ccd_total_tests = 0;
    inline std::uint64_t ccd_total_collisions = 0;

    // Inverse of any 2x2 matrix
    Mat2 matrix2d_inverse(const Mat2 &H) {
        double det = H.a11 * H.a22 - H.a12 * H.a21;
        if (std::abs(det) < 1e-12)
            throw std::runtime_error("Singular matrix in inverse()");
        double inv_det = 1.0 / det;
        return {
                H.a22 * inv_det, -H.a12 * inv_det,
                -H.a21 * inv_det, H.a11 * inv_det
        };
    }

    // Gradient of the full incremental potential for node i.
    // The block-local non-barrier terms are evaluated with local_grad_no_barrier()
    // Barrier contributions are added separately from the current global node–segment barrier pair set
    Vec2 compute_local_gradient(int i, const Vec &x_local, const Vec &xhat_local, const Vec &xpin_local,
                                const std::vector<double> &mass_local, const std::vector<double> &L_local,
                                const std::vector<char> &is_pinned_local,
                                double dt, double k, const Vec2 &g_accel,
                                const std::vector<NodeSegmentPair> &barrier_pairs,
                                double dhat, const Vec &x_global, int global_offset) {

        Vec2 gi = local_grad_no_barrier(i, x_local, xhat_local, xpin_local,
                                        mass_local, L_local, is_pinned_local,
                                        dt, k, g_accel);

        const int who_global = global_offset + i;

        Vec2 gbar{0.0, 0.0};
        for (const NodeSegmentPair &c : barrier_pairs) {
            if (c.node != who_global and c.seg0 != who_global and c.seg1 != who_global)
                continue;

            Vec2 gb = local_barrier_grad(who_global, x_global, c.node, c.seg0, c.seg1, dhat);
            gbar.x += gb.x;
            gbar.y += gb.y;
        }

        gi.x += dt * dt * gbar.x;
        gi.y += dt * dt * gbar.y;
        return gi;
    }

    // Hessian of the full incremental potential for node i.
    // The block-local non-barrier terms are evaluated with local_hess_no_barrier()
    // Barrier contributions are added separately from the current global node–segment barrier pair set
    Mat2 compute_local_hessian(int i, const Vec &x_local, const std::vector<double> &mass_local,
                               const std::vector<double> &L_local, const std::vector<char> &is_pinned_local,
                               double dt, double k,
                               const std::vector<NodeSegmentPair> &barrier_pairs,
                               double dhat, const Vec &x_global, int global_offset) {

        Mat2 Hi = local_hess_no_barrier(i, x_local, mass_local, L_local, is_pinned_local, dt, k);

        const int who_global = global_offset + i;

        for (const NodeSegmentPair &c : barrier_pairs) {
            if (c.node != who_global and c.seg0 != who_global and c.seg1 != who_global)
                continue;

            Mat2 Hb = local_barrier_hess(who_global, x_global, c.node, c.seg0, c.seg1, dhat);

            Hi.a11 += dt * dt * Hb.a11;
            Hi.a12 += dt * dt * Hb.a12;
            Hi.a21 += dt * dt * Hb.a21;
            Hi.a22 += dt * dt * Hb.a22;
        }

        return Hi;
    }

    // CCD step weight
    double compute_safe_step_ccd(int who_global, const Vec2 &dx, const Vec &x_global,
                                 const std::vector<NodeSegmentPair> &ccd_candidate_set, double eta = 0.9) {

        double omega = 1.0;

        for (const NodeSegmentPair &c : ccd_candidate_set) {
            if (who_global != c.node and who_global != c.seg0 and who_global != c.seg1)
                continue;

            Vec2 xi = get_xi(x_global, c.node);
            Vec2 xj = get_xi(x_global, c.seg0);
            Vec2 xk = get_xi(x_global, c.seg1);

            Vec2 dxi{0,0}, dxj{0,0}, dxk{0,0};
            Vec2 full{-dx.x, -dx.y};

            if (who_global == c.node) dxi = full;
            else if (who_global == c.seg0) dxj = full;
            else if (who_global == c.seg1) dxk = full;

            // Count one actual narrow-phase CCD test
            double t_hit = 0.0;
            bool collision_found = ccd_point_segment_2d(xi, dxi, xj, dxj, xk, dxk, t_hit);
            ++ccd_total_tests;

            double omega_c = 1.0;
            if (collision_found) {
                ++ccd_total_collisions;

                if (t_hit <= 1e-12) {
                    // Already in collision, don't move at all
                    omega_c = 0.0;
                } else {
                    omega_c = eta * t_hit;
                }
            }

            omega = std::min(omega, omega_c);
            if (omega <= 0.0) return 0.0;
        }

        return omega;
    }

    // Trust region step weight
    double compute_safe_step_trust_region(int who_global, const Vec2& dx, const Vec& x_global,
                                          const std::vector<NodeSegmentPair>& candidate_set, double eta = 0.4) {

        double omega = 1.0;

        for (const auto& c : candidate_set) {
            if (who_global != c.node and who_global != c.seg0 and who_global != c.seg1)
                continue;

            Vec2 xi = get_xi(x_global, c.node);
            Vec2 xj = get_xi(x_global, c.seg0);
            Vec2 xk = get_xi(x_global, c.seg1);

            Vec2 dxi{0,0}, dxj{0,0}, dxk{0,0};
            Vec2 full{-dx.x, -dx.y};

            if (who_global == c.node) dxi = full;
            else if (who_global == c.seg0) dxj = full;
            else if (who_global == c.seg1) dxk = full;

            omega = std::min(
                    omega,
                    collision_filtering::trust_region::trust_region_weight(xi, dxi, xj, dxj, xk, dxk, eta)
            );
        }

        return omega;
    }

    // Collision-free policy switch
    double compute_safe_filtering_step_policy(StepPolicy policy, int who_global, const Vec2 &dx, const Vec &x_global,
                                              const std::vector<NodeSegmentPair> &candidate_set, double eta) {

        if (policy == StepPolicy::CCD)
            return compute_safe_step_ccd(who_global, dx, x_global, candidate_set, eta);
        else if (policy == StepPolicy::TrustRegion)
            return compute_safe_step_trust_region(who_global, dx, x_global, candidate_set, eta);

        throw std::runtime_error("Unknown StepPolicy");
    }

    // Block description for one chain or one contiguous block of nodes
    struct BlockView {
        Vec* x;
        const Vec* xhat;
        const Vec* xpin;
        const std::vector<double>* mass;
        const std::vector<double>* L;
        const std::vector<char>* is_pinned;
        int offset;

        int size() const { return static_cast<int>(mass->size()); }
    };

    // Performs one Newton step for a single node
    inline void update_one_node(int local_i, const BlockView& b, Vec& x_global, BroadPhaseCache& broad_cache,
                                const Vec& v_vel_global, const std::vector<char>& segment_valid,
                                double dt, double k, const Vec2& g_accel,
                                double dhat, double eta, StepPolicy filtering_step_policy) {

        // Energy model uses current cached barrier pairs
        Vec2 gi = compute_local_gradient(local_i, *b.x, *b.xhat, *b.xpin,
                                         *b.mass, *b.L, *b.is_pinned,
                                         dt, k, g_accel,
                                         broad_cache.pairs, dhat, x_global, b.offset);

        Mat2 Hi = compute_local_hessian(local_i, *b.x, *b.mass, *b.L, *b.is_pinned,
                                        dt, k, broad_cache.pairs, dhat, x_global, b.offset);

        Vec2 dx = matvec(matrix2d_inverse(Hi), gi);

        const int who_global = b.offset + local_i;

        Vec v_newton(v_vel_global.size(), 0.0);
        set_xi(v_newton, who_global, {-dx.x / dt, -dx.y / dt});

        std::vector<NodeSegmentPair> filtering_candidate_set;

        if (filtering_step_policy == StepPolicy::CCD) {
            filtering_candidate_set = build_ccd_candidates(x_global, v_newton, segment_valid, dt);
        }
        else if (filtering_step_policy == StepPolicy::TrustRegion) {
            double motion_pad = norm(dx) / eta;
            filtering_candidate_set = build_trust_region_candidates(x_global, v_newton, segment_valid, dt, motion_pad);
        }
        else {
            throw std::runtime_error("Unknown StepPolicy");
        }

        double omega = compute_safe_filtering_step_policy(
                filtering_step_policy, who_global, dx, x_global, filtering_candidate_set, eta
        );

        Vec2 xi = get_xi(*b.x, local_i);
        xi.x -= omega * dx.x;
        xi.y -= omega * dx.y;

        set_xi(*b.x, local_i, xi);
        set_xi(x_global, who_global, xi);

        // Incrementally refresh the energy broad phase after this node move
        refresh_pairs_for_moved_node(broad_cache, x_global, v_vel_global, who_global,
                                     dt, /*node_pad=*/dhat, /*segment_pad=*/0.0);
    }

    // Global convergence residual
    double compute_global_residual(const Vec &x_local, const Vec &xhat_local, const Vec &xpin_local,
                                   const std::vector<double> &mass_local, const std::vector<double> &L_local,
                                   const std::vector<char> &is_pinned_local,
                                   double dt, double k, const Vec2 &g_accel,
                                   const std::vector<NodeSegmentPair> &barrier_pairs_eval,
                                   double dhat, const Vec &x_global, int global_offset) {

        const int N = static_cast<int>(mass_local.size());
        double r_inf = 0.0;

        for (int i = 0; i < N; ++i) {
            Vec2 g = compute_local_gradient(i, x_local, xhat_local, xpin_local,
                                            mass_local, L_local, is_pinned_local,
                                            dt, k, g_accel,
                                            barrier_pairs_eval, dhat, x_global, global_offset);

            r_inf = std::max(r_inf, std::abs(g.x));
            r_inf = std::max(r_inf, std::abs(g.y));
        }

        return r_inf;
    }

    // Nonlinear Gauss–Seidel solver.
    std::pair<double,int> global_gauss_seidel_solver(std::vector<BlockView>& blocks, Vec& x_global, const Vec& v_vel_global,
                                                     const std::vector<char>& segment_valid,
                                                     double dt, double k, const Vec2& g_accel,
                                                     double dhat, int max_global_iters, double tol_abs,
                                                     double eta, StepPolicy filtering_step_policy,
                                                     std::vector<double>* residual_history /*= nullptr*/) {

        // Persistent cache for the barrier pairs
        BroadPhaseCache barrier_cache = initialize_barrier_cache(x_global, v_vel_global, segment_valid, dt, dhat);

        // Residual history
        if (residual_history) residual_history->clear();

        auto eval_residual = [&]() {
            double residual = 0.0;
            for (const BlockView& b : blocks) {
                residual = std::max(
                        residual,
                        compute_global_residual(*b.x, *b.xhat, *b.xpin,
                                                *b.mass, *b.L, *b.is_pinned,
                                                dt, k, g_accel,barrier_cache.pairs, dhat,
                                                x_global, b.offset)
                );
            }
            return residual;
        };

        // Initial residual
        double r = eval_residual();
        if (residual_history) residual_history->push_back(r);

        if (r < tol_abs)
            return {r, 0};

        for (int it = 1; it < max_global_iters; ++it) {

            // Node sweep
            for (const BlockView& b : blocks) {
                for (int i = 0; i < b.size(); ++i) {
                    update_one_node(i, b, x_global, barrier_cache, v_vel_global, segment_valid,
                                    dt, k, g_accel, dhat, eta, filtering_step_policy);
                }
            }

            // Residual evaluation
            r = eval_residual();
            if (residual_history) residual_history->push_back(r);

            if (r < tol_abs)
                return {r, it};
        }

        return {r, max_global_iters};
    }
}

// ======================================================
// Visualization utilities
// ======================================================
namespace visualization {
    using namespace math;

    //  Export an Obj file
    void export_obj(const std::string &filename, const Vec &x, const std::vector<std::pair<int, int>> &edges) {
        std::ofstream out(filename);

        if (!out) {
            std::cerr << "Error: cannot write " << filename << "\n";
            return;
        }
        int N = (int) (x.size() / 2);

        for (int i = 0; i < N; ++i) {
            Vec2 xi = get_xi(x, i);
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

// ======================================================
// Build the chains
// ======================================================
namespace chain_model{
    using namespace math;

    // Core data structure for a single chain
    struct Chain {
        int N{}; // number of nodes
        Vec x; // positions
        Vec v; // velocities
        Vec xhat;  // predicted positions
        Vec xpin; // fixed pin targets
        std::vector<double> mass; // per-node masses
        std::vector<char> is_pinned; // per-node pin flags
        std::vector<double> rest_lengths; // rest spring lengths
        std::vector<std::pair<int, int>> edges; // connectivity list
    };

    Chain make_chain(Vec2 start, Vec2 end, int N, double mass_value) {
        Chain c;
        c.N = N;
        c.x.resize(2 * N);
        c.v.assign(2 * N, 0.0);
        c.xhat.assign(2 * N, 0.0);
        c.xpin.assign(2 * N, 0.0);
        c.mass.assign(N, mass_value);
        c.is_pinned.assign(N, 0);

        for (int i = 0; i < N; ++i) {
            double t = (N == 1) ? 0.0 : double(i) / (N - 1);
            Vec2 xi{start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)};
            set_xi(c.x, i, xi);
            set_xi(c.xpin, i, xi);
        }

        for (int i = 0; i < N - 1; ++i) {
            c.edges.emplace_back(i, i + 1);
            c.rest_lengths.push_back(node_distance(c.x, i, i + 1));
        }

        return c;
    }
}

// ======================================================
// Update the per-step linear extrapolation and velocity after solve
// ======================================================
namespace state_update{
    using namespace chain_model;
    using namespace math;

    // Build xhat
    void build_xhat(Chain &c, double dt) {
        for (int i = 0; i < c.N; ++i) {
            Vec2 xi = get_xi(c.x, i);
            Vec2 vi = get_xi(c.v, i);
            set_xi(c.xhat, i, {xi.x + dt * vi.x, xi.y + dt * vi.y});
        }
    }

    // Update node velocities from the new positions
    void update_velocity(Chain &c, const Vec &xnew, double dt) {
        for (int i = 0; i < c.N; ++i) {
            Vec2 xi_new = get_xi(xnew, i);
            Vec2 xi_old = get_xi(c.x, i);
            set_xi(c.v, i, {(xi_new.x - xi_old.x) / dt, (xi_new.y - xi_old.y) / dt});
        }
        c.x = xnew;
    }
}

// ======================================================
// Unified initial guess wrapper
// ======================================================
namespace initial_guess {
    using namespace math;
    using namespace chain_model;
    using namespace state_update;
    using physics::NodeSegmentPair;

    using namespace collision_filtering::aabb;
    using collision_filtering::ccd::ccd_get_safe_step;
    using collision_filtering::trust_region::trust_region_weight;

    enum class Type {
        Trivial,
        Affine,
        CCD,
        TrustRegion
    };

    // Description of one contiguous block in the global indexing
    struct BlockRef {
        Chain* chain;
        Vec* xnew;
        int offset;
    };

    // ------------------------------------------------------
    // Helpers for global assembly
    // ------------------------------------------------------

    inline int total_nodes(const std::vector<BlockRef>& blocks) {
        int total = 0;
        for (const auto& b : blocks)
            total += b.chain->N;
        return total;
    }

    inline void scatter_block_positions(Vec& x_combined, const Vec& x_block, int offset, int N_block) {
        for (int i = 0; i < N_block; ++i)
            set_xi(x_combined, offset + i, get_xi(x_block, i));
    }

    inline void scatter_block_velocities(Vec& v_combined, const Vec& v_block, int offset, int N_block) {
        for (int i = 0; i < N_block; ++i)
            set_xi(v_combined, offset + i, get_xi(v_block, i));
    }

    inline void build_x_combined_from_current_positions(Vec& x_combined, const std::vector<BlockRef>& blocks) {
        for (const auto& b : blocks)
            scatter_block_positions(x_combined, b.chain->x, b.offset, b.chain->N);
    }

    inline void build_x_combined_from_xnew(Vec& x_combined, const std::vector<BlockRef>& blocks) {
        for (const auto& b : blocks)
            scatter_block_positions(x_combined, *b.xnew, b.offset, b.chain->N);
    }

    inline void build_v_combined_from_chain_velocities(Vec& v_combined, const std::vector<BlockRef>& blocks) {
        for (const auto& b : blocks)
            scatter_block_velocities(v_combined, b.chain->v, b.offset, b.chain->N);
    }

    // ------------------------------------------------------
    // Affine initial guess
    // ------------------------------------------------------
    namespace affine_initial_guess {
        using namespace math;
        using namespace chain_model;

        // Affine predictor structure
        struct AffineParams {
            double omega;
            Vec2 vhat;
            Vec2 xcom;
        };

        // Fit affine field V(X) = vhat + omega R (X - X_com)
        // Pinned nodes are excluded from the least-squares fit.
        inline AffineParams compute_affine_params_global(const std::vector<BlockRef>& blocks) {
            Vec2 xcom{0.0, 0.0};
            double M = 0.0;

            for (const auto& b : blocks) {
                const Chain& c = *b.chain;
                for (int i = 0; i < c.N; ++i) {
                    if (c.is_pinned[i]) continue;
                    Vec2 xi = get_xi(c.x, i);
                    xcom.x += c.mass[i] * xi.x;
                    xcom.y += c.mass[i] * xi.y;
                    M += c.mass[i];
                }
            }

            if (M <= 1e-12)
                return {0.0, {0.0, 0.0}, {0.0, 0.0}};

            xcom.x /= M;
            xcom.y /= M;

            double G[3][3] = {{0.0}};
            double bvec[3] = {0.0, 0.0, 0.0};

            for (const auto& br : blocks) {
                const Chain& c = *br.chain;
                for (int i = 0; i < c.N; ++i) {
                    if (c.is_pinned[i]) continue;

                    Vec2 Xi = get_xi(c.x, i);
                    Vec2 Vi = get_xi(c.v, i);
                    Vec2 d{Xi.x - xcom.x, Xi.y - xcom.y};

                    Vec2 U1{-d.y, d.x};
                    Vec2 U2{1.0, 0.0};
                    Vec2 U3{0.0, 1.0};
                    Vec2 U[3] = {U1, U2, U3};

                    double w = c.mass[i];

                    for (int k = 0; k < 3; ++k) {
                        bvec[k] += w * (U[k].x * Vi.x + U[k].y * Vi.y);
                        for (int j = 0; j < 3; ++j) {
                            G[k][j] += w * (U[k].x * U[j].x + U[k].y * U[j].y);
                        }
                    }
                }
            }

            double omega  = (std::abs(G[0][0]) > 1e-12) ? bvec[0] / G[0][0] : 0.0;
            double vhat_x = (std::abs(G[1][1]) > 1e-12) ? bvec[1] / G[1][1] : 0.0;
            double vhat_y = (std::abs(G[2][2]) > 1e-12) ? bvec[2] / G[2][2] : 0.0;

            return {omega, {vhat_x, vhat_y}, xcom};
        }

        inline Vec2 affine_velocity_at(const AffineParams& ap, const Vec2& x) {
            Vec2 d{x.x - ap.xcom.x, x.y - ap.xcom.y};
            return {ap.vhat.x - ap.omega * d.y, ap.vhat.y + ap.omega * d.x};
        }

        inline void build_v_combined_from_affine(Vec& v_combined, const std::vector<BlockRef>& blocks,
                                                 const AffineParams& ap) {
            for (const auto& b : blocks) {
                const Chain& c = *b.chain;
                for (int i = 0; i < c.N; ++i) {
                    Vec2 xi = get_xi(c.x, i);
                    Vec2 vi = c.is_pinned[i] ? Vec2{0.0, 0.0} : affine_velocity_at(ap, xi);
                    set_xi(v_combined, b.offset + i, vi);
                }
            }
        }

        inline void affine_initial_guess_global(const AffineParams& ap, const BlockRef& b, double dt) {
            Chain& c = *b.chain;
            Vec& xnew = *b.xnew;
            xnew = c.x;

            for (int i = 0; i < c.N; ++i) {
                Vec2 xi = get_xi(c.x, i);

                if (c.is_pinned[i]) {
                    set_xi(xnew, i, xi);
                    continue;
                }

                Vec2 d{xi.x - ap.xcom.x, xi.y - ap.xcom.y};
                Vec2 v_aff{ap.vhat.x - ap.omega * d.y, ap.vhat.y + ap.omega * d.x};

                set_xi(xnew, i, {xi.x + dt * v_aff.x, xi.y + dt * v_aff.y});
            }
        }
    }

    // ------------------------------------------------------
    // Step-size helpers
    // ------------------------------------------------------

    inline double compute_initial_guess_ccd_step(const Vec& x_combined, const Vec& v_combined,
                                                 const std::vector<NodeSegmentPair>& barrier_pairs,
                                                 double dt, double eta = 0.9) {
        double omega = 1.0;

        for (const auto& c : barrier_pairs) {
            Vec2 xi = get_xi(x_combined, c.node);
            Vec2 xj = get_xi(x_combined, c.seg0);
            Vec2 xk = get_xi(x_combined, c.seg1);

            Vec2 vi = get_xi(v_combined, c.node);
            Vec2 vj = get_xi(v_combined, c.seg0);
            Vec2 vk = get_xi(v_combined, c.seg1);

            Vec2 dxi{dt * vi.x, dt * vi.y};
            Vec2 dxj{dt * vj.x, dt * vj.y};
            Vec2 dxk{dt * vk.x, dt * vk.y};

            double omega_c = ccd_get_safe_step(xi, dxi, xj, dxj, xk, dxk, eta);
            omega = std::min(omega, omega_c);

            if (omega <= 0.0)
                return 0.0;
        }

        return omega;
    }

    inline double compute_initial_guess_trust_region_step(const Vec& x_combined, const Vec& v_combined,
                                                          const std::vector<NodeSegmentPair>& barrier_pairs,
                                                          double dt, double eta = 0.4) {
        double omega = 1.0;

        for (const auto& c : barrier_pairs) {
            Vec2 xi = get_xi(x_combined, c.node);
            Vec2 xj = get_xi(x_combined, c.seg0);
            Vec2 xk = get_xi(x_combined, c.seg1);

            Vec2 vi = get_xi(v_combined, c.node);
            Vec2 vj = get_xi(v_combined, c.seg0);
            Vec2 vk = get_xi(v_combined, c.seg1);

            Vec2 dxi{dt * vi.x, dt * vi.y};
            Vec2 dxj{dt * vj.x, dt * vj.y};
            Vec2 dxk{dt * vk.x, dt * vk.y};

            double omega_c = trust_region_weight(xi, dxi, xj, dxj, xk, dxk, eta);
            omega = std::min(omega, omega_c);

            if (omega <= 0.0)
                return 0.0;
        }

        return omega;
    }

    // ------------------------------------------------------
    // Unified initial guess application
    // ------------------------------------------------------

    inline void apply(Type initial_guess_type,
                      const std::vector<BlockRef>& blocks,
                      Vec& x_combined, Vec& v_combined,
                      const std::vector<char>& segment_valid,
                      double dt, double dhat, double eta) {

        const int total = total_nodes(blocks);
        x_combined.assign(2 * total, 0.0);
        v_combined.assign(2 * total, 0.0);

        // Trivial initial guess
        if (initial_guess_type == Type::Trivial) {
            for (const auto& b : blocks)
                *b.xnew = b.chain->x;

            build_v_combined_from_chain_velocities(v_combined, blocks);
            build_x_combined_from_current_positions(x_combined, blocks);
            return;
        }

        // Affine rotational initial guess
        if (initial_guess_type == Type::Affine) {
            using namespace affine_initial_guess;

            AffineParams ap = compute_affine_params_global(blocks);
            build_v_combined_from_affine(v_combined, blocks, ap);

            for (const auto& b : blocks)
                affine_initial_guess_global(ap, b, dt);

            build_x_combined_from_xnew(x_combined, blocks);
            return;
        }

        // CCD-projected initial guess
        if (initial_guess_type == Type::CCD) {
            build_x_combined_from_current_positions(x_combined, blocks);
            build_v_combined_from_chain_velocities(v_combined, blocks);

            auto init_pairs = build_ccd_candidates(x_combined, v_combined, segment_valid, dt);
            double omega0 = compute_initial_guess_ccd_step(x_combined, v_combined, init_pairs, dt, eta);

            for (const auto& b : blocks) {
                Chain& c = *b.chain;
                Vec& xnew = *b.xnew;
                xnew = c.x;

                for (int i = 0; i < c.N; ++i) {
                    Vec2 xi = get_xi(c.x, i);

                    if (c.is_pinned[i]) {
                        set_xi(xnew, i, xi);
                        continue;
                    }

                    Vec2 vi = get_xi(c.v, i);
                    set_xi(xnew, i, {xi.x + omega0 * dt * vi.x,xi.y + omega0 * dt * vi.y});
                }
            }

            build_x_combined_from_xnew(x_combined, blocks);
            return;
        }

        // Trust-region projected initial guess
        if (initial_guess_type == Type::TrustRegion) {
            build_x_combined_from_current_positions(x_combined, blocks);
            build_v_combined_from_chain_velocities(v_combined, blocks);

            double vmax = 0.0;
            for (int i = 0; i < total; ++i)
                vmax = std::max(vmax, norm(get_xi(v_combined, i)));

            double motion_pad = dt * vmax / eta;
            auto barrier_pairs = build_trust_region_candidates(x_combined, v_combined, segment_valid, dt, motion_pad);

            double alpha = compute_initial_guess_trust_region_step(x_combined, v_combined, barrier_pairs, dt, eta);
            alpha = (alpha < 0.0) ? 0.0 : (alpha > 1.0) ? 1.0 : alpha;

            for (const auto& b : blocks) {
                Chain& c = *b.chain;
                Vec& xnew = *b.xnew;
                xnew = c.x;

                for (int i = 0; i < c.N; ++i) {
                    Vec2 xi = get_xi(c.x, i);

                    if (c.is_pinned[i]) {
                        set_xi(xnew, i, xi);
                        continue;
                    }

                    Vec2 vi = get_xi(c.v, i);
                    set_xi(xnew, i, {xi.x + alpha * dt * vi.x, xi.y + alpha * dt * vi.y});
                }
            }

            build_x_combined_from_xnew(x_combined, blocks);
            return;
        }

        throw std::runtime_error("Unknown initial guess type");
    }
}

// ======================================================
// Main simulation
// ======================================================
namespace simulation {
    using namespace math;
    using namespace solver;
    using namespace visualization;
    using namespace chain_model;
    using namespace state_update;
    using namespace collision_filtering::aabb;
    using namespace collision_filtering::ccd;
    using namespace initial_guess;

    enum class ExampleType {
        Example1,
        Example2,
        Example3
    };

    int sim() {
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();

        std::string outdir = "frames_spring_IPC_bvh";

        // Delete old frame folder if it exists
        if (fs::exists(outdir)) {
            fs::remove_all(outdir);
        }

        fs::create_directories(outdir);

        // Parameters
        double dt = 1.0 / 30.0;
        Vec2 g_accel{0.0, -9.81};
        double k_spring = 20.0;
        int max_global_iters = 1000;
        double tol_abs = 1e-6;
        double dhat = 0.1;
        int number_of_nodes = 11;

        // Choose examples
        // Example 1: two vertical chains moving toward each other
        // Example 2: multiple chains fall onto a pinned ground segment
//        ExampleType example_type = ExampleType::Example1;
//         ExampleType example_type = ExampleType::Example2;
        ExampleType example_type = ExampleType::Example3;

        // Choose the initial guess type (Trivial/Affine/CCD/TrustRegion)
        Type initial_guess_type = Type::CCD;

        // Choose the collision-free step filtering policy (CCD/TrustRegion)
        StepPolicy filtering_step_policy = StepPolicy::CCD;

        double eta;
        if (initial_guess_type == Type::TrustRegion && filtering_step_policy == StepPolicy::TrustRegion) {
            eta = 0.4;
        }
        else if (initial_guess_type == Type::CCD && filtering_step_policy == StepPolicy::CCD) {
            eta = 0.9;
        }
        else {
            eta = 0.9;
        }

        int total_frame = 60;

        std::vector<Chain> chains;

        // Example 1: two vertical chains moving toward each other
        if (example_type == ExampleType::Example1) {
            total_frame = 150;

            Chain chain1 = chain_model::make_chain({-0.1,  1.5}, {-0.1, -1.5}, number_of_nodes, 0.05);
            Chain chain2 = chain_model::make_chain({ 0.1,  1.5}, { 0.1, -1.5}, number_of_nodes, 0.05);

            chain1.is_pinned[0] = 1;
            chain2.is_pinned[0] = 1;

            set_xi(chain1.xpin, 0, get_xi(chain1.x, 0));
            set_xi(chain2.xpin, 0, get_xi(chain2.x, 0));

            for (int i = 0; i < chain1.N; ++i)
                set_xi(chain1.v, i, {-6.0, 0.0});

            for (int i = 0; i < chain2.N; ++i)
                set_xi(chain2.v, i, {6.0, 0.0});

            chains.push_back(chain1);
            chains.push_back(chain2);
        }
        // Example 2: multiple chains fall onto a pinned ground segment
        else if (example_type == ExampleType::Example2) {
            total_frame = 60;

            Chain chain1 = chain_model::make_chain({-0.8, 1.2}, { 1.6, 0.0}, number_of_nodes, 0.05);
            Chain chain2 = chain_model::make_chain({-0.4, 2.0}, { 2.0, 0.8}, number_of_nodes, 0.05);
            Chain chain3 = chain_model::make_chain({ 0.0, 2.8}, { 2.4, 1.6}, number_of_nodes, 0.05);
            Chain ground = chain_model::make_chain({-2.0, -1.8}, {2.0, -1.8}, 2, 1.0);

            ground.is_pinned[0] = 1;
            ground.is_pinned[1] = 1;

            set_xi(ground.xpin, 0, get_xi(ground.x, 0));
            set_xi(ground.xpin, 1, get_xi(ground.x, 1));

            for (int i = 0; i < chain1.N; ++i)
                set_xi(chain1.v, i, {0.0, 0.0});

            for (int i = 0; i < chain2.N; ++i)
                set_xi(chain2.v, i, {0.0, 0.0});

            for (int i = 0; i < chain3.N; ++i)
                set_xi(chain3.v, i, {0.0, 0.0});

            for (int i = 0; i < ground.N; ++i)
                set_xi(ground.v, i, {0.0, 0.0});

            chains.push_back(chain1);
            chains.push_back(chain2);
            chains.push_back(chain3);
            chains.push_back(ground);
        }
        // Example 3: an upper free chain falls onto a lower pinned chain
        else if (example_type == ExampleType::Example3) {
            total_frame = 60;

            Chain lower = chain_model::make_chain({-1.2, 0.2}, {1.2, 0.0}, number_of_nodes, 0.05);
            Chain upper = chain_model::make_chain({-0.4, 1.8}, {1.6, 1.0}, number_of_nodes, 0.05);

            // Pin both ends of the lower chain so it acts like a suspended obstacle
            lower.is_pinned[0] = 1;
            lower.is_pinned[lower.N - 1] = 1;

            set_xi(lower.xpin, 0, get_xi(lower.x, 0));
            set_xi(lower.xpin, lower.N - 1, get_xi(lower.x, lower.N - 1));

            // Initial velocities
            for (int i = 0; i < lower.N; ++i)
                set_xi(lower.v, i, {0.0, 0.0});

            for (int i = 0; i < upper.N; ++i)
                set_xi(upper.v, i, {0.0, 0.0});

            chains.push_back(lower);
            chains.push_back(upper);
        }

        // Global indexing data
        const int nblocks = static_cast<int>(chains.size());

        std::vector<int> offsets(nblocks, 0);
        for (int b = 1; b < nblocks; ++b)
            offsets[b] = offsets[b - 1] + chains[b - 1].N;

        int total_nodes = 0;
        for (const auto& c : chains)
            total_nodes += c.N;

        // Global valid-segment array
        std::vector<char> segment_valid(std::max(0, total_nodes - 1), 0);
        for (int b = 0; b < nblocks; ++b) {
            for (int i = 0; i + 1 < chains[b].N; ++i)
                segment_valid[offsets[b] + i] = 1;
        }

        // Combined edges
        std::vector<std::pair<int, int>> edges_combined;
        for (int b = 0; b < nblocks; ++b) {
            for (const auto& e : chains[b].edges)
                edges_combined.emplace_back(e.first + offsets[b], e.second + offsets[b]);
        }

        // Global state
        Vec x_combined(2 * total_nodes, 0.0);
        Vec v_combined(2 * total_nodes, 0.0);

        // Per-block unknowns
        std::vector<Vec> xnew_blocks(nblocks);
        for (int b = 0; b < nblocks; ++b)
            xnew_blocks[b] = chains[b].x;

        auto make_guess_blocks = [&]() {
            std::vector<BlockRef> guess_blocks;
            guess_blocks.reserve(nblocks);
            for (int b = 0; b < nblocks; ++b)
                guess_blocks.push_back({&chains[b], &xnew_blocks[b], offsets[b]});
            return guess_blocks;
        };

        auto make_solver_blocks = [&]() {
            std::vector<BlockView> blocks;
            blocks.reserve(nblocks);
            for (int b = 0; b < nblocks; ++b) {
                blocks.push_back({&xnew_blocks[b], &chains[b].xhat, &chains[b].xpin, &chains[b].mass,
                                  &chains[b].rest_lengths, &chains[b].is_pinned, offsets[b]});
            }
            return blocks;
        };

        // Initial export
        {
            std::vector<BlockRef> guess_blocks = make_guess_blocks();
            build_x_combined_from_current_positions(x_combined, guess_blocks);
            export_frame(outdir, 1, x_combined, edges_combined);
        }

        double max_global_residual = 0.0;
        int sum_global_iters_used = 0;

        // Time stepping
        for (int frame = 2; frame <= total_frame + 1; ++frame) {

            // Linear extrapolation
            for (int b = 0; b < nblocks; ++b)
                build_xhat(chains[b], dt);

            // Initial guess blocks
            std::vector<BlockRef> guess_blocks = make_guess_blocks();

            // Initial guess
            initial_guess::apply(initial_guess_type, guess_blocks,x_combined, v_combined, segment_valid, dt, dhat, eta);

            // Solver blocks
            std::vector<BlockView> blocks = make_solver_blocks();

            // Nonlinear GS solve
            std::vector<double> res_hist;
            auto result = global_gauss_seidel_solver(blocks, x_combined, v_combined,
                                                     segment_valid, dt, k_spring, g_accel, dhat,
                                                     max_global_iters, tol_abs, eta, filtering_step_policy, &res_hist);

            double global_residual = result.first;
            int iters_used = result.second;

            max_global_residual = std::max(max_global_residual, global_residual);
            sum_global_iters_used += iters_used;

            // Velocity update
            for (int b = 0; b < nblocks; ++b)
                update_velocity(chains[b], xnew_blocks[b], dt);

            // Export
            build_x_combined_from_current_positions(x_combined, guess_blocks);
            export_frame(outdir, frame, x_combined, edges_combined);

            std::cout << "Frame " << std::setw(4) << frame
                      << " | initial_residual=" << std::scientific << res_hist.front()
                      << " | final_residual="   << std::scientific << global_residual
                      << " | global_iters="     << std::setw(3) << iters_used
                      << '\n';
        }

        auto t_end = clock::now();
        std::chrono::duration<double> elapsed = t_end - t_start;

        double avg_global_iters_used = 1.0 * sum_global_iters_used / total_frame;

        std::cout << "\n===== Simulation Summary =====\n";
        std::cout << "max_global_residual = " << std::scientific << max_global_residual << "\n";
        std::cout << "avg_global_iters = " << std::fixed << avg_global_iters_used << "\n";
        std::cout << "total runtime = " << elapsed.count() << " seconds\n";
        std::cout << "ccd_total_tests = " << solver::ccd_total_tests << "\n";
        std::cout << "ccd_total_collisions = " << solver::ccd_total_collisions << "\n";

        return 0;
    }
}

int main() {
    return simulation::sim();
}
