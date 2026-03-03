#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>

// ======================================================
// Prelim stuff
// ======================================================

typedef std::vector<double> Vec;
namespace fs = std::__fs::filesystem;
using namespace std;

struct Vec2 {
    double x, y;
};

struct Mat2 {
    double a11, a12, a21, a22;
};

// rank-3 tensor of dM/ dx, stored as two 2x2 matrices: [dM/dx_x, dM/dx_y]
struct Mat2x2x2 {
    Mat2 dMx, dMy;
};

Vec2 getXi(const Vec &x, int i) {
    return {x[2*i], x[2*i+1]};
}

void setXi(Vec &x, int i, const Vec2 &v) {
    x[2*i] = v.x; x[2*i+1] = v.y;
}

double norm2(const Vec &a, int i, int j) {
    double dx = a[2*i]   - a[2*j];
    double dy = a[2*i+1] - a[2*j+1];
    return dx*dx + dy*dy;
}

double norm(const Vec &a, int i, int j){
    return std::sqrt(norm2(a,i,j));
}

// Vector multiplication
static inline Vec2 mul(const Mat2& A, const Vec2& v){
    return { A.a11*v.x + A.a12*v.y, A.a21*v.x + A.a22*v.y };
}

// Matrix-matrix multiplication
static inline Mat2 mul(const Mat2& A, const Mat2& B){
    return {
            A.a11*B.a11 + A.a12*B.a21,  A.a11*B.a12 + A.a12*B.a22,
            A.a21*B.a11 + A.a22*B.a21,  A.a21*B.a12 + A.a22*B.a22
    };
}

// Add two matrices
static inline Mat2 add(const Mat2& A, const Mat2& B){
    return { A.a11+B.a11, A.a12+B.a12, A.a21+B.a21, A.a22+B.a22 };
}

// Subtract two matrices
static inline Mat2 sub(const Mat2& A, const Mat2& B){
    return { A.a11-B.a11, A.a12-B.a12, A.a21-B.a21, A.a22-B.a22 };
}

// Subtract two vectors
static inline Vec2 sub(const Vec2& a, const Vec2& b) {
    return {a.x - b.x, a.y - b.y};
}

// Scale matrix
static inline Mat2 scale(const Mat2& A, double s){
    return { s*A.a11, s*A.a12, s*A.a21, s*A.a22 };
}

// Scale vector
static inline Vec2 scale(const Vec2& v, double s){
    return { s*v.x, s*v.y };
}

static inline Mat2 outer(const Vec2& a, const Vec2& b){
    return { a.x*b.x, a.x*b.y, a.y*b.x, a.y*b.y };
}

// Frobenius norm of a 2x2 matrix
static inline double frobNorm(const Mat2& A) {
    return std::sqrt(A.a11*A.a11 + A.a12*A.a12 + A.a21*A.a21 + A.a22*A.a22);
}

// Transpose
static inline Mat2 transpose(const Mat2& M) {
    return {M.a11, M.a21, M.a12, M.a22};
}

// ======================================================
// Spring energy
// ======================================================
// Local spring energy
double springEnergy(const Vec &x, int i, int j, double k, double L) {
    double ell = norm(x,i,j);
    return 0.5 * k / L * (ell - L) * (ell - L);
}

// Total spring energy
double totalSpringEnergy(const Vec &x, double k, const std::vector<double> &L) {
    return springEnergy(x,0,1,k,L[0]) +
           springEnergy(x,1,2,k,L[1]) +
           springEnergy(x,2,3,k,L[2]);
}

// ======================================================
// Local Gradient of spring energy
// ======================================================

// Analytic local spring gradient at node i
Vec2 localSpringGrad(int i, const Vec &x, double k, const std::vector<double> &L) {
    Vec2 g_i{0.0, 0.0};

    std::function<void(int,int,double)> contrib = [&](int a, int b, double Lref) {
        Vec2 xa = getXi(x,a), xb = getXi(x,b);
        double dx = xb.x - xa.x, dy = xb.y - xa.y;
        double ell = std::sqrt(dx*dx + dy*dy);
        if (ell < 1e-12) return;

        double coeff = k/Lref * (ell - Lref)/ell;
        double sgn = (i == b) ? +1.0 : -1.0;

        g_i.x += sgn * coeff * dx;
        g_i.y += sgn * coeff * dy;
    };

    int N = (int)L.size() + 1;
    if (i-1 >= 0)   contrib(i-1,i,L[i-1]);
    if (i+1 <= N-1) contrib(i,i+1,L[i]);

    return g_i;
}

// ======================================================
// Local Hessian of spring energy
// ======================================================

// Analytic local spring Hessian (2x2 block at node i)
Mat2 localSpringHess(int i, const Vec &x, double k, const std::vector<double> &L) {
    Mat2 H_ii{0.0, 0.0, 0.0, 0.0};

    std::function<void(int,int,double)> contrib = [&](int a, int b, double Lref) {
        Vec2 xa = getXi(x,a), xb = getXi(x,b);
        double dx = xb.x - xa.x, dy = xb.y - xa.y;
        double ell = std::sqrt(dx*dx + dy*dy);
        if (ell < 1e-12) return;

        // Derivatives from d/dx of k/L * (ell - L) * (dx/ell, dy/ell)
        // Arrange as: coeff1 * I + coeff2 * [dx;dy][dx dy]
        double coeff1 = k/Lref * (ell - Lref)/ell;          // from (ell-L)/ell term
        double coeff2 = k/Lref * (Lref) / (ell*ell*ell);    // from derivative of ell

        // K_j = [Kxx, Kxy; Kxy; Kyy]
        double Kxx = coeff1 + coeff2*dx*dx;
        double Kyy = coeff1 + coeff2*dy*dy;
        double Kxy = coeff2*dx*dy;

        H_ii.a11 += Kxx; H_ii.a12 += Kxy;
        H_ii.a21 += Kxy; H_ii.a22 += Kyy;
    };

    int N = (int)L.size() + 1;
    if (i-1 >= 0)   contrib(i-1,i,L[i-1]);
    if (i+1 <= N-1) contrib(i,i+1,L[i]);

    return H_ii;
}

// ======================================================
// Barrier term
// ======================================================
// Scalar barrier energy
double barrierEnergy(double d, double dhat) {
    if (d >= dhat) return 0.0;
    return - (d - dhat) * (d - dhat) * std::log(d / dhat);
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
// Compute (signed) point–segment distance
// ======================================================
double nodeSegmentDistance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1, double &t, Vec2 &p, Vec2 &r){
    // Segment direction
    Vec2 seg = { xjp1.x - xj.x, xjp1.y - xj.y };
    double seg_len2 = seg.x * seg.x + seg.y * seg.y;

    // Handle degenerate segment
    if (seg_len2 < 1e-14) {
        t = 0.0;
        p = xj;
        r = { xi.x - p.x, xi.y - p.y };
        return std::sqrt(r.x * r.x + r.y * r.y);
    }

    // Project point onto segment
    Vec2 q = { xi.x - xj.x, xi.y - xj.y };
    double dot = q.x * seg.x + q.y * seg.y;
    t = dot / seg_len2;

    // Clamp to segment
    t = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);

    // Closest point
    p = { xj.x + t * seg.x, xj.y + t * seg.y };

    // Vector and distance
    r = { xi.x - p.x, xi.y - p.y };
    return std::sqrt(r.x * r.x + r.y * r.y);
}

// ======================================================
// LocalBarrierGrad for a node
// ======================================================
Vec2 localBarrierGrad(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
    Vec2 xi = getXi(x, node);
    Vec2 x1 = getXi(x, seg0);
    Vec2 x2 = getXi(x, seg1);

    // Define v = x - x1
    Vec2 v = {xi.x - x1.x, xi.y - x1.y};

    double t; Vec2 p{}, r{};
    double d = nodeSegmentDistance(xi, x1, x2, t, p, r);
    if (d >= dhat) return {0,0};
    d = std::max(d, 1e-12);

    // Segment geometry
    Vec2 s{ x2.x - x1.x, x2.y - x1.y };
    double L = std::sqrt(s.x*s.x + s.y*s.y);
    if (L < 1e-12) return {0,0};
    Vec2 u{ s.x/L, s.y/L };

    // Projector P = I - uu^T
    Mat2 P{1 - u.x*u.x, -u.x*u.y,
           -u.x*u.y, 1 - u.y*u.y};

    // Barrier quantities
    Vec2 n{ r.x/d, r.y/d };              // unit normal
    double bp = barrierGrad(d, dhat);    //b'(d)
    Vec2 f{ bp*n.x, bp*n.y };            // f = b'(d) * n
    double u_dot_v = u.x*v.x + u.y*v.y;

    // Handle endpoint branches (point–point)
    if (t <= 1e-6) {
        // Closest to x1
        if (who == node) return { bp*n.x, bp*n.y }; // n = (xi-x1)/d
        if (who == seg0) return { -bp*n.x, -bp*n.y };
        return {0,0};
    }
    if (t >= 1.0 - 1e-6) {
        // Closest to x2
        if (who == node) return { bp*n.x, bp*n.y }; // n = (xi-x2)/d
        if (who == seg1) return { -bp*n.x, -bp*n.y };
        return {0,0};
    }

    // Interior branch
    if (who == node) {
        return mul(P, f);
    }

    // Build the term [ (u^T v)I + v u^T - 2 (u^T v) u u^T ] / L
    // This is T_common_T = (T/L)^T = J1^T + P
    Mat2 vuT = { v.x*u.x, v.x*u.y, v.y*u.x, v.y*u.y }; // outer(v, u) = v u^T
    Mat2 uuT = { u.x*u.x, u.x*u.y, u.y*u.x, u.y*u.y }; // outer(u, u)
    Mat2 term_T_T = add(add(scale(Mat2{1,0,0,1}, u_dot_v), vuT), scale(uuT, -2*u_dot_v)); // This is T^T
    Mat2 T_common_T = scale(term_T_T, 1.0/L); // This is (T/L)^T

    if (who == seg0) {
        // g1 = J1^T f = ( (T/L)^T - P^T ) f
        Mat2 J1_T = add(T_common_T, scale(P, -1.0)); // P is symmetric
        return mul(J1_T, f); // Returns J1^T * f
    } else if (who == seg1) {
        // g2 = J2^T f = ( -(T/L) )^T f = - (T/L)^T f
        Mat2 J2_T = scale(T_common_T, -1.0);
        return mul(J2_T, f); // Returns J2^T * f
    }

    return {0,0};
}


// ======================================================
// Analytic derivatives for Tensor Term
// ======================================================
// Gets v = r_p = x - x1
Vec2 getV(const Vec& x, int node, int seg0) {
    Vec2 xi = getXi(x, node); Vec2 x1 = getXi(x, seg0);
    return sub(xi, x1);
}
// Gets theta = u^T v
double getTheta(const Vec2& u, const Vec2& v) {
    return u.x*v.x + u.y*v.y;
}

// The gradient of projection matrix P w.r.t. x1 (but this one return -grad(P))
Mat2x2x2 getGradJP_analytic(const Vec2& u, const Mat2& P, double L) {
    Mat2x2x2 gradP{};
    // dP_ik / dx1_l = (1/L) * (P_il u_k + u_i P_kl)

    // dP/dx1_x (l=0)
    double P_il_x = P.a11, P_il_y = P.a21; // P_i0 (col 0 of P)
    gradP.dMx.a11 = (1.0/L) * (P_il_x * u.x + u.x * P_il_x);
    gradP.dMx.a12 = (1.0/L) * (P_il_x * u.y + u.x * P.a12);
    gradP.dMx.a21 = (1.0/L) * (P_il_y * u.x + u.y * P_il_x);
    gradP.dMx.a22 = (1.0/L) * (P_il_y * u.y + u.y * P.a12);

    // dP/dx1_y (l=1)
    double P_il_x_y = P.a12, P_il_y_y = P.a22; // P_i1 (col 1 of P)
    gradP.dMy.a11 = (1.0/L) * (P_il_x_y * u.x + u.x * P_il_x_y);
    gradP.dMy.a12 = (1.0/L) * (P_il_x_y * u.y + u.x * P_il_y_y);
    gradP.dMy.a21 = (1.0/L) * (P_il_y_y * u.x + u.y * P_il_x_y);
    gradP.dMy.a22 = (1.0/L) * (P_il_y_y * u.y + u.y * P_il_y_y);

    // return -grad(P)
    return {scale(gradP.dMx, -1.0), scale(gradP.dMy, -1.0)};
}

// grad(J_common) w.r.t. x1
// This implements Term A + Term B from the note
Mat2x2x2 getGradJCommon_analytic(double L, const Vec2& u, const Vec2& v, double theta, const Mat2& P, const Vec2& r){
    Mat2 I = {1,0,0,1};
    Mat2 T = add(add(scale(I, theta), outer(u, v)), scale(outer(u, u), -2*theta)); // T = J_common * L

    // Term A: d(1/L)/dx1_l * T ---
    // d(1/L)/dx1_l = u_l / L^2
    Mat2 dJ_dx = scale(T, u.x / (L*L)); // (l=0)
    Mat2 dJ_dy = scale(T, u.y / (L*L)); // (l=1)

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
    double dtheta_l_x = -r.x/L - u.x;      // d(theta)/dx1_x
    Vec2 dv_l_x = {-1, 0};                 // d(v)/dx1_x
    Vec2 du_l_x = {-P.a11/L, -P.a21/L};     // d(u)/dx1_x (col 0 of -P/L)

    Mat2 dT_l = {0,0,0,0};
    dT_l = add(dT_l, scale(I, dtheta_l_x));
    dT_l = add(dT_l, outer(du_l_x, v));
    dT_l = add(dT_l, outer(u, dv_l_x));
    dT_l = add(dT_l, scale(outer(u,u), -2.0*dtheta_l_x));
    dT_l = add(dT_l, scale(outer(du_l_x, u), -2.0*theta));
    dT_l = add(dT_l, scale(outer(u, du_l_x), -2.0*theta));

    dJ_dx = add(dJ_dx, scale(dT_l, 1.0/L)); // Add Term B (l=0)

    // Build Term B for l=1 (y-component)
    double dtheta_l_y = -r.y/L - u.y;     // d(theta)/dx1_y
    Vec2 dv_l_y = {0, -1};                 // d(v)/dx1_y
    Vec2 du_l_y = {-P.a12/L, -P.a22/L};     // d(u)/dx1_y (col 1 of -P/L)

    dT_l = {0,0,0,0}; // Reset
    dT_l = add(dT_l, scale(I, dtheta_l_y));
    dT_l = add(dT_l, outer(du_l_y, v));
    dT_l = add(dT_l, outer(u, dv_l_y));
    dT_l = add(dT_l, scale(outer(u,u), -2.0*dtheta_l_y));
    dT_l = add(dT_l, scale(outer(du_l_y, u), -2.0*theta));
    dT_l = add(dT_l, scale(outer(u, du_l_y), -2.0*theta));

    dJ_dy = add(dJ_dy, scale(dT_l, 1.0/L)); // Add Term B (l=1)

    return {dJ_dx, dJ_dy};
}

// grad(J_common) w.r.t. x2
Mat2x2x2 getGradJCommon_analytic_x2(double L, const Vec2& u, const Vec2& v, double theta, const Mat2& P, const Vec2& r){
    Mat2 I = {1,0,0,1};
    Mat2 T = add(add(scale(I, theta), outer(u, v)), scale(outer(u, u), -2*theta)); // T = J_common * L

    // Term A: d(1/L)/dx2_l * T
    // d(1/L)/dx2_l = -u_l / L^2
    Mat2 dJ_dx = scale(T, -u.x / (L*L)); // (l=0)
    Mat2 dJ_dy = scale(T, -u.y / (L*L)); // (l=1)

    // Term B: (1/L) * dT/dx2_l
    // Derivatives we need (w.r.t. x2_l)
    // d(v)/dx2_l = 0
    // d(u)/dx2_l = P_l / L (col l of P/L)
    // d(theta)/dx2_l = (d(u)/dx2_l)^T * v = (P_l/L)^T * v = (1/L) * (P v)_l = r_l / L

    // Build Term B for l=0 (x-component)
    double dtheta_l_x = r.x/L;             // d(theta)/dx2_x
    Vec2 du_l_x = {P.a11/L, P.a21/L};       // d(u)/dx2_x (col 0 of P/L)

    Mat2 dT_l = {0,0,0,0};
    dT_l = add(dT_l, scale(I, dtheta_l_x));
    dT_l = add(dT_l, outer(du_l_x, v));
    dT_l = add(dT_l, scale(outer(u,u), -2.0*dtheta_l_x));
    dT_l = add(dT_l, scale(outer(du_l_x, u), -2.0*theta));
    dT_l = add(dT_l, scale(outer(u, du_l_x), -2.0*theta));

    dJ_dx = add(dJ_dx, scale(dT_l, 1.0/L)); // Add Term B (l=0)

    // Build Term B for l=1 (y-component)
    double dtheta_l_y = r.y/L;             // d(theta)/dx2_y
    Vec2 du_l_y = {P.a12/L, P.a22/L};       // d(u)/dx2_y (col 1 of P/L)

    dT_l = {0,0,0,0}; // Reset
    dT_l = add(dT_l, scale(I, dtheta_l_y));
    dT_l = add(dT_l, outer(du_l_y, v));
    // dT_l = add(dT_l, outer(u, d(v)_l)); // 0
    dT_l = add(dT_l, scale(outer(u,u), -2.0*dtheta_l_y));
    dT_l = add(dT_l, scale(outer(du_l_y, u), -2.0*theta));
    dT_l = add(dT_l, scale(outer(u, du_l_y), -2.0*theta));

    dJ_dy = add(dJ_dy, scale(dT_l, 1.0/L)); // Add Term B (l=1)

    return {dJ_dx, dJ_dy};
}


// ==============================
// LocalBarrierHess for a node
// ==============================
Mat2 localBarrierHess(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
    Vec2 xi   = getXi(x, node);
    Vec2 x1   = getXi(x, seg0);
    Vec2 x2   = getXi(x, seg1);

    double t; Vec2 p{}, r_vec{}; // r_vec is the residual r = P(xi-x1)
    double d = nodeSegmentDistance(xi, x1, x2, t, p, r_vec);
    if (d >= dhat) return {0,0,0,0};
    d = std::max(d, 1e-12);

    // Barrier quantities
    Vec2 n{ r_vec.x/d, r_vec.y/d };
    double bp  = barrierGrad(d, dhat);
    double bpp = barrierHess(d, dhat);

    // Inner Hessian wrt residual
    Mat2 Hrr{
            bpp*n.x*n.x + (bp/d)*(1 - n.x*n.x),
            (bpp - bp/d)*n.x*n.y,
            (bpp - bp/d)*n.x*n.y,
            bpp*n.y*n.y + (bp/d)*(1 - n.y*n.y)
    };

    // ======================================================
    // Endpoint branches
    // ======================================================
    if (t <= 1e-6) {
        // Closest to x1 (point–point)
        if (who == node or who == seg0)      return Hrr;
        else return {0,0,0,0};
    }

    if (t >= 1.0 - 1e-6) {
        // Closest to x2 (point–point)
        if (who == node or who == seg1)      return Hrr;
        else return {0,0,0,0};
    }

    // ======================================================
    // Interior branch
    // ======================================================

    // Geometry
    Vec2 s{ x2.x - x1.x, x2.y - x1.y };
    double L = std::sqrt(s.x*s.x + s.y*s.y);
    if (L < 1e-12) return {0,0,0,0};
    Vec2 u{ s.x/L, s.y/L };

    // Projection matrix
    Mat2 P{1 - u.x*u.x, -u.x*u.y,
           -u.x*u.y, 1 - u.y*u.y};

    // Shared quantities
    Vec2 v = getV(x, node, seg0);      // v = r_p = xi - x1
    double theta = getTheta(u, v);     // theta = u^T v
    Vec2 f{ bp*n.x, bp*n.y };

    // Base term T_common = [(u^T v)I + u v^T - 2(u^T v)u u^T]/L
    Mat2 I{1,0,0,1};
    Mat2 uvT{ u.x*v.x, u.x*v.y, u.y*v.x, u.y*v.y }; // u v^T
    Mat2 uuT{ u.x*u.x, u.x*u.y, u.y*u.x, u.y*u.y };
    Mat2 term = add(add(scale(I, theta), uvT), scale(uuT, -2*theta));
    Mat2 T_common = scale(term, 1.0/L);

    // Jacobians
    Mat2 Jx  = P;
    Mat2 Jx1 = add(T_common, scale(P, -1.0));
    Mat2 Jx2 = scale(T_common, -1.0);

    // Helper: M^T * Hrr * M
    auto hessTerm = [&](const Mat2& M) {
        return mul(Mat2{M.a11, M.a21, M.a12, M.a22}, mul(Hrr, M)); // M^T * Hrr * M
    };

    // Base Hessians (J^T Hrr J)
    Mat2 Hnode = hessTerm(Jx);
    Mat2 H1    = hessTerm(Jx1);
    Mat2 H2    = hessTerm(Jx2);

    // ======================================================
    // TENSOR TERM CALCULATION
    // ======================================================
    Mat2 K1{};
    Mat2 K2{};

    // K1 Term
    // K1 = [d(J1^T)/dx1] * f = [d(J_common^T)/dx1 + d(J_P^T)/dx1] * f
    // (K1)_il = sum_k [ d(J_common)_ki/dx1_l + d(J_P)_ki/dx1_l ] * f_k
    {
        // Get analytic 3-tensors T_ikl = d(J_ik)/dx1_l
        Mat2x2x2 gradJC1 = getGradJCommon_analytic(L, u, v, theta, P, r_vec);
        Mat2x2x2 gradJP1 = getGradJP_analytic(u, P, L);

        // Get tensor slices for d(J1)/dx1_l = d(J_common)/dx1_l + d(J_P)/dx1_l
        Mat2 dJ1_dx = add(gradJC1.dMx, gradJP1.dMx);
        Mat2 dJ1_dy = add(gradJC1.dMy, gradJP1.dMy);

        // Contract: (K1)_il = sum_k d(J1)_ki/dx1_l * f_k
        // Col 1 (l=0): (K1)_i0 = sum_k d(J1)_ki/dx1_x * f_k = (dJ1_dx^T) * f
        Vec2 col1_K1 = mul(transpose(dJ1_dx), f);
        // Col 2 (l=1): (K1)_i1 = sum_k d(J1)_ki/dx1_y * f_k = (dJ1_dy^T) * f
        Vec2 col2_K1 = mul(transpose(dJ1_dy), f);

        K1 = {col1_K1.x, col2_K1.x, col1_K1.y, col2_K1.y};
    }

    // K2 Term
    // K2 = [d(J2^T)/dx2] * f = [d(-J_common^T)/dx2] * f
    // (K2)_il = sum_k [ -d(J_common)_ki/dx2_l ] * f_k
    {
        // Get analytic 3-tensor T_ikl = d(J_common_ik)/dx2_l
        Mat2x2x2 gradJC2 = getGradJCommon_analytic_x2(L, u, v, theta, P, r_vec);

        // Get tensor slices for d(J2)/dx2_l = -d(J_common)/dx2_l
        Mat2 dJ2_dx = scale(gradJC2.dMx, -1.0);
        Mat2 dJ2_dy = scale(gradJC2.dMy, -1.0);

        // Contract: (K2)_il = sum_k d(J2)_ki/dx2_l * f_k
        // Col 1 (l=0): (K2)_i0 = sum_k d(J2)_ki/dx2_x * f_k = (dJ2_dx^T) * f
        Vec2 col1_K2 = mul(transpose(dJ2_dx), f);
        // Col 2 (l=1): (K2)_i1 = sum_k d(J2)_ki/dx2_y * f_k = (dJ2_dy^T) * f
        Vec2 col2_K2 = mul(transpose(dJ2_dy), f);

        K2 = {col1_K2.x, col2_K2.x, col1_K2.y, col2_K2.y};
    }

    // Return appropriate block
    if (who == node)      return Hnode;
    else if (who == seg0) return add(H1, K1);
    else if (who == seg1) return add(H2, K2);
    else                  return {0,0,0,0};
}

// ======================================================
// The test sections starts here
// ======================================================
// Spring finite difference tests
std::vector<double> numericSpringGrad(Vec x, double k, const std::vector<double>& L, double h){
    int n = x.size();
    std::vector<double> g(n, 0.0);

    auto diffOne = [&](int idx)->double {
        double orig = x[idx];
        x[idx] = orig + h;
        double Ep = totalSpringEnergy(x,k,L);
        x[idx] = orig - h;
        double Em = totalSpringEnergy(x,k,L);
        x[idx] = orig;
        return (Ep - Em) / (2.0*h);
    };

    for(int i=0;i<n;i++) g[i] = diffOne(i);
    return g;
}

Mat2 numericSpringHess_block(int i, Vec x, double k, const std::vector<double>& L, double h){
    int idx_x = 2*i;
    int idx_y = 2*i + 1;
    double ox = x[idx_x], oy = x[idx_y];

    auto localG = [&](int node)->Vec2{
        return localSpringGrad(node,x,k,L);
    };

    // d/dx
    x[idx_x] = ox + h;
    Vec2 gpx = localG(i);
    x[idx_x] = ox - h;
    Vec2 gmx = localG(i);
    x[idx_x] = ox;

    Vec2 col1 = scale(sub(gpx,gmx), 1.0/(2*h));

    // d/dy
    x[idx_y] = oy + h;
    Vec2 gpy = localG(i);
    x[idx_y] = oy - h;
    Vec2 gmy = localG(i);
    x[idx_y] = oy;

    Vec2 col2 = scale(sub(gpy,gmy), 1.0/(2*h));

    return {col1.x, col2.x, col1.y, col2.y};
}

void runSpringGradFDTest(const char* title, Vec x, double k, const std::vector<double>& L){
    cout << "\n== " << title << " (SPRING GRADIENT TEST) ==" << endl;

    double E0 = totalSpringEnergy(x,k,L);
    cout << "Base spring energy E0 = " << std::setprecision(12) << E0 << endl;

    // analytic gradient
    vector<double> ga(x.size(),0.0);
    int N = (int)L.size() + 1;
    for(int i = 0; i < N; i++){
        Vec2 g = localSpringGrad(i, x, k, L);
        ga[2*i]   = g.x;
        ga[2*i+1] = g.y;
    }

    vector<double> hs{1e-2,1e-3,1e-4,1e-5};
    cout << left << setw(12)<<"h"<<setw(18)<<"||err||_2"<<setw(12)<<"rate"<<endl;

    double prevErr=-1, prevH=-1;
    for(double h:hs){
        auto gn = numericSpringGrad(x,k,L,h);
        double err=0;
        for(size_t i=0;i<gn.size();i++) err += (gn[i]-ga[i])*(gn[i]-ga[i]);
        err = sqrt(err);
        double rate = (prevErr>0)? log(prevErr/err)/log(prevH/h) : NAN;

        cout<<scientific<<setprecision(2)
            <<setw(12)<<h<<setw(18)<<err<<setw(12)<<rate<<endl;

        prevErr=err; prevH=h;
    }
}

void runSpringHessianFDTest(const char* title, Vec x, double k, const std::vector<double>& L){
    cout << "\n== " << title << " (SPRING HESSIAN TEST) ==" << endl;

    Mat2 Ha0 = localSpringHess(0,x,k,L);
    Mat2 Ha1 = localSpringHess(1,x,k,L);
    Mat2 Ha2 = localSpringHess(2,x,k,L);

    vector<double> hs{1e-2,1e-3,1e-4,1e-5};
    cout<<left
        <<setw(12)<<"h"
        <<setw(12)<<"err0"<<setw(10)<<"rate"
        <<setw(12)<<"err1"<<setw(10)<<"rate"
        <<setw(12)<<"err2"<<setw(10)<<"rate"<<endl;

    double p0=-1,p1=-1,p2=-1,ph=-1;
    for(double h:hs){
        Mat2 Hn0 = numericSpringHess_block(0,x,k,L,h);
        Mat2 Hn1 = numericSpringHess_block(1,x,k,L,h);
        Mat2 Hn2 = numericSpringHess_block(2,x,k,L,h);

        auto e0=frobNorm(sub(Ha0,Hn0));
        auto e1=frobNorm(sub(Ha1,Hn1));
        auto e2=frobNorm(sub(Ha2,Hn2));

        double r0 = (p0>0)? log(p0/e0)/log(ph/h) : NAN;
        double r1 = (p1>0)? log(p1/e1)/log(ph/h) : NAN;
        double r2 = (p2>0)? log(p2/e2)/log(ph/h) : NAN;

        cout<<scientific<<setprecision(2)
            <<setw(12)<<h
            <<setw(12)<<e0<<setw(10)<<r0
            <<setw(12)<<e1<<setw(10)<<r1
            <<setw(12)<<e2<<setw(10)<<r2<<endl;

        p0=e0;p1=e1;p2=e2;ph=h;
    }
}


// Local barrier energy: b(d; dhat) with the same branching via nodeSegmentDistance
double localBarrierEnergy(const Vec &x, int node, int seg0, int seg1, double dhat) {
    Vec2 xi = getXi(x, node);
    Vec2 x1 = getXi(x, seg0);
    Vec2 x2 = getXi(x, seg1);

    double t; Vec2 p{}, r{};
    double d = nodeSegmentDistance(xi, x1, x2, t, p, r);
    if (d >= dhat) return 0.0;
    d = std::max(d, 1e-12);
    return barrierEnergy(d, dhat);
}

// Assemble analytic gradient (6 entries: xi, x1, x2)
std::array<double,6> analyticGrad(const Vec &x, int node, int seg0, int seg1, double dhat){
    std::array<double,6> g{0,0,0,0,0,0};
    // xi
    Vec2 gi  = localBarrierGrad(node, x, node, seg0, seg1, dhat);
    g[0] = gi.x; g[1] = gi.y;
    // x1
    Vec2 g1  = localBarrierGrad(seg0, x, node, seg0, seg1, dhat);
    g[2] = g1.x; g[3] = g1.y;
    // x2
    Vec2 g2  = localBarrierGrad(seg1, x, node, seg0, seg1, dhat);
    g[4] = g2.x; g[5] = g2.y;
    return g;
}

// Numerical gradient via central difference on the 6 DOFs we care about
std::array<double,6> numericGrad(Vec x, int node, int seg0, int seg1, double dhat, double h){
    std::array<double,6> g{0,0,0,0,0,0};

    auto diffOne = [&](int who, int comp)->double {
        int idx = 2*who + comp;
        double orig = x[idx];
        x[idx] = orig + h;
        double Ep = localBarrierEnergy(x, node, seg0, seg1, dhat);
        x[idx] = orig - h;
        double Em = localBarrierEnergy(x, node, seg0, seg1, dhat);
        x[idx] = orig; // restore
        return (Ep - Em) / (2.0*h);
    };

    // xi
    g[0] = diffOne(node, 0);
    g[1] = diffOne(node, 1);
    // x1
    g[2] = diffOne(seg0, 0);
    g[3] = diffOne(seg0, 1);
    // x2
    g[4] = diffOne(seg1, 0);
    g[5] = diffOne(seg1, 1);

    return g;
}

// Numerical Hessian for a single diagonal block (e.g., d(g_who) / d(x_who))
Mat2 numericHessian(int who, Vec x, int node, int seg0, int seg1, double dhat, double h) {
    // 'who' is the node we are differentiating *with respect to* (e.g., seg0)
    // 'who_grad' is the node whose gradient we are observing (which is the same 'who')
    int who_grad = who;

    int idx_x = 2 * who;     // Differentiate w.r.t. x-component
    int idx_y = 2 * who + 1; // Differentiate w.r.t. y-component

    double orig_x = x[idx_x];
    double orig_y = x[idx_y];

    // Column 1: Differentiate g_who w.r.t. x_who_x
    x[idx_x] = orig_x + h;
    Vec2 g_p_x = localBarrierGrad(who_grad, x, node, seg0, seg1, dhat);
    x[idx_x] = orig_x - h;
    Vec2 g_m_x = localBarrierGrad(who_grad, x, node, seg0, seg1, dhat);
    x[idx_x] = orig_x; // Restore

    Vec2 col1 = scale(sub(g_p_x, g_m_x), 1.0 / (2.0 * h));

    // Column 2: Differentiate g_who w.r.t. x_who_y
    x[idx_y] = orig_y + h;
    Vec2 g_p_y = localBarrierGrad(who_grad, x, node, seg0, seg1, dhat);
    x[idx_y] = orig_y - h;
    Vec2 g_m_y = localBarrierGrad(who_grad, x, node, seg0, seg1, dhat);
    x[idx_y] = orig_y; // Restore

    Vec2 col2 = scale(sub(g_p_y, g_m_y), 1.0 / (2.0 * h));

    // Assemble Mat2 from columns {a11, a12, a21, a22}
    // col1 = {a11, a21}
    // col2 = {a12, a22}
    return {col1.x, col2.x, col1.y, col2.y};
}


double l2(const std::array<double,6>& a){
    double s=0; for(double v: a) s += v*v; return std::sqrt(s);
}

std::array<double,6> sub(const std::array<double,6>& a, const std::array<double,6>& b){
    std::array<double,6> r{};
    for (int i=0;i<6;++i) r[i] = a[i]-b[i];
    return r;
}

void printVec6(const std::array<double,6>& v){
    cout << std::fixed << std::setprecision(6)
         << "[" << v[0] << ", " << v[1] << " | "
         << v[2] << ", " << v[3] << " | "
         << v[4] << ", " << v[5] << "]";
}

void printMat2(const char* title, const Mat2& M){
    cout << std::scientific << std::setprecision(2);
    cout << title
         << "[[" << M.a11 << ", " << M.a12 << "], ["
         << M.a21 << ", " << M.a22 << "]]" << endl;
}


// Run a single scenario and print FD convergence table
void runGradFDTest(const char* title, Vec x, int node, int seg0, int seg1, double dhat){ // Renamed
    cout << "\n== " << title << " (GRADIENT TEST) ==" << endl;

    // Show base energy to confirm we're in the active region
    double E0 = localBarrierEnergy(x, node, seg0, seg1, dhat);
    cout << "Base energy E0 = " << std::setprecision(12) << E0 << endl;

    // Analytic gradient
    auto ga = analyticGrad(x, node, seg0, seg1, dhat);
    cout << "Analytic grad (xi | x1 | x2) = ";
    printVec6(ga);
    cout << endl;

    // Step sizes
    std::vector<double> hs {1e-2, 1e-3, 1e-4, 1e-5};

    cout << std::left
         << setw(12) << "h"
         << setw(18) << "||err||_2"
         << setw(12) << "rate"
         << endl;

    double prevErr = -1.0, prevH = -1.0;
    for (double h: hs){
        auto gn = numericGrad(x, node, seg0, seg1, dhat, h);
        auto err = sub(gn, ga);
        double e = l2(err);
        double rate = (prevErr > 0.0) ? std::log(prevErr/e)/std::log(prevH/h) : NAN;

        cout << std::scientific << std::setprecision(2)
             << setw(12) << h
             << setw(18) << e
             << setw(12) << rate
             << endl;

        prevErr = e; prevH = h;
    }
}

// Run Hessian FD test for a single scenario
void runHessianFDTest(const char* title, Vec x, int node, int seg0, int seg1, double dhat){
    cout << "\n== " << title << " (HESSIAN TEST) ==" << endl;

    // Analytic Hessians
    Mat2 Ha_node = localBarrierHess(node, x, node, seg0, seg1, dhat);
    Mat2 Ha_seg0 = localBarrierHess(seg0, x, node, seg0, seg1, dhat);
    Mat2 Ha_seg1 = localBarrierHess(seg1, x, node, seg0, seg1, dhat);

    cout << "Analytic Hessians (per-node):" << endl;
    printMat2("  H_xx:   ", Ha_node);
    printMat2("  H_x1x1: ", Ha_seg0);
    printMat2("  H_x2x2: ", Ha_seg1);


    // Step sizes
    std::vector<double> hs {1e-2, 1e-3, 1e-4, 1e-5};

    cout << std::left
         << setw(12) << "h"
         << setw(16) << "err_xx"
         << setw(12) << "rate_xx"
         << setw(16) << "err_x1"
         << setw(12) << "rate_x1"
         << setw(16) << "err_x2"
         << setw(12) << "rate_x2"
         << endl;

    double prevErr_n = -1.0, prevErr_0 = -1.0, prevErr_1 = -1.0;
    double prevH = -1.0;

    for (double h: hs){
        // Numerical Hessians
        Mat2 Hn_node = numericHessian(node, x, node, seg0, seg1, dhat, h);
        Mat2 Hn_seg0 = numericHessian(seg0, x, node, seg0, seg1, dhat, h);
        Mat2 Hn_seg1 = numericHessian(seg1, x, node, seg0, seg1, dhat, h);

        // Errors
        double e_node = frobNorm(sub(Ha_node, Hn_node));
        double e_seg0 = frobNorm(sub(Ha_seg0, Hn_seg0));
        double e_seg1 = frobNorm(sub(Ha_seg1, Hn_seg1));

        // Rates
        double rate_n = (prevErr_n > 0.0) ? std::log(prevErr_n/e_node)/std::log(prevH/h) : NAN;
        double rate_0 = (prevErr_0 > 0.0) ? std::log(prevErr_0/e_seg0)/std::log(prevH/h) : NAN;
        double rate_1 = (prevErr_1 > 0.0) ? std::log(prevErr_1/e_seg1)/std::log(prevH/h) : NAN;

        cout << std::scientific << std::setprecision(2)
             << setw(12) << h
             << setw(16) << e_node
             << setw(12) << rate_n
             << setw(16) << e_seg0
             << setw(12) << rate_0
             << setw(16) << e_seg1
             << setw(12) << rate_1
             << endl;

        prevErr_n = e_node;
        prevErr_0 = e_seg0;
        prevErr_1 = e_seg1;
        prevH = h;
    }
}


int main(){
    // Spring tests: 4 nodes in chain: 0-1-2-3
    Vec xs(8);
    setXi(xs,0,{0,0});
    setXi(xs,1,{1.0,0.0});
    setXi(xs,2,{2.0,0.2});
    setXi(xs,3,{3.0,0});

    double k = 100.0;
    vector<double> L = {1.0,1.0,1.0};

    runSpringGradFDTest("Spring test", xs, k, L);
    runSpringHessianFDTest("Spring test", xs, k, L);
    
    // Global vector x holds all nodes; we only use 3 nodes (node, seg0, seg1)
    // Layout: [x_node, y_node, x_seg0, y_seg0, x_seg1, y_seg1]
    Vec x(6);
    int node = 0, seg0 = 1, seg1 = 2;

    double dhat = 0.2; // pick a barrier radius

    // Interior projection
    // Segment [x1=(0,0) -- x2=(1,0)], node xi near the interior
    setXi(x, seg0, {0.0, 0.0});
    setXi(x, seg1, {1.0, 0.0});
    setXi(x, node, {0.30, 0.05}); // distance ~0.05 < dhat
    runGradFDTest("Interior projection", x, node, seg0, seg1, dhat);
    runHessianFDTest("Interior projection", x, node, seg0, seg1, dhat);


    // Closest to x1 (t <= 0)
    setXi(x, seg0, {0.0, 0.0});
    setXi(x, seg1, {1.0, 0.0});
    setXi(x, node, {-0.10, 0.05}); // projection clamps to x1
    runGradFDTest("Endpoint branch near x1", x, node, seg0, seg1, dhat);
    runHessianFDTest("Endpoint branch near x1", x, node, seg0, seg1, dhat);


    // Closest to x2 (t >= 1)
    setXi(x, seg0, {0.0, 0.0});
    setXi(x, seg1, {1.0, 0.0});
    setXi(x, node, {1.10, 0.05}); // projection clamps to x2
    runGradFDTest("Endpoint branch near x2", x, node, seg0, seg1, dhat);
    runHessianFDTest("Endpoint branch near x2", x, node, seg0, seg1, dhat);

    return 0;
}