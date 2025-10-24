#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

// ======================================================
// Prelim stuff
// ======================================================

typedef std::vector<double> Vec;
namespace fs = std::__fs::filesystem;

struct Vec2 {
    double x, y;
};

struct Mat2 {
    double a11, a12, a21, a22;
};

struct Chain {
    int N{}; // number of nodes
    Vec x; // positions (2*N)
    Vec v; // velocities (2*N)
    Vec xhat;  // predicted positions (2*N)
    std::vector<double> mass; // per-node masses
    std::vector<bool> is_fixed; // per-node fixed flags
    std::vector<double> rest_lengths; // rest spring lengths
    std::vector<std::pair<int,int>> edges; // connectivity list
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
// Local Gradient of the spring energy
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
// Local Hessian of the spring energy
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
// Barrier term (node-segment)
// ======================================================
struct BarrierPair {
    int node;   // i
    int seg0;   // j
    int seg1;   // j+1
};

std::vector<BarrierPair> build_barrier_pairs(int N) {
    std::vector<BarrierPair> pairs;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            if (i == j or i == j + 1) continue; // skip self/adjacent
            pairs.push_back({i, j, j + 1});
        }
    }
    return pairs;
}

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

double nodeSegmentSignedDistance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1){
    double t;
    Vec2 p{}, r{};
    double d = nodeSegmentDistance(xi, xj, xjp1, t, p, r);

    // Segment direction and unit normal
    Vec2 seg = { xjp1.x - xj.x, xjp1.y - xj.y };
    double seg_len = std::sqrt(seg.x * seg.x + seg.y * seg.y);
    Vec2 n = { -seg.y / seg_len, seg.x / seg_len };

    // Sign based on which side xi lies
    double side = n.x * (xi.x - xj.x) + n.y * (xi.y - xj.y);
    return (side >= 0.0 ? d : -d);
}

// ======================================================
// LocalBarrierGrad for a node
// ======================================================
// Build projectors T, P from segment direction s = x_{j+1} - x_j: T = uu^T and P = I - uu^T
static inline void buildProjectors(const Vec2& xj, const Vec2& xk, Mat2& T, Mat2& P){
    Vec2 s{ xk.x - xj.x, xk.y - xj.y };
    double len2 = s.x*s.x + s.y*s.y;
    // Assume non-degenerate
    double inv = 1.0 / len2;
    T = { s.x*s.x*inv, s.x*s.y*inv, s.x*s.y*inv, s.y*s.y*inv };
    P = { 1.0 - T.a11, -T.a12, -T.a21, 1.0 - T.a22 };
}

Vec2 localBarrierGrad(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
    Vec2 xi   = getXi(x, node);
    Vec2 xj   = getXi(x, seg0);
    Vec2 xk   = getXi(x, seg1);

    double t; Vec2 p{}, r{};
    double d = nodeSegmentDistance(xi, xj, xk, t, p, r);
    if (d >= dhat) return {0,0};
    d = std::max(d, 1e-12);

    Vec2 n{ r.x/d, r.y/d };
    double bp = barrierGrad(d, dhat);

    Mat2 T{}, P{};
    buildProjectors(xj, xk, T, P);
    Vec2 g_raw{ bp*n.x, bp*n.y };

    if (who == node) {
        // Gradient of {x_i} = P * (bp * n)
        return mul(P, g_raw);
    } else if (who == seg0 or who == seg1) {
        return {0,0};
    }
    return {0,0};
}

// ==============================
// LocalBarrierHess for a node
// ==============================
Mat2 localBarrierHess(int who, const Vec &x, int node, int seg0, int seg1, double dhat) {
    Vec2 xi   = getXi(x, node);
    Vec2 xj   = getXi(x, seg0);
    Vec2 xk   = getXi(x, seg1);

    double t; Vec2 p{}, r{};
    double d = nodeSegmentDistance(xi, xj, xk, t, p, r);
    if (d >= dhat) return {0,0,0,0};
    d = std::max(d, 1e-12);

    Vec2 n{ r.x/d, r.y/d };
    double bp  = barrierGrad(d, dhat);
    double bpp = barrierHess(d, dhat);

    // K = b'' nn^T + (b'/d)(I - nn^T)
    double nx=n.x, ny=n.y;
    Mat2 K{
            bpp*nx*nx + (bp/d)*(1 - nx*nx),
            (bpp - bp/d)*nx*ny,
            (bpp - bp/d)*nx*ny,
            bpp*ny*ny + (bp/d)*(1 - ny*ny)
    };

    Mat2 T{}, P{};
    buildProjectors(xj, xk, T, P);

    if (who == node) {
        // H_{ii} = P K P
        Mat2 PK = mul(P, K);
        Mat2 H  = mul(PK, P);
        return H;
    } else {
        return {0,0,0,0};
    }
}

// ==============================
// Function Psi
// ==============================

// Gravity energy
double gravityEnergy(const Vec& x, const std::vector<double>& mass, const Vec2& g_accel){
    double Eg = 0.0;
    int N = static_cast<int>(mass.size());
    for (int i = 0; i < N; ++i) {
        Vec2 xi = getXi(x, i);
        Eg += -mass[i] * (g_accel.x * xi.x + g_accel.y * xi.y);
    }
    return Eg;
}

// Total barrier energy that sums over all active pairs
double totalBarrierEnergy(const Vec& x, const std::vector<BarrierPair>& barriers, double dhat){
    double Eb = 0.0;
    for (const BarrierPair& c : barriers) {
        double t;
        Vec2 p{}, r{};
        double d = nodeSegmentDistance(getXi(x, c.node),getXi(x, c.seg0),getXi(x, c.seg1),t,
                                       p, r);
        Eb += barrierEnergy(d, dhat);
    }
    return Eb;
}

// Full Psi energy
double PsiEnergy(const Vec& x, const Vec& xhat, const std::vector<double>& mass, const std::vector<double>& L,
                 double dt, double k, const Vec2& g_accel, const std::vector<BarrierPair>& barriers, double dhat){
    double Ein = 0.0;
    int N = static_cast<int>(mass.size());
    for (int i = 0; i < N; ++i) {
        Vec2 xi = getXi(x, i), xhi = getXi(xhat, i);
        double dx = xi.x - xhi.x, dy = xi.y - xhi.y;
        Ein += 0.5 * mass[i] * (dx * dx + dy * dy);
    }
    double Es = totalSpringEnergy(x, k, L);
    double Eb = totalBarrierEnergy(x, barriers, dhat);
    double Eg = gravityEnergy(x, mass, g_accel);

    return Ein + dt * dt * (Es + Eb + Eg);
}

// ==============================
// Local Gradient of the function Psi
// ==============================
Vec2 PsiLocalGrad(int i, const Vec& x, const Vec& xhat, const std::vector<double>& mass,
                  const std::vector<double>& L, double dt, double k, const Vec2& g_accel,
                  const std::vector<bool>& is_fixed, const std::vector<BarrierPair>& barriers, double dhat) {
    if (is_fixed[i]) return {0.0, 0.0};

    Vec2 xi = getXi(x, i), xhi = getXi(xhat, i);
    Vec2 gi{0.0, 0.0};

    // Mass term
    gi.x += mass[i] * (xi.x - xhi.x);
    gi.y += mass[i] * (xi.y - xhi.y);

    // Spring contribution
    Vec2 gs = localSpringGrad(i, x, k, L);
    gi.x += dt * dt * gs.x;
    gi.y += dt * dt * gs.y;

    // Gravity
    gi.x -= dt * dt * mass[i] * g_accel.x;
    gi.y -= dt * dt * mass[i] * g_accel.y;

    // Barrier forces
    for (const BarrierPair& c : barriers) {
        for (int who : {c.node, c.seg0, c.seg1}) {
            if (who != i) continue;
            Vec2 gb = localBarrierGrad(i, x, c.node, c.seg0, c.seg1, dhat);
            gi.x += dt * dt * gb.x;
            gi.y += dt * dt * gb.y;
        }
    }

    return gi;
}

// ==============================
// Local Hessian of the function Psi
// ==============================
Mat2 PsiLocalHess(int i, const Vec& x, const std::vector<double>& mass, const std::vector<double>& L, double dt,
                  double k,const std::vector<bool>& is_fixed, const std::vector<BarrierPair>& barriers, double dhat){
    if (is_fixed[i]) return {0, 0, 0, 0};

    Mat2 H{mass[i], 0, 0, mass[i]};

    // Spring term
    Mat2 Hs = localSpringHess(i, x, k, L);
    H.a11 += dt * dt * Hs.a11;
    H.a12 += dt * dt * Hs.a12;
    H.a21 += dt * dt * Hs.a21;
    H.a22 += dt * dt * Hs.a22;

    // Barrier term (local node-block only)
    for (const BarrierPair & c : barriers) {
        for (int who : {c.node, c.seg0, c.seg1}) {
            if (who != i) continue;
            Mat2 Hb = localBarrierHess(who, x, c.node, c.seg0, c.seg1, dhat);
            H.a11 += dt * dt * Hb.a11;
            H.a12 += dt * dt * Hb.a12;
            H.a21 += dt * dt * Hb.a21;
            H.a22 += dt * dt * Hb.a22;
        }
    }
    return H;
}

// =======================
// 2D Point–Segment CCD
// =======================
bool ccd_point_segment_2d(const Vec2& x1, const Vec2& dx1, const Vec2& x2, const Vec2& dx2,
                          const Vec2& x3, const Vec2& dx3, double& t_out, double eps = 1e-12){

    // Vector computation
    std::function<Vec2(const Vec2&, const Vec2&)> sub = [](const Vec2& a, const Vec2& b) -> Vec2 {
        return {a.x - b.x, a.y - b.y};
    };

    std::function<Vec2(const Vec2&, const Vec2&)> add = [](const Vec2& a, const Vec2& b) -> Vec2 {
        return {a.x + b.x, a.y + b.y};
    };

    std::function<Vec2(const Vec2&, double)> mul = [](const Vec2& a, double s) -> Vec2 {
        return {a.x * s, a.y * s};
    };

    std::function<double(const Vec2&, const Vec2&)> dot = [](const Vec2& a, const Vec2& b) -> double {
        return a.x * b.x + a.y * b.y;
    };

    std::function<double(const Vec2&, const Vec2&)> cross = [](const Vec2& a, const Vec2& b) -> double {
        return a.x * b.y - a.y * b.x;
    };

    std::function<double(const Vec2&)> norm2 = [&](const Vec2& a) -> double {
        return dot(a, a);
    };

    // Compute coefficients of f(t) = a t^2 + b t + c
    Vec2 x21  = sub(x1, x2);
    Vec2 x32  = sub(x3, x2);
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
    }
    else {
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
    Vec2 x1t = add(x1, mul(dx1, t_star));
    Vec2 x2t = add(x2, mul(dx2, t_star));
    Vec2 x3t = add(x3, mul(dx3, t_star));

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

double ccd_get_safe_step(const Vec2& x1, const Vec2& dx1, const Vec2& x2, const Vec2& dx2,
                         const Vec2& x3, const Vec2& dx3,double eta = 0.9) {

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

// ==============================
// Utilities
// ==============================
double residual_norm(const std::vector<Vec2> &r) {
    double sum = 0.0;
    for (Vec2 ri:r) {
        sum += ri.x * ri.x + ri.y * ri.y;
    }

    return std::sqrt(sum);
}

// Inverse of any 2x2 matrix
Mat2 matrix2d_inverse(const Mat2 &H) {
    double det = H.a11 * H.a22 - H.a12 * H.a21;
    if (std::abs(det) < 1e-12) {
        throw std::runtime_error("Singular matrix in inverse()");
    }
    double inv_det = 1.0 / det;
    Mat2 Hi{};
    Hi.a11 =  H.a22 * inv_det;
    Hi.a12 = -H.a12 * inv_det;
    Hi.a21 = -H.a21 * inv_det;
    Hi.a22 =  H.a11 * inv_det;
    return Hi;
}

// =====================================================
//  Nonlinear Gauss–Seidel using local grad/hess
// =====================================================

// Compute local gradient + barrier
Vec2 compute_local_gradient(int i, const Vec &x_local, const Vec &xhat_local, const std::vector<double> &mass_local,
                            const std::vector<double> &L_local, double dt, double k, const Vec2 &g_accel,
                            const std::vector<bool> &is_fixed_local, const std::vector<BarrierPair> &barriers_global,
                            double dhat, const Vec &x_global, int global_offset){
    Vec2 gi = PsiLocalGrad(i, x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                           is_fixed_local, {}, dhat);

    // Add barrier contributions
    int who_global = global_offset + i;
    Vec2 gbar{0.0, 0.0};
    for (const auto &c : barriers_global) {
        if (c.node != who_global and c.seg0 != who_global and c.seg1 != who_global)
            continue;
        Vec2 gb = localBarrierGrad(who_global, x_global, c.node, c.seg0, c.seg1, dhat);
        gbar.x += gb.x; gbar.y += gb.y;
    }

    gi.x += dt * dt * gbar.x;
    gi.y += dt * dt * gbar.y;
    return gi;
}

// Compute local Hessian + barrier
Mat2 compute_local_hessian(int i, const Vec &x_local, const std::vector<double> &mass_local,
                           const std::vector<double> &L_local, double dt, double k,
                           const std::vector<bool> &is_fixed_local, const std::vector<BarrierPair> &barriers_global,
                           double dhat, const Vec &x_global, int global_offset){
    Mat2 Hi = PsiLocalHess(i, x_local, mass_local, L_local, dt, k, is_fixed_local, {}, dhat);
    int who_global = global_offset + i;

    for (const auto &c : barriers_global) {
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
                         const std::vector<BarrierPair> &barriers_global, double eta){
    double omega_hat = 1.0;
    Vec2 dx_zero = {0.0, 0.0};

    for (const auto &c : barriers_global) {
        if (c.node != who_global)
            continue;

        Vec2 xi = getXi(x_global, c.node);
        Vec2 xj = getXi(x_global, c.seg0);
        Vec2 xk = getXi(x_global, c.seg1);

        Vec2 dx_neg = {-dx.x, -dx.y};
        double omega_c = ccd_get_safe_step(xi, dx_neg, xj, dx_zero, xk, dx_zero, eta);
        omega_hat = std::min(1.0, omega_c);
    }

    return omega_hat;
}

double compute_residual(const Vec &x_local, const Vec &xhat_local, const std::vector<double> &mass_local,
                        const std::vector<double> &L_local, double dt, double k, const Vec2 &g_accel,
                        const std::vector<bool> &is_fixed_local, const std::vector<BarrierPair> &barriers_global,
                        double dhat, const Vec &x_global, int global_offset){
    const int N = (int)mass_local.size();
    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        if (is_fixed_local[i]) continue;
        Vec2 g = compute_local_gradient(i, x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                                        is_fixed_local, barriers_global, dhat, x_global, global_offset);
        sum += g.x * g.x + g.y * g.y;
    }

    return std::sqrt(sum);
}


std::pair<double,int> gauss_seidel_minimize_with_barrier_global(Vec &x_local, const Vec &xhat_local,
                                                                const std::vector<double> &mass_local,
                                                                const std::vector<double> &L_local,
                                                                double dt, double k, const Vec2 &g_accel,
                                                                const std::vector<bool> &is_fixed_local,
                                                                const std::vector<BarrierPair> &barriers_global,
                                                                double dhat, Vec &x_global, int global_offset,
                                                                int max_sweeps, double tol_abs, double eta){
    const int N_local = (int)mass_local.size();

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (int i = 0; i < N_local; ++i) {
            if (is_fixed_local[i]) continue;

            Vec2 gi = compute_local_gradient(i, x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                                             is_fixed_local, barriers_global, dhat, x_global, global_offset);

            Mat2 Hi = compute_local_hessian(i, x_local, mass_local, L_local, dt, k, is_fixed_local, barriers_global,
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

        double rn = compute_residual(x_local, xhat_local, mass_local, L_local, dt, k, g_accel, is_fixed_local,
                                     barriers_global, dhat, x_global, global_offset);
        if (rn < tol_abs)
            return {rn, sweep + 1};
    }

    double rn = compute_residual(x_local, xhat_local, mass_local, L_local, dt, k, g_accel, is_fixed_local,
                                 barriers_global, dhat, x_global, global_offset);
    return {rn, max_sweeps};
}


// =====================================================
//  Export a Obj file
// =====================================================
void export_obj(const std::string &filename, const Vec &x, const std::vector<std::pair<int,int>> &edges){
    std::ofstream out(filename);

    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }
    int N = (int)(x.size()/2);

    for (int i=0;i<N;++i){
        Vec2 xi = getXi(x,i);
        out << "v " << xi.x << " " << xi.y << " 0.0\n";
    }

    for (const std::pair<int,int> &e : edges)
        out << "l " << (e.first+1) << " " << (e.second+1) << "\n";
    out.close();
}

// Export function
void export_frame(const std::string& outdir, int frame, const Vec& x_combined,
                  const std::vector<std::pair<int,int>>& edges_combined){
    std::ostringstream ss;
    ss << outdir << "/frame_" << std::setw(4)
       << std::setfill('0') << frame << ".obj";
    export_obj(ss.str(), x_combined, edges_combined);
}

// ==============================
// Numerical Experiment: the left spring is fixed, while the right spring has only the initial node fixed
// ==============================
Chain make_chain(Vec2 start, Vec2 end, int N, bool fix_first, bool fix_last, double mass_value) {
    Chain c;
    c.N = N;

    // Allocate vectors
    c.x.resize(2 * N);
    c.v.resize(2 * N, 0.0);
    c.xhat.resize(2 * N, 0.0);
    c.mass.resize(N, mass_value);
    c.is_fixed.resize(N, false);

    // Fix endpoints if necessary
    if (fix_first) c.is_fixed[0] = true;
    if (fix_last)  c.is_fixed[N-1] = true;

    // Place nodes linearly between start and end
    for (int i = 0; i < N; ++i) {
        double t = (N == 1) ? 0.0 : double(i) / (N - 1);
        Vec2 xi{
                start.x + t * (end.x - start.x),
                start.y + t * (end.y - start.y)
        };
        setXi(c.x, i, xi);
    }

    // Build edge list and rest lengths
    for (int i = 0; i < N - 1; ++i) {
        c.edges.emplace_back(i, i + 1);
        c.rest_lengths.push_back(norm(c.x, i, i + 1));
    }

    return c;
}

void combine_positions(Vec& x_combined, const Vec& x_left, const Vec& x_right, int N_left, int N_right){
    for (int i = 0; i < N_left; ++i)
        setXi(x_combined, i, getXi(x_left, i));
    for (int i = 0; i < N_right; ++i)
        setXi(x_combined, N_left + i, getXi(x_right, i));
}

std::vector<BarrierPair> build_barriers(int N_left, int N_right, const std::vector<bool>& is_fixed_right){
    std::vector<BarrierPair> barriers;
    for (int i = 0; i < N_right; ++i) {
        if (is_fixed_right[i]) continue;
        for (int j = 0; j < N_left - 1; ++j) {
            BarrierPair bp{};
            bp.node = N_left + i; // global index of right node
            bp.seg0 = j;          // global index of left segment start
            bp.seg1 = j + 1;
            barriers.push_back(bp);
        }
    }
    return barriers;
}

int main(){
    std::string outdir = "frames_spring_IPC3";
    fs::create_directory(outdir);

    // Parameters
    double dt = 1.0 / 30.0;
    Vec2 g_accel = {0.0, -9.81};
    double k_spring = 20.0;
    int total_frame = 100;
    int max_sweeps = 300;
    double tol_abs = 1e-8;
    double dhat = 0.2;
    double eta = 0.9;

    // Create chains
    Chain left  = make_chain({-1.0, 0.0}, {0.0, -1.0}, 2, true, true, 0.05); // fully fixed
    Chain right = make_chain({-1.2, 0.2}, {-0.2, 0.2}, 2, true, false, 0.05);

    // Combined geometry for OBJ export
    std::vector<std::pair<int,int>> edges_combined = left.edges;
    for (auto &e : right.edges)
        edges_combined.emplace_back(e.first + left.N, e.second + left.N);

    Vec x_combined(2 * (left.N + right.N), 0.0);
    combine_positions(x_combined, left.x, right.x, left.N, right.N);
    export_frame(outdir, 0, x_combined, edges_combined);

    // Main simulation loop
    for (int frame = 1; frame <= total_frame; ++frame) {

        // Predictor step: xhat = x + dt * v
        for (int i = 0; i < right.N; ++i) {
            Vec2 xi = getXi(right.x, i);
            Vec2 vi = getXi(right.v, i);
            setXi(right.xhat, i, { xi.x + dt * vi.x, xi.y + dt * vi.y });
        }

        // Combine positions for barrier queries
        combine_positions(x_combined, left.x, right.x, left.N, right.N);

        // Defines contact pairs
        std::vector<BarrierPair> barriers = build_barriers(left.N, right.N, right.is_fixed);

        // Call Gauss–Seidel solver
        Vec xnew_right = right.x;
        std::pair<double, int> result = gauss_seidel_minimize_with_barrier_global( xnew_right, right.xhat,
                                                                                   right.mass, right.rest_lengths,
                                                                                   dt, k_spring, g_accel,
                                                                                   right.is_fixed,
                                                                                   barriers, dhat,
                                                                                   x_combined, left.N,
                                                                                   max_sweeps, tol_abs, eta);
        double residual = result.first;
        int sweeps = result.second;

        // Velocity update using Backward Euler
        for (int i = 0; i < right.N; ++i) {
            if (right.is_fixed[i]) continue;
            Vec2 xi_new = getXi(xnew_right, i);
            Vec2 xi_old = getXi(right.x, i);
            setXi(right.v, i, { (xi_new.x - xi_old.x) / dt, (xi_new.y - xi_old.y) / dt });
        }

        right.x = xnew_right;

        // Combine geometry for OBJ export
        combine_positions(x_combined, left.x, right.x, left.N, right.N);

        // Compute signed distances for reporting
        double min_signed_d = std::numeric_limits<double>::max();
        bool collision = false;

        for (const BarrierPair& c : barriers) {
            Vec2 xi_end = getXi(x_combined, c.node);
            Vec2 xj = getXi(x_combined, c.seg0);
            Vec2 xk = getXi(x_combined, c.seg1);
            double d_end = nodeSegmentSignedDistance(xi_end, xj, xk);
            if (d_end < min_signed_d) min_signed_d = d_end;
            if (d_end < 0.0) collision = true;
        }

        // Export frame
        export_frame(outdir, frame, x_combined, edges_combined);

        // Print info
        std::cout << "Frame " << std::setw(4) << frame
                  << " | residual=" << std::scientific << residual
                  << " | sweeps=" << sweeps
                  << " | signed_d=" << min_signed_d
                  << (collision ? "  COLLISION" : "")
                  << std::endl;
    }

    return 0;
}