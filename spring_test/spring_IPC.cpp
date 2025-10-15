#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
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
// Local Gradient
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
// Local Hessian
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
            if (i == j || i == j + 1) continue; // skip self/adjacent
            pairs.push_back({i, j, j + 1});
        }
    }
    return pairs;
}

// Barrier energy and derivatives
double barrierEnergy(double d, double dhat) {
    if (d >= dhat) return 0.0;
    return - (d - dhat) * (d - dhat) * std::log(d / dhat);
}

double barrierGrad(double d, double dhat) {
    if (d >= dhat) return 0.0;
    return -2 * (d - dhat) * std::log(d / dhat) - (d - dhat) * (d - dhat) / d;
}

double barrierHess(double d, double dhat) {
    if (d >= dhat) return 0.0;
    return -2 * std::log(d / dhat) - 4 * (d - dhat) / d + (d - dhat) * (d - dhat) / (d * d);
}

// ======================================================
// Compute point–segment distance
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
// Compute signed point–segment distance
// ======================================================
double nodeSegmentSignedDistance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1, double &t, Vec2 &p, Vec2 &r){
    double unsigned_d = nodeSegmentDistance(xi, xj, xjp1, t, p, r);

    // Segment direction
    Vec2 seg = { xjp1.x - xj.x, xjp1.y - xj.y };
    double seg_len2 = seg.x * seg.x + seg.y * seg.y;
    if (seg_len2 < 1e-14) return unsigned_d;

    // Compute normal only if projection inside segment
    if (t > 1e-10 && t < 1.0 - 1e-10){
        double seg_len = std::sqrt(seg_len2);
        Vec2 n = { -seg.y / seg_len, seg.x / seg_len };
        double signed_d = n.x * (xi.x - p.x) + n.y * (xi.y - p.y);
        return signed_d;
    } else {
        // Outside region: sign not meaningful
        return unsigned_d;
    }
}



// ==============================
// LocalBarrierGrad
// ==============================
static inline Vec2 mul(const Mat2& A, const Vec2& v){
    return { A.a11*v.x + A.a12*v.y, A.a21*v.x + A.a22*v.y };
}
static inline Mat2 mul(const Mat2& A, const Mat2& B){
    return {
            A.a11*B.a11 + A.a12*B.a21,  A.a11*B.a12 + A.a12*B.a22,
            A.a21*B.a11 + A.a22*B.a21,  A.a21*B.a12 + A.a22*B.a22
    };
}

// ======================================================
// LocalBarrierGrad
// ======================================================
// Build projectors T, P from segment direction s = x_{j+1} - x_j
// T = uu^T and P = I - uu^T
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

    Mat2 T{}, P{}; buildProjectors(xj, xk, T, P);
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
// LocalBarrierHess (node block only)
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

    Mat2 T{}, P{}; buildProjectors(xj, xk, T, P);

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
        double d = nodeSegmentDistance(getXi(x, c.node),getXi(x, c.seg0),getXi(x, c.seg1),t, p, r);
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

Vec2 mat2_mul(const Mat2 &A, const Vec2 &v) {
    return { A.a11 * v.x + A.a12 * v.y, A.a21 * v.x + A.a22 * v.y };
}

// =====================================================
//  Nonlinear Gauss–Seidel using local grad/hess
// =====================================================
std::pair<double,int> gauss_seidel_minimize_with_barrier_global(Vec &x_local, const Vec &xhat_local,
                                                                const std::vector<double> &mass_local,
                                                                const std::vector<double> &L_local,
                                                                double dt, double k, const Vec2 &g_accel,
                                                                const std::vector<bool> &is_fixed_local,
                                                                const std::vector<BarrierPair> &barriers_global,
                                                                double dhat, const Vec &x_global,
                                                                int global_offset, int max_sweeps = 200,
                                                                double tol_abs = 1e-10, double omega = 1.0){
    const int N_local = (int)mass_local.size();

    std::function<double(const std::vector<Vec2>&)> residual_norm2 = [&](const std::vector<Vec2> &r){
        double s = 0.0;
        for (const Vec2 &v : r) {
            s += v.x * v.x + v.y * v.y;
        }
        return std::sqrt(s);
    };

    for (int sweep = 0; sweep < max_sweeps; ++sweep){
        // One Gauss–Seidel sweep over all local nodes
        for (int i = 0; i < N_local; ++i){
            if (is_fixed_local[i]) continue;

            // Local gradient (mass + spring + gravity)
            Vec2 gi = PsiLocalGrad( i, x_local, xhat_local, mass_local, L_local, dt, k, g_accel,
                                    is_fixed_local, {}, dhat);

            // Add barrier gradient using global geometry
            int who_global = global_offset + i;
            Vec2 gbar{0.0, 0.0};

            for (const BarrierPair &c : barriers_global){
                if (c.node != who_global && c.seg0 != who_global && c.seg1 != who_global)
                    continue; // only relevant if this node participates

                Vec2 gb = localBarrierGrad(who_global, x_global, c.node, c.seg0, c.seg1, dhat);
                gbar.x += gb.x;
                gbar.y += gb.y;
            }

            // Scale by dt^2 (since Psi includes dt^2 * PE(x))
            gi.x += dt * dt * gbar.x;
            gi.y += dt * dt * gbar.y;

            // Local Hessian (mass + spring)
            Mat2 Hi = PsiLocalHess(
                    i, x_local, mass_local, L_local, dt, k, is_fixed_local, {}, dhat);{
            // Add barrier Hessian block
                Mat2 Hb_sum{0, 0, 0, 0};
                for (const BarrierPair &c : barriers_global){
                    if (c.node != who_global && c.seg0 != who_global && c.seg1 != who_global)
                        continue;

                    Mat2 Hb = localBarrierHess(who_global, x_global, c.node, c.seg0, c.seg1, dhat);

                    Hb_sum.a11 += Hb.a11; Hb_sum.a12 += Hb.a12;
                    Hb_sum.a21 += Hb.a21; Hb_sum.a22 += Hb.a22;
                }

                Hi.a11 += dt * dt * Hb_sum.a11;
                Hi.a12 += dt * dt * Hb_sum.a12;
                Hi.a21 += dt * dt * Hb_sum.a21;
                Hi.a22 += dt * dt * Hb_sum.a22;
            }

            // Solve and update local position
            Mat2 Hi_inv = matrix2d_inverse(Hi);
            Vec2 dx = mat2_mul(Hi_inv, gi);

            Vec2 xi = getXi(x_local, i);
            xi.x -= omega * dx.x;
            xi.y -= omega * dx.y;
            setXi(x_local, i, xi);
        }

        // Compute residual norm for stopping criterion
        std::vector<Vec2> r(N_local, {0, 0});
        for (int i = 0; i < N_local; ++i){
            if (is_fixed_local[i]) continue;

            Vec2 gi = PsiLocalGrad(i, x_local, xhat_local, mass_local, L_local,dt, k, g_accel,
                                   is_fixed_local, {}, dhat);

            int who_global = global_offset + i;
            Vec2 gbar{0.0, 0.0};
            for (const BarrierPair &c : barriers_global){
                if (c.node != who_global && c.seg0 != who_global && c.seg1 != who_global)
                    continue;

                Vec2 gb = localBarrierGrad(who_global, x_global, c.node, c.seg0, c.seg1, dhat);
                gbar.x += gb.x;
                gbar.y += gb.y;
            }

            gi.x += dt * dt * gbar.x;
            gi.y += dt * dt * gbar.y;
            r[i] = gi;
        }

        double rn = residual_norm2(r);

        // Collision check
        bool collision = false;
        double min_signed_d = std::numeric_limits<double>::max();

        for (const auto &c : barriers_global) {
            Vec2 xi = getXi(x_global, c.node);
            Vec2 xj = getXi(x_global, c.seg0);
            Vec2 xk = getXi(x_global, c.seg1);
            double t;
            Vec2 p{}, rvec{};
            double d_signed = nodeSegmentSignedDistance(xi, xj, xk, t, p, rvec);
            min_signed_d = std::min(min_signed_d, d_signed);

            if (d_signed < 0.0) {
                collision = true;
            }
        }

        // Print sweep info
        std::cout << "Sweep " << std::setw(3) << sweep + 1
                  << " | residual = " << rn
                  << " | signed_d = " << min_signed_d
                  << (collision ? "  Collision detected" : "")
                  << std::endl;


        if (rn < tol_abs)
            return {rn, sweep + 1};
    }

    // Return final residual if not converged
    std::vector<Vec2> r(N_local, {0, 0});
    for (int i = 0; i < N_local; ++i){
        if (is_fixed_local[i]) continue;

        Vec2 gi = PsiLocalGrad(i, x_local, xhat_local, mass_local, L_local,
                               dt, k, g_accel, is_fixed_local, {}, dhat);

        int who_global = global_offset + i;
        Vec2 gbar{0.0, 0.0};
        for (const BarrierPair &c : barriers_global){
            if (c.node != who_global && c.seg0 != who_global && c.seg1 != who_global)
                continue;

            Vec2 gb = localBarrierGrad(who_global, x_global, c.node, c.seg0, c.seg1, dhat);
            gbar.x += gb.x;
            gbar.y += gb.y;
        }

        gi.x += dt * dt * gbar.x;
        gi.y += dt * dt * gbar.y;
        r[i] = gi;
    }

    double rn = 0.0;
    for (Vec2 &v : r)
        rn += v.x * v.x + v.y * v.y;
    rn = std::sqrt(rn);

    return {rn, max_sweeps};
}

// =====================================================
//  Export a Obj file
// =====================================================
void export_obj(const std::string &filename, const Vec &x, const std::vector<std::pair<int,int>> &edges)
{
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

// ==============================
// Numerical Experiment: the left spring is fixed, while the right spring has only the initial node fixed
// ==============================
int main(){
    std::string outdir = "frames_spring_IPC2";
    fs::create_directory(outdir);

    // -------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------
    double dt          = 1.0 / 40.0;
    Vec2   g_accel     = {0.0, -9.81};
    double k_spring    = 20.0;
    int    total_frame = 600;
    int    max_sweeps  = 50;
    double tol_abs     = 1e-8;
    double omega       = 0.4;
    double dhat        = 0.5;  // barrier activation distance

    // -------------------------------------------------------------
    // Left chain (fixed): on the line y = -x-1
    // -------------------------------------------------------------
    const int N_left = 2;
    Vec x_left(2 * N_left, 0.0);
    setXi(x_left, 0, { -1.0,  0.0 });
    setXi(x_left, 1, {  0.0, -1.0 });

    std::vector<std::pair<int,int>> edges_left;
    for (int i = 0; i < N_left - 1; ++i)
        edges_left.emplace_back(i, i + 1);


    // -------------------------------------------------------------
    // Right chain (dynamic): horizontal at y = 0.2
    // -------------------------------------------------------------
    const int N_right = 2;
    Vec x_right(2 * N_right, 0.0);
    Vec v_right(2 * N_right, 0.0);
    Vec xhat_right(2 * N_right, 0.0);

    std::vector<double> mass_right(N_right, 0.2);
    std::vector<bool> is_fixed_right(N_right, false);
    is_fixed_right[0] = true; // pin first node

    setXi(x_right, 0, { -1.2,  0.2 });
    setXi(x_right, 1, { -0.2,  0.2 });

    std::vector<std::pair<int,int>> edges_right;
    for (int i = 0; i < N_right - 1; ++i)
        edges_right.emplace_back(i, i + 1);

    std::vector<double> L_right;
    for (std::pair<int,int> &e : edges_right)
        L_right.push_back(norm(x_right, e.first, e.second));

    // -------------------------------------------------------------
    // Combined geometry for OBJ export
    // -------------------------------------------------------------
    std::vector<std::pair<int,int>> edges_combined = edges_left;
    for (std::pair<int,int> &e : edges_right)
        edges_combined.emplace_back(e.first + N_left, e.second + N_left);

    Vec x_combined(2 * (N_left + N_right), 0.0);
    for (int i = 0; i < N_left; ++i)
        setXi(x_combined, i, getXi(x_left, i));
    for (int i = 0; i < N_right; ++i)
        setXi(x_combined, N_left + i, getXi(x_right, i));

    export_obj(outdir + "/frame_0000.obj", x_combined, edges_combined);

    // -------------------------------------------------------------
    // Main simulation loop
    // -------------------------------------------------------------
    for (int frame = 1; frame <= total_frame; ++frame)
    {
        // Predictor step: xhat = x + dt * v
        for (int i = 0; i < N_right; ++i) {
            Vec2 xi = getXi(x_right, i);
            Vec2 vi = getXi(v_right, i);
            setXi(xhat_right, i, { xi.x + dt * vi.x, xi.y + dt * vi.y });
        }

        // Combine positions for barrier queries
        for (int i = 0; i < N_left; ++i)
            setXi(x_combined, i, getXi(x_left, i));
        for (int i = 0; i < N_right; ++i)
            setXi(x_combined, N_left + i, getXi(x_right, i));

        // Build barrier pairs (right node vs. left segment)
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

        // Gauss–Seidel minimization for the right chain
        Vec xnew_right = xhat_right;
        for (int i=0;i<N_left;++i) {
            setXi(x_combined, i, getXi(x_left, i));
        }

        for (int i=0;i<N_right;++i) {
            setXi(x_combined, N_left+i, getXi(x_right, i));
        }

        std::pair<double, int> result = gauss_seidel_minimize_with_barrier_global( xnew_right, xhat_right,
                                                                                   mass_right, L_right,
                                                                                   dt, k_spring, g_accel,
                                                                                   is_fixed_right,
                                                                                   barriers, dhat,
                                                                                   x_combined, N_left,
                                                                                   max_sweeps, tol_abs, omega);
        double residual = result.first;
        int sweeps = result.second;

        // Velocity update using Backward Euler
        for (int i = 0; i < N_right; ++i) {
            if (is_fixed_right[i]) continue;
            Vec2 xi_new = getXi(xnew_right, i);
            Vec2 xi_old = getXi(x_right, i);
            setXi(v_right, i, { (xi_new.x - xi_old.x) / dt,(xi_new.y - xi_old.y) / dt });
        }

        x_right = xnew_right;

        // Combine geometry for OBJ export
        for (int i = 0; i < N_left; ++i)
            setXi(x_combined, i, getXi(x_left, i));
        for (int i = 0; i < N_right; ++i)
            setXi(x_combined, N_left + i, getXi(x_right, i));

        // Export frame
        std::ostringstream ss;
        ss << outdir << "/frame_" << std::setw(4)
           << std::setfill('0') << frame << ".obj";
        export_obj(ss.str(), x_combined, edges_combined);

        // Print simulation info
        std::cout << "Frame " << std::setw(4) << frame
                  << " | residual=" << std::scientific << residual
                  << " | sweeps=" << sweeps
                  << " | num barriers=" << barriers.size()
                  << std::endl;
    }

    return 0;
}

//// ==============================
//// Numerical Experiment: the left spring is fixed, while the right spring has only the initial node fixed
//// ==============================
//int main()
//{
//    std::string outdir = "frames_spring_IPC";
//    fs::create_directory(outdir);
//
//    // -------------------------------------------------------------
//    // Parameters
//    // -------------------------------------------------------------
//    double dt          = 1.0 / 30.0;
//    Vec2   g_accel     = {0.0, -9.81};
//    double k_spring    = 20.0;
//    int    total_frame = 300;
//    int    max_sweeps  = 50;
//    double tol_abs     = 1e-8;
//    double omega       = 1.0;
//    double dhat        = 0.5;  // barrier activation distance
//
//    // -------------------------------------------------------------
//    // Left chain (fixed)
//    // -------------------------------------------------------------
//    const int N_left = 2;
//    Vec x_left(2 * N_left, 0.0);
//    for (int i = 0; i < N_left; ++i)
//        setXi(x_left, i, { -0.5, -i * 1.0 });  // vertical downward
//
//    std::vector<std::pair<int,int>> edges_left;
//    for (int i = 0; i < N_left - 1; ++i)
//        edges_left.emplace_back(i, i + 1);
//
//    // -------------------------------------------------------------
//    // Right chain (dynamic)
//    // -------------------------------------------------------------
//    const int N_right = 2;
//    Vec x_right(2 * N_right, 0.0);
//    Vec v_right(2 * N_right, 0.0);
//    Vec xhat_right(2 * N_right, 0.0);
//
//    std::vector<double> mass_right(N_right, 0.05);
//    std::vector<bool> is_fixed_right(N_right, false);
//    is_fixed_right[0] = true; // pin first node
//
//    for (int i = 0; i < N_right; ++i)
//        setXi(x_right, i, { 0.0 + (double)i * 1.0, 0.0 });  // horizontal layout
//
//    std::vector<std::pair<int,int>> edges_right;
//    for (int i = 0; i < N_right - 1; ++i)
//        edges_right.emplace_back(i, i + 1);
//
//    std::vector<double> L_right;
//    for (std::pair<int,int> &e : edges_right)
//        L_right.push_back(norm(x_right, e.first, e.second));
//
//    // -------------------------------------------------------------
//    // Combined geometry for OBJ export
//    // -------------------------------------------------------------
//    std::vector<std::pair<int,int>> edges_combined = edges_left;
//    for (std::pair<int,int> &e : edges_right)
//        edges_combined.emplace_back(e.first + N_left, e.second + N_left);
//
//    Vec x_combined(2 * (N_left + N_right), 0.0);
//    for (int i = 0; i < N_left; ++i)
//        setXi(x_combined, i, getXi(x_left, i));
//    for (int i = 0; i < N_right; ++i)
//        setXi(x_combined, N_left + i, getXi(x_right, i));
//
//    export_obj(outdir + "/frame_0000.obj", x_combined, edges_combined);
//
//    // -------------------------------------------------------------
//    // Main simulation loop
//    // -------------------------------------------------------------
//    for (int frame = 1; frame <= total_frame; ++frame)
//    {
//        // Predictor step: xhat = x + dt * v
//        for (int i = 0; i < N_right; ++i) {
//            Vec2 xi = getXi(x_right, i);
//            Vec2 vi = getXi(v_right, i);
//            setXi(xhat_right, i, { xi.x + dt * vi.x, xi.y + dt * vi.y });
//        }
//
//        // Combine positions for barrier queries
//        for (int i = 0; i < N_left; ++i)
//            setXi(x_combined, i, getXi(x_left, i));
//        for (int i = 0; i < N_right; ++i)
//            setXi(x_combined, N_left + i, getXi(x_right, i));
//
//        // Build barrier pairs (right node vs. left segment)
//        std::vector<BarrierPair> barriers;
//        for (int i = 0; i < N_right; ++i) {
//            if (is_fixed_right[i]) continue;
//            for (int j = 0; j < N_left - 1; ++j) {
//                BarrierPair bp{};
//                bp.node = N_left + i; // global index of right node
//                bp.seg0 = j;          // global index of left segment start
//                bp.seg1 = j + 1;
//                barriers.push_back(bp);
//            }
//        }
//
//        // Gauss–Seidel minimization for the right chain
//        Vec xnew_right = xhat_right;
//        for (int i=0;i<N_left;++i) {
//            setXi(x_combined, i, getXi(x_left, i));
//        }
//
//        for (int i=0;i<N_right;++i) {
//            setXi(x_combined, N_left+i, getXi(x_right, i));
//        }
//
//        std::pair<double, int> result = gauss_seidel_minimize_with_barrier_global( xnew_right, xhat_right,
//                                                                                   mass_right, L_right,
//                                                                                   dt, k_spring, g_accel,
//                                                                                   is_fixed_right,
//                                                                                   barriers, dhat,
//                                                                                   x_combined, N_left,
//                                                                                   max_sweeps, tol_abs, omega);
//        double residual = result.first;
//        int sweeps = result.second;
//
//        // Velocity update using Backward Euler
//        for (int i = 0; i < N_right; ++i) {
//            if (is_fixed_right[i]) continue;
//            Vec2 xi_new = getXi(xnew_right, i);
//            Vec2 xi_old = getXi(x_right, i);
//            setXi(v_right, i, { (xi_new.x - xi_old.x) / dt,(xi_new.y - xi_old.y) / dt });
//        }
//
//        x_right = xnew_right;
//
//        // Combine geometry for OBJ export
//        for (int i = 0; i < N_left; ++i)
//            setXi(x_combined, i, getXi(x_left, i));
//        for (int i = 0; i < N_right; ++i)
//            setXi(x_combined, N_left + i, getXi(x_right, i));
//
//        // Export frame
//        std::ostringstream ss;
//        ss << outdir << "/frame_" << std::setw(4)
//           << std::setfill('0') << frame << ".obj";
//        export_obj(ss.str(), x_combined, edges_combined);
//
//        // Print simulation info
//        std::cout << "Frame " << std::setw(4) << frame
//                  << " | residual=" << std::scientific << residual
//                  << " | sweeps=" << sweeps
//                  << " | num barriers=" << barriers.size()
//                  << std::endl;
//    }
//
//    return 0;
//}

