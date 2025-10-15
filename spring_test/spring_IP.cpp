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

// Compute point–segment distance
double nodeSegmentDistance(const Vec2 &xi,
                           const Vec2 &xj,
                           const Vec2 &xjp1,
                           double &t, Vec2 &p, Vec2 &r)
{
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

// -------------------- Utilities --------------------

namespace fs = std::__fs::filesystem;

// The residual , over all nodes, i.e. the gradient of Psi
// Psi(x) = 1/2 sum_i m_i ||x_i - xhat_i||^2 + dt^2 * E_spring(x) - dt^2 * sum_i m_i g · x_i
// r_i = m_i (x_i - xhat_i) + dt^2 * grad_i E_spring(x) - dt^2 * m_i g

std::vector<Vec2> residual_Psi(const Vec &x, const Vec &xhat, const std::vector<double> &mass,
        const std::vector<double> &L, double dt, double k, const Vec2 &g_accel,const std::vector<bool> &is_fixed)
{
    int N = static_cast<int>(mass.size());
    std::vector<Vec2> r(N, {0.0,0.0});
    for (int i = 0; i < N; ++i) {
        Vec2 xi = getXi(x, i);
        Vec2 xhat_i = getXi(xhat, i);

        // mass term
        r[i].x += mass[i] * (xi.x - xhat_i.x);
        r[i].y += mass[i] * (xi.y - xhat_i.y);

        // spring term
        Vec2 gi = localSpringGrad(i, x, k, L);
        r[i].x += dt * dt * gi.x;
        r[i].y += dt * dt * gi.y;

        // gravity
        r[i].x -= dt * dt * mass[i] * g_accel.x;
        r[i].y -= dt * dt * mass[i] * g_accel.y;

        if (is_fixed[i]) r[i] = {0.0, 0.0};
    }
    return r;
}

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
    if (std::abs(det) < 1e-16) {
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


 // -------------------- Nonlinear GS with local Hessian inverse --------------------
std::pair<double,int> gauss_seidel_minimize(Vec &x, const Vec &xhat, const std::vector<double> &mass,
                                            const std::vector<double> &L, double dt, double k, const Vec2 &g_accel,
                                            const std::vector<bool> &is_fixed, int max_sweeps=200, double tol_abs=1e-10,
                                            double omega=1.0){
    int N = static_cast<int>(mass.size());

    for (int sweep=0; sweep<max_sweeps; ++sweep) {
        for (int i=0;i<N;++i){
            if (is_fixed[i]) continue;

            Vec2 xi     = getXi(x,i);
            Vec2 xhat_i = getXi(xhat,i);

            // --- Gradient block r_i ---
            Vec2 r_i{0.0,0.0};

            // mass term
            r_i.x += mass[i]*(xi.x - xhat_i.x);
            r_i.y += mass[i]*(xi.y - xhat_i.y);

            // springs
            Vec2 gi = localSpringGrad(i, x, k, L);
            r_i.x += dt*dt * gi.x;
            r_i.y += dt*dt * gi.y;

            // gravity
            r_i.x -= dt*dt * mass[i] * g_accel.x;
            r_i.y -= dt*dt * mass[i] * g_accel.y;

            // --- Hessian block ---
            // Compute spring contribution
            Mat2 Hspring = localSpringHess(i, x, k, L);

            // Scale spring part by dt^2
            Hspring.a11 *= dt*dt;
            Hspring.a12 *= dt*dt;
            Hspring.a21 *= dt*dt;
            Hspring.a22 *= dt*dt;

            // Full Hessian (mass + Hspring)
            Mat2 Hii{};
            Hii.a11 = mass[i] + Hspring.a11;
            Hii.a12 = Hspring.a12;
            Hii.a21 = Hspring.a21;
            Hii.a22 = mass[i] + Hspring.a22;

            // Solve Hii * delta = r_i
            Mat2 Hii_inv = matrix2d_inverse(Hii);
            Vec2 delta   = mat2_mul(Hii_inv, r_i);

            // Update
            xi.x -= omega * delta.x;
            xi.y -= omega * delta.y;
            setXi(x, i, xi);
        }

        std::vector<Vec2> r = residual_Psi(x, xhat, mass, L, dt, k, g_accel, is_fixed);
        double current_residual_norm = residual_norm(r);
        if (current_residual_norm < tol_abs)
            return {current_residual_norm, sweep+1};
    }
    std::vector<Vec2> r = residual_Psi(x, xhat, mass, L, dt, k, g_accel, is_fixed);
    return {residual_norm(r), max_sweeps};
}

// -------------------- OBJ export --------------------
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


// -------------------------------------------------------------
// Main simulation: the right spring swings while the left spring is kept fixed
// -------------------------------------------------------------
int main(){
    std::string outdir = "frames_spring_IP";
    fs::create_directory(outdir);

    // --- Parameters ---
    double dt          = 1.0 / 30.0;
    Vec2   g_accel     = {0.0, -9.81};
    double k_spring    = 20.0;
    int    total_frame = 300;
    int    max_sweeps  = 50;
    double tol_abs     = 1e-8;
    double omega       = 1.0;

    // ---------------- Left chain (fixed) ----------------
    const int N_left = 2;
    Vec x_left(2*N_left, 0.0);
    for (int i = 0; i < N_left; ++i)
        setXi(x_left, i, { -0.5, -i * 1.0 });

    std::vector<std::pair<int,int>> edges_left;
    for (int i = 0; i < N_left - 1; ++i)
        edges_left.emplace_back(i, i + 1);

    // ---------------- Right chain (dynamic) -------------
    const int N_right = 2;
    Vec x_right(2*N_right, 0.0), v_right(2*N_right, 0.0), xhat_right(2*N_right, 0.0);
    std::vector<double> mass_right(N_right, 0.05);
    std::vector<bool>   is_fixed_right(N_right, false);
    is_fixed_right[0] = true; // pin first node

    for (int i = 0; i < N_right; ++i)
        setXi(x_right, i, { 0.0 + (double)i * 1.0, 0.0 });  // unit segment initially

    std::vector<std::pair<int,int>> edges_right;
    for (int i = 0; i < N_right - 1; ++i)
        edges_right.emplace_back(i, i + 1);

    std::vector<double> L_right;
    for (std::pair<int,int> &e : edges_right)
        L_right.push_back(norm(x_right, e.first, e.second));

    // ==========================================================
    // Combine for OBJ export
    // ==========================================================
    std::vector<std::pair<int,int>> edges_combined;
    edges_combined.reserve(edges_left.size());
    for (std::pair<int,int> &e : edges_left)
        edges_combined.push_back(e);

    for (std::pair<int,int> &e : edges_right)
        edges_combined.emplace_back( e.first + N_left, e.second + N_left );

    Vec x_combined(2*(N_left + N_right), 0.0);
    for (int i = 0; i < N_left; ++i)
        setXi(x_combined, i, getXi(x_left, i));

    for (int i = 0; i < N_right; ++i)
        setXi(x_combined, N_left + i, getXi(x_right, i));

    export_obj(outdir + "/frame_0000.obj", x_combined, edges_combined);

    // ==========================================================
    // Main simulation loop
    // ==========================================================
    for (int frame = 1; frame <= total_frame; ++frame)
    {
        // ----------------------------------------------------------
        // Combine for barrier geometry updates
        // ----------------------------------------------------------
        for (int i = 0; i < N_left; ++i)
            setXi(x_combined, i, getXi(x_left, i));
        for (int i = 0; i < N_right; ++i)
            setXi(x_combined, N_left + i, getXi(x_right, i));


        // ----------------------------------------------------------
        // Predictor step
        // ----------------------------------------------------------
        for (int i = 0; i < N_right; ++i) {
            Vec2 xi = getXi(x_right, i);
            Vec2 vi = getXi(v_right, i);
            setXi(xhat_right, i, { xi.x + dt * vi.x, xi.y + dt * vi.y });
        }

        Vec xnew_right = xhat_right;
        std::pair<double,int> res_right = gauss_seidel_minimize(
                xnew_right, xhat_right, mass_right, L_right, dt, k_spring, g_accel,
                is_fixed_right,  max_sweeps, tol_abs, omega
        );

        // ----------------------------------------------------------
        // Check convergence
        // ----------------------------------------------------------
        if (res_right.second >= max_sweeps && res_right.first > tol_abs) {
            std::cerr << "\n[FATAL] Frame " << frame
                      << " failed to converge! Residual = "
                      << res_right.first << "\n";
            std::exit(EXIT_FAILURE);
        }

        // ----------------------------------------------------------
        // Velocity update
        // ----------------------------------------------------------
        for (int i = 0; i < N_right; ++i) {
            if (is_fixed_right[i]) continue;
            Vec2 xi_new = getXi(xnew_right, i);
            Vec2 xi_old = getXi(x_right, i);
            setXi(v_right, i, { (xi_new.x - xi_old.x) / dt, (xi_new.y - xi_old.y) / dt });
        }
        x_right = xnew_right;


        // ----------------------------------------------------------
        // Save frame and log summary
        // ----------------------------------------------------------
        std::ostringstream ss;
        ss << outdir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
        export_obj(ss.str(), x_combined, edges_combined);

        std::cout << "Frame " << frame
                  << " | res=" << res_right.first
                  << " | sweeps=" << res_right.second << "\n";
    }

    return 0;
}