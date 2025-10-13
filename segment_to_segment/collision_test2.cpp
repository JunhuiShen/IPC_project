#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <system_error>
#include <cstdio>

#include <filesystem>
namespace fs = std::__fs::filesystem;

#include <Eigen/Dense>
#include <Eigen/QR>                   // RealSchur

// ============================================================================
// Data types
// ============================================================================
struct P2 { double x, y; };

struct Sample {
    P2 x;   // position
    P2 v;   // velocity
    double m; // mass
};

// ============================================================================
// Step 0: This visualization helper writes a filled 2D circle so that points can be more visible
// ============================================================================
static void appendCircle(std::ofstream& ofs, double cx, double cy, double r, int nSeg, int& vCount){
    int start = vCount + 1;
    for(int i=0;i<nSeg;++i){
        double th = 2.0 * M_PI * (double)i / (double)nSeg;
        ofs << "v " << (cx + r*std::cos(th)) << " " << (cy + r*std::sin(th)) << " 0\n";
        ++vCount;
    }
    for(int i=1;i<nSeg-1;++i){
        ofs << "f " << start << " " << (start+i) << " " << (start+i+1) << "\n";
    }
}

//// ============================================================================
//// Step 1: Generate random points with random initial velocities
//// ============================================================================
//static std::vector<Sample> makeRandomSamples(int N, unsigned seed = 1234){
//    std::mt19937 rng(seed);
//    std::uniform_real_distribution<double> ux(-0.5, 0.5);
//    std::uniform_real_distribution<double> uy(-0.5, 0.5);
//    std::normal_distribution<double>       uv(0.0, 0.5); // Gaussian velocities
//
//    std::vector<Sample> S(N);
//    for(int i=0;i<N;++i){
//        S[i].x = { ux(rng), uy(rng) };
//        S[i].v = { uv(rng), uv(rng) };
//        S[i].m = 1.0;
//    }
//    return S;
//}

// ============================================================================
// Step 1: One two-point moving-upward segment along with one fixed three-point segment
// ============================================================================
std::vector<Sample> makeSegmentSamples(){
    std::vector<Sample> S;

    // Two-point moving segment (bottom, moving upward)
    S.push_back({ P2{-0.5, 0.0}, P2{0.0, 0.5}, 1.0 });  // moving endpoint A
    S.push_back({ P2{ 0.5, 0.0}, P2{0.0, 0.5}, 1.0 });  // moving endpoint B

    // Three-point fixed segment (top, stationary)
    S.push_back({ P2{-1.0, 1.0}, P2{0.0, 0.0}, 1.0 });  // fixed point 1
    S.push_back({ P2{ 0.0, 1.0}, P2{0.0, 0.0}, 1.0 });  // fixed point 2
    S.push_back({ P2{ 1.0, 1.0}, P2{0.0, 0.0}, 1.0 });  // fixed point 3

    return S;
}

// ============================================================================
// Compute the COM: x_com  = (sum_i m_i * x_i) / (sum_i m_i), y_com = (sum_i m_i * y_i) / (sum_i m_i)
// ============================================================================
static P2 centerOfMass(const std::vector<Sample>& S){
    double M = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    for(const auto& s : S){
        M += s.m;
        cx += s.m * s.x.x;
        cy += s.m * s.x.y;
    }
    return { cx / M, cy / M };
}

// ============================================================================
// Compute the basis functions for affine-body dynamics: tilde(u)^(k)(x) for k= 1, ..., 6 at point x
//   1: A1(x-xhat)=(-dy, dx) where A1 = [0 -1; 1 0]
//   2: A2(x-xhat)=( dy, dx) where A2 = [0 1; 1 0]
//   3: A3(x-xhat)=( dx, 0) where A3 = [1 0; 0 0]
//   4: A4(x-xhat)=( 0, dy) where A4 = [0 0; 0 1]
//   5: (1,0)  translation-x
//   6: (0,1)  translation-y
// ============================================================================
static inline P2 Uk(int k, const P2& x, const P2& xhat){
    double dx = x.x - xhat.x, dy = x.y - xhat.y;
    switch(k){
        case 1: return { -dy,  dx };
        case 2: return {  dy,  dx };
        case 3: return {  dx,  0  };
        case 4: return {  0 ,  dy };
        case 5: return {  1 ,  0  };
        case 6: return {  0 ,  1  };
    }
    return {0,0};
}

// ============================================================================
// Step 2: Assemble G (6x6) and b (6x1) and solve Gc=b
// ============================================================================
static void build_G_and_b(const std::vector<Sample>& S, const P2& xhat, Eigen::Matrix<double,6,6>& G, Eigen::Matrix<double,6,1>& b)
{
    G.setZero(); b.setZero();
    for(const auto& s:S){
        for(int k=0;k<6;++k){
            P2 uk = Uk(k+1, s.x, xhat);
            // b_k += m * <u^k, v>
            b(k) += s.m * (uk.x * s.v.x + uk.y * s.v.y);
            // G_{k,j} += m * <u^k, u^j>
            for(int j=0;j<6;++j){
                P2 uj = Uk(j+1, s.x, xhat);
                G(k,j) += s.m * (uk.x * uj.x + uk.y * uj.y);
            }
        }
    }
}

// ============================================================================
// Step 3: From c1..c6 build B (sum_i c_i A_i for i = 1, ... , 4)and vhat  ([c5; c6])
// ============================================================================
static void assemble_B_and_vhat(const Eigen::Matrix<double,6,1>& c, Eigen::Matrix2d& B, Eigen::Vector2d& vhat)
{
    // Basis matrices
    Eigen::Matrix2d A1, A2, A3, A4;
    A1 << 0, -1, 1,  0;
    A2 << 0,  1, 1,  0;
    A3 << 1,  0, 0,  0;
    A4 << 0,  0, 0,  1;

    B = c(0)*A1 + c(1)*A2 + c(2)*A3 + c(3)*A4;
    vhat << c(4), c(5);
}

// ============================================================================
// Step 4a: Hard-coded exact one-step update in Schur coordinates using the ODE
//   h' = T h + vtil,   T = [ a  b ]
//                      [ 0  d ]
//   vtil = constant (Q^T vhat)
// ============================================================================
static inline Eigen::Vector2d step_exact_upper2x2real(
        double a, double b, double d,
        const Eigen::Vector2d& h,
        const Eigen::Vector2d& vtil,
        double dt)
{
    constexpr double eps = 1e-14;
    const auto expz = [](double z){ return std::exp(z); };

    const double Ea = expz(a * dt);
    const double Ed = expz(d * dt);

    // --- h2 update ---
    double h2p;
    if (std::abs(d) < eps) {
        h2p = h(1) + dt * vtil(1); // limit d -> 0
    } else {
        h2p = Ed * h(1) + (Ed - 1.0) / d * vtil(1);
    }

    // --- h1 homogeneous ---
    double h1_hom;
    if (std::abs(a - d) < eps) {
        // limit a = d
        h1_hom = Ea * h(0) + b * Ea * (dt * h(1));
    } else {
        h1_hom = Ea * h(0) + (b / (d - a)) * (Ed - Ea) * h(1);
    }

    // --- h1 from v1 ---
    double h1_v1;
    if (std::abs(a) < eps) {
        h1_v1 = dt * vtil(0);
    } else {
        h1_v1 = (Ea - 1.0) / a * vtil(0);
    }

    // --- h1 from v2 ---
    double h1_v2;
    if (std::abs(d) < eps) {
        if (std::abs(a) < eps) {
            h1_v2 = 0.5 * dt * dt * vtil(1); // a=0,d=0
        } else {
            h1_v2 = ((dt / a) - (1.0 - Ea) / (a * a)) * vtil(1); // d=0
        }
    } else {
        if (std::abs(a) < eps) {
            h1_v2 = (((Ed - 1.0) / (d * d)) - dt / d) * vtil(1); // a=0
        } else {
            // general case
            h1_v2 = ( (Ed - Ea) / (d - a) - (Ea - 1.0) / a ) / d * vtil(1);
        }
    }

    double h1p = h1_hom + h1_v1 + h1_v2;

    return { h1p, h2p };
}

// ============================================================================
// Step 4b: Exact one-step update in Schur coordinates for a complex-conjugate
//          eigenpair (rotation–scaling block)
//   h' = T h + vtil,
//   T = [ a   b ]
//       [ -b  a ],   eigenvalues = a ± i b
//   vtil = constant (Q^T vhat)
// ============================================================================
static inline Eigen::Vector2d step_exact_upper2x2complex(
        double a, double b,
        const Eigen::Vector2d& h,
        const Eigen::Vector2d& vtil,
        double dt)
{
    constexpr double eps = 1e-14;

    // --- Homogeneous update ---
    const double E = std::exp(a * dt);
    const double c = std::cos(b * dt);
    const double s = std::sin(b * dt);

    Eigen::Vector2d h_hom;
    h_hom(0) = E * (c * h(0) + s * h(1));
    h_hom(1) = E * (-s * h(0) + c * h(1));

    // --- Inhomogeneous update ---
    // A = E c - 1, B = E s
    const double A = E * c - 1.0;
    const double B = E * s;

    const double determinant_T = a * a + b * b;

    Eigen::Matrix2d F;
    if (determinant_T > eps) {
        // Exact closed form:
        // F =  T^{-1}(e^{T dt} - I) = (1/(a^2+b^2)) * [ [aA+bB,  aB-bA; -(aB-bA), aA+bB] ]
        const double m11 = a * A + b * B;
        const double m12 = a * B - b * A;
        F(0,0) =  m11 / determinant_T;
        F(0,1) =  m12 / determinant_T;
        F(1,0) = -m12 / determinant_T;
        F(1,1) =  m11 / determinant_T;
    } else {
        // Degenerate / small (a^2 + b^2)
        // F = dt I + (1/2) dt^2 T + (1/6) dt^3 T^2
        // with T = [[a, b], [-b, a]] and T^2 = (a^2 - b^2) I + 2ab J where J = [0 1; -1 0]
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;

        // I coefficient: dt + (1/2) a dt^2 + (1/6)(a^2 - b^2) dt^3
        const double Icoef = dt + 0.5 * a * dt2 + (1.0/6.0) * (a*a - b*b) * dt3;

        // J coefficient: (1/2) b dt^2 + (1/3) a b dt^3  (since (1/6)*2ab = 1/3 ab)
        const double Jcoef = 0.5 * b * dt2 + (1.0/3.0) * a * b * dt3;

        F(0,0) =  Icoef;
        F(0,1) =  Jcoef;
        F(1,0) = -Jcoef;
        F(1,1) =  Icoef;
    }

    return h_hom + F * vtil;
}

// ============================================================================
// Step 4c: Dispatcher for a 2x2 Schur block
// Chooses the correct exact one-step update based on the block structure.
// - Real upper-triangular:    T = [[a, b], [0, d]]
// - Complex conjugate block:  T = [[a, b], [-b, a]]
// ============================================================================
static inline Eigen::Vector2d step_exact_schur2x2(
        const Eigen::Matrix2d& T,
        const Eigen::Vector2d& h,
        const Eigen::Vector2d& vtil,
        double dt)
{
    constexpr double tol = 1e-14;

    const double a = T(0,0);
    const double b = T(0,1);
    const double c = T(1,0);
    const double d = T(1,1);

    // Complex block test: equal diagonals AND anti-symmetric off-diagonals
    const bool is_complex_block =
            (std::abs(a - d) < tol) && (std::abs(c + b) < tol) && (std::abs(c) > tol);

    if (is_complex_block) {
        // T = [[a, b], [-b, a]]
        return step_exact_upper2x2complex(a, b, h, vtil, dt);
    }

    // Otherwise treat as real upper-triangular (c ~ 0)
    return step_exact_upper2x2real(a, b, d, h, vtil, dt);
}



//// ============================================================================
//// Step 5: Write the OBJ frames
//// ============================================================================
//static void writeOBJFrame(const std::string& path,
//                          const std::vector<Eigen::Vector2d>& H,
//                          const Eigen::Matrix2d& Q,
//                          const P2& xhat,
//                          double visSx, double visSy,
//                          double rVis, int segVis)
//{
//    std::ofstream ofs(path);
//    if(!ofs){ throw std::runtime_error("Cannot open OBJ: " + path); }
//
//    int vCount = 0;
//    ofs << "# " << path << "\n";
//
//    // particles as filled discs
//    for(const auto& h : H){
//        Eigen::Vector2d w = Q * h;                // current offset (h = tilde(phi))
//        double X = xhat.x + w(0);           // global x-position
//        double Y = xhat.y + w(1);           // global y-position
//        appendCircle(ofs, visSx*X, visSy*Y, rVis, segVis, vCount);
//    }
//}

// ============================================================================
// Step 5: Write the OBJ frames (with segment connectivity)
// ============================================================================
static void writeOBJFrame(const std::string& path,
                          const std::vector<Eigen::Vector2d>& H,
                          const Eigen::Matrix2d& Q,
                          const P2& xhat,
                          double visSx, double visSy,
                          double rVis, int segVis)
{
    std::ofstream ofs(path);
    if(!ofs){ throw std::runtime_error("Cannot open OBJ: " + path); }

    int vCount = 0;
    ofs << "# " << path << "\n";

    // --- Draw circles for visualization ---
    for(const auto& h : H){
        Eigen::Vector2d w = Q * h;
        double X = xhat.x + w(0);
        double Y = xhat.y + w(1);
        appendCircle(ofs, visSx*X, visSy*Y, rVis, segVis, vCount);
    }

    // --- Add "center vertices" for connectivity ---
    std::vector<int> centerIdx;
    for(const auto& h : H){
        Eigen::Vector2d w = Q * h;
        double X = xhat.x + w(0);
        double Y = xhat.y + w(1);
        ofs << "v " << visSx*X << " " << visSy*Y << " 0\n";
        ++vCount;
        centerIdx.push_back(vCount);
    }

    // --- Connectivity ---
    ofs << "\n# Segments\n";
    // Moving 2-point segment
    ofs << "l " << centerIdx[0] << " " << centerIdx[1] << "\n";
    // Fixed 3-point segment
    ofs << "l " << centerIdx[2] << " " << centerIdx[3] << " " << centerIdx[4] << "\n";
}


// ============================================================================
// Step 6: Main
// ============================================================================
int main(){
    // --------------------- parameters ---------------------
//    const int    N         = 100;
    const double dt        = 1.0 / 30.0;
    const int    nFrames   = 500;
    const double rVis      = 0.05;
    const int    segVis    = 30;
    const double visSx     = 5.0;
    const double visSy     = 5.0;
    const std::string outDir = "obj_affine_schur";

//    // ------------------------------------------------------
//    // Generate random particle samples in R^2
//    // ------------------------------------------------------
//    auto S = makeRandomSamples(N);

    // ------------------------------------------------------
    // Generate the fixed and moving segments
    // ------------------------------------------------------
    auto S = makeSegmentSamples();
    const int N = static_cast<int>(S.size());

    // xhat = center of mass of reference points
    P2 xhat = centerOfMass(S);

    // ------------------------------------------------------
    // Build normal equations G c = b to fit an affine map
    // Solve for affine coefficients (c1, ..., c6)
    // ------------------------------------------------------
    Eigen::Matrix<double,6,6> G;
    Eigen::Matrix<double,6,1> b;
    build_G_and_b(S, xhat, G, b);
    Eigen::Matrix<double,6,1> c = G.ldlt().solve(b);

    // ------------------------------------------------------
    // Assemble B and vhat such that original ODE is:
    //
    //   x'(t) = B (x(t) - xhat) + vhat,
    //
    // where x is in R^2 and B is a 2x2 real matrix.
    // ------------------------------------------------------
    Eigen::Matrix2d B;
    Eigen::Vector2d vhat;
    assemble_B_and_vhat(c, B, vhat);

    // ------------------------------------------------------
    // Schur decomposition: B = Q T Q^T,
    //   Q orthogonal, T upper-triangular.
    //
    // Change variables: h = Q^T (x - xhat).
    //
    // In Schur coordinates, the ODE becomes:
    //
    //   h'(t) = T h(t) + vtil,   with vtil = Q^T vhat.
    //
    // This triangular form allows an exact step via closed-form integration of the 2x2 system.
    // ------------------------------------------------------
    Eigen::RealSchur<Eigen::Matrix2d> schur;
    schur.compute(B);
    Eigen::Matrix2d Q  = schur.matrixU();   // orthogonal
    Eigen::Matrix2d T  = schur.matrixT();   // 2x2 upper-triangular
    Eigen::Matrix2d QT = Q.transpose();

    // Define constant vector vtil = Q^T vhat
    Eigen::Vector2d vtil = QT * vhat;

    // Extract T entries: T = [ b11 b12; 0 b22 ]
    const double b11 = T(0,0), b12 = T(0,1), b22 = T(1,1);

    // ------------------------------------------------------
    // Initialize Schur coordinates for all particles:
    //   h_i = Q^T (x_i - xhat)
    // ------------------------------------------------------
    std::vector<Eigen::Vector2d> H(N);
    for(int i=0;i<N;++i){
        Eigen::Vector2d rel;
        rel << S[i].x.x - xhat.x,
                S[i].x.y - xhat.y;
        H[i] = QT * rel;
    }

    // Create output directory
    std::error_code ec;
    fs::create_directories(outDir, ec);

    for(int f=0; f<nFrames; ++f){
        char nm[256];
        std::snprintf(nm, sizeof(nm), "%s/frame_%04d.obj", outDir.c_str(), f);
        writeOBJFrame(nm, H, Q, xhat, visSx, visSy, rVis, segVis);

        // Exact one-step update in Schur form:
        for(int i=0; i<N; ++i){
            // Pass the actual 2x2 Schur block T for particle i
            H[i] = step_exact_schur2x2(T, H[i], vtil, dt);
        }
    }

    // ------------------------------------------------------
    // Print the relevant outputs
    // ------------------------------------------------------
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "c^T = " << c.transpose() << "\n";
    std::cout << "B =\n" << B << "\n";
//    Eigen::EigenSolver<Eigen::Matrix2d> eig(B);
//    std::cout << "Eigenvalues of B: "
//              << eig.eigenvalues().transpose() << "\n";
    std::cout << "vhat = [" << vhat.transpose() << "]\n";
    std::cout << "xhat = (" << xhat.x << "," << xhat.y << ")\n";
    std::cout << "Wrote " << nFrames << " frames to ./" << outDir << "\n";
    return 0;
}
