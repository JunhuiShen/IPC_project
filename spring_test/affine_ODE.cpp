#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <system_error>
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <filesystem>

// ======================================================
// Prelim structure and functions
// ======================================================
typedef std::vector<double> Vec;
namespace fs = std::__fs::filesystem;

// --- Core data structure ---
using Vec = std::vector<double>;

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

namespace convert {
    using namespace math;

// ============================================================================
// Conversion Functions (Bridge between custom types and Eigen)
// ============================================================================
    static inline Eigen::Matrix2d Mat2ToEigen(const Mat2 &M) {
        Eigen::Matrix2d E;
        E << M.a11, M.a12,
                M.a21, M.a22;
        return E;
    }

    static inline Mat2 EigenToMat2(const Eigen::Matrix2d &E) {
        return {E(0, 0), E(0, 1), E(1, 0), E(1, 1)};
    }

    static inline Eigen::Vector2d Vec2ToEigen(const Vec2 &V) {
        return {V.x, V.y};
    }

    static inline Vec2 EigenToVec2(const Eigen::Vector2d &E) {
        return {E(0), E(1)};
    }
}

namespace sampling {
    using namespace math;

    struct Sample {
        Vec2 x;
        Vec2 v;
        double m;
    };

    // Generate segment samples
    std::vector<Sample> makeSegmentSamples() {
        std::vector<Sample> S;

        // Two-point moving segment (bottom, moving upward)
        S.push_back({Vec2{-0.5, 0.0}, Vec2{0.0, 0.5}, 1.0});  // moving endpoint A
        S.push_back({Vec2{0.5, 0.0}, Vec2{0.0, 0.5}, 1.0});  // moving endpoint B

        // Three-point fixed segment (top, stationary)
        S.push_back({Vec2{-1.0, 1.0}, Vec2{0.0, 0.0}, 1.0});  // fixed point 1
        S.push_back({Vec2{0.0, 1.0}, Vec2{0.0, 0.0}, 1.0});  // fixed point 2
        S.push_back({Vec2{1.0, 1.0}, Vec2{0.0, 0.0}, 1.0});  // fixed point 3

        return S;
    }

// ============================================================================
// Compute the COM: x_com, y_com
// ============================================================================
    static Vec2 centerOfMass(const std::vector<Sample> &S) {
        double M = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        for (const auto &s: S) {
            M += s.m;
            cx += s.m * s.x.x;
            cy += s.m * s.x.y;
        }
        return {cx / M, cy / M};
    }
}

namespace affine_fit {
    using namespace math;
    using namespace sampling;
    using namespace convert;

// ============================================================================
// Compute the basis functions for affine-body dynamics: tilde(u)^(k)(x)
// ============================================================================
    static inline Vec2 Uk(int k, const Vec2 &x, const Vec2 &xhat) {
        double dx = x.x - xhat.x, dy = x.y - xhat.y;
        switch (k) {
            case 1:
                return {-dy, dx};
            case 2:
                return {dy, dx};
            case 3:
                return {dx, 0};
            case 4:
                return {0, dy};
            case 5:
                return {1, 0};
            case 6:
                return {0, 1};
        }
        return {0, 0};
    }

// ============================================================================
// Assemble G (6x6) and b (6x1) using Eigen types
// ============================================================================
    static void build_G_and_b(const std::vector<Sample> &S, const Vec2 &xhat,
                              Eigen::Matrix<double, 6, 6> &G, Eigen::Matrix<double, 6, 1> &b) {
        G.setZero();
        b.setZero();
        for (const auto &s: S) {
            for (int k = 0; k < 6; ++k) {
                Vec2 uk = Uk(k + 1, s.x, xhat);
                // b_k += m * <u^k, v>
                b(k) += s.m * (uk.x * s.v.x + uk.y * s.v.y);
                // G_{k,j} += m * <u^k, u^j>
                for (int j = 0; j < 6; ++j) {
                    Vec2 uj = Uk(j + 1, s.x, xhat);
                    G(k, j) += s.m * (uk.x * uj.x + uk.y * uj.y);
                }
            }
        }
    }

// ============================================================================
// From c1..c6 build B and vhat (Outputting custom Mat2/Vec2)
// ============================================================================
    static void assemble_B_and_vhat(const Eigen::Matrix<double, 6, 1> &c, Mat2 &B, Vec2 &vhat) {
        // Basis matrices (A1, A2, A3, A4) defined using Eigen for simple arithmetic
        Eigen::Matrix2d A1, A2, A3, A4;
        A1 << 0, -1, 1, 0;
        A2 << 0, 1, 1, 0;
        A3 << 1, 0, 0, 0;
        A4 << 0, 0, 0, 1;

        // B_eigen = c1*A1 + c2*A2 + c3*A3 + c4*A4
        Eigen::Matrix2d B_eigen = c(0) * A1 + c(1) * A2 + c(2) * A3 + c(3) * A4;
        Eigen::Vector2d vhat_eigen;
        vhat_eigen << c(4), c(5);

        // Convert results back to custom structs
        B = EigenToMat2(B_eigen);
        vhat = EigenToVec2(vhat_eigen);
    }
}

namespace solve_affine_ODE {
// ============================================================================
// Exact one-step update in Schur coordinates (Real upper-triangular)
// ============================================================================
    static inline Eigen::Vector2d step_exact_upper2x2real(double a, double b, double d, const Eigen::Vector2d &h,
                                                          const Eigen::Vector2d &vtil, double dt) {
        constexpr double eps = 1e-14;

        // --- Special case: diagonal matrix ---
        if (std::abs(b) < eps) {
            const double Ea = std::exp(a * dt);
            const double Ed = std::exp(d * dt);

            Eigen::Vector2d out;
            out(0) = Ea * h(0) + (std::abs(a) < eps ? dt * vtil(0) : (Ea - 1.0) / a * vtil(0));
            out(1) = Ed * h(1) + (std::abs(d) < eps ? dt * vtil(1) : (Ed - 1.0) / d * vtil(1));
            return out;
        }

        // --- General case ---
        const auto expz = [](double z) { return std::exp(z); };

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
                h1_v2 = ((Ed - Ea) / (d - a) - (Ea - 1.0) / a) / d * vtil(1);
            }
        }

        double h1p = h1_hom + h1_v1 + h1_v2;

        return {h1p, h2p};
    }

// ============================================================================
// Exact one-step update in Schur coordinates (Complex conjugate block)
// ============================================================================
    static inline Eigen::Vector2d step_exact_upper2x2complex(double a, double b, const Eigen::Vector2d &h,
                                                             const Eigen::Vector2d &vtil, const double dt) {
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
        const double B_term = E * s;

        const double determinant_T = a * a + b * b;

        Eigen::Matrix2d F;
        if (determinant_T > eps) {
            // Exact closed form
            const double m11 = a * A + b * B_term;
            const double m12 = a * B_term - b * A;
            F(0, 0) = m11 / determinant_T;
            F(0, 1) = m12 / determinant_T;
            F(1, 0) = -m12 / determinant_T;
            F(1, 1) = m11 / determinant_T;
        } else {
            // Degenerate / small (a^2 + b^2) approximation
            const double dt2 = dt * dt;
            const double dt3 = dt2 * dt;

            // I coefficient
            const double Icoef = dt + 0.5 * a * dt2 + (1.0 / 6.0) * (a * a - b * b) * dt3;

            // J coefficient
            const double Jcoef = 0.5 * b * dt2 + (1.0 / 3.0) * a * b * dt3;

            F(0, 0) = Icoef;
            F(0, 1) = Jcoef;
            F(1, 0) = -Jcoef;
            F(1, 1) = Icoef;
        }

        return h_hom + F * vtil;
    }

// ============================================================================
// Dispatcher for a 2x2 Schur block
// ============================================================================
    static inline Eigen::Vector2d step_exact_schur2x2(const Eigen::Matrix2d &T, const Eigen::Vector2d &h,
                                                      const Eigen::Vector2d &vtil, const double dt) {
        if (T.isZero(1e-14))
            return h + dt * vtil;

        constexpr double tol = 1e-14;

        const double a = T(0, 0);
        const double b = T(0, 1);
        const double c = T(1, 0);
        const double d = T(1, 1);

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
}

namespace visualization {
    static void appendCircle(std::ofstream &ofs, double cx, double cy, double r, int nSeg, int &vCount) {
        int start = vCount + 1;
        for (int i = 0; i < nSeg; ++i) {
            double th = 2.0 * M_PI * (double) i / (double) nSeg;
            ofs << "v " << (cx + r * std::cos(th)) << " " << (cy + r * std::sin(th)) << " 0\n";
            ++vCount;
        }
        for (int i = 1; i < nSeg - 1; ++i) {
            ofs << "f " << start << " " << (start + i) << " " << (start + i + 1) << "\n";
        }
    }

    static void writeOBJFrame(const std::string &path, const std::vector<Eigen::Vector2d> &H, const Eigen::Matrix2d &Q,
                              const Vec2 &xhat, double visSx, double visSy, double rVis, int segVis) {
        std::ofstream ofs(path);
        if (!ofs) { throw std::runtime_error("Cannot open OBJ: " + path); }

        int vCount = 0;
        ofs << "# " << path << "\n";

        // Draw circles for visualization
        for (const auto &h: H) {
            Eigen::Vector2d w = Q * h; // Back to global coordinates
            double X = xhat.x + w(0);
            double Y = xhat.y + w(1);
            appendCircle(ofs, visSx * X, visSy * Y, rVis, segVis, vCount);
        }

        // Add center vertices for connectivity
        std::vector<int> centerIdx;
        for (const auto &h: H) {
            Eigen::Vector2d w = Q * h;
            double X = xhat.x + w(0);
            double Y = xhat.y + w(1);
            ofs << "v " << visSx * X << " " << visSy * Y << " 0\n";
            ++vCount;
            centerIdx.push_back(vCount);
        }

        // Connectivity
        ofs << "\n# Segments\n";
        // Moving 2-point segment (indices 0 and 1)
        ofs << "l " << centerIdx[0] << " " << centerIdx[1] << "\n";
        // Fixed 3-point segment (indices 2, 3, and 4)
        ofs << "l " << centerIdx[2] << " " << centerIdx[3] << " " << centerIdx[4] << "\n";
    }
}

namespace simulation {
    using namespace math;
    using namespace convert;
    using namespace affine_fit;
    using namespace solve_affine_ODE;
    using namespace visualization;


    struct AffineODE {
        Eigen::Matrix2d T;       // upper-triangular dynamics matrix (Schur form)
        Eigen::Matrix2d Q;       // orthonormal transform
        Eigen::Matrix2d QT;      // Q^T
        Eigen::Vector2d vtil;    // forcing term in Schur coordinates
    };


// Convert native B and vhat results into the Eigen ODE structure
    static AffineODE buildAffineODE(const Mat2 &B_mat2, const Vec2 &vhat_vec2) {
        Eigen::Matrix2d B = Mat2ToEigen(B_mat2);
        Eigen::Vector2d vhat = Vec2ToEigen(vhat_vec2);

        Eigen::RealSchur<Eigen::Matrix2d> schur;
        schur.compute(B);

        AffineODE ode;
        ode.Q = schur.matrixU();
        ode.T = schur.matrixT();
        ode.QT = schur.matrixU().transpose();
        ode.vtil = schur.matrixU().transpose() * vhat;

        return ode;
    }


// Initialize particle positions directly in Schur coordinates (Eigen Vector2d)
    static std::vector<Eigen::Vector2d>
    initStateSchur(const std::vector<Sample> &S, const Vec2 &xhat, const Eigen::Matrix2d &QT) {
        std::vector<Eigen::Vector2d> H(S.size());
        for (size_t i = 0; i < S.size(); ++i) {
            Eigen::Vector2d rel;
            rel << S[i].x.x - xhat.x, S[i].x.y - xhat.y;
            H[i] = QT * rel;
        }
        return H;
    }

    int main() {
        // --------------------- parameters ---------------------
        const double dt = 1.0 / 30.0;
        const int nFrames = 500;
//    const double visSx = 5.0, visSy = 5.0;
//    const double rVis = 0.05;
//    const int segVis = 30;
//    const std::string outDir = "obj_affine_eigen_schur";

        // ------------------------------------------------------
        // Setup
        // ------------------------------------------------------
        auto S = makeSegmentSamples();
        Vec2 xhat = centerOfMass(S);

        // Solve least-squares system for affine fit
        Eigen::Matrix<double, 6, 6> G;
        Eigen::Matrix<double, 6, 1> b;
        build_G_and_b(S, xhat, G, b);

        // Solve Gc = b using Eigen's robust LDLT solver
        Eigen::Matrix<double, 6, 1> c = G.ldlt().solve(b);

        // Extract native affine parameters (B and vhat are stored in custom structs)
        Mat2 B_mat2{};
        Vec2 vhat_vec2{};
        assemble_B_and_vhat(c, B_mat2, vhat_vec2);

        // Build ODE system (Schur form using Eigen)
        AffineODE ode = buildAffineODE(B_mat2, vhat_vec2);

        // Initialize particle positions in Schur coordinates (Eigen Vector2d)
        std::vector<Eigen::Vector2d> H = initStateSchur(S, xhat, ode.QT);

//    // Ensure OBJ output directory exists
//    std::error_code ec;
//    fs::create_directories(outDir, ec);

        // ------------------------------------------------------
        // Simulation loop
        // ------------------------------------------------------
        for (int f = 0; f < nFrames; ++f) {

//        char nm[256];
//        std::snprintf(nm, sizeof(nm), "%s/frame_%04d.obj", outDir.c_str(), f);
//
//        // Write frame using Eigen types for visualization
//        writeOBJFrame(nm, H, ode.Q, xhat, visSx, visSy, rVis, segVis);

            // Exact analytical Schur-space ODE integration (Numerically robust)
            for (auto &h: H)
                // Use the robust Eigen dispatcher and analytical step functions
                h = step_exact_schur2x2(ode.T, h, ode.vtil, dt);
        }

        // ------------------------------------------------------
        // Print the relevant outputs
        // ------------------------------------------------------
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "c^T = " << c.transpose() << "\n";
        std::cout << "Recovered B =\n" << ode.Q * ode.T * ode.Q.transpose()
                  << "\n"; // Print B using Schur decomposition

        std::cout << "vhat = [" << ode.vtil(0) << ", " << ode.vtil(1)
                  << "]\n"; // Print vtil, which is the forcing term in Schur space
        std::cout << "xhat = (" << xhat.x << "," << xhat.y << ")\n";
//    std::cout << "Wrote " << nFrames << " frames to ./" << outDir << "\n";
        return 0;
    }
}

int main() {
    return simulation::main();
}

