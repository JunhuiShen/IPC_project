#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <string>

// ======================================================
// Prelim structure and functions
// ======================================================

typedef std::vector<double> Vec;

// --- Core data structure ---
struct Vec2 {
    double x, y;
};

struct Mat2 {
    double a11, a12, a21, a22;
};

// Global Vector Utilities
// Get the 2D vector for node i from the flattened Vec array
Vec2 getXi(const Vec &x, int i) {
    return {x[2 * i], x[2 * i + 1]};
}

// Set the 2D vector for node i in the flattened Vec array
void setXi(Vec &x, int i, const Vec2 &v) {
    x[2 * i] = v.x;
    x[2 * i + 1] = v.y;
}

namespace math {
    // --- Vec2 Operations ---
    static inline Vec2 add(const Vec2 &a, const Vec2 &b) {
        return {a.x + b.x, a.y + b.y};
    }

    static inline Vec2 sub(const Vec2 &a, const Vec2 &b) {
        return {a.x - b.x, a.y - b.y};
    }

    static inline Vec2 scale(const Vec2 &a, double s) {
        return {s * a.x, s * a.y};
    }

    static inline double dot(const Vec2 &a, const Vec2 &b) {
        return a.x * b.x + a.y * b.y;
    }

    // Matrix-vector multiplication
    static inline Vec2 mul(const Mat2 &A, const Vec2 &v) {
        return {A.a11 * v.x + A.a12 * v.y, A.a21 * v.x + A.a22 * v.y};
    }

    // --- Mat2 Operations ---
    // Scale matrix
    static inline Mat2 scale(const Mat2 &A, double s) {
        return {s * A.a11, s * A.a12, s * A.a21, s * A.a22};
    }
}

// ======================================================
// Initial guess for the Gauss-Seidel solver (Section 3)
// ======================================================
namespace initial_guess {
    using namespace math;

    // Parameters of the rotational affine velocity field:
    // V(X,0) = omega * R (X - X_COM) + v_hat
    struct RotationalAffineVelocity {
        double omega;  // angular velocity omega
        Vec2  v_hat;   // translational velocity v_hat
        Vec2  X_com;   // center of mass X_COM
    };

    // Fit the rotational affine velocity field from sample positions X, velocities V, masses m,
    // EXCLUDING fixed nodes from the fit.
    RotationalAffineVelocity
    fit_rotational_affine_velocity(const Vec &X,
                                   const Vec &V,
                                   const std::vector<double> &mass,
                                   const std::vector<bool> &is_fixed)
    {
        const int N = static_cast<int>(mass.size());
        RotationalAffineVelocity params{0.0, {0.0, 0.0}, {0.0, 0.0}};

        // ------------------------------------------------
        // 1) Center of mass X_COM over *free* nodes only
        // ------------------------------------------------
        double M = 0.0;
        Vec2 X_com{0.0, 0.0};

        for (int i = 0; i < N; ++i) {
            if (is_fixed[i]) continue;  // exclude anchors
            double mi = mass[i];
            if (mi <= 0.0) continue;

            Vec2 Xi = getXi(X, i);
            M       += mi;
            X_com.x += mi * Xi.x;
            X_com.y += mi * Xi.y;
        }

        if (M <= 0.0) {
            // Degenerate: no free mass ⇒ zero field
            params.X_com = {0.0, 0.0};
            params.omega = 0.0;
            params.v_hat = {0.0, 0.0};
            return params;
        }

        X_com.x /= M;
        X_com.y /= M;
        params.X_com = X_com;

        // ------------------------------------------------
        // 2) Build normal equations G c = b in R^3
        //    using only free nodes
        // ------------------------------------------------
        double G[3][3] = {{0.0, 0.0, 0.0},
                          {0.0, 0.0, 0.0},
                          {0.0, 0.0, 0.0}};
        double b[3] = {0.0, 0.0, 0.0};

        int free_count = 0;

        for (int i = 0; i < N; ++i) {
            if (is_fixed[i]) continue;
            double mi = mass[i];
            if (mi <= 0.0) continue;

            free_count++;

            Vec2 Xi = getXi(X, i);
            Vec2 Vi = getXi(V, i);
            Vec2 X_shift = sub(Xi, X_com);  // Xi - X_COM
            double x = X_shift.x;
            double y = X_shift.y;

            // Basis vectors
            Vec2 U[3];
            U[0] = Vec2{-y, x};  // rotation
            U[1] = Vec2{ 1, 0};  // translation x
            U[2] = Vec2{ 0, 1};  // translation y

            // Accumulate G and b
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < 3; ++j) {
                    G[k][j] += mi * dot(U[k], U[j]);
                }
                b[k] += mi * dot(U[k], Vi);
            }
        }

        // ------------------------------------------------
        // 2.5) Not enough free nodes ⇒ pure translation fit
        // ------------------------------------------------
        if (free_count < 2) {
            // Fit only a uniform translation v_hat = avg(V) over free nodes
            Vec2 v_avg{0.0, 0.0};
            for (int i = 0; i < N; ++i) {
                if (is_fixed[i]) continue;
                double mi = mass[i];
                if (mi <= 0.0) continue;

                Vec2 Vi = getXi(V, i);
                v_avg.x += mi * Vi.x;
                v_avg.y += mi * Vi.y;
            }
            v_avg.x /= M;
            v_avg.y /= M;

            params.omega = 0.0;
            params.v_hat = v_avg;
            return params;
        }

        // ------------------------------------------------
        // 3) Solve the 3x3 system G c = b using Gaussian elimination
        // ------------------------------------------------
        double A[3][4] = {
                {G[0][0], G[0][1], G[0][2], b[0]},
                {G[1][0], G[1][1], G[1][2], b[1]},
                {G[2][0], G[2][1], G[2][2], b[2]}
        };

        // Forward elimination with partial pivoting
        for (int pivot = 0; pivot < 3; ++pivot) {
            int maxRow = pivot;
            double maxVal = std::fabs(A[pivot][pivot]);
            for (int r = pivot + 1; r < 3; ++r) {
                double v = std::fabs(A[r][pivot]);
                if (v > maxVal) {
                    maxVal = v;
                    maxRow = r;
                }
            }

            if (maxVal < 1e-12) {
                // Ill-conditioned; fall back to zero field
                params.omega = 0.0;
                params.v_hat = {0.0, 0.0};
                return params;
            }

            // Swap rows if needed
            if (maxRow != pivot) {
                for (int c = 0; c < 4; ++c)
                    std::swap(A[pivot][c], A[maxRow][c]);
            }

            // Eliminate
            double diag = A[pivot][pivot];
            for (int r = pivot + 1; r < 3; ++r) {
                double alpha = A[r][pivot] / diag;
                for (int c = pivot; c < 4; ++c) {
                    A[r][c] -= alpha * A[pivot][c];
                }
            }
        }

        // Back substitution
        double c_vec[3];
        for (int i = 2; i >= 0; --i) {
            double sum = A[i][3];
            for (int j = i + 1; j < 3; ++j) {
                sum -= A[i][j] * c_vec[j];
            }
            c_vec[i] = sum / A[i][i];
        }

        // c1 = omega, (c2, c3) = v_hat
        params.omega = c_vec[0];
        params.v_hat = Vec2{c_vec[1], c_vec[2]};
        // params.X_com already set
        return params;
    }

    // Apply the uniformly affine motion
    //   phi(X, dt) = (I + dt B)(X - X_COM) + (X_COM + v_hat dt)
    // where B = omega R, R = [[0, -1],[1, 0]].
    // The result is the collision-free initial guess x_guess (which is assigned to x_hat in the solver).
    void apply_uniformly_affine_predictor(const Vec &X_ref,
                                          const RotationalAffineVelocity &params,
                                          double dt,
                                          Vec &x_guess) // RENAMED xhat -> x_guess
    {
        const int N = static_cast<int>(X_ref.size() / 2);
        x_guess.resize(2 * N);

        const double omega = params.omega;
        const Vec2  &X_com = params.X_com;
        const Vec2  &v_hat = params.v_hat;

        // A(dt) = I + omega dt R
        // R = [0 -1; 1 0] => A(dt) = [1  -omega dt;
        //                             omega dt   1]
        Mat2 A_dt{1.0, -omega * dt,
                  omega * dt,  1.0};

        Vec2 X_com_adv{
                X_com.x + v_hat.x * dt,
                X_com.y + v_hat.y * dt
        };

        for (int i = 0; i < N; ++i) {
            Vec2 Xi       = getXi(X_ref, i);
            Vec2 Xi_shift = sub(Xi, X_com);      // X - X_COM
            Vec2 Y        = mul(A_dt, Xi_shift); // (I + dt B)(X - X_COM)
            Vec2 Xi_new   = add(Y, X_com_adv);   // + (X_COM + v_hat dt)
            setXi(x_guess, i, Xi_new); // RENAMED xhat -> x_guess
        }
    }

    // Apply affine motion WITH a fixed anchor:
    //   phi*(X) = X_a + A(dt)(X - X_a)
    // This guarantees that the node at anchor_index is exactly fixed.
    void apply_affine_about_anchor(const Vec &X_ref,
                                   const RotationalAffineVelocity &params,
                                   double dt,
                                   Vec &x_guess, // RENAMED xhat -> x_guess
                                   int anchor_index)
    {
        const int N = static_cast<int>(X_ref.size() / 2);
        x_guess.resize(2 * N);

        Vec2 Xa = getXi(X_ref, anchor_index);

        double w = params.omega;
        Mat2 A_dt{1.0, -w * dt,
                  w * dt,  1.0};

        for (int i = 0; i < N; ++i) {
            Vec2 Xi  = getXi(X_ref, i);
            Vec2 rel = math::sub(Xi, Xa);
            Vec2 rot = math::mul(A_dt, rel);
            Vec2 Xi_new = math::add(Xa, rot);
            setXi(x_guess, i, Xi_new); // RENAMED xhat -> x_guess
        }

        // Enforce exact position of the anchor (numerical cleanliness)
        setXi(x_guess, anchor_index, Xa); // RENAMED xhat -> x_guess
    }
}

// ======================================================
// Example Usage (Main function for testing)
// ======================================================
int main() {
    // --- Setup Parameters ---
    const double dt = 1.0/30.0; // Time step
    const int N = 4;        // Number of nodes

    // Reference Configuration (X^n, 2N elements: x0y0x1y1...)
    // A simple square centered at (1, 1)
    Vec X_ref = {
            0.0, 0.0, // X0
            2.0, 0.0, // X1
            2.0, 2.0, // X2
            0.0, 2.0  // X3
    };

    // Initial Velocities (V, 2N elements)
    // We set up velocities corresponding to:
    // Pure counter-clockwise rotation (omega=10) and a translation (v_hat=(1, 0))
    const double omega_true = 10.0;
    Vec2 v_hat_true = {1.0, 0.0};
    Vec2 X_com_true = {1.0, 1.0};

    // Rotation matrix R = [[0, -1], [1, 0]]
    Mat2 R_mat = {0.0, -1.0, 1.0, 0.0};
    Mat2 B_mat = math::scale(R_mat, omega_true); // B = omega R

    // Calculate V = B(X - X_COM) + v_hat
    Vec V(2 * N);
    for (int i = 0; i < N; ++i) {
        Vec2 Xi = getXi(X_ref, i);
        Vec2 X_shift = math::sub(Xi, X_com_true);
        Vec2 Rot = math::mul(B_mat, X_shift);
        Vec2 Vi_true = math::add(Rot, v_hat_true);
        setXi(V, i, Vi_true);
    }

    // Mass and Fixed nodes
    std::vector<double> mass(N, 1.0); // All mass = 1.0
    std::vector<bool> is_fixed(N, false);

    // --- 1. Fit the Affine Parameters ---
    initial_guess::RotationalAffineVelocity params =
            initial_guess::fit_rotational_affine_velocity(X_ref, V, mass, is_fixed);

    // --- 2. Apply the Affine Predictor ---
    Vec x_hat;
    initial_guess::apply_uniformly_affine_predictor(X_ref, params, dt, x_hat);

    // --- 3. Output Results ---
    std::cout << "--- Affine Predictor Initial Guess Test ---" << std::endl;
    std::cout << "Configuration size: " << N << " nodes" << std::endl;
    std::cout << "Time step (dt): " << dt << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    std::cout << "--- Fitted Parameters ---" << std::endl;
    std::cout << "X_COM: (" << params.X_com.x << ", " << params.X_com.y << ")" << std::endl;
    std::cout << "Omega: " << params.omega << " (Expected: " << omega_true << ")" << std::endl;
    std::cout << "v_hat: (" << params.v_hat.x << ", " << params.v_hat.y << ") (Expected: (" << v_hat_true.x << ", " << v_hat_true.y << "))" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    std::cout << "--- Configuration Update (x^n -> x^hat) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; ++i) {
        Vec2 Xi = getXi(X_ref, i);
        Vec2 Xhat_i = getXi(x_hat, i);
        Vec2 Vi = getXi(V, i);

        std::cout << "Node " << i << ":" << std::endl;
        std::cout << "  X^n:   (" << Xi.x << ", " << Xi.y << ")" << std::endl;
        std::cout << "  V:     (" << Vi.x << ", " << Vi.y << ")" << std::endl;
        std::cout << "  X^hat: (" << Xhat_i.x << ", " << Xhat_i.y << ")" << std::endl;
    }

    return 0;
}
