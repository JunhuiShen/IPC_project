#include "IPC_math.h"
#include "bending_energy.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
//  Utilities
// ===========================================================================

namespace {

constexpr double kB = 0.8;
constexpr double kCe = 1.3;

bool check_convergence(const std::string& label, double analytic,
                       const std::vector<double>& hs,
                       const std::vector<double>& errors,
                       double noise_scale = 1e-10, bool verbose = true) {
    const double noise_floor = noise_scale * (1.0 + std::abs(analytic));
    bool saw_good_slope = false;
    bool all_below_noise = true;
    bool passed = true;

    if (verbose) {
        std::cout << "  " << label
                  << "  analytic=" << std::scientific << std::setprecision(8)
                  << analytic << "\n";
    }

    for (std::size_t i = 1; i < hs.size(); ++i) {
        if (errors[i] < noise_floor || errors[i-1] < noise_floor) {
            if (verbose)
                std::cout << "    h=" << hs[i] << "  err=" << errors[i]
                          << "  (round-off regime, skipped)\n";
            continue;
        }
        all_below_noise = false;
        double slope = std::log(errors[i-1] / errors[i]) / std::log(hs[i-1] / hs[i]);
        if (verbose)
            std::cout << "    h=" << hs[i] << "  err=" << errors[i]
                      << "  slope=" << std::fixed << std::setprecision(2) << slope << "\n";
        if (slope < 1.8) {
            std::cerr << "  FAIL: slope " << slope << " < 1.8 for " << label << "\n";
            passed = false;
        } else {
            saw_good_slope = true;
        }
    }

    if (all_below_noise) {
        if (verbose)
            std::cout << "    (all errors below noise floor -- exact match)\n";
        return true;
    }

    if (!saw_good_slope) {
        std::cerr << "  FAIL: no reliable slope data for " << label << "\n";
        passed = false;
    }
    return passed;
}

HingeDef MakeTestHinge() {
    HingeDef h;
    h.x[0] = Vec3(0.0,  0.0,  0.0);
    h.x[1] = Vec3(1.2,  0.1,  0.05);
    h.x[2] = Vec3(0.3,  0.85, 0.2);
    h.x[3] = Vec3(0.55, -0.9, -0.1);
    return h;
}

HingeDef MakeFlatHinge() {
    HingeDef h;
    h.x[0] = Vec3(0.0, 0.0, 0.0);
    h.x[1] = Vec3(1.0, 0.0, 0.0);
    h.x[2] = Vec3(0.3, 0.8, 0.0);
    h.x[3] = Vec3(0.5, -0.9, 0.0);
    return h;
}

}  // anonymous namespace

// ===========================================================================
//  Test 1: theta and energy are finite
// ===========================================================================

TEST(BendingEnergy, EnergyIsFinite) {
    const auto h = MakeTestHinge();
    const double theta = bending_theta(h);
    ASSERT_TRUE(std::isfinite(theta));
    const double E = bending_energy(h, kB, kCe, 0.1);
    EXPECT_TRUE(std::isfinite(E));
}

// ===========================================================================
//  Test 2: flat configuration gives theta = 0
// ===========================================================================

TEST(BendingEnergy, FlatStateZeroTheta) {
    const auto h = MakeFlatHinge();
    const double theta = bending_theta(h);
    EXPECT_NEAR(theta, 0.0, 1e-14);
}

// ===========================================================================
//  Test 3: rest state (bar_theta = theta) has zero energy and gradient
// ===========================================================================

TEST(BendingEnergy, RestStateZeroEnergyAndGradient) {
    const auto h = MakeTestHinge();
    const double bar_theta = bending_theta(h);
    const double E = bending_energy(h, kB, kCe, bar_theta);
    EXPECT_NEAR(E, 0.0, 1e-14);
    for (int node : {2, 3}) {
        Vec3 g = bending_node_gradient(h, kB, kCe, bar_theta, node);
        EXPECT_LT(g.norm(), 1e-12) << "rest gradient should be zero (node " << node << ")";
    }
}

// ===========================================================================
//  Test 4: translation invariance of energy
// ===========================================================================

TEST(BendingEnergy, TranslationInvariance) {
    const auto h = MakeTestHinge();
    HingeDef shifted = h;
    const Vec3 t(2.5, -1.25, 0.75);
    for (auto& x : shifted.x) x += t;

    const double bar_theta = 0.0;
    const double E0 = bending_energy(h, kB, kCe, bar_theta);
    const double E1 = bending_energy(shifted, kB, kCe, bar_theta);
    EXPECT_NEAR(E0, E1, 1e-12) << "energy should be translation invariant";
}

// ===========================================================================
//  Test 5: rotation invariance of energy
// ===========================================================================

TEST(BendingEnergy, RotationInvariance) {
    const auto h = MakeTestHinge();
    Eigen::AngleAxisd aa(0.7, Vec3(1.0, 2.0, -1.0).normalized());
    const Mat33 R = aa.toRotationMatrix();
    HingeDef rotated = h;
    for (auto& x : rotated.x) x = R * x;

    const double bar_theta = 0.0;
    const double E0 = bending_energy(h, kB, kCe, bar_theta);
    const double E1 = bending_energy(rotated, kB, kCe, bar_theta);
    EXPECT_NEAR(E0, E1, 1e-10) << "energy should be rotation invariant";
}

// ===========================================================================
//  Test 6: Hessian self-block is symmetric
// ===========================================================================

TEST(BendingEnergy, HessianSymmetry) {
    const auto h = MakeTestHinge();
    const double bar_theta = 0.0;
    for (int node : {2, 3}) {
        const Mat33 H = bending_node_hessian(h, kB, kCe, bar_theta, node);
        const double asym = (H - H.transpose()).cwiseAbs().maxCoeff();
        EXPECT_LT(asym, 1e-10) << "Hessian self-block should be symmetric (node " << node << ")";
    }
}

// ===========================================================================
//  Test 7: gradient convergence (slope = 2 in finite differences)
// ===========================================================================

TEST(BendingEnergy, GradientConvergence) {
    const auto h = MakeTestHinge();
    const double bar_theta = 0.0;

    const std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;

    for (int node : {0, 1, 2, 3}) {
        const Vec3 g = bending_node_gradient(h, kB, kCe, bar_theta, node);
        for (int c = 0; c < 3; ++c) {
            const double analytic = g(c);
            if (std::abs(analytic) < 1e-14) continue;

            std::vector<double> errors;
            for (auto hh : hs) {
                HingeDef dp = h, dm = h;
                dp.x[node](c) += hh;
                dm.x[node](c) -= hh;
                const double fd = (bending_energy(dp, kB, kCe, bar_theta)
                                   - bending_energy(dm, kB, kCe, bar_theta)) / (2.0 * hh);
                errors.push_back(std::abs(fd - analytic));
            }
            const std::string label = "g[node" + std::to_string(node) + "](" + std::to_string(c) + ")";
            if (!check_convergence(label, analytic, hs, errors)) all_passed = false;
        }
    }
    EXPECT_TRUE(all_passed) << "gradient convergence failed";
}

// ===========================================================================
//  Test 8: Hessian self-block convergence (slope = 2 in finite differences)
// ===========================================================================

TEST(BendingEnergy, HessianConvergence) {
    const auto h = MakeTestHinge();
    const double bar_theta = 0.0;

    const std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;
    int tested = 0, skipped = 0;

    for (int node : {2, 3}) {
        const Mat33 H = bending_node_hessian(h, kB, kCe, bar_theta, node);

        // For each perturbation direction d at this node, collect the
        // per-node gradient so we can finite-difference it.
        for (int d = 0; d < 3; ++d) {
            std::vector<Vec3> g_plus(hs.size()), g_minus(hs.size());
            for (std::size_t hi = 0; hi < hs.size(); ++hi) {
                HingeDef dp = h, dm = h;
                dp.x[node](d) += hs[hi];
                dm.x[node](d) -= hs[hi];
                g_plus[hi]  = bending_node_gradient(dp, kB, kCe, bar_theta, node);
                g_minus[hi] = bending_node_gradient(dm, kB, kCe, bar_theta, node);
            }
            for (int c = 0; c < 3; ++c) {
                const double analytic = H(c, d);

                if (std::abs(analytic) < 1e-14) {
                    const double fd_fine = (g_plus.back()(c) - g_minus.back()(c)) / (2.0 * hs.back());
                    if (std::abs(fd_fine) > 1e-6) {
                        std::cerr << "  FAIL: H[node" << node << "](" << c << "," << d
                                  << ") analytic=0 but fd=" << fd_fine << "\n";
                        all_passed = false;
                    }
                    skipped++;
                    continue;
                }

                std::vector<double> errors;
                for (std::size_t hi = 0; hi < hs.size(); ++hi) {
                    const double fd = (g_plus[hi](c) - g_minus[hi](c)) / (2.0 * hs[hi]);
                    errors.push_back(std::abs(fd - analytic));
                }
                const std::string label = "H[node" + std::to_string(node) + "]("
                                          + std::to_string(c) + "," + std::to_string(d) + ")";
                if (!check_convergence(label, analytic, hs, errors, 1e-9, false)) {
                    check_convergence(label, analytic, hs, errors, 1e-9, true);
                    all_passed = false;
                }
                tested++;
            }
        }
    }

    std::cout << "  Tested " << tested << " entries, skipped " << skipped << " zero entries\n";
    EXPECT_TRUE(all_passed) << "Hessian convergence failed";
}

// ===========================================================================
//  Test 9: directional derivative convergence (slope = 2)
// ===========================================================================

TEST(BendingEnergy, DirectionalDerivativeConvergence) {
    const auto h = MakeTestHinge();
    const double bar_theta = 0.0;

    Vec3 dx(0.4, -0.6, 0.3);
    dx.normalize();

    const std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};

    for (int node : {0, 1, 2, 3}) {
        const Vec3 g = bending_node_gradient(h, kB, kCe, bar_theta, node);
        const double exact = g.dot(dx);

        std::vector<double> errors;
        for (auto hh : hs) {
            HingeDef dp = h, dm = h;
            dp.x[node] += hh * dx;
            dm.x[node] -= hh * dx;
            const double fd = (bending_energy(dp, kB, kCe, bar_theta)
                               - bending_energy(dm, kB, kCe, bar_theta)) / (2.0 * hh);
            errors.push_back(std::abs(fd - exact));
        }
        const std::string label = "directional[node" + std::to_string(node) + "]";
        EXPECT_TRUE(check_convergence(label, exact, hs, errors))
            << "directional derivative convergence failed (node " << node << ")";
    }
}

// ===========================================================================
//  Test 10: rest-state Hessian self-block equals the Gauss-Newton term
// ===========================================================================

TEST(BendingEnergy, RestHessianMatchesGaussNewton) {
    // At the rest state delta = 0, so the Hessian reduces to
    //   2 k_B c_e * (theta_{,p} theta_{,q}) = 2 k_B c_e * (ell^2 / Q^2) * m m^T.
    const auto h = MakeTestHinge();
    const double bar_theta = bending_theta(h);

    for (int node : {2, 3}) {
        Vec3 e, a;
        if (node == 2) {
            e = h.x[1] - h.x[0];
            a = h.x[2] - h.x[0];
        } else {
            e = h.x[0] - h.x[1];
            a = h.x[3] - h.x[1];
        }
        const Vec3 m = e.cross(a);
        const double Q = m.squaredNorm();
        const double ell = e.norm();
        const Mat33 expected = (2.0 * kB * kCe * ell * ell / (Q * Q)) * (m * m.transpose());

        const Mat33 H = bending_node_hessian(h, kB, kCe, bar_theta, node);
        const double err = (H - expected).cwiseAbs().maxCoeff();
        EXPECT_LT(err, 1e-12) << "rest Hessian mismatch (node " << node << ")";
    }
}

// ===========================================================================
//  Test 11: known fold angle -- both apex vertices rotated about the edge
// ===========================================================================

TEST(BendingEnergy, KnownFoldAngle) {
    // Place the edge along x, the flat triangles at y = +/- 1, then rotate
    // both apex vertices about the x-axis by alpha. The resulting dihedral
    // change is 2*alpha, and with our sign convention theta = -2*alpha.
    for (double alpha : {0.05, 0.3, 0.7, 1.2, -0.4, -1.1}) {
        HingeDef h;
        h.x[0] = Vec3(0.0, 0.0, 0.0);
        h.x[1] = Vec3(1.0, 0.0, 0.0);
        h.x[2] = Vec3(0.5,  std::cos(alpha),  std::sin(alpha));
        h.x[3] = Vec3(0.5, -std::cos(alpha),  std::sin(alpha));

        const double theta = bending_theta(h);
        EXPECT_NEAR(theta, -2.0 * alpha, 1e-12)
            << "theta mismatch for alpha=" << alpha;
    }
}

// ===========================================================================
//  Test 12: uniform scale invariance -- theta is a pure angle
// ===========================================================================

TEST(BendingEnergy, ScaleInvariance) {
    const auto h = MakeTestHinge();
    const double theta_ref = bending_theta(h);

    for (double s : {0.01, 0.1, 10.0, 1000.0}) {
        HingeDef hs = h;
        for (auto& x : hs.x) x *= s;
        const double theta_s = bending_theta(hs);
        EXPECT_NEAR(theta_s, theta_ref, 1e-12)
            << "theta should be scale-invariant, scale=" << s;
    }
}

// ===========================================================================
//  Test 13: energy is exactly k_B * c_e * (theta - bar_theta)^2
// ===========================================================================

TEST(BendingEnergy, EnergyQuadraticFormula) {
    const auto h = MakeTestHinge();
    const double theta = bending_theta(h);
    for (double bar : {-1.0, -0.25, 0.0, 0.25, 1.0}) {
        const double delta = theta - bar;
        const double expected = kB * kCe * delta * delta;
        const double E = bending_energy(h, kB, kCe, bar);
        EXPECT_NEAR(E, expected, 1e-14 * (1.0 + std::abs(expected)))
            << "energy mismatch for bar_theta=" << bar;
    }
}

// ===========================================================================
//  Test 14: energy is invariant under a large global translation
// ===========================================================================

TEST(BendingEnergy, LargeCoordinateOffset) {
    const auto h = MakeTestHinge();
    const double bar_theta = 0.0;
    const double E_ref = bending_energy(h, kB, kCe, bar_theta);
    const double theta_ref = bending_theta(h);

    // Large offset exercises floating-point cancellation in the
    // subtractions x[1]-x[0], x[2]-x[0], x[3]-x[0].
    for (double offset : {1.0e3, 1.0e6}) {
        HingeDef hs = h;
        for (auto& x : hs.x) x += Vec3(offset, offset, offset);

        const double theta = bending_theta(hs);
        const double E = bending_energy(hs, kB, kCe, bar_theta);
        const double tol = 1.0e-12 * (1.0 + offset);
        EXPECT_NEAR(theta, theta_ref, tol) << "theta shifted by offset=" << offset;
        EXPECT_NEAR(E, E_ref, tol) << "energy shifted by offset=" << offset;

        for (int node : {2, 3}) {
            Vec3 g = bending_node_gradient(hs, kB, kCe, bar_theta, node);
            ASSERT_TRUE(std::isfinite(g.norm())) << "grad not finite at offset=" << offset;
            Mat33 H = bending_node_hessian(hs, kB, kCe, bar_theta, node);
            ASSERT_TRUE(std::isfinite(H.norm())) << "Hessian not finite at offset=" << offset;
        }
    }
}

// ===========================================================================
//  Test 15: near-degenerate triangle -- theta, gradient, Hessian finite
// ===========================================================================

TEST(BendingEnergy, NearDegenerateTriangleA) {
    // Triangle A has vanishing area (x_2 is nearly on the edge line), while
    // triangle B is well-formed. The code should stay finite and the sign
    // conventions should still match finite differences.
    HingeDef h;
    h.x[0] = Vec3(0.0, 0.0, 0.0);
    h.x[1] = Vec3(1.0, 0.0, 0.0);
    h.x[2] = Vec3(0.5, 1.0e-3, 0.0);
    h.x[3] = Vec3(0.5, -0.9, -0.1);

    const double theta = bending_theta(h);
    ASSERT_TRUE(std::isfinite(theta));

    const double bar_theta = theta;  // keep delta small so we stay in the
                                     // regime where the 1/Q factor does not
                                     // blow up the energy gradient
    const double E = bending_energy(h, kB, kCe, bar_theta);
    EXPECT_NEAR(E, 0.0, 1e-14);

    for (int node : {2, 3}) {
        Vec3 g = bending_node_gradient(h, kB, kCe, bar_theta, node);
        ASSERT_TRUE(std::isfinite(g.norm())) << "grad not finite (node " << node << ")";
        EXPECT_LT(g.norm(), 1e-10) << "rest grad should be zero (node " << node << ")";

        Mat33 H = bending_node_hessian(h, kB, kCe, bar_theta, node);
        ASSERT_TRUE(std::isfinite(H.norm())) << "Hessian not finite (node " << node << ")";
    }

    // A small perturbation of bar_theta gives a finite-but-large gradient
    // at node 2 (Q is small). Just check finiteness.
    const double bar_small = bar_theta - 1.0e-6;
    Vec3 g2 = bending_node_gradient(h, kB, kCe, bar_small, 2);
    ASSERT_TRUE(std::isfinite(g2.norm()));
}

// ===========================================================================
//  Test 16: zero-length hinge edge -- degenerate but handled gracefully
// ===========================================================================

TEST(BendingEnergy, ZeroLengthHingeEdge) {
    HingeDef h;
    h.x[0] = Vec3(0.0, 0.0, 0.0);
    h.x[1] = Vec3(0.0, 0.0, 0.0);  // collapsed onto x_0
    h.x[2] = Vec3(0.5, 1.0, 0.0);
    h.x[3] = Vec3(0.5, -1.0, 0.0);

    const double theta = bending_theta(h);
    EXPECT_EQ(theta, 0.0);
    const double E = bending_energy(h, kB, kCe, 0.0);
    EXPECT_EQ(E, 0.0);
    for (int node : {0, 1, 2, 3}) {
        Vec3 g = bending_node_gradient(h, kB, kCe, 0.0, node);
        EXPECT_LT(g.norm(), 1e-30);
        Mat33 H_psd = bending_node_hessian_psd(h, kB, kCe, 0.0, node);
        EXPECT_LT(H_psd.norm(), 1e-30);
    }
    for (int node : {2, 3}) {
        Mat33 H = bending_node_hessian(h, kB, kCe, 0.0, node);
        EXPECT_LT(H.norm(), 1e-30);
    }
}

// ===========================================================================
//  Test 17: gradient sums to zero (translation null mode)
// ===========================================================================

TEST(BendingEnergy, GradientSumZero) {
    const auto h = MakeTestHinge();
    for (double bar_theta : {-0.2, 0.0, 0.3}) {
        Vec3 total = Vec3::Zero();
        for (int node = 0; node < 4; ++node)
            total += bending_node_gradient(h, kB, kCe, bar_theta, node);
        EXPECT_LT(total.norm(), 1e-12)
            << "gradient should sum to zero, bar_theta=" << bar_theta;
    }
}

// ===========================================================================
//  Test 18: PSD Hessian matches 2*k_B*c_e*(grad theta)(grad theta)^T
// ===========================================================================

TEST(BendingEnergy, PsdHessianMatchesGradientSquared) {
    const auto h = MakeTestHinge();
    // Use bar_theta != theta so the energy gradient != 0, then check that
    // H_psd equals the outer product of the theta gradient (= energy
    // gradient divided by 2 k_B c_e delta).
    const double bar_theta = 0.0;
    const double theta = bending_theta(h);
    const double delta = theta - bar_theta;
    ASSERT_GT(std::abs(delta), 1e-6);

    for (int node = 0; node < 4; ++node) {
        const Vec3 g = bending_node_gradient(h, kB, kCe, bar_theta, node);
        const Vec3 g_theta = g / (2.0 * kB * kCe * delta);
        const Mat33 H_expected = (2.0 * kB * kCe) * (g_theta * g_theta.transpose());
        const Mat33 H_psd = bending_node_hessian_psd(h, kB, kCe, bar_theta, node);
        const double err = (H_psd - H_expected).cwiseAbs().maxCoeff();
        EXPECT_LT(err, 1e-12) << "PSD Hessian mismatch at node " << node;

        // Symmetry
        EXPECT_LT((H_psd - H_psd.transpose()).cwiseAbs().maxCoeff(), 1e-14);

        // Positive semidefinite: rank-1 outer product always has >= 0 eigenvalues.
        Eigen::SelfAdjointEigenSolver<Mat33> es(H_psd);
        EXPECT_GE(es.eigenvalues().minCoeff(), -1e-12)
            << "PSD Hessian has negative eigenvalue at node " << node;
    }
}

// ===========================================================================
//  Test 19: PSD Hessian matches exact at rest (delta = 0) for nodes 2, 3
// ===========================================================================

TEST(BendingEnergy, PsdHessianMatchesExactAtRest) {
    const auto h = MakeTestHinge();
    const double bar_theta = bending_theta(h);  // delta = 0
    for (int node : {2, 3}) {
        const Mat33 H_exact = bending_node_hessian(h, kB, kCe, bar_theta, node);
        const Mat33 H_psd   = bending_node_hessian_psd(h, kB, kCe, bar_theta, node);
        const double err = (H_exact - H_psd).cwiseAbs().maxCoeff();
        EXPECT_LT(err, 1e-12)
            << "PSD should equal exact at delta=0, node " << node;
    }
}

// ===========================================================================
//  Test 20: PSD Hessian FD convergence at all 4 nodes, near the rest state
// ===========================================================================
//
// At the rest state (delta = 0) the PSD Hessian equals the true Hessian, so
// finite differences of the gradient converge to H_psd at slope 2. This
// gives an indirect consistency check for the gradients of nodes 0 and 1.

TEST(BendingEnergy, PsdHessianConvergenceAtRest) {
    const auto h = MakeTestHinge();
    const double bar_theta = bending_theta(h);

    const std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;

    for (int node : {0, 1, 2, 3}) {
        const Mat33 H = bending_node_hessian_psd(h, kB, kCe, bar_theta, node);

        for (int d = 0; d < 3; ++d) {
            std::vector<Vec3> g_plus(hs.size()), g_minus(hs.size());
            for (std::size_t hi = 0; hi < hs.size(); ++hi) {
                HingeDef dp = h, dm = h;
                dp.x[node](d) += hs[hi];
                dm.x[node](d) -= hs[hi];
                g_plus[hi]  = bending_node_gradient(dp, kB, kCe, bar_theta, node);
                g_minus[hi] = bending_node_gradient(dm, kB, kCe, bar_theta, node);
            }
            for (int c = 0; c < 3; ++c) {
                const double analytic = H(c, d);
                if (std::abs(analytic) < 1e-14) continue;
                std::vector<double> errors;
                for (std::size_t hi = 0; hi < hs.size(); ++hi) {
                    const double fd = (g_plus[hi](c) - g_minus[hi](c)) / (2.0 * hs[hi]);
                    errors.push_back(std::abs(fd - analytic));
                }
                const std::string label = "H_psd[node" + std::to_string(node) + "]("
                                          + std::to_string(c) + "," + std::to_string(d) + ")";
                if (!check_convergence(label, analytic, hs, errors, 1e-9, false)) {
                    check_convergence(label, analytic, hs, errors, 1e-9, true);
                    all_passed = false;
                }
            }
        }
    }
    EXPECT_TRUE(all_passed) << "PSD Hessian convergence at rest failed";
}
