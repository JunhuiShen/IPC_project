#include "sdf_penalty_energy.h"

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-8;

//  Finite-difference gradient of a scalar energy E(x).
template <typename F>
Vec3 fd_gradient(const F& energy, const Vec3& x, double h){
    Vec3 g;
    for (int k = 0; k < 3; ++k) {
        Vec3 xp = x; xp(k) += h;
        Vec3 xm = x; xm(k) -= h;
        g(k) = (energy(xp) - energy(xm)) / (2.0 * h);
    }
    return g;
}

//  Finite-difference Jacobian of a vector function grad(x).
template <typename G>
Mat33 fd_hessian(const G& gradient, const Vec3& x, double h){
    Mat33 H;
    for (int j = 0; j < 3; ++j) {
        Vec3 xp = x; xp(j) += h;
        Vec3 xm = x; xm(j) -= h;
        const Vec3 gp = gradient(xp);
        const Vec3 gm = gradient(xm);
        const Vec3 col = (gp - gm) / (2.0 * h);
        for (int i = 0; i < 3; ++i) H(i, j) = col(i);
    }
    return H;
}

//  Slope test for central-difference FD.  Copied from barrier_energy_test.cpp
//  per project convention (each energy-test file keeps its own copy).
bool check_convergence(const std::string& label, double analytic,
                       const std::vector<double>& hs,
                       const std::vector<double>& errors,
                       double noise_scale = 1e-10, bool verbose = true){
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
        double slope = std::log(errors[i-1]/errors[i]) / std::log(hs[i-1]/hs[i]);
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
        if (verbose) std::cout << "    (all errors below noise floor -- exact match)\n";
        return true;
    }
    if (!saw_good_slope) {
        std::cerr << "  FAIL: no reliable slope data for " << label << "\n";
        passed = false;
    }
    return passed;
}

using SDFClosure = std::function<SDFEvaluation(const Vec3&)>;

//  Gradient convergence test: dE/dx_k  vs  central-difference of E.
bool run_sdf_gradient_convergence(const std::string& name, const Vec3& x,
                                  const SDFClosure& sdf, double k, double eps){
    std::cout << "=== Gradient convergence: " << name << " ===\n";
    const std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    const auto energy = [&](const Vec3& q){ return sdf_penalty_energy(sdf(q), k, eps); };

    const Vec3 g_ana = sdf_penalty_gradient(sdf(x), k, eps);
    bool all_passed = true;

    for (int comp = 0; comp < 3; ++comp) {
        const double analytic = g_ana(comp);

        if (std::abs(analytic) < 1e-14) {
            const double fd_fine = fd_gradient(energy, x, hs.back())(comp);
            if (std::abs(fd_fine) > 1e-8) {
                std::cerr << "  FAIL: dE/dx_" << comp
                          << " analytic=0 but fd=" << fd_fine << "\n";
                all_passed = false;
            }
            continue;
        }

        std::vector<double> errors;
        for (double h : hs)
            errors.push_back(std::abs(fd_gradient(energy, x, h)(comp) - analytic));

        const std::string label = std::string("dE/dx_") + std::to_string(comp);
        if (!check_convergence(label, analytic, hs, errors, 1e-10)) all_passed = false;
    }

    std::cout << (all_passed ? "  PASSED\n\n" : "  FAILED\n\n");
    return all_passed;
}

//  Hessian convergence test: d^2E/dx_k dx_l  vs  central-difference of grad.
bool run_sdf_hessian_convergence(const std::string& name, const Vec3& x,
                                 const SDFClosure& sdf, double k, double eps,
                                 bool include_curvature){
    std::cout << "=== Hessian convergence: " << name
              << " (curvature=" << (include_curvature ? "true" : "false") << ") ===\n";
    const std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    const auto gradient = [&](const Vec3& q){ return sdf_penalty_gradient(sdf(q), k, eps); };

    const Mat33 H_ana = sdf_penalty_hessian(sdf(x), k, eps, include_curvature);
    bool all_passed = true;
    int tested = 0, skipped_zero = 0;

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double analytic = H_ana(row, col);

            if (std::abs(analytic) < 1e-14) {
                const double fd_fine = fd_hessian(gradient, x, hs.back())(row, col);
                if (std::abs(fd_fine) > 1e-6) {
                    std::cerr << "  FAIL: H(" << row << "," << col
                              << ") analytic=0 but fd=" << fd_fine << "\n";
                    all_passed = false;
                }
                skipped_zero++;
                continue;
            }

            std::vector<double> errors;
            for (double h : hs)
                errors.push_back(std::abs(fd_hessian(gradient, x, h)(row, col) - analytic));

            const std::string label = std::string("H(") + std::to_string(row) + ","
                                    + std::to_string(col) + ")";
            if (!check_convergence(label, analytic, hs, errors, 1e-9, false)) {
                check_convergence(label, analytic, hs, errors, 1e-9, true);
                all_passed = false;
            }
            tested++;
        }
    }

    std::cout << "  Tested " << tested << " entries, skipped " << skipped_zero << " zero entries\n";
    std::cout << (all_passed ? "  PASSED\n\n" : "  FAILED\n\n");
    return all_passed;
}

}  // namespace

// ============================================================================
//  Heaviside
// ============================================================================

TEST(SDFHeaviside, PiecewiseValues){
    const double eps = 0.1;
    EXPECT_EQ(sdf_heaviside(-1.0, eps), 1.0);
    EXPECT_EQ(sdf_heaviside( 0.0, eps), 1.0);          //  (eps-0)/eps = 1
    EXPECT_NEAR(sdf_heaviside(0.05, eps), 0.5, kTol);
    EXPECT_NEAR(sdf_heaviside(eps,  eps), 0.0, kTol);
    EXPECT_EQ(sdf_heaviside(2.0 * eps, eps), 0.0);
}

TEST(SDFHeaviside, GradientPiecewise){
    const double eps = 0.1;
    EXPECT_EQ(sdf_heaviside_gradient(-1.0, eps), 0.0);
    EXPECT_EQ(sdf_heaviside_gradient( 0.0, eps), 0.0);   //  boundary
    EXPECT_NEAR(sdf_heaviside_gradient(0.05, eps), -1.0/eps, kTol);
    EXPECT_EQ(sdf_heaviside_gradient(eps,  eps), 0.0);   //  boundary
    EXPECT_EQ(sdf_heaviside_gradient(2.0 * eps, eps), 0.0);
}

TEST(SDFHeaviside, RejectsBadEps){
    EXPECT_THROW(sdf_heaviside(0.5, 0.0),          std::runtime_error);
    EXPECT_THROW(sdf_heaviside_gradient(0.5, -1.0), std::runtime_error);
}

// ============================================================================
//  Plane SDF
// ============================================================================

TEST(PlaneSDF, Evaluate){
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};

    SDFEvaluation r = evaluate_sdf(p, Vec3(3.0, 2.0, -5.0));
    EXPECT_NEAR(r.phi, 2.0, kTol);
    EXPECT_TRUE(r.grad_phi.isApprox(Vec3(0.0, 1.0, 0.0)));
    EXPECT_TRUE(r.hess_phi.isApprox(Mat33::Zero()));

    r = evaluate_sdf(p, Vec3(0.0, -0.3, 0.0));
    EXPECT_NEAR(r.phi, -0.3, kTol);
}

TEST(PlaneSDF, EnergyOutsideTransitionIsZero){
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    const double k = 100.0;
    const double eps = 0.1;

    //  Far outside (phi > eps): zero energy, zero gradient, zero Hessian.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, 1.0, 0.0));
        EXPECT_EQ(sdf_penalty_energy(sdf, k, eps), 0.0);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero()));
        EXPECT_TRUE(sdf_penalty_hessian(sdf, k, eps).isApprox(Mat33::Zero()));
    }

    //  Deep inside (phi < 0): energy k/2, but gradient and Hessian vanish.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, -1.0, 0.0));
        EXPECT_NEAR(sdf_penalty_energy(sdf, k, eps), 0.5 * k, kTol);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero()));
        EXPECT_TRUE(sdf_penalty_hessian(sdf, k, eps).isApprox(Mat33::Zero()));
    }
}

TEST(PlaneSDF, EnergyInTransitionLayer){
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    const double k = 50.0;
    const double eps = 0.2;

    //  phi = 0.05, H = (0.2 - 0.05)/0.2 = 0.75, E = 0.5 * 50 * 0.75^2
    Vec3 x(1.0, 0.05, -2.0);
    const double expected = 0.5 * k * (0.75 * 0.75);
    EXPECT_NEAR(sdf_penalty_energy(evaluate_sdf(p, x), k, eps), expected, kTol);
}

TEST(PlaneSDF, GradientConvergence){
    //  Tilted plane, phi_target = 0.15 in eps = 0.3 -- 0.15 slack on both sides,
    //  safe for h up to 1e-2.
    Vec3 n = Vec3(1.0, 2.0, 3.0).normalized();
    PlaneSDF p{Vec3(0.1, -0.2, 0.05), n};
    const double k = 25.0;
    const double eps = 0.3;

    Vec3 x = p.point + 0.15 * n + Vec3(0.4, -0.3, 0.2);
    x += (0.15 - (x - p.point).dot(n)) * n;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(p, q); };
    EXPECT_TRUE(run_sdf_gradient_convergence("plane", x, sdf, k, eps));
}

TEST(PlaneSDF, HessianConvergence){
    Vec3 n = Vec3(-1.0, 1.0, 2.0).normalized();
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), n};
    const double k = 40.0;
    const double eps = 0.25;

    Vec3 x = 0.10 * n + Vec3(0.5, 0.5, -0.1);
    x += (0.10 - (x - p.point).dot(n)) * n;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(p, q); };
    EXPECT_TRUE(run_sdf_hessian_convergence("plane", x, sdf, k, eps, /*include_curvature=*/true));
}

TEST(PlaneSDF, HessianIsRankOne){
    Vec3 n = Vec3(-1.0, 1.0, 2.0).normalized();
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), n};
    const double k = 40.0;
    const double eps = 0.25;

    Vec3 x = 0.10 * n + Vec3(0.5, 0.5, -0.1);
    x += (0.10 - (x - p.point).dot(n)) * n;

    const Mat33 H_ana = sdf_penalty_hessian(evaluate_sdf(p, x), k, eps);

    //  Hessian should be k/eps^2 * n n^T -- rank one and symmetric.
    EXPECT_TRUE(H_ana.isApprox(H_ana.transpose(), kTol));
    const Mat33 expected = (k / (eps * eps)) * (n * n.transpose());
    EXPECT_TRUE(H_ana.isApprox(expected, kTol));

    Eigen::SelfAdjointEigenSolver<Mat33> es(H_ana);
    const Vec3 lambda = es.eigenvalues();
    EXPECT_NEAR(lambda(0), 0.0,           1e-6);
    EXPECT_NEAR(lambda(1), 0.0,           1e-6);
    EXPECT_NEAR(lambda(2), k/(eps*eps),   1e-6);
}

// ============================================================================
//  Cylinder SDF
// ============================================================================

TEST(CylinderSDF, Evaluate){
    //  Axis along z through origin, radius 0.5.
    CylinderSDF c{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), 0.5};

    //  Point at (0.6, 0, 10) -- off-axis distance 0.6, phi = 0.1.
    SDFEvaluation r = evaluate_sdf(c, Vec3(0.6, 0.0, 10.0));
    EXPECT_NEAR(r.phi, 0.1, kTol);
    EXPECT_TRUE(r.grad_phi.isApprox(Vec3(1.0, 0.0, 0.0)));
    //  hess_phi should be (I - a a^T - n n^T)/r_perp = e_y e_y^T / 0.6.
    Mat33 expected = Mat33::Zero();
    expected(1, 1) = 1.0 / 0.6;
    EXPECT_TRUE(r.hess_phi.isApprox(expected, kTol));

    //  Point on the axis: phi = -R, derivatives degenerate -> zero sentinels.
    r = evaluate_sdf(c, Vec3(0.0, 0.0, 3.0));
    EXPECT_NEAR(r.phi, -0.5, kTol);
    EXPECT_TRUE(r.grad_phi.isApprox(Vec3::Zero()));
    EXPECT_TRUE(r.hess_phi.isApprox(Mat33::Zero()));
}

TEST(CylinderSDF, GradientConvergence){
    //  Tilted axis so components mix.
    CylinderSDF c{Vec3(0.1, -0.2, 0.05),
                  Vec3(1.0, 2.0, -1.0).normalized(),
                  0.4};
    const double k = 30.0;
    const double eps = 0.2;

    //  Pick a point with perpendicular distance = R + 0.08, well inside the
    //  transition layer.  Choose a radial direction orthogonal to the axis.
    const Vec3 raw_dir(3.0, -1.0, 2.0);
    const Vec3 perp_dir = (raw_dir - c.axis * c.axis.dot(raw_dir)).normalized();
    const Vec3 x = c.point + (c.radius + 0.08) * perp_dir
                   + 0.5 * c.axis;   //  arbitrary axial offset

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(c, q); };
    EXPECT_TRUE(run_sdf_gradient_convergence("cylinder", x, sdf, k, eps));
}

TEST(CylinderSDF, HessianConvergence){
    CylinderSDF c{Vec3(0.0, 0.0, 0.0),
                  Vec3(0.0, 1.0, 2.0).normalized(),
                  0.4};
    const double k = 20.0;
    const double eps = 0.2;

    const Vec3 raw_dir(1.0, 2.0, 3.0);
    const Vec3 perp_dir = (raw_dir - c.axis * c.axis.dot(raw_dir)).normalized();
    const Vec3 x = c.point + (c.radius + 0.08) * perp_dir
                   + 0.3 * c.axis;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(c, q); };
    EXPECT_TRUE(run_sdf_hessian_convergence("cylinder", x, sdf, k, eps,
                                            /*include_curvature=*/true));
}

// ============================================================================
//  Sphere SDF
// ============================================================================

TEST(SphereSDF, Evaluate){
    //  Unit-ish sphere at origin.
    SphereSDF s{Vec3(0.0, 0.0, 0.0), 0.5};

    //  Point on +x at distance 0.8 -- phi = 0.3, n = (1,0,0).
    SDFEvaluation r = evaluate_sdf(s, Vec3(0.8, 0.0, 0.0));
    EXPECT_NEAR(r.phi, 0.3, kTol);
    EXPECT_TRUE(r.grad_phi.isApprox(Vec3(1.0, 0.0, 0.0)));
    //  hess_phi = (I - n n^T) / 0.8 -- diag(0, 1/0.8, 1/0.8).
    Mat33 expected = Mat33::Zero();
    expected(1, 1) = 1.0 / 0.8;
    expected(2, 2) = 1.0 / 0.8;
    EXPECT_TRUE(r.hess_phi.isApprox(expected, kTol));

    //  At the center: phi = -R, derivatives degenerate -> zero sentinels.
    r = evaluate_sdf(s, Vec3(0.0, 0.0, 0.0));
    EXPECT_NEAR(r.phi, -0.5, kTol);
    EXPECT_TRUE(r.grad_phi.isApprox(Vec3::Zero()));
    EXPECT_TRUE(r.hess_phi.isApprox(Mat33::Zero()));
}

TEST(SphereSDF, EnergyOutsideTransitionIsZero){
    SphereSDF s{Vec3(0.0, 0.0, 0.0), 0.5};
    const double k = 100.0;
    const double eps = 0.1;

    //  Far outside (phi > eps): zero energy, zero gradient, zero Hessian.
    {
        const SDFEvaluation sdf = evaluate_sdf(s, Vec3(2.0, 0.0, 0.0));
        EXPECT_EQ(sdf_penalty_energy(sdf, k, eps), 0.0);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero()));
        EXPECT_TRUE(sdf_penalty_hessian(sdf, k, eps).isApprox(Mat33::Zero()));
    }

    //  Deep inside (phi < 0): energy k/2, but gradient and Hessian vanish.
    {
        const SDFEvaluation sdf = evaluate_sdf(s, Vec3(0.1, 0.0, 0.0));
        EXPECT_NEAR(sdf_penalty_energy(sdf, k, eps), 0.5 * k, kTol);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero()));
        EXPECT_TRUE(sdf_penalty_hessian(sdf, k, eps).isApprox(Mat33::Zero()));
    }
}

TEST(SphereSDF, EnergyInTransitionLayer){
    SphereSDF s{Vec3(0.0, 0.0, 0.0), 0.5};
    const double k = 50.0;
    const double eps = 0.2;

    //  ||x|| = 0.55, phi = 0.05, H = (0.2 - 0.05)/0.2 = 0.75, E = 0.5 * 50 * 0.75^2.
    Vec3 x(0.55, 0.0, 0.0);
    const double expected = 0.5 * k * (0.75 * 0.75);
    EXPECT_NEAR(sdf_penalty_energy(evaluate_sdf(s, x), k, eps), expected, kTol);
}

TEST(SphereSDF, HessianStructure){
    //  Sphere penalty Hessian = k*(H')^2 * n n^T + k*H*H' * (I - n n^T)/r.
    //  Normal eigenvalue = k*(H')^2; transverse (2x) = k*H*H'/r (negative in
    //  the ramp since H' < 0).  Differs from the plane (rank 1, no curvature).
    SphereSDF s{Vec3(0.0, 0.0, 0.0), 0.5};
    const double k = 40.0;
    const double eps = 0.25;

    //  r = 0.6, phi = 0.1, H = (eps - phi)/eps = 0.6, H' = -1/eps = -4.
    const Vec3 n_dir = Vec3(1.0, 2.0, -1.0).normalized();
    const double r    = 0.6;
    const Vec3 x      = r * n_dir;
    const double phi  = r - 0.5;          // 0.1
    const double H    = (eps - phi) / eps; // 0.6
    const double Hp   = -1.0 / eps;        // -4

    const Mat33 H_ana = sdf_penalty_hessian(evaluate_sdf(s, x), k, eps);
    const Mat33 expected =
            (k * Hp * Hp) * (n_dir * n_dir.transpose())
          + (k * H * Hp / r) * (Mat33::Identity() - n_dir * n_dir.transpose());

    EXPECT_TRUE(H_ana.isApprox(H_ana.transpose(), kTol));
    EXPECT_TRUE(H_ana.isApprox(expected, kTol));

    //  Eigenvalues: one along n (= k*Hp^2), two transverse (= k*H*Hp/r).
    Eigen::SelfAdjointEigenSolver<Mat33> es(H_ana);
    const Vec3 lambda = es.eigenvalues();   // ascending order
    const double lam_normal = k * Hp * Hp;  // +640
    const double lam_trans  = k * H * Hp / r; // -160
    EXPECT_NEAR(lambda(0), lam_trans, 1e-6);
    EXPECT_NEAR(lambda(1), lam_trans, 1e-6);
    EXPECT_NEAR(lambda(2), lam_normal, 1e-6);
}

TEST(SphereSDF, GradientConvergence){
    SphereSDF s{Vec3(0.1, -0.2, 0.05), 0.4};
    const double k = 30.0;
    const double eps = 0.2;

    //  Point with ||x - center|| = R + 0.08 -- phi = 0.08 sits mid-transition
    //  layer, safe for h up to 1e-2 without leaving the ramp.
    const Vec3 dir = Vec3(3.0, -1.0, 2.0).normalized();
    const Vec3 x = s.center + (s.radius + 0.08) * dir;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(s, q); };
    EXPECT_TRUE(run_sdf_gradient_convergence("sphere", x, sdf, k, eps));
}

TEST(SphereSDF, HessianConvergence){
    SphereSDF s{Vec3(0.1, -0.2, 0.05), 0.4};
    const double k = 30.0;
    const double eps = 0.2;

    const Vec3 dir = Vec3(3.0, -1.0, 2.0).normalized();
    const Vec3 x = s.center + (s.radius + 0.08) * dir;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(s, q); };
    EXPECT_TRUE(run_sdf_hessian_convergence("sphere", x, sdf, k, eps,
                                            /*include_curvature=*/true));
}

//  Direct slope-2 test of the SDF itself (bypasses the penalty pipeline):
//  FD of phi  vs analytic grad_phi, and FD of grad_phi vs analytic hess_phi.
TEST(SphereSDF, SdfDerivativesConvergence){
    SphereSDF s{Vec3(0.1, -0.2, 0.05), 0.4};

    //  Off-center point so phi, grad_phi, hess_phi are all nondegenerate.
    const Vec3 x = s.center + Vec3(0.30, 0.15, -0.20);

    const auto phi      = [&](const Vec3& q){ return evaluate_sdf(s, q).phi; };
    const auto grad_phi = [&](const Vec3& q){ return evaluate_sdf(s, q).grad_phi; };

    const SDFEvaluation e_ana = evaluate_sdf(s, x);
    const std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};

    //  grad_phi vs central-difference of phi
    std::cout << "=== SDF gradient convergence: sphere (phi) ===\n";
    bool grad_ok = true;
    for (int k = 0; k < 3; ++k) {
        std::vector<double> errors;
        for (double h : hs)
            errors.push_back(std::abs(fd_gradient(phi, x, h)(k) - e_ana.grad_phi(k)));
        const std::string label = std::string("dphi/dx_") + std::to_string(k);
        if (!check_convergence(label, e_ana.grad_phi(k), hs, errors, 1e-10)) grad_ok = false;
    }
    EXPECT_TRUE(grad_ok);

    //  hess_phi vs central-difference of grad_phi
    std::cout << "=== SDF Hessian convergence: sphere (grad_phi) ===\n";
    bool hess_ok = true;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double analytic = e_ana.hess_phi(row, col);
            std::vector<double> errors;
            for (double h : hs)
                errors.push_back(std::abs(fd_hessian(grad_phi, x, h)(row, col) - analytic));
            const std::string label = std::string("d2phi(") + std::to_string(row) + ","
                                    + std::to_string(col) + ")";
            if (!check_convergence(label, analytic, hs, errors, 1e-9, false)) {
                check_convergence(label, analytic, hs, errors, 1e-9, true);
                hess_ok = false;
            }
        }
    }
    EXPECT_TRUE(hess_ok);
}

// ============================================================================
//  Boundary behavior
// ============================================================================

TEST(SDFPenalty, ZeroOutsideTransition){
    CylinderSDF c{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), 0.5};
    const double k = 100.0;
    const double eps = 0.1;

    //  Far outside: everything zero.
    {
        const SDFEvaluation sdf = evaluate_sdf(c, Vec3(2.0, 0.0, 0.5));
        EXPECT_EQ(sdf_penalty_energy  (sdf, k, eps), 0.0);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero()));
        EXPECT_TRUE(sdf_penalty_hessian (sdf, k, eps).isApprox(Mat33::Zero()));
    }

    //  Deep inside: energy k/2, zero derivatives.
    {
        const SDFEvaluation sdf = evaluate_sdf(c, Vec3(0.0, 0.0, 0.5));
        EXPECT_NEAR(sdf_penalty_energy(sdf, k, eps), 0.5 * k, kTol);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero()));
        EXPECT_TRUE(sdf_penalty_hessian (sdf, k, eps).isApprox(Mat33::Zero()));
    }
}

TEST(SDFPenalty, GradientPushesOutward){
    //  In the transition layer the -gradient should have a positive projection
    //  onto the outward normal (i.e. the force pushes the node away from the
    //  obstacle).
    CylinderSDF c{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), 0.5};
    const double k = 10.0;
    const double eps = 0.1;

    //  Radial direction perpendicular to the axis.
    const Vec3 radial = Vec3(1.0, 1.0, 0.0).normalized();
    //  Point just outside the surface with an arbitrary axial offset.
    const Vec3 x = c.point + (c.radius + 0.05) * radial + 0.3 * c.axis;

    const Vec3 g = sdf_penalty_gradient(evaluate_sdf(c, x), k, eps);
    const Vec3 force = -g;
    EXPECT_GT(force.dot(radial), 0.0);
}
