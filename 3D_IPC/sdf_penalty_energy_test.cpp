#include "sdf_penalty_energy.h"

#include "rigid_body_ipc.h"

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
    bool saw_reliable_slope = false;
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
                      << "  slope=" << std::fixed << std::setprecision(6) << slope << "\n";
        saw_reliable_slope = true;
        if (slope < 1.99 || slope > 2.01) {
            std::cerr << "  FAIL: slope " << slope
                      << " outside [1.99, 2.01] for " << label << "\n";
            passed = false;
        }
    }

    if (all_below_noise) {
        if (verbose) std::cout << "    (all errors below noise floor -- exact match)\n";
        return true;
    }
    if (!saw_reliable_slope) {
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
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
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
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
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

TEST(PlaneSDF, EnergyBehavior){
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    const double k = 100.0;
    //  eps=0 collapses to the hard one-sided quadratic: E = 0.5*k*phi^2 for phi<0.
    const double eps = 0.0;

    //  Outside (phi > 0): zero energy, zero gradient, zero Hessian.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, 1.0, 0.0));
        EXPECT_EQ(sdf_penalty_energy(sdf, k, eps), 0.0);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero()));
        EXPECT_TRUE(sdf_penalty_hessian(sdf, k, eps).isApprox(Mat33::Zero()));
    }

    //  Inside (phi = -0.5): E = 0.5*k*0.25, grad = k*(-0.5)*n, hess = k*n*n^T.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, -0.5, 0.0));
        EXPECT_NEAR(sdf_penalty_energy(sdf, k, eps), 0.5 * k * 0.25, kTol);
        const Vec3 n(0.0, 1.0, 0.0);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(k * (-0.5) * n, kTol));
        EXPECT_TRUE(sdf_penalty_hessian(sdf, k, eps).isApprox(k * (n * n.transpose()), kTol));
    }
}

//  Soft-barrier semantics: with eps>0, E = 0.5*k*H(phi)*(eps - phi)^2.
//  Cloth's force-free rest is at phi = eps; surface contact (phi=0) costs
//  0.5*k*eps^2 of energy and gets pushed outward.
TEST(PlaneSDF, SoftBarrierWithEps){
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)};
    const double k   = 100.0;
    const double eps = 0.1;
    const Vec3   n(0.0, 1.0, 0.0);

    //  Past the active range (phi > eps): zero everything.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, 1.0, 0.0));  //  phi=1.0
        EXPECT_EQ(sdf_penalty_energy(sdf, k, eps), 0.0);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero()));
    }
    //  At the rest distance (phi = eps): zero energy, zero force.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, eps, 0.0));
        EXPECT_NEAR(sdf_penalty_energy(sdf, k, eps), 0.0, kTol);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(Vec3::Zero(), kTol));
    }
    //  At the surface (phi = 0): E = 0.5*k*eps^2, grad = -k*eps*n.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, 0.0, 0.0));
        EXPECT_NEAR(sdf_penalty_energy(sdf, k, eps), 0.5 * k * eps * eps, kTol);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(-k * eps * n, kTol));
    }
    //  In the transition band (phi = 0.04): H = 0.6, d = 0.06.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, 0.04, 0.0));
        EXPECT_NEAR(sdf_penalty_energy(sdf, k, eps), 0.5 * k * 0.6 * 0.06 * 0.06, kTol);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(-5.4 * n, kTol));
        EXPECT_TRUE(sdf_penalty_hessian(sdf, k, eps).isApprox(180.0 * (n * n.transpose()), kTol));
    }
    //  Inside (phi = -0.2): E = 0.5*k*(eps+0.2)^2 = 0.5*k*0.09.
    {
        const SDFEvaluation sdf = evaluate_sdf(p, Vec3(0.0, -0.2, 0.0));
        EXPECT_NEAR(sdf_penalty_energy(sdf, k, eps), 0.5 * k * 0.09, kTol);
        EXPECT_TRUE(sdf_penalty_gradient(sdf, k, eps).isApprox(-k * 0.3 * n, kTol));
    }
}

TEST(PlaneSDF, GradientConvergence){
    //  phi = -0.15: inside by 0.15, safe for h up to 1e-2 without crossing phi=0.
    Vec3 n = Vec3(1.0, 2.0, 3.0).normalized();
    PlaneSDF p{Vec3(0.1, -0.2, 0.05), n};
    const double k = 25.0;
    const double eps = 0.3;

    Vec3 tangent(0.4, -0.3, 0.2);
    tangent -= tangent.dot(n) * n;
    Vec3 x = p.point - 0.15 * n + tangent;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(p, q); };
    EXPECT_TRUE(run_sdf_gradient_convergence("plane", x, sdf, k, eps));
}

TEST(PlaneSDF, HessianConvergence){
    //  phi = -0.10: inside, hessian = k*n*n^T (constant), FD recovers it exactly.
    Vec3 n = Vec3(-1.0, 1.0, 2.0).normalized();
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), n};
    const double k = 40.0;
    const double eps = 0.25;

    Vec3 tangent(0.5, 0.5, -0.1);
    tangent -= tangent.dot(n) * n;
    Vec3 x = p.point - 0.10 * n + tangent;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(p, q); };
    EXPECT_TRUE(run_sdf_hessian_convergence("plane", x, sdf, k, eps, /*include_curvature=*/true));
}

TEST(PlaneSDF, HessianIsRankOne){
    Vec3 n = Vec3(-1.0, 1.0, 2.0).normalized();
    PlaneSDF p{Vec3(0.0, 0.0, 0.0), n};
    const double k = 40.0;
    const double eps = 0.25;

    Vec3 tangent(0.5, 0.5, -0.1);
    tangent -= tangent.dot(n) * n;
    Vec3 x = p.point - 0.10 * n + tangent;

    const Mat33 H_ana = sdf_penalty_hessian(evaluate_sdf(p, x), k, eps);

    //  Hessian inside is k * n*n^T (plane has zero curvature) -- rank one and symmetric.
    EXPECT_TRUE(H_ana.isApprox(H_ana.transpose(), kTol));
    const Mat33 expected = k * (n * n.transpose());
    EXPECT_TRUE(H_ana.isApprox(expected, kTol));

    Eigen::SelfAdjointEigenSolver<Mat33> es(H_ana);
    const Vec3 lambda = es.eigenvalues();
    EXPECT_NEAR(lambda(0), 0.0, 1e-6);
    EXPECT_NEAR(lambda(1), 0.0, 1e-6);
    EXPECT_NEAR(lambda(2), k,   1e-6);
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
    //  phi = -0.08: inside by 0.08, safe for h up to 1e-2.
    CylinderSDF c{Vec3(0.1, -0.2, 0.05),
                  Vec3(1.0, 2.0, -1.0).normalized(),
                  0.4};
    const double k = 30.0;
    const double eps = 0.2;

    const Vec3 raw_dir(3.0, -1.0, 2.0);
    const Vec3 perp_dir = (raw_dir - c.axis * c.axis.dot(raw_dir)).normalized();
    const Vec3 x = c.point + (c.radius - 0.08) * perp_dir + 0.5 * c.axis;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(c, q); };
    EXPECT_TRUE(run_sdf_gradient_convergence("cylinder", x, sdf, k, eps));
    EXPECT_GT((-sdf_penalty_gradient(sdf(x), k, eps)).dot(perp_dir), 0.0);
}

TEST(CylinderSDF, HessianConvergence){
    CylinderSDF c{Vec3(0.0, 0.0, 0.0),
                  Vec3(0.0, 1.0, 2.0).normalized(),
                  0.4};
    const double k = 20.0;
    const double eps = 0.2;

    const Vec3 raw_dir(1.0, 2.0, 3.0);
    const Vec3 perp_dir = (raw_dir - c.axis * c.axis.dot(raw_dir)).normalized();
    const Vec3 x = c.point + (c.radius - 0.08) * perp_dir + 0.3 * c.axis;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(c, q); };
    EXPECT_TRUE(run_sdf_hessian_convergence("cylinder", x, sdf, k, eps,
                                            /*include_curvature=*/true));
}

// ============================================================================
//  Sphere SDF
// ============================================================================

TEST(SphereSDF, Evaluate){
    //  Sphere at origin, radius 0.5.
    SphereSDF s{Vec3(0.0, 0.0, 0.0), 0.5};

    //  Point at (0.6, 0, 0) -- distance 0.6, phi = 0.1, grad along +x.
    SDFEvaluation r = evaluate_sdf(s, Vec3(0.6, 0.0, 0.0));
    EXPECT_NEAR(r.phi, 0.1, kTol);
    EXPECT_TRUE(r.grad_phi.isApprox(Vec3(1.0, 0.0, 0.0)));
    //  hess_phi = (I - n n^T)/r_dist; with n = e_x and r_dist = 0.6, the
    //  remaining tangent plane stiffness is diag(0, 1/0.6, 1/0.6).
    Mat33 expected = Mat33::Zero();
    expected(1, 1) = 1.0 / 0.6;
    expected(2, 2) = 1.0 / 0.6;
    EXPECT_TRUE(r.hess_phi.isApprox(expected, kTol));

    //  Point on a translated sphere's center: phi = -R, derivatives degenerate
    //  -> zero sentinels (same convention as the cylinder's on-axis case).
    SphereSDF s2{Vec3(1.0, 2.0, 3.0), 0.4};
    r = evaluate_sdf(s2, s2.center);
    EXPECT_NEAR(r.phi, -0.4, kTol);
    EXPECT_TRUE(r.grad_phi.isApprox(Vec3::Zero()));
    EXPECT_TRUE(r.hess_phi.isApprox(Mat33::Zero()));
}

TEST(SphereSDF, GradientConvergence){
    //  Inside by 0.08: phi = -0.08, safe for h up to 1e-2 without crossing 0.
    SphereSDF s{Vec3(0.1, -0.2, 0.05), 0.4};
    const double k = 30.0;
    const double eps = 0.2;

    const Vec3 dir = Vec3(3.0, -1.0, 2.0).normalized();
    const Vec3 x = s.center + (s.radius - 0.08) * dir;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(s, q); };
    EXPECT_TRUE(run_sdf_gradient_convergence("sphere", x, sdf, k, eps));
    EXPECT_GT((-sdf_penalty_gradient(sdf(x), k, eps)).dot(dir), 0.0);
}

TEST(SphereSDF, HessianConvergence){
    SphereSDF s{Vec3(0.0, 0.0, 0.0), 0.4};
    const double k = 20.0;
    const double eps = 0.2;

    const Vec3 dir = Vec3(1.0, 2.0, 3.0).normalized();
    const Vec3 x = s.center + (s.radius - 0.08) * dir;

    const SDFClosure sdf = [&](const Vec3& q){ return evaluate_sdf(s, q); };
    EXPECT_TRUE(run_sdf_hessian_convergence("sphere", x, sdf, k, eps,
                                            /*include_curvature=*/true));
}

// ============================================================================
//  Rigid-body chained derivatives
// ============================================================================

namespace {


struct RigidSDFSetup {
    SphereSDF sphere{Vec3(0.0, 0.0, 0.0), 1.0};
    double k = 50.0;
    double eps = 0.2;
    double dt = 0.31;
    Vec3 X_centered{0.4, -0.25, 0.3};
    Vec4 q_n = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    Vec3 omega{0.6, -0.3, 0.7};
    Vec3 x_com;

    RigidSDFSetup() {
        const Vec3 dir = Vec3(0.7, -0.4, 0.6).normalized();
        const Vec3 rotated =
            world_space_position(X_centered, Vec3::Zero(), q_n, omega, dt);
        x_com = (sphere.radius + 0.08) * dir - rotated;
    }

    Vec3 world_x(const Vec3& t, const Vec3& w) const {
        return world_space_position(X_centered, t, q_n, w, dt);
    }
    SDFEvaluation sdf_at(const Vec3& t, const Vec3& w) const {
        return evaluate_sdf(sphere, world_x(t, w));
    }
    double energy_at(const Vec3& t, const Vec3& w) const {
        return sdf_penalty_energy(sdf_at(t, w), k, eps);
    }
    RigidSDFGradient gradient_at(const Vec3& t, const Vec3& w) const {
        return sdf_penalty_gradient_rb(sdf_at(t, w), X_centered, q_n, w, dt, k, eps);
    }
};

bool run_component_convergence(const std::string& label, double analytic,
                               const std::vector<double>& hs,
                               const std::function<double(double)>& fd_at_h,
                               double noise_scale){
    if (std::abs(analytic) < 1e-14) {
        const double fd_fine = fd_at_h(hs.back());
        if (std::abs(fd_fine) > 1e-6) {
            std::cerr << "  FAIL: " << label << " analytic=0 but fd=" << fd_fine << "\n";
            return false;
        }
        return true;
    }
    std::vector<double> errors;
    for (double h : hs) errors.push_back(std::abs(fd_at_h(h) - analytic));
    return check_convergence(label, analytic, hs, errors, noise_scale, false)
        || check_convergence(label, analytic, hs, errors, noise_scale, true);
}

}  // namespace

TEST(RigidBodySDFPenalty, GradientConvergesWithCenteredDifferences){
    const RigidSDFSetup s;
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};

    //  The smooth-branch assumption behind the slope test: every FD sample
    //  must stay strictly inside the (0, eps) band.
    const SDFEvaluation sdf0 = s.sdf_at(s.x_com, s.omega);
    ASSERT_GT(sdf0.phi, 0.05);
    ASSERT_LT(sdf0.phi, s.eps - 0.05);

    const RigidSDFGradient g_ana = s.gradient_at(s.x_com, s.omega);

    const auto energy_of_t = [&](const Vec3& t){ return s.energy_at(t, s.omega); };
    const auto energy_of_w = [&](const Vec3& w){ return s.energy_at(s.x_com, w); };

    bool all_passed = true;
    for (int comp = 0; comp < 3; ++comp) {
        all_passed &= run_component_convergence(
            "dE/dt_" + std::to_string(comp), g_ana.translation(comp), hs,
            [&](double h){ return fd_gradient(energy_of_t, s.x_com, h)(comp); }, 1e-10);
        all_passed &= run_component_convergence(
            "dE/dw_" + std::to_string(comp), g_ana.rotation(comp), hs,
            [&](double h){ return fd_gradient(energy_of_w, s.omega, h)(comp); }, 1e-10);
    }
    EXPECT_TRUE(all_passed);
}

TEST(RigidBodySDFPenalty, HessianConvergesWithCenteredDifferences){
    const RigidSDFSetup s;
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};

    const RigidSDFHessian H_ana = sdf_penalty_hessian_rb(
        s.sdf_at(s.x_com, s.omega), s.X_centered, s.q_n, s.omega, s.dt, s.k, s.eps);

    EXPECT_TRUE(H_ana.translation_translation.isApprox(
        H_ana.translation_translation.transpose(), 1e-12));
    EXPECT_TRUE(H_ana.rotation_rotation.isApprox(
        H_ana.rotation_rotation.transpose(), 1e-12));

    const auto grad_t_of_t = [&](const Vec3& t){ return s.gradient_at(t, s.omega).translation; };
    const auto grad_t_of_w = [&](const Vec3& w){ return s.gradient_at(s.x_com, w).translation; };
    const auto grad_w_of_w = [&](const Vec3& w){ return s.gradient_at(s.x_com, w).rotation; };

    bool all_passed = true;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            all_passed &= run_component_convergence(
                "H_tt(" + std::to_string(row) + "," + std::to_string(col) + ")",
                H_ana.translation_translation(row, col), hs,
                [&](double h){ return fd_hessian(grad_t_of_t, s.x_com, h)(row, col); }, 1e-9);
            all_passed &= run_component_convergence(
                "H_tw(" + std::to_string(row) + "," + std::to_string(col) + ")",
                H_ana.translation_rotation(row, col), hs,
                [&](double h){ return fd_hessian(grad_t_of_w, s.omega, h)(row, col); }, 1e-9);
            all_passed &= run_component_convergence(
                "H_ww(" + std::to_string(row) + "," + std::to_string(col) + ")",
                H_ana.rotation_rotation(row, col), hs,
                [&](double h){ return fd_hessian(grad_w_of_w, s.omega, h)(row, col); }, 1e-9);
        }
    }
    EXPECT_TRUE(all_passed);
}
