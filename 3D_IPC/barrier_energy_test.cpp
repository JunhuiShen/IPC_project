#include "barrier_energy.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
//  Utilities
// ===========================================================================

constexpr double kTol = 1e-8;

bool approx(double a, double b, double tol = kTol)
{
    return std::abs(a - b) <= tol * (1.0 + std::abs(a) + std::abs(b));
}

void require(bool cond, const std::string& msg)
{
    if (!cond) {
        std::cerr << "TEST FAILED: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------------------
//  Finite-difference helper for scalar barrier:
//      fd = ( b(delta+h) - b(delta-h) ) / (2h)
// ---------------------------------------------------------------------------

double scalar_barrier_fd(double delta, double d_hat, double h)
{
    const double bp = scalar_barrier(delta + h, d_hat);
    const double bm = scalar_barrier(delta - h, d_hat);
    return (bp - bm) / (2.0 * h);
}

// ---------------------------------------------------------------------------
//  Finite-difference helper for node--triangle barrier energy:
//      fd = ( PE(y + h*e_alpha) - PE(y - h*e_alpha) ) / (2h)
//
//  which_vec: 0 = x, 1 = x1, 2 = x2, 3 = x3
//  comp:      0, 1, 2  (spatial index k)
// ---------------------------------------------------------------------------

double node_triangle_barrier_fd(
        const Vec3& x,
        const Vec3& x1,
        const Vec3& x2,
        const Vec3& x3,
        double d_hat,
        int which_vec,
        int comp,
        double h)
{
    Vec3 xp = x, x1p = x1, x2p = x2, x3p = x3;
    Vec3 xm = x, x1m = x1, x2m = x2, x3m = x3;

    switch (which_vec) {
        case 0: xp(comp)  += h; xm(comp)  -= h; break;
        case 1: x1p(comp) += h; x1m(comp) -= h; break;
        case 2: x2p(comp) += h; x2m(comp) -= h; break;
        case 3: x3p(comp) += h; x3m(comp) -= h; break;
        default: std::exit(EXIT_FAILURE);
    }

    const double Ep = node_triangle_barrier(xp, x1p, x2p, x3p, d_hat);
    const double Em = node_triangle_barrier(xm, x1m, x2m, x3m, d_hat);
    return (Ep - Em) / (2.0 * h);
}

// ===========================================================================
//  Test 1:  scalar barrier gradient, convergence rate = 2
//
//  For central differences, the error is O(h^2).  So if we halve h,
//  the error should decrease by a factor of ~4, i.e. slope ~ 2 in
//  log-log.
// ===========================================================================

void test_scalar_barrier_gradient_convergence()
{
    std::cout << "=== Test 1: scalar_barrier_gradient convergence ===\n";

    const double d_hat = 1.0;
    const double delta = 0.4;
    const double analytic = scalar_barrier_gradient(delta, d_hat);
    const double noise_floor = 1e-10 * (1.0 + std::abs(analytic));

    std::vector<double> hs    = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    std::vector<double> errors;

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  h              |error|         slope\n";

    bool saw_good_slope = false;

    for (std::size_t i = 0; i < hs.size(); ++i) {
        const double fd = scalar_barrier_fd(delta, d_hat, hs[i]);
        const double err = std::abs(fd - analytic);
        errors.push_back(err);

        if (i == 0) {
            std::cout << "  " << hs[i] << "   " << err << "   ---\n";
        } else {
            if (errors[i] < noise_floor || errors[i-1] < noise_floor) {
                std::cout << "  " << hs[i] << "   " << err << "   (round-off regime, skipped)\n";
                continue;
            }
            const double slope = std::log(errors[i-1] / errors[i])
                                 / std::log(hs[i-1] / hs[i]);
            std::cout << "  " << hs[i] << "   " << err << "   " << slope << "\n";
            require(slope > 1.8, "scalar barrier: convergence slope should be ~2");
            saw_good_slope = true;
        }
    }
    require(saw_good_slope, "scalar barrier: no reliable slope data");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 2:  barrier is zero outside activation distance
// ===========================================================================

void test_barrier_zero_outside_activation()
{
    std::cout << "=== Test 2: barrier zero outside activation ===\n";

    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x(0.25, 0.25, 2.0);   // distance = 2.0, well outside d_hat = 1.0
    const double d_hat = 1.0;

    const auto r = node_triangle_barrier_gradient(x, x1, x2, x3, d_hat);

    require(approx(r.energy, 0.0),             "inactive barrier energy should be zero");
    require(approx(r.barrier_derivative, 0.0),  "inactive barrier derivative should be zero");

    for (int k = 0; k < 3; ++k) {
        require(approx(r.grad_x(k),  0.0), "inactive grad_x");
        require(approx(r.grad_x1(k), 0.0), "inactive grad_x1");
        require(approx(r.grad_x2(k), 0.0), "inactive grad_x2");
        require(approx(r.grad_x3(k), 0.0), "inactive grad_x3");
    }

    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 3:  partition of force (gradients sum to zero)
// ===========================================================================

void test_partition_of_force()
{
    std::cout << "=== Test 3: partition of force ===\n";

    // face case
    {
        const Vec3 x(0.25, 0.25, 0.3);
        const Vec3 x1(0.0, 0.0, 0.0), x2(1.0, 0.0, 0.0), x3(0.0, 1.0, 0.0);
        const auto r = node_triangle_barrier_gradient(x, x1, x2, x3, 1.0);
        for (int k = 0; k < 3; ++k) {
            double sum_k = r.grad_x(k) + r.grad_x1(k) + r.grad_x2(k) + r.grad_x3(k);
            require(std::abs(sum_k) < 1e-12, "face: total gradient should sum to zero");
        }
    }
    // edge case
    {
        const Vec3 x(0.5, -0.2, 0.1);
        const Vec3 x1(0.0, 0.0, 0.0), x2(1.0, 0.0, 0.0), x3(0.0, 1.0, 0.0);
        const auto r = node_triangle_barrier_gradient(x, x1, x2, x3, 1.0);
        for (int k = 0; k < 3; ++k) {
            double sum_k = r.grad_x(k) + r.grad_x1(k) + r.grad_x2(k) + r.grad_x3(k);
            require(std::abs(sum_k) < 1e-12, "edge: total gradient should sum to zero");
        }
    }
    // vertex case
    {
        const Vec3 x(-0.2, -0.3, 0.1);
        const Vec3 x1(0.0, 0.0, 0.0), x2(1.0, 0.0, 0.0), x3(0.0, 1.0, 0.0);
        const auto r = node_triangle_barrier_gradient(x, x1, x2, x3, 1.0);
        for (int k = 0; k < 3; ++k) {
            double sum_k = r.grad_x(k) + r.grad_x1(k) + r.grad_x2(k) + r.grad_x3(k);
            require(std::abs(sum_k) < 1e-12, "vertex: total gradient should sum to zero");
        }
    }

    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Convergence-rate test framework
//
//  For each DOF y_alpha, we compute:
//      error(h) = | fd(h) - analytic |
//  at several h values, then check that
//      slope = log(error(h1)/error(h2)) / log(h1/h2) ≈ 2
// ===========================================================================

struct TestPoint {
    std::string name;
    Vec3 x, x1, x2, x3;
    double d_hat;
    NodeTriangleRegion expected_region;
};

void run_convergence_test(const TestPoint& tp)
{
    std::cout << "=== Convergence test: " << tp.name << " ===\n";

    const auto r = node_triangle_barrier_gradient(
            tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat);

    require(r.distance_result.region == tp.expected_region,
            tp.name + ": unexpected region " + to_string(r.distance_result.region));

    std::cout << "  Region: " << to_string(r.distance_result.region)
              << ",  delta = " << r.distance << "\n";

    // Collect all 12 analytic gradient components
    // dof layout: (x, x1, x2, x3) x (0,1,2)
    const char* dof_names[4] = {"x", "x1", "x2", "x3"};
    double analytic[4][3];
    for (int k = 0; k < 3; ++k) {
        analytic[0][k] = r.grad_x(k);
        analytic[1][k] = r.grad_x1(k);
        analytic[2][k] = r.grad_x2(k);
        analytic[3][k] = r.grad_x3(k);
    }

    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};

    bool all_passed = true;

    for (int v = 0; v < 4; ++v) {
        for (int k = 0; k < 3; ++k) {

            // Skip components where the analytic gradient is exactly zero
            // (finite difference will also be zero up to round-off, slope is
            // meaningless).
            if (std::abs(analytic[v][k]) < 1e-14) {
                // But verify FD is also tiny at the finest h
                double fd_fine = node_triangle_barrier_fd(
                        tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v, k, hs.back());
                if (std::abs(fd_fine) > 1e-8) {
                    std::cerr << "  FAIL: " << dof_names[v] << "(" << k << ")"
                              << "  analytic=0 but fd=" << fd_fine << "\n";
                    all_passed = false;
                }
                continue;
            }

            std::vector<double> errors;
            for (std::size_t i = 0; i < hs.size(); ++i) {
                double fd = node_triangle_barrier_fd(
                        tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v, k, hs[i]);
                errors.push_back(std::abs(fd - analytic[v][k]));
            }

            // Check convergence slopes between successive h pairs.
            //
            // The central-difference error is O(h^2), so halving h should
            // reduce the error by ~4x (slope ~2 in log-log).  However, once
            // the absolute error drops below ~1e-11 * |analytic|, floating-
            // point round-off dominates and the slope becomes meaningless.
            // We only enforce the slope check when both errors are well above
            // that noise floor.

            const double noise_floor = 1e-10 * (1.0 + std::abs(analytic[v][k]));

            std::cout << "  d/d(" << dof_names[v] << ")_" << k
                      << "  analytic=" << std::scientific << std::setprecision(8)
                      << analytic[v][k] << "\n";

            bool saw_good_slope = false;

            for (std::size_t i = 1; i < hs.size(); ++i) {
                // If either error is at the noise floor, slope is unreliable
                if (errors[i] < noise_floor || errors[i-1] < noise_floor) {
                    std::cout << "    h=" << hs[i]
                              << "  err=" << errors[i]
                              << "  (round-off regime, skipped)\n";
                    continue;
                }

                double slope = std::log(errors[i-1] / errors[i])
                               / std::log(hs[i-1] / hs[i]);
                std::cout << "    h=" << hs[i]
                          << "  err=" << errors[i]
                          << "  slope=" << std::fixed << std::setprecision(2)
                          << slope << "\n";

                if (slope < 1.8) {
                    std::cerr << "  FAIL: slope " << slope << " < 1.8 for d/d("
                              << dof_names[v] << ")_" << k << "\n";
                    all_passed = false;
                } else {
                    saw_good_slope = true;
                }
            }

            // We must have seen at least one reliable slope measurement
            if (!saw_good_slope) {
                std::cerr << "  FAIL: no reliable slope data for d/d("
                          << dof_names[v] << ")_" << k << "\n";
                all_passed = false;
            }
        }
    }

    require(all_passed, tp.name + ": convergence test failed");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  main
// ===========================================================================

int main()
{
    test_scalar_barrier_gradient_convergence();
    test_barrier_zero_outside_activation();
    test_partition_of_force();

    // -------------------------------------------------------------------
    //  Convergence tests for each case
    // -------------------------------------------------------------------

    // Face interior
    run_convergence_test({
                                 "face_interior",
                                 Vec3(0.25, 0.25, 0.3),                                      // x
                                 Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                                 1.0,
                                 NodeTriangleRegion::FaceInterior
                         });

    // Edge 12
    run_convergence_test({
                                 "edge_12",
                                 Vec3(0.5, -0.2, 0.1),
                                 Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                                 1.0,
                                 NodeTriangleRegion::Edge12
                         });

    // Edge 23
    run_convergence_test({
                                 "edge_23",
                                 Vec3(0.7, 0.7, 0.1),
                                 Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                                 1.0,
                                 NodeTriangleRegion::Edge23
                         });

    // Edge 31
    run_convergence_test({
                                 "edge_31",
                                 Vec3(-0.15, 0.5, 0.1),
                                 Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                                 1.0,
                                 NodeTriangleRegion::Edge31
                         });

    // Vertex 1
    run_convergence_test({
                                 "vertex_1",
                                 Vec3(-0.2, -0.3, 0.1),
                                 Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                                 1.0,
                                 NodeTriangleRegion::Vertex1
                         });

    // Vertex 2:  need l3 <= 0 and l1 <= 0
    //   projection (1.4, -0.1, 0) -> alpha=1.4, beta=-0.1, l1=-0.3, l3=-0.1
    run_convergence_test({
                                 "vertex_2",
                                 Vec3(1.4, -0.1, 0.1),
                                 Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                                 1.0,
                                 NodeTriangleRegion::Vertex2
                         });

    // Vertex 3:  need l1 <= 0 and l2 <= 0
    //   projection (-0.1, 1.4, 0) -> alpha=-0.1, beta=1.4, l1=-0.3, l2=-0.1
    run_convergence_test({
                                 "vertex_3",
                                 Vec3(-0.1, 1.4, 0.1),
                                 Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                                 1.0,
                                 NodeTriangleRegion::Vertex3
                         });

    // -------------------------------------------------------------------
    //  A non-axis-aligned triangle to test generality
    // -------------------------------------------------------------------

    std::cout << "\n========================================\n"
              << "All barrier energy tests passed.\n"
              << "========================================\n";
    return 0;
}