#include "barrier_energy.h"

#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
//  Utilities
// ===========================================================================

namespace {

constexpr double kTol = 1e-8;

bool approx(double a, double b, double tol = kTol){
    return std::abs(a - b) <= tol * (1.0 + std::abs(a) + std::abs(b));
}

// ===========================================================================
//  Assembly helpers: reconstruct full gradient / hessian from per-DOF calls
// ===========================================================================

// Node-triangle: assemble 12-vector gradient from 4 per-DOF Vec3 calls
Eigen::Matrix<double, 12, 1> nt_assemble_gradient(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat) {
    Eigen::Matrix<double, 12, 1> g = Eigen::Matrix<double, 12, 1>::Zero();
    for (int dof = 0; dof < 4; ++dof)
        g.segment<3>(3*dof) = node_triangle_barrier_gradient(x, x1, x2, x3, d_hat, dof);
    return g;
}


// Segment-segment: assemble 12-vector gradient from 4 per-DOF Vec3 calls
Eigen::Matrix<double, 12, 1> ss_assemble_gradient(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat) {
    Eigen::Matrix<double, 12, 1> g = Eigen::Matrix<double, 12, 1>::Zero();
    for (int dof = 0; dof < 4; ++dof)
        g.segment<3>(3*dof) = segment_segment_barrier_gradient(x1, x2, x3, x4, d_hat, dof);
    return g;
}


// ===========================================================================
//  FD helpers for node-triangle barrier (gradient check)
// ===========================================================================

double node_triangle_barrier_fd(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                double d_hat, int which_vec, int comp, double h){
    Vec3 xp = x, x1p = x1, x2p = x2, x3p = x3;
    Vec3 xm = x, x1m = x1, x2m = x2, x3m = x3;
    switch (which_vec) {
        case 0: xp(comp)  += h; xm(comp)  -= h; break;
        case 1: x1p(comp) += h; x1m(comp) -= h; break;
        case 2: x2p(comp) += h; x2m(comp) -= h; break;
        case 3: x3p(comp) += h; x3m(comp) -= h; break;
        default: std::exit(EXIT_FAILURE);
    }
    return (node_triangle_barrier(xp, x1p, x2p, x3p, d_hat)
            - node_triangle_barrier(xm, x1m, x2m, x3m, d_hat)) / (2.0 * h);
}

// FD helper for node-triangle gradient (for Hessian check)
double node_triangle_gradient_fd(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                 double d_hat, int v1, int k, int v2, int l, double h){
    Vec3 xp = x, x1p = x1, x2p = x2, x3p = x3;
    Vec3 xm = x, x1m = x1, x2m = x2, x3m = x3;
    switch (v1) {
        case 0: xp(k)  += h; xm(k)  -= h; break;
        case 1: x1p(k) += h; x1m(k) -= h; break;
        case 2: x2p(k) += h; x2m(k) -= h; break;
        case 3: x3p(k) += h; x3m(k) -= h; break;
        default: std::exit(EXIT_FAILURE);
    }
    Vec3 gp = node_triangle_barrier_gradient(xp, x1p, x2p, x3p, d_hat, v2);
    Vec3 gm = node_triangle_barrier_gradient(xm, x1m, x2m, x3m, d_hat, v2);
    return (gp(l) - gm(l)) / (2.0 * h);
}

// ===========================================================================
//  FD helpers for segment-segment barrier
// ===========================================================================

double segment_segment_barrier_fd(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                  double d_hat, int which_vec, int comp, double h){
    Vec3 x1p = x1, x2p = x2, x3p = x3, x4p = x4;
    Vec3 x1m = x1, x2m = x2, x3m = x3, x4m = x4;
    switch (which_vec) {
        case 0: x1p(comp) += h; x1m(comp) -= h; break;
        case 1: x2p(comp) += h; x2m(comp) -= h; break;
        case 2: x3p(comp) += h; x3m(comp) -= h; break;
        case 3: x4p(comp) += h; x4m(comp) -= h; break;
        default: std::exit(EXIT_FAILURE);
    }
    return (segment_segment_barrier(x1p, x2p, x3p, x4p, d_hat)
            - segment_segment_barrier(x1m, x2m, x3m, x4m, d_hat)) / (2.0 * h);
}

double segment_segment_gradient_fd(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                   double d_hat, int v1, int k, int v2, int l, double h){
    Vec3 x1p = x1, x2p = x2, x3p = x3, x4p = x4;
    Vec3 x1m = x1, x2m = x2, x3m = x3, x4m = x4;
    switch (v1) {
        case 0: x1p(k) += h; x1m(k) -= h; break;
        case 1: x2p(k) += h; x2m(k) -= h; break;
        case 2: x3p(k) += h; x3m(k) -= h; break;
        case 3: x4p(k) += h; x4m(k) -= h; break;
        default: std::exit(EXIT_FAILURE);
    }
    Vec3 gp = segment_segment_barrier_gradient(x1p, x2p, x3p, x4p, d_hat, v2);
    Vec3 gm = segment_segment_barrier_gradient(x1m, x2m, x3m, x4m, d_hat, v2);
    return (gp(l) - gm(l)) / (2.0 * h);
}

// ===========================================================================
//  Convergence test infrastructure
// ===========================================================================

struct TestPoint {
    std::string name;
    Vec3 x, x1, x2, x3;
    double d_hat;
    NodeTriangleRegion expected_region;
};

struct SSTestPoint {
    std::string name;
    Vec3 x1, x2, x3, x4;
    double d_hat;
    SegmentSegmentRegion expected_region;
};

bool check_convergence(const std::string& label, double analytic, const std::vector<double>& hs,
                       const std::vector<double>& errors, double noise_scale = 1e-10, bool verbose = true){
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

// ===========================================================================
//  FD helpers for scalar barrier
// ===========================================================================

double scalar_barrier_fd(double delta, double d_hat, double h){
    return (scalar_barrier(delta+h, d_hat) - scalar_barrier(delta-h, d_hat)) / (2.0*h);
}
double scalar_barrier_gradient_fd(double delta, double d_hat, double h){
    return (scalar_barrier_gradient(delta+h, d_hat) - scalar_barrier_gradient(delta-h, d_hat)) / (2.0*h);
}

// ===========================================================================
//  Node-triangle gradient convergence helper
// ===========================================================================

bool run_gradient_convergence_test(const TestPoint& tp){
    std::cout << "=== Gradient convergence: " << tp.name << " ===\n";

    const char* dof_names[4] = {"x", "x1", "x2", "x3"};
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;

    for (int v = 0; v < 4; ++v) {
        Vec3 g = node_triangle_barrier_gradient(tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v);
        for (int k = 0; k < 3; ++k) {
            double analytic = g(k);

            if (std::abs(analytic) < 1e-14) {
                double fd_fine = node_triangle_barrier_fd(tp.x, tp.x1, tp.x2, tp.x3,
                                                          tp.d_hat, v, k, hs.back());
                if (std::abs(fd_fine) > 1e-8) {
                    std::cerr << "  FAIL: " << dof_names[v] << "(" << k
                              << ") analytic=0 but fd=" << fd_fine << "\n";
                    all_passed = false;
                }
                continue;
            }

            std::vector<double> errors;
            for (auto h : hs)
                errors.push_back(std::abs(
                        node_triangle_barrier_fd(tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v, k, h)
                        - analytic));

            std::string label = std::string("d/d(") + dof_names[v] + ")_" + std::to_string(k);
            if (!check_convergence(label, analytic, hs, errors, 1e-10)) all_passed = false;
        }
    }

    std::cout << (all_passed ? "  PASSED" : "  FAILED") << "\n\n";
    return all_passed;
}

// ===========================================================================
//  Node-triangle Hessian convergence helper
// ===========================================================================

bool run_hessian_convergence_test(const TestPoint& tp){
    std::cout << "=== Hessian convergence: " << tp.name << " ===\n";

    const char* dof_names[4] = {"x", "x1", "x2", "x3"};
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;
    int tested = 0, skipped_zero = 0;

    for (int v = 0; v < 4; ++v) {
        const Mat33 H = node_triangle_barrier_hessian(tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v);
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                const double analytic_val = H(k, l);

                if (std::abs(analytic_val) < 1e-14) {
                    const double fd_fine = node_triangle_gradient_fd(
                            tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v, k, v, l, hs.back());
                    if (std::abs(fd_fine) > 1e-6) {
                        std::cerr << "  FAIL: H(" << dof_names[v] << k << ","
                                  << dof_names[v] << l << ") analytic=0 but fd=" << fd_fine << "\n";
                        all_passed = false;
                    }
                    skipped_zero++;
                    continue;
                }

                std::vector<double> errors;
                for (auto h : hs) {
                    const double fd = node_triangle_gradient_fd(
                            tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v, k, v, l, h);
                    errors.push_back(std::abs(fd - analytic_val));
                }

                const std::string label = std::string("H(") + dof_names[v] + std::to_string(k)
                                        + "," + dof_names[v] + std::to_string(l) + ")";
                if (!check_convergence(label, analytic_val, hs, errors, 1e-9, false)) {
                    check_convergence(label, analytic_val, hs, errors, 1e-9, true);
                    all_passed = false;
                }
                tested++;
            }
        }
    }

    std::cout << "  Tested " << tested << " entries, skipped " << skipped_zero << " zero entries\n";
    std::cout << (all_passed ? "  PASSED" : "  FAILED") << "\n\n";
    return all_passed;
}

// ===========================================================================
//  Segment-segment gradient convergence helper
// ===========================================================================

bool run_ss_gradient_convergence_test(const SSTestPoint& tp){
    std::cout << "=== SS Gradient convergence: " << tp.name << " ===\n";

    const char* dof_names[4] = {"x1", "x2", "x3", "x4"};
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;

    for (int v = 0; v < 4; ++v) {
        Vec3 g = segment_segment_barrier_gradient(tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v);
        for (int k = 0; k < 3; ++k) {
            double analytic = g(k);

            if (std::abs(analytic) < 1e-14) {
                double fd_fine = segment_segment_barrier_fd(
                        tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v, k, hs.back());
                if (std::abs(fd_fine) > 1e-8) {
                    std::cerr << "  FAIL: " << dof_names[v] << "(" << k
                              << ") analytic=0 but fd=" << fd_fine << "\n";
                    all_passed = false;
                }
                continue;
            }

            std::vector<double> errors;
            for (auto h : hs)
                errors.push_back(std::abs(
                        segment_segment_barrier_fd(tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v, k, h)
                        - analytic));

            std::string label = std::string("d/d(") + dof_names[v] + ")_" + std::to_string(k);
            if (!check_convergence(label, analytic, hs, errors, 1e-9)) all_passed = false;
        }
    }

    std::cout << (all_passed ? "  PASSED" : "  FAILED") << "\n\n";
    return all_passed;
}

// ===========================================================================
//  Segment-segment Hessian convergence helper
// ===========================================================================

bool run_ss_hessian_convergence_test(const SSTestPoint& tp){
    std::cout << "=== SS Hessian convergence: " << tp.name << " ===\n";

    const char* dof_names[4] = {"x1", "x2", "x3", "x4"};
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;
    int tested = 0, skipped_zero = 0;

    for (int v = 0; v < 4; ++v) {
        const Mat33 H = segment_segment_barrier_hessian(tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v);
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                const double analytic_val = H(k, l);

                if (std::abs(analytic_val) < 1e-14) {
                    const double fd_fine = segment_segment_gradient_fd(
                            tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v, k, v, l, hs.back());
                    if (std::abs(fd_fine) > 1e-6) {
                        std::cerr << "  FAIL: H(" << dof_names[v] << k << ","
                                  << dof_names[v] << l << ") analytic=0 but fd=" << fd_fine << "\n";
                        all_passed = false;
                    }
                    skipped_zero++;
                    continue;
                }

                std::vector<double> errors;
                for (auto h : hs) {
                    const double fd = segment_segment_gradient_fd(
                            tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v, k, v, l, h);
                    errors.push_back(std::abs(fd - analytic_val));
                }

                const std::string label = std::string("H(") + dof_names[v] + std::to_string(k)
                                        + "," + dof_names[v] + std::to_string(l) + ")";
                if (!check_convergence(label, analytic_val, hs, errors, 1e-9, false)) {
                    check_convergence(label, analytic_val, hs, errors, 1e-9, true);
                    all_passed = false;
                }
                tested++;
            }
        }
    }

    std::cout << "  Tested " << tested << " entries, skipped " << skipped_zero << " zero entries\n";
    std::cout << (all_passed ? "  PASSED" : "  FAILED") << "\n\n";
    return all_passed;
}

// ===========================================================================
//  Test point data
// ===========================================================================

std::vector<TestPoint> make_nt_test_points() {
    return {
            {"face_interior",
                    Vec3(0.25,0.25,0.3), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::FaceInterior},
            {"edge_12",
                    Vec3(0.5,-0.2,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Edge12},
            {"edge_23",
                    Vec3(0.7,0.7,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Edge23},
            {"edge_31",
                    Vec3(-0.15,0.5,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Edge31},
            {"vertex_1",
                    Vec3(-0.2,-0.3,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Vertex1},
            {"vertex_2",
                    Vec3(1.4,-0.1,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Vertex2},
            {"vertex_3",
                    Vec3(-0.1,1.4,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Vertex3},
    };
}

std::vector<SSTestPoint> make_ss_test_points() {
    return {
            {"ss_interior",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0.5,-1,0.5), Vec3(0.5,1,0.5), 2.0,
                    SegmentSegmentRegion::Interior},
            {"ss_edge_s0",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(-1,-1,0.3), Vec3(-1,1,0.3), 3.0,
                    SegmentSegmentRegion::Edge_s0},
            {"ss_edge_s1",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(2,-1,0.3), Vec3(2,1,0.3), 3.0,
                    SegmentSegmentRegion::Edge_s1},
            {"ss_edge_t0",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0.5,0.3,0.3), Vec3(0.5,1.3,0.3), 2.0,
                    SegmentSegmentRegion::Edge_t0},
            {"ss_edge_t1",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0.5,-1.3,0.3), Vec3(0.5,-0.3,0.3), 2.0,
                    SegmentSegmentRegion::Edge_t1},
            {"ss_corner_s0t0",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(-0.5,-0.5,0.3), Vec3(-0.5,-1.5,0.3), 2.0,
                    SegmentSegmentRegion::Corner_s0t0},
            {"ss_corner_s0t1",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(-1.5,-1.5,0.3), Vec3(-0.5,-0.5,0.3), 2.0,
                    SegmentSegmentRegion::Corner_s0t1},
            {"ss_corner_s1t0",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(1.5,-0.5,0.3), Vec3(1.5,-1.5,0.3), 2.0,
                    SegmentSegmentRegion::Corner_s1t0},
            {"ss_corner_s1t1",
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(2.5,-1.5,0.3), Vec3(1.5,-0.5,0.3), 2.0,
                    SegmentSegmentRegion::Corner_s1t1},
    };
}

} // namespace

// ===========================================================================
//  Tests
// ===========================================================================

TEST(BarrierEnergy, ScalarBarrierGradientConvergence){
    const double d_hat = 1.0, delta = 0.4;
    const double analytic = scalar_barrier_gradient(delta, d_hat);
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    std::vector<double> errors;
    for (auto h : hs) errors.push_back(std::abs(scalar_barrier_fd(delta, d_hat, h) - analytic));
    EXPECT_TRUE(check_convergence("b'", analytic, hs, errors)) << "scalar barrier gradient convergence";
}

TEST(BarrierEnergy, ScalarBarrierHessianConvergence){
    const double d_hat = 1.0, delta = 0.4;
    const double analytic = scalar_barrier_hessian(delta, d_hat);
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    std::vector<double> errors;
    for (auto h : hs) errors.push_back(std::abs(scalar_barrier_gradient_fd(delta, d_hat, h) - analytic));
    EXPECT_TRUE(check_convergence("b''", analytic, hs, errors)) << "scalar barrier hessian convergence";
}

TEST(BarrierEnergy, ZeroOutsideActivation){
    const Vec3 x1(0,0,0), x2(1,0,0), x3(0,1,0), x(0.25,0.25,2.0);
    const double d_hat = 1.0;
    EXPECT_TRUE(approx(node_triangle_barrier(x, x1, x2, x3, d_hat), 0.0)) << "inactive energy";
    for (int dof = 0; dof < 4; ++dof) {
        Vec3 g = node_triangle_barrier_gradient(x, x1, x2, x3, d_hat, dof);
        for (int k = 0; k < 3; ++k)
            EXPECT_TRUE(approx(g(k), 0.0)) << "inactive grad dof=" << dof;
    }
}

TEST(BarrierEnergy, PartitionOfForce){
    auto check = [](const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                    const std::string& label){
        Vec3 total = Vec3::Zero();
        for (int dof = 0; dof < 4; ++dof)
            total += node_triangle_barrier_gradient(x, x1, x2, x3, 1.0, dof);
        for (int k = 0; k < 3; ++k)
            EXPECT_LT(std::abs(total(k)), 1e-12) << label << ": total gradient should sum to zero";
    };

    check(Vec3(0.25,0.25,0.3), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), "face");
    check(Vec3(0.5,-0.2,0.1),  Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), "edge");
    check(Vec3(-0.2,-0.3,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), "vertex");
}

TEST(BarrierEnergy, NTGradientConvergence){
    auto test_points = make_nt_test_points();
    for (const auto& tp : test_points) {
        EXPECT_TRUE(run_gradient_convergence_test(tp)) << tp.name << ": gradient convergence failed";
    }
}

TEST(BarrierEnergy, NTHessianConvergence){
    auto test_points = make_nt_test_points();
    for (const auto& tp : test_points) {
        EXPECT_TRUE(run_hessian_convergence_test(tp)) << tp.name << ": Hessian convergence failed";
    }
}

TEST(BarrierEnergy, SSZeroOutsideActivation){
    const Vec3 x1(0,0,0), x2(1,0,0), x3(0.5,-3,2), x4(0.5,3,2);
    const double d_hat = 1.0;
    EXPECT_TRUE(approx(segment_segment_barrier(x1, x2, x3, x4, d_hat), 0.0)) << "ss inactive energy";
    for (int dof = 0; dof < 4; ++dof) {
        Vec3 g = segment_segment_barrier_gradient(x1, x2, x3, x4, d_hat, dof);
        for (int k = 0; k < 3; ++k)
            EXPECT_TRUE(approx(g(k), 0.0)) << "ss inactive grad dof=" << dof;
    }
}

TEST(BarrierEnergy, SSPartitionOfForce){
    auto check = [](const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                    const std::string& label){
        Vec3 total = Vec3::Zero();
        for (int dof = 0; dof < 4; ++dof)
            total += segment_segment_barrier_gradient(x1, x2, x3, x4, 2.0, dof);
        for (int k = 0; k < 3; ++k)
            EXPECT_LT(std::abs(total(k)), 1e-12) << label << ": total gradient should sum to zero";
    };

    check(Vec3(0,0,0), Vec3(1,0,0), Vec3(0.5,-1,0.5), Vec3(0.5,1,0.5), "interior");
    check(Vec3(0,0,0), Vec3(1,0,0), Vec3(-1,-1,0.3),  Vec3(-1,1,0.3),  "edge_s0");
    check(Vec3(0,0,0), Vec3(1,0,0), Vec3(-1,-1,0.3),  Vec3(-1,-2,0.3), "corner");
}

TEST(BarrierEnergy, SSGradientConvergence){
    auto test_points = make_ss_test_points();
    for (const auto& tp : test_points) {
        EXPECT_TRUE(run_ss_gradient_convergence_test(tp)) << tp.name << ": SS gradient convergence failed";
    }
}

TEST(BarrierEnergy, SSHessianConvergence){
    auto test_points = make_ss_test_points();
    for (const auto& tp : test_points) {
        EXPECT_TRUE(run_ss_hessian_convergence_test(tp)) << tp.name << ": SS Hessian convergence failed";
    }
}

TEST(BarrierEnergy, StressNTNearActivation){
    const Vec3 x1(0,0,0), x2(1,0,0), x3(0,1,0);
    const double d_hat = 1.0;

    // Just inside: distance = d_hat - 1e-8 -> barrier should be small but positive
    const double d_inside = d_hat - 1e-8;
    const Vec3 x_inside(0.25, 0.25, d_inside);
    double E_inside = node_triangle_barrier(x_inside, x1, x2, x3, d_hat);
    EXPECT_GT(E_inside, 0.0) << "barrier should be positive just inside activation";
    std::cout << "  just inside: E=" << std::scientific << E_inside << "\n";

    // Just outside: distance = d_hat + 1e-8 -> barrier should be exactly 0
    const double d_outside = d_hat + 1e-8;
    const Vec3 x_outside(0.25, 0.25, d_outside);
    double E_outside = node_triangle_barrier(x_outside, x1, x2, x3, d_hat);
    EXPECT_EQ(E_outside, 0.0) << "barrier should be zero just outside activation";
    std::cout << "  just outside: E=" << E_outside << "\n";

    // Gradient should be continuous: just-inside gradient should be small
    Vec3 g_inside = node_triangle_barrier_gradient(x_inside, x1, x2, x3, d_hat, 0);
    std::cout << "  just inside gradient norm: " << g_inside.norm() << "\n";
    EXPECT_LT(g_inside.norm(), 1.0) << "gradient at boundary should be moderate";
}

TEST(BarrierEnergy, StressSSNearActivation){
    const double d_hat = 1.0;

    // Interior region, distance ~ d_hat
    const double d_inside = d_hat - 1e-8;
    const Vec3 x1(0,0,0), x2(1,0,0);
    const Vec3 x3(0.5, -1, d_inside), x4(0.5, 1, d_inside);
    double E = segment_segment_barrier(x1, x2, x3, x4, d_hat);
    EXPECT_GT(E, 0.0) << "ss barrier should be positive just inside";

    const Vec3 x3_out(0.5, -1, d_hat + 1e-8), x4_out(0.5, 1, d_hat + 1e-8);
    double E_out = segment_segment_barrier(x1, x2, x3_out, x4_out, d_hat);
    EXPECT_EQ(E_out, 0.0) << "ss barrier should be zero just outside";

    std::cout << "  inside=" << std::scientific << E << " outside=" << E_out << "\n";
}

TEST(BarrierEnergy, StressSSNearParallel){
    const double d_hat = 1.0;
    // Two nearly parallel segments, small offset in z
    const Vec3 x1(0,0,0), x2(1,0,0);
    const Vec3 x3(0.1, 0.0, 0.3), x4(0.9, 1e-10, 0.3);

    double E = segment_segment_barrier(x1, x2, x3, x4, d_hat);
    EXPECT_TRUE(std::isfinite(E)) << "near-parallel SS barrier should be finite";
    EXPECT_GT(E, 0.0) << "near-parallel SS barrier should be positive (within d_hat)";

    // Gradient should also be finite
    for (int dof = 0; dof < 4; ++dof) {
        Vec3 g = segment_segment_barrier_gradient(x1, x2, x3, x4, d_hat, dof);
        EXPECT_TRUE(std::isfinite(g.norm())) << "near-parallel SS gradient should be finite for dof=" << dof;
    }

    // Hessian should be finite
    for (int dof = 0; dof < 4; ++dof) {
        Mat33 H = segment_segment_barrier_hessian(x1, x2, x3, x4, d_hat, dof);
        EXPECT_TRUE(std::isfinite(H.norm())) << "near-parallel SS Hessian should be finite for dof=" << dof;
    }

    std::cout << "  E=" << std::scientific << E << "\n";
}

TEST(BarrierEnergy, StressSSNearParallelGradient){
    SSTestPoint tp;
    tp.name = "ss_near_parallel_stress";
    tp.x1 = Vec3(0,0,0);
    tp.x2 = Vec3(1,0,0);
    tp.x3 = Vec3(0.1, 0.3, 0.3);
    tp.x4 = Vec3(0.9, 0.4, 0.3);
    tp.d_hat = 1.0;
    tp.expected_region = SegmentSegmentRegion::Interior;
    EXPECT_TRUE(run_ss_gradient_convergence_test(tp)) << "near-parallel SS gradient convergence";
}
