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

bool approx(double a, double b, double tol = kTol){
    return std::abs(a - b) <= tol * (1.0 + std::abs(a) + std::abs(b));
}

void require(bool cond, const std::string& msg){
    if (!cond) {
        std::cerr << "TEST FAILED: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------------------
//  FD helpers for scalar barrier
// ---------------------------------------------------------------------------

double scalar_barrier_fd(double delta, double d_hat, double h){
    return (scalar_barrier(delta + h, d_hat) - scalar_barrier(delta - h, d_hat)) / (2.0 * h);
}

double scalar_barrier_gradient_fd(double delta, double d_hat, double h){
    return (scalar_barrier_gradient(delta + h, d_hat) - scalar_barrier_gradient(delta - h, d_hat)) / (2.0 * h);
}

// ---------------------------------------------------------------------------
//  FD helper for node-triangle barrier energy (for gradient check)
// ---------------------------------------------------------------------------

double node_triangle_barrier_fd(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int which_vec, int comp, double h){
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

// ---------------------------------------------------------------------------
//  FD helper for node-triangle barrier gradient (for Hessian check)
// ---------------------------------------------------------------------------

double node_triangle_gradient_fd(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int v1, int k, int v2, int l, double h){
    Vec3 xp = x, x1p = x1, x2p = x2, x3p = x3;
    Vec3 xm = x, x1m = x1, x2m = x2, x3m = x3;
    switch (v1) {
        case 0: xp(k)  += h; xm(k)  -= h; break;
        case 1: x1p(k) += h; x1m(k) -= h; break;
        case 2: x2p(k) += h; x2m(k) -= h; break;
        case 3: x3p(k) += h; x3m(k) -= h; break;
        default: std::exit(EXIT_FAILURE);
    }

    auto gp = node_triangle_barrier_gradient(xp, x1p, x2p, x3p, d_hat);
    auto gm = node_triangle_barrier_gradient(xm, x1m, x2m, x3m, d_hat);

    auto get_grad = [](const NodeTriangleBarrierResult& r, int v, int c) -> double {
        switch (v) {
            case 0: return r.grad_x(c);
            case 1: return r.grad_x1(c);
            case 2: return r.grad_x2(c);
            case 3: return r.grad_x3(c);
            default: return 0.0;
        }
    };

    return (get_grad(gp, v2, l) - get_grad(gm, v2, l)) / (2.0 * h);
}

// ---------------------------------------------------------------------------
//  FD helpers for segment-segment barrier
// ---------------------------------------------------------------------------

double segment_segment_barrier_fd(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int which_vec, int comp, double h){
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

double segment_segment_gradient_fd(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int v1, int k, int v2, int l, double h){
    Vec3 x1p = x1, x2p = x2, x3p = x3, x4p = x4;
    Vec3 x1m = x1, x2m = x2, x3m = x3, x4m = x4;
    switch (v1) {
        case 0: x1p(k) += h; x1m(k) -= h; break;
        case 1: x2p(k) += h; x2m(k) -= h; break;
        case 2: x3p(k) += h; x3m(k) -= h; break;
        case 3: x4p(k) += h; x4m(k) -= h; break;
        default: std::exit(EXIT_FAILURE);
    }

    auto gp = segment_segment_barrier_gradient(x1p, x2p, x3p, x4p, d_hat);
    auto gm = segment_segment_barrier_gradient(x1m, x2m, x3m, x4m, d_hat);

    auto get_grad = [](const SegmentSegmentBarrierResult& r, int v, int c) -> double {
        switch (v) {
            case 0: return r.grad_x1(c);
            case 1: return r.grad_x2(c);
            case 2: return r.grad_x3(c);
            case 3: return r.grad_x4(c);
            default: return 0.0;
        }
    };

    return (get_grad(gp, v2, l) - get_grad(gm, v2, l)) / (2.0 * h);
}

// ---------------------------------------------------------------------------
//  Convergence test infrastructure
// ---------------------------------------------------------------------------

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

bool check_convergence(const std::string& label, double analytic, const std::vector<double>& hs, const std::vector<double>& errors, double noise_scale = 1e-10, bool verbose = true){
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
            std::cout << "    (all errors below noise floor — exact match)\n";
        return true;
    }

    if (!saw_good_slope) {
        std::cerr << "  FAIL: no reliable slope data for " << label << "\n";
        passed = false;
    }
    return passed;
}

// ===========================================================================
//  Test 1: scalar_barrier_gradient convergence
// ===========================================================================

void test_scalar_barrier_gradient_convergence(){
    std::cout << "=== Test 1: scalar_barrier_gradient convergence ===\n";
    const double d_hat = 1.0, delta = 0.4;
    const double analytic = scalar_barrier_gradient(delta, d_hat);
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    std::vector<double> errors;
    for (auto h : hs) errors.push_back(std::abs(scalar_barrier_fd(delta, d_hat, h) - analytic));
    require(check_convergence("b'", analytic, hs, errors), "scalar barrier gradient convergence");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 2: scalar_barrier_hessian convergence
// ===========================================================================

void test_scalar_barrier_hessian_convergence(){
    std::cout << "=== Test 2: scalar_barrier_hessian convergence ===\n";
    const double d_hat = 1.0, delta = 0.4;
    const double analytic = scalar_barrier_hessian(delta, d_hat);
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    std::vector<double> errors;
    for (auto h : hs) errors.push_back(std::abs(scalar_barrier_gradient_fd(delta, d_hat, h) - analytic));
    require(check_convergence("b''", analytic, hs, errors), "scalar barrier hessian convergence");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 3: barrier zero outside activation
// ===========================================================================

void test_barrier_zero_outside_activation(){
    std::cout << "=== Test 3: barrier zero outside activation ===\n";
    const Vec3 x1(0,0,0), x2(1,0,0), x3(0,1,0), x(0.25,0.25,2.0);
    const auto r = node_triangle_barrier_gradient(x, x1, x2, x3, 1.0);
    require(approx(r.energy, 0.0), "inactive energy");
    require(approx(r.barrier_derivative, 0.0), "inactive derivative");
    for (int k = 0; k < 3; ++k) {
        require(approx(r.grad_x(k), 0.0), "inactive grad");
        require(approx(r.grad_x1(k), 0.0), "inactive grad");
        require(approx(r.grad_x2(k), 0.0), "inactive grad");
        require(approx(r.grad_x3(k), 0.0), "inactive grad");
    }
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 4: partition of force (node-triangle)
// ===========================================================================

void test_partition_of_force(){
    std::cout << "=== Test 4: partition of force ===\n";

    auto check = [](const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, const std::string& label){
        const auto r = node_triangle_barrier_gradient(x, x1, x2, x3, 1.0);
        for (int k = 0; k < 3; ++k) {
            double sum_k = r.grad_x(k) + r.grad_x1(k) + r.grad_x2(k) + r.grad_x3(k);
            require(std::abs(sum_k) < 1e-12, label + ": total gradient should sum to zero");
        }
    };

    check(Vec3(0.25,0.25,0.3), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), "face");
    check(Vec3(0.5,-0.2,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), "edge");
    check(Vec3(-0.2,-0.3,0.1), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), "vertex");

    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Node-triangle gradient convergence
// ===========================================================================

void run_gradient_convergence_test(const TestPoint& tp){
    std::cout << "=== Gradient convergence: " << tp.name << " ===\n";

    const auto r = node_triangle_barrier_gradient(tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat);
    require(r.distance_result.region == tp.expected_region,
            tp.name + ": unexpected region " + to_string(r.distance_result.region));

    std::cout << "  Region: " << to_string(r.distance_result.region)
              << ",  delta = " << r.distance << "\n";

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
            if (std::abs(analytic[v][k]) < 1e-14) {
                double fd_fine = node_triangle_barrier_fd(tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v, k, hs.back());
                if (std::abs(fd_fine) > 1e-8) {
                    std::cerr << "  FAIL: " << dof_names[v] << "(" << k << ") analytic=0 but fd=" << fd_fine << "\n";
                    all_passed = false;
                }
                continue;
            }

            std::vector<double> errors;
            for (auto h : hs)
                errors.push_back(std::abs(node_triangle_barrier_fd(tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v, k, h) - analytic[v][k]));

            std::string label = std::string("d/d(") + dof_names[v] + ")_" + std::to_string(k);
            if (!check_convergence(label, analytic[v][k], hs, errors, 1e-10)) all_passed = false;
        }
    }

    require(all_passed, tp.name + ": gradient convergence failed");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Node-triangle Hessian convergence
// ===========================================================================

void run_hessian_convergence_test(const TestPoint& tp){
    std::cout << "=== Hessian convergence: " << tp.name << " ===\n";

    const auto hr = node_triangle_barrier_hessian(tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat);
    require(hr.distance_result.region == tp.expected_region,
            tp.name + ": unexpected region " + to_string(hr.distance_result.region));

    std::cout << "  Region: " << to_string(hr.distance_result.region)
              << ",  delta = " << hr.distance << "\n";

    // Symmetry check
    double max_asym = 0.0;
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            max_asym = std::max(max_asym, std::abs(hr.hessian(i,j) - hr.hessian(j,i)));
    std::cout << "  Symmetry check: max |H - H^T| = " << std::scientific << max_asym << "\n";
    require(max_asym < 1e-12, tp.name + ": Hessian not symmetric");

    const char* dof_names[4] = {"x", "x1", "x2", "x3"};
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;
    int tested = 0, skipped_zero = 0;

    for (int v1 = 0; v1 < 4; ++v1) {
        for (int k = 0; k < 3; ++k) {
            for (int v2 = 0; v2 < 4; ++v2) {
                for (int l = 0; l < 3; ++l) {

                    double analytic_val = hr.hessian(3*v1+k, 3*v2+l);

                    if (std::abs(analytic_val) < 1e-14) {
                        double fd_fine = node_triangle_gradient_fd(
                                tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v1, k, v2, l, hs.back());
                        if (std::abs(fd_fine) > 1e-6) {
                            std::cerr << "  FAIL: H(" << dof_names[v1] << k << ","
                                      << dof_names[v2] << l << ") analytic=0 but fd=" << fd_fine << "\n";
                            all_passed = false;
                        }
                        skipped_zero++;
                        continue;
                    }

                    std::vector<double> errors;
                    for (auto h : hs) {
                        double fd = node_triangle_gradient_fd(
                                tp.x, tp.x1, tp.x2, tp.x3, tp.d_hat, v1, k, v2, l, h);
                        errors.push_back(std::abs(fd - analytic_val));
                    }

                    std::string label = std::string("H(") + dof_names[v1] + std::to_string(k)
                                        + "," + dof_names[v2] + std::to_string(l) + ")";

                    if (!check_convergence(label, analytic_val, hs, errors, 1e-9, false)) {
                        check_convergence(label, analytic_val, hs, errors, 1e-9, true);
                        all_passed = false;
                    }
                    tested++;
                }
            }
        }
    }

    std::cout << "  Tested " << tested << " entries, skipped " << skipped_zero << " zero entries\n";
    require(all_passed, tp.name + ": Hessian convergence failed");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Segment-segment: barrier zero outside activation
// ===========================================================================

void test_ss_barrier_zero_outside_activation(){
    std::cout << "=== SS Test: barrier zero outside activation ===\n";
    const Vec3 x1(0,0,0), x2(1,0,0), x3(0.5,-3,2), x4(0.5,3,2);
    const auto r = segment_segment_barrier_gradient(x1, x2, x3, x4, 1.0);
    require(approx(r.energy, 0.0), "ss inactive energy");
    require(approx(r.barrier_derivative, 0.0), "ss inactive derivative");
    for (int k = 0; k < 3; ++k) {
        require(approx(r.grad_x1(k), 0.0), "ss inactive grad");
        require(approx(r.grad_x2(k), 0.0), "ss inactive grad");
        require(approx(r.grad_x3(k), 0.0), "ss inactive grad");
        require(approx(r.grad_x4(k), 0.0), "ss inactive grad");
    }
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Segment-segment: partition of force
// ===========================================================================

void test_ss_partition_of_force(){
    std::cout << "=== SS Test: partition of force ===\n";

    auto check = [](const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, const std::string& label){
        const auto r = segment_segment_barrier_gradient(x1, x2, x3, x4, 2.0);
        for (int k = 0; k < 3; ++k) {
            double sum_k = r.grad_x1(k) + r.grad_x2(k) + r.grad_x3(k) + r.grad_x4(k);
            require(std::abs(sum_k) < 1e-12, label + ": total gradient should sum to zero");
        }
    };

    check(Vec3(0,0,0), Vec3(1,0,0), Vec3(0.5,-1,0.5), Vec3(0.5,1,0.5), "interior");
    check(Vec3(0,0,0), Vec3(1,0,0), Vec3(-1,-1,0.3), Vec3(-1,1,0.3), "edge_s0");
    check(Vec3(0,0,0), Vec3(1,0,0), Vec3(-1,-1,0.3), Vec3(-1,-2,0.3), "corner");

    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Segment-segment gradient convergence
// ===========================================================================

void run_ss_gradient_convergence_test(const SSTestPoint& tp){
    std::cout << "=== SS Gradient convergence: " << tp.name << " ===\n";

    const auto r = segment_segment_barrier_gradient(tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat);
    require(r.distance_result.region == tp.expected_region,
            tp.name + ": unexpected region " + to_string(r.distance_result.region));

    std::cout << "  Region: " << to_string(r.distance_result.region)
              << ",  delta = " << r.distance << "\n";

    const char* dof_names[4] = {"x1", "x2", "x3", "x4"};
    double analytic[4][3];
    for (int k = 0; k < 3; ++k) {
        analytic[0][k] = r.grad_x1(k);
        analytic[1][k] = r.grad_x2(k);
        analytic[2][k] = r.grad_x3(k);
        analytic[3][k] = r.grad_x4(k);
    }

    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;

    for (int v = 0; v < 4; ++v) {
        for (int k = 0; k < 3; ++k) {
            if (std::abs(analytic[v][k]) < 1e-14) {
                double fd_fine = segment_segment_barrier_fd(tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v, k, hs.back());
                if (std::abs(fd_fine) > 1e-8) {
                    std::cerr << "  FAIL: " << dof_names[v] << "(" << k << ") analytic=0 but fd=" << fd_fine << "\n";
                    all_passed = false;
                }
                continue;
            }

            std::vector<double> errors;
            for (auto h : hs)
                errors.push_back(std::abs(segment_segment_barrier_fd(tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v, k, h) - analytic[v][k]));

            std::string label = std::string("d/d(") + dof_names[v] + ")_" + std::to_string(k);
            if (!check_convergence(label, analytic[v][k], hs, errors, 1e-9)) all_passed = false;
        }
    }

    require(all_passed, tp.name + ": SS gradient convergence failed");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Segment-segment Hessian convergence
// ===========================================================================

void run_ss_hessian_convergence_test(const SSTestPoint& tp){
    std::cout << "=== SS Hessian convergence: " << tp.name << " ===\n";

    const auto hr = segment_segment_barrier_hessian(tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat);
    require(hr.distance_result.region == tp.expected_region,
            tp.name + ": unexpected region " + to_string(hr.distance_result.region));

    std::cout << "  Region: " << to_string(hr.distance_result.region)
              << ",  delta = " << hr.distance << "\n";

    // Symmetry check
    double max_asym = 0.0;
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            max_asym = std::max(max_asym, std::abs(hr.hessian(i,j) - hr.hessian(j,i)));
    std::cout << "  Symmetry check: max |H - H^T| = " << std::scientific << max_asym << "\n";
    require(max_asym < 1e-12, tp.name + ": SS Hessian not symmetric");

    const char* dof_names[4] = {"x1", "x2", "x3", "x4"};
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;
    int tested = 0, skipped_zero = 0;

    for (int v1 = 0; v1 < 4; ++v1) {
        for (int k = 0; k < 3; ++k) {
            for (int v2 = 0; v2 < 4; ++v2) {
                for (int l = 0; l < 3; ++l) {

                    double analytic_val = hr.hessian(3*v1+k, 3*v2+l);

                    if (std::abs(analytic_val) < 1e-14) {
                        double fd_fine = segment_segment_gradient_fd(
                                tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v1, k, v2, l, hs.back());
                        if (std::abs(fd_fine) > 1e-6) {
                            std::cerr << "  FAIL: H(" << dof_names[v1] << k << ","
                                      << dof_names[v2] << l << ") analytic=0 but fd=" << fd_fine << "\n";
                            all_passed = false;
                        }
                        skipped_zero++;
                        continue;
                    }

                    std::vector<double> errors;
                    for (auto h : hs) {
                        double fd = segment_segment_gradient_fd(
                                tp.x1, tp.x2, tp.x3, tp.x4, tp.d_hat, v1, k, v2, l, h);
                        errors.push_back(std::abs(fd - analytic_val));
                    }

                    std::string label = std::string("H(") + dof_names[v1] + std::to_string(k)
                                        + "," + dof_names[v2] + std::to_string(l) + ")";

                    if (!check_convergence(label, analytic_val, hs, errors, 1e-9, false)) {
                        check_convergence(label, analytic_val, hs, errors, 1e-9, true);
                        all_passed = false;
                    }
                    tested++;
                }
            }
        }
    }

    std::cout << "  Tested " << tested << " entries, skipped " << skipped_zero << " zero entries\n";
    require(all_passed, tp.name + ": SS Hessian convergence failed");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  main
// ===========================================================================

int main(){
    // --- Scalar barrier tests ---
    test_scalar_barrier_gradient_convergence();
    test_scalar_barrier_hessian_convergence();

    // --- Node-triangle tests ---
    test_barrier_zero_outside_activation();
    test_partition_of_force();

    std::vector<TestPoint> nt_test_points = {
            {"face_interior",
                    Vec3(0.25, 0.25, 0.3),
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::FaceInterior},

            {"edge_12",
                    Vec3(0.5, -0.2, 0.1),
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Edge12},

            {"edge_23",
                    Vec3(0.7, 0.7, 0.1),
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Edge23},

            {"edge_31",
                    Vec3(-0.15, 0.5, 0.1),
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Edge31},

            {"vertex_1",
                    Vec3(-0.2, -0.3, 0.1),
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Vertex1},

            {"vertex_2",
                    Vec3(1.4, -0.1, 0.1),
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Vertex2},

            {"vertex_3",
                    Vec3(-0.1, 1.4, 0.1),
                    Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), 1.0,
                    NodeTriangleRegion::Vertex3},
    };

    for (const auto& tp : nt_test_points)
        run_gradient_convergence_test(tp);

    for (const auto& tp : nt_test_points)
        run_hessian_convergence_test(tp);

    // --- Segment-segment tests ---
    test_ss_barrier_zero_outside_activation();
    test_ss_partition_of_force();

    std::vector<SSTestPoint> ss_test_points = {
            // Interior: skew segments, distance < d_hat
            {"ss_interior",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(0.5,-1,0.5), Vec3(0.5,1,0.5), 2.0,
                    SegmentSegmentRegion::Interior},

            // Edge s=0: x1 vs segment (x3,x4)
            {"ss_edge_s0",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(-1,-1,0.3), Vec3(-1,1,0.3), 3.0,
                    SegmentSegmentRegion::Edge_s0},

            // Edge s=1: x2 vs segment (x3,x4)
            {"ss_edge_s1",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(2,-1,0.3), Vec3(2,1,0.3), 3.0,
                    SegmentSegmentRegion::Edge_s1},

            // Edge t=0: x3 vs segment (x1,x2)
            {"ss_edge_t0",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(0.5,0.3,0.3), Vec3(0.5,1.3,0.3), 2.0,
                    SegmentSegmentRegion::Edge_t0},

            // Edge t=1: x4 vs segment (x1,x2)
            {"ss_edge_t1",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(0.5,-1.3,0.3), Vec3(0.5,-0.3,0.3), 2.0,
                    SegmentSegmentRegion::Edge_t1},

            // Corner s=0,t=0: x1 vs x3
            {"ss_corner_s0t0",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(-0.5,-0.5,0.3), Vec3(-0.5,-1.5,0.3), 2.0,
                    SegmentSegmentRegion::Corner_s0t0},

            // Corner s=0,t=1: x1 vs x4
            {"ss_corner_s0t1",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(-1.5,-1.5,0.3), Vec3(-0.5,-0.5,0.3), 2.0,
                    SegmentSegmentRegion::Corner_s0t1},

            // Corner s=1,t=0: x2 vs x3
            {"ss_corner_s1t0",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(1.5,-0.5,0.3), Vec3(1.5,-1.5,0.3), 2.0,
                    SegmentSegmentRegion::Corner_s1t0},

            // Corner s=1,t=1: x2 vs x4
            {"ss_corner_s1t1",
                    Vec3(0,0,0), Vec3(1,0,0),
                    Vec3(2.5,-1.5,0.3), Vec3(1.5,-0.5,0.3), 2.0,
                    SegmentSegmentRegion::Corner_s1t1},
    };

    for (const auto& tp : ss_test_points)
        run_ss_gradient_convergence_test(tp);

    for (const auto& tp : ss_test_points)
        run_ss_hessian_convergence_test(tp);

    std::cout << "\n========================================\n"
              << "All barrier energy tests passed.\n"
              << "========================================\n";
    return 0;
}