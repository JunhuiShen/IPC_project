#include "IPC_math.h"
#include "corotated_energy.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
//  Utilities
// ===========================================================================

void require(bool cond, const std::string& msg){
    if (!cond) {
        std::cerr << "TEST FAILED: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

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
//  Test configurations
// ===========================================================================

constexpr double kMu = 2.0;
constexpr double kLambda = 5.0;

struct TestTriangle {
    double ref_area;
    Mat22 Dm_inv;
};

TestTriangle MakeTestTriangle(){
    Vec2 X0(0.0, 0.0), X1(1.2, 0.1), X2(0.2, 0.9);
    Mat22 Dm_local;
    Dm_local.col(0) = X1 - X0;
    Dm_local.col(1) = X2 - X0;
    TestTriangle t;
    t.ref_area = 0.5 * std::abs(Dm_local.determinant());
    t.Dm_inv = Dm_local.inverse();
    return t;
}

TriangleDef EmbedRestTriangle(){
    TriangleDef def;
    def.x[0] = Vec3(0.0, 0.0, 0.0);
    def.x[1] = Vec3(1.2, 0.1, 0.0);
    def.x[2] = Vec3(0.2, 0.9, 0.0);
    return def;
}

TriangleDef MakeDeformedTriangle(){
    TriangleDef def;
    def.x[0] = Vec3(0.1, -0.2, 0.3);
    def.x[1] = Vec3(1.4, 0.2, -0.1);
    def.x[2] = Vec3(0.0, 1.0, 0.4);
    return def;
}

Mat32 compute_F(const Mat22& Dm_inv, const TriangleDef& def){
    Mat32 Ds;
    Ds.col(0) = def.x[1] - def.x[0];
    Ds.col(1) = def.x[2] - def.x[0];
    return Ds * Dm_inv;
}

// ===========================================================================
//  Test 1: energy is finite
// ===========================================================================

void test_energy_is_finite(){
    std::cout << "=== Test 1: energy is finite ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();
    const double E = corotated_energy(tri.ref_area, tri.Dm_inv, def, kMu, kLambda);
    require(std::isfinite(E), "energy is not finite");
    std::cout << "  E = " << E << "\n";
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 2: rest state has zero energy and zero gradient
// ===========================================================================

void test_rest_state(){
    std::cout << "=== Test 2: rest state has zero energy and gradient ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = EmbedRestTriangle();
    const double E = corotated_energy(tri.ref_area, tri.Dm_inv, def, kMu, kLambda);

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const auto g = corotated_node_gradient(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    std::cout << "  E = " << std::scientific << E << "\n";
    require(std::abs(E) < 1e-10, "rest energy should be zero");
    for (int i = 0; i < 3; ++i)
        require(g[i].norm() < 1e-10, "rest gradient should be zero");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 3: rotation invariance — rotated rest state has zero energy
// ===========================================================================

void test_rotation_invariance(){
    std::cout << "=== Test 3: rotation invariance ===\n";
    const auto tri = MakeTestTriangle();
    const auto def0 = EmbedRestTriangle();

    Eigen::AngleAxisd aa(0.7, Vec3(1.0, 2.0, -1.0).normalized());
    const Mat33 R = aa.toRotationMatrix();
    TriangleDef def1;
    for (int i = 0; i < 3; ++i) def1.x[i] = R * def0.x[i];

    const double E = corotated_energy(tri.ref_area, tri.Dm_inv, def1, kMu, kLambda);

    const Mat32 F = compute_F(tri.Dm_inv, def1);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const auto g = corotated_node_gradient(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    std::cout << "  E = " << std::scientific << E << "\n";
    require(std::abs(E) < 1e-10, "rotated rest energy should be zero");
    for (int i = 0; i < 3; ++i)
        require(g[i].norm() < 1e-10, "rotated rest gradient should be zero");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 4: translation invariance
// ===========================================================================

void test_translation_invariance(){
    std::cout << "=== Test 4: translation invariance ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    TriangleDef shifted = def;
    Vec3 t(2.5, -1.25, 0.75);
    for (auto& x : shifted.x) x += t;

    const double E0 = corotated_energy(tri.ref_area, tri.Dm_inv, def, kMu, kLambda);
    const double E1 = corotated_energy(tri.ref_area, tri.Dm_inv, shifted, kMu, kLambda);

    std::cout << "  E0 = " << E0 << ",  E1 = " << E1 << ",  diff = " << std::abs(E0 - E1) << "\n";
    require(std::abs(E0 - E1) < 1e-10, "energy should be translation invariant");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 5: nodal gradients sum to zero
// ===========================================================================

void test_gradient_sum_zero(){
    std::cout << "=== Test 5: nodal gradients sum to zero ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const auto g = corotated_node_gradient(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);
    const Vec3 total = g[0] + g[1] + g[2];

    std::cout << "  |sum| = " << total.norm() << "\n";
    require(total.norm() < 1e-10, "nodal gradients should sum to zero");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 6: Hessian symmetry
// ===========================================================================

void test_hessian_symmetry(){
    std::cout << "=== Test 6: Hessian symmetry ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const Mat99 H = corotated_node_hessian(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    double max_asym = (H - H.transpose()).cwiseAbs().maxCoeff();
    std::cout << "  max |H - H^T| = " << std::scientific << max_asym << "\n";
    require(max_asym < 1e-8, "Hessian should be symmetric");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 7: Hessian has translation null modes
// ===========================================================================

void test_hessian_translation_null(){
    std::cout << "=== Test 7: Hessian translation null modes ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const Mat99 H = corotated_node_hessian(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    for (int axis = 0; axis < 3; ++axis) {
        Vec9 t = Vec9::Zero();
        for (int node = 0; node < 3; ++node)
            t(3 * node + axis) = 1.0;
        double res = (H * t).cwiseAbs().maxCoeff();
        std::cout << "  axis " << axis << ": |H * t| = " << std::scientific << res << "\n";
        require(res < 1e-8, "Hessian should have translation null mode");
    }
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 8: gradient convergence (slope = 2)
// ===========================================================================

void test_gradient_convergence(){
    std::cout << "=== Test 8: gradient convergence (slope = 2) ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const auto g = corotated_node_gradient(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;

    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 3; ++c) {
            double analytic = g[i](c);

            if (std::abs(analytic) < 1e-14) continue;

            std::vector<double> errors;
            for (auto h : hs) {
                TriangleDef dp = def, dm = def;
                set_dof(dp, i, c, get_dof(dp, i, c) + h);
                set_dof(dm, i, c, get_dof(dm, i, c) - h);
                double fd = (corotated_energy(tri.ref_area, tri.Dm_inv, dp, kMu, kLambda)
                             - corotated_energy(tri.ref_area, tri.Dm_inv, dm, kMu, kLambda)) / (2.0 * h);
                errors.push_back(std::abs(fd - analytic));
            }

            std::string label = "g[" + std::to_string(i) + "](" + std::to_string(c) + ")";
            if (!check_convergence(label, analytic, hs, errors)) all_passed = false;
        }
    }

    require(all_passed, "gradient convergence failed");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 9: Hessian convergence (slope = 2)
// ===========================================================================

void test_hessian_convergence(){
    std::cout << "=== Test 9: Hessian convergence (slope = 2) ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const Mat99 H = corotated_node_hessian(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    bool all_passed = true;
    int tested = 0, skipped = 0;

    for (int j = 0; j < 3; ++j) {
        for (int d = 0; d < 3; ++d) {
            std::vector<std::array<Vec3, 3>> g_plus_list, g_minus_list;

            for (auto h : hs) {
                TriangleDef dp = def, dm = def;
                set_dof(dp, j, d, get_dof(dp, j, d) + h);
                set_dof(dm, j, d, get_dof(dm, j, d) - h);

                Mat32 Fp = compute_F(tri.Dm_inv, dp);
                CorotatedCache32 cp = buildCorotatedCache(Fp);
                g_plus_list.push_back(corotated_node_gradient(cp, Fp, tri.ref_area, tri.Dm_inv, kMu, kLambda));

                Mat32 Fm = compute_F(tri.Dm_inv, dm);
                CorotatedCache32 cm = buildCorotatedCache(Fm);
                g_minus_list.push_back(corotated_node_gradient(cm, Fm, tri.ref_area, tri.Dm_inv, kMu, kLambda));
            }

            for (int i = 0; i < 3; ++i) {
                for (int c = 0; c < 3; ++c) {
                    double analytic = H(3*i+c, 3*j+d);

                    if (std::abs(analytic) < 1e-14) {
                        double fd_fine = (g_plus_list.back()[i](c) - g_minus_list.back()[i](c)) / (2.0 * hs.back());
                        if (std::abs(fd_fine) > 1e-6) {
                            std::cerr << "  FAIL: H(" << i << c << "," << j << d << ") analytic=0 but fd=" << fd_fine << "\n";
                            all_passed = false;
                        }
                        skipped++;
                        continue;
                    }

                    std::vector<double> errors;
                    for (std::size_t hi = 0; hi < hs.size(); ++hi) {
                        double fd = (g_plus_list[hi][i](c) - g_minus_list[hi][i](c)) / (2.0 * hs[hi]);
                        errors.push_back(std::abs(fd - analytic));
                    }

                    std::string label = "H(" + std::to_string(3*i+c) + "," + std::to_string(3*j+d) + ")";
                    if (!check_convergence(label, analytic, hs, errors, 1e-9, false)) {
                        check_convergence(label, analytic, hs, errors, 1e-9, true);
                        all_passed = false;
                    }
                    tested++;
                }
            }
        }
    }

    std::cout << "  Tested " << tested << " entries, skipped " << skipped << " zero entries\n";
    require(all_passed, "Hessian convergence failed");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  Test 10: directional derivative convergence (slope = 2)
// ===========================================================================

void test_directional_derivative_convergence(){
    std::cout << "=== Test 10: directional derivative convergence (slope = 2) ===\n";
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    TriangleDef dx = ZeroTriangleDef();
    dx.x[0] = Vec3(0.3, -0.7, 0.2);
    dx.x[1] = Vec3(-0.4, 0.1, 0.5);
    dx.x[2] = Vec3(0.25, 0.6, -0.35);
    double norm = std::sqrt(dx.x[0].squaredNorm() + dx.x[1].squaredNorm() + dx.x[2].squaredNorm());
    for (auto& v : dx.x) v /= norm;

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const auto g = corotated_node_gradient(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);
    const double exact = flatten_gradient(g).dot(flatten_def(dx));

    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    std::vector<double> errors;

    for (auto h : hs) {
        const TriangleDef dp = add_scale(def, dx, h);
        const TriangleDef dm = add_scale(def, dx, -h);
        double fd = (corotated_energy(tri.ref_area, tri.Dm_inv, dp, kMu, kLambda)
                     - corotated_energy(tri.ref_area, tri.Dm_inv, dm, kMu, kLambda)) / (2.0 * h);
        errors.push_back(std::abs(fd - exact));
    }

    require(check_convergence("directional", exact, hs, errors), "directional derivative convergence failed");
    std::cout << "  PASSED\n\n";
}

// ===========================================================================
//  main
// ===========================================================================

int main(){
    test_energy_is_finite();
    test_rest_state();
    test_rotation_invariance();
    test_translation_invariance();
    test_gradient_sum_zero();
    test_hessian_symmetry();
    test_hessian_translation_null();
    test_gradient_convergence();
    test_hessian_convergence();
    test_directional_derivative_convergence();

    std::cout << "\n========================================\n"
              << "All corotated energy tests passed.\n"
              << "========================================\n";
    return 0;
}
