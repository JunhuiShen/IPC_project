#include "sdf_penalty_energy.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1.0e-10;

double component(const Vec2& v, int c) {
    return c == 0 ? v.x : v.y;
}

double component(const Mat2& A, int r, int c) {
    if (r == 0 && c == 0) return A.a11;
    if (r == 0 && c == 1) return A.a12;
    if (r == 1 && c == 0) return A.a21;
    return A.a22;
}

void perturb(Vec2& x, int comp, double h) {
    if (comp == 0) {
        x.x += h;
    } else {
        x.y += h;
    }
}

bool check_slope_near_two(const std::string& label,
                          const std::vector<double>& hs,
                          const std::vector<double>& errors,
                          double analytic) {
    const double noise_floor = 1.0e-10 * (1.0 + std::abs(analytic));
    bool saw_reliable_slope = false;
    bool all_below_noise = true;
    bool passed = true;

    std::cout << "  " << label
              << " analytic=" << std::scientific << std::setprecision(8)
              << analytic << "\n";

    for (std::size_t i = 1; i < hs.size(); ++i) {
        if (errors[i] < noise_floor || errors[i - 1] < noise_floor) {
            std::cout << "    h=" << std::scientific << hs[i]
                      << " err=" << errors[i]
                      << " (round-off regime, skipped)\n";
            continue;
        }

        all_below_noise = false;
        const double slope = std::log(errors[i - 1] / errors[i]) /
                             std::log(hs[i - 1] / hs[i]);
        std::cout << "    h=" << std::scientific << hs[i]
                  << " err=" << errors[i]
                  << " slope=" << std::fixed << std::setprecision(2)
                  << slope << "\n";

        saw_reliable_slope = true;
        if (std::abs(slope - 2.0) > 0.05) {
            ADD_FAILURE() << label << " finite-difference slope " << slope
                          << " is outside [1.95, 2.05]";
            passed = false;
        }
    }

    if (all_below_noise) {
        std::cout << "    all errors below noise floor; exact match\n";
        return true;
    }

    if (!saw_reliable_slope) {
        ADD_FAILURE() << label << " had no reliable finite-difference slope data";
        return false;
    }
    return passed;
}

template <typename EnergyFn>
bool check_energy_gradient_component(const std::string& label,
                                     const Vec2& x,
                                     int comp,
                                     double analytic,
                                     EnergyFn energy) {
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    std::vector<double> errors;
    errors.reserve(hs.size());

    for (double h : hs) {
        Vec2 xp = x;
        Vec2 xm = x;
        perturb(xp, comp, h);
        perturb(xm, comp, -h);
        const double fd = (energy(xp) - energy(xm)) / (2.0 * h);
        errors.push_back(std::abs(fd - analytic));
    }

    return check_slope_near_two(label, hs, errors, analytic);
}

template <typename GradFn>
bool check_gradient_hessian_component(const std::string& label,
                                      const Vec2& x,
                                      int perturb_comp,
                                      int grad_comp,
                                      double analytic,
                                      GradFn gradient) {
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    std::vector<double> errors;
    errors.reserve(hs.size());

    for (double h : hs) {
        Vec2 xp = x;
        Vec2 xm = x;
        perturb(xp, perturb_comp, h);
        perturb(xm, perturb_comp, -h);
        const double fd = (component(gradient(xp), grad_comp) -
                           component(gradient(xm), grad_comp)) / (2.0 * h);
        errors.push_back(std::abs(fd - analytic));
    }

    return check_slope_near_two(label, hs, errors, analytic);
}

template <typename SDF>
bool check_sdf_convergence(const std::string& name, const SDF& sdf, const Vec2& x,
                           double k, double eps) {
    auto energy = [&](const Vec2& x_eval) {
        return sdf_penalty_energy(evaluate_sdf(sdf, x_eval), k, eps);
    };
    auto gradient = [&](const Vec2& x_eval) {
        return sdf_penalty_gradient(evaluate_sdf(sdf, x_eval), k, eps);
    };

    const Vec2 g = sdf_penalty_gradient(evaluate_sdf(sdf, x), k, eps);
    const Mat2 H = sdf_penalty_hessian(evaluate_sdf(sdf, x), k, eps);

    bool passed = true;
    for (int comp = 0; comp < 2; ++comp) {
        passed &= check_energy_gradient_component(
            name + " dE/dx comp " + std::to_string(comp),
            x, comp, component(g, comp), energy);
    }

    for (int perturb_comp = 0; perturb_comp < 2; ++perturb_comp) {
        for (int grad_comp = 0; grad_comp < 2; ++grad_comp) {
            passed &= check_gradient_hessian_component(
                name + " dgrad/dx row " + std::to_string(grad_comp) +
                    " col " + std::to_string(perturb_comp),
                x, perturb_comp, grad_comp,
                component(H, grad_comp, perturb_comp), gradient);
        }
    }

    return passed;
}

} // namespace

TEST(GroundSDF, Evaluate) {
    const GroundSDF ground{0.25};
    const SDFEvaluation sdf = evaluate_sdf(ground, Vec2{2.0, -0.75});

    EXPECT_NEAR(sdf.phi, -1.0, kTol);
    EXPECT_NEAR(sdf.grad_phi.x, 0.0, kTol);
    EXPECT_NEAR(sdf.grad_phi.y, 1.0, kTol);
    EXPECT_NEAR(sdf.hess_phi.a11, 0.0, kTol);
    EXPECT_NEAR(sdf.hess_phi.a12, 0.0, kTol);
    EXPECT_NEAR(sdf.hess_phi.a21, 0.0, kTol);
    EXPECT_NEAR(sdf.hess_phi.a22, 0.0, kTol);
}

TEST(CircleSDF, Evaluate) {
    const CircleSDF circle{{1.0, -2.0}, 0.5};
    const SDFEvaluation sdf = evaluate_sdf(circle, Vec2{1.3, -1.6});

    EXPECT_NEAR(sdf.phi, 0.0, kTol);
    EXPECT_NEAR(sdf.grad_phi.x, 0.6, kTol);
    EXPECT_NEAR(sdf.grad_phi.y, 0.8, kTol);
}

TEST(SDFHeaviside2D, PiecewiseValues) {
    const double eps = 0.1;
    EXPECT_EQ(sdf_heaviside(-1.0, eps), 1.0);
    EXPECT_EQ(sdf_heaviside(0.0, eps), 1.0);
    EXPECT_NEAR(sdf_heaviside(0.04, eps), 0.6, kTol);
    EXPECT_NEAR(sdf_heaviside(eps, eps), 0.0, kTol);
    EXPECT_EQ(sdf_heaviside(2.0 * eps, eps), 0.0);

    EXPECT_THROW(sdf_heaviside(0.0, 0.0), std::runtime_error);
}

TEST(SDFHeaviside2D, GradientPiecewise) {
    const double eps = 0.1;
    EXPECT_EQ(sdf_heaviside_gradient(-1.0, eps), 0.0);
    EXPECT_EQ(sdf_heaviside_gradient(0.0, eps), 0.0);
    EXPECT_NEAR(sdf_heaviside_gradient(0.04, eps), -10.0, kTol);
    EXPECT_EQ(sdf_heaviside_gradient(eps, eps), 0.0);
    EXPECT_EQ(sdf_heaviside_gradient(2.0 * eps, eps), 0.0);

    EXPECT_THROW(sdf_heaviside_gradient(0.0, -1.0), std::runtime_error);
}

TEST(SDFPenalty2D, EnergyBehavior) {
    const GroundSDF ground{0.0};
    const double k = 100.0;
    const double eps = 0.1;

    EXPECT_EQ(sdf_penalty_energy(evaluate_sdf(ground, {0.0, 0.5}), k, eps), 0.0);
    EXPECT_NEAR(sdf_penalty_energy(evaluate_sdf(ground, {0.0, -0.2}), k, eps),
                0.5 * k * 0.3 * 0.3, kTol);
    EXPECT_NEAR(sdf_penalty_energy(evaluate_sdf(ground, {0.0, 0.04}), k, eps),
                0.5 * k * 0.06 * 0.06, kTol);

    const Vec2 g = sdf_penalty_gradient(evaluate_sdf(ground, {0.0, 0.04}), k, eps);
    EXPECT_NEAR(g.x, 0.0, kTol);
    EXPECT_NEAR(g.y, -6.0, kTol);

    const Mat2 H = sdf_penalty_hessian(evaluate_sdf(ground, {0.0, 0.04}), k, eps);
    EXPECT_NEAR(H.a11, 0.0, kTol);
    EXPECT_NEAR(H.a12, 0.0, kTol);
    EXPECT_NEAR(H.a21, 0.0, kTol);
    EXPECT_NEAR(H.a22, k, kTol);

    const Vec2 inside_g = sdf_penalty_gradient(evaluate_sdf(ground, {0.0, -0.2}), k, eps);
    const Mat2 inside_H = sdf_penalty_hessian(evaluate_sdf(ground, {0.0, -0.2}), k, eps);
    EXPECT_NEAR(inside_g.x, 0.0, kTol);
    EXPECT_NEAR(inside_g.y, -30.0, kTol);
    EXPECT_NEAR(inside_H.a11, 0.0, kTol);
    EXPECT_NEAR(inside_H.a12, 0.0, kTol);
    EXPECT_NEAR(inside_H.a21, 0.0, kTol);
    EXPECT_NEAR(inside_H.a22, k, kTol);
}

TEST(SDFPenalty2D, HardQuadraticWithZeroEps) {
    const GroundSDF ground{0.0};
    const double k = 50.0;
    const double eps = 0.0;

    EXPECT_EQ(sdf_penalty_energy(evaluate_sdf(ground, {0.0, 0.5}), k, eps), 0.0);
    EXPECT_NEAR(sdf_penalty_energy(evaluate_sdf(ground, {0.0, -0.2}), k, eps),
                0.5 * k * 0.2 * 0.2, kTol);

    const Vec2 g = sdf_penalty_gradient(evaluate_sdf(ground, {0.0, -0.2}), k, eps);
    EXPECT_NEAR(g.x, 0.0, kTol);
    EXPECT_NEAR(g.y, -10.0, kTol);
}

TEST(SDFPenalty2D, GroundFiniteDifferenceConvergence) {
    const GroundSDF ground{0.0};
    EXPECT_TRUE(check_sdf_convergence("ground", ground, Vec2{0.4, 0.04}, 37.0, 0.1));
}

TEST(SDFPenalty2D, CircleFiniteDifferenceConvergence) {
    const CircleSDF circle{{0.0, 0.0}, 1.0};
    EXPECT_TRUE(check_sdf_convergence("circle", circle, Vec2{1.04, 0.2}, 23.0, 0.15));
}
