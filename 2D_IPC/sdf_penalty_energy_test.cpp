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

double component(const RigidSDFGradient& g, int c) {
    if (c == 0) return g.translation.x;
    if (c == 1) return g.translation.y;
    return g.rotation;
}

double component(const RigidSDFHessian& H, int r, int c) {
    if (r < 2 && c < 2) return component(H.translation_translation, r, c);
    if (r < 2 && c == 2) return component(H.translation_rotation, r);
    if (r == 2 && c < 2) return component(H.translation_rotation, c);
    return H.rotation_rotation;
}

Vec2 rotate(const Vec2& x, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    return {c * x.x - s * x.y, s * x.x + c * x.y};
}

void perturb(Vec2& x, int comp, double h) {
    if (comp == 0) {
        x.x += h;
    } else {
        x.y += h;
    }
}

void perturb_rigid_dof(Vec2& y, double& theta, int dof, double h) {
    if (dof == 0) {
        y.x += h;
    } else if (dof == 1) {
        y.y += h;
    } else {
        theta += h;
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

template <typename EnergyFn>
bool check_rigid_energy_gradient_component(const std::string& label,
                                           const Vec2& y,
                                           double theta,
                                           int dof,
                                           double analytic,
                                           EnergyFn energy) {
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    std::vector<double> errors;
    errors.reserve(hs.size());

    for (double h : hs) {
        Vec2 yp = y;
        Vec2 ym = y;
        double thetap = theta;
        double thetam = theta;
        perturb_rigid_dof(yp, thetap, dof, h);
        perturb_rigid_dof(ym, thetam, dof, -h);
        const double fd = (energy(yp, thetap) - energy(ym, thetam)) / (2.0 * h);
        errors.push_back(std::abs(fd - analytic));
    }

    return check_slope_near_two(label, hs, errors, analytic);
}

template <typename GradFn>
bool check_rigid_gradient_hessian_component(const std::string& label,
                                            const Vec2& y,
                                            double theta,
                                            int perturb_dof,
                                            int grad_dof,
                                            double analytic,
                                            GradFn gradient) {
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    std::vector<double> errors;
    errors.reserve(hs.size());

    for (double h : hs) {
        Vec2 yp = y;
        Vec2 ym = y;
        double thetap = theta;
        double thetam = theta;
        perturb_rigid_dof(yp, thetap, perturb_dof, h);
        perturb_rigid_dof(ym, thetam, perturb_dof, -h);
        const double fd = (component(gradient(yp, thetap), grad_dof) -
                           component(gradient(ym, thetam), grad_dof)) / (2.0 * h);
        errors.push_back(std::abs(fd - analytic));
    }

    return check_slope_near_two(label, hs, errors, analytic);
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
                0.5 * k * 0.6 * 0.06 * 0.06, kTol);

    const Vec2 g = sdf_penalty_gradient(evaluate_sdf(ground, {0.0, 0.04}), k, eps);
    EXPECT_NEAR(g.x, 0.0, kTol);
    EXPECT_NEAR(g.y, -5.4, kTol);

    const Mat2 H = sdf_penalty_hessian(evaluate_sdf(ground, {0.0, 0.04}), k, eps);
    EXPECT_NEAR(H.a11, 0.0, kTol);
    EXPECT_NEAR(H.a12, 0.0, kTol);
    EXPECT_NEAR(H.a21, 0.0, kTol);
    EXPECT_NEAR(H.a22, 180.0, kTol);

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

TEST(SDFPenalty2D, RigidBodyGradientAndHessianChainRuleHaveSlopeTwo) {
    const CircleSDF circle{{0.0, 0.0}, 1.0};
    const Vec2 material{0.2, 0.15};
    const Vec2 y{0.9, 0.0};
    const double theta = 0.3;
    const double k = 23.0;
    const double eps = 0.2;

    auto position = [&](const Vec2& y_eval, double theta_eval) {
        return y_eval + rotate(material, theta_eval);
    };
    auto energy = [&](const Vec2& y_eval, double theta_eval) {
        const Vec2 x_eval = position(y_eval, theta_eval);
        return sdf_penalty_energy(evaluate_sdf(circle, x_eval), k, eps);
    };
    auto gradient = [&](const Vec2& y_eval, double theta_eval) {
        const Vec2 x_eval = position(y_eval, theta_eval);
        return sdf_penalty_gradient_rb(evaluate_sdf(circle, x_eval), x_eval, y_eval, k, eps);
    };

    const Vec2 x = position(y, theta);
    const RigidSDFGradient g = sdf_penalty_gradient_rb(evaluate_sdf(circle, x), x, y, k, eps);
    const RigidSDFHessian H = sdf_penalty_hessian_rb(evaluate_sdf(circle, x), x, y, k, eps);

    bool passed = true;
    for (int dof = 0; dof < 3; ++dof) {
        passed &= check_rigid_energy_gradient_component(
                "sdf rigid dE/dq dof " + std::to_string(dof),
                y, theta, dof, component(g, dof), energy);
    }

    for (int perturb_dof = 0; perturb_dof < 3; ++perturb_dof) {
        for (int grad_dof = 0; grad_dof < 3; ++grad_dof) {
            passed &= check_rigid_gradient_hessian_component(
                    "sdf rigid dgrad/dq row " + std::to_string(grad_dof) +
                            " col " + std::to_string(perturb_dof),
                    y, theta, perturb_dof, grad_dof,
                    component(H, grad_dof, perturb_dof), gradient);
        }
    }

    EXPECT_TRUE(passed);
}
