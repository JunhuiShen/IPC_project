#include "rb_inertia.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

double component(const Vec2& v, int c) {
    return c == 0 ? v.x : v.y;
}

double component(const Mat2& A, int r, int c) {
    if (r == 0 && c == 0) return A.a11;
    if (r == 0 && c == 1) return A.a12;
    if (r == 1 && c == 0) return A.a21;
    return A.a22;
}

double vec_entry(const Vec2& v, int index) {
    return index == 0 ? v.x : v.y;
}

void perturb(Vec2& y, int comp, double h) {
    if (comp == 0) {
        y.x += h;
    } else {
        y.y += h;
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
                          << " is not near 2";
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

template <typename GradFn>
bool check_translation_gradient_hessian_component(const std::string& label,
                                                  const Vec2& y,
                                                  int perturb_comp,
                                                  int grad_comp,
                                                  double analytic,
                                                  GradFn gradient) {
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    std::vector<double> errors;
    errors.reserve(hs.size());

    for (double h : hs) {
        Vec2 yp = y;
        Vec2 ym = y;
        perturb(yp, perturb_comp, h);
        perturb(ym, perturb_comp, -h);
        const double fd = (component(gradient(yp), grad_comp) -
                           component(gradient(ym), grad_comp)) / (2.0 * h);
        errors.push_back(std::abs(fd - analytic));
    }

    return check_slope_near_two(label, hs, errors, analytic);
}

template <typename GradFn>
bool check_rotation_gradient_hessian(const std::string& label,
                                     double theta,
                                     double analytic,
                                     GradFn gradient) {
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    std::vector<double> errors;
    errors.reserve(hs.size());

    for (double h : hs) {
        const double fd = (gradient(theta + h) - gradient(theta - h)) / (2.0 * h);
        errors.push_back(std::abs(fd - analytic));
    }

    return check_slope_near_two(label, hs, errors, analytic);
}

} // namespace

TEST(RigidBodyInertia, TranslationGradientHessianHasSlopeTwo) {
    const Vec2 y{0.31, -0.42};
    const Vec2 y_n{-0.13, 0.27};
    const Vec2 vhat_n{1.7, -0.6};
    const double dt = 0.037;
    const double m_total = 4.25;

    auto gradient = [&](const Vec2& y_eval) {
        return inertia_translation_gradient(y_eval, y_n, vhat_n, dt, m_total);
    };

    const Mat2 H = inertia_translation_hessian(m_total);

    bool passed = true;
    for (int perturb_comp = 0; perturb_comp < 2; ++perturb_comp) {
        for (int grad_comp = 0; grad_comp < 2; ++grad_comp) {
            passed &= check_translation_gradient_hessian_component(
                "rb inertia translation dgrad/dy row " + std::to_string(grad_comp) +
                    " col " + std::to_string(perturb_comp),
                y, perturb_comp, grad_comp,
                component(H, grad_comp, perturb_comp), gradient);
        }
    }

    EXPECT_TRUE(passed);
}

TEST(RigidBodyInertia, RotationGradientHessianHasSlopeTwo) {
    const Vec U = {
        {-0.8, 0.35},
        {0.45, -0.55},
        {0.9, 0.7},
    };
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const double theta = 0.61;
    const double theta_n = -0.28;
    const double omega_n = 1.35;
    const double dt = 0.041;
    const Mat2 inertia_tensor = inertia_body_tensor(U, masses);

    auto gradient = [&](double theta_eval) {
        return inertia_rotation_gradient(theta_eval, theta_n, omega_n, inertia_tensor, dt);
    };

    const double H = inertia_rotation_hessian(theta, theta_n, omega_n, inertia_tensor, dt);

    bool passed = true;
    passed &= check_rotation_gradient_hessian("rb inertia rotation dgrad/dtheta", theta, H, gradient);

    EXPECT_TRUE(passed);
}

TEST(RigidBodyInertia, EnergyGradientTestTranslation) {
    const Vec U = {
        {-0.8, 0.35},
        {0.45, -0.55},
        {0.9, 0.7},
    };
    const Vec2 y{0.31, -0.42};
    const Vec2 y_n{-0.13, 0.27};
    const Vec2 vhat_n{1.7, -0.6};
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const double m_total = 3.8;
    const double theta = 0.61;
    const double theta_n = -0.28;
    const double omega_n = 1.35;
    const double dt = 0.041;
    const Mat2 I = inertia_body_tensor(U, masses);

    const Vec2 analytic_grad = inertia_translation_gradient(y, y_n, vhat_n, dt, m_total);

    bool passed = true;
    for (int comp = 0; comp < 2; ++comp) {
        const double y_comp = comp == 0 ? y.x : y.y;
        auto energy_of_comp = [&, comp](double val) {
            Vec2 y_eval = y;
            if (comp == 0) y_eval.x = val; else y_eval.y = val;
            return incremental_potential_energy(y_eval, theta, y_n, theta_n, vhat_n, omega_n, dt, m_total, I);
        };
        passed &= check_rotation_gradient_hessian(
            "dE/dy[" + std::to_string(comp) + "]",
            y_comp, component(analytic_grad, comp), energy_of_comp);
    }
    EXPECT_TRUE(passed);
}

TEST(RigidBodyInertia, EnergyGradientTestRotation) {
    const Vec U = {
        {-0.8, 0.35},
        {0.45, -0.55},
        {0.9, 0.7},
    };
    const Vec2 y{0.31, -0.42};
    const Vec2 y_n{-0.13, 0.27};
    const Vec2 vhat_n{1.7, -0.6};
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const double m_total = 3.8;
    const double theta = 0.61;
    const double theta_n = -0.28;
    const double omega_n = 1.35;
    const double dt = 0.041;
    const Mat2 I = inertia_body_tensor(U, masses);

    const double analytic_grad = inertia_rotation_gradient(theta, theta_n, omega_n, I, dt);

    auto energy_of_theta = [&](double theta_eval) {
        return incremental_potential_energy(y, theta_eval, y_n, theta_n, vhat_n, omega_n, dt, m_total, I);
    };

    const bool passed = check_rotation_gradient_hessian(
        "dE/dtheta", theta, analytic_grad, energy_of_theta);
    EXPECT_TRUE(passed);
}
