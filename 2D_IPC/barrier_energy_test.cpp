#include "barrier_energy.h"

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

void perturb(Vec& x, int node, int comp, double h) {
    if (comp == 0) {
        x[node].x += h;
    } else {
        x[node].y += h;
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

template <typename EnergyFn>
bool check_energy_gradient_component(const std::string& label,
                                     const Vec& x,
                                     int node,
                                     int comp,
                                     double analytic,
                                     EnergyFn energy) {
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    std::vector<double> errors;
    errors.reserve(hs.size());

    for (double h : hs) {
        Vec xp = x;
        Vec xm = x;
        perturb(xp, node, comp, h);
        perturb(xm, node, comp, -h);
        const double fd = (energy(xp) - energy(xm)) / (2.0 * h);
        errors.push_back(std::abs(fd - analytic));
    }

    return check_slope_near_two(label, hs, errors, analytic);
}

template <typename GradFn>
bool check_gradient_hessian_component(const std::string& label,
                                      const Vec& x,
                                      int node,
                                      int perturb_comp,
                                      int grad_comp,
                                      double analytic,
                                      GradFn gradient) {
    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    std::vector<double> errors;
    errors.reserve(hs.size());

    for (double h : hs) {
        Vec xp = x;
        Vec xm = x;
        perturb(xp, node, perturb_comp, h);
        perturb(xm, node, perturb_comp, -h);
        const double fd = (component(gradient(xp), grad_comp) -
                           component(gradient(xm), grad_comp)) / (2.0 * h);
        errors.push_back(std::abs(fd - analytic));
    }

    return check_slope_near_two(label, hs, errors, analytic);
}

} // namespace

TEST(BarrierEnergy, EnergyGradientAndGradientHessianHaveSlopeTwo) {
    const Vec x = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.35, 0.22},
    };
    const int seg0 = 0;
    const int seg1 = 1;
    const int node = 2;
    const double dhat = 1.0;

    auto energy = [&](const Vec& x_eval) {
        return node_segment_barrier_energy(x_eval, node, seg0, seg1, dhat);
    };

    bool passed = true;
    for (int who : {node, seg0, seg1}) {
        const Vec2 g = local_barrier_grad(who, x, node, seg0, seg1, dhat);
        const Mat2 H = local_barrier_hess(who, x, node, seg0, seg1, dhat);
        auto gradient = [&](const Vec& x_eval) {
            return local_barrier_grad(who, x_eval, node, seg0, seg1, dhat);
        };

        for (int c = 0; c < 2; ++c) {
            passed &= check_energy_gradient_component(
                "barrier dE/dx node " + std::to_string(who) +
                    " comp " + std::to_string(c),
                x, who, c, component(g, c), energy);
        }

        for (int perturb_comp = 0; perturb_comp < 2; ++perturb_comp) {
            for (int grad_comp = 0; grad_comp < 2; ++grad_comp) {
                passed &= check_gradient_hessian_component(
                    "barrier dgrad/dx node " + std::to_string(who) +
                        " row " + std::to_string(grad_comp) +
                        " col " + std::to_string(perturb_comp),
                    x, who, perturb_comp, grad_comp,
                    component(H, grad_comp, perturb_comp), gradient);
            }
        }
    }

    EXPECT_TRUE(passed);
}
