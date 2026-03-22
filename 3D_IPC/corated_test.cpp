#include "corotated_energy.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

// Tests
void FiniteDifferenceGradientTest(const TriangleRest& rest, const TriangleDef& def_in, double mu, double lambda, double eps = 1e-6) {
    auto g = corotated_node_gradient(rest, def_in, mu, lambda);

    std::cout << std::setprecision(16);
    std::cout << "===== Energy vs node-gradient FD test =====\n";
    std::cout << "Analytic gradient:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "g[" << i << "] = " << g[i].transpose() << "\n";
    }

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 3; ++c) {
            TriangleDef def_plus = def_in;
            TriangleDef def_minus = def_in;

            set_dof(def_plus, i, c, get_dof(def_plus, i, c) + eps);
            set_dof(def_minus, i, c, get_dof(def_minus, i, c) - eps);

            double Eplus = corotated_energy(rest, def_plus, mu, lambda);
            double Eminus = corotated_energy(rest, def_minus, mu, lambda);
            double fd = (Eplus - Eminus) / (2.0 * eps);
            double an = g[i](c);

            double abs_err = std::abs(fd - an);
            double denom = std::max(1.0, std::max(std::abs(fd), std::abs(an)));
            double rel_err = abs_err / denom;

            max_abs_err = std::max(max_abs_err, abs_err);
            max_rel_err = std::max(max_rel_err, rel_err);
        }
    }

    std::cout << "\nMax abs err = " << max_abs_err << "\n";
    std::cout << "Max rel err = " << max_rel_err << "\n\n";
}

void FiniteDifferenceHessianTest(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda, double eps = 1e-6) {
    std::cout << std::setprecision(16);
    std::cout << "===== Node-gradient vs node-Hessian FD test =====\n";

    Mat99 H = corotated_node_hessian(rest, def, mu, lambda);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    for (int j = 0; j < 3; ++j) {
        for (int c = 0; c < 3; ++c) {
            TriangleDef def_plus = def;
            TriangleDef def_minus = def;

            set_dof(def_plus, j, c, get_dof(def_plus, j, c) + eps);
            set_dof(def_minus, j, c, get_dof(def_minus, j, c) - eps);

            auto g_plus = corotated_node_gradient(rest, def_plus, mu, lambda);
            auto g_minus = corotated_node_gradient(rest, def_minus, mu, lambda);

            for (int i = 0; i < 3; ++i) {
                Vec3 fd = (g_plus[i] - g_minus[i]) / (2.0 * eps);
                Vec3 an = H.block<3,1>(3 * i, 3 * j + c);
                Vec3 err = fd - an;

                double abs_err = err.cwiseAbs().maxCoeff();
                double rel_err = abs_err / std::max(
                        1.0,
                        std::max(fd.cwiseAbs().maxCoeff(), an.cwiseAbs().maxCoeff()));

                max_abs_err = std::max(max_abs_err, abs_err);
                max_rel_err = std::max(max_rel_err, rel_err);
            }
        }
    }

    std::cout << "Global max abs err = " << max_abs_err << "\n";
    std::cout << "Global max rel err = " << max_rel_err << "\n\n";
}

void ConvergenceTestEnergyGradient(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda) {
    std::cout << "\n===== Energy to Gradient convergence test =====\n";

    TriangleDef dx = ZeroTriangleDef();
    dx.x[0] = Vec3(0.3, -0.7, 0.2);
    dx.x[1] = Vec3(-0.4, 0.1, 0.5);
    dx.x[2] = Vec3(0.25, 0.6, -0.35);

    double norm = std::sqrt(dx.x[0].squaredNorm() + dx.x[1].squaredNorm() + dx.x[2].squaredNorm());
    for (auto& v : dx.x) v /= norm;

    auto g = corotated_node_gradient(rest, def, mu, lambda);
    Vec9 g_flat = flatten_gradient(g);
    Vec9 dx_flat = flatten_def(dx);

    double g_dot_dx = g_flat.dot(dx_flat);

    std::vector<double> eps_list = {1e-1, 5e-2, 2.5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4};
    std::vector<double> errors;

    for (double eps : eps_list) {
        TriangleDef def_plus = add_scale(def, dx, eps);
        TriangleDef def_minus = add_scale(def, dx, -eps);

        double Eplus = corotated_energy(rest, def_plus, mu, lambda);
        double Eminus = corotated_energy(rest, def_minus, mu, lambda);

        double fd = (Eplus - Eminus) / (2.0 * eps);
        double err = std::abs(fd - g_dot_dx);
        errors.push_back(err);
    }

    std::cout << "\nEstimated convergence rates:\n";
    for (size_t i = 1; i < errors.size(); ++i) {
        double rate = std::log(errors[i] / errors[i - 1]) /
                      std::log(eps_list[i] / eps_list[i - 1]);
        std::cout << "rate(" << eps_list[i - 1] << " -> " << eps_list[i]
                  << ") = " << rate << "\n";
    }
}

void ConvergenceTestHessian(const TriangleRest& rest, const TriangleDef& def,
                            double mu,
                            double lambda) {
    std::cout << "\n===== Gradient to Hessian convergence test =====\n";

    TriangleDef dx = ZeroTriangleDef();
    dx.x[0] = Vec3(0.3, -0.7, 0.2);
    dx.x[1] = Vec3(-0.4, 0.1, 0.5);
    dx.x[2] = Vec3(0.25, 0.6, -0.35);

    double norm = std::sqrt(dx.x[0].squaredNorm() + dx.x[1].squaredNorm() + dx.x[2].squaredNorm());
    for (auto& v : dx.x) v /= norm;

    Mat99 H = corotated_node_hessian(rest, def, mu, lambda);
    Vec9 dx_flat = flatten_def(dx);
    Vec9 v_true = H * dx_flat;

    std::vector<double> eps_list = {1e-1, 5e-2, 2.5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4};
    std::vector<double> errors;

    for (double eps : eps_list) {
        TriangleDef def_plus = add_scale(def, dx, eps);
        TriangleDef def_minus = add_scale(def, dx, -eps);

        auto g_plus = corotated_node_gradient(rest, def_plus, mu, lambda);
        auto g_minus = corotated_node_gradient(rest, def_minus, mu, lambda);

        std::array<Vec3, 3> dg_fd;
        for (int i = 0; i < 3; ++i) {
            dg_fd[i] = (g_plus[i] - g_minus[i]) / (2.0 * eps);
        }

        Vec9 v_fd = flatten_gradient(dg_fd);
        Vec9 err = v_fd - v_true;

        double e = err.norm();
        errors.push_back(e);
    }

    std::cout << "\nEstimated convergence rates:\n";
    for (size_t i = 1; i < errors.size(); ++i) {
        double rate = std::log(errors[i] / errors[i - 1]) /
                      std::log(eps_list[i] / eps_list[i - 1]);
        std::cout << "rate(" << eps_list[i - 1] << " -> " << eps_list[i]
                  << ") = " << rate << "\n";
    }
}

void BasicHessianChecks(const TriangleRest& rest,
                        const TriangleDef& def,
                        double mu,
                        double lambda) {
    Mat99 H = corotated_node_hessian(rest, def, mu, lambda);
    std::cout << "\n===== Basic Hessian checks =====\n";
    std::cout << "||H - H^T||_inf = "
              << (H - H.transpose()).cwiseAbs().maxCoeff() << "\n";

    Vec3 row0 = H.block<3,3>(0,0).rowwise().sum() + H.block<3,3>(0,3).rowwise().sum() + H.block<3,3>(0,6).rowwise().sum();
    Vec3 row1 = H.block<3,3>(3,0).rowwise().sum() + H.block<3,3>(3,3).rowwise().sum() + H.block<3,3>(3,6).rowwise().sum();
    Vec3 row2 = H.block<3,3>(6,0).rowwise().sum() + H.block<3,3>(6,3).rowwise().sum() + H.block<3,3>(6,6).rowwise().sum();

    std::cout << "block-row sum 0 = " << row0.transpose() << "\n";
    std::cout << "block-row sum 1 = " << row1.transpose() << "\n";
    std::cout << "block-row sum 2 = " << row2.transpose() << "\n";
}

int main() {
    TriangleRest rest;
    rest.X[0] = Vec2(0.0, 0.0);
    rest.X[1] = Vec2(1.2, 0.1);
    rest.X[2] = Vec2(0.2, 0.9);

    TriangleDef def;
    def.x[0] = Vec3(0.1, -0.2, 0.3);
    def.x[1] = Vec3(1.4, 0.2, -0.1);
    def.x[2] = Vec3(0.0, 1.0, 0.4);

    double mu = 2.0;
    double lambda = 5.0;

    std::cout << std::setprecision(16);
    std::cout << "Triangle energy = " << corotated_energy(rest, def, mu, lambda) << "\n\n";

    FiniteDifferenceGradientTest(rest, def, mu, lambda, 1e-6);
    FiniteDifferenceHessianTest(rest, def, mu, lambda, 1e-6);
    ConvergenceTestEnergyGradient(rest, def, mu, lambda);
    ConvergenceTestHessian(rest, def, mu, lambda);
    BasicHessianChecks(rest, def, mu, lambda);

    return 0;
}
