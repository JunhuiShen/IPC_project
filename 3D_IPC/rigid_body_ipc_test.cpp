#include "rigid_body_ipc.h"

#include "algebra/algebra.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kDt = 0.03;

struct OmegaNodeEnergy {
    double energy = 0.0;
    Vec3 gradient = Vec3::Zero();
    Mat33 hessian = Mat33::Zero();
};

struct QuaternionNodeEnergy {
    double energy = 0.0;
    Vec4 gradient = Vec4::Zero();
    Mat44 hessian = Mat44::Zero();
};

OmegaNodeEnergy evaluate_omega_node_energy(const Vec3& X_centered, const Vec3& target, const Vec3& x_com, const Vec4& q0, const Vec3& omega, double dt) {
    const Vec3 x = world_space_position(X_centered, x_com, q0, omega, dt);
    const Mat33 J_xomega = dx_domega(X_centered, q0, omega, dt);
    const std::array<Mat33, 3> H_xomega = d2x_domega2(X_centered, q0, omega, dt);
    const Vec3 gx = x - target;
    const Vec3 omega_gradient = rigid_node_omega_gradient(gx, J_xomega);
    const Mat33 omega_hessian = rigid_node_omega_hessian(gx, Mat33::Identity(), J_xomega, H_xomega);

    OmegaNodeEnergy result;
    result.energy = 0.5 * gx.squaredNorm();
    result.gradient = omega_gradient;
    result.hessian = omega_hessian;
    return result;
}

QuaternionNodeEnergy evaluate_quaternion_node_energy(const Vec3& X_centered, const Vec3& target, const Vec3& x_com, const Vec4& quat) {
    const Vec3 gx = x_com + quaternion_rotate(quat, X_centered) - target;
    const Mat34 J_xq = dx_dq(X_centered, quat);
    const std::array<Mat44, 3> H_xq = d2x_dq2(X_centered);

    QuaternionNodeEnergy result;
    result.energy = 0.5 * gx.squaredNorm();
    result.gradient = J_xq.transpose() * gx;
    result.hessian = J_xq.transpose() * J_xq;
    for (int c = 0; c < 3; ++c)
        result.hessian += gx[c] * H_xq[c];
    result.hessian = 0.5 * (result.hessian + result.hessian.transpose());
    return result;
}

const std::vector<double> kConvergenceHs = {
    1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4
};

void expect_quadratic_convergence(
    const std::vector<double>& hs,
    const std::vector<double>& errors) {
    ASSERT_EQ(hs.size(), errors.size());

    constexpr double noise_floor = 1.0e-10;
    bool all_below_noise = true;
    bool saw_reliable_slope = false;

    for (std::size_t i = 1; i < errors.size(); ++i) {
        if (errors[i - 1] < noise_floor || errors[i] < noise_floor)
            continue;

        all_below_noise = false;
        if (errors[i] == 0.0)
            continue;

        const double slope = std::log(errors[i - 1] / errors[i])
            / std::log(hs[i - 1] / hs[i]);
        saw_reliable_slope = true;
        EXPECT_GT(slope, 1.99);
        EXPECT_LT(slope, 2.01);
    }

    EXPECT_TRUE(all_below_noise || saw_reliable_slope)
        << "no reliable finite-difference slope data";
}

TEST(RigidBodyIPCQuaternionWrappers, MultiplyAndConjugateUseScalarFirstHamiltonConvention) {
    const Vec4 a(0.5, -0.2, 0.4, 0.1);
    const Vec4 b(-0.3, 0.6, -0.1, 0.2);

    EXPECT_TRUE(quaternion_multiply(a, b).isApprox(
        Rigid_Body::ALGEBRA::QuaternionMultiply(a, b), 1.0e-14));
    EXPECT_TRUE(quaternion_conjugate(a).isApprox(Vec4(0.5, 0.2, -0.4, -0.1), 1.0e-14));
}

TEST(RigidBodyIPCQuaternionWrappers, InverseAndNormalizeHandleNonunitQuaternion) {
    const Vec4 quat(2.0, -1.0, 3.0, 4.0);
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);

    EXPECT_TRUE(quaternion_multiply(quat, quaternion_inverse(quat)).isApprox(identity, 1.0e-14));
    EXPECT_NEAR(quaternion_normalize(quat).norm(), 1.0, 1.0e-14);
    EXPECT_THROW(quaternion_inverse(Vec4::Zero()), std::invalid_argument);
    EXPECT_THROW(quaternion_normalize(Vec4::Zero()), std::invalid_argument);
}

TEST(RigidBodyIPCQuaternionWrappers, SignAlignmentUsesReferenceHemisphere) {
    const Vec4 reference = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec4 opposite = -reference;

    EXPECT_TRUE(quaternion_align_sign(opposite, reference).isApprox(reference, 1.0e-14));
    EXPECT_TRUE(quaternion_align_sign(reference, reference).isApprox(reference, 1.0e-14));
}

TEST(RigidBodyIPCQuaternionWrappers, TimeDerivativeUsesWorldSpaceAngularVelocity) {
    const Vec4 q = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.7, -0.4, 0.2);
    const Vec4 omega_quaternion(0.0, omega[0], omega[1], omega[2]);
    const Vec4 expected = 0.5 * quaternion_multiply(omega_quaternion, q);
    const Vec4 q_dot = quaternion_time_derivative(q, omega);

    EXPECT_TRUE(q_dot.isApprox(expected, 1.0e-14));
    EXPECT_NEAR(q.dot(q_dot), 0.0, 1.0e-14);
}

TEST(RigidBodyIPCQuaternionWrappers, ForwardAndInverseRotationRoundTrip) {
    const double half_angle = 0.25 * std::acos(-1.0);
    const Vec4 quat(std::cos(half_angle), 0.0, 0.0, std::sin(half_angle));
    const Vec3 vector(1.0, 0.0, 0.0);

    const Vec3 rotated = quaternion_rotate(quat, vector);
    EXPECT_TRUE(rotated.isApprox(Vec3(0.0, 1.0, 0.0), 1.0e-14));
    EXPECT_TRUE(quaternion_inverse_rotate(quat, rotated).isApprox(vector, 1.0e-14));
}

TEST(RigidBodyIPCOmegaNodeKinematics, MaterialAndWorldSpacePositionRoundTrip) {
    const Vec3 X_centered(0.7, -0.4, 1.2);
    const Vec3 x_com(-0.3, 0.8, 1.5);
    const Vec4 q0 = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.7, -0.4, 0.2);
    const Vec3 x = world_space_position(X_centered, x_com, q0, omega, kDt);
    const Vec3 recovered = material_space_position(x, x_com, q0, omega, kDt);

    EXPECT_TRUE(recovered.isApprox(X_centered, 1.0e-14));
}

TEST(RigidBodyIPCQuaternionOmega, ProductTensorMatchesHamiltonProduct) {
    const Vec4 a(0.5, -0.2, 0.4, 0.1);
    const Vec4 b(-0.3, 0.6, -0.1, 0.2);
    Vec4 product_from_tensor = Vec4::Zero();
    for (int alpha = 0; alpha < 4; ++alpha) {
        for (int beta = 0; beta < 4; ++beta) {
            for (int gamma = 0; gamma < 4; ++gamma) {
                product_from_tensor[alpha] +=
                    quaternion_product_tensor(alpha, beta, gamma) * a[beta] * b[gamma];
            }
        }
    }

    const Vec4 expected = Rigid_Body::ALGEBRA::QuaternionMultiply(a, b);
    EXPECT_TRUE(product_from_tensor.isApprox(expected, 1.0e-14));
}

TEST(RigidBodyIPCQuaternionExp, FirstOrderTaylorRemainderConvergesQuadratically) {
    const Vec3 omega(0.7, -0.4, 0.2);
    const Vec3 direction = Vec3(0.3, -0.4, 0.5).normalized();
    constexpr double dt = 1.3;
    const Vec4 value = exp(omega, dt);
    const Mat43 J = dexp_domega(omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        const Vec4 remainder = exp(omega + h * direction, dt) - value - h * J * direction;
        errors[hi] = remainder.norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionExp, JacobianConvergesQuadratically) {
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const Mat43 exact = dexp_domega(omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        Mat43 finite_difference = Mat43::Zero();
        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            finite_difference.col(beta) =
                (exp(omega + step, dt) - exp(omega - step, dt)) / (2.0 * h);
        }
        errors[hi] = (finite_difference - exact).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionExp, SecondDerivativeConvergesQuadratically) {
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const std::array<Mat33, 4> exact = d2exp_domega2(omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        double squared_error = 0.0;
        for (int gamma = 0; gamma < 3; ++gamma) {
            Vec3 step = Vec3::Zero();
            step[gamma] = h;
            const Mat43 finite_difference =
                (dexp_domega(omega + step, dt) - dexp_domega(omega - step, dt)) / (2.0 * h);
            for (int alpha = 0; alpha < 4; ++alpha) {
                const Vec3 error = finite_difference.row(alpha).transpose()   - exact[alpha].col(gamma);
                squared_error += error.squaredNorm();
            }
        }
        errors[hi] = std::sqrt(squared_error);
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionExp, SmallAngleTaylorBranchMatchesLimit) {
    constexpr double dt = 1.3;
    const double a = 0.5 * dt;
    const Vec3 omega(3.0e-5, -4.0e-5, 2.0e-5);
    ASSERT_GT(a * omega.norm(), 1.0e-6);
    ASSERT_LT(a * omega.norm(), 1.0e-4);

    const double angle = a * omega.norm();
    const double angle2 = angle * angle;
    const double angle4 = angle2 * angle2;
    const double sinc = 1.0 - angle2 / 6.0 + angle4 / 120.0;
    const Mat33 omega_outer = omega * omega.transpose();
    const double C1 = -a * a * sinc;
    const double C2 = a * sinc;
    const double C3 = a * a * a * (-1.0 / 3.0 + angle2 / 30.0 - angle4 / 840.0);

    Mat43 expected_dexp = Mat43::Zero();
    expected_dexp.row(0) = C1 * omega.transpose();
    expected_dexp.bottomRows<3>() = C2 * Mat33::Identity() + C3 * omega_outer;
    EXPECT_TRUE(dexp_domega(omega, dt).isApprox(expected_dexp, 1.0e-14));

    const std::array<Mat33, 4> d2exp = d2exp_domega2(omega, dt);
    const double scalar_outer_coefficient = std::pow(a, 4) * (1.0 / 3.0 - angle2 / 30.0 + angle4 / 840.0);
    const Mat33 expected_scalar_hessian = C1 * Mat33::Identity() + scalar_outer_coefficient * omega_outer;
    EXPECT_TRUE(d2exp[0].isApprox(expected_scalar_hessian, 1.0e-14));

    const double A = C3;
    const double B = std::pow(a, 5) * (1.0 / 15.0 - angle2 / 210.0 + angle4 / 7560.0);
    for (int i = 0; i < 3; ++i) {
        Mat33 expected_vector_hessian = Mat33::Zero();
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                expected_vector_hessian(j, k) = B * omega[i] * omega[j] * omega[k]
                    + A * ((i == j ? omega[k] : 0.0) + (i == k ? omega[j] : 0.0) + (j == k ? omega[i] : 0.0));
            }
        }
        EXPECT_TRUE(d2exp[1 + i].isApprox(expected_vector_hessian, 1.0e-14));
    }
}

TEST(RigidBodyIPCQuaternionOmega, MatchesLeftQuaternionUpdate) {
    const Vec4 q0 = Rigid_Body::ALGEBRA::QuaternionFromVector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);

    const Vec4 expected = Rigid_Body::ALGEBRA::QuaternionMultiply(
        Rigid_Body::ALGEBRA::QuaternionFromVector(kDt * omega), q0);
    const Vec4 actual = quaternion_from_angular_velocity(q0, omega, kDt);

    EXPECT_TRUE(actual.isApprox(expected, 1.0e-14));
}

TEST(RigidBodyIPCQuaternionOmega, ZeroAngularVelocityTaylorLimit) {
    const Vec4 q0(1.0, 0.0, 0.0, 0.0);
    const Vec3 zero_omega = Vec3::Zero();

    EXPECT_TRUE(exp(zero_omega, kDt).isApprox(q0, 1.0e-14));
    EXPECT_TRUE(quaternion_from_angular_velocity(q0, zero_omega, kDt).isApprox(q0, 1.0e-14));

    const Mat43 J = dq_domega(q0, zero_omega, kDt);
    Mat43 expected = Mat43::Zero();
    expected.bottomRows<3>() = 0.5 * kDt * Mat33::Identity();
    EXPECT_TRUE(J.isApprox(expected, 1.0e-14));

    const std::array<Mat33, 4> H_exp = d2exp_domega2(zero_omega, kDt);
    EXPECT_TRUE(H_exp[0].isApprox(-0.25 * kDt * kDt * Mat33::Identity(), 1.0e-14));
    for (int alpha = 1; alpha < 4; ++alpha)
        EXPECT_TRUE(H_exp[alpha].isZero(1.0e-14));
}

TEST(RigidBodyIPCQuaternionOmega, JacobianConvergesQuadratically) {
    const Vec4 q0 = Rigid_Body::ALGEBRA::QuaternionFromVector(Vec3(-0.3, 0.2, 0.1));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const Mat43 exact = dq_domega(q0, omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        Mat43 finite_difference = Mat43::Zero();
        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            finite_difference.col(beta) = (quaternion_from_angular_velocity(q0, omega + step, dt) - quaternion_from_angular_velocity(q0, omega - step, dt)) / (2.0 * h);
        }
        errors[hi] = (finite_difference - exact).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionOmega, SecondDerivativeConvergesQuadratically) {
    const Vec4 q0 = Rigid_Body::ALGEBRA::QuaternionFromVector(Vec3(-0.3, 0.2, 0.1));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const std::array<Mat33, 4> exact = d2q_domega2(q0, omega, dt);
    std::vector<double> errors(kConvergenceHs.size());
    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        double squared_error = 0.0;
        for (int gamma = 0; gamma < 3; ++gamma) {
            Vec3 step = Vec3::Zero();
            step[gamma] = h;
            const Mat43 plus = dq_domega(q0, omega + step, dt);
            const Mat43 minus = dq_domega(q0, omega - step, dt);
            const Mat43 finite_difference = (plus - minus) / (2.0 * h);
            for (int alpha = 0; alpha < 4; ++alpha) {
                const Vec3 error = finite_difference.row(alpha).transpose()  - exact[alpha].col(gamma);
                squared_error += error.squaredNorm();
            }
        }
        errors[hi] = std::sqrt(squared_error);
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionDerivatives, DxDqTaylorRemainderConvergesQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec4 q(0.8, -0.3, 0.4, 0.2);
    const Vec4 direction = Vec4(0.2, -0.3, 0.4, -0.5).normalized();
    const Mat34 J = dx_dq(X_centered, q);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        const Vec3 remainder = Rigid_Body::ALGEBRA::QuaternionRotate(q + h * direction, X_centered)
            - Rigid_Body::ALGEBRA::QuaternionRotate(q, X_centered) - h * J * direction;
        errors[hi] = remainder.norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCQuaternionDerivatives, D2xDq2MatchesFiniteDifferenceOfDxDq) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec4 q(0.8, -0.3, 0.4, 0.2);
    const std::array<Mat44, 3> exact = d2x_dq2(X_centered);

    for (double h : kConvergenceHs) {
        for (int gamma = 0; gamma < 4; ++gamma) {
            Vec4 step = Vec4::Zero();
            step[gamma] = h;
            const Mat34 finite_difference =
                (dx_dq(X_centered, q + step) - dx_dq(X_centered, q - step)) / (2.0 * h);

            for (int c = 0; c < 3; ++c) {
                const Vec4 error = finite_difference.row(c).transpose()
                    - exact[c].col(gamma);
                EXPECT_LT(error.norm(), 1.0e-10)
                    << "h=" << h << ", output coordinate=" << c
                    << ", quaternion coordinate=" << gamma;
            }
        }
    }
}

TEST(RigidBodyIPCQuaternionDerivatives, NodeEnergyDerivativesConvergeQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec3 target(-0.2, 0.6, 0.3);
    const Vec3 x_com(0.1, -0.2, 0.3);
    const Vec4 quat(0.8, -0.3, 0.4, 0.2);
    std::vector<double> gradient_errors(kConvergenceHs.size());
    std::vector<double> hessian_errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        const QuaternionNodeEnergy exact =
            evaluate_quaternion_node_energy(X_centered, target, x_com, quat);
        Vec4 gradient_fd = Vec4::Zero();
        Mat44 hessian_fd = Mat44::Zero();
        for (int beta = 0; beta < 4; ++beta) {
            Vec4 step = Vec4::Zero();
            step[beta] = h;
            const QuaternionNodeEnergy plus =
                evaluate_quaternion_node_energy(X_centered, target, x_com, quat + step);
            const QuaternionNodeEnergy minus =
                evaluate_quaternion_node_energy(X_centered, target, x_com, quat - step);
            gradient_fd[beta] = (plus.energy - minus.energy) / (2.0 * h);
            hessian_fd.col(beta) = (plus.gradient - minus.gradient) / (2.0 * h);
        }
        gradient_errors[hi] = (gradient_fd - exact.gradient).norm();
        hessian_errors[hi] = (hessian_fd - exact.hessian).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, gradient_errors);
    expect_quadratic_convergence(kConvergenceHs, hessian_errors);
}

TEST(RigidBodyIPCOmegaNodeKinematics, JacobianConvergesQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec3 x_com(0.1, -0.2, 0.3);
    const Vec4 q0 = Rigid_Body::ALGEBRA::QuaternionFromVector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const Mat33 exact = dx_domega(X_centered, q0, omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        Mat33 finite_difference = Mat33::Zero();
        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            finite_difference.col(beta) =
                (world_space_position(X_centered, x_com, q0, omega + step, dt)
                 - world_space_position(X_centered, x_com, q0, omega - step, dt)) / (2.0 * h);
        }
        errors[hi] = (finite_difference - exact).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCOmegaNodeKinematics, SecondDerivativeConvergesQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec4 q0 = Rigid_Body::ALGEBRA::QuaternionFromVector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    const std::array<Mat33, 3> exact = d2x_domega2(X_centered, q0, omega, dt);
    std::vector<double> errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        double squared_error = 0.0;
        for (int gamma = 0; gamma < 3; ++gamma) {
            Vec3 step = Vec3::Zero();
            step[gamma] = h;
            const Mat33 finite_difference =
                (dx_domega(X_centered, q0, omega + step, dt)
                 - dx_domega(X_centered, q0, omega - step, dt)) / (2.0 * h);
            for (int c = 0; c < 3; ++c) {
                const Vec3 error = finite_difference.row(c).transpose()
                    - exact[c].col(gamma);
                squared_error += error.squaredNorm();
            }
        }
        errors[hi] = std::sqrt(squared_error);
    }

    expect_quadratic_convergence(kConvergenceHs, errors);
}

TEST(RigidBodyIPCOmegaNodeDerivatives, CenteredDifferenceConvergesQuadratically) {
    const Vec3 X_centered(0.4, -0.7, 1.1);
    const Vec3 target(-0.2, 0.6, 0.3);
    const Vec3 x_com(0.1, -0.2, 0.3);
    const Vec4 q0 = Rigid_Body::ALGEBRA::QuaternionFromVector(Vec3(0.2, -0.1, 0.3));
    const Vec3 omega(0.7, -0.4, 0.2);
    constexpr double dt = 1.3;
    std::vector<double> gradient_errors(kConvergenceHs.size());
    std::vector<double> hessian_errors(kConvergenceHs.size());
    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        const OmegaNodeEnergy exact =
            evaluate_omega_node_energy(X_centered, target, x_com, q0, omega, dt);
        Vec3 gradient_fd = Vec3::Zero();
        Mat33 hessian_fd = Mat33::Zero();

        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            const OmegaNodeEnergy plus =
                evaluate_omega_node_energy(X_centered, target, x_com, q0, omega + step, dt);
            const OmegaNodeEnergy minus =
                evaluate_omega_node_energy(X_centered, target, x_com, q0, omega - step, dt);
            gradient_fd[beta] = (plus.energy - minus.energy) / (2.0 * h);
            hessian_fd.col(beta) = (plus.gradient - minus.gradient) / (2.0 * h);
        }

        gradient_errors[hi] = (gradient_fd - exact.gradient).norm();
        hessian_errors[hi] = (hessian_fd - exact.hessian).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, gradient_errors);
    expect_quadratic_convergence(kConvergenceHs, hessian_errors);
}

TEST(RigidBodyIPCInertialEnergy, TranslationDerivativesMatchEnergy) {
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const std::vector<Vec3> R_p = {
        Vec3(-0.8, 0.35, 0.2),
        Vec3(0.45, -0.55, 0.9),
        Vec3(0.3394736842105263, -0.0184210526315789, -0.4578947368421053),
    };
    const double total_mass = 3.8;
    const Vec3 x_com(0.31, -0.42, 0.18);
    const Vec3 x_com_n(-0.13, 0.27, -0.22);
    const Vec3 v_com_n(1.7, -0.6, 0.4);
    const Vec4 q_n = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.6, -0.3, 0.7);
    const Vec3 omega_n(-0.2, 0.5, 0.4);
    constexpr double dt = 0.31;

    ASSERT_TRUE((masses[0] * R_p[0] + masses[1] * R_p[1] + masses[2] * R_p[2]).isZero(1.0e-14));

    const Vec3 exact_gradient = inertia_translation_gradient(x_com, x_com_n, v_com_n, dt, total_mass);
    const Mat33 exact_hessian = inertia_translation_hessian(total_mass);
    Vec3 gradient_fd = Vec3::Zero();
    Mat33 hessian_fd = Mat33::Zero();
    constexpr double h = 1.0e-5;

    for (int alpha = 0; alpha < 3; ++alpha) {
        Vec3 step = Vec3::Zero();
        step[alpha] = h;
        const double plus_energy = incremental_potential_energy(x_com + step, omega, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, masses, R_p);
        const double minus_energy = incremental_potential_energy(x_com - step, omega, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, masses, R_p);
        gradient_fd[alpha] = (plus_energy - minus_energy) / (2.0 * h);

        const Vec3 plus_gradient = inertia_translation_gradient(x_com + step, x_com_n, v_com_n, dt, total_mass);
        const Vec3 minus_gradient = inertia_translation_gradient(x_com - step, x_com_n, v_com_n, dt, total_mass);
        hessian_fd.col(alpha) = (plus_gradient - minus_gradient) / (2.0 * h);
    }

    EXPECT_TRUE(gradient_fd.isApprox(exact_gradient, 1.0e-9));
    EXPECT_TRUE(hessian_fd.isApprox(exact_hessian, 1.0e-9));
}

TEST(RigidBodyIPCInertialEnergy, OmegaDerivativesConvergeQuadratically) {
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const std::vector<Vec3> R_p = {
        Vec3(-0.8, 0.35, 0.2),
        Vec3(0.45, -0.55, 0.9),
        Vec3(0.3394736842105263, -0.0184210526315789, -0.4578947368421053),
    };
    const double total_mass = 3.8;
    const Vec3 x_com_n(-0.13, 0.27, -0.22);
    const Vec3 v_com_n(1.7, -0.6, 0.4);
    constexpr double dt = 0.31;
    const Vec3 x_com = x_com_n + dt * v_com_n;
    const Vec4 q_n = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.6, -0.3, 0.7);
    const Vec3 omega_n(-0.2, 0.5, 0.4);
    const auto [exact_gradient, exact_hessian] = inertia_rotation_gradient_hessian(omega, q_n, omega_n, dt, masses, R_p);
    std::vector<double> gradient_errors(kConvergenceHs.size());
    std::vector<double> hessian_errors(kConvergenceHs.size());

    for (std::size_t hi = 0; hi < kConvergenceHs.size(); ++hi) {
        const double h = kConvergenceHs[hi];
        Vec3 gradient_fd = Vec3::Zero();
        Mat33 hessian_fd = Mat33::Zero();

        for (int beta = 0; beta < 3; ++beta) {
            Vec3 step = Vec3::Zero();
            step[beta] = h;
            const double plus_energy = incremental_potential_energy(x_com, omega + step, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, masses, R_p);
            const double minus_energy = incremental_potential_energy(x_com, omega - step, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, masses, R_p);
            gradient_fd[beta] = (plus_energy - minus_energy) / (2.0 * h);

            const Vec3 plus_gradient = inertia_rotation_gradient_hessian(omega + step, q_n, omega_n, dt, masses, R_p).first;
            const Vec3 minus_gradient = inertia_rotation_gradient_hessian(omega - step, q_n, omega_n, dt, masses, R_p).first;
            hessian_fd.col(beta) = (plus_gradient - minus_gradient) / (2.0 * h);
        }

        gradient_errors[hi] = (gradient_fd - exact_gradient).norm();
        hessian_errors[hi] = (hessian_fd - exact_hessian).norm();
    }

    expect_quadratic_convergence(kConvergenceHs, gradient_errors);
    expect_quadratic_convergence(kConvergenceHs, hessian_errors);
}

TEST(RigidBodyIPCInertialEnergy, ReducedEnergyMatchesFullNodalMassQuadratic) {
    const std::vector<double> masses = {1.2, 0.7, 1.9};
    const std::vector<Vec3> R_p = {
        Vec3(-0.8, 0.35, 0.2),
        Vec3(0.45, -0.55, 0.9),
        Vec3(0.3394736842105263, -0.0184210526315789, -0.4578947368421053),
    };
    const double total_mass = 3.8;
    const Vec3 x_com(0.31, -0.42, 0.18);
    const Vec3 x_com_n(-0.13, 0.27, -0.22);
    const Vec3 v_com_n(1.7, -0.6, 0.4);
    const Vec4 q_n = quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const Vec3 omega(0.6, -0.3, 0.7);
    const Vec3 omega_n(-0.2, 0.5, 0.4);
    constexpr double dt = 0.31;

    ASSERT_TRUE((masses[0] * R_p[0] + masses[1] * R_p[1] + masses[2] * R_p[2]).isZero(1.0e-14));

    const Vec4 q = quaternion_from_angular_velocity(q_n, omega, dt);
    double full_nodal_energy = 0.0;

    for (std::size_t p = 0; p < R_p.size(); ++p) {
        const Vec3 r_p = quaternion_rotate(q, R_p[p]);
        const Vec3 r_p_n = quaternion_rotate(q_n, R_p[p]);
        const Vec3 x_p = x_com + r_p;
        const Vec3 v_p_n = v_com_n + omega_n.cross(r_p_n);
        const Vec3 x_hat_p = x_com_n + r_p_n + dt * v_p_n;
        const Vec3 residual = x_p - x_hat_p;
        full_nodal_energy += 0.5 * masses[p] * residual.squaredNorm();
    }

    const double reduced_energy = incremental_potential_energy(x_com, omega, x_com_n, v_com_n, q_n, omega_n, dt, total_mass, masses, R_p);

    EXPECT_NEAR(reduced_energy, full_nodal_energy, 1.0e-14);
}

}  // namespace
