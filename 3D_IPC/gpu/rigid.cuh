#pragma once

// Cartesian rigid derivative pullbacks.
#include "rigid_types.h"

#ifdef __CUDACC__
#define IPC_RIGID_HD __host__ __device__
#else
#define IPC_RIGID_HD
#endif

namespace ipc_gpu {
namespace rigid_detail {

IPC_RIGID_HD inline void multiply(const double* a, const double* b, double* c) {
    for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
            c[3*r+s] = (a[3*r]*b[s] + a[3*r+1]*b[3+s]) + a[3*r+2]*b[6+s];
}

IPC_RIGID_HD inline void transpose_multiply(const double* a, const double* b, double* c) {
    for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
            c[3*r+s] = (a[r]*b[s] + a[3+r]*b[3+s]) + a[6+r]*b[6+s];
}

IPC_RIGID_HD inline void add(double* a, const double* b) {
    for (int k = 0; k < 9; ++k) a[k] += b[k];
}

IPC_RIGID_HD inline void symmetrize(double* a) {
    // Preserve the CPU reference's column-major, in-place symmetrization.
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            a[3*r+c] = 0.5 * (a[3*r+c] + a[3*c+r]);
}

IPC_RIGID_HD inline double inertia_trace(const double* a, const double* b, const double* moment) {
    double product[9];
    transpose_multiply(a, b, product);
    double diagonal[3];
    for (int i = 0; i < 3; ++i)
        diagonal[i] = (product[3*i]*moment[i] + product[3*i+1]*moment[3+i])
            + product[3*i+2]*moment[6+i];
    return (diagonal[0] + diagonal[1]) + diagonal[2];
}

IPC_RIGID_HD inline double skew_entry(int axis, int r, int c) {
    if (r == c || r == axis || c == axis) return 0.0;
    return (r == (axis+1)%3 && c == (axis+2)%3) ? -1.0 : 1.0;
}

} // namespace rigid_detail

IPC_RIGID_HD inline RigidResult evaluate_rigid(const RigidPreparedContact& p) {
    RigidResult result{};
    auto& b = result.barrier;
    const bool tg = p.flags & RigidTranslationGradient;
    const bool og = p.flags & RigidOrientationGradient;
    const bool th = p.flags & RigidTranslationHessian;
    const bool oh = p.flags & RigidOrientationHessian;
    const bool mh = p.flags & RigidMixedHessian;
    const bool upper = p.flags & RigidUpperCrossBlocks;
    // PT COM-only assembly uses its precomputed point self block. The old
    // cross-block sum was overwritten below, so do not read or sum it.
    const bool point_translation = !p.segment_segment && th && !og && !oh && !mh;
    for (int k = 0; k < 9; ++k) b.translation_translation_hessian[k] = p.translation_hessian[k];
    for (int i = 0; i < 4; ++i) {
        const double* g = p.gradients[i];
        const double* J = p.jacobians[i];
        for (int r = 0; r < 3; ++r) {
            if (tg) b.translation_gradient[r] += g[r];
            if (og) b.orientation_gradient[r] += (J[r]*g[0] + J[3+r]*g[1]) + J[6+r]*g[2];
        }
        if ((th || oh || mh) && !point_translation) {
            for (int j = upper ? i : 0; j < 4; ++j) {
                const double* H = p.hessians[i][j];
                if (th) rigid_detail::add(b.translation_translation_hessian, H);
                double temporary[9], contribution[9];
                if (mh) {
                    rigid_detail::multiply(H, p.jacobians[j], temporary);
                    rigid_detail::add(b.translation_orientation_hessian, temporary);
                }
                if (oh) {
                    rigid_detail::transpose_multiply(J, H, temporary);
                    rigid_detail::multiply(temporary, p.jacobians[j], contribution);
                    if (upper && i != j) {
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c)
                                b.orientation_orientation_hessian[3*r+c] += contribution[3*r+c] + contribution[3*c+r];
                    } else rigid_detail::add(b.orientation_orientation_hessian, contribution);
                }
            }
        }
        if (oh)
            for (int axis = 0; axis < 3; ++axis)
                for (int k = 0; k < 9; ++k)
                    b.orientation_orientation_hessian[k] += g[axis] * p.curvature[i][axis][k];
    }
    if (th && !og && !p.segment_segment)
        for (int k = 0; k < 9; ++k) b.translation_translation_hessian[k] = p.translation_hessian[k];
    if (th) rigid_detail::symmetrize(b.translation_translation_hessian);
    if (oh) rigid_detail::symmetrize(b.orientation_orientation_hessian);

    if (!(p.flags & RigidFriction)) return result;
    auto& f = result.friction;
    double weight = 0.0, FJ[9]{};
    for (int i = 0; i < 4; ++i) {
        weight += p.friction_weights[i];
        if (og)
            for (int k = 0; k < 9; ++k) FJ[k] += p.friction_weights[i] * p.jacobians[i][k];
    }
    double HJ[9];
    if (mh || oh) rigid_detail::multiply(p.friction_hessian, FJ, HJ);
    const bool translation = p.update_mask & 1u;
    const bool orientation = p.update_mask & 2u;
    if (oh && orientation) rigid_detail::transpose_multiply(FJ, HJ, f.orientation_orientation_hessian);
    for (int r = 0; r < 3; ++r) {
        if (tg && translation) f.translation_gradient[r] = weight * p.friction_gradient[r];
        if (og && orientation)
            f.orientation_gradient[r] = (FJ[r]*p.friction_gradient[0] + FJ[3+r]*p.friction_gradient[1])
                + FJ[6+r]*p.friction_gradient[2];
        for (int c = 0; c < 3; ++c) {
            if (th && translation) f.translation_translation_hessian[3*r+c] = (weight * p.friction_hessian[3*r+c]) * weight;
            if (mh && translation && orientation) f.translation_orientation_hessian[3*r+c] = weight * HJ[3*r+c];
        }
    }
    // Frozen friction intentionally has no quaternion-curvature contribution.
    return result;
}

IPC_RIGID_HD inline Derivatives evaluate_rigid_inertia(const RigidInertia& p) {
    Derivatives result{};
    if (!p.rotational) {
        for (int r = 0; r < 3; ++r) {
            result.gradient[r] = p.mass * (p.position[r] - p.previous_position[r] - p.dt*p.previous_velocity[r]);
            result.hessian[3*r+r] = p.mass;
        }
        result.gradient[1] -= p.dt * p.dt * p.mass * p.gravity;
        return result;
    }

    const double* q = p.orientation;
    const double vector_norm2 = (q[1]*q[1] + q[2]*q[2]) + q[3]*q[3];
    double D[9], G[4][9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double skew = 0.0;
            for (int l = 0; l < 3; ++l) skew += q[l+1] * rigid_detail::skew_entry(l, r, c);
            const double identity = r == c ? 1.0 : 0.0;
            D[3*r+c] = ((q[0]*q[0] - vector_norm2)*identity + 2.0*q[r+1]*q[c+1])
                + 2.0*q[0]*skew - p.rotation_predictor[3*r+c];
            G[0][3*r+c] = 2.0*q[0]*identity + 2.0*skew;
            for (int l = 0; l < 3; ++l)
                G[l+1][3*r+c] = ((-2.0*q[l+1]*identity + 2.0*(r == l)*q[c+1])
                    + 2.0*q[r+1]*(c == l)) + 2.0*q[0]*rigid_detail::skew_entry(l,r,c);
        }
    }
    double gradient_q[4], hessian_q[4][4];
    for (int alpha = 0; alpha < 4; ++alpha) {
        gradient_q[alpha] = rigid_detail::inertia_trace(D, G[alpha], p.second_moment);
        for (int beta = 0; beta < 4; ++beta) {
            double K[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    if (alpha == 0 && beta == 0) K[3*r+c] = 2.0*(r == c);
                    else if (alpha == 0 || beta == 0)
                        K[3*r+c] = 2.0*rigid_detail::skew_entry((alpha == 0 ? beta : alpha)-1,r,c);
                    else K[3*r+c] = -2.0*(alpha == beta)*(r == c)
                        + 2.0*(r == alpha-1)*(c == beta-1) + 2.0*(r == beta-1)*(c == alpha-1);
                }
            }
            hessian_q[alpha][beta] = rigid_detail::inertia_trace(G[alpha], G[beta], p.second_moment)
                + rigid_detail::inertia_trace(D, K, p.second_moment);
        }
    }
    for (int beta = 0; beta < 3; ++beta) {
        for (int alpha = 0; alpha < 4; ++alpha)
            result.gradient[beta] += gradient_q[alpha] * p.orientation_jacobian[alpha][beta];
        for (int kappa = 0; kappa < 3; ++kappa) {
            for (int alpha = 0; alpha < 4; ++alpha) {
                for (int delta = 0; delta < 4; ++delta)
                    result.hessian[3*beta+kappa] += p.orientation_jacobian[alpha][beta]
                        * hessian_q[alpha][delta] * p.orientation_jacobian[delta][kappa];
                result.hessian[3*beta+kappa] += gradient_q[alpha] * p.orientation_hessians[alpha][3*beta+kappa];
            }
        }
    }
    return result;
}

} // namespace ipc_gpu

#undef IPC_RIGID_HD

// Quaternion kinematics and material-point derivatives.
#include <cmath>

#ifdef __CUDACC__
#define IPC_GPU_RIGID_HD __host__ __device__
#else
#define IPC_GPU_RIGID_HD
#endif

namespace ipc_gpu::solver {

// Scalar-first quaternions and world-space angular velocity, matching
// rigid_body_ipc.cpp. Matrices are row-major. The caller supplies a finite,
// unit previous orientation and owns device status/error handling. These
// routines do not normalize, truncate rotations, or fall back to host work.
struct RigidKinematics {
    double orientation[4];
    double orientation_jacobian[4][3];
    double orientation_hessians[4][9];
};

IPC_GPU_RIGID_HD inline int rigid_epsilon(int i, int j, int k) {
    if (i == j || j == k || i == k) return 0;
    return ((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) ||
            (i == 2 && j == 0 && k == 1)) ? 1 : -1;
}

IPC_GPU_RIGID_HD inline int rigid_quaternion_product_tensor(int a, int b, int c) {
    if (a == 0 && b == 0 && c == 0) return 1;
    if (a == 0 && b == c && b != 0) return -1;
    if (a != 0 && b == 0 && a == c) return 1;
    if (a != 0 && c == 0 && a == b) return 1;
    if (a && b && c) return rigid_epsilon(a - 1, b - 1, c - 1);
    return 0;
}

IPC_GPU_RIGID_HD inline void rigid_quaternion_multiply(const double* a, const double* b, double* out) {
    // Temporaries make aliased input/output safe while retaining CPU order.
    const double w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    const double x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    const double y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    const double z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

IPC_GPU_RIGID_HD inline void rigid_quaternion_exp(const double* omega, double dt, double* out) {
    const double r = sqrt((omega[0]*omega[0] + omega[1]*omega[1]) + omega[2]*omega[2]);
    const double a = 0.5 * dt, angle = a * r;
    double coefficient;
    if (fabs(angle) < 1.0e-4) {
        const double angle2 = angle * angle, angle4 = angle2 * angle2;
        coefficient = a * (1.0 - angle2 / 6.0 + angle4 / 120.0);
    } else coefficient = sin(angle) / r;
    out[0] = cos(angle);
    for (int k = 0; k < 3; ++k) out[k + 1] = coefficient * omega[k];
}

IPC_GPU_RIGID_HD inline void rigid_orientation(const double* previous_q,
                                                   const double* omega, double dt, double* q) {
    double exponential[4];
    rigid_quaternion_exp(omega, dt, exponential);
    rigid_quaternion_multiply(exponential, previous_q, q);
}

// exp((dt/2) omega), including the same small-angle Taylor branch as CPU.
IPC_GPU_RIGID_HD inline void rigid_exp_derivatives(const double* omega, double dt,
                                                       double J[4][3], double H[4][9]) {
    const double r = sqrt((omega[0]*omega[0] + omega[1]*omega[1]) + omega[2]*omega[2]);
    const double a = 0.5 * dt, angle = a * r;
    double C1, C2, C3, H1, H2, A, B;
    if (fabs(angle) < 1.0e-4) {
        const double angle2 = angle * angle, angle4 = angle2 * angle2;
        const double sinc = 1.0 - angle2 / 6.0 + angle4 / 120.0;
        C1 = -a * a * sinc;
        C2 = a * sinc;
        C3 = a * a * a * (-1.0 / 3.0 + angle2 / 30.0 - angle4 / 840.0);
        H1 = -a * a * sinc;
        H2 = pow(a, 4) * (1.0 / 3.0 - angle2 / 30.0 + angle4 / 840.0);
        A = a * a * a * (-1.0 / 3.0 + angle2 / 30.0 - angle4 / 840.0);
        B = pow(a, 5) * (1.0 / 15.0 - angle2 / 210.0 + angle4 / 7560.0);
    } else {
        const double s = sin(angle), c = cos(angle);
        C1 = -a * s / r;
        C2 = s / r;
        C3 = (a * r * c - s) / (r * r * r);
        H1 = -a * s / r;
        H2 = -a * a * c / (r * r) + a * s / (r * r * r);
        const double numerator = a * r * c - s;
        A = numerator / (r * r * r);
        B = -a * a * s / (r * r * r) - 3.0 * numerator / (r * r * r * r * r);
    }
    for (int beta = 0; beta < 3; ++beta) {
        J[0][beta] = C1 * omega[beta];
        for (int i = 0; i < 3; ++i)
            J[i + 1][beta] = C2 * (i == beta) + C3 * omega[i] * omega[beta];
        for (int gamma = 0; gamma < 3; ++gamma) {
            H[0][3*beta+gamma] = H1 * (beta == gamma) + H2 * omega[beta] * omega[gamma];
            for (int i = 0; i < 3; ++i) {
                // CPU forms omega_outer first, then B * omega[i] * outer.
                H[i + 1][3*beta+gamma] = (B * omega[i]) * (omega[beta] * omega[gamma]);
                H[i + 1][3*beta+gamma] += A * ((i == beta) * omega[gamma] +
                    (i == gamma) * omega[beta] + (beta == gamma) * omega[i]);
            }
        }
    }
}

IPC_GPU_RIGID_HD inline RigidKinematics rigid_kinematics(const double* previous_q,
                                                            const double* omega, double dt) {
    RigidKinematics result{};
    rigid_orientation(previous_q, omega, dt, result.orientation);
    double Jexp[4][3], Hexp[4][9];
    rigid_exp_derivatives(omega, dt, Jexp, Hexp);
    for (int alpha = 0; alpha < 4; ++alpha) {
        for (int beta = 0; beta < 3; ++beta)
            for (int gamma = 0; gamma < 4; ++gamma)
                for (int delta = 0; delta < 4; ++delta)
                    result.orientation_jacobian[alpha][beta] +=
                        rigid_quaternion_product_tensor(alpha,gamma,delta) * Jexp[gamma][beta] * previous_q[delta];
        for (int gamma = 0; gamma < 4; ++gamma)
            for (int delta = 0; delta < 4; ++delta)
                for (int k = 0; k < 9; ++k)
                    result.orientation_hessians[alpha][k] +=
                        (rigid_quaternion_product_tensor(alpha,gamma,delta) * Hexp[gamma][k]) * previous_q[delta];
    }
    return result;
}

IPC_GPU_RIGID_HD inline void rigid_rotation_predictor(const double* previous_q,
                                                          const double* previous_omega, double dt,
                                                          double* predictor) {
    double negative_omega[3], q_nm1[4], q_dot[4];
    for (int k = 0; k < 3; ++k) negative_omega[k] = -previous_omega[k];
    rigid_orientation(previous_q, negative_omega, dt, q_nm1);
    const double angular_quaternion[4] = {0.0, previous_omega[0], previous_omega[1], previous_omega[2]};
    rigid_quaternion_multiply(angular_quaternion, q_nm1, q_dot);
    for (int k = 0; k < 4; ++k) q_dot[k] *= 0.5;
    const double* q = previous_q;
    const double vector_norm2 = (q[1]*q[1] + q[2]*q[2]) + q[3]*q[3];
    const double q_dot_product = ((q[0]*q_dot[0] + q[1]*q_dot[1]) + q[2]*q_dot[2]) + q[3]*q_dot[3];
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
        double skew = 0.0;
        for (int k = 0; k < 3; ++k) skew += q[k+1] * rigid_detail::skew_entry(k,r,c);
        const double identity = r == c ? 1.0 : 0.0;
        const double R = ((q[0]*q[0] - vector_norm2) * identity + 2.0*q[r+1]*q[c+1]) + 2.0*q[0]*skew;
        double Rdot = q_dot[0] * (2.0*q[0]*identity + 2.0*skew);
        for (int l = 0; l < 3; ++l) {
            const double G = ((-2.0*q[l+1]*identity + 2.0*(r == l)*q[c+1]) +
                              2.0*q[r+1]*(c == l)) + 2.0*q[0]*rigid_detail::skew_entry(l,r,c);
            Rdot += q_dot[l+1] * G;
        }
        Rdot -= (2.0 * q_dot_product) * R;
        predictor[3*r+c] = R + dt * Rdot;
    }
}

// Complete rotational-inertia preparation is device-callable: unlike the
// original hybrid bridge, no CPU quaternion derivatives/predictor are needed.
IPC_GPU_RIGID_HD inline RigidInertia build_rigid_inertia(const double* previous_q,
    const double* previous_omega, const double* omega, double dt, const double* second_moment) {
    RigidInertia result{};
    result.dt = dt; result.rotational = 1;
    const RigidKinematics kinematics = rigid_kinematics(previous_q, omega, dt);
    for (int a = 0; a < 4; ++a) {
        result.orientation[a] = kinematics.orientation[a];
        for (int b = 0; b < 3; ++b) result.orientation_jacobian[a][b] = kinematics.orientation_jacobian[a][b];
        for (int k = 0; k < 9; ++k) result.orientation_hessians[a][k] = kinematics.orientation_hessians[a][k];
    }
    for (int k = 0; k < 9; ++k) result.second_moment[k] = second_moment[k];
    rigid_rotation_predictor(previous_q, previous_omega, dt, result.rotation_predictor);
    return result;
}

IPC_GPU_RIGID_HD inline void rigid_reference_derivatives(const double* X,
    const RigidKinematics& kinematics, double* jacobian, double curvature[3][9]) {
    const double* q = kinematics.orientation;
    double Jxq[3][4]{}, Hxq[3][4][4]{};
    for (int c = 0; c < 3; ++c) {
        Jxq[c][0] = 2.0 * q[0] * X[c];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
            Jxq[c][0] += 2.0 * rigid_epsilon(c,i,j) * q[i+1] * X[j];
        for (int a = 0; a < 3; ++a) {
            Jxq[c][a+1] = 2.0 * (-q[a+1]*X[c] + q[c+1]*X[a]);
            for (int j = 0; j < 3; ++j)
                Jxq[c][a+1] += 2.0 * ((c == a)*q[j+1]*X[j] + q[0]*rigid_epsilon(c,a,j)*X[j]);
        }
        Hxq[c][0][0] = 2.0 * X[c];
        for (int a = 0; a < 3; ++a) {
            for (int j = 0; j < 3; ++j) {
                const double value = 2.0 * rigid_epsilon(c,a,j) * X[j];
                Hxq[c][0][a+1] += value;
                Hxq[c][a+1][0] += value;
            }
            for (int b = 0; b < 3; ++b)
                Hxq[c][a+1][b+1] = 2.0 * (-(a == b)*X[c] + (a == c)*X[b] + (b == c)*X[a]);
        }
        for (int beta = 0; beta < 3; ++beta) {
            jacobian[3*c+beta] = 0.0;
            for (int alpha = 0; alpha < 4; ++alpha)
                jacobian[3*c+beta] += Jxq[c][alpha] * kinematics.orientation_jacobian[alpha][beta];
        }
        double temporary[3][4]{};
        for (int i = 0; i < 3; ++i) for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                temporary[i][a] += kinematics.orientation_jacobian[b][i] * Hxq[c][b][a];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
            curvature[c][3*i+j] = 0.0;
            for (int a = 0; a < 4; ++a)
                curvature[c][3*i+j] += temporary[i][a] * kinematics.orientation_jacobian[a][j];
            for (int a = 0; a < 4; ++a)
                curvature[c][3*i+j] += Jxq[c][a] * kinematics.orientation_hessians[a][3*i+j];
        }
        // Retain the column-major in-place symmetrization of CPU Eigen code.
        for (int column = 0; column < 3; ++column) for (int row = 0; row < 3; ++row)
            curvature[c][3*row+column] = 0.5 * (curvature[c][3*row+column] + curvature[c][3*column+row]);
    }
}

IPC_GPU_RIGID_HD inline void rigid_world_position(const double* X, const double* com,
                                                      const double* orientation, double* position) {
    const double reference[4] = {0.0, X[0], X[1], X[2]};
    const double conjugate[4] = {orientation[0], -orientation[1], -orientation[2], -orientation[3]};
    double right[4], rotated[4];
    rigid_quaternion_multiply(reference, conjugate, right);
    rigid_quaternion_multiply(orientation, right, rotated);
    for (int k = 0; k < 3; ++k) position[k] = com[k] + rotated[k+1];
}

} // namespace ipc_gpu::solver

#undef IPC_GPU_RIGID_HD

// Finite material-point pin derivatives.
#include "rigid_types.h"
#include <cmath>

#ifdef __CUDACC__
#define IPC_RIGID_PIN_HD __host__ __device__
#else
#define IPC_RIGID_PIN_HD
#endif

namespace ipc_gpu::solver {
namespace rigid_pin_detail {

IPC_RIGID_PIN_HD inline bool finite(double x) {
#ifdef __CUDA_ARCH__
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}
IPC_RIGID_PIN_HD inline bool finite3(const double* x) {
    return finite(x[0]) && finite(x[1]) && finite(x[2]);
}
IPC_RIGID_PIN_HD inline bool finite_derivatives(const RigidDerivatives& d) {
    for (int k=0;k<3;++k)
        if (!finite(d.translation_gradient[k]) || !finite(d.orientation_gradient[k])) return false;
    for (int k=0;k<9;++k)
        if (!finite(d.translation_translation_hessian[k]) ||
            !finite(d.translation_orientation_hessian[k]) ||
            !finite(d.orientation_orientation_hessian[k])) return false;
    return true;
}
IPC_RIGID_PIN_HD inline bool valid_pin(const RigidPinConstraint& p) {
    return finite3(p.local_position) && finite3(p.initial_target) &&
        finite3(p.rotation_center) && finite(p.angular_velocity_z);
}

} // namespace rigid_pin_detail

IPC_RIGID_PIN_HD inline bool rigid_pin_target(const RigidPinConstraint& pin,
    double time, double* target) {
    if (!rigid_pin_detail::valid_pin(pin) || !rigid_pin_detail::finite(time)) return false;
    if(pin.angular_velocity_z==0) {
        for(int k=0;k<3;++k)target[k]=pin.initial_target[k];
        return true;
    }
    if(!rigid_pin_detail::finite3(pin.rotation_axis))return false;
    double scale=0;
    for(int k=0;k<3;++k)scale=fmax(scale,fabs(pin.rotation_axis[k]));
    if(!(scale>0))return false;
    double axis[3],norm2=0;
    for(int k=0;k<3;++k){axis[k]=pin.rotation_axis[k]/scale;norm2+=axis[k]*axis[k];}
    const double norm=sqrt(norm2);
    if(!(norm>0)||!rigid_pin_detail::finite(norm))return false;
    for(int k=0;k<3;++k)axis[k]/=norm;
    const double theta = pin.angular_velocity_z*time;
    if (!rigid_pin_detail::finite(theta)) return false;
    const double c = cos(theta), s = sin(theta);
    double offset[3];
    for (int k=0;k<3;++k) offset[k] = pin.initial_target[k]-pin.rotation_center[k];
    if(!rigid_pin_detail::finite3(offset))return false;
    if(pin.rotation_axis[0]==0&&pin.rotation_axis[1]==0&&pin.rotation_axis[2]>0) {
    target[0] = pin.rotation_center[0]+(c*offset[0]-s*offset[1]);
    target[1] = pin.rotation_center[1]+(s*offset[0]+c*offset[1]);
    target[2] = pin.rotation_center[2]+offset[2];
    } else {
        const double dot=(axis[0]*offset[0]+axis[1]*offset[1])+axis[2]*offset[2];
        const double cross[3]{axis[1]*offset[2]-axis[2]*offset[1],
            axis[2]*offset[0]-axis[0]*offset[2],axis[0]*offset[1]-axis[1]*offset[0]};
        for(int k=0;k<3;++k)
            target[k]=pin.rotation_center[k]+(c*offset[k]+s*cross[k]+((1-c)*dot)*axis[k]);
    }
    return rigid_pin_detail::finite3(target);
}

// Exact pullback of dt^2*kpin/2*||com+R(q(omega))*X-target(time)||^2.
// with_orientation_hessian includes the signed quaternion-curvature term;
// it is deliberately not a positive-semidefinite Gauss-Newton replacement.
IPC_RIGID_PIN_HD inline bool evaluate_rigid_pin(const RigidPinConstraint& pin,
    const double* com, const RigidKinematics& kin, double dt, double kpin,
    double time, bool with_orientation_hessian, RigidDerivatives& out) {
    out = {};
    if (!rigid_pin_detail::finite(kpin)) return false;
    if (kpin<=0) return true; // Same inactive-spring rule as CPU.
    if (!rigid_pin_detail::finite(dt) || dt<0 || !rigid_pin_detail::finite3(com)) return false;
    const double scale = dt*dt*kpin;
    if (!rigid_pin_detail::finite(scale)) return false;
    for (int a=0;a<4;++a) {
        if (!rigid_pin_detail::finite(kin.orientation[a])) return false;
        for (int k=0;k<3;++k)
            if (!rigid_pin_detail::finite(kin.orientation_jacobian[a][k])) return false;
        if (with_orientation_hessian) for (int k=0;k<9;++k)
            if (!rigid_pin_detail::finite(kin.orientation_hessians[a][k])) return false;
    }
    double target[3], position[3];
    if (!rigid_pin_target(pin,time,target)) return false;
    rigid_world_position(pin.local_position,com,kin.orientation,position);
    double error[3], J[9], curvature[3][9];
    for (int k=0;k<3;++k) error[k] = position[k]-target[k];
    if (!rigid_pin_detail::finite3(error)) return false;
    rigid_reference_derivatives(pin.local_position,kin,J,curvature);
    for (int r=0;r<3;++r) {
        out.translation_gradient[r] = scale*error[r];
        out.orientation_gradient[r] = ((scale*J[r])*error[0]+(scale*J[3+r])*error[1])
            +(scale*J[6+r])*error[2];
        out.translation_translation_hessian[3*r+r] = scale;
    }
    for (int k=0;k<9;++k) out.translation_orientation_hessian[k] = scale*J[k];
    if (with_orientation_hessian) {
        double H[9];
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
            double value = (J[r]*J[c]+J[3+r]*J[3+c])+J[6+r]*J[6+c];
            for (int axis=0;axis<3;++axis) value += error[axis]*curvature[axis][3*r+c];
            H[3*r+c] = value;
        }
        for (int r=0;r<3;++r) for (int c=0;c<3;++c)
            out.orientation_orientation_hessian[3*r+c] = scale*(0.5*(H[3*r+c]+H[3*c+r]));
    }
    return rigid_pin_detail::finite_derivatives(out);
}

// One body owns its aggregate; there are no floating-point atomics or races.
// Invalid metadata/derivatives return an error without publishing a partial sum.
IPC_RIGID_PIN_HD inline int evaluate_rigid_body_pins(const RigidPinScene& pins,
    std::size_t body, const double* com, const RigidKinematics& kin, double dt,
    double kpin, double time, bool with_orientation_hessian, RigidDerivatives& out) {
    out = {};
    if (!rigid_pin_detail::finite(kpin)) return RigidPinInvalidInput;
    if (kpin<=0 || pins.count==0) return 0;
    if (body>=pins.bodies || !pins.pins || !pins.offsets ||
        pins.offsets[0]!=0 || pins.offsets[pins.bodies]!=pins.count) return RigidPinInvalidInput;
    const auto begin=pins.offsets[body], end=pins.offsets[body+1];
    if (begin>end || end>pins.count) return RigidPinInvalidInput;
    RigidDerivatives sum{};
    for (std::size_t i=begin;i<end;++i) {
        if (pins.pins[i].body!=static_cast<int>(body) || !rigid_pin_detail::valid_pin(pins.pins[i]))
            return RigidPinInvalidInput;
        RigidDerivatives value{};
        if (!evaluate_rigid_pin(pins.pins[i],com,kin,dt,kpin,time,with_orientation_hessian,value))
            return RigidPinInvalidDerivative;
        for (int k=0;k<3;++k) {
            sum.translation_gradient[k] += value.translation_gradient[k];
            sum.orientation_gradient[k] += value.orientation_gradient[k];
        }
        for (int k=0;k<9;++k) {
            sum.translation_translation_hessian[k] += value.translation_translation_hessian[k];
            sum.translation_orientation_hessian[k] += value.translation_orientation_hessian[k];
            sum.orientation_orientation_hessian[k] += value.orientation_orientation_hessian[k];
        }
    }
    if (!rigid_pin_detail::finite_derivatives(sum)) return RigidPinInvalidDerivative;
    out = sum;
    return 0;
}

} // namespace ipc_gpu::solver

#undef IPC_RIGID_PIN_HD

// Pivoted local LDLT solve.
#include <cmath>
#include <cfloat>

// Same lower-triangle pivoted LDLT and diagonal pseudoinverse policy as
// Eigen 3.4 LDLT::solve used by our CPU rigid GS solver. Extracted without
// numerical changes from the previous rigid_pipeline.cu device function.
// Algorithm adapted from Eigen/src/Cholesky/LDLT.h (MPL-2.0).
// Copyright (C) 2008-2010 Gael Guennebaud and Benoit Jacob.
// This Source Code Form is subject to the Mozilla Public License, v. 2.0.
// If a copy was not distributed, obtain it at https://mozilla.org/MPL/2.0/.
#if defined(__CUDACC__)
#define IPC_RIGID_SOLVE_HD __host__ __device__
#else
#define IPC_RIGID_SOLVE_HD
#endif

namespace ipc_gpu::solver {
using std::isfinite;
IPC_RIGID_SOLVE_HD inline bool rigid_ldlt3(const double* H,const double* g,double* x)
{
    double a[3][3],rhs[3]; int swaps[3]={0,1,2}; bool zero_seen=false;
    for (int i=0;i<3;++i) {
        if (!isfinite(g[i])) return false;
        rhs[i]=g[i];
        for (int j=0;j<3;++j) {
            if (!isfinite(H[3*i+j])) return false;
            a[i][j]=H[3*i+j];
        }
    }
    for (int k=0;k<3;++k) {
        int pivot=k;
        for (int i=k+1;i<3;++i) if (fabs(a[i][i])>fabs(a[pivot][pivot])) pivot=i;
        swaps[k]=pivot;
        if (pivot!=k) {
            for (int j=0;j<k;++j) { const double t=a[k][j]; a[k][j]=a[pivot][j]; a[pivot][j]=t; }
            for (int i=pivot+1;i<3;++i) { const double t=a[i][k]; a[i][k]=a[i][pivot]; a[i][pivot]=t; }
            const double t=a[k][k]; a[k][k]=a[pivot][pivot]; a[pivot][pivot]=t;
            for (int i=k+1;i<pivot;++i) { const double u=a[i][k]; a[i][k]=a[pivot][i]; a[pivot][i]=u; }
        }
        if (k>0) {
            double temp[3]{},diagonal=0;
            for (int j=0;j<k;++j) { temp[j]=a[j][j]*a[k][j]; diagonal+=a[k][j]*temp[j]; }
            a[k][k]-=diagonal;
            for (int i=k+1;i<3;++i) {
                double product=0;
                for (int j=0;j<k;++j) product+=a[i][j]*temp[j];
                a[i][k]-=product;
            }
        }
        const double diagonal=a[k][k];
        if (!isfinite(diagonal)) return false;
        const bool nonzero=fabs(diagonal)>0;
        if (k==0 && !nonzero) {
            for (int i=0;i<3;++i) for (int j=0;j<i;++j) if (a[i][j]!=0) return false;
            for (int i=0;i<3;++i) x[i]=0;
            return true;
        }
        if (zero_seen && nonzero) return false;
        if (!nonzero) zero_seen=true;
        for (int i=k+1;i<3;++i) {
            if (nonzero) a[i][k]/=diagonal;
            else if (a[i][k]!=0) return false;
            if (!isfinite(a[i][k])) return false;
        }
    }
    for (int k=0;k<3;++k) { const double t=rhs[k]; rhs[k]=rhs[swaps[k]]; rhs[swaps[k]]=t; }
    for (int i=0;i<3;++i) for (int j=0;j<i;++j) rhs[i]-=a[i][j]*rhs[j];
    for (int i=0;i<3;++i) rhs[i]=fabs(a[i][i])>DBL_MIN ? rhs[i]/a[i][i] : 0.0;
    for (int i=2;i>=0;--i) for (int j=i+1;j<3;++j) rhs[i]-=a[j][i]*rhs[j];
    for (int k=2;k>=0;--k) { const double t=rhs[k]; rhs[k]=rhs[swaps[k]]; rhs[swaps[k]]=t; }
    for (int i=0;i<3;++i) { x[i]=rhs[i]; if (!isfinite(x[i])) return false; }
    return true;
}
} // namespace ipc_gpu::solver

#undef IPC_RIGID_SOLVE_HD

// Analytic rotational continuous collision detection.
#include "broadphase_types.h"
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define IPC_RCCD_HD __host__ __device__
#else
#define IPC_RCCD_HD
#endif

namespace ipc_gpu::solver {

// Port of the current analytic rotational CCD in ccd.cpp. These are curved
// trajectories, not a linear chord approximation. CPU code uses trigonometric
// roots and a quadratic swept-edge surface (there is no cubic root stage).
// Quaternion signs retain arcs beyond pi; endpoints alone cannot encode an
// arbitrary number of complete turns. The caller must not discard winding.
enum class RigidRotationStatus : unsigned { Success, InvalidInput, AmbiguousFullTurn, OutsideCap, UnsupportedWinding };
struct RigidRotationQuery {
    double positions[4][3];
    double com[3], current[4], target[4];
    unsigned vertex_face;
    // Nonzero winding cannot be represented by this endpoint-only CPU API.
    int extra_turns = 0;
};
struct RigidRotationResult {
    RigidRotationStatus status = RigidRotationStatus::Success;
    bool collision = false;
    double time = 0.0;
};

namespace rotation_detail {
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double two_pi = 2.0 * pi;
IPC_RCCD_HD inline bool finite(double x) { return x-x == 0.0; }
IPC_RCCD_HD inline double minimum(double a, double b) { return a < b ? a : b; }
IPC_RCCD_HD inline double maximum(double a, double b) { return a > b ? a : b; }
IPC_RCCD_HD inline double clamp(double x, double lo, double hi) { return maximum(lo, minimum(hi,x)); }
struct V2 {
    double x, y;
    IPC_RCCD_HD V2 operator+(V2 b) const { return {x+b.x,y+b.y}; }
    IPC_RCCD_HD V2 operator-(V2 b) const { return {x-b.x,y-b.y}; }
    IPC_RCCD_HD V2 operator*(double a) const { return {x*a,y*a}; }
    IPC_RCCD_HD V2 operator/(double a) const { return {x/a,y/a}; }
    IPC_RCCD_HD double dot(V2 b) const { return x*b.x+y*b.y; }
    IPC_RCCD_HD double norm() const { return sqrt(dot(*this)); }
};
struct V3 {
    double x,y,z;
    IPC_RCCD_HD V3 operator+(V3 b) const { return {x+b.x,y+b.y,z+b.z}; }
    IPC_RCCD_HD V3 operator-(V3 b) const { return {x-b.x,y-b.y,z-b.z}; }
    IPC_RCCD_HD V3 operator*(double a) const { return {x*a,y*a,z*a}; }
    IPC_RCCD_HD V3 operator/(double a) const { return {x/a,y/a,z/a}; }
    IPC_RCCD_HD double dot(V3 b) const { return (x*b.x+y*b.y)+z*b.z; }
    IPC_RCCD_HD double squaredNorm() const { return dot(*this); }
    IPC_RCCD_HD double norm() const { return sqrt(squaredNorm()); }
    IPC_RCCD_HD V3 cross(V3 b) const { return {y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x}; }
};
IPC_RCCD_HD inline V3 vec(const double* p) { return {p[0],p[1],p[2]}; }
IPC_RCCD_HD inline void basis(V3 n, V3& e1, V3& e2) {
    const V3 ref = fabs(n.x) < .9 ? V3{1,0,0} : V3{0,1,0};
    e1 = n.cross(ref); e1 = e1/e1.norm(); e2 = n.cross(e1);
}
IPC_RCCD_HD inline V2 project(V3 p, V3 com, V3 e1, V3 e2) {
    const V3 d = p-com; return {d.dot(e1),d.dot(e2)};
}
IPC_RCCD_HD inline bool normalize(const double* q, double* out) {
    const double norm = sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]);
    if (!(norm > 0.0) || !finite(norm)) return false;
    for (int k = 0; k < 4; ++k) out[k] = q[k]/norm;
    return true;
}
IPC_RCCD_HD inline void relative(const double* target, const double* current, double* q) {
    const double conjugate[4] = {current[0],-current[1],-current[2],-current[3]};
    rigid_quaternion_multiply(target,conjugate,q);
}

IPC_RCCD_HD inline bool point_segment_2d(V2 x, V2 com, double theta_n,
    double theta_new, V2 x0, V2 x1, double& step) {
    step = 0.0;
    const V2 dx = x-com;
    const double cn = cos(theta_n), sn = sin(theta_n);
    const V2 r{cn*dx.x+sn*dx.y,-sn*dx.x+cn*dx.y};
    const V2 d = x1-x0;
    const double length = d.norm();
    if (length < 1e-14) return false;
    const V2 direction = d/length;
    const double A = r.x*direction.y-r.y*direction.x;
    const double B = -r.x*direction.x-r.y*direction.y;
    const double C = (com.x-x0.x)*direction.y-(com.y-x0.y)*direction.x;
    const double amplitude = sqrt(A*A+B*B);
    if (fabs(C) > amplitude+1e-14) return false;
    if (amplitude < 1e-14) {
        if (fabs(C) > 1e-14) return false;
        const double t = (x-x0).dot(d)/(length*length);
        return t >= -1e-12 && t <= 1.0+1e-12;
    }
    const double phi = atan2(B,A), ac = acos(clamp(-C/amplitude,-1,1));
    const double candidates[2] = {phi+ac,phi-ac};
    const double dtheta = theta_new-theta_n;
    if (fabs(dtheta) < 1e-14) return false;
    double best = INFINITY;
    for (int i = 0; i < 2; ++i) {
        const double k = dtheta > 0 ? ceil((theta_n-candidates[i]-1e-12)/two_pi)
                                  : floor((theta_n-candidates[i]+1e-12)/two_pi);
        const double theta = candidates[i]+two_pi*k;
        double s = (theta-theta_n)/dtheta;
        if (s < -1e-12 || s > 1.0+1e-12) continue;
        s = clamp(s,0,1);
        const V2 position = V2{cos(theta)*r.x-sin(theta)*r.y,sin(theta)*r.x+cos(theta)*r.y}+com;
        const double t = (position-x0).dot(d)/(length*length);
        if (t < -1e-12 || t > 1.0+1e-12) continue;
        if (s < best) best = s;
    }
    if (best == INFINITY) return false;
    step = best; return true;
}

IPC_RCCD_HD inline bool inside(V3 p, V3 origin, V3 e1, V3 e2,
    double a11, double a12, double a22, double det) {
    const V3 r = p-origin;
    const double b1 = r.dot(e1), b2 = r.dot(e2);
    const double alpha = (b1*a22-b2*a12)/det;
    const double beta = (-b1*a12+b2*a11)/det;
    return alpha >= -1e-10 && beta >= -1e-10 && 1.0-alpha-beta >= -1e-10;
}

// Raw counterparts intentionally preserve the CPU epsilon/rejection behavior.
// Use the checked public query below so full-turn ambiguity cannot be mistaken
// for a certified collision-free step.
IPC_RCCD_HD inline bool point_triangle(V3 x, V3 com, const double* target,
    const double* current, V3 a, V3 b, V3 c, double& s) {
    constexpr double eps = 1e-10;
    s = 0.0;
    double qr[4]; relative(target,current,qr);
    const V3 v = vec(qr+1); const double norm = v.norm();
    // The CPU rigid_rotation_epsilon is zero: a small nonzero rotation can
    // still reach contact and must not be treated as an unchanged pose.
    if (norm == 0.0) return false;
    const V3 n = v/norm, dx = x-com;
    const V3 parallel = n*dx.dot(n), perpendicular = dx-parallel;
    if (perpendicular.norm() < eps) return false;
    const double dtheta = 2.0*atan2(norm,qr[0]);
    if (dtheta == 0.0) return false;
    const V3 crossed = n.cross(perpendicular), e1 = b-a, e2 = c-a;
    const double a11 = e1.dot(e1), a12 = e1.dot(e2), a22 = e2.dot(e2);
    const V3 normal = e1.cross(e2); const double normal_length = normal.norm();
    if (normal_length*normal_length <= eps*a11*a22) return false;
    const V3 unit_normal = normal/normal_length;
    const double A = perpendicular.dot(unit_normal), B = crossed.dot(unit_normal), C = (com+parallel-a).dot(unit_normal);
    const double det = a11*a22-a12*a12;
    if (fabs(det) <= eps*a11*a22) return false;
    const double amplitude = sqrt(A*A+B*B);
    if (amplitude < eps) {
        if (fabs(C) > eps) return false;
        if (inside(x,a,e1,e2,a11,a12,a22,det)) return true;
        V3 be1,be2; basis(n,be1,be2);
        const V2 p = project(x,com,be1,be2);
        const V2 vertices[3] = {project(a,com,be1,be2),project(b,com,be1,be2),project(c,com,be1,be2)};
        const double theta = atan2(p.y,p.x);
        double best = INFINITY;
        for (int i = 0; i < 3; ++i) {
            double candidate;
            if (point_segment_2d(p,{0,0},theta,theta+dtheta,vertices[i],vertices[(i+1)%3],candidate)
                && candidate < best) best = candidate;
        }
        if (best == INFINITY) return false;
        s = best; return true;
    }
    if (fabs(C) > amplitude+eps) return false;
    const double phi = atan2(B,A), ac = acos(clamp(-C/amplitude,-1,1));
    const double candidates[2] = {phi+ac,phi-ac};
    double best = INFINITY;
    for (int i = 0; i < 2; ++i) {
        const double k = dtheta > 0 ? ceil((-candidates[i]-eps)/two_pi) : floor((-candidates[i]+eps)/two_pi);
        double candidate = (candidates[i]+two_pi*k)/dtheta;
        if (candidate < -eps || candidate > 1.0+eps) continue;
        candidate = clamp(candidate,0,1);
        const double theta = candidate*dtheta;
        const V3 position = ((com+parallel)+perpendicular*cos(theta))+crossed*sin(theta);
        if (inside(position,a,e1,e2,a11,a12,a22,det) && candidate < best) best = candidate;
    }
    if (best == INFINITY) return false;
    s = best; return true;
}

// Closed-form counterpart of ccd_rotation_family_detail. CUDA long double has
// no extra precision, so all bounds explicitly use binary64 roundoff. Only the
// legacy near-zero radial-polynomial branch enters this finite-primitive repair.
struct FamilyRoots { double value[2]{}; int count=0; };
IPC_RCCD_HD inline FamilyRoots family_roots(double a,double b,double c) {
    FamilyRoots result;
    if(a==0) {
        if(b!=0)result.value[result.count++]=-c/b;
        return result;
    }
    const double bb=b*b,ac4=4*a*c;
    double disc=fma(-4*a,c,bb);
    const double error=32*DBL_EPSILON*(fabs(bb)+fabs(ac4));
    if(disc < -error)return result;
    disc=maximum(0,disc);
    const double q=-.5*(b+copysign(sqrt(disc),b));
    if(q==0)result.value[result.count++]=-b/(2*a);
    else {result.value[result.count++]=q/a;result.value[result.count++]=c/q;}
    return result;
}
struct FamilyRange {
    V3 p,v,q,e,n;
    double roundoff,arc,direction,best=INFINITY;
    IPC_RCCD_HD void consider(double w,bool zero_event=false) {
        const double parameter_error=64*DBL_EPSILON;
        if(!finite(w)||w < -parameter_error||w > 1+parameter_error)return;
        w=clamp(w,0,1);
        const V3 a=p+v*w,b=q+e*w;
        const double ar2=a.squaredNorm(),br2=b.squaredNorm();
        const double a_error=roundoff*(p.norm()+fabs(w)*v.norm());
        const double b_error=roundoff*(q.norm()+fabs(w)*e.norm());
        const double radial_error=roundoff*(ar2+br2)
            +2*(sqrt(ar2)*a_error+sqrt(br2)*b_error)
            +a_error*a_error+b_error*b_error;
        if(fabs(br2-ar2)>radial_error)return;
        if(ar2==0&&br2==0){best=0;return;}
        if(ar2==0||br2==0)return;
        const double x=a.dot(b),y=direction*n.dot(a.cross(b));
        double angle;
        if(zero_event&&x>0&&fabs(y)<=roundoff*sqrt(ar2*br2))angle=0;
        else {angle=atan2(y,x);if(angle<0)angle+=two_pi;}
        const double fraction=angle/arc;
        if(fraction<=1+1e-10)best=minimum(best,clamp(fraction,0,1));
    }
    IPC_RCCD_HD void roots(double a,double b,double c,bool zero_event=false) {
        const auto candidates=family_roots(a,b,c);
        for(int i=0;i<candidates.count;++i)consider(candidates.value[i],zero_event);
    }
};
IPC_RCCD_HD inline V3 family_moving_at(V3 p,V3 delta,double h0,double h2,
    double en,double dh,double u,int endpoint) {
    if(endpoint==0)return p;
    if(endpoint==1)return p+delta;
    return p+delta*clamp((h2+u*en-h0)/dh,0,1);
}
IPC_RCCD_HD inline bool family_range(V3 moving0,V3 moving_delta,V3 stationary0,
    V3 stationary_delta,V3 axis,double h0,double h1,double h2,double en,
    double coordinate_scale,double dtheta,double lo,double hi,double& s,int endpoint=-1) {
    const double dh=h1-h0;
    const V3 p=family_moving_at(moving0,moving_delta,h0,h2,en,dh,lo,endpoint);
    const V3 v=family_moving_at(moving0,moving_delta,h0,h2,en,dh,hi,endpoint)-p;
    const V3 q=stationary0+stationary_delta*lo,e=stationary_delta*(hi-lo);
    const double scale=maximum(maximum(p.norm(),(p+v).norm()),maximum(q.norm(),(q+e).norm()));
    if(scale==0){s=0;return true;}
    FamilyRange range{p/scale,v/scale,q/scale,e/scale,axis/axis.norm(),
        DBL_EPSILON*(128+64*coordinate_scale/scale),fabs(dtheta),dtheta<0 ? -1.0 : 1.0};
    const double fa=range.e.squaredNorm()-range.v.squaredNorm();
    const double fb=2*(range.q.dot(range.e)-range.p.dot(range.v));
    const double fc=range.q.squaredNorm()-range.p.squaredNorm();
    const double ea=range.roundoff*(range.e.squaredNorm()+range.v.squaredNorm());
    const double eb=2*range.roundoff*(range.q.norm()*range.e.norm()+range.p.norm()*range.v.norm());
    const double ec=range.roundoff*(range.q.squaredNorm()+range.p.squaredNorm());
    const bool family=fabs(fa)<=ea&&fabs(fb)<=eb&&fabs(fc)<=ec;
    range.consider(0);range.consider(1);
    if(!family) {
        // A small nonzero polynomial has only actual roots. In particular,
        // constant unequal radii do not become a swept-surface family.
        range.roots(fa,fb,fc);
    } else {
        const double kp=range.n.dot(range.p.cross(range.v));
        const double kq=range.n.dot(range.q.cross(range.e));
        range.roots(kq*range.v.squaredNorm()-kp*range.e.squaredNorm(),
            2*(kq*range.p.dot(range.v)-kp*range.q.dot(range.e)),
            kq*range.p.squaredNorm()-kp*range.q.squaredNorm());
        // Interior angle-zero and radial-axis events split wrapped branches.
        range.roots(range.n.dot(range.v.cross(range.e)),
            range.n.dot(range.p.cross(range.e)+range.v.cross(range.q)),
            range.n.dot(range.p.cross(range.q)),true);
        range.roots(range.v.squaredNorm(),2*range.p.dot(range.v),range.p.squaredNorm(),true);
        range.roots(range.e.squaredNorm(),2*range.q.dot(range.e),range.q.squaredNorm(),true);
    }
    if(!finite(range.best))return false;
    s=range.best;return true;
}
IPC_RCCD_HD inline bool swept_family(V3 moving0,V3 moving_delta,V3 stationary0,
    V3 stationary_delta,V3 axis,double h0,double h1,double h2,double en,
    double coordinate_scale,double dtheta,double& s) {
    double lo=0,hi=1,best=INFINITY,candidate=0;
    const double lower=minimum(h0,h1),upper=maximum(h0,h1);
    bool nominal=true,found=false;
    if(en==0)nominal=h2>=lower&&h2<=upper;
    else {
        double enter=(lower-h2)/en,exit=(upper-h2)/en;
        if(enter>exit){const double saved=enter;enter=exit;exit=saved;}
        lo=maximum(lo,enter);hi=minimum(hi,exit);nominal=lo<=hi;
    }
    if(nominal&&family_range(moving0,moving_delta,stationary0,stationary_delta,
            axis,h0,h1,h2,en,coordinate_scale,dtheta,lo,hi,candidate)) {
        found=true;best=candidate;
    }
    // Roundoff seams use the actual finite moving endpoint. They never extend
    // the moving edge or substitute an extrapolated infinite-line contact.
    const double height_error=64*DBL_EPSILON*(coordinate_scale
        +maximum(maximum(fabs(h0),fabs(h1)),maximum(fabs(h2),fabs(h2+en))));
    for(int endpoint=0;endpoint<2;++endpoint) {
        const double height=endpoint==0 ? h0 : h1;
        double begin=0,end=1;
        if(en==0) {if(fabs(h2-height)>height_error)continue;}
        else {
            begin=(height-height_error-h2)/en;end=(height+height_error-h2)/en;
            if(begin>end){const double saved=begin;begin=end;end=saved;}
            begin=maximum(begin,0);end=minimum(end,1);if(begin>end)continue;
        }
        if(family_range(moving0,moving_delta,stationary0,stationary_delta,
                axis,h0,h1,h2,en,coordinate_scale,dtheta,begin,end,candidate,endpoint)) {
            found=true;best=minimum(best,candidate);
        }
    }
    if(found)s=best;
    return found;
}

IPC_RCCD_HD inline bool segment_segment(V3 x0, V3 x1, V3 com,
    const double* target, const double* current, V3 x2, V3 x3, double& s) {
    constexpr double eps = 1e-10;
    s = 0.0;
    double qr[4]; relative(target,current,qr);
    const V3 v = vec(qr+1); const double norm = v.norm();
    if (norm == 0.0) return false;
    const V3 n = v/norm;
    const double h0 = (x0-com).dot(n), h1 = (x1-com).dot(n);
    const V3 p0 = (x0-com)-n*h0, p1 = (x1-com)-n*h1;
    const double r0 = p0.norm(), r1 = p1.norm();
    if (maximum(r0,r1) < eps) return false;
    const double dtheta = 2.0*atan2(norm,qr[0]);
    if (dtheta == 0.0) return false;
    const V3 ref = r0 >= r1 ? p0 : p1, dp = p1-p0;
    const V3 q2 = x2-com, q3 = x3-com, e = q3-q2;
    const double h2 = q2.dot(n), en = e.dot(n);
    const V3 q2p = q2-n*h2, ep = e-n*en;
    V3 e1,e2; basis(n,e1,e2);
    const double theta_n = atan2(ref.dot(e2),ref.dot(e1));
    bool found = false; double best = INFINITY;
    if (fabs(h1-h0) > eps) {
        const double dh = h1-h0, beta = en/dh;
        const V3 p0prime = p0+dp*((h2-h0)/dh);
        const double A = ep.squaredNorm()-beta*beta*dp.squaredNorm();
        const double B = 2.0*(q2p.dot(ep)-beta*p0prime.dot(dp));
        const double C = q2p.squaredNorm()-p0prime.squaredNorm();
        if(fabs(A)<eps&&fabs(B)<eps&&fabs(C)<eps) {
            const V3 points[5]={x0,x1,x2,x3,com};double coordinate_scale=0;
            for(int i=0;i<5;++i)coordinate_scale=maximum(coordinate_scale,
                maximum(maximum(fabs(points[i].x),fabs(points[i].y)),fabs(points[i].z)));
            return swept_family(p0,dp,q2p,ep,n,h0,h1,h2,en,coordinate_scale,dtheta,s);
        }
        const double disc = B*B-4.0*A*C;
        if (disc >= 0.0) for (int sign = -1; sign <= 1; sign += 2) {
            double u;
            if (fabs(A) < eps) {
                if (fabs(B) < eps) {
                    if (fabs(C) >= eps || sign == 1) continue;
                    u = 0.0;
                } else u = -C/B;
            } else u = (-B+sign*sqrt(maximum(0.0,disc)))/(2.0*A);
            if (u < -eps || u > 1.0+eps) continue;
            u = clamp(u,0,1);
            double t = (h2+u*en-h0)/dh;
            if (t < -eps || t > 1.0+eps) continue;
            t = clamp(t,0,1);
            const V3 moving = p0+dp*t;
            if (moving.norm() < eps) continue;
            const V3 star_p = q2p+ep*u;
            // Recover the relative angle directly, as on the CPU. Subtracting
            // two world-scale angles loses near-start impacts.
            double angle = atan2(n.dot(moving.cross(star_p)),moving.dot(star_p));
            while (dtheta > 0 && angle < -eps) angle += two_pi;
            while (dtheta < 0 && angle > eps) angle -= two_pi;
            double candidate = angle/dtheta;
            if (candidate < -eps || candidate > 1.0+eps) continue;
            candidate = clamp(candidate,0,1);
            if (candidate < best) { best = candidate; found = true; }
        }
    } else {
        const double outer = maximum(r0,r1), dp2 = dp.squaredNorm();
        double inner;
        if (dp2 < eps) inner = r0;
        else {
            const double t = -p0.dot(dp)/dp2;
            inner = t > 0.0 && t < 1.0 ? (p0+dp*t).norm() : minimum(r0,r1);
        }
        if (fabs(en) > eps) {
            double u = (h0-h2)/en;
            if (u >= -eps && u <= 1.0+eps) {
                u = clamp(u,0,1);
                const V3 star = x2*(1.0-u)+x3*u;
                const V3 star_p = (star-com)-n*(star-com).dot(n);
                const double radius = star_p.norm();
                if (radius >= inner-eps && radius <= outer+eps) {
                    double candidate;
                    if (point_segment_2d(project(star,com,e1,e2),{0,0},0,-dtheta,
                        project(x0,com,e1,e2),project(x1,com,e1,e2),candidate)) { best = candidate; found = true; }
                }
            }
        } else if (fabs(h0-h2) < eps) {
            const V2 a = project(x0,com,e1,e2), b = project(x1,com,e1,e2);
            const V2 c = project(x2,com,e1,e2), d = project(x3,com,e1,e2);
            const V2 points[4] = {a,b,c,d};
            for (int i = 0; i < 4; ++i) {
                double candidate;
                const bool forward = i < 2;
                if (point_segment_2d(points[i],{0,0},forward ? theta_n : 0,
                    forward ? theta_n+dtheta : -dtheta,forward ? c : a,forward ? d : b,candidate)
                    && candidate < best) { best = candidate; found = true; }
            }
        }
    }
    if (!found) return false;
    s = best; return true;
}

} // namespace rotation_detail

IPC_RCCD_HD inline RigidRotationResult rigid_rotation_ccd(const RigidRotationQuery& q) {
    using namespace rotation_detail;
    RigidRotationResult result;
    if (q.extra_turns != 0) { result.status = RigidRotationStatus::UnsupportedWinding; return result; }
    for (int i = 0; i < 4; ++i) {
        if (!finite(q.current[i]) || !finite(q.target[i])) { result.status = RigidRotationStatus::InvalidInput; return result; }
        // This branch-heavy CPU algorithm has no scaled-arithmetic fallback.
        // Reject extreme magnitudes before finite inputs can overflow an
        // intermediate discriminant and be misclassified as a non-collision.
        for (int k = 0; k < 3; ++k) if (!finite(q.positions[i][k]) || !finite(q.com[k]) ||
            fabs(q.positions[i][k]) > 1e30 || fabs(q.com[k]) > 1e30) {
            result.status = RigidRotationStatus::InvalidInput; return result;
        }
    }
    double current_norm = 0, target_norm = 0;
    for (int k = 0; k < 4; ++k) { current_norm += q.current[k]*q.current[k]; target_norm += q.target[k]*q.target[k]; }
    if (!finite(current_norm) || !finite(target_norm) || fabs(current_norm-1)>1e-8 || fabs(target_norm-1)>1e-8) {
        result.status = RigidRotationStatus::InvalidInput; return result;
    }
    double relative_q[4]; relative(q.target,q.current,relative_q);
    // Raw CPU kernels treat this as negligible even on a nearly full turn.
    // Fail closed here instead of allowing that ambiguity to authorize motion.
    if (relative_q[0] < 0 && vec(relative_q+1).norm() < 1e-10) {
        result.status = RigidRotationStatus::AmbiguousFullTurn; return result;
    }
    result.collision = q.vertex_face
        ? point_triangle(vec(q.positions[0]),vec(q.com),q.target,q.current,vec(q.positions[1]),vec(q.positions[2]),vec(q.positions[3]),result.time)
        : segment_segment(vec(q.positions[0]),vec(q.positions[1]),vec(q.com),q.target,q.current,vec(q.positions[2]),vec(q.positions[3]),result.time);
    if (!finite(result.time)) result.status = RigidRotationStatus::InvalidInput;
    return result;
}

IPC_RCCD_HD inline RigidRotationStatus interpolate_rigid_orientation(const double* q_current,
    const double* q_target, double alpha, double* out) {
    using namespace rotation_detail;
    if (!finite(alpha) || alpha < 0 || alpha > 1) return RigidRotationStatus::InvalidInput;
    double current[4], target[4], relative_q[4];
    if (!normalize(q_current,current) || !normalize(q_target,target)) return RigidRotationStatus::InvalidInput;
    if (alpha == 0 || alpha == 1) {
        for (int k = 0; k < 4; ++k) out[k] = alpha == 0 ? current[k] : target[k];
        return RigidRotationStatus::Success;
    }
    relative(target,current,relative_q);
    if (!normalize(relative_q,relative_q)) return RigidRotationStatus::InvalidInput;
    const double sine = vec(relative_q+1).norm();
    if (relative_q[0] < 0 && sine <= 64.0*DBL_EPSILON) return RigidRotationStatus::AmbiguousFullTurn;
    if (relative_q[0] >= 0 && sine < 1e-12) {
        for (int k = 0; k < 4; ++k) out[k] = (1-alpha)*current[k]+alpha*target[k];
    } else {
        const double factor = 2.0*alpha*atan2(sine,relative_q[0])/sine;
        const double vector[3] = {factor*relative_q[1],factor*relative_q[2],factor*relative_q[3]};
        rigid_orientation(current,vector,1.0,out);
    }
    return normalize(out,out) ? RigidRotationStatus::Success : RigidRotationStatus::InvalidInput;
}

IPC_RCCD_HD inline RigidRotationStatus rigid_angular_velocity_from_orientation(const double* q,
    const double* previous_q, double dt, double* omega) {
    using namespace rotation_detail;
    if (!finite(dt) || dt <= 0) return RigidRotationStatus::InvalidInput;
    double target[4], previous[4], relative_q[4];
    if (!normalize(q,target) || !normalize(previous_q,previous)) return RigidRotationStatus::InvalidInput;
    relative(target,previous,relative_q);
    if (!normalize(relative_q,relative_q)) return RigidRotationStatus::InvalidInput;
    const double sine = vec(relative_q+1).norm();
    if (relative_q[0] < 0 && sine <= 64.0*DBL_EPSILON) return RigidRotationStatus::AmbiguousFullTurn;
    const double factor = relative_q[0] >= 0 && sine < 1e-12 ? 2.0/dt : 2.0*atan2(sine,relative_q[0])/(dt*sine);
    for (int k = 0; k < 3; ++k) omega[k] = factor*relative_q[k+1];
    return RigidRotationStatus::Success;
}

IPC_RCCD_HD inline RigidRotationStatus bound_rigid_orientation(const double* q_anchor,
    const double* q_current, const double* q_target, double theta_bound, double* out) {
    using namespace rotation_detail;
    if (!finite(theta_bound)) return RigidRotationStatus::InvalidInput;
    double current[4],target[4],rel[4];
    if (!normalize(q_current,current) || !normalize(q_target,target)) return RigidRotationStatus::InvalidInput;
    relative(target,current,rel);
    if (!normalize(rel,rel)) return RigidRotationStatus::InvalidInput;
    const double sine = vec(rel+1).norm();
    if (rel[0] < 0 && sine <= 64.0*DBL_EPSILON) return RigidRotationStatus::AmbiguousFullTurn;
    if (theta_bound >= pi || (rel[0] >= 0 && sine == 0)) {
        for (int k = 0; k < 4; ++k) out[k] = target[k];
        return RigidRotationStatus::Success;
    }
    double anchor[4], tangent[4];
    if (!normalize(q_anchor,anchor)) return RigidRotationStatus::InvalidInput;
    const double axis_q[4] = {0,rel[1]/sine,rel[2]/sine,rel[3]/sine};
    rigid_quaternion_multiply(axis_q,current,tangent);
    double anchor_current = 0, anchor_tangent = 0;
    for (int k = 0; k < 4; ++k) { anchor_current += anchor[k]*current[k]; anchor_tangent += anchor[k]*tangent[k]; }
    const double a = fabs(anchor_current), b = (anchor_current < 0 ? -1.0 : 1.0)*anchor_tangent;
    const double cap_cosine = cos(.5*maximum(0,theta_bound));
    if (a+1e-12 < cap_cosine) return RigidRotationStatus::OutsideCap;
    const double amplitude = hypot(a,b), phase = atan2(b,a);
    const double boundary_sine = sqrt(maximum(0,amplitude*amplitude-cap_cosine*cap_cosine));
    const double first_exit = maximum(0,phase+atan2(boundary_sine,cap_cosine));
    const double half_arc = atan2(sine,rel[0]);
    const double alpha = first_exit >= half_arc ? 1 : clamp(first_exit/half_arc,0,1);
    return interpolate_rigid_orientation(current,target,alpha,out);
}

// Same spherical-cap envelope as parallel_helper.cpp; translation padding is
// applied by the caller. This encloses the full arc, not only its endpoints.
IPC_RCCD_HD inline RigidRotationStatus rigid_spherical_cap_box(const double* com,
    const double* orientation, const double* X, double theta_bound, BpAabb& box) {
    using namespace rotation_detail;
    if (!finite(theta_bound)) return RigidRotationStatus::InvalidInput;
    for (int k = 0; k < 3; ++k) if (!finite(com[k]) || !finite(X[k])) return RigidRotationStatus::InvalidInput;
    double q[4]; if (!normalize(orientation,q)) return RigidRotationStatus::InvalidInput;
    const double zero[3] = {0,0,0}; double offset[3];
    rigid_world_position(X,zero,q,offset);
    const double radius = vec(offset).norm();
    if (!finite(radius)) return RigidRotationStatus::InvalidInput;
    const double extent = maximum(0,theta_bound), cosine = cos(extent), sine = sin(extent);
    for (int k = 0; k < 3; ++k) {
        if (radius == 0) { box.lo[k] = com[k]; box.hi[k] = com[k]; continue; }
        if (extent >= pi) { box.lo[k] = com[k]-radius; box.hi[k] = com[k]+radius; continue; }
        const double d = clamp(offset[k]/radius,-1,1), tangent = sqrt(maximum(0,1-d*d));
        const double hi = d >= cosine ? 1.0 : d*cosine+tangent*sine;
        const double lo = -d >= cosine ? -1.0 : d*cosine-tangent*sine;
        box.lo[k] = com[k]+radius*lo; box.hi[k] = com[k]+radius*hi;
    }
    return RigidRotationStatus::Success;
}

} // namespace ipc_gpu::solver

#undef IPC_RCCD_HD
