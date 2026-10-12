#pragma once

#include "rigid_math.cuh"
#include <cmath>

#ifdef __CUDACC__
#define IPC_RESIDENT_RIGID_HD __host__ __device__
#else
#define IPC_RESIDENT_RIGID_HD
#endif

namespace ipc_gpu::resident {

// Scalar-first quaternions and world-space angular velocity, matching
// rigid_body_ipc.cpp. Matrices are row-major. The caller supplies a finite,
// unit previous orientation and owns device status/error handling. These
// routines do not normalize, truncate rotations, or fall back to host work.
struct RigidKinematics {
    double orientation[4];
    double orientation_jacobian[4][3];
    double orientation_hessians[4][9];
};

IPC_RESIDENT_RIGID_HD inline int rigid_epsilon(int i, int j, int k) {
    if (i == j || j == k || i == k) return 0;
    return ((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) ||
            (i == 2 && j == 0 && k == 1)) ? 1 : -1;
}

IPC_RESIDENT_RIGID_HD inline int rigid_quaternion_product_tensor(int a, int b, int c) {
    if (a == 0 && b == 0 && c == 0) return 1;
    if (a == 0 && b == c && b != 0) return -1;
    if (a != 0 && b == 0 && a == c) return 1;
    if (a != 0 && c == 0 && a == b) return 1;
    if (a && b && c) return rigid_epsilon(a - 1, b - 1, c - 1);
    return 0;
}

IPC_RESIDENT_RIGID_HD inline void rigid_quaternion_multiply(const double* a, const double* b, double* out) {
    // Temporaries make aliased input/output safe while retaining CPU order.
    const double w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    const double x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    const double y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    const double z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

IPC_RESIDENT_RIGID_HD inline void rigid_quaternion_exp(const double* omega, double dt, double* out) {
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

IPC_RESIDENT_RIGID_HD inline void rigid_orientation(const double* previous_q,
                                                   const double* omega, double dt, double* q) {
    double exponential[4];
    rigid_quaternion_exp(omega, dt, exponential);
    rigid_quaternion_multiply(exponential, previous_q, q);
}

// exp((dt/2) omega), including the same small-angle Taylor branch as CPU.
IPC_RESIDENT_RIGID_HD inline void rigid_exp_derivatives(const double* omega, double dt,
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

IPC_RESIDENT_RIGID_HD inline RigidKinematics rigid_kinematics(const double* previous_q,
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

IPC_RESIDENT_RIGID_HD inline void rigid_rotation_predictor(const double* previous_q,
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
IPC_RESIDENT_RIGID_HD inline RigidInertia build_rigid_inertia(const double* previous_q,
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

IPC_RESIDENT_RIGID_HD inline void rigid_reference_derivatives(const double* X,
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

IPC_RESIDENT_RIGID_HD inline void rigid_world_position(const double* X, const double* com,
                                                      const double* orientation, double* position) {
    const double reference[4] = {0.0, X[0], X[1], X[2]};
    const double conjugate[4] = {orientation[0], -orientation[1], -orientation[2], -orientation[3]};
    double right[4], rotated[4];
    rigid_quaternion_multiply(reference, conjugate, right);
    rigid_quaternion_multiply(orientation, right, rotated);
    for (int k = 0; k < 3; ++k) position[k] = com[k] + rotated[k+1];
}

} // namespace ipc_gpu::resident

#undef IPC_RESIDENT_RIGID_HD
