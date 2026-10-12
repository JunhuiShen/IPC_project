// The symmetric_eigen2 QR iteration and real Givens construction below are
// adapted from Eigen's Eigenvalues/SelfAdjointEigenSolver.h and Jacobi/Jacobi.h.
// Eigen is a lightweight C++ template library for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "kernels.h"

#include <cmath>

// The host spelling permits the same arithmetic to be checked against the
// scalar reference without CUDA hardware. Production calls are device-only.
#if defined(__CUDACC__)
#define IPC_GPU_CLOTH_DEVICE __device__
#else
#define IPC_GPU_CLOTH_DEVICE
#endif

namespace ipc_gpu {
namespace cloth_detail {

struct Vector {
    double v[3];
    IPC_GPU_CLOTH_DEVICE double& operator[](int i) { return v[i]; }
    IPC_GPU_CLOTH_DEVICE double operator[](int i) const { return v[i]; }
};

IPC_GPU_CLOTH_DEVICE inline Vector difference(const double* a, const double* b) {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

IPC_GPU_CLOTH_DEVICE inline double dot(const Vector& a, const Vector& b) {
    return ::fma(a[2], b[2], a[0] * b[0] + a[1] * b[1]);
}

IPC_GPU_CLOTH_DEVICE inline Vector cross(const Vector& a, const Vector& b) {
    return {{::fma(a[1], b[2], -(a[2] * b[1])),
             ::fma(a[2], b[0], -(a[0] * b[2])),
             ::fma(a[0], b[1], -(a[1] * b[0]))}};
}

IPC_GPU_CLOTH_DEVICE inline bool finite(double x) {
    return (x - x) == 0.0;
}

IPC_GPU_CLOTH_DEVICE inline Derivatives invalid_derivatives() {
    Derivatives result;
    for (int i = 0; i < 3; ++i) result.gradient[i] = NAN;
    for (int i = 0; i < 9; ++i) result.hessian[i] = NAN;
    return result;
}

// The CPU cache uses SelfAdjointEigenSolver::compute, not computeDirect.
// Its two-dimensional case is a scaled symmetric QR iteration. Retain that
// algorithm, its stopping criterion and its sorted eigenvectors because the
// small eigenvalue is subsequently clamped at exactly 1e-12.
IPC_GPU_CLOTH_DEVICE inline bool symmetric_eigen2(
    const double c[2][2], double eigenvalues[2], double u[2][2]) {
    if (!finite(c[0][0]) || !finite(c[1][0]) || !finite(c[1][1])) return false;
    double scale = ::fabs(c[0][0]);
    if (::fabs(c[1][0]) > scale) scale = ::fabs(c[1][0]);
    if (::fabs(c[1][1]) > scale) scale = ::fabs(c[1][1]);
    if (scale == 0.0) scale = 1.0;
    double a = c[0][0] / scale;
    double b = c[1][0] / scale;
    double d = c[1][1] / scale;
    u[0][0] = u[1][1] = 1.0;
    u[0][1] = u[1][0] = 0.0;
    constexpr double inverse_epsilon = 4503599627370496.0;
    constexpr double minimum_normal = 2.2250738585072014e-308;
    int iteration = 0;
    for (;;) {
        const double scaled_b = inverse_epsilon * b;
        if (::fabs(b) < minimum_normal
            || scaled_b * scaled_b <= ::fabs(a) + ::fabs(d)) b = 0.0;
        if (b == 0.0) break;
        if (++iteration > 60) return false;

        const double half_difference = (a - d) * 0.5;
        double shift = d;
        if (half_difference == 0.0) {
            shift -= ::fabs(b);
        } else {
            const double square = b * b;
            const double h = ::hypot(half_difference, b);
            const double denominator = half_difference + (half_difference > 0.0 ? h : -h);
            shift -= square == 0.0 ? b / (denominator / b) : square / denominator;
        }

        const double x = a - shift;
        double cosine, sine;
        if (x == 0.0) {
            cosine = 0.0;
            sine = b < 0.0 ? 1.0 : -1.0;
        } else if (::fabs(x) > ::fabs(b)) {
            const double tangent = b / x;
            double norm = ::sqrt(1.0 + tangent * tangent);
            if (x < 0.0) norm = -norm;
            cosine = 1.0 / norm;
            sine = -tangent * cosine;
        } else {
            const double tangent = x / b;
            double norm = ::sqrt(1.0 + tangent * tangent);
            if (b < 0.0) norm = -norm;
            sine = -1.0 / norm;
            cosine = -tangent * sine;
        }
        const double left = sine * a + cosine * b;
        const double right = sine * b + cosine * d;
        a = cosine * (cosine * a - sine * b)
            - sine * (cosine * b - sine * d);
        d = sine * left + cosine * right;
        b = cosine * left - sine * right;
        for (int row = 0; row < 2; ++row) {
            const double first = u[row][0], second = u[row][1];
            u[row][0] = cosine * first - sine * second;
            u[row][1] = sine * first + cosine * second;
        }
    }
    if (d < a) {
        const double temp = a; a = d; d = temp;
        for (int row = 0; row < 2; ++row) {
            const double first = u[row][0];
            u[row][0] = u[row][1]; u[row][1] = first;
        }
    }
    eigenvalues[0] = a * scale;
    eigenvalues[1] = d * scale;
    return true;
}

IPC_GPU_CLOTH_DEVICE inline void inverse2(const double a[2][2], double inverse[2][2]) {
    // C is intentionally not regularized: only the eigenvalues used by S are
    // clamped in buildCorotatedCache. Rank-deficient C retains its nonfinite
    // reference behavior instead of silently changing the constitutive model.
    const double reciprocal = 1.0 / (a[0][0] * a[1][1] - a[0][1] * a[1][0]);
    inverse[0][0] = a[1][1] * reciprocal;
    inverse[0][1] = -a[0][1] * reciprocal;
    inverse[1][0] = -a[1][0] * reciprocal;
    inverse[1][1] = a[0][0] * reciprocal;
}

} // namespace cloth_detail

IPC_GPU_CLOTH_DEVICE inline Derivatives evaluate(const Point& input, const Parameters& params) {
    Derivatives result{};
    const double pin_scale = params.dt2 * params.kpin;
    for (int axis = 0; axis < 3; ++axis) {
        double gradient;
        if (input.solid) {
            const double gravity_force = (-input.mass) * params.gravity[axis];
            const double gravity_step = params.dt2 * gravity_force;
            gradient = ::fma(input.mass, input.position[axis]-input.predicted[axis], gravity_step);
            if (input.pinned) gradient = ::fma(pin_scale,input.position[axis]-input.target[axis],gradient);
        } else {
            gradient = 0.0 + input.mass * (input.position[axis] - input.predicted[axis]);
            gradient += params.dt2 * ((-input.mass) * params.gravity[axis]);
            if (input.pinned) gradient += pin_scale * (input.position[axis] - input.target[axis]);
        }
        result.gradient[axis] = gradient;
        result.hessian[3 * axis + axis] = input.pinned ? input.mass + pin_scale : input.mass;
    }
    return result;
}

IPC_GPU_CLOTH_DEVICE inline Derivatives evaluate(const Triangle& input, const Parameters& params) {
    double f[3][2], c[2][2];
    for (int row = 0; row < 3; ++row) {
        const double a = input.positions[3 + row] - input.positions[row];
        const double b = input.positions[6 + row] - input.positions[row];
        for (int column = 0; column < 2; ++column)
            f[row][column] = a * input.dm_inverse[column] + b * input.dm_inverse[2 + column];
    }
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 2; ++column)
            c[row][column] = (f[0][row] * f[0][column] + f[1][row] * f[1][column])
                + f[2][row] * f[2][column];
    double eigenvalues[2], u[2][2];
    if (!cloth_detail::symmetric_eigen2(c, eigenvalues, u))
        return cloth_detail::invalid_derivatives();
    const double singular[2] = {
        ::sqrt(eigenvalues[0] < 1e-12 ? 1e-12 : eigenvalues[0]),
        ::sqrt(eigenvalues[1] < 1e-12 ? 1e-12 : eigenvalues[1])};
    double s[2][2], s_inverse[2][2], c_inverse[2][2];
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 2; ++column)
            s[row][column] = (u[row][0] * singular[0]) * u[column][0]
                + (u[row][1] * singular[1]) * u[column][1];
    cloth_detail::inverse2(s, s_inverse);
    cloth_detail::inverse2(c, c_inverse);
    double rotation[3][2], b[3][2], stress[3][2];
    const double determinant = singular[0] * singular[1];
    const double trace = s[0][0] + s[1][1];
    const double twice_mu = 2.0 * params.mu;
    const double volumetric = params.lambda * (determinant - 1.0) * determinant;
    const double positive = (0.5 * params.lambda) * (2.0 * determinant - 1.0) * determinant;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 2; ++column) {
            rotation[row][column] = f[row][0] * s_inverse[0][column]
                + f[row][1] * s_inverse[1][column];
            b[row][column] = f[row][0] * c_inverse[0][column]
                + f[row][1] * c_inverse[1][column];
            stress[row][column] = twice_mu * (f[row][column] - rotation[row][column])
                + volumetric * b[row][column];
        }
    Derivatives result{};
    for (int row = 0; row < 3; ++row) {
        double gradient = 0.0;
        for (int beta = 0; beta < 2; ++beta)
            gradient += stress[row][beta] * input.shape[beta];
        result.gradient[row] = input.area * gradient;
        for (int column = 0; column < 3; ++column) {
            const double rrt = ::fma(rotation[row][1], rotation[column][1],
                rotation[row][0] * rotation[column][0]);
            const double q = ::fma(b[row][1], f[column][1], b[row][0] * f[column][0]);
            double sum = 0.0;
            for (int beta = 0; beta < 2; ++beta)
                for (int eta = 0; eta < 2; ++eta) {
                    const double sinv = s_inverse[eta][beta];
                    const double cinv = c_inverse[eta][beta];
                    const double dcd_f = (eta == 0 ? -rotation[column][1] : rotation[column][0]) / trace;
                    const double re = beta == 0 ? rotation[row][1] : -rotation[row][0];
                    double dr = row == column ? sinv : 0.0;
                    dr = ::fma(-rrt, sinv, dr);
                    dr = ::fma(-dcd_f, re, dr);
                    double dp = row == column ? volumetric * cinv : 0.0;
                    dp = ::fma(-volumetric,
                        ::fma(b[row][eta], b[column][beta], q * cinv), dp);
                    const double product = b[column][eta] * b[row][beta];
                    dp = ::fma(positive, product + product, dp);
                    dp = ::fma(twice_mu, (row == column && beta == eta ? 1.0 : 0.0) - dr, dp);
                    sum = ::fma(dp * input.shape[beta], input.shape[eta], sum);
                }
            result.hessian[3 * row + column] = input.area * sum;
        }
    }
    return result;
}

IPC_GPU_CLOTH_DEVICE inline Derivatives evaluate(const Hinge& input, const Parameters& params) {
    using cloth_detail::Vector;
    using cloth_detail::difference;
    using cloth_detail::dot;
    using cloth_detail::cross;
    const Vector edge = difference(input.positions + 3, input.positions);
    const Vector a = difference(input.positions + 6, input.positions);
    const Vector b = difference(input.positions + 9, input.positions);
    const Vector normal_a = cross(edge, a), normal_b = cross(b, edge);
    const double length = ::sqrt(dot(edge, edge));
    const double normal_a_squared = dot(normal_a, normal_a);
    const double normal_b_squared = dot(normal_b, normal_b);
    Derivatives result{};
    if (length <= 0.0 || normal_a_squared <= 0.0 || normal_b_squared <= 0.0) return result;
    const Vector unit_edge{{edge[0] / length, edge[1] / length, edge[2] / length}};
    const double x = dot(normal_a, normal_b);
    const double y = dot(cross(normal_a, normal_b), unit_edge);
    const double angle = ::atan2(y, x);
    Vector dx{}, dy{};
    if (input.active_node == 0) {
        const Vector ca = difference(input.positions + 6, input.positions + 3);
        const Vector cb = difference(input.positions + 9, input.positions + 3);
        const Vector first = cross(normal_b, ca), second = cross(normal_a, cb);
        const double coefficient = dot(ca, normal_b) + dot(normal_a, cb);
        const double ea = dot(unit_edge, ca), eb = dot(unit_edge, cb);
        for (int axis = 0; axis < 3; ++axis) {
            dx[axis] = first[axis] - second[axis];
            dy[axis] = ::fma(-eb, normal_a[axis],
                ::fma(-ea, normal_b[axis], coefficient * unit_edge[axis]));
        }
    } else if (input.active_node == 1) {
        const Vector first = cross(normal_a, b), second = cross(normal_b, a);
        const double coefficient = -(dot(a, normal_b) + dot(normal_a, b));
        const double ea = dot(unit_edge, a), eb = dot(unit_edge, b);
        for (int axis = 0; axis < 3; ++axis) {
            dx[axis] = first[axis] - second[axis];
            dy[axis] = ::fma(eb, normal_a[axis],
                ::fma(ea, normal_b[axis], coefficient * unit_edge[axis]));
        }
    } else {
        dx = input.active_node == 2 ? cross(normal_b, edge) : cross(edge, normal_a);
        for (int axis = 0; axis < 3; ++axis)
            dy[axis] = (-length) * (input.active_node == 2 ? normal_b[axis] : normal_a[axis]);
    }
    Vector angle_gradient;
    const double denominator = normal_a_squared * normal_b_squared;
    const double scale = (2.0 * params.bending_stiffness) * input.coefficient;
    for (int axis = 0; axis < 3; ++axis) {
        angle_gradient[axis] = ::fma(-y, dx[axis], x * dy[axis]) / denominator;
        result.gradient[axis] = (scale * (angle - input.rest_angle)) * angle_gradient[axis];
    }
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            result.hessian[3 * row + column] = (scale * angle_gradient[row]) * angle_gradient[column];
    return result;
}

} // namespace ipc_gpu

#undef IPC_GPU_CLOTH_DEVICE
