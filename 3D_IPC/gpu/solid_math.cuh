/*
The signed QR-SVD below is adapted from third_party/tgsl/ImplicitQRSVD.h:
Copyright (c) 2016 Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

If the code is used in an article, the following paper shall be cited:
Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran,
"Implicit-shifted Symmetric QR Singular Value Decomposition of 3x3 Matrices",
University of California Los Angeles, 2016.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

#include "solid_types.h"

#include <cmath>

// Host spelling supports an explicitly selected arithmetic test executable.
// Production never calls this arithmetic on the host.
#if defined(__CUDACC__)
#define IPC_GPU_SOLID_DEVICE __device__
#else
#define IPC_GPU_SOLID_DEVICE
#endif

namespace ipc_gpu {
namespace solid_detail {

struct Matrix { double v[3][3]; };

IPC_GPU_SOLID_DEVICE inline Matrix identity() {
    Matrix result{};
    for (int i = 0; i < 3; ++i) result.v[i][i] = 1.0;
    return result;
}

IPC_GPU_SOLID_DEVICE inline bool finite(double value) {
    return (value - value) == 0.0;
}

IPC_GPU_SOLID_DEVICE inline Derivatives invalid_derivatives() {
    Derivatives result;
    for (int i = 0; i < 3; ++i) result.gradient[i] = NAN;
    for (int i = 0; i < 9; ++i) result.hessian[i] = NAN;
    return result;
}

struct Givens {
    double c = 1.0, s = 0.0;
    IPC_GPU_SOLID_DEVICE Givens() = default;
    IPC_GPU_SOLID_DEVICE Givens(double a, double b) { compute(a, b); }
    IPC_GPU_SOLID_DEVICE void compute(double a, double b) {
        const double d = a * a + b * b;
        c = 1.0; s = 0.0;
        if (d != 0.0) {
            const double t = 1.0 / ::sqrt(d);
            c = a * t; s = -b * t;
        }
    }
    IPC_GPU_SOLID_DEVICE void unconventional(double a, double b) {
        const double d = a * a + b * b;
        c = 0.0; s = 1.0;
        if (d != 0.0) {
            const double t = 1.0 / ::sqrt(d);
            s = a * t; c = b * t;
        }
    }
    IPC_GPU_SOLID_DEVICE void row(Matrix& a, int i, int k) const {
        for (int j = 0; j < 3; ++j) {
            const double x = a.v[i][j], y = a.v[k][j];
            a.v[i][j] = c * x - s * y;
            a.v[k][j] = s * x + c * y;
        }
    }
    IPC_GPU_SOLID_DEVICE void column(Matrix& a, int i, int k) const {
        for (int j = 0; j < 3; ++j) {
            const double x = a.v[j][i], y = a.v[j][k];
            a.v[j][i] = c * x - s * y;
            a.v[j][k] = s * x + c * y;
        }
    }
};

IPC_GPU_SOLID_DEVICE inline void zero_chase(Matrix& b, Matrix& u, Matrix& v) {
    const Givens r1(b.v[0][0], b.v[1][0]);
    Givens r2;
    if (b.v[1][0] != 0.0)
        r2.compute(b.v[0][0] * b.v[0][1] + b.v[1][0] * b.v[1][1],
                   b.v[0][0] * b.v[0][2] + b.v[1][0] * b.v[1][2]);
    else r2.compute(b.v[0][1], b.v[0][2]);
    r1.row(b, 0, 1);
    r2.column(b, 1, 2); r2.column(v, 1, 2);
    const Givens r3(b.v[1][1], b.v[2][1]);
    r3.row(b, 1, 2);
    r1.column(u, 0, 1); r3.column(u, 1, 2);
}

template <int t>
IPC_GPU_SOLID_DEVICE inline void process(Matrix& b, Matrix& u, double (&sigma)[3], Matrix& v) {
    constexpr int other = t == 1 ? 0 : 2;
    sigma[other] = b.v[other][other];
    const double a00 = b.v[t][t], a01 = b.v[t][t + 1];
    const double a10 = b.v[t + 1][t], a11 = b.v[t + 1][t + 1];
    const double x0 = a00 + a11, x1 = a10 - a01;
    const double denominator = ::sqrt(x0 * x0 + x1 * x1);
    Givens gu, gv;
    if (denominator != 0.0) { gu.c = x0 / denominator; gu.s = -x1 / denominator; }
    const double x = gu.c * a00 - gu.s * a10;
    const double y = gu.c * a01 - gu.s * a11;
    const double z = gu.s * a01 + gu.c * a11;
    double cosine = 1.0, sine = 0.0, s0 = x, s1 = z;
    if (y != 0.0) {
        const double tau = 0.5 * (x - z);
        const double w = ::sqrt(tau * tau + y * y);
        const double ratio = y / (tau > 0.0 ? tau + w : tau - w);
        cosine = 1.0 / ::sqrt(ratio * ratio + 1.0);
        sine = -ratio * cosine;
        const double c2 = cosine * cosine, csy = 2.0 * cosine * sine * y;
        const double s2 = sine * sine;
        s0 = c2 * x - csy + s2 * z;
        s1 = s2 * x + csy + c2 * z;
    }
    if (s0 < s1) {
        const double old = s0; s0 = s1; s1 = old;
        gv.c = -sine; gv.s = cosine;
    } else { gv.c = cosine; gv.s = sine; }
    const double uc = gu.c * gv.c - gu.s * gv.s;
    const double us = gu.s * gv.c + gu.c * gv.s;
    gu.c = uc; gu.s = us;
    sigma[t] = s0; sigma[t + 1] = s1;
    gu.column(u, t, t + 1); gv.column(v, t, t + 1);
}

IPC_GPU_SOLID_DEVICE inline void flip(Matrix& u, double (&sigma)[3], int column) {
    sigma[column] = -sigma[column];
    for (int i = 0; i < 3; ++i) u.v[i][column] = -u.v[i][column];
}
IPC_GPU_SOLID_DEVICE inline void swap_columns(Matrix& a, int i, int j) {
    for (int row = 0; row < 3; ++row) {
        const double x = a.v[row][i]; a.v[row][i] = a.v[row][j]; a.v[row][j] = x;
    }
}
IPC_GPU_SOLID_DEVICE inline void swap_values(double& a, double& b) {
    const double old = a; a = b; b = old;
}
IPC_GPU_SOLID_DEVICE inline void negate_columns(Matrix& u, Matrix& v, int column) {
    for (int row = 0; row < 3; ++row) {
        u.v[row][column] = -u.v[row][column];
        v.v[row][column] = -v.v[row][column];
    }
}
template <int t>
IPC_GPU_SOLID_DEVICE inline void sort(Matrix& u, double (&sigma)[3], Matrix& v) {
    if constexpr (t == 0) {
        if (::fabs(sigma[1]) >= ::fabs(sigma[2])) {
            if (sigma[1] < 0.0) { flip(u, sigma, 1); flip(u, sigma, 2); }
            return;
        }
        if (sigma[2] < 0.0) { flip(u, sigma, 1); flip(u, sigma, 2); }
        swap_values(sigma[1], sigma[2]);
        swap_columns(u, 1, 2); swap_columns(v, 1, 2);
        if (sigma[1] > sigma[0]) {
            swap_values(sigma[0], sigma[1]);
            swap_columns(u, 0, 1); swap_columns(v, 0, 1);
        } else negate_columns(u, v, 2);
    } else {
        if (::fabs(sigma[0]) >= sigma[1]) {
            if (sigma[0] < 0.0) { flip(u, sigma, 0); flip(u, sigma, 2); }
            return;
        }
        swap_values(sigma[0], sigma[1]);
        swap_columns(u, 0, 1); swap_columns(v, 0, 1);
        if (::fabs(sigma[1]) < ::fabs(sigma[2])) {
            swap_values(sigma[1], sigma[2]);
            swap_columns(u, 1, 2); swap_columns(v, 1, 2);
        } else negate_columns(u, v, 1);
        if (sigma[1] < 0.0) { flip(u, sigma, 1); flip(u, sigma, 2); }
    }
}

// Match the CPU Eigen 3x3 products: two packet rows with explicit FMA,
// followed by the scalar row's right-associated non-contracted reduction.
// Build with --fmad=false: every other QR-SVD operation has strict rounding.
IPC_GPU_SOLID_DEVICE inline double product3(int row, double a0, double a1,
    double a2, double b0, double b1, double b2) {
    if (row < 2) return ::fma(a2, b2, ::fma(a1, b1, a0 * b0));
    return a0 * b0 + (a1 * b1 + a2 * b2);
}

IPC_GPU_SOLID_DEVICE inline bool signed_polar(Matrix b, Matrix& rotation) {
    Matrix u = identity(), v = identity();
    const Givens first(b.v[1][0], b.v[2][0]);
    first.row(b, 1, 2); first.column(u, 1, 2);
    zero_chase(b, u, v);
    double a1 = b.v[0][0], b1 = b.v[0][1], a2 = b.v[1][1];
    double a3 = b.v[2][2], b2 = b.v[1][2];
    double g1 = a1 * b1, g2 = a2 * b2;
    const double scale = 0.5 * ::sqrt(a1 * a1 + a2 * a2 + a3 * a3 + b1 * b1 + b2 * b2);
    const double tolerance = 2.2737367544323206e-13 * (scale > 1.0 ? scale : 1.0);
    int iterations = 0;
    while (::fabs(b2) > tolerance && ::fabs(b1) > tolerance
        && ::fabs(a1) > tolerance && ::fabs(a2) > tolerance && ::fabs(a3) > tolerance) {
        // The reference converges rapidly for 3x3 inputs. A failed record is
        // surfaced as a CUDA computation error, never replaced by CPU output.
        if (++iterations > 256) return false;
        const double w1 = a2 * a2 + b1 * b1, w2 = a3 * a3 + b2 * b2;
        const double d = 0.5 * (w1 - w2), bs = g2 * g2;
        const double shift = w2 - ::copysign(bs / (::fabs(d) + ::sqrt(d * d + bs)), d);
        const Givens r(a1 * a1 - shift, g1);
        r.column(b, 0, 1); r.column(v, 0, 1);
        zero_chase(b, u, v);
        a1 = b.v[0][0]; b1 = b.v[0][1]; a2 = b.v[1][1];
        a3 = b.v[2][2]; b2 = b.v[1][2];
        g1 = a1 * b1; g2 = a2 * b2;
    }
    double sigma[3];
    if (::fabs(b2) <= tolerance) {
        process<0>(b, u, sigma, v); sort<0>(u, sigma, v);
    } else if (::fabs(b1) <= tolerance) {
        process<1>(b, u, sigma, v); sort<1>(u, sigma, v);
    } else if (::fabs(a2) <= tolerance) {
        Givens r; r.unconventional(b.v[1][2], b.v[2][2]);
        r.row(b, 1, 2); r.column(u, 1, 2);
        process<0>(b, u, sigma, v); sort<0>(u, sigma, v);
    } else if (::fabs(a3) <= tolerance) {
        const Givens r1(b.v[1][1], b.v[1][2]);
        r1.column(b, 1, 2); r1.column(v, 1, 2);
        const Givens r2(b.v[0][0], b.v[0][2]);
        r2.column(b, 0, 2); r2.column(v, 0, 2);
        process<0>(b, u, sigma, v); sort<0>(u, sigma, v);
    } else if (::fabs(a1) <= tolerance) {
        Givens r1; r1.unconventional(b.v[0][1], b.v[1][1]);
        r1.row(b, 0, 1); r1.column(u, 0, 1);
        Givens r2; r2.unconventional(b.v[0][2], b.v[2][2]);
        r2.row(b, 0, 2); r2.column(u, 0, 2);
        process<1>(b, u, sigma, v); sort<1>(u, sigma, v);
    } else return false;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column) {
            rotation.v[row][column] = product3(row,
                u.v[row][0], u.v[row][1], u.v[row][2],
                v.v[column][0], v.v[column][1], v.v[column][2]);
            if (!finite(rotation.v[row][column])) return false;
        }
    return true;
}

} // namespace solid_detail

IPC_GPU_SOLID_DEVICE inline Derivatives evaluate(const Solid& input, const SolidParameters& params) {
    using solid_detail::product3;
    solid_detail::Matrix f, rotation, cofactor, stress;
    for (int row = 0; row < 3; ++row) {
        const double a = input.positions[3 + row] - input.positions[row];
        const double b = input.positions[6 + row] - input.positions[row];
        const double c = input.positions[9 + row] - input.positions[row];
        for (int column = 0; column < 3; ++column) {
            f.v[row][column] = product3(row, a, b, c, input.dm_inverse[column],
                input.dm_inverse[3 + column], input.dm_inverse[6 + column]);
            if (!solid_detail::finite(f.v[row][column])) return solid_detail::invalid_derivatives();
        }
    }
    if (!solid_detail::signed_polar(f, rotation)) return solid_detail::invalid_derivatives();
    // Cofactors are polynomial, so inverted and singular deformations retain
    // the scalar model's behavior without an inverse or regularization.
    cofactor.v[0][0] = f.v[1][1] * f.v[2][2] - f.v[2][1] * f.v[1][2];
    cofactor.v[0][1] = f.v[2][0] * f.v[1][2] - f.v[1][0] * f.v[2][2];
    cofactor.v[0][2] = f.v[1][0] * f.v[2][1] - f.v[2][0] * f.v[1][1];
    cofactor.v[1][0] = f.v[2][1] * f.v[0][2] - f.v[0][1] * f.v[2][2];
    cofactor.v[1][1] = f.v[0][0] * f.v[2][2] - f.v[2][0] * f.v[0][2];
    cofactor.v[1][2] = f.v[2][0] * f.v[0][1] - f.v[0][0] * f.v[2][1];
    cofactor.v[2][0] = f.v[0][1] * f.v[1][2] - f.v[1][1] * f.v[0][2];
    cofactor.v[2][1] = f.v[1][0] * f.v[0][2] - f.v[0][0] * f.v[1][2];
    cofactor.v[2][2] = f.v[0][0] * f.v[1][1] - f.v[1][0] * f.v[0][1];
    const double determinant = (f.v[0][0] * (f.v[1][1] * f.v[2][2] - f.v[1][2] * f.v[2][1])
        - f.v[0][1] * (f.v[1][0] * f.v[2][2] - f.v[1][2] * f.v[2][0]))
        + f.v[0][2] * (f.v[1][0] * f.v[2][1] - f.v[1][1] * f.v[2][0]);
    const double twice_mu = 2.0 * params.mu;
    const double volumetric = params.lambda * (determinant - 1.0);
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            stress.v[row][column] = twice_mu * (f.v[row][column] - rotation.v[row][column])
                + volumetric * cofactor.v[row][column];
    double u[3];
    Derivatives result{};
    for (int row = 0; row < 3; ++row) {
        result.gradient[row] = product3(row, stress.v[row][0], stress.v[row][1], stress.v[row][2],
            input.shape[0], input.shape[1], input.shape[2]) * input.measure;
        u[row] = product3(row, cofactor.v[row][0], cofactor.v[row][1], cofactor.v[row][2],
            input.shape[0], input.shape[1], input.shape[2]);
    }
    const double shape_norm = (input.shape[0] * input.shape[0] + input.shape[1] * input.shape[1])
        + input.shape[2] * input.shape[2];
    const double diagonal = twice_mu * shape_norm * input.measure;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            result.hessian[3 * row + column] = (row == column ? diagonal : 0.0)
                + params.lambda * u[row] * u[column] * input.measure;
    return result;
}

} // namespace ipc_gpu

#undef IPC_GPU_SOLID_DEVICE
