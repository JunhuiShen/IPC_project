/*
Adapted from third_party/tgsl/ImplicitQRSVD.h:
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
#include "batched_polar.h"
#include "third_party/tgsl/ImplicitQRSVD.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace volumetric_detail {
namespace {
void scalar_polar(const Mat33& input, Mat33& rotation) {
    if (!input.allFinite()) throw std::invalid_argument("deformation gradient must be finite");
    Mat33 U, V;
    Vec3 sigma;
    JIXIE::singularValueDecomposition(input, U, sigma, V);
    rotation.noalias() = U * V.transpose();
}

#if defined(__AVX512F__)
using Mask = __mmask8;
constexpr Mask all = 0xff;
struct P {
    __m512d v;
    P() = default;
    explicit P(double x) : v(_mm512_set1_pd(x)) {}
    explicit P(__m512d x) : v(x) {}
    friend P operator+(P a, P b) { return P(_mm512_add_pd(a.v, b.v)); }
    friend P operator-(P a, P b) { return P(_mm512_sub_pd(a.v, b.v)); }
    friend P operator*(P a, P b) { return P(_mm512_mul_pd(a.v, b.v)); }
    friend P operator/(P a, P b) { return P(_mm512_div_pd(a.v, b.v)); }
    friend P operator-(P a) {
        return P(_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a.v),
            _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL)))));
    }
};
P choose(Mask mask, P yes, P no) { return P(_mm512_mask_blend_pd(mask, no.v, yes.v)); }
P abs(P a) {
    return P(_mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a.v),
        _mm512_set1_epi64(0x7fffffffffffffffLL))));
}
P sqrt(P a) { return P(_mm512_sqrt_pd(a.v)); }
P copysign(P magnitude, P sign) {
    const auto bits = _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL));
    return P(_mm512_castsi512_pd(_mm512_or_si512(
        _mm512_andnot_si512(bits, _mm512_castpd_si512(magnitude.v)),
        _mm512_and_si512(bits, _mm512_castpd_si512(sign.v)))));
}
Mask lt(P a, P b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ); }
Mask le(P a, P b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ); }
Mask gt(P a, P b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ); }
Mask ge(P a, P b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ); }
Mask ne(P a, P b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_NEQ_UQ); }
P max(P a, P b) { return choose(lt(a, b), b, a); }
struct Matrix { P v[3][3]; };
Matrix identity() {
    Matrix result;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) result.v[i][j] = P(i == j ? 1.0 : 0.0);
    return result;
}
struct Givens {
    P c{1.0}, s{0.0};
    Givens() = default;
    Givens(P a, P b) { compute(a, b); }
    void compute(P a, P b) {
        const P d = a * a + b * b;
        const Mask nonzero = ne(d, P(0.0));
        const P t = P(1.0) / sqrt(choose(nonzero, d, P(1.0)));
        c = choose(nonzero, a * t, P(1.0));
        s = choose(nonzero, -b * t, P(0.0));
    }
    void unconventional(P a, P b) {
        const P d = a * a + b * b;
        const Mask nonzero = ne(d, P(0.0));
        const P t = P(1.0) / sqrt(choose(nonzero, d, P(1.0)));
        s = choose(nonzero, a * t, P(1.0));
        c = choose(nonzero, b * t, P(0.0));
    }
    void row(Matrix& a, int i, int k, Mask mask = all) const {
        for (int j = 0; j < 3; ++j) {
            const P x = a.v[i][j], y = a.v[k][j];
            a.v[i][j] = choose(mask, c * x - s * y, x);
            a.v[k][j] = choose(mask, s * x + c * y, y);
        }
    }
    void column(Matrix& a, int i, int k, Mask mask = all) const {
        for (int j = 0; j < 3; ++j) {
            const P x = a.v[j][i], y = a.v[j][k];
            a.v[j][i] = choose(mask, c * x - s * y, x);
            a.v[j][k] = choose(mask, s * x + c * y, y);
        }
    }
};

void zero_chase(Matrix& b, Matrix& u, Matrix& v, Mask mask = all) {
    Givens r1(b.v[0][0], b.v[1][0]);
    const Mask nonzero = ne(b.v[1][0], P(0.0));
    Givens r2(choose(nonzero, b.v[0][0] * b.v[0][1] + b.v[1][0] * b.v[1][1], b.v[0][1]),
               choose(nonzero, b.v[0][0] * b.v[0][2] + b.v[1][0] * b.v[1][2], b.v[0][2]));
    r1.row(b, 0, 1, mask);
    r2.column(b, 1, 2, mask);
    r2.column(v, 1, 2, mask);
    Givens r3(b.v[1][1], b.v[2][1]);
    r3.row(b, 1, 2, mask);
    r1.column(u, 0, 1, mask);
    r3.column(u, 1, 2, mask);
}

template <int t>
void process(Matrix& b, Matrix& u, P (&sigma)[3], Matrix& v, Mask mask) {
    if (!mask) return;
    constexpr int other = t == 1 ? 0 : 2;
    sigma[other] = choose(mask, b.v[other][other], sigma[other]);
    const P a00 = choose(mask, b.v[t][t], P(1.0));
    const P a01 = choose(mask, b.v[t][t + 1], P(0.0));
    const P a10 = choose(mask, b.v[t + 1][t], P(0.0));
    const P a11 = choose(mask, b.v[t + 1][t + 1], P(1.0));
    const P x0 = a00 + a11, x1 = a10 - a01;
    const P denominator = sqrt(x0 * x0 + x1 * x1);
    const Mask nonzero = ne(denominator, P(0.0));
    const P safe = choose(nonzero, denominator, P(1.0));
    Givens gu, gv;
    gu.c = choose(nonzero, x0 / safe, P(1.0));
    gu.s = choose(nonzero, -x1 / safe, P(0.0));
    const P x = gu.c * a00 - gu.s * a10;
    const P y = gu.c * a01 - gu.s * a11;
    const P z = gu.s * a01 + gu.c * a11;
    const Mask ynonzero = ne(y, P(0.0));
    const P tau = P(0.5) * (x - z);
    const P w = sqrt(tau * tau + y * y);
    const P divisor = choose(ynonzero, choose(gt(tau, P(0.0)), tau + w, tau - w), P(1.0));
    const P ratio = y / divisor;
    const P cosine = choose(ynonzero, P(1.0) / sqrt(ratio * ratio + P(1.0)), P(1.0));
    const P sine = choose(ynonzero, -ratio * cosine, P(0.0));
    const P c2 = cosine * cosine;
    const P csy = P(2.0) * cosine * sine * y;
    const P s2 = sine * sine;
    P s0 = choose(ynonzero, c2 * x - csy + s2 * z, x);
    P s1 = choose(ynonzero, s2 * x + csy + c2 * z, z);
    const Mask swap = lt(s0, s1);
    const P old = s0;
    s0 = choose(swap, s1, s0);
    s1 = choose(swap, old, s1);
    gv.c = choose(swap, -sine, cosine);
    gv.s = choose(swap, cosine, sine);
    const P uc = gu.c * gv.c - gu.s * gv.s;
    const P us = gu.s * gv.c + gu.c * gv.s;
    gu.c = uc;
    gu.s = us;
    sigma[t] = choose(mask, s0, sigma[t]);
    sigma[t + 1] = choose(mask, s1, sigma[t + 1]);
    gu.column(u, t, t + 1, mask);
    gv.column(v, t, t + 1, mask);
}

void flip(Matrix& u, P (&sigma)[3], int column, Mask mask) {
    sigma[column] = choose(mask, -sigma[column], sigma[column]);
    for (int i = 0; i < 3; ++i) u.v[i][column] = choose(mask, -u.v[i][column], u.v[i][column]);
}
void swap_columns(Matrix& a, int i, int j, Mask mask) {
    for (int row = 0; row < 3; ++row) {
        const P x = a.v[row][i], y = a.v[row][j];
        a.v[row][i] = choose(mask, y, x);
        a.v[row][j] = choose(mask, x, y);
    }
}
void swap_values(P& a, P& b, Mask mask) {
    const P old = a;
    a = choose(mask, b, a);
    b = choose(mask, old, b);
}
void negate_columns(Matrix& u, Matrix& v, int column, Mask mask) {
    for (int row = 0; row < 3; ++row) {
        u.v[row][column] = choose(mask, -u.v[row][column], u.v[row][column]);
        v.v[row][column] = choose(mask, -v.v[row][column], v.v[row][column]);
    }
}
template <int t>
void sort(Matrix& u, P (&sigma)[3], Matrix& v, Mask mask) {
    if (!mask) return;
    if constexpr (t == 0) {
        const Mask ordered = mask & ge(abs(sigma[1]), abs(sigma[2]));
        const Mask negative = ordered & lt(sigma[1], P(0.0));
        flip(u, sigma, 1, negative); flip(u, sigma, 2, negative);
        mask &= ~ordered;
        const Mask fix = mask & lt(sigma[2], P(0.0));
        flip(u, sigma, 1, fix); flip(u, sigma, 2, fix);
        swap_values(sigma[1], sigma[2], mask);
        swap_columns(u, 1, 2, mask); swap_columns(v, 1, 2, mask);
        const Mask reverse = mask & gt(sigma[1], sigma[0]);
        swap_values(sigma[0], sigma[1], reverse);
        swap_columns(u, 0, 1, reverse); swap_columns(v, 0, 1, reverse);
        negate_columns(u, v, 2, mask & ~reverse);
    } else {
        const Mask ordered = mask & ge(abs(sigma[0]), sigma[1]);
        const Mask negative = ordered & lt(sigma[0], P(0.0));
        flip(u, sigma, 0, negative); flip(u, sigma, 2, negative);
        mask &= ~ordered;
        swap_values(sigma[0], sigma[1], mask);
        swap_columns(u, 0, 1, mask); swap_columns(v, 0, 1, mask);
        const Mask reverse = mask & lt(abs(sigma[1]), abs(sigma[2]));
        swap_values(sigma[1], sigma[2], reverse);
        swap_columns(u, 1, 2, reverse); swap_columns(v, 1, 2, reverse);
        negate_columns(u, v, 1, mask & ~reverse);
        const Mask fix = mask & lt(sigma[1], P(0.0));
        flip(u, sigma, 1, fix); flip(u, sigma, 2, fix);
    }
}

void polar_eight(const Mat33* inputs, Mat33* outputs, int count) {
    Matrix b, u = identity(), v = identity();
    alignas(64) double lanes[8];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 8; ++k) lanes[k] = inputs[std::min(k, count - 1)](i, j);
            b.v[i][j] = P(_mm512_load_pd(lanes));
        }
    Givens first(b.v[1][0], b.v[2][0]);
    first.row(b, 1, 2); first.column(u, 1, 2);
    zero_chase(b, u, v);
    P a1 = b.v[0][0], b1 = b.v[0][1], a2 = b.v[1][1], a3 = b.v[2][2], b2 = b.v[1][2];
    P g1 = a1 * b1, g2 = a2 * b2;
    const P tolerance = P(1024.0 * std::numeric_limits<double>::epsilon())
        * max(P(0.5) * sqrt(a1 * a1 + a2 * a2 + a3 * a3 + b1 * b1 + b2 * b2), P(1.0));
    for (;;) {
        const Mask active = gt(abs(b2), tolerance) & gt(abs(b1), tolerance)
            & gt(abs(a1), tolerance) & gt(abs(a2), tolerance) & gt(abs(a3), tolerance);
        if (!active) break;
        const P w1 = a2 * a2 + b1 * b1, w2 = a3 * a3 + b2 * b2;
        const P d = P(0.5) * (w1 - w2), bs = g2 * g2;
        const P divisor = choose(active, abs(d) + sqrt(d * d + bs), P(1.0));
        const P mu = w2 - copysign(bs / divisor, d);
        Givens r(a1 * a1 - mu, g1);
        r.column(b, 0, 1, active); r.column(v, 0, 1, active);
        zero_chase(b, u, v, active);
        a1 = b.v[0][0]; b1 = b.v[0][1]; a2 = b.v[1][1]; a3 = b.v[2][2]; b2 = b.v[1][2];
        g1 = a1 * b1; g2 = a2 * b2;
    }
    P sigma[3] = {P(0.0), P(0.0), P(0.0)};
    Mask remaining = all;
    Mask branch = remaining & le(abs(b2), tolerance);
    process<0>(b, u, sigma, v, branch); sort<0>(u, sigma, v, branch); remaining &= ~branch;
    branch = remaining & le(abs(b1), tolerance);
    process<1>(b, u, sigma, v, branch); sort<1>(u, sigma, v, branch); remaining &= ~branch;
    branch = remaining & le(abs(a2), tolerance);
    if (branch) {
        Givens r; r.unconventional(b.v[1][2], b.v[2][2]);
        r.row(b, 1, 2, branch); r.column(u, 1, 2, branch);
        process<0>(b, u, sigma, v, branch); sort<0>(u, sigma, v, branch);
    }
    remaining &= ~branch;
    branch = remaining & le(abs(a3), tolerance);
    if (branch) {
        Givens r1(b.v[1][1], b.v[1][2]);
        r1.column(b, 1, 2, branch); r1.column(v, 1, 2, branch);
        Givens r2(b.v[0][0], b.v[0][2]);
        r2.column(b, 0, 2, branch); r2.column(v, 0, 2, branch);
        process<0>(b, u, sigma, v, branch); sort<0>(u, sigma, v, branch);
    }
    remaining &= ~branch;
    branch = remaining & le(abs(a1), tolerance);
    if (branch) {
        Givens r1; r1.unconventional(b.v[0][1], b.v[1][1]);
        r1.row(b, 0, 1, branch); r1.column(u, 0, 1, branch);
        Givens r2; r2.unconventional(b.v[0][2], b.v[2][2]);
        r2.row(b, 0, 2, branch); r2.column(u, 0, 2, branch);
        process<1>(b, u, sigma, v, branch); sort<1>(u, sigma, v, branch);
    }
    // Match Eigen's packet rows and its scalar tail independently. The
    // reference uses explicit FMAs in packet rows despite contraction being
    // disabled for the surrounding scalar SVD.
    using Packet = typename Eigen::internal::find_best_packet<double, 3>::type;
    constexpr int packet_rows = Eigen::internal::unpacket_traits<Packet>::size;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const P a0 = u.v[row][0], a1 = u.v[row][1], a2 = u.v[row][2];
            const P b0 = v.v[column][0], b1 = v.v[column][1], b2 = v.v[column][2];
            P result;
            if constexpr (packet_rows > 1) {
                if (row < 3 / packet_rows * packet_rows) {
                    const P first = a0 * b0;
                    result = P(_mm512_fmadd_pd(a2.v, b2.v,
                        _mm512_fmadd_pd(a1.v, b1.v, first.v)));
                } else result = a0 * b0 + (a1 * b1 + a2 * b2);
            } else result = a0 * b0 + (a1 * b1 + a2 * b2);
            _mm512_store_pd(lanes, result.v);
            for (int lane = 0; lane < count; ++lane) outputs[lane](row, column) = lanes[lane];
        }
    }
}
#endif
} // namespace

void batched_signed_polar(const Mat33* inputs, Mat33* rotations, std::size_t count) {
    for (std::size_t first = 0; first < count; first += 8) {
        const int lanes = static_cast<int>(std::min<std::size_t>(8, count - first));
#if defined(__AVX512F__)
        bool ordinary = lanes >= 4;
        for (int i = 0; i < lanes; ++i) {
            const double scale = inputs[first + i].cwiseAbs().maxCoeff();
            ordinary = ordinary && inputs[first + i].allFinite()
                && scale <= 1e20 && (scale == 0.0 || scale >= 1e-20);
        }
        if (ordinary) { polar_eight(inputs + first, rotations + first, lanes); continue; }
#endif
        for (int i = 0; i < lanes; ++i) scalar_polar(inputs[first + i], rotations[first + i]);
    }
}
} // namespace volumetric_detail
