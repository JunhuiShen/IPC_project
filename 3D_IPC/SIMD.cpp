#include "SIMD.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace ipc_simd {
namespace {

// Explicit double-precision instructions ensure vectorization across elements,
// independently of Eigen's vectorization within an individual small matrix.
struct Pack {
#if defined(__AVX2__)
    using Native = __m256d;
    static constexpr int width = 4;
#elif defined(__aarch64__) || defined(_M_ARM64)
    using Native = float64x2_t;
    static constexpr int width = 2;
#elif defined(__SSE2__) || defined(_M_X64)
    using Native = __m128d;
    static constexpr int width = 2;
#else
    using Native = double;
    static constexpr int width = 1;
#endif
    Native value;

    Pack() = default;
    explicit Pack(double x) {
#if defined(__AVX2__)
        value = _mm256_set1_pd(x);
#elif defined(__aarch64__) || defined(_M_ARM64)
        value = vdupq_n_f64(x);
#elif defined(__SSE2__) || defined(_M_X64)
        value = _mm_set1_pd(x);
#else
        value = x;
#endif
    }
    static Pack raw(Native x) { Pack p; p.value = x; return p; }
    static Pack load(const double* x) {
#if defined(__AVX2__)
        return raw(_mm256_loadu_pd(x));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vld1q_f64(x));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_loadu_pd(x));
#else
        return raw(*x);
#endif
    }
#if defined(__AVX2__)
    template <typename Get>
    static Pack gather(const Get& get) {
        return raw(_mm256_setr_pd(get(0), get(1), get(2), get(3)));
    }
#endif
    void store(double* x) const {
#if defined(__AVX2__)
        _mm256_storeu_pd(x, value);
#elif defined(__aarch64__) || defined(_M_ARM64)
        vst1q_f64(x, value);
#elif defined(__SSE2__) || defined(_M_X64)
        _mm_storeu_pd(x, value);
#else
        *x = value;
#endif
    }
    friend Pack operator+(Pack a, Pack b) {
#if defined(__AVX2__)
        return raw(_mm256_add_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vaddq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_add_pd(a.value, b.value));
#else
        return raw(a.value + b.value);
#endif
    }
    friend Pack operator-(Pack a, Pack b) {
#if defined(__AVX2__)
        return raw(_mm256_sub_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vsubq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_sub_pd(a.value, b.value));
#else
        return raw(a.value - b.value);
#endif
    }
    friend Pack operator*(Pack a, Pack b) {
#if defined(__AVX2__)
        return raw(_mm256_mul_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vmulq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_mul_pd(a.value, b.value));
#else
        return raw(a.value * b.value);
#endif
    }
    friend Pack operator/(Pack a, Pack b) {
#if defined(__AVX2__)
        return raw(_mm256_div_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vdivq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_div_pd(a.value, b.value));
#else
        return raw(a.value / b.value);
#endif
    }
    friend Pack sqrt(Pack a) {
#if defined(__AVX2__)
        return raw(_mm256_sqrt_pd(a.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vsqrtq_f64(a.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_sqrt_pd(a.value));
#else
        return raw(std::sqrt(a.value));
#endif
    }
    // Comparison masks are represented by all-one bits in each selected lane.
    friend Pack greater(Pack a, Pack b) {
#if defined(__AVX2__)
        return raw(_mm256_cmp_pd(a.value, b.value, _CMP_GT_OQ));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vreinterpretq_f64_u64(vcgtq_f64(a.value, b.value)));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_cmpgt_pd(a.value, b.value));
#else
        return Pack(a.value > b.value ? 1.0 : 0.0);
#endif
    }
    friend Pack not_less_equal(Pack a, Pack b) {
#if defined(__AVX2__)
        return raw(_mm256_cmp_pd(a.value, b.value, _CMP_NLE_UQ));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_u64(vcleq_f64(a.value, b.value)))));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_cmpnle_pd(a.value, b.value));
#else
        return Pack(!(a.value <= b.value) ? 1.0 : 0.0);
#endif
    }
    friend Pack equal(Pack a, Pack b) {
#if defined(__AVX2__)
        return raw(_mm256_cmp_pd(a.value, b.value, _CMP_EQ_OQ));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vreinterpretq_f64_u64(vceqq_f64(a.value, b.value)));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_cmpeq_pd(a.value, b.value));
#else
        return Pack(a.value == b.value ? 1.0 : 0.0);
#endif
    }
    friend Pack mask_and(Pack a, Pack b) {
#if defined(__AVX2__)
        return raw(_mm256_and_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a.value), vreinterpretq_u64_f64(b.value))));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_and_pd(a.value, b.value));
#else
        return Pack(a.value != 0.0 && b.value != 0.0 ? 1.0 : 0.0);
#endif
    }
    friend Pack select(Pack mask, Pack yes, Pack no) {
#if defined(__AVX2__)
        return raw(_mm256_or_pd(_mm256_and_pd(mask.value, yes.value), _mm256_andnot_pd(mask.value, no.value)));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vbslq_f64(vreinterpretq_u64_f64(mask.value), yes.value, no.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_or_pd(_mm_and_pd(mask.value, yes.value), _mm_andnot_pd(mask.value, no.value)));
#else
        return mask.value != 0.0 ? yes : no;
#endif
    }
    friend Pack abs(Pack a) {
#if defined(__AVX2__)
        return raw(_mm256_andnot_pd(_mm256_set1_pd(-0.0), a.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vabsq_f64(a.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_andnot_pd(_mm_set1_pd(-0.0), a.value));
#else
        return Pack(std::abs(a.value));
#endif
    }
    friend Pack copy_sign(Pack magnitude, Pack sign) {
#if defined(__AVX2__)
        const auto signbit = _mm256_set1_pd(-0.0);
        return raw(_mm256_or_pd(_mm256_andnot_pd(signbit, magnitude.value), _mm256_and_pd(signbit, sign.value)));
#elif defined(__aarch64__) || defined(_M_ARM64)
        const auto signbit = vreinterpretq_u64_f64(vdupq_n_f64(-0.0));
        return raw(vbslq_f64(signbit, sign.value, magnitude.value));
#elif defined(__SSE2__) || defined(_M_X64)
        const auto signbit = _mm_set1_pd(-0.0);
        return raw(_mm_or_pd(_mm_andnot_pd(signbit, magnitude.value), _mm_and_pd(signbit, sign.value)));
#else
        return Pack(std::copysign(magnitude.value, sign.value));
#endif
    }
};

constexpr int W = Pack::width;

// Atan polynomial coefficients and split pi constants are from fdlibm:
// https://www.netlib.org/fdlibm/s_atan.c and e_atan2.c.
// The ratio reduction and SIMD masks below replace its scalar interval tree.
/*
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this software is freely
 * granted, provided that this notice is preserved.
 */
Pack packed_atan2(Pack y, Pack x) {
    const Pack zero(0.0), one(1.0);
    const Pack ax = abs(x), ay = abs(y);
    const Pack swap = greater(ay, ax);
    const Pack numerator = select(swap, ax, ay);
    const Pack denominator = select(swap, ay, ax);
    const Pack both_inf = equal(numerator, Pack(std::numeric_limits<double>::infinity()));
    // No overflow in the ratio. Avoid 0/0 and infinity/infinity before blend.
    const Pack r = select(both_inf, one, numerator)
        / select(both_inf, one, select(greater(denominator, zero), denominator, one));
    const Pack fold = greater(r, Pack(4.14213562373095048802e-01));
    const Pack t = select(fold, (r - one) / (r + one), r);
    const Pack z = t * t, w = z * z;
    const Pack odd = z * (Pack(3.33333333333329318027e-01)
        + w * (Pack(1.42857142725034663711e-01)
        + w * (Pack(9.09088713343650656196e-02)
        + w * (Pack(6.66107313738753120669e-02)
        + w * (Pack(4.97687799461593236017e-02)
        + w * Pack(1.62858201153657823623e-02))))));
    const Pack even = w * (Pack(-1.99999999998764832476e-01)
        + w * (Pack(-1.11111104054623557880e-01)
        + w * (Pack(-7.69187620504482999495e-02)
        + w * (Pack(-5.83357013379057348645e-02)
        + w * Pack(-3.65315727442169155270e-02)))));
    const Pack correction = t * (odd + even);
    Pack angle = select(fold,
        Pack(7.85398163397448278999e-01) - ((correction - Pack(3.06161699786838301793e-17)) - t),
        t - correction);
    angle = select(swap, Pack(1.57079632679489655800e+00)
        - (angle - Pack(6.12323399573676603587e-17)), angle);
    const Pack negative_x = greater(zero, copy_sign(one, x));
    angle = select(negative_x, Pack(3.14159265358979311600e+00)
        - (angle - Pack(1.22464679914735317720e-16)), angle);
    angle = copy_sign(angle, y);
    const Pack ordered = mask_and(equal(x, x), equal(y, y));
    return select(ordered, angle, x + y);
}

using PackedVec3 = std::array<Pack, 3>;
Pack dot(const PackedVec3& a, const PackedVec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
PackedVec3 cross(const PackedVec3& a, const PackedVec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

void accumulate_scalar(
    const RefMesh& mesh, const std::vector<Vec3>& x,
    int triangle, int role, const std::vector<ShapeGrads>* rest_shape_grads,
    double mu, double lambda, double dt2, Vec3& g, Mat33& H) {
    const int* v = &mesh.tris[3 * triangle];
    Mat32 Ds;
    Ds.col(0) = x[v[1]] - x[v[0]];
    Ds.col(1) = x[v[2]] - x[v[0]];
    const Mat32 F = Ds * mesh.Dm_inverse[triangle];
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const ShapeGrads gradN = rest_shape_grads
        ? (*rest_shape_grads)[triangle]
        : shape_function_gradients(mesh.Dm_inverse[triangle]);
    const Mat32 P = PCorotated32(cache, F, mu, lambda);
    Mat66 dPdF;
    dPdFCorotated32(cache, mu, lambda, dPdF);
    g += dt2 * corotated_node_gradient(P, mesh.area[triangle], gradN, role);
    H += dt2 * corotated_node_hessian(dPdF, mesh.area[triangle], gradN, role);
}

} // namespace

int lane_width() { return W; }

const char* backend_name() {
#if defined(__AVX2__)
    return "AVX2 (4 doubles)";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "NEON (2 doubles)";
#elif defined(__SSE2__) || defined(_M_X64)
    return "SSE2 (2 doubles)";
#else
    return "scalar (1 double)";
#endif
}

void atan2_batch(const double* y, const double* x, double* angles, std::size_t count) {
    for (std::size_t begin = 0; begin < count; begin += W) {
        const int active = static_cast<int>(std::min<std::size_t>(W, count - begin));
        if (active == W) {
            packed_atan2(Pack::load(y + begin), Pack::load(x + begin)).store(angles + begin);
        } else {
            alignas(32) double local_x[W], local_y[W], output[W];
            for (int lane = 0; lane < W; ++lane) {
                local_x[lane] = lane < active ? x[begin + lane] : 1.0;
                local_y[lane] = lane < active ? y[begin + lane] : 0.0;
            }
            packed_atan2(Pack::load(local_y), Pack::load(local_x)).store(output);
            for (int lane = 0; lane < active; ++lane) angles[begin + lane] = output[lane];
        }
    }
}

void accumulate_point_terms(
    double mass, const Vec3& x, const Vec3& xhat, const Vec3& gravity,
    const Vec3* pin_target, double kpin, double dt2,
    Vec3& gradient, Mat33& hessian) {
    const Pack m(mass), minus_m(-mass), timestep2(dt2), pin_scale(dt2 * kpin);
    for (int begin = 0; begin < 3; begin += W) {
        const int count = std::min(W, 3 - begin);
        alignas(32) double x_data[W] = {}, xhat_data[W] = {}, gravity_data[W] = {};
        alignas(32) double pin_data[W] = {}, g_data[W] = {}, H_data[W] = {};
        for (int lane = 0; lane < count; ++lane) {
            const int axis = begin + lane;
            x_data[lane] = x[axis]; xhat_data[lane] = xhat[axis];
            gravity_data[lane] = gravity[axis];
            if (pin_target) pin_data[lane] = (*pin_target)[axis];
            g_data[lane] = gradient[axis]; H_data[lane] = hessian(axis, axis);
        }
        const Pack current = Pack::load(x_data);
        Pack g = Pack::load(g_data) + m * (current - Pack::load(xhat_data));
        g = g + timestep2 * (minus_m * Pack::load(gravity_data));
        Pack H = Pack::load(H_data) + m;
        if (pin_target) {
            g = g + pin_scale * (current - Pack::load(pin_data));
            H = H + pin_scale;
        }
        g.store(g_data); H.store(H_data);
        for (int lane = 0; lane < count; ++lane) {
            const int axis = begin + lane;
            gradient[axis] = g_data[lane]; hessian(axis, axis) = H_data[lane];
        }
    }
}

void accumulate_bending(
    const RefMesh& mesh, const std::vector<Vec3>& positions,
    const std::vector<std::pair<int, int>>& incident,
    double kB, double dt2, Vec3& gradient, Mat33& hessian) {
    const Pack zero(0.0), one(1.0), twice_kB(2.0 * kB);
    for (std::size_t begin = 0; begin < incident.size(); begin += W) {
        const int count = static_cast<int>(std::min<std::size_t>(W, incident.size() - begin));
        alignas(32) double points[4][3][W], ce_data[W], rest_data[W], role_data[W];
        for (int lane = 0; lane < W; ++lane) {
            const auto [hi, role] = incident[begin + std::min(lane, count - 1)];
            const Hinge& hinge = mesh.hinges[hi];
            for (int vertex = 0; vertex < 4; ++vertex)
                for (int axis = 0; axis < 3; ++axis)
                    points[vertex][axis][lane] = positions[hinge.v[vertex]][axis];
            ce_data[lane] = hinge.c_e; rest_data[lane] = hinge.bar_theta;
            role_data[lane] = static_cast<double>(role);
        }
        const Pack role = Pack::load(role_data);
        const Pack role0 = equal(role, zero), role2 = equal(role, Pack(2.0));
        const Pack role3 = equal(role, Pack(3.0));
        PackedVec3 e, a, b, cA, cB;
        for (int axis = 0; axis < 3; ++axis) {
            const Pack p0 = Pack::load(points[0][axis]), p1 = Pack::load(points[1][axis]);
            const Pack p2 = Pack::load(points[2][axis]), p3 = Pack::load(points[3][axis]);
            e[axis] = p1 - p0; a[axis] = p2 - p0; b[axis] = p3 - p0;
            cA[axis] = p2 - p1; cB[axis] = p3 - p1;
        }
        const PackedVec3 mA = cross(e, a), mB = cross(b, e);
        const Pack muA2 = dot(mA, mA), muB2 = dot(mB, mB), ell = sqrt(dot(e, e));
        // Negate the reference's <= 0 checks, including its NaN behavior;
        // nonfinite geometry is not silently converted into zero force.
        const Pack valid = mask_and(not_less_equal(ell, zero),
            mask_and(not_less_equal(muA2, zero), not_less_equal(muB2, zero)));
        const Pack safe_ell = select(valid, ell, one);
        PackedVec3 ehat;
        for (int axis = 0; axis < 3; ++axis) ehat[axis] = e[axis] / safe_ell;
        const Pack X = dot(mA, mB), Y = dot(cross(mA, mB), ehat);
        const Pack theta = packed_atan2(select(valid, Y, zero), select(valid, X, one));

        // For edge endpoints, write the existing chain rule using u/v:
        // role 0: u=x2-x1, v=x3-x1; role 1: u=-a, v=-b.
        // Apex derivatives retain their direct formulas to avoid extra
        // cancellation from dot products that are theoretically zero.
        PackedVec3 u, v;
        for (int axis = 0; axis < 3; ++axis) {
            u[axis] = select(role0, cA[axis], zero - a[axis]);
            v[axis] = select(role0, cB[axis], zero - b[axis]);
        }
        const PackedVec3 mB_cross_u = cross(mB, u), mA_cross_v = cross(mA, v);
        const PackedVec3 role2_dX = cross(mB, e), role3_dX = cross(e, mA);
        const Pack edge_coefficient = dot(u, mB) + dot(mA, v);
        const Pack ehat_u = dot(ehat, u), ehat_v = dot(ehat, v);
        const Pack denominator = select(valid, muA2 * muB2, one);
        const Pack scale = twice_kB * Pack::load(ce_data);
        const Pack g_scale = scale * (theta - Pack::load(rest_data));
        PackedVec3 gtheta;
        alignas(32) double g_data[3][W], H_data[3][3][W];
        for (int axis = 0; axis < 3; ++axis) {
            Pack dX = mB_cross_u[axis] - mA_cross_v[axis];
            Pack dY = edge_coefficient * ehat[axis] - ehat_u * mB[axis] - ehat_v * mA[axis];
            dX = select(role2, role2_dX[axis], select(role3, role3_dX[axis], dX));
            dY = select(role2, (zero - ell) * mB[axis], select(role3, (zero - ell) * mA[axis], dY));
            gtheta[axis] = select(valid, (X * dY - Y * dX) / denominator, zero);
            select(valid, g_scale * gtheta[axis], zero).store(g_data[axis]);
        }
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                select(valid, scale * (gtheta[row] * gtheta[col]), zero).store(H_data[row][col]);
        for (int lane = 0; lane < count; ++lane)
            for (int row = 0; row < 3; ++row) {
                gradient[row] += dt2 * g_data[row][lane];
                for (int col = 0; col < 3; ++col)
                    hessian(row, col) += dt2 * H_data[row][col][lane];
            }
    }
}

void accumulated_corotated_elasticity(
    const RefMesh& mesh, const std::vector<Vec3>& positions,
    const IncidentTriangles& incident,
    const std::vector<ShapeGrads>* rest_shape_grads,
    double mu, double lambda, double dt2, Vec3& gradient, Mat33& hessian) {
    const Pack zero(0.0), one(1.0), two(2.0);
    const Pack twice_mu(2.0 * mu), bulk(lambda);
    for (std::size_t begin = 0; begin < incident.size(); begin += W) {
        const int count = static_cast<int>(std::min<std::size_t>(W, incident.size() - begin));

#if defined(__AVX2__)
        // Gather directly into SIMD registers; preserve original tail inputs.
        std::array<int, W> triangles, roles;
        std::array<const Mat22*, W> dm;
        std::array<std::array<const Vec3*, W>, 3> points;
        for (int lane = 0; lane < W; ++lane) {
            const auto [triangle, role] = incident[begin + std::min(lane, count - 1)];
            triangles[lane] = triangle;
            roles[lane] = role;
            dm[lane] = &mesh.Dm_inverse[triangle];
            const int* v = &mesh.tris[3 * triangle];
            for (int vertex = 0; vertex < 3; ++vertex)
                points[vertex][lane] = &positions[v[vertex]];
        }
        Pack F[3][2];
        const Pack dm00 = Pack::gather([&](int lane) { return (*dm[lane])(0, 0); });
        const Pack dm01 = Pack::gather([&](int lane) { return (*dm[lane])(0, 1); });
        const Pack dm10 = Pack::gather([&](int lane) { return (*dm[lane])(1, 0); });
        const Pack dm11 = Pack::gather([&](int lane) { return (*dm[lane])(1, 1); });
        for (int row = 0; row < 3; ++row) {
            const Pack e0 = Pack::gather([&](int lane) { return (*points[1][lane])[row] - (*points[0][lane])[row]; });
            const Pack e1 = Pack::gather([&](int lane) { return (*points[2][lane])[row] - (*points[0][lane])[row]; });
            F[row][0] = e0 * dm00 + e1 * dm10;
            F[row][1] = e0 * dm01 + e1 * dm11;
        }
#else
        // Gather a small AoSoA packet. Tail lanes repeat the last real triangle
        // and are never scattered, so they cannot create singular dummy input.
        alignas(32) double edges[3][2][W], dm[2][2][W], b_data[2][W], area_data[W];
        for (int lane = 0; lane < W; ++lane) {
            const auto [triangle, role] = incident[begin + std::min(lane, count - 1)];
            const int* v = &mesh.tris[3 * triangle];
            const Mat22& Dm_inv = mesh.Dm_inverse[triangle];
            for (int row = 0; row < 3; ++row) {
                edges[row][0][lane] = positions[v[1]][row] - positions[v[0]][row];
                edges[row][1][lane] = positions[v[2]][row] - positions[v[0]][row];
            }
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 2; ++col)
                    dm[row][col][lane] = Dm_inv(row, col);
            for (int col = 0; col < 2; ++col) {
                b_data[col][lane] = rest_shape_grads
                    ? (*rest_shape_grads)[triangle][role][col]
                    : (role == 0 ? -Dm_inv(0, col) - Dm_inv(1, col) : Dm_inv(role - 1, col));
            }
            area_data[lane] = mesh.area[triangle];
        }

        Pack F[3][2];
        const Pack dm00 = Pack::load(dm[0][0]), dm01 = Pack::load(dm[0][1]);
        const Pack dm10 = Pack::load(dm[1][0]), dm11 = Pack::load(dm[1][1]);
        for (int row = 0; row < 3; ++row) {
            const Pack e0 = Pack::load(edges[row][0]), e1 = Pack::load(edges[row][1]);
            F[row][0] = e0 * dm00 + e1 * dm10;
            F[row][1] = e0 * dm01 + e1 * dm11;
        }
#endif
        Pack c00 = F[0][0] * F[0][0] + F[1][0] * F[1][0] + F[2][0] * F[2][0];
        Pack c01 = F[0][0] * F[0][1] + F[1][0] * F[1][1] + F[2][0] * F[2][1];
        Pack c11 = F[0][1] * F[0][1] + F[1][1] * F[1][1] + F[2][1] * F[2][1];
        Pack det = c00 * c11 - c01 * c01;
        Pack trace = c00 + c11;

        // The analytic square root applies to the positive-definite, unclamped
        // branch. Preserve Eigen's 1e-12 eigenvalue clamp and its behavior on
        // ill-conditioned inputs by evaluating those individual lanes exactly
        // through the established scalar kernel. det/trace bounds lambda_min;
        // det/trace^2 > 1e-6 also limits determinant cancellation to the tested
        // regime (roughly condition(C) < 1e6).
        alignas(32) double det_data[W], trace_data[W];
        det.store(det_data);
        trace.store(trace_data);
        bool fallback[W];
        bool any_fallback = false;
        int fallback_count = 0;
        for (int lane = 0; lane < W; ++lane) {
            fallback[lane] = !std::isfinite(det_data[lane])
                || !std::isfinite(trace_data[lane])
                || !(det_data[lane] > 1e-10 * trace_data[lane])
                || !(det_data[lane] > 1e-6 * trace_data[lane] * trace_data[lane]);
            any_fallback = any_fallback || fallback[lane];
            if (lane < count && fallback[lane]) ++fallback_count;
        }
        if (fallback_count == count) {
            for (int lane = 0; lane < count; ++lane) {
                const auto [triangle, role] = incident[begin + lane];
                accumulate_scalar(mesh, positions, triangle, role, rest_shape_grads,
                                  mu, lambda, dt2, gradient, hessian);
            }
            continue;
        }
        if (any_fallback) {
            // Substitute benign matrices before SIMD division/sqrt. These
            // lanes' outputs are ignored in favor of scalar results below.
            alignas(32) double values[W];
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 2; ++col) {
                    F[row][col].store(values);
                    for (int lane = 0; lane < W; ++lane)
                        if (fallback[lane]) values[lane] = row == col ? 1.0 : 0.0;
                    F[row][col] = Pack::load(values);
                }
            const auto replace = [&](Pack p, double benign) {
                p.store(values);
                for (int lane = 0; lane < W; ++lane)
                    if (fallback[lane]) values[lane] = benign;
                return Pack::load(values);
            };
            c00 = replace(c00, 1.0);
            c01 = replace(c01, 0.0);
            c11 = replace(c11, 1.0);
            det = replace(det, 1.0);
            trace = replace(trace, 2.0);
        }

        // For C=F^T F, sqrt(C)=(C+sqrt(det(C))*I)/sqrt(tr(C)+2sqrt(det(C))).
        const Pack J = sqrt(det);
        const Pack traceS = sqrt(trace + two * J);
        const Pack inv_s_denominator = one / (J * traceS);
        const Pack s00 = (c11 + J) * inv_s_denominator;
        const Pack s01 = (zero - c01) * inv_s_denominator;
        const Pack s11 = (c00 + J) * inv_s_denominator;
        const Pack inv_det = one / det;
        const Pack ci00 = c11 * inv_det, ci01 = (zero - c01) * inv_det, ci11 = c00 * inv_det;
#if defined(__AVX2__)
        const auto gather_b = [&](int col) {
            return Pack::gather([&](int lane) {
                const int role = roles[lane];
                const Mat22& Dm_inv = *dm[lane];
                return rest_shape_grads
                    ? (*rest_shape_grads)[triangles[lane]][role][col]
                    : (role == 0 ? -Dm_inv(0, col) - Dm_inv(1, col) : Dm_inv(role - 1, col));
            });
        };
        const Pack b0 = gather_b(0), b1 = gather_b(1);
#else
        const Pack b0 = Pack::load(b_data[0]), b1 = Pack::load(b_data[1]);
#endif
        const Pack b_norm2 = b0 * b0 + b1 * b1;
        const Pack b_s_b = b0 * (s00 * b0 + s01 * b1) + b1 * (s01 * b0 + s11 * b1);
        const Pack b_c_b = b0 * (ci00 * b0 + ci01 * b1) + b1 * (ci01 * b0 + ci11 * b1);
        const Pack volumetric = bulk * (J - one) * J;
        const Pack lambda_J2 = bulk * J * J;
        const Pack inv_traceS = one / traceS;
#if defined(__AVX2__)
        const Pack area = Pack::gather([&](int lane) { return mesh.area[triangles[lane]]; });
#else
        const Pack area = Pack::load(area_data);
#endif
        Pack R[3][2], B[3][2], Bb[3], Reb[3];
        alignas(32) double g_data[3][W], H_data[3][3][W];
        for (int row = 0; row < 3; ++row) {
            R[row][0] = F[row][0] * s00 + F[row][1] * s01;
            R[row][1] = F[row][0] * s01 + F[row][1] * s11;
            B[row][0] = F[row][0] * ci00 + F[row][1] * ci01;
            B[row][1] = F[row][0] * ci01 + F[row][1] * ci11;
            Bb[row] = B[row][0] * b0 + B[row][1] * b1;
            Reb[row] = R[row][1] * b0 - R[row][0] * b1;
            const Pack p0 = twice_mu * (F[row][0] - R[row][0]) + volumetric * B[row][0];
            const Pack p1 = twice_mu * (F[row][1] - R[row][1]) + volumetric * B[row][1];
            (area * (p0 * b0 + p1 * b1)).store(g_data[row]);
        }

        // Contract the existing 6x6 dP/dF analytically into the node's 3x3
        // diagonal block. With b=grad(N), B=F C^-1 and Q=B F^T:
        // H/A = 2mu[|b|^2 I-(I-RR^T)(b^T S^-1 b)
        //                  -(Re b)(Re b)^T/tr(S)]
        //       +lambda(J-1)J(I-Q)(b^T C^-1 b)+lambda J^2(Bb)(Bb)^T.
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                const Pack identity = row == col ? one : zero;
                const Pack RRT = R[row][0] * R[col][0] + R[row][1] * R[col][1];
                const Pack Q = B[row][0] * F[col][0] + B[row][1] * F[col][1];
                const Pack dR = (identity - RRT) * b_s_b + Reb[row] * Reb[col] * inv_traceS;
                const Pack H = twice_mu * (identity * b_norm2 - dR)
                    + volumetric * (identity - Q) * b_c_b + lambda_J2 * Bb[row] * Bb[col];
                (area * H).store(H_data[row][col]);
            }
        }

        // Scatter only into this active vertex, in exactly the original
        // incident order. No shared element output or atomic addition is used.
        for (int lane = 0; lane < count; ++lane) {
            if (fallback[lane]) {
                const auto [triangle, role] = incident[begin + lane];
                accumulate_scalar(mesh, positions, triangle, role, rest_shape_grads,
                                  mu, lambda, dt2, gradient, hessian);
            } else {
                for (int row = 0; row < 3; ++row) {
                    gradient[row] += dt2 * g_data[row][lane];
                    for (int col = 0; col < 3; ++col)
                        hessian(row, col) += dt2 * H_data[row][col][lane];
                }
            }
        }
    }
}

// V2 kernels retain the scalar spectral/angle evaluation and arithmetic order.
// Gathered inputs are transposed locally; independent element lanes evaluate
// derivatives and return AoS contributions for the caller's ordered reduction.
#if defined(__AVX512F__)
struct ElementPack {
    using Native = __m512d;
    static constexpr int width = 8;
    Native value;
    ElementPack() = default;
    explicit ElementPack(double x) : value(_mm512_set1_pd(x)) {}
    static ElementPack raw(Native x) { ElementPack p; p.value=x; return p; }
    static ElementPack load(const double* x) { return raw(_mm512_loadu_pd(x)); }
    void store(double* x) const { _mm512_storeu_pd(x,value); }
    friend ElementPack operator+(ElementPack a, ElementPack b) { return raw(_mm512_add_pd(a.value,b.value)); }
    friend ElementPack operator-(ElementPack a, ElementPack b) { return raw(_mm512_sub_pd(a.value,b.value)); }
    friend ElementPack operator*(ElementPack a, ElementPack b) { return raw(_mm512_mul_pd(a.value,b.value)); }
    friend ElementPack operator/(ElementPack a, ElementPack b) { return raw(_mm512_div_pd(a.value,b.value)); }
    friend ElementPack equal(ElementPack a, ElementPack b) {
        return raw(_mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(a.value,b.value,_CMP_EQ_OQ),-1)));
    }
    friend ElementPack select(ElementPack mask, ElementPack yes, ElementPack no) {
        const auto bits=_mm512_cmp_epi64_mask(_mm512_castpd_si512(mask.value),_mm512_setzero_si512(),_MM_CMPINT_NE);
        return raw(_mm512_mask_blend_pd(bits,no.value,yes.value));
    }
};
#else
using ElementPack = Pack;
#endif

const char* tile_backend_name() {
#if defined(__AVX512F__)
    return "AVX-512 (8 doubles)";
#else
    return backend_name();
#endif
}

// Match the scalar kernel's fused operations where hardware supports them.
// The operand order matters: algebraic reassociation changes long trajectories.
static ElementPack element_multiply_add(ElementPack a, ElementPack b, ElementPack c) {
#if defined(__AVX512F__)
    return ElementPack::raw(_mm512_fmadd_pd(a.value, b.value, c.value));
#elif defined(__AVX2__) && defined(__FMA__)
    return ElementPack::raw(_mm256_fmadd_pd(a.value, b.value, c.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
    return ElementPack::raw(vfmaq_f64(c.value, a.value, b.value));
#else
    return a * b + c;
#endif
}

void corotated_derivatives_tile(
    const Vec3* positions, const Mat22* dm_inverse, const double* areas,
    const Vec2* shape_gradients, std::size_t entry_count,
    double mu, double lambda, Vec3* gradients, Mat33* hessians) {
    constexpr int W = ElementPack::width;
    assert(entry_count <= tile_width);
    const ElementPack zero(0.0), one(1.0), twice_mu(2.0 * mu), bulk(lambda);
    for (std::size_t begin = 0; begin < entry_count; begin += W) {
        const int count = static_cast<int>(std::min<std::size_t>(W, entry_count - begin));
        alignas(64) double s[2][2][W], ci[2][2][W], r[3][2][W], b[3][2][W];
        alignas(64) double f_data[3][2][W], q[2][W], area[W], jd[W], tr[W];
        alignas(64) double p[3][2][W], gd[3][W], hd[3][3][W];
        // Layout conversion uses only the already-gathered AoS tile.
        alignas(64) double points[3][3][W], material[2][2][W];
        for (int lane = 0; lane < W; ++lane) {
            const std::size_t entry = begin + std::min(lane, count - 1);
            for (int vertex = 0; vertex < 3; ++vertex)
                for (int axis = 0; axis < 3; ++axis)
                    points[vertex][axis][lane] = positions[3*entry+vertex][axis];
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 2; ++col)
                    material[row][col][lane] = dm_inverse[entry](row,col);
        }
        // Preserve the scalar eigensolver and its clamping decisions. Only
        // local tile data are read; the derivative contractions below use SIMD.
        for (int lane = 0; lane < W; ++lane) {
            const std::size_t entry = begin + std::min(lane, count - 1);
            Mat32 ds;
            Mat22 dm;
            for (int axis = 0; axis < 3; ++axis) {
                ds(axis,0) = points[1][axis][lane] - points[0][axis][lane];
                ds(axis,1) = points[2][axis][lane] - points[0][axis][lane];
            }
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 2; ++col) dm(row,col) = material[row][col][lane];
            const Mat32 f = ds * dm;
            const CorotatedCache32 cache = buildCorotatedCache(f);
            const Mat32 stress = PCorotated32(cache, f, mu, lambda);
            for (int i = 0; i < 2; ++i) {
                q[i][lane] = shape_gradients[entry][i];
                for (int j = 0; j < 2; ++j) {
                    s[i][j][lane] = cache.SInv(i,j);
                    ci[i][j][lane] = cache.FTFinv(i,j);
                }
            }
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 2; ++j) {
                    r[i][j][lane] = cache.R(i,j);
                    b[i][j][lane] = cache.FFTFInv(i,j);
                    p[i][j][lane] = stress(i,j);
                    f_data[i][j][lane] = f(i,j);
                }
            }
            area[lane] = areas[entry];
            jd[lane] = cache.J;
            tr[lane] = cache.traceS;
        }
        const ElementPack J = ElementPack::load(jd), trace = ElementPack::load(tr), A = ElementPack::load(area);
        const ElementPack volumetric = bulk * (J - one) * J;
        const ElementPack positive = ElementPack(0.5 * lambda) * (ElementPack(2.0) * J - one) * J;
        const ElementPack shape[2] = {ElementPack::load(q[0]), ElementPack::load(q[1])};
        // Retain the scalar beta/eta contraction order for each self block.
        #pragma GCC unroll 3
        for (int i = 0; i < 3; ++i) {
            ElementPack g = zero;
            for (int beta = 0; beta < 2; ++beta)
                g = g + ElementPack::load(p[i][beta]) * shape[beta];
            (A * g).store(gd[i]);
            #pragma GCC unroll 3
            for (int j = 0; j < 3; ++j) {
                const ElementPack RRT = element_multiply_add(ElementPack::load(r[i][1]), ElementPack::load(r[j][1]),
                    ElementPack::load(r[i][0]) * ElementPack::load(r[j][0]));
                const ElementPack Q = element_multiply_add(ElementPack::load(b[i][1]), ElementPack::load(f_data[j][1]),
                    ElementPack::load(b[i][0]) * ElementPack::load(f_data[j][0]));
                ElementPack sum = zero;
                #pragma GCC unroll 2
                for (int beta = 0; beta < 2; ++beta) {
                    #pragma GCC unroll 2
                    for (int eta = 0; eta < 2; ++eta) {
                        const ElementPack sinv = ElementPack::load(s[eta][beta]);
                        const ElementPack cinv = ElementPack::load(ci[eta][beta]);
                        const ElementPack dcdF = (eta == 0 ? zero - ElementPack::load(r[j][1]) : ElementPack::load(r[j][0])) / trace;
                        const ElementPack re = beta == 0 ? ElementPack::load(r[i][1]) : zero - ElementPack::load(r[i][0]);
                        ElementPack dr = i == j ? sinv : zero;
                        dr = element_multiply_add(ElementPack(-1.0) * RRT, sinv, dr);
                        dr = element_multiply_add(ElementPack(-1.0) * dcdF, re, dr);
                        ElementPack dp = i == j ? volumetric * cinv : zero;
                        dp = element_multiply_add(ElementPack(-1.0) * volumetric, element_multiply_add(ElementPack::load(b[i][eta]), ElementPack::load(b[j][beta]), Q * cinv), dp);
                        const ElementPack product = ElementPack::load(b[j][eta]) * ElementPack::load(b[i][beta]);
                        dp = element_multiply_add(positive, product + product, dp);
                        dp = element_multiply_add(twice_mu, (i == j && beta == eta ? one : zero) - dr, dp);
                        sum = element_multiply_add(dp * shape[beta], shape[eta], sum);
                    }
                }
                (A * sum).store(hd[i][j]);
            }
        }
        for (int lane = 0; lane < count; ++lane)
            for (int i = 0; i < 3; ++i) {
                gradients[begin + lane][i] = gd[i][lane];
                for (int j = 0; j < 3; ++j)
                    hessians[begin + lane](i,j) = hd[i][j][lane];
            }
    }
}


void bending_derivatives_tile(
    const Vec3* positions, const int* active_nodes, const double* coefficients,
    const double* rest_angles, std::size_t entry_count, double kB,
    Vec3* gradients, Mat33* hessians) {
    assert(entry_count <= tile_width);
    constexpr int W = ElementPack::width;
    const ElementPack zero(0.0), one(1.0);
    for (std::size_t begin=0; begin<entry_count; begin+=W) {
        const int count=static_cast<int>(std::min<std::size_t>(W,entry_count-begin));
        alignas(64) double c[4][3][W], ca[3][W], cb[3][W], aa[3][W], bb[3][W];
        alignas(64) double X[W],Y[W],theta[W],ell[W],denominator[W],scale[W],roles[W],valid[W];
        alignas(64) double points[4][3][W];
        for(int lane=0;lane<W;++lane) {
            const auto e=begin+std::min(lane,count-1);
            for(int vertex=0;vertex<4;++vertex)
                for(int axis=0;axis<3;++axis) points[vertex][axis][lane]=positions[4*e+vertex][axis];
        }
        for(int lane=0;lane<W;++lane) {
            const auto e=begin+std::min(lane,count-1);
            assert(active_nodes[e] >= 0 && active_nodes[e] < 4);
            HingeDef def;
            for(int vertex=0;vertex<4;++vertex)
                for(int axis=0;axis<3;++axis) def.x[vertex][axis]=points[vertex][axis][lane];
            const auto cache=make_bending_cache(def);
            const Vec3 A=def.x[2]-def.x[1],B=def.x[3]-def.x[1];
            for(int i=0;i<3;++i) {
                c[0][i][lane]=cache.mA[i];c[1][i][lane]=cache.mB[i];
                c[2][i][lane]=cache.e_hat[i];c[3][i][lane]=cache.e[i];
                ca[i][lane]=A[i];cb[i][lane]=B[i];
                aa[i][lane]=cache.a[i];bb[i][lane]=cache.b[i];
            }
            X[lane]=cache.X;Y[lane]=cache.Y;theta[lane]=cache.theta-rest_angles[e];
            ell[lane]=cache.ell;denominator[lane]=cache.degenerate?1.0:cache.muA2*cache.muB2;
            scale[lane]=2.0*kB*coefficients[e];roles[lane]=active_nodes[e];valid[lane]=cache.degenerate?0.0:1.0;
        }
        std::array<ElementPack,3> mA,mB,ehat,e,A,B,a,b;
        for(int i=0;i<3;++i) {
            mA[i]=ElementPack::load(c[0][i]);mB[i]=ElementPack::load(c[1][i]);
            ehat[i]=ElementPack::load(c[2][i]);e[i]=ElementPack::load(c[3][i]);
            A[i]=ElementPack::load(ca[i]);B[i]=ElementPack::load(cb[i]);
            a[i]=ElementPack::load(aa[i]);b[i]=ElementPack::load(bb[i]);
        }
        const auto ordered_dot=[](const std::array<ElementPack,3>& a,const std::array<ElementPack,3>& b) {
            // Match Eigen's pair reduction before the fused third product.
            // Volatile prevents contraction across the pair's rounding boundary.
            volatile ElementPack::Native p0=(a[0]*b[0]).value,p1=(a[1]*b[1]).value;
            return element_multiply_add(a[2],b[2],ElementPack::raw(p0)+ElementPack::raw(p1));
        };
        const auto ordered_cross=[](const std::array<ElementPack,3>& a,const std::array<ElementPack,3>& b) {
            return std::array<ElementPack,3>{
                element_multiply_add(a[1],b[2],ElementPack(-1.0)*(a[2]*b[1])),
                element_multiply_add(a[2],b[0],ElementPack(-1.0)*(a[0]*b[2])),
                element_multiply_add(a[0],b[1],ElementPack(-1.0)*(a[1]*b[0]))};
        };
        const auto mbA=ordered_cross(mB,A),maB=ordered_cross(mA,B);
        const auto mab=ordered_cross(mA,b),mba=ordered_cross(mB,a);
        const auto mbe=ordered_cross(mB,e),ema=ordered_cross(e,mA);
        const ElementPack coef0=ordered_dot(A,mB)+ordered_dot(mA,B);
        const ElementPack coef1=ElementPack(-1.0)*(ordered_dot(a,mB)+ordered_dot(mA,b));
        const ElementPack ehA=ordered_dot(ehat,A),ehB=ordered_dot(ehat,B);
        const ElementPack eha=ordered_dot(ehat,a),ehb=ordered_dot(ehat,b);
        const ElementPack r=ElementPack::load(roles),r0=equal(r,zero),r1=equal(r,one),r2=equal(r,ElementPack(2.0));
        const ElementPack v=equal(ElementPack::load(valid),one),xx=ElementPack::load(X),yy=ElementPack::load(Y);
        const ElementPack ss=ElementPack::load(scale),gs=ss*ElementPack::load(theta),den=ElementPack::load(denominator),negative_ell=ElementPack(-1.0)*ElementPack::load(ell);
        ElementPack gtheta[3];
        alignas(64) double gg[3][W],hh[3][3][W];
        for(int i=0;i<3;++i) {
            const ElementPack dX=select(r0,mbA[i]-maB[i],select(r1,mab[i]-mba[i],select(r2,mbe[i],ema[i])));
            ElementPack dY0=element_multiply_add(ElementPack(-1.0)*ehA,mB[i],coef0*ehat[i]);
            dY0=element_multiply_add(ElementPack(-1.0)*ehB,mA[i],dY0);
            ElementPack dY1=element_multiply_add(eha,mB[i],coef1*ehat[i]);
            dY1=element_multiply_add(ehb,mA[i],dY1);
            const ElementPack dY=select(r0,dY0,select(r1,dY1,negative_ell*select(r2,mB[i],mA[i])));
            gtheta[i]=select(v,element_multiply_add(ElementPack(-1.0)*yy,dX,xx*dY)/den,zero);
            select(v, gs*gtheta[i], zero).store(gg[i]);
        }
        // Eigen scales the left vector before forming the outer product.
        for(int i=0;i<3;++i)
            for(int j=0;j<3;++j) select(v, (ss*gtheta[i])*gtheta[j], zero).store(hh[i][j]);
        for(int lane=0;lane<count;++lane)
            for(int i=0;i<3;++i) {
                gradients[begin+lane][i]=gg[i][lane];
                for(int j=0;j<3;++j) hessians[begin+lane](i,j)=hh[i][j][lane];
            }
    }
}

} // namespace ipc_simd
