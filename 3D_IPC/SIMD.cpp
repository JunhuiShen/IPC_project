#include "SIMD.h"
#include "corotated_energy.h"
#include "friction_energy.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

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
};

constexpr int W = Pack::width;

} // namespace

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

// V2 receives gathered AoS tiles, transposes locally, and returns per-entry
// AoS derivatives. Regular elasticity and bending arithmetic use element lanes.
const char* tile_backend_name() {
#if defined(__AVX512F__)
    return "AVX-512 (8 doubles)";
#else
    return backend_name();
#endif
}

void point_derivatives_tile(const PointInput* inputs, std::size_t entry_count,
    const Vec3& gravity, double kpin, double dt2,
    Vec3* gradients, Mat33* hessians) {
    assert(entry_count <= tile_width);
    const Pack zero(0.0), one(1.0), timestep2(dt2), pin_scale(dt2 * kpin);
    for (std::size_t begin = 0; begin < entry_count; begin += W) {
        const int count = static_cast<int>(std::min<std::size_t>(W, entry_count-begin));
        alignas(32) double mass[W], pinned[W], position[3][W], predicted[3][W], target[3][W];
        for (int lane = 0; lane < W; ++lane) {
            const auto& input = inputs[begin + std::min(lane,count-1)];
            mass[lane] = input.mass;
            pinned[lane] = input.pin_target ? 1.0 : 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                position[axis][lane] = input.position[axis];
                predicted[axis][lane] = input.predicted_position[axis];
                target[axis][lane] = input.pin_target ? (*input.pin_target)[axis] : input.position[axis];
            }
        }
        const Pack m = Pack::load(mass), minus_m = Pack(-1.0) * m;
        const Pack has_pin = equal(Pack::load(pinned), one);
        const Pack masked_pin_scale = select(has_pin, pin_scale, zero);
        alignas(32) double g[3][W], diagonal[W];
        for (int axis = 0; axis < 3; ++axis) {
            const Pack current = Pack::load(position[axis]);
            Pack value = zero + m * (current - Pack::load(predicted[axis]));
            value = value + timestep2 * (minus_m * Pack(gravity[axis]));
            select(has_pin, value + masked_pin_scale * (current - Pack::load(target[axis])), value).store(g[axis]);
        }
        const Pack inertial_hessian = zero + m;
        select(has_pin, inertial_hessian + masked_pin_scale, inertial_hessian).store(diagonal);
        for (int lane = 0; lane < count; ++lane) {
            const auto entry = begin + lane;
            hessians[entry].setZero();
            for (int axis = 0; axis < 3; ++axis) {
                gradients[entry][axis] = g[axis][lane];
                hessians[entry](axis,axis) = diagonal[lane];
            }
        }
    }
}

namespace {

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
    friend ElementPack sqrt(ElementPack a) { return raw(_mm512_sqrt_pd(a.value)); }
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

// Preserve the rounded product without spilling it to volatile memory.
static ElementPack element_separate_product(ElementPack a, ElementPack b) {
    auto value = (a * b).value;
#if defined(__GNUC__) && (defined(__AVX2__) || defined(__SSE2__))
    __asm__("" : "+v"(value));
#elif defined(__GNUC__) && defined(__aarch64__)
    __asm__("" : "+w"(value));
#else
    volatile ElementPack::Native rounded = value;
    value = rounded;
#endif
    return ElementPack::raw(value);
}

template <typename T, std::size_t N>
static void pad_prepared_lanes(T (&values)[N], int count) {
    if constexpr (std::is_same_v<T, double>)
        std::fill(values + count, values + N, values[count - 1]);
    else
        for (auto& row : values) pad_prepared_lanes(row, count);
}

} // namespace

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
        for (int lane = 0; lane < count; ++lane) {
            const std::size_t entry = begin + lane;
            for (int vertex = 0; vertex < 3; ++vertex)
                for (int axis = 0; axis < 3; ++axis)
                    points[vertex][axis][lane] = positions[3*entry+vertex][axis];
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 2; ++col)
                    material[row][col][lane] = dm_inverse[entry](row,col);
        }
        // Preserve the scalar eigensolver and its clamping decisions. Only
        // local tile data are read; the derivative contractions below use SIMD.
        for (int lane = 0; lane < count; ++lane) {
            const std::size_t entry = begin + lane;
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
        // Fill unused lanes from prepared values without repeating the eigensolve.
        if (count < W) {
            pad_prepared_lanes(s, count); pad_prepared_lanes(ci, count);
            pad_prepared_lanes(r, count); pad_prepared_lanes(b, count);
            pad_prepared_lanes(f_data, count); pad_prepared_lanes(p, count);
            pad_prepared_lanes(q, count); pad_prepared_lanes(area, count);
            pad_prepared_lanes(jd, count); pad_prepared_lanes(tr, count);
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
        alignas(64) double points[4][3][W];
        for (int lane = 0; lane < count; ++lane) {
            const auto entry = begin + lane;
            for (int vertex = 0; vertex < 4; ++vertex)
                for (int axis = 0; axis < 3; ++axis)
                    points[vertex][axis][lane] = positions[4 * entry + vertex][axis];
        }
        if (count < W) pad_prepared_lanes(points, count);
        const auto ordered_dot=[](const std::array<ElementPack,3>& a,const std::array<ElementPack,3>& b) {
            // Match Eigen's pair reduction before the fused third product.
            return element_multiply_add(a[2], b[2],
                element_separate_product(a[0], b[0]) + element_separate_product(a[1], b[1]));
        };
        const auto ordered_cross=[](const std::array<ElementPack,3>& a,const std::array<ElementPack,3>& b) {
            return std::array<ElementPack,3>{
                element_multiply_add(a[1],b[2],ElementPack(-1.0)*(a[2]*b[1])),
                element_multiply_add(a[2],b[0],ElementPack(-1.0)*(a[0]*b[2])),
                element_multiply_add(a[0],b[1],ElementPack(-1.0)*(a[1]*b[0]))};
        };
        std::array<ElementPack, 3> e, a, b, A, B, ehat;
        for (int axis = 0; axis < 3; ++axis) {
            const auto x0 = ElementPack::load(points[0][axis]);
            const auto x1 = ElementPack::load(points[1][axis]);
            const auto x2 = ElementPack::load(points[2][axis]);
            const auto x3 = ElementPack::load(points[3][axis]);
            e[axis] = x1 - x0;
            a[axis] = x2 - x0;
            b[axis] = x3 - x0;
            A[axis] = x2 - x1;
            B[axis] = x3 - x1;
        }
        const auto mA = ordered_cross(e, a), mB = ordered_cross(b, e);
        alignas(64) double muA2[W], muB2[W], ell[W], safe_ell[W];
        alignas(64) double X[W], Y[W], theta[W], denominator[W], scale[W], roles[W], valid[W];
        ordered_dot(mA, mA).store(muA2);
        ordered_dot(mB, mB).store(muB2);
        sqrt(ordered_dot(e, e)).store(ell);
        for (int lane = 0; lane < W; ++lane) {
            const auto entry = begin + std::min(lane, count - 1);
            assert(active_nodes[entry] >= 0 && active_nodes[entry] < 4);
            const bool degenerate = ell[lane] <= 0.0 || muA2[lane] <= 0.0 || muB2[lane] <= 0.0;
            valid[lane] = degenerate ? 0.0 : 1.0;
            safe_ell[lane] = degenerate ? 1.0 : ell[lane];
            denominator[lane] = degenerate ? 1.0 : muA2[lane] * muB2[lane];
            scale[lane] = 2.0 * kB * coefficients[entry];
            roles[lane] = active_nodes[entry];
        }
        const auto nondegenerate = equal(ElementPack::load(valid), one);
        for (int axis = 0; axis < 3; ++axis)
            ehat[axis] = select(nondegenerate, e[axis] / ElementPack::load(safe_ell), zero);
        select(nondegenerate, ordered_dot(mA, mB), zero).store(X);
        select(nondegenerate, ordered_dot(ordered_cross(mA, mB), ehat), zero).store(Y);
        // Keep the reference angle function and evaluate only real entries.
        for (int lane = 0; lane < count; ++lane)
            theta[lane] = (valid[lane] ? std::atan2(Y[lane], X[lane]) : 0.0) - rest_angles[begin + lane];
        if (count < W) pad_prepared_lanes(theta, count);
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

namespace {

// Contact features use independent lanes in the same hardware pack.
using ContactElementPack = Pack;
static ContactElementPack contact_vector_multiply_add(ContactElementPack a, ContactElementPack b, ContactElementPack c) {
#if defined(__AVX2__) && defined(__FMA__)
    return ContactElementPack::raw(_mm256_fmadd_pd(a.value, b.value, c.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
    return ContactElementPack::raw(vfmaq_f64(c.value, a.value, b.value));
#else
    return a * b + c;
#endif
}

// Arithmetic lanes are independent contacts; no mesh indices enter this layer.
struct ContactPack {
    ContactElementPack value;
    ContactPack() : value(0.0) {}
    ContactPack(double x) : value(x) {}
    ContactPack(ContactElementPack x) : value(x) {}
    static ContactPack load(const double* x) { return ContactElementPack::load(x); }
    void store(double* x) const { value.store(x); }
    friend ContactPack operator+(ContactPack a, ContactPack b) { return a.value + b.value; }
    friend ContactPack operator-(ContactPack a, ContactPack b) { return a.value - b.value; }
    friend ContactPack operator*(ContactPack a, ContactPack b) { return a.value * b.value; }
    friend ContactPack operator/(ContactPack a, ContactPack b) { return a.value / b.value; }
    friend ContactPack operator-(ContactPack a) { return ContactElementPack(-1.0) * a.value; }
    ContactPack& operator+=(ContactPack b) { value = value + b.value; return *this; }
    ContactPack& operator-=(ContactPack b) { value = value - b.value; return *this; }
    ContactPack& operator/=(ContactPack b) { value = value / b.value; return *this; }
};

static ContactElementPack contact_greater(ContactElementPack a, ContactElementPack b) {
    return greater(a, b);
}
static ContactPack contact_sqrt(ContactPack x) {
    return sqrt(x.value);
}
static ContactPack contact_multiply_add(ContactPack a, ContactPack b, ContactPack c) {
    return contact_vector_multiply_add(a.value, b.value, c.value);
}
// Keep the scalar kernel's rounded products separate from later fused sums.
// The empty register constraint emits no instructions or memory accesses.
static ContactPack contact_separate_product(ContactPack a, ContactPack b) {
    auto result = (a * b).value;
#if defined(__GNUC__) && (defined(__AVX2__) || defined(__SSE2__))
    __asm__("" : "+v"(result.value));
#elif defined(__GNUC__) && defined(__aarch64__)
    __asm__("" : "+w"(result.value));
#endif
    return result;
}
// The scalar reduction rounds its xy products before adding the fused z tail.
static ContactPack contact_ordered_dot(const ContactPack* a, const ContactPack* b) {
    return contact_multiply_add(a[2], b[2],
        (ContactPack(0.0) + contact_separate_product(a[0], b[0])) + contact_separate_product(a[1], b[1]));
}

static ContactPack contact_sign(ContactPack x) {
    return select(contact_greater(x.value, ContactElementPack(0.0)), ContactElementPack(1.0),
        select(contact_greater(ContactElementPack(0.0), x.value), ContactElementPack(-1.0), ContactElementPack(0.0)));
}
struct ContactVector {
    ContactPack data[3];
    ContactPack& operator()(int i) { return data[i]; }
    const ContactPack& operator()(int i) const { return data[i]; }
};
struct ContactMatrix {
    ContactPack data[3][3];
    ContactPack& operator()(int i, int j) { return data[i][j]; }
    const ContactPack& operator()(int i, int j) const { return data[i][j]; }
};
struct ContactPacket {
    ContactVector x[4], separation;
    ContactPack delta, bp, bpp, sa, sb, sw, query, edge_a;
};

enum class ContactFeature { Point, Edge, Face, Interior };
struct PreparedMeshContact {
    std::array<Vec3, 4> positions;
    // Face normal, or closest-point displacement for other features.
    Vec3 gradient_vector, separation;
    double gradient_scale;
    double delta, bp, bpp;
    double sa, sb, sw, query, edge_a;
    ContactFeature feature;
    bool active;
};
static ContactMatrix contact_point_hessian(const ContactPacket& data) {
    ContactMatrix H;
    ContactPack u[3];
    for (int i = 0; i < 3; ++i) u[i] = (data.x[0](i) - data.x[1](i)) / data.delta;
    const ContactPack c2 = data.bp / data.delta;
    for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
    {
        const ContactPack normal = contact_multiply_add(-u[k], u[l], k == l ? 1.0 : 0.0);
        const ContactPack left = data.bpp * u[k];
        // Preserve which product the scalar self block contracts into the sum.
        H(k,l) = data.query * (k == 1 && l == 1
            ? contact_multiply_add(c2, normal, left * u[l])
            : contact_multiply_add(left, u[l], c2 * normal));
    }
    return H;
}

// Use the scalar reference for lanes with poor conditioning or cancellation.
// Regular lanes evaluate the original derivative expressions below.
static ContactPack contact_expression_fallback(const ContactPacket& data, ContactFeature feature) {
    if (feature == ContactFeature::Point) return 0.0;
    ContactPack a[3], b[3], r[3];
    for (int i = 0; i < 3; ++i) {
        r[i] = data.separation(i);
        a[i] = feature == ContactFeature::Interior
            ? data.x[1](i) - data.x[0](i) : data.x[2](i) - data.x[1](i);
        b[i] = feature == ContactFeature::Interior
            ? data.x[3](i) - data.x[2](i) : data.x[3](i) - data.x[1](i);
    }
    const ContactPack A = contact_ordered_dot(a, a);
    const auto stationarity_bad = [&](const ContactPack* direction, ContactPack length2) {
        const ContactPack residual = contact_multiply_add(r[2], direction[2],
            contact_multiply_add(r[1], direction[1], r[0] * direction[0]));
        const ContactPack bound = ContactPack(1e-24) * data.delta * data.delta * length2;
        return contact_greater((residual * residual).value, bound.value);
    };
    ContactPack fallback;
    if (feature == ContactFeature::Edge) {
        ContactPack offset[3];
        for (int i = 0; i < 3; ++i) offset[i] = data.x[0](i) - data.x[1](i);
        const ContactPack t = contact_ordered_dot(offset, a) / A;
        const auto valid = select(contact_greater(A.value, ContactElementPack(0.0)),
            select(contact_greater(t.value, ContactElementPack(0.0)),
                contact_greater(ContactElementPack(1.0), t.value), ContactElementPack(0.0)),
            ContactElementPack(0.0));
        fallback = select(valid, ContactElementPack(0.0), ContactElementPack(1.0));
        fallback = select(stationarity_bad(a, A), ContactElementPack(1.0), fallback.value);
    } else {
        const ContactPack B = contact_ordered_dot(a, b), C = contact_ordered_dot(b, b);
        const ContactPack det = contact_multiply_add(A, C, -(B * B));
        fallback = select(contact_greater(det.value, (ContactPack(1e-8) * A * C).value),
            ContactElementPack(0.0), ContactElementPack(1.0));
        fallback = select(stationarity_bad(a, A), ContactElementPack(1.0), fallback.value);
        fallback = select(stationarity_bad(b, C), ContactElementPack(1.0), fallback.value);
    }
    return fallback;
}

static ContactMatrix contact_edge_hessian(const ContactPacket& data) {
    const int p = 0, q = 0, requested_dof_count = 1;
    const int requested_dofs[1] = {0};
    const ContactPack delta = data.delta, bp = data.bp, bpp = data.bpp;
    ContactMatrix H;
    const ContactPack omega[4] = {data.sa, 0.0, 0.0, 0.0};
    const ContactPack epsilon[4] = {data.sb, 0.0, 0.0, 0.0};
    const auto& x = data.x[0];
    const auto& xa = data.x[1];
    const auto& xb = data.x[2];
    ContactPack e[3], w[3];
    for (int i = 0; i < 3; ++i) { e[i] = xb(i) - xa(i); w[i] = x(i) - xa(i); }

    const ContactPack alpha = contact_ordered_dot(w, e), beta = contact_ordered_dot(e, e);
    const ContactPack t = alpha / beta;

    ContactPack r[3], u[3];
    for (int i = 0; i < 3; ++i) {
        r[i] = x(i) - (xa(i) + t * e[i]);
        u[i] = r[i] / delta;
    }

    ContactPack t_d[4][3];
    ContactPack r_d[4][3][3];
    for (int di = 0; di < requested_dof_count; ++di) {
        const int pp = requested_dofs[di];
        for (int k = 0; k < 3; ++k) {
            const ContactPack alpha_pk = omega[pp] * e[k] + epsilon[pp] * w[k];
            const ContactPack beta_pk  = 2.0 * epsilon[pp] * e[k];
            t_d[pp][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);
            for (int i = 0; i < 3; ++i) {
                const ContactPack dik = (i == k) ? 1.0 : 0.0;
                const ContactPack dpa = data.edge_a;
                const ContactPack dpx = data.query;
                const ContactPack q_d = dpa * dik + t_d[pp][k] * e[i] + t * epsilon[pp] * dik;
                r_d[pp][k][i] = dpx * dik - q_d;
            }
        }
    }

    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            const ContactPack dkl = (k == l) ? 1.0 : 0.0;

            const ContactPack alpha_pk = omega[p] * e[k] + epsilon[p] * w[k];
            const ContactPack alpha_ql = omega[q] * e[l] + epsilon[q] * w[l];
            const ContactPack alpha_pkql =
                    (omega[p] * epsilon[q] + epsilon[p] * omega[q]) * dkl;
            const ContactPack beta_pk   = 2.0 * epsilon[p] * e[k];
            const ContactPack beta_ql   = 2.0 * epsilon[q] * e[l];
            const ContactPack beta_pkql = 2.0 * epsilon[p] * epsilon[q] * dkl;

            const ContactPack t_pkql = alpha_pkql / beta
                                 - contact_multiply_add(alpha, beta_pkql, contact_multiply_add(alpha_pk, beta_ql, alpha_ql * beta_pk)) / (beta * beta)
                                 + 2.0 * alpha * beta_pk * beta_ql / (beta * beta * beta);

            ContactPack ddelta_pk = 0.0, ddelta_ql = 0.0;
            for (int i = 0; i < 3; ++i) {
                ddelta_pk += u[i] * r_d[p][k][i];
                ddelta_ql += u[i] * r_d[q][l][i];
            }

            ContactPack proj_term = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const ContactPack dij = (i == j) ? 1.0 : 0.0;
                    proj_term = contact_multiply_add(contact_multiply_add(-u[i], u[j], dij) * r_d[p][k][i], r_d[q][l][j], proj_term);
                }
            }
            proj_term /= delta;

            ContactPack uq_term = 0.0;
            for (int i = 0; i < 3; ++i) {
                const ContactPack dik = (i == k) ? 1.0 : 0.0;
                const ContactPack dil = (i == l) ? 1.0 : 0.0;
                const ContactPack q_ipkql = t_pkql * e[i]
                                     + t_d[p][k] * epsilon[q] * dil
                                     + t_d[q][l] * epsilon[p] * dik;
                uq_term += u[i] * q_ipkql;
            }

            const ContactPack d2delta = proj_term - uq_term;
            H(k, l) = contact_multiply_add(bpp * ddelta_pk, ddelta_ql, bp * d2delta);
        }
    }
    return H;
}

// Keep the SIMD coefficient lookup independent of scalar-kernel inlining.
static constexpr int contact_levi_civita(int i, int j, int k) {
    if (i == j || j == k || i == k) return 0;
    return ((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) || (i == 2 && j == 0 && k == 1)) ? 1 : -1;
}

static ContactMatrix contact_face_hessian(const ContactPacket& data) {
    const int p = 0, q = 0, requested_dof_count = 1;
    const int requested_dofs[1] = {0};
    const ContactPack delta = data.delta, bp = data.bp, bpp = data.bpp;
    ContactMatrix H;
    const auto& x = data.x[0];
    const auto& x1 = data.x[1];
    const auto& x2 = data.x[2];
    const auto& x3 = data.x[3];

    const ContactPack sig_a[4] = {data.sa, 0.0, 0.0, 0.0};
    const ContactPack sig_b[4] = {data.sb, 0.0, 0.0, 0.0};
    const ContactPack sig_w[4] = {data.sw, 0.0, 0.0, 0.0};

    ContactPack a[3], b[3], w[3];
    for (int i = 0; i < 3; ++i) {
        a[i] = x2(i) - x1(i);
        b[i] = x3(i) - x1(i);
        w[i] = x(i)  - x1(i);
    }

    ContactPack N[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        for (int m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
                N[i] = i < 2
                    ? N[i] + contact_separate_product(contact_levi_civita(i, m, n) * a[m], b[n])
                    : contact_multiply_add(contact_levi_civita(i, m, n) * a[m], b[n], N[i]);
            }
        }
    }

    ContactPack eta = 0.0;
    for (int i = 0; i < 3; ++i) eta = contact_multiply_add(N[i], N[i], eta);
    eta = contact_sqrt(eta);

    ContactPack n[3];
    for (int i = 0; i < 3; ++i) n[i] = N[i] / eta;

    const ContactPack psi = contact_ordered_dot(N, w);
    const ContactPack phi = psi / eta;
    const ContactPack s_sign = contact_sign(phi);

    ContactPack Nd[4][3][3];
    for (int di = 0; di < requested_dof_count; ++di) {
        const int pp = requested_dofs[di];
        for (int k = 0; k < 3; ++k) {
            for (int i = 0; i < 3; ++i) {
                ContactPack val = 0.0;
                for (int nn = 0; nn < 3; ++nn) val += sig_a[pp] * contact_levi_civita(i, k, nn) * b[nn];
                for (int m = 0; m < 3; ++m)    val += sig_b[pp] * contact_levi_civita(i, m, k) * a[m];
                Nd[pp][k][i] = val;
            }
        }
    }

    ContactPack eta_d[4][3], psi_d[4][3], phi_d[4][3];
    for (int di = 0; di < requested_dof_count; ++di) {
        const int pp = requested_dofs[di];
        for (int k = 0; k < 3; ++k) {
            ContactPack eta_pk = 0.0;
            for (int i = 0; i < 3; ++i) eta_pk += n[i] * Nd[pp][k][i];
            eta_d[pp][k] = eta_pk;

            ContactPack psi_pk = 0.0;
            for (int i = 0; i < 3; ++i) psi_pk += Nd[pp][k][i] * w[i];
            psi_pk += sig_w[pp] * N[k];
            psi_d[pp][k] = psi_pk;

            phi_d[pp][k] = psi_pk / eta - psi * eta_pk / (eta * eta);
        }
    }

    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            // p == q: the cross product's second derivative for one vertex
            // is exactly zero. Exceptional geometry still uses the fallback.
            ContactPack proj_NN = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const ContactPack dij = (i == j) ? 1.0 : 0.0;
                    proj_NN = contact_multiply_add(contact_multiply_add(-n[i], n[j], dij) * Nd[p][k][i], Nd[q][l][j], proj_NN);
                }
            }

            const ContactPack eta_pkql = ContactPack(0.0) + proj_NN / eta;

            ContactPack psi_pkql = 0.0;
            psi_pkql += sig_w[q] * Nd[p][k][l];
            psi_pkql += sig_w[p] * Nd[q][l][k];

            const ContactPack phi_pkql = psi_pkql / eta
                                  - contact_multiply_add(psi, eta_pkql, contact_multiply_add(psi_d[p][k], eta_d[q][l], psi_d[q][l] * eta_d[p][k])) / (eta * eta)
                                  + 2.0 * psi * eta_d[p][k] * eta_d[q][l] / (eta * eta * eta);

            H(k, l) = contact_multiply_add(bpp * phi_d[p][k], phi_d[q][l], s_sign * bp * phi_pkql);
        }
    }
    return H;
}

static ContactMatrix contact_interior_hessian(const ContactPacket& data) {
    const int p = 0, q = 0, requested_dof_count = 1;
    const int requested_dofs[1] = {0};
    const ContactPack delta = data.delta, bp = data.bp, bpp = data.bpp;
    ContactMatrix H;
    const auto& x1 = data.x[0];
    const auto& x2 = data.x[1];
    const auto& x3 = data.x[2];
    const auto& x4 = data.x[3];

    const ContactPack sig_a[4] = {data.sa, 0.0, 0.0, 0.0};
    const ContactPack sig_b[4] = {data.sb, 0.0, 0.0, 0.0};
    const ContactPack sig_c[4] = {data.sw, 0.0, 0.0, 0.0};

    ContactPack a[3], b[3], c[3];
    for (int i = 0; i < 3; ++i) {
        a[i] = x2(i) - x1(i);
        b[i] = x4(i) - x3(i);
        c[i] = x1(i) - x3(i);
    }

    const ContactPack A = contact_ordered_dot(a, a), B = contact_ordered_dot(a, b),
        C = contact_ordered_dot(b, b), D = contact_ordered_dot(a, c), E = contact_ordered_dot(b, c);

    const ContactPack Delta = contact_multiply_add(A, C, -(B * B));
    const ContactPack nu    = contact_multiply_add(B, E, -(C * D));
    const ContactPack zeta  = contact_multiply_add(A, E, -(B * D));
    const ContactPack s_val = nu / Delta;
    const ContactPack t_val = zeta / Delta;

    ContactPack Ad[4][3], Bd[4][3], Cd[4][3], Dd[4][3], Ed[4][3];
    for (int di = 0; di < requested_dof_count; ++di) {
        const int pp = requested_dofs[di];
        for (int k = 0; k < 3; ++k) {
            Ad[pp][k] = 2.0 * sig_a[pp] * a[k];
            Bd[pp][k] = sig_a[pp] * b[k] + sig_b[pp] * a[k];
            Cd[pp][k] = 2.0 * sig_b[pp] * b[k];
            Dd[pp][k] = sig_a[pp] * c[k] + sig_c[pp] * a[k];
            Ed[pp][k] = sig_b[pp] * c[k] + sig_c[pp] * b[k];
        }
    }

    ContactPack nu_d[4][3], zeta_d[4][3], Delta_d[4][3];
    for (int di = 0; di < requested_dof_count; ++di) {
        const int pp = requested_dofs[di];
        for (int k = 0; k < 3; ++k) {
            nu_d[pp][k] = contact_multiply_add(Bd[pp][k], E, B * Ed[pp][k]);
            nu_d[pp][k] = contact_multiply_add(-Cd[pp][k], D, nu_d[pp][k]);
            nu_d[pp][k] = contact_multiply_add(-C, Dd[pp][k], nu_d[pp][k]);
            zeta_d[pp][k] = k < 2
                ? contact_multiply_add(A, Ed[pp][k], Ad[pp][k] * E)
                : contact_multiply_add(Ad[pp][k], E, A * Ed[pp][k]);
            zeta_d[pp][k] = contact_multiply_add(-Bd[pp][k], D, zeta_d[pp][k]);
            zeta_d[pp][k] = contact_multiply_add(-B, Dd[pp][k], zeta_d[pp][k]);
            Delta_d[pp][k] = k < 2
                ? contact_multiply_add(A, Cd[pp][k], Ad[pp][k] * C)
                : contact_multiply_add(Ad[pp][k], C, A * Cd[pp][k]);
            Delta_d[pp][k] = contact_multiply_add(-2.0 * B, Bd[pp][k], Delta_d[pp][k]);
        }
    }

    ContactPack s_d[4][3], t_d[4][3];
    for (int di = 0; di < requested_dof_count; ++di) {
        const int pp = requested_dofs[di];
        for (int k = 0; k < 3; ++k) {
            s_d[pp][k] = nu_d[pp][k] / Delta   - nu   * Delta_d[pp][k] / (Delta * Delta);
            t_d[pp][k] = zeta_d[pp][k] / Delta - zeta * Delta_d[pp][k] / (Delta * Delta);
        }
    }

    ContactPack r_vec[3], u[3];
    for (int i = 0; i < 3; ++i) {
        r_vec[i] = (x1(i) + s_val * a[i]) - (x3(i) + t_val * b[i]);
        u[i] = r_vec[i] / delta;
    }

    ContactPack p_d[4][3][3], q_d_arr[4][3][3], r_d[4][3][3];
    for (int di = 0; di < requested_dof_count; ++di) {
        const int pp = requested_dofs[di];
        for (int k = 0; k < 3; ++k) {
            for (int i = 0; i < 3; ++i) {
                const ContactPack dik = (i == k) ? 1.0 : 0.0;
                const ContactPack dp0 = data.query;
                const ContactPack dp2 = data.edge_a;
                p_d[pp][k][i]     = dp0 * dik + s_d[pp][k] * a[i] + s_val * sig_a[pp] * dik;
                q_d_arr[pp][k][i] = dp2 * dik + t_d[pp][k] * b[i] + t_val * sig_b[pp] * dik;
                r_d[pp][k][i]     = p_d[pp][k][i] - q_d_arr[pp][k][i];
            }
        }
    }

    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            const ContactPack dkl = (k == l) ? 1.0 : 0.0;

            const ContactPack A_pkql = 2.0 * sig_a[p] * sig_a[q] * dkl;
            const ContactPack B_pkql =
                    (sig_a[p] * sig_b[q] + sig_b[p] * sig_a[q]) * dkl;
            const ContactPack C_pkql = 2.0 * sig_b[p] * sig_b[q] * dkl;
            const ContactPack D_pkql =
                    (sig_a[p] * sig_c[q] + sig_c[p] * sig_a[q]) * dkl;
            const ContactPack E_pkql =
                    (sig_b[p] * sig_c[q] + sig_c[p] * sig_b[q]) * dkl;

            ContactPack nu_pkql = contact_multiply_add(B_pkql, E, Bd[p][k] * Ed[q][l]);
            nu_pkql = contact_multiply_add(Bd[q][l], Ed[p][k], nu_pkql);
            nu_pkql = contact_multiply_add(B, E_pkql, nu_pkql);
            nu_pkql = contact_multiply_add(-C_pkql, D, nu_pkql);
            nu_pkql = contact_multiply_add(-Cd[p][k], Dd[q][l], nu_pkql);
            nu_pkql = contact_multiply_add(-Cd[q][l], Dd[p][k], nu_pkql);
            nu_pkql = contact_multiply_add(-C, D_pkql, nu_pkql);
            ContactPack Delta_pkql = contact_multiply_add(A_pkql, C, Ad[p][k] * Cd[q][l]);
            Delta_pkql = contact_multiply_add(Ad[q][l], Cd[p][k], Delta_pkql);
            Delta_pkql = contact_multiply_add(A, C_pkql, Delta_pkql);
            Delta_pkql -= 2.0 * contact_multiply_add(Bd[p][k], Bd[q][l], B * B_pkql);
            ContactPack zeta_pkql = contact_multiply_add(A_pkql, E, Ad[p][k] * Ed[q][l]);
            zeta_pkql = contact_multiply_add(Ad[q][l], Ed[p][k], zeta_pkql);
            zeta_pkql = contact_multiply_add(A, E_pkql, zeta_pkql);
            zeta_pkql = contact_multiply_add(-B_pkql, D, zeta_pkql);
            zeta_pkql = contact_multiply_add(-Bd[p][k], Dd[q][l], zeta_pkql);
            zeta_pkql = contact_multiply_add(-Bd[q][l], Dd[p][k], zeta_pkql);
            zeta_pkql = contact_multiply_add(-B, D_pkql, zeta_pkql);

            const ContactPack s_pkql = nu_pkql / Delta
                                - contact_multiply_add(nu, Delta_pkql, contact_multiply_add(nu_d[p][k], Delta_d[q][l], nu_d[q][l] * Delta_d[p][k])) / (Delta * Delta)
                                + 2.0 * nu * Delta_d[p][k] * Delta_d[q][l] / (Delta * Delta * Delta);
            const ContactPack t_pkql = zeta_pkql / Delta
                                - contact_multiply_add(zeta, Delta_pkql, contact_multiply_add(zeta_d[p][k], Delta_d[q][l], zeta_d[q][l] * Delta_d[p][k])) / (Delta * Delta)
                                + 2.0 * zeta * Delta_d[p][k] * Delta_d[q][l] / (Delta * Delta * Delta);

            ContactPack ddelta_pk = 0.0, ddelta_ql = 0.0;
            for (int i = 0; i < 3; ++i) {
                ddelta_pk += u[i] * r_d[p][k][i];
                ddelta_ql += u[i] * r_d[q][l][i];
            }

            ContactPack proj_term = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const ContactPack dij = (i == j) ? 1.0 : 0.0;
                    proj_term = contact_multiply_add(contact_multiply_add(-u[i], u[j], dij) * r_d[p][k][i], r_d[q][l][j], proj_term);
                }
            }
            proj_term /= delta;

            ContactPack ur_term = 0.0;
            for (int i = 0; i < 3; ++i) {
                const ContactPack dik = (i == k) ? 1.0 : 0.0;
                const ContactPack dil = (i == l) ? 1.0 : 0.0;
                ContactPack p_ipkql = contact_multiply_add(s_pkql, a[i], s_d[p][k] * sig_a[q] * dil);
                p_ipkql = contact_multiply_add(s_d[q][l] * sig_a[p], dik, p_ipkql);
                ContactPack q_ipkql = contact_multiply_add(t_pkql, b[i], t_d[p][k] * sig_b[q] * dil);
                q_ipkql = contact_multiply_add(t_d[q][l] * sig_b[p], dik, q_ipkql);
                ur_term += u[i] * (p_ipkql - q_ipkql);
            }

            H(k, l) = contact_multiply_add(bpp * ddelta_pk, ddelta_ql, bp * (proj_term + ur_term));
        }
    }
    return H;
}
static void prepare_point(PreparedMeshContact& out, const MeshContactInput& input,
                          int first, int second) {
    out.feature = ContactFeature::Point;
    out.positions[0] = input.positions[first];
    out.positions[1] = input.positions[second];
    out.query = input.role == first || input.role == second ? 1.0 : 0.0;
}
static void prepare_edge(PreparedMeshContact& out, const MeshContactInput& input,
                         int query, int first, int second) {
    out.feature = ContactFeature::Edge;
    out.positions[0] = input.positions[query];
    out.positions[1] = input.positions[first];
    out.positions[2] = input.positions[second];
    out.query = input.role == query ? 1.0 : 0.0;
    out.edge_a = input.role == first ? 1.0 : 0.0;
    out.sa = out.query - out.edge_a;
    out.sb = input.role == second ? 1.0 : -out.edge_a;
}

} // namespace

void mesh_contact_derivatives_tile(const MeshContactInput* inputs, std::size_t count,
    double d_hat, double k_barrier, double friction, double dt, double eps_v,
    MeshContactOutput* outputs, unsigned char* derivative_active) {
    assert(count <= contact_tile_width);
    if (derivative_active) std::fill(derivative_active, derivative_active + count, 0);
    std::array<PreparedMeshContact, contact_tile_width> prepared;
    std::optional<std::array<FrozenFrictionContact, contact_tile_width>> frozen;
    if (friction != 0.0) frozen.emplace();
    std::array<int, contact_tile_width> roles;
    for (std::size_t e = 0; e < count; ++e) {
        outputs[e].gradient.setZero();
        outputs[e].hessian.setZero();
        outputs[e].friction_gradient.setZero();
        outputs[e].friction_hessian.setZero();
        const auto& input = inputs[e];
        const auto& x = input.positions;
        const int role = input.role;
        if (role < 0 || role > 3) throw std::invalid_argument("SIMD contact: role must be in [0, 3].");
        roles[e] = role;
        auto& out = prepared[e];
        out.active = false;
        out.feature = ContactFeature::Point;
        out.sa = out.sb = out.sw = out.query = out.edge_a = 0.0;
        if (!input.segment_segment) {
            const auto evaluation = make_node_triangle_contact_evaluation(x, d_hat,
                friction != 0.0 ? k_barrier : 1.0);
            if (friction != 0.0)
                (*frozen)[e] = make_node_triangle_frozen_friction_contact(x, input.previous_positions,
                    evaluation, dt, eps_v);
            if (!evaluation.active) continue;
            const auto& dr = evaluation.dr;
            out.positions = x;
            out.active = true; out.delta = dr.distance;
            out.bp = evaluation.b_prime; out.bpp = evaluation.b_double_prime;
            const auto weights = friction != 0.0 && (*frozen)[e].active
                ? (*frozen)[e].weights
                : node_triangle_contact_weights(x[0], x[1], x[2], x[3], 1e-12, &dr);
            out.separation = x[0] - dr.closest_point;
            auto region = dr.region;
            if (region == NodeTriangleRegion::FaceInterior) out.separation = dr.phi * dr.normal;
            if (region == NodeTriangleRegion::DegenerateTriangle) {
                if (weights[1] == -1.0) region = NodeTriangleRegion::Vertex1;
                else if (weights[2] == -1.0) region = NodeTriangleRegion::Vertex2;
                else if (weights[3] == -1.0) region = NodeTriangleRegion::Vertex3;
                else if (weights[3] == 0.0) region = NodeTriangleRegion::Edge12;
                else if (weights[1] == 0.0) region = NodeTriangleRegion::Edge23;
                else region = NodeTriangleRegion::Edge31;
            }
            if (region == NodeTriangleRegion::FaceInterior) {
                out.feature = ContactFeature::Face;
                out.sa = role == 2 ? 1.0 : (role == 1 ? -1.0 : 0.0);
                out.sb = role == 3 ? 1.0 : (role == 1 ? -1.0 : 0.0);
                out.sw = role == 0 ? 1.0 : (role == 1 ? -1.0 : 0.0);
                const double sign = dr.phi > 0.0 ? 1.0 : (dr.phi < 0.0 ? -1.0 : 0.0);
                out.gradient_scale = (out.bp * sign) * weights[role];
                out.gradient_vector = dr.normal;
            } else {
                out.gradient_scale = out.bp * weights[role];
                out.gradient_vector = out.separation;
                switch (region) {
                    case NodeTriangleRegion::Edge12: prepare_edge(out,input,0,1,2); break;
                    case NodeTriangleRegion::Edge23: prepare_edge(out,input,0,2,3); break;
                    case NodeTriangleRegion::Edge31: prepare_edge(out,input,0,3,1); break;
                    case NodeTriangleRegion::Vertex1: prepare_point(out,input,0,1); break;
                    case NodeTriangleRegion::Vertex2: prepare_point(out,input,0,2); break;
                    case NodeTriangleRegion::Vertex3: prepare_point(out,input,0,3); break;
                    default: break;
                }
            }
        } else {
            const auto evaluation = make_segment_segment_contact_evaluation(x, d_hat,
                friction != 0.0 ? k_barrier : 1.0);
            if (friction != 0.0)
                (*frozen)[e] = make_segment_segment_frozen_friction_contact(x, input.previous_positions,
                    evaluation, dt, eps_v);
            if (!evaluation.active) continue;
            const auto& dr = evaluation.dr;
            out.positions = x;
            out.active = true; out.delta = dr.distance;
            out.bp = evaluation.b_prime; out.bpp = evaluation.b_double_prime;
            const auto weights = friction != 0.0 && (*frozen)[e].active
                ? (*frozen)[e].weights
                : segment_segment_contact_weights(x[0],x[1],x[2],x[3],1e-12,&dr);
            out.separation = dr.closest_point_1 - dr.closest_point_2;
            out.gradient_scale = out.bp * weights[role];
            out.gradient_vector = out.separation;
            auto region = dr.region;
            if (region == SegmentSegmentRegion::ParallelSegments) {
                const bool s0 = dr.s <= 1e-14, s1 = dr.s >= 1.0-1e-14;
                const bool t0 = dr.t <= 1e-14, t1 = dr.t >= 1.0-1e-14;
                if (s0 && t0) region = SegmentSegmentRegion::Corner_s0t0;
                else if (s0 && t1) region = SegmentSegmentRegion::Corner_s0t1;
                else if (s1 && t0) region = SegmentSegmentRegion::Corner_s1t0;
                else if (s1 && t1) region = SegmentSegmentRegion::Corner_s1t1;
                else if (s0) region = SegmentSegmentRegion::Edge_s0;
                else if (s1) region = SegmentSegmentRegion::Edge_s1;
                else if (t0) region = SegmentSegmentRegion::Edge_t0;
                else if (t1) region = SegmentSegmentRegion::Edge_t1;
                else {
                    const auto value = segment_segment_barrier_self_gradient_and_hessian(
                        x[0],x[1],x[2],x[3],role,evaluation);
                    outputs[e].gradient=value.first;outputs[e].hessian=value.second;
                    if (derivative_active) derivative_active[e] = 1;
                    out.active=false;continue;
                }
            }
            switch (region) {
                case SegmentSegmentRegion::Corner_s0t0: prepare_point(out,input,0,2); break;
                case SegmentSegmentRegion::Corner_s0t1: prepare_point(out,input,0,3); break;
                case SegmentSegmentRegion::Corner_s1t0: prepare_point(out,input,1,2); break;
                case SegmentSegmentRegion::Corner_s1t1: prepare_point(out,input,1,3); break;
                case SegmentSegmentRegion::Edge_s0: prepare_edge(out,input,0,2,3); break;
                case SegmentSegmentRegion::Edge_s1: prepare_edge(out,input,1,2,3); break;
                case SegmentSegmentRegion::Edge_t0: prepare_edge(out,input,2,0,1); break;
                case SegmentSegmentRegion::Edge_t1: prepare_edge(out,input,3,0,1); break;
                case SegmentSegmentRegion::Interior:
                    out.feature=ContactFeature::Interior;
                    out.sa=role==1?1.0:(role==0?-1.0:0.0);
                    out.sb=role==3?1.0:(role==2?-1.0:0.0);
                    out.sw=role==0?1.0:(role==2?-1.0:0.0);
                    out.query=role==0?1.0:0.0;out.edge_a=role==2?1.0:0.0;
                    break;
                default: break;
            }
        }
    }
    std::array<std::array<std::size_t, contact_tile_width>, 4> feature_indices;
    std::array<std::size_t, 4> feature_counts{};
    for (std::size_t e = 0; e < count; ++e) {
        auto& value = prepared[e];
        if (!value.active) continue;
        if (value.feature == ContactFeature::Point && value.query == 0.0) value.active = false;
        if (value.feature == ContactFeature::Edge && value.sa == 0.0 && value.sb == 0.0) value.active = false;
        if (value.active) {
            const auto feature = static_cast<std::size_t>(value.feature);
            feature_indices[feature][feature_counts[feature]++] = e;
            if (derivative_active) derivative_active[e] = 1;
        }
    }
    constexpr int W=ContactElementPack::width;
    for (auto feature : {ContactFeature::Point, ContactFeature::Edge, ContactFeature::Face, ContactFeature::Interior}) {
        const auto& indices = feature_indices[static_cast<std::size_t>(feature)];
        const auto entries = feature_counts[static_cast<std::size_t>(feature)];
        for (std::size_t begin=0;begin<entries;begin+=W) {
            const int active=static_cast<int>(std::min<std::size_t>(W,entries-begin));
            alignas(64) double x[4][3][W],parameters[8][W],direction[3][W],scale[W],separation[3][W];
            for (int lane=0;lane<W;++lane) {
                const auto& record=prepared[indices[begin+std::min(lane,active-1)]];
                for(int v=0;v<4;++v) for(int axis=0;axis<3;++axis) x[v][axis][lane]=record.positions[v][axis];
                const double fields[]={record.delta,record.bp,record.bpp,record.sa,record.sb,record.sw,record.query,record.edge_a};
                for(int f=0;f<8;++f) parameters[f][lane]=fields[f];
                for(int axis=0;axis<3;++axis) direction[axis][lane]=record.gradient_vector[axis];
                scale[lane]=record.gradient_scale;
                for(int axis=0;axis<3;++axis)separation[axis][lane]=record.separation[axis];
            }
            ContactPacket packet;
            for(int axis=0;axis<3;++axis)packet.separation(axis)=ContactPack::load(separation[axis]);
            for(int v=0;v<4;++v) for(int axis=0;axis<3;++axis) packet.x[v](axis)=ContactPack::load(x[v][axis]);
            packet.delta=ContactPack::load(parameters[0]);packet.bp=ContactPack::load(parameters[1]);packet.bpp=ContactPack::load(parameters[2]);
            packet.sa=ContactPack::load(parameters[3]);packet.sb=ContactPack::load(parameters[4]);packet.sw=ContactPack::load(parameters[5]);
            packet.query=ContactPack::load(parameters[6]);packet.edge_a=ContactPack::load(parameters[7]);
            ContactMatrix H;
            switch (feature) {
                case ContactFeature::Point: H = contact_point_hessian(packet); break;
                case ContactFeature::Edge: H = contact_edge_hessian(packet); break;
                case ContactFeature::Face: H = contact_face_hessian(packet); break;
                case ContactFeature::Interior: H = contact_interior_hessian(packet); break;
            }
            alignas(64) double g[3][W],h[3][3][W],fallback[W];
            contact_expression_fallback(packet,feature).store(fallback);
            for (int axis = 0; axis < 3; ++axis) {
                const auto vector = ContactPack::load(direction[axis]);
                const auto unit = feature == ContactFeature::Face ? vector : vector / packet.delta;
                (ContactPack::load(scale) * unit).store(g[axis]);
            }
            for(int row=0;row<3;++row) for(int col=0;col<3;++col) H(row,col).store(h[row][col]);
            for(int lane=0;lane<active;++lane) {
                auto& out=outputs[indices[begin+lane]];
                for(int row=0;row<3;++row) {
                    out.gradient[row]=g[row][lane];
                    for(int col=0;col<3;++col) out.hessian(row,col)=h[row][col][lane];
                }
                if(fallback[lane]!=0.0 || !out.hessian.allFinite()) {
                    const auto& input=inputs[indices[begin+lane]];
                    const auto& p=input.positions;
                    out.hessian=input.segment_segment
                        ? segment_segment_barrier_self_gradient_and_hessian(p[0],p[1],p[2],p[3],d_hat,input.role).second
                        : node_triangle_barrier_self_gradient_and_hessian(p[0],p[1],p[2],p[3],d_hat,input.role).second;
                }
            }
        }
    }
    if (friction != 0.0) {
        std::array<Vec3,contact_tile_width> gradients;
        std::array<Mat33,contact_tile_width> hessians;
        friction_derivatives_tile(frozen->data(),roles.data(),count,friction,dt*dt,gradients.data(),hessians.data());
        for(std::size_t e=0;e<count;++e) {
            outputs[e].friction_gradient=gradients[e];outputs[e].friction_hessian=hessians[e];
            if (derivative_active && !derivative_active[e] && (*frozen)[e].active) {
                // Preserve nonzero and exceptional friction outputs, including
                // a possible nonfinite intermediate at a zero-weight role.
                if ((*frozen)[e].weights[roles[e]] != 0.0
                    || !(gradients[e].array() == 0.0).all()
                    || !(hessians[e].array() == 0.0).all())
                    derivative_active[e] = 1;
            }
        }
    }
}

void friction_derivatives_tile(const FrozenFrictionContact* contacts, const int* roles,
    std::size_t count, double friction, double dt2, Vec3* gradients, Mat33* hessians) {
    assert(count <= contact_tile_width);
    constexpr int W=ContactElementPack::width;
    for(std::size_t begin=0;begin<count;begin+=W) {
        const int active=static_cast<int>(std::min<std::size_t>(W,count-begin));
        alignas(64) double slips[W],eps[W],normal[W],weights[W],u[3][W],projector[3][3][W];
        for(int lane=0;lane<W;++lane) {
            const std::size_t entry=begin+std::min(lane,active-1);
            const auto& c=contacts[entry];
            if(roles[entry]<0 || roles[entry]>=4)
                throw std::invalid_argument("frozen friction: role index must be in [0, 3].");
            const bool on=c.active && friction!=0.0 && dt2!=0.0;
            if(!std::isfinite(friction) || friction<0 || !std::isfinite(dt2) || dt2<0
                || (on && (!(c.eps_u>0) || !std::isfinite(c.eps_u)
                    || !(c.normal_force>0) || !std::isfinite(c.normal_force)
                    || !c.tangential_displacement.allFinite()))) {
                // Preserve the scalar API's validation and exception type.
                (void)frozen_friction_role_gradient_and_hessian(c,roles[entry],friction,dt2);
            }
            const double slip=on?c.tangential_displacement.norm():0.0;
            if(!std::isfinite(slip)) throw std::runtime_error("frozen friction: slip is not finite.");
            slips[lane]=slip;eps[lane]=on?c.eps_u:1.0;normal[lane]=on?c.normal_force:0.0;
            weights[lane]=c.weights[roles[entry]];
            for(int i=0;i<3;++i) {
                u[i][lane]=on?c.tangential_displacement[i]:0.0;
                for(int j=0;j<3;++j) projector[i][j][lane]=on?c.projector(i,j):0.0;
            }
        }
        const ContactElementPack zero(0.0),one(1.0),two(2.0);
        const auto slip=ContactElementPack::load(slips),epsilon=ContactElementPack::load(eps);
        const auto smooth=contact_greater(epsilon,slip);
        const auto safe_slip=select(smooth,one,slip);
        const auto mollifier=select(smooth,two/epsilon-slip/(epsilon*epsilon),one/safe_slip);
        const auto load=ContactElementPack::load(normal);
        const auto common=select(contact_greater(load,zero),ContactElementPack(dt2*friction),zero);
        const auto scale=common*load*mollifier;
        alignas(64) double scale_data[W],g[3][W],H[3][3][W];
        scale.store(scale_data);
        for(int lane=0;lane<active;++lane) {
            if(!std::isfinite(scale_data[lane]))
                throw std::runtime_error("frozen friction: derivative scale is not finite.");
            if(scale_data[lane]!=0.0 && !contacts[begin+lane].projector.allFinite())
                throw std::invalid_argument("frozen friction: contact data must be finite.");
        }
        const auto nonzero=contact_greater(scale,zero);
        const auto weight=ContactElementPack::load(weights),weight2=weight*weight;
        for(int i=0;i<3;++i) {
            (weight*select(nonzero,scale*ContactElementPack::load(u[i]),zero)).store(g[i]);
            for(int j=0;j<3;++j)
                (weight2*select(nonzero,scale*ContactElementPack::load(projector[i][j]),zero)).store(H[i][j]);
        }
        for(int lane=0;lane<active;++lane) for(int i=0;i<3;++i) {
            gradients[begin+lane][i]=g[i][lane];
            for(int j=0;j<3;++j) hessians[begin+lane](i,j)=H[i][j][lane];
        }
    }
}

void sdf_derivatives_tile(const SDFEvaluation* evaluations, std::size_t count,
    double stiffness, double epsilon, Vec3* gradients, Mat33* hessians, bool include_curvature) {
    assert(count <= tile_width);
    constexpr int W=ContactElementPack::width;
    for(std::size_t begin=0;begin<count;begin+=W) {
        const int active=static_cast<int>(std::min<std::size_t>(W,count-begin));
        alignas(64) double phi[W],normal[3][W],curvature[3][3][W];
        bool fallback[W]{};
        for(int lane=0;lane<W;++lane) {
            const auto& s=evaluations[begin+std::min(lane,active-1)];
            fallback[lane]=!std::isfinite(s.phi) || !std::isfinite(stiffness) || !std::isfinite(epsilon)
                || !s.grad_phi.allFinite() || (include_curvature && !s.hess_phi.allFinite());
            phi[lane]=fallback[lane]?0.0:s.phi;
            for(int i=0;i<3;++i) {
                normal[i][lane]=fallback[lane]?0.0:s.grad_phi[i];
                for(int j=0;j<3;++j) curvature[i][j][lane]=!include_curvature || fallback[lane]?0.0:s.hess_phi(i,j);
            }
        }
        const ContactElementPack zero(0.0),one(1.0),two(2.0),four(4.0),k(stiffness);
        const auto z=ContactElementPack::load(phi);
        ContactElementPack first,second,enabled;
        if(epsilon<=0.0) {
            enabled=contact_greater(zero,z);first=k*z;second=k;
        } else {
            const ContactElementPack eps(epsilon);
            enabled=contact_greater(eps,z);
            const auto H=select(contact_greater(zero,z),one,select(enabled,(eps-z)/eps,zero));
            const auto Hp=select(contact_greater(z,zero),select(enabled,zero-one/eps,zero),zero);
            const auto d=eps-z;
            first=ContactElementPack(0.5*stiffness)*(Hp*d*d-two*H*d);
            second=ContactElementPack(0.5*stiffness)*((zero-four*Hp)*d+two*H);
        }
        alignas(64) double g[3][W],h[3][3][W];
        for(int i=0;i<3;++i) {
            const auto ni=ContactElementPack::load(normal[i]);
            select(enabled,first*ni,zero).store(g[i]);
            for(int j=0;j<3;++j) {
                auto value=second*(ni*ContactElementPack::load(normal[j]));
                if(include_curvature) value=contact_vector_multiply_add(first,ContactElementPack::load(curvature[i][j]),value);
                select(enabled,value,zero).store(h[i][j]);
            }
        }
        for(int lane=0;lane<active;++lane) {
            const std::size_t entry=begin+lane;
            if(fallback[lane]) {
                gradients[entry]=sdf_penalty_gradient(evaluations[entry],stiffness,epsilon);
                hessians[entry]=sdf_penalty_hessian(evaluations[entry],stiffness,epsilon,include_curvature);
            } else for(int i=0;i<3;++i) {
                gradients[entry][i]=g[i][lane];
                for(int j=0;j<3;++j) hessians[entry](i,j)=h[i][j][lane];
            }
        }
    }
}

} // namespace ipc_simd
