#include "general_simd.h"

#include "batched_polar.h"
#include "friction_energy.h"
#include "parallel_helper.h"
#include "rigid_contact_scheduling.h"
#include "volumetric_corotated_energy.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

// Solid kernels.
namespace ipc_simd {
namespace {

// Independent element/node records occupy hardware lanes, as in SIMD.cpp.
// Keep this experimental kernel isolated from the existing cloth kernels.
struct SolidPack {
#if defined(__AVX512F__)
    using Native = __m512d;
    static constexpr int width = 8;
#elif defined(__AVX2__)
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
    SolidPack() = default;
    explicit SolidPack(double x) {
#if defined(__AVX512F__)
        value = _mm512_set1_pd(x);
#elif defined(__AVX2__)
        value = _mm256_set1_pd(x);
#elif defined(__aarch64__) || defined(_M_ARM64)
        value = vdupq_n_f64(x);
#elif defined(__SSE2__) || defined(_M_X64)
        value = _mm_set1_pd(x);
#else
        value = x;
#endif
    }
    static SolidPack raw(Native value) { SolidPack result; result.value = value; return result; }
    static SolidPack load(const double* values) {
#if defined(__AVX512F__)
        return raw(_mm512_loadu_pd(values));
#elif defined(__AVX2__)
        return raw(_mm256_loadu_pd(values));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vld1q_f64(values));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_loadu_pd(values));
#else
        return raw(*values);
#endif
    }
    void store(double* values) const {
#if defined(__AVX512F__)
        _mm512_storeu_pd(values, value);
#elif defined(__AVX2__)
        _mm256_storeu_pd(values, value);
#elif defined(__aarch64__) || defined(_M_ARM64)
        vst1q_f64(values, value);
#elif defined(__SSE2__) || defined(_M_X64)
        _mm_storeu_pd(values, value);
#else
        *values = value;
#endif
    }
    friend SolidPack operator+(SolidPack a, SolidPack b) {
#if defined(__AVX512F__)
        return raw(_mm512_add_pd(a.value, b.value));
#elif defined(__AVX2__)
        return raw(_mm256_add_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vaddq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_add_pd(a.value, b.value));
#else
        return raw(a.value + b.value);
#endif
    }
    friend SolidPack operator-(SolidPack a, SolidPack b) {
#if defined(__AVX512F__)
        return raw(_mm512_sub_pd(a.value, b.value));
#elif defined(__AVX2__)
        return raw(_mm256_sub_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vsubq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_sub_pd(a.value, b.value));
#else
        return raw(a.value - b.value);
#endif
    }
    friend SolidPack operator*(SolidPack a, SolidPack b) {
#if defined(__AVX512F__)
        return raw(_mm512_mul_pd(a.value, b.value));
#elif defined(__AVX2__)
        return raw(_mm256_mul_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vmulq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_mul_pd(a.value, b.value));
#else
        return raw(a.value * b.value);
#endif
    }
};

SolidPack multiply_add(SolidPack a, SolidPack b, SolidPack c) {
#if defined(__AVX512F__)
    return SolidPack::raw(_mm512_fmadd_pd(a.value, b.value, c.value));
#elif defined(__AVX2__) && defined(__FMA__)
    return SolidPack::raw(_mm256_fmadd_pd(a.value, b.value, c.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
    return SolidPack::raw(vfmaq_f64(c.value, a.value, b.value));
#else
    return a * b + c;
#endif
}

SolidPack separate_product(SolidPack a, SolidPack b) {
    auto value = (a * b).value;
#if defined(__GNUC__) && (defined(__AVX2__) || defined(__SSE2__))
    __asm__("" : "+v"(value));
#elif defined(__GNUC__) && defined(__aarch64__)
    __asm__("" : "+w"(value));
#else
    volatile SolidPack::Native rounded = value;
    value = rounded;
#endif
    return SolidPack::raw(value);
}

// Match Eigen's fixed-size 3x3 product in the strict-rounding volumetric
// reference. Its two packet rows use explicit multiply-add intrinsics, while
// the remaining scalar row uses a right-associated, non-contracted reduction.
// The lanes here still represent different tetrahedra, not matrix rows.
SolidPack matrix_row_product(int row, const SolidPack (&a)[3],
    const SolidPack (&b)[3]) {
    using EigenPacket = typename Eigen::internal::find_best_packet<double, 3>::type;
    constexpr int packet_rows = Eigen::internal::unpacket_traits<EigenPacket>::size;
    if constexpr (packet_rows > 1) {
        if (row < 3 / packet_rows * packet_rows)
            return multiply_add(a[2], b[2],
                multiply_add(a[1], b[1], separate_product(a[0], b[0])));
    }
    return separate_product(a[0], b[0])
        + (separate_product(a[1], b[1]) + separate_product(a[2], b[2]));
}

template <typename T, std::size_t N>
void pad_lanes(T (&values)[N], int count) {
    if constexpr (std::is_same_v<T, double>)
        std::fill(values + count, values + N, values[count - 1]);
    else
        for (auto& row : values) pad_lanes(row, count);
}

} // namespace

void solid_derivatives_tile(
    const Vec3* positions, const Mat33* dm_inverse, const double* measures,
    const Vec3* shape_gradients, std::size_t entry_count, double mu, double lambda,
    Vec3* gradients, Mat33* pbgs_blocks) {
    assert(entry_count <= tile_width);
    constexpr int width = SolidPack::width;
    const SolidPack zero(0.0), one(1.0), twice_mu(2.0 * mu), bulk(lambda);
    for (std::size_t begin = 0; begin < entry_count; begin += width) {
        const int count = static_cast<int>(std::min<std::size_t>(width, entry_count - begin));
        alignas(64) double deformation[3][3][width], rotation[3][3][width];
        alignas(64) double cofactor[3][3][width];
        alignas(64) double shape[3][width], measure[width], determinant[width];
        alignas(64) double gradient[3][width], block[3][3][width];
        alignas(64) double ds_values[3][3][width], inverse[3][3][width];
        // ElementF is compiled with contraction disabled, except for Eigen's
        // explicit packet FMAs. Reproduce both its reduction order and those
        // rounding boundaries while computing different elements in SIMD.
        for (int lane = 0; lane < count; ++lane) {
            const std::size_t entry = begin + lane;
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column) {
                    ds_values[row][column][lane] = positions[4 * entry + column + 1][row]
                        - positions[4 * entry][row];
                    inverse[row][column][lane] = dm_inverse[entry](row, column);
                }
        }
        if (count < width) {
            pad_lanes(ds_values, count);
            pad_lanes(inverse, count);
        }
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column) {
                const SolidPack a[3] = {SolidPack::load(ds_values[row][0]),
                    SolidPack::load(ds_values[row][1]), SolidPack::load(ds_values[row][2])};
                const SolidPack b[3] = {SolidPack::load(inverse[0][column]),
                    SolidPack::load(inverse[1][column]), SolidPack::load(inverse[2][column])};
                matrix_row_product(row, a, b).store(deformation[row][column]);
            }
        // Keep v2's deformation, polar factor and cofactors in SIMD lanes.
        // The AoS cache path remains the reference/fallback on other backends.
#if defined(__AVX512F__)
        volumetric_detail::signed_polar_soa_tile(&deformation[0][0][0],
            &rotation[0][0][0], count);
        SolidPack f[3][3], c[3][3];
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                f[row][column] = SolidPack::load(deformation[row][column]);
        const auto minor = [&](int a, int b, int d, int e) {
            return separate_product(f[a][b], f[d][e])
                - separate_product(f[d][b], f[a][e]);
        };
        // GradJ's exact operand order, including the negative cofactor entries.
        c[0][0] = minor(1, 1, 2, 2);
        c[0][1] = minor(2, 0, 1, 2);
        c[0][2] = minor(1, 0, 2, 1);
        c[1][0] = minor(2, 1, 0, 2);
        c[1][1] = minor(0, 0, 2, 2);
        c[1][2] = minor(2, 0, 0, 1);
        c[2][0] = minor(0, 1, 1, 2);
        c[2][1] = minor(1, 0, 0, 2);
        c[2][2] = minor(0, 0, 1, 1);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                c[row][column].store(cofactor[row][column]);
        // Eigen's det3 expands row zero, with separately rounded products
        // and a left-associated subtract/add. Do not rewrite as F:cofactor.
        const SolidPack a = separate_product(f[0][0],
            separate_product(f[1][1], f[2][2]) - separate_product(f[1][2], f[2][1]));
        const SolidPack b = separate_product(f[0][1],
            separate_product(f[1][0], f[2][2]) - separate_product(f[1][2], f[2][0]));
        const SolidPack d = separate_product(f[0][2],
            separate_product(f[1][0], f[2][1]) - separate_product(f[1][1], f[2][0]));
        ((a - b) + d).store(determinant);
#else
        Mat33 matrices[width];
        CorotatedCache caches[width];
        for (int lane = 0; lane < count; ++lane) {
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    matrices[lane](row, column) = deformation[row][column][lane];
        }
        volumetric_detail::update_corotated_cache_batch(matrices, caches, count);
#endif
        for (int lane = 0; lane < count; ++lane) {
            const std::size_t entry = begin + lane;
            measure[lane] = measures[entry];
            for (int row = 0; row < 3; ++row)
                shape[row][lane] = shape_gradients[entry][row];
#if !defined(__AVX512F__)
            const CorotatedCache& cache = caches[lane];
            determinant[lane] = cache.J_cache;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    rotation[row][column][lane] = cache.R_cache(row, column);
                    cofactor[row][column][lane] = cache.JFinvT_cache(row, column);
                }
            }
#endif
        }
        if (count < width) {
            pad_lanes(deformation, count); pad_lanes(rotation, count);
            pad_lanes(cofactor, count); pad_lanes(determinant, count);
            pad_lanes(shape, count); pad_lanes(measure, count);
        }
        const SolidPack q[3] = {
            SolidPack::load(shape[0]), SolidPack::load(shape[1]), SolidPack::load(shape[2])};
        const SolidPack volume = SolidPack::load(measure);
        const SolidPack volumetric_stress = bulk * (SolidPack::load(determinant) - one);
        SolidPack stress[3][3];
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column) {
                const SolidPack strain = SolidPack::load(deformation[row][column])
                    - SolidPack::load(rotation[row][column]);
                // Preserve the two independently rounded products in Eigen's
                // coefficient-wise scalar constitutive expression.
                stress[row][column] = separate_product(twice_mu, strain)
                    + separate_product(volumetric_stress, SolidPack::load(cofactor[row][column]));
            }
        // Eigen's three-component dot uses a pair sum then the third term.
        const SolidPack norm2 = (separate_product(q[0], q[0])
            + separate_product(q[1], q[1])) + separate_product(q[2], q[2]);
        const SolidPack diagonal = separate_product(twice_mu * norm2, volume);
        SolidPack u[3];
        for (int row = 0; row < 3; ++row) {
            const SolidPack c[3] = {SolidPack::load(cofactor[row][0]),
                SolidPack::load(cofactor[row][1]), SolidPack::load(cofactor[row][2])};
            const SolidPack g = matrix_row_product(row, stress[row], q);
            u[row] = matrix_row_product(row, c, q);
            (g * volume).store(gradient[row]);
        }
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column) {
                const SolidPack volumetric = (bulk * u[row]) * u[column];
                (separate_product(volumetric, volume)
                    + (row == column ? diagonal : zero)).store(block[row][column]);
            }
        for (int lane = 0; lane < count; ++lane)
            for (int row = 0; row < 3; ++row) {
                gradients[begin + lane][row] = gradient[row][lane];
                for (int column = 0; column < 3; ++column)
                    pbgs_blocks[begin + lane](row, column) = block[row][column][lane];
            }
    }
}

} // namespace ipc_simd

// Rigid contact kernels.
namespace ipc_simd {
namespace {

// Every hardware lane belongs to an independent contact. These are explicit
// SIMD products across contacts, separate from Eigen's within-matrix SIMD.
struct Pack {
#if defined(__AVX512F__)
    using Native = __m512d;
    static constexpr int width = 8;
#elif defined(__AVX2__)
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
    Pack() : Pack(0.0) {}
    explicit Pack(double x) {
#if defined(__AVX512F__)
        value = _mm512_set1_pd(x);
#elif defined(__AVX2__)
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
#if defined(__AVX512F__)
        return raw(_mm512_loadu_pd(x));
#elif defined(__AVX2__)
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
#if defined(__AVX512F__)
        _mm512_storeu_pd(x, value);
#elif defined(__AVX2__)
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
#if defined(__AVX512F__)
        return raw(_mm512_add_pd(a.value, b.value));
#elif defined(__AVX2__)
        return raw(_mm256_add_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vaddq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_add_pd(a.value, b.value));
#else
        return raw(a.value + b.value);
#endif
    }
    friend Pack operator*(Pack a, Pack b) {
#if defined(__AVX512F__)
        return raw(_mm512_mul_pd(a.value, b.value));
#elif defined(__AVX2__)
        return raw(_mm256_mul_pd(a.value, b.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
        return raw(vmulq_f64(a.value, b.value));
#elif defined(__SSE2__) || defined(_M_X64)
        return raw(_mm_mul_pd(a.value, b.value));
#else
        return raw(a.value * b.value);
#endif
    }
    Pack& operator+=(Pack b) { *this = *this + b; return *this; }
};

using Matrix = std::array<std::array<Pack, 3>, 3>;
using Vector = std::array<Pack, 3>;
using ReferencePacket = typename Eigen::internal::find_best_packet<double, 3>::type;
constexpr int reference_packet_width = Eigen::internal::unpacket_traits<ReferencePacket>::size;

Pack multiply_add(Pack a, Pack b, Pack c) {
#if defined(__AVX512F__)
    return Pack::raw(_mm512_fmadd_pd(a.value, b.value, c.value));
#elif defined(__AVX2__) && defined(__FMA__)
    return Pack::raw(_mm256_fmadd_pd(a.value, b.value, c.value));
#elif defined(__aarch64__) || defined(_M_ARM64)
    return Pack::raw(vfmaq_f64(c.value, a.value, b.value));
#else
    return a * b + c;
#endif
}

Pack select(Pack condition, Pack yes, Pack no) {
#if defined(__AVX512F__)
    return Pack::raw(_mm512_mask_blend_pd(
        _mm512_cmp_pd_mask(condition.value, _mm512_setzero_pd(), _CMP_NEQ_OQ),
        no.value, yes.value));
#elif defined(__AVX2__)
    return Pack::raw(_mm256_blendv_pd(no.value, yes.value,
        _mm256_cmp_pd(condition.value, _mm256_setzero_pd(), _CMP_NEQ_OQ)));
#elif defined(__aarch64__) || defined(_M_ARM64)
    return Pack::raw(vbslq_f64(vcgtq_f64(condition.value, vdupq_n_f64(0.0)),
        yes.value, no.value));
#elif defined(__SSE2__) || defined(_M_X64)
    const auto mask = _mm_cmpneq_pd(condition.value, _mm_setzero_pd());
    return Pack::raw(_mm_or_pd(_mm_and_pd(mask, yes.value), _mm_andnot_pd(mask, no.value)));
#else
    return condition.value != 0.0 ? yes : no;
#endif
}

Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix c;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if constexpr (reference_packet_width > 1)
                c[i][j] = (a[i][0] * b[0][j] + a[i][1] * b[1][j])
                    + a[i][2] * b[2][j];
            else
                c[i][j] = a[i][0] * b[0][j]
                    + (a[i][1] * b[1][j] + a[i][2] * b[2][j]);
        }
    return c;
}

Matrix multiply_column_major(const Matrix& a, const Matrix& b) {
    Matrix c;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            // Eigen's fixed 3x3 product evaluates its first two rows with
            // packet multiply-add, and its final scalar row with the balanced
            // three-term reduction. SIMD lanes here remain separate contacts.
            if (reference_packet_width > 1 && i < 2)
                c[i][j] = multiply_add(a[i][2], b[2][j],
                    multiply_add(a[i][1], b[1][j], a[i][0] * b[0][j]));
            else
                c[i][j] = a[i][0] * b[0][j]
                    + (a[i][1] * b[1][j] + a[i][2] * b[2][j]);
        }
    return c;
}

Matrix transpose(const Matrix& a) {
    Matrix b;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) b[i][j] = a[j][i];
    return b;
}

void add(Matrix& a, const Matrix& b) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) a[i][j] += b[i][j];
}

void symmetrize(Matrix& a) {
    // Match the reference's in-place, column-major Eigen assignment. Using
    // a snapshot would give a slightly different result for roundoff-sized
    // asymmetry in the input matrix.
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            a[i][j] = Pack(0.5) * (a[i][j] + a[j][i]);
}

struct Modes {
    bool tg, og, th, oh, mh;
    explicit Modes(RigidDerivativeMode mode)
        : tg(mode != RigidDerivativeMode::OrientationHessian),
          og(mode != RigidDerivativeMode::TranslationHessian),
          th(mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::TranslationHessian),
          oh(mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::OrientationHessian),
          mh(mode == RigidDerivativeMode::Full) {}
};

struct PreparedContact {
    std::array<Vec3, 4> gradients;
    std::array<Mat33, 4> jacobians;
    std::array<std::array<Mat33, 4>, 4> hessians;
    std::array<std::array<Mat33, 3>, 4> curvature;
    std::array<double, 4> friction_weights;
    Mat33 translation_hessian;
    void clear(const Modes& flags, bool cross_hessians) {
        friction_weights.fill(0.0);
        translation_hessian.setZero();
        for (int i = 0; i < 4; ++i) {
            gradients[i].setZero();
            if (flags.og) jacobians[i].setZero();
            if ((flags.th || flags.oh) && cross_hessians)
                for (auto& h : hessians[i]) h.setZero();
            if (flags.oh)
                for (auto& h : curvature[i]) h.setZero();
        }
    }
};

// Reuse node pullbacks within an immutable contact evaluation range. The
// kinematics objects remain unchanged until the range joins. Match their
// address and all reference-coordinate bits; retain at most sixteen nodes.
// Every public tile call and every range starts with an empty cache, so no
// body update, derivative call or helper range can inherit another's entries.
struct NodeKinematicsCache {
    struct Entry {
        const QuaternionOmegaKinematics* kinematics;
        Vec3 reference;
        Mat33 jacobian;
        std::array<Mat33, 3> curvature;
        bool has_curvature;
    };
    std::array<Entry, 16> entries;
    std::size_t used = 0, next = 0;

    const Entry& get(const Vec3& reference,
                     const QuaternionOmegaKinematics* kinematics,
                     bool with_curvature) {
        Entry* found = nullptr;
        for (std::size_t i = 0; i < used; ++i) {
            auto& entry = entries[i];
            if (entry.kinematics == kinematics
                && std::memcmp(entry.reference.data(), reference.data(),
                               3 * sizeof(double)) == 0) {
                found = &entry;
                break;
            }
        }
        if (!found) {
            const std::size_t slot = used < entries.size() ? used++ : next;
            if (used == entries.size()) next = (slot + 1) % entries.size();
            found = &entries[slot];
            found->reference = reference;
            found->kinematics = kinematics;
            found->jacobian = dx_domega(reference, *kinematics);
            found->has_curvature = false;
        }
        if (with_curvature && !found->has_curvature) {
            found->curvature = d2x_domega2(reference, *kinematics);
            found->has_curvature = true;
        }
        return *found;
    }
};

template <class Evaluation>
void prepare_contact(const RigidContactInput& input, const Evaluation& evaluation,
    RigidDerivativeMode mode, PreparedContact& prepared,
    FrozenFrictionContact* frozen, double dt, double eps_v,
    NodeKinematicsCache& node_kinematics) {
    constexpr bool segment = std::is_same_v<Evaluation, SegmentSegmentContactEvaluation>;
    const auto& x = input.positions;
    // The scalar builder validates dt/eps_v even for a geometrically inactive
    // admitted contact, but ignores its previous positions in that case.
    if (frozen) {
        if constexpr (segment)
            *frozen = make_segment_segment_frozen_friction_contact(x, input.previous_positions, evaluation, dt, eps_v);
        else
            *frozen = make_node_triangle_frozen_friction_contact(x, input.previous_positions, evaluation, dt, eps_v);
    }
    if (!evaluation.active) return;
    const int first = input.side == RigidBarrierSide::FirstPrimitive ? 0 : (segment ? 2 : 1);
    const int last = input.side == RigidBarrierSide::FirstPrimitive ? (segment ? 1 : 0) : 3;
    const Modes flags(mode);
    const auto gradient = [&](int i) {
        if constexpr (segment)
            return segment_segment_barrier_gradient(x[0], x[1], x[2], x[3], i, evaluation);
        else
            return node_triangle_barrier_gradient(x[0], x[1], x[2], x[3], i, evaluation);
    };
    const auto hessian = [&](int i, int j) {
        if constexpr (segment)
            return segment_segment_barrier_cross_hessian(x[0], x[1], x[2], x[3], i, j, evaluation);
        else
            return node_triangle_barrier_cross_hessian(x[0], x[1], x[2], x[3], i, j, evaluation);
    };
    if (flags.og && (!input.kinematics || (flags.oh && !input.kinematics->has_second_derivatives)))
        throw std::invalid_argument("SIMD rigid contact: requested orientation kinematics are missing.");
    for (int i = first; i <= last; ++i) {
        prepared.gradients[i] = gradient(i);
        if (flags.og) {
            const auto& node = node_kinematics.get(input.body_references[i],
                input.kinematics, flags.oh);
            prepared.jacobians[i] = node.jacobian;
            if (flags.oh) prepared.curvature[i] = node.curvature;
        }
    }
    if (mode == RigidDerivativeMode::TranslationHessian) {
        // These are the scalar solver's exact COM shortcuts. In particular,
        // translating all three triangle vertices gives the point self block.
        if constexpr (segment) {
            prepared.hessians[first][first] = hessian(first, first);
            prepared.hessians[first][last] = hessian(first, last);
            prepared.hessians[last][first] = prepared.hessians[first][last].transpose();
            prepared.hessians[last][last] = hessian(last, last);
        } else prepared.translation_hessian = hessian(0, 0);
    } else if (mode != RigidDerivativeMode::Gradient) {
        for (int i = first; i <= last; ++i)
            for (int j = mode == RigidDerivativeMode::OrientationHessian ? i : first; j <= last; ++j)
                prepared.hessians[i][j] = hessian(i, j);
    }
    if (frozen) {
        if (frozen->active) {
            const unsigned mask = input.body_role_mask ? input.body_role_mask
                : ((1u << (last + 1)) - (1u << first));
            for (int i = 0; i < 4; ++i) {
                if (!(mask & (1u << i))) continue;
                prepared.friction_weights[i] = frozen->weights[i];
                if (flags.og && updates_rigid_orientation(input.update_mode)
                    && (i < first || i > last))
                    prepared.jacobians[i] = node_kinematics.get(input.body_references[i],
                        input.kinematics, false).jacobian;
            }
        }
        // Existing SIMD friction then computes the common relative gradient
        // and Hessian. Signed body weights are pulled back only once below.
        frozen->weights[0] = 1.0;
    }
}

template <class Getter>
Pack gather(int active, const Getter& get) {
    alignas(64) double values[Pack::width];
    for (int lane = 0; lane < Pack::width; ++lane)
        values[lane] = get(std::min(lane, active - 1));
    return Pack::load(values);
}



} // namespace

static void rigid_contact_derivatives_tile_cached(const RigidContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, RigidDerivativeMode mode,
    RigidContactOutput* outputs, NodeKinematicsCache& node_kinematics) {
    assert(count <= contact_tile_width);
    if (count == 0) return;
    // Every packet of an all-NT translation tile bypasses the cross-block
    // loop below. Only its point self Hessian is live; mixed tiles retain
    // all cross-block initialization, including NT lanes in mixed packets.
    const bool point_translation_tile = mode == RigidDerivativeMode::TranslationHessian
        && std::none_of(inputs, inputs + count,
            [](const RigidContactInput& input) { return input.segment_segment; });
    std::array<PreparedContact, contact_tile_width> prepared;
    std::array<FrozenFrictionContact, contact_tile_width> frozen;
    std::array<Vec3, contact_tile_width> friction_gradient;
    std::array<Mat33, contact_tile_width> friction_hessian;
    const Modes flags(mode);
    for (std::size_t e = 0; e < count; ++e) {
        // Tail packets and gradient/COM-only requests never read the unused
        // records or orientation-Hessian fields. Initialize only live work.
        prepared[e].clear(flags, !point_translation_tile);
        outputs[e] = RigidContactOutput{};
        const auto& input = inputs[e];
        const auto& x = input.positions;
        auto* contact = friction != 0.0 ? &frozen[e] : nullptr;
        if (input.segment_segment) {
            SegmentSegmentContactEvaluation evaluation;
            if (contact) evaluation = make_segment_segment_contact_evaluation(x, d_hat, k_barrier, 1e-12, nullptr, input.exact_computation_fallback);
            else {
                evaluation.dr = segment_segment_distance(x[0], x[1], x[2], x[3], 1e-12, input.exact_computation_fallback);
                evaluation.d_hat = d_hat;
                evaluation.active = !(d_hat > 0.0 && evaluation.dr.distance >= d_hat);
                if (evaluation.active) {
                    evaluation.b_prime = scalar_barrier_gradient(evaluation.dr.distance, d_hat);
                    if (mode != RigidDerivativeMode::Gradient)
                        evaluation.b_double_prime = scalar_barrier_hessian(evaluation.dr.distance, d_hat);
                }
            }
            prepare_contact(input, evaluation, mode, prepared[e], contact, dt, eps_v, node_kinematics);
        } else {
            NodeTriangleContactEvaluation evaluation;
            if (contact) evaluation = make_node_triangle_contact_evaluation(x, d_hat, k_barrier, 1e-12, nullptr, input.exact_computation_fallback);
            else {
                evaluation.dr = node_triangle_distance(x[0], x[1], x[2], x[3]);
                evaluation.d_hat = d_hat;
                evaluation.active = !(d_hat > 0.0 && evaluation.dr.distance >= d_hat);
                if (evaluation.active) {
                    evaluation.b_prime = scalar_barrier_gradient(evaluation.dr.distance, d_hat);
                    if (mode != RigidDerivativeMode::Gradient)
                        evaluation.b_double_prime = scalar_barrier_hessian(evaluation.dr.distance, d_hat);
                }
            }
            prepare_contact(input, evaluation, mode, prepared[e], contact, dt, eps_v, node_kinematics);
        }
    }
    if (friction != 0.0) {
        std::array<int, contact_tile_width> roles{};
        friction_derivatives_tile(frozen.data(), roles.data(), count,
            friction, dt * dt, friction_gradient.data(), friction_hessian.data());
    }

    for (std::size_t begin = 0; begin < count; begin += Pack::width) {
        const int active = static_cast<int>(std::min<std::size_t>(Pack::width, count - begin));
        // NT COM uses the point self block below. Its temporary cross-block
        // sum is overwritten, so an all-NT packet needs none of that work.
        const bool point_translation_packet = mode == RigidDerivativeMode::TranslationHessian
            && std::none_of(inputs + begin, inputs + begin + active,
                [](const RigidContactInput& input) { return input.segment_segment; });
        const auto scalar = [&](const auto& get) {
            return gather(active, [&](int lane) { return get(prepared[begin + lane]); });
        };
        const auto matrix = [&](const auto& get) {
            Matrix result;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    result[r][c] = scalar([&](const auto& p) { return get(p)(r, c); });
            return result;
        };
        const auto store_vector = [&](const Vector& value, const auto& get) {
            for (int i = 0; i < 3; ++i) {
                alignas(64) double lanes[Pack::width];
                value[i].store(lanes);
                for (int lane = 0; lane < active; ++lane) get(outputs[begin + lane])[i] = lanes[lane];
            }
        };
        const auto store_matrix = [&](const Matrix& value, const auto& get) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    alignas(64) double lanes[Pack::width];
                    value[r][c].store(lanes);
                    for (int lane = 0; lane < active; ++lane) get(outputs[begin + lane])(r, c) = lanes[lane];
                }
        };
        std::array<Matrix, 4> J;
        for (int i = 0; i < 4; ++i)
            if (flags.og) J[i] = matrix([&](const auto& p) -> const Mat33& { return p.jacobians[i]; });
        Vector tg, og;
        Matrix tt = matrix([](const auto& p) -> const Mat33& { return p.translation_hessian; }), to, oo;
        for (int i = 0; i < 4; ++i) {
            Vector g;
            for (int r = 0; r < 3; ++r) g[r] = scalar([&](const auto& p) { return p.gradients[i][r]; });
            for (int r = 0; r < 3; ++r) {
                if (flags.tg) tg[r] += g[r];
                if (flags.og) og[r] += (J[i][0][r] * g[0] + J[i][1][r] * g[1]) + J[i][2][r] * g[2];
            }
            if (mode != RigidDerivativeMode::Gradient && !point_translation_packet) {
                for (int j = mode == RigidDerivativeMode::OrientationHessian ? i : 0; j < 4; ++j) {
                    const Matrix H = matrix([&](const auto& p) -> const Mat33& { return p.hessians[i][j]; });
                    if (flags.th) add(tt, H);
                    if (flags.mh) add(to, multiply_column_major(H, J[j]));
                    if (flags.oh) {
                        const Matrix contribution = multiply_column_major(multiply(transpose(J[i]), H), J[j]);
                        if (mode == RigidDerivativeMode::OrientationHessian && i != j) {
                            Matrix pair = contribution;
                            add(pair, transpose(contribution));
                            add(oo, pair);
                        } else add(oo, contribution);
                    }
                }
            }
            if (flags.oh)
                for (int axis = 0; axis < 3; ++axis) {
                    const Matrix curvature = matrix([&](const auto& p) -> const Mat33& { return p.curvature[i][axis]; });
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c) oo[r][c] += g[axis] * curvature[r][c];
                }
        }
        if (mode == RigidDerivativeMode::TranslationHessian) {
            const Pack segment = gather(active, [&](int lane) {
                return inputs[begin + lane].segment_segment ? 1.0 : 0.0;
            });
            const Matrix point_block = matrix([](const auto& p) -> const Mat33& {
                return p.translation_hessian;
            });
            // NT translation uses its self block directly in the reference.
            // Adding the unused zero cross blocks would erase negative zeros.
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    tt[r][c] = select(segment, tt[r][c], point_block[r][c]);
        }
        if (flags.th) symmetrize(tt);
        if (flags.oh) symmetrize(oo);
        if (flags.tg) store_vector(tg, [](auto& o) -> Vec3& { return o.barrier.translation_gradient; });
        if (flags.og) store_vector(og, [](auto& o) -> Vec3& { return o.barrier.orientation_gradient; });
        if (flags.th) store_matrix(tt, [](auto& o) -> Mat33& { return o.barrier.translation_translation_hessian; });
        if (flags.mh) store_matrix(to, [](auto& o) -> Mat33& { return o.barrier.translation_orientation_hessian; });
        if (flags.oh) store_matrix(oo, [](auto& o) -> Mat33& { return o.barrier.orientation_orientation_hessian; });

        if (friction == 0.0) continue;
        Pack weight;
        Matrix FJ;
        for (int i = 0; i < 4; ++i) {
            const Pack w = scalar([&](const auto& p) { return p.friction_weights[i]; });
            weight += w;
            if (flags.og)
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) FJ[r][c] += w * J[i][r][c];
        }
        Vector fg, ftg, fog;
        Matrix fH, ftt, fto, foo;
        for (int r = 0; r < 3; ++r) {
            fg[r] = gather(active, [&](int lane) { return friction_gradient[begin + lane][r]; });
            for (int c = 0; c < 3; ++c)
                fH[r][c] = gather(active, [&](int lane) { return friction_hessian[begin + lane](r, c); });
        }
        const Pack translation = gather(active, [&](int lane) { return updates_rigid_translation(inputs[begin + lane].update_mode) ? 1.0 : 0.0; });
        const Pack orientation = gather(active, [&](int lane) { return updates_rigid_orientation(inputs[begin + lane].update_mode) ? 1.0 : 0.0; });
        Matrix fHj;
        if (flags.mh || flags.oh) fHj = multiply_column_major(fH, FJ);
        if (flags.oh) foo = multiply(transpose(FJ), fHj);
        for (int r = 0; r < 3; ++r) {
            if (flags.tg) ftg[r] = translation * (weight * fg[r]);
            if (flags.og) fog[r] = orientation * ((FJ[0][r] * fg[0] + FJ[1][r] * fg[1]) + FJ[2][r] * fg[2]);
            for (int c = 0; c < 3; ++c) {
                if (flags.th) ftt[r][c] = translation * ((weight * fH[r][c]) * weight);
                if (flags.mh) fto[r][c] = (translation * orientation) * (weight * fHj[r][c]);
                if (flags.oh) foo[r][c] = orientation * foo[r][c];
            }
        }
        // Frozen friction intentionally has no quaternion-curvature term.
        if (flags.tg) store_vector(ftg, [](auto& o) -> Vec3& { return o.friction.translation_gradient; });
        if (flags.og) store_vector(fog, [](auto& o) -> Vec3& { return o.friction.orientation_gradient; });
        if (flags.th) store_matrix(ftt, [](auto& o) -> Mat33& { return o.friction.translation_translation_hessian; });
        if (flags.mh) store_matrix(fto, [](auto& o) -> Mat33& { return o.friction.translation_orientation_hessian; });
        if (flags.oh) store_matrix(foo, [](auto& o) -> Mat33& { return o.friction.orientation_orientation_hessian; });
    }
}

void rigid_contact_derivatives_tile(const RigidContactInput* inputs,
    std::size_t count, double d_hat, double k_barrier, double friction,
    double dt, double eps_v, RigidDerivativeMode mode,
    RigidContactOutput* outputs) {
    NodeKinematicsCache node_kinematics;
    rigid_contact_derivatives_tile_cached(inputs, count, d_hat, k_barrier,
        friction, dt, eps_v, mode, outputs, node_kinematics);
}

RigidContactOutput rigid_contact_derivatives(
    int rb, const RefMesh& ref_mesh, const DeformedState& state,
    const BroadPhase::Cache& cache,
    const std::vector<int>& nt_pair_indices,
    const std::vector<int>& ss_pair_indices,
    const std::vector<int>& node_to_rb_local,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& omega_new,
    const SimParams& params, double dt, RigidDerivativeMode mode,
    const QuaternionOmegaKinematics* kinematics, bool cooperative,
    const std::function<void()>* leader_work) {
    RigidContactOutput total;
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0) {
        if (leader_work) (*leader_work)();
        return total;
    }
    const std::size_t nt_count = nt_pair_indices.size();
    const std::size_t count = nt_count + ss_pair_indices.size();
    QuaternionOmegaKinematics owned_kinematics;
    if (!kinematics && count != 0 && mode != RigidDerivativeMode::TranslationHessian) {
        owned_kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt,
            mode != RigidDerivativeMode::Gradient);
        kinematics = &owned_kinematics;
    }
    const double distance2 = params.d_hat * params.d_hat;
    const auto gather_contact = [&](std::size_t index, RigidContactInput& input) {
        std::array<int, 4> nodes;
        const bool segment = index >= nt_count;
        if (segment) {
            const auto& pair = cache.ss_pairs[ss_pair_indices[index - nt_count]];
            nodes = {pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
        } else {
            const auto& pair = cache.nt_pairs[nt_pair_indices[index]];
            nodes = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
        }
        const int first_owner = owning_rb_for_node(ref_mesh.node_to_rb, nodes[0]);
        const int second_owner = owning_rb_for_node(ref_mesh.node_to_rb, nodes[segment ? 2 : 1]);
        if (first_owner == second_owner || (first_owner != rb && second_owner != rb)) return false;
        const bool nearby = segment
            ? segment_aabbs_within_distance(positions[nodes[0]], positions[nodes[1]], positions[nodes[2]], positions[nodes[3]], distance2)
            : node_triangle_aabbs_within_distance(positions[nodes[0]], positions[nodes[1]], positions[nodes[2]], positions[nodes[3]], distance2);
        if (!nearby) return false;
        input.segment_segment = segment;
        input.exact_computation_fallback = params.exact_computation_fallback;
        input.side = first_owner == rb ? RigidBarrierSide::FirstPrimitive : RigidBarrierSide::SecondPrimitive;
        input.kinematics = kinematics;
        input.update_mode = ref_mesh.rb_update_modes[rb];
        input.body_role_mask = 0;
        for (int role = 0; role < 4; ++role) {
            const int node = nodes[role];
            input.positions[role] = positions[node];
            input.body_references[role].setZero();
            if (owning_rb_for_node(ref_mesh.node_to_rb, node) == rb) {
                input.body_role_mask |= 1u << role;
                input.body_references[role] = ref_mesh.ref_positions[rb][node_to_rb_local[node]];
            }
            if (params.friction_coefficient != 0.0)
                input.previous_positions[role] = state.deformed_positions[node];
        }
        return true;
    };
    const auto evaluate = [&](std::size_t begin, std::size_t end, const auto& emit) {
        // Input kinematics and body references are fixed for this range.
        // Helpers own separate range stacks; all finish before the body moves.
        NodeKinematicsCache node_kinematics;
        std::array<RigidContactInput, contact_tile_width> inputs;
        std::array<RigidContactOutput, contact_tile_width> outputs;
        std::array<std::size_t, contact_tile_width> indices;
        std::size_t active = 0;
        const auto flush = [&] {
            if (active == 0) return;
            rigid_contact_derivatives_tile_cached(inputs.data(), active, params.d_hat,
                params.k_barrier, params.friction_coefficient, dt,
                params.friction_velocity_epsilon, mode, outputs.data(), node_kinematics);
            for (std::size_t e = 0; e < active; ++e) emit(indices[e], outputs[e]);
            active = 0;
        };
        for (std::size_t index = begin; index < end; ++index) {
            if (!gather_contact(index, inputs[active])) continue;
            indices[active++] = index;
            if (active == contact_tile_width) flush();
        }
        flush();
    };
    solver_detail::ordered_rigid_contact_ranges(static_cast<int>(count),
        cooperative, mode,
        [&](int begin, int end, const auto& emit) {
            evaluate(begin, end, [&](std::size_t index, const auto& value) {
                emit(index, value.barrier, value.friction);
            });
        }, total.barrier, params.friction_coefficient != 0.0 ? &total.friction : nullptr,
        static_cast<int>(contact_tile_width), leader_work);
    return total;
}

} // namespace ipc_simd

// Vertex assembly.
namespace solver_detail {

void prepare_general_simd_batch(
    const int* nodes, std::size_t count, const RefMesh& mesh,
    const std::vector<IncidentTriangles>& incident,
    const std::vector<ShapeGrads>& shape_gradients,
    const std::vector<Pin>& pins, const PinMap& pin_map,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>& predicted, const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask,
    GeneralSimdVertexSystem* outputs,
    const GeneralSimdMaterials* materials) {
    constexpr std::size_t width = ipc_simd::tile_width;
    assert(count <= width);
    const double dt2 = params.dt2();
    std::array<ipc_simd::PointInput, width> points;
    std::array<std::size_t, width> point_owners;
    std::array<Vec3, width> gradients;
    std::array<Mat33, width> hessians;
    std::size_t cloth_count = 0;
    for (std::size_t i = 0; i < count; ++i)
        cloth_count += !solid_mask[nodes[i]];
    std::size_t cloth_entry = 0, solid_entry = cloth_count;
    for (std::size_t i = 0; i < count; ++i) {
        const int node = nodes[i];
        const auto entry = solid_mask[node] ? solid_entry++ : cloth_entry++;
        point_owners[entry] = i;
        points[entry].mass = mesh.mass[node];
        points[entry].position = positions[node];
        points[entry].predicted_position = predicted[node];
        if (pin_map[node] >= 0) points[entry].pin_target = pins[pin_map[node]].target_position;
        outputs[i] = GeneralSimdVertexSystem{};
    }
    // A color batch can contain both materials. Preserve the cloth point
    // kernel, and use scalar-solid rounding only for the solid entries.
    ipc_simd::point_derivatives_tile(points.data(), cloth_count, params.gravity,
        params.kpin, dt2, gradients.data(), hessians.data());
    ipc_simd::solid_point_derivatives_tile(points.data() + cloth_count, count - cloth_count,
        params.gravity, params.kpin, dt2, gradients.data() + cloth_count, hessians.data() + cloth_count);
    for (std::size_t i = 0; i < count; ++i) {
        outputs[point_owners[i]].gradient = gradients[i];
        outputs[point_owners[i]].hessian = hessians[i];
    }

    // Pack across complete independent nodes, retaining every node's incident
    // element order. Distribute each tile before reusing its private storage.
    std::array<Vec3, 4 * width> gathered;
    std::array<std::size_t, width> owners;
    std::array<double, width> measures;
    std::size_t pending = 0;
    const auto distribute = [&] {
        for (std::size_t e = 0; e < pending; ++e) {
            outputs[owners[e]].gradient += dt2 * gradients[e];
            outputs[owners[e]].hessian += dt2 * hessians[e];
        }
        pending = 0;
    };
    if (materials) {
        // Resolve rest data and topology once between solver calls. Only live
        // positions are gathered here. Singleton batches can pass a contiguous
        // incident span straight to the kernel without copying static values.
        const auto gather_positions = [&](const auto& records, std::size_t begin,
                                          std::size_t size, std::size_t offset) {
            for (std::size_t e = 0; e < size; ++e) {
                const auto& corners = records.nodes[begin + e];
                for (std::size_t corner = 0; corner < corners.size(); ++corner)
                    gathered[corners.size() * (offset + e) + corner] = positions[corners[corner]];
            }
        };
        const auto assemble_elements = [&](const auto& records, const auto& kernel) {
            using Matrix = typename std::decay_t<decltype(records.dm_inverse)>::value_type;
            using Shape = typename std::decay_t<decltype(records.shape_gradients)>::value_type;
            if (count == 1) {
                const auto end = records.node_offsets[nodes[0] + 1];
                for (auto e = records.node_offsets[nodes[0]]; e < end; e += width) {
                    pending = std::min(width, end - e);
                    gather_positions(records, e, pending, 0);
                    std::fill_n(owners.data(), pending, std::size_t(0));
                    kernel(records.dm_inverse.data() + e, records.measures.data() + e,
                        records.shape_gradients.data() + e);
                    distribute();
                }
                return;
            }
            std::array<Matrix, width> inverse;
            std::array<Shape, width> shapes;
            const auto flush = [&] {
                if (!pending) return;
                kernel(inverse.data(), measures.data(), shapes.data());
                distribute();
            };
            for (std::size_t i = 0; i < count; ++i) {
                auto e = records.node_offsets[nodes[i]];
                const auto end = records.node_offsets[nodes[i] + 1];
                while (e < end) {
                    const auto size = std::min(width - pending, end - e);
                    gather_positions(records, e, size, pending);
                    std::copy_n(records.dm_inverse.data() + e, size, inverse.data() + pending);
                    std::copy_n(records.measures.data() + e, size, measures.data() + pending);
                    std::copy_n(records.shape_gradients.data() + e, size, shapes.data() + pending);
                    std::fill_n(owners.data() + pending, size, i);
                    e += size; pending += size;
                    if (pending == width) flush();
                }
            }
            flush();
        };
        assemble_elements(materials->triangles, [&](const Mat22* inverse,
                                                   const double* area, const Vec2* shape) {
            // Share basic v2's membrane arithmetic as well as its physics.
            ipc_simd::corotated_derivatives_tile(gathered.data(), inverse, area,
                shape, pending, params.mu, params.lambda, gradients.data(), hessians.data());
        });
        assemble_elements(materials->tets, [&](const Mat33* inverse,
                                              const double* measure, const Vec3* shape) {
            ipc_simd::solid_derivatives_tile(gathered.data(), inverse, measure,
                shape, pending, params.solid_mu, params.solid_lambda, gradients.data(), hessians.data());
        });
        if (params.kB > 0.0) {
            const auto& records = materials->hinges;
            if (count == 1) {
                const auto end = records.node_offsets[nodes[0] + 1];
                for (auto e = records.node_offsets[nodes[0]]; e < end; e += width) {
                    pending = std::min(width, end - e);
                    gather_positions(records, e, pending, 0);
                    std::fill_n(owners.data(), pending, std::size_t(0));
                    ipc_simd::bending_derivatives_tile(gathered.data(),
                        records.active_nodes.data() + e, records.coefficients.data() + e,
                        records.rest_angles.data() + e, pending, params.kB,
                        gradients.data(), hessians.data());
                    distribute();
                }
            } else {
                std::array<int, width> roles;
                std::array<double, width> angles;
                const auto flush = [&] {
                    if (!pending) return;
                    ipc_simd::bending_derivatives_tile(gathered.data(), roles.data(),
                        measures.data(), angles.data(), pending, params.kB,
                        gradients.data(), hessians.data());
                    distribute();
                };
                for (std::size_t i = 0; i < count; ++i) {
                    auto e = records.node_offsets[nodes[i]];
                    const auto end = records.node_offsets[nodes[i] + 1];
                    while (e < end) {
                        const auto size = std::min(width - pending, end - e);
                        gather_positions(records, e, size, pending);
                        std::copy_n(records.active_nodes.data() + e, size, roles.data() + pending);
                        std::copy_n(records.coefficients.data() + e, size, measures.data() + pending);
                        std::copy_n(records.rest_angles.data() + e, size, angles.data() + pending);
                        std::fill_n(owners.data() + pending, size, i);
                        e += size; pending += size;
                        if (pending == width) flush();
                    }
                }
                flush();
            }
        }
    } else {
        std::array<Mat22, width> triangle_dm;
        std::array<Vec2, width> triangle_shapes;
        const auto flush_triangles = [&] {
            if (!pending) return;
            ipc_simd::corotated_derivatives_tile(gathered.data(), triangle_dm.data(),
                measures.data(), triangle_shapes.data(), pending, params.mu,
                params.lambda, gradients.data(), hessians.data());
            distribute();
        };
        for (std::size_t i = 0; i < count; ++i) {
            if (solid_mask[nodes[i]]) continue;
            for (const auto& [triangle, corner] : incident[nodes[i]]) {
                if (triangle < 0 || static_cast<std::size_t>(triangle) >= mesh.Dm_inverse.size()) continue;
                for (int j = 0; j < 3; ++j)
                    gathered[3 * pending + j] = positions[mesh.tris[3 * triangle + j]];
                triangle_dm[pending] = mesh.Dm_inverse[triangle];
                triangle_shapes[pending] = shape_gradients[triangle][corner];
                measures[pending] = mesh.area[triangle];
                owners[pending++] = i;
                if (pending == width) flush_triangles();
            }
        }
        flush_triangles();

        std::array<Mat33, width> tet_dm;
        std::array<Vec3, width> tet_shapes;
        const auto flush_tets = [&] {
            if (!pending) return;
            ipc_simd::solid_derivatives_tile(gathered.data(), tet_dm.data(),
                measures.data(), tet_shapes.data(), pending, params.solid_mu,
                params.solid_lambda, gradients.data(), hessians.data());
            distribute();
        };
        for (std::size_t i = 0; i < count; ++i) {
            if (!solid_mask[nodes[i]]) continue;
            for (const auto& [tet, corner] : mesh.tet_adj[nodes[i]]) {
                const auto& rest = mesh.tet_rest_data[tet];
                for (int j = 0; j < 4; ++j)
                    gathered[4 * pending + j] = positions[mesh.tets[4 * tet + j]];
                tet_dm[pending] = rest.Dm_inverse;
                tet_shapes[pending] = rest.grad_N[corner];
                measures[pending] = rest.measure;
                owners[pending++] = i;
                if (pending == width) flush_tets();
            }
        }
        flush_tets();

        if (params.kB > 0.0) {
            std::array<int, width> roles;
            std::array<double, width> angles;
            const auto flush_hinges = [&] {
                if (!pending) return;
                ipc_simd::bending_derivatives_tile(gathered.data(), roles.data(),
                    measures.data(), angles.data(), pending, params.kB,
                    gradients.data(), hessians.data());
                distribute();
            };
            for (std::size_t i = 0; i < count; ++i) {
                if (solid_mask[nodes[i]]) continue;
                const auto found = mesh.hinge_adj.find(nodes[i]);
                if (found == mesh.hinge_adj.end()) continue;
                for (const auto& [hinge, role] : found->second) {
                    const auto& value = mesh.hinges[hinge];
                    for (int j = 0; j < 4; ++j)
                        gathered[4 * pending + j] = positions[value.v[j]];
                    roles[pending] = role;
                    measures[pending] = value.c_e;
                    angles[pending] = value.bar_theta;
                    owners[pending++] = i;
                    if (pending == width) flush_hinges();
                }
            }
            flush_hinges();
        }
    }

    if (params.k_sdf > 0.0 && (!params.sdf_planes.empty()
        || !params.sdf_cylinders.empty() || !params.sdf_spheres.empty())) {
        std::array<Vec3, width> current, prior;
        std::array<physics_detail::SdfDerivatives, width> values;
        std::size_t active = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const int node = nodes[i];
            if (solid_mask[node] && !surface_mask[node]) continue;
            owners[active] = i;
            current[active] = positions[node];
            if (params.friction_coefficient > 0.0) prior[active] = (*previous)[node];
            ++active;
        }
        physics_detail::compute_sdf_derivatives_tile(params, current.data(),
            params.friction_coefficient > 0.0 ? prior.data() : nullptr,
            active, values.data());
        for (std::size_t e = 0; e < active; ++e) {
            auto& output = outputs[owners[e]];
            output.gradient += dt2 * values[e].gradient;
            output.hessian += dt2 * values[e].hessian;
            if (solid_mask[nodes[owners[e]]]) {
                // Solid reference adds SDF friction after mesh friction.
                output.sdf_friction_gradient = values[e].friction_gradient;
                output.sdf_friction_hessian = values[e].friction_hessian;
            } else if (params.friction_coefficient > 0.0) {
                output.gradient += values[e].friction_gradient;
                output.hessian += values[e].friction_hessian;
            }
        }
    }
}

void accumulate_general_simd_contacts(
    int node, bool solid, const BroadPhase::Cache& cache,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask, bool cooperative,
    GeneralSimdVertexSystem& output,
    safe_step_detail::VertexAabbRejections* rejections,
    const std::function<void()>* leader_work) {
    if (rejections) rejections->distance = 0.0;
    const double scale = params.dt2() * params.k_barrier;
    GeneralSimdVertexSystem contact;
    // Cloth v2 evaluates positive-distance candidates even with a zero barrier
    // coefficient. Solids retain their original positive-stiffness condition.
    if (params.d_hat > 0.0 && (!solid || params.k_barrier > 0.0)) {
        constexpr std::size_t width = ipc_simd::contact_tile_width;
        const auto& nt = cache.vertex_nt[node];
        const auto& ss = cache.vertex_ss[node];
        const std::size_t total = nt.size() + ss.size();
        if (rejections) {
            rejections->clear.assign(total, 0);
            rejections->distance = params.d_hat;
        }
        const bool use_derivative_mask = std::isfinite(scale);
        const auto gather = [&](std::size_t index, ipc_simd::MeshContactInput& input) {
            std::array<int, 4> nodes;
            const bool segment = index >= nt.size();
            const auto& entry = segment ? ss[index - nt.size()] : nt[index];
            if (segment) {
                const auto& pair = cache.ss_pairs[entry.pair_index];
                nodes = {pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
            } else {
                const auto& pair = cache.nt_pairs[entry.pair_index];
                nodes = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
            }
            if (solid && !cache.excludes_tet_interior_nt_queries) {
                for (int n : nodes)
                    if (solid_mask[n] && !surface_mask[n]) return false;
            }
            bool clear = false;
            const double d2 = params.d_hat * params.d_hat;
            const bool active = segment
                ? segment_aabbs_within_distance(positions[nodes[0]], positions[nodes[1]],
                    positions[nodes[2]], positions[nodes[3]], d2, &clear)
                : node_triangle_aabbs_within_distance(positions[nodes[0]], positions[nodes[1]],
                    positions[nodes[2]], positions[nodes[3]], d2, &clear);
            // Only the conservative distance certificate may suppress CCD.
            // A derivative mask, an interior filter or a plane-side rejection
            // alone does not prove that the proposed step is collision-free.
            if (rejections) rejections->clear[index] = clear;
            if (!active) return false;
            input.role = entry.dof;
            input.segment_segment = segment;
            for (int j = 0; j < 4; ++j) {
                input.positions[j] = positions[nodes[j]];
                if (params.friction_coefficient != 0.0)
                    input.previous_positions[j] = (*previous)[nodes[j]];
            }
            return true;
        };
        const auto accumulate_value = [&](const ipc_simd::MeshContactOutput& value) {
            auto& target = solid ? contact : output;
            target.gradient += scale * value.gradient;
            target.hessian += scale * value.hessian;
            if (params.friction_coefficient != 0.0) {
                if (solid) {
                    contact.sdf_friction_gradient += value.friction_gradient;
                    contact.sdf_friction_hessian += value.friction_hessian;
                } else {
                    output.gradient += value.friction_gradient;
                    output.hessian += value.friction_hessian;
                }
            }
        };
        if (cooperative && total >= 32 && omp_get_num_threads() > 1) {
            ordered_nodal_contact_ranges(static_cast<int>(total), true,
                params.friction_coefficient != 0.0,
                [&](int begin, int end, const auto& emit) {
                    for (std::size_t first = begin; first < static_cast<std::size_t>(end); first += width) {
                        std::array<ipc_simd::MeshContactInput, width> inputs;
                        std::array<ipc_simd::MeshContactOutput, width> values;
                        std::array<unsigned char, width> active;
                        std::array<std::size_t, width> indices;
                        std::size_t count = 0;
                        const auto stop = std::min(static_cast<std::size_t>(end), first + width);
                        for (std::size_t index = first; index < stop; ++index) {
                            if (gather(index, inputs[count])) indices[count++] = index;
                        }
                        ipc_simd::general_mesh_contact_derivatives_tile(inputs.data(), count,
                            params.d_hat, params.k_barrier, params.friction_coefficient,
                            params.dt(), params.friction_velocity_epsilon, values.data(),
                            use_derivative_mask ? active.data() : nullptr);
                        for (std::size_t e = 0; e < count; ++e)
                            if (!use_derivative_mask || active[e])
                                emit(indices[e], values[e].gradient, values[e].hessian,
                                    values[e].friction_gradient, values[e].friction_hessian);
                    }
                }, [&](const Vec3& gradient, const Mat33& hessian,
                       const Vec3& friction_gradient, const Mat33& friction_hessian) {
                    ipc_simd::MeshContactOutput value;
                    value.gradient = gradient; value.hessian = hessian;
                    value.friction_gradient = friction_gradient; value.friction_hessian = friction_hessian;
                    accumulate_value(value);
                }, leader_work);
        } else {
            if (leader_work) (*leader_work)();
            std::array<ipc_simd::MeshContactInput, width> inputs;
            std::array<ipc_simd::MeshContactOutput, width> values;
            std::array<unsigned char, width> active;
            std::size_t pending = 0;
            const auto flush = [&] {
                if (!pending) return;
                ipc_simd::general_mesh_contact_derivatives_tile(inputs.data(), pending,
                    params.d_hat, params.k_barrier, params.friction_coefficient,
                    params.dt(), params.friction_velocity_epsilon, values.data(),
                    use_derivative_mask ? active.data() : nullptr);
                for (std::size_t e = 0; e < pending; ++e)
                    if (!use_derivative_mask || active[e]) accumulate_value(values[e]);
                pending = 0;
            };
            // Rejected candidates do not consume lanes or force a partial
            // packet. Match basic v2's full-row active-contact compaction.
            for (std::size_t index = 0; index < total; ++index)
                if (gather(index, inputs[pending]) && ++pending == width) flush();
            flush();
        }
    } else if (leader_work) {
        (*leader_work)();
    }
    if (solid) {
        output.gradient += contact.gradient;
        output.hessian += contact.hessian;
        if (params.friction_coefficient != 0.0) {
            output.gradient += contact.sdf_friction_gradient;
            output.hessian += contact.sdf_friction_hessian;
            output.gradient += output.sdf_friction_gradient;
            output.hessian += output.sdf_friction_hessian;
        }
    }
}

} // namespace solver_detail
