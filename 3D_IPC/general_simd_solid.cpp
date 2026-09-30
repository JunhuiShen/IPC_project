#include "general_simd_solid.h"
#include "volumetric_corotated_energy.h"

#include <algorithm>
#include <cassert>
#include <type_traits>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

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
        // Batch the signed QR-SVD as well as the surrounding material work.
        // The helper preserves the scalar reference's rounding boundaries,
        // including determinant/cofactor calculations and partial batches.
        Mat33 matrices[width];
        CorotatedCache caches[width];
        for (int lane = 0; lane < count; ++lane) {
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    matrices[lane](row, column) = deformation[row][column][lane];
        }
        volumetric_detail::update_corotated_cache_batch(matrices, caches, count);
        for (int lane = 0; lane < count; ++lane) {
            const std::size_t entry = begin + lane;
            const CorotatedCache& cache = caches[lane];
            measure[lane] = measures[entry];
            determinant[lane] = cache.J_cache;
            for (int row = 0; row < 3; ++row) {
                shape[row][lane] = shape_gradients[entry][row];
                for (int column = 0; column < 3; ++column) {
                    rotation[row][column][lane] = cache.R_cache(row, column);
                    cofactor[row][column][lane] = cache.JFinvT_cache(row, column);
                }
            }
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
