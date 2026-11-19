#pragma once
#include <cmath>
#include <cfloat>
#include <cstdint>

#ifdef __CUDACC__
#define IPC_LOCAL_SOLVE32_HD __host__ __device__
#else
#define IPC_LOCAL_SOLVE32_HD
#endif

namespace ipc_gpu { namespace local_solve_fp32 {

// CPU-style inverse/pseudoinverse policy, specialized to IEEE binary32.
// This is not Eigen's exact operation sequence or its binary64 rank decision.
// No eigenvalue clipping, SPD projection, or descent test is performed.
enum Status : int {
    Adjugate = 0,
    MinimumNormQR = 1,
    Zero = 2,
    InvalidInput = -1,
    NumericalFailure = -2
};

IPC_LOCAL_SOLVE32_HD inline uint32_t float_bits(float value) {
    union Bits { float value; uint32_t bits; } repr;
    repr.value = value;
    return repr.bits;
}

IPC_LOCAL_SOLVE32_HD inline float from_bits(uint32_t bits) {
    union Bits { float value; uint32_t bits; } repr;
    repr.bits = bits;
    return repr.value;
}

IPC_LOCAL_SOLVE32_HD inline bool finite_number(float value) {
    return (float_bits(value) & uint32_t(0x7f800000u)) != uint32_t(0x7f800000u);
}

IPC_LOCAL_SOLVE32_HD inline bool is_zero(float value) {
    return (float_bits(value) & uint32_t(0x7fffffffu)) == 0u;
}

IPC_LOCAL_SOLVE32_HD inline float absolute(float value) {
    return from_bits(float_bits(value) & uint32_t(0x7fffffffu));
}

IPC_LOCAL_SOLVE32_HD inline float max_absolute(float a, float b) {
    const uint32_t aa = float_bits(a) & uint32_t(0x7fffffffu);
    const uint32_t bb = float_bits(b) & uint32_t(0x7fffffffu);
    return from_bits(aa > bb ? aa : bb);
}

// Explicit PTX, with no .ftz modifier, isolates this solve from the inherited
// material kernel's --use_fast_math/FMA flags. In particular, CUDA compiler
// flags cannot turn these into reciprocal approximations or fused operations.
IPC_LOCAL_SOLVE32_HD inline float add(float a, float b) {
#if defined(__CUDA_ARCH__)
    float result;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    volatile float result = a + b;
    return result;
#endif
}

IPC_LOCAL_SOLVE32_HD inline float subtract(float a, float b) {
#if defined(__CUDA_ARCH__)
    float result;
    asm volatile("sub.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    volatile float result = a - b;
    return result;
#endif
}

IPC_LOCAL_SOLVE32_HD inline float multiply(float a, float b) {
#if defined(__CUDA_ARCH__)
    float result;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    volatile float result = a * b;
    return result;
#endif
}

IPC_LOCAL_SOLVE32_HD inline float divide(float a, float b) {
#if defined(__CUDA_ARCH__)
    float result;
    asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    volatile float result = a / b;
    return result;
#endif
}

IPC_LOCAL_SOLVE32_HD inline float square_root(float value) {
#if defined(__CUDA_ARCH__)
    float result;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
#else
    volatile float result = sqrtf(value);
    return result;
#endif
}

IPC_LOCAL_SOLVE32_HD inline float norm3(const float* values, int begin) {
    float scale = 0.0f;
    for (int i = begin; i < 3; ++i) scale = max_absolute(scale, values[i]);
    if (is_zero(scale)) return 0.0f;
    float sum = 0.0f;
    for (int i = begin; i < 3; ++i) {
        const float v = divide(values[i], scale);
        sum = add(sum, multiply(v, v));
    }
    return multiply(scale, square_root(sum));
}

// Left-multiply a by a Householder reflector. Optionally transform its RHS
// and right-multiply the accumulated orthogonal matrix q by the reflector.
IPC_LOCAL_SOLVE32_HD inline void householder(
    float a[3][3], int column, int begin, float (*q)[3], float* rhs,
    int columns_end) {
    float v[3] = {0.0f, 0.0f, 0.0f};
    for (int row = begin; row < 3; ++row) v[row] = a[row][column];
    const float norm = norm3(v, begin);
    if (is_zero(norm)) return;
    const float beta = v[begin] >= 0.0f ? -norm : norm;
    // Normalize before subtraction so v0-beta cannot overflow.
    for (int row = begin; row < 3; ++row) v[row] = divide(v[row], norm);
    v[begin] = subtract(v[begin], divide(beta, norm));
    float vv = 0.0f;
    for (int row = begin; row < 3; ++row) vv = add(vv, multiply(v[row], v[row]));
    const float tau = divide(2.0f, vv);
    for (int col = column; col < columns_end; ++col) {
        float product = 0.0f;
        for (int row = begin; row < 3; ++row) product = add(product, multiply(v[row], a[row][col]));
        for (int row = begin; row < 3; ++row) a[row][col] = subtract(a[row][col], multiply(multiply(tau, v[row]), product));
    }
    a[begin][column] = beta;
    for (int row = begin + 1; row < 3; ++row) a[row][column] = 0.0f;
    if (rhs != nullptr) {
        float product = 0.0f;
        for (int row = begin; row < 3; ++row) product = add(product, multiply(v[row], rhs[row]));
        for (int row = begin; row < 3; ++row) rhs[row] = subtract(rhs[row], multiply(multiply(tau, v[row]), product));
    }
    if (q != nullptr) {
        for (int row = 0; row < 3; ++row) {
            float product = 0.0f;
            for (int col = begin; col < 3; ++col) product = add(product, multiply(q[row][col], v[col]));
            for (int col = begin; col < 3; ++col) q[row][col] = subtract(q[row][col], multiply(multiply(tau, product), v[col]));
        }
    }
}

// Exact frexp-style bit decomposition, including subnormals, independent of
// compiler FTZ flags. Call only for finite nonzero values.
IPC_LOCAL_SOLVE32_HD inline float split_exponent(float value, int& exponent) {
    const uint32_t bits = float_bits(value);
    const uint32_t sign = bits & uint32_t(0x80000000u);
    const uint32_t encoded = (bits >> 23) & uint32_t(0xffu);
    uint32_t fraction = bits & uint32_t(0x007fffffu);
    if (encoded == 0u) {
        exponent = -125;
        while ((fraction & uint32_t(0x00800000u)) == 0u) { fraction <<= 1; --exponent; }
        fraction &= uint32_t(0x007fffffu);
    } else {
        exponent = int(encoded) - 126;
    }
    return from_bits(sign | uint32_t(0x3f000000u) | fraction);
}

// Exact power-of-two scaling of a represented mantissa, with round-to-nearest
// ties-to-even at the subnormal boundary. Integer construction avoids FTZ.
IPC_LOCAL_SOLVE32_HD inline float join_exponent(float value, int exponent) {
    if (is_zero(value)) return value;
    int adjust = 0;
    const uint32_t bits = float_bits(split_exponent(value, adjust));
    exponent += adjust;
    const uint32_t sign = bits & uint32_t(0x80000000u);
    const uint32_t fraction = bits & uint32_t(0x007fffffu);
    if (exponent > 128) return from_bits(sign | uint32_t(0x7f800000u));
    if (exponent >= -125) return from_bits(sign | (uint32_t(exponent + 126) << 23) | fraction);
    const int shift = -125 - exponent;
    if (shift > 24) return from_bits(sign);
    const uint32_t significand = uint32_t(0x00800000u) | fraction;
    uint32_t rounded = significand >> shift;
    const uint32_t remainder = significand & ((uint32_t(1u) << shift) - uint32_t(1u));
    const uint32_t half = uint32_t(1u) << (shift - 1);
    if (remainder > half || (remainder == half && (rounded & uint32_t(1u)) != 0u)) ++rounded;
    return from_bits(sign | rounded);
}

// Avoid intermediate overflow in rhs_scale/matrix_scale*damping. All
// mantissas/arithmetic are float; exponents are combined with integer math.
IPC_LOCAL_SOLVE32_HD inline float restore_scale(
    float component, float rhs_scale, float matrix_scale, float damping) {
    if (is_zero(component)) return component;
    int component_exponent = 0, rhs_exponent = 0;
    int matrix_exponent = 0, damping_exponent = 0;
    const float component_mantissa = split_exponent(component, component_exponent);
    const float rhs_mantissa = split_exponent(rhs_scale, rhs_exponent);
    const float matrix_mantissa = split_exponent(matrix_scale, matrix_exponent);
    const float damping_mantissa = split_exponent(damping, damping_exponent);
    const float mantissa = multiply(divide(multiply(component_mantissa, rhs_mantissa), matrix_mantissa), damping_mantissa);
    return join_exponent(mantissa, component_exponent + rhs_exponent
                                  - matrix_exponent + damping_exponent);
}

// Return +damping * H^+ f, where f is force (= -energy gradient). H is row
// major and is not assumed positive definite or forcibly symmetrized.
// All nonfinite inputs/results fail closed with a zero output and negative
// status. Zero H, zero force, or zero damping have the zero minimum-norm result.
IPC_LOCAL_SOLVE32_HD inline int solve3(
    const float* h, const float* force, float damping, float* solution) {
    for (int i = 0; i < 3; ++i) solution[i] = 0.0f;
    if (!finite_number(damping) || (!is_zero(damping) && (float_bits(damping) >> 31) != 0u)) return InvalidInput;
    float matrix_scale = 0.0f, rhs_scale = 0.0f;
    for (int i = 0; i < 9; ++i) {
        if (!finite_number(h[i])) return InvalidInput;
        matrix_scale = max_absolute(matrix_scale, h[i]);
    }
    for (int i = 0; i < 3; ++i) {
        if (!finite_number(force[i])) return InvalidInput;
        rhs_scale = max_absolute(rhs_scale, force[i]);
    }
    if (is_zero(matrix_scale) || is_zero(rhs_scale) || is_zero(damping)) return Zero;

    float a[3][3], rhs[3], x[3] = {0.0f, 0.0f, 0.0f};
    for (int row = 0; row < 3; ++row) {
        rhs[row] = divide(force[row], rhs_scale);
        for (int col = 0; col < 3; ++col) a[row][col] = divide(h[3 * row + col], matrix_scale);
    }
    const float c[3][3] = {
        {subtract(multiply(a[1][1],a[2][2]),multiply(a[1][2],a[2][1])), subtract(multiply(a[0][2],a[2][1]),multiply(a[0][1],a[2][2])), subtract(multiply(a[0][1],a[1][2]),multiply(a[0][2],a[1][1]))},
        {subtract(multiply(a[1][2],a[2][0]),multiply(a[1][0],a[2][2])), subtract(multiply(a[0][0],a[2][2]),multiply(a[0][2],a[2][0])), subtract(multiply(a[0][2],a[1][0]),multiply(a[0][0],a[1][2]))},
        {subtract(multiply(a[1][0],a[2][1]),multiply(a[1][1],a[2][0])), subtract(multiply(a[0][1],a[2][0]),multiply(a[0][0],a[2][1])), subtract(multiply(a[0][0],a[1][1]),multiply(a[0][1],a[1][0]))}
    };
    const float determinant = add(add(multiply(a[0][0],c[0][0]), multiply(a[0][1],c[1][0])), multiply(a[0][2],c[2][0]));
    // det(H)/maxabs(H)^3 removes the original threshold's overflow/underflow.
    const float determinant_tolerance = 64.0f * FLT_EPSILON;
    int status = Adjugate;
    if (finite_number(determinant) && absolute(determinant) > determinant_tolerance) {
        const float inverse_det = divide(1.0f, determinant);
        for (int row = 0; row < 3; ++row) {
            x[row] = add(add(multiply(multiply(c[row][0],inverse_det),rhs[0]), multiply(multiply(c[row][1],inverse_det),rhs[1])),
                         multiply(multiply(c[row][2],inverse_det),rhs[2]));
        }
    } else {
        status = MinimumNormQR;
        int permutation[3] = {0, 1, 2};
        float max_pivot = 0.0f;
        for (int k = 0; k < 3; ++k) {
            int best = k;
            float largest = -1.0f;
            for (int col = k; col < 3; ++col) {
                const float values[3] = {a[0][col], a[1][col], a[2][col]};
                const float norm = norm3(values, k);
                if (norm > largest) { largest = norm; best = col; }
            }
            if (best != k) {
                for (int row = 0; row < 3; ++row) {
                    const float value = a[row][k]; a[row][k] = a[row][best]; a[row][best] = value;
                }
                const int value = permutation[k]; permutation[k] = permutation[best]; permutation[best] = value;
            }
            householder(a, k, k, nullptr, rhs, 3);
            max_pivot = max_absolute(max_pivot, a[k][k]);
        }
        const float rank_tolerance = multiply(3.0f * FLT_EPSILON, max_pivot);
        int rank = 0;
        // Column pivoting orders the independent prefix. Discard its trailing
        // numerical nullspace rather than forming normal equations.
        while (rank < 3 && absolute(a[rank][rank]) > rank_tolerance) ++rank;
        if (rank == 0) return Zero;

        // H/scale * P = Q R, then R_top^T = V T.
        // Solve T^T z = (Q^T rhs)_top and x = P V [z;0]. This second QR
        // supplies the minimum-norm solution instead of setting free pivoted
        // coordinates to zero (which would only give a basic solution).
        float t[3][3] = {}, v[3][3] = {};
        for (int row = 0; row < 3; ++row) {
            v[row][row] = 1.0f;
            for (int col = 0; col < rank; ++col) t[row][col] = a[col][row];
        }
        for (int k = 0; k < rank; ++k) householder(t, k, k, v, nullptr, rank);
        float z[3] = {0.0f, 0.0f, 0.0f};
        for (int row = 0; row < rank; ++row) {
            float sum = rhs[row];
            for (int col = 0; col < row; ++col) sum = subtract(sum, multiply(t[col][row], z[col]));
            if (is_zero(t[row][row]) || !finite_number(t[row][row])) return NumericalFailure;
            z[row] = divide(sum, t[row][row]);
        }
        for (int row = 0; row < 3; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < rank; ++col) sum = add(sum, multiply(v[row][col], z[col]));
            x[permutation[row]] = sum;
        }
    }

    float restored[3];
    for (int i = 0; i < 3; ++i) {
        if (!finite_number(x[i])) return NumericalFailure;
        restored[i] = restore_scale(x[i], rhs_scale, matrix_scale, damping);
        if (!finite_number(restored[i])) return NumericalFailure;
    }
    for (int i = 0; i < 3; ++i) solution[i] = restored[i];
    return status;
}

} } // namespace ipc_gpu::local_solve_fp32
#undef IPC_LOCAL_SOLVE32_HD
