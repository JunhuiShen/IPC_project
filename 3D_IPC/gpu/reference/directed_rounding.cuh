#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__CUDACC__)
#define IPC_GPU_DIRECTED_ROUNDING_HD __host__ __device__
#else
#define IPC_GPU_DIRECTED_ROUNDING_HD
#endif

namespace ipc_gpu::directed_rounding {
namespace detail {

static_assert(sizeof(double) == sizeof(std::uint64_t)
    && std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::radix == 2
    && std::numeric_limits<double>::digits == 53
    && std::numeric_limits<double>::min_exponent == -1021
    && std::numeric_limits<double>::max_exponent == 1024,
    "GPU directed_rounding requires IEEE-754 binary64");

IPC_GPU_DIRECTED_ROUNDING_HD inline std::uint64_t bits(double x) {
#if defined(__CUDA_ARCH__)
    return static_cast<std::uint64_t>(__double_as_longlong(x));
#else
    std::uint64_t pattern;
    std::memcpy(&pattern, &x, sizeof(pattern));
    return pattern;
#endif
}

IPC_GPU_DIRECTED_ROUNDING_HD inline double from_bits(std::uint64_t pattern) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(pattern));
#else
    double result;
    std::memcpy(&result, &pattern, sizeof(result));
    return result;
#endif
}

// Direction is +1 for +infinity, -1 for -infinity, and 0 for positive zero.
template<int Direction>
IPC_GPU_DIRECTED_ROUNDING_HD inline double step(double x) {
    static_assert(Direction >= -1 && Direction <= 1, "invalid step direction");
    std::uint64_t pattern = bits(x);
    const std::uint64_t magnitude = pattern & UINT64_C(0x7fffffffffffffff);

    // Only normal -> normal transitions take the integer path. Keep zeros,
    // subnormals, normal-range endpoints, infinities and every NaN on the
    // platform nextafter path. Host errno/fenv and CUDA's cold-case handling
    // therefore retain their existing behavior.
    if (magnitude > UINT64_C(0x0010000000000000)
        && magnitude < UINT64_C(0x7fefffffffffffff)) {
        if constexpr (Direction == 0) {
            --pattern;
        } else {
            const bool negative = (pattern >> 63) != 0;
            if (negative == (Direction > 0)) --pattern;
            else ++pattern;
        }
        return from_bits(pattern);
    }

    constexpr double target = Direction > 0 ? double(INFINITY)
        : Direction < 0 ? -double(INFINITY) : 0.0;
    return ::nextafter(x, target);
}

} // namespace detail

IPC_GPU_DIRECTED_ROUNDING_HD inline double next_up(double x) { return detail::step<1>(x); }
IPC_GPU_DIRECTED_ROUNDING_HD inline double next_down(double x) { return detail::step<-1>(x); }

// Match nextafter(x,+0.0), including -0.0 -> +0.0. Callers still perform each
// arithmetic operation separately before widening; no reassociation is added.
IPC_GPU_DIRECTED_ROUNDING_HD inline double toward_positive_zero(double x) {
    return detail::step<0>(x);
}

} // namespace ipc_gpu::directed_rounding

#undef IPC_GPU_DIRECTED_ROUNDING_HD
