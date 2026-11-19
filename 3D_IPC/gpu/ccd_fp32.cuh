#pragma once
#ifndef IPC_GPU_CCD_FP32_H
#define IPC_GPU_CCD_FP32_H

// Conservative CCD of the ACTUAL represented FP32 linear trajectories.
// Every acceptance uses outward-rounded FP32 projection intervals. Failure to
// prove separation is Uncertain, never a miss. A valid zero-motion query is
// Safe because it introduces no new trajectory, including at initial contact.
// This file has no exact-arithmetic dependency or higher-precision arithmetic.

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define IPC_CCD32_HD __host__ __device__
#else
#define IPC_CCD32_HD
#endif

namespace ipc_gpu {
namespace ccd32 {

enum class Status : unsigned { Safe, Uncertain, Invalid };

struct LinearQuery {
    float start[4][3]{};
    float end[4][3]{};
    bool vertex_face = true;
    float gap_floor = 1e-8f;
};

struct ClipResult {
    Status status = Status::Uncertain;
    float fraction = 0.0f;
    float endpoint[4][3]{};
    int checks = 0;
    bool full_step = false;
};

namespace detail {

IPC_CCD32_HD inline std::uint32_t bits(float value) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
#endif
}

IPC_CCD32_HD inline float from_bits(std::uint32_t value) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(value);
#else
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
#endif
}

IPC_CCD32_HD inline bool finite(float value) {
    return (bits(value) & 0x7f800000u) != 0x7f800000u;
}
IPC_CCD32_HD inline bool zero(float value) {
    return (bits(value) & 0x7fffffffu) == 0u;
}
IPC_CCD32_HD inline bool subnormal(float value) {
    const auto raw = bits(value) & 0x7fffffffu;
    return raw != 0u && raw < 0x00800000u;
}
IPC_CCD32_HD inline bool same_value(float a, float b) {
    return bits(a) == bits(b) || (zero(a) && zero(b));
}
IPC_CCD32_HD inline float infinity() { return from_bits(0x7f800000u); }
IPC_CCD32_HD inline float nan() { return from_bits(0x7fc00000u); }
IPC_CCD32_HD inline float absolute(float value) {
    return from_bits(bits(value) & 0x7fffffffu);
}

// Integer comparisons preserve subnormal bounds even in an enclosing FTZ
// module. All callers reject NaN before using these order operations.
IPC_CCD32_HD inline std::uint32_t order(float value) {
    const auto raw = bits(value);
    return raw & 0x80000000u ? ~raw : raw | 0x80000000u;
}
IPC_CCD32_HD inline float minimum(float a, float b) {
    return order(a) < order(b) ? a : b;
}
IPC_CCD32_HD inline float maximum(float a, float b) {
    return order(a) > order(b) ? a : b;
}
IPC_CCD32_HD inline float clamp01(float value) {
    return maximum(0.0f, minimum(1.0f, value));
}

// The bit equivalent of nextafterf, independent of ambient FTZ/DAZ flags.
IPC_CCD32_HD inline float neighbor(float value, bool upward) {
    auto raw = bits(value);
    if ((raw & 0x7fffffffu) > 0x7f800000u) return value;
    if ((upward && raw == 0x7f800000u)
        || (!upward && raw == 0xff800000u)) return value;
    if ((raw & 0x7fffffffu) == 0u)
        return from_bits(upward ? 1u : 0x80000001u);
    raw += upward == ((raw & 0x80000000u) == 0u) ? 1u : std::uint32_t(-1);
    return from_bits(raw);
}

// CUDA instructions explicitly omit .ftz and carry their own rounding mode.
// Host operations force a represented float result before moving one ULP
// outward. Host subnormal/underflow arithmetic is conservatively rejected so
// an external DAZ/FTZ setting cannot make an inward interval look certified.
IPC_CCD32_HD inline float add_bound(float a, float b, bool upward) {
#if defined(__CUDA_ARCH__)
    float result;
    if (upward) asm("add.rp.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    else asm("add.rm.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    if (zero(a)) return b;
    if (zero(b)) return a;
    if ((bits(a) ^ bits(b)) == 0x80000000u) return 0.0f;
    if (subnormal(a) || subnormal(b)) return nan();
    volatile float rounded = a + b;
    const float result = rounded;
    if (zero(result) || subnormal(result)) return nan();
    return neighbor(result, upward);
#endif
}
IPC_CCD32_HD inline float subtract_bound(float a, float b, bool upward) {
    return add_bound(a, from_bits(bits(b) ^ 0x80000000u), upward);
}
IPC_CCD32_HD inline float multiply_bound(float a, float b, bool upward) {
#if defined(__CUDA_ARCH__)
    float result;
    if (upward) asm("mul.rp.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    else asm("mul.rm.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    if (zero(a) || zero(b)) return 0.0f;
    if (subnormal(a) || subnormal(b)) return nan();
    volatile float rounded = a * b;
    const float result = rounded;
    if (zero(result) || subnormal(result)) return nan();
    return neighbor(result, upward);
#endif
}

struct Interval { float lo, hi; };
struct Vec { float v[3]; };

IPC_CCD32_HD inline Interval whole() { return {-infinity(), infinity()}; }
IPC_CCD32_HD inline Interval point(float value) { return {value, value}; }
IPC_CCD32_HD inline bool bounded(Interval value) {
    return finite(value.lo) && finite(value.hi);
}
IPC_CCD32_HD inline Interval add(Interval a, Interval b) {
    if (!bounded(a) || !bounded(b)) return whole();
    const float lo = add_bound(a.lo, b.lo, false);
    const float hi = add_bound(a.hi, b.hi, true);
    return finite(lo) && finite(hi) ? Interval{lo, hi} : whole();
}
IPC_CCD32_HD inline Interval subtract(Interval a, Interval b) {
    if (!bounded(a) || !bounded(b)) return whole();
    const float lo = subtract_bound(a.lo, b.hi, false);
    const float hi = subtract_bound(a.hi, b.lo, true);
    return finite(lo) && finite(hi) ? Interval{lo, hi} : whole();
}
IPC_CCD32_HD inline Interval multiply(Interval a, Interval b) {
    if (!bounded(a) || !bounded(b)) return whole();
    float lo = infinity(), hi = -infinity();
    const float av[2] = {a.lo, a.hi}, bv[2] = {b.lo, b.hi};
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        const float lower = multiply_bound(av[i], bv[j], false);
        const float upper = multiply_bound(av[i], bv[j], true);
        if (!finite(lower) || !finite(upper)) return whole();
        lo = minimum(lo, lower);
        hi = maximum(hi, upper);
    }
    return {lo, hi};
}
IPC_CCD32_HD inline float sqrt_upper(float value) {
    if (!finite(value) || (bits(value) & 0x80000000u)) return infinity();
    if (zero(value)) return 0.0f;
#if defined(__CUDA_ARCH__)
    float result;
    asm("sqrt.rp.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
#else
    if (subnormal(value)) return infinity();
    volatile float rounded = ::sqrtf(value);
    return neighbor(rounded, true);
#endif
}

IPC_CCD32_HD inline Vec difference(const float* a, const float* b) {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}
IPC_CCD32_HD inline Vec cross(Vec a, Vec b) {
    return {{a.v[1] * b.v[2] - a.v[2] * b.v[1],
             a.v[2] * b.v[0] - a.v[0] * b.v[2],
             a.v[0] * b.v[1] - a.v[1] * b.v[0]}};
}

// This is only a proposal. Its output is re-certified before use. Callers
// must use this same operation when committing a returned clipping fraction.
// Endpoint identities are exact; a subtract overflow yields NaN and cannot
// authorize a motion. Explicit RN instructions also make replay independent
// of the surrounding compiler's contraction and FTZ settings.
IPC_CCD32_HD inline float interpolate(float start, float end, float fraction) {
    if (fraction <= 0.0f) return start;
    if (fraction >= 1.0f) return end;
    float delta, result;
#if defined(__CUDA_ARCH__)
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(delta) : "f"(end), "f"(start));
    if (!finite(delta)) return nan();
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result)
        : "f"(fraction), "f"(delta), "f"(start));
#else
    volatile float rounded_delta = end - start;
    delta = rounded_delta;
    if (!finite(delta)) return nan();
    result = ::fmaf(fraction, delta, start);
#endif
    return result;
}

IPC_CCD32_HD inline bool valid(const LinearQuery& q) {
    if (!finite(q.gap_floor) || zero(q.gap_floor)
        || (bits(q.gap_floor) & 0x80000000u)) return false;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        if (!finite(q.start[i][k]) || !finite(q.end[i][k])) return false;
    return true;
}
IPC_CCD32_HD inline bool unchanged(const LinearQuery& q) {
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        if (!same_value(q.start[i][k], q.end[i][k])) return false;
    return true;
}

// Exact real trajectory position relative to one represented local origin,
// enclosed using FP32 operations. At either endpoint use its stored position
// directly, avoiding unnecessary dependency error from start + (end-start).
IPC_CCD32_HD inline Interval position(const LinearQuery& q, int i, int k, float time) {
    if (time == 0.0f) return subtract(point(q.start[i][k]), point(q.start[3][k]));
    if (time == 1.0f) return subtract(point(q.end[i][k]), point(q.start[3][k]));
    const Interval base = subtract(point(q.start[i][k]), point(q.start[3][k]));
    const Interval motion = subtract(point(q.end[i][k]), point(q.start[i][k]));
    return add(base, multiply(point(time), motion));
}

IPC_CCD32_HD inline bool projection(const LinearQuery& q, Vec axis,
                                    float time_lo, float time_hi, float gap) {
    float scale = 0.0f;
    for (int k = 0; k < 3; ++k) {
        if (!finite(axis.v[k])) return false;
        scale = maximum(scale, absolute(axis.v[k]));
    }
    if (!(scale > 0.0f)) return false;
    for (int k = 0; k < 3; ++k) axis.v[k] /= scale;
    Interval norm2 = point(0.0f);
    for (int k = 0; k < 3; ++k)
        norm2 = add(norm2, multiply(point(axis.v[k]), point(axis.v[k])));
    const float norm = sqrt_upper(norm2.hi);
    if (!finite(norm) || !(norm > 0.0f)) return false;
    const float threshold = multiply(point(gap), point(norm)).hi;
    if (!finite(threshold)) return false;
    const int split = q.vertex_face ? 1 : 2;
    float first_lo = infinity(), first_hi = -infinity();
    float second_lo = infinity(), second_hi = -infinity();
    const int endpoints = time_lo == time_hi ? 1 : 2;
    for (int endpoint = 0; endpoint < endpoints; ++endpoint) {
        const float time = endpoint ? time_hi : time_lo;
        for (int i = 0; i < 4; ++i) {
            Interval projected = point(0.0f);
            for (int k = 0; k < 3; ++k)
                projected = add(projected, multiply(point(axis.v[k]), position(q, i, k, time)));
            if (!bounded(projected)) return false;
            if (i < split) {
                first_lo = minimum(first_lo, projected.lo);
                first_hi = maximum(first_hi, projected.hi);
            } else {
                second_lo = minimum(second_lo, projected.lo);
                second_hi = maximum(second_hi, projected.hi);
            }
        }
    }
    const float ab = subtract(point(first_lo), point(second_hi)).lo;
    const float ba = subtract(point(second_lo), point(first_hi)).lo;
    const float distance = maximum(ab, ba);
    // Both must be finite and the lower bound strictly positive. Use integer
    // ordering so a positive subnormal threshold is never flushed to zero.
    return finite(distance) && (bits(distance) & 0x80000000u) == 0u
        && !zero(distance) && order(distance) > order(threshold);
}

// Axes only propose directions; rounded normals and edges cannot weaken a
// proof. They can at worst fail to find a useful separating direction.
IPC_CCD32_HD inline bool separated(const LinearQuery& q, float time_lo,
                                  float time_hi, float gap) {
    for (int k = 0; k < 3; ++k) {
        Vec axis{{0.0f, 0.0f, 0.0f}};
        axis.v[k] = 1.0f;
        if (projection(q, axis, time_lo, time_hi, gap)) return true;
    }
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const float (*p)[3] = endpoint ? q.end : q.start;
        if (q.vertex_face) {
            const Vec a = difference(p[2], p[1]), b = difference(p[3], p[1]);
            const Vec normal = cross(a, b);
            if (projection(q, normal, time_lo, time_hi, gap)) return true;
            for (int i = 1; i < 4; ++i) {
                const Vec edge = difference(p[i == 3 ? 1 : i + 1], p[i]);
                if (projection(q, cross(normal, edge), time_lo, time_hi, gap)) return true;
            }
        } else {
            const Vec a = difference(p[1], p[0]), b = difference(p[3], p[2]);
            const Vec w = difference(p[2], p[0]);
            if (projection(q, cross(a, b), time_lo, time_hi, gap)
                || projection(q, cross(cross(a, w), a), time_lo, time_hi, gap)
                || projection(q, cross(cross(b, w), b), time_lo, time_hi, gap)) return true;
        }
        const int split = q.vertex_face ? 1 : 2;
        for (int i = 0; i < split; ++i) for (int j = split; j < 4; ++j)
            if (projection(q, difference(p[i], p[j]), time_lo, time_hi, gap)) return true;
    }
    return false;
}

} // namespace detail

// A moving query must end beyond gap_floor. If the starting floor is proved,
// it is preserved throughout the trajectory. Near the floor, a move that
// restores the endpoint floor may be accepted only after strict positive
// separation is proved for its entire trajectory. Exact initial contact,
// coplanarity without a separating axis and unresolved rounding stay blocked.
IPC_CCD32_HD inline Status validate_linear(const LinearQuery& q, int subdivision_depth = 6) {
    using namespace detail;
    if (!valid(q) || subdivision_depth < 0 || subdivision_depth > 10) return Status::Invalid;
    if (unchanged(q)) return Status::Safe;
    if (separated(q, 0.0f, 1.0f, q.gap_floor)) return Status::Safe;
    if (!separated(q, 1.0f, 1.0f, q.gap_floor)) return Status::Uncertain;
    const float required_gap = separated(q, 0.0f, 0.0f, q.gap_floor) ? q.gap_floor : 0.0f;
    if (separated(q, 0.0f, 1.0f, required_gap)) return Status::Safe;
    const int intervals = 1 << subdivision_depth;
    const float width = 1.0f / static_cast<float>(intervals);
    for (int i = 0; i < intervals; ++i) {
        const float a = static_cast<float>(i) * width;
        const float b = static_cast<float>(i + 1) * width;
        if (!separated(q, a, b, required_gap)) return Status::Uncertain;
    }
    return Status::Safe;
}

// This returns a certified represented endpoint, not an earliest-impact time.
// Unresolved motion falls back to fraction zero/unchanged endpoint with Safe
// status. Callers must distinguish this no-progress result from a full step.
IPC_CCD32_HD inline ClipResult clip_linear(const LinearQuery& q,
                                           int max_attempts = 64, int subdivision_depth = 6) {
    ClipResult result;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        result.endpoint[i][k] = q.start[i][k];
    if (!detail::valid(q) || max_attempts < 1 || max_attempts > 64
        || subdivision_depth < 0 || subdivision_depth > 10) {
        result.status = Status::Invalid;
        return result;
    }
    LinearQuery trial = q;
    float fraction = 1.0f;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        if (attempt) {
            bool valid_endpoint = true;
            for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
                trial.end[i][k] = detail::interpolate(q.start[i][k], q.end[i][k], fraction);
                valid_endpoint = valid_endpoint && detail::finite(trial.end[i][k]);
            }
            if (!valid_endpoint) break;
            if (detail::unchanged(trial)) break;
        }
        ++result.checks;
        const Status status = validate_linear(trial, subdivision_depth);
        if (status == Status::Safe) {
            result.status = status;
            result.fraction = fraction;
            result.full_step = attempt == 0;
            for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
                result.endpoint[i][k] = trial.end[i][k];
            return result;
        }
        fraction *= 0.5f;
    }
    result.status = Status::Safe;
    return result;
}

} // namespace ccd32
} // namespace ipc_gpu

#undef IPC_CCD32_HD
#endif // IPC_GPU_CCD_FP32_H
