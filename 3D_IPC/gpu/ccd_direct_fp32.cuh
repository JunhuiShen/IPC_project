#pragma once
#ifndef IPC_GPU_CCD_DIRECT_FP32_H
#define IPC_GPU_CCD_DIRECT_FP32_H

#include "ccd_fp32.cuh"

#if defined(__CUDACC__)
#define IPC_DIRECT32_HD __host__ __device__
#else
#define IPC_DIRECT32_HD
#endif

// Direct FP32 clipping requires an outward-rounded separation proof for the
// represented motion. Grouped updates first try a coordinate-axis clearance
// bound, then the closest-feature axis. An unresolved start can give zero.
namespace ipc_gpu { namespace ccd32direct {
using ccd32::LinearQuery;
using ccd32::ClipResult;
using ccd32::Status;

namespace detail {
using namespace ccd32::detail;

IPC_DIRECT32_HD inline int moving_vertex(const LinearQuery& q) {
    int result = -1;
    for (int i = 0; i < 4; ++i) {
        bool changed = false;
        for (int k = 0; k < 3; ++k)
            changed = changed || !same_value(q.start[i][k], q.end[i][k]);
        if (changed) {
            if (result >= 0) return -2;
            result = i;
        }
    }
    return result;
}

IPC_DIRECT32_HD inline float dot(Vec a, Vec b) {
    return a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2];
}

IPC_DIRECT32_HD inline Vec negate(Vec a) {
    for (int k = 0; k < 3; ++k) a.v[k] = -a.v[k];
    return a;
}

IPC_DIRECT32_HD inline Vec point_segment_residual(const float* p,
                                                   const float* a,
                                                   const float* b) {
    const Vec edge = difference(b, a), offset = difference(p, a);
    const float length = dot(edge, edge);
    if (!(length > 0.0f)) return offset;
    const float numerator = dot(offset, edge);
    if (!(numerator > 0.0f)) return offset;
    if (numerator >= length) return difference(p, b);
    const float parameter = numerator / length;
    Vec result;
    for (int k = 0; k < 3; ++k)
        result.v[k] = ::fmaf(-parameter, edge.v[k], offset.v[k]);
    return result;
}

// Translation and scaling here only improve the proposed feature direction.
// They are not used to certify geometry, so their ordinary rounding cannot
// produce a false acceptance. An incorrect feature can only cause rejection.
IPC_DIRECT32_HD inline bool closest_axis(const LinearQuery& q, Vec& axis) {
    float p[4][3], scale = 0.0f;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        p[i][k] = q.start[i][k] - q.start[3][k];
        if (!finite(p[i][k])) return false;
        scale = maximum(scale, absolute(p[i][k]));
    }
    if (!(scale > 0.0f) || subnormal(scale)) return false;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        p[i][k] /= scale;

    if (q.vertex_face) {
        const Vec a = difference(p[2], p[1]), b = difference(p[3], p[1]);
        const Vec offset = difference(p[0], p[1]), normal = cross(a, b);
        const float area = dot(normal, normal);
        const float aa = dot(a, a), ab = dot(a, b), bb = dot(b, b);
        const float ar = dot(a, offset), br = dot(b, offset);
        const float u = bb*ar - ab*br, v = aa*br - ab*ar;
        if (area > 0.0f && u >= 0.0f && v >= 0.0f && u+v <= area) {
            // The interior residual is proportional to the face normal.
            // Avoid reconstructing and subtracting nearly equal closest points.
            axis = normal;
        } else {
            axis = point_segment_residual(p[0], p[1], p[2]);
            float best = dot(axis, axis);
            for (int edge = 2; edge <= 3; ++edge) {
                const Vec candidate = point_segment_residual(p[0], p[edge], p[edge == 3 ? 1 : 3]);
                const float squared = dot(candidate, candidate);
                if (squared < best) { axis = candidate; best = squared; }
            }
        }
    } else {
        const Vec a = difference(p[1], p[0]), b = difference(p[3], p[2]);
        const Vec offset = difference(p[0], p[2]);
        const float aa = dot(a, a), ab = dot(a, b), bb = dot(b, b);
        const float ar = dot(a, offset), br = dot(b, offset);
        const float denominator = aa*bb - ab*ab;
        const float u = ab*br - bb*ar, v = aa*br - ab*ar;
        if (denominator > 0.0f && u >= 0.0f && u <= denominator
            && v >= 0.0f && v <= denominator) {
            axis = cross(a, b);
        } else {
            axis = point_segment_residual(p[0], p[2], p[3]);
            float best = dot(axis, axis);
            for (int endpoint = 1; endpoint < 4; ++endpoint) {
                Vec candidate = endpoint == 1
                    ? point_segment_residual(p[1], p[2], p[3])
                    : negate(point_segment_residual(p[endpoint], p[0], p[1]));
                const float squared = dot(candidate, candidate);
                if (squared < best) { axis = candidate; best = squared; }
            }
        }
    }
    scale = 0.0f;
    for (int k = 0; k < 3; ++k) {
        if (!finite(axis.v[k])) return false;
        scale = maximum(scale, absolute(axis.v[k]));
    }
    if (!(scale > 0.0f) || subnormal(scale)) return false;
    for (int k = 0; k < 3; ++k) axis.v[k] /= scale;
    return true;
}

IPC_DIRECT32_HD inline Interval projected_gap(const float* a, const float* b, Vec axis) {
    Interval result = point(0.0f);
    for (int k = 0; k < 3; ++k) {
        const Interval delta = subtract(point(a[k]), point(b[k]));
        if (!bounded(delta)) return whole();
        const bool negative = (bits(axis.v[k]) & 0x80000000u) != 0u;
        const float lo = multiply_bound(axis.v[k], negative ? delta.hi : delta.lo, false);
        const float hi = multiply_bound(axis.v[k], negative ? delta.lo : delta.hi, true);
        if (!finite(lo) || !finite(hi)) return whole();
        result = add(result, Interval{lo, hi});
        if (!bounded(result)) return whole();
    }
    return result;
}

IPC_DIRECT32_HD inline bool prepare_axis(const LinearQuery& q, Vec& axis, float& threshold) {
    if (!closest_axis(q, axis)) return false;
    const Interval first = projected_gap(q.start[0], q.start[q.vertex_face ? 1 : 2], axis);
    if (!bounded(first)) return false;
    if (first.hi < 0.0f) axis = negate(axis);
    float norm2_upper = 0.0f;
    for (int k = 0; k < 3; ++k)
        norm2_upper = add_bound(norm2_upper,
            multiply_bound(absolute(axis.v[k]), absolute(axis.v[k]), true), true);
    if (!finite(norm2_upper)) return false;
    threshold = multiply_bound(q.gap_floor, sqrt_upper(norm2_upper), true);
    return finite(threshold) && threshold > 0.0f;
}

IPC_DIRECT32_HD inline bool certify_axis(const LinearQuery& q, Vec axis, float threshold) {
    const int split = q.vertex_face ? 1 : 2;
    for (int a = 0; a < split; ++a) for (int b = split; b < 4; ++b) {
        const Interval start = projected_gap(q.start[a], q.start[b], axis);
        const Interval end = projected_gap(q.end[a], q.end[b], axis);
        if (!bounded(start) || !bounded(end)
            || !(start.lo > threshold) || !(end.lo > threshold)) return false;
    }
    return true;
}

IPC_DIRECT32_HD inline float rounded_delta(float start, float end) {
#if defined(__CUDA_ARCH__)
    float value;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(end), "f"(start));
    return value;
#else
    volatile float value = end - start;
    return value;
#endif
}

IPC_DIRECT32_HD inline float divide_down(float numerator, float denominator) {
    if (!finite(numerator) || !finite(denominator)
        || !(numerator > 0.0f) || !(denominator > 0.0f)
        || subnormal(numerator) || subnormal(denominator)) return 0.0f;
#if defined(__CUDA_ARCH__)
    float value;
    asm("div.rm.f32 %0, %1, %2;" : "=f"(value) : "f"(numerator), "f"(denominator));
    return finite(value) ? value : 1.0f;
#else
    volatile float rounded = numerator / denominator;
    const float value = rounded;
    if (!finite(value)) return 1.0f;
    if (zero(value) || subnormal(value)) return 0.0f;
    return maximum(0.0f, neighbor(value, false));
#endif
}

// For 0<beta<1, interpolate uses D=RN(end-start), then RN_fma(start+beta*D).
// Keeping D itself in the certified slope accounts for subtraction rounding.
// RN error is bounded by u*|start|+beta*u*|D|+nu, with u=2^-24 and
// nu=2^-126 (a deliberately conservative normal underflow allowance).
// Thus |axis dot error| <= E0 + beta*E1. All sums/products below round upward.
// The resulting envelope holds for EVERY independently reduced beta, not only
// the originally proposed fraction. The beta=1 identity is handled separately.
IPC_DIRECT32_HD inline bool rounding_envelope(const LinearQuery& q, int mover,
                                             Vec axis, Interval& slope,
                                             float& e0, float& e1) {
    constexpr float unit = 0x1p-24f;
    constexpr float tiny = 0x1p-126f;
    float absolute_start = 0.0f, absolute_delta = 0.0f, axis_l1 = 0.0f;
    slope = point(0.0f);
    for (int k = 0; k < 3; ++k) {
        const float start = q.start[mover][k];
        const float delta = rounded_delta(start, q.end[mover][k]);
        if (!finite(delta) || subnormal(delta)) return false;
        // The exact FMA arguments lie between start and start+D. Require a
        // finite enclosure so the RN relative/absolute error bound applies.
        const Interval limit = add(point(start), point(delta));
        if (!bounded(limit)) return false;
        const float weight = absolute(axis.v[k]);
        absolute_start = add_bound(absolute_start,
            multiply_bound(weight, absolute(start), true), true);
        absolute_delta = add_bound(absolute_delta,
            multiply_bound(weight, absolute(delta), true), true);
        axis_l1 = add_bound(axis_l1, weight, true);
        slope = add(slope, multiply(point(axis.v[k]), point(delta)));
        if (!finite(absolute_start) || !finite(absolute_delta)
            || !finite(axis_l1) || !bounded(slope)) return false;
    }
    // axis has maximum absolute component one, hence axis_l1>=1. Accumulating
    // the normal tiny allowance here avoids subnormal padding multiplications
    // in host interval helpers, which deliberately reject subnormal arithmetic.
    e0 = add_bound(multiply_bound(unit, absolute_start, true),
                   multiply_bound(tiny, axis_l1, true), true);
    e1 = multiply_bound(unit, absolute_delta, true);
    return finite(e0) && finite(e1);
}

// A represented prefix is monotone in each coordinate up to beta<1;
// beta=1 uses the original endpoint. Use the largest interior beta to
// enclose the whole family, including FP32 interpolation rounding.
IPC_DIRECT32_HD inline bool coordinate_clearance(const LinearQuery& q, int mover, float retained) {
    const int split = q.vertex_face ? 1 : 2;
    float extent[3];
    for (int k = 0; k < 3; ++k) {
        const float start = q.start[mover][k], end = q.end[mover][k];
        const float interior = interpolate(start, end, 0x1.fffffep-1f);
        if (!finite(interior))
            return false;
        const float lo = minimum(start, minimum(end, interior));
        const float hi = maximum(start, maximum(end, interior));
        extent[k] = subtract_bound(hi, lo, true);
    }
    for (int k = 0; k < 3; ++k) {
        float alo = infinity(), ahi = -infinity(), blo = infinity(), bhi = -infinity();
        for (int i = 0; i < 4; ++i) {
            if (i < split) {
                alo = minimum(alo, q.start[i][k]);
                ahi = maximum(ahi, q.start[i][k]);
            } else {
                blo = minimum(blo, q.start[i][k]);
                bhi = maximum(bhi, q.start[i][k]);
            }
        }
        const float gap = maximum(subtract_bound(alo, bhi, false), subtract_bound(blo, ahi, false));
        const float budget =
            multiply_bound(1.0f - retained, subtract_bound(gap, q.gap_floor, false), false);
        if (finite(budget) && extent[k] < budget)
            return true;
    }
    return false;
}

} // namespace detail

IPC_DIRECT32_HD inline Status validate_linear(const LinearQuery& q) {
    if (!ccd32::detail::valid(q)) return Status::Invalid;
    const int mover = detail::moving_vertex(q);
    if (mover == -2) return Status::Invalid;
    if (mover == -1) return Status::Safe;
    ccd32::detail::Vec axis;
    float threshold;
    return detail::prepare_axis(q, axis, threshold)
        && detail::certify_axis(q, axis, threshold) ? Status::Safe : Status::Uncertain;
}

// Grouped GS also certifies the complete stored move. Reserving clearance
// when moving towards a contact keeps subsequent updates off the rounding floor.
IPC_DIRECT32_HD inline float propose_fraction(const LinearQuery& q, bool* full_step = nullptr) {
    using namespace detail;
    if (full_step) *full_step = false;
    if (!valid(q)) return -1.0f;
    const int mover = moving_vertex(q);
    if (mover == -2) return -1.0f;
    if (mover == -1) {
        if (full_step) *full_step = true;
        return 1.0f;
    }
    constexpr float retained_clearance = 0.8f;
    if (full_step && detail::coordinate_clearance(q,mover,retained_clearance)) {
        *full_step = true;
        return 1.0f;
    }
    Vec axis;
    float threshold;
    if (!prepare_axis(q, axis, threshold)) return 0.0f;

    const int split = q.vertex_face ? 1 : 2;
    float moving_gap = infinity();
    bool end_safe = true;
    for (int a = 0; a < split; ++a) for (int b = split; b < 4; ++b) {
        const Interval gap = projected_gap(q.start[a], q.start[b], axis);
        if (!bounded(gap) || !(gap.lo > threshold)) return 0.0f;
        if (a == mover || b == mover) {
            moving_gap = minimum(moving_gap, gap.lo);
            if (full_step && end_safe) {
                const Interval end = projected_gap(q.end[a], q.end[b], axis);
                const float reserve = threshold + retained_clearance * (gap.lo - threshold);
                end_safe = bounded(end) && end.lo > threshold && end.lo >= reserve;
            }
        }
    }
    if (full_step) *full_step = end_safe;
    Interval slope;
    float e0, e1;
    if (!rounding_envelope(q, mover, axis, slope, e0, e1)) return 0.0f;
    if (mover >= split) slope = {-slope.hi, -slope.lo};
    const float closing = maximum(0.0f, -slope.lo);
    float budget = subtract_bound(moving_gap, threshold, false);
    if (full_step && closing > 0.0f) budget = multiply_bound(1.0f-retained_clearance, budget, false);
    const float numerator = subtract_bound(budget, e0, false);
    const float denominator = add_bound(closing, e1, true);
    if (!finite(numerator) || !finite(denominator) || !(numerator > 0.0f)) return 0.0f;
    // The batch takes a minimum across pairs. A full proposal must cover all
    // smaller, rounded prefixes as well as the original endpoint at beta=1.
    if (numerator > denominator && (full_step ? end_safe : certify_axis(q, axis, threshold))) return 1.0f;
    float fraction = denominator > 0.0f ? divide_down(numerator, denominator) : 1.0f;
    fraction = minimum(1.0f, maximum(0.0f, fraction));
    // Strictly interior, including when downward division happened to be
    // exact. This also keeps the proposal on interpolate's FMA branch.
    return fraction > 0.0f ? maximum(0.0f, neighbor(fraction, false)) : 0.0f;
}

IPC_DIRECT32_HD inline ClipResult clip_linear(const LinearQuery& q) {
    ClipResult result;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        result.endpoint[i][k] = q.start[i][k];
    const float fraction = ccd32direct::propose_fraction(q);
    result.checks = 1;
    if (fraction < 0.0f) { result.status = Status::Invalid; return result; }
    result.status = Status::Safe;
    if (!(fraction > 0.0f)) return result;
    LinearQuery trial = q;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        trial.end[i][k] = ccd32::detail::interpolate(q.start[i][k], q.end[i][k], fraction);
    ++result.checks;
    if (ccd32direct::validate_linear(trial) != Status::Safe) return result;
    result.fraction = fraction;
    result.full_step = fraction == 1.0f;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        result.endpoint[i][k] = trial.end[i][k];
    return result;
}

} } // namespace ipc_gpu::ccd32direct
#undef IPC_DIRECT32_HD
#endif
