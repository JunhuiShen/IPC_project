#pragma once
#include "ccd.cuh"
#include "ccd_fp32.cuh"

#if defined(__CUDACC__)
#define IPC_CLEARANCE_HD __host__ __device__
#else
#define IPC_CLEARANCE_HD
#endif

namespace ipc_gpu::ccd_clearance {
using namespace ccd::detail;

// This direction proposes a correction only. The caller rebuilds swept
// candidates and proves every motion again before changing the solver state.
IPC_CLEARANCE_HD inline Vec best_axis(const float start[4][3], bool face) {
    Vec p[4], best{};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k)
            p[i].v[k] = double(start[i][k]) - double(start[3][k]);
    double best_gap = 0.0;
    const int split = face ? 1 : 2;
    const auto consider = [&](Vec n) {
        const double length = sqrt(n.v[0] * n.v[0] + n.v[1] * n.v[1] + n.v[2] * n.v[2]);
        if (!(length > 0.0) || !finite(length))
            return;
        for (int k = 0; k < 3; ++k)
            n.v[k] /= length;
        double lo = INFINITY, hi = -INFINITY;
        for (int i = 0; i < split; ++i)
            for (int j = split; j < 4; ++j) {
                const Vec d = difference(p[i].v, p[j].v);
                const double g = d.v[0] * n.v[0] + d.v[1] * n.v[1] + d.v[2] * n.v[2];
                lo = minimum(lo, g);
                hi = maximum(hi, g);
            }
        if (maximum(lo, -hi) <= best_gap)
            return;
        best_gap = maximum(lo, -hi);
        for (int k = 0; k < 3; ++k)
            best.v[k] = lo > -hi ? n.v[k] : -n.v[k];
    };
    const Vec u = difference(p[1].v, p[0].v), v = difference(p[3].v, p[2].v);
    if (face) {
        consider(cross(difference(p[2].v, p[1].v), v));
        for (int i = 1; i < 4; ++i) {
            const Vec e = difference(p[i == 3 ? 1 : i + 1].v, p[i].v);
            const Vec d = difference(p[0].v, p[i].v);
            consider(d);
            consider(cross(e, cross(d, e)));
        }
    } else {
        consider(cross(u, v));
        for (int i = 0; i < 2; ++i)
            for (int j = 2; j < 4; ++j) {
                const Vec d = difference(p[i].v, p[j].v);
                consider(d);
                consider(cross(u, cross(d, u)));
                consider(cross(v, cross(d, v)));
            }
    }
    return best;
}

// The direction is a proposal; all decisions below use outward intervals.
// A four-slice bound on rounded endpoints is cheap on this rare path. FMA
// interpolation is coordinatewise monotone for interior FP32 fractions;
// beta=1 (the stored target) is checked separately.
IPC_CLEARANCE_HD inline float grouped_fraction(const ccd32::LinearQuery& q, bool& full) {
    full = false;
    const bool face = q.vertex_face;
    const int split = face ? 1 : 2;
    const Vec n = best_axis(q.start, face);
    Interval norm2 = point(0.0);
    for (int k = 0; k < 3; ++k)
        norm2 = add(norm2, multiply(point(n.v[k]), point(n.v[k])));
    const double threshold = multiply(point(double(q.gap_floor)), point(sqrt_upper(norm2.hi))).hi;
    if (!(threshold > 0.0) || !finite(threshold))
        return 0.0f;
    const auto gap = [&](const float* a, const float* b) {
        Interval d = point(0.0);
        for (int k = 0; k < 3; ++k)
            d = add(d, multiply(point(n.v[k]), subtract(point(double(a[k])), point(double(b[k])))));
        return d.lo;
    };
    int mover = -1;
    for (int i = 0; i < 4; ++i) {
        if (q.start[i][0] != q.end[i][0] || q.start[i][1] != q.end[i][1] ||
            q.start[i][2] != q.end[i][2]) {
            if (mover >= 0)
                return 0.0f;
            mover = i;
        }
    }
    if (mover < 0) {
        full = true;
        return 1.0f;
    }
    double cap = 1.0;
    bool target_safe = true;
    for (int a = 0; a < split; ++a)
        for (int b = split; b < 4; ++b) {
            const double begin = gap(q.start[a], q.start[b]);
            if (!finite(begin) || !(begin > threshold))
                return 0.0f;
            if (a != mover && b != mover)
                continue;
            const double end = gap(q.end[a], q.end[b]);
            if (!finite(end))
                return 0.0f;
            const double budget = 0.2 * (begin - threshold);
            target_safe = target_safe && end > threshold && begin - end <= budget;
            if (begin > end)
                cap = minimum(cap, budget / (begin - end));
        }
    full = target_safe;
    const float fraction = static_cast<float>(cap);
    if (!(fraction > 0.0f))
        return 0.0f;
    const float max_interior = ccd32::detail::minimum(fraction, 0x1.fffffep-1f);
    float left[3], right[3];
    for (int k = 0; k < 3; ++k)
        left[k] = q.start[mover][k];
    for (int slice = 1; slice <= 4; ++slice) {
        const float beta = max_interior * (0.25f * static_cast<float>(slice));
        for (int k = 0; k < 3; ++k) {
            right[k] = ccd32::detail::interpolate(q.start[mover][k], q.end[mover][k], beta);
            if (!ccd32::detail::finite(right[k]))
                return 0.0f;
        }
        float worst[3];
        for (int k = 0; k < 3; ++k) {
            const bool low = (mover < split) == (n.v[k] >= 0.);
            worst[k] = low ? ccd32::detail::minimum(left[k], right[k])
                           : ccd32::detail::maximum(left[k], right[k]);
        }
        for (int a = 0; a < split; ++a)
            for (int b = split; b < 4; ++b) {
                if (a != mover && b != mover)
                    continue;
                if (!(gap(a == mover ? worst : q.start[a], b == mover ? worst : q.start[b]) > threshold))
                    return 0.0f;
            }
        for (int k = 0; k < 3; ++k)
            left[k] = right[k];
    }
    if (fraction == 1.0f && !full)
        return max_interior;
    return fraction;
}
} // namespace ipc_gpu::ccd_clearance
#undef IPC_CLEARANCE_HD
