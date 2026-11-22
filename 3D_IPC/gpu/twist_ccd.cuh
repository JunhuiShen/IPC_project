#pragma once
#include "ccd.cuh"

#if defined(__CUDACC__)
#define IPC_TWIST_HD __host__ __device__
#else
#define IPC_TWIST_HD
#endif

namespace ipc_gpu::twist_ccd {
using namespace ccd::detail;

struct Bounds3 {
    Interval v[3];
};

IPC_TWIST_HD inline Bounds3 relative(const float* a, const float* b) {
    Bounds3 r;
    for (int k = 0; k < 3; ++k)
        r.v[k] = subtract(point(double(a[k])), point(double(b[k])));
    return r;
}
IPC_TWIST_HD inline Bounds3 mix(const Bounds3& x, const Bounds3& y, double t) {
    if (t == 0.)
        return x;
    if (t == 1.)
        return y;
    Bounds3 r;
    for (int k = 0; k < 3; ++k)
        r.v[k] = add(multiply(point(1. - t), x.v[k]), multiply(point(t), y.v[k]));
    return r;
}
IPC_TWIST_HD inline Bounds3 normal(const Bounds3& x, const Bounds3& y) {
    Bounds3 r;
    for (int k = 0; k < 3; ++k) {
        const int i = (k + 1) % 3, j = (k + 2) % 3;
        r.v[k] = subtract(multiply(x.v[i], y.v[j]), multiply(x.v[j], y.v[i]));
    }
    return r;
}
IPC_TWIST_HD inline Interval inner(const Bounds3& x, const Bounds3& y) {
    Interval sum = point(0.);
    for (int k = 0; k < 3; ++k)
        sum = add(sum, multiply(x.v[k], y.v[k]));
    return sum;
}

// The supporting-plane distance for PT (or supporting-line distance for EE)
// is |det(a,b,w)| / |a x b|. The scalar triple product is cubic in time;
// the Bernstein control intervals enclose its entire range for affine motion.
IPC_TWIST_HD inline bool volume_separates(const Bounds3* x, const Bounds3* y, double gap) {
    const auto n0 = normal(x[0], x[1]), n2 = normal(y[0], y[1]);
    const auto left = normal(x[0], y[1]), right = normal(y[0], x[1]);
    Bounds3 n1;
    Interval norm2 = point(0.);
    for (int k = 0; k < 3; ++k) {
        n1.v[k] = multiply(point(.5), add(left.v[k], right.v[k]));
        const double lo = minimum(n0.v[k].lo, minimum(n1.v[k].lo, n2.v[k].lo));
        const double hi = maximum(n0.v[k].hi, maximum(n1.v[k].hi, n2.v[k].hi));
        const double magnitude = maximum(fabs(lo), fabs(hi));
        norm2 = add(norm2, multiply(point(magnitude), point(magnitude)));
    }
    const double h = multiply(point(gap), point(sqrt_upper(norm2.hi))).hi;
    const double h3 = multiply(point(3.), point(h)).hi;
    if (!finite(h3) || !(h > 0.))
        return false;
    const auto v0 = inner(n0, x[2]), v3 = inner(n2, y[2]);
    const auto v1 = add(inner(n0, y[2]), multiply(point(2.), inner(n1, x[2])));
    const auto v2 = add(inner(n2, x[2]), multiply(point(2.), inner(n1, y[2])));
    return (v0.lo > h && v1.lo > h3 && v2.lo > h3 && v3.lo > h) ||
           (v0.hi < -h && v1.hi < -h3 && v2.hi < -h3 && v3.hi < -h);
}
// Midpoint directions only propose a separating axis. Projection at both
// ends of a time slice certifies every synchronized affine corner gap on it.
IPC_TWIST_HD inline bool slice_separated(const float start[4][3], const float end[4][3], bool face,
                                         double h, double l, double r) {
    Vec p[4];
    Bounds3 low[4], high[4];
    for (int i = 0; i < 4; ++i) {
        const auto a = relative(start[i], start[3]);
        const auto b = relative(end[i], end[3]);
        low[i] = mix(a, b, l);
        high[i] = mix(a, b, r);
        for (int k = 0; k < 3; ++k)
            p[i].v[k] = .5 * (low[i].v[k].lo + high[i].v[k].hi);
    }
    const int split = face ? 1 : 2;
    const auto separates = [&](Vec axis) {
        const double scale = maximum(fabs(axis.v[0]), maximum(fabs(axis.v[1]), fabs(axis.v[2])));
        if (!(scale > 0.) || !finite(scale))
            return false;
        Bounds3 n;
        for (int k = 0; k < 3; ++k)
            n.v[k] = point(axis.v[k] / scale);
        const double threshold = multiply(point(h), point(sqrt_upper(inner(n, n).hi))).hi;
        if (!finite(threshold))
            return false;
        bool forward = true, reverse = true;
        for (int time = 0; time < 2 && (forward || reverse); ++time) {
            const auto* corners = time ? high : low;
            double amin = INFINITY, amax = -INFINITY, bmin = INFINITY, bmax = -INFINITY;
            for (int i = 0; i < 4; ++i) {
                const auto gap = inner(n, corners[i]);
                if (!finite(gap.lo) || !finite(gap.hi))
                    return false;
                if (i < split) {
                    amin = minimum(amin, gap.lo);
                    amax = maximum(amax, gap.hi);
                } else {
                    bmin = minimum(bmin, gap.lo);
                    bmax = maximum(bmax, gap.hi);
                }
            }
            forward = forward && subtract(point(amin), point(bmax)).lo > threshold;
            reverse = reverse && subtract(point(bmin), point(amax)).lo > threshold;
        }
        return forward || reverse;
    };
    const Vec u = difference(p[1].v, p[0].v), v = difference(p[3].v, p[2].v);
    if (face) {
        const Vec n = cross(difference(p[2].v, p[1].v), v);
        if (separates(n))
            return true;
        for (int i = 1; i < 4; ++i) {
            const Vec edge = difference(p[i == 3 ? 1 : i + 1].v, p[i].v);
            const Vec offset = difference(p[0].v, p[i].v);
            if (separates(cross(edge, cross(offset, edge))) || separates(offset))
                return true;
        }
    } else {
        if (separates(cross(u, v)))
            return true;
        for (int i = 0; i < 2; ++i)
            for (int j = 2; j < 4; ++j) {
                const Vec offset = difference(p[i].v, p[j].v);
                if (separates(offset) || separates(cross(u, cross(offset, u))) ||
                    separates(cross(v, cross(offset, v))))
                    return true;
            }
    }
    return false;
}

IPC_TWIST_HD inline bool swept_certificate(const float start[4][3], const float end[4][3], bool face,
                                           double gap) {
    const int first[3] = {face ? 2 : 1, 3, face ? 0 : 2};
    const int second[3] = {face ? 1 : 0, face ? 1 : 2, face ? 1 : 0};
    Bounds3 a[3], b[3];
    for (int j = 0; j < 3; ++j) {
        a[j] = relative(start[first[j]], start[second[j]]);
        b[j] = relative(end[first[j]], end[second[j]]);
    }
    // Depth-first dyadic slices. Exhaustion is inconclusive.
    int node = 1, depth = 0;
    for (int attempt = 0; attempt < 1024; ++attempt) {
        int width = 1 << depth;
        double l = double(node - width) / width, r = double(node - width + 1) / width;
        Bounds3 lo[3], hi[3];
        for (int j = 0; j < 3; ++j) {
            lo[j] = mix(a[j], b[j], l);
            hi[j] = mix(a[j], b[j], r);
        }
        if (volume_separates(lo, hi, gap) || slice_separated(start, end, face, gap, l, r)) {
            while (node > 1 && (node & 1)) {
                node >>= 1;
                --depth;
            }
            if (node == 1)
                return true;
            ++node;
        } else {
            if (depth == 16)
                return false;
            node *= 2;
            ++depth;
        }
    }
    return false;
}
} // namespace ipc_gpu::twist_ccd

#undef IPC_TWIST_HD
