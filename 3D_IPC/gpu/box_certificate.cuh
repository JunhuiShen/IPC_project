#pragma once

#include "broadphase.h"
#include "ccd.cuh"
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define IPC_MOVEMENT_BOX_HD __host__ __device__
#else
#define IPC_MOVEMENT_BOX_HD
#endif

namespace ipc_gpu::solver {

// Optional diagnostic witness. Production callers need only the boolean.
// A successful witness proves gap > required_gap along its max-component-
// normalized axis (projection compensates its non-unit Euclidean norm)
// for EVERY choice of the four vertices inside their respective closed boxes.
struct MovementBoxCertificateWitness {
    double axis[3]{};
    double required_gap = 0.0;
};

namespace movement_box_detail {
using ccd::detail::Interval;
using ccd::detail::Vec;
using ccd::detail::add;
using ccd::detail::subtract;
using ccd::detail::multiply;
using ccd::detail::point;
using ccd::detail::minimum;
using ccd::detail::maximum;
using ccd::detail::finite;

IPC_MOVEMENT_BOX_HD inline Vec normalized_axis(Vec axis) {
    double scale = 0.0;
    for (int k = 0; k < 3; ++k) {
        if (!finite(axis.v[k])) return {{0, 0, 0}};
        scale = maximum(scale, ::fabs(axis.v[k]));
    }
    if (!(scale > 0.0)) return {{0, 0, 0}};
    for (int k = 0; k < 3; ++k) axis.v[k] /= scale;
    return axis;
}

IPC_MOVEMENT_BOX_HD inline bool projection(const BpAabb* boxes,
    const double* origin, bool vertex_face, double required_gap, Vec proposed,
    MovementBoxCertificateWitness* witness) {
    const Vec axis = normalized_axis(proposed);
    if (axis.v[0] == 0.0 && axis.v[1] == 0.0 && axis.v[2] == 0.0) return false;
    Interval norm2 = point(0.0);
    for (int k = 0; k < 3; ++k)
        norm2 = add(norm2, multiply(point(axis.v[k]), point(axis.v[k])));
    const double norm_upper = ccd::detail::sqrt_upper(norm2.hi);
    const double threshold = multiply(point(required_gap), point(norm_upper)).hi;
    if (!finite(threshold)) return false;
    double lo[2] = {INFINITY, INFINITY}, hi[2] = {-INFINITY, -INFINITY};
    const int split = vertex_face ? 1 : 2;
    for (int i = 0; i < 4; ++i) {
        Interval projected = point(0.0);
        for (int k = 0; k < 3; ++k) {
            // Origin subtraction also rounds outwards. Approximate axes only
            // propose directions; every acceptance bounds original box data.
            const Interval coordinate = subtract({boxes[i].lo[k], boxes[i].hi[k]}, point(origin[k]));
            projected = add(projected, multiply(point(axis.v[k]), coordinate));
        }
        const int side = i < split ? 0 : 1;
        lo[side] = minimum(lo[side], projected.lo);
        hi[side] = maximum(hi[side], projected.hi);
    }
    const double forward = subtract(point(lo[0]), point(hi[1])).lo;
    const double backward = subtract(point(lo[1]), point(hi[0])).lo;
    if (!((finite(forward) && forward > threshold)
        || (finite(backward) && backward > threshold))) return false;
    if (witness) {
        for (int k = 0; k < 3; ++k) witness->axis[k] = axis.v[k];
        witness->required_gap = required_gap;
    }
    return true;
}
} // namespace movement_box_detail

// False means UNKNOWN: retain the pair and its ordinary contact/CCD checks.
// True proves strict separation of the complete permitted primitive regions.
// Threshold >= d_hat makes the physical barrier/friction identically inactive.
// Threshold >= max(1e-8, upper(128*DBL_EPSILON*box_coordinate_scale), 1e-10)
// dominates the GPU linear CCD endpoint floor AND path-validation band for
// every later represented position in these boxes, not just the rebuild state.
//
// Preconditions for REUSE: all current and proposed vertices stay in these
// same boxes; every trajectory is inside them; parameters remain unchanged.
// Linear vertex motion satisfies this by convexity. Endpoint-only rigid
// rotation bounds do NOT. Rebuild invalidates the certificate. Keep the full
// conflict graph/coloring; filter only post-color contact/CCD work incidence.
// Materials/indices/error state still need their ordinary validation. This
// geometric proof does not license hiding a nonfinite input or stale box.
// Nor is exact mathematical inactivity alone a universal bitwise proof for
// every approximate floating-point feature implementation: differential and
// full-simulation parity gates are required before replacing its execution.
IPC_MOVEMENT_BOX_HD inline bool certify_movement_box_contact(const BpAabb* boxes,
    const double positions[4][3], bool vertex_face, double d_hat,
    MovementBoxCertificateWitness* witness = nullptr) {
    using namespace movement_box_detail;
    if (witness) *witness = MovementBoxCertificateWitness{};
    if (!boxes || !positions || !finite(d_hat) || !(d_hat >= 0.0)) return false;
    double coordinate_scale = 0.0;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        if (!finite(boxes[i].lo[k]) || !finite(boxes[i].hi[k])
            || !finite(positions[i][k]) || boxes[i].lo[k] > boxes[i].hi[k]
            || positions[i][k] < boxes[i].lo[k] || positions[i][k] > boxes[i].hi[k]) return false;
        coordinate_scale = maximum(coordinate_scale,
            maximum(::fabs(boxes[i].lo[k]), ::fabs(boxes[i].hi[k])));
    }
    const double scaled_floor = multiply(point(128.0 * DBL_EPSILON), point(coordinate_scale)).hi;
    const double required_gap = maximum(d_hat, maximum(1e-8, maximum(1e-10, scaled_floor)));
    if (!finite(required_gap)) return false;
    const double* origin = positions[3];
    for (int k = 0; k < 3; ++k) {
        Vec axis{{0, 0, 0}}; axis.v[k] = 1.0;
        if (projection(boxes, origin, vertex_face, required_gap, axis, witness)) return true;
    }
    using ccd::detail::difference;
    using ccd::detail::cross;
    if (vertex_face) {
        const Vec a = normalized_axis(difference(positions[2], positions[1]));
        const Vec b = normalized_axis(difference(positions[3], positions[1]));
        const Vec normal = normalized_axis(cross(a, b));
        if (projection(boxes, origin, true, required_gap, normal, witness)) return true;
        for (int i = 1; i < 4; ++i) {
            const Vec edge = normalized_axis(difference(positions[i == 3 ? 1 : i+1], positions[i]));
            if (projection(boxes, origin, true, required_gap, cross(normal, edge), witness)) return true;
        }
    } else {
        const Vec a = normalized_axis(difference(positions[1], positions[0]));
        const Vec b = normalized_axis(difference(positions[3], positions[2]));
        const Vec w = normalized_axis(difference(positions[2], positions[0]));
        if (projection(boxes, origin, false, required_gap, cross(a,b), witness)
            || projection(boxes, origin, false, required_gap, cross(cross(a,w),a), witness)
            || projection(boxes, origin, false, required_gap, cross(cross(b,w),b), witness)) return true;
    }
    const int split = vertex_face ? 1 : 2;
    for (int i = 0; i < split; ++i) for (int j = split; j < 4; ++j)
        if (projection(boxes, origin, vertex_face, required_gap,
                difference(positions[i], positions[j]), witness)) return true;
    return false;
}
} // namespace ipc_gpu::solver

#undef IPC_MOVEMENT_BOX_HD
