#pragma once

// ORIGINAL (exact_computation_fallback == false) color-predictor helpers.
//
// These reproduce the CPU false predictor's floating-point policy from
// initial_guess.cpp/solver.cpp without touching the qualified
// original_contact/original_ccd component bodies:
//   * the false projected supporting-plane certificate
//     (contact_boxes_separated with exact_computation_fallback == false),
//   * the fixed 1e-8 predictor gap,
//   * the per-pair min(startgap, 1e-8) tested-endpoint refinement applied
//     after each positive original step.
//
// All routines are device-capable ordinary binary64 arithmetic. They never
// perform a host geometry query and never alter a qualified component.

#include "broadphase.h"
#include "level_schedule.cuh"     // stable_norm3 (Eigen stableNorm order)
#include "original_contact.cuh"   // original CPU PT/EE distance ports

#include <cfloat>
#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#define IPC_ORIGINAL_GUESS_HD __host__ __device__
#else
#define IPC_ORIGINAL_GUESS_HD
#endif

namespace ipc_gpu::solver::original_guess {

namespace geometry = ::ipc_gpu::solver::original_contact_detail;

// CPU colored_guess_min_separation.
constexpr double kMinimumSeparation = 1.0e-8;

// CPU guess_point_triangle_distance (initial_guess.cpp). The ported
// point_triangle_distance already minimizes over the three closed edges
// outside the face, matching node_triangle_distance's exterior branch, so its
// distance is the same quantity the CPU predictor compares.
IPC_ORIGINAL_GUESS_HD inline double point_triangle_guess_distance(
    const double x[4][3]) {
    geometry::PtGeometry g;
    geometry::point_triangle_distance(x, 1.0e-12, g);
    return g.distance;
}

// CPU segment_segment_distance(..., 1e-12, false) == the fast false solve.
IPC_ORIGINAL_GUESS_HD inline double edge_edge_guess_distance(
    const double x[4][3]) {
    geometry::EeGeometry g;
    geometry::segment_segment_distance_fast(x, 1.0e-12, g);
    return g.distance;
}

IPC_ORIGINAL_GUESS_HD inline double contact_guess_distance(
    const double x[4][3], bool face) {
    return face ? point_triangle_guess_distance(x)
                : edge_edge_guess_distance(x);
}

// CPU solver_detail::contact_boxes_separated for the ORIGINAL false solver.
// The original and robust variants share the projected four-box formula; only
// the segment-segment distance backend differs. This port always uses the
// original fast EE distance and the CPU's strict
// 512*eps*(magnitude+threshold)+512*denorm_min padding.
IPC_ORIGINAL_GUESS_HD inline bool boxes_separated_original(
    const double positions[4][3], const BpAabb boxes[4],
    bool segment_segment, double d_hat) {
    if (!(d_hat > 0.0) || !level_predictor::finite_value(d_hat)) return false;
    for (int node = 0; node < 4; ++node) {
        for (int axis = 0; axis < 3; ++axis) {
            const double lo = boxes[node].lo[axis];
            const double hi = boxes[node].hi[axis];
            if (!level_predictor::finite_value(positions[node][axis]) || !level_predictor::finite_value(lo)
                || !level_predictor::finite_value(hi))
                return false;
            if (!(hi - lo >= 2e-10) || !(lo + 1e-10 <= hi - 1e-10)
                || positions[node][axis] < lo || positions[node][axis] > hi)
                return false;
        }
    }
    double direction[3];
    if (segment_segment) {
        geometry::EeGeometry g;
        geometry::segment_segment_distance_fast(positions, 1.0e-12, g);
        for (int k = 0; k < 3; ++k)
            direction[k] = g.closest_point_1[k] - g.closest_point_2[k];
    } else {
        geometry::PtGeometry g;
        geometry::point_triangle_distance(positions, 1.0e-12, g);
        for (int k = 0; k < 3; ++k)
            direction[k] = positions[0][k] - g.closest[k];
    }
    bool finite = true;
    double scale = 0.0;
    for (int k = 0; k < 3; ++k) {
        finite = finite && level_predictor::finite_value(direction[k]);
        const double magnitude = ::fabs(direction[k]);
        if (magnitude > scale) scale = magnitude;
    }
    if (!finite || !(scale > 0.0) || !level_predictor::finite_value(scale)) return false;
    for (int k = 0; k < 3; ++k) direction[k] /= scale;

    double lower[4], upper[4], magnitude = 0.0;
    for (int node = 0; node < 4; ++node) {
        lower[node] = 0.0;
        upper[node] = 0.0;
        double absolute_sum = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double lo = direction[axis] >= 0.0
                ? boxes[node].lo[axis] : boxes[node].hi[axis];
            const double hi = direction[axis] >= 0.0
                ? boxes[node].hi[axis] : boxes[node].lo[axis];
            lower[node] += direction[axis] * lo;
            upper[node] += direction[axis] * hi;
            absolute_sum += ::fabs(direction[axis])
                * ::fmax(::fabs(lo), ::fabs(hi));
        }
        if (!level_predictor::finite_value(lower[node]) || !level_predictor::finite_value(upper[node])
            || !level_predictor::finite_value(absolute_sum))
            return false;
        magnitude = ::fmax(magnitude, absolute_sum);
    }
    const int split = segment_segment ? 2 : 1;
    double alo = lower[0], ahi = upper[0];
    double blo = lower[split], bhi = upper[split];
    for (int node = 1; node < split; ++node) {
        alo = ::fmin(alo, lower[node]);
        ahi = ::fmax(ahi, upper[node]);
    }
    for (int node = split + 1; node < 4; ++node) {
        blo = ::fmin(blo, lower[node]);
        bhi = ::fmax(bhi, upper[node]);
    }
    double norm_squared = 0.0;
    for (int k = 0; k < 3; ++k) norm_squared += direction[k] * direction[k];
    const double threshold = d_hat * ::sqrt(norm_squared);
    const double gap = ::fmax(alo - bhi, blo - ahi);
    // 0x1p-1074 is exactly std::numeric_limits<double>::denorm_min() and is
    // safe to use from device code without a constexpr library call.
    const double padding = 512.0 * DBL_EPSILON * (magnitude + threshold)
        + 512.0 * 0x1p-1074;
    return level_predictor::finite_value(threshold) && level_predictor::finite_value(gap)
        && level_predictor::finite_value(padding) && gap > threshold + padding;
}

// Distance from the moving vertex to one incident primitive, with `candidate`
// replacing that vertex's coordinate and every other role taken from the
// immutable current state.
IPC_ORIGINAL_GUESS_HD inline double incident_distance(
    const BpContact& contact, int node, const double (*positions)[3],
    const double candidate[3]) {
    double x[4][3];
    for (int role = 0; role < 4; ++role) {
        const int vertex = contact.vertices[role];
        for (int k = 0; k < 3; ++k)
            x[role][k] = vertex == node ? candidate[k] : positions[vertex][k];
    }
    return contact_guess_distance(x, contact.kind == 0);
}

}  // namespace ipc_gpu::solver::original_guess

#undef IPC_ORIGINAL_GUESS_HD
