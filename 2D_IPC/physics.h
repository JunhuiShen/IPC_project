#pragma once

#include "barrier_energy.h"
#include "ipc_math.h"
#include "node_segment_distance.h"
#include "spring_energy.h"
#include <vector>

// ======================================================
// Physics: spring energy, IPC barrier energy,
// and their analytic gradients and Hessians
// ======================================================

namespace physics {
    using namespace math;

    struct NodeSegmentPair {
        int node;
        int seg0;
        int seg1;
    };

    // --- Incremental potential (no barrier) ---

    Vec2 local_grad_no_barrier(int i, const Vec &x, const Vec &xhat, const Vec &xpin,
                               const std::vector<double> &mass,
                               const std::vector<double> &L,
                               int rest_offset,
                               const std::vector<char> &is_pinned,
                               double dt, double k, const Vec2 &g_accel);

    Mat2 local_hess_no_barrier(int i, const Vec &x,
                               const std::vector<double> &mass,
                               const std::vector<double> &L,
                               int rest_offset,
                               const std::vector<char> &is_pinned,
                               double dt, double k);

}
