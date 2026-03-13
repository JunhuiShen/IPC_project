#pragma once

#include "ipc_math.h"
#include <vector>

// ======================================================
// Physics: spring energy, IPC barrier energy,
// and their analytic gradients and Hessians
// ======================================================

namespace physics {
    using namespace math;

    struct NodeSegmentPair {
        int node;   // i
        int seg0;   // j
        int seg1;   // j+1
    };

    // --- Spring energy ---

    Vec2 local_spring_grad(int i, const Vec &x, double k, const std::vector<double> &L);
    Mat2 local_spring_hess(int i, const Vec &x, double k, const std::vector<double> &L);

    // --- Barrier (scalar) ---

    double barrier_energy(double d, double dhat);
    double barrier_grad  (double d, double dhat);
    double barrier_hess  (double d, double dhat);

    // --- Point-segment geometry ---

    double node_segment_distance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1,
                                 double &t, Vec2 &p, Vec2 &r);

    // --- Barrier gradient / Hessian for a node-segment pair ---

    Vec2 local_barrier_grad(int who, const Vec &x, int node, int seg0, int seg1, double dhat);
    Mat2 local_barrier_hess(int who, const Vec &x, int node, int seg0, int seg1, double dhat);

    // --- Incremental potential (no barrier) ---

    Vec2 local_grad_no_barrier(int i, const Vec &x, const Vec &xhat, const Vec &xpin,
                               const std::vector<double> &mass, const std::vector<double> &L,
                               double dt, double k, const Vec2 &g_accel);

    Mat2 local_hess_no_barrier(int i, const Vec &x,
                               const std::vector<double> &mass, const std::vector<double> &L,
                               double dt, double k);

} // namespace physics
