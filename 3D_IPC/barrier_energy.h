#pragma once

#include "IPC_math.h"
#include "node_triangle_distance.h"

//  Scalar barrier
double scalar_barrier(double delta, double d_hat);

//  Scalar barrier gradient
double scalar_barrier_gradient(double delta, double d_hat);

struct NodeTriangleBarrierResult{
    double energy{0.0};
    double distance{0.0};
    double barrier_derivative{0.0};   // db/d(delta)

    Vec3 grad_x  = Vec3::Zero();      // dPE/dx_k
    Vec3 grad_x1 = Vec3::Zero();      // dPE/d(x1)_k
    Vec3 grad_x2 = Vec3::Zero();      // dPE/d(x2)_k
    Vec3 grad_x3 = Vec3::Zero();      // dPE/d(x3)_k

    NodeTriangleDistanceResult distance_result;
};

//  Barrier energy with node-triangle distance
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps = 1.0e-12);

//  Barrier energy gradient with node-triangle distance
NodeTriangleBarrierResult node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                                         double d_hat, double eps = 1.0e-12);
