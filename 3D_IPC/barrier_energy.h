#pragma once

#include "IPC_math.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

//  Scalar barrier
double scalar_barrier(double delta, double d_hat);

//  Scalar barrier gradient  db/d(delta)
double scalar_barrier_gradient(double delta, double d_hat);

//  Scalar barrier hessian  d2b/d(delta)2
double scalar_barrier_hessian(double delta, double d_hat);

// ====================================================================
//  Node--triangle barrier
// ====================================================================
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

// 12x12 Hessian: rows/cols ordered as (x, x1, x2, x3) x (0,1,2)
using Mat12 = Eigen::Matrix<double, 12, 12>;

struct NodeTriangleBarrierHessianResult{
    double energy{0.0};
    double distance{0.0};

    Vec3 grad_x  = Vec3::Zero();
    Vec3 grad_x1 = Vec3::Zero();
    Vec3 grad_x2 = Vec3::Zero();
    Vec3 grad_x3 = Vec3::Zero();

    Mat12 hessian = Mat12::Zero();  // d2PE / dy_{pk} dy_{ql}

    NodeTriangleDistanceResult distance_result;
};

//  Barrier energy with node-triangle distance
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps = 1.0e-12);

//  Barrier energy gradient with node-triangle distance
NodeTriangleBarrierResult node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                                         double d_hat, double eps = 1.0e-12);

//  Barrier energy hessian with node-triangle distance
NodeTriangleBarrierHessianResult node_triangle_barrier_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                                               double d_hat, double eps = 1.0e-12);

// ====================================================================
//  Segment--segment barrier
// ====================================================================

struct SegmentSegmentBarrierResult{
    double energy{0.0};
    double distance{0.0};
    double barrier_derivative{0.0}; // db/d(delta)

    Vec3 grad_x1 = Vec3::Zero(); // dPE/d(x1)_k
    Vec3 grad_x2 = Vec3::Zero();  // dPE/d(x2)_k
    Vec3 grad_x3 = Vec3::Zero();  // dPE/d(x3)_k
    Vec3 grad_x4 = Vec3::Zero();  // dPE/d(x4)_k

    SegmentSegmentDistanceResult distance_result;
};

struct SegmentSegmentBarrierHessianResult{
    double energy{0.0};
    double distance{0.0};

    Vec3 grad_x1 = Vec3::Zero();
    Vec3 grad_x2 = Vec3::Zero();
    Vec3 grad_x3 = Vec3::Zero();
    Vec3 grad_x4 = Vec3::Zero();

    Mat12 hessian = Mat12::Zero(); // d2PE / dy_{pk} dy_{ql}, ordered (x1,x2,x3,x4)

    SegmentSegmentDistanceResult distance_result;
};

double segment_segment_barrier(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12);

SegmentSegmentBarrierResult segment_segment_barrier_gradient(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12);

SegmentSegmentBarrierHessianResult segment_segment_barrier_hessian(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12);