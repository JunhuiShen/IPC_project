#pragma once

#include "IPC_math.h"
#include "node_triangle_distance.h"
#include "rigid_body_ipc.h"
#include "segment_segment_distance.h"

#include <array>

//  Scalar barrier
double scalar_barrier(double delta, double d_hat);
double scalar_barrier_gradient(double delta, double d_hat);
double scalar_barrier_hessian(double delta, double d_hat);

// Node--triangle barrier with DOF ordering: 0=x, 1=x1, 2=x2, 3=x3
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps = 1.0e-12);

Vec3 node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                    double d_hat, int dof, double eps = 1.0e-12,
                                    const NodeTriangleDistanceResult* precomputed_dr = nullptr);

// Hessian block H(row_dof, col_dof), where H(k,l) is the derivative of
// gradient(row_dof)(k) with respect to coordinate l of col_dof.
Mat33 node_triangle_barrier_cross_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
        double d_hat, int row_dof, int col_dof, double eps = 1.0e-12,
        const NodeTriangleDistanceResult* precomputed_dr = nullptr);

// Self/diagonal Hessian block H(dof, dof).
Mat33 node_triangle_barrier_self_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                         double d_hat, int dof, double eps = 1.0e-12,
                                         const NodeTriangleDistanceResult* precomputed_dr = nullptr);

// Returns both the gradient and self/diagonal Hessian block for dof.
std::pair<Vec3, Mat33> node_triangle_barrier_self_gradient_and_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                                                       double d_hat, int dof, double eps = 1.0e-12);

// Segment--segment barrier with DOF ordering: 0=x1, 1=x2, 2=x3, 3=x4
double segment_segment_barrier(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12);

Vec3 segment_segment_barrier_gradient(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                      double d_hat, int dof, double eps = 1.0e-12,
                                      const SegmentSegmentDistanceResult* precomputed_dr = nullptr);

// Hessian block H(row_dof, col_dof), where H(k,l) is the derivative of
// gradient(row_dof)(k) with respect to coordinate l of col_dof.
Mat33 segment_segment_barrier_cross_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double d_hat, int row_dof, int col_dof, double eps = 1.0e-12,
        const SegmentSegmentDistanceResult* precomputed_dr = nullptr);

// Self/diagonal Hessian block H(dof, dof).
Mat33 segment_segment_barrier_self_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double d_hat, int dof, double eps = 1.0e-12,
        const SegmentSegmentDistanceResult* precomputed_dr = nullptr);

// Returns both the gradient and self/diagonal Hessian block for dof.
std::pair<Vec3, Mat33> segment_segment_barrier_self_gradient_and_hessian(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                                                         double d_hat, int dof, double eps = 1.0e-12);

enum class RigidBarrierSide {
    FirstPrimitive,
    SecondPrimitive
};

// FirstPrimitive selects the node; SecondPrimitive selects the triangle.
// X_centered entries on the unselected side are ignored and may be zero.
RigidEnergyDerivatives node_triangle_barrier_rb(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, const std::array<Vec3, 4>& X_centered, RigidBarrierSide side, const Vec4& q_n, const Vec3& omega, double dt, double d_hat, double eps = 1.0e-12);

// FirstPrimitive selects (x1,x2); SecondPrimitive selects (x3,x4).
// X_centered entries on the unselected side are ignored and may be zero.
RigidEnergyDerivatives segment_segment_barrier_rb(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, const std::array<Vec3, 4>& X_centered, RigidBarrierSide side, const Vec4& q_n, const Vec3& omega, double dt, double d_hat, double eps = 1.0e-12);
