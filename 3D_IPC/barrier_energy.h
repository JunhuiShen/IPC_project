#pragma once

#include "IPC_math.h"
#include "node_triangle_distance.h"
#include "rigid_body_ipc.h"
#include "segment_segment_distance.h"

#include <array>
#include <utility>

//  Scalar barrier
double scalar_barrier(double delta, double d_hat);
double scalar_barrier_gradient(double delta, double d_hat);
double scalar_barrier_hessian(double delta, double d_hat);

// Ephemeral closest-distance and scalar-barrier data shared by the normal and
// friction kernels for one mesh contact evaluation point. The normal load is
// unscaled by dt^2. Friction-only data (feature weights and tangent frame) is
// intentionally formed by the friction builder, so normal-only evaluations do
// not pay for it.
template <typename DistanceResult>
struct MeshContactEvaluation {
    DistanceResult dr{};
    bool active = false;
    double b_prime = 0.0;
    double b_double_prime = 0.0;
    double normal_load = 0.0;
    double d_hat = 0.0;
};

using NodeTriangleContactEvaluation =
        MeshContactEvaluation<NodeTriangleDistanceResult>;
using SegmentSegmentContactEvaluation =
        MeshContactEvaluation<SegmentSegmentDistanceResult>;

NodeTriangleContactEvaluation make_node_triangle_contact_evaluation(
        const std::array<Vec3, 4>& positions,
        double d_hat, double k_barrier, double eps = 1.0e-12,
        const NodeTriangleDistanceResult* precomputed_dr = nullptr);

SegmentSegmentContactEvaluation make_segment_segment_contact_evaluation(
        const std::array<Vec3, 4>& positions,
        double d_hat, double k_barrier, double eps = 1.0e-12,
        const SegmentSegmentDistanceResult* precomputed_dr = nullptr);

std::array<double, 4> node_triangle_contact_weights(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
        double eps = 1.0e-12,
        const NodeTriangleDistanceResult* precomputed_dr = nullptr);

std::array<double, 4> segment_segment_contact_weights(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double eps = 1.0e-12,
        const SegmentSegmentDistanceResult* precomputed_dr = nullptr);

// Node--triangle barrier with DOF ordering: 0=x, 1=x1, 2=x2, 3=x3
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps = 1.0e-12);

Vec3 node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                    double d_hat, int dof, double eps = 1.0e-12,
                                    const NodeTriangleDistanceResult* precomputed_dr = nullptr,
                                    const double* precomputed_scalar_gradient = nullptr);

Vec3 node_triangle_barrier_gradient(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
        int dof, const NodeTriangleContactEvaluation& evaluation,
        double eps = 1.0e-12);

// Hessian block H(row_dof, col_dof), where H(k,l) is the derivative of
// gradient(row_dof)(k) with respect to coordinate l of col_dof.
Mat33 node_triangle_barrier_cross_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
        double d_hat, int row_dof, int col_dof, double eps = 1.0e-12,
        const NodeTriangleDistanceResult* precomputed_dr = nullptr,
        const double* precomputed_scalar_gradient = nullptr,
        const double* precomputed_scalar_hessian = nullptr);

Mat33 node_triangle_barrier_cross_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
        int row_dof, int col_dof,
        const NodeTriangleContactEvaluation& evaluation,
        double eps = 1.0e-12);

// Self/diagonal Hessian block H(dof, dof).
Mat33 node_triangle_barrier_self_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                         double d_hat, int dof, double eps = 1.0e-12,
                                         const NodeTriangleDistanceResult* precomputed_dr = nullptr,
                                         const double* precomputed_scalar_gradient = nullptr,
                                         const double* precomputed_scalar_hessian = nullptr);

Mat33 node_triangle_barrier_self_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
        int dof, const NodeTriangleContactEvaluation& evaluation,
        double eps = 1.0e-12);

// Returns both the gradient and self/diagonal Hessian block for dof.
std::pair<Vec3, Mat33> node_triangle_barrier_self_gradient_and_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                                                       double d_hat, int dof, double eps = 1.0e-12,
                                                                       const NodeTriangleDistanceResult* precomputed_dr = nullptr,
                                                                       const double* precomputed_scalar_gradient = nullptr,
                                                                       const double* precomputed_scalar_hessian = nullptr);

std::pair<Vec3, Mat33> node_triangle_barrier_self_gradient_and_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
        int dof, const NodeTriangleContactEvaluation& evaluation,
        double eps = 1.0e-12);

// Segment--segment barrier with DOF ordering: 0=x1, 1=x2, 2=x3, 3=x4
double segment_segment_barrier(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12);

Vec3 segment_segment_barrier_gradient(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                      double d_hat, int dof, double eps = 1.0e-12,
                                      const SegmentSegmentDistanceResult* precomputed_dr = nullptr,
                                      const double* precomputed_scalar_gradient = nullptr);

Vec3 segment_segment_barrier_gradient(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        int dof, const SegmentSegmentContactEvaluation& evaluation,
        double eps = 1.0e-12);

// Hessian block H(row_dof, col_dof), where H(k,l) is the derivative of
// gradient(row_dof)(k) with respect to coordinate l of col_dof.
Mat33 segment_segment_barrier_cross_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double d_hat, int row_dof, int col_dof, double eps = 1.0e-12,
        const SegmentSegmentDistanceResult* precomputed_dr = nullptr,
        const double* precomputed_scalar_gradient = nullptr,
        const double* precomputed_scalar_hessian = nullptr);

Mat33 segment_segment_barrier_cross_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        int row_dof, int col_dof,
        const SegmentSegmentContactEvaluation& evaluation,
        double eps = 1.0e-12);

// Self/diagonal Hessian block H(dof, dof).
Mat33 segment_segment_barrier_self_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double d_hat, int dof, double eps = 1.0e-12,
        const SegmentSegmentDistanceResult* precomputed_dr = nullptr,
        const double* precomputed_scalar_gradient = nullptr,
        const double* precomputed_scalar_hessian = nullptr);

Mat33 segment_segment_barrier_self_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        int dof, const SegmentSegmentContactEvaluation& evaluation,
        double eps = 1.0e-12);

// Returns both the gradient and self/diagonal Hessian block for dof.
std::pair<Vec3, Mat33> segment_segment_barrier_self_gradient_and_hessian(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                                                         double d_hat, int dof, double eps = 1.0e-12,
                                                                         const SegmentSegmentDistanceResult* precomputed_dr = nullptr,
                                                                         const double* precomputed_scalar_gradient = nullptr,
                                                                         const double* precomputed_scalar_hessian = nullptr);

std::pair<Vec3, Mat33> segment_segment_barrier_self_gradient_and_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        int dof, const SegmentSegmentContactEvaluation& evaluation,
        double eps = 1.0e-12);

enum class RigidBarrierSide {
    FirstPrimitive,
    SecondPrimitive
};

enum class RigidDerivativeMode {
    Full,
    Gradient,
    TranslationHessian,
    OrientationHessian
};

// FirstPrimitive selects the node; SecondPrimitive selects the triangle.
// X_centered entries on the unselected side are ignored and may be zero.
RigidEnergyDerivatives node_triangle_barrier_rb(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, const std::array<Vec3, 4>& X_centered, RigidBarrierSide side, const Vec4& q_n, const Vec3& omega, double dt, double d_hat, RigidDerivativeMode mode = RigidDerivativeMode::Full, double eps = 1.0e-12, const QuaternionOmegaKinematics* cached_kinematics = nullptr, const NodeTriangleDistanceResult* precomputed_dr = nullptr, const double* precomputed_scalar_gradient = nullptr, const double* precomputed_scalar_hessian = nullptr);

// FirstPrimitive selects (x1,x2); SecondPrimitive selects (x3,x4).
// X_centered entries on the unselected side are ignored and may be zero.
RigidEnergyDerivatives segment_segment_barrier_rb(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, const std::array<Vec3, 4>& X_centered, RigidBarrierSide side, const Vec4& q_n, const Vec3& omega, double dt, double d_hat, RigidDerivativeMode mode = RigidDerivativeMode::Full, double eps = 1.0e-12, const QuaternionOmegaKinematics* cached_kinematics = nullptr, const SegmentSegmentDistanceResult* precomputed_dr = nullptr, const double* precomputed_scalar_gradient = nullptr, const double* precomputed_scalar_hessian = nullptr);
