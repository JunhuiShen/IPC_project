#pragma once

#include "IPC_math.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

//  Scalar barrier
double scalar_barrier(double delta, double d_hat);
double scalar_barrier_gradient(double delta, double d_hat);
double scalar_barrier_hessian(double delta, double d_hat);

// Node--triangle barrier with DOF ordering: 0=x, 1=x1, 2=x2, 3=x3
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps = 1.0e-12);

Vec3 node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12);

Mat312 node_triangle_barrier_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                     double d_hat, int dof, double eps = 1.0e-12);

// Returns both gradient and hessian row for dof
std::pair<Vec3, Mat312> node_triangle_barrier_gradient_and_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                                                   double d_hat, int dof, double eps = 1.0e-12);

// Segment--segment barrier with DOF ordering: 0=x1, 1=x2, 2=x3, 3=x4
double segment_segment_barrier(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12);

Vec3 segment_segment_barrier_gradient(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12);

Mat312 segment_segment_barrier_hessian(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12);

// Returns both gradient and hessian row for dof
std::pair<Vec3, Mat312> segment_segment_barrier_gradient_and_hessian(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                                                     double d_hat, int dof, double eps = 1.0e-12);