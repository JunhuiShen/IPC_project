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

// Node--triangle barrier with DOF ordering: p = 0(x), 1(x1), 2(x2), 3(x3)
//  Scalar barrier energy only (no gradient)
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps = 1.0e-12);

//  Gradient for a single DOF p in {0,1,2,3}.
//  Returns the 3-vector dE/d(y_p).
Vec3 node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12);

//  Hessian row for a single DOF p in {0,1,2,3}.
//  Returns the 3x12 block  d^2E / (d(y_p) d(y_all)) where columns are ordered (x, x1, x2, x3) x (0,1,2).
Mat312 node_triangle_barrier_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12);

//  Segment--segment barrier with DOF ordering: p = 0(x1), 1(x2), 2(x3), 3(x4)
//  Scalar barrier energy only
double segment_segment_barrier(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12);

//  Gradient for a single DOF p in {0,1,2,3}.
//  Returns the 3-vector dE/d(y_p).
Vec3 segment_segment_barrier_gradient(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12);

//  Hessian row for a single DOF p in {0,1,2,3}.
//  Returns the 3x12 block  d^2E / (d(y_p) d(y_all)) where columns are ordered (x1, x2, x3, x4) x (0,1,2).
Mat312 segment_segment_barrier_hessian(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12);