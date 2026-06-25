#pragma once

#include "ipc_math.h"

#include <vector>

struct RigidBarrierGradient {
    Vec2 translation{0.0, 0.0};
    double rotation = 0.0;
};

struct RigidBarrierHessian {
    Mat2 translation_translation{0.0, 0.0, 0.0, 0.0};
    Vec2 translation_rotation{0.0, 0.0};
    double rotation_rotation = 0.0;
};

double barrier_energy(double d, double dhat);
double barrier_grad(double d, double dhat);
double barrier_hess(double d, double dhat);

double node_segment_barrier_energy(const Vec& x, int node, int seg0, int seg1, double dhat);
Vec2 local_barrier_grad(int who, const Vec& x, int node, int seg0, int seg1, double dhat);
Mat2 local_barrier_hess(int who, const Vec& x, int node, int seg0, int seg1, double dhat);

RigidBarrierGradient local_barrier_grad_rb(const std::vector<int>& rb_nodes, const Vec& x, const Vec2& x_com, int node, int seg0, int seg1, double dhat);
RigidBarrierHessian local_barrier_hess_rb(const std::vector<int>& rb_nodes, const Vec& x, const Vec2& x_com, int node, int seg0, int seg1, double dhat);
