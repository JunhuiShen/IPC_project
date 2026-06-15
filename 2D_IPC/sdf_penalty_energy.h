#pragma once

#include "ipc_math.h"

struct SDFEvaluation {
    double phi = 0.0;
    Vec2 grad_phi{0.0, 0.0};
    Mat2 hess_phi{0.0, 0.0, 0.0, 0.0};
};

double sdf_heaviside(double z, double eps);

double sdf_heaviside_gradient(double z, double eps);

struct GroundSDF {
    double height = 0.0;
};

SDFEvaluation evaluate_sdf(const GroundSDF& sdf, const Vec2& x);

struct CircleSDF {
    Vec2 center{0.0, 0.0};
    double radius = 1.0;
};

SDFEvaluation evaluate_sdf(const CircleSDF& sdf, const Vec2& x);

double sdf_penalty_energy(const SDFEvaluation& sdf, double k, double eps);

Vec2 sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps);

Mat2 sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps,
                         bool include_curvature = true);
