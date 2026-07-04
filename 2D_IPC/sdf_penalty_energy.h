#pragma once

#include "ipc_math.h"

struct SDFEvaluation {
    double phi = 0.0;
    Vec2 grad_phi{0.0, 0.0};
    Mat2 hess_phi{0.0, 0.0, 0.0, 0.0};
};

struct RigidSDFGradient {
    Vec2 translation{0.0, 0.0};
    double rotation = 0.0;
};

struct RigidSDFHessian {
    Mat2 translation_translation{0.0, 0.0, 0.0, 0.0};
    Vec2 translation_rotation{0.0, 0.0};
    double rotation_rotation = 0.0;
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

struct PlaneSDF {
    Vec2   normal{0.0, 1.0}; // default normal pointing up like a ground plane.
    double offset = 0.0;
};

SDFEvaluation evaluate_sdf(const PlaneSDF& sdf, const Vec2& x);

double sdf_penalty_energy(const SDFEvaluation& sdf, double k, double eps);

Vec2 sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps);

Mat2 sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps, bool include_curvature = true);

RigidSDFGradient sdf_penalty_gradient_rb(const SDFEvaluation& sdf, const Vec2& x, const Vec2& x_com, double k, double eps);

RigidSDFHessian sdf_penalty_hessian_rb(const SDFEvaluation& sdf, const Vec2& x, const Vec2& x_com, double k, double eps, bool include_sdf_curvature = true, bool include_rigid_curvature = true);
