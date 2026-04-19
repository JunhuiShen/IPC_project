#pragma once

#include "IPC_math.h"

//  SDF-based penalty energy

//  Heaviside and its derivative with respect to z.
double sdf_heaviside(double z, double eps);

double sdf_heaviside_gradient(double z, double eps);

struct SDFEvaluation {
    double phi;
    Vec3   grad_phi;
    Mat33  hess_phi;

    SDFEvaluation() : phi(0.0), grad_phi(Vec3::Zero()), hess_phi(Mat33::Zero()) {}
};

//  Infinite half-space
struct PlaneSDF {
    Vec3 point;
    Vec3 normal;   //  must be unit length
};

SDFEvaluation evaluate_sdf(const PlaneSDF& s, const Vec3& x);

//  Infinite solid cylinder 
struct CylinderSDF {
    Vec3   point;
    Vec3   axis;     //  must be unit length
    double radius;
};

SDFEvaluation evaluate_sdf(const CylinderSDF& s, const Vec3& x);

double sdf_penalty_energy(const SDFEvaluation& sdf, double k, double eps);

Vec3   sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps);

Mat33  sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps,
                           bool include_curvature = true);
