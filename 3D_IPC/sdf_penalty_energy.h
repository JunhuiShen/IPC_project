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

//  Solid sphere
struct SphereSDF {
    Vec3   center;
    double radius;
};

SDFEvaluation evaluate_sdf(const SphereSDF& s, const Vec3& x);

double sdf_penalty_energy(const SDFEvaluation& sdf, double k, double eps);

Vec3   sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps);

Mat33  sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps,
                           bool include_curvature = true);



struct RigidSDFGradient {
    Vec3 translation;  //  dE/dx_com
    Vec3 rotation;     //  dE/domega

    RigidSDFGradient() : translation(Vec3::Zero()), rotation(Vec3::Zero()) {}
};

struct RigidSDFHessian {
    Mat33 translation_translation;  //  d2E/dx_com2
    Mat33 translation_rotation;     //  d2E/dx_com domega (rows x_com, cols omega)
    Mat33 rotation_rotation;        //  d2E/domega2

    RigidSDFHessian()
        : translation_translation(Mat33::Zero()),
          translation_rotation(Mat33::Zero()),
          rotation_rotation(Mat33::Zero()) {}
};

RigidSDFGradient sdf_penalty_gradient_rb(
    const SDFEvaluation& sdf, const Vec3& X_centered,
    const Vec4& q_n, const Vec3& omega, double dt,
    double k, double eps);

RigidSDFHessian sdf_penalty_hessian_rb(
    const SDFEvaluation& sdf, const Vec3& X_centered,
    const Vec4& q_n, const Vec3& omega, double dt,
    double k, double eps,
    bool include_sdf_curvature = true,
    bool include_rigid_curvature = true);
