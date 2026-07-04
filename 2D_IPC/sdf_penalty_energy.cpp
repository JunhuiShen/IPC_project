#include "sdf_penalty_energy.h"

#include <cmath>
#include <stdexcept>

double sdf_heaviside(double z, double eps) {
    if (eps <= 0.0) throw std::runtime_error("sdf_heaviside: eps must be positive.");
    if (z < 0.0) return 1.0;
    if (z > eps) return 0.0;
    return (eps - z) / eps;
}

double sdf_heaviside_gradient(double z, double eps) {
    if (eps <= 0.0) throw std::runtime_error("sdf_heaviside_gradient: eps must be positive.");
    if (z <= 0.0) return 0.0;
    if (z >= eps) return 0.0;
    return -1.0 / eps;
}

SDFEvaluation evaluate_sdf(const GroundSDF& sdf, const Vec2& x) {
    SDFEvaluation r;
    r.phi = x.y - sdf.height;
    r.grad_phi = {0.0, 1.0};
    r.hess_phi = {0.0, 0.0, 0.0, 0.0};
    return r;
}

SDFEvaluation evaluate_sdf(const CircleSDF& sdf, const Vec2& x) {
    SDFEvaluation r;
    const Vec2 d = sub(x, sdf.center);
    const double r_dist = norm(d);
    r.phi = r_dist - sdf.radius;

    if (r_dist > 0.0) {
        r.grad_phi = scale(d, 1.0 / r_dist);
        const Mat2 I{1.0, 0.0, 0.0, 1.0};
        r.hess_phi = scale(add(I, scale(outer(r.grad_phi, r.grad_phi), -1.0)), 1.0 / r_dist);
    } else {
        r.grad_phi = {0.0, 0.0};
        r.hess_phi = {0.0, 0.0, 0.0, 0.0};
    }

    return r;
}

SDFEvaluation evaluate_sdf(const PlaneSDF& sdf, const Vec2& x) {
    SDFEvaluation r;
    r.phi      = dot(sdf.normal, x) - sdf.offset;
    r.grad_phi = sdf.normal;
    r.hess_phi = {0.0, 0.0, 0.0, 0.0};
    return r;
}

double sdf_penalty_energy(const SDFEvaluation& sdf, double k, double eps) {
    if (sdf.phi >= eps) return 0.0;
    const double d = eps - sdf.phi;
    return 0.5 * k * d * d;
}

Vec2 sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps) {
    if (sdf.phi >= eps) return {0.0, 0.0};
    return scale(sdf.grad_phi, -k * (eps - sdf.phi));
}

Mat2 sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps, bool include_curvature) {
    if (sdf.phi >= eps) return {0.0, 0.0, 0.0, 0.0};

    const double d = eps - sdf.phi;
    Mat2 Hess = scale(outer(sdf.grad_phi, sdf.grad_phi), k);
    if (include_curvature) {
        Hess = add(Hess, scale(sdf.hess_phi, -k * d));
    }
    return Hess;
}

RigidSDFGradient sdf_penalty_gradient_rb(const SDFEvaluation& sdf, const Vec2& x, const Vec2& x_com, double k, double eps) {
    const Vec2 gx = sdf_penalty_gradient(sdf, k, eps);
    const Vec2 r = x - x_com;
    const Vec2 dx_dtheta{-r.y, r.x};

    RigidSDFGradient result;
    result.translation = gx;
    result.rotation = dot(dx_dtheta, gx);
    return result;
}

RigidSDFHessian sdf_penalty_hessian_rb(const SDFEvaluation& sdf, const Vec2& x, const Vec2& x_com, double k, double eps, bool include_sdf_curvature, bool include_rigid_curvature) {
    const Vec2 gx = sdf_penalty_gradient(sdf, k, eps);
    const Mat2 Hx = sdf_penalty_hessian(sdf, k, eps, include_sdf_curvature);
    const Vec2 r = x - x_com;
    const Vec2 dx_dtheta{-r.y, r.x};
    const Vec2 d2x_dtheta2{-r.x, -r.y};
    const Vec2 Hx_dx_dtheta = matvec(Hx, dx_dtheta);

    RigidSDFHessian result;
    result.translation_translation = Hx;
    result.translation_rotation = Hx_dx_dtheta;
    result.rotation_rotation = dot(dx_dtheta, Hx_dx_dtheta);
    if (include_rigid_curvature) {
        result.rotation_rotation += dot(gx, d2x_dtheta2);
    }
    return result;
}
