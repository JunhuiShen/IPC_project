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
    if (z < 0.0) return 0.0;
    if (z > eps) return 0.0;
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

double sdf_penalty_energy(const SDFEvaluation& sdf, double k, double eps) {
    const double H = sdf_heaviside(sdf.phi, eps);
    return 0.5 * k * H * H;
}

Vec2 sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps) {
    const double H = sdf_heaviside(sdf.phi, eps);
    const double Hp = sdf_heaviside_gradient(sdf.phi, eps);
    return scale(sdf.grad_phi, k * H * Hp);
}

Mat2 sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps,
                         bool include_curvature) {
    const double H = sdf_heaviside(sdf.phi, eps);
    const double Hp = sdf_heaviside_gradient(sdf.phi, eps);

    Mat2 Hess = scale(outer(sdf.grad_phi, sdf.grad_phi), k * Hp * Hp);
    if (include_curvature) {
        Hess = add(Hess, scale(sdf.hess_phi, k * H * Hp));
    }
    return Hess;
}
