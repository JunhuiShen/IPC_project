#include "sdf_penalty_energy.h"

#include <cmath>
#include <stdexcept>

//  H(z) = 1 for z<0, (eps-z)/eps on [0,eps], 0 for z>eps.
double sdf_heaviside(double z, double eps){
    if (eps <= 0.0) throw std::runtime_error("sdf_heaviside: eps must be positive.");
    if (z < 0.0)    return 1.0;
    if (z > eps)    return 0.0;
    return (eps - z) / eps;
}

//  H'(z) = -1/eps on (0,eps), 0 elsewhere.  At the breakpoints we return 0.
double sdf_heaviside_gradient(double z, double eps){
    if (eps <= 0.0) throw std::runtime_error("sdf_heaviside_gradient: eps must be positive.");
    if (z <= 0.0) return 0.0;
    if (z >= eps) return 0.0;
    return -1.0 / eps;
}

//  Plane SDF
SDFEvaluation evaluate_sdf(const PlaneSDF& s, const Vec3& x){
    SDFEvaluation r;
    r.phi      = (x - s.point).dot(s.normal);
    r.grad_phi = s.normal;
    r.hess_phi = Mat33::Zero();
    return r;
}

//  Infinite cylinder SDF
SDFEvaluation evaluate_sdf(const CylinderSDF& s, const Vec3& x){
    SDFEvaluation r;
    const Vec3 v = x - s.point;
    const Vec3 w = v - s.axis * (s.axis.dot(v));
    const double r_perp = w.norm();
    r.phi = r_perp - s.radius;
    if (r_perp > 0.0) {
        r.grad_phi = w / r_perp;
        r.hess_phi = (Mat33::Identity()
                      - s.axis * s.axis.transpose()
                      - r.grad_phi * r.grad_phi.transpose()) / r_perp;
    } else {
        r.grad_phi = Vec3::Zero();
        r.hess_phi = Mat33::Zero();
    }
    return r;
}

//  Sphere SDF
SDFEvaluation evaluate_sdf(const SphereSDF& s, const Vec3& x){
    SDFEvaluation r;
    const Vec3   d      = x - s.center;
    const double r_dist = d.norm();
    r.phi = r_dist - s.radius;
    if (r_dist > 0.0) {
        r.grad_phi = d / r_dist;
        r.hess_phi = (Mat33::Identity() - r.grad_phi * r.grad_phi.transpose()) / r_dist;
    } else {
        r.grad_phi = Vec3::Zero();
        r.hess_phi = Mat33::Zero();
    }
    return r;
}


double sdf_penalty_energy(const SDFEvaluation& sdf, double k, double eps){
    if (eps <= 0.0) {
        if (sdf.phi >= 0.0) return 0.0;
        return 0.5 * k * sdf.phi * sdf.phi;
    }

    const double H = sdf_heaviside(sdf.phi, eps);
    if (H <= 0.0) return 0.0;

    const double d = eps - sdf.phi;
    return 0.5 * k * H * d * d;
}

Vec3 sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps){
    if (eps <= 0.0) {
        if (sdf.phi >= 0.0) return Vec3::Zero();
        return k * sdf.phi * sdf.grad_phi;
    }

    const double H = sdf_heaviside(sdf.phi, eps);
    if (H <= 0.0) return Vec3::Zero();

    const double Hp = sdf_heaviside_gradient(sdf.phi, eps);
    const double d = eps - sdf.phi;
    const double dE_dphi = 0.5 * k * (Hp * d * d - 2.0 * H * d);
    return dE_dphi * sdf.grad_phi;
}

Mat33 sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps, bool include_curvature){
    double dE_dphi = 0.0;
    double d2E_dphi2 = 0.0;

    if (eps <= 0.0) {
        if (sdf.phi >= 0.0) return Mat33::Zero();
        dE_dphi = k * sdf.phi;
        d2E_dphi2 = k;
    } else {
        const double H = sdf_heaviside(sdf.phi, eps);
        if (H <= 0.0) return Mat33::Zero();

        const double Hp = sdf_heaviside_gradient(sdf.phi, eps);
        const double d = eps - sdf.phi;
        dE_dphi = 0.5 * k * (Hp * d * d - 2.0 * H * d);
        d2E_dphi2 = 0.5 * k * (-4.0 * Hp * d + 2.0 * H);
    }

    Mat33 Hess = d2E_dphi2 * (sdf.grad_phi * sdf.grad_phi.transpose());
    if (include_curvature) {
        Hess += dE_dphi * sdf.hess_phi;
    }
    return Hess;
}
