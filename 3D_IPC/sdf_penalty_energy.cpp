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
    if (sdf.phi >= eps) return 0.0;
    const double d = eps - sdf.phi;
    return 0.5 * k * d * d;
}

Vec3 sdf_penalty_gradient(const SDFEvaluation& sdf, double k, double eps){
    if (sdf.phi >= eps) return Vec3::Zero();
    return -k * (eps - sdf.phi) * sdf.grad_phi;
}

Mat33 sdf_penalty_hessian(const SDFEvaluation& sdf, double k, double eps, bool include_curvature){
    if (sdf.phi >= eps) return Mat33::Zero();
    Mat33 Hess = k * (sdf.grad_phi * sdf.grad_phi.transpose());
    if (include_curvature)
        Hess -= k * (eps - sdf.phi) * sdf.hess_phi;
    return Hess;
}
