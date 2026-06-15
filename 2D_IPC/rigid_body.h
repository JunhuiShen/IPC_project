#pragma once

#include "algebra/algebra.h"
#include <Eigen/Dense>
#include <limits>
#include <vector>

// ======================================================
// 2D rigid body: mass properties, momenta, velocities,
// and impulse-based collision response.
//
// Ported from TGSL's RigidBody (TWO_D specialization).
//
// Orientation is stored as a quaternion (w,x,y,z) restricted to
// rotation about z: orientation = (cos(theta/2), 0, 0, sin(theta/2)).
// omega is a 3-vector with only omega.z() meaningful in 2D.
// inertia_tensor_inv is the scalar 1/I_zz (TGSL's TV is a single-element
// vector in the TWO_D specialization).
// ======================================================

namespace Rigid_Body {

struct RigidBody {
    double infinite_mass_threshold = double(1e10);
    bool infinite_mass = false;
    double total_mass;
    std::vector<double> inertia_tensor_inv;

    Eigen::Vector2d x_com;
    Eigen::Vector2d v_com;
    Eigen::Vector4d orientation;
    Eigen::Vector3d omega;

    RigidBody() {
        infinite_mass = true;
        total_mass = std::numeric_limits<double>::max();
        inertia_tensor_inv = {double(0)};
        orientation = Eigen::Vector4d(double(1), double(0), double(0), double(0));
        omega = Eigen::Vector3d::Zero();
        x_com = Eigen::Vector2d::Zero();
        v_com = Eigen::Vector2d::Zero();
    }

    // X: reference positions of particles, nodal_mass: per-particle mass
    RigidBody(const std::vector<Eigen::Vector2d>& X, const std::vector<double>& nodal_mass);

    // Same as above, restricted to a subset of particles
    RigidBody(const std::vector<int>& particle_indices, const std::vector<Eigen::Vector2d>& X, const std::vector<double>& nodal_mass);

    // Also computes v_com and omega from particle velocities
    RigidBody(const std::vector<Eigen::Vector2d>& X, const std::vector<Eigen::Vector2d>& v, const std::vector<double>& nodal_mass);

    // World-frame velocity of the material point currently at world position x
    Eigen::Vector2d Velocity(const Eigen::Vector2d& x) const;

    void UpdatePositionAndOrientation(double dt);

    void ApplyImpulse(const Eigen::Vector2d& x, const Eigen::Vector2d& impulse);

    double NKN(const Eigen::Vector2d& x, const Eigen::Vector2d& N) const;

    double NKNFriction(const Eigen::Vector2d& x, const Eigen::Vector2d& N, const Eigen::Vector2d& tangent_rel_V, const double mu) const;

    void VelocityFromMomenta(const Eigen::Vector2d& p, const Eigen::Vector3d& l);

    Eigen::Vector2d WorldSpacePosition(const Eigen::Vector2d& X) const;

    Eigen::Vector2d MaterialSpacePosition(const Eigen::Vector2d& x) const;

    Eigen::Matrix2d MatrixK(const Eigen::Vector2d& x) const;
};


inline double NormalRelativeVelocity(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N){
    // the convention is that N points from b1 to b0
    Eigen::Vector2d v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    Eigen::Vector2d relative_velocity = v1 - v0;
    return relative_velocity.dot(N);
}

inline double CollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, const double coeff_rest = 0.0){
    // the convention is that N points from b1 to b0
    Eigen::Vector2d v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    Eigen::Vector2d relative_velocity = v1 - v0;
    double vN = relative_velocity.dot(N);
    if (vN <= double(0))
        return double(0);
    else
        return -(double(1) + coeff_rest) * vN / (b0.NKN(x_contact, N) + b1.NKN(x_contact, N));
}

inline bool InsideFrictionCone(const Eigen::Vector2d& impulse, const Eigen::Vector2d& N, const double static_coefficient = double(0)){
    Eigen::Vector2d temp = impulse - impulse.dot(N) * N;
    return temp.norm() <= static_coefficient * impulse.dot(N); 
}

inline Eigen::Vector2d StaticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, const double coeff_rest = double(0)){
    // the convention is that N points from b1 to b0
    Eigen::Vector2d v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    Eigen::Vector2d relative_velocity = v1 - v0;
    double vN = relative_velocity.dot(N);

    Eigen::Vector2d j = Eigen::Vector2d::Zero(), rhs = Eigen::Vector2d::Zero();
    rhs = -coeff_rest * vN * N - relative_velocity;

    Eigen::Matrix2d K0 = b0.MatrixK(x_contact);
    Eigen::Matrix2d K1 = b1.MatrixK(x_contact);
    Eigen::Matrix2d K_inv = K0 + K1;
    K_inv = K_inv.inverse();
    j = K_inv * rhs;
    return j;
}

inline Eigen::Vector2d KineticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, const double kinetic_friction = double(0), const double coeff_rest = double(0)){
    // the convention is that N points from b1 to b0
    Eigen::Vector2d v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    Eigen::Vector2d relative_velocity = v1 - v0;
    double vN = relative_velocity.dot(N);

    // Compute tangential relative velocity
    Eigen::Vector2d tangent = relative_velocity - vN * N;
    tangent.normalize();

    Eigen::Vector2d j;
    if(vN <= double(0)){
        j = Eigen::Vector2d::Zero();
    }
    else {
        double j_mag = double(0);
        double val = b0.NKNFriction(x_contact, N, tangent, kinetic_friction) + b1.NKNFriction(x_contact, N, tangent, kinetic_friction);
        j_mag = -((double(1) + coeff_rest) * vN) / val;
        j = j_mag * N + kinetic_friction * j_mag * tangent;
    }
    return j;
}

inline void LinearAndAngularMomentum(const std::vector<int>& particle_indices, const std::vector<Eigen::Vector2d>& x, const std::vector<Eigen::Vector2d>& v, const std::vector<double>& mass, Eigen::Vector2d& p, Eigen::Vector3d& l){
    // compute center of mass and p
    double total_mass = double(0);
    Eigen::Vector2d x_com = Eigen::Vector2d::Zero();
    p = Eigen::Vector2d::Zero();
    for(size_t j = 0; j < particle_indices.size(); j++){
        size_t i = size_t(particle_indices[j]);
        total_mass += mass[i];
        x_com += mass[i] * x[i];
        p += mass[i] * v[i];
    }
    x_com /= total_mass;

    // compute total angular momentum l
    l = Eigen::Vector3d::Zero();
    size_t angular_alpha_start = 2;
    for(size_t j = 0; j < particle_indices.size(); j++){
        size_t i = size_t(particle_indices[j]);
        Eigen::Vector2d r = x[i] - x_com;
        for(size_t alpha = angular_alpha_start; alpha < 3; alpha++){
            for(size_t beta = 0; beta < 2; beta++){
                for(size_t gamma = 0; gamma < 2; gamma++){
                    l[alpha] += ALGEBRA::LeviCivita(int(alpha),int(beta),int(gamma)) * r[beta] * mass[i] * v[i][gamma];
                }
            }
        }
    }
}

}  // namespace Rigid_Body
