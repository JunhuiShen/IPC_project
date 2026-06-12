#pragma once

#include "ipc_math.h"
#include <Eigen/Dense>
#include <vector>

// ======================================================
// 2D rigid body: mass properties, momenta, velocities,
// and impulse-based collision response.
// Ported from TGSL's RigidBody (TWO_D specialization).
//
// Orientation is stored as a quaternion (w,x,y,z) restricted
// to rotation about z: orientation = (cos(theta/2), 0, 0, sin(theta/2)).
// omega is a 3-vector with only omega.z() meaningful in 2D.
// Use RigidBody::Theta() to recover the raw angle used by
// step_filter::ccd's theta_n / theta_new.
// ======================================================

namespace rigid_body {

namespace algebra {

    // Levi-Civita symbol for indices in {0,1} (TGSL's 2D specialization)
    inline int LeviCivita(int alpha, int beta) {
        if (alpha == 0 && beta == 1) return 1;
        if (alpha == 1 && beta == 0) return -1;
        return 0;
    }

    // Quaternions are stored as (w, x, y, z)
    Eigen::Vector4d QuaternionMultiply(const Eigen::Vector4d& a, const Eigen::Vector4d& b);
    Eigen::Vector4d ConjugateQuaternion(const Eigen::Vector4d& q);
    Eigen::Vector3d QuaternionRotate(const Eigen::Vector4d& q, const Eigen::Vector3d& v);
    Eigen::Vector4d VectorToQuaternion(const Eigen::Vector3d& v);
    // Exponential map: angular velocity * dt -> unit quaternion
    Eigen::Vector4d QuaternionFromVector(const Eigen::Vector3d& w);

} // namespace algebra

struct RigidBody {
    double infinite_mass_threshold = 1e10;
    bool infinite_mass = false;
    double total_mass = 0.0;

    // 1 / I_zz (moment of inertia about the COM z-axis)
    double inertia_tensor_inv = 0.0;

    Eigen::Vector2d x_com = Eigen::Vector2d::Zero();
    Eigen::Vector2d v_com = Eigen::Vector2d::Zero();

    Eigen::Vector4d orientation     = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector4d orientation_dot = Eigen::Vector4d::Zero();
    Eigen::Vector3d omega           = Eigen::Vector3d::Zero();

    RigidBody() {
        infinite_mass = true;
        total_mass = infinite_mass_threshold;
    }

    // X: reference positions of particles, nodal_mass: per-particle mass
    RigidBody(const std::vector<Vec2>& X, const std::vector<double>& nodal_mass);

    // Same as above, restricted to a subset of particles
    RigidBody(const std::vector<int>& particle_indices, const std::vector<Vec2>& X, const std::vector<double>& nodal_mass);

    // Also computes v_com and omega from particle velocities
    RigidBody(const std::vector<Vec2>& X, const std::vector<Vec2>& v, const std::vector<double>& nodal_mass);

    // World-frame velocity of the material point currently at world position x
    Eigen::Vector2d Velocity(const Eigen::Vector2d& x) const;

    void UpdatePositionAndOrientation(double dt);

    void ToLinearNIF(const Eigen::Vector2d& x_frame, const Eigen::Vector2d& v_frame);
    void ToLinearIF(const Eigen::Vector2d& x_frame, const Eigen::Vector2d& v_frame);
    void DirectCOMUpdate(const Eigen::Vector2d& x, const Eigen::Vector2d& v);

    void ApplyImpulse(const Eigen::Vector2d& x, const Eigen::Vector2d& impulse);
    Eigen::Matrix2d MatrixK(const Eigen::Vector2d& x) const;
    double NKN(const Eigen::Vector2d& x, const Eigen::Vector2d& N) const;
    double NKNFriction(const Eigen::Vector2d& x, const Eigen::Vector2d& N, const Eigen::Vector2d& tangent_rel_V, double mu) const;
    void VelocityFromMomenta(const Eigen::Vector2d& p, const Eigen::Vector3d& l);

    Eigen::Vector2d WorldSpacePosition(const Eigen::Vector2d& X) const;
    Eigen::Vector2d MaterialSpacePosition(const Eigen::Vector2d& x) const;

    double KineticEnergy() const;

    // Rotation angle theta such that orientation = (cos(theta/2), 0, 0, sin(theta/2)),
    // matching theta_n / theta_new in step_filter::ccd.
    double Theta() const;
};

// the convention is that N points from b1 to b0
double NormalRelativeVelocity(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N);

double CollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, double coeff_rest = 0.0);

bool InsideFrictionCone(const Eigen::Vector2d& impulse, const Eigen::Vector2d& N, double static_coefficient = 0.0);

Eigen::Vector2d StaticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, double coeff_rest = 0.0);

Eigen::Vector2d KineticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, double kinetic_friction = 0.0, double coeff_rest = 0.0);

void LinearAndAngularMomentum(const std::vector<int>& particle_indices, const std::vector<Vec2>& x, const std::vector<Vec2>& v, const std::vector<double>& mass, Eigen::Vector2d& p, Eigen::Vector3d& l);

} // namespace rigid_body
