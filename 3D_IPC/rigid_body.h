#pragma once

#include "IPC_math.h"
#include <Eigen/Dense>
#include <vector>

// ======================================================
// 3D rigid body: mass properties, momenta, velocities,
// and impulse-based collision response.
// Ported from TGSL's RigidBody (general 3D case).
//
// Orientation is stored as a quaternion (w,x,y,z).
// inertia_tensor / inertia_tensor_inv are the body-frame
// 3x3 inertia tensor (about the COM) and its inverse,
// computed once at construction; world-frame quantities
// are obtained via RotationMatrix() / WorldInertiaTensorInv().
// ======================================================

namespace rigid_body {

namespace algebra {

    // Levi-Civita symbol for indices in {0,1,2}
    inline int LeviCivita(int alpha, int beta, int gamma) {
        if (alpha == 0 && beta == 1 && gamma == 2) return 1;
        if (alpha == 1 && beta == 2 && gamma == 0) return 1;
        if (alpha == 2 && beta == 0 && gamma == 1) return 1;
        if (alpha == 2 && beta == 1 && gamma == 0) return -1;
        if (alpha == 1 && beta == 0 && gamma == 2) return -1;
        if (alpha == 0 && beta == 2 && gamma == 1) return -1;
        return 0;
    }

    // Quaternions are stored as (w, x, y, z)
    Eigen::Vector4d QuaternionMultiply(const Eigen::Vector4d& a, const Eigen::Vector4d& b);
    Eigen::Vector4d ConjugateQuaternion(const Eigen::Vector4d& q);
    Vec3 QuaternionRotate(const Eigen::Vector4d& q, const Vec3& v);
    Eigen::Vector4d VectorToQuaternion(const Vec3& v);
    // Exponential map: angular velocity * dt -> unit quaternion
    Eigen::Vector4d QuaternionFromVector(const Vec3& w);
    // Rotation matrix corresponding to a unit quaternion
    Mat33 QuaternionToRotationMatrix(const Eigen::Vector4d& q);

} // namespace algebra

struct RigidBody {
    double infinite_mass_threshold = 1e10;
    bool infinite_mass = false;
    double total_mass = 0.0;

    // Body-frame inertia tensor (about the COM) and its inverse
    Mat33 inertia_tensor     = Mat33::Zero();
    Mat33 inertia_tensor_inv = Mat33::Zero();

    Vec3 x_com = Vec3::Zero();
    Vec3 v_com = Vec3::Zero();

    Eigen::Vector4d orientation     = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector4d orientation_dot = Eigen::Vector4d::Zero();
    Vec3 omega = Vec3::Zero();

    RigidBody() {
        infinite_mass = true;
        total_mass = infinite_mass_threshold;
    }

    // X: reference positions of particles, nodal_mass: per-particle mass
    RigidBody(const std::vector<Vec3>& X, const std::vector<double>& nodal_mass);

    // Same as above, restricted to a subset of particles
    RigidBody(const std::vector<int>& particle_indices, const std::vector<Vec3>& X, const std::vector<double>& nodal_mass);

    // Also computes v_com and omega from particle velocities
    RigidBody(const std::vector<Vec3>& X, const std::vector<Vec3>& v, const std::vector<double>& nodal_mass);

    // World-frame velocity of the material point currently at world position x
    Vec3 Velocity(const Vec3& x) const;

    void UpdatePositionAndOrientation(double dt);

    void ToLinearNIF(const Vec3& x_frame, const Vec3& v_frame);
    void ToLinearIF(const Vec3& x_frame, const Vec3& v_frame);
    void DirectCOMUpdate(const Vec3& x, const Vec3& v);

    void ApplyImpulse(const Vec3& x, const Vec3& impulse);
    Mat33 MatrixK(const Vec3& x) const;
    double NKN(const Vec3& x, const Vec3& N) const;
    double NKNFriction(const Vec3& x, const Vec3& N, const Vec3& tangent_rel_V, double mu) const;
    void VelocityFromMomenta(const Vec3& p, const Vec3& l);

    Vec3 WorldSpacePosition(const Vec3& X) const;
    Vec3 MaterialSpacePosition(const Vec3& x) const;

    double KineticEnergy() const;

    // World-frame rotation matrix corresponding to `orientation`
    Mat33 RotationMatrix() const;
    // World-frame inverse inertia tensor: R * inertia_tensor_inv * R^T
    Mat33 WorldInertiaTensorInv() const;
    // World-frame inertia tensor: R * inertia_tensor * R^T
    Mat33 WorldInertiaTensor() const;
};

// the convention is that N points from b1 to b0
double NormalRelativeVelocity(const RigidBody& b0, const RigidBody& b1, const Vec3& x_contact, const Vec3& N);

double CollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Vec3& x_contact, const Vec3& N, double coeff_rest = 0.0);

bool InsideFrictionCone(const Vec3& impulse, const Vec3& N, double static_coefficient = 0.0);

Vec3 StaticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Vec3& x_contact, const Vec3& N, double coeff_rest = 0.0);

Vec3 KineticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Vec3& x_contact, const Vec3& N, double kinetic_friction = 0.0, double coeff_rest = 0.0);

void LinearAndAngularMomentum(const std::vector<int>& particle_indices, const std::vector<Vec3>& x, const std::vector<Vec3>& v, const std::vector<double>& mass, Vec3& p, Vec3& l);

} // namespace rigid_body
