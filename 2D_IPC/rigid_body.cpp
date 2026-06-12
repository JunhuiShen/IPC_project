#include "rigid_body.h"
#include <cassert>
#include <cmath>

namespace rigid_body {

namespace algebra {

Eigen::Vector4d QuaternionMultiply(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
    double aw = a[0], ax = a[1], ay = a[2], az = a[3];
    double bw = b[0], bx = b[1], by = b[2], bz = b[3];
    return Eigen::Vector4d(
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    );
}

Eigen::Vector4d ConjugateQuaternion(const Eigen::Vector4d& q) {
    return Eigen::Vector4d(q[0], -q[1], -q[2], -q[3]);
}

Eigen::Vector3d QuaternionRotate(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    Eigen::Vector4d v_quat(0.0, v.x(), v.y(), v.z());
    Eigen::Vector4d result = QuaternionMultiply(QuaternionMultiply(q, v_quat), ConjugateQuaternion(q));
    return result.tail<3>();
}

Eigen::Vector4d VectorToQuaternion(const Eigen::Vector3d& v) {
    return Eigen::Vector4d(0.0, v.x(), v.y(), v.z());
}

Eigen::Vector4d QuaternionFromVector(const Eigen::Vector3d& w) {
    double angle = w.norm();
    if (angle < 1e-14) {
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
    Eigen::Vector3d axis = w / angle;
    double half = 0.5 * angle;
    double s = std::sin(half);
    return Eigen::Vector4d(std::cos(half), axis.x() * s, axis.y() * s, axis.z() * s);
}

} // namespace algebra

namespace {

inline double cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

inline Eigen::Vector2d perp(const Eigen::Vector2d& r) {
    return Eigen::Vector2d(-r.y(), r.x());
}

} // namespace

RigidBody::RigidBody(const std::vector<Vec2>& X, const std::vector<double>& nodal_mass) {
    total_mass = 0.0;
    x_com = Eigen::Vector2d::Zero();
    for (std::size_t i = 0; i < X.size(); ++i) {
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * Eigen::Vector2d(X[i].x, X[i].y);
    }
    x_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        x_com = Eigen::Vector2d::Zero();
        inertia_tensor_inv = 0.0;
    } else {
        double I_zz = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) {
            Eigen::Vector2d r = Eigen::Vector2d(X[i].x, X[i].y) - x_com;
            I_zz += nodal_mass[i] * r.squaredNorm();
        }
        inertia_tensor_inv = 1.0 / I_zz;
    }
    orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
}

RigidBody::RigidBody(const std::vector<int>& particle_indices, const std::vector<Vec2>& X, const std::vector<double>& nodal_mass) {
    total_mass = 0.0;
    x_com = Eigen::Vector2d::Zero();
    for (int idx : particle_indices) {
        total_mass += nodal_mass[idx];
        x_com += nodal_mass[idx] * Eigen::Vector2d(X[idx].x, X[idx].y);
    }
    x_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        x_com = Eigen::Vector2d::Zero();
        inertia_tensor_inv = 0.0;
    } else {
        double I_zz = 0.0;
        for (int idx : particle_indices) {
            Eigen::Vector2d r = Eigen::Vector2d(X[idx].x, X[idx].y) - x_com;
            I_zz += nodal_mass[idx] * r.squaredNorm();
        }
        inertia_tensor_inv = 1.0 / I_zz;
    }
    orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
}

RigidBody::RigidBody(const std::vector<Vec2>& X, const std::vector<Vec2>& v, const std::vector<double>& nodal_mass) {
    total_mass = 0.0;
    x_com = Eigen::Vector2d::Zero();
    v_com = Eigen::Vector2d::Zero();
    for (std::size_t i = 0; i < X.size(); ++i) {
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * Eigen::Vector2d(X[i].x, X[i].y);
        v_com += nodal_mass[i] * Eigen::Vector2d(v[i].x, v[i].y);
    }
    x_com /= total_mass;
    v_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        x_com = Eigen::Vector2d::Zero();
        v_com = Eigen::Vector2d::Zero();
        inertia_tensor_inv = 0.0;
    } else {
        double I_zz = 0.0;
        double l_z = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) {
            Eigen::Vector2d r = Eigen::Vector2d(X[i].x, X[i].y) - x_com;
            Eigen::Vector2d vi(v[i].x, v[i].y);
            I_zz += nodal_mass[i] * r.squaredNorm();
            l_z += nodal_mass[i] * cross2(r, vi);
        }
        inertia_tensor_inv = 1.0 / I_zz;
        omega.z() = l_z * inertia_tensor_inv;
    }
    orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
}

Eigen::Vector2d RigidBody::Velocity(const Eigen::Vector2d& x) const {
    Eigen::Vector2d r = x - x_com;
    return v_com + omega.z() * perp(r);
}

void RigidBody::UpdatePositionAndOrientation(double dt) {
    x_com += dt * v_com;
    Eigen::Vector4d q_hat = algebra::QuaternionFromVector(dt * omega);
    orientation = algebra::QuaternionMultiply(q_hat, orientation);
}

void RigidBody::ToLinearNIF(const Eigen::Vector2d& x_frame, const Eigen::Vector2d& v_frame) {
    x_com -= x_frame;
    v_com -= v_frame;
}

void RigidBody::ToLinearIF(const Eigen::Vector2d& x_frame, const Eigen::Vector2d& v_frame) {
    x_com += x_frame;
    v_com += v_frame;
}

void RigidBody::DirectCOMUpdate(const Eigen::Vector2d& x, const Eigen::Vector2d& v) {
    x_com = x;
    v_com = v;
}

void RigidBody::ApplyImpulse(const Eigen::Vector2d& x, const Eigen::Vector2d& impulse) {
    if (infinite_mass) return;

    v_com += impulse / total_mass;

    Eigen::Vector2d r = x - x_com;
    omega.z() += cross2(r, impulse) * inertia_tensor_inv;
}

Eigen::Matrix2d RigidBody::MatrixK(const Eigen::Vector2d& x) const {
    Eigen::Matrix2d K;
    if (infinite_mass) {
        K.setZero();
        return K;
    }
    Eigen::Vector2d r = x - x_com;
    K(0, 0) = 1.0 / total_mass + r.y() * r.y() * inertia_tensor_inv;
    K(0, 1) = -r.x() * r.y() * inertia_tensor_inv;
    K(1, 0) = K(0, 1);
    K(1, 1) = 1.0 / total_mass + r.x() * r.x() * inertia_tensor_inv;
    return K;
}

double RigidBody::NKN(const Eigen::Vector2d& x, const Eigen::Vector2d& N) const {
    if (infinite_mass) return 0.0;

    Eigen::Vector2d r = x - x_com;
    double rXN = cross2(r, N);
    return N.squaredNorm() / total_mass + inertia_tensor_inv * rXN * rXN;
}

double RigidBody::NKNFriction(const Eigen::Vector2d& x, const Eigen::Vector2d& N, const Eigen::Vector2d& tangent_rel_V, double mu) const {
    if (infinite_mass) return 0.0;

    Eigen::Vector2d r = x - x_com;
    Eigen::Vector2d N_muT = N - mu * tangent_rel_V;

    double result = N.dot(N_muT) / total_mass;

    double rXN_muT = cross2(r, N_muT);
    double rXN = cross2(r, N);
    result += inertia_tensor_inv * rXN_muT * rXN;

    return result;
}

void RigidBody::VelocityFromMomenta(const Eigen::Vector2d& p, const Eigen::Vector3d& l) {
    v_com = p / total_mass;
    omega.setZero();
    omega.z() = l.z() * inertia_tensor_inv;
}

Eigen::Vector2d RigidBody::WorldSpacePosition(const Eigen::Vector2d& X) const {
    Eigen::Vector3d R(X.x(), X.y(), 0.0);
    Eigen::Vector3d r = algebra::QuaternionRotate(orientation, R);
    return x_com + r.head<2>();
}

Eigen::Vector2d RigidBody::MaterialSpacePosition(const Eigen::Vector2d& x) const {
    Eigen::Vector3d r(x.x() - x_com.x(), x.y() - x_com.y(), 0.0);
    Eigen::Vector3d X = algebra::QuaternionRotate(algebra::ConjugateQuaternion(orientation), r);
    return X.head<2>();
}

double RigidBody::KineticEnergy() const {
    assert(!infinite_mass);

    double linear_part = 0.5 * total_mass * v_com.squaredNorm();
    // I_zz is invariant under in-plane rotation, so TGSL's body-frame
    // round trip (rotate omega into body frame, scale by I, rotate back)
    // collapses to a single scalar term in 2D.
    double angular_part = 0.5 * omega.z() * omega.z() / inertia_tensor_inv;

    return linear_part + angular_part;
}

double RigidBody::Theta() const {
    double theta = 2.0 * std::atan2(orientation[3], orientation[0]);
    if (theta > M_PI) {
        theta -= 2.0 * M_PI;
    } else if (theta < -M_PI) {
        theta += 2.0 * M_PI;
    }
    return theta;
}

double NormalRelativeVelocity(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N) {
    Eigen::Vector2d v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    return (v1 - v0).dot(N);
}

double CollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, double coeff_rest) {
    double vN = NormalRelativeVelocity(b0, b1, x_contact, N);
    if (vN <= 0.0) return 0.0;
    return -(1.0 + coeff_rest) * vN / (b0.NKN(x_contact, N) + b1.NKN(x_contact, N));
}

bool InsideFrictionCone(const Eigen::Vector2d& impulse, const Eigen::Vector2d& N, double static_coefficient) {
    Eigen::Vector2d tangent = impulse - N * impulse.dot(N);
    return tangent.norm() <= static_coefficient * impulse.dot(N);
}

Eigen::Vector2d StaticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, double coeff_rest) {
    Eigen::Vector2d v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    Eigen::Vector2d relative_velocity = v1 - v0;
    double vN = relative_velocity.dot(N);

    Eigen::Vector2d rhs = -coeff_rest * vN * N - relative_velocity;

    Eigen::Matrix2d K = b0.MatrixK(x_contact) + b1.MatrixK(x_contact);
    return K.inverse() * rhs;
}

Eigen::Vector2d KineticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Eigen::Vector2d& x_contact, const Eigen::Vector2d& N, double kinetic_friction, double coeff_rest) {
    Eigen::Vector2d v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    Eigen::Vector2d relative_velocity = v1 - v0;
    double vN = relative_velocity.dot(N);

    if (vN <= 0.0) return Eigen::Vector2d::Zero();

    Eigen::Vector2d tangent = relative_velocity - vN * N;
    double tangent_norm = tangent.norm();
    if (tangent_norm > 1e-14) {
        tangent /= tangent_norm;
    } else {
        tangent.setZero();
    }

    double val = b0.NKNFriction(x_contact, N, tangent, kinetic_friction) + b1.NKNFriction(x_contact, N, tangent, kinetic_friction);
    double j_mag = -((1.0 + coeff_rest) * vN) / val;

    return j_mag * N + kinetic_friction * j_mag * tangent;
}

void LinearAndAngularMomentum(const std::vector<int>& particle_indices, const std::vector<Vec2>& x, const std::vector<Vec2>& v, const std::vector<double>& mass, Eigen::Vector2d& p, Eigen::Vector3d& l) {
    double total_mass = 0.0;
    Eigen::Vector2d x_com = Eigen::Vector2d::Zero();
    p = Eigen::Vector2d::Zero();
    for (int idx : particle_indices) {
        total_mass += mass[idx];
        x_com += mass[idx] * Eigen::Vector2d(x[idx].x, x[idx].y);
        p += mass[idx] * Eigen::Vector2d(v[idx].x, v[idx].y);
    }
    x_com /= total_mass;

    l = Eigen::Vector3d::Zero();
    for (int idx : particle_indices) {
        Eigen::Vector2d r(x[idx].x - x_com.x(), x[idx].y - x_com.y());
        Eigen::Vector2d vi(v[idx].x, v[idx].y);
        l.z() += mass[idx] * cross2(r, vi);
    }
}

} // namespace rigid_body
