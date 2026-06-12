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

Vec3 QuaternionRotate(const Eigen::Vector4d& q, const Vec3& v) {
    Eigen::Vector4d v_quat(0.0, v.x(), v.y(), v.z());
    Eigen::Vector4d result = QuaternionMultiply(QuaternionMultiply(q, v_quat), ConjugateQuaternion(q));
    return result.tail<3>();
}

Eigen::Vector4d VectorToQuaternion(const Vec3& v) {
    return Eigen::Vector4d(0.0, v.x(), v.y(), v.z());
}

Eigen::Vector4d QuaternionFromVector(const Vec3& w) {
    double angle = w.norm();
    if (angle < 1e-14) {
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
    Vec3 axis = w / angle;
    double half = 0.5 * angle;
    double s = std::sin(half);
    return Eigen::Vector4d(std::cos(half), axis.x() * s, axis.y() * s, axis.z() * s);
}

Mat33 QuaternionToRotationMatrix(const Eigen::Vector4d& q) {
    Mat33 R;
    R.col(0) = QuaternionRotate(q, Vec3(1.0, 0.0, 0.0));
    R.col(1) = QuaternionRotate(q, Vec3(0.0, 1.0, 0.0));
    R.col(2) = QuaternionRotate(q, Vec3(0.0, 0.0, 1.0));
    return R;
}

} // namespace algebra

namespace {

// Skew-symmetric cross-product matrix: skew(r) * v == r.cross(v)
inline Mat33 skew(const Vec3& r) {
    Mat33 S;
    S <<   0.0, -r.z(),  r.y(),
         r.z(),    0.0, -r.x(),
        -r.y(),  r.x(),    0.0;
    return S;
}

// Body-frame inertia tensor about the COM via the parallel-axis theorem:
// I = sum_i mass_i * (|r_i|^2 * Identity - r_i r_i^T)
Mat33 inertia_tensor_about_com(const std::vector<Vec3>& r, const std::vector<double>& mass) {
    Mat33 I = Mat33::Zero();
    for (std::size_t i = 0; i < r.size(); ++i) {
        I += mass[i] * (r[i].squaredNorm() * Mat33::Identity() - r[i] * r[i].transpose());
    }
    return I;
}

} // namespace

RigidBody::RigidBody(const std::vector<Vec3>& X, const std::vector<double>& nodal_mass) {
    total_mass = 0.0;
    x_com = Vec3::Zero();
    for (std::size_t i = 0; i < X.size(); ++i) {
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * X[i];
    }
    x_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        x_com = Vec3::Zero();
        inertia_tensor = Mat33::Zero();
        inertia_tensor_inv = Mat33::Zero();
    } else {
        std::vector<Vec3> r(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) r[i] = X[i] - x_com;
        inertia_tensor = inertia_tensor_about_com(r, nodal_mass);
        inertia_tensor_inv = inertia_tensor.inverse();
    }
    orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
}

RigidBody::RigidBody(const std::vector<int>& particle_indices, const std::vector<Vec3>& X, const std::vector<double>& nodal_mass) {
    total_mass = 0.0;
    x_com = Vec3::Zero();
    for (int idx : particle_indices) {
        total_mass += nodal_mass[idx];
        x_com += nodal_mass[idx] * X[idx];
    }
    x_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        x_com = Vec3::Zero();
        inertia_tensor = Mat33::Zero();
        inertia_tensor_inv = Mat33::Zero();
    } else {
        std::vector<Vec3> r;
        std::vector<double> mass;
        r.reserve(particle_indices.size());
        mass.reserve(particle_indices.size());
        for (int idx : particle_indices) {
            r.push_back(X[idx] - x_com);
            mass.push_back(nodal_mass[idx]);
        }
        inertia_tensor = inertia_tensor_about_com(r, mass);
        inertia_tensor_inv = inertia_tensor.inverse();
    }
    orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
}

RigidBody::RigidBody(const std::vector<Vec3>& X, const std::vector<Vec3>& v, const std::vector<double>& nodal_mass) {
    total_mass = 0.0;
    x_com = Vec3::Zero();
    v_com = Vec3::Zero();
    for (std::size_t i = 0; i < X.size(); ++i) {
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * X[i];
        v_com += nodal_mass[i] * v[i];
    }
    x_com /= total_mass;
    v_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        x_com = Vec3::Zero();
        v_com = Vec3::Zero();
        inertia_tensor = Mat33::Zero();
        inertia_tensor_inv = Mat33::Zero();
    } else {
        std::vector<Vec3> r(X.size());
        Vec3 l = Vec3::Zero();
        for (std::size_t i = 0; i < X.size(); ++i) {
            r[i] = X[i] - x_com;
            l += nodal_mass[i] * r[i].cross(v[i]);
        }
        inertia_tensor = inertia_tensor_about_com(r, nodal_mass);
        inertia_tensor_inv = inertia_tensor.inverse();
        omega = inertia_tensor_inv * l;
    }
    orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
}

Vec3 RigidBody::Velocity(const Vec3& x) const {
    Vec3 r = x - x_com;
    return v_com + omega.cross(r);
}

void RigidBody::UpdatePositionAndOrientation(double dt) {
    x_com += dt * v_com;
    Eigen::Vector4d q_hat = algebra::QuaternionFromVector(dt * omega);
    orientation = algebra::QuaternionMultiply(q_hat, orientation);
}

void RigidBody::ToLinearNIF(const Vec3& x_frame, const Vec3& v_frame) {
    x_com -= x_frame;
    v_com -= v_frame;
}

void RigidBody::ToLinearIF(const Vec3& x_frame, const Vec3& v_frame) {
    x_com += x_frame;
    v_com += v_frame;
}

void RigidBody::DirectCOMUpdate(const Vec3& x, const Vec3& v) {
    x_com = x;
    v_com = v;
}

void RigidBody::ApplyImpulse(const Vec3& x, const Vec3& impulse) {
    if (infinite_mass) return;

    v_com += impulse / total_mass;

    Vec3 r = x - x_com;
    omega += WorldInertiaTensorInv() * r.cross(impulse);
}

Mat33 RigidBody::MatrixK(const Vec3& x) const {
    if (infinite_mass) return Mat33::Zero();

    Vec3 r = x - x_com;
    Mat33 r_cross = skew(r);
    return Mat33::Identity() / total_mass - r_cross * WorldInertiaTensorInv() * r_cross;
}

double RigidBody::NKN(const Vec3& x, const Vec3& N) const {
    if (infinite_mass) return 0.0;

    Vec3 r = x - x_com;
    Vec3 rXN = r.cross(N);
    return N.squaredNorm() / total_mass + rXN.dot(WorldInertiaTensorInv() * rXN);
}

double RigidBody::NKNFriction(const Vec3& x, const Vec3& N, const Vec3& tangent_rel_V, double mu) const {
    if (infinite_mass) return 0.0;

    Vec3 r = x - x_com;
    Vec3 N_muT = N - mu * tangent_rel_V;

    double result = N.dot(N_muT) / total_mass;

    Vec3 rXN_muT = r.cross(N_muT);
    Vec3 rXN = r.cross(N);
    result += rXN_muT.dot(WorldInertiaTensorInv() * rXN);

    return result;
}

void RigidBody::VelocityFromMomenta(const Vec3& p, const Vec3& l) {
    v_com = p / total_mass;
    omega = WorldInertiaTensorInv() * l;
}

Vec3 RigidBody::WorldSpacePosition(const Vec3& X) const {
    return x_com + algebra::QuaternionRotate(orientation, X);
}

Vec3 RigidBody::MaterialSpacePosition(const Vec3& x) const {
    return algebra::QuaternionRotate(algebra::ConjugateQuaternion(orientation), x - x_com);
}

double RigidBody::KineticEnergy() const {
    assert(!infinite_mass);

    double linear_part = 0.5 * total_mass * v_com.squaredNorm();
    double angular_part = 0.5 * omega.dot(WorldInertiaTensor() * omega);

    return linear_part + angular_part;
}

Mat33 RigidBody::RotationMatrix() const {
    return algebra::QuaternionToRotationMatrix(orientation);
}

Mat33 RigidBody::WorldInertiaTensorInv() const {
    Mat33 R = RotationMatrix();
    return R * inertia_tensor_inv * R.transpose();
}

Mat33 RigidBody::WorldInertiaTensor() const {
    Mat33 R = RotationMatrix();
    return R * inertia_tensor * R.transpose();
}

double NormalRelativeVelocity(const RigidBody& b0, const RigidBody& b1, const Vec3& x_contact, const Vec3& N) {
    Vec3 v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    return (v1 - v0).dot(N);
}

double CollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Vec3& x_contact, const Vec3& N, double coeff_rest) {
    double vN = NormalRelativeVelocity(b0, b1, x_contact, N);
    if (vN <= 0.0) return 0.0;
    return -(1.0 + coeff_rest) * vN / (b0.NKN(x_contact, N) + b1.NKN(x_contact, N));
}

bool InsideFrictionCone(const Vec3& impulse, const Vec3& N, double static_coefficient) {
    Vec3 tangent = impulse - N * impulse.dot(N);
    return tangent.norm() <= static_coefficient * impulse.dot(N);
}

Vec3 StaticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Vec3& x_contact, const Vec3& N, double coeff_rest) {
    Vec3 v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    Vec3 relative_velocity = v1 - v0;
    double vN = relative_velocity.dot(N);

    Vec3 rhs = -coeff_rest * vN * N - relative_velocity;

    Mat33 K = b0.MatrixK(x_contact) + b1.MatrixK(x_contact);
    return K.inverse() * rhs;
}

Vec3 KineticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Vec3& x_contact, const Vec3& N, double kinetic_friction, double coeff_rest) {
    Vec3 v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
    Vec3 relative_velocity = v1 - v0;
    double vN = relative_velocity.dot(N);

    if (vN <= 0.0) return Vec3::Zero();

    Vec3 tangent = relative_velocity - vN * N;
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

void LinearAndAngularMomentum(const std::vector<int>& particle_indices, const std::vector<Vec3>& x, const std::vector<Vec3>& v, const std::vector<double>& mass, Vec3& p, Vec3& l) {
    double total_mass = 0.0;
    Vec3 x_com = Vec3::Zero();
    p = Vec3::Zero();
    for (int idx : particle_indices) {
        total_mass += mass[idx];
        x_com += mass[idx] * x[idx];
        p += mass[idx] * v[idx];
    }
    x_com /= total_mass;

    l = Vec3::Zero();
    for (int idx : particle_indices) {
        Vec3 r = x[idx] - x_com;
        l += mass[idx] * r.cross(v[idx]);
    }
}

} // namespace rigid_body
