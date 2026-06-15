#include "rigid_body.h"

namespace Rigid_Body {

RigidBody::RigidBody(const std::vector<Eigen::Vector2d>& X, const std::vector<double>& nodal_mass) {
    v_com = Eigen::Vector2d::Zero();
    omega = Eigen::Vector3d::Zero();

    // compute total mass and center of mass
    total_mass = double(0);
    x_com = Eigen::Vector2d::Zero();
    for (std::size_t i = 0; i < X.size(); ++i) {
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * X[i];
    }
    x_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        orientation = Eigen::Vector4d(double(1),double(0), double(0), double(0));
        x_com = Eigen::Vector2d::Zero();
        inertia_tensor_inv = {double(0)};
    } 
    else {
        // Compute inertia tensor and total momenta
        size_t angular_alpha_start = 2;
        Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
        for(size_t i = 0; i < X.size(); ++i) {
            Eigen::Vector2d r = X[i] - x_com;

            for(size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
                for(size_t beta = angular_alpha_start; beta < 3; beta++) {
                    for(size_t delta = 0; delta < 2; delta++){
                        for(size_t gamma = 0; gamma < 2; gamma++){
                            for(size_t epsilon = 0; epsilon < 2; epsilon++){
                                I(alpha,beta) += ALGEBRA::LeviCivita(int(gamma),int(delta),int(alpha)) * r[delta] * nodal_mass[i] * ALGEBRA::LeviCivita(int(gamma), int(epsilon), int(beta)) * r[epsilon];
                            }
                        }
                    }
                }
            }

        }

        inertia_tensor_inv = {double(1) / I(2, 2)};
        orientation = Eigen::Vector4d(double(1), double(0), double(0), double(0));
    }
}

RigidBody::RigidBody(const std::vector<int>& particle_indices, const std::vector<Eigen::Vector2d>& X, const std::vector<double>& nodal_mass) {
    v_com = Eigen::Vector2d::Zero();
    omega = Eigen::Vector3d::Zero();

    // compute total mass and center of mass
    total_mass = double(0);
    x_com = Eigen::Vector2d::Zero();

    for (size_t j = 0; j < particle_indices.size(); j++) {
        size_t i = size_t(particle_indices[j]);
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * X[i];
    }
    x_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
        x_com = Eigen::Vector2d::Zero();
        inertia_tensor_inv = {double(0)};
    } 
    
    else {
        // Compute inertia tensor and total momenta
        size_t angular_alpha_start = 2;
        Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
        for(size_t j = 0; j < particle_indices.size(); j++) {
            size_t i = size_t(particle_indices[j]);
            Eigen::Vector2d r = X[i] - x_com;
            for(size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
                for(size_t beta = angular_alpha_start; beta < 3; beta++) {
                    for(size_t delta = 0; delta < 2; delta++){
                        for(size_t gamma = 0; gamma < 2; gamma++){
                            for(size_t epsilon = 0; epsilon < 2; epsilon++){
                                I(alpha,beta) += ALGEBRA::LeviCivita(int(gamma),int(delta),int(alpha)) * r[delta] * nodal_mass[i] * ALGEBRA::LeviCivita(int(gamma), int(epsilon), int(beta)) * r[epsilon];
                            }
                        }
                    }
                }
            }

        }

        inertia_tensor_inv = {double(1) / I(2, 2)};
        orientation = Eigen::Vector4d(double(1), double(0), double(0), double(0));
    }
}

RigidBody::RigidBody(const std::vector<Eigen::Vector2d>& X, const std::vector<Eigen::Vector2d>& v, const std::vector<double>& nodal_mass) {
    // compute total mass, center of mass, and center of mass velocity
    total_mass = double(0);
    x_com = Eigen::Vector2d::Zero();
    v_com = Eigen::Vector2d::Zero();

    for (size_t i = 0; i < X.size(); ++i) {
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * X[i];
        v_com += nodal_mass[i] * v[i];
    }
    x_com /= total_mass;
    v_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        orientation = Eigen::Vector4d(double(1), double(0), double(0), double(0));
        x_com = Eigen::Vector2d::Zero();
        v_com = Eigen::Vector2d::Zero();

        inertia_tensor_inv = {double(0)};
    } 
    else {
        // compute inertia tensor and total momenta
        Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
        Eigen::Vector2d p = Eigen::Vector2d::Zero();
        Eigen::Vector3d l = Eigen::Vector3d::Zero();
        size_t angular_alpha_start = 2;
        for(size_t i = 0; i < X.size(); ++i) {
            Eigen::Vector2d r = X[i] - x_com;
            p += nodal_mass[i] * v[i];
            for(size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
                for(size_t beta = 0; beta < 2; beta++){
                    for(size_t gamma = 0; gamma < 2; gamma++){
                        l[alpha] += ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * nodal_mass[i] * v[i][gamma];
                    }
                }
            }

            for(size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
                for(size_t beta = angular_alpha_start; beta < 3; beta++) {
                    for(size_t delta = 0; delta < 2; delta++){
                        for(size_t gamma = 0; gamma < 2; gamma++){
                            for(size_t epsilon = 0; epsilon < 2; epsilon++){
                                I(alpha,beta) += ALGEBRA::LeviCivita(int(gamma),int(delta),int(alpha)) * r[delta] * nodal_mass[i] * ALGEBRA::LeviCivita(int(gamma), int(epsilon), int(beta)) * r[epsilon];
                            }
                        }
                    }
                }
            }

        }
        inertia_tensor_inv = {double(1) / I(2, 2)};
        omega = {double(0), double(0), l[2] * inertia_tensor_inv[0]};
        orientation = {double(0),double(0), double(0), double(1)};
    }
}

Eigen::Vector2d RigidBody::Velocity(const Eigen::Vector2d& x) const {
    Eigen::Vector2d result = v_com;
    size_t angular_alpha_start = 2;
    Eigen::Vector2d r = x - x_com;
    for(size_t alpha = 0; alpha < 2; alpha++){
        for(size_t beta = angular_alpha_start; beta < 3; beta++){
            for(size_t gamma = 0; gamma < 2; gamma++){
                result[alpha] += ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma)) * omega[beta] * r[gamma];
            }
        }
    }
    return result;
}

void RigidBody::UpdatePositionAndOrientation(const double dt) {
    x_com += dt * v_com;
    Eigen::Vector4d q_hat = ALGEBRA::QuaternionFromVector(dt * omega);
    orientation = ALGEBRA::QuaternionMultiply(q_hat, orientation);
}

void RigidBody::ApplyImpulse(const Eigen::Vector2d& x, const Eigen::Vector2d& impulse) {
    if (infinite_mass)
        return;
    else{
        v_com += impulse / total_mass;

        size_t angular_alpha_start = 2;
        Eigen::Vector2d r = x - x_com;
        
        Eigen::Vector3d rXi = Eigen::Vector3d::Zero();
        for(size_t alpha = angular_alpha_start; alpha < 3; alpha++){
            for(size_t beta = 0; beta < 2; beta++){
                for(size_t gamma = 0; gamma < 2; gamma++){
                    rXi[alpha] += ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * impulse[gamma];
                }
            }
        }
        omega[2] += rXi[2] * inertia_tensor_inv[0];
    }
}

Eigen::Matrix2d RigidBody::MatrixK(const Eigen::Vector2d& x) const {

    Eigen::Vector2d r = x - x_com;

    Eigen::Matrix2d K;
    K(0, 0) = (double(1) / total_mass) + (r[1] * r[1]) * inertia_tensor_inv[0];
    K(0, 1) = (-r[0] * r[1]) * inertia_tensor_inv[0];
    K(1, 0) = K(0, 1);
    K(1, 1) = (double(1) / total_mass) + (r[0] * r[0]) * inertia_tensor_inv[0];

    if (infinite_mass) {
        K.setZero();
    }

    return K;
}

double RigidBody::NKN(const Eigen::Vector2d& x, const Eigen::Vector2d& N) const {
    if (infinite_mass)
        return double(0);
    else{
        double result = N.norm() / total_mass;

        //(1) rXN
        size_t angular_alpha_start = 2;
        Eigen::Vector2d r = x - x_com;

        Eigen::Vector3d rXN = Eigen::Vector3d::Zero();
        for(size_t alpha = angular_alpha_start; alpha < 3; alpha++){
            for(size_t beta = 0; beta < 2; beta++){
                for(size_t gamma = 0; gamma < 2; gamma++){
                    rXN[alpha] += ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * N[gamma];
                }
            }
        }

        //(2) I^{-1}(rXN)
        Eigen::Vector3d Iinv_rXN = Eigen::Vector3d::Zero();
        Iinv_rXN[2] = rXN[2] * inertia_tensor_inv[0];

        //(3) (I^{-1}(rXN))Xr
        Eigen::Vector3d AN = Eigen::Vector3d::Zero();
        for(size_t alpha = 0; alpha < 2; alpha++){
            for(size_t beta = angular_alpha_start; beta < 3; beta++){
                for(size_t gamma = 0; gamma < 2; gamma++){
                    AN[alpha] += ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma)) * Iinv_rXN[beta] * r[gamma];
                }
            }
        }

        for(size_t alpha = 0; alpha < 2; alpha++){
            result += AN[alpha] * N[alpha];
        }


        return result;
    }
}

double RigidBody::NKNFriction(const Eigen::Vector2d& x, const Eigen::Vector2d& N, const Eigen::Vector2d& tangent_rel_V, const double mu) const {
    if (infinite_mass)
        return double(0);
    else{
        Eigen::Vector2d N_muT = N - mu * tangent_rel_V;
        double result = N.dot(N_muT) / total_mass;
        
        //(1) rX(N-muT)
        size_t angular_alpha_start = 2;
        Eigen::Vector2d r = x - x_com;

        Eigen::Vector3d rXN = Eigen::Vector3d::Zero();
        for(size_t alpha = angular_alpha_start; alpha < 3; alpha++){
            for(size_t beta = 0; beta < 2; beta++){
                for(size_t gamma = 0; gamma < 2; gamma++){
                    rXN[alpha] += ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * N_muT[gamma];
                }
            }
        }

        //(2) I^{-1}(rX(N-muT))
        Eigen::Vector3d Iinv_rXN = Eigen::Vector3d::Zero();
        Iinv_rXN[2] = rXN[2] * inertia_tensor_inv[0];

        //(3) (I^{-1}(rX(N-muT)))Xr
        Eigen::Vector3d AN = Eigen::Vector3d::Zero();
        for(size_t alpha = 0; alpha < 2; alpha++){
            for(size_t beta = angular_alpha_start; beta < 3; beta++){
                for(size_t gamma = 0; gamma < 2; gamma++){
                    AN[alpha] += ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma)) * Iinv_rXN[beta] * r[gamma];
                }
            }
        }

        for(size_t alpha = 0; alpha < 2; alpha++){
            result += AN[alpha] * N[alpha];
        }


        return result;
    }
}

void RigidBody::VelocityFromMomenta(const Eigen::Vector2d& p, const Eigen::Vector3d& l) {
    v_com = p / total_mass;
    omega = Eigen::Vector3d::Zero();
    omega[2] = l[2] * inertia_tensor_inv[0];
}

Eigen::Vector2d RigidBody::WorldSpacePosition(const Eigen::Vector2d& X) const {
    Eigen::Vector2d x = x_com;

    Eigen::Vector3d R = Eigen::Vector3d::Zero(), r = Eigen::Vector3d::Zero();
    for(size_t alpha = 0; alpha < 2; alpha++){
        R[alpha] = X[alpha];
    }
    r = ALGEBRA::QuaternionRotate(orientation,R);
    for(size_t alpha = 0; alpha < 2; alpha++){
        x[alpha] += r[alpha];
    }
    return x;
}

Eigen::Vector2d RigidBody::MaterialSpacePosition(const Eigen::Vector2d& x) const {
    Eigen::Vector3d r = Eigen::Vector3d::Zero();
    for(size_t alpha = 0; alpha < 2; alpha++){
        r[alpha] = x[alpha] - x_com[alpha];
    }
    Eigen::Vector3d X_full = ALGEBRA::QuaternionRotate(ALGEBRA::ConjugateQuaternion(orientation), r);
    Eigen::Vector2d X = {X_full[0], X_full[1]};
    return X;
}

}  // namespace Rigid_Body
