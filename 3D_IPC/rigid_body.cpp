#include "rigid_body.h"

namespace Rigid_Body {

RigidBody::RigidBody(const std::vector<Eigen::Vector3d>& X, const std::vector<double>& nodal_mass) {
    v_com = Eigen::Vector3d::Zero();
    omega = Eigen::Vector3d::Zero();

    total_mass = double(0);
    x_com = Eigen::Vector3d::Zero();

    for (std::size_t i = 0; i < X.size(); ++i) {
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * X[i];
    }
    x_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        orientation = {double(1), double(0), double(0), double(0)};
        x_com = Eigen::Vector3d::Zero();
        inertia_tensor_inv = {double(0), double(0), double(0)};
    } 
    else {
        size_t angular_alpha_start = 0;
        Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
        for (std::size_t i = 0; i < X.size(); ++i) {
            Eigen::Vector3d r = X[i] - x_com;
            for(size_t alpha = angular_alpha_start; alpha < 3; alpha++){
                for(size_t beta = angular_alpha_start; beta < 3; beta++){
                    for(size_t delta = 0; delta < 3; delta++){
                        for(size_t gamma = 0; gamma < 3; gamma++){
                            for(size_t epsilon = 0; epsilon < 3; epsilon++){
                                I(alpha,beta)+=ALGEBRA::LeviCivita(int(gamma), int(delta), int(alpha)) * r[delta] * nodal_mass[i] * ALGEBRA::LeviCivita(int(gamma), int(epsilon), int(beta)) * r[epsilon];
                            }
                        }
                    }
                }
            }
        }
        // Diagonalize Inertia tensor
        std::vector<double> lambda;
        Eigen::MatrixXd Q_temp;
        ALGEBRA::SymmetricEigenDecomposition(I, Q_temp, lambda, 20, true);
        Eigen::Matrix3d Q = Q_temp;
        if(Q.determinant() < double(0)){
            Q(0, 2) = -Q(0, 2);
            Q(1, 2) = -Q(1, 2);
            Q(2, 2) = -Q(2, 2);
        }
        orientation = ALGEBRA::QuaternionFromRotationMatrix(Q);
        inertia_tensor_inv = {double(1)/lambda[0], double(1)/lambda[1], double(1)/lambda[2]};
    }
}

RigidBody::RigidBody(const std::vector<int>& particle_indices, const std::vector<Eigen::Vector3d>& X, const std::vector<double>& nodal_mass) {
    v_com = Eigen::Vector3d::Zero();
    omega = Eigen::Vector3d::Zero();

    total_mass = 0.0;
    x_com = Eigen::Vector3d::Zero();
    for (size_t j = 0; j < particle_indices.size(); ++j) {
        size_t i = size_t(particle_indices[j]);
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * X[i];
    }
    x_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        orientation = {double(1),double(0),double(0),double(0)};
        x_com = Eigen::Vector3d::Zero();
        inertia_tensor_inv = {double(0),double(0),double(0)};
    }

    // Compute inertia tensor and total momenta
    size_t angular_alpha_start = 0;

    Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
    for(size_t j = 0; j < particle_indices.size(); j++){
        size_t i = size_t(particle_indices[j]);
        Eigen::Vector3d r = X[i] - x_com;
        for(size_t alpha = angular_alpha_start; alpha < 3; alpha++){
            for(size_t beta = angular_alpha_start; beta < 3; beta++){
                for(size_t delta = 0; delta < 3; delta++){
                    for(size_t gamma = 0; gamma < 3; gamma++){
                        for(size_t epsilon = 0; epsilon < 3; epsilon++){
                            I(alpha,beta) += ALGEBRA::LeviCivita(int(gamma),int(delta),int(alpha)) * r[delta] * nodal_mass[i] * ALGEBRA::LeviCivita(int(gamma), int(epsilon), int(beta)) * r[epsilon];
                        }
                    }
                }
            }
        }
    }
    // Diagonalize inertia tensor
    std::vector<double> lambda;
    Eigen::MatrixXd Q_temp;
    ALGEBRA::SymmetricEigenDecomposition(I,Q_temp,lambda,20,true);
    Eigen::Matrix3d Q = Q_temp;

    if(Q.determinant() < double(0)){
        Q(0,2) = -Q(0,2);
        Q(1,2) = -Q(1,2);
        Q(2,2) = -Q(2,2);
    }

    orientation = ALGEBRA::QuaternionFromRotationMatrix(Q);
    inertia_tensor_inv = {double(1)/lambda[0], double(1)/lambda[1], double(1)/lambda[2]};

}

RigidBody::RigidBody(const std::vector<Eigen::Vector3d>& X, const std::vector<Eigen::Vector3d>& v, const std::vector<double>& nodal_mass) {
    // compute total mass, center of mass and center of mass velocities
    total_mass = double(0);
    x_com = Eigen::Vector3d::Zero();
    v_com = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < X.size(); ++i) {
        total_mass += nodal_mass[i];
        x_com += nodal_mass[i] * X[i];
        v_com += nodal_mass[i] * v[i];
    }
    x_com /= total_mass;
    v_com /= total_mass;

    if (total_mass > infinite_mass_threshold) {
        infinite_mass = true;
        orientation = {double(1),double(0),double(0),double(0)};
        x_com = Eigen::Vector3d::Zero();
        v_com = Eigen::Vector3d::Zero();
        inertia_tensor_inv = {double(0),double(0),double(0)};
    }

    // compute inertia tensor and total momenta
    Eigen::Matrix3d I = Eigen::Matrix3d::Zero();

    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d l = Eigen::Vector3d::Zero();

    size_t angular_alpha_start = 0;
    for(size_t i = 0; i < X.size(); i++){
        Eigen::Vector3d r = X[i] - x_com;
        p += nodal_mass[i] * v[i];
        for(size_t alpha = angular_alpha_start; alpha < 3; alpha++){
            for(size_t beta = 0; beta < 3; beta++){
                for(size_t gamma = 0; gamma < 3; gamma++){
                    l[alpha] += ALGEBRA::LeviCivita(int(alpha),int(beta),int(gamma)) * r[beta] * nodal_mass[i] * v[i][gamma];
                }
            }
        }

        for(size_t alpha=angular_alpha_start;alpha<3;alpha++){
            for(size_t beta=angular_alpha_start;beta<3;beta++){
                for(size_t delta=0;delta<3;delta++){
                    for(size_t gamma=0;gamma<3;gamma++){
                        for(size_t epsilon=0;epsilon<3;epsilon++){
                            I(alpha,beta)+=ALGEBRA::LeviCivita(int(gamma), int(delta), int(alpha))*r[delta]*nodal_mass[i]*ALGEBRA::LeviCivita(int(gamma), int(epsilon), int(beta))*r[epsilon];
                        }
                    }
                }
            }
        }
    }
    //solve for omega from angular momentum and intertia tensor
    std::vector<double> l_rhs={l[0],l[1],l[2]};
    std::vector<double> omega_solve(3);
    ALGEBRA::PLUSolve(I,l_rhs,omega_solve);
    omega={omega_solve[0],omega_solve[1],omega_solve[2]};

    //diagonalize intertia tensor  
    std::vector<double> lambda;
    Eigen::MatrixXd Q_temp;
    ALGEBRA::SymmetricEigenDecomposition(I,Q_temp,lambda,20,true);
    Eigen::Matrix3d Q = Q_temp;
    if(Q.determinant()<double(0)){
        Q(0,2)=-Q(0,2);
        Q(1,2)=-Q(1,2);
        Q(2,2)=-Q(2,2);
    }
    orientation=ALGEBRA::QuaternionFromRotationMatrix(Q);
    inertia_tensor_inv={double(1)/lambda[0],double(1)/lambda[1],double(1)/lambda[2]};
}

Eigen::Vector3d RigidBody::Velocity(const Eigen::Vector3d& x) const {
	Eigen::Vector3d result=v_com;
	size_t angular_start=0;
	Eigen::Vector3d r = x - x_com;
	for(size_t alpha=0;alpha<3;alpha++){
		for(size_t beta=angular_start;beta<3;beta++){
			for(size_t gamma=0;gamma<3;gamma++){
				result[alpha]+=ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma))*omega[beta]*r[gamma];
			}
		}
	}
	return result;
}

void RigidBody::UpdatePositionAndOrientation(const double dt) {
    for(size_t c=0;c<3;c++)
        x_com[c]+=dt*v_com[c];

	Eigen::Vector4d q_hat=ALGEBRA::QuaternionFromVector({dt*omega[0],dt*omega[1],dt*omega[2]});
	orientation=ALGEBRA::QuaternionMultiply(q_hat,orientation);
}

void RigidBody::ApplyImpulse(const Eigen::Vector3d& x, const Eigen::Vector3d& impulse) {
	if(infinite_mass)
		return;
	else{
		//linear momentum change
        v_com += impulse / total_mass;
		//angular momentum change
		//(1) rXi
		size_t angular_start=0;
		Eigen::Vector3d r = x - x_com;

		Eigen::Vector3d rXi= Eigen::Vector3d::Zero();
		for(size_t alpha=angular_start;alpha<3;alpha++){
			for(size_t beta=0;beta<3;beta++){
				for(size_t gamma=0;gamma<3;gamma++){
					rXi[alpha]+=ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma))*r[beta]*impulse[gamma];
				}
			}
		}

		//(2) I^{-1}(rXi)
		Eigen::Vector4d q_T=ALGEBRA::ConjugateQuaternion(orientation);
		Eigen::Vector3d Iinv_rXi=ALGEBRA::QuaternionRotate(q_T,rXi);
		for(size_t alpha=0;alpha<3;alpha++){
			Iinv_rXi[alpha]*=inertia_tensor_inv[alpha];
		}
		Iinv_rXi=ALGEBRA::QuaternionRotate(orientation,Iinv_rXi);
		for(size_t alpha=0;alpha<3;alpha++)
			omega[alpha]+=Iinv_rXi[alpha];
	}
}

Eigen::Matrix3d RigidBody::MatrixK(const Eigen::Vector3d& x) const {
    Eigen::Vector3d r = x - x_com;

    Eigen::Matrix3d r_cross = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d I_inv = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    // r*
    r_cross(0,0) = double(0);  r_cross(0,1) = r[2]; r_cross(0,2) = -r[1];
    r_cross(1,0) = -r[2]; r_cross(1,1) = double(0); r_cross(1,2) = r[0];
    r_cross(2,0) = r[1]; r_cross(2,1) = -r[0]; r_cross(2,2) = double(0);

    // Quaternion to Matrix
    R(0,0) = double(1)-double(2)*orientation[2]*orientation[2]-2*orientation[3]*orientation[3];  R(0,1) = double(2)*(orientation[1]*orientation[2] - orientation[0]*orientation[3]); R(0,2) = double(2)*(orientation[1]*orientation[3]+orientation[0]*orientation[2]);
    R(1,0) = double(2)*(orientation[1]*orientation[2]+orientation[0]*orientation[3]); R(1,1) = double(1)-double(2)*orientation[1]*orientation[1]-double(2)*orientation[3]*orientation[3]; R(1,2) = double(2)*(orientation[2]*orientation[3] - orientation[0]*orientation[1]);
    R(2,0) = double(2)*(orientation[1]*orientation[3] - orientation[0]*orientation[2]); R(2,1) = double(2)*(orientation[2]*orientation[3]+orientation[0]*orientation[1]); R(2,2) = double(1)-double(2)*orientation[1]*orientation[1]-double(2)*orientation[2]*orientation[2];

    // Inertia tensor inverse
    I_inv(0,0) = inertia_tensor_inv[0];  I_inv(0,1) = double(0); I_inv(0,2) = double(0);
    I_inv(1,0) = double(0); I_inv(1,1) = inertia_tensor_inv[1]; I_inv(1,2) = double(0);
    I_inv(2,0) = double(0); I_inv(2,1) = double(0); I_inv(2,2) = inertia_tensor_inv[2];

    I_inv = R * I_inv * R.transpose();
    K = r_cross.transpose() * I_inv * r_cross;

    for (size_t alpha = 0; alpha <3 ; alpha++) {
        K(alpha,alpha) += double(1)/total_mass;
    }

    // Immovable static obejcts like the ground plane can be created by seting K = 0
    if (infinite_mass) {
        for (size_t alpha = 0; alpha < 3 ; alpha++) {
            for (size_t beta = 0; beta < 3; beta++) {
                K(alpha,beta) = double(0);
            }
        }
    }
    return K;
}

double RigidBody::NKN(const Eigen::Vector3d& x, const Eigen::Vector3d& N) const {
	if(infinite_mass)
		return double(0);
	else{
		double result = N.norm()/total_mass;

		//(1) rXN
		size_t angular_start=0;
		Eigen::Vector3d r = x - x_com;

		Eigen::Vector3d rXN = Eigen::Vector3d::Zero();
		for(size_t alpha=angular_start;alpha<3;alpha++){
			for(size_t beta=0;beta<3;beta++){
				for(size_t gamma=0;gamma<3;gamma++){
					rXN[alpha]+=ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma))*r[beta]*N[gamma];
				}
			}
		}

		//(2) I^{-1}(rXN)
		Eigen::Vector3d Iinv_rXN=Eigen::Vector3d::Zero();
		Eigen::Vector4d q_T=ALGEBRA::ConjugateQuaternion(orientation);
		Iinv_rXN=ALGEBRA::QuaternionRotate(q_T,rXN);
		for(size_t alpha=0;alpha<3;alpha++){
			Iinv_rXN[alpha]*=inertia_tensor_inv[alpha];
		}
		Iinv_rXN=ALGEBRA::QuaternionRotate(orientation,Iinv_rXN);

		//(3) (I^{-1}(rXN))Xr
		Eigen::Vector3d AN=Eigen::Vector3d::Zero(); 
		for(size_t alpha=0;alpha<3;alpha++){
			for(size_t beta=angular_start;beta<3;beta++){
				for(size_t gamma=0;gamma<3;gamma++){
					AN[alpha]+=ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma))*Iinv_rXN[beta]*r[gamma];
				}
			}
		}

		for(size_t alpha=0;alpha<3;alpha++)
			result+=AN[alpha]*N[alpha];

		return result;
	}
}

double RigidBody::NKNFriction(const Eigen::Vector3d& x, const Eigen::Vector3d& N, const Eigen::Vector3d& tangent_rel_V, double mu) const {
    if(infinite_mass)
    return double(0);
    else{
        Eigen::Vector3d N_muT = N - mu * tangent_rel_V;

        double result = N.dot(N_muT)/total_mass;

        //(1) rX(N-muT)
        size_t angular_start=0;
        Eigen::Vector3d r = x - x_com;
        Eigen::Vector3d rXN = Eigen::Vector3d::Zero();
        for(size_t alpha=angular_start;alpha<3;alpha++){
            for(size_t beta=0;beta<3;beta++){
                for(size_t gamma=0;gamma<3;gamma++){
                    rXN[alpha]+=ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma))*r[beta]*N_muT[gamma];
                }
            }
        }

        //(2) I^{-1}(rX(N-muT))
        Eigen::Vector3d Iinv_rXN=Eigen::Vector3d::Zero();
        Eigen::Vector4d q_T=ALGEBRA::ConjugateQuaternion(orientation);
        Iinv_rXN=ALGEBRA::QuaternionRotate(q_T,rXN);
        for(size_t alpha=0;alpha<3;alpha++){
            Iinv_rXN[alpha]*=inertia_tensor_inv[alpha];
        }
        Iinv_rXN=ALGEBRA::QuaternionRotate(orientation,Iinv_rXN);

        //(3) (I^{-1}(rX(N-muT)))Xr
        Eigen::Vector3d AN = Eigen::Vector3d::Zero();
        for(size_t alpha=0;alpha<3;alpha++){
            for(size_t beta=angular_start;beta<3;beta++){
                for(size_t gamma=0;gamma<3;gamma++){
                    AN[alpha]+=ALGEBRA::LeviCivita(int(alpha), int(beta), int(gamma))*Iinv_rXN[beta]*r[gamma];
                }
            }
        }

        for(size_t alpha=0;alpha<3;alpha++)
            result+=AN[alpha]*N[alpha];

        return result;
    }
}

void RigidBody::VelocityFromMomenta(const Eigen::Vector3d& p, const Eigen::Vector3d& l) {
  for(size_t alpha=0;alpha<3;alpha++)
    v_com[alpha]=p[alpha]/total_mass;

  Eigen::Vector4d q_T=ALGEBRA::ConjugateQuaternion(orientation);
  omega=ALGEBRA::QuaternionRotate(q_T,l);
  for(size_t alpha=0;alpha<3;alpha++){
    omega[alpha]*=inertia_tensor_inv[alpha];
  }
  omega=ALGEBRA::QuaternionRotate(orientation,omega);

}

Eigen::Vector3d RigidBody::WorldSpacePosition(const Eigen::Vector3d& X) const {
    Eigen::Vector3d x=x_com;

    Eigen::Vector3d R = Eigen::Vector3d::Zero(), r = Eigen::Vector3d::Zero();

    R = X;
    r = ALGEBRA::QuaternionRotate(orientation,R);

    x += r;

    return x;
}

Eigen::Vector3d RigidBody::MaterialSpacePosition(const Eigen::Vector3d& x) const {
    Eigen::Vector3d r = x - x_com;

    Eigen::Vector3d X_full=ALGEBRA::QuaternionRotate(ALGEBRA::ConjugateQuaternion(orientation),r);
   
    Eigen::Vector3d X=X_full;
    return X;
}
}  // namespace rigid_body
