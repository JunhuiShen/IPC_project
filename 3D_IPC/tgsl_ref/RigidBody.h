#pragma once
#include <float.h>

#include <core/Definitions.h>
#include <core/algebra/MatricesAndVectors.h>
#include <core/algebra/PLU.h>
#include <core/algebra/QR.h>
#include <core/algebra/MathTools.h>

namespace TGSL{

struct RigidBody{
  T infinite_mass_threshold = T(1e10);
  bool infinite_mass = false;
  T total_mass;
  TV inertia_tensor_inv;
  ALGEBRA::DenseMatrix I_hat, InertiaC4;
  Eigen::Matrix3d inertia_tensor_material;

  Particle x_com;
  Vector v_com;
  Vector4T orientation;
  Vector4T orientation_dot;
  Vector3T omega;
 
  RigidBody(){
    infinite_mass = true;
    total_mass = FLT_MAX;
    #ifdef TWO_D
    inertia_tensor_inv = {T(0)};
    #else
    inertia_tensor_inv = {T(0), T(0), T(0)};
    #endif
    orientation = {T(1), T(0), T(0), T(0)};
    x_com = Particle();
    v_com = Vector();
    omega = Vector3T();
    orientation_dot = Vector4T();
    inertia_tensor_material = Eigen::Matrix3d::Zero();
    I_hat = ALGEBRA::DenseMatrix(4, 4, T(0));
    InertiaC4 = ALGEBRA::DenseMatrix(16, 16, T(0));
  }
  RigidBody(const TVP& X, const TV& nodal_mass);
  RigidBody(const IV& particle_indices, const TVP& X, const TV& nodal_mass);
  RigidBody(const TVP& X, const TVV& v, const TV& nodal_mass);
  Vector Velocity(const Particle& x) const;
  void UpdatePositionAndOrientation(const T dt);

  void UpdateByPositionAndOmega(const T dt, const RigidBody& rb_input, const TV& y);
  TV GetPositionAndOmega() const;

  void UpdateByPositionAndTheta(const T dt, const RigidBody& rb_input, const TV& y);

  void UpdateByPartialGridInput(const TV& input);
  void UpdateByGridOutput(const T dt, const TV& output);

  TV GetPartialGridInput(const nm input_dim) const;
  TV GetGridOutput(const nm output_dim) const;

  void ToLinearNIF(const Particle& x_frame, const Vector& v_frame);
  void ToLinearIF(const Particle& x_frame, const Vector& v_frame);

  void DirectCOMUpdate(const Particle& x, const Vector& v);

  void ApplyImpulse(const Particle& x, const Vector& i);
  T NKN(const Particle& x, const Vector& N) const;
  T NKNFriction(const Particle& x, const Vector& N, const Vector& tangent_rel_V, const T mu) const;
  void VelocityFromMomenta(const Vector& p, const Vector3T& l);
  Particle WorldSpacePosition(const Particle& X) const;
  Particle MaterialSpacePosition(const Particle& x) const;
  ALGEBRA::DenseMatrix MatrixK(const Particle& x) const;
  T KineticEnergy();
  Vector3T AngularMomentum(const TVP& X, const TV& nodal_mass, const T dt) const;
  Vector3T AltAngularMomentum(const TVP& X, const TV& nodal_mass, const T dt) const;

  Eigen::Matrix2d RotationMatrix2d() const;
};

inline T NormalRelativeVelocity(const RigidBody& b0, const RigidBody& b1, const Particle& x_contact, const Vector& N){
  //the convention is that N points from b1 to b0
  Vector v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
  Vector relative_velocity;
  for (size_t alpha = 0; alpha < d; alpha++)
    relative_velocity[alpha] = v1[alpha] - v0[alpha];
  T vN = ALGEBRA::DotProduct(relative_velocity, N);
  return vN;
}

inline T CollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Particle& x_contact, const Vector& N, const T coeff_rest=T(0)){
  //the convention is that N points from b1 to b0
  Vector v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
  Vector relative_velocity;
  for (size_t alpha = 0; alpha < d; alpha++)
    relative_velocity[alpha] = v1[alpha] - v0[alpha];
  T vN = ALGEBRA::DotProduct(relative_velocity, N);

  if (vN <= T(0))
    return T(0);
  else{
    return -(T(1) + coeff_rest) * vN / (b0.NKN(x_contact, N) + b1.NKN(x_contact, N));
  }
}

inline bool InsideFrictionCone(const Vector& impulse, const Vector& N, const T static_coefficient = T(0)){
  Vector temp = ALGEBRA::Difference(impulse, ALGEBRA::Scale(N, ALGEBRA::DotProduct(impulse, N)));
  return ALGEBRA::Norm(temp) <= (static_coefficient * ALGEBRA::DotProduct(impulse, N));
}

inline Vector StaticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Particle& x_contact, const Vector& N, const T coeff_rest=T(0)){
  //the convention is that N points from b1 to b0
  Vector v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
  Vector relative_velocity = Vector();
  relative_velocity = ALGEBRA::Difference(v1, v0);
  T vN = ALGEBRA::DotProduct(relative_velocity, N);

  Vector j = Vector(), rhs = Vector();
  for (size_t alpha=0; alpha<d; alpha++){
    rhs[alpha] = -coeff_rest * vN * N[alpha] - relative_velocity[alpha];
  }

  #ifdef TWO_D
  TGSL::ALGEBRA::DenseMatrix K0 = b0.MatrixK(x_contact);
  TGSL::ALGEBRA::DenseMatrix K1 = b1.MatrixK(x_contact);
  TGSL::ALGEBRA::DenseMatrix K_inv(2,2);
  for (size_t alpha = 0; alpha < d ; alpha++){
    for (size_t beta = 0; beta < d; beta++) {
      K_inv(alpha,beta) = K0(alpha,beta)+K1(alpha,beta);
    }
  }
  K_inv = K_inv.Inverse();
  j = K_inv*rhs;
  #else
  TGSL::ALGEBRA::DenseMatrix K0 = b0.MatrixK(x_contact);
  TGSL::ALGEBRA::DenseMatrix K1 = b1.MatrixK(x_contact);
  TGSL::ALGEBRA::DenseMatrix K_inv(3,3);
  for (size_t alpha = 0; alpha < d ; alpha++){
    for (size_t beta = 0; beta < d; beta++) {
      K_inv(alpha, beta) = K0(alpha, beta) + K1(alpha, beta);
    }
  }
  K_inv = K_inv.Inverse();
  j = K_inv * rhs;
  #endif
  return j;
}

inline Vector KineticFrictionCollisionImpulse(const RigidBody& b0, const RigidBody& b1, const Particle& x_contact, const Vector& N, const T kinetic_friction = T(0), const T coeff_rest=T(0)){
  //the convention is that N points from b1 to b0
  Vector v0 = b0.Velocity(x_contact), v1 = b1.Velocity(x_contact);
  Vector relative_velocity;
  for(size_t alpha=0;alpha<d;alpha++){
    relative_velocity[alpha] = v1[alpha] - v0[alpha];
  }

  T vN = ALGEBRA::DotProduct(relative_velocity, N);

  // Computing tangential relative velocity
  Vector tangent = Vector();
  for (size_t alpha = 0; alpha<d; alpha++) {
    tangent[alpha] = relative_velocity[alpha] - vN * N[alpha];
  }

  // Normalize tangential relative velocity
  ALGEBRA::Normalize(tangent);

  Vector j = Vector();
  if(vN<=T(0)) {
	for (size_t alpha = 0; alpha<d; alpha++) {
	  j[alpha] = T(0);
	}
	return j;
  }
  else {
	T j_mag = T(0);
	T val = b0.NKNFriction(x_contact, N, tangent, kinetic_friction) + b1.NKNFriction(x_contact, N, tangent, kinetic_friction);
	j_mag = -((T(1) + coeff_rest) * vN) / val;

	for (size_t alpha = 0; alpha < d; alpha++) {
	  j[alpha] = j_mag * N[alpha] + kinetic_friction * j_mag * tangent[alpha];
	}
	return j;
  }
}

inline void LinearAndAngularMomentum(const IV& particle_indices, const TVP& x, const TVV& v, const TV& mass, Vector& p, Vector3T& l){
  //compute center of mass and p
  T total_mass = T(0);
  Particle x_com = Particle();
  p = Vector();
  for(sz j = 0; j < particle_indices.size(); j++) {
    sz i = sz(particle_indices[j]);
    total_mass += mass[i];
    for(size_t c = 0; c < d; c++) {
      x_com[c] += mass[i] * x[i][c];
      p[c] += mass[i] * v[i][c];
    }
  }
  for(size_t c = 0; c < d; c++){
    x_com[c] /= total_mass;
  }
  
  //compute total angular momentum  
  l = Vector3T();
  size_t angular_alpha_start = 0;
  #ifdef TWO_D
  angular_alpha_start = 2;
  #endif
  
  for(sz j = 0; j < particle_indices.size(); j++) {
    sz i = sz(particle_indices[j]);
    Vector r = Vector();
    for(size_t alpha = 0; alpha < d; alpha++) {
      r[alpha] = x[i][alpha] - x_com[alpha];
    }
    for(size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
      for(size_t beta = 0; beta < d; beta++) {
        for(size_t gamma = 0; gamma < d; gamma++) {
          l[alpha] += LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * mass[i] * v[i][gamma];
        }
      }
    }
  }
}

inline T ClosestInteger(const T x) {
  TGSLAssert((x - floor(x) != T(.5)), "RigidBody::ClosestInteger: Cannot determine closest integer.");
  if (x - floor(x) < T(.5)) {
    return floor(x);
  }
  else {
    return ceil(x);
  }
}

template<typename func>
inline void ShiftOmega(const T period, const Vector3T& w_n, Vector3T& w) {
  // TODO: need to think about how using norm affects whether the result will be in the correct valley
  T w_norm = ALGEBRA::Norm(w);
  T k = ClosestInteger((w_norm - ALGEBRA::Norm(w_n)) / period);
  w = ALGEBRA::Scale(w, (w_norm - k * period) / w_norm);
}

}