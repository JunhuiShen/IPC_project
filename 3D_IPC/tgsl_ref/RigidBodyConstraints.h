#pragma once
#include <float.h>
#include <algorithm>
#include <iomanip> // For manipulators

#include <core/Definitions.h>
#include <core/algebra/MatricesAndVectors.h>
#include <core/algebra/MathTools.h>
#include <core/algebra/VectorTools.h>
#include <rigid_body/RigidBody.h>
#include <rigid_body/RBConstraintInfo.h>
#include <LineSearch.h>
#include <elasticity/PointEdgeBarrier.h>


namespace TGSL {

struct RBsInfo {
  nm current_rb;
  //nm rb_hat_op;
  std::vector<RigidBody> rigid_bodies_hat;
  std::vector<RigidBody> rigid_bodies_input;
};

struct NewtonsMethodInfo {
  bool hard_tol; //
  bool solve_full_system;
  bool do_line_search;
  bool failing_status; //
  bool only_gradient_descent;
  nm line_search_op;
  sz max_it;
  T tol;
  TV energy; //
  TV residual_norm; //
  std::vector<Vector4T> direction_info;
};

template<typename VectorType>
void PrintVector(const std::string& name, const VectorType& v) {
  std::cout << name << ": ";
  for (sz i = 0; i < v.size(); ++i) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

void GetDqDw(Eigen::Matrix<T, 4, 3>& DqDw, const Vector4T& q_n, const T dt) {
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 3; beta++) {
      for (size_t gamma = 0; gamma < 4; gamma++) {
        DqDw(alpha, beta) += T(.5) * dt * QPT(alpha, beta + 1, gamma) * q_n[gamma];
      }
    }
  }
}

Eigen::Matrix<T, 4, 3> F_matrix(const Vector4T& q) {
  // q (0, xi) = F(q) * xi
  Eigen::Matrix<T, 4, 3> F = Eigen::Matrix<T, 4, 3>::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t gamma = 0; gamma < 3; gamma++) {
        F(alpha, gamma) += QPT(alpha, beta, gamma + 1) * q[beta];
      }
    }
  }
  return F;
}

void GetdydY(const Vector4T& q_n, Eigen::Matrix<T, 7, 6>& dydY) {
  //
  for (size_t alpha = 0; alpha < 3; alpha++) {
    dydY(alpha, alpha) = T(1);
  }
  //
  Eigen::Matrix<T, 4, 3> F = F_matrix(q_n);
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 3; beta++) {
      dydY(alpha + 3, beta + 3) = F(alpha, beta);
    }
  }
}

Eigen::Vector4d ExpOf(const Vector3T& xi) {
  Eigen::Vector4d exp_xi(0, 0, 0, 0);
  T xi_norm = ALGEBRA::Norm(xi);
  if (xi_norm < 1e-8) {
    exp_xi << 1, 0, 0, 0;
  }
  else {
    exp_xi[0] = cos(xi_norm);
    for (size_t alpha = 1; alpha < 4; alpha++) {
      exp_xi[alpha] = sin(xi_norm) * xi[alpha - 1] / xi_norm;
    }
  }
  return exp_xi;
}

Vector4T ExpOf4T(const Vector3T& xi) {
  Vector4T exp;
  T xi_norm = ALGEBRA::Norm(xi);
  if (xi_norm < 1e-8) {
    exp = { T(1), T(0), T(0), T(0) };
  }
  else {
    exp[0] = cos(xi_norm);
    for (size_t alpha = 1; alpha < 4; alpha++) {
      exp[alpha] = sin(xi_norm) * xi[alpha - 1] / xi_norm;
    }
  }
  return exp;
}

Eigen::Matrix<T, 4, 3> DerivExpOf(const T dt, const Vector3T& w) {
  Eigen::Matrix<T, 4, 3> D = Eigen::Matrix<T, 4, 3>::Zero();
  Eigen::Matrix3d I_3 = Eigen::Matrix3d::Identity();
  T w_norm = ALGEBRA::Norm(w);
  if (w_norm == 0) {
    for (size_t alpha = 0; alpha < 3; alpha++) {
      D(alpha + 1, alpha) = dt / T(2);
    }
  }
  else {
    T angle = w_norm * dt / T(2);
    // first row of D
    for (size_t beta = 0; beta < 3; beta++) {
      D(0, beta) = -sin(angle) * dt / T(2) * w[beta] / w_norm;
    }
    // rest rows of D
    for (size_t alpha = 1; alpha < 4; alpha++) {
      for (size_t beta = 0; beta < 3; beta++) {
        D(alpha, beta) = cos(angle) * dt / T(2) * w[beta] * w[alpha - 1] / (w_norm * w_norm) + sin(angle) * (I_3(alpha - 1, beta) * w_norm - w[alpha - 1] * w[beta] / w_norm) / (w_norm * w_norm);
      }
    }
  }
  

  return D;
}

void DqDw(const T dt, const Vector3T& w, const Vector4T& q_input, Eigen::Matrix<T, 4, 3>& dqdw) {
  Eigen::Matrix<T, 4, 3> dexpdw = DerivExpOf(dt, w);
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 3; beta++) {
      for (size_t gamma = 0; gamma < 4; gamma++) {
        for (size_t delta = 0; delta < 4; delta++) {
          dqdw(alpha, beta) += QPT(alpha, gamma, delta) * dexpdw(gamma, beta) * q_input[delta];
        }
      }
    }
  }
}

// Implicit PCP

//void AddLAPCPGradientHessian(const T k, const RigidBody& rb_0, const RigidBody& rb_1, const Particle& X_0, const Particle& X_1, const Vector4T& q_n, const T dt, Eigen::Matrix<T, 6, 1>& g0_new, Eigen::Matrix<T, 6, 6>& H0_new) {
//  Eigen::Matrix<T, 7, 1> g0_old = Eigen::Matrix<T, 7, 1>::Zero();
//  Eigen::Matrix<T, 7, 7> H0_old = Eigen::Matrix<T, 7, 7>::Zero();
//  PCPGradientHessian(k, rb_0, rb_1, X_0, X_1, g0_old, H0_old);
//  g0_old = g0_old * dt * dt / T(4);
//  H0_old = H0_old * dt * dt / T(4);
//  Eigen::Matrix<T, 7, 6> dydY_n = Eigen::Matrix<T, 7, 6>::Zero(), dydY_np1 = Eigen::Matrix<T, 7, 6>::Zero();
//  GetdydY(q_n, dydY_n);
//  GetdydY(rb_0.orientation, dydY_np1);
//  g0_new = g0_new - dydY_n.transpose() * g0_old; // negative sign since L = KE - PE
//  H0_new = H0_new - dydY_n.transpose() * H0_old * dydY_np1; // negative sign since L = KE - PE
//}
//
//void Add7DPCPGradientHessian(const T k, const RigidBody& rb_0, const RigidBody& rb_1, const Particle& X_0, const Particle& X_1, const T dt, Eigen::Matrix<T, 7, 1>& g0_new, Eigen::Matrix<T, 7, 7>& H0_new) {
//  Eigen::Matrix<T, 7, 1> g0_old = Eigen::Matrix<T, 7, 1>::Zero();
//  Eigen::Matrix<T, 7, 7> H0_old = Eigen::Matrix<T, 7, 7>::Zero();
//  PCPGradientHessian(k, rb_0, rb_1, X_0, X_1, g0_old, H0_old);
//  g0_new -= g0_old * dt * dt / T(4);
//  H0_new -= H0_old * dt * dt / T(4);
//}
//
//void AddAllLAPCPGradientHessian(const IV& segment_mesh, const IV& incident_elements_rb, const std::vector<TVP>& constraint_segments_info, const std::vector<RigidBody>& rigid_bodies, const nm& current_rb, const RigidBody& rb_n, const RigidBody& rb, const T& dt, const T& k, Eigen::Matrix<T, 6, 1>& g_new, Eigen::Matrix<T, 6, 6>& H_new) {
//  for (sz n = 0; n < incident_elements_rb.size(); n++) {
//    std::div_t result = std::div(incident_elements_rb[n], nm(2));
//    nm segment_number = result.quot;
//    nm segment_order = result.rem;
//    nm rb_number = segment_mesh[incident_elements_rb[n]];
//    TGSLAssert(rb_number == current_rb, "RigidBodyConstraints::NewtonsMethod_MultiConstraints: rb_number and current_rb do not match.");
//    nm other_rb_number = segment_mesh[incident_elements_rb[n] + nm(1) - nm(2) * segment_order];
//    // segment_order = 0 --> incident_elements_rb[n] + nm(1) - nm(2) * segment_order = incident_elements_rb[n] + 1
//    // segment_order = 1 --> incident_elements_rb[n] + nm(1) - nm(2) * segment_order = incident_elements_rb[n] - 1
//    TVP segment_info = constraint_segments_info[segment_number];
//    if (segment_order == 0) {
//      AddLAPCPGradientHessian(k, rb, rigid_bodies[other_rb_number], segment_info[0], segment_info[1], rb_n.orientation, dt, g_new, H_new);
//    }
//    else {
//      AddLAPCPGradientHessian(k, rb, rigid_bodies[other_rb_number], segment_info[1], segment_info[0], rb_n.orientation, dt, g_new, H_new);
//    }
//  }
//}

// Other PE
void AddGravityGradient(const RigidBody& rb, const T dt, const T gravity, Eigen::Matrix<T, 6, 1>& g_new) {
  g_new(1) -= dt * dt * rb.total_mass * gravity / T(4);
}

void AddBEGravityGradient(const RigidBody& rb, const T dt, const T gravity, Eigen::Matrix<T, 6, 1>& g_new) {
  g_new[1] += dt * dt * rb.total_mass * gravity;
}

void Add7DGravityGradient(const RigidBody& rb, const T dt, const T gravity, Eigen::Matrix<T, 7, 1>& g_new) {
  g_new[1] -= dt * dt * rb.total_mass * gravity / T(4);
}


// New Lagrangian (Using unit quaternion and midpoint rule)

T NewLagrangian(const RigidBody& rb_n, const RigidBody& rb_np1, const T dt) {
  T L = T(0), KE_linear = T(0), KE_angular = T(0), PE = T(0);

  // KE_linear
  //Vector v_COM = ALGEBRA::Scale(ALGEBRA::Difference(rb_np1.x_com, rb_n.x_com), T(1) / dt);
  //KE_linear = T(.5) * rb_n.total_mass * ALGEBRA::NormSquared(v_COM);

  // KE_angular
  Vector3T w = ALGEBRA::VectorOfQuaternion(ALGEBRA::QuaternionMultiply(ALGEBRA::ConjugateQuaternion(rb_n.orientation), rb_np1.orientation));
  Eigen::Vector3d omega_material(w[0], w[1], w[2]);
  KE_angular = T(2) / (dt * dt) * omega_material.transpose() * rb_n.inertia_tensor_material * omega_material;

  L = KE_linear + KE_angular - PE;
  return L;
}

void OmegaDerivatives(const Vector4T& p, const Vector4T& q, Eigen::Vector3d& w, Eigen::Matrix<T, 3, 4>& DwDp, Eigen::Matrix<T, 3, 4>& DwDq, std::vector<Eigen::Matrix<T, 3, 4>>& D2wDpDq) {
  // "Omega" = Im(p* q)
  w = Eigen::Vector3d::Zero();
  DwDp = Eigen::Matrix<T, 3, 4>::Zero();
  DwDq = Eigen::Matrix<T, 3, 4>::Zero();
  Eigen::Vector4d Sign(T(1), T(-1), T(-1), T(-1));
  for (size_t alpha = 0; alpha < 3; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t gamma = 0; gamma < 4; gamma++) {
        w[alpha] += QPT(alpha + 1, beta, gamma) * Sign[beta] * p[beta] * q[gamma]; // w = Im(q0* q1), material_omega = 1/dt * w = q* q_dot
        DwDp(alpha, beta) += QPT(alpha + 1, beta, gamma) * Sign[beta] * q[gamma]; // regular omega = 2 q_dot q*
        DwDq(alpha, beta) += QPT(alpha + 1, gamma, beta) * Sign[gamma] * p[gamma];
        D2wDpDq[gamma](alpha, beta) = QPT(alpha + 1, beta, gamma) * Sign[beta];
      }
    }
  }
}

void QuaternionGradientHessian(const Vector4T& q_0, const Vector4T& q_1, const Vector4T& q_2, const Eigen::Matrix3d& J, Eigen::Vector4d& dLdq1, Eigen::Matrix4d& d2Ldq1dq2) {
  Eigen::Vector3d w_1, w_2;
  Eigen::Matrix<T, 3, 4> Dw1Dq0, Dw1Dq1, Dw2Dq1, Dw2Dq2;
  std::vector<Eigen::Matrix<T, 3, 4>> D2w1Dq0Dq1, D2w2Dq1Dq2;
  D2w1Dq0Dq1.resize(4);
  D2w2Dq1Dq2.resize(4);
  for (sz e = 0; e < 4; e++) {
    D2w1Dq0Dq1[e] = Eigen::Matrix<T, 3, 4>::Zero();
    D2w2Dq1Dq2[e] = Eigen::Matrix<T, 3, 4>::Zero();
  }
  OmegaDerivatives(q_0, q_1, w_1, Dw1Dq0, Dw1Dq1, D2w1Dq0Dq1); // w_1 = Im(q0* q1)
  OmegaDerivatives(q_1, q_2, w_2, Dw2Dq1, Dw2Dq2, D2w2Dq1Dq2); // w_2 = Im(q1* q2)
  dLdq1 = Dw1Dq1.transpose() * J * w_1 + Dw2Dq1.transpose() * J * w_2; // gradient = actual_gradient * dt^2 / 4
  //
  d2Ldq1dq2 = (Dw2Dq1.transpose() * J * Dw2Dq2);
  Eigen::Vector3d Jw = J * w_2;
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t gamma = 0; gamma < 3; gamma++) {
        d2Ldq1dq2(alpha, beta) += D2w2Dq1Dq2[beta](gamma, alpha) * Jw[gamma];
      }
    }
  }
}

void OmegaGradientHessian_paper(const Vector4T& q_0, const Vector4T& q_1, const Vector4T& q_2, const Eigen::Matrix3d& J, Eigen::Vector3d& g_omega, Eigen::Matrix3d& H_omega) {
  Eigen::Vector4d g_quaternion = Eigen::Vector4d::Zero();
  Eigen::Matrix4d H_quaternion = Eigen::Matrix4d::Zero();
  Eigen::Matrix<T, 4, 3> F_1 = F_matrix(q_1);
  Eigen::Matrix<T, 4, 3> F_2 = F_matrix(q_2);
  QuaternionGradientHessian(q_0, q_1, q_2, J, g_quaternion, H_quaternion);
  g_omega = F_1.transpose() * g_quaternion;
  H_omega = F_1.transpose() * H_quaternion * F_2;
}

void OmegaNewtonsMethod_paper(RigidBody& rb, const T dt) {
  sz N_Newton = 15;
  T tol = 1e-18;
  RigidBody rb_n = rb;
  Vector4T q_nm1 = ALGEBRA::Difference(rb_n.orientation, ALGEBRA::Scale(rb_n.orientation_dot, dt));
  //Vector4T q_nm1 = ALGEBRA::QuaternionMultiply(ALGEBRA::QuaternionFromVector(ALGEBRA::Scale(rb_n.omega, T(-0.5) * dt)), rb_n.orientation);
  Vector3T xi = { T(0),T(0),T(0) };
  TV res_norm;
  res_norm.resize(N_Newton);

  Vector3T omega = { T(0),T(0),T(1) };
  Vector3T angle = ALGEBRA::Scale(omega, T(dt));
  //rb.orientation = ALGEBRA::QuaternionMultiply(ALGEBRA::QuaternionFromVector(angle), rb.orientation);
  //Vector4T q_nm1 = ALGEBRA::QuaternionMultiply(ALGEBRA::QuaternionFromVector(ALGEBRA::Scale(angle, T(-1))), rb.orientation);

  // Newton's iteration
  for (sz k = 0; k < N_Newton; k++) {
    // computing Gradient function
    Eigen::Matrix<T, 3, 1> g_new = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 3> H_inv, H_new = Eigen::Matrix<T, 3, 3>::Zero();
    OmegaGradientHessian_paper(q_nm1, rb_n.orientation, rb.orientation, rb.inertia_tensor_material, g_new, H_new);
    //Eigen::Matrix<T, 4, 1> g_old = Eigen::Matrix<T, 4, 1>::Zero();
    //Eigen::Matrix<T, 4, 4> H_old = Eigen::Matrix<T, 4, 4>::Zero();
    //QuaternionGradientHessian(q_nm1, rb_n.orientation, rb.orientation, rb.inertia_tensor_material, g_old, H_old);
    res_norm[k] = g_new.norm();
    //std::cout << "k = " << k << ", 3d res_norm = " << res_norm[k] << " and 4d res_norm = " << g_old.norm() << std::endl;
    //std::cout << "k = " << k << ", 3d res_norm = " << res_norm[k] << std::endl;
    if ((g_new.norm() < tol) || (k == N_Newton - 1)) {
      /*std::cout << "Newton's Method ends at " << k << "-th iteration, with 3d residual norm = " << g_new.norm() << std::endl;
      std::cout << "And 4d residual norm = " << g_old.norm() << std::endl;*/
      //std::cout << "LA Newton's Method ends at " << k << "-th iteration, with 3d residual norm = " << std::scientific << std::setprecision(4) << g_new.norm() << std::endl;
      break;
    }

    H_inv = H_new.inverse();
    // computing update for omega
    Eigen::Matrix<T, 3, 1> delta_xi = -H_inv * g_new;
    Vector3T d_xi = { delta_xi[0],delta_xi[1],delta_xi[2] };
    xi = ALGEBRA::Sum(xi, d_xi);
    Vector4T exp_xi = { ExpOf(d_xi)[0],ExpOf(d_xi)[1], ExpOf(d_xi)[2], ExpOf(d_xi)[3] };
    rb.orientation = ALGEBRA::QuaternionMultiply(rb.orientation, exp_xi);

  }
  // Update q_dot_np1 and omega_np1 after Newton's Method
  rb.orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rb.orientation, rb_n.orientation), T(1) / dt);
  //rb.orientation_dot = ALGEBRA::QuaternionMultiply(rb_n.orientation, ALGEBRA::VectorToQuaternion(xi));
  Vector4T Omega_temp = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(rb.orientation_dot, ALGEBRA::ConjugateQuaternion(rb_n.orientation)), T(2));
  rb.omega = ALGEBRA::VectorOfQuaternion(Omega_temp);
}

void OmegaNewtonsMethod_paper_new(const T dt, RigidBody& rb) {
  sz N_Newton = 15;
  T tol = 1e-18;
  RigidBody rb_n = rb;
  Vector4T q_hat_n = { ExpOf(rb_n.omega)[0], ExpOf(rb_n.omega)[1] ,ExpOf(rb_n.omega)[2] ,ExpOf(rb_n.omega)[3] };
  Vector4T q_nm1 = ALGEBRA::QuaternionMultiply(rb_n.orientation, ALGEBRA::ConjugateQuaternion(q_hat_n));
  Vector3T xi = { T(0),T(0),T(0) };
  TV res_norm;
  res_norm.resize(N_Newton);


  // Newton's iteration
  for (sz k = 0; k < N_Newton; k++) {
    // computing Gradient function
    Eigen::Matrix<T, 3, 1> g_new = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 3> H_inv, H_new = Eigen::Matrix<T, 3, 3>::Zero();
    OmegaGradientHessian_paper(q_nm1, rb_n.orientation, rb.orientation, rb.inertia_tensor_material, g_new, H_new);
    res_norm[k] = g_new.norm();
    if ((g_new.norm() < tol) || (k == N_Newton - 1)) {
      break;
    }

    H_inv = H_new.inverse();
    // computing update for omega
    Eigen::Matrix<T, 3, 1> delta_xi = -H_inv * g_new;
    Vector3T d_xi = { delta_xi[0],delta_xi[1],delta_xi[2] };
    xi = ALGEBRA::Sum(xi, d_xi);
    Vector4T exp_xi = { ExpOf(xi)[0],ExpOf(xi)[1], ExpOf(xi)[2], ExpOf(xi)[3] };
    rb.orientation = ALGEBRA::QuaternionMultiply(rb_n.orientation, exp_xi);
  }
  // Update q_dot_np1 and omega_np1 after Newton's Method
  rb.omega = xi;
}

//void AddKEGradientHessian(const Vector4T& q_nm1, const RigidBody& rb_n, const RigidBody& rb_np1, const T dt, Eigen::Matrix<T, 6, 1>& g_new, Eigen::Matrix<T, 6, 6>& H_new) {
//  Eigen::Vector3d g_omega = Eigen::Vector3d::Zero();
//  Eigen::Matrix3d H_omega = Eigen::Matrix3d::Zero();
//  OmegaGradientHessian_paper(q_nm1, rb_n.orientation, rb_np1.orientation, rb_n.inertia_tensor_material, g_omega, H_omega);
//
//  for (size_t alpha = 0; alpha < 3; alpha++) {
//    g_new[alpha] += (rb_n.total_mass / T(4)) * (-rb_np1.x_com[alpha] + rb_n.x_com[alpha] + dt * rb_n.v_com[alpha]);
//    g_new[alpha + 3] += g_omega[alpha];
//    H_new(alpha, alpha) += -rb_n.total_mass / T(4);
//  }
//  H_new.block<3, 3>(3, 3) += H_omega;
//
//}
//
//void Add7DKEGradientHessian(const Vector4T& q_nm1, const RigidBody& rb_n, const RigidBody& rb_np1, const T dt, Eigen::Matrix<T, 7, 1>& g_new, Eigen::Matrix<T, 7, 7>& H_new) {
//  Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
//  Eigen::Matrix4d H_q = Eigen::Matrix4d::Zero();
//  QuaternionGradientHessian(q_nm1, rb_n.orientation, rb_np1.orientation, rb_n.inertia_tensor_material, g_q, H_q);
//
//  for (size_t alpha = 0; alpha < 3; alpha++) {
//    g_new[alpha] += (rb_n.total_mass / T(4)) * (-rb_np1.x_com[alpha] + rb_n.x_com[alpha] + dt * rb_n.v_com[alpha]);
//    H_new(alpha, alpha) += -rb_n.total_mass / T(4);
//  }
//  g_new.block<4, 1>(3, 0) += g_q;
//  H_new.block<4, 4>(3, 3) += H_q;
//}
//
//void NewtonsMethod(RigidBody& rb, const T dt, const T k0, const RigidBody& rb_1, const Particle& X_0, const Particle& X_1, const T gravity) {
//  sz N_Newton = 15;
//  T tol = 1e-15;
//  RigidBody rb_n = rb;
//  Vector4T q_nm1 = ALGEBRA::Difference(rb_n.orientation, ALGEBRA::Scale(rb_n.orientation_dot, dt));
//  Vector3T xi = { T(0),T(0),T(0) };
//  TV residual_norm;
//  residual_norm.resize(N_Newton);
//  
//
//  // Newton's iteration
//  for (sz k = 0; k < N_Newton; k++) {
//    // computing Gradient function
//    Eigen::Matrix<T, 6, 1> g_new = Eigen::Matrix<T, 6, 1>::Zero();
//    Eigen::Matrix<T, 6, 6> H_inv, H_new = Eigen::Matrix<T, 6, 6>::Zero();
//    AddKEGradientHessian(q_nm1, rb_n, rb, dt, g_new, H_new);
//    AddLAPCPGradientHessian(k0, rb, rb_1, X_0, X_1, rb_n.orientation, dt, g_new, H_new);
//    AddGravityGradient(rb, dt, gravity, g_new);
//    residual_norm[k] = g_new.norm();
//    //std::cout << "At " << k << "-th iteration, g_norm = " << g_new.norm() << std::endl;
//    if (residual_norm[k] < tol) {
//      std::cout << "Newton's Method ends at " << k << "-th iteration with residual norm = " << residual_norm[k] << std::endl;
//      break;
//    }
//    /*if ((k > 0) && (abs(residual_norm[k] - residual_norm[k - 1]) < tol)) {
//      std::cout << "Newton's Method ends at " << k << "-th iteration with residual norm = " << residual_norm[k] << std::endl;
//      break;
//    }*/
//    H_inv = H_new.inverse();
//    // computing update for x_com
//    Eigen::Matrix<T, 6, 1> delta_Y = -H_inv * g_new;
//    rb.x_com = ALGEBRA::Sum(rb.x_com, { delta_Y[0],delta_Y[1],delta_Y[2] });
//    // computing update for omega
//    Vector3T d_xi = { delta_Y[3],delta_Y[4],delta_Y[5] };
//    xi = ALGEBRA::Sum(xi, d_xi);
//    Vector4T exp_xi = { ExpOf(d_xi)[0],ExpOf(d_xi)[1], ExpOf(d_xi)[2], ExpOf(d_xi)[3] };
//    rb.orientation = ALGEBRA::QuaternionMultiply(rb.orientation, exp_xi);
//    if (k == N_Newton - 1) {
//      std::cout << "Newton's Method ends at last iteration with second to last residual norm = " << residual_norm[k] << std::endl;
//    }
//  }
//  // Update q_dot_np1 and omega_np1 after Newton's Method
//  rb.v_com = ALGEBRA::Scale(ALGEBRA::Difference(rb.x_com, rb_n.x_com), T(1) / dt);
//  rb.orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rb.orientation, rb_n.orientation), T(1) / dt);
//  Vector4T Omega_temp = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(rb.orientation_dot, ALGEBRA::QuaternionInverse(rb_n.orientation)), T(2));
//  rb.omega = ALGEBRA::VectorOfQuaternion(Omega_temp);
//}
//
//void ModifiedNewtonsMethod(const IV& segment_mesh, const bool& use_incident_elements_rb, const IV& incident_elements_rb, const std::vector<TVP>& constraint_segments_info, std::vector<RigidBody>& rigid_bodies,
//  const nm& current_rb, const T& dt, const T& k0, const T& gravity, const sz& N_Newton, const T& tol, const RigidBody& rb_n, RigidBody& rb) {
//  
//  Vector4T q_nm1 = ALGEBRA::Difference(rb_n.orientation, ALGEBRA::Scale(rb_n.orientation_dot, dt));
//  Vector3T xi = { T(0),T(0),T(0) };
//  TV residual_norm;
//  residual_norm.resize(N_Newton + 1);
//
//  // Newton's iteration
//  for (sz k = 0; k < N_Newton + 1; k++) {
//    // computing Gradient function
//    Eigen::Matrix<T, 6, 1> g_new = Eigen::Matrix<T, 6, 1>::Zero();
//    Eigen::Matrix<T, 6, 6> H_new = Eigen::Matrix<T, 6, 6>::Zero();
//    AddKEGradientHessian(q_nm1, rb_n, rb, dt, g_new, H_new);
//    AddGravityGradient(rb, dt, gravity, g_new);
//    // Add all point constraint potential part for the current rigid body
//    if (use_incident_elements_rb) {
//      for (sz n = 0; n < incident_elements_rb.size(); n++) {
//        std::div_t result = std::div(incident_elements_rb[n], nm(2));
//        nm segment_number = result.quot;
//        nm segment_order = result.rem;
//        nm rb_number = segment_mesh[incident_elements_rb[n]];
//        TGSLAssert(rb_number == current_rb, "RigidBodyConstraints::NewtonsMethod_MultiConstraints: rb_number and current_rb do not match.");
//        nm other_rb_number = segment_mesh[incident_elements_rb[n] + nm(1) - nm(2) * segment_order];
//        // segment_order = 0 --> incident_elements_rb[n] + nm(1) - nm(2) * segment_order = incident_elements_rb[n] + 1
//        // segment_order = 1 --> incident_elements_rb[n] + nm(1) - nm(2) * segment_order = incident_elements_rb[n] - 1
//        TVP segment_info = constraint_segments_info[segment_number];
//        if (segment_order == 0) {
//          AddLAPCPGradientHessian(k0, rb, rigid_bodies[other_rb_number], segment_info[0], segment_info[1], rb_n.orientation, dt, g_new, H_new);
//        }
//        else {
//          AddLAPCPGradientHessian(k0, rb, rigid_bodies[other_rb_number], segment_info[1], segment_info[0], rb_n.orientation, dt, g_new, H_new);
//        }
//      }
//    }
//    residual_norm[k] = g_new.norm();
//    //std::cout << "At " << k << "-th iteration, g_norm = " << g_new.norm() << std::endl;
//    //if ((residual_norm[k] < tol) || ((abs(residual_norm[k] - residual_norm[k - 1]) < tol) && (k > 0)) || (k == N_Newton)) {
//    if ((residual_norm[k] < tol) || (k == N_Newton)) {
//      /*std::cout << "For rigid body [" << current_rb << "], Newton's Method starts with res_norm = "
//        << std::scientific << std::setprecision(4) << residual_norm[0] <<
//        " and ends after " << k << " Newton iterations with res_norm = " << residual_norm[k] << std::endl;*/
//      break;
//    }
//    // solving for x_com
//    Eigen::Vector3d g_x = g_new.block<3, 1>(0, 0);
//    Eigen::Matrix3d H_x = H_new.block<3, 3>(0, 0);
//    Eigen::Vector3d dx = -H_x.inverse() * g_x;
//    Vector d_x = { dx[0],dx[1],dx[2] };
//    rb.x_com = ALGEBRA::Sum(rb.x_com, d_x);
//
//    // computing update for x_com
//    Eigen::Vector3d g_xi = g_new.block<3, 1>(3, 0);
//    Eigen::Matrix3d H_xi = H_new.block<3, 3>(3, 3);
//    Eigen::Vector3d dxi = -H_xi.inverse() * g_xi;
//    Vector d_xi = { dxi[0],dxi[1],dxi[2] };
//    xi = ALGEBRA::Sum(xi, d_xi);
//    Vector4T exp_xi = { ExpOf(d_xi)[0],ExpOf(d_xi)[1], ExpOf(d_xi)[2], ExpOf(d_xi)[3] };
//    rb.orientation = ALGEBRA::QuaternionMultiply(rb.orientation, exp_xi);
//  }
//  // Update q_dot_np1 and omega_np1 after Newton's Method
//  rb.v_com = ALGEBRA::Scale(ALGEBRA::Difference(rb.x_com, rb_n.x_com), T(1) / dt);
//  rb.orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rb.orientation, rb_n.orientation), T(1) / dt);
//  Vector4T Omega_temp = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(rb.orientation_dot, ALGEBRA::QuaternionInverse(rb_n.orientation)), T(2));
//  rb.omega = ALGEBRA::VectorOfQuaternion(Omega_temp);
//}
//
//void ModifiedNewtonsMethod_PCPMP(const IV& segment_mesh, const bool& use_incident_elements_rb, const IV& incident_elements_rb, const std::vector<TVP>& constraint_segments_info, const std::vector<RigidBody>& rigid_bodies_n,
//  std::vector<RigidBody>& rigid_bodies, const nm& current_rb, const T& a0, const T& dt, const T& k0, const T& gravity, const sz& N_Newton, const T& tol, const RigidBody& rb_n, RigidBody& rb) {
//
//  Vector4T q_nm1 = ALGEBRA::Difference(rb_n.orientation, ALGEBRA::Scale(rb_n.orientation_dot, dt));
//  Vector3T xi = { T(0),T(0),T(0) };
//  TV residual_norm;
//  residual_norm.resize(N_Newton + 1);
//
//  // Newton's iteration
//  for (sz k = 0; k < N_Newton + 1; k++) {
//    // computing Gradient function
//    Eigen::Matrix<T, 6, 1> g_new = Eigen::Matrix<T, 6, 1>::Zero();
//    Eigen::Matrix<T, 6, 6> H_new = Eigen::Matrix<T, 6, 6>::Zero();
//    AddKEGradientHessian(q_nm1, rb_n, rb, dt, g_new, H_new);
//    AddGravityGradient(rb, dt, gravity, g_new);
//    // Add all point constraint potential part for the current rigid body
//    if (use_incident_elements_rb) {
//      for (sz n = 0; n < incident_elements_rb.size(); n++) {
//        std::div_t result = std::div(incident_elements_rb[n], nm(2));
//        nm segment_number = result.quot;
//        nm segment_order = result.rem;
//        nm rb_number = segment_mesh[incident_elements_rb[n]];
//        TGSLAssert(rb_number == current_rb, "RigidBodyConstraints::NewtonsMethod_MultiConstraints: rb_number and current_rb do not match.");
//        nm other_rb_number = segment_mesh[incident_elements_rb[n] + nm(1) - nm(2) * segment_order];
//        // segment_order = 0 --> incident_elements_rb[n] + nm(1) - nm(2) * segment_order = incident_elements_rb[n] + 1
//        // segment_order = 1 --> incident_elements_rb[n] + nm(1) - nm(2) * segment_order = incident_elements_rb[n] - 1
//        TVP segment_info = constraint_segments_info[segment_number];
//        if (segment_order == 0) {
//          AddPCPMPGradientHessian(a0, k0, rb_n, rb, rigid_bodies_n[other_rb_number], rigid_bodies[other_rb_number], segment_info[0], segment_info[1], rb_n.orientation, dt, g_new, H_new);
//        }
//        else {
//          AddPCPMPGradientHessian(a0, k0, rb_n, rb, rigid_bodies_n[other_rb_number], rigid_bodies[other_rb_number], segment_info[1], segment_info[0], rb_n.orientation, dt, g_new, H_new);
//        }
//      }
//    }
//    residual_norm[k] = g_new.norm();
//    //std::cout << "At " << k << "-th iteration, g_norm = " << g_new.norm() << std::endl;
//    //if ((residual_norm[k] < tol) || ((abs(residual_norm[k] - residual_norm[k - 1]) < tol) && (k > 0)) || (k == N_Newton)) {
//    if ((residual_norm[k] < tol) || (k == N_Newton)) {
//      /*std::cout << "For rigid body [" << current_rb << "], Newton's Method starts with res_norm = " 
//        << std::scientific << std::setprecision(4) << residual_norm[0] <<
//        " and ends after " << k << " Newton iterations with res_norm = " << residual_norm[k] << std::endl;*/
//      break;
//    }
//    // solving for x_com
//    Eigen::Vector3d g_x = g_new.block<3, 1>(0, 0);
//    Eigen::Matrix3d H_x = H_new.block<3, 3>(0, 0);
//    Eigen::Vector3d dx = -H_x.inverse() * g_x;
//    Vector d_x = { dx[0],dx[1],dx[2] };
//    rb.x_com = ALGEBRA::Sum(rb.x_com, d_x);
//
//    // computing update for x_com
//    Eigen::Vector3d g_xi = g_new.block<3, 1>(3, 0);
//    Eigen::Matrix3d H_xi = H_new.block<3, 3>(3, 3);
//    Eigen::Vector3d dxi = -H_xi.inverse() * g_xi;
//    Vector d_xi = { dxi[0],dxi[1],dxi[2] };
//    xi = ALGEBRA::Sum(xi, d_xi);
//    Vector4T exp_xi = { ExpOf(d_xi)[0],ExpOf(d_xi)[1], ExpOf(d_xi)[2], ExpOf(d_xi)[3] };
//    rb.orientation = ALGEBRA::QuaternionMultiply(rb.orientation, exp_xi);
//  }
//  // Update q_dot_np1 and omega_np1 after Newton's Method
//  rb.v_com = ALGEBRA::Scale(ALGEBRA::Difference(rb.x_com, rb_n.x_com), T(1) / dt);
//  rb.orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rb.orientation, rb_n.orientation), T(1) / dt);
//  Vector4T Omega_temp = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(rb.orientation_dot, ALGEBRA::QuaternionInverse(rb_n.orientation)), T(2));
//  rb.omega = ALGEBRA::VectorOfQuaternion(Omega_temp);
//}
//
//void NewtonsMethod_PCPMP(const IV& segment_mesh, const bool& use_incident_elements_rb, const IV& incident_elements_rb, const std::vector<TVP>& constraint_segments_info, const std::vector<RigidBody>& rigid_bodies_n,
//  std::vector<RigidBody>& rigid_bodies, const nm& current_rb, const T& a0, const T& dt, const T& k0, const T& gravity, const sz& N_Newton, const T& tol, const RigidBody& rb_n, RigidBody& rb) {
//
//  Vector4T q_nm1 = ALGEBRA::Difference(rb_n.orientation, ALGEBRA::Scale(rb_n.orientation_dot, dt));
//  Vector3T xi = { T(0),T(0),T(0) };
//  TV residual_norm;
//  residual_norm.resize(N_Newton + 1);
//
//  // Newton's iteration
//  for (sz k = 0; k < N_Newton + 1; k++) {
//    // computing Gradient function
//    Eigen::Matrix<T, 6, 1> g_new = Eigen::Matrix<T, 6, 1>::Zero();
//    Eigen::Matrix<T, 6, 6> H_new = Eigen::Matrix<T, 6, 6>::Zero();
//    AddKEGradientHessian(q_nm1, rb_n, rb, dt, g_new, H_new);
//    AddGravityGradient(rb, dt, gravity, g_new);
//    // Add all point constraint potential part for the current rigid body
//    if (use_incident_elements_rb) {
//      for (sz n = 0; n < incident_elements_rb.size(); n++) {
//        std::div_t result = std::div(incident_elements_rb[n], nm(2));
//        nm segment_number = result.quot;
//        nm segment_order = result.rem;
//        nm rb_number = segment_mesh[incident_elements_rb[n]];
//        TGSLAssert(rb_number == current_rb, "RigidBodyConstraints::NewtonsMethod_MultiConstraints: rb_number and current_rb do not match.");
//        nm other_rb_number = segment_mesh[incident_elements_rb[n] + nm(1) - nm(2) * segment_order];
//        // segment_order = 0 --> incident_elements_rb[n] + nm(1) - nm(2) * segment_order = incident_elements_rb[n] + 1
//        // segment_order = 1 --> incident_elements_rb[n] + nm(1) - nm(2) * segment_order = incident_elements_rb[n] - 1
//        TVP segment_info = constraint_segments_info[segment_number];
//        if (segment_order == 0) {
//          AddPCPMPGradientHessian(a0, k0, rb_n, rb, rigid_bodies_n[other_rb_number], rigid_bodies[other_rb_number], segment_info[0], segment_info[1], rb_n.orientation, dt, g_new, H_new);
//        }
//        else {
//          AddPCPMPGradientHessian(a0, k0, rb_n, rb, rigid_bodies_n[other_rb_number], rigid_bodies[other_rb_number], segment_info[1], segment_info[0], rb_n.orientation, dt, g_new, H_new);
//        }
//      }
//    }
//    residual_norm[k] = g_new.norm();
//    //std::cout << "At " << k << "-th iteration, g_norm = " << g_new.norm() << std::endl;
//    //if ((residual_norm[k] < tol) || ((abs(residual_norm[k] - residual_norm[k - 1]) < tol) && (k > 0)) || (k == N_Newton)) {
//    if ((residual_norm[k] < tol) || (k == N_Newton)) {
//      /*std::cout << "For rigid body [" << current_rb << "], Newton's Method starts with res_norm = "
//        << std::scientific << std::setprecision(4) << residual_norm[0] <<
//        " and ends after " << k << " Newton iterations with res_norm = " << residual_norm[k] << std::endl;*/
//      break;
//    }
//
//    Eigen::Matrix<T, 6, 1> dY = -H_new.inverse() * g_new;
//
//    // solving for x_com
//    Vector d_x = { dY[0],dY[1],dY[2] };
//    rb.x_com = ALGEBRA::Sum(rb.x_com, d_x);
//
//    // computing update for x_com
//    Vector d_xi = { dY[3],dY[4],dY[5] };
//    xi = ALGEBRA::Sum(xi, d_xi);
//    Vector4T exp_xi = { ExpOf(d_xi)[0],ExpOf(d_xi)[1], ExpOf(d_xi)[2], ExpOf(d_xi)[3] };
//    rb.orientation = ALGEBRA::QuaternionMultiply(rb.orientation, exp_xi);
//  }
//  // Update q_dot_np1 and omega_np1 after Newton's Method
//  rb.v_com = ALGEBRA::Scale(ALGEBRA::Difference(rb.x_com, rb_n.x_com), T(1) / dt);
//  rb.orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rb.orientation, rb_n.orientation), T(1) / dt);
//  Vector4T Omega_temp = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(rb.orientation_dot, ALGEBRA::QuaternionInverse(rb_n.orientation)), T(2));
//  rb.omega = ALGEBRA::VectorOfQuaternion(Omega_temp);
//}

// Using general line search

// Traditional Backward Euler

T TraditionalIncrementalPotential(const T dt, const RigidBody& rb_input, const RigidBody& rb) {
  T IP = T(0);

  // linear part
  Particle x_hat = ALGEBRA::Sum(rb_input.x_com, ALGEBRA::Scale(rb_input.v_com, dt));
  IP += T(.5) * rb.total_mass * ALGEBRA::NormSquared(ALGEBRA::Difference(rb.x_com, x_hat));

  // angular part
  Vector4T q, q_conj, q_n, q_conj_n, q_dot_n, q_conj_dot_n, q_nm1, w_n;
  q = rb.orientation;
  q_conj = ALGEBRA::ConjugateQuaternion(q);
  q_n = rb_input.orientation;
  q_conj_n = ALGEBRA::ConjugateQuaternion(q_n);
  w_n = ALGEBRA::VectorToQuaternion(rb_input.omega);
  Vector3T angle = ALGEBRA::Scale(rb_input.omega, T(-1) * dt);
  q_nm1 = ALGEBRA::QuaternionMultiply(ALGEBRA::QuaternionFromVector(angle), q_n);
  q_dot_n = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(w_n, q_nm1), T(.5)); // q_dot_n = 0.5 * w_n * q_nm1 = 0.5 * w_n * exp(-w_n * dt / 2) * q_n
  q_conj_dot_n = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(q_conj_n, ALGEBRA::QuaternionMultiply(q_dot_n, q_conj_n)), T(-1));

  for (size_t beta = 0; beta < 4; beta++) {
    for (size_t sigma = 0; sigma < 4; sigma++) {
      for (size_t epsilon = 0; epsilon < 4; epsilon++) {
        for (size_t theta = 0; theta < 4; theta++) {
          IP += rb_input.InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) * q[beta] * q_conj[epsilon]
            * (T(.5) * q[sigma] * q_conj[theta]
              - q_n[sigma] * q_conj_n[theta]
              - dt * q_dot_n[sigma] * q_conj_n[theta]
              - dt * q_n[sigma] * q_conj_dot_n[theta]);
        }
      }
    }
  }

  return IP;
}

void AddTraditionalIPGradient(const T dt, const RBsInfo& rbs_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient) {
  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rbs_info.current_rb].omega, rbs_info.rigid_bodies_input[rbs_info.current_rb].orientation, dqdw);

  // linear part
  Particle x = rigid_bodies[rbs_info.current_rb].x_com;
  Particle x_n = rbs_info.rigid_bodies_input[rbs_info.current_rb].x_com;
  Vector v_n = rbs_info.rigid_bodies_input[rbs_info.current_rb].v_com;
  Particle x_hat = ALGEBRA::Sum(x_n, ALGEBRA::Scale(v_n, dt));
  for (size_t alpha = 0; alpha < 3; alpha++) {
    gradient[alpha] += rigid_bodies[rbs_info.current_rb].total_mass * (x[alpha] - x_hat[alpha]);
  }

  // compute gradient
  Vector4T q, q_conj, q_n, q_conj_n, q_dot_n, q_conj_dot_n, q_nm1, w_n;
  q = rigid_bodies[rbs_info.current_rb].orientation;
  q_conj = ALGEBRA::ConjugateQuaternion(q);
  q_n = rbs_info.rigid_bodies_input[rbs_info.current_rb].orientation;
  q_conj_n = ALGEBRA::ConjugateQuaternion(q_n);
  w_n = ALGEBRA::VectorToQuaternion(rbs_info.rigid_bodies_input[rbs_info.current_rb].omega);
  Vector3T angle = ALGEBRA::Scale(rbs_info.rigid_bodies_input[rbs_info.current_rb].omega, T(-1) * dt);
  q_nm1 = ALGEBRA::QuaternionMultiply(ALGEBRA::QuaternionFromVector(angle), q_n);
  q_dot_n = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(w_n, q_nm1), T(.5)); // q_dot_n = 0.5 * w_n * q_nm1 = 0.5 * w_n * exp(-w_n * dt / 2) * q_n
  q_conj_dot_n = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(q_conj_n, ALGEBRA::QuaternionMultiply(q_dot_n, q_conj_n)), T(-1));

  Eigen::Matrix4d Dq = Eigen::Matrix4d::Identity(), Dq_conj;
  Dq_conj = -Dq;
  Dq_conj(0, 0) = T(1); //diag(Dq_conj) = [1, -1, -1, -1]^T

  Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t sigma = 0; sigma < 4; sigma++) {
        for (size_t epsilon = 0; epsilon < 4; epsilon++) {
          for (size_t theta = 0; theta < 4; theta++) {
            g_q[alpha] += rigid_bodies[rbs_info.current_rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
              (Dq(alpha, beta) * q_conj[epsilon] + q[beta] * Dq_conj(alpha, epsilon))
              * (q[sigma] * q_conj[theta] - q_n[sigma] * q_conj_n[theta] - dt * q_dot_n[sigma] * q_conj_n[theta] - dt * q_n[sigma] * q_conj_dot_n[theta]);
          }
        }
      }
    }
  }
  gradient.block<3, 1>(3, 0) += dqdw.transpose() * g_q;
}

void AddTraditionalIPGradientHessian(const T dt, const RBsInfo& rbs_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient, Matrix6d& Hessian) {
  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rbs_info.current_rb].omega, rbs_info.rigid_bodies_input[rbs_info.current_rb].orientation, dqdw);

  // linear part
  Particle x = rigid_bodies[rbs_info.current_rb].x_com;
  Particle x_n = rbs_info.rigid_bodies_input[rbs_info.current_rb].x_com;
  Vector v_n = rbs_info.rigid_bodies_input[rbs_info.current_rb].v_com;
  Particle x_hat = ALGEBRA::Sum(x_n, ALGEBRA::Scale(v_n, dt));
  for (size_t alpha = 0; alpha < 3; alpha++) {
    gradient[alpha] += rigid_bodies[rbs_info.current_rb].total_mass * (x[alpha] - x_hat[alpha]);
    Hessian(alpha, alpha) += rigid_bodies[rbs_info.current_rb].total_mass;
  }
  
  // compute gradient
  Vector4T q, q_conj, q_n, q_conj_n, q_dot_n, q_conj_dot_n, q_nm1, w_n;
  q = rigid_bodies[rbs_info.current_rb].orientation;
  q_conj = ALGEBRA::ConjugateQuaternion(q);
  q_n = rbs_info.rigid_bodies_input[rbs_info.current_rb].orientation;
  q_conj_n = ALGEBRA::ConjugateQuaternion(q_n);
  w_n = ALGEBRA::VectorToQuaternion(rbs_info.rigid_bodies_input[rbs_info.current_rb].omega);
  Vector3T angle = ALGEBRA::Scale(rbs_info.rigid_bodies_input[rbs_info.current_rb].omega, T(-1) * dt);
  q_nm1 = ALGEBRA::QuaternionMultiply(ALGEBRA::QuaternionFromVector(angle), q_n);
  q_dot_n = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(w_n, q_nm1), T(.5)); // q_dot_n = 0.5 * w_n * q_nm1 = 0.5 * w_n * exp(-w_n * dt / 2) * q_n
  q_conj_dot_n = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(q_conj_n, ALGEBRA::QuaternionMultiply(q_dot_n, q_conj_n)), T(-1));

  Eigen::Matrix4d Dq = Eigen::Matrix4d::Identity(), Dq_conj;
  Dq_conj = -Dq;
  Dq_conj(0, 0) = T(1); //diag(Dq_conj) = [1, -1, -1, -1]^T

  Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t sigma = 0; sigma < 4; sigma++) {
        for (size_t epsilon = 0; epsilon < 4; epsilon++) {
          for (size_t theta = 0; theta < 4; theta++) {
            g_q[alpha] += rigid_bodies[rbs_info.current_rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
                 (Dq(alpha, beta) * q_conj[epsilon] + q[beta] * Dq_conj(alpha, epsilon))
               * (q[sigma] * q_conj[theta] - q_n[sigma] * q_conj_n[theta] - dt * q_dot_n[sigma] * q_conj_n[theta] - dt * q_n[sigma] * q_conj_dot_n[theta]);
          }
        }
      }
    }
  }
  gradient.block<3, 1>(3, 0) += dqdw.transpose() * g_q;
  // compute Hessian
  Eigen::Matrix4d H_q = Eigen::Matrix4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t gamma = 0; gamma < 4; gamma++) {
      for (size_t beta = 0; beta < 4; beta++) {
        for (size_t sigma = 0; sigma < 4; sigma++) {
          for (size_t epsilon = 0; epsilon < 4; epsilon++) {
            for (size_t theta = 0; theta < 4; theta++) {
              H_q(alpha, gamma) += rigid_bodies[rbs_info.current_rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta)
                * ((Dq(alpha, beta) * Dq_conj(gamma, epsilon) + Dq(gamma, beta) * Dq_conj(alpha, epsilon))
                   * (q[sigma] * q_conj[theta] - q_n[sigma] * q_conj_n[theta] - dt * q_dot_n[sigma] * q_conj_n[theta] - dt * q_n[sigma] * q_conj_dot_n[theta])
                 + (Dq(alpha, beta) * q_conj[epsilon] + q[beta] * Dq_conj(alpha, epsilon))
                   * (Dq(gamma, sigma) * q_conj[theta] + q[sigma] * Dq_conj(gamma, theta)));
            }
          }
        }
      }
    }
  }
  Hessian.block<3, 3>(3, 3) += dqdw.transpose() * H_q * dqdw;
}

// Backward Euler -- momentum conserving assumption

// Codes compatible with line search

/////////////////
// Compute rb_hat
/////////////////
void ComputeRBHat(const T dt, RBsInfo& rbs_info) {
  // update COM part
  rbs_info.rigid_bodies_hat[rbs_info.current_rb].x_com = ALGEBRA::Sum(rbs_info.rigid_bodies_hat[rbs_info.current_rb].x_com, ALGEBRA::Scale(rbs_info.rigid_bodies_hat[rbs_info.current_rb].v_com, dt));

  // update quaternion part
  OmegaNewtonsMethod_paper(rbs_info.rigid_bodies_hat[rbs_info.current_rb], dt);

}

//////////////////////////////////
// IP and its gradient and hessian
//////////////////////////////////
T NewIncrementalPotential(const sz rb, const RBsInfo& rbs_info, const std::vector<RigidBody>& rigid_bodies) {
  T ICP = T(0);

  Particle x = rigid_bodies[rb].x_com, x_hat = rbs_info.rigid_bodies_hat[rb].x_com;
  Vector4T q = rigid_bodies[rb].orientation, q_conj = ALGEBRA::ConjugateQuaternion(q);
  Vector4T q_hat = rbs_info.rigid_bodies_hat[rb].orientation, q_hat_conj = ALGEBRA::ConjugateQuaternion(q_hat);

  // linear part of "KE"
  ICP += T(.5) * rigid_bodies[rb].total_mass * ALGEBRA::NormSquared(ALGEBRA::Difference(x, x_hat));

  // angular part of "KE"
  for (size_t beta = 0; beta < 4; beta++) {
    for (size_t sigma = 0; sigma < 4; sigma++) {
      for (size_t epsilon = 0; epsilon < 4; epsilon++) {
        for (size_t theta = 0; theta < 4; theta++) {
          ICP += T(.5) * rigid_bodies[rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
            (q[beta] * q_conj[epsilon] * q[sigma] * q_conj[theta]
              - T(2) * q[beta] * q_conj[epsilon] * q_hat[sigma] * q_hat_conj[theta]
              + q_hat[beta] * q_hat_conj[epsilon] * q_hat[sigma] * q_hat_conj[theta]);
        }
      }
    }
  }
  return ICP;
}

void AddNewIncrementalGradient(const sz rb, const T dt, const RBsInfo& rbs_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient) {
  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb].omega, rbs_info.rigid_bodies_input[rb].orientation, dqdw);

  Particle x = rigid_bodies[rb].x_com, x_hat = rbs_info.rigid_bodies_hat[rb].x_com;
  Vector4T q = rigid_bodies[rb].orientation, q_hat = rbs_info.rigid_bodies_hat[rb].orientation;
  Vector4T q_conj = ALGEBRA::ConjugateQuaternion(q), q_hat_conj = ALGEBRA::ConjugateQuaternion(q_hat);

  Vector4T sgn = { T(1),T(-1),T(-1),T(-1) };
  Eigen::Matrix4d I_4 = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d sign = I_4;
  sign(1, 1) = T(-1);
  sign(2, 2) = T(-1);
  sign(3, 3) = T(-1);

  // gradient for x_com
  for (size_t alpha = 0; alpha < 3; alpha++) {
    gradient[alpha] += rigid_bodies[rb].total_mass * (x[alpha] - x_hat[alpha]);
  }

  // gradient for quaternion
  Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t sigma = 0; sigma < 4; sigma++) {
        for (size_t epsilon = 0; epsilon < 4; epsilon++) {
          for (size_t theta = 0; theta < 4; theta++) {
            g_q[alpha] += rigid_bodies[rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
              (I_4(alpha, beta) * q_conj[epsilon] + q[beta] * sign(alpha, epsilon)) * (q[sigma] * q_conj[theta] - q_hat[sigma] * q_hat_conj[theta]);
          }
        }
      }
    }
  }
  
  gradient.block<3, 1>(3, 0) += dqdw.transpose() * g_q;
}

void AddNewIncremental7DGradient(const sz rb, const T dt, const RBsInfo& rbs_info, const std::vector<RigidBody>& rigid_bodies, Eigen::Matrix<T, 7, 1>& gradient) {
  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb].omega, rbs_info.rigid_bodies_input[rb].orientation, dqdw);

  Particle x = rigid_bodies[rb].x_com, x_hat = rbs_info.rigid_bodies_hat[rb].x_com;
  Vector4T q = rigid_bodies[rb].orientation, q_hat = rbs_info.rigid_bodies_hat[rb].orientation;
  Vector4T q_conj = ALGEBRA::ConjugateQuaternion(q), q_hat_conj = ALGEBRA::ConjugateQuaternion(q_hat);

  Vector4T sgn = { T(1),T(-1),T(-1),T(-1) };
  Eigen::Matrix4d I_4 = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d sign = I_4;
  sign(1, 1) = T(-1);
  sign(2, 2) = T(-1);
  sign(3, 3) = T(-1);

  // gradient for x_com
  for (size_t alpha = 0; alpha < 3; alpha++) {
    gradient[alpha] += rigid_bodies[rb].total_mass * (x[alpha] - x_hat[alpha]);
  }

  // gradient for quaternion
  Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t sigma = 0; sigma < 4; sigma++) {
        for (size_t epsilon = 0; epsilon < 4; epsilon++) {
          for (size_t theta = 0; theta < 4; theta++) {
            gradient[alpha + 3] += rigid_bodies[rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
              (I_4(alpha, beta) * q_conj[epsilon] + q[beta] * sign(alpha, epsilon)) * (q[sigma] * q_conj[theta] - q_hat[sigma] * q_hat_conj[theta]);
          }
        }
      }
    }
  }

}

void AddNewIncrementalOmegaGradient(const sz rb, const T dt, const RBsInfo& rbs_info, const std::vector<RigidBody>& rigid_bodies, Vector3d& gradient) {
  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb].omega, rbs_info.rigid_bodies_input[rb].orientation, dqdw);

  Vector4T q = rigid_bodies[rb].orientation, q_hat = rbs_info.rigid_bodies_hat[rb].orientation;
  Vector4T q_conj = ALGEBRA::ConjugateQuaternion(q), q_hat_conj = ALGEBRA::ConjugateQuaternion(q_hat);

  Vector4T sgn = { T(1),T(-1),T(-1),T(-1) };
  Eigen::Matrix4d I_4 = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d sign = I_4;
  sign(1, 1) = T(-1);
  sign(2, 2) = T(-1);
  sign(3, 3) = T(-1);

  // gradient for quaternion
  Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t sigma = 0; sigma < 4; sigma++) {
        for (size_t epsilon = 0; epsilon < 4; epsilon++) {
          for (size_t theta = 0; theta < 4; theta++) {
            g_q[alpha] += rigid_bodies[rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
              (I_4(alpha, beta) * q_conj[epsilon] + q[beta] * sign(alpha, epsilon)) * (q[sigma] * q_conj[theta] - q_hat[sigma] * q_hat_conj[theta]);
          }
        }
      }
    }
  }
  gradient += dqdw.transpose() * g_q;
}

void AddNewIncrementalGradientHessian(const sz rb, const T dt, const RBsInfo& rbs_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient, Matrix6d& Hessian) {
  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb].omega, rbs_info.rigid_bodies_input[rb].orientation, dqdw);

  Particle x = rigid_bodies[rb].x_com, x_hat = rbs_info.rigid_bodies_hat[rb].x_com;

  Vector4T q = rigid_bodies[rb].orientation, q_hat = rbs_info.rigid_bodies_hat[rb].orientation;
  Vector4T q_conj = ALGEBRA::ConjugateQuaternion(q), q_hat_conj = ALGEBRA::ConjugateQuaternion(q_hat);

  Vector4T sgn = { T(1),T(-1),T(-1),T(-1) };
  Eigen::Matrix4d I_4 = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d sign = I_4;
  sign(1, 1) = T(-1);
  sign(2, 2) = T(-1);
  sign(3, 3) = T(-1);

  // gradient for x_com
  for (size_t alpha = 0; alpha < 3; alpha++) {
    gradient[alpha] += rigid_bodies[rb].total_mass * (x[alpha] - x_hat[alpha]);
    Hessian(alpha, alpha) += rigid_bodies[rb].total_mass;
  }

  // gradient for quaternion
  Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t sigma = 0; sigma < 4; sigma++) {
        for (size_t epsilon = 0; epsilon < 4; epsilon++) {
          for (size_t theta = 0; theta < 4; theta++) {
            g_q[alpha] += rigid_bodies[rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
              (I_4(alpha, beta) * q_conj[epsilon] + q[beta] * sign(alpha, epsilon)) * (q[sigma] * q_conj[theta] - q_hat[sigma] * q_hat_conj[theta]);
          }
        }
      }
    }
  }
  gradient.block<3, 1>(3, 0) += dqdw.transpose() * g_q;

  // hessian for quaternion
  Eigen::Matrix4d H_q = Eigen::Matrix4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t gamma = 0; gamma < 4; gamma++) {
      for (size_t beta = 0; beta < 4; beta++) {
        for (size_t sigma = 0; sigma < 4; sigma++) {
          for (size_t epsilon = 0; epsilon < 4; epsilon++) {
            for (size_t theta = 0; theta < 4; theta++) {
              H_q(alpha, gamma) += rigid_bodies[rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
                ((I_4(alpha, beta) * sign(epsilon, gamma) + I_4(beta, gamma) * sign(alpha, epsilon)) *
                  (q[sigma] * q_conj[theta] - q_hat[sigma] * q_hat_conj[theta]) +
                  (I_4(alpha, beta) * q_conj[epsilon] + q[beta] * sign(alpha, epsilon)) *
                  (I_4(sigma, gamma) * q_conj[theta] + q[sigma] * sign(theta, gamma)));
            }
          }
        }
      }
    }
  }
  Hessian.block<3, 3>(3, 3) += dqdw.transpose() * H_q * dqdw;
}

void AddNewIncrementalOmegaGradientHessian(const sz rb, const T dt, const RBsInfo& rbs_info, const std::vector<RigidBody>& rigid_bodies, Vector3d& gradient, Matrix3d& Hessian) {
  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb].omega, rbs_info.rigid_bodies_input[rb].orientation, dqdw);

  Vector4T q = rigid_bodies[rb].orientation, q_hat = rbs_info.rigid_bodies_hat[rb].orientation;
  Vector4T q_conj = ALGEBRA::ConjugateQuaternion(q), q_hat_conj = ALGEBRA::ConjugateQuaternion(q_hat);

  Vector4T sgn = { T(1),T(-1),T(-1),T(-1) };
  Eigen::Matrix4d I_4 = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d sign = I_4;
  sign(1, 1) = T(-1);
  sign(2, 2) = T(-1);
  sign(3, 3) = T(-1);

  // gradient for quaternion
  Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t beta = 0; beta < 4; beta++) {
      for (size_t sigma = 0; sigma < 4; sigma++) {
        for (size_t epsilon = 0; epsilon < 4; epsilon++) {
          for (size_t theta = 0; theta < 4; theta++) {
            g_q[alpha] += rigid_bodies[rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
              (I_4(alpha, beta) * q_conj[epsilon] + q[beta] * sign(alpha, epsilon)) * (q[sigma] * q_conj[theta] - q_hat[sigma] * q_hat_conj[theta]);
          }
        }
      }
    }
  }
  gradient += dqdw.transpose() * g_q;

  // hessian for quaternion
  Eigen::Matrix4d H_q = Eigen::Matrix4d::Zero();
  for (size_t alpha = 0; alpha < 4; alpha++) {
    for (size_t gamma = 0; gamma < 4; gamma++) {
      for (size_t beta = 0; beta < 4; beta++) {
        for (size_t sigma = 0; sigma < 4; sigma++) {
          for (size_t epsilon = 0; epsilon < 4; epsilon++) {
            for (size_t theta = 0; theta < 4; theta++) {
              H_q(alpha, gamma) += rigid_bodies[rb].InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) *
                ((I_4(alpha, beta) * sign(epsilon, gamma) + I_4(beta, gamma) * sign(alpha, epsilon)) *
                  (q[sigma] * q_conj[theta] - q_hat[sigma] * q_hat_conj[theta]) +
                  (I_4(alpha, beta) * q_conj[epsilon] + q[beta] * sign(alpha, epsilon)) *
                  (I_4(sigma, gamma) * q_conj[theta] + q[sigma] * sign(theta, gamma)));
            }
          }
        }
      }
    }
  }
  Hessian += dqdw.transpose() * H_q * dqdw;
}

////////////
// gravity
////////////
T GravitationalPotential(const sz rb, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  return rigid_bodies[rb].total_mass * rbc_info.gravity * rigid_bodies[rb].x_com[1];
}

void GravitationalGradient(const sz rb, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Eigen::Matrix<T, 6, 1>& gradient) {
  gradient.setZero();
  gradient[1] = rigid_bodies[rb].total_mass * rbc_info.gravity;
}

////////////////////////////////////
// PCP and its gradient and hessian
////////////////////////////////////

T ConstraintBasedPointConstraintPotential(const sz constraint_segment_number, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  sz rb0 = rbc_info.segment_mesh[2 * constraint_segment_number];
  sz rb1 = rbc_info.segment_mesh[2 * constraint_segment_number + 1];
  Particle X_0 = rbc_info.constraint_segments_info[constraint_segment_number][0];
  Particle X_1 = rbc_info.constraint_segments_info[constraint_segment_number][1];
  Vector distance = ALGEBRA::Difference(rigid_bodies[rb1].WorldSpacePosition(X_1), rigid_bodies[rb0].WorldSpacePosition(X_0));

  return T(.5) * rbc_info.pcp_stiffness * ALGEBRA::NormSquared(distance);
}

T RigidBodyBasedPointConstraintPotential(const sz rb, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T potential = T(0);
  nm order = rbc_info.constrained_rigid_body_count[rb];
  if (order != nm(-1)) {
    for (sz count = 0; count < rbc_info.incident_elements_info[order].size(); ++count) {
      sz c = rbc_info.incident_elements_info[order][count][0];
      potential += ConstraintBasedPointConstraintPotential(c, rbc_info, rigid_bodies);
    }
  }
  return potential;
}

// PCP gradient

void ConstraintBasedPointConstraintGradient(const sz constraint_segment_number, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, std::vector<Eigen::Matrix<T, 6, 1>>& gradients) {
#ifndef TWO_D
  sz rb0 = rbc_info.segment_mesh[2 * constraint_segment_number];
  sz rb1 = rbc_info.segment_mesh[2 * constraint_segment_number + 1];
  Vector2I rb = { rb0,rb1 };
  Particle X_0 = rbc_info.constraint_segments_info[constraint_segment_number][0];
  Particle X_1 = rbc_info.constraint_segments_info[constraint_segment_number][1];
  Vector distance = ALGEBRA::Difference(rigid_bodies[rb1].WorldSpacePosition(X_1), rigid_bodies[rb0].WorldSpacePosition(X_0));
  Vector4T sign = { T(1),T(-1),T(-1),T(-1) };
#pragma omp parallel for
  for (sz i = 0; i < 2; ++i) {
    gradients[i].setZero();
    T sgn = std::pow(T(-1), T(i + 1));

    // g_x
    for (sz alpha = 0; alpha < 3; ++alpha) {
      gradients[i][alpha] = sgn * rbc_info.pcp_stiffness * distance[alpha];
    }

    // g_w
    Eigen::Matrix<T, 3, 4> DdDq = Eigen::Matrix<T, 3, 4>::Zero();
    for (size_t alpha = 0; alpha < 3; alpha++) {
      for (size_t beta = 0; beta < 4; beta++) {
        for (size_t epsilon = 0; epsilon < 3; epsilon++) {
          for (size_t sigma = 0; sigma < 4; sigma++) {
            T QPT_scalar_0 = QPT_QPT(alpha + 1, beta, epsilon + 1, sigma);
            T QPT_scalar_1 = QPT_QPT(alpha + 1, sigma, epsilon + 1, beta);
            T scalar = QPT_scalar_0 * sign[sigma] + QPT_scalar_1 * sign[beta];
            DdDq(alpha, beta) += sgn * scalar * rbc_info.constraint_segments_info[constraint_segment_number][i][epsilon] * rigid_bodies[rb[i]].orientation[sigma];
          }
        }
      }
    }

    Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
    for (size_t alpha = 0; alpha < 4; alpha++) {
      for (size_t beta = 0; beta < 3; beta++) {
        g_q[alpha] += rbc_info.pcp_stiffness * distance[beta] * DdDq(beta, alpha);
      }
    }

    Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
    TGSL::DqDw(dt, rigid_bodies[rb[i]].omega, rbs_info.rigid_bodies_input[rb[i]].orientation, dqdw);

    gradients[i].block<3, 1>(3, 0) = dqdw.transpose() * g_q;
}
#else
  TGSLAssert(false, "RigidBodyConstraints::ConstraintBasedPointConstraintGradient: not defined for 2d.");
#endif // !TWO_D
}

void RigidBodyBasedPointConstraintGradient(const sz rb, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Eigen::Matrix<T, 6, 1>& gradient) {
  gradient.setZero();
  nm order = rbc_info.constrained_rigid_body_count[rb];
  if (order != nm(-1)) {
    for (sz count = 0; count < rbc_info.incident_elements_info[order].size(); ++count) {
      std::vector<Eigen::Matrix<T, 6, 1>> gradients(2);
      sz c = rbc_info.incident_elements_info[order][count][0];
      sz segment_pair_order = rbc_info.incident_elements_info[order][count][1];
      
      ConstraintBasedPointConstraintGradient(c, dt, rbs_info, rbc_info, rigid_bodies, gradients);
      gradient += gradients[segment_pair_order];
    }
  }
}

// PCP gradient and hessian
void ConstraintBasedPointConstraintGradientHessian(const sz constraint_segment_number, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, std::vector<Eigen::Matrix<T, 6, 1>>& gradients, std::vector<Eigen::Matrix<T, 6, 6>>& Hessians) {
  sz rb0 = rbc_info.segment_mesh[2 * constraint_segment_number];
  sz rb1 = rbc_info.segment_mesh[2 * constraint_segment_number + 1];
  Vector2I rb = { rb0,rb1 };
  Particle X_0 = rbc_info.constraint_segments_info[constraint_segment_number][0];
  Particle X_1 = rbc_info.constraint_segments_info[constraint_segment_number][1];
  Vector distance = ALGEBRA::Difference(rigid_bodies[rb1].WorldSpacePosition(X_1), rigid_bodies[rb0].WorldSpacePosition(X_0));
  //PrintVector("x_1", rigid_bodies[rb1].WorldSpacePosition(X_1));
  //PrintVector("x_0", rigid_bodies[rb0].WorldSpacePosition(X_0));
  Vector4T sign = { T(1),T(-1),T(-1),T(-1) };
  
  #pragma omp parallel for
  for (sz i = 0; i < 2; ++i) {
    gradients[i].setZero();
    Hessians[i].setZero();
    T sgn = std::pow(T(-1), T(i + 1));

    // g_x
    for (sz alpha = 0; alpha < 3; ++alpha) {
      gradients[i][alpha] = sgn * rbc_info.pcp_stiffness * distance[alpha];
      Hessians[i](alpha, alpha) = rbc_info.pcp_stiffness;
    }

    // g_w
    Eigen::Matrix<T, 3, 4> DdDq = Eigen::Matrix<T, 3, 4>::Zero();
    Eigen::Matrix4d D = Eigen::Matrix4d::Zero();
    for (size_t alpha = 0; alpha < 3; alpha++) {
      for (size_t beta = 0; beta < 4; beta++) {
        for (size_t epsilon = 0; epsilon < 3; epsilon++) {
          for (size_t sigma = 0; sigma < 4; sigma++) {
            T QPT_scalar_0 = QPT_QPT(alpha + 1, beta, epsilon + 1, sigma);
            T QPT_scalar_1 = QPT_QPT(alpha + 1, sigma, epsilon + 1, beta);
            T scalar = QPT_scalar_0 * sign[sigma] + QPT_scalar_1 * sign[beta];
            DdDq(alpha, beta) += sgn * scalar * rbc_info.constraint_segments_info[constraint_segment_number][i][epsilon] * rigid_bodies[rb[i]].orientation[sigma];
            D(sigma, beta) += sgn * scalar * rbc_info.constraint_segments_info[constraint_segment_number][i][epsilon] * distance[alpha];
          }
        }
      }
    }

    Eigen::Vector4d g_q = Eigen::Vector4d::Zero();
    for (size_t alpha = 0; alpha < 4; alpha++) {
      for (size_t beta = 0; beta < 3; beta++) {
        g_q[alpha] += rbc_info.pcp_stiffness * distance[beta] * DdDq(beta, alpha);
      }
    }

    Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
    TGSL::DqDw(dt, rigid_bodies[rb[i]].omega, rbs_info.rigid_bodies_input[rb[i]].orientation, dqdw);

    gradients[i].block<3, 1>(3, 0) = dqdw.transpose() * g_q;
    Hessians[i].block<3, 3>(0, 3) = sgn * rbc_info.pcp_stiffness * DdDq * dqdw;
    Hessians[i].block<3, 3>(3, 0) = Hessians[i].block<3, 3>(0, 3).transpose();
    Hessians[i].block<3, 3>(3, 3) = rbc_info.pcp_stiffness * dqdw.transpose() * (DdDq.transpose() * DdDq + D) * dqdw;
  }
}

void RigidBodyBasedPointConstraintGradientHessian(const sz rb, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Eigen::Matrix<T, 6, 1>& gradient, Eigen::Matrix<T, 6, 6>& Hessian) {
  gradient.setZero();
  Hessian.setZero();
  nm order = rbc_info.constrained_rigid_body_count[rb];
  if (order != nm(-1)) {
    for (sz count = 0; count < rbc_info.incident_elements_info[order].size(); ++count) {
      std::vector<Eigen::Matrix<T, 6, 1>> gradients(2);
      std::vector<Eigen::Matrix<T, 6, 6>> Hessians(2);

      sz c = rbc_info.incident_elements_info[order][count][0];
      sz segment_pair_order = rbc_info.incident_elements_info[order][count][1];

      ConstraintBasedPointConstraintGradientHessian(c, dt, rbs_info, rbc_info, rigid_bodies, gradients, Hessians);
      gradient += gradients[segment_pair_order];
      Hessian += Hessians[segment_pair_order];
    }
  }
}

///////////////////////////////////////
// IPC and its gradient and Hessian (cube - ceiling)
//////////////////////////////////////
T QuaternionBasedCubeCeilingIPC2DTemp(const sz rb0, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  // Assuming
  // -----***-----*------
  //       |  rb0 |
  //       |      |
  //       *------*
  // rb0 is fixed to the ceiling on its left, and IPC is only dependent on the relative angle between rb0 and ceiling

  // theta_min = 0
  // theta_max = pi/2

  // dist_hat = epsilon
  // dist = defined through q0
  
  T left_bound = T(1) - rigid_bodies[rb0].orientation[0];
  T right_bound = rigid_bodies[rb0].orientation[0] - std::sqrt(T(2)) / T(2);
  T dist;
  if (left_bound <= right_bound) {
    dist = left_bound;
  }
  else
  {
    dist = right_bound;
  }
  return IPCBarrierPotential(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist);
}

void QuaternionBasedCubeCeilingIPCGradient2DTemp(const sz rb0, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient) {
  // Assuming
  // -----***-----*------
  //       |  rb0 |
  //       |      |
  //       *------*
  // rb0 is fixed to the ceiling on its left, and IPC is only dependent on the relative angle between rb0 and ceiling

  // theta_min = 0
  // theta_max = pi/2

  // dist_hat = epsilon
  // dist = defined through q0
  gradient.setZero();

  T left_bound = T(1) - rigid_bodies[rb0].orientation[0];
  T right_bound = rigid_bodies[rb0].orientation[0] - std::sqrt(T(2)) / T(2);
  //std::cout << rigid_bodies[rb0].orientation[0] << ", " << left_bound << ", " << right_bound << std::endl;
  //TGSLAssert(((left_bound >= T(0)) || (right_bound >= T(0))), "RigidBodyConstraints::QuaternionBasedIPC2DTemp: the left and right bounds of quaternion are out of bound.");

  // compute distance and sign
  T dist, sgn;
  if (left_bound <= right_bound) {
    dist = left_bound;
    sgn = T(-1);
  }
  else
  {
    dist = right_bound;
    sgn = T(1);
  }
  Vector4d g_q = Vector4d::Zero();
  g_q[0] = sgn * IPCBarrierFirstDerivative(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist);

  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb0].omega, rbs_info.rigid_bodies_input[rb0].orientation, dqdw);

  gradient.block<3, 1>(3, 0) = dqdw.transpose() * g_q;
}

void QuaternionBasedCubeCeilingIPCGradientHessian2DTemp(const sz rb0, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient, Matrix6d& hessian) {
  // Assuming
  // -----***-----*------
  //       |  rb0 |
  //       |      |
  //       *------*
  // rb0 is fixed to the ceiling on its left, and IPC is only dependent on the relative angle between rb0 and ceiling

  // theta_min = 0
  // theta_max = pi/2

  // dist_hat = epsilon
  // dist = defined through q0
  gradient.setZero();
  hessian.setZero();

  T left_bound = T(1) - rigid_bodies[rb0].orientation[0];
  T right_bound = rigid_bodies[rb0].orientation[0] - std::sqrt(T(2)) / T(2);
  //std::cout << rigid_bodies[rb0].orientation[0] << ", " << left_bound << ", " << right_bound << std::endl;
  //TGSLAssert(((left_bound >= T(0)) && (right_bound >= T(0))), "RigidBodyConstraints::QuaternionBasedCubeCeilingIPCGradientHessian2DTemp: the left and right bounds of quaternion are out of bound.");

  // compute distance and sign
  T dist, sgn;
  if (left_bound <= right_bound) {
    dist = left_bound;
    sgn = T(-1);
  }
  else
  {
    dist = right_bound;
    sgn = T(1);
  }

  Vector4d g_q = Vector4d::Zero();
  Matrix4d h_q = Matrix4d::Zero();
  g_q[0] = sgn * IPCBarrierFirstDerivative(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist);
  h_q(0, 0) = IPCBarrierSecondDerivative(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist);

  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb0].omega, rbs_info.rigid_bodies_input[rb0].orientation, dqdw);

  gradient.block<3, 1>(3, 0) = dqdw.transpose() * g_q;
  hessian.block<3, 3>(3, 3) = dqdw.transpose() * h_q * dqdw;
}

///////////////////////////////////////
// IPC and its gradient and Hessian (cube - cube)
//////////////////////////////////////
T QuaternionBasedCubeCubeIPC2DTemp(const sz rb0, const sz rb1, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  // Assuming
  //        *------*
  //        |  rb0 |
  //        |      |
  // *------*------*
  // |  rb1 |
  // |      |
  // *------*
  // rb1 is fixed to the rb0 on its upper right corner, and IPC is only dependent on the relative angle between rb1 and rb0

  // theta_min = -pi/2
  // theta_max = pi/2

  // dist_hat = epsilon
  // dist = defined through q

  Vector4T q = ALGEBRA::QuaternionMultiply(rigid_bodies[rb0].orientation, ALGEBRA::ConjugateQuaternion(rigid_bodies[rb1].orientation));
  T dist = q[0] - std::sqrt(T(2)) / T(2);
  return IPCBarrierPotential(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist);
}

void QuaternionBasedCubeCubeIPCGradient2DTemp(const sz rb1, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient) {
  // Assuming
  //        *------*
  //        |  rb0 |
  //        |      |
  // *------*------*
  // |  rb1 |
  // |      |
  // *------*
  // rb1 is fixed to the rb0 on its upper right corner, and IPC is only dependent on the relative angle between rb1 and rb0

  // theta_min = -pi/2
  // theta_max = pi/2

  // dist_hat = epsilon
  // dist = defined through q
  gradient.setZero();

  Vector4T q = ALGEBRA::QuaternionMultiply(rigid_bodies[rbc_info.ipc_type[rb1]].orientation, ALGEBRA::ConjugateQuaternion(rigid_bodies[rb1].orientation));
  T dist = q[0] - std::sqrt(T(2)) / T(2);

  // TODO: uncomment this later please!
  TGSLAssert((dist >= T(0)), "RigidBodyConstraints::QuaternionBasedCubeCubeIPC2DTemp: the dist of quaternion are out of bound.");

  Vector4d g_q = Vector4d::Zero();
  Matrix4d D = ALGEBRA::QuaternionInverseDerivative(rigid_bodies[rb1].orientation);
  for (size_t alpha = 0; alpha < 4; ++alpha) {
    for (size_t beta = 0; beta < 4; ++beta) {
      for (size_t gamma = 0; gamma < 4; ++gamma) {
        g_q[alpha] += QPT(0, beta, gamma) * rigid_bodies[rbc_info.ipc_type[rb1]].orientation[beta] * D(gamma, alpha);
      }
    }
  }
  g_q *= IPCBarrierFirstDerivative(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist);

  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb1].omega, rbs_info.rigid_bodies_input[rb1].orientation, dqdw);

  gradient.block<3, 1>(3, 0) = dqdw.transpose() * g_q;
}

void QuaternionBasedCubeCubeIPCGradientHessian2DTemp(const sz rb1, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient, Matrix6d& hessian) {
  // Assuming
  //        *------*
  //        |  rb0 |
  //        |      |
  // *------*------*
  // |  rb1 |
  // |      |
  // *------*
  // rb1 is fixed to the rb0 on its upper right corner, and IPC is only dependent on the relative angle between rb1 and rb0

  // theta_min = -pi/2
  // theta_max = pi/2

  // dist_hat = epsilon
  // dist = defined through q
  gradient.setZero();
  hessian.setZero();

  Vector4T q = ALGEBRA::QuaternionMultiply(rigid_bodies[rbc_info.ipc_type[rb1]].orientation, ALGEBRA::ConjugateQuaternion(rigid_bodies[rb1].orientation));
  T dist = q[0] - std::sqrt(T(2)) / T(2);
  // TODO: uncomment this later
  //std::cout << "theta_0 = " << ALGEBRA::AngleFrom2DQuaternion(rigid_bodies[rbc_info.ipc_type[rb1]].orientation) << std::endl;
  //std::cout << "theta_1 = " << ALGEBRA::AngleFrom2DQuaternion(rigid_bodies[rb1].orientation) << std::endl;
  //std::cout << "theta = " << ALGEBRA::AngleFrom2DQuaternion(q) << std::endl;
  //std::cout << "dist = " << dist << std::endl;
  //TGSLAssert((dist >= T(0)), "RigidBodyConstraints::QuaternionBasedCubeCubeIPCGradientHessian2DTemp: the dist of quaternion are out of bound.");

  Vector4d g_q = Vector4d::Zero();
  Matrix4d h_q = Matrix4d::Zero();
  Matrix4d D = ALGEBRA::QuaternionInverseDerivative(rigid_bodies[rb1].orientation);
  std::vector<Matrix4d> D2_q = ALGEBRA::QuaternionInverseSecondDerivative(rigid_bodies[rb1].orientation);
  for (size_t alpha = 0; alpha < 4; ++alpha) {
    for (size_t beta = 0; beta < 4; ++beta) {
      for (size_t gamma = 0; gamma < 4; ++gamma) {
        g_q[alpha] += QPT(0, beta, gamma) * rigid_bodies[rbc_info.ipc_type[rb1]].orientation[beta] * D(gamma, alpha);
        for (size_t sigma = 0; sigma < 4; ++sigma) {
          h_q(alpha, beta) += QPT(0, gamma, sigma) * rigid_bodies[rbc_info.ipc_type[rb1]].orientation[gamma] * D2_q[beta](sigma, alpha);
        }
      }
    }
  }
  h_q *= IPCBarrierFirstDerivative(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist);
  h_q += IPCBarrierSecondDerivative(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist) * g_q * g_q.transpose();
  g_q *= IPCBarrierFirstDerivative(rbc_info.ipc_stiffness, rbc_info.ipc_d_hat, dist);

  Eigen::Matrix<T, 4, 3> dqdw = Eigen::Matrix<T, 4, 3>::Zero();
  TGSL::DqDw(dt, rigid_bodies[rb1].omega, rbs_info.rigid_bodies_input[rb1].orientation, dqdw);

  gradient.block<3, 1>(3, 0) = dqdw.transpose() * g_q;
  hessian.block<3, 3>(3, 3) = dqdw.transpose() * h_q * dqdw;
}

///////////////////////////////////////////////////////////////
// All angle constraint potential and its gradient and hessian
///////////////////////////////////////////////////////////////
T ConstraintBasedAngleConstraintPotential(const sz constraint_segment_number, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T energy = T(0);
  sz rb0 = rbc_info.segment_mesh[2 * constraint_segment_number];
  sz rb1 = rbc_info.segment_mesh[2 * constraint_segment_number + 1];
  if (rigid_bodies[rb0].infinite_mass) {
    energy = QuaternionBasedCubeCeilingIPC2DTemp(rb1, rbc_info, rigid_bodies);
  }
  else if (rigid_bodies[rb1].infinite_mass) {
    energy = QuaternionBasedCubeCeilingIPC2DTemp(rb0, rbc_info, rigid_bodies);
  }
  else {
    energy = QuaternionBasedCubeCubeIPC2DTemp(rb0, rb1, rbc_info, rigid_bodies);
  }
  return energy;
}

T RigidBodyBasedAngleConstraintPotential(const sz rb, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T potential = T(0);
  nm order = rbc_info.constrained_rigid_body_count[rb];
  if (order != nm(-1)) {
    for (sz count = 0; count < rbc_info.incident_elements_info[order].size(); ++count) {
      sz c = rbc_info.incident_elements_info[order][count][0];
      potential += ConstraintBasedAngleConstraintPotential(c, rbc_info, rigid_bodies);
    }
  }
  return potential;
}

// ACP gradient
void ConstraintBasedAngleConstraintGradient(const sz constraint_segment_number, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, std::vector<Eigen::Matrix<T, 6, 1>>& gradients) {
  sz rb0 = rbc_info.segment_mesh[2 * constraint_segment_number];
  sz rb1 = rbc_info.segment_mesh[2 * constraint_segment_number + 1];
}

///////////////////////////////////////
// Non-Inertial Frame Acceleration Gradient
//////////////////////////////////////////////
void NIFAccelerationGradient(const sz rb, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& g) {
  g.setZero();
  //if (rbc_info.constraint_accelerations[rbc_info.constrained_rigid_body_count[rb]].size() > 1) {
  //  TGSLAssert(false, "RigidBodyConstraints::NIFAccelerationGradient: not set up for multiple constraints per rigid body yet.");
  //}
  for (sz i = 0; i < d; ++i) {
    g[i] = rigid_bodies[rb].total_mass * rbc_info.constraint_accelerations[rbc_info.constrained_rigid_body_count[rb]][0][i];
  }
}

//////////////////////////////////////////
// rigid body based energy for one rigid body
//////////////////////////////////////////
T RigidBodyBasedEnergy(const sz rb, const T dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T energy = T(0);
  energy += NewIncrementalPotential(rb, rbs_info, rigid_bodies);
  energy += dt * dt * rigid_bodies[rbs_info.current_rb].total_mass * rbc_info.gravity * rigid_bodies[rbs_info.current_rb].x_com[1];
  return energy;
}

T ConstraintBasedEnergy(const sz c, const T dt, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T energy = T(0);
  for (sz c = 0; c < rbc_info.constraint_segments_info.size(); ++c) {
    energy += dt * dt * ConstraintBasedPointConstraintPotential(c, rbc_info, rigid_bodies);
    energy += dt * dt * ConstraintBasedAngleConstraintPotential(c, rbc_info, rigid_bodies);
  }
  return energy;
}

void RigidBodyBasedGradient(const sz rb, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Vector6d& gradient) {
  // Add "KE"-like gradient and Hessian
  AddNewIncrementalGradient(rb, dt, rbs_info, rigid_bodies, gradient);

  // Add gravity gradient
  Vector6d g_temp;
  GravitationalGradient(rb, rbc_info, rigid_bodies, g_temp);
  gradient += dt * dt * g_temp;
}

void ConstraintBasedGradient(const sz c, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, std::vector<Vector6d>& gradients) {
  // Add PCP gradient and Hessian for all constraints
  if (!rbc_info.segment_mesh.empty()) {
    //g_temp.setZero();
    //RigidBodyBasedPointConstraintGradient(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
    //gradient += dt * dt * g_temp;
  }

  // Add IPC gradient
  //if (rbc_info.use_ipc) {
  //  g_temp.setZero();
  //  if (rbc_info.ipc_type[rb] == nm(-2)) {
  //    QuaternionBasedCubeCeilingIPCGradient2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
  //    //PrintVector("IPC_force_0", -g_temp);
  //    gradient += dt * dt * g_temp;
  //  }
  //  else if (rbc_info.ipc_type[rb] != nm(-1)) {
  //    QuaternionBasedCubeCubeIPCGradient2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
  //    //PrintVector("IPC_force_1", -g_temp);
  //    gradient += dt * dt * g_temp;
  //  }
  //  if (g_temp.norm() > T(0)) {
  //    rbc_info.used_acp = true;
  //  }
  //}
}


////////////////////////////////////////////
// total energy and its gradient and hessian
////////////////////////////////////////////

T RigidBodyBasedTotalEnergy(const sz rb, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T energy = T(0);
  energy += NewIncrementalPotential(rb, rbs_info, rigid_bodies);
  energy += dt * dt * rigid_bodies[rbs_info.current_rb].total_mass * rbc_info.gravity * rigid_bodies[rbs_info.current_rb].x_com[1];
  if (!rbc_info.segment_mesh.empty()) {
    energy += dt * dt * RigidBodyBasedPointConstraintPotential(rb, rbc_info, rigid_bodies);
  }
  // Add IPC energy
  if (rbc_info.use_ipc) {
    T e_temp = T(0);
    if (rbc_info.ipc_type[rb] == nm(-2)) {
      e_temp = dt * dt * QuaternionBasedCubeCeilingIPC2DTemp(rb, rbc_info, rigid_bodies);
      energy += e_temp;
    }
    else if (rbc_info.ipc_type[rb] != nm(-1)) {
      e_temp = dt * dt * QuaternionBasedCubeCubeIPC2DTemp(rbc_info.ipc_type[rb], rb, rbc_info, rigid_bodies);
      energy += e_temp;
    }
    if (e_temp > T(0)) {
      rbc_info.used_acp = true;
    }
  }
  return energy;
}

void RigidBodyBasedTotalGradient(const sz rb, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Eigen::Matrix<T, 6, 1>& gradient) {
  // Add "KE"-like gradient and Hessian
  AddNewIncrementalGradient(rb, dt, rbs_info, rigid_bodies, gradient);

  // Add gravity gradient
  Eigen::Matrix<T, 6, 1> g_temp;
  GravitationalGradient(rb, rbc_info, rigid_bodies, g_temp);
  gradient += dt * dt * g_temp;

  // Add PCP gradient and Hessian for all constraints
  if (!rbc_info.segment_mesh.empty()) {
    g_temp.setZero();
    RigidBodyBasedPointConstraintGradient(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
    gradient += dt * dt * g_temp;
  }
  
  // Add NIF acceleration gradient
  if (rbc_info.include_constraint_acceleration && !rbc_info.constraint_accelerations[rbc_info.constrained_rigid_body_count[rb]].empty()) {
    g_temp.setZero();
    NIFAccelerationGradient(rb, rbc_info, rigid_bodies, g_temp);
    gradient += dt * dt * g_temp;
  }
  
  // Add IPC gradient
  if (rbc_info.use_ipc) {
    g_temp.setZero();
    if (rbc_info.ipc_type[rb] == nm(-2)) {
      QuaternionBasedCubeCeilingIPCGradient2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
      //PrintVector("IPC_force_0", -g_temp);
      gradient += dt * dt * g_temp;
    }
    else if (rbc_info.ipc_type[rb] != nm(-1)) {
      QuaternionBasedCubeCubeIPCGradient2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
      //PrintVector("IPC_force_1", -g_temp);
      gradient += dt * dt * g_temp;
    }
    if (g_temp.norm() > T(0)) {
      rbc_info.used_acp = true;
    }
  }
}

void RigidBodyBasedTotalGradientHessian(const sz rb, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies,
  Eigen::Matrix<T, 6, 1>& gradient, Eigen::Matrix<T, 6, 6>& Hessian) {
  // Add "KE"-like gradient and Hessian
  AddNewIncrementalGradientHessian(rb, dt, rbs_info, rigid_bodies, gradient, Hessian);

  // Add gravity gradient
  Eigen::Matrix<T, 6, 1> g_temp;
  GravitationalGradient(rb, rbc_info, rigid_bodies, g_temp);
  gradient += dt * dt * g_temp;

  Eigen::Matrix<T, 6, 6> H_temp;
  // Add PCP gradient and Hessian for all constraints
  if (!rbc_info.segment_mesh.empty()) {
    g_temp.setZero();
    H_temp.setZero();
    RigidBodyBasedPointConstraintGradientHessian(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp, H_temp);
    gradient += dt * dt * g_temp;
    Hessian += dt * dt * H_temp;
  }
  
  // Add NIF acceleration gradient
  if (rbc_info.include_constraint_acceleration && !rbc_info.constraint_accelerations[rbc_info.constrained_rigid_body_count[rb]].empty()) {
    g_temp.setZero();
    NIFAccelerationGradient(rb, rbc_info, rigid_bodies, g_temp);
    gradient += dt * dt * g_temp;
  }

  // Add IPC gradient and Hessian
  if (rbc_info.use_ipc) {
    g_temp.setZero();
    H_temp.setZero();
    if (rbc_info.ipc_type[rb] == nm(-2)) {
      QuaternionBasedCubeCeilingIPCGradientHessian2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp, H_temp);
      //PrintVector("IPC_force_0", -g_temp);
      gradient += dt * dt * g_temp;
      Hessian += dt * dt * H_temp;
    }
    else if (rbc_info.ipc_type[rb] != nm(-1)) {
      QuaternionBasedCubeCubeIPCGradientHessian2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp, H_temp);
      //PrintVector("IPC_force_1", -g_temp);
      gradient += dt * dt * g_temp;
      Hessian += dt * dt * H_temp;
    }
    if (g_temp.norm() > T(0)) {
      rbc_info.used_acp = true;
    }
  }
  
}

T TraditionalRigidBodyBasedTotalEnergy(const sz rb, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T energy = T(0);
  energy += TraditionalIncrementalPotential(dt, rbs_info.rigid_bodies_input[rb], rigid_bodies[rb]);
  energy += dt * dt * rigid_bodies[rbs_info.current_rb].total_mass * rbc_info.gravity * rigid_bodies[rbs_info.current_rb].x_com[1];
  if (!rbc_info.segment_mesh.empty()) {
    energy += dt * dt * RigidBodyBasedPointConstraintPotential(rb, rbc_info, rigid_bodies);
  }
  //// Add IPC energy
  //if (rbc_info.use_ipc) {
  //  T e_temp = T(0);
  //  if (rbc_info.ipc_type[rb] == nm(-2)) {
  //    e_temp = dt * dt * QuaternionBasedCubeCeilingIPC2DTemp(rb, rbc_info, rigid_bodies);
  //    energy += e_temp;
  //  }
  //  else if (rbc_info.ipc_type[rb] != nm(-1)) {
  //    e_temp = dt * dt * QuaternionBasedCubeCubeIPC2DTemp(rbc_info.ipc_type[rb], rb, rbc_info, rigid_bodies);
  //    energy += e_temp;
  //  }
  //  if (e_temp > T(0)) {
  //    rbc_info.used_acp = true;
  //  }
  //}
  return energy;
}

void TraditionalRigidBodyBasedTotalGradient(const sz rb, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Eigen::Matrix<T, 6, 1>& gradient) {
  // Add "KE"-like gradient and Hessian
  AddTraditionalIPGradient(dt, rbs_info, rigid_bodies, gradient);
  //PrintVector("gradient", gradient);
  // Add gravity gradient
  Eigen::Matrix<T, 6, 1> g_temp;
  GravitationalGradient(rb, rbc_info, rigid_bodies, g_temp);
  gradient += dt * dt * g_temp;

  // Add PCP gradient and Hessian for all constraints
  if (!rbc_info.segment_mesh.empty()) {
    g_temp.setZero();
    RigidBodyBasedPointConstraintGradient(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
    gradient += dt * dt * g_temp;
  }

  //// Add NIF acceleration gradient
  //if (rbc_info.include_constraint_acceleration && !rbc_info.constraint_accelerations[rbc_info.constrained_rigid_body_count[rb]].empty()) {
  //  g_temp.setZero();
  //  NIFAccelerationGradient(rb, rbc_info, rigid_bodies, g_temp);
  //  gradient += dt * dt * g_temp;
  //}

  //// Add IPC gradient
  //if (rbc_info.use_ipc) {
  //  g_temp.setZero();
  //  if (rbc_info.ipc_type[rb] == nm(-2)) {
  //    QuaternionBasedCubeCeilingIPCGradient2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
  //    //PrintVector("IPC_force_0", -g_temp);
  //    gradient += dt * dt * g_temp;
  //  }
  //  else if (rbc_info.ipc_type[rb] != nm(-1)) {
  //    QuaternionBasedCubeCubeIPCGradient2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
  //    //PrintVector("IPC_force_1", -g_temp);
  //    gradient += dt * dt * g_temp;
  //  }
  //  if (g_temp.norm() > T(0)) {
  //    rbc_info.used_acp = true;
  //  }
  //}
}

void TraditionalRigidBodyBasedTotalGradientHessian(const sz rb, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies,
  Eigen::Matrix<T, 6, 1>& gradient, Eigen::Matrix<T, 6, 6>& Hessian) {
  // Add "KE"-like gradient and Hessian
  AddTraditionalIPGradientHessian(dt, rbs_info, rigid_bodies, gradient, Hessian);
  //PrintVector("Gradient from GradientHessian", gradient);
  // Add gravity gradient
  Eigen::Matrix<T, 6, 1> g_temp;
  GravitationalGradient(rb, rbc_info, rigid_bodies, g_temp);
  gradient += dt * dt * g_temp;

  Eigen::Matrix<T, 6, 6> H_temp;
  // Add PCP gradient and Hessian for all constraints
  if (!rbc_info.segment_mesh.empty()) {
    g_temp.setZero();
    H_temp.setZero();
    RigidBodyBasedPointConstraintGradientHessian(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp, H_temp);
    gradient += dt * dt * g_temp;
    Hessian += dt * dt * H_temp;
  }

  //// Add NIF acceleration gradient
  //if (rbc_info.include_constraint_acceleration && !rbc_info.constraint_accelerations[rbc_info.constrained_rigid_body_count[rb]].empty()) {
  //  g_temp.setZero();
  //  NIFAccelerationGradient(rb, rbc_info, rigid_bodies, g_temp);
  //  gradient += dt * dt * g_temp;
  //}

  //// Add IPC gradient and Hessian
  //if (rbc_info.use_ipc) {
  //  g_temp.setZero();
  //  H_temp.setZero();
  //  if (rbc_info.ipc_type[rb] == nm(-2)) {
  //    QuaternionBasedCubeCeilingIPCGradientHessian2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp, H_temp);
  //    //PrintVector("IPC_force_0", -g_temp);
  //    gradient += dt * dt * g_temp;
  //    Hessian += dt * dt * H_temp;
  //  }
  //  else if (rbc_info.ipc_type[rb] != nm(-1)) {
  //    QuaternionBasedCubeCubeIPCGradientHessian2DTemp(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp, H_temp);
  //    //PrintVector("IPC_force_1", -g_temp);
  //    gradient += dt * dt * g_temp;
  //    Hessian += dt * dt * H_temp;
  //  }
  //  if (g_temp.norm() > T(0)) {
  //    rbc_info.used_acp = true;
  //  }
  //}

}

////////////////////////////////////////////////////
// system total energy and its gradient and hessian
////////////////////////////////////////////////////
T RigidBodiesSystemEnergy(const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T energy = T(0);
  for (sz rb = 0; rb < rigid_bodies.size(); ++rb) {
    energy += RigidBodyBasedEnergy(rb, dt, rbs_info, rbc_info, rigid_bodies);
  }
  for (sz c = 0; c < rbc_info.constraint_segments_info.size(); ++c) {
    energy += ConstraintBasedEnergy(c, dt, rbc_info, rigid_bodies);
  }
  return energy;
}

void RigidBodiesSystemGradient(const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Eigen::VectorXd& gradient) {
  gradient.resize(nm(6) * rigid_bodies.size());
  gradient.setZero();
  // rigid-body-based gradients
  for (sz rb = 0; rb < rigid_bodies.size(); ++rb) {
    Vector6d g_temp = Vector6d::Zero();
    RigidBodyBasedGradient(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp);
    gradient.segment(6 * rb, 6) += g_temp;
  }
  // segment-based gradients
  for (sz c = 0; c < rbc_info.constraint_segments_info.size(); ++c) {
    sz rb0 = rbc_info.segment_mesh[2 * c];
    sz rb1 = rbc_info.segment_mesh[2 * c + 1];
    std::vector<Vector6d> gradients;
    ConstraintBasedGradient(c, dt, rbs_info, rbc_info, rigid_bodies, gradients);
    gradient.segment(6 * rb0, 6) += gradients[0];
    gradient.segment(6 * rb1, 6) += gradients[1];
  }
}

void RigidBodiesSystemGradientHessian(const T dt, const RBsInfo& rbs_info,  RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies,
  Eigen::VectorXd& gradient, Eigen::MatrixXd& Hessian) {
  sz N = rigid_bodies.size();
  gradient.resize(6 * N);
  gradient.setZero();
  Hessian.resize(6 * N, 6 * N);
  Hessian.setZero();
  for (sz rb = 0; rb < N; ++rb) {
    Vector6d g_temp = Vector6d::Zero();
    Matrix6d H_temp = Matrix6d::Zero();
    //RigidBodyBasedGradientHessian(rb, dt, rbs_info, rbc_info, rigid_bodies, g_temp, H_temp);
    gradient.segment(6 * rb, 6) += g_temp;
    Hessian.block<6, 6>(6 * rb, 6 * rb) += H_temp;
  }
  for (sz c = 0; c < rbc_info.constraint_segments_info.size(); ++c) {
    sz rb0 = rbc_info.segment_mesh[2 * c];
    sz rb1 = rbc_info.segment_mesh[2 * c + 1];
    std::vector<Vector6d> gradients;
    std::vector<Matrix6d> hessians;
    //ConstraintBasedGradientHessian(c, dt, rbs_info, rbc_info, rigid_bodies, gradients, hessians);
    gradient.segment(6 * rb0, 6) += gradients[0];
    gradient.segment(6 * rb1, 6) += gradients[1];
    Hessian.block<6, 6>(6 * rb0, 6 * rb0) += hessians[0];
    Hessian.block<6, 6>(6 * rb0, 6 * rb1) += hessians[1];
    Hessian.block<6, 6>(6 * rb1, 6 * rb0) += hessians[2];
    Hessian.block<6, 6>(6 * rb1, 6 * rb1) += hessians[3];
  }
}

////////////////////////////////////////////
// total energy and its gradient and hessian
////////////////////////////////////////////
T RigidBodyBasedTotalTraditionalEnergy(const sz rb, const T dt, const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies) {
  T energy = T(0);
  energy += TraditionalIncrementalPotential(dt, rbs_info.rigid_bodies_input[rb], rigid_bodies[rb]);
  energy += dt * dt * rigid_bodies[rbs_info.current_rb].total_mass * rbc_info.gravity * rigid_bodies[rbs_info.current_rb].x_com[1];
  if (!rbc_info.segment_mesh.empty()) {
    energy += dt * dt * RigidBodyBasedPointConstraintPotential(rb, rbc_info, rigid_bodies);
  }
  // Add IPC energy
  if (rbc_info.use_ipc) {
    T e_temp = T(0);
    if (rbc_info.ipc_type[rb] == nm(-2)) {
      e_temp = dt * dt * QuaternionBasedCubeCeilingIPC2DTemp(rb, rbc_info, rigid_bodies);
      energy += e_temp;
    }
    else if (rbc_info.ipc_type[rb] != nm(-1)) {
      e_temp = dt * dt * QuaternionBasedCubeCubeIPC2DTemp(rbc_info.ipc_type[rb], rb, rbc_info, rigid_bodies);
      energy += e_temp;
    }
    if (e_temp > T(0)) {
      rbc_info.used_acp = true;
    }
  }
  return energy;
}


void TraditionalTotalIPCGradient(const T& dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies, Eigen::Matrix<T, 6, 1>& gradient) {
  // Add "KE"-like gradient and Hessian
  AddTraditionalIPGradient(dt, rbs_info, rigid_bodies, gradient);

  // Add gravity gradient
  Eigen::Matrix<T, 6, 1> g_temp;
  GravitationalGradient(rbs_info.current_rb, rbc_info, rigid_bodies, g_temp);
  gradient += dt * dt * g_temp;

  //// Add PCP gradient and Hessian for all constraints
  //if (rbc_info.use_incident_elements_rb) {
  //  //AddAllBEPCPGradient(dt, rbs_info, rbc_info, rigid_bodies, gradient);
  //}
}

void TraditionalTotalIPCGradientHessian(const T& dt, const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, const std::vector<RigidBody>& rigid_bodies,
  Eigen::Matrix<T, 6, 1>& gradient, Eigen::Matrix<T, 6, 6>& Hessian) {
  // Add "KE"-like gradient and Hessian
  AddTraditionalIPGradientHessian(dt, rbs_info, rigid_bodies, gradient, Hessian);

  // Add gravity gradient
  Eigen::Matrix<T, 6, 1> g_temp;
  GravitationalGradient(rbs_info.current_rb, rbc_info, rigid_bodies, g_temp);
  gradient += dt * dt * g_temp;

  //// Add PCP gradient and Hessian for all constraints
  //if (rbc_info.use_incident_elements_rb) {
  //  //AddAllBEPCPGradientHessian(dt, rbs_info, rbc_info, rigid_bodies, gradient, Hessian);
  //}
}

////////////////////////////
// data conversion functions
////////////////////////////
void YToRigidBody6D(const T dt, const TV& y, const RigidBody& rb_input, RigidBody& rb) {
  for (sz i = 0; i < 3; i++) {
    rb.x_com[i] = y[i];
    rb.v_com[i] = (y[i] - rb_input.x_com[i]) / dt;
    rb.omega[i] = y[i + 3];
  }
  Vector3T angle = { dt * y[3],dt * y[4],dt * y[5] };
  rb.orientation = ALGEBRA::QuaternionMultiply(ALGEBRA::QuaternionFromVector(angle), rb_input.orientation);
  rb.orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(rb.omega), rb.orientation), T(.5));
}

void YToRigidBody6DOld(const T dt, const TV& y, const RigidBody& rb_input, RigidBody& rb) {
  for (sz i = 0; i < 3; i++) {
    rb.x_com[i] = y[i];
    rb.omega[i] = y[i + 3];
  }
}

void RigidBodyToY6D(const RigidBody& rb, TV& y) {
  for (sz i = 0; i < 3; i++) {
    y[i] = rb.x_com[i];
    y[i + 3] = rb.omega[i];
  }
}

//////////////////////////
// 2D omega shifting
/////////////////////////
template<typename func>
void Shift1DOmega(const func& gradient, const T period, const TV& y0, T& w_np1) {
  // shifting
  // 1. compute initial guess gradient
  T g = gradient(y0)[5];
  //std::cout << "f'(0) = " << g << std::endl;
  //std::cout << "unshifted = " << w_np1 << std::endl;

  if (g < T(0)) {
    w_np1 -= std::floor(w_np1 / period) * period;
  }
  else {
    w_np1 -= std::ceil(w_np1 / period) * period;
  }
  //std::cout << "Shifted = " << w_np1 << std::endl;
  //PrintVector("Actual Output", newton_output[i]);
}

template<typename func>
void Shift3DOmegaFor2DSim(const func& gradient, const T period, const TV& y0, TV& y) {
  // shifting
  // 1. compute initial guess gradient
  T g = gradient(y0)[5];
  //std::cout << "f'(0) = " << g << std::endl;
  //std::cout << "unshifted = " << w_np1 << std::endl;

  Vector3T w = { y[3],y[4],y[5] };
  T w_signed_norm;
  if (y[5] >= T(0)) {
    w_signed_norm = ALGEBRA::Norm(w);
  }
  else {
    w_signed_norm = T(-1) * ALGEBRA::Norm(w);
  }
  w = ALGEBRA::Scale(w, T(1) / w_signed_norm);

  if (g < T(0)) {
    w_signed_norm -= std::floor(w_signed_norm / period) * period;
  }
  else {
    w_signed_norm -= std::ceil(w_signed_norm / period) * period;
  }

  for (size_t i = 0; i < 3; ++i) {
    y[3 + i] = w_signed_norm * w[i];
  }

  //std::cout << "Shifted = " << w_np1 << std::endl;
  //PrintVector("Actual Output", newton_output[i]);
}


///////////////////////////
// Newton related functions
///////////////////////////
void Solve6DLinearSystem(const bool& solve_full_system, const Eigen::Matrix<T, 6, 1>& g_new, const Eigen::Matrix<T, 6, 6>& H_new, TV& dY) {
  if (!solve_full_system) {
    // solving for d_x
    Eigen::Vector3d g_Y0 = g_new.block<3, 1>(0, 0);
    Eigen::Matrix3d H_Y0 = H_new.block<3, 3>(0, 0);
    Eigen::Vector3d dY0 = -H_Y0.completeOrthogonalDecomposition().pseudoInverse() * g_Y0;
    //Eigen::Vector3d dY0 = -H_Y0.inverse() * g_Y0;
    // solving for d_w
    Eigen::Vector3d g_Y1 = g_new.block<3, 1>(3, 0);
    Eigen::Matrix3d H_Y1 = H_new.block<3, 3>(3, 3);
    Eigen::Vector3d dY1 = -H_Y1.completeOrthogonalDecomposition().pseudoInverse() * g_Y1;
    //Eigen::Vector3d dY1 = -H_Y1.inverse() * g_Y1;
    // compute dY
    dY = { dY0[0], dY0[1], dY0[2], dY1[0], dY1[1] ,dY1[2] };
  }
  else {
    Eigen::Matrix<T, 6, 1> d_Y = -H_new.completeOrthogonalDecomposition().pseudoInverse() * g_new;
    //Eigen::Matrix<T, 6, 1> d_Y = -H_new.inverse() * g_new;
    dY = { d_Y[0], d_Y[1], d_Y[2], d_Y[3], d_Y[4], d_Y[5] };
  }
}

// This Newton keeps track of 6D only
template<typename Func1, typename Func2, typename Func3>
void BENewtonsMethod6D(const Func1& energy, const Func2& gradient, const Func3& gradientandhessian, const T dt,
  const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, NewtonsMethodInfo& newton_info, LineSearchInfo& ls_info, std::vector<RigidBody>& rigid_bodies) {
  newton_info.failing_status = true;
  // obtain y and compute energy
  // initial guess: x = x_n, omega = 0 (i.e., v = 0, q = q_n, q_dot = 0)
  TV y(nm(6), T(0));
  RigidBodyToY6D(rigid_bodies[rbs_info.current_rb], y);
  for (size_t i = 3; i < 6; ++i) {
    y[i] = T(0);
  }
  newton_info.energy.emplace_back(energy(y));

  // Newton's iteration
  for (sz k = 0; k < newton_info.max_it + sz(1); k++) {
    if (newton_info.only_gradient_descent) {
      // compute 6D gradient and Hessian
      TV g = gradient(y);

      // record 6D residual norm (true norm) and break if residual is small 
      newton_info.residual_norm.emplace_back(VectorTools::Norm<T, TV>(g, ErrorNorm::L2));
      /*if ((newton_info.residual_norm[k] < newton_info.tol) || (newton_info.residual_norm[k] < newton_info.tol * newton_info.residual_norm[0])) {
        break;
      }*/
      if (newton_info.residual_norm[k] < newton_info.tol) {
        break;
      }

      // FAIL if residual norm is still too big after max newton iterations
      if (k == newton_info.max_it) {
        newton_info.failing_status = false;
      }

      // line search

      // compute 6D direction
      TV dy_k(g.size());
      VectorTools::Scale(T(-1), g, dy_k);
      T phi_prime_0 = VectorTools::DotProduct(dy_k, g);

      ls_info.step_size = T(1);
      newton_info.direction_info.emplace_back(Vector4T{ T(0),T(0),T(1),T(1) });

      // Update line search direction and do line search
      ls_info.direction = dy_k;

      switch (newton_info.line_search_op) {
      case 0:
        BackTracking(energy, gradient, ls_info, y);
        break;
      case 1:
        StrongWolfe(energy, gradient, ls_info, y);
        break;
      default:
        throw "BENewtonsMethod: Invalid line_search_op.";
      }
      newton_info.direction_info[k][3] = ls_info.step_size;
      YToRigidBody6D(dt, y, rbs_info.rigid_bodies_input[rbs_info.current_rb], rigid_bodies[rbs_info.current_rb]);
      newton_info.energy.emplace_back(energy(y));
      // Assert to make sure energy goes down
      if (newton_info.energy[k + 1] >= newton_info.energy[k]) {
        std::cout << "At k = " << k << ", and line_search_op = " << newton_info.line_search_op << "..." << std::endl;
        std::cout << "RigidBodyConstraints::BENewtonsMethod6D: Line Search did not reduce energy." << std::endl;
        //TGSLAssert(false, "RigidBodyConstraints::BENewtonsMethod6D: Line Search did not reduce energy.");
      }


    }
    else {
      TV y_k = y;
      // compute 6D gradient and Hessian
      Eigen::Matrix<T, 6, 1> g_k = Eigen::Matrix<T, 6, 1>::Zero();
      Eigen::Matrix<T, 6, 6> H_k = Eigen::Matrix<T, 6, 6>::Zero();
      gradientandhessian(y, rigid_bodies, g_k, H_k);
      

      // record 6D residual norm (true norm) and break if residual is small 
      newton_info.residual_norm.emplace_back(g_k.norm());
      /*if ((newton_info.residual_norm[k] < newton_info.tol) || (newton_info.residual_norm[k] < newton_info.tol * newton_info.residual_norm[0])) {
        break;
      }*/
      if (newton_info.residual_norm[k] < newton_info.tol) {
        break;
      }

      // FAIL if residual norm is still too big after max newton iterations
      if (k == newton_info.max_it) {
        newton_info.failing_status = false;
      }

      // compute 6D direction
      TV dy_k(y.size(), T(0));
      Solve6DLinearSystem(newton_info.solve_full_system, g_k, H_k, dy_k);

      // line search
      if (newton_info.do_line_search) {
        // compute temp energy[k+1]
        TV y_temp(y.size());
        VectorTools::LinearCombination(T(1), y_k, T(1), dy_k, y_temp);
        T energy_kp1_temp = energy(y_temp);
        TV gradient_kp1_temp = gradient(y_temp);

        // check if energy_kp1 is descending, if not, use line search
        if (VectorTools::Norm<T, TV>(gradient_kp1_temp, ErrorNorm::L2) < newton_info.residual_norm[k]) {
          //if (energy_kp1_temp < newton_info.energy[k]) {
          newton_info.direction_info.emplace_back(Vector4T{ T(1),T(0),T(0),T(1) });
          //std::cout << "Newton's decreases energy successfully." << std::endl;
          y = y_temp;
          YToRigidBody6D(dt, y, rbs_info.rigid_bodies_input[rbs_info.current_rb], rigid_bodies[rbs_info.current_rb]);
          newton_info.energy.emplace_back(energy_kp1_temp);
        }
        else {
          // use line search
          // first, re-initialize step_size to 1
          ls_info.step_size = T(1);
          newton_info.direction_info.emplace_back(Vector4T{ T(0),T(1),T(0),T(1) });
          //std::cout << "At k = " << k << ", Newton's did not decrease energy..." << std::endl;
          // compute 6D gradient and phi'(0) to check if Newton's descending
          TV g = { g_k[0],g_k[1],g_k[2],g_k[3],g_k[4],g_k[5] };
          T phi_prime_0 = VectorTools::DotProduct(dy_k, g);

          // if not, use gradient descent direction
          if (phi_prime_0 >= T(-1e-8)) {
            newton_info.direction_info[k] = { nm(0),nm(0),nm(1),T(1) };
            //std::cout << "and Newton's does not give a descending direction." << std::endl;
            VectorTools::Scale(T(-1), g, dy_k);
            phi_prime_0 = VectorTools::DotProduct(dy_k, g);
          }

          // Update line search direction and do line search
          ls_info.direction = dy_k;
          switch (newton_info.line_search_op) {
          case 0:
            BackTracking(energy, gradient, ls_info, y);
            break;
          case 1:
            StrongWolfe(energy, gradient, ls_info, y);
            break;
          default:
            throw "BENewtonsMethod: Invalid line_search_op.";
          }
          newton_info.direction_info[k][3] = ls_info.step_size;
          YToRigidBody6D(dt, y, rbs_info.rigid_bodies_input[rbs_info.current_rb], rigid_bodies[rbs_info.current_rb]);
          newton_info.energy.emplace_back(energy(y));
          // Assert to make sure energy goes down
          if (newton_info.energy[k + 1] >= newton_info.energy[k]) {
            std::cout << "At k = " << k << ", and line_search_op = " << newton_info.line_search_op << "..." << std::endl;
            std::cout << "RigidBodyConstraints::BENewtonsMethod6D: Line Search did not reduce energy." << std::endl;
            //TGSLAssert(false, "RigidBodyConstraints::BENewtonsMethod6D: Line Search did not reduce energy.");
          }


        }
      }
      else {
        VectorTools::LinearCombination(T(1), y_k, T(1), dy_k, y);
        YToRigidBody6D(dt, y, rbs_info.rigid_bodies_input[rbs_info.current_rb], rigid_bodies[rbs_info.current_rb]);
        newton_info.energy.emplace_back(energy(y));
      }
    }

  }
  // update velocities
  rigid_bodies[rbs_info.current_rb].v_com = ALGEBRA::Scale(ALGEBRA::Difference(rigid_bodies[rbs_info.current_rb].x_com, rbs_info.rigid_bodies_input[rbs_info.current_rb].x_com), T(1) / dt);
  
  //PrintVector("x", rigid_bodies[rbs_info.current_rb].x_com);
  //PrintVector("x_n", rbs_info.rigid_bodies_hat[rbs_info.current_rb].x_com);
  
  rigid_bodies[rbs_info.current_rb].orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rigid_bodies[rbs_info.current_rb].orientation, rbs_info.rigid_bodies_input[rbs_info.current_rb].orientation), T(1) / dt);
  rigid_bodies[rbs_info.current_rb].omega = { y[3],y[4],y[5] };
}

template<typename Func1, typename Func2, typename Func3>
void BENewtonsMethod6DEXP(const Func1& energy, const Func2& gradient, const Func3& gradientandhessian, const RigidBody& rb_input,
  NewtonsMethodInfo& newton_info, LineSearchInfo& ls_info, TV& y) {
  newton_info.failing_status = true;
  // obtain y and compute energy
  // initial guess: x = x_n, omega = 0 (i.e., v = 0, q = q_n, q_dot = 0)
  TGSLAssert((y.size() == nm(6)), "RigidBodyConstraints::BENewtonsMethod6DEXP: incorrect y-dimension.");
  for (size_t i = 0; i < 3; ++i) {
    y[i] = rb_input.x_com[i];
    y[i + 3] = T(0);
  }
  newton_info.energy.emplace_back(energy(y));

  // Newton's iteration
  for (sz k = 0; k < newton_info.max_it + sz(1); k++) {
    if (newton_info.only_gradient_descent) {
      // compute 6D gradient and Hessian
      TV g = gradient(y);

      // record 6D residual norm (true norm) and break if residual is small 
      newton_info.residual_norm.emplace_back(VectorTools::Norm<T, TV>(g, ErrorNorm::L2));
      /*if ((newton_info.residual_norm[k] < newton_info.tol) || (newton_info.residual_norm[k] < newton_info.tol * newton_info.residual_norm[0])) {
        break;
      }*/
      if (newton_info.residual_norm[k] < newton_info.tol) {
        break;
      }

      // FAIL if residual norm is still too big after max newton iterations
      if (k == newton_info.max_it) {
        newton_info.failing_status = false;
      }

      // line search

      // compute 6D direction
      TV dy_k(g.size());
      VectorTools::Scale(T(-1), g, dy_k);
      T phi_prime_0 = VectorTools::DotProduct(dy_k, g);

      ls_info.step_size = T(1);
      newton_info.direction_info.emplace_back(Vector4T{ T(0),T(0),T(1),T(1) });

      // Update line search direction and do line search
      ls_info.direction = dy_k;

      switch (newton_info.line_search_op) {
      case 0:
        BackTracking(energy, gradient, ls_info, y);
        break;
      case 1:
        StrongWolfe(energy, gradient, ls_info, y);
        break;
      default:
        throw "BENewtonsMethod: Invalid line_search_op.";
      }
      newton_info.direction_info[k][3] = ls_info.step_size;
      newton_info.energy.emplace_back(energy(y));
      // Assert to make sure energy goes down
      if (newton_info.energy[k + 1] >= newton_info.energy[k]) {
        std::cout << "At k = " << k << ", and line_search_op = " << newton_info.line_search_op << "..." << std::endl;
        std::cout << "RigidBodyConstraints::BENewtonsMethod6D: Line Search did not reduce energy." << std::endl;
        //TGSLAssert(false, "RigidBodyConstraints::BENewtonsMethod6D: Line Search did not reduce energy.");
      }


    }
    else {
      TV y_k = y;
      // compute 6D gradient and Hessian
      Eigen::Matrix<T, 6, 1> g_k = Eigen::Matrix<T, 6, 1>::Zero();
      Eigen::Matrix<T, 6, 6> H_k = Eigen::Matrix<T, 6, 6>::Zero();
      gradientandhessian(y, g_k, H_k);


      // record 6D residual norm (true norm) and break if residual is small 
      newton_info.residual_norm.emplace_back(g_k.norm());
      /*if ((newton_info.residual_norm[k] < newton_info.tol) || (newton_info.residual_norm[k] < newton_info.tol * newton_info.residual_norm[0])) {
        break;
      }*/
      if (newton_info.residual_norm[k] < newton_info.tol) {
        break;
      }

      // FAIL if residual norm is still too big after max newton iterations
      if (k == newton_info.max_it) {
        newton_info.failing_status = false;
      }

      // compute 6D direction
      TV dy_k(y.size(), T(0));
      Solve6DLinearSystem(newton_info.solve_full_system, g_k, H_k, dy_k);

      // line search
      if (newton_info.do_line_search) {
        // compute temp energy[k+1]
        TV y_temp(y.size());
        VectorTools::LinearCombination(T(1), y_k, T(1), dy_k, y_temp);
        T energy_kp1_temp = energy(y_temp);
        TV gradient_kp1_temp = gradient(y_temp);

        // check if energy_kp1 is descending, if not, use line search
        if (VectorTools::Norm<T, TV>(gradient_kp1_temp, ErrorNorm::L2) < newton_info.residual_norm[k]) {
          //if (energy_kp1_temp < newton_info.energy[k]) {
          newton_info.direction_info.emplace_back(Vector4T{ T(1),T(0),T(0),T(1) });
          //std::cout << "Newton's decreases energy successfully." << std::endl;
          y = y_temp;
          newton_info.energy.emplace_back(energy_kp1_temp);
        }
        else {
          // use line search
          // first, re-initialize step_size to 1
          ls_info.step_size = T(1);
          newton_info.direction_info.emplace_back(Vector4T{ T(0),T(1),T(0),T(1) });
          //std::cout << "At k = " << k << ", Newton's did not decrease energy..." << std::endl;
          // compute 6D gradient and phi'(0) to check if Newton's descending
          TV g = { g_k[0], g_k[1], g_k[2], g_k[3], g_k[4], g_k[5] };
          T phi_prime_0 = VectorTools::DotProduct(dy_k, g);
          //std::cout << "Newton direction phi'(0) = " << phi_prime_0 << std::endl;

          // if not, use gradient descent direction
          if (phi_prime_0 > T(-1) * g_k.squaredNorm()) {
            newton_info.direction_info[k] = { nm(0),nm(0),nm(1),T(1) };
            //std::cout << "and Newton's does not give a descending direction." << std::endl;
            VectorTools::Scale(T(-1), g, dy_k);
            phi_prime_0 = VectorTools::DotProduct(dy_k, g);
            //std::cout << "gradient descent direction phi'(0) = " << T(-1) * g_k.squaredNorm() << " = " << phi_prime_0 << std::endl;
          
          }

          // Update line search direction and do line search
          ls_info.direction = dy_k;
          switch (newton_info.line_search_op) {
          case 0:
            BackTracking(energy, gradient, ls_info, y);
            break;
          case 1:
            StrongWolfe(energy, gradient, ls_info, y);
            break;
          default:
            throw "BENewtonsMethod: Invalid line_search_op.";
          }
          newton_info.direction_info[k][3] = ls_info.step_size;
          newton_info.energy.emplace_back(energy(y));
          // Assert to make sure energy goes down
          if (newton_info.energy[k + 1] >= newton_info.energy[k]) {
            std::cout << "k = " << k << ", energy[k] = " << newton_info.energy[k] << ", energy[k+1] = " << newton_info.energy[k + 1] << std::endl;
            std::cout << "residual_norm[k] = " << newton_info.residual_norm[k] << std::endl;
            //TGSLAssert(false, "RigidBodyConstraints::BENewtonsMethod6D: Line Search did not reduce energy.");
            newton_info.failing_status = false;
          }
        }
      }
      else {
        VectorTools::LinearCombination(T(1), y_k, T(1), dy_k, y);
        newton_info.energy.emplace_back(energy(y));
      }
    }
  }
}

template<typename Func>
void BENewtonsMethod6DEXPWOLS(const Func& gradientandhessian, const RigidBody& rb_input, NewtonsMethodInfo& newton_info, LineSearchInfo& ls_info, TV& y) {
  newton_info.failing_status = true;
  // obtain y
  // initial guess: x = x_n, omega = 0 (i.e., v = 0, q = q_n, q_dot = 0)
  TGSLAssert((y.size() == nm(6)), "RigidBodyConstraints::BENewtonsMethod6DEXP: incorrect y-dimension.");
  for (size_t i = 0; i < 3; ++i) {
    y[i] = rb_input.x_com[i];
    y[i + 3] = T(0);
  }

  // Newton's iteration
  for (sz k = 0; k < newton_info.max_it + sz(1); k++) {
    std::cout << "k = " << k << std::endl;
    TV y_k = y;
    // compute 6D gradient and Hessian
    Eigen::Matrix<T, 6, 1> g_k = Eigen::Matrix<T, 6, 1>::Zero();
    Eigen::Matrix<T, 6, 6> H_k = Eigen::Matrix<T, 6, 6>::Zero();
    gradientandhessian(y, g_k, H_k);
    PrintVector("g_k", g_k);
    std::cout << "H_k = " << std::endl;
    std::cout << H_k << std::endl;

    // record 6D residual norm (true norm) and break if residual is small 
    newton_info.residual_norm.emplace_back(g_k.norm());
    /*if ((newton_info.residual_norm[k] < newton_info.tol) || (newton_info.residual_norm[k] < newton_info.tol * newton_info.residual_norm[0])) {
      break;
    }*/
    if (newton_info.residual_norm[k] < newton_info.tol) {
      break;
    }

    // FAIL if residual norm is still too big after max newton iterations
    if (k == newton_info.max_it) {
      newton_info.failing_status = false;
    }

    // compute 6D direction
    TV dy_k(y.size(), T(0));
    Solve6DLinearSystem(newton_info.solve_full_system, g_k, H_k, dy_k);
    PrintVector("dy_k", dy_k);

    VectorTools::LinearCombination(T(1), y_k, T(1), dy_k, y);

    //// shift
    //T period = T(60) * pi;
    //Vector3T w_k = { y_k[3], y_k[4], y_k[5] };
    //T w_n = ALGEBRA::Norm(w_k);
    //Vector3T w_temp = { y[3], y[4], y[5] };
    //T w_np1 = ALGEBRA::Norm(w_temp);
    //ALGEBRA::Normalize(w_temp);
    //T kk = ClosestInteger((w_np1 - w_n) / period);
    //w_np1 -= kk * period;
    //w_temp = ALGEBRA::Scale(w_temp, w_np1);
    //for (size_t alpha = 0; alpha < 3; ++alpha) {
    //  y[alpha + 3] = w_temp[alpha];
    //}
  }
}

// WOLS = WithOut Line Search
void BENewtonsMethod6DWOLS(const T dt,
  const RBsInfo& rbs_info, RBConstraintInfo& rbc_info, NewtonsMethodInfo& newton_info, LineSearchInfo& ls_info, std::vector<RigidBody>& rigid_bodies) {
  newton_info.failing_status = true;
  // obtain y and compute energy
  TV y(nm(6), T(0));
  RigidBodyToY6D(rigid_bodies[rbs_info.current_rb], y);

  // Newton's iteration
  for (sz k = 0; k < newton_info.max_it + sz(1); k++) {
    TV y_k = y;
    // compute 6D gradient and Hessian
    Eigen::Matrix<T, 6, 1> g_k = Eigen::Matrix<T, 6, 1>::Zero();
    Eigen::Matrix<T, 6, 6> H_k = Eigen::Matrix<T, 6, 6>::Zero();
    RigidBodyBasedTotalGradientHessian(rbs_info.current_rb, dt, rbs_info, rbc_info, rigid_bodies, g_k, H_k);

    // record 6D residual norm (true norm) and break if residual is small 
    newton_info.residual_norm.emplace_back(g_k.norm());
    if (newton_info.residual_norm[k] < newton_info.tol) {
      break;
    }

    // FAIL if residual norm is still too big after max newton iterations
    if (k == newton_info.max_it) {
      newton_info.failing_status = false;
    }

    // compute 6D direction
    TV dy_k(y.size(), T(0));
    Solve6DLinearSystem(newton_info.solve_full_system, g_k, H_k, dy_k);

    VectorTools::LinearCombination(T(1), y_k, T(1), dy_k, y);
    YToRigidBody6D(dt, y, rbs_info.rigid_bodies_input[rbs_info.current_rb], rigid_bodies[rbs_info.current_rb]);
  }
  // update velocities
  rigid_bodies[rbs_info.current_rb].v_com = ALGEBRA::Scale(ALGEBRA::Difference(rigid_bodies[rbs_info.current_rb].x_com, rbs_info.rigid_bodies_input[rbs_info.current_rb].x_com), T(1) / dt);
  rigid_bodies[rbs_info.current_rb].orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rigid_bodies[rbs_info.current_rb].orientation, rbs_info.rigid_bodies_input[rbs_info.current_rb].orientation), T(1) / dt);
  rigid_bodies[rbs_info.current_rb].omega = { y[3],y[4],y[5] };
}


// This Newton keeps track of 6D only
template<typename Func1, typename Func2, typename Func3>
void BENewtonsMethodFixX(const Func1& energy, const Func2& gradient, const Func3& gradientandhessian, const T dt,
  const RBsInfo& rbs_info, const RBConstraintInfo& rbc_info, NewtonsMethodInfo& newton_info, LineSearchInfo& ls_info, std::vector<RigidBody>& rigid_bodies) {
  newton_info.failing_status = true;
  // obtain y and compute energy
  TV y(nm(6), T(0));
  RigidBodyToY6D(rigid_bodies[rbs_info.current_rb], y); // initial y[3-5] are zeros as we have initial guess q_np1 = q_n
  newton_info.energy.emplace_back(energy(y));

  // Newton's iteration
  for (sz k = 0; k < newton_info.max_it + sz(1); k++) {
    TV y_k = y;
    // compute 6D gradient and Hessian
    Eigen::Matrix<T, 6, 1> g_k = Eigen::Matrix<T, 6, 1>::Zero();
    Eigen::Matrix<T, 6, 6> H_k = Eigen::Matrix<T, 6, 6>::Zero();
    gradientandhessian(y, rigid_bodies, g_k, H_k);

    Eigen::Vector3d g_3D = g_k.block<3, 1>(3, 0);
    Eigen::Matrix3d H_3D = H_k.block<3, 3>(3, 3);

    // record 6D residual norm (true norm) and break if residual is small 
    newton_info.residual_norm.emplace_back(g_3D.norm());
    /*if ((newton_info.residual_norm[k] < newton_info.tol) || (newton_info.residual_norm[k] < newton_info.tol * newton_info.residual_norm[0])) {
      break;
    }*/
    if (newton_info.residual_norm[k] < newton_info.tol) {
      break;
    }

    // FAIL if residual norm is still too big after max newton iterations
    if (k == newton_info.max_it) {
      newton_info.failing_status = false;
    }

    // compute 6D direction
    TV dy_k(y.size(), T(0));
    Eigen::Vector3d dw = -H_3D.inverse() * g_3D;
    for (sz alpha = 0; alpha < 3; ++alpha) {
      dy_k[alpha + 3] = dw[alpha];
    }

      // line search
    VectorTools::LinearCombination(T(1), y_k, T(1), dy_k, y);
    YToRigidBody6D(dt, y, rbs_info.rigid_bodies_input[rbs_info.current_rb], rigid_bodies[rbs_info.current_rb]);
    newton_info.energy.emplace_back(energy(y));

  }
  // update velocities
  rigid_bodies[rbs_info.current_rb].v_com = ALGEBRA::Scale(ALGEBRA::Difference(rigid_bodies[rbs_info.current_rb].x_com, rbs_info.rigid_bodies_input[rbs_info.current_rb].x_com), T(1) / dt);
  rigid_bodies[rbs_info.current_rb].orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rigid_bodies[rbs_info.current_rb].orientation, rbs_info.rigid_bodies_input[rbs_info.current_rb].orientation), T(1) / dt);
  rigid_bodies[rbs_info.current_rb].omega = { y[3],y[4],y[5] };
}

template<typename Func3>
void BENewtonsMethodFixXEXP(const Func3& gradientandhessian, const RigidBody& rb_input,
  NewtonsMethodInfo& newton_info, LineSearchInfo& ls_info, TV& y) {
  newton_info.failing_status = true;
  // obtain y and compute energy
  // initial guess: x = x_n, omega = 0 (i.e., v = 0, q = q_n, q_dot = 0)
  TGSLAssert((y.size() == nm(6)), "RigidBodyConstraints::BENewtonsMethod6DEXP: incorrect y-dimension.");
  for (size_t i = 0; i < 3; ++i) {
    y[i] = rb_input.x_com[i];
    y[i + 3] = T(0);
  }

  //PrintVector("y0", y);

  // Newton's iteration
  for (sz k = 0; k < newton_info.max_it + sz(1); k++) {
    //std::cout << "k = " << k << std::endl;
    TV y_k = y;
    // compute 6D gradient and Hessian
    Eigen::Matrix<T, 6, 1> g_k = Eigen::Matrix<T, 6, 1>::Zero();
    Eigen::Matrix<T, 6, 6> H_k = Eigen::Matrix<T, 6, 6>::Zero();
    gradientandhessian(y, g_k, H_k);

    Eigen::Vector3d g_3D = g_k.block<3, 1>(3, 0);
    Eigen::Matrix3d H_3D = H_k.block<3, 3>(3, 3);

    //PrintVector("g_3D", g_3D);
    //std::cout << "H_3D = " << std::endl;
    //std::cout << H_3D << std::endl;
    //std::cout << "f(" << y[5] << ") = " << g_3D(2) << std::endl;
    //std::cout << "f'(" << y[5] << ") = " << H_3D(2, 2) << std::endl;

    // record 6D residual norm (true norm) and break if residual is small 
    newton_info.residual_norm.emplace_back(g_3D.norm());
    /*if ((newton_info.residual_norm[k] < newton_info.tol) || (newton_info.residual_norm[k] < newton_info.tol * newton_info.residual_norm[0])) {
      break;
    }*/
    if (newton_info.residual_norm[k] < newton_info.tol) {
      break;
    }

    // FAIL if residual norm is still too big after max newton iterations
    if (k == newton_info.max_it) {
      newton_info.failing_status = false;
    }

    // compute 6D direction
    TV dy_k(y.size(), T(0));
    Eigen::Vector3d dw = -H_3D.inverse() * g_3D;
    for (sz alpha = 0; alpha < 3; ++alpha) {
      dy_k[alpha + 3] = dw[alpha];
    }
    //std::cout << "dw = " << dw(2) << std::endl;

    // No line search yet
    VectorTools::LinearCombination(T(1), y_k, T(1), dy_k, y);
    //PrintVector("y_k", y_k);
  }
}


// This Newton keeps track of 6D only
template<typename Func>
void NewtonsMethod1D(const Func& gradientandhessian, const T dt,
  const RBsInfo& rbs_info, NewtonsMethodInfo& newton_info, LineSearchInfo& ls_info, std::vector<RigidBody>& rigid_bodies) {
  newton_info.failing_status = true;
  // obtain y and compute energy
  TV y(nm(6), T(0));
  RigidBodyToY6D(rigid_bodies[rbs_info.current_rb], y);
  TGSLAssert((y.size() == ls_info.direction.size()), "RigidBodyConstraints::NewtonsMethod1D: dimensions of y and d do not match.");
  T alpha = T(1);

  TV y_alpha(y.size());
  // Newton's iteration
  for (sz k = 0; k < newton_info.max_it + sz(1); k++) {
    std::cout << "After " << k << " iterations, alpha = " << alpha << std::endl;
    
    VectorTools::LinearCombination(T(1), y, alpha, ls_info.direction, y_alpha);
    Eigen::Matrix<T, 6, 1> g = Eigen::Matrix<T, 6, 1>::Zero();
    Eigen::Matrix<T, 6, 6> H = Eigen::Matrix<T, 6, 6>::Zero();
    //gradientandhessian(y_alpha, g, H);
    T g_k = T(0), H_k = T(0);
    for (sz i = 0; i < g.size(); ++i) {
      g_k += g[i] * ls_info.direction[i];
      for (sz j = 0; j < g.size(); ++j) {
        H_k += ls_info.direction[i] * H(i, j) * ls_info.direction[j];
      }
    }
    //std::cout << "g_norm = " << g.norm() << std::endl;
    //std::cout << "|g_k| = " << std::abs(g_k) << std::endl;
    // record 1D residual norm (true norm) and break if residual is small 
    newton_info.residual_norm.emplace_back(std::abs(g_k));
    if ((newton_info.residual_norm[k] < newton_info.tol) || (newton_info.residual_norm[k] < newton_info.tol * newton_info.residual_norm[0])) {
      break; 
    }

    // FAIL if residual norm is still too big after max newton iterations
    if (k == newton_info.max_it) {
      newton_info.failing_status = false;
      break;
    }

    TGSLAssert((std::abs(H_k) > 1e-10), "RigidBodyConstraints::NewtonsMethod1D: 1D Hessian is too small. :(");
    alpha -= g_k / H_k;
  }
  //VectorTools::LinearCombination(T(1), y, T(1), ls_info.direction, y_alpha);
  // update velocities
  YToRigidBody6D(dt, y_alpha, rbs_info.rigid_bodies_input[rbs_info.current_rb], rigid_bodies[rbs_info.current_rb]);
  rigid_bodies[rbs_info.current_rb].v_com = ALGEBRA::Scale(ALGEBRA::Difference(rigid_bodies[rbs_info.current_rb].x_com, rbs_info.rigid_bodies_input[rbs_info.current_rb].x_com), T(1) / dt);
  rigid_bodies[rbs_info.current_rb].orientation_dot = ALGEBRA::Scale(ALGEBRA::Difference(rigid_bodies[rbs_info.current_rb].orientation, rbs_info.rigid_bodies_input[rbs_info.current_rb].orientation), T(1) / dt);
}

template<typename Func1>
void LambdaTest(const Func1& energy, const TV& x, T& m) {
  std::cout << std::endl;
  std::cout << "When m = " << m << std::endl;
  std::cout << "Energy = " << energy(x) << std::endl;
  m += T(1);
}

}