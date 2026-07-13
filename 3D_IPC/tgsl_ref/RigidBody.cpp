#include <rigid_body/RigidBody.h>

namespace TGSL {

RigidBody::RigidBody(const TVP& X, const TV& nodal_mass) {
  v_com = Vector();
  omega = Vector3T();
  orientation_dot = Vector4T();
  inertia_tensor_material = Eigen::Matrix3d::Zero();
  I_hat = ALGEBRA::DenseMatrix(4, 4, T(0));
  InertiaC4 = ALGEBRA::DenseMatrix(16, 16, T(0));

  // compute total mass and center of mass
  total_mass = T(0);
  x_com = Particle();

  for (sz i = 0; i < X.size(); i++) {
    total_mass += nodal_mass[i];
    for (size_t c = 0; c < d; c++) {
      x_com[c] += nodal_mass[i] * X[i][c];
    }
  }
  for (size_t c = 0; c < d; c++) {
    x_com[c] /= total_mass;
  }

  if (total_mass > infinite_mass_threshold) {
    infinite_mass = true;
    orientation = {T(1), T(0), T(0), T(0)};
    x_com = Particle();
  #ifdef TWO_D
    inertia_tensor_inv = {T(0)};
  #else
    inertia_tensor_inv = {T(0), T(0), T(0)};
  #endif
  } else {
    // compute intertia tensor and total momenta
    size_t angular_alpha_start = 0;
  #ifdef TWO_D
    angular_alpha_start = 2;
  #endif

    // J tensor
    Eigen::Matrix3d I_3 = Eigen::Matrix3d::Identity();
    for (sz i = 0; i < X.size(); i++) {
      Particle R = ALGEBRA::Difference(X[i], x_com);
      for (size_t alpha = 0; alpha < d; alpha++) {
        for (size_t beta = 0; beta < d; beta++) {
          inertia_tensor_material(alpha, beta) += nodal_mass[i] * R[alpha] * R[beta];
        }
      }
    }
    inertia_tensor_material = inertia_tensor_material.trace() * I_3 - inertia_tensor_material;
    

    // I_hat tensor
    for (sz i = 0; i < X.size(); i++) {
      Particle R = ALGEBRA::Difference(X[i], x_com);
      for (size_t alpha = 1; alpha < d + 1; alpha++) {
        for (size_t beta = 1; beta < d + 1; beta++) {
          I_hat(alpha, beta) += nodal_mass[i] * R[alpha - 1] * R[beta - 1];
        }
      }
    }

    // InertiaC4 tensor
    for (int beta = 0; beta < 4; beta++) {
      for (int sigma = 0; sigma < 4; sigma++) {
        for (int epsilon = 0; epsilon < 4; epsilon++) {
          for (int theta = 0; theta < 4; theta++) {
            for (int alpha = 1; alpha < 4; alpha++) {
              for (int delta = 1; delta < 4; delta++) {
                for (int rho = 1; rho < 4; rho++) {
                  InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) += I_hat(delta, rho) * QPT_QPT(alpha, beta, delta, epsilon) * QPT_QPT(alpha, sigma, rho, theta);
                }
              }
            }
          }
        }
      }
    }

    // old inertia tensor
    TGSL::ALGEBRA::DenseMatrix I(3, 3);
    for (sz i = 0; i < X.size(); i++) {
      Vector r = Vector();
      for (size_t alpha = 0; alpha < d; alpha++) {
        r[alpha] = X[i][alpha] - x_com[alpha];
      }
      for (size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
        for (size_t beta = angular_alpha_start; beta < 3; beta++) {
          for (size_t delta = 0; delta < d; delta++) {
            for (size_t gamma = 0; gamma < d; gamma++) {
              for (size_t epsilon = 0; epsilon < d; epsilon++) {
                I(alpha, beta) += LeviCivita(int(gamma), int(delta), int(alpha)) * r[delta] * nodal_mass[i] * LeviCivita(int(gamma), int(epsilon), int(beta)) * r[epsilon];
              }
            }
          }
        }
      }
    }
#ifndef TWO_D
    // diagonalize intertia tensor
    TV lambda;
    ALGEBRA::DenseMatrix Q(3, 3);
    SymmetricEigenDecomposition(I, Q, lambda, 20, true);
    if (Q.Determinant() < T(0)) {
      Q(0, 2) = -Q(0, 2);
      Q(1, 2) = -Q(1, 2);
      Q(2, 2) = -Q(2, 2);
    }
    orientation = QuaternionFromRotationMatrix(Q);
    inertia_tensor_inv = {T(1) / lambda[0], T(1) / lambda[1], T(1) / lambda[2]};
#else
    inertia_tensor_inv = {T(1) / I(2, 2)};
    orientation = {T(1), T(0), T(0), T(0)};
#endif
  }
}

RigidBody::RigidBody(const IV& particle_indices, const TVP& X, const TV& nodal_mass) {
  v_com = Vector();
  omega = Vector3T();
  orientation_dot = Vector4T();
  inertia_tensor_material = Eigen::Matrix3d::Zero();
  I_hat = ALGEBRA::DenseMatrix(4, 4, T(0));
  InertiaC4 = ALGEBRA::DenseMatrix(16, 16, T(0));

  // compute total mass and center of mass
  total_mass = T(0);
  x_com = Particle();

  for (sz j = 0; j < particle_indices.size(); j++) {
    sz i = sz(particle_indices[j]);
    total_mass += nodal_mass[i];
    for (size_t c = 0; c < d; c++) {
      x_com[c] += nodal_mass[i] * X[i][c];
    }
  }
  for (size_t c = 0; c < d; c++) {
    x_com[c] /= total_mass;
  }

  if (total_mass > infinite_mass_threshold) {
    infinite_mass = true;
    orientation = {T(1), T(0), T(0), T(0)};
    x_com = Particle();
#ifdef TWO_D
    inertia_tensor_inv = {T(0)};
#else
    inertia_tensor_inv = {T(0), T(0), T(0)};
#endif
  }

  // J tensor
  Eigen::Matrix3d I_3 = Eigen::Matrix3d::Identity();
  for (sz i = 0; i < X.size(); i++) {
    Particle R = ALGEBRA::Difference(X[i], x_com);
    for (size_t alpha = 0; alpha < 3; alpha++) {
      for (size_t beta = 0; beta < 3; beta++) {
        inertia_tensor_material(alpha, beta) += nodal_mass[i] * R[alpha] * R[beta];
      }
    }
  }
  inertia_tensor_material = inertia_tensor_material.trace() * I_3 - inertia_tensor_material;

  // I_hat tensor
  for (sz i = 0; i < X.size(); i++) {
    Particle R = ALGEBRA::Difference(X[i], x_com);
    for (size_t alpha = 1; alpha < 4; alpha++) {
      for (size_t beta = 1; beta < 4; beta++) {
        I_hat(alpha, beta) += nodal_mass[i] * R[alpha - 1] * R[beta - 1];
      }
    }
  }

  // InertiaC4 tensor
  for (int beta = 0; beta < 4; beta++) {
    for (int sigma = 0; sigma < 4; sigma++) {
      for (int epsilon = 0; epsilon < 4; epsilon++) {
        for (int theta = 0; theta < 4; theta++) {
          for (int alpha = 1; alpha < 4; alpha++) {
            for (int delta = 1; delta < 4; delta++) {
              for (int rho = 1; rho < 4; rho++) {
                InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) += I_hat(delta, rho) * QPT_QPT(alpha, beta, delta, epsilon) * QPT_QPT(alpha, sigma, rho, theta);
              }
            }
          }
        }
      }
}
    }

  // compute intertia tensor and total momenta
  size_t angular_alpha_start = 0;
#ifdef TWO_D
  angular_alpha_start = 2;
#endif

  TGSL::ALGEBRA::DenseMatrix I(3, 3);
  for (sz j = 0; j < particle_indices.size(); j++) {
    sz i = sz(particle_indices[j]);
    Vector r = Vector();
    for (size_t alpha = 0; alpha < d; alpha++) {
      r[alpha] = X[i][alpha] - x_com[alpha];
    }
    for (size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
      for (size_t beta = angular_alpha_start; beta < 3; beta++) {
        for (size_t delta = 0; delta < d; delta++) {
          for (size_t gamma = 0; gamma < d; gamma++) {
            for (size_t epsilon = 0; epsilon < d; epsilon++) {
              I(alpha, beta) += LeviCivita(int(gamma), int(delta), int(alpha)) * r[delta] * nodal_mass[i] * LeviCivita(int(gamma), int(epsilon), int(beta)) * r[epsilon];
            }
          }
        }
      }
    }
  }
  #ifndef TWO_D
  // diagonalize intertia tensor
  TV lambda;
  ALGEBRA::DenseMatrix Q(3, 3);
  SymmetricEigenDecomposition(I, Q, lambda, 20, true);
  if (Q.Determinant() < T(0)) {
    Q(0, 2) = -Q(0, 2);
    Q(1, 2) = -Q(1, 2);
    Q(2, 2) = -Q(2, 2);
  }
  orientation = QuaternionFromRotationMatrix(Q);
  inertia_tensor_inv = {T(1) / lambda[0], T(1) / lambda[1], T(1) / lambda[2]};
  #else
    inertia_tensor_inv = {T(1) / I(2, 2)};
    orientation = {T(1), T(0), T(0), T(0)};
  #endif
}

RigidBody::RigidBody(const TVP& X, const TVV& v, const TV& nodal_mass) {
  // compute total mass, center of mass and center of mass velocities
  total_mass = T(0);
  x_com = Particle();
  v_com = Vector();
  inertia_tensor_material = Eigen::Matrix3d::Zero();
  I_hat = ALGEBRA::DenseMatrix(4, 4, T(0));
  InertiaC4 = ALGEBRA::DenseMatrix(16, 16, T(0));

  for (sz i = 0; i < X.size(); i++) {
    total_mass += nodal_mass[i];
    for (size_t c = 0; c < d; c++) {
      x_com[c] += nodal_mass[i] * X[i][c];
      v_com[c] += nodal_mass[i] * v[i][c];
    }
  }
  for (size_t c = 0; c < d; c++) {
    x_com[c] /= total_mass;
    v_com[c] /= total_mass;
  }

  if (total_mass > infinite_mass_threshold) {
    infinite_mass = true;
    orientation = {T(1), T(0), T(0), T(0)};
    x_com = Particle();
    v_com = Vector();
#ifdef TWO_D
    inertia_tensor_inv = {T(0)};
#else
    inertia_tensor_inv = {T(0), T(0), T(0)};
#endif
  }

  // J tensor
  Eigen::Matrix3d I_3 = Eigen::Matrix3d::Identity();
  for (sz i = 0; i < X.size(); i++) {
    Particle R = ALGEBRA::Difference(X[i], x_com);
    for (size_t alpha = 0; alpha < 3; alpha++) {
      for (size_t beta = 0; beta < 3; beta++) {
        inertia_tensor_material(alpha, beta) += nodal_mass[i] * R[alpha] * R[beta];
      }
    }
  }
  inertia_tensor_material = inertia_tensor_material.trace() * I_3 - inertia_tensor_material;

  // I_hat tensor
  for (sz i = 0; i < X.size(); i++) {
    Particle R = ALGEBRA::Difference(X[i], x_com);
    for (size_t alpha = 1; alpha < 4; alpha++) {
      for (size_t beta = 1; beta < 4; beta++) {
        I_hat(alpha, beta) += nodal_mass[i] * R[alpha - 1] * R[beta - 1];
      }
    }
  }

  // InertiaC4 tensor
  for (int beta = 0; beta < 4; beta++) {
    for (int sigma = 0; sigma < 4; sigma++) {
      for (int epsilon = 0; epsilon < 4; epsilon++) {
        for (int theta = 0; theta < 4; theta++) {
          for (int alpha = 1; alpha < 4; alpha++) {
            for (int delta = 1; delta < 4; delta++) {
              for (int rho = 1; rho < 4; rho++) {
                InertiaC4(beta * 4 + sigma, epsilon * 4 + theta) += I_hat(delta, rho) * QPT_QPT(alpha, beta, delta, epsilon) * QPT_QPT(alpha, sigma, rho, theta);
              }
            }
          }
        }
      }
    }
  }

  // compute intertia tensor and total momenta
  TGSL::ALGEBRA::DenseMatrix I(3, 3);
  Vector p = Vector();
  Vector3T l = Vector3T();

  size_t angular_alpha_start = 0;
#ifdef TWO_D
  angular_alpha_start = 2;
#endif

  for (sz i = 0; i < X.size(); i++) {
    Vector r = Vector();
    for (size_t alpha = 0; alpha < d; alpha++) {
      r[alpha] = X[i][alpha] - x_com[alpha];
      p[alpha] += v[i][alpha] * nodal_mass[i];
    }
    for (size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
      for (size_t beta = 0; beta < d; beta++) {
        for (size_t gamma = 0; gamma < d; gamma++) {
          l[alpha] += LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * nodal_mass[i] * v[i][gamma];
        }
      }
    }

    for (size_t alpha = angular_alpha_start; alpha < 3; alpha++) {
      for (size_t beta = angular_alpha_start; beta < 3; beta++) {
        for (size_t delta = 0; delta < d; delta++) {
          for (size_t gamma = 0; gamma < d; gamma++) {
            for (size_t epsilon = 0; epsilon < d; epsilon++) {
              I(alpha, beta) += LeviCivita(int(gamma), int(delta), int(alpha)) * r[delta] * nodal_mass[i] * LeviCivita(int(gamma), int(epsilon), int(beta)) * r[epsilon];
            }
          }
        }
      }
    }
  }
#ifndef TWO_D
  // solve for omega from angular momentum and intertia tensor
  TV l_rhs = {l[0], l[1], l[2]};
  TV omega_solve(3);
  TGSL::ALGEBRA::PLUSolve(I, l_rhs, omega_solve);
  omega = {omega_solve[0], omega_solve[1], omega_solve[2]};

  // diagonalize intertia tensor
  TV lambda;
  ALGEBRA::DenseMatrix Q(3, 3);
  SymmetricEigenDecomposition(I, Q, lambda, 20, true);
  if (Q.Determinant() < T(0)) {
    Q(0, 2) = -Q(0, 2);
    Q(1, 2) = -Q(1, 2);
    Q(2, 2) = -Q(2, 2);
  }
  orientation = QuaternionFromRotationMatrix(Q);
  Vector4T Omega = { T(0),omega[0],omega[1],omega[2] };
  orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(Omega, orientation), T(.5));
  inertia_tensor_inv = {T(1) / lambda[0], T(1) / lambda[1], T(1) / lambda[2]};
#else
  inertia_tensor_inv = {T(1) / I(2, 2)};
  omega = {T(0), T(0), l[2] * inertia_tensor_inv[0]};
  orientation = {T(0), T(0), T(0), T(1)};
#endif
}

Vector RigidBody::Velocity(const Particle& x) const {
  Vector result = v_com;
#ifdef TWO_D
  size_t angular_start = 2;
  Vector r = {x[0] - x_com[0], x[1] - x_com[1]};
#else
  size_t angular_start = 0;
  Vector r = {x[0] - x_com[0], x[1] - x_com[1], x[2] - x_com[2]};
#endif
  for (size_t alpha = 0; alpha < d; alpha++) {
    for (size_t beta = angular_start; beta < 3; beta++) {
      for (size_t gamma = 0; gamma < d; gamma++) {
        result[alpha] += LeviCivita(int(alpha), int(beta), int(gamma)) * omega[beta] * r[gamma];
      }
    }
  }
  return result;
}

void RigidBody::UpdatePositionAndOrientation(const T dt) {
  for (size_t c = 0; c < d; c++)
    x_com[c] += dt * v_com[c];

  Vector4T q_hat = ALGEBRA::QuaternionFromVector({dt * omega[0], dt * omega[1], dt * omega[2]});
  orientation = ALGEBRA::QuaternionMultiply(q_hat, orientation);
}

void RigidBody::UpdateByPositionAndOmega(const T dt, const RigidBody& rb_input, const TV& y) {
#ifdef TWO_D
  TGSLAssert(false, "RigidBody::UpdateByPositionAndOmega: Not defined nor used for 2D.");
#else
  TGSLAssert((y.size() == nm(6)), "RigidBody::UpdateByPositionAndOmega: dimension of y should be 6.");

  x_com = { y[0], y[1], y[2] };
  omega = { y[3], y[4], y[5] };

  v_com = ALGEBRA::Scale(ALGEBRA::Difference(x_com, rb_input.x_com), T(1) / dt);
  Vector4T q_hat = ALGEBRA::QuaternionFromVector({ dt * omega[0], dt * omega[1], dt * omega[2] });
  orientation = ALGEBRA::QuaternionMultiply(q_hat, orientation);
  orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));
#endif // TWO_D
  
}

void RigidBody::UpdateByPositionAndTheta(const T dt, const RigidBody& rb_input, const TV& y) {
#ifdef TWO_D
  TGSLAssert((y.size() == nm(3)), "RigidBody::UpdateByPositionAndTheta: dimension of y should be 3.");

  x_com = { y[0], y[1] };
  orientation = { cos(y[2] / T(2)), T(0), T(0), sin(y[2] / T(2)) };

  v_com = ALGEBRA::Scale(ALGEBRA::Difference(x_com, rb_input.x_com), T(1) / dt);
  T c = orientation[0] * rb_input.orientation[0] + orientation[3] * rb_input.orientation[3];
  omega = { T(0), T(0), T(2) * std::acos(c) / dt };
  orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));

#else
  TGSLAssert(false, "RigidBody::UpdateByPositionAndTheta: not defined for 3D.");
#endif // TWO_D

}

TV RigidBody::GetPositionAndOmega() const {
#ifdef TWO_D
  TGSLAssert(false, "RigidBody::GetPositionAndOmega: Not defined nor used for 2D.");
#endif // TWO_D
  return { x_com[0], x_com[1], x_com[2], omega[0], omega[1], omega[2] };
}

void RigidBody::UpdateByPartialGridInput(const TV& input) {
#ifndef TWO_D
  
  nm input_dim = input.size();
  if (input_dim == 2) {
    // input = { omega[2], theta }, where theta = 2 * arctan(orientation[3] / orientation[0])
    omega = { T(0), T(0), input[0] };
    orientation = { cos(input[1] / T(2)), T(0), T(0), sin(input[1] / T(2)) };
    orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));
  }
  else if (input_dim == 6) {
    // input = { v_com[0], v_com[1], x_com[0], x_com[1], omega[2], theta }
    v_com = { input[0], input[1], T(0) };
    x_com = { input[2], input[3], T(0) };
    omega = { T(0), T(0), input[4] };
    orientation = { cos(input[5] / T(2)), T(0), T(0), sin(input[5] / T(2)) };
    orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));
  }
  else if (input_dim == 8) {
    // input = { v_com[0], v_com[1], x_com[0], x_com[1], omega[2], theta, a_frame[0], a_frame[1] }
    // TODO: nothing is done with a_frame. Be careful!
    v_com = { input[0], input[1], T(0) };
    x_com = { input[2], input[3], T(0) };
    omega = { T(0), T(0), input[4] };
    orientation = { cos(input[5] / T(2)), T(0), T(0), sin(input[5] / T(2)) };
    orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));
  }
  else if (input_dim == 9) {
    // input = { v_com[0], v_com[1], x_com[0], x_com[1], omega[2], theta_this, a_frame[0], a_frame[1], theta_other }
    // TODO: nothing is done with a_frame and theta_other. Be careful!
    v_com = { input[0], input[1], T(0) };
    x_com = { input[2], input[3], T(0) };
    omega = { T(0), T(0), input[4] };
    orientation = { cos(input[5] / T(2)), T(0), T(0), sin(input[5] / T(2)) };
    orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));
  }
  else if (input_dim == 10) {
    // input = { v_com[0], v_com[1], x_com[0], x_com[1], omega[2], theta, a_frame[0], a_frame[1], y2[0], y2[1] }
    // TODO: nothing is done with a_frame and y2. Be careful!
    v_com = { input[0], input[1], T(0) };
    x_com = { input[2], input[3], T(0) };
    omega = { T(0), T(0), input[4] };
    orientation = { cos(input[5] / T(2)), T(0), T(0), sin(input[5] / T(2)) };
    orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));
  }
  else {
    TGSLAssert(false, "RigidBody::UpdateByGridInput: invalid grid input dimension.");
  }
#endif
}

void RigidBody::UpdateByGridOutput(const T dt, const TV& output) {
  nm output_dim = output.size();
  if (output_dim == 1) {
    // output = { omega[2] }
    omega[2] = output[0];
    Vector4T q_hat = ALGEBRA::QuaternionFromVector({ dt * omega[0], dt * omega[1], dt * omega[2] });
    orientation = ALGEBRA::QuaternionMultiply(q_hat, orientation);
    orientation_dot = orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));
  }
  else if (output_dim == 3) {
    // output = { v_com[0], v_com[1], omega[2] }
    v_com[0] = output[0];
    v_com[1] = output[1];
    for (size_t c = 0; c < d; ++c) {
      x_com[c] += dt * v_com[c];
    }

    omega[2] = output[2];
    Vector4T q_hat = ALGEBRA::QuaternionFromVector({ dt * omega[0], dt * omega[1], dt * omega[2] });
    orientation = ALGEBRA::QuaternionMultiply(q_hat, orientation);
    orientation_dot = orientation_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(ALGEBRA::VectorToQuaternion(omega), orientation), T(.5));
  }
  else {
    TGSLAssert(false, "RigidBody::UpdateByGridOutput: Invalid output dimension.");
  }
}

TV RigidBody::GetPartialGridInput(const nm input_dim) const {
  TV grid_input(input_dim);
  if (input_dim == 2) {
    // grid_input = { omega[2], theta }
    grid_input[0] = omega[2];
    T theta = T(2) * atan2(orientation[3], orientation[0]);
    if (theta > pi) {
      theta -= T(2) * pi;
    }
    else if (theta < -pi) {
      theta += T(2) * pi;
    }
    grid_input[1] = theta;
  }
  else if ((input_dim == 6) || (input_dim == 8) || (input_dim == 10) || (input_dim == 9)) {
    // grid_input = { v_com[0], v_com[1], x_com[0], x_com[1], omega[2], theta }
    grid_input[0] = v_com[0];
    grid_input[1] = v_com[1];
    grid_input[2] = x_com[0];
    grid_input[3] = x_com[1];
    grid_input[4] = omega[2];
    T theta = T(2) * atan2(orientation[3], orientation[0]);
    if (theta > pi) {
      theta -= T(2) * pi;
    }
    else if (theta < -pi) {
      theta += T(2) * pi;
    }
    grid_input[5] = theta;
  }
  else {
    TGSLAssert(false, "RigidBody::GetPartialGridInput: Invalid grid input dimension.");
  }
  return grid_input;
}

TV RigidBody::GetGridOutput(const nm output_dim) const {
  TV grid_output(output_dim);
  if (output_dim == 1) {
    // grid_output = { omega[2] }
    grid_output[0] = omega[2];
  }
  else if (output_dim == 3) {
    // grid_output = { v_com[0], v_com[1], omega[2] }  
    grid_output[0] = v_com[0];
    grid_output[1] = v_com[1];
    grid_output[2] = omega[2];
  }
  else {
    TGSLAssert(false, "RigidBody::GetGridOutput: Invalid output dimension");
  }
  return grid_output;
}

void RigidBody::ToLinearNIF(const Particle& x_frame, const Vector& v_frame) {
  // x_NIF = x_IF - x_frame
  for (size_t c = 0; c < d; ++c) {
    x_com[c] -= x_frame[c];
    v_com[c] -= v_frame[c];
  }
}

void RigidBody::ToLinearIF(const Particle& x_frame, const Vector& v_frame) {
  // x_IF = x_NIF + x_frame
  for (size_t c = 0; c < d; ++c) {
    x_com[c] += x_frame[c];
    v_com[c] += v_frame[c];
  }
}

void RigidBody::DirectCOMUpdate(const Particle& x, const Vector& v) {
  x_com = x;
  v_com = v;
}

void RigidBody::ApplyImpulse(const Particle& x, const Vector& i) {
  if (infinite_mass)
    return;
  else {
    // linear momentum change
    for (size_t alpha = 0; alpha < d; alpha++) {
      v_com[alpha] += i[alpha] / total_mass;
    }
// angular momentum change
//(1) rXi
#ifdef TWO_D
    size_t angular_start = 2;
    Vector r = {x[0] - x_com[0], x[1] - x_com[1]};
#else
    size_t angular_start = 0;
    Vector r = {x[0] - x_com[0], x[1] - x_com[1], x[2] - x_com[2]};
#endif

    Vector3T rXi = Vector3T();
    for (size_t alpha = angular_start; alpha < 3; alpha++) {
      for (size_t beta = 0; beta < d; beta++) {
        for (size_t gamma = 0; gamma < d; gamma++) {
          rXi[alpha] += LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * i[gamma];
        }
      }
    }

//(2) I^{-1}(rXi)
#ifndef TWO_D
    Vector4T q_T = ALGEBRA::ConjugateQuaternion(orientation);
    Vector3T Iinv_rXi = ALGEBRA::QuaternionRotate(q_T, rXi);
    for (size_t alpha = 0; alpha < 3; alpha++) {
      Iinv_rXi[alpha] *= inertia_tensor_inv[alpha];
    }
    Iinv_rXi = ALGEBRA::QuaternionRotate(orientation, Iinv_rXi);
    for (size_t alpha = 0; alpha < 3; alpha++)
      omega[alpha] += Iinv_rXi[alpha];
#else
    omega[2] += rXi[2] * inertia_tensor_inv[0];
#endif
  }
}

ALGEBRA::DenseMatrix RigidBody::MatrixK(const Particle& x) const {
#ifdef TWO_D
  TGSL::ALGEBRA::DenseMatrix K(2, 2);

  Vector r = {x[0] - x_com[0], x[1] - x_com[1]};

  // K
  K(0, 0) = (T(1) / total_mass) + ((r[1] * r[1]) * inertia_tensor_inv[0]);
  K(0, 1) = (-r[0] * r[1]) * inertia_tensor_inv[0];
  K(1, 0) = (-r[0] * r[1]) * inertia_tensor_inv[0];
  K(1, 1) = (T(1) / total_mass) + ((r[0] * r[0]) * inertia_tensor_inv[0]);

  if (infinite_mass) {
    for (size_t alpha = 0; alpha < d; alpha++) {
      for (size_t beta = 0; beta < d; beta++) {
        K(alpha, beta) = T(0);
      }
    }
  }
#else
  Vector r = {x[0] - x_com[0], x[1] - x_com[1], x[2] - x_com[2]};

  TGSL::ALGEBRA::DenseMatrix r_cross(3, 3);
  TGSL::ALGEBRA::DenseMatrix R(3, 3);
  TGSL::ALGEBRA::DenseMatrix I_inv(3, 3);
  TGSL::ALGEBRA::DenseMatrix K(3, 3);
  // r*
  r_cross(0, 0) = T(0);
  r_cross(0, 1) = r[2];
  r_cross(0, 2) = -r[1];
  r_cross(1, 0) = -r[2];
  r_cross(1, 1) = T(0);
  r_cross(1, 2) = r[0];
  r_cross(2, 0) = r[1];
  r_cross(2, 1) = -r[0];
  r_cross(2, 2) = T(0);

  // Quaternion to Matrix
  R(0, 0) = 1 - 2 * orientation[2] * orientation[2] - 2 * orientation[3] * orientation[3];
  R(0, 1) = 2 * (orientation[1] * orientation[2] - orientation[0] * orientation[3]);
  R(0, 2) = 2 * (orientation[1] * orientation[3] + orientation[0] * orientation[2]);
  R(1, 0) = 2 * (orientation[1] * orientation[2] + orientation[0] * orientation[3]);
  R(1, 1) = 1 - 2 * orientation[1] * orientation[1] - 2 * orientation[3] * orientation[3];
  R(1, 2) = 2 * (orientation[2] * orientation[3] - orientation[0] * orientation[1]);
  R(2, 0) = 2 * (orientation[1] * orientation[3] - orientation[0] * orientation[2]);
  R(2, 1) = 2 * (orientation[2] * orientation[3] + orientation[0] * orientation[1]);
  R(2, 2) = 1 - 2 * orientation[1] * orientation[1] - 2 * orientation[2] * orientation[2];

  // Inertia tensor inverse
  I_inv(0, 0) = inertia_tensor_inv[0];
  I_inv(0, 1) = T(0);
  I_inv(0, 2) = T(0);
  I_inv(1, 0) = T(0);
  I_inv(1, 1) = inertia_tensor_inv[1];
  I_inv(1, 2) = T(0);
  I_inv(2, 0) = T(0);
  I_inv(2, 1) = T(0);
  I_inv(2, 2) = inertia_tensor_inv[2];

  I_inv = R * I_inv * R.Transpose();
  K = r_cross.Transpose() * I_inv * r_cross;

  for (size_t alpha = 0; alpha < 3; alpha++) {
    K(alpha, alpha) += T(1) / total_mass;
  }

  // Immovable static obejcts like the ground plane can be created by seting K = 0
  if (infinite_mass) {
    for (size_t alpha = 0; alpha < d; alpha++) {
      for (size_t beta = 0; beta < d; beta++) {
        K(alpha, beta) = T(0);
      }
    }
  }
#endif
  return K;
}

T RigidBody::NKNFriction(const Particle& x, const Vector& N, const Vector& tangent_rel_V, const T mu) const {
  if (infinite_mass)
    return T(0);
  else {
    Vector N_muT = Vector();
    for (size_t alpha = 0; alpha < d; alpha++) {
      N_muT[alpha] = N[alpha] - mu * tangent_rel_V[alpha];
    }
    T result = ALGEBRA::DotProduct(N, N_muT) / total_mass;

//(1) rX(N-muT)
#ifdef TWO_D
    size_t angular_start = 2;
    Vector r = {x[0] - x_com[0], x[1] - x_com[1]};
#else
    size_t angular_start = 0;
    Vector r = {x[0] - x_com[0], x[1] - x_com[1], x[2] - x_com[2]};
#endif
    Vector3T rXN = Vector3T();
    for (size_t alpha = angular_start; alpha < 3; alpha++) {
      for (size_t beta = 0; beta < d; beta++) {
        for (size_t gamma = 0; gamma < d; gamma++) {
          rXN[alpha] += LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * N_muT[gamma];
        }
      }
    }

    //(2) I^{-1}(rX(N-muT))
    Vector3T Iinv_rXN = Vector3T();
#ifndef TWO_D
    Vector4T q_T = ALGEBRA::ConjugateQuaternion(orientation);
    Iinv_rXN = ALGEBRA::QuaternionRotate(q_T, rXN);
    for (size_t alpha = 0; alpha < 3; alpha++) {
      Iinv_rXN[alpha] *= inertia_tensor_inv[alpha];
    }
    Iinv_rXN = ALGEBRA::QuaternionRotate(orientation, Iinv_rXN);
#else
    Iinv_rXN[2] = rXN[2] * inertia_tensor_inv[0];
#endif

    //(3) (I^{-1}(rX(N-muT)))Xr
    Vector3T AN = Vector3T();
    for (size_t alpha = 0; alpha < d; alpha++) {
      for (size_t beta = angular_start; beta < 3; beta++) {
        for (size_t gamma = 0; gamma < d; gamma++) {
          AN[alpha] += LeviCivita(int(alpha), int(beta), int(gamma)) * Iinv_rXN[beta] * r[gamma];
        }
      }
    }

    for (size_t alpha = 0; alpha < d; alpha++)
      result += AN[alpha] * N[alpha];

    return result;
  }
}

T RigidBody::NKN(const Particle& x, const Vector& N) const {
  if (infinite_mass)
    return T(0);
  else {
    T result = ALGEBRA::Norm(N) / total_mass;

//(1) rXN
#ifdef TWO_D
    size_t angular_start = 2;
    Vector r = {x[0] - x_com[0], x[1] - x_com[1]};
#else
    size_t angular_start = 0;
    Vector r = {x[0] - x_com[0], x[1] - x_com[1], x[2] - x_com[2]};
#endif

    Vector3T rXN = Vector3T();
    for (size_t alpha = angular_start; alpha < 3; alpha++) {
      for (size_t beta = 0; beta < d; beta++) {
        for (size_t gamma = 0; gamma < d; gamma++) {
          rXN[alpha] += LeviCivita(int(alpha), int(beta), int(gamma)) * r[beta] * N[gamma];
        }
      }
    }

    //(2) I^{-1}(rXN)
    Vector3T Iinv_rXN = Vector3T();
#ifndef TWO_D
    Vector4T q_T = ALGEBRA::ConjugateQuaternion(orientation);
    Iinv_rXN = ALGEBRA::QuaternionRotate(q_T, rXN);
    for (size_t alpha = 0; alpha < 3; alpha++) {
      Iinv_rXN[alpha] *= inertia_tensor_inv[alpha];
    }
    Iinv_rXN = ALGEBRA::QuaternionRotate(orientation, Iinv_rXN);
#else
    Iinv_rXN[2] = rXN[2] * inertia_tensor_inv[0];
#endif

    //(3) (I^{-1}(rXN))Xr
    Vector3T AN = Vector3T();
    for (size_t alpha = 0; alpha < d; alpha++) {
      for (size_t beta = angular_start; beta < 3; beta++) {
        for (size_t gamma = 0; gamma < d; gamma++) {
          AN[alpha] += LeviCivita(int(alpha), int(beta), int(gamma)) * Iinv_rXN[beta] * r[gamma];
        }
      }
    }

    for (size_t alpha = 0; alpha < d; alpha++)
      result += AN[alpha] * N[alpha];

    return result;
  }
}

void RigidBody::VelocityFromMomenta(const Vector& p, const Vector3T& l) {
  for (size_t alpha = 0; alpha < d; alpha++)
    v_com[alpha] = p[alpha] / total_mass;

#ifndef TWO_D
  Vector4T q_T = ALGEBRA::ConjugateQuaternion(orientation);
  omega = ALGEBRA::QuaternionRotate(q_T, l);
  for (size_t alpha = 0; alpha < 3; alpha++) {
    omega[alpha] *= inertia_tensor_inv[alpha];
  }
  omega = ALGEBRA::QuaternionRotate(orientation, omega);
#else
  omega = Vector3T();
  omega[2] = l[2] * inertia_tensor_inv[0];
#endif
}

Particle RigidBody::WorldSpacePosition(const Particle& X) const {
  Particle x = x_com;

  Vector3T R = Vector3T(), r = Vector3T();
  for (size_t alpha = 0; alpha < d; alpha++) {
    R[alpha] = X[alpha];
  }
  r = ALGEBRA::QuaternionRotate(orientation, R);
  // only for testing gradient purpose
  /*
  Vector4T R_temp = { T(0),R[0],R[1],R[2] };
  Vector4T r_temp = ALGEBRA::QuaternionMultiply(orientation, ALGEBRA::QuaternionMultiply(R_temp, ALGEBRA::QuaternionInverse(orientation)));
  r = { r_temp[1],r_temp[2],r_temp[3] };
  */
  for (size_t alpha = 0; alpha < d; alpha++) {
    x[alpha] += r[alpha];
  }
  return x;
}

Particle RigidBody::MaterialSpacePosition(const Particle& x) const {
  Vector3T r = Vector3T();
  for (size_t alpha = 0; alpha < d; alpha++) {
    r[alpha] = x[alpha] - x_com[alpha];
  }

  Vector3T X_full = ALGEBRA::QuaternionRotate(ALGEBRA::ConjugateQuaternion(orientation), r);
#ifdef TWO_D
  Particle X = {X_full[0], X_full[1]};
#else
  Particle X = X_full;
#endif

  return X;
}

T RigidBody::KineticEnergy() {
  TGSLAssert(!infinite_mass, "RigidBody::KineticEnergy: not well defined for infinte mass case.");
  TGSLAssert(total_mass >= T(0), "RigidBody::KineticEnergy: not well defined for negative mass.");

  T linear_part = T(.5) * total_mass * ALGEBRA::DotProduct(v_com, v_com);

  Vector4T q_T = ALGEBRA::ConjugateQuaternion(orientation);
  Vector3T I_omega = ALGEBRA::QuaternionRotate(q_T, omega);
  for (size_t alpha = 0; alpha < 3; alpha++) {
    TGSLAssert(inertia_tensor_inv[alpha] > T(0), "RigidBody::KineticEnergy: degenerate intertia tensor.");
    I_omega[alpha] *= T(1) / inertia_tensor_inv[alpha];
  }
  I_omega = ALGEBRA::QuaternionRotate(orientation, I_omega);

  T angular_part = T(.5) * ALGEBRA::DotProduct(omega, I_omega);

  return linear_part + angular_part;
}

Vector3T RigidBody::AngularMomentum(const TVP& X, const TV& nodal_mass, const T dt) const {
  // angular momentum using intertia_tensor_inv and omega
  Vector4T q_inv = ALGEBRA::QuaternionInverse(orientation);
  Vector3T I_omega = ALGEBRA::QuaternionRotate(q_inv, omega);
  for (size_t alpha = 0; alpha < 3; alpha++) {
    I_omega[alpha] *= T(1) / inertia_tensor_inv[alpha];
  }
  I_omega = ALGEBRA::QuaternionRotate(orientation, I_omega);
  return I_omega;
}

Vector3T RigidBody::AltAngularMomentum(const TVP& X, const TV& nodal_mass, const T dt) const {
  // angular momentum using quaternions
  Vector3T l = Vector3T();
  #ifndef TWO_D
  RigidBody rb(X, nodal_mass);
  Vector4T q_inv, q_nm1, q_inv_nm1, q_inv_dot;
  q_inv = ALGEBRA::QuaternionInverse(orientation);
  q_nm1 = ALGEBRA::Difference(orientation, ALGEBRA::Scale(orientation_dot, dt));
  q_inv_nm1 = ALGEBRA::QuaternionInverse(q_nm1);
  //q_inv_dot = ALGEBRA::Scale(ALGEBRA::Difference(q_inv, q_inv_nm1), T(1) / dt);
  q_inv_dot = ALGEBRA::Scale(ALGEBRA::QuaternionMultiply(q_inv, ALGEBRA::QuaternionMultiply(orientation_dot, q_inv)), T(-1));
  for (sz p = 0; p < nodal_mass.size(); p++) {
    #ifdef TWO_D
      TGSLAssert(false, "RigidBody::AltAngularMomentum not defined in 2D.");
    #else
    Particle R_p = ALGEBRA::Difference(X[p], rb.x_com);
    Vector4T R_temp = { T(0),R_p[0],R_p[1],R_p[2] };
    Particle r_p = ALGEBRA::QuaternionRotate(orientation, R_p);
    Vector4T v_temp = ALGEBRA::Sum(ALGEBRA::QuaternionMultiply(orientation_dot, ALGEBRA::QuaternionMultiply(R_temp, q_inv)),
      ALGEBRA::QuaternionMultiply(orientation, ALGEBRA::QuaternionMultiply(R_temp, q_inv_dot)));
    Vector v_p = { v_temp[1],v_temp[2],v_temp[3] };
    l = ALGEBRA::Sum(l, ALGEBRA::Scale(ALGEBRA::CrossProduct(r_p, v_p), nodal_mass[p]));
    #endif
  }
  #endif
  return l;
  
}

Eigen::Matrix2d RigidBody::RotationMatrix2d() const {
  Eigen::Matrix2d Q;
#ifdef TWO_D
  Q(0, 0) = orientation[0] * orientation[0] - orientation[3] * orientation[3]; // cos(theta) = cos(theta/2)^2 - sin(theta/2)^2
  Q(1, 1) = Q(0, 0);
  Q(1, 0) = T(2) * orientation[0] * orientation[3]; // sin(theta) = 2 * sin(theta/2) * cos(theta/2)
  Q(0, 1) = T(-1) * Q(1, 0);
#else
  TGSLAssert(false, "RigidBody::RotationMatrix2d: not defined for 3d.");
#endif
  return Q;
}
}  // namespace TGSL