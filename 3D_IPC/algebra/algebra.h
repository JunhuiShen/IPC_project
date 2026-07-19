#pragma once

#include "IPC_math.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "quaternion_math.h"

namespace Rigid_Body {

inline void Assert(bool success, const std::string& flag) {
    if (!success) {
        throw std::runtime_error(flag);
    }
}

namespace ALGEBRA {

// Levi-Civita symbol for indices in {0,1,2}
inline int LeviCivita(int alpha, int beta, int gamma) {
    Assert((0 <= alpha) && (alpha <= 2), "ALGEBRA: Invalid first argument for LeviCivita 3D.");
    Assert((0 <= beta) && (beta <= 2), "ALGEBRA: Invalid second argument for LeviCivita 3D.");
    Assert((0 <= gamma) && (gamma <= 2), "ALGEBRA: Invalid third argument for LeviCivita 3D.");
    if (alpha == 0 && beta == 1 && gamma == 2)
        return 1;
    else if (alpha == 1 && beta == 2 && gamma == 0)
        return 1;
    else if (alpha == 2 && beta == 0 && gamma == 1)
        return 1;
    else if (alpha == 2 && beta == 1 && gamma == 0)
        return -1;
    else if (alpha == 1 && beta == 0 && gamma == 2)
        return -1;
    else if (alpha == 0 && beta == 2 && gamma == 1)
        return -1;
    else
        return 0;
}

// Levi-Civita symbol for indices in {0,1}
inline int LeviCivita(int alpha, int beta) {
    Assert((0 <= alpha) && (alpha <= 1), "ALGEBRA: Invalid first argument for LeviCivita 2D.");
    Assert((0 <= beta) && (beta <= 1), "ALGEBRA: Invalid second argument for LeviCivita 2D.");
    if (alpha == 0 && beta == 1)
        return 1;
    else if (alpha == 1 && beta == 0)
        return -1;
    else
        return 0;
}

// Quaternions are stored as (w, x, y, z)
inline Eigen::Vector4d ConjugateQuaternion(const Eigen::Vector4d& q) {
    return Eigen::Vector4d(q[0], -q[1], -q[2], -q[3]);
}

inline Eigen::Vector4d QuaternionMultiply(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
    return Eigen::Vector4d(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
}

inline Eigen::Matrix<double,4,3> DexpDw(const double dt, const Vec3& w){
    Eigen::Matrix<double,4,3> dexpdw = Eigen::Matrix<double,4,3>::Zero();
    Mat33 I_3 = Mat33::Identity();
    double w_norm = w.norm();
    double angle  = w_norm * dt / double(2);          // (1/2) dt |w|
 
    if(angle < 1e-4){
        // series branch:  sin(x)/x ~ 1 - x^2/6,   coefficients written in dt, |w|
        double c1 = double(.5)*dt      * (double(1) - angle*angle/double(6));   // sin(angle)/|w|
        double h  = -dt*dt*dt/double(24) * (double(1) - angle*angle/double(10)); // bracket coeff
        double g  = -dt*dt/double(4)   * (double(1) - angle*angle/double(6));   // scalar-row coeff
        for(size_t beta = 0; beta < 3; beta++){
            dexpdw(0,beta) = g * w[beta];
        }
        for(size_t alpha = 1; alpha < 4; alpha++){
            for(size_t beta = 0; beta < 3; beta++){
                dexpdw(alpha,beta) = c1 * I_3(alpha-1,beta) + h * w[alpha-1]*w[beta];
            }
        }
    }
    else{
        for(size_t beta = 0; beta < 3; beta++){
            dexpdw(0,beta) = -sin(angle) * dt / double(2) * w[beta] / w_norm;
        }
        for(size_t alpha = 1; alpha < 4; alpha++){
            for(size_t beta = 0; beta < 3; beta++){
                dexpdw(alpha,beta) = cos(angle)*dt/double(2)*w[alpha-1]*w[beta]/(w_norm*w_norm)
                                   + sin(angle)*(I_3(alpha-1,beta)*w_norm - w[alpha-1]*w[beta]/w_norm)/(w_norm*w_norm);
            }
        }
    }
    return dexpdw;
}
 

inline std::array<Eigen::Matrix<double,4,3>,3> D2expDw2(const double dt, const Vec3& w){
    std::array<Eigen::Matrix<double,4,3>,3> d2;
    for(size_t gamma = 0; gamma < 3; gamma++) d2[gamma].setZero();
 
    Mat33 I_3 = Mat33::Identity();
    double w_norm = w.norm();
    double angle  = w_norm * dt / double(2);
 
    double g, g_t, h, h_t;
    if(angle < 1e-2){
        g   = -dt*dt/double(4)              * (double(1) - angle*angle/double(6));
        g_t =  dt*dt*dt*dt/double(48)       * (double(1) - angle*angle/double(10));
        h   = -dt*dt*dt/double(24)          * (double(1) - angle*angle/double(10));
        h_t =  dt*dt*dt*dt*dt/double(480)   * (double(1) - angle*angle/double(14));
    }
    else{
        double s  = sin(angle);
        double c  = cos(angle);
        double n2 = w_norm*w_norm;
        double n3 = n2*w_norm;
        double n4 = n2*n2;
        double n5 = n4*w_norm;
        g   = -dt/double(2) * s / w_norm;
        g_t = -dt*dt/double(4) * c / n2 + dt/double(2) * s / n3;
        h   =  dt/double(2) * c / n2 - s / n3;
        h_t = -dt*dt/double(4) * s / n3 - double(3)*dt/double(2) * c / n4 + double(3) * s / n5;
    }
 
    for(size_t gamma = 0; gamma < 3; gamma++){
        // scalar row (alpha = 0)
        for(size_t beta = 0; beta < 3; beta++){
            d2[gamma](0,beta) = g * I_3(beta,gamma) + g_t * w[beta]*w[gamma];
        }
        // vector rows (alpha = 1..3, component a = alpha-1)
        for(size_t alpha = 1; alpha < 4; alpha++){
            size_t a = alpha - 1;
            for(size_t beta = 0; beta < 3; beta++){
                d2[gamma](alpha,beta) = h * ( I_3(beta,gamma) * w[a]
                                            + I_3(gamma,a)    * w[beta]
                                            + I_3(a,beta)     * w[gamma] )
                                      + h_t * w[a]*w[beta]*w[gamma];
            }
        }
    }
    return d2;
}


inline Eigen::Matrix<double,4,3> DqDw(const double dt, const Vec3& w, const Vec4& q_n){
    Eigen::Matrix<double,4,3> dexpdw = DexpDw(dt,w);
    Eigen::Matrix<double,4,3> DqDw = Eigen::Matrix<double,4,3>::Zero();
    for(size_t alpha = 0; alpha < 4; alpha++){
      for(size_t beta = 0; beta < 3; beta++){
        for(size_t gamma = 0; gamma < 4; gamma++){
          for(size_t delta = 0; delta < 4; delta++){
            DqDw(alpha,beta) += quaternion_product_tensor(alpha,gamma,delta) * dexpdw(gamma,beta) * q_n[delta];
          }
        }
      }
    }
    return DqDw;
}

inline std::array<Eigen::Matrix<double,4,3>,3> D2qDw2(const double dt, const Vec3& w, const Vec4& q_n){
    std::array<Eigen::Matrix<double,4,3>,3> d2expdw2 = D2expDw2(dt, w);
    std::array<Eigen::Matrix<double,4,3>,3> d2qdw2;
    for(size_t gamma = 0; gamma < 3; gamma++) d2qdw2[gamma].setZero();

    for(size_t theta = 0; theta < 3; theta++){
      for(size_t alpha = 0; alpha < 4; alpha++){
        for(size_t beta = 0; beta < 3; beta++){
          for(size_t gamma = 0; gamma < 4; gamma++){
            for(size_t delta = 0; delta < 4; delta++){
                d2qdw2[theta](alpha,beta) += quaternion_product_tensor(alpha,gamma,delta) * d2expdw2[theta](gamma,beta) * q_n[delta];
            }
          }
        }
      }
    }
    return d2qdw2;
}



inline Eigen::Vector3d QuaternionRotate(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    Eigen::Vector4d a = q;
    Eigen::Vector4d v_quat(double(0), v[0], v[1], v[2]);
    Eigen::Vector4d b = QuaternionMultiply(v_quat, ConjugateQuaternion(q));
    Eigen::Vector4d quat_product(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
    return Eigen::Vector3d(quat_product[1], quat_product[2], quat_product[3]);
}

// Exponential map: angular velocity * dt -> unit quaternion
inline Eigen::Vector4d QuaternionFromVector(const Eigen::Vector3d& w) {
    double omega = w.norm();
    Eigen::Vector4d q;
    if (omega < 1e-10) {
        q = {double(1), double(0), double(0), double(0)};
    }
    else{
        double s = std::sin(double(0.5) * omega);
        q = {std::cos(double(0.5) * omega), s * w[0] / omega, s * w[1] / omega, s * w[2] / omega};
    }
    return q;
}

// Rotation matrix corresponding to a unit quaternion
inline Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d& q) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    Eigen::Matrix3d R;
    R(0, 0) = double(1) - double(2) * y * y - double(2) * z * z;  R(0, 1) = double(2) * (x * y - w * z);      R(0, 2) = double(2) * (x * z + w * y);
    R(1, 0) = double(2) * (x * y + w * z);       R(1, 1) = double(1) - double(2) * x * x - double(2) * z * z; R(1, 2) = double(2) * (y * z - w * x);
    R(2, 0) = double(2) * (x * z - w * y);       R(2, 1) = double(2) * (y * z + w * x);       R(2, 2) = double(1) - double(2) * x * x - double(2) * y * y;
    return R;
}

// Unit quaternion corresponding to a (proper) rotation matrix
inline Eigen::Vector4d QuaternionFromRotationMatrix(const Eigen::Matrix3d& R) {
    Eigen::Vector4d q;
    Assert(R.determinant() > double(0), "ALGEBRA: input must be a rotation matrix.");

    double trace = R.trace();
    if (trace > double(0)) { // I change M_EPSILON to 0
        double s = double(0.5) / std::sqrt(trace + double(1.0));
        q[0] = double(0.25) / s;
        q[1] = (R(2, 1) - R(1, 2)) * s;
        q[2] = (R(0, 2) - R(2, 0)) * s;
        q[3] = (R(1, 0) - R(0, 1)) * s;
    }
    else {
        if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
            double s = double(2) * std::sqrt(double(1.0) + R(0, 0) - R(1, 1) - R(2, 2));
            q[0] = (R(2, 1) - R(1, 2)) / s;
            q[1] = double(0.25) * s;
            q[2] = (R(0, 1) + R(1, 0)) / s;
            q[3] = (R(0, 2) + R(2, 0)) / s;
        }
        else if (R(1, 1) > R(2, 2)) {
            double s = double(2) * std::sqrt(double(1.0) + R(1, 1) - R(0, 0) - R(2, 2));
            q[0] = (R(0, 2) - R(2, 0)) / s;
            q[1] = (R(0, 1) + R(1, 0)) / s;
            q[2] = double(0.25) * s;
            q[3] = (R(1, 2) + R(2, 1)) / s;
        }
        else {
            double s = double(2) * std::sqrt(double(1.0) + R(2, 2) - R(0, 0) - R(1, 1));
            q[0] = (R(1, 0) - R(0, 1)) / s;
            q[1] = (R(0, 2) + R(2, 0)) / s;
            q[2] = (R(1, 2) + R(2, 1)) / s;
            q[3] = double(0.25) * s;
        }
    }
    q.normalize();
    return q;
}

inline double SubFrobeniusNorm(const Eigen::MatrixXd& A, const std::array<int,2>& i_range, const std::array<int,2>& j_range) {
    //returns Aij Aij for i_range[0] <= i <= i_range[1] and j_range[0] <= j <= j_range[1]
    Assert(i_range[1]>=i_range[0] && i_range[0]>=0 && i_range[1]<int(A.rows()),"ALGEBRA::SubFrobeniusNorm: i_ranges are degenerate");
    Assert(j_range[1]>=j_range[0] && j_range[0]>=0 && j_range[1]<int(A.cols()),"ALGEBRA::SubFrobeniusNorm: j_ranges are degenerate");

    std::vector<double> column_sums(size_t(j_range[1] - j_range[0] + 1));
    for(size_t j=size_t(j_range[0]);j<size_t(j_range[1]+1);j++){
        double result = double(0);
        #pragma omp parallel for reduction(+: result)
        for(int i=int(i_range[0]);i<int(i_range[1]+1);++i){
            result += A(size_t(i),j)*A(size_t(i),j);
        }
        column_sums[j-j_range[0]] = result;
    }
    double norm_squared=double(0);
    #pragma omp parallel for reduction(+: norm_squared)
    for(int i = 0; i < int(column_sums.size()); ++i) {
        norm_squared += column_sums[i];
    }

    return std::sqrt(norm_squared);
}

inline void HouseholderTriDiagonalize(const Eigen::MatrixXd& A, std::vector<std::vector<double>>& housedholder_vectors, std::vector<double>& diagonal, std::vector<double>& subdiagonal, const double tol = double(1e-10)) {
  std::array<int,2> dims = {int(A.rows()),int(A.cols())};
  Assert(dims[0] == dims[1], "ALGEBRA::HouseholderTriDiagonalize: only works for symmetric square matrices");
  Assert(dims[0] > 2, "ALGEBRA::HouseholderTriDiagonalize: only makes sense for mxm matrices with m>2");
  size_t m = size_t(dims[0]);
  housedholder_vectors.resize(m - 2);
  diagonal.resize(m, double(0));
  subdiagonal.resize(m - 1, double(0));
  Eigen::MatrixXd B(m, m);
  B = A;

  // get the Householder vector
  for (size_t j = 0; j < m - 2; j++) {
    housedholder_vectors[j].resize(m - j - 1);
    std::fill(std::begin(housedholder_vectors[j]), std::end(housedholder_vectors[j]), double(0));

    diagonal[j] = B(j, j);
    double beta = SubFrobeniusNorm(B,{int(j + 1), int(m - 1)}, {int(j), int(j)});
    // serial version
    /*T beta=T(0);
    for(sz i=j+1;i<m;i++){
            beta+=B(i,j)*B(i,j);
    }
    beta=std::sqrt(beta);*/

    double sign = B(j + 1, j) < double(0) ? double(-1) : double(1);
    housedholder_vectors[j][0] = B(j + 1, j) + sign * beta;
    double D_mag = housedholder_vectors[j][0] * housedholder_vectors[j][0];
    B(j + 1, j) = -sign * beta;
    subdiagonal[j] = B(j + 1, j);

    double result = double(0);
 		#pragma omp parallel for reduction(+: result)
    for (int i = int(j+2); i < int(m); ++i) {
      housedholder_vectors[j][i - (j + 1)] = B(i, j);
      result += housedholder_vectors[j][i - (j + 1)] * housedholder_vectors[j][i - (j + 1)];
      B(i, j) = double(0);
    }
    D_mag += result;

    D_mag = std::sqrt(D_mag);

    // serial version
    /*for(sz i=j+2;i<m;i++){
            housedholder_vectors[j][i-(j+1)]=B(i,j);
            D_mag+=housedholder_vectors[j][i-(j+1)]*housedholder_vectors[j][i-(j+1)];
            B(i,j)=T(0);
    }
    D_mag=std::sqrt(D_mag);*/

    if (D_mag > tol) {
      #pragma omp parallel for
      for(int i=int(j+1);i<int(m);++i){
        housedholder_vectors[j][i - (j + 1)] /= D_mag;
      }

      // serial version
      /*for(sz i=j+1;i<m;i++){
              housedholder_vectors[j][i-(j+1)]/=D_mag;
      }*/
      //////////////////////////////

      // apply projection to remaining columns of A: (A_j^hat)
      #pragma omp parallel for
      for(int k=int(j+1);k<int(m);++k){
        double dot = double(0);
        for (size_t i = j + 1; i < m; i++) {
          dot += B(i, k) * housedholder_vectors[j][i - (j + 1)];
        }
        for (size_t i = j + 1; i < m; i++) {
          B(i, k) -= double(2) * dot * housedholder_vectors[j][i - (j + 1)];
        }
      }
      // serial version
      /*for(sz k=j+1;k<m;k++){
              T dot=T(0);
              for(sz i=j+1;i<m;i++){
                      dot+=B(i,k)*housedholder_vectors[j][i-(j+1)];
              }
              for(sz i=j+1;i<m;i++){
                      B(i,k)-=T(2)*dot*housedholder_vectors[j][i-(j+1)];
              }
      }*/

      // now do the same from the right
      #pragma omp parallel for
      for (int k=int(j+1);k<int(m);++k){
        double dot = double(0);
        for (size_t i = j + 1; i < m; i++) {
          dot += B(k, i) * housedholder_vectors[j][i - (j + 1)];
        }
        for (size_t i = j + 1; i < m; i++) {
          B(k, i) -= double(2) * dot * housedholder_vectors[j][i - (j + 1)];
        }
      }

      // serial version (Q_J A_j^hat) Q_j^T = (Q_j (Q_J A_j^hat)^T )^T
      /*for(sz k=j+1;k<m;k++){
              T dot=T(0);
              for(sz i=j+1;i<m;i++){
                      dot+=B(k,i)*housedholder_vectors[j][i-(j+1)];
              }
              for(sz i=j+1;i<m;i++){
                      B(k,i)-=T(2)*dot*housedholder_vectors[j][i-(j+1)];
              }
      }*/
    }
  }
  diagonal[m - 2] = B(m - 2, m - 2);
  diagonal[m - 1] = B(m - 1, m - 1);
  subdiagonal[m - 2] = B(m - 1, m - 2);
}

inline void GivensRotate(const Eigen::Vector2d& givens, Eigen::Vector2d& rotated){
  double x=rotated[0], y=rotated[1];
  double c=givens[0], s=givens[1];
  rotated={c*x - s*y, s*x + c*y};
}

inline double Givens(const Eigen::Vector2d& x, Eigen::Vector2d& givens, const double tol = double(1e-13)){
  double norm=std::sqrt(x[0]*x[0] + x[1]*x[1]);
  if(norm>tol)
    givens={x[0]/norm,-x[1]/norm};
  else
    givens={double(1),0};
  return norm;
}

inline void GivensSymmetricTridiagonalQR(const std::vector<double>& diagonal, const std::vector<double>& sub_diagonal, std::vector<Eigen::Vector2d>& givens_rotations, std::vector<double>& r_diag, std::vector<double>& r_super_diag, std::vector<double>& r_super_super_diag) {
  /*
          Input:
                          Symmetric tridiagonal matrix T[n], represented by "diagonal", "sub_diagonal"
          Output:
                          1. Givens rotations G0,...,Gn-2 that construct Q = G0*...*Gn-2
                          2. Upper triangular matrix R, represented by "r_diag", "r_super_diag", "r_super_super_diag"
          Comment:
                          T = QR is the QR decomposition
  */
  size_t m = diagonal.size();
  if (m == 1) {  // trivial case, but let's support it
    r_diag.resize(m);
    r_diag[0] = diagonal[0];

    r_super_diag.resize(0);
    r_super_super_diag.resize(0);
    givens_rotations.resize(0);
    return;
  }
  Assert(sub_diagonal.size() == m - 1 && m > 0, "ALGEBRA::GivensTridiagonalQR: input symmetric tridiagonal matrix is sized incorrectly");
  // size outputs
  givens_rotations.resize(m - 1);
  r_diag.resize(m);
  r_super_diag.resize(m - 1);
  m > 1 ? r_super_super_diag.resize(m - 2) : r_super_super_diag.resize(0);

  double alpha_hat = diagonal[0], beta_hat = sub_diagonal[0];
  for (size_t j = 0; j < m - 1; j++) {
    r_diag[j] = Givens({alpha_hat, sub_diagonal[j]}, givens_rotations[j]);
    Eigen::Vector2d rotated = {beta_hat, diagonal[j + 1]};
    GivensRotate(givens_rotations[j], rotated);
    r_super_diag[j] = rotated[0];
    alpha_hat = rotated[1];
    if (j < m - 2) {
      Eigen::Vector2d rotated = {double(0), sub_diagonal[j + 1]};
      GivensRotate(givens_rotations[j], rotated);
      r_super_super_diag[j] = rotated[0];
      beta_hat = rotated[1];
    }
  }
  r_diag[m - 1] = alpha_hat;
}

inline void GivensSymmetricTridiagonalRQ(const std::vector<Eigen::Vector2d>& givens_rotations, const std::vector<double>& r_diag, const std::vector<double>& r_super_diag, const std::vector<double>& r_super_super_diag, std::vector<double>& diagonal, std::vector<double>& sub_diagonal) {
  /*
          Input:
                        1. Givens rotations G0,...,Gn-2, Q = G0*...*Gn-2
                        2. Upper triangular matrix R, represented by "r_diag", "r_super_diag", "r_super_super_diag"
          Output:
                        1. Symmetric tridiagonal matrix T_hat = RQ, represented by "diagonal", "sub_diagonal"
                        2. T_hat = QtTQ
  */
  size_t m = r_diag.size();
  Assert(givens_rotations.size() == m - 1 && r_super_diag.size() == m - 1 && m >= 2, "ALGEBRA::GivensTridiagonalRQ: input R and/or Q are/is sized incorrectly");
  if (m > 2) {
    Assert(r_super_super_diag.size() == m - 2, "ALGEBRA::GivensTridiagonalRQ: input R and/or Q are/is sized incorrectly");
  }
  // size outputs
  diagonal.resize(m);
  sub_diagonal.resize(m - 1);

  // subdiagonal
  #pragma omp parallel for
  for(int i=0;i<int(m-1);++i){
    sub_diagonal[i] = -r_diag[i + 1] * givens_rotations[i][1];
  }

  // diagonal
  diagonal[0] = r_diag[0] * givens_rotations[0][0] - r_super_diag[0] * givens_rotations[0][1];
  #pragma omp parallel for
  for (int i=1;i<int(m-1);++i){
    diagonal[i] = r_diag[i] * givens_rotations[i][0] * givens_rotations[i - 1][0] - r_super_diag[i] * givens_rotations[i][1];
  }
  diagonal[m - 1] = r_diag[m - 1] * givens_rotations[m - 2][0];
}

inline void QRIteration(const Eigen::MatrixXd& A, std::vector<std::vector<double>>& housedholder_vectors, std::vector<std::vector<Eigen::Vector2d>>& givens_rotations, std::vector<double>& lambda, const size_t max_it = 3,
                        bool explicit_shifts = true) {
  /*
          input:
                  A: 						A = Q * lambda * Q.Transpose()
          output:
                  lambda: 					eigenvalues, Q * lambda * Q.Transpose(), Q = Qh * Qg
                  householder_vectors:  	Qh^T * A * Qh = tri_diag where Qh is constructed from the householder_vectors using
                                                                                                    ConstructQFromHouseholderVectors
                  givens_rotations: 		vector of the givens rotations used in each iteration of QR, Qg = Q0 * ... * Qit where
                                                                                                          it = givens_rotations.size() and Qi is constructed using
     ConstructQFromHouseholderVectors(givens_rotations[i])
  */

  std::array<int,2> dims = {int(A.rows()), int(A.cols())};
  Assert(dims[0] == dims[1] && dims[0] >= 3, "QRIteration: symmetric eigen problem can only be done with a square matrix.");
  Assert(dims[0] >= 3, "QRIteration: matrix dimension is less than 3, you sould use a different eigen solver.");
  size_t m = size_t(dims[0]);

  // first reduce to tridiagonal form
  std::vector<double> sub_diagonal;
  HouseholderTriDiagonalize(A, housedholder_vectors, lambda, sub_diagonal);

  // now do QR iteration on the tridiagonal form
  std::vector<double> r_diag, r_super_diag, r_super_super_diag;
  givens_rotations.resize(max_it);
  double mu = double(0);
  for (size_t it = 0; it < max_it; it++) {
    if (explicit_shifts) {
      // Wilkinson shift
      double a = lambda[m - 2], b = sub_diagonal[m - 2], c = lambda[m - 1];
      double delta = double(.5) * (a - c);
      double sign = (delta < 0 ? double(-1) : double(1));
      double denom=(std::abs(delta) + std::sqrt(delta * delta + b * b));
      if(denom>double(1e-4)){
        mu = c - sign * b * b / denom;
      }
      else
        mu=double(0);
      // explicit shift
      #pragma omp parallel for
      for (int k=0;k<int(m);++k){
        lambda[k] -= mu;
      }
    }

    std::vector<Eigen::Vector2d> Git;
    ALGEBRA::GivensSymmetricTridiagonalQR(lambda, sub_diagonal, Git, r_diag, r_super_diag, r_super_super_diag);
    givens_rotations[it] = Git;
    ALGEBRA::GivensSymmetricTridiagonalRQ(Git, r_diag, r_super_diag, r_super_super_diag, lambda, sub_diagonal);
    if (explicit_shifts) {
      // explicit shift
      #pragma omp parallel for
      for(int k=0;k<int(m);++k){
        lambda[k] += mu;
      }
    }
  }
}

inline void ConstructQFromGivensRotations(const std::vector<Eigen::Vector2d>& givens_rotations, const size_t m, Eigen::MatrixXd& Q) {
  /*
          Input:
                        m:	square matrix size
                        Givens rotations: size = m - 1, givens_rotations[k] = {c_k,s_k}
          Output:
                        Q  = G0'*G1'*...Gn-2'
                        Gk = [I_k 						 ]
                             [  	c_k, -s_k 			 ]
                             [  	s_k, c_k			 ]
                             [ 					I_{n-2-k}]
  */

  Assert(m > 1 && givens_rotations.size() == m - 1, "TGSL::ALGEBRA::ConstructQFromGivensRotations: matrix not sized correctly.");
  // set Q to identity initially
  Q = Eigen::MatrixXd::Zero(m,m);
  for (size_t i = 0; i < m; i++)
    Q(i, i) = double(1);

  for (int k = m - 2; k >= 0; k--) {
    for (size_t j = 0; j < m; j++) {
      Eigen::Vector2d Gk = {givens_rotations[k][0], -givens_rotations[k][1]}, rotated = {Q(k, j), Q(k + 1, j)};
      GivensRotate(Gk, rotated);
      Q(k, j) = rotated[0];
      Q(k + 1, j) = rotated[1];
    }
  }
}

inline void ConstructQFromHouseholderVectors(const std::vector<std::vector<double>>& housedholder_vectors, const size_t m, Eigen::MatrixXd& Q) {
  /*
          Input:	m = square matrix size
                          housedholder_vectors: size = n, Ni = housedholder_vectors[i].size() is between 0 and m,
                                          the convention is that the real Householder vector is [0,...,0,housedholder_vectors[i][0],...housedholder_vectors[i][Ni-1]]^T
                          Q = Q0*Q1*...Qn-1
  */

  size_t n = housedholder_vectors.size();
  Assert(n > 0, "ALGEBRA::ConstructQFromHouseholderVectors: housedholder_vectors not sized correctly.");

  // set Q to identity initially
  Q = Eigen::MatrixXd::Zero(m,m);
  for (size_t i = 0; i < m; i++)
    Q(i, i) = double(1);

  // multiply Q by the Qk (Householder projections)
  for (int k = int(n - 1); k >= 0; k--) {
    // multiply each Qk times each column (j) of Q
    Assert(housedholder_vectors[k].size() <= m, "ALGEBRA::ConstructQFromHouseholderVectors: housedholder_vectors not sized correctly.");
    size_t i_start = m - housedholder_vectors[k].size();
    for (size_t j = 0; j < m; j++) {
      double dot = double(0);
      for (size_t i = i_start; i < m; i++) {
        dot += Q(i, j) * housedholder_vectors[k][i - i_start];
      }
      for (size_t i = i_start; i < m; i++) {
        Q(i, j) -= double(2) * dot * housedholder_vectors[k][i - i_start];
      }
    }
  }
}

inline void ConstructQFromQRGivensAndHouseholder(const size_t m, std::vector<std::vector<double>>& housedholder_vectors, std::vector<std::vector<Eigen::Vector2d>>& givens_rotations, Eigen::MatrixXd& Q) {
  Q = Eigen::MatrixXd(m,m);

  Eigen::MatrixXd QG(m, m);
  for (size_t i = 0; i < m; i++)
    QG(i, i) = double(1);

  for (size_t i = 0; i < givens_rotations.size(); i++) {
    Eigen::MatrixXd QGi(m, m);
    ConstructQFromGivensRotations(givens_rotations[i], m, QGi);
    QG = QG * QGi;
  }

  Eigen::MatrixXd QH(m, m);
  ConstructQFromHouseholderVectors(housedholder_vectors, m, QH);
  Q = QH * QG;
}

inline void SymmetricEigenDecomposition(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, std::vector<double>& lambda, const size_t max_it = 10, bool explicit_shifts = false) {
    std::array<int,2> dims = {int(A.rows()), int(A.cols())};
    Assert(dims[0] == dims[1], "SymmetricEigenDecomposition: Matrix isn't square.");
    size_t m = size_t(dims[0]);
    std::vector<std::vector<double>> housedholder_vectors;
    std::vector<std::vector<Eigen::Vector2d>> givens_rotations;
    QRIteration(A, housedholder_vectors, givens_rotations, lambda, max_it, explicit_shifts);
    ConstructQFromQRGivensAndHouseholder(m, housedholder_vectors, givens_rotations, Q);
}

  inline std::array<int,2> MaxAbsEntry(const Eigen::MatrixXd& A, const std::array<int,2>& i_range,const std::array<int,2>& j_range){
      std::array<int,2>result={i_range[0],j_range[0]};
      double max_abs=double(0);
      for(size_t i=size_t(i_range[0]);i<size_t(i_range[1]);i++){
        for(size_t j=size_t(j_range[0]);j<size_t(j_range[1]);j++){
          if(abs(A(i,j))>max_abs){
            result={int(i),int(j)};
            max_abs=abs(A(i,j));
          }
        }
      }
      return result;
  }

  inline void SwapRows(const std::array<int,2>& row_swap, Eigen::MatrixXd& A){
      Assert(size_t(row_swap[0])<size_t(A.rows()) && size_t(row_swap[0])<size_t(A.rows()) && size_t(row_swap[0])>=0 && size_t(row_swap[1])>=0,"ALGEBRA::SwapRows: rows not in dims of matrix.");

      if(row_swap[0]==row_swap[1]) 
        return;

      for(size_t j=0;j<size_t(A.cols());j++){
        double temp=A(row_swap[0], j);
        A(row_swap[0], j)=A(row_swap[1] + j);
        A(row_swap[1] + j)=temp;
      }
  }

inline void PLU(const Eigen::MatrixXd& A, std::vector<std::array<int,2>>& row_swaps, Eigen::MatrixXd& LU){
    std::array<int,2> dims = {int(A.rows()), int(A.cols())};
    Assert(dims[0]==dims[1],"PLU only supports square matrices.");
    LU=A;
    size_t n=dims[0];
    
    row_swaps.resize(n-1);
    
    for(size_t k=0;k<n-1;k++){
      std::array<int,2> pivot_entry= MaxAbsEntry(LU,{int(k),int(n)},{int(k),int(k+1)});
      row_swaps[k]={int(k),pivot_entry[0]};
      Assert(abs(LU(pivot_entry[0],k)),"PLU: rank deficient matrix.");
      SwapRows(row_swaps[k], LU);
      for(size_t i=k+1;i<n;i++){
        LU(i,k)=LU(i,k)/LU(k,k);
        #pragma omp parallel for
        for(int j=int(k+1);j<int(n);++j){
            LU(i,j)-=LU(i,k)*LU(k,j);
          }
      }
    }
}

inline void PLUSolve(const Eigen::MatrixXd& LU, const std::vector<std::array<int,2>>& row_swaps, const std::vector<double>& b, std::vector<double>& x){
	std::array<int,2> dims={int(LU.rows()),int(LU.cols())};
	Assert(dims[0]==dims[1],"PLU only supports square matrices.");
	Assert(size_t(dims[1])==b.size(),"PLUSolve: rhs dims do not match matrix dims.");
	size_t n=dims[1];
	x.resize(n);

	//P^T
	std::vector<double> permuted_rhs=b;
	for(size_t i=0;i<n-1;i++){
		double temp=permuted_rhs[row_swaps[i][0]];
		permuted_rhs[row_swaps[i][0]]=permuted_rhs[row_swaps[i][1]];
		permuted_rhs[row_swaps[i][1]]=temp;
	}

	//forward substitution
	std::vector<double> y(n);
	for(size_t i=0;i<n;i++){
		y[i]=permuted_rhs[i];
		double result = double(0);
		#pragma omp parallel for reduction(+: result)
		for (int j = 0; j < int(i); ++j) {
			result += LU(i,j)*y[j];
		}
		y[i] -= result;
		/*for(nm j=0;j<nm(i);j++){//serial version
			y[i]-=LU(i,j)*y[j];
		}*/
	}

	//back substitution
	for(int i=n-1;i>=0;i--){
		x[i]=y[i];
		double result = double(0);
		#pragma omp parallel for reduction(+: result)
		for (int j = int(i+1); j < int(n); ++j) {
			result += LU(i,j)*x[j];
		}
		x[i]-= result;
		/*for(sz j=i+1;j<n;j++){//serial version
			x[i]-=LU(i,j)*x[j];
		}*/
		Assert(abs(LU(i,i))>0,"PLUSolve: system is rank deficient.");
		x[i]/=LU(i,i);
	}

}

inline void PLUSolve(const Eigen::MatrixXd& A, const std::vector<double>& b, std::vector<double>& x){
	std::array<int,2> dims = {int(A.rows()), int(A.cols())};
	Assert(dims[0]==dims[1],"PLU only supports square matrices.");
	Assert(size_t(dims[1])==b.size(),"PLUSolve: rhs dims do not match matrix dims.");
	size_t n=dims[1];
	x.resize(n);
	
	Eigen::MatrixXd LU(n,n);
	std::vector<std::array<int,2>> row_swaps;
	PLU(A,row_swaps,LU);

	PLUSolve(LU, row_swaps, b, x);

}

}  // namespace ALGEBRA
}  // namespace Rigid_Body
