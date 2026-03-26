#include "Corotated32.h"

namespace TGSL {
void Corotated32Cache::UpdateCache(const ALGEBRA::DenseMatrix& F, const bool update_damping){
  ALGEBRA::DenseMatrix U, D, R;
  ALGEBRA::DenseMatrix FTF = F.Transpose() * F;
  Jacobi(FTF, D, U);

  ALGEBRA::DenseMatrix S_mat = ALGEBRA::Identity(2);
  S_mat(0, 0) = std::sqrt(D(0,0));
  S_mat(1, 1) = std::sqrt(D(1,1));
  ALGEBRA::DenseMatrix S = U * S_mat * U.Transpose();
  R = F * S.Inverse();
  R_cache = {R(0,0),R(0,1),R(1,0),R(1,1),R(2,0),R(2,1)};
  J_cache = T(S_mat(0, 0) * S_mat(1, 1));
  ALGEBRA::DenseMatrix FTFinv = FTF.Inverse();
  FTFinv_cache = {FTFinv(0,0), FTFinv(0,1), FTFinv(1,0), FTFinv(1,1)};
  trace_S_cache=S.Trace();
  if (update_damping) {
    Rn_cache=R_cache;
    Fn_cache={F(0,0),F(0,1),F(1,0),
              F(1,1),F(2,0),F(2,1)};
  }

}

void Corotated32Cache::P(const ALGEBRA::DenseMatrix& F, const T mu, const T lambda, ALGEBRA::DenseMatrix& P){
  TGSLAssert(F.m == 3 && F.n == 2, "Corotated32Cache::P: F needs to be 3*2.");
  ALGEBRA::DenseMatrix R(3,2), FTFinv(2,2);
  R(0,0) = R_cache[0];
  R(0,1) = R_cache[1];
  R(1,0) = R_cache[2];
  R(1,1) = R_cache[3];
  R(2,0) = R_cache[4];
  R(2,1) = R_cache[5];
  FTFinv(0,0) = FTFinv_cache[0];
  FTFinv(0,1) = FTFinv_cache[1];
  FTFinv(1,0) = FTFinv_cache[2];
  FTFinv(1,1) = FTFinv_cache[3];
  P = T(2) * T(mu) * (F - R) + T(lambda) * T(J_cache - T(1)) * J_cache * F * FTFinv;
}
void Corotated32Cache::deltaP(const ALGEBRA::DenseMatrix& F, const ALGEBRA::DenseMatrix& dF, const T mu, const T lambda, ALGEBRA::DenseMatrix& dP){
  TGSLAssert(F.m == 3 && F.n == 2, "Corotated32Cache::deltaP: F needs to be 3*2.");
  ALGEBRA::DenseMatrix R(3,2), FTFinv(2,2);
  R(0,0) = R_cache[0];
  R(0,1) = R_cache[1];
  R(1,0) = R_cache[2];
  R(1,1) = R_cache[3];
  R(2,0) = R_cache[4];
  R(2,1) = R_cache[5];
  FTFinv(0,0) = FTFinv_cache[0];
  FTFinv(0,1) = FTFinv_cache[1];
  FTFinv(1,0) = FTFinv_cache[2];
  FTFinv(1,1) = FTFinv_cache[3];
  ALGEBRA::DenseMatrix S = R.Transpose()*F;
  ALGEBRA::DenseMatrix RTdF = R.Transpose()*dF;
  T c = (RTdF(0,1)-RTdF(1,0))/trace_S_cache;
  ALGEBRA::DenseMatrix RTdR(2, 2);
  RTdR(0, 0) = T(0);
  RTdR(0, 1) = c;
  RTdR(1, 0) = -c;
  RTdR(1, 1) = T(0);
  ALGEBRA::DenseMatrix dS = RTdF - RTdR*S;
  ALGEBRA::DenseMatrix dR = dF*S.Inverse() - R*dS*S.Inverse();
  ALGEBRA::DenseMatrix deltaE = dF.Transpose() * F + F.Transpose() * dF;
  dP = T(2) * T(mu) * dF - T(2) * T(mu) * dR + T(lambda) * (J_cache - T(1)) * J_cache * dF * FTFinv - T(lambda) * (J_cache - T(1)) * J_cache * F * FTFinv * (deltaE) * FTFinv +
      0.5 * T(lambda) * (2.0 * J_cache - T(1)) * J_cache * (FTFinv * deltaE).Trace() * F * FTFinv;
  
}

void Corotated32Cache::dPdF(const ALGEBRA::DenseMatrix& F,const T mu, const T lambda, ALGEBRA::DenseMatrix& dPdF) {
  TGSLAssert(F.m == 3 && F.n == 2, "Corotated32Cache::deltaP: F needs to be 3*2.");
  ALGEBRA::DenseMatrix R(3,2), FTFinv(2,2);
  R(0,0) = R_cache[0];
  R(0,1) = R_cache[1];
  R(1,0) = R_cache[2];
  R(1,1) = R_cache[3];
  R(2,0) = R_cache[4];
  R(2,1) = R_cache[5];
  FTFinv(0,0) = FTFinv_cache[0];
  FTFinv(0,1) = FTFinv_cache[1];
  FTFinv(1,0) = FTFinv_cache[2];
  FTFinv(1,1) = FTFinv_cache[3];
  ALGEBRA::DenseMatrix S = R.Transpose()*F;
  ALGEBRA::DenseMatrix SInv = S.Inverse();
  ALGEBRA::DenseMatrix RRT = R*R.Transpose();
  ALGEBRA::DenseMatrix Re(3,2);
  Re(0,0) = R(0,1);
  Re(0,1) = -R(0,0);
  Re(1,0) = R(1,1);
  Re(1,1) = -R(1,0);
  Re(2,0) = R(2,1);
  Re(2,1) = -R(2,0);
  TV dcdF = {-R(0,1)/trace_S_cache, R(0,0)/trace_S_cache, -R(1,1)/trace_S_cache, R(1,0)/trace_S_cache, -R(2,1)/trace_S_cache, R(2,0)/trace_S_cache};
  ALGEBRA::DenseMatrix dRdF(6,6);
  IV indices = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1};
  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      dRdF(c1, c2) = 0;
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m == i) {
        dRdF(c1, c2) += SInv(j, n);
      }
      dRdF(c1, c2) -= RRT(m, i) * SInv(j ,n);
      dRdF(c1, c2) -= dcdF[c2]*Re(m,n);
    }
  }

  ALGEBRA::DenseMatrix FFTFInv = F*FTFinv;
  ALGEBRA::DenseMatrix FFTFInvFT = F*FTFinv*F.Transpose();

  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m > 0 && i > 0) {}
      dPdF(c1, c2) = 0;
      if (m == i){
        dPdF(c1, c2) += T(lambda) * (J_cache - T(1)) * J_cache *FTFinv(j,n);
      }
      dPdF(c1, c2) -= T(lambda) * (J_cache - T(1)) * J_cache * (FFTFInv(m, j)*FFTFInv(i, n)+FFTFInvFT(m, i) * FTFinv(j,n));
      dPdF(c1, c2) += 0.5 * T(lambda) * (2.0 * J_cache - T(1)) * J_cache * (FFTFInv(i, j)*FFTFInv(m, n)+ FFTFInv(i, j)*FFTFInv(m, n));
    }
  }
  
  dPdF =  dPdF+ T(2) * T(mu)*(ALGEBRA::Identity(6) -dRdF);

}

void Corotated32Cache::PRayleignDamping(const ALGEBRA::DenseMatrix& dVdX, const T alpha_rd, const T mu, const T lambda, ALGEBRA::DenseMatrix& P) {
  TGSLAssert(dVdX.m == 3 && dVdX.n == 2, "Corotated32Cache::PRayleignDamping: dVdX needs to be 3*2.");
  ALGEBRA::DenseMatrix R(3,2);
  R(0,0) = Rn_cache[0];
  R(0,1) = Rn_cache[1];
  R(1,0) = Rn_cache[2];
  R(1,1) = Rn_cache[3];
  R(2,0) = Rn_cache[4];
  R(2,1) = Rn_cache[5];
  ALGEBRA::DenseMatrix RTdF = R.Transpose()*dVdX;
  T c = (RTdF(0,1)-RTdF(1,0))/T(2);
  ALGEBRA::DenseMatrix RTdR(2, 2);
  RTdR(0, 0) = T(0);
  RTdR(0, 1) = c;
  RTdR(1, 0) = -c;
  RTdR(1, 1) = T(0);
  ALGEBRA::DenseMatrix dS = RTdF - RTdR;
  ALGEBRA::DenseMatrix dR = dVdX - R*dS;
  ALGEBRA::DenseMatrix deltaE = dVdX.Transpose() * R + R.Transpose() * dVdX;
  P = T(2) * T(mu) * dVdX - T(2) * T(mu) * dR  +
      0.5 * T(lambda) * (deltaE).Trace() * R ;
  P = P * alpha_rd;
}


void Corotated32Cache::dPdFRayleignDamping(const ALGEBRA::DenseMatrix& F, const T alpha_rd, const T mu, const T lambda, ALGEBRA::DenseMatrix& dPdF){
  TGSLAssert(F.m == 3 && F.n == 2, "Corotated32Cache::deltaP: F needs to be 3*2.");
  ALGEBRA::DenseMatrix R(3,2), FTFinv(2,2);
  R(0,0) = Rn_cache[0];
  R(0,1) = Rn_cache[1];
  R(1,0) = Rn_cache[2];
  R(1,1) = Rn_cache[3];
  R(2,0) = Rn_cache[4];
  R(2,1) = Rn_cache[5];
  ALGEBRA::DenseMatrix Id = ALGEBRA::Identity(2);
  ALGEBRA::DenseMatrix RRT = R*R.Transpose();
  ALGEBRA::DenseMatrix Re(3,2);
  Re(0,0) = R(0,1);
  Re(0,1) = -R(0,0);
  Re(1,0) = R(1,1);
  Re(1,1) = -R(1,0);
  Re(2,0) = R(2,1);
  Re(2,1) = -R(2,0);
  TV dcdF = {-R(0,1)/T(2), R(0,0)/T(2), -R(1,1)/T(2), R(1,0)/T(2), -R(2,1)/T(2), R(2,0)/T(2)};
  ALGEBRA::DenseMatrix dRdF(6,6);
  IV indices = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1};
  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      dRdF(c1, c2) = 0;
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m == i) {
        dRdF(c1, c2) += Id(j, n);
      }
      dRdF(c1, c2) -= RRT(m, i) * Id(j ,n);
      dRdF(c1, c2) -= dcdF[c2]*Re(m,n);
    }
  }

  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m > 0 && i > 0) {}
      dPdF(c1, c2) = 0;
      dPdF(c1, c2) += 0.5 * T(lambda) * (R(i, j)*R(m, n)+ R(i, j)*R(m, n));
    }
  }
  
  dPdF = dPdF+ T(2) * T(mu)*(ALGEBRA::Identity(6) -dRdF);
  dPdF = dPdF*alpha_rd;
}

void Corotated32Cache::UpdateCache(const Eigen::Matrix<T, d, d-1>& F, const bool update_damping){
  Eigen::Matrix<T, d-1, d-1> U, D;
  Eigen::Matrix<T, d, d-1> R;
  Eigen::Matrix<T, d-1, d-1> FTF = F.transpose() * F;
  //comment out in 2D, otherwise the code doesn't compile
  #ifdef TWO_D
  U(0) = T(1);
  D(0) = FTF(0);
  #else
  Jacobi(FTF, D, U);
  Eigen::Matrix<T, d-1, d-1> S_mat = Eigen::Matrix<T, d-1, d-1>::Identity();
  S_mat(0, 0) = std::sqrt(D(0,0));
  S_mat(1, 1) = std::sqrt(D(1,1));
  Eigen::Matrix<T, d-1, d-1> S = U * S_mat * U.transpose();
  R = F * S.inverse();
  R_cache = {R(0,0),R(0,1),R(1,0),R(1,1),R(2,0),R(2,1)};
  J_cache = T(S_mat(0, 0) * S_mat(1, 1));
  Eigen::Matrix<T, d-1, d-1> FTFinv = FTF.inverse();
  FTFinv_cache = {FTFinv(0,0), FTFinv(0,1), FTFinv(1,0), FTFinv(1,1)};
  trace_S_cache=S.trace();
  if (update_damping) {
    Rn_cache=R_cache;
    Fn_cache={F(0,0),F(0,1),F(1,0),
              F(1,1),F(2,0),F(2,1)};
  }
  #endif

}

void Corotated32Cache::P(const Eigen::Matrix<T, d, d-1>& F, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& P){
  #ifndef TWO_D
  //TGSLAssert(d == 3, "Corotated32Cache::PEigen: F needs to be 3*2.");
  Eigen::Matrix<T, d, d-1> R;
  // Eigen::Matrix<T, d-1, d-1> FTFinv;
  R(0,0) = R_cache[0];
  R(0,1) = R_cache[1];
  R(1,0) = R_cache[2];
  R(1,1) = R_cache[3];
  R(2,0) = R_cache[4];
  R(2,1) = R_cache[5];
  Eigen::Matrix<T, d-1, d-1> FTFinv;
  FTFinv(0,0) = FTFinv_cache[0];
  FTFinv(0,1) = FTFinv_cache[1];
  FTFinv(1,0) = FTFinv_cache[2];
  FTFinv(1,1) = FTFinv_cache[3];
  P = T(2) * T(mu) * (F - R) + T(lambda) * T(J_cache - T(1)) * J_cache * F * FTFinv;
  #endif
}
	
void Corotated32Cache::deltaP(const Eigen::Matrix<T, d, d-1>& F, const Eigen::Matrix<T, d, d-1>& dF, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& dP) {
  #ifndef TWO_D
  //TGSLAssert(d==3, "Corotated32Cache::deltaPEigen: F needs to be 3*2.");
  Eigen::Matrix<T, d, d-1> R; 
  Eigen::Matrix<T, d-1, d-1> FTFinv;
  R(0,0) = R_cache[0];
  R(0,1) = R_cache[1];
  R(1,0) = R_cache[2];
  R(1,1) = R_cache[3];
  R(2,0) = R_cache[4];
  R(2,1) = R_cache[5];
  FTFinv(0,0) = FTFinv_cache[0];
  FTFinv(0,1) = FTFinv_cache[1];
  FTFinv(1,0) = FTFinv_cache[2];
  FTFinv(1,1) = FTFinv_cache[3];
  Eigen::Matrix<T, d-1, d-1> S = R.transpose()*F;
  Eigen::Matrix<T, d-1, d-1> RTdF = R.transpose()*dF;
  T c = (RTdF(0,1)-RTdF(1,0))/trace_S_cache;
  Eigen::Matrix<T, d-1, d-1> RTdR;
  RTdR(0, 0) = T(0);
  RTdR(0, 1) = c;
  RTdR(1, 0) = -c;
  RTdR(1, 1) = T(0);
  Eigen::Matrix<T, d-1, d-1> dS = RTdF - RTdR*S;
  Eigen::Matrix<T, d, d-1> dR = dF*S.inverse() - R*dS*S.inverse();
  Eigen::Matrix<T, d-1, d-1> deltaE = dF.transpose() * F + F.transpose() * dF;
  dP = T(2) * T(mu) * dF - T(2) * T(mu) * dR + T(lambda) * (J_cache - T(1)) * J_cache * dF * FTFinv - T(lambda) * (J_cache - T(1)) * J_cache * F * FTFinv * (deltaE) * FTFinv +
      0.5 * T(lambda) * (2.0 * J_cache - T(1)) * J_cache * (FTFinv * deltaE).trace() * F * FTFinv;
  #endif
}

void Corotated32Cache::deltaPSimpleDefinite(const Eigen::Matrix<T, d, d - 1>& F, const Eigen::Matrix<T, d, d - 1>& dF, const T mu, const T lambda, Eigen::Matrix<T, d, d - 1>& dP) {
#ifndef TWO_D
  // TGSLAssert(d==3, "Corotated32Cache::deltaPEigen: F needs to be 3*2.");
  Eigen::Matrix<T, d, d - 1> R;
  Eigen::Matrix<T, d - 1, d - 1> FTFinv;
  R(0, 0) = R_cache[0];
  R(0, 1) = R_cache[1];
  R(1, 0) = R_cache[2];
  R(1, 1) = R_cache[3];
  R(2, 0) = R_cache[4];
  R(2, 1) = R_cache[5];
  FTFinv(0, 0) = FTFinv_cache[0];
  FTFinv(0, 1) = FTFinv_cache[1];
  FTFinv(1, 0) = FTFinv_cache[2];
  FTFinv(1, 1) = FTFinv_cache[3];
  Eigen::Matrix<T, d - 1, d - 1> S = R.transpose() * F;
  Eigen::Matrix<T, d - 1, d - 1> RTdF = R.transpose() * dF;
  T c = (RTdF(0, 1) - RTdF(1, 0)) / trace_S_cache;
  Eigen::Matrix<T, d - 1, d - 1> RTdR;
  RTdR(0, 0) = T(0);
  RTdR(0, 1) = c;
  RTdR(1, 0) = -c;
  RTdR(1, 1) = T(0);
  Eigen::Matrix<T, d - 1, d - 1> dS = RTdF - RTdR * S;
  Eigen::Matrix<T, d, d - 1> dR = dF * S.inverse() - R * dS * S.inverse();
  // Eigen::Matrix<T, d - 1, d - 1> deltaE = dF.transpose() * F + F.transpose() * dF;
  dP = T(2) * T(mu) * dF - T(2) * T(mu) * dR + T(lambda) * (J_cache - T(1)) * J_cache * dF * FTFinv;
#endif
}
	
void Corotated32Cache::dPdF(const Eigen::Matrix<T, d, d-1>& F,const T mu, const T lambda, Eigen::Matrix<T, d*(d-1), d*(d-1)>& dPdF) {
  #ifndef TWO_D
  //TGSLAssert(d == 3, "Corotated32Cache::deltaP: F needs to be 3*2.");
  Eigen::Matrix<T, d, d-1> R; 
  Eigen::Matrix<T, d-1, d-1> FTFinv;
  R(0,0) = R_cache[0];
  R(0,1) = R_cache[1];
  R(1,0) = R_cache[2];
  R(1,1) = R_cache[3];
  R(2,0) = R_cache[4];
  R(2,1) = R_cache[5];
  FTFinv(0,0) = FTFinv_cache[0];
  FTFinv(0,1) = FTFinv_cache[1];
  FTFinv(1,0) = FTFinv_cache[2];
  FTFinv(1,1) = FTFinv_cache[3];
  Eigen::Matrix<T, d-1, d-1> S = R.transpose()*F;
  Eigen::Matrix<T, d-1, d-1> SInv = S.inverse();
  Eigen::Matrix<T, d, d> RRT = R*R.transpose();
  Eigen::Matrix<T, d, d-1> Re;
  Re(0,0) = R(0,1);
  Re(0,1) = -R(0,0);
  Re(1,0) = R(1,1);
  Re(1,1) = -R(1,0);
  Re(2,0) = R(2,1);
  Re(2,1) = -R(2,0);
  TV dcdF = {-R(0,1)/trace_S_cache, R(0,0)/trace_S_cache, -R(1,1)/trace_S_cache, R(1,0)/trace_S_cache, -R(2,1)/trace_S_cache, R(2,0)/trace_S_cache};
  Eigen::Matrix<T, d*(d-1), d*(d-1)> dRdF;
  IV indices = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1};
  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      dRdF(c1, c2) = 0;
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m == i) {
        dRdF(c1, c2) += SInv(j, n);
      }
      dRdF(c1, c2) -= RRT(m, i) * SInv(j ,n);
      dRdF(c1, c2) -= dcdF[c2]*Re(m,n);
    }
  }

  Eigen::Matrix<T, d, d-1> FFTFInv = F*FTFinv;
  Eigen::Matrix<T, d, d> FFTFInvFT = F*FTFinv*F.transpose();

  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m > 0 && i > 0) {}
      dPdF(c1, c2) = 0;
      if (m == i){
        dPdF(c1, c2) += T(lambda) * (J_cache - T(1)) * J_cache *FTFinv(j,n);
      }
      dPdF(c1, c2) -= T(lambda) * (J_cache - T(1)) * J_cache * (FFTFInv(m, j)*FFTFInv(i, n)+FFTFInvFT(m, i) * FTFinv(j,n));
      dPdF(c1, c2) += 0.5 * T(lambda) * (2.0 * J_cache - T(1)) * J_cache * (FFTFInv(i, j)*FFTFInv(m, n)+ FFTFInv(i, j)*FFTFInv(m, n));
    }
  }
  
  dPdF =  dPdF+ T(2) * T(mu)*(Eigen::Matrix<T, d*(d-1), d*(d-1)>::Identity() -dRdF);
  #endif
}

void Corotated32Cache::PRayleignDamping(const Eigen::Matrix<T, d, d-1>& dVdX, const T alpha_rd, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& P) {
  #ifndef TWO_D
  //TGSLAssert(d == 3, "Corotated32Cache::PRayleignDamping: dVdX needs to be 3*2.");
  Eigen::Matrix<T, d, d-1> R;
  R(0,0) = Rn_cache[0];
  R(0,1) = Rn_cache[1];
  R(1,0) = Rn_cache[2];
  R(1,1) = Rn_cache[3];
  R(2,0) = Rn_cache[4];
  R(2,1) = Rn_cache[5];
  Eigen::Matrix<T, d-1, d-1> RTdF = R.transpose()*dVdX;
  T c = (RTdF(0,1)-RTdF(1,0))/T(2);
  Eigen::Matrix<T, d-1, d-1> RTdR;
  RTdR(0, 0) = T(0);
  RTdR(0, 1) = c;
  RTdR(1, 0) = -c;
  RTdR(1, 1) = T(0);
  Eigen::Matrix<T, d-1, d-1> dS = RTdF - RTdR;
  Eigen::Matrix<T, d, d-1> dR = dVdX - R*dS;
  Eigen::Matrix<T, d-1, d-1> deltaE = dVdX.transpose() * R + R.transpose() * dVdX;
  P = T(2) * T(mu) * dVdX - T(2) * T(mu) * dR  +
      0.5 * T(lambda) * (deltaE).trace() * R ;
  P = P * alpha_rd;
  #endif
}

void Corotated32Cache::dPdFRayleignDamping(const Eigen::Matrix<T, d, d-1>& F, const T alpha_rd, const T mu, const T lambda, Eigen::Matrix<T, d*(d-1), d*(d-1)>& dPdF){
  #ifndef TWO_D
  //TGSLAssert(d == 3, "Corotated32Cache::deltaP: F needs to be 3*2.");
  Eigen::Matrix<T, d, d-1> R; 
  Eigen::Matrix<T, d-1, d-1> FTFinv;
  R(0,0) = Rn_cache[0];
  R(0,1) = Rn_cache[1];
  R(1,0) = Rn_cache[2];
  R(1,1) = Rn_cache[3];
  R(2,0) = Rn_cache[4];
  R(2,1) = Rn_cache[5];
  Eigen::Matrix<T, d-1, d-1> Id = Eigen::Matrix<T, d-1, d-1>::Identity();
  Eigen::Matrix<T, d, d> RRT = R*R.transpose();
  Eigen::Matrix<T, d, d-1> Re;
  Re(0,0) = R(0,1);
  Re(0,1) = -R(0,0);
  Re(1,0) = R(1,1);
  Re(1,1) = -R(1,0);
  Re(2,0) = R(2,1);
  Re(2,1) = -R(2,0);
  TV dcdF = {-R(0,1)/T(2), R(0,0)/T(2), -R(1,1)/T(2), R(1,0)/T(2), -R(2,1)/T(2), R(2,0)/T(2)};
  Eigen::Matrix<T, d*(d-1), d*(d-1)> dRdF(6,6);
  IV indices = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1};
  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      dRdF(c1, c2) = 0;
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m == i) {
        dRdF(c1, c2) += Id(j, n);
      }
      dRdF(c1, c2) -= RRT(m, i) * Id(j ,n);
      dRdF(c1, c2) -= dcdF[c2]*Re(m,n);
    }
  }

  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m > 0 && i > 0) {}
      dPdF(c1, c2) = 0;
      dPdF(c1, c2) += 0.5 * T(lambda) * (R(i, j)*R(m, n)+ R(i, j)*R(m, n));
    }
  }
  
  dPdF = dPdF+ T(2) * T(mu)*(Eigen::Matrix<T, d*(d-1), d*(d-1)>::Identity() -dRdF);
  dPdF = dPdF*alpha_rd;
  #endif
}

T PsiCorotated32(const ALGEBRA::DenseMatrix& F, const T mu, const T lambda) {
  TGSLAssert(F.m == 3 && F.n == 2, "Corotated32::psiCorotated32: F needs to be 3*2.");
  ALGEBRA::DenseMatrix U, D, R;
  Jacobi(F.Transpose() * F, D, U);

  ALGEBRA::DenseMatrix S_mat = ALGEBRA::Identity(2);
  S_mat(0, 0) = std::sqrt(D(0,0));
  S_mat(1, 1) = std::sqrt(D(1,1));
  ALGEBRA::DenseMatrix S = U * S_mat * U.Transpose();
  R = F * S.Inverse();
  T J = T(S_mat(0, 0) * S_mat(1, 1));
  return T(mu) * ALGEBRA::Norm(F - R) * ALGEBRA::Norm(F - R) + T(lambda) / T(2) * (T(J) - T(1)) * (T(J) - T(1));
}

T PsiCorotated32(const Eigen::Matrix<T, d, d-1>& F, const T mu, const T lambda) {
  #ifdef TWO_D
  TGSLAssert(false,"PsiCorotated32: currently only supports TWO_D.");
  return T(0);
  #else
  Eigen::Matrix<T,2,2> U, D, S, S_diag;
  Eigen::Matrix<T,2,2> C=F.transpose() * F;
  Jacobi(C, D, U);

  S_diag.setZero();
  S_diag(0, 0) = std::sqrt(D(0,0));
  S_diag(1, 1) = std::sqrt(D(1,1));
  S = U * S_diag * U.transpose();
  Eigen::Matrix<T,3,2> R = F * S.inverse();
  T J = T(S_diag(0, 0) * S_diag(1, 1));
  Eigen::Matrix<T,3,2> FmR=F - R;
  T contraction=FmR(0,0)*FmR(0,0) + FmR(0,1)*FmR(0,1) + FmR(1,0)*FmR(1,0) + FmR(1,1)*FmR(1,1) + FmR(2,0)*FmR(2,0) + FmR(2,1)*FmR(2,1);
  return T(mu) * contraction + T(lambda) / T(2) * (T(J) - T(1)) * (T(J) - T(1));
  #endif
}

void PCorotated32(const ALGEBRA::DenseMatrix& F, const T mu, const T lambda, ALGEBRA::DenseMatrix& P) {
  TGSLAssert(F.m == 3 && F.n == 2, "Corotated32::PCorotated32: F needs to be 3*2.");
  ALGEBRA::DenseMatrix U, D, R;
  ALGEBRA::DenseMatrix FTF = F.Transpose() * F;
  Jacobi(FTF, D, U);

  ALGEBRA::DenseMatrix S_mat = ALGEBRA::Identity(2);
  S_mat(0, 0) = std::sqrt(D(0,0));
  S_mat(1, 1) = std::sqrt(D(1,1));
  ALGEBRA::DenseMatrix S = U * S_mat * U.Transpose();
  R = F * S.Inverse();
  T J = T(S_mat(0, 0) * S_mat(1, 1));
  
  P = T(2) * T(mu) * (F - R) + T(lambda) * T(T(J) - T(1)) * T(J) * F * (FTF.Inverse());

}

void PCorotated32(const Eigen::Matrix<T, d, d-1>& F, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& P) {
  #ifdef TWO_D
  TGSLAssert(false,"PsiCorotated32: currently only defined in THREE_D.");
  #else
  Eigen::Matrix<T, d-1, d-1> U, D;
  Eigen::Matrix<T, d, d-1> R;
  Eigen::Matrix<T, d-1, d-1> FTF = F.transpose() * F;
  Jacobi(FTF, D, U);

  Eigen::Matrix<T, d-1, d-1> S_diag;
  S_diag<<std::sqrt(D(0,0)),T(0),T(0),std::sqrt(D(1,1));
  Eigen::Matrix<T, d-1, d-1> S = U * S_diag * U.transpose();
  R = F * S.inverse();
  T J = T(S_diag(0, 0) * S_diag(1, 1));
  
  P = T(2) * T(mu) * (F - R) + T(lambda) * T(T(J) - T(1)) * T(J) * F * (FTF.inverse());
  //P<<P_dm(0,0),P_dm(0,1),P_dm(1,0),P_dm(1,1),P_dm(2,0),P_dm(2,1);
  #endif
}

void deltaPCorotated32(const ALGEBRA::DenseMatrix& F, const ALGEBRA::DenseMatrix& dF, const T mu, const T lambda, ALGEBRA::DenseMatrix& dP) {
  TGSLAssert(F.m == 3 && F.n == 2, "Corotated32::deltaPCorotated32: F needs to be 3*2.");
  ALGEBRA::DenseMatrix U, D, R;
  ALGEBRA::DenseMatrix FTF = F.Transpose() * F;
  ALGEBRA::DenseMatrix FTFinv = FTF.Inverse();
  Jacobi(FTF, D, U);

  ALGEBRA::DenseMatrix S_mat = ALGEBRA::Identity(2);
  S_mat(0, 0) = std::sqrt(D(0,0));
  S_mat(1, 1) = std::sqrt(D(1,1));
  ALGEBRA::DenseMatrix S = U * S_mat * U.Transpose();
  R = F * S.Inverse();
  T J = T(S_mat(0, 0) * S_mat(1, 1));

  ALGEBRA::DenseMatrix A(2,1), B(2,1);

  ALGEBRA::DenseMatrix s_0 = S_mat;  // R.Transpose()*F;

  B(0, 0) = (R.Transpose() * dF - dF.Transpose() * R)(0, 1);
  B(1, 0) = (R.Transpose() * dF - dF.Transpose() * R)(1, 0);
  S_mat(0, 0) = s_0(0, 0) + s_0(1, 1);
  S_mat(0, 1) = s_0(0, 1);
  S_mat(1, 0) = s_0(1, 0);
  S_mat(1, 1) = s_0(0, 0) + s_0(1, 1);
  A = S_mat.Inverse() * B;

  ALGEBRA::DenseMatrix RtdR(2, 2);
  ALGEBRA::DenseMatrix dR(3, 2);

  RtdR(0, 0) = T(0);
  RtdR(0, 1) = A(0, 0);
  RtdR(1, 0) = A(1, 0);
  RtdR(1, 1) = T(0);
  ALGEBRA::DenseMatrix S_matori = R.Transpose() * F;
  ALGEBRA::DenseMatrix deltaS;
  deltaS = R.Transpose() * dF - RtdR * S_matori;
  dR = dF * S_matori.Inverse() - R * deltaS * S_matori.Inverse(); 
  
  ALGEBRA::DenseMatrix deltaE = dF.Transpose() * F + F.Transpose() * dF;
  dP = T(2) * T(mu) * dF - T(2) * T(mu) * dR + T(lambda) * (T(J) - T(1)) * T(J) * dF * FTFinv - T(lambda) * T(T(J) - T(1)) * T(J) * F * FTFinv * (deltaE) * FTFinv +
      0.5 * T(lambda) * (2.0 * T(J) - T(1)) * T(J) * (FTFinv * deltaE).Trace() * F * FTFinv;
}

void deltaPCorotated32(const Eigen::Matrix<T, d, d-1>& F, const Eigen::Matrix<T, d, d-1>& dF, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& dP) {
  #ifdef TWO_D
  TGSLAssert(false,"deltaPCorotated32: currently only defined in THREE_D.");
  #else
  Eigen::Matrix<T, 2, 2> U, C, C_inv, D, S_diag, S, s_0, RtdR, S_matori, deltaS, deltaE;
  C = F.transpose() * F;
  C_inv = C.inverse();
  Jacobi(C, D, U);

  S_diag.setZero();
  S_diag(0, 0) = std::sqrt(D(0,0));
  S_diag(1, 1) = std::sqrt(D(1,1));
  S = U * S_diag * U.transpose();
  Eigen::Matrix<T, 3, 2> R = F * S.inverse();
  T J = T(S_diag(0, 0) * S_diag(1, 1));

  Eigen::Matrix<T, 2, 1> A, B;

  s_0 = S_diag;  // R.Transpose()*F;

  B(0, 0) = (R.transpose() * dF - dF.transpose() * R)(0, 1);
  B(1, 0) = (R.transpose() * dF - dF.transpose() * R)(1, 0);
  S_diag(0, 0) = s_0(0, 0) + s_0(1, 1);
  S_diag(0, 1) = s_0(0, 1);
  S_diag(1, 0) = s_0(1, 0);
  S_diag(1, 1) = s_0(0, 0) + s_0(1, 1);
  A = S_diag.inverse() * B;

  Eigen::Matrix<T, 3, 2> dR;

  RtdR(0, 0) = T(0);
  RtdR(0, 1) = A(0, 0);
  RtdR(1, 0) = A(1, 0);
  RtdR(1, 1) = T(0);
  S_matori = R.transpose() * F;
  deltaS = R.transpose() * dF - RtdR * S_matori;
  dR = dF * S_matori.inverse() - R * deltaS * S_matori.inverse(); 
  
  deltaE = dF.transpose() * F + F.transpose() * dF;
  dP = T(2) * T(mu) * dF - T(2) * T(mu) * dR + T(lambda) * (T(J) - T(1)) * T(J) * dF * C_inv - T(lambda) * T(T(J) - T(1)) * T(J) * F * C_inv * (deltaE) * C_inv +
      0.5 * T(lambda) * (2.0 * T(J) - T(1)) * T(J) * (C_inv * deltaE).trace() * F * C_inv;
  #endif
}

void dPdFCorotated32(const ALGEBRA::DenseMatrix& F, const T mu, const T lambda, ALGEBRA::DenseMatrix& dPdF) {
  TGSLAssert(F.m == 3 && F.n == 2, "Corotated32Cache::deltaP: F needs to be 3*2.");
  ALGEBRA::DenseMatrix U, D, R;
  ALGEBRA::DenseMatrix FTF = F.Transpose() * F;
  ALGEBRA::DenseMatrix FTFinv = FTF.Inverse();
  Jacobi(FTF, D, U);

  ALGEBRA::DenseMatrix S_mat = ALGEBRA::Identity(2);
  S_mat(0, 0) = std::sqrt(D(0,0));
  S_mat(1, 1) = std::sqrt(D(1,1));
  ALGEBRA::DenseMatrix S = U * S_mat * U.Transpose();
  R = F * S.Inverse();
  //ALGEBRA::DenseMatrix S = R.Transpose()*F;
  T J_cache = T(S_mat(0, 0) * S_mat(1, 1));
  ALGEBRA::DenseMatrix SInv = S.Inverse();
  ALGEBRA::DenseMatrix RRT = R*R.Transpose();
  ALGEBRA::DenseMatrix Re(3,2);
  Re(0,0) = R(0,1);
  Re(0,1) = -R(0,0);
  Re(1,0) = R(1,1);
  Re(1,1) = -R(1,0);
  Re(2,0) = R(2,1);
  Re(2,1) = -R(2,0);
  T trace_S_cache=S.Trace();
  TV dcdF = {-R(0,1)/trace_S_cache, R(0,0)/trace_S_cache, -R(1,1)/trace_S_cache, R(1,0)/trace_S_cache, -R(2,1)/trace_S_cache, R(2,0)/trace_S_cache};
  ALGEBRA::DenseMatrix dRdF(6,6);
  IV indices = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1};
  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      dRdF(c1, c2) = 0;
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m == i) {
        dRdF(c1, c2) += SInv(j, n);
      }
      dRdF(c1, c2) -= RRT(m, i) * SInv(j ,n);
      dRdF(c1, c2) -= dcdF[c2]*Re(m,n);
    }
  }

  ALGEBRA::DenseMatrix FFTFInv = F*FTFinv;
  ALGEBRA::DenseMatrix FFTFInvFT = F*FTFinv*F.Transpose();

  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m > 0 && i > 0) {}
      dPdF(c1, c2) = 0;
      if (m == i){
        dPdF(c1, c2) += T(lambda) * (J_cache - T(1)) * J_cache *FTFinv(j,n);
      }
      dPdF(c1, c2) -= T(lambda) * (J_cache - T(1)) * J_cache * (FFTFInv(m, j)*FFTFInv(i, n)+FFTFInvFT(m, i) * FTFinv(j,n));
      dPdF(c1, c2) += 0.5 * T(lambda) * (2.0 * J_cache - T(1)) * J_cache * (FFTFInv(i, j)*FFTFInv(m, n)+ FFTFInv(i, j)*FFTFInv(m, n));
    }
  }
  
  dPdF =  dPdF+ T(2) * T(mu)*(ALGEBRA::Identity(6) -dRdF);
}

void dPdFCorotated32(const Eigen::Matrix<T,d,d-1>& F, const T mu, const T lambda, Eigen::Matrix<T,d*(d-1),d*(d-1)>& dPdF) {
  #ifdef TWO_D
  TGSLAssert(false,"dPdFCorotated32: currently only supports THREE_D.");
  #else
  Eigen::Matrix<T,2,2> U, D, FTF, FTFinv, S_diag, S, SInv;
  Eigen::Matrix<T,3,2> R, Re, FFTFInv;
  Eigen::Matrix<T,3,3> RRT, FFTFInvFT;
  Eigen::Matrix<T,6,6> dRdF;
  FTF = F.transpose() * F;
  FTFinv = FTF.inverse();
  Jacobi(FTF, D, U);

  S_diag.setZero();
  S_diag(0, 0) = std::sqrt(D(0,0));
  S_diag(1, 1) = std::sqrt(D(1,1));
  S = U * S_diag * U.transpose();
  R = F * S.inverse();
  T J_cache = T(S_diag(0, 0) * S_diag(1, 1));
  SInv = S.inverse();
  RRT = R*R.transpose();
  Re(0,0) = R(0,1);
  Re(0,1) = -R(0,0);
  Re(1,0) = R(1,1);
  Re(1,1) = -R(1,0);
  Re(2,0) = R(2,1);
  Re(2,1) = -R(2,0);
  T trace_S_cache=S.trace();
  TV dcdF = {-R(0,1)/trace_S_cache, R(0,0)/trace_S_cache, -R(1,1)/trace_S_cache, R(1,0)/trace_S_cache, -R(2,1)/trace_S_cache, R(2,0)/trace_S_cache};
  IV indices = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1};
  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      dRdF(c1, c2) = 0;
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m == i) {
        dRdF(c1, c2) += SInv(j, n);
      }
      dRdF(c1, c2) -= RRT(m, i) * SInv(j ,n);
      dRdF(c1, c2) -= dcdF[c2]*Re(m,n);
    }
  }

  FFTFInv = F*FTFinv;
  FFTFInvFT = F*FTFinv*F.transpose();

  for (sz c1 = 0; c1 < 6; c1++) {
    for (sz c2 = 0; c2 < 6; c2++) {
      sz m = indices[2*c1];
      sz n = indices[2*c1+1];
      sz i = indices[2*c2];
      sz j = indices[2*c2+1];
      if (m > 0 && i > 0) {}
      dPdF(c1, c2) = 0;
      if (m == i){
        dPdF(c1, c2) += T(lambda) * (J_cache - T(1)) * J_cache *FTFinv(j,n);
      }
      dPdF(c1, c2) -= T(lambda) * (J_cache - T(1)) * J_cache * (FFTFInv(m, j)*FFTFInv(i, n)+FFTFInvFT(m, i) * FTFinv(j,n));
      dPdF(c1, c2) += 0.5 * T(lambda) * (2.0 * J_cache - T(1)) * J_cache * (FFTFInv(i, j)*FFTFInv(m, n)+ FFTFInv(i, j)*FFTFInv(m, n));
    }
  }
  

  Eigen::Matrix<T,6,6> I;
  I.setIdentity();
  dPdF =  dPdF+ T(2) * T(mu)*(I -dRdF);
  #endif
}

void dJdF32(const Eigen::Matrix<T, d, d-1>& F, Eigen::Matrix<T, d, d-1>& dJ){
  Eigen::Matrix<T, d-1, d-1> FTF=F.transpose()*F;
  T J2=FTF.determinant();

  Eigen::Matrix<T, d-1, d-1> J2invT;
  J2invT<<FTF(1,1),-FTF(1,0),-FTF(0,1),FTF(0,0);

  if(J2>T(0))
    dJ=(T(1)/sqrt(J2))*F*J2invT;
  else
    dJ=Eigen::Matrix<T, d, d-1>::Zero();
}

}  // namespace TGSL
