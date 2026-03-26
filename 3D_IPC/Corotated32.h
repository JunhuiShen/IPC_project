#pragma once 
#include <core/algebra/MatricesAndVectors.h>
#include <core/Definitions.h>
#include <core/algebra/MathTools.h>
#include <io/IO.h>

namespace TGSL {
// For F = dPhi/dX that is 3*2, e.g. cloth in 3D
struct Corotated32Cache{
	TV FTFinv_cache;
	TV R_cache;
	T J_cache;
	T trace_S_cache;

	//damping cache
	TV Rn_cache;
	TV Fn_cache;

	Corotated32Cache():FTFinv_cache(d*(d-1)),R_cache(d*(d-1)), Rn_cache(d*(d-1)), Fn_cache(d*(d-1)){}
	void UpdateCache(const ALGEBRA::DenseMatrix& F, const bool update_damping = false);
	void P(const ALGEBRA::DenseMatrix& F, const T mu, const T lambda, ALGEBRA::DenseMatrix& P);
	void deltaP(const ALGEBRA::DenseMatrix& F, const ALGEBRA::DenseMatrix& dF, const T mu, const T lambda, ALGEBRA::DenseMatrix& dP);
	void dPdF(const ALGEBRA::DenseMatrix& F,const T mu, const T lambda, ALGEBRA::DenseMatrix& dPdF);
	void PRayleignDamping(const ALGEBRA::DenseMatrix& dVdX, const T alpha_rd, const T mu, const T lambda, ALGEBRA::DenseMatrix& P);
	void dPdFRayleignDamping(const ALGEBRA::DenseMatrix& F, const T alpha_rd, const T mu, const T lambda, ALGEBRA::DenseMatrix& dPdF);
	void UpdateCache(const Eigen::Matrix<T, d, d-1>& F, const bool update_damping = false);
	void P(const Eigen::Matrix<T, d, d-1>& F, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& P);
    void deltaP(const Eigen::Matrix<T, d, d - 1>& F, const Eigen::Matrix<T, d, d - 1>& dF, const T mu, const T lambda, Eigen::Matrix<T, d, d - 1>& dP);
    void deltaPSimpleDefinite(const Eigen::Matrix<T, d, d - 1>& F, const Eigen::Matrix<T, d, d - 1>& dF, const T mu, const T lambda, Eigen::Matrix<T, d, d - 1>& dP);
	void dPdF(const Eigen::Matrix<T, d, d-1>& F,const T mu, const T lambda, Eigen::Matrix<T, d*(d-1), d*(d-1)>& dPdF);
	void PRayleignDamping(const Eigen::Matrix<T, d, d-1>& dVdX, const T alpha_rd, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& P);
	void dPdFRayleignDamping(const Eigen::Matrix<T, d, d-1>& F, const T alpha_rd, const T mu, const T lambda, Eigen::Matrix<T, d*(d-1), d*(d-1)>& dPdF);
};

T PsiCorotated32(const ALGEBRA::DenseMatrix& F, const T mu, const T lambda);
T PsiCorotated32(const Eigen::Matrix<T, d, d-1>& F, const T mu, const T lambda);

void PCorotated32(const ALGEBRA::DenseMatrix& F, const T mu, const T lambda, ALGEBRA::DenseMatrix& P);
void PCorotated32(const Eigen::Matrix<T, d, d-1>& F, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& P);

void deltaPCorotated32(const ALGEBRA::DenseMatrix& F, const ALGEBRA::DenseMatrix& dF, const T mu, const T lambda, ALGEBRA::DenseMatrix& dP);
void deltaPCorotated32(const Eigen::Matrix<T, d, d-1>& F, const Eigen::Matrix<T, d, d-1>& dF, const T mu, const T lambda, Eigen::Matrix<T, d, d-1>& dP);

void dPdFCorotated32(const ALGEBRA::DenseMatrix& F, const T mu, const T lambda, ALGEBRA::DenseMatrix& dPdF);
void dPdFCorotated32(const Eigen::Matrix<T, d, d-1>& F, const T mu, const T lambda, Eigen::Matrix<T, d*(d-1), d*(d-1)>& dPdF);
void dJdF32(const Eigen::Matrix<T, d, d-1>& F,Eigen::Matrix<T, d, d-1>& dJ);
}  // namespace TGSL