#pragma once

#include <iostream>
#include <numeric>

#include <core/Definitions.h>
#include <Eigen/Geometry>

namespace TGSL{
namespace ALGEBRA{
	struct DenseMatrix{
	  TV A;
	  sz m,n;

		DenseMatrix(){
	    m=1;n=1;
	    A.resize(1,T(0));
	  }

	  DenseMatrix(const sz _m,const sz _n){
	    m=_m;n=_n;
	    A.resize(m*n,T(0));
	  }

	  DenseMatrix(const sz _m,const sz _n,const T value){
	    m=_m;n=_n;
	    A.resize(m*n,value);
	  }
	  
	  inline void Resize(const sz _m,const sz _n){
	    m=_m;n=_n;
	    A.resize(m*n);
	  }

	  inline void Resize(const sz _m,const sz _n, const T value){
	    m=_m;n=_n;
	    A.resize(m*n);
	    std::fill(std::begin(A), std::end(A), value);
	  }

	  inline T SubFrobeniusNorm(const Vector2I& i_range, const Vector2I& j_range) const {
	  	//returns Aij Aij for i_range[0] <= i <= i_range[1] and j_range[0] <= j <= j_range[1]
	  	TGSLAssert(i_range[1]>=i_range[0] && i_range[0]>=0 && i_range[1]<nm(m),"DenseMatrix::SubFrobeniusNorm: i_ranges are degenerate");
	  	TGSLAssert(j_range[1]>=j_range[0] && j_range[0]>=0 && j_range[1]<nm(n),"DenseMatrix::SubFrobeniusNorm: j_ranges are degenerate");

	  	TV column_sums(sz(j_range[1] - j_range[0] + 1));
	  	for(sz j=sz(j_range[0]);j<sz(j_range[1]+1);j++){
		    T result = T(0);
            #pragma omp parallel for reduction(+: result)
			for(nm i=nm(i_range[0]);i<nm(i_range[1]+1);++i){
				result += A[n*sz(i) + j]*A[n*sz(i) + j];
			}
			column_sums[j-j_range[0]] = result;
	  	}
	  	T norm_squared=T(0);
		#pragma omp parallel for reduction(+: norm_squared)
		for(nm i = 0; i < nm(column_sums.size()); ++i) {
			norm_squared += column_sums[i];
		}

	  	return std::sqrt(norm_squared);
	  }

	  inline T Trace()const{
	  	TGSLAssert(m==n,"DenseMatrix::Trace only defined for square matrices");
	  	T trace=T(0);
		#pragma omp parallel for reduction(+: trace)
		for(nm i=0; i < nm(m); ++i) {
			trace += A[n*sz(i) + sz(i)];
		}

	  	return trace;
	  }

	  inline T Determinant()const{
	  	TGSLAssert(m==n,"DenseMatrix::Determinant only defined for square matrices");
	  	TGSLAssert(m<4,"DenseMatrix::Determinant currently only defined for dimensions less than 4.");
	  	if(m==1)
	  		return A[0];
	  	if(m==2)
	  		return A[n*sz(0) + sz(0)]*A[n*sz(1) + sz(1)] - A[n*sz(1) + sz(0)]*A[n*sz(0) + sz(1)];
	  	if(m==3)
	  		return A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])+A[2]*(A[3]*A[7]-A[4]*A[6]);

	  	return T(0);
	  }
	  
	  inline T operator()(const sz i,const sz j)const{
	    TGSLAssert(i<m && j<n,"DenseMatrix: i and j past matrix bounds.");
	    return A[n*i + j];
	  }

	  inline void SwapRows(const Vector2I& row_swap){
	  	TGSLAssert(sz(row_swap[0])<m && sz(row_swap[0])<m && sz(row_swap[0])>=0 && sz(row_swap[1])>=0,"DenseMatrix::SwapRows: rows not in dims of matrix.");

	  	if(row_swap[0]==row_swap[1]) 
	  		return;
	  	
	  	for(sz j=0;j<n;j++){
	  		T temp=A[n*row_swap[0] + j];
	  		A[n*row_swap[0] + j]=A[n*row_swap[1] + j];
	  		A[n*row_swap[1] + j]=temp;
	  	}
	  }

	  inline DenseMatrix Inverse() const{
	  	TGSLAssert(m==n,"DenseMatrix::Determinant only defined for square matrices");
	  	TGSLAssert(m<4,"DenseMatrix::Determinant currently only defined for dimensions less than 4.");
	  	DenseMatrix inverse(m,m);
	  	if(m==1)
	  		inverse.A={T(1)/A[0]};
	  	else if(m==2){
	  		T det=A[0]*A[3]-A[2]*A[1];
	  		inverse.A={A[3]/det,-A[1]/det,-A[2]/det,A[0]/det};
	  	}
	  	else{
		  	T det=A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7]-A[4]*A[6]);
		  	inverse.A={(A[4]*A[8]-A[5]*A[7])/det,(A[2]*A[7]-A[1]*A[8])/det,(A[1]*A[5]-A[2]*A[4])/det,
		  			   (A[5]*A[6]-A[3]*A[8])/det,(A[0]*A[8]-A[2]*A[6])/det,(A[2]*A[3]-A[0]*A[5])/det,
		  			   (A[3]*A[7]-A[4]*A[6])/det,(A[1]*A[6]-A[0]*A[7])/det,(A[0]*A[4]-A[1]*A[3])/det};
		  }
	  	return inverse;
	  }

	  inline DenseMatrix Transpose() const{
	  	DenseMatrix transpose(n,m);
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<n;j++){
	  			transpose(j,i)=A[n*i + j];
	  		}
	  	}
	  	return transpose;
	  }

	  inline DenseMatrix SymmetricPart() const{
	  	TGSLAssert(m==n,"DenseMatrix::SymmetricPart only defined for square matrices.");
	  	DenseMatrix result(m,m);
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<m;j++){
	  			result(i,j)=T(.5)*(A[n*i + j] + A[n*j + i]);
	  		}
	  	}
	  	return result;
	  }
	  
	  inline T& operator()(const sz i,const sz j){
	    TGSLAssert(i<m && j<n,"DenseMatrix: i and j past matrix bounds.");
	    return A[n*i + j];
	  }
	  
	  inline void operator=(const DenseMatrix& B){
	    Vector2I dims=B.Size();
	    m=dims[0];n=dims[1];
	    A=B.A;
	  }

	  inline Vector2I MaxAbsEntry(const Vector2I& i_range,const Vector2I& j_range){
	  	Vector2I result={i_range[0],j_range[0]};
	  	T max_abs=T(0);
	  	for(sz i=sz(i_range[0]);i<sz(i_range[1]);i++){
	  		for(sz j=sz(j_range[0]);j<sz(j_range[1]);j++){
	  			if(abs(A[n*i + j])>max_abs){
	  				result={nm(i),nm(j)};
	  				max_abs=abs(A[n*i + j]);
	  			}
	  		}
	  	}
	  	return result;
	  }

	  inline DenseMatrix operator-(const DenseMatrix& B)const{
	  	Vector2I dims=B.Size();
	  	TGSLAssert(this->Size()==dims,"DenseMatrix::operator+: matrix dims do not match.");
	  	DenseMatrix result(m,n,T(0));
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<n;j++){
	  			result.A[n*i + j]=A[n*i + j] - B.A[n*i + j];
	  		}
	  	}
	  	return result;
	  }

	  inline DenseMatrix operator+(const DenseMatrix& B)const{
	  	Vector2I dims=B.Size();
	  	TGSLAssert(this->Size()==dims,"DenseMatrix::operator+: matrix dims do not match.");
	  	DenseMatrix result(m,n,T(0));
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<n;j++){
	  			result.A[n*i + j]=A[n*i + j] + B.A[n*i + j];
	  		}
	  	}
	  	return result;
	  }

	  inline DenseMatrix operator*(const T scale)const{
	  	DenseMatrix result(m,n,T(0));
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<n;j++){
	  			result.A[n*i + j]=A[n*i + j]*scale;
	  		}
	  	}
	  	return result;
	  }

	  inline DenseMatrix operator*(const DenseMatrix& B)const{
	  	Vector2I dims=B.Size();
	  	TGSLAssert(sz(dims[0])==n,"DenseMatrix::operator*: matrix dims do not match.");
	  	DenseMatrix result(m,dims[1],T(0));
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<sz(dims[1]);j++){
	  			for(sz k=0;k<n;k++){
	  				result(i,j)+=A[n*i + k]*B(k,j);
	  			}
	  		}
	  	}
	  	return result;
	  }

	  inline Vector3T operator*(const Vector3T& x)const{
	  	Vector3T result=Vector3T();
	  	TGSLAssert(3==n && 3==m,"DenseMatrix::operator*: matrix and vector dims do not match.");
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<n;j++){
	  			result[i]+=A[n*i + j]*x[j];
	  		}
	  	}
	  	return result;
	  }

	  inline Vector2T operator*(const Vector2T& x)const{
	  	Vector2T result=Vector2T();
	  	TGSLAssert(2==n && 2==m,"DenseMatrix::operator*: matrix and vector dims do not match.");
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<n;j++){
	  			result[i]+=A[n*i + j]*x[j];
	  		}
	  	}
	  	return result;
	  }
	  
	  inline TV operator*(const TV& x)const{
	  	TV result(m,T(0));
	  	TGSLAssert(x.size()==n,"DenseMatrix::operator*: matrix and vector dims do not match.");
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<n;j++){
	  			result[i]+=A[n*i + j]*x[j];
	  		}
	  	}
	  	return result;
	  }

	  inline DenseMatrix operator/(const T scale)const{
	  	DenseMatrix result(m,n,T(0));
	  	for(sz i=0;i<m;i++){
	  		for(sz j=0;j<n;j++){
	  			result.A[n*i + j]=A[n*i + j]/scale;
	  		}
	  	}
	  	return result;
	  }
	  
	  inline Vector2I Size()const{
	    return {nm(m),nm(n)};
	  }

	  //this finds decoupled blocks of the matrix
	  void GroupRows(IVV& connected_groups)const {
			DenseMatrix A(m,n);
			A=(*this);
	    size_t num_groups = A.n;
	    TGSLAssert(A.n == A.m, "DenseMatrix::GroupRows: Matrix must be square.");
	    std::vector<size_t> buckets(num_groups);
	    // iota makes the vector 0, 1, 2, 3, ...
	    std::iota(buckets.begin(), buckets.end(), 0);

	    // loop over groups
	    for (size_t i = 0; i < num_groups; ++i) {
	      // loop over (other) buckets
	      for (size_t j = 0; j < num_groups; ++j) {
	        if (buckets[i] == buckets[j]) // don't process your own bucket
	          continue;
	        // check if the row that holds j also holds i
	        if (A(buckets[j], i) == 1) {
	          // if so, add everything in that bucket to the bucket holding i
	          for (size_t l = 0; l < num_groups; ++l) {
	            A(buckets[i], l) = std::max(A(buckets[i], l), A(buckets[j], l));
	            A(buckets[j], l) = 0;
	          }
	          // if a bucket grabs another, make every member aware of its new bucket
	          size_t prior_bucket = buckets[j];
	          for (size_t l = 0; l < num_groups; ++l) {
	            if (buckets[l] == prior_bucket)
	              buckets[l] = buckets[i];
	          }
	        }
	      }
	    }

	    // count and identify the number of unique buckets left at the end
	    std::sort(buckets.begin(), buckets.end());
	    auto last = std::unique(buckets.begin(), buckets.end());

	    // use the bucket numbers to loop over the row and grab the group
	    connected_groups.resize(last - buckets.begin());
	    size_t count = 0;
	    for (size_t i = 0; i < connected_groups.size(); ++i) {
	      IV current_group;
	      for (size_t l = 0; l < num_groups; ++l) {
	      	if (A(buckets[i], l) == 1)
	          current_group.emplace_back(l);
	      }
	      if (current_group.size() > 0) {
	      	connected_groups[count++] = current_group;
	      }
	    }
	    connected_groups.resize(count);
  	}
};

inline DenseMatrix operator*(const T scale, const DenseMatrix& A){
  	return A*scale;
}

inline DenseMatrix Identity(sz m){
	DenseMatrix result(m,m,T(0));
	for(sz i=0;i<m;i++){
		result.A[m*i + i]=T(1);
	}
	return result;
}

inline DenseMatrix DenseMatrixFromSerial(const sz e, const sz m, const sz n, const TV& A_serial){
	DenseMatrix result(m,n);
	TGSLAssert(m*n*e+m*n-1 < A_serial.size(),"ALGEBRA::DenseMatrixFromSerial: matrix cache not sized consistently with desired output.");
	for(sz i=0;i<m*n;i++){
		result.A[i]=A_serial[m*n*e + i];
	}
	return result;
}

inline void Normalize(Vector2T& x){
	T norm=std::sqrt(x[0]*x[0] + x[1]*x[1]);
	if(norm>0){
		x={x[0]/norm,x[1]/norm};
	}
}

inline void Normalize(Vector3T& x){
	T norm=std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
	if(norm>0){
		x={x[0]/norm,x[1]/norm,x[2]/norm};
	}
}

inline void Normalize(Vector4T& q){
	T norm=std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
	if(norm>0){
		q={q[0]/norm,q[1]/norm,q[2]/norm,q[3]/norm};
	}
}

inline Vector4T QuaternionFromRotationMatrix(const DenseMatrix& R){
	Vector4T q=Vector4T();
	TGSLAssert(R.Determinant()>T(0),"QuaternionFromRotationMatrix: input must be a rotation matrix.");

	T trace = R.Trace();
  if( trace > 0 ) {// I changed M_EPSILON to 0
    T s = T(0.5) / std::sqrt(trace + T(1.0));
    q[0] = T(0.25) / s;
    q[1] = ( R(2,1) - R(1,2) ) * s;
    q[2] = ( R(0,2) - R(2,0) ) * s;
    q[3] = ( R(1,0) - R(0,1) ) * s;
  } else {
    if ( R(0,0) > R(1,1) && R(0,0) > R(2,2) ) {
      T s = T(2) * std::sqrt( T(1) + R(0,0) - R(1,1) - R(2,2));
      q[0] = (R(2,1) - R(1,2) ) / s;
      q[1] = T(.25) * s;
      q[2] = (R(0,1) + R(1,0) ) / s;
      q[3] = (R(0,2) + R(2,0) ) / s;
    } else if (R(1,1) > R(2,2)) {
      T s = T(2) * std::sqrt( T(1) + R(1,1) - R(0,0) - R(2,2));
      q[0] = (R(0,2) - R(2,0) ) / s;
      q[1] = (R(0,1) + R(1,0) ) / s;
      q[2] = T(.25) * s;
      q[3] = (R(1,2) + R(2,1) ) / s;
    } else {
      T s = T(2) * sqrtf( T(1) + R(2,2) - R(0,0) - R(1,1) );
      q[0] = (R(1,0) - R(0,1) ) / s;
      q[1] = (R(0,2) + R(2,0) ) / s;
      q[2] = (R(1,2) + R(2,1) ) / s;
      q[3] = T(.25) * s;
    }
  }

  Normalize(q);

	return q;
}

inline Eigen::Matrix2d RotationMatrix2D(const T theta) {
	Eigen::Matrix2d Q;
	Q(0, 0) = std::cos(theta);
	Q(1, 1) = Q(0, 0);
	Q(1, 0) = std::sin(theta);
	Q(0, 1) = T(-1) * Q(1, 0);
	return Q;
}

inline Vector2T Scale(const Vector2T& x,const T s){
	return {s*x[0],s*x[1]};
}

inline Vector3T Scale(const Vector3T& x,const T s){
	return {s*x[0],s*x[1],s*x[2]};
}

inline Vector4T Scale(const Vector4T& x,const T s){
	return {s*x[0],s*x[1],s*x[2],s*x[3]};
}

inline Vector2T Sum(const Vector2T& x,const Vector2T& y){
	return {x[0]+y[0],x[1]+y[1]};
}

inline Vector3T Sum(const Vector3T& x,const Vector3T& y){
	return {x[0]+y[0],x[1]+y[1],x[2]+y[2]};
}

inline Vector4T Sum(const Vector4T& x,const Vector4T& y){
	return {x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3]};
}

inline Vector2T Difference(const Vector2T& x,const Vector2T& y){
	return {x[0]-y[0],x[1]-y[1]};
}

inline Vector3T Difference(const Vector3T& x,const Vector3T& y){
	return {x[0]-y[0],x[1]-y[1],x[2]-y[2]};
}

inline Vector4T Difference(const Vector4T& x,const Vector4T& y){
	return {x[0]-y[0],x[1]-y[1],x[2]-y[2],x[3]-y[3]};
}

inline T DotProduct(const Vector2T& a,const Vector2T& b){
	return a[0]*b[0] + a[1]*b[1];
}

inline T DotProduct(const Vector3T& a,const Vector3T& b){
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline Vector3T CrossProduct(const Vector3T& a,const Vector3T& b){
	return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}

inline DenseMatrix CrossProductMatrix(const Vector3T& v) {
	DenseMatrix v_star(3, 3, T(0));
	v_star(0, 1) = -v[2];
	v_star(0, 2) = v[1];
	v_star(1, 0) = v[2];
	v_star(1, 2) = -v[0];
	v_star(2, 0) = -v[1];
	v_star(2, 1) = v[0];
	return v_star;
}

inline T NormSquared(const Vector2T& x){
	return x[0]*x[0] + x[1]*x[1];
}

inline T NormSquared(const Vector3T& x){
	return x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
}

inline T NormSquared(const Vector4T& q){
	return q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];	
}

inline T Norm(const Vector2T& x){
	return std::sqrt(x[0]*x[0] + x[1]*x[1]);
}

inline T Norm(const Vector3T& x){
	return std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}

inline T Norm(const Vector4T& q){
	return std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);	
}

inline T Norm(const DenseMatrix& A){
	Vector2I i_range={0,nm(A.m)-1},j_range={0,nm(A.n)-1};
	return A.SubFrobeniusNorm(i_range,j_range);
}

inline T Contract(const DenseMatrix& A,const DenseMatrix& B){
	/*
	Note: not checking sizing of A and B for efficiency, be warned.
	*/
	T result=T(0);
	for(sz i=0;i<A.A.size();i++)
		result+=A.A[i]*B.A[i];
	return result;
}

inline Vector4T QuaternionFromVector(const Vector3T& v){
	T omega=Norm(v);
	Vector4T q;
	if (omega < 1e-10) {
		q={T(1),T(0),T(0),T(0)};
	}
	else {
		q={cos(T(.5)*omega),sin(T(.5)*omega)*v[0]/omega,sin(T(.5)*omega)*v[1]/omega,sin(T(.5)*omega)*v[2]/omega};
	}
	return q;
}

inline Vector4T QuaternionExponential(const Vector3T& theta) {
	T angle = Norm(theta);
	Vector3T u = Scale(theta, T(1) / angle);
	Vector4T q;
	if (angle < 1e-10) {
		q = { T(1),T(0),T(0),T(0) };
	}
	else {
		q = { cos(T(.5) * angle), sin(T(.5) * angle) * u[0], sin(T(.5) * angle) * u[1],sin(T(.5) * angle) * u[2] };
	}
	return q;
}

inline void QuaternionExponentialFirstDerivative(const Vector3T& theta, Eigen::Matrix<T, 4, 3>& D) {
	D.setZero();
	Eigen::Matrix3d I_3 = Eigen::Matrix3d::Identity();
	T angle = Norm(theta);
	Vector3T u = theta;
	Normalize(u);
	if (angle < T(1e-8)) {
		for (size_t alpha = 0; alpha < 3; alpha++) {
			D(alpha + 1, alpha) = T(.5);
		}
	}
	else {
		// row 0
		for (size_t beta = 0; beta < 3; ++beta) {
			D(0, beta) = -T(.5) * sin(T(.5) * angle) * u[beta];
		}
		// row 1-3
		for (size_t alpha = 1; alpha < 4; ++alpha) {
			for (size_t beta = 0; beta < 3; ++beta) {
				D(alpha, beta) = u[alpha - 1] * u[beta] * (T(.5) * cos(T(.5) * angle) - sin(T(.5) * angle) / angle) + I_3(alpha - 1, beta) * sin(T(.5) * angle) / angle;
			}
		}
	}
}

inline Vector4T DebugQuaternionExponential(const Vector3T& theta) {
	T angle = Norm(theta);
	Vector3T u = Scale(theta, T(1) / angle);
	Vector4T q;
	if (angle < 1e-10) {
		q = { T(1),T(0),T(0),T(0) };
	}
	else {
		q = { cos(angle), sin(angle) * u[0], sin(angle) * u[1],sin(angle) * u[2] };
	}
	return q;
}

inline void DebugQuaternionExponentialFirstDerivative(const Vector3T& theta, Eigen::Matrix<T, 4, 3>& D) {
	Eigen::Matrix3d I_3 = Eigen::Matrix3d::Identity();
	T angle = Norm(theta);
	Vector3T u = theta;
	Normalize(u);
	if (angle == T(0)) {
		for (size_t alpha = 0; alpha < 3; alpha++) {
			D(alpha + 1, alpha) = T(1);
		}
	}
	else {
		// row 0
		for (size_t beta = 0; beta < 3; ++beta) {
			D(0, beta) = -sin(angle) * u[beta];
		}
		// row 1-3
		for (size_t alpha = 1; alpha < 4; ++alpha) {
			for (size_t beta = 0; beta < 3; ++beta) {
				D(alpha, beta) = u[alpha - 1] * u[beta] * (cos(angle) - sin(angle) / angle) + I_3(alpha - 1, beta) * sin(angle) / angle;
			}
		}
	}
}

inline void DebugQuaternionExponentialSecondDerivative(const Vector3T& theta, std::vector<Eigen::Matrix<T, 4, 3>>& D) {
	Eigen::Matrix3d I_3 = Eigen::Matrix3d::Identity();
	T angle = Norm(theta);
	Vector3T u = Scale(theta, T(1) / angle);
	T f_theta = cos(angle) / angle - sin(angle) / (angle * angle);
	// D[gamma](alpha, beta) = D2(exp(theta)[alpha])/D(theta[beta])D(theta[gamma])
	if (angle == T(0)) {
		for (size_t beta = 0; beta < 3; ++beta) {
			D[beta](0, beta) = T(-1);
		}
	}
	else {
		// row 0
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t gamma = 0; gamma < 3; ++gamma) {
				D[gamma](0, beta) = u[beta] * u[gamma] * (sin(angle) / angle - cos(angle)) - I_3(beta, gamma) * sin(angle) / angle;
			}
		}
		// row 1-3
		for (size_t alpha = 1; alpha < 4; ++alpha) {
			for (size_t beta = 0; beta < 3; ++beta) {
				for (size_t gamma = 0; gamma < 3; ++gamma) {
					D[gamma](alpha, beta) = u[alpha - 1] * u[beta] * u[gamma] * (T(-3) * f_theta - sin(angle))
						+ (I_3(alpha - 1, beta) * u[gamma] + I_3(alpha - 1, gamma) * u[beta] + I_3(beta, gamma) * u[alpha - 1]) * f_theta;
				}
			}
		}
	}
}

// TODO: needs debugging with new definition
inline void QuaternionExponentialSecondDerivative(const Vector3T& theta, std::vector<Eigen::Matrix<T, 4, 3>>& D) {
	D.resize(3);
	for (size_t i = 0; i < 3; ++i) {
		D[i].setZero();
	}
	Eigen::Matrix3d I_3 = Eigen::Matrix3d::Identity();
	T angle = Norm(theta);
	Vector3T u = Scale(theta, T(1) / angle);
	T f_theta = T(.5) * cos(T(.5) * angle) / angle - sin(T(.5) * angle) / (angle * angle);
	// D[gamma](alpha, beta) = D2(exp(theta)[alpha])/D(theta[beta])D(theta[gamma])
	if (angle < T(1e-8)) {
		for (size_t beta = 0; beta < 3; ++beta) {
			D[beta](0, beta) = T(-.25);
		}
	}
	else {
		// row 0
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t gamma = 0; gamma < 3; ++gamma) {
				D[gamma](0, beta) = u[beta] * u[gamma] * (T(.5) * sin(T(.5) * angle) / angle - T(.25) * cos(T(.5) * angle)) - I_3(beta, gamma) * T(.5) * sin(T(.5) * angle) / angle;
			}
		}
		// row 1-3
		for (size_t alpha = 1; alpha < 4; ++alpha) {
			for (size_t beta = 0; beta < 3; ++beta) {
				for (size_t gamma = 0; gamma < 3; ++gamma) {
					D[gamma](alpha, beta) = u[alpha - 1] * u[beta] * u[gamma] * (T(-3) * f_theta - T(.25) * sin(T(.5) * angle))
						+ (I_3(alpha - 1, beta) * u[gamma] + I_3(alpha - 1, gamma) * u[beta] + I_3(beta, gamma) * u[alpha - 1]) * f_theta;
				}
			}
		}
	}
}

inline Vector4T ConjugateQuaternion(const Vector4T& q){
	return {q[0],-q[1],-q[2],-q[3]};
}

inline Vector4T QuaternionInverse(const Vector4T& q) {
	T scale = T(1) / NormSquared(q);
	Vector4T q_inv = { q[0],-q[1],-q[2],-q[3] };
	return Scale(q_inv, scale);
}

inline Vector4T VectorToQuaternion(const Vector3T& v) {
	return { T(0),v[0],v[1],v[2] };
}

inline Vector3T VectorOfQuaternion(const Vector4T& q) {
	return { q[1],q[2],q[3] };
}

inline Eigen::Matrix4d QuaternionInverseDerivative(const Vector4T& q) {
	// D(alpha, beta) = Dq_inv[alpha] / Dq[beta]
	T scale = T(1) / NormSquared(q);
	Eigen::Matrix4d I, D;
	Eigen::Vector4d signs(1, -1, -1, -1);
	I << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
	for (size_t alpha = 0; alpha < 4; alpha++) {
		for (size_t beta = 0; beta < 4; beta++) {
			D(alpha, beta) = signs[alpha] * (-2 * scale * scale * q[alpha] * q[beta] + scale * I(alpha, beta));
		}
	}
	return D;
}

inline std::vector<Eigen::Matrix4d> QuaternionInverseSecondDerivative(const Vector4T& q) {
	// D2_q[gamma](alpha, beta) = D (D(alpha, beta)) / Dq[gamma]
	T scale = T(1) / NormSquared(q);
	Eigen::Matrix4d I;
	std::vector<Eigen::Matrix4d> D2_q;
	Eigen::Vector4d signs(1, -1, -1, -1);
	I << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
	D2_q.resize(4);
	
	for (size_t gamma = 0; gamma < 4; gamma++) {
		for (size_t alpha = 0; alpha < 4; alpha++) {
			for (size_t beta = 0; beta < 4; beta++) {
				D2_q[gamma](alpha, beta) = signs[alpha]
					* (-2 * pow(scale, T(2)) * (I(alpha, gamma) * q[beta] + I(beta, gamma) * q[alpha] + I(alpha, beta) * q[gamma])
					  + 8 * pow(scale, T(3)) * q[alpha] * q[beta] * q[gamma]);
			}
		}
	}
	return D2_q;
}

inline Vector4T QuaternionMultiply(const Vector4T& a, const Vector4T& b){
	return {a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
					a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
					a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
					a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};
}

inline Vector3T QuaternionRotate(const Vector4T& q, const Vector3T& x){
  Vector4T a=q;
	Vector4T b=QuaternionMultiply({T(0),x[0],x[1],x[2]},ConjugateQuaternion(q));
	Vector4T quat_produc={a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
					a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
					a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
					a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};
	return {quat_produc[1],quat_produc[2],quat_produc[3]};
}

inline Vector2T QuaternionRotate(const Vector4T& q, const Vector2T& x) {
	T c = q[0] * q[0] - q[3] * q[3];
	T s = T(2) * q[0] * q[3];
	return { c * x[0] - s * x[1], s * x[0] + c * x[1] };
}

inline int LeviCivita1(int alpha, int beta, int gamma) {
	TGSLAssert((0 <= alpha) && (alpha <= 2), "MathTools: Invalid first argument for LeviCivita 3D.");
	TGSLAssert((0 <= beta) && (beta <= 2), "MathTools: Invalid second argument for LeviCivita 3D.");
	TGSLAssert((0 <= gamma) && (gamma <= 2), "MathTools: Invalid second argument for LeviCivita 3D.");
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

// Quaternion Product Tensor
inline int QPT1(int alpha, int beta, int gamma) {
	TGSLAssert((0 <= alpha) && (alpha <= 3), "MathTools: Invalid first argument for Quaternion Product Tensor.");
	TGSLAssert((0 <= beta) && (beta <= 3), "MathTools: Invalid second argument for Quaternion Product Tensor.");
	TGSLAssert((0 <= gamma) && (gamma <= 3), "MathTools: Invalid second argument for Quaternion Product Tensor.");
	if (alpha == 0 && beta == 0 && gamma == 0)
		return 1;
	else if (alpha == 0 && beta == gamma && beta != 0)
		return -1;
	else if (alpha != 0 && beta == 0 && alpha == gamma)
		return 1;
	else if (alpha != 0 && gamma == 0 && alpha == beta)
		return 1;
	else if (alpha != 0 && beta != 0 && gamma != 0)
		return LeviCivita1(alpha - 1, beta - 1, gamma - 1);
	else
		return 0;
}

inline int QPT2(int alpha, int beta, int delta, int epsilon) {
	TGSLAssert((0 <= alpha) && (alpha <= 3), "ALGEBRA: Invalid first argument for Quaternion Product Tensor.");
	TGSLAssert((0 <= beta) && (beta <= 3), "ALGEBRA: Invalid second argument for Quaternion Product Tensor.");
	TGSLAssert((0 <= delta) && (delta <= 3), "ALGEBRA: Invalid second argument for Quaternion Product Tensor.");
	TGSLAssert((0 <= epsilon) && (epsilon <= 3), "ALGEBRA: Invalid second argument for Quaternion Product Tensor.");
	T result = 0;
	for (int gamma = 0; gamma < 4; gamma++) {
		result += QPT1(alpha, beta, gamma) * QPT1(gamma, delta, epsilon);
	}
	return result;
}

inline void QuaternionRotateFirstDerivative(const Vector4T& q, const Vector3T& x, Eigen::Matrix<T, 3, 4>& D) {
	// D (q*x*q_inv) / Dq
	D.setZero();
	Vector4T q_inv = QuaternionInverse(q);
	Eigen::Matrix4d DqinvDq = QuaternionInverseDerivative(q);
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 4; ++beta) {
			for (size_t delta = 1; delta < 4; ++delta) {
				for (size_t theta = 0; theta < 4; ++theta) {
					D(alpha, beta) += QPT2(alpha + 1, beta, delta, theta) * x[delta - 1] * q_inv[theta];
					for (size_t eta = 0; eta < 4; ++eta) {
						D(alpha, beta) += QPT2(alpha + 1, eta, delta, theta) * q[eta] * x[delta - 1] * DqinvDq(theta, beta);
					}
				}
			}
		}
	}
}

inline void QuaternionRotateSecondDerivative(const Vector4T& q, const Vector3T& x, std::vector<Eigen::Matrix<T, 3, 4>>& DD) {
	// D (q*x*q_inv) / (Dq * Dq)
	DD.resize(4);
	for (size_t i = 0; i < 4; ++i) {
		DD[i].setZero();
	}
	Vector4T q_inv = QuaternionInverse(q);
	Eigen::Matrix4d DqinvDq = QuaternionInverseDerivative(q);
	std::vector<Eigen::Matrix4d> DqinvDqq = QuaternionInverseSecondDerivative(q);
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 4; ++beta) {
			for (size_t gamma = 0; gamma < 4; ++gamma) {
				for (size_t delta = 1; delta < 4; ++delta) {
					for (size_t theta = 0; theta < 4; ++theta) {
						DD[gamma](alpha, beta) += QPT2(alpha + 1, beta, delta, theta) * x[delta - 1] * DqinvDq(theta, gamma);
						DD[gamma](alpha, beta) += QPT2(alpha + 1, gamma, delta, theta) * x[delta - 1] * DqinvDq(theta, beta);
						for (size_t eta = 0; eta < 4; ++eta) {
							DD[gamma](alpha, beta) += QPT2(alpha + 1, eta, delta, theta) * q[eta] * x[delta - 1] * DqinvDqq[gamma](theta, beta);
						}
					}
				}
			}
		}
	}
}

inline void QuaternionFromTwoUnitVectors(const Vector& u, const Vector& v, Vector4T& q) {
	// compute q such that v = q * u * q_inv
	T eps = T(1e-8);
	TGSLAssert(((ALGEBRA::Norm(u) - T(1)) < eps), "ALGEBRA::QuaternionFromTwoVectors: vector u is not normalized");
	TGSLAssert(((ALGEBRA::Norm(v) - T(1)) < eps), "ALGEBRA::QuaternionFromTwoVectors: vector v is not normalized");
	T cos = ALGEBRA::DotProduct(u, v);
	T cos_half = sqrt((T(1) + cos) / T(2));
	T sin_half = sqrt((T(1) - cos) / T(2));
	Vector3T u_temp, v_temp;
#ifdef TWO_D
	u_temp = { u[0], u[1], T(0) };
	v_temp = { v[0], v[1], T(0) };
#else
	u_temp = u;
	v_temp = v;
#endif // TWO_D
	Vector3T n = ALGEBRA::CrossProduct(u_temp, v_temp);
	ALGEBRA::Normalize(n);
	q = { cos_half, sin_half * n[0], sin_half * n[1], sin_half * n[2] };
}

inline void QuaternionFromTwoUnitVectors(const Vector3d& u, const Vector3d& v, Vector4d& q) {
	// compute q such that v = q * u * q_inv
	//T eps = T(1e-8);
	//TGSLAssert((std::abs(u.norm() - T(1)) < eps), "ALGEBRA::QuaternionFromTwoVectors: vector u is not normalized");
	//TGSLAssert((v.norm() < eps), "ALGEBRA::QuaternionFromTwoVectors: vector v is not normalized");
	Vector3d v_hat = v.normalized();
	
	T cos = u.dot(v_hat);
	T cos_half = sqrt((T(1) + cos) / T(2));
	T sin_half = sqrt((T(1) - cos) / T(2));
#ifdef TWO_D
	TGSLAssert(false, "ALGEBRA::QuaternionFromTwoUnitVectors: not defined for 2d.");
#endif // TWO_D
	Vector3d n = u.cross(v_hat);
	n.normalize();
	q << cos_half, sin_half * n[0], sin_half * n[1], sin_half * n[2];
}

inline T Epsilon(const int alpha, const int beta, const int gamma) {
	TGSLAssert((0 <= alpha) && (alpha <= 2), "ALGEBRA: Invalid first argument for Epsilon.");
	TGSLAssert((0 <= beta) && (beta <= 2), "ALGEBRA: Invalid second argument for Epsilon.");
	TGSLAssert((0 <= gamma) && (gamma <= 2), "ALGEBRA: Invalid second argument for Epsilon.");
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

template <typename EigenVectorType, typename EigenMatrixType>
inline void EigenNormalizedFirstDerivative(const EigenVectorType& v, EigenMatrixType& DvhatDv) {
	// Compute D(v_hat) / Dv, where v_hat = v / |v|
	EigenVectorType v_hat = v.normalized();
	DvhatDv = (EigenMatrixType::Identity() - v_hat * v_hat.transpose()) / v.norm();
}

template <typename EigenVectorType, typename EigenMatrixType>
inline void EigenNormalizedSecondDerivative(const EigenVectorType& v, std::vector<EigenMatrixType>& DDvhatDvv) {
	// Compute DD(v_hat) / Dvv, where v_hat = v / |v|
	nm n = v.size();
	DDvhatDvv.resize(n);
	EigenMatrixType I_n = EigenMatrixType::Identity();
	EigenVectorType v_hat = v.normalized();
	for (size_t alpha = 0; alpha < n; ++alpha) {
		for (size_t beta = 0; beta < n; ++beta) {
			for (size_t gamma = 0; gamma < n; ++gamma) {
				DDvhatDvv[gamma](alpha, beta) = (
					T(3) * v_hat[alpha] * v_hat[beta] * v_hat[gamma]
					- I_n(alpha, beta) * v_hat[gamma]
					- I_n(alpha, gamma) * v_hat[beta]
					- I_n(beta, gamma) * v_hat[alpha]
					) / v.squaredNorm();
			}
		}
	}
}

template <typename EigenVectorType, typename EigenMatrixType>
inline void EigenNormalizedThirdDerivative(const EigenVectorType& v, std::vector<std::vector<EigenMatrixType>>& DDDvhatDvvv) {
	// Compute DDD(v_hat) / Dvvv, where v_hat = v / |v|
	nm n = v.size();
	DDDvhatDvvv.resize(n);
	for (size_t i = 0; i < n; ++i) {
		DDDvhatDvvv[i].resize(n);
	}
	EigenMatrixType I_n = EigenMatrixType::Identity();
	EigenVectorType v_hat = v.normalized();
	for (size_t alpha = 0; alpha < n; ++alpha) {
		for (size_t beta = 0; beta < n; ++beta) {
			for (size_t gamma = 0; gamma < n; ++gamma) {
				for (size_t sigma = 0; sigma < n; ++sigma) {
					DDDvhatDvvv[sigma][gamma](alpha, beta) =
						(T(-15) * v_hat[alpha] * v_hat[beta] * v_hat[gamma] * v_hat[sigma] + T(3) * (
							I_n(gamma, sigma) * v_hat[alpha] * v_hat[beta] + I_n(beta, sigma) * v_hat[alpha] * v_hat[gamma] +
							I_n(alpha, sigma) * v_hat[beta] * v_hat[gamma] + I_n(beta, gamma) * v_hat[alpha] * v_hat[sigma] +
							I_n(alpha, gamma) * v_hat[beta] * v_hat[sigma] + I_n(alpha, beta) * v_hat[gamma] * v_hat[sigma])
							- (I_n(alpha, beta) * I_n(gamma, sigma) + I_n(alpha, gamma) * I_n(beta, sigma)
								+ I_n(alpha, sigma) * I_n(beta, gamma))) / std::pow(v.norm(), T(3));
				}
			}
		}
	}
}

inline void QuaternionVDerivativeFromTwoUnitVectors(const Vector3d& u, const Vector3d& v, Eigen::Matrix<T, 4, 3>& DqDv) {
	// compute q such that v = q * u * q_inv
	//T eps = T(1e-8);
	//TGSLAssert((std::abs(u.norm() - T(1)) < eps), "ALGEBRA::QuaternionFromTwoVectors: vector u is not normalized");
	//TGSLAssert((v.norm() < eps), "ALGEBRA::QuaternionFromTwoVectors: vector v is not normalized");
	Vector3d v_hat = v.normalized();
	Matrix3d DvhatDv;
	EigenNormalizedFirstDerivative(v, DvhatDv);

	T cos = u.dot(v_hat);
	Vector3d DcosDv = DvhatDv.transpose() * u;

	T cos_half = sqrt((T(1) + cos) / T(2));
	T DcoshDcos = T(.25) / cos_half;
	Vector3d DcoshDv = DcoshDcos * DcosDv;

	T sin_half = sqrt((T(1) - cos) / T(2));
	T DsinhDcos = T(-0.25) / sin_half;
	Vector3d DsinhDv = DsinhDcos * DcosDv;

#ifdef TWO_D
	TGSLAssert(false, "ALGEBRA::QuaternionFromTwoUnitVectors: not defined for 2d.");
#endif // TWO_D
	Vector3d uxv = u.cross(v_hat);
	T norm = uxv.norm(); // norm = ||uxv_hat|| = ||1 - (u * v_hat)^2|| = || 1 - cos^2 ||
	Vector3d n = uxv / norm;

	Matrix3d DcrossDv = Matrix3d::Zero();
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t sigma = 0; sigma < 3; ++sigma) {
				for (size_t eta = 0; eta < 3; ++eta) {
					DcrossDv(alpha, beta) += Epsilon(alpha, sigma, eta) * u[sigma] * DvhatDv(eta, beta);
				}
			}
		}
	}

	T DnormDcos = -cos / norm;
	Vector3d DnormDv = DnormDcos * DcosDv;

	Matrix3d DnDv = (DcrossDv - n * DnormDv.transpose()) / norm;

	DqDv.block<1, 3>(0, 0) = DcoshDv.transpose();
	DqDv.block<3, 3>(1, 0) = n * DsinhDv.transpose() + sin_half * DnDv;
}

inline void QuaternionBothVDerivativesFromTwoUnitVectors(const Vector3d& u, const Vector3d& v, Eigen::Matrix<T, 4, 3>& DqDv, std::vector<Eigen::Matrix<T, 4, 3>>& DDqDvv) {
	// compute q such that v = q * u * q_inv
	//T eps = T(1e-8);
	//TGSLAssert((std::abs(u.norm() - T(1)) < eps), "ALGEBRA::QuaternionFromTwoVectors: vector u is not normalized");
	//TGSLAssert((v.norm() < eps), "ALGEBRA::QuaternionFromTwoVectors: vector v is not normalized");
	Vector3d v_hat = v.normalized();
	Matrix3d DvhatDv;
	EigenNormalizedFirstDerivative(v, DvhatDv);
	std::vector<Matrix3d> DDvhatDvv;
	EigenNormalizedSecondDerivative(v, DDvhatDvv);
	
	T cos = u.dot(v_hat);
	Vector3d DcosDv = DvhatDv.transpose() * u;
	Matrix3d DDcosDvv = Matrix3d::Zero();
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t gamma = 0; gamma < 3; ++gamma) {
				DDcosDvv(alpha, beta) += u[gamma] * DDvhatDvv[beta](gamma, alpha);
			}
		}
	}

	T cos_half = sqrt((T(1) + cos) / T(2));
	T DcoshDcos = T(.25) / cos_half;
	T DDcoshDcos = T(-1) / (T(16) * std::pow(cos_half, T(3)));
	Vector3d DcoshDv = DcoshDcos * DcosDv;
	Matrix3d DDcoshDvv = DDcoshDcos * DcosDv * DcosDv.transpose() + DcoshDcos * DDcosDvv;

	T sin_half = sqrt((T(1) - cos) / T(2));
	T DsinhDcos = T(-0.25) / sin_half;
	T DDsinhDcos = T(-1) / (T(16) * std::pow(sin_half, T(3)));
	Vector3d DsinhDv = DsinhDcos * DcosDv;
	Matrix3d DDsinhDvv = DDsinhDcos * DcosDv * DcosDv.transpose() + DsinhDcos * DDcosDvv;

#ifdef TWO_D
	TGSLAssert(false, "ALGEBRA::QuaternionFromTwoUnitVectors: not defined for 2d.");
#endif // TWO_D
	Vector3d uxv = u.cross(v_hat);
	T norm = uxv.norm(); // norm = ||uxv_hat|| = ||1 - (u * v_hat)^2|| = || 1 - cos^2 ||
	Vector3d n = uxv / norm;

	Matrix3d DcrossDv = Matrix3d::Zero();
	std::vector<Matrix3d> DDcrossDvv;
	DDcrossDvv.resize(3);
	for (size_t gamma = 0; gamma < 3; ++gamma) {
		DDcrossDvv[gamma].setZero();
	}
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t sigma = 0; sigma < 3; ++sigma) {
				for (size_t eta = 0; eta < 3; ++eta) {
					DcrossDv(alpha, beta) += Epsilon(alpha, sigma, eta) * u[sigma] * DvhatDv(eta, beta);
					for (size_t gamma = 0; gamma < 3; ++gamma) {
						DDcrossDvv[gamma](alpha, beta) += Epsilon(alpha, sigma, eta) * u[sigma] * DDvhatDvv[gamma](eta, beta);
					}
				}
			}
		}
	}

	T DnormDcos = -cos / norm;
	T DDnormDcos = T(-1) / norm + cos * DnormDcos / (norm * norm);
	Vector3d DnormDv = DnormDcos * DcosDv;
	Matrix3d DDnormDvv = DDnormDcos * DcosDv * DcosDv.transpose() + DnormDcos * DDcosDvv;

	Matrix3d DnDv = (DcrossDv - n * DnormDv.transpose()) / norm;
	std::vector<Matrix3d> DDnDvv;
	DDnDvv.resize(3);
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t gamma = 0; gamma < 3; ++gamma) {
				DDnDvv[gamma](alpha, beta) = (
					DDcrossDvv[gamma](alpha, beta)
					- DnDv(alpha, beta) * DnormDv[gamma] - DnDv(alpha, gamma) * DnormDv[beta]
					- n[alpha] * DDnormDvv(beta, gamma)
					) / norm;
			}
		}
	}

	DqDv.block<1, 3>(0, 0) = DcoshDv.transpose();
	DqDv.block<3, 3>(1, 0) = n * DsinhDv.transpose() + sin_half * DnDv;

	DDqDvv.resize(3);
	for (size_t beta = 0; beta < 3; ++beta) {
		for (size_t gamma = 0; gamma < 3; ++gamma) {
			DDqDvv[gamma](0, beta) = DDcoshDvv(beta, gamma);
			for (size_t alpha = 0; alpha < 3; ++alpha) {
				DDqDvv[gamma](alpha + 1, beta) =
					n[alpha] * DDsinhDvv(beta, gamma)
					+ DsinhDv[beta] * DnDv(alpha, gamma) + DsinhDv[gamma] * DnDv(alpha, beta)
					+ sin_half * DDnDvv[gamma](alpha, beta);
			}
		}
	}
}

inline void QuaternionThreeVDerivativesFromTwoUnitVectors(const Vector3d& u, const Vector3d& v, Eigen::Matrix<T, 4, 3>& DqDv, std::vector<Eigen::Matrix<T, 4, 3>>& DDqDvv, std::vector<std::vector<Eigen::Matrix<T, 4, 3>>>& DDDqDvvv) {
	// compute q such that v = q * u * q_inv
	//T eps = T(1e-8);
	//TGSLAssert((std::abs(u.norm() - T(1)) < eps), "ALGEBRA::QuaternionFromTwoVectors: vector u is not normalized");
	//TGSLAssert((v.norm() < eps), "ALGEBRA::QuaternionFromTwoVectors: vector v is not normalized");
	Vector3d v_hat = v.normalized();
	Matrix3d DvhatDv;
	EigenNormalizedFirstDerivative(v, DvhatDv);
	std::vector<Matrix3d> DDvhatDvv;
	EigenNormalizedSecondDerivative(v, DDvhatDvv);
	std::vector<std::vector<Matrix3d>> DDDvhatDvvv;
	EigenNormalizedThirdDerivative(v, DDDvhatDvvv);

	///////////////////////////////////
	// cos
	///////////////////////////////////
	T cos = u.dot(v_hat);
	Vector3d DcosDv = DvhatDv.transpose() * u;
	Matrix3d DDcosDvv = Matrix3d::Zero();
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t sigma = 0; sigma < 3; ++sigma) {
				DDcosDvv(alpha, beta) += u[sigma] * DDvhatDvv[beta](sigma, alpha);
			}
		}
	}
	std::vector<Matrix3d> DDDcosDvvv;
	DDDcosDvvv.resize(3);
	for (size_t gamma = 0; gamma < 3; ++gamma) {
		DDDcosDvvv[gamma].setZero();
		for (size_t alpha = 0; alpha < 3; ++alpha) {
			for (size_t beta = 0; beta < 3; ++beta) {
				for (size_t sigma = 0; sigma < 3; ++sigma) {
					DDDcosDvvv[gamma](alpha, beta) += u[sigma] * DDDvhatDvvv[gamma][beta](sigma, alpha);
				}
			}
		}
	}

	///////////////////////////////////
	// cos_half
	///////////////////////////////////
	T cos_half = sqrt((T(1) + cos) / T(2));
	T DcoshDcos = T(.25) / cos_half;
	T DDcoshDcos = T(-1) / (T(16) * std::pow(cos_half, T(3)));
	T DDDcoshDcos = T(3) / (T(64) * std::pow(cos_half, T(5)));
	Vector3d DcoshDv = DcoshDcos * DcosDv;
	Matrix3d DDcoshDvv = DDcoshDcos * DcosDv * DcosDv.transpose() + DcoshDcos * DDcosDvv;
	std::vector<Matrix3d> DDDcoshDvvv;
	DDDcoshDvvv.resize(3);
	for (size_t gamma = 0; gamma < 3; ++gamma) {
		for (size_t alpha = 0; alpha < 3; ++alpha) {
			for (size_t beta = 0; beta < 3; ++beta) {
				DDDcoshDvvv[gamma](alpha, beta) =
					DDDcoshDcos * DcosDv[alpha] * DcosDv[beta] * DcosDv[gamma] +
					DDcoshDcos * (
						DDcosDvv(alpha, gamma) * DcosDv[beta] +
						DDcosDvv(beta, gamma) * DcosDv[alpha] +
						DDcosDvv(alpha, beta) * DcosDv[gamma]) +
					DcoshDcos * DDDcosDvvv[gamma](alpha, beta);
			}
		}
	}

	///////////////////////////////////
	// sin_half
	///////////////////////////////////
	T sin_half = sqrt((T(1) - cos) / T(2));
	T DsinhDcos = T(-0.25) / sin_half;
	T DDsinhDcos = T(-1) / (T(16) * std::pow(sin_half, T(3)));
	T DDDsinhDcos = T(-3) / (T(64) * std::pow(sin_half, T(5)));
	Vector3d DsinhDv = DsinhDcos * DcosDv;
	Matrix3d DDsinhDvv = DDsinhDcos * DcosDv * DcosDv.transpose() + DsinhDcos * DDcosDvv;
	std::vector<Matrix3d> DDDsinhDvvv;
	DDDsinhDvvv.resize(3);
	for (size_t gamma = 0; gamma < 3; ++gamma) {
		for (size_t alpha = 0; alpha < 3; ++alpha) {
			for (size_t beta = 0; beta < 3; ++beta) {
				DDDsinhDvvv[gamma](alpha, beta) =
					DDDsinhDcos * DcosDv[alpha] * DcosDv[beta] * DcosDv[gamma] +
					DDsinhDcos * (
						DDcosDvv(alpha, gamma) * DcosDv[beta] +
						DDcosDvv(beta, gamma) * DcosDv[alpha] +
						DDcosDvv(alpha, beta) * DcosDv[gamma]) +
					DsinhDcos * DDDcosDvvv[gamma](alpha, beta);
			}
		}
	}

#ifdef TWO_D
	TGSLAssert(false, "ALGEBRA::QuaternionFromTwoUnitVectors: not defined for 2d.");
#endif // TWO_D
	Vector3d uxv = u.cross(v_hat);
	T norm = uxv.norm(); // norm = ||uxv_hat|| = ||1 - (u * v_hat)^2|| = || 1 - cos^2 ||
	Vector3d n = uxv / norm;

	///////////////////////////////////
	// u x v_hat
	///////////////////////////////////
	Matrix3d DcrossDv = Matrix3d::Zero();
	std::vector<Matrix3d> DDcrossDvv; DDcrossDvv.resize(3);
	std::vector<std::vector<Matrix3d>> DDDcrossDvvv; DDDcrossDvvv.resize(3);
	for (size_t gamma = 0; gamma < 3; ++gamma) {
		DDcrossDvv[gamma].setZero();
		DDDcrossDvvv[gamma].resize(3);
		for (size_t sigma = 0; sigma < 3; ++sigma) {
			DDDcrossDvvv[gamma][sigma].setZero();
		}
	}
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t sigma = 0; sigma < 3; ++sigma) {
				for (size_t eta = 0; eta < 3; ++eta) {
					DcrossDv(alpha, beta) += Epsilon(alpha, sigma, eta) * u[sigma] * DvhatDv(eta, beta);
					for (size_t gamma = 0; gamma < 3; ++gamma) {
						DDcrossDvv[gamma](alpha, beta) += Epsilon(alpha, sigma, eta) * u[sigma] * DDvhatDvv[gamma](eta, beta);
						for (size_t delta = 0; delta < 3; ++delta) {
							DDDcrossDvvv[delta][gamma](alpha, beta) +=
								Epsilon(alpha, sigma, eta) * u[sigma] * DDDvhatDvvv[delta][gamma](eta, beta);
						}
					}
				}
			}
		}
	}

	///////////////////////////////////
	// || u x v_hat ||
	///////////////////////////////////
	T DnormDcos = -cos / norm;
	//T DDnormDcos = T(-1) / norm + cos * DnormDcos / (norm * norm);
	T DDnormDcos = T(-1) / std::pow(norm, T(3));
	T DDDnormDcos = T(-3) * cos / std::pow(norm, T(5));
	Vector3d DnormDv = DnormDcos * DcosDv;
	Matrix3d DDnormDvv = DDnormDcos * DcosDv * DcosDv.transpose() + DnormDcos * DDcosDvv;
	std::vector<Matrix3d> DDDnormDvvv; DDDnormDvvv.resize(3);
	for (size_t gamma = 0; gamma < 3; ++gamma) {
		for (size_t alpha = 0; alpha < 3; ++alpha) {
			for (size_t beta = 0; beta < 3; ++beta) {
				DDDnormDvvv[gamma](alpha, beta) =
					DDDnormDcos * DcosDv[alpha] * DcosDv[beta] * DcosDv[gamma] +
					DDnormDcos * (
						DDcosDvv(alpha, gamma) * DcosDv[beta] +
						DDcosDvv(beta, gamma) * DcosDv[alpha] +
						DDcosDvv(alpha, beta) * DcosDv[gamma]) +
					DnormDcos * DDDcosDvvv[gamma](alpha, beta);
			}
		}
	}

	///////////////////////////////////
	// n = (uxv_hat).normalize()
	///////////////////////////////////
	Matrix3d DnDv = (DcrossDv - n * DnormDv.transpose()) / norm;
	std::vector<Matrix3d> DDnDvv; DDnDvv.resize(3);
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t gamma = 0; gamma < 3; ++gamma) {
				DDnDvv[gamma](alpha, beta) = (
					DDcrossDvv[gamma](alpha, beta)
					- DnDv(alpha, beta) * DnormDv[gamma] - DnDv(alpha, gamma) * DnormDv[beta]
					- n[alpha] * DDnormDvv(beta, gamma)
					) / norm;
			}
		}
	}
	std::vector<std::vector<Matrix3d>> DDDnDvvv; DDDnDvvv.resize(3);
	for (size_t delta = 0; delta < 3; ++delta) {
		DDDnDvvv[delta].resize(3);
	}
	for (size_t alpha = 0; alpha < 3; ++alpha) {
		for (size_t beta = 0; beta < 3; ++beta) {
			for (size_t gamma = 0; gamma < 3; ++gamma) {
				for (size_t delta = 0; delta < 3; ++delta) {
					DDDnDvvv[delta][gamma](alpha, beta) = (
						DDDcrossDvvv[delta][gamma](alpha, beta) - (
							DDnDvv[delta](alpha, gamma) * DnormDv[beta] +
							DDnDvv[delta](alpha, beta) * DnormDv[gamma] +
							DDnDvv[gamma](alpha, beta) * DnormDv[delta]) - (
								DnDv(alpha, gamma) * DDnormDvv(beta, delta) +
								DnDv(alpha, beta) * DDnormDvv(gamma, delta) +
								DnDv(alpha, delta) * DDnormDvv(beta, gamma)) -
						n[alpha] * DDDnormDvvv[delta](beta, gamma)) / norm;
				}
			}
		}
	}

	DqDv.block<1, 3>(0, 0) = DcoshDv.transpose();
	DqDv.block<3, 3>(1, 0) = n * DsinhDv.transpose() + sin_half * DnDv;

	DDqDvv.resize(3); DDDqDvvv.resize(3);
	for (size_t delta = 0; delta < 3; ++delta) {
		DDDqDvvv[delta].resize(3);
	}
	for (size_t beta = 0; beta < 3; ++beta) {
		for (size_t gamma = 0; gamma < 3; ++gamma) {
			DDqDvv[gamma](0, beta) = DDcoshDvv(beta, gamma);
			for (size_t alpha = 0; alpha < 3; ++alpha) {
				DDqDvv[gamma](alpha + 1, beta) =
					n[alpha] * DDsinhDvv(beta, gamma)
					+ DsinhDv[beta] * DnDv(alpha, gamma) + DsinhDv[gamma] * DnDv(alpha, beta)
					+ sin_half * DDnDvv[gamma](alpha, beta);
			}
			for (size_t delta = 0; delta < 3; ++delta) {
				DDDqDvvv[delta][gamma](0, beta) = DDDcoshDvvv[delta](beta, gamma);
				for (size_t alpha = 0; alpha < 3; ++alpha) {
					DDDqDvvv[delta][gamma](alpha + 1, beta) =
						n[alpha] * DDDsinhDvvv[delta](beta, gamma) + 
						sin_half * DDDnDvvv[delta][gamma](alpha, beta) +
						DDsinhDvv(beta, gamma) * DnDv(alpha, delta) +
						DDsinhDvv(beta, delta) * DnDv(alpha, gamma) +
						DDsinhDvv(gamma, delta) * DnDv(alpha, beta) +
						DsinhDv[beta] * DDnDvv[delta](alpha, gamma) +
						DsinhDv[gamma] * DDnDvv[delta](alpha, beta) +
						DsinhDv[delta] * DDnDvv[gamma](alpha, beta);
				}
			}
		}
	}
}


inline T AngleFrom2DQuaternion(const Vector4T& q) {
	// theta ranges from -pi to pi
	// Assuming q[1] = q[2] = 0
	//T tol = 1e-12;
	//TGSLAssert((std::abs(q[1]) < tol) && (std::abs(q[2]) < tol), "q is not a 2D rotation.");
	T theta = T(2) * atan2(q[3], q[0]);
	if (theta > pi) {
		theta -= T(2) * pi;
	}
	else if (theta < -pi) {
		theta += T(2) * pi;
	}
	return theta * T(180) / pi;
}

}}

inline std::ostream& operator<<(std::ostream& os, const TGSL::ALGEBRA::DenseMatrix& A)
{
	TGSL::Vector2I dims=A.Size();
	TGSL::sz m=dims[0],n=dims[1];
	os << std::endl;
	for(TGSL::sz i=0;i<m;i++){
		for(TGSL::sz j=0;j<n-1;j++){
			os << A(i,j) << " , ";
		}
		os << A(i,n-1) << std::endl;
	}
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TGSL::Vector4T& v)
{
	os << std::endl;
	for(TGSL::sz i=0;i<3;i++){
		os << v[i] << " , ";
	}
	os << v[3] << std::endl;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TGSL::Vector& v)
{
	os << std::endl;
	for(TGSL::sz i=0;i<TGSL::d;i++){
		os << v[i] << " , ";
	}
	os << std::endl;
  return os;
}