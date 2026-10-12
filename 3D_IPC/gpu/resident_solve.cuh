#pragma once
#include <cmath>
#include <cfloat>
#ifdef __CUDACC__
#define IPC_SOLVE_HD __host__ __device__
#else
#define IPC_SOLVE_HD
#endif

namespace ipc_gpu::resident {
// Small rank-revealing solve. CPU matrix3d_inverse uses its explicit inverse
// in the regular case and complete orthogonal decomposition otherwise. This
// device implementation uses column-pivoted QR followed by QR of R^T for the
// same minimum-norm problem, without dynamic memory or squared condition
// numbers. It is not a promise of bitwise Eigen arithmetic equivalence.
IPC_SOLVE_HD inline double norm3(const double* v,int begin) {
    double scale=0;
    for(int i=begin;i<3;++i)scale=fmax(scale,fabs(v[i]));
    if(scale==0)return 0;
    double sum=0;for(int i=begin;i<3;++i){double q=v[i]/scale;sum+=q*q;}
    return scale*sqrt(sum);
}
IPC_SOLVE_HD inline bool finite_number(double v){return (v-v)==0.0;}
IPC_SOLVE_HD inline void householder(double a[3][3],int column,int rows_begin,
    double q[3][3],double* rhs,int columns_end) {
    double v[3]={0,0,0};
    for(int i=rows_begin;i<3;++i)v[i]=a[i][column];
    const double norm=norm3(v,rows_begin);
    if(norm==0)return;
    const double beta=v[rows_begin]>=0?-norm:norm;
    // Scale before subtraction to avoid overflow in v0-beta.
    for(int i=rows_begin;i<3;++i)v[i]/=norm;
    v[rows_begin]-=beta/norm;
    double vv=0;for(int i=rows_begin;i<3;++i)vv+=v[i]*v[i];
    const double tau=2.0/vv;
    for(int j=column;j<columns_end;++j) {
        double d=0;for(int i=rows_begin;i<3;++i)d+=v[i]*a[i][j];
        for(int i=rows_begin;i<3;++i)a[i][j]-=tau*v[i]*d;
    }
    a[rows_begin][column]=beta;
    for(int i=rows_begin+1;i<3;++i)a[i][column]=0;
    if(rhs){double d=0;for(int i=rows_begin;i<3;++i)d+=v[i]*rhs[i];for(int i=rows_begin;i<3;++i)rhs[i]-=tau*v[i]*d;}
    if(q)for(int r=0;r<3;++r){double d=0;for(int i=rows_begin;i<3;++i)d+=q[r][i]*v[i];for(int i=rows_begin;i<3;++i)q[r][i]-=tau*d*v[i];}
}
IPC_SOLVE_HD inline bool solve3(const double* h,const double* g,double* solution) {
    double scale=0;
    for(int i=0;i<3;++i){solution[i]=0;if(!finite_number(g[i]))return false;}
    for(int i=0;i<9;++i){if(!finite_number(h[i]))return false;scale=fmax(scale,fabs(h[i]));}
    if(scale==0)return true;
    double c[9]={h[4]*h[8]-h[5]*h[7],h[2]*h[7]-h[1]*h[8],h[1]*h[5]-h[2]*h[4],
        h[5]*h[6]-h[3]*h[8],h[0]*h[8]-h[2]*h[6],h[2]*h[3]-h[0]*h[5],
        h[3]*h[7]-h[4]*h[6],h[1]*h[6]-h[0]*h[7],h[0]*h[4]-h[1]*h[3]};
    const double det=(h[0]*c[0]+h[1]*c[3])+h[2]*c[6];
    const double tolerance=64.0*DBL_EPSILON*scale*scale*scale;
    if(finite_number(det)&&fabs(det)>tolerance) {
        const double inv_det=1.0/det;
        for(int r=0;r<3;++r)solution[r]=((c[3*r]*inv_det)*g[0]+(c[3*r+1]*inv_det)*g[1])+(c[3*r+2]*inv_det)*g[2];
    }else{
        double a[3][3],y[3];int permutation[3]={0,1,2};
        for(int r=0;r<3;++r){y[r]=g[r]/scale;for(int col=0;col<3;++col)a[r][col]=h[3*r+col]/scale;}
        double max_pivot=0;
        for(int k=0;k<3;++k){
            int best=k;double largest=-1;
            for(int col=k;col<3;++col){double v[3]={a[0][col],a[1][col],a[2][col]};double n=norm3(v,k);if(n>largest){largest=n;best=col;}}
            if(best!=k){for(int r=0;r<3;++r){double t=a[r][k];a[r][k]=a[r][best];a[r][best]=t;}int t=permutation[k];permutation[k]=permutation[best];permutation[best]=t;}
            householder(a,k,k,nullptr,y,3);max_pivot=fmax(max_pivot,fabs(a[k][k]));
        }
        int rank=0;for(int k=0;k<3;++k)if(fabs(a[k][k])>3.0*DBL_EPSILON*max_pivot)++rank;
        if(rank==0)return true;
        // R_top^T = V T; solve T^T z = Q^T g, then x = P V z.
        double t[3][3]{},v[3][3]{};
        for(int r=0;r<3;++r){v[r][r]=1;for(int col=0;col<rank;++col)t[r][col]=a[col][r];}
        for(int k=0;k<rank;++k)householder(t,k,k,v,nullptr,rank);
        double z[3]={0,0,0};
        for(int r=0;r<rank;++r){double sum=y[r];for(int j=0;j<r;++j)sum-=t[j][r]*z[j];z[r]=sum/t[r][r];}
        for(int r=0;r<3;++r){double sum=0;for(int j=0;j<rank;++j)sum+=v[r][j]*z[j];solution[permutation[r]]=sum;}
    }
    for(int i=0;i<3;++i)if(!finite_number(solution[i]))return false;
    return true;
}
} // namespace ipc_gpu::resident
#undef IPC_SOLVE_HD
