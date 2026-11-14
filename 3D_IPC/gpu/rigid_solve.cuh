#pragma once

#include <cmath>
#include <cfloat>

// Same lower-triangle pivoted LDLT and diagonal pseudoinverse policy as
// Eigen 3.4 LDLT::solve used by our CPU rigid GS solver. Extracted without
// numerical changes from the previous rigid_pipeline.cu device function.
// Algorithm adapted from Eigen/src/Cholesky/LDLT.h (MPL-2.0).
// Copyright (C) 2008-2010 Gael Guennebaud and Benoit Jacob.
// This Source Code Form is subject to the Mozilla Public License, v. 2.0.
// If a copy was not distributed, obtain it at https://mozilla.org/MPL/2.0/.
#if defined(__CUDACC__)
#define IPC_RIGID_SOLVE_HD __host__ __device__
#else
#define IPC_RIGID_SOLVE_HD
#endif

namespace ipc_gpu::solver {
using std::isfinite;
IPC_RIGID_SOLVE_HD inline bool rigid_ldlt3(const double* H,const double* g,double* x)
{
    double a[3][3],rhs[3]; int swaps[3]={0,1,2}; bool zero_seen=false;
    for (int i=0;i<3;++i) {
        if (!isfinite(g[i])) return false;
        rhs[i]=g[i];
        for (int j=0;j<3;++j) {
            if (!isfinite(H[3*i+j])) return false;
            a[i][j]=H[3*i+j];
        }
    }
    for (int k=0;k<3;++k) {
        int pivot=k;
        for (int i=k+1;i<3;++i) if (fabs(a[i][i])>fabs(a[pivot][pivot])) pivot=i;
        swaps[k]=pivot;
        if (pivot!=k) {
            for (int j=0;j<k;++j) { const double t=a[k][j]; a[k][j]=a[pivot][j]; a[pivot][j]=t; }
            for (int i=pivot+1;i<3;++i) { const double t=a[i][k]; a[i][k]=a[i][pivot]; a[i][pivot]=t; }
            const double t=a[k][k]; a[k][k]=a[pivot][pivot]; a[pivot][pivot]=t;
            for (int i=k+1;i<pivot;++i) { const double u=a[i][k]; a[i][k]=a[pivot][i]; a[pivot][i]=u; }
        }
        if (k>0) {
            double temp[3]{},diagonal=0;
            for (int j=0;j<k;++j) { temp[j]=a[j][j]*a[k][j]; diagonal+=a[k][j]*temp[j]; }
            a[k][k]-=diagonal;
            for (int i=k+1;i<3;++i) {
                double product=0;
                for (int j=0;j<k;++j) product+=a[i][j]*temp[j];
                a[i][k]-=product;
            }
        }
        const double diagonal=a[k][k];
        if (!isfinite(diagonal)) return false;
        const bool nonzero=fabs(diagonal)>0;
        if (k==0 && !nonzero) {
            for (int i=0;i<3;++i) for (int j=0;j<i;++j) if (a[i][j]!=0) return false;
            for (int i=0;i<3;++i) x[i]=0;
            return true;
        }
        if (zero_seen && nonzero) return false;
        if (!nonzero) zero_seen=true;
        for (int i=k+1;i<3;++i) {
            if (nonzero) a[i][k]/=diagonal;
            else if (a[i][k]!=0) return false;
            if (!isfinite(a[i][k])) return false;
        }
    }
    for (int k=0;k<3;++k) { const double t=rhs[k]; rhs[k]=rhs[swaps[k]]; rhs[swaps[k]]=t; }
    for (int i=0;i<3;++i) for (int j=0;j<i;++j) rhs[i]-=a[i][j]*rhs[j];
    for (int i=0;i<3;++i) rhs[i]=fabs(a[i][i])>DBL_MIN ? rhs[i]/a[i][i] : 0.0;
    for (int i=2;i>=0;--i) for (int j=i+1;j<3;++j) rhs[i]-=a[j][i]*rhs[j];
    for (int k=2;k>=0;--k) { const double t=rhs[k]; rhs[k]=rhs[swaps[k]]; rhs[swaps[k]]=t; }
    for (int i=0;i<3;++i) { x[i]=rhs[i]; if (!isfinite(x[i])) return false; }
    return true;
}
} // namespace ipc_gpu::solver

#undef IPC_RIGID_SOLVE_HD
