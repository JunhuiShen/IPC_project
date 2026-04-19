// gpu_solver.cu
// CUDA implementation of gpu_solver.h  (gpu_build_jacobi_predictions and
// gpu_parallel_commit).  The CPU stub is gpu_solver_stub.cpp — read that
// first; the algorithm is identical, only the execution model changes.
//
// TODO 3 — add CUDA to LANGUAGES in CMakeLists.txt and swap stub for this file

#include "gpu_solver.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cg = cooperative_groups;

// ============================================================================
// Scalar helpers
// ============================================================================

__device__ static double dev_clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ static double dev_min2(double a, double b) { return a < b ? a : b; }
__device__ static double dev_max2(double a, double b) { return a > b ? a : b; }

// ============================================================================
// 3-vector helpers  (plain double[3])
// ============================================================================

__device__ static double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ static double norm3(const double v[3]) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

__device__ static void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ static void sub3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[0]-b[0]; out[1] = a[1]-b[1]; out[2] = a[2]-b[2];
}

__device__ static void add3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[0]+b[0]; out[1] = a[1]+b[1]; out[2] = a[2]+b[2];
}

__device__ static void scale3(double s, const double v[3], double out[3]) {
    out[0] = s*v[0]; out[1] = s*v[1]; out[2] = s*v[2];
}

__device__ static void addscale3(double s, const double v[3], double acc[3]) {
    acc[0] += s*v[0]; acc[1] += s*v[1]; acc[2] += s*v[2];
}

// out = a + s*b
__device__ static void addscaled3(const double a[3], double s, const double b[3], double out[3]) {
    out[0] = a[0]+s*b[0]; out[1] = a[1]+s*b[1]; out[2] = a[2]+s*b[2];
}

// Load vertex i from flat position array [nv*3]
__device__ static void loadv(const double* x, int i, double v[3]) {
    v[0] = x[i*3]; v[1] = x[i*3+1]; v[2] = x[i*3+2];
}

// ============================================================================
// 3x3 matrix helpers  (row-major, double H[9])
// ============================================================================

__device__ static void mat33_inverse(const double H[9], double inv[9]) {
    double det = H[0]*(H[4]*H[8] - H[5]*H[7])
               - H[1]*(H[3]*H[8] - H[5]*H[6])
               + H[2]*(H[3]*H[7] - H[4]*H[6]);
    double id = 1.0 / det;
    inv[0] =  id * (H[4]*H[8] - H[5]*H[7]);
    inv[1] = -id * (H[1]*H[8] - H[2]*H[7]);
    inv[2] =  id * (H[1]*H[5] - H[2]*H[4]);
    inv[3] = -id * (H[3]*H[8] - H[5]*H[6]);
    inv[4] =  id * (H[0]*H[8] - H[2]*H[6]);
    inv[5] = -id * (H[0]*H[5] - H[2]*H[3]);
    inv[6] =  id * (H[3]*H[7] - H[4]*H[6]);
    inv[7] = -id * (H[0]*H[7] - H[1]*H[6]);
    inv[8] =  id * (H[0]*H[4] - H[1]*H[3]);
}

// H += s * (a ⊗ a)  (rank-1 update)
__device__ static void outer3_add(double s, const double a[3], double H[9]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            H[r*3+c] += s * a[r] * a[c];
}

// H += s * I3
__device__ static void adddiag3(double s, double H[9]) {
    H[0] += s; H[4] += s; H[8] += s;
}

// mat-vec: out = M * v  (3x3 row-major * vec3)
__device__ static void matvec3(const double M[9], const double v[3], double out[3]) {
    out[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
    out[1] = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
    out[2] = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
}

// ============================================================================
// Corotated elastic — raw doubles, no Eigen
// ============================================================================
//
// F is 3x2 stored row-major: F[i*2+j] = F_{ij}
// Dm_inv stored Eigen col-major: dm[0]=(0,0), dm[1]=(1,0), dm[2]=(0,1), dm[3]=(1,1)
//
// Computes elastic contribution to g[3] and H[9] for local vertex `a` in triangle ti.

__device__ static void corotated_vertex_gh(
    int a,                   // local vertex index 0/1/2 within triangle
    const double* x_v0,     // positions of the 3 triangle vertices (each double[3])
    const double* x_v1,
    const double* x_v2,
    const double* dm,        // Dm_inv for this triangle (4 doubles, col-major)
    double ref_area,
    double mu, double lambda, double dt2,
    double g[3], double H[9])  // accumulated OUT
{
    // Build deformation gradient F (3x2)
    double Ds0[3], Ds1[3];  // columns of Ds
    sub3(x_v1, x_v0, Ds0);
    sub3(x_v2, x_v0, Ds1);

    // F = Ds * Dm_inv: F[i][j] = Ds0[i]*dm_at(0,j) + Ds1[i]*dm_at(1,j)
    // dm_at(r,c) = dm[c*2+r]  (col-major 2x2)
    // dm_at(0,0)=dm[0], dm_at(1,0)=dm[1], dm_at(0,1)=dm[2], dm_at(1,1)=dm[3]
    double F[6];  // row-major 3x2
    for (int i = 0; i < 3; ++i) {
        F[i*2+0] = Ds0[i]*dm[0] + Ds1[i]*dm[1];  // col j=0
        F[i*2+1] = Ds0[i]*dm[2] + Ds1[i]*dm[3];  // col j=1
    }

    // C = F^T F  (2x2 symmetric)
    double c00 = 0, c01 = 0, c11 = 0;
    for (int k = 0; k < 3; ++k) {
        c00 += F[k*2]*F[k*2];
        c01 += F[k*2]*F[k*2+1];
        c11 += F[k*2+1]*F[k*2+1];
    }

    // 2x2 symmetric eigensolver for C
    double d_half = (c00 - c11) * 0.5;
    double disc   = sqrt(d_half*d_half + c01*c01);
    double lam0   = dev_max2((c00+c11)*0.5 - disc, 1e-24);
    double lam1   = dev_max2((c00+c11)*0.5 + disc, 1e-24);

    // Eigenvector for lam0: unnorm = [-c01, d_half+disc]
    double ex = -c01, ey = d_half + disc;
    double elen = sqrt(ex*ex + ey*ey);
    if (elen < 1e-14) { ex = 1.0; ey = 0.0; }
    else { ex /= elen; ey /= elen; }
    // U = [[ex, -ey],[ey, ex]] (cols = eigenvectors)

    double s0 = sqrt(lam0), s1 = sqrt(lam1);
    double J = s0 * s1;
    double traceS = s0 + s1;

    // SInv = U * diag(1/s0, 1/s1) * U^T
    double si0 = 1.0/s0, si1 = 1.0/s1;
    double SInv00 = ex*ex*si0 + ey*ey*si1;
    double SInv01 = ex*ey*(si0 - si1);
    double SInv11 = ey*ey*si0 + ex*ex*si1;

    // R = F * SInv  (3x2)
    double R[6];
    for (int i = 0; i < 3; ++i) {
        R[i*2+0] = F[i*2+0]*SInv00 + F[i*2+1]*SInv01;
        R[i*2+1] = F[i*2+0]*SInv01 + F[i*2+1]*SInv11;
    }

    // P = 2*mu*(F-R) + lambda*(J-1)*J * FFTFInv
    // Need: FTFinv = C^{-1}, FFTFInv = F * C^{-1}
    double det_C = c00*c11 - c01*c01;
    double idet  = (det_C > 1e-30) ? 1.0/det_C : 0.0;
    double FTFinv00 =  c11*idet, FTFinv01 = -c01*idet, FTFinv11 = c00*idet;

    double FFTFInv[6];  // 3x2
    for (int i = 0; i < 3; ++i) {
        FFTFInv[i*2+0] = F[i*2+0]*FTFinv00 + F[i*2+1]*FTFinv01;
        FFTFInv[i*2+1] = F[i*2+0]*FTFinv01 + F[i*2+1]*FTFinv11;
    }

    double lJ1J = lambda * (J-1.0) * J;
    double P[6];
    for (int i = 0; i < 6; ++i)
        P[i] = 2.0*mu*(F[i]-R[i]) + lJ1J*FFTFInv[i];

    // Shape function gradients (2-vectors):
    // gradN[1] = [dm[0], dm[2]], gradN[2] = [dm[1], dm[3]], gradN[0] = -gradN[1]-gradN[2]
    double gradN[3][2];
    gradN[1][0] = dm[0]; gradN[1][1] = dm[2];
    gradN[2][0] = dm[1]; gradN[2][1] = dm[3];
    gradN[0][0] = -gradN[1][0] - gradN[2][0];
    gradN[0][1] = -gradN[1][1] - gradN[2][1];

    // Gradient contribution: g[gamma] += ref_area * sum_beta P[gamma][beta] * gradN[a][beta]
    double dg_scale = dt2 * ref_area;
    for (int gamma = 0; gamma < 3; ++gamma) {
        double val = P[gamma*2+0]*gradN[a][0] + P[gamma*2+1]*gradN[a][1];
        g[gamma] += dg_scale * val;
    }

    // dPdF (6x6, flat index flatF(m,n)=2*m+n)
    // RRT = R*R^T (3x3)
    double RRT[9] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            RRT[i*3+j] = R[i*2+0]*R[j*2+0] + R[i*2+1]*R[j*2+1];

    // Re: column-flip of R: Re[i][0]=R[i][1], Re[i][1]=-R[i][0]
    // dcdF[2*m+n] = Re[m][n] / traceS
    double dcdF[6];
    for (int m = 0; m < 3; ++m) {
        dcdF[2*m+0] = -R[m*2+1] / traceS;   // flatF(m,0)
        dcdF[2*m+1] =  R[m*2+0] / traceS;    // flatF(m,1)
    }

    // FFTFInvFT = FFTFInv * F^T  (3x3)
    double FFTFInvFT[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            FFTFInvFT[i*3+j] = FFTFInv[i*2+0]*F[j*2+0] + FFTFInv[i*2+1]*F[j*2+1];

    // Build dPdF (6x6)
    // idx[k] -> (m,n) pairs: k=0:(0,0), k=1:(0,1), k=2:(1,0), k=3:(1,1), k=4:(2,0), k=5:(2,1)
    double dRdF[36] = {};
    for (int c1 = 0; c1 < 6; ++c1) {
        int m = c1/2, n = c1%2;
        for (int c2 = 0; c2 < 6; ++c2) {
            int ii = c2/2, j = c2%2;
            double SInv_jn = (j==0&&n==0)?SInv00:(j==1&&n==1)?SInv11:SInv01;
            double v = 0.0;
            if (m == ii) v += SInv_jn;
            v -= RRT[m*3+ii] * SInv_jn;
            // Re[m][n] = R[m*2+(1-n)] * (n==0 ? 1 : -1)
            double Re_mn = (n==0) ? R[m*2+1] : -R[m*2+0];
            v -= dcdF[c2] * Re_mn;
            dRdF[c1*6+c2] = v;
        }
    }

    double dPdF[36] = {};
    for (int c1 = 0; c1 < 6; ++c1) {
        int m = c1/2, n = c1%2;
        for (int c2 = 0; c2 < 6; ++c2) {
            int ii = c2/2, j = c2%2;
            double FTFinv_jn = (j==0&&n==0)?FTFinv00:(j==1&&n==1)?FTFinv11:FTFinv01;
            double v = 0.0;
            if (m == ii) v += lJ1J * FTFinv_jn;
            v -= lJ1J * (FFTFInv[m*2+j]*FFTFInv[ii*2+n] + FFTFInvFT[m*3+ii]*FTFinv_jn);
            double coef2 = 0.5 * lambda * (2.0*J - 1.0) * J;
            v += coef2 * (FFTFInv[ii*2+j]*FFTFInv[m*2+n] + FFTFInv[ii*2+j]*FFTFInv[m*2+n]);
            dPdF[c1*6+c2] = v;
        }
    }
    // dPdF += 2*mu*(I - dRdF)
    for (int k = 0; k < 6; ++k) dPdF[k*6+k] += 2.0*mu;
    for (int k = 0; k < 36; ++k) dPdF[k] -= 2.0*mu * dRdF[k];

    // Hessian contribution
    for (int gamma = 0; gamma < 3; ++gamma) {
        for (int delta = 0; delta < 3; ++delta) {
            double val = 0.0;
            for (int beta = 0; beta < 2; ++beta)
                for (int eta = 0; eta < 2; ++eta)
                    val += dPdF[(gamma*2+beta)*6+(delta*2+eta)] * gradN[a][beta] * gradN[a][eta];
            H[gamma*3+delta] += dg_scale * val;
        }
    }
}

// ============================================================================
// Bending energy — raw doubles, no Eigen
// ============================================================================

__device__ static void bending_vertex_gh_psd(
    int role,
    const double* hv0, const double* hv1, const double* hv2, const double* hv3,
    double kB, double ce, double bar_theta, double dt2,
    double g[3], double H[9])
{
    // e = hv1 - hv0, a = hv2 - hv0, b = hv3 - hv0
    double e[3], a[3], b[3];
    sub3(hv1, hv0, e);
    sub3(hv2, hv0, a);
    sub3(hv3, hv0, b);

    double mA[3], mB[3];
    cross3(e, a, mA);
    cross3(b, e, mB);

    double muA2 = dot3(mA, mA);
    double muB2 = dot3(mB, mB);
    double ell  = norm3(e);

    if (ell < 1e-14 || muA2 < 1e-28 || muB2 < 1e-28) return;

    double e_hat[3]; scale3(1.0/ell, e, e_hat);

    double X = dot3(mA, mB);
    double mAxmB[3]; cross3(mA, mB, mAxmB);
    double Y = dot3(mAxmB, e_hat);
    double theta = atan2(Y, X);

    // grad_theta_node via dX/dnode and dY/dnode
    double dXn[3] = {0,0,0}, dYn[3] = {0,0,0};
    switch (role) {
        case 0: {
            double cA[3], cB[3];
            sub3(hv2, hv1, cA); sub3(hv3, hv1, cB);
            double t1[3], t2[3]; cross3(mB, cA, t1); cross3(mA, cB, t2);
            sub3(t1, t2, dXn);
            double coef = dot3(cA, mB) + dot3(mA, cB);
            double tmp1[3], tmp2[3];
            scale3(coef, e_hat, dYn);
            scale3(-dot3(e_hat, cA), mB, tmp1);
            scale3(-dot3(e_hat, cB), mA, tmp2);
            for(int k=0;k<3;k++) dYn[k] += tmp1[k] + tmp2[k];
            break;
        }
        case 1: {
            double t1[3], t2[3];
            cross3(mA, b, t1); cross3(mB, a, t2);
            sub3(t1, t2, dXn);
            double coef = -(dot3(a, mB) + dot3(mA, b));
            double tmp1[3], tmp2[3];
            scale3(coef, e_hat, dYn);
            scale3(dot3(e_hat, a), mB, tmp1);
            scale3(dot3(e_hat, b), mA, tmp2);
            for(int k=0;k<3;k++) dYn[k] += tmp1[k] + tmp2[k];
            break;
        }
        case 2:
            cross3(mB, e, dXn);
            scale3(-ell, mB, dYn);
            break;
        case 3:
            cross3(e, mA, dXn);
            scale3(-ell, mA, dYn);
            break;
    }

    double denom = muA2 * muB2;
    double gtheta[3];
    for (int k = 0; k < 3; ++k)
        gtheta[k] = (X*dYn[k] - Y*dXn[k]) / denom;

    double delta_theta = theta - bar_theta;
    double dg_scale = dt2 * 2.0 * kB * ce * delta_theta;
    addscale3(dg_scale, gtheta, g);

    // PSD hessian: 2*kB*ce * gtheta ⊗ gtheta
    outer3_add(dt2 * 2.0 * kB * ce, gtheta, H);
}

// ============================================================================
// Segment closest-point helpers (for distance functions)
// ============================================================================

__device__ static void seg_closest(const double* x, const double* a, const double* b,
                                    double* q, double& t) {
    double ab[3]; sub3(b, a, ab);
    double len2 = dot3(ab, ab);
    if (len2 < 1e-30) { t = 0.0; q[0]=a[0]; q[1]=a[1]; q[2]=a[2]; return; }
    double xma[3]; sub3(x, a, xma);
    t = dev_clamp(dot3(xma, ab) / len2, 0.0, 1.0);
    addscaled3(a, t, ab, q);
}

// ============================================================================
// Node-triangle distance (returns region enum 0-7 and fills result)
// Regions: 0=FaceInterior,1=Edge12,2=Edge23,3=Edge31,4=V1,5=V2,6=V3,7=Degen
// ============================================================================

struct NTDist {
    double closest[3];
    double tilde_x[3];
    double normal[3];
    double bary[3];    // barycentric of tilde_x: [l1,l2,l3]
    double phi;
    double dist;
    int    region;     // 0..7
};

__device__ static void node_triangle_distance_dev(
    const double* x, const double* x1, const double* x2, const double* x3,
    double eps, NTDist& out)
{
    out.region = 7; // Degenerate
    out.phi = 0; out.dist = 0;
    for (int k=0;k<3;k++) { out.closest[k]=0; out.tilde_x[k]=0; out.normal[k]=0; out.bary[k]=0; }

    double e12[3], e13[3];
    sub3(x2, x1, e12);
    sub3(x3, x1, e13);
    double n_raw[3]; cross3(e12, e13, n_raw);
    double n_norm = norm3(n_raw);

    if (n_norm <= eps) {
        // Degenerate: find closest of 3 edge closest points
        double q12[3], q23[3], q31[3];
        double t12, t23, t31;
        seg_closest(x, x1, x2, q12, t12);
        seg_closest(x, x2, x3, q23, t23);
        seg_closest(x, x3, x1, q31, t31);
        double d12=0,d23=0,d31=0;
        double tmp[3];
        sub3(x,q12,tmp); d12=norm3(tmp);
        sub3(x,q23,tmp); d23=norm3(tmp);
        sub3(x,q31,tmp); d31=norm3(tmp);
        if (d12 <= d23 && d12 <= d31) {
            for(int k=0;k<3;k++) out.closest[k]=q12[k];
            out.dist = d12;
        } else if (d23 <= d12 && d23 <= d31) {
            for(int k=0;k<3;k++) out.closest[k]=q23[k];
            out.dist = d23;
        } else {
            for(int k=0;k<3;k++) out.closest[k]=q31[k];
            out.dist = d31;
        }
        for(int k=0;k<3;k++) out.tilde_x[k]=out.closest[k];
        return;
    }

    double inv_n = 1.0/n_norm;
    double n[3]; for(int k=0;k<3;k++) n[k]=n_raw[k]*inv_n;
    double x_mx1[3]; sub3(x,x1,x_mx1);
    double phi = dot3(n, x_mx1);
    double tilde_x[3]; addscaled3(x, -phi, n, tilde_x);

    for(int k=0;k<3;k++) { out.normal[k]=n[k]; out.tilde_x[k]=tilde_x[k]; }
    out.phi = phi;

    // Barycentric coords of tilde_x in triangle (x1,x2,x3)
    // p = l1*x1 + l2*x2 + l3*x3, l1+l2+l3=1
    double v0[3], v1[3], v2[3];
    sub3(x2, x1, v0);  // e12
    sub3(x3, x1, v1);  // e13
    sub3(tilde_x, x1, v2);
    double d00 = dot3(v0,v0), d01 = dot3(v0,v1), d11 = dot3(v1,v1);
    double d20 = dot3(v2,v0), d21 = dot3(v2,v1);
    double denom2 = d00*d11 - d01*d01;
    double l2, l3;
    if (fabs(denom2) < 1e-30) { l2=0; l3=0; }
    else { l2=(d11*d20-d01*d21)/denom2; l3=(d00*d21-d01*d20)/denom2; }
    double l1 = 1.0 - l2 - l3;
    out.bary[0]=l1; out.bary[1]=l2; out.bary[2]=l3;

    if (l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0) {
        for(int k=0;k<3;k++) out.closest[k]=tilde_x[k];
        out.dist = fabs(phi);
        out.region = 0; // FaceInterior
        return;
    }

    // Vertex regions
    if (l2 <= 0.0 && l3 <= 0.0) {
        for(int k=0;k<3;k++) out.closest[k]=x1[k];
        double tmp[3]; sub3(x,x1,tmp); out.dist=norm3(tmp);
        out.region = 4; return;
    }
    if (l3 <= 0.0 && l1 <= 0.0) {
        for(int k=0;k<3;k++) out.closest[k]=x2[k];
        double tmp[3]; sub3(x,x2,tmp); out.dist=norm3(tmp);
        out.region = 5; return;
    }
    if (l1 <= 0.0 && l2 <= 0.0) {
        for(int k=0;k<3;k++) out.closest[k]=x3[k];
        double tmp[3]; sub3(x,x3,tmp); out.dist=norm3(tmp);
        out.region = 6; return;
    }

    // Edge regions
    if (l3 < 0.0) {
        double q[3]; double t; seg_closest(tilde_x,x1,x2,q,t);
        for(int k=0;k<3;k++) out.closest[k]=q[k];
        double tmp[3]; sub3(x,q,tmp); out.dist=norm3(tmp);
        out.region = 1; return;
    }
    if (l1 < 0.0) {
        double q[3]; double t; seg_closest(tilde_x,x2,x3,q,t);
        for(int k=0;k<3;k++) out.closest[k]=q[k];
        double tmp[3]; sub3(x,q,tmp); out.dist=norm3(tmp);
        out.region = 2; return;
    }
    if (l2 < 0.0) {
        double q[3]; double t; seg_closest(tilde_x,x3,x1,q,t);
        for(int k=0;k<3;k++) out.closest[k]=q[k];
        double tmp[3]; sub3(x,q,tmp); out.dist=norm3(tmp);
        out.region = 3; return;
    }

    // Numerical fallback: closest edge
    double q12[3],q23[3],q31[3]; double t12,t23,t31;
    seg_closest(tilde_x,x1,x2,q12,t12);
    seg_closest(tilde_x,x2,x3,q23,t23);
    seg_closest(tilde_x,x3,x1,q31,t31);
    double d12=0,d23=0,d31=0; double tmp[3];
    sub3(x,q12,tmp); d12=norm3(tmp);
    sub3(x,q23,tmp); d23=norm3(tmp);
    sub3(x,q31,tmp); d31=norm3(tmp);
    if (d12 <= d23 && d12 <= d31) {
        for(int k=0;k<3;k++) out.closest[k]=q12[k];
        out.dist=d12; out.region=1;
    } else if (d23<=d12 && d23<=d31) {
        for(int k=0;k<3;k++) out.closest[k]=q23[k];
        out.dist=d23; out.region=2;
    } else {
        for(int k=0;k<3;k++) out.closest[k]=q31[k];
        out.dist=d31; out.region=3;
    }
}

// ============================================================================
// Segment-segment distance
// Regions: 0=Interior,1=Edge_s0,2=Edge_s1,3=Edge_t0,4=Edge_t1,
//          5=Corner_s0t0,6=Corner_s0t1,7=Corner_s1t0,8=Corner_s1t1,9=Parallel
// ============================================================================

struct SSDist {
    double cp1[3], cp2[3];
    double s, t, dist;
    int region;
};

__device__ static void classify_st(double s, double t, double tol, int& region) {
    bool s0=(s<=tol), s1=(s>=1.0-tol), t0=(t<=tol), t1=(t>=1.0-tol);
    if (s0&&t0) { region=5; return; }
    if (s0&&t1) { region=6; return; }
    if (s1&&t0) { region=7; return; }
    if (s1&&t1) { region=8; return; }
    if (s0) { region=1; return; }
    if (s1) { region=2; return; }
    if (t0) { region=3; return; }
    if (t1) { region=4; return; }
    region = 0;
}

__device__ static double opt_t_fixed_s(const double* x1, const double* aa,
                                        const double* x3, const double* bb,
                                        double s, double C, double& t_out) {
    double p[3]; addscaled3(x1, s, aa, p);
    if (C <= 0.0) { t_out = 0.0; }
    else {
        double pmx3[3]; sub3(p, x3, pmx3);
        t_out = dev_clamp(dot3(pmx3, bb)/C, 0.0, 1.0);
    }
    double q[3]; addscaled3(x3, t_out, bb, q);
    double diff[3]; sub3(p, q, diff);
    return norm3(diff);
}

__device__ static double opt_s_fixed_t(const double* x1, const double* aa,
                                        const double* x3, const double* bb,
                                        double t, double A, double& s_out) {
    double q[3]; addscaled3(x3, t, bb, q);
    if (A <= 0.0) { s_out = 0.0; }
    else {
        double qmx1[3]; sub3(q, x1, qmx1);
        s_out = dev_clamp(dot3(qmx1, aa)/A, 0.0, 1.0);
    }
    double p[3]; addscaled3(x1, s_out, aa, p);
    double diff[3]; sub3(p, q, diff);
    return norm3(diff);
}

__device__ static void segment_segment_distance_dev(
    const double* x1, const double* x2, const double* x3, const double* x4,
    double eps, SSDist& out)
{
    double aa[3], bb[3], cc[3];
    sub3(x2, x1, aa);
    sub3(x4, x3, bb);
    sub3(x1, x3, cc);

    double A = dot3(aa,aa), B = dot3(aa,bb), C = dot3(bb,bb);
    double D = dot3(aa,cc), E = dot3(bb,cc);
    double Delta = A*C - B*B;

    const bool non_parallel = (Delta > eps * eps);

    if (non_parallel) {
        double s_u = (B*E - C*D) / Delta;
        double t_u = (A*E - B*D) / Delta;
        if (s_u >= 0.0 && s_u <= 1.0 && t_u >= 0.0 && t_u <= 1.0) {
            out.s = s_u; out.t = t_u;
            addscaled3(x1, out.s, aa, out.cp1);
            addscaled3(x3, out.t, bb, out.cp2);
            double diff[3]; sub3(out.cp1, out.cp2, diff);
            out.dist = norm3(diff);
            out.region = 0; return;
        }
    }

    // Check 4 boundary edges
    double best = 1e300; double bs=0, bt=0;
    double t_c, s_c, d;
    d = opt_t_fixed_s(x1,aa,x3,bb,0.0,C,t_c);
    if(d<best){best=d;bs=0.0;bt=t_c;}
    d = opt_t_fixed_s(x1,aa,x3,bb,1.0,C,t_c);
    if(d<best){best=d;bs=1.0;bt=t_c;}
    d = opt_s_fixed_t(x1,aa,x3,bb,0.0,A,s_c);
    if(d<best){best=d;bs=s_c;bt=0.0;}
    d = opt_s_fixed_t(x1,aa,x3,bb,1.0,A,s_c);
    if(d<best){best=d;bs=s_c;bt=1.0;}

    out.s = bs; out.t = bt;
    addscaled3(x1, out.s, aa, out.cp1);
    addscaled3(x3, out.t, bb, out.cp2);
    double diff[3]; sub3(out.cp1, out.cp2, diff);
    out.dist = norm3(diff);
    if (non_parallel) classify_st(out.s, out.t, 1e-14, out.region);
    else out.region = 9;
}

// ============================================================================
// Scalar barrier
// ============================================================================

__device__ static double scalar_barrier_g(double delta, double d_hat) {
    if (delta >= d_hat || delta <= 0.0) return 0.0;
    double s = delta - d_hat;
    return -2.0*s*log(delta/d_hat) - s*s/delta;
}

__device__ static double scalar_barrier_h(double delta, double d_hat) {
    if (delta >= d_hat || delta <= 0.0) return 0.0;
    double ratio = d_hat/delta;
    return ratio*ratio + 2.0*ratio - 3.0 - 2.0*log(delta/d_hat);
}

__device__ static double seg_param_from_closest(const double* q,
                                                  const double* a, const double* b) {
    double ab[3]; sub3(b,a,ab);
    double denom = dot3(ab,ab);
    if (denom <= 0.0) return 0.0;
    double qa[3]; sub3(q,a,qa);
    return dev_clamp(dot3(qa,ab)/denom, 0.0, 1.0);
}

// ============================================================================
// NT barrier gradient + Hessian for vertex `dof`
// ============================================================================

__device__ static void nt_barrier_gh(
    const double* x,  const double* x1, const double* x2, const double* x3,
    double d_hat, int dof,
    double g[3], double H[9])
{
    NTDist dr;
    node_triangle_distance_dev(x, x1, x2, x3, 1e-12, dr);
    double delta = dr.dist;
    double bp  = scalar_barrier_g(delta, d_hat);
    double bpp = scalar_barrier_h(delta, d_hat);
    if (bp == 0.0 && bpp == 0.0) return;
    if (delta <= 0.0) return;

    double u[3]; sub3(x, dr.closest, u);
    scale3(1.0/delta, u, u);

    // Gradient
    double coeff[4] = {0,0,0,0};
    double face_n[3] = {0,0,0};
    bool use_normal = false;

    if (dr.region == 0) { // FaceInterior
        double sphi = (dr.phi > 0) ? 1.0 : (dr.phi < 0) ? -1.0 : 0.0;
        coeff[0] =  bp*sphi;
        coeff[1] = -bp*sphi*dr.bary[0];
        coeff[2] = -bp*sphi*dr.bary[1];
        coeff[3] = -bp*sphi*dr.bary[2];
        for(int k=0;k<3;k++) face_n[k]=dr.normal[k];
        use_normal = true;
    } else if (dr.region == 1) { // Edge12
        double t = seg_param_from_closest(dr.closest, x1, x2);
        coeff[0]=bp; coeff[1]=-bp*(1-t); coeff[2]=-bp*t; coeff[3]=0;
    } else if (dr.region == 2) { // Edge23
        double t = seg_param_from_closest(dr.closest, x2, x3);
        coeff[0]=bp; coeff[1]=0; coeff[2]=-bp*(1-t); coeff[3]=-bp*t;
    } else if (dr.region == 3) { // Edge31
        double t = seg_param_from_closest(dr.closest, x3, x1);
        coeff[0]=bp; coeff[1]=-bp*t; coeff[2]=0; coeff[3]=-bp*(1-t);
    } else if (dr.region == 4) { // V1
        coeff[0]=bp; coeff[1]=-bp;
    } else if (dr.region == 5) { // V2
        coeff[0]=bp; coeff[2]=-bp;
    } else if (dr.region == 6) { // V3
        coeff[0]=bp; coeff[3]=-bp;
    } else if (dr.region == 7) { // Degen
        coeff[0]=bp;
        double d1=0,d2=0,d3=0; double tmp[3];
        sub3(dr.closest,x1,tmp); d1=norm3(tmp);
        sub3(dr.closest,x2,tmp); d2=norm3(tmp);
        sub3(dr.closest,x3,tmp); d3=norm3(tmp);
        if (d1<=d2&&d1<=d3) coeff[1]=-bp;
        else if (d2<=d3) coeff[2]=-bp;
        else coeff[3]=-bp;
    }

    if (use_normal) {
        for(int k=0;k<3;k++) g[k] += coeff[dof] * face_n[k];
    } else {
        for(int k=0;k<3;k++) g[k] += coeff[dof] * u[k];
    }

    // Hessian: dispatch by region
    if (dr.region == 7) return; // degenerate skip

    if (dr.region == 4 || dr.region == 5 || dr.region == 6) {
        int a_idx = (dr.region==4)?1:(dr.region==5)?2:3;
        double sp[4]={0,0,0,0}; sp[0]=1.0; sp[a_idx]=-1.0;
        if (sp[dof]==0.0) return;
        const double* Ya[4] = {x,x1,x2,x3};
        double uu[3]; sub3(Ya[0], Ya[a_idx], uu); scale3(1.0/delta, uu, uu);
        double c1=bpp, c2=bp/delta, sq=sp[dof]*sp[dof];
        for(int k=0;k<3;k++)
            for(int l=0;l<3;l++)
                H[k*3+l] += sq*(c1*uu[k]*uu[l] + c2*((k==l?1.0:0.0)-uu[k]*uu[l]));
        return;
    }

    if (dr.region==1||dr.region==2||dr.region==3) {
        int a_idx, b_idx;
        if (dr.region==1){a_idx=1;b_idx=2;}
        else if(dr.region==2){a_idx=2;b_idx=3;}
        else{a_idx=3;b_idx=1;}

        const double* Ya[4]={x,x1,x2,x3};
        double omega[4]={0,0,0,0}, eps4[4]={0,0,0,0};
        omega[0]=1.0; omega[a_idx]=-1.0;
        eps4[a_idx]=-1.0; eps4[b_idx]=1.0;

        const double* xa=Ya[a_idx]; const double* xb=Ya[b_idx];
        double ee[3], ww[3];
        sub3(xb,xa,ee); sub3(x,xa,ww);
        double alpha=dot3(ww,ee), beta=dot3(ee,ee);
        double tt = alpha/beta;
        double rr[3]; for(int i=0;i<3;i++) rr[i]=ww[i]-tt*ee[i];
        double uu2[3]; for(int i=0;i<3;i++) uu2[i]=rr[i]/delta;

        double t_d[4][3], r_d[4][3][3];
        for(int pp=0;pp<4;pp++) {
            for(int k=0;k<3;k++) {
                double apk=omega[pp]*ee[k]+eps4[pp]*ww[k];
                double bpk=2.0*eps4[pp]*ee[k];
                t_d[pp][k]=apk/beta - alpha*bpk/(beta*beta);
                for(int i=0;i<3;i++) {
                    double dik=(i==k)?1.0:0.0;
                    double dpa=(pp==a_idx)?1.0:0.0;
                    double dpx=(pp==0)?1.0:0.0;
                    double q_d=dpa*dik+t_d[pp][k]*ee[i]+tt*eps4[pp]*dik;
                    r_d[pp][k][i]=dpx*dik-q_d;
                }
            }
        }
        int p=dof;
        for(int k=0;k<3;k++) {
            for(int l=0;l<3;l++) {
                double dkl=(k==l)?1.0:0.0;
                double apk=omega[p]*ee[k]+eps4[p]*ww[k];
                double aql=omega[p]*ee[l]+eps4[p]*ww[l];
                double apkql=2.0*omega[p]*eps4[p]*dkl;
                double bpk=2.0*eps4[p]*ee[k];
                double bql=2.0*eps4[p]*ee[l];
                double bpkql=2.0*eps4[p]*eps4[p]*dkl;
                double tpkql=apkql/beta-(apk*bql+aql*bpk+alpha*bpkql)/(beta*beta)+2.0*alpha*bpk*bql/(beta*beta*beta);
                double ddelta_pk=0,ddelta_ql=0;
                for(int i=0;i<3;i++){ddelta_pk+=uu2[i]*r_d[p][k][i]; ddelta_ql+=uu2[i]*r_d[p][l][i];}
                double proj=0;
                for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
                    double dij=(i==j)?1.0:0.0;
                    proj+=(dij-uu2[i]*uu2[j])*r_d[p][k][i]*r_d[p][l][j];
                }
                proj/=delta;
                double uq=0;
                for(int i=0;i<3;i++){
                    double dik=(i==k)?1.0:0.0, dil=(i==l)?1.0:0.0;
                    double qipkql=tpkql*ee[i]+t_d[p][k]*eps4[p]*dil+t_d[p][l]*eps4[p]*dik;
                    uq+=uu2[i]*qipkql;
                }
                H[k*3+l]+=bpp*ddelta_pk*ddelta_ql+bp*(proj-uq);
            }
        }
        return;
    }

    if (dr.region==0) { // FaceInterior
        double sig_a[4]={0,-1,1,0}, sig_b[4]={0,-1,0,1}, sig_w[4]={1,-1,0,0};
        double aa2[3],bb2[3],ww[3];
        sub3(x2,x1,aa2); sub3(x3,x1,bb2); sub3(x,x1,ww);
        double N[3]={0,0,0};
        // N = a x b using levi-civita
        N[0]=aa2[1]*bb2[2]-aa2[2]*bb2[1];
        N[1]=aa2[2]*bb2[0]-aa2[0]*bb2[2];
        N[2]=aa2[0]*bb2[1]-aa2[1]*bb2[0];
        double eta=norm3(N);
        double nn[3]; scale3(1.0/eta,N,nn);
        double psi=dot3(N,ww);
        double phi2=psi/eta;
        double s_sign=(phi2>0)?1.0:(phi2<0)?-1.0:0.0;

        // Nd[pp][k][i] = sig_a[pp] * ekxb[i][k] + sig_b[pp] * axek[i][k]
        // ekxb[i][k] = sum_n eps_{ikn}*bb2[n]  (precomputed from bb2)
        // axek[i][k] = sum_m eps_{imk}*aa2[m]  (precomputed from aa2)
        // Both are antisymmetric 3x3 matrices:
        //   ekxb[i][k]: [0,bb2[2],-bb2[1]; -bb2[2],0,bb2[0]; bb2[1],-bb2[0],0]  (row=i, col=k)
        //   axek[i][k]: [0,-aa2[2],aa2[1]; aa2[2],0,-aa2[0]; -aa2[1],aa2[0],0]
        double ekxb[3][3] = {{0,bb2[2],-bb2[1]},{-bb2[2],0,bb2[0]},{bb2[1],-bb2[0],0}};
        double axek[3][3] = {{0,-aa2[2],aa2[1]},{aa2[2],0,-aa2[0]},{-aa2[1],aa2[0],0}};

        double Nd[4][3][3]={};
        for(int pp2=0;pp2<4;pp2++)
            for(int k=0;k<3;k++)
                for(int i=0;i<3;i++)
                    Nd[pp2][k][i] = sig_a[pp2]*ekxb[i][k] + sig_b[pp2]*axek[i][k];

        double eta_d[4][3]={}, psi_d[4][3]={}, phi_d[4][3]={};
        for(int pp2=0;pp2<4;pp2++) {
            for(int k=0;k<3;k++) {
                double eta_pk=0;
                for(int i=0;i<3;i++) eta_pk+=nn[i]*Nd[pp2][k][i];
                eta_d[pp2][k]=eta_pk;
                double psi_pk=0;
                for(int i=0;i<3;i++) psi_pk+=Nd[pp2][k][i]*ww[i];
                psi_pk+=sig_w[pp2]*N[k];
                psi_d[pp2][k]=psi_pk;
                phi_d[pp2][k]=psi_pk/eta - psi*eta_pk/(eta*eta);
            }
        }

        int p=dof;
        for(int k=0;k<3;k++) {
            for(int l=0;l<3;l++) {
                double dkl=(k==l)?1.0:0.0;
                // coeff_N2 = sig_a[p]*sig_b[p]-sig_a[p]*sig_b[p] = 0 (as in CPU reference)
                // nN2 = 0
                double proj_NN=0;
                for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
                    double dij=(i==j)?1.0:0.0;
                    proj_NN+=(dij-nn[i]*nn[j])*Nd[p][k][i]*Nd[p][l][j];
                }
                double eta_pkql = proj_NN/eta;
                double psi_pkql=sig_w[p]*Nd[p][k][l]+sig_w[p]*Nd[p][l][k];
                double phi_pkql=psi_pkql/eta
                    -(psi_d[p][k]*eta_d[p][l]+psi_d[p][l]*eta_d[p][k]+psi*eta_pkql)/(eta*eta)
                    +2.0*psi*eta_d[p][k]*eta_d[p][l]/(eta*eta*eta);
                H[k*3+l]+=bpp*phi_d[p][k]*phi_d[p][l]+s_sign*bp*phi_pkql;
            }
        }
    }
}

// ============================================================================
// SS barrier gradient + Hessian for vertex `dof`
// ============================================================================

__device__ static void ss_barrier_gh(
    const double* x1, const double* x2, const double* x3, const double* x4,
    double d_hat, int dof,
    double g[3], double H[9])
{
    SSDist dr;
    segment_segment_distance_dev(x1,x2,x3,x4,1e-12,dr);
    double delta=dr.dist;
    double bp=scalar_barrier_g(delta,d_hat);
    double bpp=scalar_barrier_h(delta,d_hat);
    if (bp==0.0&&bpp==0.0) return;
    if (delta<=0.0) return;

    double r[3]; sub3(dr.cp1,dr.cp2,r);
    double uu[3]; scale3(1.0/delta,r,uu);
    double s=dr.s, t=dr.t;

    // Gradient
    double mu4[4]={0,0,0,0};
    int reg=dr.region;
    if (reg==0) { mu4[0]=bp*(1-s); mu4[1]=bp*s; mu4[2]=-bp*(1-t); mu4[3]=-bp*t; }
    else if(reg==1){mu4[0]=bp;mu4[2]=-bp*(1-t);mu4[3]=-bp*t;}
    else if(reg==2){mu4[1]=bp;mu4[2]=-bp*(1-t);mu4[3]=-bp*t;}
    else if(reg==3){mu4[0]=bp*(1-s);mu4[1]=bp*s;mu4[2]=-bp;}
    else if(reg==4){mu4[0]=bp*(1-s);mu4[1]=bp*s;mu4[3]=-bp;}
    else if(reg==5){mu4[0]=bp;mu4[2]=-bp;}
    else if(reg==6){mu4[0]=bp;mu4[3]=-bp;}
    else if(reg==7){mu4[1]=bp;mu4[2]=-bp;}
    else if(reg==8){mu4[1]=bp;mu4[3]=-bp;}
    else { // parallel
        double muf[4]={1-s,s,-(1-t),-t};
        for(int k=0;k<3;k++) g[k]+=bp*muf[dof]*uu[k];
        return;
    }
    for(int k=0;k<3;k++) g[k]+=mu4[dof]*uu[k];

    // Hessian
    if (reg==5||reg==6||reg==7||reg==8) { // corner
        int a_idx=(reg==5||reg==6)?0:(reg==7)?1:1;
        int b_idx=(reg==5||reg==7)?2:(reg==6)?3:3;
        if(reg==5){a_idx=0;b_idx=2;}else if(reg==6){a_idx=0;b_idx=3;}
        else if(reg==7){a_idx=1;b_idx=2;}else{a_idx=1;b_idx=3;}
        double sp4[4]={0,0,0,0}; sp4[a_idx]=1.0; sp4[b_idx]=-1.0;
        if(sp4[dof]==0.0) return;
        const double* Y4[4]={x1,x2,x3,x4};
        double uu2[3]; sub3(Y4[a_idx],Y4[b_idx],uu2); scale3(1.0/delta,uu2,uu2);
        double c1=bpp,c2=bp/delta,sq=sp4[dof]*sp4[dof];
        for(int k=0;k<3;k++) for(int l=0;l<3;l++)
            H[k*3+l]+=sq*(c1*uu2[k]*uu2[l]+c2*((k==l?1.0:0.0)-uu2[k]*uu2[l]));
        return;
    }

    if (reg==1||reg==2||reg==3||reg==4) { // edge
        int query_idx,ea_idx,eb_idx;
        if(reg==1){query_idx=0;ea_idx=2;eb_idx=3;}
        else if(reg==2){query_idx=1;ea_idx=2;eb_idx=3;}
        else if(reg==3){query_idx=2;ea_idx=0;eb_idx=1;}
        else{query_idx=3;ea_idx=0;eb_idx=1;}
        const double* Y4[4]={x1,x2,x3,x4};
        const double* xq=Y4[query_idx]; const double* xea=Y4[ea_idx]; const double* xeb=Y4[eb_idx];
        double omega4[4]={0,0,0,0},eps4[4]={0,0,0,0};
        omega4[query_idx]=1.0; omega4[ea_idx]=-1.0; eps4[ea_idx]=-1.0; eps4[eb_idx]=1.0;
        double ee[3],ww[3]; sub3(xeb,xea,ee); sub3(xq,xea,ww);
        double alpha=dot3(ww,ee),beta2=dot3(ee,ee);
        double tp=alpha/beta2;
        double rr[3]; for(int i=0;i<3;i++) rr[i]=ww[i]-tp*ee[i];
        double uu2[3]; scale3(1.0/delta,rr,uu2);
        double t_d[4][3]; double r_d[4][3][3]={};
        for(int pp=0;pp<4;pp++) for(int k=0;k<3;k++) {
            double apk=omega4[pp]*ee[k]+eps4[pp]*ww[k];
            double bpk=2.0*eps4[pp]*ee[k];
            t_d[pp][k]=apk/beta2-alpha*bpk/(beta2*beta2);
            for(int i=0;i<3;i++){
                double dik=(i==k)?1.0:0.0,dp_ea=(pp==ea_idx)?1.0:0.0,dp_q=(pp==query_idx)?1.0:0.0;
                double q_d=dp_ea*dik+t_d[pp][k]*ee[i]+tp*eps4[pp]*dik;
                r_d[pp][k][i]=dp_q*dik-q_d;
            }
        }
        int p=dof;
        for(int k=0;k<3;k++) for(int l=0;l<3;l++) {
            double dkl=(k==l)?1.0:0.0;
            double apk=omega4[p]*ee[k]+eps4[p]*ww[k];
            double aql=omega4[p]*ee[l]+eps4[p]*ww[l];
            double apkql=2.0*omega4[p]*eps4[p]*dkl;
            double bpk=2.0*eps4[p]*ee[k],bql=2.0*eps4[p]*ee[l];
            double bpkql=2.0*eps4[p]*eps4[p]*dkl;
            double tpkql=apkql/beta2-(apk*bql+aql*bpk+alpha*bpkql)/(beta2*beta2)+2.0*alpha*bpk*bql/(beta2*beta2*beta2);
            double ddelta_pk=0,ddelta_ql=0;
            for(int i=0;i<3;i++){ddelta_pk+=uu2[i]*r_d[p][k][i];ddelta_ql+=uu2[i]*r_d[p][l][i];}
            double proj=0;
            for(int i=0;i<3;i++) for(int j=0;j<3;j++){double dij=(i==j)?1.0:0.0;proj+=(dij-uu2[i]*uu2[j])*r_d[p][k][i]*r_d[p][l][j];}
            proj/=delta;
            double uq=0;
            for(int i=0;i<3;i++){double dik=(i==k)?1.0:0.0,dil=(i==l)?1.0:0.0;double qipkql=tpkql*ee[i]+t_d[p][k]*eps4[p]*dil+t_d[p][l]*eps4[p]*dik;uq+=uu2[i]*qipkql;}
            H[k*3+l]+=bpp*ddelta_pk*ddelta_ql+bp*(proj-uq);
        }
        return;
    }

    if (reg==0) { // interior
        double sig_a[4]={-1,1,0,0},sig_b[4]={0,0,-1,1},sig_c[4]={1,0,-1,0};
        double aa2[3],bb2[3],cc2[3];
        sub3(x2,x1,aa2); sub3(x4,x3,bb2); sub3(x1,x3,cc2);
        double A=dot3(aa2,aa2),B=dot3(aa2,bb2),C=dot3(bb2,bb2);
        double D=dot3(aa2,cc2),E=dot3(bb2,cc2);
        double Delta2=A*C-B*B;
        double nu=B*E-C*D,zeta_v=A*E-B*D;
        double s_val=nu/Delta2,t_val=zeta_v/Delta2;
        double Ad[4][3],Bd[4][3],Cd[4][3],Dd[4][3],Ed_arr[4][3];
        for(int pp=0;pp<4;pp++) for(int k=0;k<3;k++){
            Ad[pp][k]=2.0*sig_a[pp]*aa2[k];
            Bd[pp][k]=sig_a[pp]*bb2[k]+sig_b[pp]*aa2[k];
            Cd[pp][k]=2.0*sig_b[pp]*bb2[k];
            Dd[pp][k]=sig_a[pp]*cc2[k]+sig_c[pp]*aa2[k];
            Ed_arr[pp][k]=sig_b[pp]*cc2[k]+sig_c[pp]*bb2[k];
        }
        double nu_d[4][3],zeta_d[4][3],Delta_d[4][3];
        for(int pp=0;pp<4;pp++) for(int k=0;k<3;k++){
            nu_d[pp][k]=Bd[pp][k]*E+B*Ed_arr[pp][k]-Cd[pp][k]*D-C*Dd[pp][k];
            zeta_d[pp][k]=Ad[pp][k]*E+A*Ed_arr[pp][k]-Bd[pp][k]*D-B*Dd[pp][k];
            Delta_d[pp][k]=Ad[pp][k]*C+A*Cd[pp][k]-2.0*B*Bd[pp][k];
        }
        double s_d[4][3],t_d[4][3];
        for(int pp=0;pp<4;pp++) for(int k=0;k<3;k++){
            s_d[pp][k]=nu_d[pp][k]/Delta2-nu*Delta_d[pp][k]/(Delta2*Delta2);
            t_d[pp][k]=zeta_d[pp][k]/Delta2-zeta_v*Delta_d[pp][k]/(Delta2*Delta2);
        }
        double r_vec[3]; for(int i=0;i<3;i++) r_vec[i]=(x1[i]+s_val*aa2[i])-(x3[i]+t_val*bb2[i]);
        double uu3[3]; scale3(1.0/delta,r_vec,uu3);
        double p_d[4][3][3],q_darr[4][3][3],r_d[4][3][3];
        for(int pp=0;pp<4;pp++) for(int k=0;k<3;k++) for(int i=0;i<3;i++){
            double dik=(i==k)?1.0:0.0,dp0=(pp==0)?1.0:0.0,dp2=(pp==2)?1.0:0.0;
            p_d[pp][k][i]=dp0*dik+s_d[pp][k]*aa2[i]+s_val*sig_a[pp]*dik;
            q_darr[pp][k][i]=dp2*dik+t_d[pp][k]*bb2[i]+t_val*sig_b[pp]*dik;
            r_d[pp][k][i]=p_d[pp][k][i]-q_darr[pp][k][i];
        }
        int p=dof;
        for(int k=0;k<3;k++) for(int l=0;l<3;l++){
            double dkl=(k==l)?1.0:0.0;
            double A_pkql=2.0*sig_a[p]*sig_a[p]*dkl;
            double B_pkql=(sig_a[p]*sig_b[p]+sig_a[p]*sig_b[p])*dkl;
            double C_pkql=2.0*sig_b[p]*sig_b[p]*dkl;
            double D_pkql=(sig_a[p]*sig_c[p]+sig_a[p]*sig_c[p])*dkl;
            double E_pkql=(sig_b[p]*sig_c[p]+sig_b[p]*sig_c[p])*dkl;
            double nu_pkql=B_pkql*E+Bd[p][k]*Ed_arr[p][l]+Bd[p][l]*Ed_arr[p][k]+B*E_pkql-C_pkql*D-Cd[p][k]*Dd[p][l]-Cd[p][l]*Dd[p][k]-C*D_pkql;
            double Delta_pkql=A_pkql*C+Ad[p][k]*Cd[p][l]+Ad[p][l]*Cd[p][k]+A*C_pkql-2.0*(Bd[p][k]*Bd[p][l]+B*B_pkql);
            double zeta_pkql=A_pkql*E+Ad[p][k]*Ed_arr[p][l]+Ad[p][l]*Ed_arr[p][k]+A*E_pkql-B_pkql*D-Bd[p][k]*Dd[p][l]-Bd[p][l]*Dd[p][k]-B*D_pkql;
            double s_pkql=nu_pkql/Delta2-(nu_d[p][k]*Delta_d[p][l]+nu_d[p][l]*Delta_d[p][k]+nu*Delta_pkql)/(Delta2*Delta2)+2.0*nu*Delta_d[p][k]*Delta_d[p][l]/(Delta2*Delta2*Delta2);
            double t_pkql=zeta_pkql/Delta2-(zeta_d[p][k]*Delta_d[p][l]+zeta_d[p][l]*Delta_d[p][k]+zeta_v*Delta_pkql)/(Delta2*Delta2)+2.0*zeta_v*Delta_d[p][k]*Delta_d[p][l]/(Delta2*Delta2*Delta2);
            double ddelta_pk=0,ddelta_ql=0;
            for(int i=0;i<3;i++){ddelta_pk+=uu3[i]*r_d[p][k][i];ddelta_ql+=uu3[i]*r_d[p][l][i];}
            double proj=0;
            for(int i=0;i<3;i++) for(int j=0;j<3;j++){double dij=(i==j)?1.0:0.0;proj+=(dij-uu3[i]*uu3[j])*r_d[p][k][i]*r_d[p][l][j];}
            proj/=delta;
            double ur_term=0;
            for(int i=0;i<3;i++){double dik=(i==k)?1.0:0.0,dil=(i==l)?1.0:0.0;
                double p_ipkql=s_pkql*aa2[i]+s_d[p][k]*sig_a[p]*dil+s_d[p][l]*sig_a[p]*dik;
                double q_ipkql=t_pkql*bb2[i]+t_d[p][k]*sig_b[p]*dil+t_d[p][l]*sig_b[p]*dik;
                ur_term+=uu3[i]*(p_ipkql-q_ipkql);}
            H[k*3+l]+=bpp*ddelta_pk*ddelta_ql+bp*(proj+ur_term);
        }
    }
}

// ============================================================================
// Kernel parameter structs (POD, passed by value)
// ============================================================================

struct MeshPtrs {
    const int*    tris;
    const double* Dm_inv;
    const double* area;
    const double* mass;
    const int*    hinge_v;
    const double* hinge_bar_theta;
    const double* hinge_ce;
    const int*    hinge_adj_offsets;
    const int*    hinge_adj_hi;
    const int*    hinge_adj_role;
};

struct AdjPtrs {
    const int* offsets;
    const int* tri_idx;
    const int* tri_local;
};

struct BpPtrs {
    const int* vnt_offsets;
    const int* vnt_pair_idx;
    const int* vnt_dof;
    const int* nt_data;
    const int* vss_offsets;
    const int* vss_pair_idx;
    const int* vss_dof;
    const int* ss_data;
};

struct CertPtrs {
    const int*    ntt_offsets;
    const int*    ntt_data;
    const int*    nte_offsets;
    const int*    nte_data;
    const double* tri_box_min;
    const double* tri_box_max;
    const double* edge_box_min;
    const double* edge_box_max;
};

struct PinPtrs {
    const double* targets;
    const int*    pin_map;
};

// BVH node packed for device. Layout matches CPU BVHNode scalars.
struct BVHNodeGPU {
    double bmin[3];
    double bmax[3];
    int    left;
    int    right;
    int    leafIndex;
    int    pad;   // keeps sizeof multiple of 8
};

struct BVHPtrs {
    const BVHNodeGPU* tri_nodes;
    const BVHNodeGPU* node_nodes;
    const BVHNodeGPU* edge_nodes;
    const int*        edges;  // num_edges * 2, [v0, v1]
    int               tri_root;
    int               node_root;
    int               edge_root;
};

// ============================================================================
// CCD device helpers — exact ports of node_triangle_only_one_node_moves and
// segment_segment_only_one_node_moves (ccd.cpp). Same scalar math, same order
// of operations, same acceptance criteria.
// ============================================================================

__device__ static bool ccd_in_unit_interval(double t, double eps) {
    return t >= -eps && t <= 1.0 + eps;
}

__device__ static bool ccd_nearly_zero(double v, double eps) {
    return fabs(v) <= eps;
}

// barycentric point-in-triangle, matches point_in_triangle_on_plane / triangle_plane_barycentric_coordinates
__device__ static bool ccd_point_in_triangle(
    const double pt[3],
    const double x1[3], const double x2[3], const double x3[3],
    double eps)
{
    double e1[3], e2[3], r[3];
    sub3(x2, x1, e1);
    sub3(x3, x1, e2);
    sub3(pt, x1, r);
    const double a11 = dot3(e1, e1);
    const double a12 = dot3(e1, e2);
    const double a22 = dot3(e2, e2);
    const double b1  = dot3(r,  e1);
    const double b2  = dot3(r,  e2);
    const double det = a11 * a22 - a12 * a12;
    if (fabs(det) <= eps) return false;
    const double alpha = ( b1 * a22 - b2 * a12) / det;
    const double beta  = (-b1 * a12 + b2 * a11) / det;
    const double lam1  = 1.0 - alpha - beta;
    return (lam1 >= -eps) && (alpha >= -eps) && (beta >= -eps);
}

// segment_segment_parameters_if_not_parallel (IPC_math.cpp)
__device__ static bool ccd_seg_seg_params(
    const double x1[3], const double x2[3],
    const double x3[3], const double x4[3],
    double& s_out, double& u_out, double eps)
{
    double a[3], b[3], c[3];
    sub3(x2, x1, a);
    sub3(x4, x3, b);
    sub3(x3, x1, c);
    double n[3]; cross3(a, b, n);
    const double denom = dot3(n, n);
    if (fabs(denom) <= eps) return false;
    double cb[3], ca[3];
    cross3(c, b, cb);
    cross3(c, a, ca);
    s_out = dot3(cb, n) / denom;
    u_out = dot3(ca, n) / denom;
    return true;
}

// General CCD for vertex-triangle: any/all nodes may move.
// Returns true (and writes out_t in [0,1]) iff a collision is detected.
// Direct port of node_triangle_only_one_node_moves (ccd.cpp).
__device__ __noinline__ static bool ccd_nt_general(
    const double x [3], const double dx [3],
    const double x1[3], const double dx1[3],
    const double x2[3], const double dx2[3],
    const double x3[3], const double dx3[3],
    double eps, double& out_t)
{
    double p0[3], dp[3];  sub3(x2, x1, p0); sub3(dx2, dx1, dp);
    double q0[3], dq[3];  sub3(x3, x1, q0); sub3(dx3, dx1, dq);
    double r0[3], dr[3];  sub3(x,  x1, r0); sub3(dx,  dx1, dr);

    double p0xq0[3]; cross3(p0, q0, p0xq0);
    const double d = dot3(p0xq0, r0);

    // c = (dp x q0).r0 + (p0 x dq).r0 + (p0 x q0).dr
    double tmp[3];
    double c = 0.0;
    cross3(dp, q0, tmp); c += dot3(tmp, r0);
    cross3(p0, dq, tmp); c += dot3(tmp, r0);
    c += dot3(p0xq0, dr);

    if (ccd_nearly_zero(c, eps)) {
        // Matches CPU: coplanar_entire_step / parallel_or_no_crossing — neither yields a candidate t
        return false;
    }
    double t = -d / c;
    if (!ccd_in_unit_interval(t, eps)) return false;
    t = dev_clamp(t, 0.0, 1.0);

    double xt[3], x1t[3], x2t[3], x3t[3];
    for (int k = 0; k < 3; ++k) {
        xt [k] = x [k] + t * dx [k];
        x1t[k] = x1[k] + t * dx1[k];
        x2t[k] = x2[k] + t * dx2[k];
        x3t[k] = x3[k] + t * dx3[k];
    }
    if (!ccd_point_in_triangle(xt, x1t, x2t, x3t, eps)) return false;
    out_t = t;
    return true;
}

// CCD for segment-segment where only x1 moves. Matches
// segment_segment_only_one_node_moves (ccd.cpp).
__device__ __noinline__ static bool ccd_ss_single(
    const double x1[3], const double dx1[3],
    const double x2[3], const double x3[3], const double x4[3],
    double eps, double& out_t)
{
    double a[3], b[3], c0[3];
    sub3(x2, x1, a);
    sub3(x4, x3, b);
    sub3(x3, x1, c0);

    double axb[3]; cross3(a, b, axb);
    const double d = dot3(axb, c0);

    // c = -(a x b).dx1 - (dx1 x b).c0
    double tmp[3];
    double c = -dot3(axb, dx1);
    cross3(dx1, b, tmp);
    c -= dot3(tmp, c0);

    if (ccd_nearly_zero(c, eps)) return false;
    double t = -d / c;
    if (!ccd_in_unit_interval(t, eps)) return false;
    t = dev_clamp(t, 0.0, 1.0);

    double x1_star[3];
    for (int k = 0; k < 3; ++k) x1_star[k] = x1[k] + t * dx1[k];

    double s, u;
    if (!ccd_seg_seg_params(x1_star, x2, x3, x4, s, u, eps)) return false;
    if (!ccd_in_unit_interval(s, eps)) return false;
    if (!ccd_in_unit_interval(u, eps)) return false;
    out_t = t;
    return true;
}

// Trust-region scalar step for vertex-triangle Gauss-Seidel.
// Matches trust_region_vertex_triangle_gauss_seidel (trust_region.cpp) with
// default eta = 0.4. Also returns d0 so the caller can gate on d0 < d_hat.
__device__ __noinline__ static double tr_nt_gs_dev(
    const double x [3], const double x1[3], const double x2[3], const double x3[3],
    const double delta[3], double& d0_out)
{
    NTDist nt;
    node_triangle_distance_dev(x, x1, x2, x3, 1.0e-12, nt);
    const double d0 = nt.dist;
    const double M  = norm3(delta);
    d0_out = d0;

    // clamp_eta(0.4) — lies within [1e-12, 0.5-1e-12] already.
    const double eta = 0.4;
    if (d0 <= 0.0) return 0.0;
    if (M  <= 1.0e-30) return 1.0;
    double omega = eta * d0 / M;
    if (omega > 1.0) omega = 1.0;
    return omega;
}

__device__ __noinline__ static double tr_ee_gs_dev(
    const double a1[3], const double a2[3], const double b1[3], const double b2[3],
    const double delta[3], double& d0_out)
{
    SSDist ss;
    segment_segment_distance_dev(a1, a2, b1, b2, 1.0e-12, ss);
    const double d0 = ss.dist;
    const double M  = norm3(delta);
    d0_out = d0;

    const double eta = 0.4;
    if (d0 <= 0.0) return 0.0;
    if (M  <= 1.0e-30) return 1.0;
    double omega = eta * d0 / M;
    if (omega > 1.0) omega = 1.0;
    return omega;
}

// AABB intersection check.
__device__ static bool aabb_intersects_dev(
    const double amin[3], const double amax[3],
    const double bmin[3], const double bmax[3])
{
    for (int k = 0; k < 3; ++k) {
        if (amax[k] < bmin[k]) return false;
        if (amin[k] > bmax[k]) return false;
    }
    return true;
}

// Stack-based BVH traversal. Matches query_bvh (broad_phase.cpp): visits node,
// if bbox intersects query and it's a leaf, records its leafIndex; else push
// both children. Stack size 256 mirrors the CPU path.
__device__ static int query_bvh_device(
    const BVHNodeGPU* nodes, int root,
    const double qmin[3], const double qmax[3],
    int* hits, int hit_cap)
{
    int hit_count = 0;
    if (root < 0) return 0;
    int stack[64];
    int top = 0;
    stack[top++] = root;
    while (top > 0) {
        const int idx = stack[--top];
        const BVHNodeGPU& n = nodes[idx];
        if (!aabb_intersects_dev(n.bmin, n.bmax, qmin, qmax)) continue;
        if (n.leafIndex >= 0) {
            if (hit_count < hit_cap) hits[hit_count++] = n.leafIndex;
        } else {
            if (top + 2 <= 64) {
                stack[top++] = n.left;
                stack[top++] = n.right;
            }
        }
    }
    return hit_count;
}

// Orchestrates the per-vertex safe-step computation.
// Direct structural port of compute_safe_step_for_vertex (parallel_helper.cpp).
// Uses BVH queries against live positions (same as CPU) and runs CCD or trust
// region according to params.use_trust_region.
// Barrier-pair-based safe step: walks the session's vnt/vss CSR lists to find
// candidate NT/SS pairs involving vi, runs CCD or trust-region for each.
//
// Why this is correct (and MUCH faster than a BVH query): the conflict-graph
// coloring + certified-region clip guarantee that vi's motion this color group
// stays within d_hat of its starting position. Any pair that could actually
// collide during this step must therefore be in the precomputed barrier pair
// list (vertices within d_hat pad). BVH queries during commit (as the CPU's
// query_pairs_for_vertex does) return the same pairs plus some extras that
// can't possibly collide — so skipping BVH and iterating the CSR is both
// correct and dramatically cheaper (no random memory access, no divergent
// traversal, no wasted CCD tests on impossible pairs).
__device__ static double compute_safe_step_for_vertex_device(
    int vi,
    const double* x,
    const double  delta[3],
    GPUSimParams  params,
    MeshPtrs      mesh,
    BpPtrs        bp)
{
    if (params.d_hat <= 0.0) return 1.0;
    const double eps = 1.0e-12;
    const double dx[3] = { -delta[0], -delta[1], -delta[2] };
    const double zero3[3] = {0.0, 0.0, 0.0};

    const bool tr = params.use_trust_region;
    double safe_min = 1.0;

    // ---- Node-triangle barrier pairs involving vi ----
    for (int i = bp.vnt_offsets[vi]; i < bp.vnt_offsets[vi+1]; ++i) {
        const int pair_idx = bp.vnt_pair_idx[i];
        const int dof      = bp.vnt_dof[i];   // 0=node, 1/2/3=tri corner a/b/c
        const int node = bp.nt_data[pair_idx*4 + 0];
        const int ta   = bp.nt_data[pair_idx*4 + 1];
        const int tb   = bp.nt_data[pair_idx*4 + 2];
        const int tc   = bp.nt_data[pair_idx*4 + 3];

        double xn[3], xa[3], xb[3], xc[3];
        loadv(x, node, xn); loadv(x, ta, xa); loadv(x, tb, xb); loadv(x, tc, xc);

        if (tr) {
            double d0;
            const double omega = tr_nt_gs_dev(xn, xa, xb, xc, delta, d0);
            if (d0 < params.d_hat && omega < safe_min) safe_min = omega;
        } else {
            // Move whichever DOF is vi. Others get zero displacement.
            double dxn[3] = {0,0,0}, dxa[3] = {0,0,0}, dxb[3] = {0,0,0}, dxc[3] = {0,0,0};
            if      (dof == 0) { dxn[0]=dx[0]; dxn[1]=dx[1]; dxn[2]=dx[2]; }
            else if (dof == 1) { dxa[0]=dx[0]; dxa[1]=dx[1]; dxa[2]=dx[2]; }
            else if (dof == 2) { dxb[0]=dx[0]; dxb[1]=dx[1]; dxb[2]=dx[2]; }
            else if (dof == 3) { dxc[0]=dx[0]; dxc[1]=dx[1]; dxc[2]=dx[2]; }
            double toi;
            if (ccd_nt_general(xn, dxn, xa, dxa, xb, dxb, xc, dxc, eps, toi)) {
                if (toi < safe_min) safe_min = toi;
            }
        }
        (void)zero3;  // suppress unused warning on tr path
    }

    // ---- Segment-segment barrier pairs involving vi ----
    for (int i = bp.vss_offsets[vi]; i < bp.vss_offsets[vi+1]; ++i) {
        const int pair_idx = bp.vss_pair_idx[i];
        const int dof      = bp.vss_dof[i];   // 0/1 = in edge1, 2/3 = in edge2
        const int v0 = bp.ss_data[pair_idx*4 + 0];
        const int v1 = bp.ss_data[pair_idx*4 + 1];
        const int v2 = bp.ss_data[pair_idx*4 + 2];
        const int v3 = bp.ss_data[pair_idx*4 + 3];

        double x0[3], x1[3], x2[3], x3[3];
        loadv(x, v0, x0); loadv(x, v1, x1); loadv(x, v2, x2); loadv(x, v3, x3);

        if (tr) {
            double d0;
            const double omega = tr_ee_gs_dev(x0, x1, x2, x3, delta, d0);
            if (d0 < params.d_hat && omega < safe_min) safe_min = omega;
        } else {
            // ccd_ss_single signature: (x1_moving, dx1, x2, x3, x4) — first
            // point moves, other 3 static. Swap arguments so vi's point is x1.
            double toi;
            bool collision = false;
            if      (dof == 0) collision = ccd_ss_single(x0, dx, x1, x2, x3, eps, toi);
            else if (dof == 1) collision = ccd_ss_single(x1, dx, x0, x2, x3, eps, toi);
            else if (dof == 2) collision = ccd_ss_single(x2, dx, x3, x0, x1, eps, toi);
            else if (dof == 3) collision = ccd_ss_single(x3, dx, x2, x0, x1, eps, toi);
            if (collision && toi < safe_min) safe_min = toi;
        }
    }

    // CPU: return tr ? safe_min : ((safe_min >= 1.0) ? 1.0 : 0.9 * safe_min);
    if (tr) return safe_min;
    return (safe_min >= 1.0) ? 1.0 : 0.9 * safe_min;
}

// ============================================================================
// compute_local_newton_direction_device
// ============================================================================

__device__ static void compute_local_newton_direction_device(
    int vi,
    MeshPtrs mesh, AdjPtrs adj, BpPtrs bp, PinPtrs pins,
    GPUSimParams params,
    const double* x, const double* xhat,
    double g[3], double H[9], double delta_out[3])
{
    double dt2 = params.dt2_val;
    for(int i=0;i<3;i++) g[i]=0;
    for(int i=0;i<9;i++) H[i]=0;

    // Inertia
    double xi[3], xhi[3];
    loadv(x,vi,xi); loadv(xhat,vi,xhi);
    double m = mesh.mass[vi];
    double grav[3] = {params.gx, params.gy, params.gz};
    for(int k=0;k<3;k++) g[k] += m*(xi[k]-xhi[k]) + dt2*(-m*grav[k]);
    adddiag3(m, H);

    // Pin spring
    int pi = pins.pin_map[vi];
    if (pi >= 0) {
        double tx=pins.targets[pi*3], ty=pins.targets[pi*3+1], tz=pins.targets[pi*3+2];
        g[0] += dt2*params.kpin*(xi[0]-tx);
        g[1] += dt2*params.kpin*(xi[1]-ty);
        g[2] += dt2*params.kpin*(xi[2]-tz);
        adddiag3(dt2*params.kpin, H);
    }

    // Corotated elastic (CSR adjacency)
    for(int idx=adj.offsets[vi]; idx<adj.offsets[vi+1]; ++idx) {
        int ti = adj.tri_idx[idx];
        int a  = adj.tri_local[idx];
        int v0=mesh.tris[ti*3+0], v1=mesh.tris[ti*3+1], v2=mesh.tris[ti*3+2];
        double xv0[3], xv1[3], xv2[3];
        loadv(x,v0,xv0); loadv(x,v1,xv1); loadv(x,v2,xv2);
        const double* dm = mesh.Dm_inv + ti*4;
        corotated_vertex_gh(a, xv0,xv1,xv2, dm, mesh.area[ti],
                            params.mu, params.lambda, dt2, g, H);
    }

    // Bending (CSR hinge adjacency)
    if (params.kB > 0.0) {
        for(int idx=mesh.hinge_adj_offsets[vi]; idx<mesh.hinge_adj_offsets[vi+1]; ++idx) {
            int hi   = mesh.hinge_adj_hi[idx];
            int role = mesh.hinge_adj_role[idx];
            double hv0[3],hv1[3],hv2[3],hv3[3];
            loadv(x, mesh.hinge_v[hi*4+0], hv0);
            loadv(x, mesh.hinge_v[hi*4+1], hv1);
            loadv(x, mesh.hinge_v[hi*4+2], hv2);
            loadv(x, mesh.hinge_v[hi*4+3], hv3);
            bending_vertex_gh_psd(role, hv0,hv1,hv2,hv3,
                                  params.kB, mesh.hinge_ce[hi],
                                  mesh.hinge_bar_theta[hi], dt2, g, H);
        }
    }

    // NT barrier
    if (params.d_hat > 0.0) {
        for(int idx=bp.vnt_offsets[vi]; idx<bp.vnt_offsets[vi+1]; ++idx) {
            int pair_idx = bp.vnt_pair_idx[idx];
            int dof      = bp.vnt_dof[idx];
            int node=bp.nt_data[pair_idx*4+0];
            int tv0=bp.nt_data[pair_idx*4+1];
            int tv1=bp.nt_data[pair_idx*4+2];
            int tv2=bp.nt_data[pair_idx*4+3];
            double xn[3],xt0[3],xt1[3],xt2[3];
            loadv(x,node,xn); loadv(x,tv0,xt0); loadv(x,tv1,xt1); loadv(x,tv2,xt2);
            double bg[3]={0,0,0}, bH[9]={};
            nt_barrier_gh(xn,xt0,xt1,xt2, params.d_hat, dof, bg, bH);
            if (vi==1 && (bH[4] > 1e6 || bH[4] < -1e6))
                printf("GPU NT vi=1 pair=%d dof=%d node=%d tv=[%d,%d,%d] H[1][1]=%.3e\n",
                       pair_idx,dof,node,tv0,tv1,tv2,bH[4]);
            addscale3(dt2, bg, g);
            for(int i=0;i<9;i++) H[i] += dt2*bH[i];
        }

        // SS barrier
        for(int idx=bp.vss_offsets[vi]; idx<bp.vss_offsets[vi+1]; ++idx) {
            int pair_idx = bp.vss_pair_idx[idx];
            int dof      = bp.vss_dof[idx];
            int vv0=bp.ss_data[pair_idx*4+0];
            int vv1=bp.ss_data[pair_idx*4+1];
            int vv2=bp.ss_data[pair_idx*4+2];
            int vv3=bp.ss_data[pair_idx*4+3];
            double xs0[3],xs1[3],xs2[3],xs3[3];
            loadv(x,vv0,xs0); loadv(x,vv1,xs1); loadv(x,vv2,xs2); loadv(x,vv3,xs3);
            double bg[3]={0,0,0}, bH[9]={};
            ss_barrier_gh(xs0,xs1,xs2,xs3, params.d_hat, dof, bg, bH);
            if (vi==1 && (bH[4] > 1e6 || bH[4] < -1e6)) {
                printf("GPU SS vi=1 pair=%d dof=%d v=[%d,%d,%d,%d] H[1][1]=%.3e\n",
                       pair_idx,dof,vv0,vv1,vv2,vv3,bH[4]);
                // dump SSDist details
                SSDist dr2;
                segment_segment_distance_dev(xs0,xs1,xs2,xs3,1e-12,dr2);
                double aa2[3],bb2[3]; sub3(xs1,xs0,aa2); sub3(xs3,xs2,bb2);
                double A2=dot3(aa2,aa2),B2=dot3(aa2,bb2),C2=dot3(bb2,bb2);
                double D2=0,E2=0; double cc2[3]; sub3(xs0,xs2,cc2);
                D2=dot3(aa2,cc2); E2=dot3(bb2,cc2);
                double Del2=A2*C2-B2*B2;
                double s_u2=(B2*E2-C2*D2)/Del2, t_u2=(A2*E2-B2*D2)/Del2;
                printf("  Del=%.3e s_u=%.3e t_u=%.3e s=%.3e t=%.3e region=%d dist=%.6e\n",
                       Del2,s_u2,t_u2,dr2.s,dr2.t,dr2.region,dr2.dist);
            }
            addscale3(dt2, bg, g);
            for(int i=0;i<9;i++) H[i] += dt2*bH[i];
        }
    }

    // delta = H^{-1} * g
    double Hinv[9];
    mat33_inverse(H, Hinv);
    matvec3(Hinv, g, delta_out);
}

// ============================================================================
// build_certified_region_device
// Fills box_min[3] and box_max[3] with the certified region AABB for vertex vi.
// ============================================================================

__device__ static void build_certified_region_device(
    int vi, const double* x, const double delta[3],
    CertPtrs cert,
    double d_hat,
    double box_min[3], double box_max[3])
{
    double xi[3]; loadv(x, vi, xi);
    double half_dhat = 0.5 * d_hat;
    double step_extent = norm3(delta);

    for(int k=0;k<3;k++) {
        box_min[k] = xi[k] - step_extent - half_dhat;
        box_max[k] = xi[k] + step_extent + half_dhat;
    }

    // Union with incident triangle boxes (padded by d_hat/2)
    for(int i=cert.ntt_offsets[vi]; i<cert.ntt_offsets[vi+1]; ++i) {
        int tri_idx = cert.ntt_data[i];
        for(int k=0;k<3;k++) {
            box_min[k] = dev_min2(box_min[k], cert.tri_box_min[tri_idx*3+k] - half_dhat);
            box_max[k] = dev_max2(box_max[k], cert.tri_box_max[tri_idx*3+k] + half_dhat);
        }
    }

    // Union with incident edge boxes (already pre-padded)
    for(int i=cert.nte_offsets[vi]; i<cert.nte_offsets[vi+1]; ++i) {
        int edge_idx = cert.nte_data[i];
        for(int k=0;k<3;k++) {
            box_min[k] = dev_min2(box_min[k], cert.edge_box_min[edge_idx*3+k]);
            box_max[k] = dev_max2(box_max[k], cert.edge_box_max[edge_idx*3+k]);
        }
    }
}

// ============================================================================
// atomicMaxDouble — needed by residual reduction kernel. CAS loop mirroring
// the atomicMinDouble in gpu_ccd.cu.
// ============================================================================
__device__ static double atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long  old_ull  = *addr_ull;
    unsigned long long  assumed;
    do {
        assumed = old_ull;
        const double old_d = __longlong_as_double(assumed);
        if (old_d >= val) break;
        old_ull = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old_ull);
    return __longlong_as_double(old_ull);
}

// ============================================================================
// residual_kernel — device port of compute_global_residual (physics.cpp:179).
// One thread per vertex. Each thread computes its local gradient (same
// formula as the Newton direction), optionally mass-normalizes it, and
// atomically reduces max |g| into d_rmax.
// ============================================================================
__global__ static void residual_kernel(
    int           nv,
    MeshPtrs      mesh,
    AdjPtrs       adj,
    BpPtrs        bp,
    PinPtrs       pins,
    GPUSimParams  params,
    const double* d_x,
    const double* d_xhat,
    int           normalize,
    double*       d_rmax)
{
    const int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;

    // Reuse Newton-direction device function: the g it returns is identical
    // to compute_local_gradient's output. The H / delta work is wasted but
    // the marginal compute is cheap vs the global-memory latency.
    double g[3], H[9], delta[3];
    compute_local_newton_direction_device(
        vi, mesh, adj, bp, pins, params, d_x, d_xhat, g, H, delta);

    if (normalize) {
        const double m = mesh.mass[vi];
        if (m > 0.0) { g[0] /= m; g[1] /= m; g[2] /= m; }
    }

    const double a0 = fabs(g[0]);
    const double a1 = fabs(g[1]);
    const double a2 = fabs(g[2]);
    double gmax = a0;
    if (a1 > gmax) gmax = a1;
    if (a2 > gmax) gmax = a2;

    atomicMaxDouble(d_rmax, gmax);
}

// Max per-vertex neighbor capacity for the GPU conflict graph. Sized for
// dense contact scenes (close-contacting cloth, stacked layers) where a
// single vertex can see thousands of barrier pairs + swept-region overlaps.
// Overflow sets d_overflow flag.
static constexpr int MAX_CONFLICT_DEG = 512;

// ============================================================================
// ============================================================================
// LBVH construction on device (Karras 2012).
//
// Pipeline, given N box AABBs:
//   1. lbvh_morton_kernel     — compute 30-bit Morton codes from box centers.
//   2. thrust::sort_by_key     — sort leaf indices by Morton key.
//   3. lbvh_build_kernel       — Karras internal-node topology from sorted keys.
//   4. lbvh_refit_kernel       — bottom-up AABB union (leaves up through root).
//
// Output: BVHNodeGPU array sized 2N-1 (leaves then internals). Root is at
// index N (first internal node).
// ============================================================================

// Interleave low 10 bits of v with zeros: ... z0 z1 ... z9 → b0 0 0 b1 0 0 ...
__host__ __device__ static inline uint32_t lbvh_spread3(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

__device__ static inline uint32_t lbvh_morton3(uint32_t x, uint32_t y, uint32_t z) {
    return (lbvh_spread3(x) << 2) | (lbvh_spread3(y) << 1) | lbvh_spread3(z);
}

// Morton code per box, using scene-wide bounds passed by value.
__global__ static void lbvh_morton_kernel(
    int           N,
    const double* box_min,    // N*3
    const double* box_max,    // N*3
    double sxmin, double symin, double szmin,
    double sxmax, double symax, double szmax,
    uint32_t*     mortons,    // N
    int*          indices)    // N  (init to 0..N-1 here)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double cx = 0.5 * (box_min[i*3+0] + box_max[i*3+0]);
    const double cy = 0.5 * (box_min[i*3+1] + box_max[i*3+1]);
    const double cz = 0.5 * (box_min[i*3+2] + box_max[i*3+2]);

    const double sx = (sxmax - sxmin) > 1e-30 ? (sxmax - sxmin) : 1.0;
    const double sy = (symax - symin) > 1e-30 ? (symax - symin) : 1.0;
    const double sz = (szmax - szmin) > 1e-30 ? (szmax - szmin) : 1.0;

    double nx = (cx - sxmin) / sx * 1023.999;
    double ny = (cy - symin) / sy * 1023.999;
    double nz = (cz - szmin) / sz * 1023.999;
    if (nx < 0) nx = 0; else if (nx > 1023) nx = 1023;
    if (ny < 0) ny = 0; else if (ny > 1023) ny = 1023;
    if (nz < 0) nz = 0; else if (nz > 1023) nz = 1023;

    mortons[i] = lbvh_morton3((uint32_t)nx, (uint32_t)ny, (uint32_t)nz);
    indices[i] = i;
}

// Common-prefix length between Morton[i] and Morton[j], with index tiebreaking
// (needed when adjacent keys are equal).  Returns -1 if j out of range.
__device__ static inline int lbvh_delta(
    int i, int j, int N, const uint32_t* sorted_mortons)
{
    if (j < 0 || j >= N) return -1;
    const uint32_t a = sorted_mortons[i];
    const uint32_t b = sorted_mortons[j];
    if (a != b) return __clz(a ^ b);
    // Tiebreak on index.  32 bits of Morton + some prefix of index.
    return 32 + __clz((uint32_t)i ^ (uint32_t)j);
}

// One thread per internal node (there are N-1 internal nodes for N leaves).
// Leaf indices [0, N); internal node indices [N, 2N-1) in the output array.
// Root is internal node 0, stored at output index N.
__global__ static void lbvh_build_kernel(
    int               N,
    const uint32_t*   sorted_mortons,   // N
    BVHNodeGPU*       nodes,            // 2N-1
    int*              parent)           // 2N-1  (filled here for refit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N - 1) return;

    // Direction d: +1 extends right, -1 extends left.
    const int dL = lbvh_delta(i, i-1, N, sorted_mortons);
    const int dR = lbvh_delta(i, i+1, N, sorted_mortons);
    const int d  = (dR > dL) ? 1 : -1;
    const int dmin = (d == 1) ? dL : dR;  // prefix just outside the range

    // Find upper bound for range length by exponentially expanding.
    int lmax = 2;
    while (lbvh_delta(i, i + lmax*d, N, sorted_mortons) > dmin) lmax *= 2;

    // Binary search for exact length.
    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2) {
        if (lbvh_delta(i, i + (l + t)*d, N, sorted_mortons) > dmin) l += t;
    }
    const int j = i + l*d;

    // Find split position inside [min(i,j), max(i,j)]: largest s such that
    // delta(i, i + s*d) > delta(i, j).
    const int dnode = lbvh_delta(i, j, N, sorted_mortons);
    int s = 0;
    for (int t = (l + 1) / 2; t >= 1; t = (t == 1) ? 0 : (t + 1) / 2) {
        if (lbvh_delta(i, i + (s + t)*d, N, sorted_mortons) > dnode) s += t;
        if (t == 1) break;
    }
    // Correct off-by-one: handle remaining step.
    int split = i + s*d + (d < 0 ? -1 : 0);  // lower of the two children's range

    // Children: if child's range has length 1, it's a leaf at split (or split+1).
    const int left_lo  = min(i, j);
    const int right_hi = max(i, j);

    int left_child, right_child;
    if (min(i, j) == split) left_child  = split;        // leaf
    else                    left_child  = N + split;    // internal
    if (max(i, j) == split + 1) right_child = split + 1;    // leaf
    else                        right_child = N + split + 1;// internal

    const int me = N + i;  // this internal node's output index
    BVHNodeGPU& n = nodes[me];
    n.left      = left_child;
    n.right     = right_child;
    n.leafIndex = -1;
    n.pad       = 0;
    // bbox filled by refit kernel

    parent[left_child]  = me;
    parent[right_child] = me;
    // avoid unused-warning
    (void)left_lo; (void)right_hi;
}

// Initialize leaf nodes from sorted box indices.
__global__ static void lbvh_init_leaves_kernel(
    int N,
    const int*    sorted_indices,   // N  — leaf i corresponds to original box sorted_indices[i]
    const double* box_min,          // (original) N*3
    const double* box_max,          // (original) N*3
    BVHNodeGPU*   nodes,            // 2N-1
    int*          node_ready)       // 2N-1 atomic counter for refit (init to 0)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int orig = sorted_indices[i];
    BVHNodeGPU& n = nodes[i];
    n.bmin[0] = box_min[orig*3+0]; n.bmin[1] = box_min[orig*3+1]; n.bmin[2] = box_min[orig*3+2];
    n.bmax[0] = box_max[orig*3+0]; n.bmax[1] = box_max[orig*3+1]; n.bmax[2] = box_max[orig*3+2];
    n.left = -1;
    n.right = -1;
    n.leafIndex = orig;  // leaf points to ORIGINAL box index (so query_bvh_device's caller gets the un-permuted index)
    n.pad = 0;

    node_ready[i] = 1;  // leaves are ready
}

// Bottom-up refit: each leaf's parent is incremented; when a parent sees both
// children ready, it computes its AABB and bubbles up to its own parent.
__global__ static void lbvh_refit_kernel(
    int N,
    BVHNodeGPU* nodes,            // 2N-1
    const int*  parent,           // 2N-1
    int*        node_ready)       // 2N-1 — incremented atomically per child arrival
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Start from leaf i, walk up propagating until we either hit an internal node
    // whose other child isn't ready yet (bail) or we reach the root.
    int cur = parent[i];
    while (cur >= N) {   // internal node index
        const int arrivals = atomicAdd(&node_ready[cur], 1) + 1;
        if (arrivals < 2) return;   // first child to arrive — let the second handle it
        // Both children ready: union their AABBs.
        BVHNodeGPU& n = nodes[cur];
        const BVHNodeGPU& L = nodes[n.left];
        const BVHNodeGPU& R = nodes[n.right];
        n.bmin[0] = fmin(L.bmin[0], R.bmin[0]);
        n.bmin[1] = fmin(L.bmin[1], R.bmin[1]);
        n.bmin[2] = fmin(L.bmin[2], R.bmin[2]);
        n.bmax[0] = fmax(L.bmax[0], R.bmax[0]);
        n.bmax[1] = fmax(L.bmax[1], R.bmax[1]);
        n.bmax[2] = fmax(L.bmax[2], R.bmax[2]);
        if (cur == N) return;        // at root
        cur = parent[cur];
    }
}

// Gather-active kernel: compacts full-nv cert_bmin/bmax into n_active-sized
// active_bmin/bmax via the active_ids[] mapping. One thread per active vertex.
__global__ static void lbvh_gather_active_boxes_kernel(
    int n_active,
    const int*    active_ids,     // n_active
    const double* full_bmin,      // nv*3
    const double* full_bmax,      // nv*3
    double*       active_bmin,    // n_active*3
    double*       active_bmax)    // n_active*3
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_active) return;
    const int vi = active_ids[i];
    active_bmin[i*3+0] = full_bmin[vi*3+0];
    active_bmin[i*3+1] = full_bmin[vi*3+1];
    active_bmin[i*3+2] = full_bmin[vi*3+2];
    active_bmax[i*3+0] = full_bmax[vi*3+0];
    active_bmax[i*3+1] = full_bmax[vi*3+1];
    active_bmax[i*3+2] = full_bmax[vi*3+2];
}

// Host wrapper: builds an LBVH over `nbox` AABBs already on device.
// Returns root index in `nodes` (always N for non-empty input).
// Session must hold pre-allocated scratch: d_cert_bvh (>= 2*nbox-1 nodes),
// plus temp buffers for mortons, indices, parents, ready counter.
struct LBVHScratch {
    uint32_t* d_mortons   = nullptr;
    int*      d_indices   = nullptr;
    int*      d_parent    = nullptr;
    int*      d_ready     = nullptr;
    int       capacity    = 0;  // N, not 2N-1
};

static void lbvh_scratch_ensure(LBVHScratch& s, int N) {
    if (N <= s.capacity) return;
    auto F = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F(s.d_mortons); F(s.d_indices); F(s.d_parent); F(s.d_ready);
    s.capacity = N;
    cudaMalloc(&s.d_mortons, N * sizeof(uint32_t));
    cudaMalloc(&s.d_indices, N * sizeof(int));
    cudaMalloc(&s.d_parent,  (2*N - 1) * sizeof(int));
    cudaMalloc(&s.d_ready,   (2*N - 1) * sizeof(int));
}

static void lbvh_scratch_free(LBVHScratch& s) {
    auto F = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F(s.d_mortons); F(s.d_indices); F(s.d_parent); F(s.d_ready);
    s.capacity = 0;
}

// Build LBVH in-place into `nodes` (caller-allocated, >= 2N-1 BVHNodeGPU).
// Returns the root index; returns -1 for N <= 0.
static int lbvh_build(
    int               N,
    const double*     d_box_min,
    const double*     d_box_max,
    double            sxmin, double symin, double szmin,
    double            sxmax, double symax, double szmax,
    BVHNodeGPU*       d_nodes,       // 2N-1
    LBVHScratch&      s)
{
    if (N <= 0) return -1;
    lbvh_scratch_ensure(s, N);

    const int block = 256;
    const int grid  = (N + block - 1) / block;

    // 1. Morton codes.
    lbvh_morton_kernel<<<grid, block>>>(
        N, d_box_min, d_box_max,
        sxmin, symin, szmin, sxmax, symax, szmax,
        s.d_mortons, s.d_indices);

    // 2. Sort (Morton, index) pairs by Morton.
    thrust::sort_by_key(
        thrust::device_ptr<uint32_t>(s.d_mortons),
        thrust::device_ptr<uint32_t>(s.d_mortons + N),
        thrust::device_ptr<int>(s.d_indices));

    // 3. Fill leaves from sorted permutation (clear ready counter).
    cudaMemset(s.d_ready, 0, (2*N - 1) * sizeof(int));
    lbvh_init_leaves_kernel<<<grid, block>>>(
        N, s.d_indices, d_box_min, d_box_max, d_nodes, s.d_ready);

    if (N == 1) {
        // Single leaf — treat node 0 as the root.
        return 0;
    }

    // 4. Build internal-node topology.
    const int ig = (N - 1 + block - 1) / block;
    lbvh_build_kernel<<<ig, block>>>(N, s.d_mortons, d_nodes, s.d_parent);

    // 5. Bottom-up refit (AABB union).
    lbvh_refit_kernel<<<grid, block>>>(N, d_nodes, s.d_parent, s.d_ready);

    return N;   // root = first internal node
}

// build_conflict_kernel — GPU port of build_conflict_graph (parallel_helper.cpp).
// One thread per vertex collects all neighbors from four edge sources:
//   1. Elastic coupling (shared triangles) — via session's gpu_adj CSR.
//   2. NT barrier pairs — via session's gpu_bp.vnt_* CSR.
//   3. SS barrier pairs — via session's gpu_bp.vss_* CSR.
//   4. Swept-region overlap — BVH query over per-iter certified AABBs.
// Then sorts + dedups the neighbor list in-thread. Output layout: flat
// [nv, MAX_CONFLICT_DEG] int array with a per-vertex count array.
//
// The CPU version writes bidirectional edges with per-thread locals + merge/
// dedup. The GPU version has each vertex collect only its own outgoing edges
// (avoids race conditions). Since every edge source is symmetric, the result
// is identical after dedup.
// ============================================================================
__global__ static void build_conflict_kernel(
    int                      nv,
    AdjPtrs                  adj,
    const int*               tris,
    BpPtrs                   bp,
    const BVHNodeGPU*        cert_bvh,
    int                      cert_root,
    const int*               active_ids,
    const double*            cert_bmin,
    const double*            cert_bmax,
    const int*               active_flag,
    int*                     out_nbrs,        // nv * MAX_CONFLICT_DEG
    int*                     out_counts,      // nv
    int*                     d_overflow)
{
    const int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;

    if (!active_flag[vi]) { out_counts[vi] = 0; return; }

    // Write neighbors directly into the global output buffer — avoids a huge
    // per-thread local array. Duplicates are fine: greedy_color_conflict_graph
    // tolerates them (it just marks the same color twice). The CPU version
    // dedups for memory efficiency, not correctness.
    int* row = out_nbrs + vi * MAX_CONFLICT_DEG;
    int count = 0;

    auto add = [&](int vj) {
        if (vj == vi || vj < 0 || vj >= nv) return;
        if (!active_flag[vj]) return;
        if (count < MAX_CONFLICT_DEG) { row[count++] = vj; }
        else { atomicExch(d_overflow, 1); }
    };

    // 1. Elastic coupling — iterate incident triangles.
    for (int i = adj.offsets[vi]; i < adj.offsets[vi+1]; ++i) {
        const int ti = adj.tri_idx[i];
        add(tris[ti*3+0]);
        add(tris[ti*3+1]);
        add(tris[ti*3+2]);
    }

    // 2. NT barrier pairs — the vnt_* CSR already lists pairs involving vi.
    for (int i = bp.vnt_offsets[vi]; i < bp.vnt_offsets[vi+1]; ++i) {
        const int pair_idx = bp.vnt_pair_idx[i];
        add(bp.nt_data[pair_idx*4+0]);
        add(bp.nt_data[pair_idx*4+1]);
        add(bp.nt_data[pair_idx*4+2]);
        add(bp.nt_data[pair_idx*4+3]);
    }

    // 3. SS barrier pairs.
    for (int i = bp.vss_offsets[vi]; i < bp.vss_offsets[vi+1]; ++i) {
        const int pair_idx = bp.vss_pair_idx[i];
        add(bp.ss_data[pair_idx*4+0]);
        add(bp.ss_data[pair_idx*4+1]);
        add(bp.ss_data[pair_idx*4+2]);
        add(bp.ss_data[pair_idx*4+3]);
    }

    // 4. Swept-region BVH query — find vertices whose certified region overlaps vi's.
    if (cert_root >= 0) {
        const double qmin[3] = { cert_bmin[vi*3+0], cert_bmin[vi*3+1], cert_bmin[vi*3+2] };
        const double qmax[3] = { cert_bmax[vi*3+0], cert_bmax[vi*3+1], cert_bmax[vi*3+2] };
        int hits[128];
        const int nh = query_bvh_device(cert_bvh, cert_root, qmin, qmax, hits, 128);
        for (int h = 0; h < nh; ++h) {
            add(active_ids[hits[h]]);
        }
    }

    // Sort + dedup in place — drastically reduces effective degree because
    // NT/SS barrier pairs contribute the same neighbor many times (a vertex
    // appearing in N barrier pairs with a common partner adds that partner N
    // times). Insertion sort is fine for modest counts; worst case O(count²).
    for (int i = 1; i < count; ++i) {
        const int key = row[i];
        int j = i - 1;
        while (j >= 0 && row[j] > key) { row[j+1] = row[j]; --j; }
        row[j+1] = key;
    }
    int dedup = 0;
    for (int i = 0; i < count; ++i) {
        if (i == 0 || row[i] != row[i-1]) row[dedup++] = row[i];
    }
    out_counts[vi] = dedup;
}

// ============================================================================
// jacobi_predict_kernel  — one thread per vertex
// ============================================================================

__global__ static void jacobi_predict_kernel(
    int nv,
    MeshPtrs mesh, AdjPtrs adj, BpPtrs bp, CertPtrs cert, PinPtrs pins,
    GPUSimParams params,
    const double* d_x, const double* d_xhat,
    double* d_g, double* d_H, double* d_delta,
    double* d_box_min, double* d_box_max)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;

    double g[3], H[9], delta[3];
    compute_local_newton_direction_device(
        vi, mesh, adj, bp, pins, params, d_x, d_xhat, g, H, delta);

    for(int k=0;k<3;k++) d_g[vi*3+k]     = g[k];
    for(int k=0;k<9;k++) d_H[vi*9+k]     = H[k];
    for(int k=0;k<3;k++) d_delta[vi*3+k] = delta[k];

    double box_min[3], box_max[3];
    build_certified_region_device(vi, d_x, delta, cert, params.d_hat, box_min, box_max);
    for(int k=0;k<3;k++) { d_box_min[vi*3+k]=box_min[k]; d_box_max[vi*3+k]=box_max[k]; }
}

// ============================================================================
// Sweep-persistent session
// Keeps static-per-sweep GPU buffers resident so gpu_build_jacobi_predictions
// and gpu_parallel_commit skip their upload paths when a session is active.
// ============================================================================

struct GpuSolverSession {
    int nv = 0;
    GPUSimParams gparams{};

    // Static mesh / adj / broadphase / pins.
    GPURefMesh         gpu_mesh;
    GPUAdjacency       gpu_adj;
    GPUBroadPhaseCache gpu_bp;
    GPUPins            gpu_pins;
    GPUPinMap          gpu_pin_map;

    // Cert CSR + flattened box arrays.
    int *d_ntt_off = nullptr, *d_ntt_data = nullptr;
    int *d_nte_off = nullptr, *d_nte_data = nullptr;
    double *d_tbmin = nullptr, *d_tbmax = nullptr;
    double *d_ebmin = nullptr, *d_ebmax = nullptr;

    // BVH.
    BVHNodeGPU *d_tri_bvh = nullptr;
    BVHNodeGPU *d_node_bvh = nullptr;
    BVHNodeGPU *d_edge_bvh = nullptr;
    int        *d_edges    = nullptr;
    int tri_root = -1, node_root = -1, edge_root = -1;

    // xhat — static per sweep.
    double *d_xhat = nullptr;

    // x_current — re-uploaded each predict/commit call.
    double *d_x = nullptr;

    // Predictions (populated by predict kernel, consumed by commit kernel).
    double *d_pred_delta = nullptr;
    double *d_pred_bmin  = nullptr;
    double *d_pred_bmax  = nullptr;
    double *d_pred_g     = nullptr;
    double *d_pred_H     = nullptr;

    // Commit scratch — sized to max group.
    int    *d_group         = nullptr;
    int     group_capacity  = 0;
    double *d_commit_delta  = nullptr;
    double *d_commit_alpha  = nullptr;
    double *d_commit_ccd    = nullptr;
    double *d_commit_xafter = nullptr;

    // Conflict graph scratch — allocated up front, reused per iter.
    // Per-iter uploads: certified region AABBs, active flags, active IDs,
    // and a newly-built BVH over the certified regions.
    double *d_cert_bmin = nullptr;      // nv*3
    double *d_cert_bmax = nullptr;      // nv*3
    int    *d_active_flag = nullptr;    // nv
    int    *d_active_ids = nullptr;     // at most nv
    int     cert_bvh_capacity = 0;
    BVHNodeGPU *d_cert_bvh = nullptr;   // at most 2*nv-1
    // Compact-active box buffers for GPU LBVH construction.
    double *d_active_bmin = nullptr;    // n_active * 3
    double *d_active_bmax = nullptr;    // n_active * 3
    LBVHScratch lbvh_scratch;           // Morton codes + sort indices + parent + ready
    int    *d_conflict_nbrs = nullptr;  // nv * MAX_CONFLICT_DEG
    int    *d_conflict_counts = nullptr;// nv
    int    *d_overflow = nullptr;       // single int

    // Per-vertex color (int per vi). Uploaded after CPU greedy coloring so
    // the fused sweep kernel knows which color each vertex belongs to.
    int    *d_vertex_color = nullptr;   // nv
};

static GpuSolverSession* g_session = nullptr;

static void session_free(GpuSolverSession* s) {
    if (!s) return;
    auto F = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F(s->d_ntt_off); F(s->d_ntt_data); F(s->d_nte_off); F(s->d_nte_data);
    F(s->d_tbmin); F(s->d_tbmax); F(s->d_ebmin); F(s->d_ebmax);
    F(s->d_tri_bvh); F(s->d_node_bvh); F(s->d_edge_bvh); F(s->d_edges);
    F(s->d_xhat); F(s->d_x);
    F(s->d_pred_delta); F(s->d_pred_bmin); F(s->d_pred_bmax);
    F(s->d_pred_g); F(s->d_pred_H);
    F(s->d_group);
    F(s->d_commit_delta); F(s->d_commit_alpha); F(s->d_commit_ccd); F(s->d_commit_xafter);
    F(s->d_cert_bmin); F(s->d_cert_bmax); F(s->d_active_flag); F(s->d_active_ids);
    F(s->d_cert_bvh);
    F(s->d_conflict_nbrs); F(s->d_conflict_counts); F(s->d_overflow);
    F(s->d_vertex_color);
    F(s->d_active_bmin); F(s->d_active_bmax);
    lbvh_scratch_free(s->lbvh_scratch);
}

void gpu_solver_begin_sweep(
    const RefMesh&           ref_mesh,
    const VertexTriangleMap& adj_map,
    const std::vector<Pin>&  pins_vec,
    const SimParams&         params,
    const std::vector<Vec3>& xhat,
    const BroadPhase::Cache& bp_cache,
    const PinMap*            pin_map_ptr)
{
    if (g_session) gpu_solver_end_sweep();
    g_session = new GpuSolverSession();
    GpuSolverSession& s = *g_session;

    const int nv = static_cast<int>(xhat.size());
    s.nv = nv;
    s.gparams = GPUSimParams::from(params);

    // Clear any previously-persisted error that didn't belong to us.
    cudaError_t pre_err = cudaGetLastError();
    if (pre_err != cudaSuccess) {
        fprintf(stderr, "[begin_sweep] clearing prior CUDA error: %s\n", cudaGetErrorString(pre_err));
    }

    s.gpu_mesh.upload(ref_mesh);
    cudaError_t after_mesh = cudaGetLastError();
    if (after_mesh != cudaSuccess) {
        fprintf(stderr, "[begin_sweep] error after gpu_mesh.upload: %s\n", cudaGetErrorString(after_mesh));
    }
    s.gpu_adj.upload(adj_map, nv);
    s.gpu_bp.upload(bp_cache, nv);
    s.gpu_pins.upload(pins_vec);
    if (pin_map_ptr) {
        s.gpu_pin_map.upload(*pin_map_ptr);
    } else {
        std::vector<int> pm(nv, -1);
        s.gpu_pin_map.data.upload(pm.data(), nv);
        s.gpu_pin_map.num_verts = nv;
    }

    // xhat.
    std::vector<double> xhat_flat(nv*3);
    for (int i = 0; i < nv; ++i) {
        xhat_flat[i*3+0]=xhat[i](0); xhat_flat[i*3+1]=xhat[i](1); xhat_flat[i*3+2]=xhat[i](2);
    }
    cudaMalloc(&s.d_xhat, nv*3*sizeof(double));
    cudaMemcpy(s.d_xhat, xhat_flat.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);

    // x (allocate only; uploaded per call).
    cudaMalloc(&s.d_x, nv*3*sizeof(double));

    // Cert CSR.
    std::vector<int> ntt_off(nv+1, 0), nte_off(nv+1, 0);
    for (int vi = 0; vi < nv; ++vi) {
        if (vi < (int)bp_cache.node_to_tris.size())  ntt_off[vi+1] = (int)bp_cache.node_to_tris[vi].size();
        if (vi < (int)bp_cache.node_to_edges.size()) nte_off[vi+1] = (int)bp_cache.node_to_edges[vi].size();
    }
    for (int vi = 0; vi < nv; ++vi) { ntt_off[vi+1] += ntt_off[vi]; nte_off[vi+1] += nte_off[vi]; }
    std::vector<int> ntt_data(ntt_off[nv]), nte_data(nte_off[nv]);
    for (int vi = 0; vi < nv; ++vi) {
        if (vi < (int)bp_cache.node_to_tris.size()) {
            int pos = ntt_off[vi];
            for (int ti : bp_cache.node_to_tris[vi]) ntt_data[pos++] = ti;
        }
        if (vi < (int)bp_cache.node_to_edges.size()) {
            int pos = nte_off[vi];
            for (int ei : bp_cache.node_to_edges[vi]) nte_data[pos++] = ei;
        }
    }
    auto hmax = [](int a, int b){ return a > b ? a : b; };
    cudaMalloc(&s.d_ntt_off,  (nv+1)*sizeof(int));
    cudaMalloc(&s.d_ntt_data, hmax(1,(int)ntt_data.size())*sizeof(int));
    cudaMalloc(&s.d_nte_off,  (nv+1)*sizeof(int));
    cudaMalloc(&s.d_nte_data, hmax(1,(int)nte_data.size())*sizeof(int));
    cudaMemcpy(s.d_ntt_off, ntt_off.data(), (nv+1)*sizeof(int), cudaMemcpyHostToDevice);
    if (!ntt_data.empty()) cudaMemcpy(s.d_ntt_data, ntt_data.data(), ntt_data.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_nte_off, nte_off.data(), (nv+1)*sizeof(int), cudaMemcpyHostToDevice);
    if (!nte_data.empty()) cudaMemcpy(s.d_nte_data, nte_data.data(), nte_data.size()*sizeof(int), cudaMemcpyHostToDevice);

    // tri/edge box flat arrays.
    const int nt_bp = (int)bp_cache.tri_boxes.size();
    const int ne_bp = (int)bp_cache.edge_boxes.size();
    std::vector<double> tri_bmin(hmax(1,nt_bp)*3), tri_bmax(hmax(1,nt_bp)*3);
    std::vector<double> edg_bmin(hmax(1,ne_bp)*3), edg_bmax(hmax(1,ne_bp)*3);
    for (int i = 0; i < nt_bp; ++i) {
        tri_bmin[i*3+0]=bp_cache.tri_boxes[i].min(0); tri_bmin[i*3+1]=bp_cache.tri_boxes[i].min(1); tri_bmin[i*3+2]=bp_cache.tri_boxes[i].min(2);
        tri_bmax[i*3+0]=bp_cache.tri_boxes[i].max(0); tri_bmax[i*3+1]=bp_cache.tri_boxes[i].max(1); tri_bmax[i*3+2]=bp_cache.tri_boxes[i].max(2);
    }
    for (int i = 0; i < ne_bp; ++i) {
        edg_bmin[i*3+0]=bp_cache.edge_boxes[i].min(0); edg_bmin[i*3+1]=bp_cache.edge_boxes[i].min(1); edg_bmin[i*3+2]=bp_cache.edge_boxes[i].min(2);
        edg_bmax[i*3+0]=bp_cache.edge_boxes[i].max(0); edg_bmax[i*3+1]=bp_cache.edge_boxes[i].max(1); edg_bmax[i*3+2]=bp_cache.edge_boxes[i].max(2);
    }
    cudaMalloc(&s.d_tbmin, tri_bmin.size()*sizeof(double));
    cudaMalloc(&s.d_tbmax, tri_bmax.size()*sizeof(double));
    cudaMalloc(&s.d_ebmin, edg_bmin.size()*sizeof(double));
    cudaMalloc(&s.d_ebmax, edg_bmax.size()*sizeof(double));
    cudaMemcpy(s.d_tbmin, tri_bmin.data(), tri_bmin.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_tbmax, tri_bmax.data(), tri_bmax.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_ebmin, edg_bmin.data(), edg_bmin.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_ebmax, edg_bmax.data(), edg_bmax.size()*sizeof(double), cudaMemcpyHostToDevice);

    // BVH.
    auto upload_bvh = [&](const std::vector<BVHNode>& src, BVHNodeGPU*& d_out) {
        const int n = (int)src.size();
        const int cap = hmax(1, n);
        std::vector<BVHNodeGPU> flat(cap);
        for (int i = 0; i < n; ++i) {
            flat[i].bmin[0]=src[i].bbox.min(0); flat[i].bmin[1]=src[i].bbox.min(1); flat[i].bmin[2]=src[i].bbox.min(2);
            flat[i].bmax[0]=src[i].bbox.max(0); flat[i].bmax[1]=src[i].bbox.max(1); flat[i].bmax[2]=src[i].bbox.max(2);
            flat[i].left=src[i].left; flat[i].right=src[i].right;
            flat[i].leafIndex=src[i].leafIndex; flat[i].pad=0;
        }
        cudaMalloc(&d_out, cap*sizeof(BVHNodeGPU));
        cudaMemcpy(d_out, flat.data(), cap*sizeof(BVHNodeGPU), cudaMemcpyHostToDevice);
    };
    upload_bvh(bp_cache.tri_bvh_nodes,  s.d_tri_bvh);
    upload_bvh(bp_cache.node_bvh_nodes, s.d_node_bvh);
    upload_bvh(bp_cache.edge_bvh_nodes, s.d_edge_bvh);
    s.tri_root  = bp_cache.tri_root;
    s.node_root = bp_cache.node_root;
    s.edge_root = bp_cache.edge_root;

    const int num_edges = (int)bp_cache.edges.size();
    std::vector<int> edges_flat(hmax(1, num_edges)*2, 0);
    for (int i = 0; i < num_edges; ++i) {
        edges_flat[i*2+0] = bp_cache.edges[i][0];
        edges_flat[i*2+1] = bp_cache.edges[i][1];
    }
    cudaMalloc(&s.d_edges, hmax(1,num_edges)*2*sizeof(int));
    cudaMemcpy(s.d_edges, edges_flat.data(), hmax(1,num_edges)*2*sizeof(int), cudaMemcpyHostToDevice);

    // Prediction / output buffers.
    cudaMalloc(&s.d_pred_delta, nv*3*sizeof(double));
    cudaMalloc(&s.d_pred_bmin,  nv*3*sizeof(double));
    cudaMalloc(&s.d_pred_bmax,  nv*3*sizeof(double));
    cudaMalloc(&s.d_pred_g,     nv*3*sizeof(double));
    cudaMalloc(&s.d_pred_H,     nv*9*sizeof(double));

    // Conflict-graph buffers.
    cudaMalloc(&s.d_cert_bmin, nv*3*sizeof(double));
    cudaMalloc(&s.d_cert_bmax, nv*3*sizeof(double));
    cudaMalloc(&s.d_active_flag, nv*sizeof(int));
    cudaMalloc(&s.d_active_ids, nv*sizeof(int));
    cudaMalloc(&s.d_active_bmin, nv*3*sizeof(double));
    cudaMalloc(&s.d_active_bmax, nv*3*sizeof(double));
    s.cert_bvh_capacity = hmax(1, 2*nv - 1);
    cudaMalloc(&s.d_cert_bvh, s.cert_bvh_capacity * sizeof(BVHNodeGPU));
    cudaError_t alloc_err = cudaMalloc(&s.d_conflict_nbrs, (size_t)nv * MAX_CONFLICT_DEG * sizeof(int));
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "[begin_sweep] d_conflict_nbrs alloc failed (nv=%d, MAX_DEG=%d, %.1f MB): %s\n",
                nv, MAX_CONFLICT_DEG,
                (double)((size_t)nv * MAX_CONFLICT_DEG * sizeof(int)) / (1024.0*1024.0),
                cudaGetErrorString(alloc_err));
    }
    cudaMalloc(&s.d_conflict_counts, nv * sizeof(int));
    cudaMalloc(&s.d_overflow, sizeof(int));
    cudaMalloc(&s.d_vertex_color, nv * sizeof(int));

}

void gpu_solver_end_sweep() {
    if (!g_session) return;
    session_free(g_session);
    delete g_session;
    g_session = nullptr;
}

// ----------------------------------------------------------------------------
// gpu_build_conflict_graph — host wrapper for build_conflict_kernel.
// Builds the per-iter swept-region BVH on CPU (small, fast), uploads it, then
// launches the GPU kernel which collects all 4 edge sources per vertex.
// ----------------------------------------------------------------------------
std::vector<std::vector<int>> gpu_build_conflict_graph(
    const std::vector<JacobiPrediction>& predictions)
{
    if (!g_session) return {};
    GpuSolverSession& s = *g_session;
    const int nv = s.nv;
    if ((int)predictions.size() != nv) return {};

    std::vector<std::vector<int>> graph(nv);

    // Flatten per-vertex certified AABBs + compute scene bounds + active list.
    std::vector<int>    active_ids;
    std::vector<int>    active_flag(nv, 0);
    std::vector<double> cert_bmin(nv*3), cert_bmax(nv*3);
    active_ids.reserve(nv);
    double sxmin =  std::numeric_limits<double>::infinity();
    double symin =  sxmin, szmin = sxmin;
    double sxmax = -std::numeric_limits<double>::infinity();
    double symax =  sxmax, szmax = sxmax;
    for (int i = 0; i < nv; ++i) {
        cert_bmin[i*3+0] = predictions[i].certified_region.min(0);
        cert_bmin[i*3+1] = predictions[i].certified_region.min(1);
        cert_bmin[i*3+2] = predictions[i].certified_region.min(2);
        cert_bmax[i*3+0] = predictions[i].certified_region.max(0);
        cert_bmax[i*3+1] = predictions[i].certified_region.max(1);
        cert_bmax[i*3+2] = predictions[i].certified_region.max(2);
        if (!predictions[i].active) continue;
        active_flag[i] = 1;
        active_ids.push_back(i);
        for (int k = 0; k < 3; ++k) {
            const double bmin_k = cert_bmin[i*3+k];
            const double bmax_k = cert_bmax[i*3+k];
            if (k == 0) { sxmin = std::min(sxmin, bmin_k); sxmax = std::max(sxmax, bmax_k); }
            if (k == 1) { symin = std::min(symin, bmin_k); symax = std::max(symax, bmax_k); }
            if (k == 2) { szmin = std::min(szmin, bmin_k); szmax = std::max(szmax, bmax_k); }
        }
    }
    const int n_active = (int)active_ids.size();

    // Upload per-vertex boxes + active metadata.
    cudaMemcpy(s.d_cert_bmin, cert_bmin.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_cert_bmax, cert_bmax.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_active_flag, active_flag.data(), nv*sizeof(int), cudaMemcpyHostToDevice);
    if (n_active > 0) {
        cudaMemcpy(s.d_active_ids, active_ids.data(), n_active*sizeof(int), cudaMemcpyHostToDevice);
    }

    // Build BVH over active certified AABBs, either on GPU (LBVH, still WIP —
    // the Karras split search has a correctness bug producing ~5% missing
    // edges) or on CPU (build_bvh — slower at 100k but correct).
    int sw_root = -1;
#ifdef GPU_USE_LBVH
    if (n_active > 0) {
        const int gblock = 256;
        const int ggrid  = (n_active + gblock - 1) / gblock;
        lbvh_gather_active_boxes_kernel<<<ggrid, gblock>>>(
            n_active, s.d_active_ids,
            s.d_cert_bmin, s.d_cert_bmax,
            s.d_active_bmin, s.d_active_bmax);

        const int need_nodes = std::max(1, 2*n_active - 1);
        if (need_nodes > s.cert_bvh_capacity) {
            cudaFree(s.d_cert_bvh);
            s.cert_bvh_capacity = need_nodes;
            cudaMalloc(&s.d_cert_bvh, s.cert_bvh_capacity * sizeof(BVHNodeGPU));
        }
        sw_root = lbvh_build(
            n_active, s.d_active_bmin, s.d_active_bmax,
            sxmin, symin, szmin, sxmax, symax, szmax,
            s.d_cert_bvh, s.lbvh_scratch);
    }
#else
    // CPU BVH build (legacy path — known-correct). Falls back if LBVH toggle off.
    std::vector<AABB> active_boxes;
    active_boxes.reserve(n_active);
    for (int vi : active_ids) active_boxes.push_back(predictions[vi].certified_region);
    std::vector<BVHNode> sw_bvh_nodes;
    sw_root = build_bvh(active_boxes, sw_bvh_nodes);
    std::vector<BVHNodeGPU> bvh_flat(std::max(1, (int)sw_bvh_nodes.size()));
    for (int i = 0; i < (int)sw_bvh_nodes.size(); ++i) {
        bvh_flat[i].bmin[0]=sw_bvh_nodes[i].bbox.min(0); bvh_flat[i].bmin[1]=sw_bvh_nodes[i].bbox.min(1); bvh_flat[i].bmin[2]=sw_bvh_nodes[i].bbox.min(2);
        bvh_flat[i].bmax[0]=sw_bvh_nodes[i].bbox.max(0); bvh_flat[i].bmax[1]=sw_bvh_nodes[i].bbox.max(1); bvh_flat[i].bmax[2]=sw_bvh_nodes[i].bbox.max(2);
        bvh_flat[i].left=sw_bvh_nodes[i].left; bvh_flat[i].right=sw_bvh_nodes[i].right;
        bvh_flat[i].leafIndex=sw_bvh_nodes[i].leafIndex; bvh_flat[i].pad=0;
    }
    if ((int)bvh_flat.size() > s.cert_bvh_capacity) {
        cudaFree(s.d_cert_bvh);
        s.cert_bvh_capacity = (int)bvh_flat.size();
        cudaMalloc(&s.d_cert_bvh, s.cert_bvh_capacity * sizeof(BVHNodeGPU));
    }
    cudaMemcpy(s.d_cert_bvh, bvh_flat.data(), bvh_flat.size()*sizeof(BVHNodeGPU), cudaMemcpyHostToDevice);
#endif

    const int zero = 0;
    cudaMemcpy(s.d_overflow, &zero, sizeof(int), cudaMemcpyHostToDevice);

    AdjPtrs adj_p{ s.gpu_adj.offsets.ptr, s.gpu_adj.tri_idx.ptr, s.gpu_adj.tri_local.ptr };
    BpPtrs  bp_p{
        s.gpu_bp.vnt_offsets.ptr, s.gpu_bp.vnt_pair_idx.ptr, s.gpu_bp.vnt_dof.ptr, s.gpu_bp.nt_data.ptr,
        s.gpu_bp.vss_offsets.ptr, s.gpu_bp.vss_pair_idx.ptr, s.gpu_bp.vss_dof.ptr, s.gpu_bp.ss_data.ptr
    };

    const int block = 256;
    const int grid  = (nv + block - 1) / block;
    build_conflict_kernel<<<grid, block>>>(
        nv, adj_p, s.gpu_mesh.tris.ptr, bp_p,
        s.d_cert_bvh, sw_root, s.d_active_ids,
        s.d_cert_bmin, s.d_cert_bmax, s.d_active_flag,
        s.d_conflict_nbrs, s.d_conflict_counts, s.d_overflow);
    cudaError_t ksync = cudaDeviceSynchronize();
    if (ksync != cudaSuccess) {
        fprintf(stderr, "[gpu_build_conflict_graph] kernel error: %s\n", cudaGetErrorString(ksync));
    }

    int overflow = 0;
    cudaMemcpy(&overflow, s.d_overflow, sizeof(int), cudaMemcpyDeviceToHost);

    // Download per-vertex counts first so we know the actual row lengths.
    std::vector<int> h_counts(nv);
    cudaMemcpy(h_counts.data(), s.d_conflict_counts, nv*sizeof(int), cudaMemcpyDeviceToHost);

    // Find max count + overflow rate. max_c caps at MAX_CONFLICT_DEG since we
    // stop writing beyond that — overflow_count tells us how many vertices hit
    // the cap.
    int max_c = 0;
    int overflow_count = 0;
    for (int vi = 0; vi < nv; ++vi) {
        if (h_counts[vi] > max_c) max_c = h_counts[vi];
        if (h_counts[vi] >= MAX_CONFLICT_DEG) ++overflow_count;
    }
    // overflow flag is set pre-dedup, so ignore it — only warn if vertices
    // actually exceeded the cap after dedup (i.e., real unique-neighbor loss).
    if (overflow_count > 0) {
        fprintf(stderr, "[conflict] %d/%d verts clamped at cap=%d (max observed=%d)\n",
                overflow_count, nv, MAX_CONFLICT_DEG, max_c);
    }
    if (max_c == 0) return graph;

    // Download a contiguous slab [0..nv) x [0..max_c). Same stride as the
    // allocated buffer (MAX_CONFLICT_DEG) but we use cudaMemcpy2D to copy
    // only max_c per row.
    std::vector<int> h_nbrs(nv * max_c);
    cudaMemcpy2D(
        h_nbrs.data(),          max_c * sizeof(int),          // dst, dst_pitch
        s.d_conflict_nbrs,      MAX_CONFLICT_DEG * sizeof(int), // src, src_pitch
        max_c * sizeof(int),                                    // width bytes
        nv,                                                     // height (rows)
        cudaMemcpyDeviceToHost);

    for (int vi = 0; vi < nv; ++vi) {
        const int c = h_counts[vi];
        if (c <= 0) continue;
        graph[vi].assign(h_nbrs.begin() + vi*max_c,
                         h_nbrs.begin() + vi*max_c + c);
    }
    return graph;
}

// Forward decl — colored_gs_sweep_kernel is defined after the commit kernel.
__global__ static void colored_gs_sweep_kernel(
    int nv, int num_colors, const int* vertex_color,
    const double* pred_delta, const double* pred_bmin, const double* pred_bmax,
    MeshPtrs mesh, AdjPtrs adj, BpPtrs bp, CertPtrs cert, BVHPtrs bvh, PinPtrs pins,
    GPUSimParams params, double* d_x, const double* d_xhat);

// ----------------------------------------------------------------------------
// gpu_fused_sweep — host wrapper for colored_gs_sweep_kernel.
// Uploads vertex_color, launches cooperatively, downloads d_x into xnew.
// ----------------------------------------------------------------------------
bool gpu_fused_sweep(
    const std::vector<JacobiPrediction>& predictions,
    const std::vector<std::vector<int>>& color_groups,
    std::vector<Vec3>&                   xnew)
{
    if (!g_session) return false;
    GpuSolverSession& s = *g_session;
    const int nv = s.nv;
    if ((int)predictions.size() != nv) return false;
    if ((int)xnew.size() != nv) return false;

    // Flatten color_groups -> per-vertex color. Inactive vertices stay -1.
    std::vector<int> vertex_color(nv, -1);
    const int num_colors = (int)color_groups.size();
    for (int c = 0; c < num_colors; ++c) {
        for (int vi : color_groups[c]) {
            if (vi >= 0 && vi < nv) vertex_color[vi] = c;
        }
    }
    cudaMemcpy(s.d_vertex_color, vertex_color.data(), nv*sizeof(int), cudaMemcpyHostToDevice);

    // Upload x into d_x, predictions into d_pred_*.
    std::vector<double> x_flat(nv*3), pred_delta(nv*3), pred_bmin(nv*3), pred_bmax(nv*3);
    for (int i = 0; i < nv; ++i) {
        x_flat[i*3+0]=xnew[i](0); x_flat[i*3+1]=xnew[i](1); x_flat[i*3+2]=xnew[i](2);
        pred_delta[i*3+0]=predictions[i].delta(0);
        pred_delta[i*3+1]=predictions[i].delta(1);
        pred_delta[i*3+2]=predictions[i].delta(2);
        pred_bmin[i*3+0]=predictions[i].certified_region.min(0);
        pred_bmin[i*3+1]=predictions[i].certified_region.min(1);
        pred_bmin[i*3+2]=predictions[i].certified_region.min(2);
        pred_bmax[i*3+0]=predictions[i].certified_region.max(0);
        pred_bmax[i*3+1]=predictions[i].certified_region.max(1);
        pred_bmax[i*3+2]=predictions[i].certified_region.max(2);
    }
    cudaMemcpy(s.d_x, x_flat.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_pred_delta, pred_delta.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_pred_bmin,  pred_bmin.data(),  nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_pred_bmax,  pred_bmax.data(),  nv*3*sizeof(double), cudaMemcpyHostToDevice);

    MeshPtrs mesh_p{
        s.gpu_mesh.tris.ptr, s.gpu_mesh.Dm_inv.ptr, s.gpu_mesh.area.ptr, s.gpu_mesh.mass.ptr,
        s.gpu_mesh.hinge_v.ptr, s.gpu_mesh.hinge_bar_theta.ptr, s.gpu_mesh.hinge_ce.ptr,
        s.gpu_mesh.hinge_adj_offsets.ptr, s.gpu_mesh.hinge_adj_hi.ptr, s.gpu_mesh.hinge_adj_role.ptr
    };
    AdjPtrs adj_p{ s.gpu_adj.offsets.ptr, s.gpu_adj.tri_idx.ptr, s.gpu_adj.tri_local.ptr };
    BpPtrs  bp_p{
        s.gpu_bp.vnt_offsets.ptr, s.gpu_bp.vnt_pair_idx.ptr, s.gpu_bp.vnt_dof.ptr, s.gpu_bp.nt_data.ptr,
        s.gpu_bp.vss_offsets.ptr, s.gpu_bp.vss_pair_idx.ptr, s.gpu_bp.vss_dof.ptr, s.gpu_bp.ss_data.ptr
    };
    CertPtrs cert_p{ s.d_ntt_off, s.d_ntt_data, s.d_nte_off, s.d_nte_data, s.d_tbmin, s.d_tbmax, s.d_ebmin, s.d_ebmax };
    BVHPtrs  bvh_p { s.d_tri_bvh, s.d_node_bvh, s.d_edge_bvh, s.d_edges, s.tri_root, s.node_root, s.edge_root };
    PinPtrs  pins_p{ s.gpu_pins.targets.ptr, s.gpu_pin_map.data.ptr };

    // Block size 32 = one warp per block. Spreading across many SMs lets the
    // memory subsystem have more concurrent requests in flight, partially
    // hiding the BVH/CCD random-access latency that dominates per-thread time.
    const int block = 32;
    const int grid  = (nv + block - 1) / block;

    // Cooperative launch: grid.sync() requires cudaLaunchCooperativeKernel.
    // All blocks must fit on SMs concurrently. Verify via occupancy query.
    int num_sms = 0, max_blocks_per_sm = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, colored_gs_sweep_kernel, block, 0);
    const int max_blocks = num_sms * max_blocks_per_sm;
    if (grid > max_blocks) {
        fprintf(stderr, "[gpu_fused_sweep] grid=%d exceeds cooperative limit=%d — falling back\n",
                grid, max_blocks);
        return false;
    }

    void* args[] = {
        (void*)&nv, (void*)&num_colors, (void*)&s.d_vertex_color,
        (void*)&s.d_pred_delta, (void*)&s.d_pred_bmin, (void*)&s.d_pred_bmax,
        (void*)&mesh_p, (void*)&adj_p, (void*)&bp_p,
        (void*)&cert_p, (void*)&bvh_p, (void*)&pins_p,
        (void*)&s.gparams, (void*)&s.d_x, (void*)&s.d_xhat
    };
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)colored_gs_sweep_kernel, grid, block, args, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_fused_sweep] cooperative launch error: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaError_t ksync = cudaDeviceSynchronize();
    if (ksync != cudaSuccess) {
        fprintf(stderr, "[gpu_fused_sweep] kernel error: %s\n", cudaGetErrorString(ksync));
        return false;
    }

    // Download updated x into xnew.
    std::vector<double> h_x(nv*3);
    cudaMemcpy(h_x.data(), s.d_x, nv*3*sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nv; ++i) {
        xnew[i] = Vec3(h_x[i*3+0], h_x[i*3+1], h_x[i*3+2]);
    }
    return true;
}

// ----------------------------------------------------------------------------
// gpu_compute_global_residual — host wrapper for residual_kernel. Session-only.
// ----------------------------------------------------------------------------
double gpu_compute_global_residual(const std::vector<Vec3>& x_current) {
    if (!g_session) return -1.0;
    GpuSolverSession& s = *g_session;
    const int nv = s.nv;
    if ((int)x_current.size() != nv) return -1.0;

    // Upload current x into the session buffer.
    std::vector<double> x_flat(nv*3);
    for (int i = 0; i < nv; ++i) {
        x_flat[i*3+0]=x_current[i](0); x_flat[i*3+1]=x_current[i](1); x_flat[i*3+2]=x_current[i](2);
    }
    cudaMemcpy(s.d_x, x_flat.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);

    // Single double for atomic max reduction.
    double* d_rmax = nullptr;
    cudaMalloc(&d_rmax, sizeof(double));
    const double zero = 0.0;
    cudaMemcpy(d_rmax, &zero, sizeof(double), cudaMemcpyHostToDevice);

    MeshPtrs mesh_p{
        s.gpu_mesh.tris.ptr, s.gpu_mesh.Dm_inv.ptr, s.gpu_mesh.area.ptr, s.gpu_mesh.mass.ptr,
        s.gpu_mesh.hinge_v.ptr, s.gpu_mesh.hinge_bar_theta.ptr, s.gpu_mesh.hinge_ce.ptr,
        s.gpu_mesh.hinge_adj_offsets.ptr, s.gpu_mesh.hinge_adj_hi.ptr, s.gpu_mesh.hinge_adj_role.ptr
    };
    AdjPtrs adj_p{ s.gpu_adj.offsets.ptr, s.gpu_adj.tri_idx.ptr, s.gpu_adj.tri_local.ptr };
    BpPtrs  bp_p{
        s.gpu_bp.vnt_offsets.ptr, s.gpu_bp.vnt_pair_idx.ptr, s.gpu_bp.vnt_dof.ptr, s.gpu_bp.nt_data.ptr,
        s.gpu_bp.vss_offsets.ptr, s.gpu_bp.vss_pair_idx.ptr, s.gpu_bp.vss_dof.ptr, s.gpu_bp.ss_data.ptr
    };
    PinPtrs pins_p{ s.gpu_pins.targets.ptr, s.gpu_pin_map.data.ptr };

    const int block = 256;
    const int grid  = (nv + block - 1) / block;
    residual_kernel<<<grid, block>>>(
        nv, mesh_p, adj_p, bp_p, pins_p, s.gparams,
        s.d_x, s.d_xhat,
        s.gparams.mass_normalize_residual ? 1 : 0,
        d_rmax);
    cudaError_t launch_err = cudaGetLastError();
    cudaError_t ksync = cudaDeviceSynchronize();
    if (launch_err != cudaSuccess || ksync != cudaSuccess) {
        fprintf(stderr, "[gpu_compute_global_residual] launch=%s sync=%s  (nv=%d block=%d grid=%d)\n",
                cudaGetErrorString(launch_err), cudaGetErrorString(ksync),
                nv, block, grid);
    }

    double rmax = 0.0;
    cudaMemcpy(&rmax, d_rmax, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rmax);
    return rmax;
}

// Ensure session commit scratch is at least `ng` entries; grow if needed.
static void session_reserve_group(GpuSolverSession& s, int ng) {
    if (ng <= s.group_capacity) return;
    auto F = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F(s.d_group);
    F(s.d_commit_delta); F(s.d_commit_alpha); F(s.d_commit_ccd); F(s.d_commit_xafter);
    s.group_capacity = ng;
    cudaMalloc(&s.d_group,         ng*sizeof(int));
    cudaMalloc(&s.d_commit_delta,  ng*3*sizeof(double));
    cudaMalloc(&s.d_commit_alpha,  ng*sizeof(double));
    cudaMalloc(&s.d_commit_ccd,    ng*sizeof(double));
    cudaMalloc(&s.d_commit_xafter, ng*3*sizeof(double));
}

// ============================================================================
// gpu_build_jacobi_predictions — host wrapper
// ============================================================================

void gpu_build_jacobi_predictions(
    const RefMesh&                 ref_mesh,
    const VertexTriangleMap&       adj_map,
    const std::vector<Pin>&        pins_vec,
    const SimParams&               params,
    const std::vector<Vec3>&       x,
    const std::vector<Vec3>&       xhat,
    const BroadPhase::Cache&       bp_cache,
    std::vector<JacobiPrediction>& predictions,
    const PinMap*                  pin_map_ptr)
{
    const int nv = static_cast<int>(x.size());
    predictions.clear();
    predictions.resize(nv);
    for(auto& p: predictions) p.active = true;

    // ---------------------------------------------------------------------
    // Fast path: when a sweep session is active, the static data (mesh, adj,
    // broad-phase, BVH, pins, xhat, cert CSR) is already resident on device.
    // Only x_current needs to be uploaded; output goes into session buffers.
    // ---------------------------------------------------------------------
    if (g_session && g_session->nv == nv) {
        GpuSolverSession& s = *g_session;
        std::vector<double> x_flat(nv*3);
        for (int i = 0; i < nv; ++i) {
            x_flat[i*3+0]=x[i](0); x_flat[i*3+1]=x[i](1); x_flat[i*3+2]=x[i](2);
        }
        cudaMemcpy(s.d_x, x_flat.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);

        MeshPtrs mesh_p{
            s.gpu_mesh.tris.ptr, s.gpu_mesh.Dm_inv.ptr, s.gpu_mesh.area.ptr, s.gpu_mesh.mass.ptr,
            s.gpu_mesh.hinge_v.ptr, s.gpu_mesh.hinge_bar_theta.ptr, s.gpu_mesh.hinge_ce.ptr,
            s.gpu_mesh.hinge_adj_offsets.ptr, s.gpu_mesh.hinge_adj_hi.ptr, s.gpu_mesh.hinge_adj_role.ptr
        };
        AdjPtrs adj_p{ s.gpu_adj.offsets.ptr, s.gpu_adj.tri_idx.ptr, s.gpu_adj.tri_local.ptr };
        BpPtrs  bp_p{
            s.gpu_bp.vnt_offsets.ptr, s.gpu_bp.vnt_pair_idx.ptr, s.gpu_bp.vnt_dof.ptr, s.gpu_bp.nt_data.ptr,
            s.gpu_bp.vss_offsets.ptr, s.gpu_bp.vss_pair_idx.ptr, s.gpu_bp.vss_dof.ptr, s.gpu_bp.ss_data.ptr
        };
        CertPtrs cert_p{ s.d_ntt_off, s.d_ntt_data, s.d_nte_off, s.d_nte_data, s.d_tbmin, s.d_tbmax, s.d_ebmin, s.d_ebmax };
        PinPtrs  pins_p{ s.gpu_pins.targets.ptr, s.gpu_pin_map.data.ptr };

        const int block = 256;
        const int grid  = (nv + block - 1) / block;
        jacobi_predict_kernel<<<grid, block>>>(
            nv, mesh_p, adj_p, bp_p, cert_p, pins_p, s.gparams,
            s.d_x, s.d_xhat,
            s.d_pred_g, s.d_pred_H, s.d_pred_delta, s.d_pred_bmin, s.d_pred_bmax);
        cudaDeviceSynchronize();

        // Download predictions into CPU vector so conflict graph / coloring can use them.
        std::vector<double> h_g(nv*3), h_H(nv*9), h_delta(nv*3), h_bmin(nv*3), h_bmax(nv*3);
        cudaMemcpy(h_g.data(),     s.d_pred_g,     nv*3*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_H.data(),     s.d_pred_H,     nv*9*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_delta.data(), s.d_pred_delta, nv*3*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bmin.data(),  s.d_pred_bmin,  nv*3*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bmax.data(),  s.d_pred_bmax,  nv*3*sizeof(double), cudaMemcpyDeviceToHost);
        for (int vi = 0; vi < nv; ++vi) {
            auto& pred = predictions[vi];
            pred.active = true;
            pred.g      = Vec3(h_g[vi*3], h_g[vi*3+1], h_g[vi*3+2]);
            pred.delta  = Vec3(h_delta[vi*3], h_delta[vi*3+1], h_delta[vi*3+2]);
            pred.certified_region.min = Vec3(h_bmin[vi*3], h_bmin[vi*3+1], h_bmin[vi*3+2]);
            pred.certified_region.max = Vec3(h_bmax[vi*3], h_bmax[vi*3+1], h_bmax[vi*3+2]);
        }
        return;
    }

    GPUSimParams gparams = GPUSimParams::from(params);

    // --- Upload x, xhat ---
    std::vector<double> x_flat(nv*3), xhat_flat(nv*3);
    for(int i=0;i<nv;i++){
        x_flat[i*3+0]=x[i](0); x_flat[i*3+1]=x[i](1); x_flat[i*3+2]=x[i](2);
        xhat_flat[i*3+0]=xhat[i](0); xhat_flat[i*3+1]=xhat[i](1); xhat_flat[i*3+2]=xhat[i](2);
    }
    double *d_x, *d_xhat;
    cudaMalloc(&d_x,    nv*3*sizeof(double));
    cudaMalloc(&d_xhat, nv*3*sizeof(double));
    cudaMemcpy(d_x,    x_flat.data(),    nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xhat, xhat_flat.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);

    // --- Upload mesh (GPURefMesh) ---
    GPURefMesh gpu_mesh;
    gpu_mesh.upload(ref_mesh);

    // --- Upload adjacency (GPUAdjacency) ---
    GPUAdjacency gpu_adj;
    gpu_adj.upload(adj_map, nv);

    // --- Upload broad-phase CSR (GPUBroadPhaseCache) ---
    GPUBroadPhaseCache gpu_bp;
    gpu_bp.upload(bp_cache, nv);

    // --- Upload pins ---
    GPUPins gpu_pins;
    gpu_pins.upload(pins_vec);
    GPUPinMap gpu_pin_map;
    if (pin_map_ptr) {
        gpu_pin_map.upload(*pin_map_ptr);
    } else {
        std::vector<int> pm(nv, -1);
        gpu_pin_map.data.upload(pm.data(), nv);
        gpu_pin_map.num_verts = nv;
    }

    // --- Build certified region CSR arrays ---
    // node_to_tris CSR
    std::vector<int> ntt_off(nv+1, 0);
    for(int vi=0;vi<nv;vi++) {
        if(vi < (int)bp_cache.node_to_tris.size())
            ntt_off[vi+1] = (int)bp_cache.node_to_tris[vi].size();
    }
    for(int vi=0;vi<nv;vi++) ntt_off[vi+1]+=ntt_off[vi];
    std::vector<int> ntt_data(ntt_off[nv]);
    for(int vi=0;vi<nv;vi++) {
        if(vi>=(int)bp_cache.node_to_tris.size()) continue;
        int pos=ntt_off[vi];
        for(int ti: bp_cache.node_to_tris[vi]) ntt_data[pos++]=ti;
    }

    // node_to_edges CSR
    std::vector<int> nte_off(nv+1, 0);
    for(int vi=0;vi<nv;vi++) {
        if(vi < (int)bp_cache.node_to_edges.size())
            nte_off[vi+1] = (int)bp_cache.node_to_edges[vi].size();
    }
    for(int vi=0;vi<nv;vi++) nte_off[vi+1]+=nte_off[vi];
    std::vector<int> nte_data(nte_off[nv]);
    for(int vi=0;vi<nv;vi++) {
        if(vi>=(int)bp_cache.node_to_edges.size()) continue;
        int pos=nte_off[vi];
        for(int ei: bp_cache.node_to_edges[vi]) nte_data[pos++]=ei;
    }

    // Flatten tri_boxes and edge_boxes
    int num_tris  = (int)bp_cache.tri_boxes.size();
    int num_edges = (int)bp_cache.edge_boxes.size();
    std::vector<double> tri_bmin(num_tris*3), tri_bmax(num_tris*3);
    std::vector<double> edge_bmin(num_edges*3), edge_bmax(num_edges*3);
    for(int i=0;i<num_tris;i++) {
        tri_bmin[i*3+0]=bp_cache.tri_boxes[i].min(0);
        tri_bmin[i*3+1]=bp_cache.tri_boxes[i].min(1);
        tri_bmin[i*3+2]=bp_cache.tri_boxes[i].min(2);
        tri_bmax[i*3+0]=bp_cache.tri_boxes[i].max(0);
        tri_bmax[i*3+1]=bp_cache.tri_boxes[i].max(1);
        tri_bmax[i*3+2]=bp_cache.tri_boxes[i].max(2);
    }
    for(int i=0;i<num_edges;i++) {
        edge_bmin[i*3+0]=bp_cache.edge_boxes[i].min(0);
        edge_bmin[i*3+1]=bp_cache.edge_boxes[i].min(1);
        edge_bmin[i*3+2]=bp_cache.edge_boxes[i].min(2);
        edge_bmax[i*3+0]=bp_cache.edge_boxes[i].max(0);
        edge_bmax[i*3+1]=bp_cache.edge_boxes[i].max(1);
        edge_bmax[i*3+2]=bp_cache.edge_boxes[i].max(2);
    }

    // Upload certified region arrays
    int *d_ntt_off, *d_ntt_data, *d_nte_off, *d_nte_data;
    double *d_tbmin, *d_tbmax, *d_ebmin, *d_ebmax;
    auto hmax = [](int a, int b){ return a > b ? a : b; };
    cudaMalloc(&d_ntt_off,  (nv+1)*sizeof(int));
    cudaMalloc(&d_ntt_data, hmax(1,(int)ntt_data.size())*sizeof(int));
    cudaMalloc(&d_nte_off,  (nv+1)*sizeof(int));
    cudaMalloc(&d_nte_data, hmax(1,(int)nte_data.size())*sizeof(int));
    cudaMalloc(&d_tbmin, hmax(1,num_tris*3)*sizeof(double));
    cudaMalloc(&d_tbmax, hmax(1,num_tris*3)*sizeof(double));
    cudaMalloc(&d_ebmin, hmax(1,num_edges*3)*sizeof(double));
    cudaMalloc(&d_ebmax, hmax(1,num_edges*3)*sizeof(double));
    cudaMemcpy(d_ntt_off,  ntt_off.data(),  (nv+1)*sizeof(int),            cudaMemcpyHostToDevice);
    if(!ntt_data.empty()) cudaMemcpy(d_ntt_data, ntt_data.data(), ntt_data.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nte_off,  nte_off.data(),  (nv+1)*sizeof(int),            cudaMemcpyHostToDevice);
    if(!nte_data.empty()) cudaMemcpy(d_nte_data, nte_data.data(), nte_data.size()*sizeof(int), cudaMemcpyHostToDevice);
    if(num_tris>0) {
        cudaMemcpy(d_tbmin, tri_bmin.data(), num_tris*3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tbmax, tri_bmax.data(), num_tris*3*sizeof(double), cudaMemcpyHostToDevice);
    }
    if(num_edges>0) {
        cudaMemcpy(d_ebmin, edge_bmin.data(), num_edges*3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ebmax, edge_bmax.data(), num_edges*3*sizeof(double), cudaMemcpyHostToDevice);
    }

    // Output arrays
    double *d_g, *d_H, *d_delta, *d_box_min, *d_box_max;
    cudaMalloc(&d_g,       nv*3*sizeof(double));
    cudaMalloc(&d_H,       nv*9*sizeof(double));
    cudaMalloc(&d_delta,   nv*3*sizeof(double));
    cudaMalloc(&d_box_min, nv*3*sizeof(double));
    cudaMalloc(&d_box_max, nv*3*sizeof(double));

    // Assemble parameter structs
    MeshPtrs mesh_p{
        gpu_mesh.tris.ptr, gpu_mesh.Dm_inv.ptr, gpu_mesh.area.ptr, gpu_mesh.mass.ptr,
        gpu_mesh.hinge_v.ptr, gpu_mesh.hinge_bar_theta.ptr, gpu_mesh.hinge_ce.ptr,
        gpu_mesh.hinge_adj_offsets.ptr, gpu_mesh.hinge_adj_hi.ptr, gpu_mesh.hinge_adj_role.ptr
    };
    AdjPtrs adj_p{ gpu_adj.offsets.ptr, gpu_adj.tri_idx.ptr, gpu_adj.tri_local.ptr };
    BpPtrs bp_p{
        gpu_bp.vnt_offsets.ptr, gpu_bp.vnt_pair_idx.ptr, gpu_bp.vnt_dof.ptr, gpu_bp.nt_data.ptr,
        gpu_bp.vss_offsets.ptr, gpu_bp.vss_pair_idx.ptr, gpu_bp.vss_dof.ptr, gpu_bp.ss_data.ptr
    };
    CertPtrs cert_p{ d_ntt_off, d_ntt_data, d_nte_off, d_nte_data,
                     d_tbmin, d_tbmax, d_ebmin, d_ebmax };
    PinPtrs pins_p{ gpu_pins.targets.ptr, gpu_pin_map.data.ptr };

    // Launch kernel
    int block = 256;
    int grid  = (nv + block - 1) / block;
    jacobi_predict_kernel<<<grid, block>>>(
        nv, mesh_p, adj_p, bp_p, cert_p, pins_p, gparams,
        d_x, d_xhat,
        d_g, d_H, d_delta, d_box_min, d_box_max);
    cudaDeviceSynchronize();

    // Download results
    std::vector<double> h_g(nv*3), h_H(nv*9), h_delta(nv*3), h_bmin(nv*3), h_bmax(nv*3);
    cudaMemcpy(h_g.data(),     d_g,       nv*3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_H.data(),     d_H,       nv*9*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta.data(), d_delta,   nv*3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bmin.data(),  d_box_min, nv*3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bmax.data(),  d_box_max, nv*3*sizeof(double), cudaMemcpyDeviceToHost);

    // Pack into JacobiPrediction structs
    for(int vi=0;vi<nv;vi++) {
        auto& pred = predictions[vi];
        pred.active = true;
        pred.g      = Vec3(h_g[vi*3], h_g[vi*3+1], h_g[vi*3+2]);
        pred.delta  = Vec3(h_delta[vi*3], h_delta[vi*3+1], h_delta[vi*3+2]);
        pred.certified_region.min = Vec3(h_bmin[vi*3], h_bmin[vi*3+1], h_bmin[vi*3+2]);
        pred.certified_region.max = Vec3(h_bmax[vi*3], h_bmax[vi*3+1], h_bmax[vi*3+2]);
    }

    // Free temporaries
    cudaFree(d_x); cudaFree(d_xhat);
    cudaFree(d_ntt_off); cudaFree(d_ntt_data);
    cudaFree(d_nte_off); cudaFree(d_nte_data);
    cudaFree(d_tbmin); cudaFree(d_tbmax);
    cudaFree(d_ebmin); cudaFree(d_ebmax);
    cudaFree(d_g); cudaFree(d_H); cudaFree(d_delta);
    cudaFree(d_box_min); cudaFree(d_box_max);
}

// ============================================================================
// clip_step_to_certified_region_device
// Scalar port of clip_step_to_certified_region (parallel_helper.cpp).
// Algorithm is identical: walk each axis, clip against the certified AABB
// face the step is about to cross. Returns alpha in [0, 1].
// ============================================================================
__device__ static double clip_step_to_certified_region_device(
    int vi, const double* x, const double delta[3],
    const double box_min[3], const double box_max[3])
{
    double alpha = 1.0;
    for (int k = 0; k < 3; ++k) {
        const double x0 = x[vi*3 + k];
        const double x1 = x0 - delta[k];
        const double lo = box_min[k];
        const double hi = box_max[k];
        if (x1 < lo) {
            const double denom = x0 - x1;
            if (fabs(denom) > 1.0e-16) {
                const double a = (x0 - lo) / denom;
                if (a < alpha) alpha = a;
            }
        } else if (x1 > hi) {
            const double denom = x1 - x0;
            if (fabs(denom) > 1.0e-16) {
                const double a = (hi - x0) / denom;
                if (a < alpha) alpha = a;
            }
        }
    }
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;
    return alpha;
}

// ============================================================================
// colored_gs_commit_kernel — one thread per color-group member.
// Algorithmic port of compute_parallel_commit_for_vertex:
//   if (use_cached_prediction):
//       delta = pred.delta
//       alpha_clip = 1.0
//   else:
//       compute_local_newton_direction_device -> g,H,delta_fresh
//       alpha_clip = clip_step_to_certified_region(x, delta_fresh, pred.region)
//       delta = alpha_clip * delta_fresh
//   ccd_step = 1.0  (CCD not yet on GPU — matches CPU when d_hat == 0)
//   x_after = x[vi] - ccd_step * delta
// ============================================================================
__global__ static void colored_gs_commit_kernel(
    int                ng,
    const int*         group,
    int                use_cached_prediction,
    const double*      pred_delta,    // nv*3
    const double*      pred_bmin,     // nv*3
    const double*      pred_bmax,     // nv*3
    MeshPtrs           mesh,
    AdjPtrs            adj,
    BpPtrs             bp,
    CertPtrs           cert,
    BVHPtrs            bvh,
    PinPtrs            pins,
    GPUSimParams       params,
    const double*      d_x,
    const double*      d_xhat,
    double*            out_delta,       // ng*3
    double*            out_alpha_clip,  // ng
    double*            out_ccd_step,    // ng
    double*            out_x_after)     // ng*3
{
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= ng) return;

    const int vi = group[li];

    double delta[3];
    double alpha_clip = 1.0;

    if (use_cached_prediction) {
        delta[0] = pred_delta[vi*3 + 0];
        delta[1] = pred_delta[vi*3 + 1];
        delta[2] = pred_delta[vi*3 + 2];
    } else {
        double g_fresh[3], H_fresh[9], delta_fresh[3];
        compute_local_newton_direction_device(
            vi, mesh, adj, bp, pins, params, d_x, d_xhat,
            g_fresh, H_fresh, delta_fresh);

        const double bmin[3] = { pred_bmin[vi*3+0], pred_bmin[vi*3+1], pred_bmin[vi*3+2] };
        const double bmax[3] = { pred_bmax[vi*3+0], pred_bmax[vi*3+1], pred_bmax[vi*3+2] };
        alpha_clip = clip_step_to_certified_region_device(vi, d_x, delta_fresh, bmin, bmax);

        delta[0] = alpha_clip * delta_fresh[0];
        delta[1] = alpha_clip * delta_fresh[1];
        delta[2] = alpha_clip * delta_fresh[2];
    }

    // CCD / trust-region safe step. Uses the barrier pair CSR (no BVH query).
    const double ccd_step = compute_safe_step_for_vertex_device(
        vi, d_x, delta, params, mesh, bp);

    out_delta[li*3 + 0] = delta[0];
    out_delta[li*3 + 1] = delta[1];
    out_delta[li*3 + 2] = delta[2];
    out_alpha_clip[li]  = alpha_clip;
    out_ccd_step[li]    = ccd_step;
    out_x_after[li*3+0] = d_x[vi*3+0] - ccd_step * delta[0];
    out_x_after[li*3+1] = d_x[vi*3+1] - ccd_step * delta[1];
    out_x_after[li*3+2] = d_x[vi*3+2] - ccd_step * delta[2];
}

// ============================================================================
// colored_gs_sweep_kernel — fused commit sweep over ALL color groups in one
// kernel launch. Uses a cooperative-groups grid barrier between colors so
// later colors read the in-place updates of earlier ones directly from d_x,
// eliminating per-color launch overhead AND the CPU round-trip that
// apply_parallel_commits previously required.
//
// Launch with cudaLaunchCooperativeKernel so cg::this_grid().sync() works.
// nv threads; each thread handles exactly one vertex, inactive for the
// colors it doesn't belong to. Writes d_x in place (non-conflicting per
// color by coloring construction).
// ============================================================================
__global__ static void colored_gs_sweep_kernel(
    int                nv,
    int                num_colors,
    const int*         vertex_color,     // nv
    const double*      pred_delta,       // nv*3
    const double*      pred_bmin,        // nv*3
    const double*      pred_bmax,        // nv*3
    MeshPtrs           mesh,
    AdjPtrs            adj,
    BpPtrs             bp,
    CertPtrs           cert,
    BVHPtrs            bvh,
    PinPtrs            pins,
    GPUSimParams       params,
    double*            d_x,              // written in place
    const double*      d_xhat)
{
    auto grid = cg::this_grid();
    const int vi = blockIdx.x * blockDim.x + threadIdx.x;
    const int my_color = (vi < nv) ? vertex_color[vi] : -1;

    for (int c = 0; c < num_colors; ++c) {
        if (vi < nv && my_color == c) {
            double delta[3];
            double alpha_clip = 1.0;
            const bool use_cached = (c == 0);

            if (use_cached) {
                delta[0] = pred_delta[vi*3 + 0];
                delta[1] = pred_delta[vi*3 + 1];
                delta[2] = pred_delta[vi*3 + 2];
            } else {
                double g_fresh[3], H_fresh[9], delta_fresh[3];
                compute_local_newton_direction_device(
                    vi, mesh, adj, bp, pins, params, d_x, d_xhat,
                    g_fresh, H_fresh, delta_fresh);

                const double bmin[3] = { pred_bmin[vi*3+0], pred_bmin[vi*3+1], pred_bmin[vi*3+2] };
                const double bmax[3] = { pred_bmax[vi*3+0], pred_bmax[vi*3+1], pred_bmax[vi*3+2] };
                alpha_clip = clip_step_to_certified_region_device(vi, d_x, delta_fresh, bmin, bmax);
                delta[0] = alpha_clip * delta_fresh[0];
                delta[1] = alpha_clip * delta_fresh[1];
                delta[2] = alpha_clip * delta_fresh[2];
            }

            const double ccd_step = compute_safe_step_for_vertex_device(
                vi, d_x, delta, params, mesh, bp);

            // In-place: non-conflicting per color-graph coloring.
            d_x[vi*3+0] -= ccd_step * delta[0];
            d_x[vi*3+1] -= ccd_step * delta[1];
            d_x[vi*3+2] -= ccd_step * delta[2];
        }
        grid.sync();
    }
}

// ============================================================================
// gpu_parallel_commit — host wrapper for colored_gs_commit_kernel.
// Upload-per-call pattern matches gpu_build_jacobi_predictions; persistent
// device buffers are a separate optimization. CCD is not implemented on GPU
// yet, so ccd_step is 1.0 for every commit — equality with CPU holds only
// when d_hat <= 0 (CPU early-returns 1.0 in compute_safe_step_for_vertex).
// ============================================================================
std::vector<ParallelCommit> gpu_parallel_commit(
    const std::vector<int>&              group,
    bool                                 use_cached_prediction,
    const std::vector<JacobiPrediction>& predictions,
    const RefMesh&                       ref_mesh,
    const VertexTriangleMap&             adj_map,
    const std::vector<Pin>&              pins_vec,
    const SimParams&                     params,
    const std::vector<Vec3>&             x,
    const std::vector<Vec3>&             xhat,
    const BroadPhase&                    broad_phase,
    const PinMap*                        pin_map_ptr)
{
    const int ng = static_cast<int>(group.size());
    std::vector<ParallelCommit> commits(ng);
    if (ng == 0) return commits;

    const int nv = static_cast<int>(x.size());
    const auto& bp_cache = broad_phase.cache();

    // ---------------------------------------------------------------------
    // Fast path: session active — static data already resident. Upload only
    // x_current + group[] and launch.
    // ---------------------------------------------------------------------
    if (g_session && g_session->nv == nv) {
        GpuSolverSession& s = *g_session;
        session_reserve_group(s, ng);

        std::vector<double> x_flat(nv*3);
        for (int i = 0; i < nv; ++i) {
            x_flat[i*3+0]=x[i](0); x_flat[i*3+1]=x[i](1); x_flat[i*3+2]=x[i](2);
        }
        cudaMemcpy(s.d_x, x_flat.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(s.d_group, group.data(), ng*sizeof(int), cudaMemcpyHostToDevice);

        MeshPtrs mesh_p{
            s.gpu_mesh.tris.ptr, s.gpu_mesh.Dm_inv.ptr, s.gpu_mesh.area.ptr, s.gpu_mesh.mass.ptr,
            s.gpu_mesh.hinge_v.ptr, s.gpu_mesh.hinge_bar_theta.ptr, s.gpu_mesh.hinge_ce.ptr,
            s.gpu_mesh.hinge_adj_offsets.ptr, s.gpu_mesh.hinge_adj_hi.ptr, s.gpu_mesh.hinge_adj_role.ptr
        };
        AdjPtrs adj_p{ s.gpu_adj.offsets.ptr, s.gpu_adj.tri_idx.ptr, s.gpu_adj.tri_local.ptr };
        BpPtrs  bp_p{
            s.gpu_bp.vnt_offsets.ptr, s.gpu_bp.vnt_pair_idx.ptr, s.gpu_bp.vnt_dof.ptr, s.gpu_bp.nt_data.ptr,
            s.gpu_bp.vss_offsets.ptr, s.gpu_bp.vss_pair_idx.ptr, s.gpu_bp.vss_dof.ptr, s.gpu_bp.ss_data.ptr
        };
        CertPtrs cert_p{ s.d_ntt_off, s.d_ntt_data, s.d_nte_off, s.d_nte_data, s.d_tbmin, s.d_tbmax, s.d_ebmin, s.d_ebmax };
        BVHPtrs  bvh_p { s.d_tri_bvh, s.d_node_bvh, s.d_edge_bvh, s.d_edges, s.tri_root, s.node_root, s.edge_root };
        PinPtrs  pins_p{ s.gpu_pins.targets.ptr, s.gpu_pin_map.data.ptr };

        const int block = 256;
        const int grid  = (ng + block - 1) / block;
        colored_gs_commit_kernel<<<grid, block>>>(
            ng, s.d_group, use_cached_prediction ? 1 : 0,
            s.d_pred_delta, s.d_pred_bmin, s.d_pred_bmax,
            mesh_p, adj_p, bp_p, cert_p, bvh_p, pins_p, s.gparams,
            s.d_x, s.d_xhat,
            s.d_commit_delta, s.d_commit_alpha, s.d_commit_ccd, s.d_commit_xafter);
        cudaError_t ksync = cudaDeviceSynchronize();
        if (ksync != cudaSuccess) {
            fprintf(stderr, "[gpu_parallel_commit session] kernel error: %s (ng=%d cached=%d)\n",
                    cudaGetErrorString(ksync), ng, (int)use_cached_prediction);
        }

        std::vector<double> h_delta(ng*3), h_alpha(ng), h_ccd(ng), h_xafter(ng*3);
        cudaMemcpy(h_delta.data(),  s.d_commit_delta,  ng*3*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_alpha.data(),  s.d_commit_alpha,  ng*sizeof(double),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ccd.data(),    s.d_commit_ccd,    ng*sizeof(double),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_xafter.data(), s.d_commit_xafter, ng*3*sizeof(double), cudaMemcpyDeviceToHost);
        for (int li = 0; li < ng; ++li) {
            auto& c = commits[li];
            c.vi         = group[li];
            c.delta      = Vec3(h_delta[li*3+0], h_delta[li*3+1], h_delta[li*3+2]);
            c.alpha_clip = h_alpha[li];
            c.ccd_step   = h_ccd[li];
            c.x_after    = Vec3(h_xafter[li*3+0], h_xafter[li*3+1], h_xafter[li*3+2]);
            c.valid      = true;
        }
        return commits;
    }

    GPUSimParams gparams = GPUSimParams::from(params);

    // --- Upload x, xhat ---
    std::vector<double> x_flat(nv*3), xhat_flat(nv*3);
    for (int i = 0; i < nv; ++i) {
        x_flat[i*3+0]=x[i](0); x_flat[i*3+1]=x[i](1); x_flat[i*3+2]=x[i](2);
        xhat_flat[i*3+0]=xhat[i](0); xhat_flat[i*3+1]=xhat[i](1); xhat_flat[i*3+2]=xhat[i](2);
    }
    double *d_x = nullptr, *d_xhat = nullptr;
    cudaMalloc(&d_x,    nv*3*sizeof(double));
    cudaMalloc(&d_xhat, nv*3*sizeof(double));
    cudaMemcpy(d_x,    x_flat.data(),    nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xhat, xhat_flat.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);

    // --- Upload mesh / adj / broad-phase / pins (same pattern as predict) ---
    GPURefMesh gpu_mesh;       gpu_mesh.upload(ref_mesh);
    GPUAdjacency gpu_adj;      gpu_adj.upload(adj_map, nv);
    GPUBroadPhaseCache gpu_bp; gpu_bp.upload(bp_cache, nv);
    GPUPins gpu_pins;          gpu_pins.upload(pins_vec);
    GPUPinMap gpu_pin_map;
    if (pin_map_ptr) {
        gpu_pin_map.upload(*pin_map_ptr);
    } else {
        std::vector<int> pm(nv, -1);
        gpu_pin_map.data.upload(pm.data(), nv);
        gpu_pin_map.num_verts = nv;
    }

    // --- Flatten + upload per-vertex predictions (delta + certified AABB) ---
    std::vector<double> pred_delta(nv*3), pred_bmin(nv*3), pred_bmax(nv*3);
    for (int vi = 0; vi < nv; ++vi) {
        pred_delta[vi*3+0] = predictions[vi].delta(0);
        pred_delta[vi*3+1] = predictions[vi].delta(1);
        pred_delta[vi*3+2] = predictions[vi].delta(2);
        pred_bmin[vi*3+0]  = predictions[vi].certified_region.min(0);
        pred_bmin[vi*3+1]  = predictions[vi].certified_region.min(1);
        pred_bmin[vi*3+2]  = predictions[vi].certified_region.min(2);
        pred_bmax[vi*3+0]  = predictions[vi].certified_region.max(0);
        pred_bmax[vi*3+1]  = predictions[vi].certified_region.max(1);
        pred_bmax[vi*3+2]  = predictions[vi].certified_region.max(2);
    }
    double *d_pred_delta, *d_pred_bmin, *d_pred_bmax;
    cudaMalloc(&d_pred_delta, nv*3*sizeof(double));
    cudaMalloc(&d_pred_bmin,  nv*3*sizeof(double));
    cudaMalloc(&d_pred_bmax,  nv*3*sizeof(double));
    cudaMemcpy(d_pred_delta, pred_delta.data(), nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pred_bmin,  pred_bmin.data(),  nv*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pred_bmax,  pred_bmax.data(),  nv*3*sizeof(double), cudaMemcpyHostToDevice);

    // --- node_to_tris / node_to_edges CSR (cert data) ---
    std::vector<int> ntt_off(nv+1, 0), nte_off(nv+1, 0);
    for (int vi = 0; vi < nv; ++vi) {
        if (vi < (int)bp_cache.node_to_tris.size())  ntt_off[vi+1] = (int)bp_cache.node_to_tris[vi].size();
        if (vi < (int)bp_cache.node_to_edges.size()) nte_off[vi+1] = (int)bp_cache.node_to_edges[vi].size();
    }
    for (int vi = 0; vi < nv; ++vi) { ntt_off[vi+1] += ntt_off[vi]; nte_off[vi+1] += nte_off[vi]; }
    std::vector<int> ntt_data(ntt_off[nv]), nte_data(nte_off[nv]);
    for (int vi = 0; vi < nv; ++vi) {
        if (vi < (int)bp_cache.node_to_tris.size()) {
            int pos = ntt_off[vi];
            for (int ti : bp_cache.node_to_tris[vi]) ntt_data[pos++] = ti;
        }
        if (vi < (int)bp_cache.node_to_edges.size()) {
            int pos = nte_off[vi];
            for (int ei : bp_cache.node_to_edges[vi]) nte_data[pos++] = ei;
        }
    }
    auto hmax = [](int a, int b){ return a > b ? a : b; };
    int *d_ntt_off, *d_ntt_data, *d_nte_off, *d_nte_data;
    cudaMalloc(&d_ntt_off,  (nv+1)*sizeof(int));
    cudaMalloc(&d_ntt_data, hmax(1,(int)ntt_data.size())*sizeof(int));
    cudaMalloc(&d_nte_off,  (nv+1)*sizeof(int));
    cudaMalloc(&d_nte_data, hmax(1,(int)nte_data.size())*sizeof(int));
    cudaMemcpy(d_ntt_off, ntt_off.data(), (nv+1)*sizeof(int), cudaMemcpyHostToDevice);
    if (!ntt_data.empty()) cudaMemcpy(d_ntt_data, ntt_data.data(), ntt_data.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nte_off, nte_off.data(), (nv+1)*sizeof(int), cudaMemcpyHostToDevice);
    if (!nte_data.empty()) cudaMemcpy(d_nte_data, nte_data.data(), nte_data.size()*sizeof(int), cudaMemcpyHostToDevice);

    // --- Flatten + upload BVH nodes (tri / node / edge) + edges ---
    auto upload_bvh = [&](const std::vector<BVHNode>& src, BVHNodeGPU*& d_out) -> int {
        const int n = (int)src.size();
        const int cap = hmax(1, n);
        std::vector<BVHNodeGPU> flat(cap);
        for (int i = 0; i < n; ++i) {
            flat[i].bmin[0] = src[i].bbox.min(0); flat[i].bmin[1] = src[i].bbox.min(1); flat[i].bmin[2] = src[i].bbox.min(2);
            flat[i].bmax[0] = src[i].bbox.max(0); flat[i].bmax[1] = src[i].bbox.max(1); flat[i].bmax[2] = src[i].bbox.max(2);
            flat[i].left      = src[i].left;
            flat[i].right     = src[i].right;
            flat[i].leafIndex = src[i].leafIndex;
            flat[i].pad       = 0;
        }
        cudaMalloc(&d_out, cap * sizeof(BVHNodeGPU));
        cudaMemcpy(d_out, flat.data(), cap * sizeof(BVHNodeGPU), cudaMemcpyHostToDevice);
        return n;
    };
    BVHNodeGPU *d_tri_bvh = nullptr, *d_node_bvh = nullptr, *d_edge_bvh = nullptr;
    upload_bvh(bp_cache.tri_bvh_nodes,  d_tri_bvh);
    upload_bvh(bp_cache.node_bvh_nodes, d_node_bvh);
    upload_bvh(bp_cache.edge_bvh_nodes, d_edge_bvh);

    const int num_edges = (int)bp_cache.edges.size();
    std::vector<int> edges_flat(hmax(1, num_edges) * 2, 0);
    for (int i = 0; i < num_edges; ++i) {
        edges_flat[i*2+0] = bp_cache.edges[i][0];
        edges_flat[i*2+1] = bp_cache.edges[i][1];
    }
    int* d_edges = nullptr;
    cudaMalloc(&d_edges, hmax(1, num_edges) * 2 * sizeof(int));
    cudaMemcpy(d_edges, edges_flat.data(), hmax(1, num_edges) * 2 * sizeof(int), cudaMemcpyHostToDevice);

    // tri_boxes / edge_boxes flat buffers for CertPtrs (reused from predict pattern).
    const int num_tris_bp  = (int)bp_cache.tri_boxes.size();
    const int num_edges_bp = (int)bp_cache.edge_boxes.size();
    std::vector<double> tri_bmin_f(hmax(1, num_tris_bp)*3), tri_bmax_f(hmax(1, num_tris_bp)*3);
    std::vector<double> edge_bmin_f(hmax(1, num_edges_bp)*3), edge_bmax_f(hmax(1, num_edges_bp)*3);
    for (int i = 0; i < num_tris_bp; ++i) {
        tri_bmin_f[i*3+0]=bp_cache.tri_boxes[i].min(0); tri_bmin_f[i*3+1]=bp_cache.tri_boxes[i].min(1); tri_bmin_f[i*3+2]=bp_cache.tri_boxes[i].min(2);
        tri_bmax_f[i*3+0]=bp_cache.tri_boxes[i].max(0); tri_bmax_f[i*3+1]=bp_cache.tri_boxes[i].max(1); tri_bmax_f[i*3+2]=bp_cache.tri_boxes[i].max(2);
    }
    for (int i = 0; i < num_edges_bp; ++i) {
        edge_bmin_f[i*3+0]=bp_cache.edge_boxes[i].min(0); edge_bmin_f[i*3+1]=bp_cache.edge_boxes[i].min(1); edge_bmin_f[i*3+2]=bp_cache.edge_boxes[i].min(2);
        edge_bmax_f[i*3+0]=bp_cache.edge_boxes[i].max(0); edge_bmax_f[i*3+1]=bp_cache.edge_boxes[i].max(1); edge_bmax_f[i*3+2]=bp_cache.edge_boxes[i].max(2);
    }
    double *d_tbmin, *d_tbmax, *d_ebmin, *d_ebmax;
    cudaMalloc(&d_tbmin, tri_bmin_f.size()*sizeof(double));
    cudaMalloc(&d_tbmax, tri_bmax_f.size()*sizeof(double));
    cudaMalloc(&d_ebmin, edge_bmin_f.size()*sizeof(double));
    cudaMalloc(&d_ebmax, edge_bmax_f.size()*sizeof(double));
    cudaMemcpy(d_tbmin, tri_bmin_f.data(), tri_bmin_f.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tbmax, tri_bmax_f.data(), tri_bmax_f.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ebmin, edge_bmin_f.data(), edge_bmin_f.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ebmax, edge_bmax_f.data(), edge_bmax_f.size()*sizeof(double), cudaMemcpyHostToDevice);

    // --- Upload group ---
    int* d_group = nullptr;
    cudaMalloc(&d_group, ng*sizeof(int));
    cudaMemcpy(d_group, group.data(), ng*sizeof(int), cudaMemcpyHostToDevice);

    // --- Output buffers ---
    double *d_out_delta = nullptr, *d_out_alpha = nullptr, *d_out_ccd = nullptr, *d_out_xafter = nullptr;
    cudaMalloc(&d_out_delta,  ng*3*sizeof(double));
    cudaMalloc(&d_out_alpha,  ng*sizeof(double));
    cudaMalloc(&d_out_ccd,    ng*sizeof(double));
    cudaMalloc(&d_out_xafter, ng*3*sizeof(double));

    MeshPtrs mesh_p{
        gpu_mesh.tris.ptr, gpu_mesh.Dm_inv.ptr, gpu_mesh.area.ptr, gpu_mesh.mass.ptr,
        gpu_mesh.hinge_v.ptr, gpu_mesh.hinge_bar_theta.ptr, gpu_mesh.hinge_ce.ptr,
        gpu_mesh.hinge_adj_offsets.ptr, gpu_mesh.hinge_adj_hi.ptr, gpu_mesh.hinge_adj_role.ptr
    };
    AdjPtrs adj_p{ gpu_adj.offsets.ptr, gpu_adj.tri_idx.ptr, gpu_adj.tri_local.ptr };
    BpPtrs  bp_p{
        gpu_bp.vnt_offsets.ptr, gpu_bp.vnt_pair_idx.ptr, gpu_bp.vnt_dof.ptr, gpu_bp.nt_data.ptr,
        gpu_bp.vss_offsets.ptr, gpu_bp.vss_pair_idx.ptr, gpu_bp.vss_dof.ptr, gpu_bp.ss_data.ptr
    };
    CertPtrs cert_p{ d_ntt_off, d_ntt_data, d_nte_off, d_nte_data, d_tbmin, d_tbmax, d_ebmin, d_ebmax };
    BVHPtrs  bvh_p {
        d_tri_bvh, d_node_bvh, d_edge_bvh, d_edges,
        bp_cache.tri_root, bp_cache.node_root, bp_cache.edge_root
    };
    PinPtrs pins_p{ gpu_pins.targets.ptr, gpu_pin_map.data.ptr };

    const int block = 256;
    const int grid  = (ng + block - 1) / block;
    colored_gs_commit_kernel<<<grid, block>>>(
        ng, d_group, use_cached_prediction ? 1 : 0,
        d_pred_delta, d_pred_bmin, d_pred_bmax,
        mesh_p, adj_p, bp_p, cert_p, bvh_p, pins_p, gparams,
        d_x, d_xhat,
        d_out_delta, d_out_alpha, d_out_ccd, d_out_xafter);
    cudaError_t ksync = cudaDeviceSynchronize();
    if (ksync != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel_commit] kernel error: %s  (ng=%d, cached=%d)\n",
                cudaGetErrorString(ksync), ng, (int)use_cached_prediction);
    }

    // --- Download results ---
    std::vector<double> h_delta(ng*3), h_alpha(ng), h_ccd(ng), h_xafter(ng*3);
    cudaMemcpy(h_delta.data(),  d_out_delta,  ng*3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alpha.data(),  d_out_alpha,  ng*sizeof(double),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ccd.data(),    d_out_ccd,    ng*sizeof(double),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xafter.data(), d_out_xafter, ng*3*sizeof(double), cudaMemcpyDeviceToHost);

    for (int li = 0; li < ng; ++li) {
        auto& c = commits[li];
        c.vi         = group[li];
        c.delta      = Vec3(h_delta[li*3+0], h_delta[li*3+1], h_delta[li*3+2]);
        c.alpha_clip = h_alpha[li];
        c.ccd_step   = h_ccd[li];
        c.x_after    = Vec3(h_xafter[li*3+0], h_xafter[li*3+1], h_xafter[li*3+2]);
        c.valid      = true;
    }

    cudaFree(d_x); cudaFree(d_xhat);
    cudaFree(d_pred_delta); cudaFree(d_pred_bmin); cudaFree(d_pred_bmax);
    cudaFree(d_ntt_off); cudaFree(d_ntt_data); cudaFree(d_nte_off); cudaFree(d_nte_data);
    cudaFree(d_tri_bvh); cudaFree(d_node_bvh); cudaFree(d_edge_bvh); cudaFree(d_edges);
    cudaFree(d_tbmin); cudaFree(d_tbmax); cudaFree(d_ebmin); cudaFree(d_ebmax);
    cudaFree(d_group);
    cudaFree(d_out_delta); cudaFree(d_out_alpha); cudaFree(d_out_ccd); cudaFree(d_out_xafter);

    return commits;
}
