// ============================================================================
// gpu_elastic.cu — minimal no-collision GPU solver for cloth / shells.
//
// Corotated StVK elasticity + bending + pin + inertia. Everything in
// GPU_Sim/gpu_solver.cu that's barrier/CCD related is omitted.
//
// Precision: FP32 throughout the device path. Consumer Ada GPUs run FP64 at
// 1/64 the FP32 rate, so keeping the Newton direction computation (SVD,
// matrix inverse, Hessian PSD proj) in double was dominating the runtime.
// Host-side still exchanges `Vec3` (double) — we convert at the boundaries.
// Storage: per-vertex scratch + a float mirror of the time-invariant mesh
// arrays (Dm_inv, area, mass, hinge_bar_theta, hinge_ce) live in FP32 on
// device; the shared GPURefMesh/GPUAdjacency/GPUPins uploads still run in
// double (so collision code can share the same infra if reused later), and
// we just D2H→convert→H2D once in init to populate the float mirrors.
//
// Lifecycle per simulation:
//   gpu_elastic_init(mesh, adj, pins, params)
//   for frame: for substep:
//       gpu_elastic_set_pin_targets(pins)
//       gpu_elastic_run_substep(x, xhat, max_iters, x_out)
//       residual = gpu_elastic_last_residual()
//   gpu_elastic_shutdown()
// ============================================================================

#include "gpu_elastic.h"
#include "gpu_mesh.h"              // GPURefMesh, GPUAdjacency, GPUPins, GPUPinMap
#include "../parallel_helper.h"    // build_vertex_adjacency_map, greedy_color
#include "../make_shape.h"         // build_pin_map
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include <vector>

using Real = float;

// ============================================================================
// Device math helpers (FP32).
// ============================================================================
__device__ static Real dev_max2(Real a, Real b) { return a > b ? a : b; }
__device__ static Real dot3(const Real a[3], const Real b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
__device__ static Real norm3(const Real v[3]) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}
__device__ static void cross3(const Real a[3], const Real b[3], Real o[3]) {
    o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0];
}
__device__ static void sub3(const Real a[3], const Real b[3], Real o[3]) {
    o[0]=a[0]-b[0]; o[1]=a[1]-b[1]; o[2]=a[2]-b[2];
}
__device__ static void scale3(Real s, const Real v[3], Real o[3]) {
    o[0]=s*v[0]; o[1]=s*v[1]; o[2]=s*v[2];
}
__device__ static void addscale3(Real s, const Real v[3], Real acc[3]) {
    acc[0]+=s*v[0]; acc[1]+=s*v[1]; acc[2]+=s*v[2];
}
__device__ static void loadv(const Real* x, int i, Real v[3]) {
    v[0]=x[i*3]; v[1]=x[i*3+1]; v[2]=x[i*3+2];
}
__device__ static void mat33_inverse(const Real H[9], Real inv[9]) {
    Real det = H[0]*(H[4]*H[8]-H[5]*H[7]) - H[1]*(H[3]*H[8]-H[5]*H[6]) + H[2]*(H[3]*H[7]-H[4]*H[6]);
    Real id = 1.0f/det;
    inv[0]= id*(H[4]*H[8]-H[5]*H[7]); inv[1]=-id*(H[1]*H[8]-H[2]*H[7]); inv[2]= id*(H[1]*H[5]-H[2]*H[4]);
    inv[3]=-id*(H[3]*H[8]-H[5]*H[6]); inv[4]= id*(H[0]*H[8]-H[2]*H[6]); inv[5]=-id*(H[0]*H[5]-H[2]*H[3]);
    inv[6]= id*(H[3]*H[7]-H[4]*H[6]); inv[7]=-id*(H[0]*H[7]-H[1]*H[6]); inv[8]= id*(H[0]*H[4]-H[1]*H[3]);
}
__device__ static void outer3_add(Real s, const Real a[3], Real H[9]) {
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) H[r*3+c] += s*a[r]*a[c];
}
__device__ static void adddiag3(Real s, Real H[9]) { H[0]+=s; H[4]+=s; H[8]+=s; }
__device__ static void matvec3(const Real M[9], const Real v[3], Real o[3]) {
    o[0]=M[0]*v[0]+M[1]*v[1]+M[2]*v[2];
    o[1]=M[3]*v[0]+M[4]*v[1]+M[5]*v[2];
    o[2]=M[6]*v[0]+M[7]*v[1]+M[8]*v[2];
}

// atomicMax for non-negative floats via bitwise CAS.
__device__ static float atomicMaxFloat(float* addr, float val) {
    unsigned int* p = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *p, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(p, assumed, __float_as_uint(val));
    } while (assumed != old);
    return __uint_as_float(old);
}

// ============================================================================
// Plain StVK elastic (per-triangle, per-vertex contribution).
// Ported from Newton's evaluate_stvk_force_hessian. Green strain G = ½(FᵀF−I),
// psi = μ·||G||_F² + ½·λ·tr(G)². No rotation extraction → no sqrtf / 1/s in
// the numerics, so no catastrophic cancellation under large stretch. Less
// robust than corotated for large compression (J → 0), but cloth rarely
// compresses.
// ============================================================================
__device__ static void stvk_vertex_gh(
    int a,
    const Real* x_v0, const Real* x_v1, const Real* x_v2,
    const Real* xp_v0, const Real* xp_v1, const Real* xp_v2,
    const Real* dm, Real ref_area,
    Real mu, Real lambda, Real dt2, Real dt, Real damping,
    Real g_out[3], Real H_out[9])
{
    // Edge vectors and F columns.
    Real x01[3], x02[3];
    sub3(x_v1, x_v0, x01);
    sub3(x_v2, x_v0, x02);
    const Real Dm00 = dm[0], Dm01 = dm[1], Dm10 = dm[2], Dm11 = dm[3];
    Real f0[3] = { x01[0]*Dm00 + x02[0]*Dm10,
                   x01[1]*Dm00 + x02[1]*Dm10,
                   x01[2]*Dm00 + x02[2]*Dm10 };
    Real f1[3] = { x01[0]*Dm01 + x02[0]*Dm11,
                   x01[1]*Dm01 + x02[1]*Dm11,
                   x01[2]*Dm01 + x02[2]*Dm11 };
    const Real f0_dot_f0 = dot3(f0, f0);
    const Real f1_dot_f1 = dot3(f1, f1);
    const Real f0_dot_f1 = dot3(f0, f1);

    // Green strain (2x2 symmetric).
    const Real G00 = Real(0.5) * (f0_dot_f0 - Real(1));
    const Real G11 = Real(0.5) * (f1_dot_f1 - Real(1));
    const Real G01 = Real(0.5) * f0_dot_f1;
    const Real trace_G = G00 + G11;

    // Degenerate-triangle guard (matches Newton).
    const Real GF2 = G00*G00 + G11*G11 + Real(2)*G01*G01;
    if (GF2 < Real(1e-20)) return;

    // PK1 stress (3x2 as two 3-vectors).
    const Real twoMuG00 = Real(2)*mu*G00;
    const Real twoMuG11 = Real(2)*mu*G11;
    const Real twoMuG01 = Real(2)*mu*G01;
    const Real lamTr    = lambda * trace_G;
    const Real PK1_c0[3] = {
        f0[0]*(twoMuG00 + lamTr) + f1[0]*twoMuG01,
        f0[1]*(twoMuG00 + lamTr) + f1[1]*twoMuG01,
        f0[2]*(twoMuG00 + lamTr) + f1[2]*twoMuG01,
    };
    const Real PK1_c1[3] = {
        f0[0]*twoMuG01 + f1[0]*(twoMuG11 + lamTr),
        f0[1]*twoMuG01 + f1[1]*(twoMuG11 + lamTr),
        f0[2]*twoMuG01 + f1[2]*(twoMuG11 + lamTr),
    };

    // Vertex-order masks (v_order == a).
    const Real m0 = (a == 0) ? Real(1) : Real(0);
    const Real m1 = (a == 1) ? Real(1) : Real(0);
    const Real m2 = (a == 2) ? Real(1) : Real(0);

    // dF/dx for this vertex (scalars multiplying identity blocks).
    const Real df0_dx = Dm00 * (m1 - m0) + Dm10 * (m2 - m0);
    const Real df1_dx = Dm01 * (m1 - m0) + Dm11 * (m2 - m0);

    // Force = −PK1·(dF/dx); our g accumulates +gradient so add +PK1·(dF/dx).
    const Real dg_scale = dt2 * ref_area;
    g_out[0] += dg_scale * (PK1_c0[0] * df0_dx + PK1_c1[0] * df1_dx);
    g_out[1] += dg_scale * (PK1_c0[1] * df0_dx + PK1_c1[1] * df1_dx);
    g_out[2] += dg_scale * (PK1_c0[2] * df0_dx + PK1_c1[2] * df1_dx);

    // Hessian via Cauchy-Green invariants (per Newton).
    const Real Ic = f0_dot_f0 + f1_dot_f1;
    const Real two_dpsi_dIc = -mu + (Real(0.5) * Ic - Real(1)) * lambda;

    // 3×3 outer products as flat 9 arrays (row-major).
    auto add_outer = [&](Real s, const Real* a_, const Real* b_, Real out[9]) {
        out[0] += s*a_[0]*b_[0]; out[1] += s*a_[0]*b_[1]; out[2] += s*a_[0]*b_[2];
        out[3] += s*a_[1]*b_[0]; out[4] += s*a_[1]*b_[1]; out[5] += s*a_[1]*b_[2];
        out[6] += s*a_[2]*b_[0]; out[7] += s*a_[2]*b_[1]; out[8] += s*a_[2]*b_[2];
    };
    auto add_scaled_I = [&](Real s, Real out[9]) { out[0] += s; out[4] += s; out[8] += s; };

    // Assemble d²E/dF² block-by-block then contract.
    // H = df0_dx² · D00 + df1_dx² · D11 + df0·df1 · (D01 + D01ᵀ)
    const Real A00 = df0_dx * df0_dx;
    const Real A11 = df1_dx * df1_dx;
    const Real A01 = df0_dx * df1_dx;

    Real block[9] = {0,0,0, 0,0,0, 0,0,0};

    // Contribution from D00 = λ·f0⊗f0 + two_dpsi_dIc·I + μ·(f0·f0·I + 2·f0⊗f0 + f1⊗f1)
    {
        const Real coef_ff0_f0 = A00 * (lambda + Real(2)*mu);
        const Real coef_I      = A00 * (two_dpsi_dIc + mu * f0_dot_f0);
        const Real coef_ff1_f1 = A00 * mu;
        add_outer(coef_ff0_f0, f0, f0, block);
        add_scaled_I(coef_I, block);
        add_outer(coef_ff1_f1, f1, f1, block);
    }
    // D11 = λ·f1⊗f1 + two_dpsi_dIc·I + μ·(f1·f1·I + 2·f1⊗f1 + f0⊗f0)
    {
        const Real coef_ff1_f1 = A11 * (lambda + Real(2)*mu);
        const Real coef_I      = A11 * (two_dpsi_dIc + mu * f1_dot_f1);
        const Real coef_ff0_f0 = A11 * mu;
        add_outer(coef_ff1_f1, f1, f1, block);
        add_scaled_I(coef_I, block);
        add_outer(coef_ff0_f0, f0, f0, block);
    }
    // D01 + D01ᵀ = λ·(f0⊗f1 + f1⊗f0) + μ·(f0·f1·2·I + f1⊗f0 + f0⊗f1)
    // (H_IIc01 = μ·(f0·f1·I + f1⊗f0); adding transpose gives 2·I·μ·f0·f1 + μ·(f1⊗f0+f0⊗f1))
    {
        const Real scale = A01;
        const Real coef_I     = scale * (Real(2) * mu * f0_dot_f1);
        const Real coef_01 = scale * (lambda + mu);
        const Real coef_10 = scale * (lambda + mu);
        add_scaled_I(coef_I, block);
        add_outer(coef_01, f0, f1, block);
        add_outer(coef_10, f1, f0, block);
    }

    const Real h_scale = dt2 * ref_area;
    #pragma unroll
    for (int i = 0; i < 9; ++i) H_out[i] += h_scale * block[i];

    // ------------------------------------------------------------------
    // Per-constraint Rayleigh damping (Newton's recipe).
    // Cμ = ||G||_F,  Cλ = tr(G); damp each separately.
    //   f_d  = -k · (dC/dt) · (dC/dx)           (contributes to −grad, so our g gets +sign)
    //   H_d  = +k / dt · (dC/dx) ⊗ (dC/dx)
    // Both scaled by dt²·ref_area like the elastic terms. In our g
    // convention g = +grad(E), so the f_d sign flips to + for g.
    // ------------------------------------------------------------------
    if (damping > Real(0)) {
        // Previous-step edges and F-columns.
        Real x01p[3], x02p[3];
        sub3(xp_v1, xp_v0, x01p);
        sub3(xp_v2, xp_v0, x02p);
        const Real f0p[3] = { x01p[0]*Dm00 + x02p[0]*Dm10,
                              x01p[1]*Dm00 + x02p[1]*Dm10,
                              x01p[2]*Dm00 + x02p[2]*Dm10 };
        const Real f1p[3] = { x01p[0]*Dm01 + x02p[0]*Dm11,
                              x01p[1]*Dm01 + x02p[1]*Dm11,
                              x01p[2]*Dm01 + x02p[2]*Dm11 };
        const Real inv_dt = Real(1) / dt;
        const Real df0dt[3] = { (f0[0] - f0p[0]) * inv_dt,
                                (f0[1] - f0p[1]) * inv_dt,
                                (f0[2] - f0p[2]) * inv_dt };
        const Real df1dt[3] = { (f1[0] - f1p[0]) * inv_dt,
                                (f1[1] - f1p[1]) * inv_dt,
                                (f1[2] - f1p[2]) * inv_dt };

        // dG/dt components.
        const Real dG00_dt = dot3(f0, df0dt);
        const Real dG11_dt = dot3(f1, df1dt);
        const Real dG01_dt = Real(0.5) * (dot3(f0, df1dt) + dot3(f1, df0dt));

        // Cλ = tr(G); dCλ/dt = dG00 + dG11; dCλ/dF = F; dCλ/dx = f0·df0_dx + f1·df1_dx.
        const Real dClmbd_dt = dG00_dt + dG11_dt;
        const Real dClmbd_dx[3] = {
            f0[0] * df0_dx + f1[0] * df1_dx,
            f0[1] * df0_dx + f1[1] * df1_dx,
            f0[2] * df0_dx + f1[2] * df1_dx
        };
        const Real kd_lmbd = lambda * damping;
        // g += dt²·area · kd_lmbd · dCλ/dt · dCλ/dx  (Newton's force is −; our grad gets +)
        const Real scale_l_g = h_scale * kd_lmbd * dClmbd_dt;
        g_out[0] += scale_l_g * dClmbd_dx[0];
        g_out[1] += scale_l_g * dClmbd_dx[1];
        g_out[2] += scale_l_g * dClmbd_dx[2];
        // H += dt²·area · kd_lmbd/dt · (dCλ/dx) ⊗ (dCλ/dx)
        const Real scale_l_h = h_scale * kd_lmbd * inv_dt;
        H_out[0] += scale_l_h * dClmbd_dx[0]*dClmbd_dx[0];
        H_out[1] += scale_l_h * dClmbd_dx[0]*dClmbd_dx[1];
        H_out[2] += scale_l_h * dClmbd_dx[0]*dClmbd_dx[2];
        H_out[3] += scale_l_h * dClmbd_dx[1]*dClmbd_dx[0];
        H_out[4] += scale_l_h * dClmbd_dx[1]*dClmbd_dx[1];
        H_out[5] += scale_l_h * dClmbd_dx[1]*dClmbd_dx[2];
        H_out[6] += scale_l_h * dClmbd_dx[2]*dClmbd_dx[0];
        H_out[7] += scale_l_h * dClmbd_dx[2]*dClmbd_dx[1];
        H_out[8] += scale_l_h * dClmbd_dx[2]*dClmbd_dx[2];

        // Cμ = ||G||_F. Skip when G≈0 (undeformed).
        const Real Cmu_sq = GF2;   // computed earlier: G00²+G11²+2·G01²
        if (Cmu_sq > Real(1e-20)) {
            const Real Cmu = sqrt(Cmu_sq);
            const Real invCmu = Real(1) / Cmu;
            const Real G00n = G00 * invCmu;
            const Real G11n = G11 * invCmu;
            const Real G01n = G01 * invCmu;
            const Real dCmu_dt = G00n*dG00_dt + G11n*dG11_dt + Real(2)*G01n*dG01_dt;
            // dCμ/dF col0 = G00n·f0 + G01n·f1;  col1 = G01n·f0 + G11n·f1.
            const Real dCmuF0[3] = { G00n*f0[0] + G01n*f1[0],
                                     G00n*f0[1] + G01n*f1[1],
                                     G00n*f0[2] + G01n*f1[2] };
            const Real dCmuF1[3] = { G01n*f0[0] + G11n*f1[0],
                                     G01n*f0[1] + G11n*f1[1],
                                     G01n*f0[2] + G11n*f1[2] };
            const Real dCmu_dx[3] = {
                dCmuF0[0]*df0_dx + dCmuF1[0]*df1_dx,
                dCmuF0[1]*df0_dx + dCmuF1[1]*df1_dx,
                dCmuF0[2]*df0_dx + dCmuF1[2]*df1_dx
            };
            const Real kd_mu = mu * damping;
            const Real scale_m_g = h_scale * kd_mu * dCmu_dt;
            g_out[0] += scale_m_g * dCmu_dx[0];
            g_out[1] += scale_m_g * dCmu_dx[1];
            g_out[2] += scale_m_g * dCmu_dx[2];
            const Real scale_m_h = h_scale * kd_mu * inv_dt;
            H_out[0] += scale_m_h * dCmu_dx[0]*dCmu_dx[0];
            H_out[1] += scale_m_h * dCmu_dx[0]*dCmu_dx[1];
            H_out[2] += scale_m_h * dCmu_dx[0]*dCmu_dx[2];
            H_out[3] += scale_m_h * dCmu_dx[1]*dCmu_dx[0];
            H_out[4] += scale_m_h * dCmu_dx[1]*dCmu_dx[1];
            H_out[5] += scale_m_h * dCmu_dx[1]*dCmu_dx[2];
            H_out[6] += scale_m_h * dCmu_dx[2]*dCmu_dx[0];
            H_out[7] += scale_m_h * dCmu_dx[2]*dCmu_dx[1];
            H_out[8] += scale_m_h * dCmu_dx[2]*dCmu_dx[2];
        }
    }
}

// ============================================================================
// Corotated elastic (per-triangle, per-vertex contribution)
// ============================================================================
__device__ static void corotated_vertex_gh(
    int a,
    const Real* x_v0, const Real* x_v1, const Real* x_v2,
    const Real* dm, Real ref_area,
    Real mu, Real lambda, Real dt2,
    Real g[3], Real H[9])
{
    Real Ds0[3], Ds1[3];
    sub3(x_v1, x_v0, Ds0);
    sub3(x_v2, x_v0, Ds1);

    Real F[6];
    for (int i=0;i<3;++i) {
        F[i*2+0] = Ds0[i]*dm[0] + Ds1[i]*dm[1];
        F[i*2+1] = Ds0[i]*dm[2] + Ds1[i]*dm[3];
    }

    Real c00=0,c01=0,c11=0;
    for (int k=0;k<3;++k) {
        c00 += F[k*2]*F[k*2];
        c01 += F[k*2]*F[k*2+1];
        c11 += F[k*2+1]*F[k*2+1];
    }

    Real d_half=(c00-c11)*0.5f, disc=sqrt(d_half*d_half+c01*c01);
    Real lam0=dev_max2((c00+c11)*0.5f-disc,1e-20f);
    Real lam1=dev_max2((c00+c11)*0.5f+disc,1e-20f);

    Real ex=-c01, ey=d_half+disc;
    Real elen=sqrt(ex*ex+ey*ey);
    if (elen<1e-12f) { ex=1.0f; ey=0.0f; } else { ex/=elen; ey/=elen; }

    Real s0=sqrt(lam0), s1=sqrt(lam1), J=s0*s1, traceS=s0+s1;
    Real si0=1.0f/s0, si1=1.0f/s1;
    Real SInv00=ex*ex*si0+ey*ey*si1, SInv01=ex*ey*(si0-si1), SInv11=ey*ey*si0+ex*ex*si1;

    Real R[6];
    for (int i=0;i<3;++i) {
        R[i*2+0] = F[i*2+0]*SInv00 + F[i*2+1]*SInv01;
        R[i*2+1] = F[i*2+0]*SInv01 + F[i*2+1]*SInv11;
    }

    Real det_C=c00*c11-c01*c01;
    Real idet=(det_C>1e-24f)?1.0f/det_C:0.0f;
    Real FTFinv00=c11*idet, FTFinv01=-c01*idet, FTFinv11=c00*idet;

    Real FFTFInv[6];
    for (int i=0;i<3;++i) {
        FFTFInv[i*2+0] = F[i*2+0]*FTFinv00 + F[i*2+1]*FTFinv01;
        FFTFInv[i*2+1] = F[i*2+0]*FTFinv01 + F[i*2+1]*FTFinv11;
    }

    Real lJ1J = lambda*(J-1.0f)*J;
    Real P[6];
    for (int i=0;i<6;++i) P[i] = 2.0f*mu*(F[i]-R[i]) + lJ1J*FFTFInv[i];

    Real gradN[3][2];
    gradN[1][0]=dm[0]; gradN[1][1]=dm[2];
    gradN[2][0]=dm[1]; gradN[2][1]=dm[3];
    gradN[0][0]=-gradN[1][0]-gradN[2][0];
    gradN[0][1]=-gradN[1][1]-gradN[2][1];

    Real dg_scale = dt2*ref_area;
    for (int gamma=0;gamma<3;++gamma) {
        g[gamma] += dg_scale*(P[gamma*2+0]*gradN[a][0] + P[gamma*2+1]*gradN[a][1]);
    }

    Real RRT[9]={};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        RRT[i*3+j] = R[i*2+0]*R[j*2+0] + R[i*2+1]*R[j*2+1];

    Real dcdF[6];
    for (int m=0;m<3;++m) {
        dcdF[2*m+0] = -R[m*2+1] / traceS;
        dcdF[2*m+1] =  R[m*2+0] / traceS;
    }

    Real FFTFInvFT[9];
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        FFTFInvFT[i*3+j] = FFTFInv[i*2+0]*F[j*2+0] + FFTFInv[i*2+1]*F[j*2+1];

    // Hessian contribution: instead of materializing dRdF[36] and dPdF[36]
    // (72 floats → register pressure), compute each scalar on the fly inside
    // the 4-nested (gamma,delta,beta,eta) projection loop. Each dPdF[...] is
    // used exactly once (contracted against gradN[a][beta]*gradN[a][eta]), so
    // no recomputation — pure register-footprint win.
    Real coef2 = 0.5f*lambda*(2.0f*J-1.0f)*J;
    for (int gamma=0;gamma<3;++gamma) {
        for (int delta=0;delta<3;++delta) {
            Real val=0.0f;
            for (int beta=0;beta<2;++beta) for (int eta=0;eta<2;++eta) {
                // c1=gamma*2+beta, c2=delta*2+eta → m=gamma, n=beta, ii=delta, j=eta.
                const int m=gamma, n=beta, ii=delta, j=eta;

                // dPdF scalar.
                Real FTFinv_jn = (j==0&&n==0)?FTFinv00:(j==1&&n==1)?FTFinv11:FTFinv01;
                Real dPdF_val = 0.0f;
                if (m==ii) dPdF_val += lJ1J*FTFinv_jn;
                dPdF_val -= lJ1J*(FFTFInv[m*2+j]*FFTFInv[ii*2+n] + FFTFInvFT[m*3+ii]*FTFinv_jn);
                dPdF_val += coef2*(FFTFInv[ii*2+j]*FFTFInv[m*2+n] + FFTFInv[ii*2+j]*FFTFInv[m*2+n]);
                if (gamma==delta && beta==eta) dPdF_val += 2.0f*mu; // diagonal c1==c2

                // dRdF scalar, then subtract 2*mu * dRdF.
                Real SInv_jn = FTFinv_jn; // SAME index pattern as FTFinv_jn above, but different value
                SInv_jn = (j==0&&n==0)?SInv00:(j==1&&n==1)?SInv11:SInv01;
                Real dRdF_val = 0.0f;
                if (m==ii) dRdF_val += SInv_jn;
                dRdF_val -= RRT[m*3+ii]*SInv_jn;
                Real Re_mn = (n==0) ? R[m*2+1] : -R[m*2+0];
                Real dcdF_c2 = (j==0) ? -R[ii*2+1]/traceS : R[ii*2+0]/traceS;
                dRdF_val -= dcdF_c2 * Re_mn;
                dPdF_val -= 2.0f*mu*dRdF_val;

                val += dPdF_val * gradN[a][beta] * gradN[a][eta];
            }
            H[gamma*3+delta] += dg_scale*val;
        }
    }
}

// ============================================================================
// Bending (per-hinge, per-vertex contribution, PSD-projected Hessian)
// ============================================================================
__device__ static void bending_vertex_gh_psd(
    int role,
    const Real* hv0, const Real* hv1, const Real* hv2, const Real* hv3,
    Real kB, Real ce, Real bar_theta, Real dt2,
    Real g[3], Real H[9])
{
    Real e[3], a[3], b[3];
    sub3(hv1, hv0, e); sub3(hv2, hv0, a); sub3(hv3, hv0, b);

    Real mA[3], mB[3];
    cross3(e, a, mA); cross3(b, e, mB);

    Real muA2=dot3(mA,mA), muB2=dot3(mB,mB), ell=norm3(e);
    if (ell<1e-12f || muA2<1e-22f || muB2<1e-22f) return;

    Real e_hat[3]; scale3(1.0f/ell, e, e_hat);

    Real X = dot3(mA, mB);
    Real mAxmB[3]; cross3(mA, mB, mAxmB);
    Real Y = dot3(mAxmB, e_hat);
    Real theta = atan2f(Y, X);

    Real dXn[3]={0,0,0}, dYn[3]={0,0,0};
    switch (role) {
        case 0: {
            Real cA[3], cB[3];
            sub3(hv2, hv1, cA); sub3(hv3, hv1, cB);
            Real t1[3], t2[3]; cross3(mB, cA, t1); cross3(mA, cB, t2);
            sub3(t1, t2, dXn);
            Real coef = dot3(cA, mB) + dot3(mA, cB);
            Real tmp1[3], tmp2[3];
            scale3(coef, e_hat, dYn);
            scale3(-dot3(e_hat, cA), mB, tmp1);
            scale3(-dot3(e_hat, cB), mA, tmp2);
            for (int k=0;k<3;++k) dYn[k] += tmp1[k] + tmp2[k];
            break;
        }
        case 1: {
            Real t1[3], t2[3];
            cross3(mA, b, t1); cross3(mB, a, t2);
            sub3(t1, t2, dXn);
            Real coef = -(dot3(a, mB) + dot3(mA, b));
            Real tmp1[3], tmp2[3];
            scale3(coef, e_hat, dYn);
            scale3(dot3(e_hat, a), mB, tmp1);
            scale3(dot3(e_hat, b), mA, tmp2);
            for (int k=0;k<3;++k) dYn[k] += tmp1[k] + tmp2[k];
            break;
        }
        case 2: cross3(mB, e, dXn); scale3(-ell, mB, dYn); break;
        case 3: cross3(e, mA, dXn); scale3(-ell, mA, dYn); break;
    }

    Real denom = muA2*muB2;
    Real gtheta[3];
    for (int k=0;k<3;++k) gtheta[k] = (X*dYn[k] - Y*dXn[k]) / denom;

    Real delta_theta = theta - bar_theta;
    addscale3(dt2*2.0f*kB*ce*delta_theta, gtheta, g);
    outer3_add(dt2*2.0f*kB*ce, gtheta, H);
}

// ============================================================================
// Pointer bundles for kernel params.
// ============================================================================
struct MeshP {
    const int* tris;
    const Real* Dm_inv;
    const Real* area;
    const Real* mass;
    const int* hinge_v;
    const Real* hinge_bar_theta;
    const Real* hinge_ce;
    const int* hinge_adj_offsets;
    const int* hinge_adj_hi;
    const int* hinge_adj_role;
};
struct AdjP {
    const int* offsets;
    const int* tri_idx;
    const int* tri_local;
};
struct PinsP {
    const int* pin_map;
    const Real* targets;
};
struct GP {
    Real dt2;
    Real dt;
    Real mu, lambda, kB, kpin;
    Real gx, gy, gz;
    Real damping;   // Rayleigh-style coefficient β; damp adds (β/dt)·H_elastic to H
    int mass_normalize_residual;
    int use_stvk;   // 1 = plain StVK (Newton-style), 0 = corotated StVK (default)
};

// ============================================================================
// Per-vertex local Newton direction (elastic + bending + pin + inertia).
// ============================================================================
__device__ static void local_newton_direction(
    int vi,
    MeshP mesh, AdjP adj, PinsP pins, GP p,
    const Real* x, const Real* xhat,
    Real g[3], Real H[9], Real delta_out[3])
{
    for (int i=0;i<3;++i) g[i] = 0;
    for (int i=0;i<9;++i) H[i] = 0;

    // Inertia + gravity.
    Real xi[3], xhi[3];
    loadv(x, vi, xi); loadv(xhat, vi, xhi);
    Real m = mesh.mass[vi];
    Real grav[3] = {p.gx, p.gy, p.gz};
    for (int k=0;k<3;++k) g[k] += m*(xi[k]-xhi[k]) + p.dt2*(-m*grav[k]);
    adddiag3(m, H);

    // Pin spring.
    int pi = pins.pin_map[vi];
    if (pi >= 0) {
        Real t[3] = { pins.targets[pi*3], pins.targets[pi*3+1], pins.targets[pi*3+2] };
        for (int k=0;k<3;++k) g[k] += p.dt2*p.kpin*(xi[k]-t[k]);
        adddiag3(p.dt2*p.kpin, H);
    }

    // Elastic — each incident triangle.
    for (int idx=adj.offsets[vi]; idx<adj.offsets[vi+1]; ++idx) {
        int ti = adj.tri_idx[idx];
        int a  = adj.tri_local[idx];
        int v0 = mesh.tris[ti*3+0], v1 = mesh.tris[ti*3+1], v2 = mesh.tris[ti*3+2];
        Real xv0[3], xv1[3], xv2[3];
        loadv(x, v0, xv0); loadv(x, v1, xv1); loadv(x, v2, xv2);
        if (p.use_stvk) {
            // Untiled path: no d_x_prev plumbed through local_newton_direction,
            // so force damping=0 here. Tiled path (the benchmark default)
            // does the full per-constraint damping.
            stvk_vertex_gh(a, xv0, xv1, xv2, xv0, xv1, xv2,
                           mesh.Dm_inv + ti*4, mesh.area[ti],
                           p.mu, p.lambda, p.dt2, p.dt, Real(0), g, H);
        } else {
            corotated_vertex_gh(a, xv0, xv1, xv2, mesh.Dm_inv + ti*4, mesh.area[ti],
                                p.mu, p.lambda, p.dt2, g, H);
        }
    }

    // Bending — each incident hinge.
    if (p.kB > 0.0f) {
        for (int idx=mesh.hinge_adj_offsets[vi]; idx<mesh.hinge_adj_offsets[vi+1]; ++idx) {
            int hi   = mesh.hinge_adj_hi[idx];
            int role = mesh.hinge_adj_role[idx];
            Real hv0[3], hv1[3], hv2[3], hv3[3];
            loadv(x, mesh.hinge_v[hi*4+0], hv0);
            loadv(x, mesh.hinge_v[hi*4+1], hv1);
            loadv(x, mesh.hinge_v[hi*4+2], hv2);
            loadv(x, mesh.hinge_v[hi*4+3], hv3);
            bending_vertex_gh_psd(role, hv0, hv1, hv2, hv3,
                                  p.kB, mesh.hinge_ce[hi], mesh.hinge_bar_theta[hi],
                                  p.dt2, g, H);
        }
    }

    // delta = H^{-1} g
    Real Hinv[9]; mat33_inverse(H, Hinv);
    matvec3(Hinv, g, delta_out);
}

// ============================================================================
// Kernels
// ============================================================================

// Device kernels for the per-substep driver kept on-device.
// build_xhat_kernel:   xhat = x + dt * v          (inertial prediction)
// copy_nv3_kernel:     dst[0..3*nv] = src[0..3*nv]
// update_velocity_kernel: v = (x_new - x_prev) / dt
__global__ static void build_xhat_kernel(
    int nv, Real dt, const Real* x, const Real* v, Real* xhat)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;
    xhat[vi*3+0] = x[vi*3+0] + dt * v[vi*3+0];
    xhat[vi*3+1] = x[vi*3+1] + dt * v[vi*3+1];
    xhat[vi*3+2] = x[vi*3+2] + dt * v[vi*3+2];
}

__global__ static void copy_nv3_kernel(int nv, const Real* src, Real* dst) {
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;
    dst[vi*3+0] = src[vi*3+0];
    dst[vi*3+1] = src[vi*3+1];
    dst[vi*3+2] = src[vi*3+2];
}

__global__ static void update_velocity_kernel(
    int nv, Real inv_dt, const Real* x_new, const Real* x_prev, Real* v)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;
    v[vi*3+0] = (x_new[vi*3+0] - x_prev[vi*3+0]) * inv_dt;
    v[vi*3+1] = (x_new[vi*3+1] - x_prev[vi*3+1]) * inv_dt;
    v[vi*3+2] = (x_new[vi*3+2] - x_prev[vi*3+2]) * inv_dt;
}

__global__ static void predict_kernel(
    int nv, MeshP mesh, AdjP adj, PinsP pins, GP p,
    const Real* x, const Real* xhat, Real* pred_delta)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;
    Real g[3], H[9], d[3];
    local_newton_direction(vi, mesh, adj, pins, p, x, xhat, g, H, d);
    pred_delta[vi*3+0]=d[0]; pred_delta[vi*3+1]=d[1]; pred_delta[vi*3+2]=d[2];
}

// Tiled color sweep: half-warp (TILE_SIZE=16) per particle. Threads stride
// through the particle's incident triangles and hinges, each accumulating a
// partial (f, H); a warp-shuffle sum collapses the 16 partials; thread 0
// adds inertia/pin/gravity, inverts H, and writes the position update.
// Matches Newton's SolverVBD TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE = 16.
//
// Mirrors Newton's SolverVBD `solve_elasticity_tile` pattern — spreads the
// per-vertex accumulation across a warp instead of running it all in one
// thread. Trades launch surface (grid = num_verts_in_color blocks) for
// register pressure (small per-thread state) and memory parallelism
// (concurrent loads across 32 threads).
// Untiled baseline: one thread per particle; kept for A/B-ing against the
// tiled version. Selected at build_graph time via GPU_ELASTIC_UNTILED=1.
__global__ static void sweep_color_untiled_kernel(
    int target_color,
    const int* group_indices, const int* group_offsets,
    MeshP mesh, AdjP adj, PinsP pins, GP p,
    Real* d_x, const Real* d_xhat)
{
    const int lo = group_offsets[target_color];
    const int hi = group_offsets[target_color + 1];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hi - lo) return;
    const int vi = group_indices[lo + idx];
    Real g[3], H[9], delta[3];
    local_newton_direction(vi, mesh, adj, pins, p, d_x, d_xhat, g, H, delta);
    d_x[vi*3+0] -= delta[0];
    d_x[vi*3+1] -= delta[1];
    d_x[vi*3+2] -= delta[2];
}

__global__ static void sweep_color_kernel(
    int target_color,
    const int* group_indices, const int* group_offsets,
    MeshP mesh, AdjP adj, PinsP pins, GP p,
    Real* d_x, const Real* d_xhat, const Real* d_x_prev)
{
    const int lo = group_offsets[target_color];
    const int hi = group_offsets[target_color + 1];
    const int tile_id = blockIdx.x;
    if (tile_id >= hi - lo) return;
    const int vi = group_indices[lo + tile_id];
    const int tid = threadIdx.x;   // 0..31

    Real f[3] = { Real(0), Real(0), Real(0) };
    Real H[9] = { Real(0), Real(0), Real(0),
                  Real(0), Real(0), Real(0),
                  Real(0), Real(0), Real(0) };

    // Elastic — each incident triangle, strided across the warp.
    const int tri_lo = adj.offsets[vi];
    const int tri_hi = adj.offsets[vi + 1];
    for (int idx = tri_lo + tid; idx < tri_hi; idx += 16) {
        int ti = adj.tri_idx[idx];
        int a  = adj.tri_local[idx];
        int v0 = mesh.tris[ti*3+0], v1 = mesh.tris[ti*3+1], v2 = mesh.tris[ti*3+2];
        Real xv0[3], xv1[3], xv2[3];
        loadv(d_x, v0, xv0); loadv(d_x, v1, xv1); loadv(d_x, v2, xv2);
        if (p.use_stvk) {
            Real xp0[3], xp1[3], xp2[3];
            loadv(d_x_prev, v0, xp0); loadv(d_x_prev, v1, xp1); loadv(d_x_prev, v2, xp2);
            stvk_vertex_gh(a, xv0, xv1, xv2, xp0, xp1, xp2,
                           mesh.Dm_inv + ti*4, mesh.area[ti],
                           p.mu, p.lambda, p.dt2, p.dt, p.damping, f, H);
        } else {
            corotated_vertex_gh(a, xv0, xv1, xv2, mesh.Dm_inv + ti*4, mesh.area[ti],
                                p.mu, p.lambda, p.dt2, f, H);
        }
    }

    // Bending — each incident hinge, strided across the warp.
    if (p.kB > Real(0)) {
        const int he_lo = mesh.hinge_adj_offsets[vi];
        const int he_hi = mesh.hinge_adj_offsets[vi + 1];
        for (int idx = he_lo + tid; idx < he_hi; idx += 16) {
            int hidx = mesh.hinge_adj_hi[idx];
            int role = mesh.hinge_adj_role[idx];
            Real hv0[3], hv1[3], hv2[3], hv3[3];
            loadv(d_x, mesh.hinge_v[hidx*4+0], hv0);
            loadv(d_x, mesh.hinge_v[hidx*4+1], hv1);
            loadv(d_x, mesh.hinge_v[hidx*4+2], hv2);
            loadv(d_x, mesh.hinge_v[hidx*4+3], hv3);
            bending_vertex_gh_psd(role, hv0, hv1, hv2, hv3,
                                  p.kB, mesh.hinge_ce[hidx], mesh.hinge_bar_theta[hidx],
                                  p.dt2, f, H);
        }
    }

    // Warp-shuffle reduction: sum f (3) and H (9) across the 16 lanes of the
    // lower half-warp. Block size is 16 so mask 0x0000FFFF covers all live
    // lanes; threads 16..31 are not launched.
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) f[i] += __shfl_down_sync(0x0000FFFFu, f[i], offset);
        #pragma unroll
        for (int i = 0; i < 9; ++i) H[i] += __shfl_down_sync(0x0000FFFFu, H[i], offset);
    }

    if (tid != 0) return;

    // At this point f, H are the reduced elastic+bending partials for vi.
    // Rayleigh-style damping proportional to the (elastic+bending) Hessian,
    // matching Newton's damp_force_and_hessian. In our dt²-scaled convention
    // the ratio (β/dt) is the same coefficient Newton uses:
    //   f_damp = (β/dt) · H_elastic · (x_prev − x)
    //   H_damp = (β/dt) · H_elastic
    // Applied BEFORE inertia/pin so damping scales only the stiffness.
    Real xi[3], xhi[3];
    loadv(d_x, vi, xi);
    loadv(d_xhat, vi, xhi);

    const Real damping_dt = p.damping / p.dt;
    if (damping_dt > Real(0)) {
        // Real x_prev (saved into d_x_prev by copy_nv3_kernel at substep
        // start, so invariant across the iteration loop). Our convention
        // stores g = grad(E); Rayleigh damping contributes +β/dt · H · (x − x_prev).
        Real xprev[3];
        loadv(d_x_prev, vi, xprev);
        const Real dvec[3] = { xi[0] - xprev[0], xi[1] - xprev[1], xi[2] - xprev[2] };
        const Real Hd[3] = {
            H[0]*dvec[0] + H[1]*dvec[1] + H[2]*dvec[2],
            H[3]*dvec[0] + H[4]*dvec[1] + H[5]*dvec[2],
            H[6]*dvec[0] + H[7]*dvec[1] + H[8]*dvec[2]
        };
        f[0] += damping_dt * Hd[0];
        f[1] += damping_dt * Hd[1];
        f[2] += damping_dt * Hd[2];
        const Real scale = Real(1) + damping_dt;
        #pragma unroll
        for (int i = 0; i < 9; ++i) H[i] *= scale;
    }

    const Real m = mesh.mass[vi];
    const Real grav[3] = { p.gx, p.gy, p.gz };
    for (int k = 0; k < 3; ++k) f[k] += m*(xi[k] - xhi[k]) + p.dt2*(-m*grav[k]);
    adddiag3(m, H);

    const int pi = pins.pin_map[vi];
    if (pi >= 0) {
        Real t[3] = { pins.targets[pi*3+0], pins.targets[pi*3+1], pins.targets[pi*3+2] };
        for (int k = 0; k < 3; ++k) f[k] += p.dt2 * p.kpin * (xi[k] - t[k]);
        adddiag3(p.dt2 * p.kpin, H);
    }

    Real Hinv[9]; mat33_inverse(H, Hinv);
    Real delta[3]; matvec3(Hinv, f, delta);
    d_x[vi*3+0] -= delta[0];
    d_x[vi*3+1] -= delta[1];
    d_x[vi*3+2] -= delta[2];
}

__global__ static void residual_kernel(
    int nv, MeshP mesh, AdjP adj, PinsP pins, GP p,
    const Real* d_x, const Real* d_xhat, Real* d_rmax)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;
    Real g[3], H[9], dummy[3];
    local_newton_direction(vi, mesh, adj, pins, p, d_x, d_xhat, g, H, dummy);
    Real gn = norm3(g);
    if (p.mass_normalize_residual) {
        Real m = mesh.mass[vi];
        if (m > 1e-24f) gn /= m;
    }
    atomicMaxFloat(d_rmax, gn);
}

// ============================================================================
// Session (persistent device state).
// ============================================================================
struct Session {
    int nv = 0, num_colors = 0;
    int num_tris = 0, num_hinges = 0, num_pins = 0;
    GP gp{};

    // Double-precision mesh mirrors (for shared GPU infra; not read by kernels).
    GPURefMesh   mesh;
    GPUAdjacency adj;
    GPUPins      pins;
    GPUPinMap    pin_map;

    // FP32 mirrors of time-invariant mesh data (read by kernels).
    Real *d_Dm_invf = nullptr;           // num_tris*4
    Real *d_areaf = nullptr;             // num_tris
    Real *d_massf = nullptr;             // nv
    Real *d_hinge_bar_theta_f = nullptr; // num_hinges
    Real *d_hinge_ce_f = nullptr;        // num_hinges
    Real *d_pin_targets_f = nullptr;     // num_pins*3 (refreshed per substep)
    Real *h_pin_targets_pinned = nullptr;// pinned mirror so per-substep uploads
                                         // can be async on s.stream (avoid
                                         // default-stream sync with classic
                                         // cudaMemcpy semantics).

    // Per-vertex device state (FP32).
    Real *d_x = nullptr, *d_xhat = nullptr, *d_pred_delta = nullptr;
    Real *d_v = nullptr;       // velocity, nv*3, device-resident across substeps
    Real *d_x_prev = nullptr;  // position at start of substep (for velocity update)
    Real *d_residual = nullptr;
    // Pinned host buffer holding one residual per substep in the current frame.
    // Async D2H per substep + a single cudaStreamSynchronize at end_frame lets
    // the CPU submit all substep graphs back-to-back while GPU pipelines them.
    Real *h_residuals_pinned = nullptr;
    int   h_residuals_cap = 0;
    int   h_residuals_count = 0;

    // Precomputed coloring (fixed once, since no collision).
    int *d_group_indices = nullptr;
    int *d_group_offsets = nullptr;
    std::vector<int> h_group_sizes;   // per-color vertex count (host copy)

    cudaStream_t stream = nullptr;
    double last_residual = -1.0;

    // CUDA Graph: captured once, replayed per substep. Graph encodes
    // (predict + sweep_c × num_colors) × max_iters + residual kernel.
    // Rebuilt if max_iters changes (otherwise same graph reused forever).
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int             cached_max_iters = -1;
};

static Session* g_sess = nullptr;

static void destroy_graph(Session* s) {
    if (!s) return;
    if (s->graph_exec) { cudaGraphExecDestroy(s->graph_exec); s->graph_exec = nullptr; }
    if (s->graph)      { cudaGraphDestroy(s->graph);          s->graph      = nullptr; }
    s->cached_max_iters = -1;
}

static void free_scratch(Session* s) {
    if (!s) return;
    destroy_graph(s);
    auto F = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F(s->d_x); F(s->d_xhat); F(s->d_pred_delta); F(s->d_residual);
    F(s->d_v); F(s->d_x_prev);
    if (s->h_residuals_pinned) { cudaFreeHost(s->h_residuals_pinned); s->h_residuals_pinned = nullptr; }
    s->h_residuals_cap = 0;
    s->h_residuals_count = 0;
    F(s->d_Dm_invf); F(s->d_areaf); F(s->d_massf);
    F(s->d_hinge_bar_theta_f); F(s->d_hinge_ce_f); F(s->d_pin_targets_f);
    if (s->h_pin_targets_pinned) { cudaFreeHost(s->h_pin_targets_pinned); s->h_pin_targets_pinned = nullptr; }
    F(s->d_group_indices); F(s->d_group_offsets);
    if (s->stream) { cudaStreamDestroy(s->stream); s->stream = nullptr; }
}

// Capture one full substep into a single CUDA graph:
//   1) build d_xhat from d_x + dt*d_v
//   2) save d_x → d_x_prev (for velocity update)
//   3) initial guess: d_x := d_xhat       (since d_hat==0 path)
//   4) (predict + per-color sweep) × max_iters
//   5) d_v := (d_x - d_x_prev)/dt
//   6) residual kernel
// All pointers baked from stable session buffers; pin targets refreshed via
// a pre-substep H2D that does not require recapture.
static void build_graph(Session& s, int max_iters) {
    destroy_graph(&s);

    MeshP mesh{
        s.mesh.tris.ptr, s.d_Dm_invf, s.d_areaf, s.d_massf,
        s.mesh.hinge_v.ptr, s.d_hinge_bar_theta_f, s.d_hinge_ce_f,
        s.mesh.hinge_adj_offsets.ptr, s.mesh.hinge_adj_hi.ptr, s.mesh.hinge_adj_role.ptr
    };
    AdjP  adj { s.adj.offsets.ptr, s.adj.tri_idx.ptr, s.adj.tri_local.ptr };
    PinsP pins{ s.pin_map.data.ptr, s.d_pin_targets_f };

    const int block  = 128;
    const int grid   = (s.nv + block - 1) / block;
    // Tiled sweep: half-warp (16 threads) per particle. Grid = num_verts_in_color.
    const int sblock = 16;
    static const bool untiled = [] {
        const char* e = std::getenv("GPU_ELASTIC_UNTILED");
        return e && e[0] == '1';
    }();
    const int sblock_untiled = 128;
    const Real dt     = s.gp.dt;
    const Real inv_dt = (dt > Real(0)) ? Real(1) / dt : Real(0);

    cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeGlobal);

    // 1. xhat = x + dt*v
    build_xhat_kernel<<<grid, block, 0, s.stream>>>(
        s.nv, dt, s.d_x, s.d_v, s.d_xhat);

    // 2. x_prev := x  (for velocity update at the end)
    //    Also doubles as the initial guess for iter 0 — with fixed 10-iter
    //    budgets the previous-step warm-start produces smoother dynamics
    //    than the full-inertial jump `x := xhat`.
    copy_nv3_kernel<<<grid, block, 0, s.stream>>>(s.nv, s.d_x, s.d_x_prev);

    // 4. Gauss-Seidel iterations. No predict_kernel: for the no-collision
    //    path every color recomputes its Newton direction fresh. When
    //    collisions come back, re-introduce a predict pass for the
    //    certified-region bounds it also produced.
    for (int iter = 0; iter < max_iters; ++iter) {
        for (int c = 0; c < s.num_colors; ++c) {
            const int csize = s.h_group_sizes[c];
            if (csize == 0) continue;
            if (untiled) {
                const int sgrid = (csize + sblock_untiled - 1) / sblock_untiled;
                sweep_color_untiled_kernel<<<sgrid, sblock_untiled, 0, s.stream>>>(
                    c, s.d_group_indices, s.d_group_offsets,
                    mesh, adj, pins, s.gp,
                    s.d_x, s.d_xhat);
            } else {
                // One block (= one half-warp) per particle in this color.
                sweep_color_kernel<<<csize, sblock, 0, s.stream>>>(
                    c, s.d_group_indices, s.d_group_offsets,
                    mesh, adj, pins, s.gp,
                    s.d_x, s.d_xhat, s.d_x_prev);
            }
        }
    }

    // 5. v := (x - x_prev) / dt
    update_velocity_kernel<<<grid, block, 0, s.stream>>>(
        s.nv, inv_dt, s.d_x, s.d_x_prev, s.d_v);

    // 6. residual reduction.
    cudaMemsetAsync(s.d_residual, 0, sizeof(Real), s.stream);
    residual_kernel<<<grid, block, 0, s.stream>>>(
        s.nv, mesh, adj, pins, s.gp, s.d_x, s.d_xhat, s.d_residual);

    cudaStreamEndCapture(s.stream, &s.graph);
    cudaGraphInstantiate(&s.graph_exec, s.graph, nullptr, nullptr, 0);
    s.cached_max_iters = max_iters;
}

// Utility: D2H from a double device buffer, convert to float, H2D to a newly
// allocated float device buffer. One-time init cost per mesh array.
static void alloc_and_convert_to_float(Real*& dst, const double* src_dev, int n) {
    std::vector<double> tmp_d(n);
    cudaMemcpy(tmp_d.data(), src_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<Real> tmp_f(n);
    for (int i = 0; i < n; ++i) tmp_f[i] = static_cast<Real>(tmp_d[i]);
    cudaMalloc(&dst, n*sizeof(Real));
    cudaMemcpy(dst, tmp_f.data(), n*sizeof(Real), cudaMemcpyHostToDevice);
}

// ============================================================================
// Public API
// ============================================================================

void gpu_elastic_init(const RefMesh& ref_mesh,
                      const VertexTriangleMap& adj,
                      const std::vector<Pin>& pins,
                      const SimParams& params)
{
    gpu_elastic_shutdown();
    g_sess = new Session();
    Session& s = *g_sess;

    const int nv = (int)ref_mesh.mass.size();
    s.nv = nv;

    // Cache scalars (cast to float for the device-side GP).
    s.gp.dt     = static_cast<Real>(params.dt());
    s.gp.dt2    = static_cast<Real>(params.dt() * params.dt());
    s.gp.mu     = static_cast<Real>(params.mu);
    s.gp.lambda = static_cast<Real>(params.lambda);
    s.gp.kB     = static_cast<Real>(params.kB);
    s.gp.damping= Real(0);
    {
        const char* e = std::getenv("GPU_ELASTIC_STVK");
        s.gp.use_stvk = (e && e[0] == '1') ? 1 : 0;
    }
    s.gp.kpin   = static_cast<Real>(params.kpin);
    s.gp.gx     = static_cast<Real>(params.gravity(0));
    s.gp.gy     = static_cast<Real>(params.gravity(1));
    s.gp.gz     = static_cast<Real>(params.gravity(2));
    s.gp.mass_normalize_residual = 1;

    // Mesh + adjacency + pins via existing GPU mirrors (double).
    s.mesh.upload(ref_mesh);
    s.adj.upload(adj, nv);
    s.pins.upload(pins);
    PinMap pm = build_pin_map(pins, nv);
    s.pin_map.upload(pm);

    s.num_tris   = s.mesh.num_tris;
    s.num_hinges = s.mesh.num_hinges;
    s.num_pins   = s.pins.count;

    // Build float mirrors from the double buffers.
    alloc_and_convert_to_float(s.d_Dm_invf, s.mesh.Dm_inv.ptr, s.num_tris * 4);
    alloc_and_convert_to_float(s.d_areaf,   s.mesh.area.ptr,   s.num_tris);
    alloc_and_convert_to_float(s.d_massf,   s.mesh.mass.ptr,   nv);
    if (s.num_hinges > 0) {
        alloc_and_convert_to_float(s.d_hinge_bar_theta_f, s.mesh.hinge_bar_theta.ptr, s.num_hinges);
        alloc_and_convert_to_float(s.d_hinge_ce_f,        s.mesh.hinge_ce.ptr,        s.num_hinges);
    }
    if (s.num_pins > 0) {
        alloc_and_convert_to_float(s.d_pin_targets_f, s.pins.targets.ptr, s.num_pins * 3);
        cudaMallocHost(&s.h_pin_targets_pinned, s.num_pins * 3 * sizeof(Real));
    }

    // Per-vertex scratch (FP32).
    cudaMalloc(&s.d_x,          nv*3*sizeof(Real));
    cudaMalloc(&s.d_xhat,       nv*3*sizeof(Real));
    cudaMalloc(&s.d_pred_delta, nv*3*sizeof(Real));
    cudaMalloc(&s.d_v,          nv*3*sizeof(Real));
    cudaMalloc(&s.d_x_prev,     nv*3*sizeof(Real));
    cudaMalloc(&s.d_residual,   sizeof(Real));
    cudaMemset(s.d_v, 0, nv*3*sizeof(Real));

    // Precomputed coloring (DSATUR). Static-order greedy (index or
    // degree-ordered) misses 3-coloring on regular grids because every
    // interior vertex ties on degree; DSATUR re-picks by current saturation
    // each step and finds it. O(V²) here is fine — init-time only.
    // Parallel Jones-Plassmann is the plan for the contact-time case.
    auto vadj = build_vertex_adjacency_map(ref_mesh.tris);
    auto color_groups = [&]() {
        std::vector<int> color(nv, -1);
        std::vector<int> sat(nv, 0);
        std::vector<int> deg(nv, 0);
        std::vector<std::unordered_set<int>> nbr_colors(nv);
        for (auto& [v, n] : vadj) if (v >= 0 && v < nv) deg[v] = (int)n.size();

        for (int step = 0; step < nv; ++step) {
            // Pick uncolored vertex with max saturation, tie-break by degree.
            int best = -1, best_sat = -1, best_deg = -1;
            for (int v = 0; v < nv; ++v) {
                if (color[v] >= 0) continue;
                if (sat[v] > best_sat ||
                    (sat[v] == best_sat && deg[v] > best_deg)) {
                    best = v; best_sat = sat[v]; best_deg = deg[v];
                }
            }
            if (best < 0) break;

            std::unordered_set<int> used;
            auto it = vadj.find(best);
            if (it != vadj.end())
                for (int n : it->second)
                    if (color[n] >= 0) used.insert(color[n]);
            int c = 0; while (used.count(c)) c++;
            color[best] = c;

            if (it != vadj.end())
                for (int n : it->second)
                    if (color[n] < 0 && !nbr_colors[n].count(c)) {
                        nbr_colors[n].insert(c);
                        sat[n]++;
                    }
        }

        int nc = 0;
        for (int c : color) if (c + 1 > nc) nc = c + 1;
        std::vector<std::vector<int>> groups(nc);
        for (int v = 0; v < nv; ++v) if (color[v] >= 0) groups[color[v]].push_back(v);
        return groups;
    }();
    s.num_colors = (int)color_groups.size();

    std::vector<int> h_gi(nv), h_go(s.num_colors + 1, 0);
    s.h_group_sizes.assign(s.num_colors, 0);
    int pos = 0;
    for (int c = 0; c < s.num_colors; ++c) {
        h_go[c] = pos;
        s.h_group_sizes[c] = (int)color_groups[c].size();
        for (int vi : color_groups[c]) h_gi[pos++] = vi;
    }
    h_go[s.num_colors] = pos;
    cudaMalloc(&s.d_group_indices, h_gi.size()*sizeof(int));
    cudaMalloc(&s.d_group_offsets, h_go.size()*sizeof(int));
    cudaMemcpy(s.d_group_indices, h_gi.data(), h_gi.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_group_offsets, h_go.data(), h_go.size()*sizeof(int), cudaMemcpyHostToDevice);

    cudaStreamCreate(&s.stream);
    fprintf(stderr, "[gpu_elastic] init nv=%d tris=%d hinges=%d pins=%d colors=%d (FP32)\n",
            nv, s.num_tris, s.num_hinges, s.num_pins, s.num_colors);
}

void gpu_elastic_shutdown() {
    if (!g_sess) return;
    free_scratch(g_sess);
    delete g_sess;
    g_sess = nullptr;
}

void gpu_elastic_set_pin_targets(const std::vector<Pin>& pins) {
    if (!g_sess) return;
    Session& s = *g_sess;
    if ((int)pins.size() != s.num_pins) {
        fprintf(stderr, "[gpu_elastic] pin count changed (%d -> %zu); reallocating float mirror\n",
                s.num_pins, pins.size());
        if (s.d_pin_targets_f) { cudaFree(s.d_pin_targets_f); s.d_pin_targets_f = nullptr; }
        if (s.h_pin_targets_pinned) { cudaFreeHost(s.h_pin_targets_pinned); s.h_pin_targets_pinned = nullptr; }
        s.num_pins = (int)pins.size();
        if (s.num_pins > 0) {
            cudaMalloc(&s.d_pin_targets_f, s.num_pins*3*sizeof(Real));
            cudaMallocHost(&s.h_pin_targets_pinned, s.num_pins*3*sizeof(Real));
        }
        // Graph baked the old pointer; must rebuild.
        destroy_graph(&s);
    }
    if (s.num_pins == 0) return;
    if (!s.h_pin_targets_pinned) {
        cudaMallocHost(&s.h_pin_targets_pinned, s.num_pins * 3 * sizeof(Real));
    }
    for (int pi = 0; pi < s.num_pins; ++pi) {
        s.h_pin_targets_pinned[pi*3+0] = static_cast<Real>(pins[pi].target_position(0));
        s.h_pin_targets_pinned[pi*3+1] = static_cast<Real>(pins[pi].target_position(1));
        s.h_pin_targets_pinned[pi*3+2] = static_cast<Real>(pins[pi].target_position(2));
    }
    // Enqueue on s.stream so it orders ahead of the next captured graph launch
    // and doesn't force a global sync (classic default-stream cudaMemcpy would).
    cudaMemcpyAsync(s.d_pin_targets_f, s.h_pin_targets_pinned,
                    s.num_pins * 3 * sizeof(Real),
                    cudaMemcpyHostToDevice, s.stream);
}

// Ensure the pinned residual buffer has room for `n` substeps.
static void ensure_residual_capacity(Session& s, int n) {
    if (n <= s.h_residuals_cap) return;
    int new_cap = s.h_residuals_cap > 0 ? s.h_residuals_cap : 64;
    while (new_cap < n) new_cap *= 2;
    if (s.h_residuals_pinned) cudaFreeHost(s.h_residuals_pinned);
    cudaMallocHost(&s.h_residuals_pinned, new_cap * sizeof(Real));
    s.h_residuals_cap = new_cap;
}

// Begin-of-frame upload: push x and v into device-resident buffers.
// Kept paired with gpu_elastic_end_frame to bracket the per-frame substep
// loop so substeps don't pay H2D/D2H for x/v and only sync once per frame.
void gpu_elastic_begin_frame(const std::vector<Vec3>& x, const std::vector<Vec3>& v) {
    if (!g_sess) return;
    Session& s = *g_sess;
    const int nv = s.nv;
    if ((int)x.size() != nv || (int)v.size() != nv) return;

    std::vector<Real> h_x(nv*3), h_v(nv*3);
    for (int i = 0; i < nv; ++i) {
        h_x[i*3+0]=(Real)x[i](0); h_x[i*3+1]=(Real)x[i](1); h_x[i*3+2]=(Real)x[i](2);
        h_v[i*3+0]=(Real)v[i](0); h_v[i*3+1]=(Real)v[i](1); h_v[i*3+2]=(Real)v[i](2);
    }
    cudaMemcpy(s.d_x, h_x.data(), nv*3*sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_v, h_v.data(), nv*3*sizeof(Real), cudaMemcpyHostToDevice);

    s.h_residuals_count = 0;
}

// Run one substep fully on-device. No H2D/D2H of x or v. The captured graph
// computes xhat, saves x_prev, runs max_iters GS iterations, updates v, and
// reduces the residual. Residual D2H is async into a pinned per-frame slot;
// no cudaStreamSynchronize here — the sync happens in end_frame. Substep
// residuals are readable after end_frame via gpu_elastic_substep_residual().
bool gpu_elastic_run_substep_device(int max_iters) {
    if (!g_sess) return false;
    Session& s = *g_sess;

    if (s.cached_max_iters != max_iters) {
        build_graph(s, max_iters);
    }

    ensure_residual_capacity(s, s.h_residuals_count + 1);

    cudaGraphLaunch(s.graph_exec, s.stream);
    cudaMemcpyAsync(&s.h_residuals_pinned[s.h_residuals_count],
                    s.d_residual, sizeof(Real),
                    cudaMemcpyDeviceToHost, s.stream);
    s.h_residuals_count++;
    return true;
}

// End-of-frame download: single cudaStreamSynchronize (covers all pipelined
// substeps + async residual D2Hs), then pull x and v back. After this call,
// h_residuals_pinned[0..count-1] holds this frame's per-substep residuals.
void gpu_elastic_end_frame(std::vector<Vec3>& x, std::vector<Vec3>& v) {
    if (!g_sess) return;
    Session& s = *g_sess;
    const int nv = s.nv;

    // Drain the stream: all graph launches + residual D2Hs retire here.
    cudaStreamSynchronize(s.stream);

    // Cache the last substep's residual for gpu_elastic_last_residual().
    if (s.h_residuals_count > 0) {
        s.last_residual = static_cast<double>(
            s.h_residuals_pinned[s.h_residuals_count - 1]);
    }

    std::vector<Real> h_x(nv*3), h_v(nv*3);
    cudaMemcpy(h_x.data(), s.d_x, nv*3*sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v.data(), s.d_v, nv*3*sizeof(Real), cudaMemcpyDeviceToHost);

    if ((int)x.size() != nv) x.resize(nv);
    if ((int)v.size() != nv) v.resize(nv);
    for (int i = 0; i < nv; ++i) {
        x[i] = Vec3(h_x[i*3+0], h_x[i*3+1], h_x[i*3+2]);
        v[i] = Vec3(h_v[i*3+0], h_v[i*3+1], h_v[i*3+2]);
    }
}

// Read the residual recorded at substep index `sub` during the current frame.
// Only valid after gpu_elastic_end_frame has drained the stream. Returns -1.0
// if out-of-range or the session is gone.
double gpu_elastic_substep_residual(int sub) {
    if (!g_sess) return -1.0;
    Session& s = *g_sess;
    if (sub < 0 || sub >= s.h_residuals_count) return -1.0;
    return static_cast<double>(s.h_residuals_pinned[sub]);
}

bool gpu_elastic_run_substep(
    const std::vector<Vec3>& x_in,
    const std::vector<Vec3>& xhat,
    int max_iters,
    std::vector<Vec3>& x_out)
{
    // Legacy path kept for the stub API signature. With the new in-graph
    // xhat/velocity handling, the host-supplied `xhat` is ignored; we derive
    // it from the session's d_x + d_v. Callers wanting the fast path should
    // bracket substep loops with gpu_elastic_begin_frame / end_frame instead.
    if (!g_sess) return false;
    Session& s = *g_sess;
    const int nv = s.nv;
    if ((int)x_in.size() != nv || (int)xhat.size() != nv) return false;
    (void)xhat;

    using clk = std::chrono::high_resolution_clock;
    auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };

    auto t0 = clk::now();

    std::vector<Real> h_x(nv*3);
    for (int i = 0; i < nv; ++i) {
        h_x[i*3+0]=(Real)x_in[i](0); h_x[i*3+1]=(Real)x_in[i](1); h_x[i*3+2]=(Real)x_in[i](2);
    }
    auto t1 = clk::now();
    cudaMemcpy(s.d_x, h_x.data(), nv*3*sizeof(Real), cudaMemcpyHostToDevice);
    auto t2 = clk::now();

    if (s.cached_max_iters != max_iters) build_graph(s, max_iters);
    cudaGraphLaunch(s.graph_exec, s.stream);
    cudaStreamSynchronize(s.stream);
    auto t3 = clk::now();

    Real rmax = 0.0f;
    cudaMemcpyAsync(h_x.data(), s.d_x,        nv*3*sizeof(Real), cudaMemcpyDeviceToHost, s.stream);
    cudaMemcpyAsync(&rmax,      s.d_residual, sizeof(Real),       cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    auto t4 = clk::now();

    s.last_residual = static_cast<double>(rmax);
    if ((int)x_out.size() != nv) x_out.resize(nv);
    for (int i = 0; i < nv; ++i) {
        x_out[i] = Vec3(h_x[i*3+0], h_x[i*3+1], h_x[i*3+2]);
    }
    auto t5 = clk::now();

    fprintf(stderr,
        "[elastic-prof] iters=%d pack=%.2fms upload=%.2fms graph=%.2fms dl=%.2fms unpack=%.2fms | total=%.2fms\n",
        max_iters, ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t3,t4), ms(t4,t5), ms(t0,t5));
    return true;
}

double gpu_elastic_last_residual() {
    return g_sess ? g_sess->last_residual : -1.0;
}
