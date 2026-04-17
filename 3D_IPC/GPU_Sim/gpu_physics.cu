// gpu_physics.cu
// Real CUDA implementation of gpu_physics.h.
// The per-vertex computation is factored into a __device__ function so it
// can be called directly from any kernel.

#include "gpu_physics.h"
#include "corotated_energy.h"
#include "bending_energy.h"
#include "barrier_energy.h"
#include "IPC_math.h"
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// gpu_local_grad_hess_device
// Pure __device__ function — called once per thread inside a kernel.
// Writes gradient (3 doubles) and hessian (9 doubles, row-major) into
// caller-provided stack arrays rather than returning Eigen types, to keep
// register pressure manageable.
// --------------------------------------------------------------------------
__device__ static void gpu_local_grad_hess_device(
    int vi,
    const int*    tris,
    const double* Dm_inv,
    const double* area,
    const double* mass,
    const int*    hinge_v,
    const double* hinge_bar_theta,
    const double* hinge_ce,
    const int*    hinge_adj_offsets,
    const int*    hinge_adj_hi,
    const int*    hinge_adj_role,
    const int*    adj_offsets,
    const int*    adj_tri_idx,
    const int*    adj_tri_local,
    const int*    pin_indices,
    const double* pin_targets,
    const int*    pin_map,
    const GPUSimParams params,
    const double* x,
    const double* xhat,
    double* g_out,   // [3]
    double* H_out)   // [9], row-major
{
    const double dt2 = params.dt2();

    double gx = 0, gy = 0, gz = 0;
    double H[9] = {};  // 3x3 row-major, zero-init

    // --- inertia ---
    const double m = mass[vi];
    const double xi0 = x[vi*3], xi1 = x[vi*3+1], xi2 = x[vi*3+2];
    const double xh0 = xhat[vi*3], xh1 = xhat[vi*3+1], xh2 = xhat[vi*3+2];

    gx += m * (xi0 - xh0) + dt2 * (-m * params.gx);
    gy += m * (xi1 - xh1) + dt2 * (-m * params.gy);
    gz += m * (xi2 - xh2) + dt2 * (-m * params.gz);
    H[0] += m; H[4] += m; H[8] += m;

    // --- pin spring ---
    const int pi = pin_map[vi];
    if (pi >= 0) {
        const double tx = pin_targets[pi*3], ty = pin_targets[pi*3+1], tz = pin_targets[pi*3+2];
        gx += dt2 * params.kpin * (xi0 - tx);
        gy += dt2 * params.kpin * (xi1 - ty);
        gz += dt2 * params.kpin * (xi2 - tz);
        H[0] += dt2 * params.kpin;
        H[4] += dt2 * params.kpin;
        H[8] += dt2 * params.kpin;
    }

    // --- corotated elastic (CSR adjacency) ---
    for (int idx = adj_offsets[vi]; idx < adj_offsets[vi+1]; ++idx) {
        const int ti = adj_tri_idx[idx];
        const int a  = adj_tri_local[idx];

        const int v0 = tris[ti*3+0];
        const int v1 = tris[ti*3+1];
        const int v2 = tris[ti*3+2];

        TriangleDef def;
        def.x[0] = Vec3(x[v0*3], x[v0*3+1], x[v0*3+2]);
        def.x[1] = Vec3(x[v1*3], x[v1*3+1], x[v1*3+2]);
        def.x[2] = Vec3(x[v2*3], x[v2*3+1], x[v2*3+2]);

        const double* dm = Dm_inv + ti * 4;
        Mat22 Dm_inv_mat;
        Dm_inv_mat(0,0) = dm[0]; Dm_inv_mat(1,0) = dm[1];
        Dm_inv_mat(0,1) = dm[2]; Dm_inv_mat(1,1) = dm[3];

        const double A = area[ti];
        Mat32 Ds;
        Ds.col(0) = def.x[1] - def.x[0];
        Ds.col(1) = def.x[2] - def.x[0];
        const Mat32 F = Ds * Dm_inv_mat;

        const CorotatedCache32 cache = buildCorotatedCache(F);
        const ShapeGrads gradN = shape_function_gradients(Dm_inv_mat);
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        Mat66 dPdF;
        dPdFCorotated32(cache, params.mu, params.lambda, dPdF);

        Vec3  dg = dt2 * corotated_node_gradient(P, A, gradN, a);
        Mat33 dH = dt2 * corotated_node_hessian(dPdF, A, gradN, a);

        gx += dg(0); gy += dg(1); gz += dg(2);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                H[r*3+c] += dH(r, c);
    }

    // --- bending ---
    if (params.kB > 0.0) {
        for (int idx = hinge_adj_offsets[vi]; idx < hinge_adj_offsets[vi+1]; ++idx) {
            const int hi   = hinge_adj_hi[idx];
            const int role = hinge_adj_role[idx];

            HingeDef hdef;
            for (int k = 0; k < 4; ++k) {
                const int hv = hinge_v[hi*4+k];
                hdef.x[k] = Vec3(x[hv*3], x[hv*3+1], x[hv*3+2]);
            }
            Vec3  dg = dt2 * bending_node_gradient(hdef, params.kB, hinge_ce[hi], hinge_bar_theta[hi], role);
            Mat33 dH = dt2 * bending_node_hessian_psd(hdef, params.kB, hinge_ce[hi], hinge_bar_theta[hi], role);

            gx += dg(0); gy += dg(1); gz += dg(2);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    H[r*3+c] += dH(r, c);
        }
    }

    g_out[0] = gx; g_out[1] = gy; g_out[2] = gz;
    for (int i = 0; i < 9; ++i) H_out[i] = H[i];
}

// --------------------------------------------------------------------------
// gpu_compute_local_gradient_and_hessian_no_barrier  (host wrapper)
// Launches a trivial 1-thread kernel and reads back the result.
// In production the kernel body would be embedded in the full GS sweep
// kernel rather than called per-vertex like this.
// --------------------------------------------------------------------------

__global__ static void single_vertex_grad_hess_kernel(
    int vi,
    const int* tris, const double* Dm_inv, const double* area, const double* mass,
    const int* hinge_v, const double* hinge_bar_theta, const double* hinge_ce,
    const int* hinge_adj_offsets, const int* hinge_adj_hi, const int* hinge_adj_role,
    const int* adj_offsets, const int* adj_tri_idx, const int* adj_tri_local,
    const int* pin_indices, const double* pin_targets, const int* pin_map,
    GPUSimParams params,
    const double* x, const double* xhat,
    double* g_out, double* H_out)
{
    gpu_local_grad_hess_device(
        vi, tris, Dm_inv, area, mass,
        hinge_v, hinge_bar_theta, hinge_ce,
        hinge_adj_offsets, hinge_adj_hi, hinge_adj_role,
        adj_offsets, adj_tri_idx, adj_tri_local,
        pin_indices, pin_targets, pin_map,
        params, x, xhat, g_out, H_out);
}

std::pair<Vec3, Mat33> gpu_compute_local_gradient_and_hessian_no_barrier(
    int vi,
    const GPURefMesh&           mesh,
    const GPUAdjacency&         adj,
    const GPUPins&              pins,
    const GPUPinMap&            pin_map,
    const GPUSimParams&         params,
    const DeviceBuffer<double>& d_x,
    const DeviceBuffer<double>& d_xhat) {

    double *d_g, *d_H;
    cudaMalloc(&d_g, 3 * sizeof(double));
    cudaMalloc(&d_H, 9 * sizeof(double));

    single_vertex_grad_hess_kernel<<<1, 1>>>(
        vi,
        mesh.tris.ptr, mesh.Dm_inv.ptr, mesh.area.ptr, mesh.mass.ptr,
        mesh.hinge_v.ptr, mesh.hinge_bar_theta.ptr, mesh.hinge_ce.ptr,
        mesh.hinge_adj_offsets.ptr, mesh.hinge_adj_hi.ptr, mesh.hinge_adj_role.ptr,
        adj.offsets.ptr, adj.tri_idx.ptr, adj.tri_local.ptr,
        pins.indices.ptr, pins.targets.ptr, pin_map.data.ptr,
        params, d_x.ptr, d_xhat.ptr,
        d_g, d_H);

    double h_g[3], h_H[9];
    cudaMemcpy(h_g, d_g, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_H, d_H, 9 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_g); cudaFree(d_H);

    Vec3  g(h_g[0], h_g[1], h_g[2]);
    Mat33 H;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            H(r, c) = h_H[r*3+c];

    return {g, H};
}

// --------------------------------------------------------------------------
// gpu_compute_local_gradient_and_hessian  (host wrapper, with barrier)
// --------------------------------------------------------------------------
std::pair<Vec3, Mat33> gpu_compute_local_gradient_and_hessian(
    int vi,
    const GPURefMesh&           mesh,
    const GPUAdjacency&         adj,
    const GPUPins&              pins,
    const GPUPinMap&            pin_map,
    const GPUSimParams&         params,
    const GPUBroadPhaseCache&   bp,
    const DeviceBuffer<double>& d_x,
    const DeviceBuffer<double>& d_xhat) {

    auto [g, H] = gpu_compute_local_gradient_and_hessian_no_barrier(
        vi, mesh, adj, pins, pin_map, params, d_x, d_xhat);

    if (params.d_hat <= 0.0) return {g, H};

    const double dt2 = params.dt2();

    // Download positions to host for barrier evaluation.
    // (In a production kernel the per-vertex barrier loop would be inside
    // the device function; for the single-vertex wrapper we pull to host.)
    const int nv = bp.num_verts;
    std::vector<double> x_h(nv * 3);
    cudaMemcpy(x_h.data(), d_x.ptr, nv * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    auto load = [&](int idx) {
        return Vec3(x_h[idx*3], x_h[idx*3+1], x_h[idx*3+2]);
    };

    // Download CSR per-vertex NT data.
    std::vector<int> vnt_off(nv + 1), vnt_pi, vnt_dof;
    cudaMemcpy(vnt_off.data(), bp.vnt_offsets.ptr, (nv+1)*sizeof(int), cudaMemcpyDeviceToHost);
    {
        const int n = vnt_off[nv];
        vnt_pi.resize(n); vnt_dof.resize(n);
        if (n > 0) {
            cudaMemcpy(vnt_pi.data(),  bp.vnt_pair_idx.ptr, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(vnt_dof.data(), bp.vnt_dof.ptr,      n*sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    const int num_nt = bp.num_nt;
    std::vector<int> nt_data(num_nt * 4);
    if (num_nt > 0)
        cudaMemcpy(nt_data.data(), bp.nt_data.ptr, num_nt*4*sizeof(int), cudaMemcpyDeviceToHost);

    for (int idx = vnt_off[vi]; idx < vnt_off[vi+1]; ++idx) {
        const int pair_idx = vnt_pi[idx];
        const int dof      = vnt_dof[idx];
        const int node = nt_data[pair_idx*4+0];
        const int tv0  = nt_data[pair_idx*4+1];
        const int tv1  = nt_data[pair_idx*4+2];
        const int tv2  = nt_data[pair_idx*4+3];
        auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
            load(node), load(tv0), load(tv1), load(tv2), params.d_hat, dof);
        g += dt2 * bg;
        H += dt2 * bH;
    }

    // Download CSR per-vertex SS data.
    std::vector<int> vss_off(nv + 1), vss_pi, vss_dof;
    cudaMemcpy(vss_off.data(), bp.vss_offsets.ptr, (nv+1)*sizeof(int), cudaMemcpyDeviceToHost);
    {
        const int n = vss_off[nv];
        vss_pi.resize(n); vss_dof.resize(n);
        if (n > 0) {
            cudaMemcpy(vss_pi.data(),  bp.vss_pair_idx.ptr, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(vss_dof.data(), bp.vss_dof.ptr,      n*sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    const int num_ss = bp.num_ss;
    std::vector<int> ss_data(num_ss * 4);
    if (num_ss > 0)
        cudaMemcpy(ss_data.data(), bp.ss_data.ptr, num_ss*4*sizeof(int), cudaMemcpyDeviceToHost);

    for (int idx = vss_off[vi]; idx < vss_off[vi+1]; ++idx) {
        const int pair_idx = vss_pi[idx];
        const int dof      = vss_dof[idx];
        const int v0 = ss_data[pair_idx*4+0];
        const int v1 = ss_data[pair_idx*4+1];
        const int v2 = ss_data[pair_idx*4+2];
        const int v3 = ss_data[pair_idx*4+3];
        auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
            load(v0), load(v1), load(v2), load(v3), params.d_hat, dof);
        g += dt2 * bg;
        H += dt2 * bH;
    }

    return {g, H};
}
