// gpu_physics_stub.cpp
// CPU implementation of gpu_physics.h.
// Uses DeviceBuffer::to_cpu() so it works whether ptr is CPU heap (stub) or
// GPU device memory (real CUDA build).

#include "gpu_physics.h"
#include "barrier_energy.h"
#include "corotated_energy.h"
#include "bending_energy.h"
#include "IPC_math.h"

std::pair<Vec3, Mat33> gpu_compute_local_gradient_and_hessian_no_barrier(
    int vi,
    const GPURefMesh&         mesh,
    const GPUAdjacency&       adj,
    const GPUPins&            pins,
    const GPUPinMap&          pin_map,
    const GPUSimParams&       params,
    const DeviceBuffer<double>& d_x,
    const DeviceBuffer<double>& d_xhat) {

    // Download all device buffers to CPU vectors.
    const auto x_cpu    = d_x.to_cpu();
    const auto xhat_cpu = d_xhat.to_cpu();
    const double* x    = x_cpu.data();
    const double* xhat = xhat_cpu.data();

    const auto mass_cpu          = mesh.mass.to_cpu();
    const auto tris_cpu          = mesh.tris.to_cpu();
    const auto Dm_inv_cpu        = mesh.Dm_inv.to_cpu();
    const auto area_cpu          = mesh.area.to_cpu();
    const auto adj_offsets_cpu   = adj.offsets.to_cpu();
    const auto adj_tri_idx_cpu   = adj.tri_idx.to_cpu();
    const auto adj_tri_local_cpu = adj.tri_local.to_cpu();
    const auto pin_map_cpu       = pin_map.data.to_cpu();
    const auto pin_targets_cpu   = pins.targets.to_cpu();

    const double dt2 = params.dt2();

    Vec3  g = Vec3::Zero();
    Mat33 H = Mat33::Zero();

    // --- inertia ---
    const double mass_vi = mass_cpu[vi];
    const Vec3 xi    (x   [vi*3], x   [vi*3+1], x   [vi*3+2]);
    const Vec3 xhati (xhat[vi*3], xhat[vi*3+1], xhat[vi*3+2]);
    const Vec3 grav  (params.gx, params.gy, params.gz);

    g += mass_vi * (xi - xhati);
    g += dt2 * (-mass_vi * grav);
    H += mass_vi * Mat33::Identity();

    // --- pin spring ---
    const int pi = pin_map_cpu[vi];
    if (pi >= 0) {
        const Vec3 tgt(pin_targets_cpu[pi*3],
                       pin_targets_cpu[pi*3+1],
                       pin_targets_cpu[pi*3+2]);
        g += dt2 * params.kpin * (xi - tgt);
        H += dt2 * params.kpin * Mat33::Identity();
    }

    // --- corotated elastic over incident triangles (CSR adjacency) ---
    for (int idx = adj_offsets_cpu[vi]; idx < adj_offsets_cpu[vi+1]; ++idx) {
        const int ti = adj_tri_idx_cpu[idx];
        const int a  = adj_tri_local_cpu[idx];

        const int v0 = tris_cpu[ti*3+0];
        const int v1 = tris_cpu[ti*3+1];
        const int v2 = tris_cpu[ti*3+2];

        TriangleDef def;
        def.x[0] = Vec3(x[v0*3], x[v0*3+1], x[v0*3+2]);
        def.x[1] = Vec3(x[v1*3], x[v1*3+1], x[v1*3+2]);
        def.x[2] = Vec3(x[v2*3], x[v2*3+1], x[v2*3+2]);

        // Dm_inv stored column-major: [a00, a10, a01, a11]
        const double* dm = Dm_inv_cpu.data() + ti * 4;
        Mat22 Dm_inv;
        Dm_inv(0,0) = dm[0]; Dm_inv(1,0) = dm[1];
        Dm_inv(0,1) = dm[2]; Dm_inv(1,1) = dm[3];

        const double A = area_cpu[ti];

        Mat32 Ds;
        Ds.col(0) = def.x[1] - def.x[0];
        Ds.col(1) = def.x[2] - def.x[0];
        const Mat32 F = Ds * Dm_inv;

        const CorotatedCache32 cache = buildCorotatedCache(F);
        const ShapeGrads gradN = shape_function_gradients(Dm_inv);
        const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
        Mat66 dPdF;
        dPdFCorotated32(cache, params.mu, params.lambda, dPdF);

        g += dt2 * corotated_node_gradient(P, A, gradN, a);
        H += dt2 * corotated_node_hessian(dPdF, A, gradN, a);
    }

    // --- bending (CSR hinge adjacency, only when kB > 0) ---
    if (params.kB > 0.0) {
        const auto hinge_adj_offsets_cpu = mesh.hinge_adj_offsets.to_cpu();
        const auto hinge_adj_hi_cpu      = mesh.hinge_adj_hi.to_cpu();
        const auto hinge_adj_role_cpu    = mesh.hinge_adj_role.to_cpu();
        const auto hinge_v_cpu           = mesh.hinge_v.to_cpu();
        const auto hinge_bar_theta_cpu   = mesh.hinge_bar_theta.to_cpu();
        const auto hinge_ce_cpu          = mesh.hinge_ce.to_cpu();

        for (int idx = hinge_adj_offsets_cpu[vi];
                 idx < hinge_adj_offsets_cpu[vi+1]; ++idx) {
            const int hi   = hinge_adj_hi_cpu[idx];
            const int role = hinge_adj_role_cpu[idx];

            HingeDef hdef;
            for (int k = 0; k < 4; ++k) {
                const int hv = hinge_v_cpu[hi*4+k];
                hdef.x[k] = Vec3(x[hv*3], x[hv*3+1], x[hv*3+2]);
            }
            const double bar_theta = hinge_bar_theta_cpu[hi];
            const double ce        = hinge_ce_cpu[hi];

            g += dt2 * bending_node_gradient(hdef, params.kB, ce, bar_theta, role);
            H += dt2 * bending_node_hessian_psd(hdef, params.kB, ce, bar_theta, role);
        }
    }

    return {g, H};
}

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

    const auto x_cpu = d_x.to_cpu();
    const double* x  = x_cpu.data();
    const double  dt2 = params.dt2();

    const auto vnt_offsets_cpu  = bp.vnt_offsets.to_cpu();
    const auto vnt_pair_idx_cpu = bp.vnt_pair_idx.to_cpu();
    const auto vnt_dof_cpu      = bp.vnt_dof.to_cpu();
    const auto nt_data_cpu      = bp.nt_data.to_cpu();
    const auto vss_offsets_cpu  = bp.vss_offsets.to_cpu();
    const auto vss_pair_idx_cpu = bp.vss_pair_idx.to_cpu();
    const auto vss_dof_cpu      = bp.vss_dof.to_cpu();
    const auto ss_data_cpu      = bp.ss_data.to_cpu();

    auto load = [&](int idx) {
        return Vec3(x[idx*3], x[idx*3+1], x[idx*3+2]);
    };

    // --- node-triangle barrier (CSR per-vertex lookup) ---
    for (int idx = vnt_offsets_cpu[vi]; idx < vnt_offsets_cpu[vi+1]; ++idx) {
        const int pair_idx = vnt_pair_idx_cpu[idx];
        const int dof      = vnt_dof_cpu[idx];

        const int node = nt_data_cpu[pair_idx*4+0];
        const int tv0  = nt_data_cpu[pair_idx*4+1];
        const int tv1  = nt_data_cpu[pair_idx*4+2];
        const int tv2  = nt_data_cpu[pair_idx*4+3];

        auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
            load(node), load(tv0), load(tv1), load(tv2), params.d_hat, dof);
        g += dt2 * bg;
        H += dt2 * bH;
    }

    // --- segment-segment barrier (CSR per-vertex lookup) ---
    for (int idx = vss_offsets_cpu[vi]; idx < vss_offsets_cpu[vi+1]; ++idx) {
        const int pair_idx = vss_pair_idx_cpu[idx];
        const int dof      = vss_dof_cpu[idx];

        const int v0 = ss_data_cpu[pair_idx*4+0];
        const int v1 = ss_data_cpu[pair_idx*4+1];
        const int v2 = ss_data_cpu[pair_idx*4+2];
        const int v3 = ss_data_cpu[pair_idx*4+3];

        auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
            load(v0), load(v1), load(v2), load(v3), params.d_hat, dof);
        g += dt2 * bg;
        H += dt2 * bH;
    }

    return {g, H};
}
