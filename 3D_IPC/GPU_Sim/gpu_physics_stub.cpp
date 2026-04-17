// gpu_physics_stub.cpp
// CPU stub for gpu_physics.h.
// DeviceBuffer::ptr is a plain CPU heap pointer in the stub so all
// device array accesses are legal without any cudaMemcpy.

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

    const double* x    = d_x.ptr;
    const double* xhat = d_xhat.ptr;
    const double  dt2  = params.dt2();

    Vec3  g = Vec3::Zero();
    Mat33 H = Mat33::Zero();

    // --- inertia ---
    const double mass_vi = mesh.mass.ptr[vi];
    const Vec3 xi    (x   [vi*3], x   [vi*3+1], x   [vi*3+2]);
    const Vec3 xhati (xhat[vi*3], xhat[vi*3+1], xhat[vi*3+2]);
    const Vec3 grav  (params.gx, params.gy, params.gz);

    g += mass_vi * (xi - xhati);
    g += dt2 * (-mass_vi * grav);
    H += mass_vi * Mat33::Identity();

    // --- pin spring ---
    const int pi = pin_map.data.ptr[vi];
    if (pi >= 0) {
        const Vec3 tgt(pins.targets.ptr[pi*3],
                       pins.targets.ptr[pi*3+1],
                       pins.targets.ptr[pi*3+2]);
        g += dt2 * params.kpin * (xi - tgt);
        H += dt2 * params.kpin * Mat33::Identity();
    }

    // --- corotated elastic over incident triangles (CSR adjacency) ---
    for (int idx = adj.offsets.ptr[vi]; idx < adj.offsets.ptr[vi+1]; ++idx) {
        const int ti = adj.tri_idx.ptr[idx];
        const int a  = adj.tri_local.ptr[idx];

        const int v0 = mesh.tris.ptr[ti*3+0];
        const int v1 = mesh.tris.ptr[ti*3+1];
        const int v2 = mesh.tris.ptr[ti*3+2];

        TriangleDef def;
        def.x[0] = Vec3(x[v0*3], x[v0*3+1], x[v0*3+2]);
        def.x[1] = Vec3(x[v1*3], x[v1*3+1], x[v1*3+2]);
        def.x[2] = Vec3(x[v2*3], x[v2*3+1], x[v2*3+2]);

        // Dm_inv stored column-major: [a00, a10, a01, a11]
        const double* dm = mesh.Dm_inv.ptr + ti * 4;
        Mat22 Dm_inv;
        Dm_inv(0,0) = dm[0]; Dm_inv(1,0) = dm[1];
        Dm_inv(0,1) = dm[2]; Dm_inv(1,1) = dm[3];

        const double A = mesh.area.ptr[ti];

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
        for (int idx = mesh.hinge_adj_offsets.ptr[vi];
                 idx < mesh.hinge_adj_offsets.ptr[vi+1]; ++idx) {
            const int hi   = mesh.hinge_adj_hi.ptr[idx];
            const int role = mesh.hinge_adj_role.ptr[idx];

            HingeDef hdef;
            for (int k = 0; k < 4; ++k) {
                const int hv = mesh.hinge_v.ptr[hi*4+k];
                hdef.x[k] = Vec3(x[hv*3], x[hv*3+1], x[hv*3+2]);
            }
            const double bar_theta = mesh.hinge_bar_theta.ptr[hi];
            const double ce        = mesh.hinge_ce.ptr[hi];

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

    const double* x   = d_x.ptr;
    const double  dt2 = params.dt2();

    auto load = [&](int idx) {
        return Vec3(x[idx*3], x[idx*3+1], x[idx*3+2]);
    };

    // --- node-triangle barrier (CSR per-vertex lookup) ---
    for (int idx = bp.vnt_offsets.ptr[vi]; idx < bp.vnt_offsets.ptr[vi+1]; ++idx) {
        const int pair_idx = bp.vnt_pair_idx.ptr[idx];
        const int dof      = bp.vnt_dof.ptr[idx];

        const int node = bp.nt_data.ptr[pair_idx*4+0];
        const int tv0  = bp.nt_data.ptr[pair_idx*4+1];
        const int tv1  = bp.nt_data.ptr[pair_idx*4+2];
        const int tv2  = bp.nt_data.ptr[pair_idx*4+3];

        auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
            load(node), load(tv0), load(tv1), load(tv2), params.d_hat, dof);
        g += dt2 * bg;
        H += dt2 * bH;
    }

    // --- segment-segment barrier (CSR per-vertex lookup) ---
    for (int idx = bp.vss_offsets.ptr[vi]; idx < bp.vss_offsets.ptr[vi+1]; ++idx) {
        const int pair_idx = bp.vss_pair_idx.ptr[idx];
        const int dof      = bp.vss_dof.ptr[idx];

        const int v0 = bp.ss_data.ptr[pair_idx*4+0];
        const int v1 = bp.ss_data.ptr[pair_idx*4+1];
        const int v2 = bp.ss_data.ptr[pair_idx*4+2];
        const int v3 = bp.ss_data.ptr[pair_idx*4+3];

        auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
            load(v0), load(v1), load(v2), load(v3), params.d_hat, dof);
        g += dt2 * bg;
        H += dt2 * bH;
    }

    return {g, H};
}
