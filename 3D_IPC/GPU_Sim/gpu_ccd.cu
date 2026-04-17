// gpu_ccd.cu
// Real CUDA implementation of CCD reduction kernels.
// Compile this instead of gpu_ccd_stub.cpp on a machine with a CUDA toolchain.

#include "gpu_ccd.h"
#include "ccd.h"
#include <cuda_runtime.h>
#include <vector>

// --------------------------------------------------------------------------
// atomicMinDouble
// CUDA does not provide atomicMin for doubles natively.
// This uses the standard CAS-loop trick: reinterpret as uint64 and loop until
// the value in memory is already <= val (nothing to do) or we win the CAS.
// --------------------------------------------------------------------------
__device__ static double atomicMinDouble(double* addr, double val) {
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long  old_ull  = *addr_ull;
    unsigned long long  assumed;
    do {
        assumed = old_ull;
        double old_d = __longlong_as_double(assumed);
        if (old_d <= val) break;  // already smaller, nothing to do
        old_ull = atomicCAS(addr_ull, assumed,
                            __double_as_longlong(val));
    } while (assumed != old_ull);
    return __longlong_as_double(old_ull);
}

// --------------------------------------------------------------------------
// Helper: load a Vec3 from a flat interleaved double buffer
// --------------------------------------------------------------------------
__device__ static Vec3 load_vec3(const double* buf, int vi) {
    return Vec3(buf[vi*3+0], buf[vi*3+1], buf[vi*3+2]);
}

// --------------------------------------------------------------------------
// nt_ccd_kernel
// Each thread handles one node-triangle pair and atomically reduces toi_min.
// nt_data layout: [node, tv0, tv1, tv2] per pair.
// --------------------------------------------------------------------------
__global__ static void nt_ccd_kernel(const int*    nt_data,
                                     int           num_nt,
                                     const double* x,
                                     const double* dx,
                                     double*       toi_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nt) return;

    const int node = nt_data[i*4+0];
    const int tv0  = nt_data[i*4+1];
    const int tv1  = nt_data[i*4+2];
    const int tv2  = nt_data[i*4+3];

    const double t = node_triangle_general_ccd(
        load_vec3(x,  node), load_vec3(dx, node),
        load_vec3(x,  tv0),  load_vec3(dx, tv0),
        load_vec3(x,  tv1),  load_vec3(dx, tv1),
        load_vec3(x,  tv2),  load_vec3(dx, tv2));

    atomicMinDouble(toi_out, t);
}

// --------------------------------------------------------------------------
// ss_ccd_kernel
// Each thread handles one segment-segment pair.
// ss_data layout: [v0, v1, v2, v3] per pair.
// --------------------------------------------------------------------------
__global__ static void ss_ccd_kernel(const int*    ss_data,
                                     int           num_ss,
                                     const double* x,
                                     const double* dx,
                                     double*       toi_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_ss) return;

    const int v0 = ss_data[i*4+0];
    const int v1 = ss_data[i*4+1];
    const int v2 = ss_data[i*4+2];
    const int v3 = ss_data[i*4+3];

    const double t = segment_segment_general_ccd(
        load_vec3(x,  v0), load_vec3(dx, v0),
        load_vec3(x,  v1), load_vec3(dx, v1),
        load_vec3(x,  v2), load_vec3(dx, v2),
        load_vec3(x,  v3), load_vec3(dx, v3));

    atomicMinDouble(toi_out, t);
}

// --------------------------------------------------------------------------
// gpu_ccd_min_toi
// --------------------------------------------------------------------------

double gpu_ccd_min_toi(const GPUBroadPhaseCache& bp,
                       const DeviceBuffer<double>& d_x,
                       const DeviceBuffer<double>& d_dx) {
    // Allocate a single double on the device, initialised to 1.0.
    double init = 1.0;
    double* d_toi;
    cudaMalloc(&d_toi, sizeof(double));
    cudaMemcpy(d_toi, &init, sizeof(double), cudaMemcpyHostToDevice);

    constexpr int kBlock = 256;

    if (bp.num_nt > 0) {
        const int grid = (bp.num_nt + kBlock - 1) / kBlock;
        nt_ccd_kernel<<<grid, kBlock>>>(
            bp.nt_data.ptr, bp.num_nt, d_x.ptr, d_dx.ptr, d_toi);
    }

    if (bp.num_ss > 0) {
        const int grid = (bp.num_ss + kBlock - 1) / kBlock;
        ss_ccd_kernel<<<grid, kBlock>>>(
            bp.ss_data.ptr, bp.num_ss, d_x.ptr, d_dx.ptr, d_toi);
    }

    double toi_min = 1.0;
    cudaMemcpy(&toi_min, d_toi, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_toi);
    return toi_min;
}

// --------------------------------------------------------------------------
// gpu_ccd_initial_guess
// --------------------------------------------------------------------------

std::vector<Vec3> gpu_ccd_initial_guess(const std::vector<Vec3>& x,
                                        const std::vector<Vec3>& xhat,
                                        const RefMesh& ref_mesh) {
    const int nv = static_cast<int>(x.size());

    // Compute dx on CPU.
    std::vector<Vec3> dx(nv);
    for (int i = 0; i < nv; ++i) dx[i] = xhat[i] - x[i];

    // BroadPhase build stays on CPU.
    BroadPhase ccd_bp;
    ccd_bp.build_ccd_candidates(x, dx, ref_mesh, 1.0);

    // Upload positions and pairs to device.
    std::vector<double> x_flat(nv * 3), dx_flat(nv * 3);
    for (int i = 0; i < nv; ++i) {
        x_flat[i*3+0]  = x[i](0);  x_flat[i*3+1]  = x[i](1);  x_flat[i*3+2]  = x[i](2);
        dx_flat[i*3+0] = dx[i](0); dx_flat[i*3+1] = dx[i](1); dx_flat[i*3+2] = dx[i](2);
    }
    DeviceBuffer<double> d_x, d_dx;
    d_x.upload(x_flat.data(), nv * 3);
    d_dx.upload(dx_flat.data(), nv * 3);

    GPUBroadPhaseCache gpu_bp;
    gpu_bp.upload(ccd_bp.cache(), nv);

    // GPU reduction.
    const double toi_min = gpu_ccd_min_toi(gpu_bp, d_x, d_dx);
    const double omega   = (toi_min >= 1.0) ? 1.0 : 0.9 * toi_min;

    std::vector<Vec3> xnew(nv);
    for (int i = 0; i < nv; ++i) xnew[i] = x[i] + omega * dx[i];
    return xnew;
}
