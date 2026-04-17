// gpu_ccd_stub.cpp
// CPU stub for gpu_ccd.h. Runs the CCD reductions serially on the CPU.
// On a real CUDA build, compile gpu_ccd.cu instead.

#include "gpu_ccd.h"
#include "ccd.h"
#include <algorithm>
#include <vector>

// --------------------------------------------------------------------------
// gpu_ccd_min_toi  (stub: serial CPU loop)
// --------------------------------------------------------------------------

double gpu_ccd_min_toi(const GPUBroadPhaseCache& bp,
                       const DeviceBuffer<double>& d_x,
                       const DeviceBuffer<double>& d_dx) {
    // Download positions back to CPU so we can call the existing CCD math.
    std::vector<double> x_cpu(d_x.count);
    std::vector<double> dx_cpu(d_dx.count);
    d_x.download(x_cpu.data());
    d_dx.download(dx_cpu.data());

    // Download pair lists.
    std::vector<int> nt_cpu(bp.nt_data.count);
    std::vector<int> ss_cpu(bp.ss_data.count);
    if (bp.num_nt > 0) bp.nt_data.download(nt_cpu.data());
    if (bp.num_ss > 0) bp.ss_data.download(ss_cpu.data());

    auto get_pos = [&](const std::vector<double>& buf, int vi) {
        return Vec3(buf[vi*3+0], buf[vi*3+1], buf[vi*3+2]);
    };

    double toi_min = 1.0;

    for (int i = 0; i < bp.num_nt; ++i) {
        int node = nt_cpu[i*4+0];
        int tv0  = nt_cpu[i*4+1];
        int tv1  = nt_cpu[i*4+2];
        int tv2  = nt_cpu[i*4+3];
        double t = node_triangle_general_ccd(
            get_pos(x_cpu, node),  get_pos(dx_cpu, node),
            get_pos(x_cpu, tv0),   get_pos(dx_cpu, tv0),
            get_pos(x_cpu, tv1),   get_pos(dx_cpu, tv1),
            get_pos(x_cpu, tv2),   get_pos(dx_cpu, tv2));
        toi_min = std::min(toi_min, t);
    }

    for (int i = 0; i < bp.num_ss; ++i) {
        int v0 = ss_cpu[i*4+0];
        int v1 = ss_cpu[i*4+1];
        int v2 = ss_cpu[i*4+2];
        int v3 = ss_cpu[i*4+3];
        double t = segment_segment_general_ccd(
            get_pos(x_cpu, v0), get_pos(dx_cpu, v0),
            get_pos(x_cpu, v1), get_pos(dx_cpu, v1),
            get_pos(x_cpu, v2), get_pos(dx_cpu, v2),
            get_pos(x_cpu, v3), get_pos(dx_cpu, v3));
        toi_min = std::min(toi_min, t);
    }

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

    // Upload positions and pairs to "device" (stub: CPU heap).
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

    // GPU reduction (stub: serial CPU).
    const double toi_min = gpu_ccd_min_toi(gpu_bp, d_x, d_dx);
    const double omega   = (toi_min >= 1.0) ? 1.0 : 0.9 * toi_min;

    std::vector<Vec3> xnew(nv);
    for (int i = 0; i < nv; ++i) xnew[i] = x[i] + omega * dx[i];
    return xnew;
}
