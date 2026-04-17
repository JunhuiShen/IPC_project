#include "GPU_Sim/gpu_physics.h"
#include "GPU_Sim/gpu_mesh.h"
#include "physics.h"
#include "make_shape.h"
#include "broad_phase.h"
#include "barrier_energy.h"
#include "solver.h"

#include <gtest/gtest.h>

namespace {

constexpr double kTol = 1e-10;

// Build a small square mesh and upload all GPU structs.
struct Scene {
    RefMesh          ref_mesh;
    DeformedState    state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    SimParams        params;

    GPURefMesh       gpu_mesh;
    GPUAdjacency     gpu_adj;
    GPUPins          gpu_pins;
    GPUPinMap        gpu_pin_map;
    GPUSimParams     gpu_params;
    DeviceBuffer<double> d_x, d_xhat;

    VertexTriangleMap adj;
    PinMap            pin_map;
    std::vector<Vec3> xhat;

    Scene(bool with_bending = false, bool with_pins = false) {
        params.fps       = 30.0;
        params.substeps  = 1;
        params.mu        = 5.0;
        params.lambda    = 5.0;
        params.density   = 1.0;
        params.thickness = 0.1;
        params.kpin      = 1e5;
        params.kB        = with_bending ? 1e-3 : 0.0;
        params.gravity   = Vec3(0.0, -9.81, 0.0);

        clear_model(ref_mesh, state, X, pins);
        build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
        state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
        ref_mesh.build_lumped_mass(params.density, params.thickness);

        if (with_pins) {
            append_pin(pins, 0, state.deformed_positions);
            append_pin(pins, static_cast<int>(state.deformed_positions.size()) - 1,
                       state.deformed_positions);
        }

        // Perturb positions slightly so gradients are non-trivial.
        xhat = state.deformed_positions;
        for (int i = 0; i < static_cast<int>(xhat.size()); ++i)
            xhat[i] += Vec3(0.01 * i, -0.005 * i, 0.003 * i) * params.dt();

        adj     = build_incident_triangle_map(ref_mesh.tris);
        pin_map = build_pin_map(pins, static_cast<int>(state.deformed_positions.size()));

        // Upload to GPU structs.
        gpu_mesh.upload(ref_mesh);
        gpu_adj.upload(adj, static_cast<int>(state.deformed_positions.size()));
        gpu_pins.upload(pins);
        gpu_pin_map.upload(pin_map);
        gpu_params = GPUSimParams::from(params);

        const auto& x_cpu = state.deformed_positions;
        const int nv = static_cast<int>(x_cpu.size());
        std::vector<double> x_flat(nv * 3), xhat_flat(nv * 3);
        for (int i = 0; i < nv; ++i) {
            x_flat[i*3+0]    = x_cpu[i](0); x_flat[i*3+1]    = x_cpu[i](1); x_flat[i*3+2]    = x_cpu[i](2);
            xhat_flat[i*3+0] = xhat[i](0);  xhat_flat[i*3+1] = xhat[i](1);  xhat_flat[i*3+2] = xhat[i](2);
        }
        d_x.upload(x_flat.data(), nv * 3);
        d_xhat.upload(xhat_flat.data(), nv * 3);
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// GradientMatchesCPU — no bending, no pins
// ---------------------------------------------------------------------------
TEST(GPUPhysics, GradientMatchesCPU) {
    Scene s;
    const int nv = static_cast<int>(s.state.deformed_positions.size());

    for (int vi = 0; vi < nv; ++vi) {
        auto [g_cpu, H_cpu] = compute_local_gradient_and_hessian_no_barrier(
            vi, s.ref_mesh, s.adj, s.pins, s.params,
            s.state.deformed_positions, s.xhat, &s.pin_map);

        auto [g_gpu, H_gpu] = gpu_compute_local_gradient_and_hessian_no_barrier(
            vi, s.gpu_mesh, s.gpu_adj, s.gpu_pins, s.gpu_pin_map, s.gpu_params,
            s.d_x, s.d_xhat);

        EXPECT_NEAR(g_cpu(0), g_gpu(0), kTol) << "vi=" << vi << " gx";
        EXPECT_NEAR(g_cpu(1), g_gpu(1), kTol) << "vi=" << vi << " gy";
        EXPECT_NEAR(g_cpu(2), g_gpu(2), kTol) << "vi=" << vi << " gz";

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(H_cpu(r,c), H_gpu(r,c), kTol)
                    << "vi=" << vi << " H[" << r << "," << c << "]";
    }
}

// ---------------------------------------------------------------------------
// GradientMatchesCPU_WithPins
// ---------------------------------------------------------------------------
TEST(GPUPhysics, GradientMatchesCPUWithPins) {
    Scene s(false, true);
    const int nv = static_cast<int>(s.state.deformed_positions.size());

    for (int vi = 0; vi < nv; ++vi) {
        auto [g_cpu, H_cpu] = compute_local_gradient_and_hessian_no_barrier(
            vi, s.ref_mesh, s.adj, s.pins, s.params,
            s.state.deformed_positions, s.xhat, &s.pin_map);

        auto [g_gpu, H_gpu] = gpu_compute_local_gradient_and_hessian_no_barrier(
            vi, s.gpu_mesh, s.gpu_adj, s.gpu_pins, s.gpu_pin_map, s.gpu_params,
            s.d_x, s.d_xhat);

        EXPECT_NEAR(g_cpu(0), g_gpu(0), kTol) << "vi=" << vi << " gx";
        EXPECT_NEAR(g_cpu(1), g_gpu(1), kTol) << "vi=" << vi << " gy";
        EXPECT_NEAR(g_cpu(2), g_gpu(2), kTol) << "vi=" << vi << " gz";

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(H_cpu(r,c), H_gpu(r,c), kTol)
                    << "vi=" << vi << " H[" << r << "," << c << "]";
    }
}

// ---------------------------------------------------------------------------
// GradientMatchesCPU_WithBending
// ---------------------------------------------------------------------------
TEST(GPUPhysics, GradientMatchesCPUWithBending) {
    Scene s(true, false);
    const int nv = static_cast<int>(s.state.deformed_positions.size());

    for (int vi = 0; vi < nv; ++vi) {
        auto [g_cpu, H_cpu] = compute_local_gradient_and_hessian_no_barrier(
            vi, s.ref_mesh, s.adj, s.pins, s.params,
            s.state.deformed_positions, s.xhat, &s.pin_map);

        auto [g_gpu, H_gpu] = gpu_compute_local_gradient_and_hessian_no_barrier(
            vi, s.gpu_mesh, s.gpu_adj, s.gpu_pins, s.gpu_pin_map, s.gpu_params,
            s.d_x, s.d_xhat);

        EXPECT_NEAR(g_cpu(0), g_gpu(0), kTol) << "vi=" << vi << " gx";
        EXPECT_NEAR(g_cpu(1), g_gpu(1), kTol) << "vi=" << vi << " gy";
        EXPECT_NEAR(g_cpu(2), g_gpu(2), kTol) << "vi=" << vi << " gz";

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(H_cpu(r,c), H_gpu(r,c), kTol)
                    << "vi=" << vi << " H[" << r << "," << c << "]";
    }
}

// ---------------------------------------------------------------------------
// GradientMatchesCPU_WithBarrier
// Two close parallel sheets so BroadPhase generates barrier pairs.
// ---------------------------------------------------------------------------
TEST(GPUPhysics, GradientMatchesCPUWithBarrier) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;

    SimParams params;
    params.fps       = 30.0;
    params.substeps  = 1;
    params.mu        = 5.0;
    params.lambda    = 5.0;
    params.density   = 1.0;
    params.thickness = 0.1;
    params.kpin      = 1e5;
    params.kB        = 0.0;
    params.d_hat     = 0.5;
    params.gravity   = Vec3(0.0, -9.81, 0.0);

    clear_model(ref_mesh, state, X, pins);
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
    // Offset second sheet in y (perpendicular to the mesh plane) so that
    // the two sheets are parallel and within barrier range.
    const double sep = params.d_hat * 0.4;
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, sep, 0.0));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_lumped_mass(params.density, params.thickness);

    const int nv = static_cast<int>(state.deformed_positions.size());
    std::vector<Vec3> xhat = state.deformed_positions;
    for (int i = 0; i < nv; ++i)
        xhat[i] += Vec3(0.001 * i, -0.0005 * i, 0.0003 * i) * params.dt();

    auto adj     = build_incident_triangle_map(ref_mesh.tris);
    auto pin_map = build_pin_map(pins, nv);

    BroadPhase bp;
    bp.initialize(state.deformed_positions, state.velocities, ref_mesh, params.dt(), params.d_hat);
    const auto& bp_cache = bp.cache();

    GPURefMesh   gpu_mesh;    gpu_mesh.upload(ref_mesh);
    GPUAdjacency gpu_adj;     gpu_adj.upload(adj, nv);
    GPUPins      gpu_pins;    gpu_pins.upload(pins);
    GPUPinMap    gpu_pin_map; gpu_pin_map.upload(pin_map);
    GPUSimParams gpu_params = GPUSimParams::from(params);
    GPUBroadPhaseCache gpu_bp;
    gpu_bp.upload(bp_cache, nv);

    std::vector<double> x_flat(nv * 3), xhat_flat(nv * 3);
    for (int i = 0; i < nv; ++i) {
        x_flat[i*3+0] = state.deformed_positions[i](0);
        x_flat[i*3+1] = state.deformed_positions[i](1);
        x_flat[i*3+2] = state.deformed_positions[i](2);
        xhat_flat[i*3+0] = xhat[i](0);
        xhat_flat[i*3+1] = xhat[i](1);
        xhat_flat[i*3+2] = xhat[i](2);
    }
    DeviceBuffer<double> d_x, d_xhat;
    d_x.upload(x_flat.data(), nv * 3);
    d_xhat.upload(xhat_flat.data(), nv * 3);

    const double dt2 = params.dt2();
    const auto& x = state.deformed_positions;

    for (int vi = 0; vi < nv; ++vi) {
        auto [g_cpu, H_cpu] = compute_local_gradient_and_hessian_no_barrier(
            vi, ref_mesh, adj, pins, params, x, xhat, &pin_map);
        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g_cpu += dt2 * bg;
            H_cpu += dt2 * bH;
        }
        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
                x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g_cpu += dt2 * bg;
            H_cpu += dt2 * bH;
        }

        auto [g_gpu, H_gpu] = gpu_compute_local_gradient_and_hessian(
            vi, gpu_mesh, gpu_adj, gpu_pins, gpu_pin_map, gpu_params, gpu_bp, d_x, d_xhat);

        EXPECT_NEAR(g_cpu(0), g_gpu(0), kTol) << "vi=" << vi << " gx";
        EXPECT_NEAR(g_cpu(1), g_gpu(1), kTol) << "vi=" << vi << " gy";
        EXPECT_NEAR(g_cpu(2), g_gpu(2), kTol) << "vi=" << vi << " gz";
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(H_cpu(r,c), H_gpu(r,c), kTol)
                    << "vi=" << vi << " H[" << r << "," << c << "]";
    }
}
