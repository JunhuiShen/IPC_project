#include <gtest/gtest.h>

#include "make_shape.h"
#include "physics.h"
#include "barrier_energy.h"
#include "broad_phase.h"
#include "rigid_body_ipc.h"
#include "simulation.h"
#include "solver.h"

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

using VecX = Eigen::VectorXd;

namespace {

// =====================================================================
//  Total energy: no_barrier base + barrier for active pairs
// =====================================================================

double total_energy(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params,
                    const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                    const std::vector<NodeTrianglePair>& nt_pairs,
                    const std::vector<SegmentSegmentPair>& ss_pairs){
    double E = compute_incremental_potential_no_barrier(ref_mesh, pins, params, x, xhat);
    double dt2 = params.dt() * params.dt();

    for (const auto& p : nt_pairs)
        E += dt2 * node_triangle_barrier(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat);

    for (const auto& p : ss_pairs)
        E += dt2 * segment_segment_barrier(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat);

    return E;
}

// =====================================================================
//  Local gradient: no_barrier base + barrier for pairs involving vi
// =====================================================================

Vec3 local_gradient(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                    const std::vector<Pin>& pins, const SimParams& params,
                    const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                    const std::vector<NodeTrianglePair>& nt_pairs,
                    const std::vector<SegmentSegmentPair>& ss_pairs){
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat);
    double dt2 = params.dt() * params.dt();

    for (const auto& p : nt_pairs) {
        // dof: 0=node, 1=tri_v[0], 2=tri_v[1], 3=tri_v[2]
        int dof = -1;
        if      (vi == p.node)      dof = 0;
        else if (vi == p.tri_v[0])  dof = 1;
        else if (vi == p.tri_v[1])  dof = 2;
        else if (vi == p.tri_v[2])  dof = 3;
        if (dof < 0) continue;
        g += dt2 * node_triangle_barrier_gradient(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
    }

    for (const auto& p : ss_pairs) {
        // dof: 0=v[0], 1=v[1], 2=v[2], 3=v[3]
        int dof = -1;
        if      (vi == p.v[0]) dof = 0;
        else if (vi == p.v[1]) dof = 1;
        else if (vi == p.v[2]) dof = 2;
        else if (vi == p.v[3]) dof = 3;
        if (dof < 0) continue;
        g += dt2 * segment_segment_barrier_gradient(
                x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, dof);
    }

    return g;
}

// =====================================================================
//  Local Hessian (diagonal block): no_barrier base + barrier
// =====================================================================

Mat33 local_hessian(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                    const std::vector<Pin>& pins, const SimParams& params,
                    const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                    const std::vector<NodeTrianglePair>& nt_pairs,
                    const std::vector<SegmentSegmentPair>& ss_pairs){
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat);
    double dt2 = params.dt() * params.dt();

    for (const auto& p : nt_pairs) {
        int dof = -1;
        if      (vi == p.node)      dof = 0;
        else if (vi == p.tri_v[0])  dof = 1;
        else if (vi == p.tri_v[1])  dof = 2;
        else if (vi == p.tri_v[2])  dof = 3;
        if (dof < 0) continue;
        H += dt2 * node_triangle_barrier_self_hessian(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
    }

    for (const auto& p : ss_pairs) {
        int dof = -1;
        if      (vi == p.v[0]) dof = 0;
        else if (vi == p.v[1]) dof = 1;
        else if (vi == p.v[2]) dof = 2;
        else if (vi == p.v[3]) dof = 3;
        if (dof < 0) continue;
        H += dt2 * segment_segment_barrier_self_hessian(
                x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, dof);
    }

    return H;
}

// =====================================================================
//  FD helpers
// =====================================================================

Vec3 local_gradient_fd(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                       const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                       const std::vector<NodeTrianglePair>& nt_pairs,
                       const std::vector<SegmentSegmentPair>& ss_pairs, double eps){
    Vec3 gfd = Vec3::Zero();
    for (int d = 0; d < 3; ++d) {
        auto xp = x, xm = x;
        xp[vi](d) += eps; xm[vi](d) -= eps;
        double Ep = total_energy(ref_mesh, pins, params, xp, xhat, nt_pairs, ss_pairs);
        double Em = total_energy(ref_mesh, pins, params, xm, xhat, nt_pairs, ss_pairs);
        gfd(d) = (Ep - Em) / (2.0 * eps);
    }
    return gfd;
}

Mat33 local_hessian_fd(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                       const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                       const std::vector<NodeTrianglePair>& nt_pairs,
                       const std::vector<SegmentSegmentPair>& ss_pairs, double eps){
    Mat33 Hfd = Mat33::Zero();
    for (int d = 0; d < 3; ++d) {
        auto xp = x, xm = x;
        xp[vi](d) += eps; xm[vi](d) -= eps;
        Vec3 gp = local_gradient(vi, ref_mesh, adj, pins, params, xp, xhat, nt_pairs, ss_pairs);
        Vec3 gm = local_gradient(vi, ref_mesh, adj, pins, params, xm, xhat, nt_pairs, ss_pairs);
        Hfd.col(d) = (gp - gm) / (2.0 * eps);
    }
    return Hfd;
}

} // anonymous namespace

// =====================================================================
//  Test fixture
// =====================================================================

class TotalEnergyTest : public ::testing::Test {
protected:
    SimParams params = SimParams::zeros();
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    VertexTriangleMap adj;
    std::vector<Vec3> xhat;
    std::vector<Vec3> x;
    std::vector<NodeTrianglePair> nt_pairs;
    std::vector<SegmentSegmentPair> ss_pairs;

    void SetUp() override {
        std::cout << std::setprecision(12);

        params.fps      = 30.0;
        params.substeps = 1;
        params.mu = 100.0;
        params.lambda = 100.0;
        params.density = 1.0;
        params.thickness = 0.1;
        params.kpin = 1e3;
        params.gravity = Vec3(0, -9.81, 0);
        params.max_global_iters = 50;
        params.tol_abs = 1e-8;
        params.d_hat = 1.0;

        clear_model(ref_mesh, state, X, pins);

        build_square_mesh(ref_mesh, state, X, 1, 1, 1.0, 1.0, Vec3(0, 0, 0));
        build_square_mesh(ref_mesh, state, X, 1, 1, 1.0, 1.0, Vec3(0, 0, 0.4));

        state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

        state.deformed_positions[0] += Vec3(0.02, -0.01,  0.03);
        state.deformed_positions[1] += Vec3(-0.01, 0.03, -0.02);
        state.deformed_positions[2] += Vec3(0.01,  0.02,  0.01);
        state.deformed_positions[3] += Vec3(-0.02,-0.01,  0.02);
        state.deformed_positions[4] += Vec3(0.02,  0.01, -0.02);
        state.deformed_positions[5] += Vec3(-0.01, 0.02,  0.01);
        state.deformed_positions[6] += Vec3(0.01, -0.01,  0.02);
        state.deformed_positions[7] += Vec3(-0.02, 0.01, -0.01);

        append_pin(pins, 6, state.deformed_positions);
        append_pin(pins, 7, state.deformed_positions);

        ref_mesh.build_lumped_mass(params.density, params.thickness);
        adj = build_incident_triangle_map(ref_mesh.tris);

        xhat = state.deformed_positions;
        x    = state.deformed_positions;

        nt_pairs.push_back({4, {0, 1, 3}});
        nt_pairs.push_back({5, {0, 1, 3}});

        ss_pairs.push_back({{0, 1, 4, 5}});
        ss_pairs.push_back({{0, 3, 4, 7}});
    }
};

// -----------------------------------------------------------------
//  Per-vertex Hessian check
// -----------------------------------------------------------------

TEST_F(TotalEnergyTest, CachedIncidentRowsAndRestGradientsMatchFallbackExactly) {
    std::vector<IncidentTriangles> dense_incident(x.size());
    for (const auto& [vi, row] : adj)
        dense_incident[vi] = row;

    std::vector<ShapeGrads> rest_shape_grads(ref_mesh.Dm_inverse.size());
    for (int ti = 0; ti < static_cast<int>(ref_mesh.Dm_inverse.size()); ++ti)
        rest_shape_grads[ti] = shape_function_gradients(ref_mesh.Dm_inverse[ti]);

    for (int vi = 0; vi < static_cast<int>(x.size()); ++vi) {
        const auto [g_fallback, H_fallback] =
                compute_local_gradient_and_hessian_no_barrier(
                        vi, ref_mesh, adj, pins, params, x, xhat);
        const auto [g_cached, H_cached] =
                compute_local_gradient_and_hessian_no_barrier(
                        vi, ref_mesh, adj, pins, params, x, xhat, nullptr,
                        &dense_incident[vi], &rest_shape_grads);

        EXPECT_EQ((g_cached - g_fallback).cwiseAbs().maxCoeff(), 0.0)
                << "gradient mismatch at vertex " << vi;
        EXPECT_EQ((H_cached - H_fallback).cwiseAbs().maxCoeff(), 0.0)
                << "Hessian mismatch at vertex " << vi;
    }
}

TEST_F(TotalEnergyTest, PerVertexHessian) {
    double eps = 1e-6;
    for (int vi = 0; vi < (int)x.size(); ++vi) {
        Mat33 H   = local_hessian(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
        Mat33 Hfd = local_hessian_fd(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs, eps);
        double err = (H - Hfd).lpNorm<Eigen::Infinity>();
        std::cout << "v " << vi << " err=" << err << "\n";
        ASSERT_LT(err, 1e-3) << "Hessian mismatch at vertex " << vi;
    }
}

// =====================================================================
//  Bending enabled: check the bending term is correctly wired into the
//  total incremental potential and per-vertex gradient alongside elastic,
//  barrier, pinning, gravity, and inertia.
// =====================================================================
//
// NOTE: at non-rest configurations the bending Hessian block uses the
// Gauss-Newton PSD approximation (see bending_node_hessian_psd), which
// differs from the true Hessian by 2*k_B*c_e*delta * d^2 theta. We
// therefore only check gradient consistency here; bending Hessian
// consistency is covered at the rest state below.

TEST_F(TotalEnergyTest, PerVertexGradientWithBending) {
    params.kB = 50.0;  // enable bending on top of the existing config

    const double eps = 1e-6;
    double max_err = 0.0;
    int worst_vi = -1;

    for (int vi = 0; vi < (int)x.size(); ++vi) {
        Vec3 g   = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
        Vec3 gfd = local_gradient_fd(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs, eps);
        const double err = (g - gfd).norm();
        if (err > max_err) { max_err = err; worst_vi = vi; }
    }
    std::cout << "bending: worst vertex=" << worst_vi << " err=" << max_err << "\n";
    ASSERT_LT(max_err, 1e-4) << "gradient mismatch with bending enabled";
}

// =====================================================================
//  compute_global_residual (production path) vs FD
//
//  Compares the production gradient code reached via compute_global_residual
//  against both an independently assembled test gradient and direct finite
//  differences of total_energy. Any drift between the paths fails here.
// =====================================================================

TEST_F(TotalEnergyTest, GlobalResidualMatchesFiniteDifference) {
    const int nv = static_cast<int>(x.size());

    BroadPhase bp;
    bp.initialize(x, state.velocities, ref_mesh, params.dt(), params.d_hat);

    const auto& bp_nt = bp.cache().nt_pairs;
    const auto& bp_ss = bp.cache().ss_pairs;
    ASSERT_FALSE(bp_nt.empty() && bp_ss.empty())
        << "broad phase produced no barrier pairs; test would degenerate";

    const double r_prod = compute_global_residual(ref_mesh, adj, pins, params, x, xhat, bp, nullptr);

    VecX g_test(3 * nv);
    for (int vi = 0; vi < nv; ++vi)
        g_test.segment<3>(3*vi) = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, bp_nt, bp_ss);

    const double eps = 1e-6;
    VecX g_fd(3 * nv);
    for (int vi = 0; vi < nv; ++vi) {
        Vec3 gfd = local_gradient_fd(vi, ref_mesh, adj, pins, params, x, xhat, bp_nt, bp_ss, eps);
        g_fd.segment<3>(3*vi) = gfd;
    }

    const double test_vs_fd = (g_test - g_fd).norm();

    // compute_global_residual reports the infinity norm of mass-normalized
    // local gradients, so normalize the test and FD paths the same way.
    VecX g_test_mass_norm = g_test;
    VecX g_fd_mass_norm = g_fd;
    for (int vi = 0; vi < nv; ++vi) {
        const double m = ref_mesh.mass[vi];
        if (m > 0.0) {
            g_test_mass_norm.segment<3>(3 * vi) /= m;
            g_fd_mass_norm.segment<3>(3 * vi) /= m;
        }
    }
    const double r_test_infty = g_test_mass_norm.cwiseAbs().maxCoeff();
    const double r_fd = g_fd_mass_norm.cwiseAbs().maxCoeff();

    std::cout << "r_prod=" << r_prod
              << " r_test_infty=" << r_test_infty
              << " r_fd=" << r_fd
              << " ||g_test - g_fd||=" << test_vs_fd << "\n";

    EXPECT_LT(test_vs_fd, 1e-4)
        << "test-path gradient disagrees with FD of total_energy";

    EXPECT_NEAR(r_prod, r_test_infty, 1e-12 * std::max(r_prod, 1.0))
        << "production compute_global_residual disagrees with test-path gradient infinity norm";

    EXPECT_NEAR(r_prod, r_fd, 1e-5 * std::max(r_prod, 1.0))
        << "production compute_global_residual disagrees with FD infinity norm";
}

// =====================================================================
//  Bending enabled at the rest configuration: in this regime delta = 0
//  so the PSD Hessian coincides with the true Hessian and FD of the
//  per-vertex gradient must match it, including the bending contribution.
// =====================================================================

TEST_F(TotalEnergyTest, PerVertexHessianWithBendingAtRest) {
    // Rebuild the mesh in its flat rest configuration so that the dihedral
    // complement is zero at every hinge. Keep barriers off so the only
    // contributions are inertia + elastic + pin + gravity + bending.
    clear_model(ref_mesh, state, X, pins);
    build_square_mesh(ref_mesh, state, X, 2, 2, 1.0, 1.0, Vec3(0, 0, 0));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_lumped_mass(params.density, params.thickness);
    adj = build_incident_triangle_map(ref_mesh.tris);
    xhat = state.deformed_positions;
    x    = state.deformed_positions;

    params.d_hat = 0.0;
    params.kB    = 50.0;
    nt_pairs.clear();
    ss_pairs.clear();
    pins.clear();

    const double eps = 1e-5;
    double max_err = 0.0;
    int worst_vi = -1;
    for (int vi = 0; vi < (int)x.size(); ++vi) {
        Mat33 H   = local_hessian   (vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
        Mat33 Hfd = local_hessian_fd(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs, eps);
        const double err = (H - Hfd).cwiseAbs().maxCoeff();
        if (err > max_err) { max_err = err; worst_vi = vi; }
    }
    std::cout << "bending rest: worst vertex=" << worst_vi << " err=" << max_err << "\n";
    ASSERT_LT(max_err, 1e-3) << "Hessian mismatch at rest with bending enabled";
}

// =====================================================================
//  SDF penalty is wired into the incremental potential and per-vertex
//  (gradient, Hessian). Verify:
//    - k_sdf == 0 reproduces the baseline (bit-for-bit);
//    - k_sdf > 0 with a PlaneSDF placed so one vertex sits in the
//      transition shell raises the total energy, pushes that vertex's
//      gradient along the outward normal, and adds a PSD block to H.
// =====================================================================

TEST_F(TotalEnergyTest, SdfPenaltyDisabledIsBaseline) {
    params.d_hat = 0.0;
    nt_pairs.clear();
    ss_pairs.clear();

    const double E_before = total_energy(ref_mesh, pins, params, x, xhat, nt_pairs, ss_pairs);

    // k_sdf == 0 with a non-empty obstacle list must reproduce the baseline exactly.
    params.k_sdf       = 0.0;
    params.eps_sdf     = 0.05;
    params.sdf_planes  = {PlaneSDF{Vec3::Zero(), Vec3(0.0, 1.0, 0.0)}};

    const double E_after = total_energy(ref_mesh, pins, params, x, xhat, nt_pairs, ss_pairs);
    EXPECT_EQ(E_before, E_after);

    Vec3 g = local_gradient(0, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
    Vec3 g_ref;
    {
        SimParams p2 = params;
        p2.sdf_planes.clear();
        g_ref = local_gradient(0, ref_mesh, adj, pins, p2, x, xhat, nt_pairs, ss_pairs);
    }
    EXPECT_TRUE((g - g_ref).isZero(0.0));
}

TEST(RigidBodySolverUpdate, ComUpdateUsesInertiaAndGravity) {
    DeformedState state;
    state.x_coms = {Vec3(-0.2, 0.4, 0.1)};
    state.v_coms = {Vec3(0.7, -0.3, 0.2)};

    SimParams params = SimParams::zeros();
    params.gravity = Vec3(0.0, -9.81, 0.0);

    constexpr double dt = 0.08;
    constexpr double total_mass = 3.7;
    const Vec3 x_com(0.5, -0.1, 0.3);
    const rb_solver::ComUpdate update = rb_solver::compute_com_update(
        0, state, x_com, params, dt, total_mass);

    std::vector<Vec3> x_coms = {x_com};
    rb_solver::commit_com_update(update, x_coms);
    Vec3 expected_x_com = state.x_coms[0] + dt * state.v_coms[0];
    expected_x_com.y() += dt * dt * params.gravity.y();

    EXPECT_EQ(update.rb, 0);
    EXPECT_DOUBLE_EQ(update.step, 1.0);
    EXPECT_TRUE(x_coms[0].isApprox(expected_x_com, 1.0e-14));
}

TEST(RigidBodySolverUpdate, OrientationUpdateSolvesNewtonSystem) {
    DeformedState state;
    state.orientations = {
        quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4))
    };
    state.omega = {Vec3(-0.2, 0.5, 0.4)};

    constexpr double dt = 0.31;
    const Vec3 omega(0.6, -0.3, 0.7);
    const Mat33 I_hat = (Mat33() <<
        1.4, 0.2, -0.1,
        0.2, 0.9, 0.15,
        -0.1, 0.15, 1.1).finished();
    RefMesh ref_mesh;
    ref_mesh.I_hat = {I_hat};

    const auto [gradient, hessian] = inertia_rotation_gradient_hessian(
        omega, state.orientations[0], state.omega[0], dt, I_hat);
    const rb_solver::OrientationUpdate update =
        rb_solver::compute_orientation_update(0, state, ref_mesh, omega, dt);

    EXPECT_EQ(update.rb, 0);
    EXPECT_DOUBLE_EQ(update.step, 1.0);
    EXPECT_TRUE((hessian * update.domega).isApprox(gradient, 1.0e-12));
}

TEST(RigidBodySolverUpdate, OrientationCommitRoundTripsThroughQuaternion) {
    DeformedState state;
    state.orientations = {
        quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4))
    };

    constexpr double dt = 0.31;
    std::vector<Vec3> omega = {Vec3(0.6, -0.3, 0.7)};
    std::vector<Vec4> orientations = {state.orientations[0]};
    const rb_solver::OrientationUpdate update{
        0, Vec3(0.15, -0.05, 0.2), 0.4
    };
    const Vec3 expected_omega = omega[0] - update.step * update.domega;

    rb_solver::commit_orientation_update(
        update, state, orientations, omega, dt);

    const Vec4 expected_orientation = quaternion_align_sign(
        quaternion_normalize(quaternion_from_angular_velocity(
            state.orientations[0], expected_omega, dt)),
        state.orientations[0]);
    const Vec4 reconstructed_orientation = quaternion_align_sign(
        quaternion_normalize(quaternion_from_angular_velocity(
            state.orientations[0], omega[0], dt)),
        orientations[0]);

    EXPECT_TRUE(orientations[0].isApprox(expected_orientation, 1.0e-14));
    EXPECT_TRUE(omega[0].isApprox(expected_omega, 1.0e-14));
    EXPECT_TRUE(reconstructed_orientation.isApprox(orientations[0], 1.0e-14));
}

TEST(RigidBodySolver, BasicCollisionFreeSweepConverges) {
    DeformedState state;
    state.x_coms = {Vec3(-0.2, 0.4, 0.1), Vec3(0.8, -0.5, 0.3)};
    state.v_coms = {Vec3(0.7, -0.3, 0.2), Vec3(-0.2, 0.6, -0.1)};
    state.orientations = {
        quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4)),
        quaternion_normalize(Vec4(0.7, 0.1, -0.5, 0.2))
    };
    state.omega = {Vec3::Zero(), Vec3::Zero()};

    RefMesh ref_mesh;
    ref_mesh.total_mass = {3.7, 1.9};
    ref_mesh.I_hat = {
        (Mat33() <<
            1.4, 0.2, -0.1,
            0.2, 0.9, 0.15,
            -0.1, 0.15, 1.1).finished(),
        (Mat33() <<
            0.8, -0.1, 0.05,
            -0.1, 1.2, 0.2,
            0.05, 0.2, 1.0).finished()
    };

    SimParams params = SimParams::zeros();
    params.fps = 20.0;
    params.substeps = 1;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 20;
    params.tol_abs = 1.0e-11;
    params.damping = 1.0;

    std::vector<Vec3> x_coms = {
        Vec3(0.5, -0.1, 0.3), Vec3(-0.4, 0.2, -0.8)
    };
    std::vector<Vec4> orientations = state.orientations;
    std::vector<Vec3> omega = {
        Vec3(0.2, -0.1, 0.15), Vec3(-0.1, 0.25, 0.05)
    };

    const SolverResult result = global_gauss_seidel_solver_basic_rb(
        ref_mesh, state, params, x_coms, orientations, omega);

    ASSERT_TRUE(result.converged);
    EXPECT_TRUE(result.has_residual);
    EXPECT_GT(result.initial_residual, result.final_residual);
    EXPECT_LE(result.final_residual, params.tol_abs);
    EXPECT_GE(result.iterations, 1);
    EXPECT_LE(result.iterations, params.max_global_iters);
    for (int rb = 0; rb < 2; ++rb) {
        Vec3 expected_com =
            state.x_coms[rb] + params.dt() * state.v_coms[rb];
        expected_com.y() += params.dt2() * params.gravity.y();
        EXPECT_TRUE(x_coms[rb].isApprox(expected_com, 1.0e-12));
        EXPECT_TRUE(omega[rb].isZero(1.0e-10));
        EXPECT_TRUE(orientations[rb].isApprox(
            state.orientations[rb], 1.0e-10));
    }
}

TEST(RigidBodySolver, ConvergedInitialGuessStillAdvancesOrientation) {
    DeformedState state;
    state.x_coms = {Vec3::Zero()};
    state.v_coms = {Vec3::Zero()};
    state.orientations = {Vec4(1.0, 0.0, 0.0, 0.0)};
    state.omega = {Vec3(5.0, 0.0, 0.0)};

    RefMesh ref_mesh;
    ref_mesh.total_mass = {1.0};
    ref_mesh.I_hat = {Mat33::Identity()};

    SimParams params = SimParams::zeros();
    params.fps = 30.0;
    params.substeps = 1;
    params.max_global_iters = 20;
    params.tol_abs = 1.0e6;

    std::vector<Vec3> x_coms = state.x_coms;
    std::vector<Vec4> orientations = state.orientations;
    std::vector<Vec3> omega = state.omega;
    const Vec4 expected_orientation = quaternion_align_sign(
        quaternion_normalize(quaternion_from_angular_velocity(
            state.orientations[0], state.omega[0], params.dt())),
        state.orientations[0]);

    const SolverResult result = global_gauss_seidel_solver_basic_rb(
        ref_mesh, state, params, x_coms, orientations, omega);

    ASSERT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_TRUE(orientations[0].isApprox(expected_orientation, 1.0e-14));
    EXPECT_FALSE(orientations[0].isApprox(state.orientations[0], 1.0e-14));
    EXPECT_TRUE(omega[0].isApprox(state.omega[0], 1.0e-14));
}

TEST(RigidBodySolver, BasicCollisionFreeSweepHonorsFixedIterations) {
    DeformedState state;
    state.x_coms = {Vec3::Zero()};
    state.v_coms = {Vec3::Zero()};
    state.orientations = {Vec4(1.0, 0.0, 0.0, 0.0)};
    state.omega = {Vec3::Zero()};

    RefMesh ref_mesh;
    ref_mesh.total_mass = {1.0};
    ref_mesh.I_hat = {Mat33::Identity()};

    SimParams params = SimParams::zeros();
    params.max_global_iters = 3;
    params.fixed_iters = true;

    std::vector<Vec3> x_coms = {Vec3(1.0, -2.0, 3.0)};
    std::vector<Vec4> orientations = state.orientations;
    std::vector<Vec3> omega = {Vec3(0.1, -0.2, 0.3)};

    const SolverResult result = global_gauss_seidel_solver_basic_rb(
        ref_mesh, state, params, x_coms, orientations, omega);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.has_residual);
    EXPECT_EQ(result.iterations, params.max_global_iters);
}

TEST(RigidBodySimulation, AdvanceOneFrameCommitsEachSubstep) {
    const Vec3 x_com_0(-0.2, 0.4, 0.1);
    const Vec3 v_com_0(0.7, -0.3, 0.2);
    const Vec4 orientation_0 =
        quaternion_normalize(Vec4(0.8, -0.2, 0.3, 0.4));
    const std::vector<Vec3> body_offsets = {
        Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0), Vec3(0.0, -1.0, 0.0),
        Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, -1.0)
    };

    DeformedState state;
    std::vector<Vec3> x;
    for (const Vec3& offset : body_offsets) {
        x.push_back(
            x_com_0 + quaternion_rotate(orientation_0, offset));
    }

    RefMesh ref_mesh;
    create_rigid_body(
        x, v_com_0, orientation_0, Vec3::Zero(), 3.7,
        ref_mesh, state);
    const std::vector<int>& nodes = ref_mesh.rb_nodes[0];

    SimParams params = SimParams::zeros();
    params.fps = 20.0;
    params.substeps = 2;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 1;
    params.fixed_iters = true;

    const SolverResult result = advance_one_frame_rb(
        state, ref_mesh, params);

    const double dt = params.dt();
    const Vec3 expected_v = v_com_0 + 2.0 * dt * params.gravity;
    const Vec3 expected_x =
        x_com_0 + 2.0 * dt * v_com_0
        + 3.0 * dt * dt * params.gravity;

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.has_residual);
    EXPECT_GT(result.initial_residual, result.final_residual);
    EXPECT_EQ(result.iterations, params.substeps);
    EXPECT_TRUE(state.x_coms[0].isApprox(expected_x, 1.0e-12));
    EXPECT_TRUE(state.v_coms[0].isApprox(expected_v, 1.0e-12));
    EXPECT_TRUE(state.orientations[0].isApprox(orientation_0, 1.0e-12));
    EXPECT_TRUE(state.omega[0].isZero(1.0e-12));
    for (std::size_t local = 0; local < nodes.size(); ++local) {
        EXPECT_TRUE(state.deformed_positions[nodes[local]].isApprox(
            expected_x + quaternion_rotate(orientation_0, body_offsets[local]),
            1.0e-12));
        EXPECT_TRUE(state.velocities[nodes[local]].isApprox(
            expected_v, 1.0e-12));
    }
}
