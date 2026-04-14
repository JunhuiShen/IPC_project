#include <gtest/gtest.h>

#include "make_shape.h"
#include "physics.h"
#include "barrier_energy.h"
#include "broad_phase.h"

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

using VecX = Eigen::VectorXd;

namespace {

VecX flatten_positions(const std::vector<Vec3>& x){
    VecX q(3 * x.size());
    for (int i = 0; i < (int)x.size(); ++i) q.segment<3>(3*i) = x[i];
    return q;
}

std::vector<Vec3> unflatten_positions(const VecX& q){
    std::vector<Vec3> x(q.size()/3);
    for (int i = 0; i < (int)x.size(); ++i) x[i] = q.segment<3>(3*i);
    return x;
}

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
        H += dt2 * node_triangle_barrier_hessian(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
    }

    for (const auto& p : ss_pairs) {
        int dof = -1;
        if      (vi == p.v[0]) dof = 0;
        else if (vi == p.v[1]) dof = 1;
        else if (vi == p.v[2]) dof = 2;
        else if (vi == p.v[3]) dof = 3;
        if (dof < 0) continue;
        H += dt2 * segment_segment_barrier_hessian(
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

// =====================================================================
//  Slope-2 check
// =====================================================================

bool slope2_check(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                  const std::vector<Pin>& pins, const SimParams& params,
                  const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                  const std::vector<NodeTrianglePair>& nt_pairs,
                  const std::vector<SegmentSegmentPair>& ss_pairs){
    std::cout << "\n=== slope-2 check vertex " << vi << " ===\n";

    Vec3  g = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
    Mat33 H = local_hessian (vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);

    Vec3 dir(0.3, -0.5, 0.8); dir.normalize();

    std::vector<double> hs = {1e-2, 5e-3, 2.5e-3, 1.25e-3};
    std::vector<double> errs;

    for (double h : hs) {
        auto xh = x;
        xh[vi] += h * dir;
        Vec3 gh  = local_gradient(vi, ref_mesh, adj, pins, params, xh, xhat, nt_pairs, ss_pairs);
        Vec3 lin = g + h * H * dir;
        double err = (gh - lin).norm();
        errs.push_back(err);
    }

    double noise_floor = 1e-10;
    bool saw_good_slope = false;
    for (int i = 1; i < (int)errs.size(); ++i) {
        if (errs[i] < noise_floor && errs[i-1] < noise_floor) continue;
        if (errs[i] == 0.0) continue;
        double slope = std::log(errs[i-1]/errs[i]) / std::log(hs[i-1]/hs[i]);
        std::cout << "    h=" << hs[i] << "  err=" << errs[i]
                  << "  slope=" << std::fixed << std::setprecision(2) << slope << "\n";
        if (slope >= 1.8) saw_good_slope = true;
    }
    if (!saw_good_slope) {
        std::cerr << "  FAIL: no slope >= 1.8 for slope-2 vertex " << vi << "\n";
    }
    return saw_good_slope;
}

} // anonymous namespace

// =====================================================================
//  Test fixture
// =====================================================================

class TotalEnergyTest : public ::testing::Test {
protected:
    SimParams params;
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
        params.step_weight = 1.0;
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
//  Barrier activation check
// -----------------------------------------------------------------

TEST_F(TotalEnergyTest, BarrierActivation) {
    for (int i = 0; i < (int)nt_pairs.size(); ++i) {
        const auto& p = nt_pairs[i];
        auto dr = node_triangle_distance(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]]);
        double e = node_triangle_barrier(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat);
        std::cout << "NT pair " << i << ": distance=" << dr.distance
                  << " barrier=" << e << " region=" << to_string(dr.region) << "\n";
        EXPECT_GT(dr.distance, 0.0) << "NT pair " << i << " has non-positive distance";
    }
    for (int i = 0; i < (int)ss_pairs.size(); ++i) {
        const auto& p = ss_pairs[i];
        auto dr = segment_segment_distance(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]]);
        double e = segment_segment_barrier(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat);
        std::cout << "SS pair " << i << ": distance=" << dr.distance
                  << " barrier=" << e << " region=" << to_string(dr.region) << "\n";
        EXPECT_GT(dr.distance, 0.0) << "SS pair " << i << " has non-positive distance";
    }
}

// -----------------------------------------------------------------
//  Directional derivative check
// -----------------------------------------------------------------

TEST_F(TotalEnergyTest, DirectionalDerivative) {
    VecX q   = flatten_positions(x);
    VecX dir = VecX::Random(q.size()); dir.normalize();
    double eps = 1e-6;

    auto xp = unflatten_positions(q + eps * dir);
    auto xm = unflatten_positions(q - eps * dir);
    double fd = (total_energy(ref_mesh, pins, params, xp, xhat, nt_pairs, ss_pairs)
                 - total_energy(ref_mesh, pins, params, xm, xhat, nt_pairs, ss_pairs)) / (2.0 * eps);

    VecX g(3 * x.size());
    for (int vi = 0; vi < (int)x.size(); ++vi)
        g.segment<3>(3*vi) = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);

    double an = g.dot(dir);
    double err = std::abs(fd - an);
    std::cout << "FD=" << fd << " analytic=" << an << " error=" << err << "\n";
    ASSERT_LT(err, 1e-4) << "directional derivative error too large";
}

// -----------------------------------------------------------------
//  Per-vertex gradient check
// -----------------------------------------------------------------

TEST_F(TotalEnergyTest, PerVertexGradient) {
    double eps = 1e-6;
    for (int vi = 0; vi < (int)x.size(); ++vi) {
        Vec3 g   = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
        Vec3 gfd = local_gradient_fd(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs, eps);
        double err = (g - gfd).norm();
        std::cout << "v " << vi << " err=" << err << "\n";
        ASSERT_LT(err, 1e-4) << "gradient mismatch at vertex " << vi;
    }
}

// -----------------------------------------------------------------
//  Per-vertex Hessian check
// -----------------------------------------------------------------

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

// -----------------------------------------------------------------
//  Slope-2 checks
// -----------------------------------------------------------------

TEST_F(TotalEnergyTest, Slope2Vertex2) {
    EXPECT_TRUE(slope2_check(2, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs))
        << "slope-2 check failed for vertex 2";
}

TEST_F(TotalEnergyTest, Slope2Vertex0) {
    EXPECT_TRUE(slope2_check(0, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs))
        << "slope-2 check failed for vertex 0";
}

TEST_F(TotalEnergyTest, Slope2Vertex4) {
    EXPECT_TRUE(slope2_check(4, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs))
        << "slope-2 check failed for vertex 4";
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
// consistency is covered at the rest state in bending_physics_test.cpp.

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

TEST_F(TotalEnergyTest, DirectionalDerivativeWithBending) {
    params.kB = 50.0;

    VecX q   = flatten_positions(x);
    VecX dir = VecX::Random(q.size()); dir.normalize();
    const double eps = 1e-6;

    auto xp = unflatten_positions(q + eps * dir);
    auto xm = unflatten_positions(q - eps * dir);
    const double fd = (total_energy(ref_mesh, pins, params, xp, xhat, nt_pairs, ss_pairs)
                       - total_energy(ref_mesh, pins, params, xm, xhat, nt_pairs, ss_pairs)) / (2.0 * eps);

    VecX g(3 * x.size());
    for (int vi = 0; vi < (int)x.size(); ++vi)
        g.segment<3>(3*vi) = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);

    const double an = g.dot(dir);
    const double err = std::abs(fd - an);
    std::cout << "bending directional: FD=" << fd << " analytic=" << an << " err=" << err << "\n";
    ASSERT_LT(err, 1e-4) << "directional derivative mismatch with bending enabled";
}

// =====================================================================
//  compute_global_residual (production path) vs FD
//
//  Bridges the production gradient code (compute_local_gradient in
//  physics.cpp, reached via compute_global_residual) to the test-side
//  local_gradient / total_energy pair that the existing FD tests have
//  already slope-2 validated. Also asserts a round-off-level match
//  between the two independently implemented gradient paths, so any
//  drift between them would fire here.
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
    const double r_test_infty = g_test.cwiseAbs().maxCoeff();

    const double eps = 1e-6;
    VecX g_fd(3 * nv);
    for (int vi = 0; vi < nv; ++vi) {
        Vec3 gfd = local_gradient_fd(vi, ref_mesh, adj, pins, params, x, xhat, bp_nt, bp_ss, eps);
        g_fd.segment<3>(3*vi) = gfd;
    }
    const double r_fd = g_fd.cwiseAbs().maxCoeff();

    const double test_vs_fd = (g_test - g_fd).norm();
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
