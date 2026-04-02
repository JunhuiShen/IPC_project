#include "make_shape.h"
#include "physics.h"
#include "barrier_energy.h"

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

using VecX = Eigen::VectorXd;

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
//  Contact pair types
// =====================================================================

struct NodeTrianglePair {
    int node;
    int tri_v[3];
};

struct SegmentSegmentPair {
    int v[4];
};

// =====================================================================
//  Total energy: no_barrier base + barrier for active pairs
// =====================================================================

double total_energy(const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs){
    double E = compute_incremental_potential_no_barrier(ref_mesh, pins, params, x, xhat);
    double dt2 = params.dt * params.dt;

    for (const auto& p : nt_pairs)
        E += dt2 * node_triangle_barrier(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat);

    for (const auto& p : ss_pairs)
        E += dt2 * segment_segment_barrier(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat);

    return E;
}

// =====================================================================
//  Local gradient: no_barrier base + barrier for pairs involving vi
// =====================================================================

Vec3 local_gradient(int vi, const RefMesh& ref_mesh, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs){
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat);
    double dt2 = params.dt * params.dt;

    for (const auto& p : nt_pairs) {
        if (vi != p.node && vi != p.tri_v[0] && vi != p.tri_v[1] && vi != p.tri_v[2]) continue;
        auto br = node_triangle_barrier_gradient(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat);
        if (vi == p.node)      g += dt2 * br.grad_x;
        if (vi == p.tri_v[0])  g += dt2 * br.grad_x1;
        if (vi == p.tri_v[1])  g += dt2 * br.grad_x2;
        if (vi == p.tri_v[2])  g += dt2 * br.grad_x3;
    }

    for (const auto& p : ss_pairs) {
        if (vi != p.v[0] && vi != p.v[1] && vi != p.v[2] && vi != p.v[3]) continue;
        auto br = segment_segment_barrier_gradient(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat);
        if (vi == p.v[0]) g += dt2 * br.grad_x1;
        if (vi == p.v[1]) g += dt2 * br.grad_x2;
        if (vi == p.v[2]) g += dt2 * br.grad_x3;
        if (vi == p.v[3]) g += dt2 * br.grad_x4;
    }

    return g;
}

// =====================================================================
//  Local Hessian (diagonal block): no_barrier base + barrier
// =====================================================================

Mat33 local_hessian(int vi, const RefMesh& ref_mesh, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs){
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat);
    double dt2 = params.dt * params.dt;

    for (const auto& p : nt_pairs) {
        int slot = -1;
        if (vi == p.node)      slot = 0;
        if (vi == p.tri_v[0])  slot = 1;
        if (vi == p.tri_v[1])  slot = 2;
        if (vi == p.tri_v[2])  slot = 3;
        if (slot < 0) continue;
        auto hr = node_triangle_barrier_hessian(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat);
        H += dt2 * hr.hessian.block<3,3>(3*slot, 3*slot);
    }

    for (const auto& p : ss_pairs) {
        int slot = -1;
        if (vi == p.v[0]) slot = 0;
        if (vi == p.v[1]) slot = 1;
        if (vi == p.v[2]) slot = 2;
        if (vi == p.v[3]) slot = 3;
        if (slot < 0) continue;
        auto hr = segment_segment_barrier_hessian(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat);
        H += dt2 * hr.hessian.block<3,3>(3*slot, 3*slot);
    }

    return H;
}

// =====================================================================
//  FD helpers
// =====================================================================

Vec3 local_gradient_fd(int vi, const RefMesh& ref_mesh, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs, double eps){
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

Mat33 local_hessian_fd(int vi, const RefMesh& ref_mesh, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs, double eps){
    Mat33 H = Mat33::Zero();
    for (int d = 0; d < 3; ++d) {
        auto xp = x, xm = x;
        xp[vi](d) += eps; xm[vi](d) -= eps;
        Vec3 gp = local_gradient(vi, ref_mesh, adj, pins, params, xp, xhat, nt_pairs, ss_pairs);
        Vec3 gm = local_gradient(vi, ref_mesh, adj, pins, params, xm, xhat, nt_pairs, ss_pairs);
        H.col(d) = (gp - gm) / (2.0 * eps);
    }
    return H;
}

// =====================================================================
//  Slope-2 check: g(x+h*d) - [g(x) + h*H*d] = O(h^2) => ratio ~4
// =====================================================================

void slope2_check(int vi, const RefMesh& ref_mesh, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs){
    std::cout << "\n=== slope-2 check vertex " << vi << " ===\n";

    Vec3 g = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
    Mat33 H = local_hessian(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);

    Vec3 dir(0.3, -0.5, 0.8); dir.normalize();

    std::vector<double> hs = {1e-2, 5e-3, 2.5e-3, 1.25e-3};
    std::vector<double> errs;

    for (double h : hs) {
        auto xh = x;
        xh[vi] += h * dir;
        Vec3 gh = local_gradient(vi, ref_mesh, adj, pins, params, xh, xhat, nt_pairs, ss_pairs);
        Vec3 lin = g + h * H * dir;
        double err = (gh - lin).norm();
        errs.push_back(err);
        std::cout << "h=" << h << " err=" << err << "\n";
    }

    for (int i = 1; i < (int)errs.size(); ++i)
        std::cout << "ratio=" << errs[i-1] / errs[i] << " (~4 expected)\n";
}

// =====================================================================
//  main
// =====================================================================

int main(){
    std::cout << std::setprecision(12);

    // -----------------------------------------------------------------
    //  Setup: two square meshes
    //    Mesh A: vertices 0-3 at z=0
    //    Mesh B: vertices 4-7 at z=0.4 (within d_hat=1.0)
    //
    //  All vertices get inertia + elastic + gravity + pin.
    //  Vertices in contact pairs also get barrier terms.
    //  Vertices NOT in any pair get no barrier (barrier adds zero).
    // -----------------------------------------------------------------

    SimParams params;
    params.dt = 1.0 / 30.0;
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

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;

    clear_model(ref_mesh, state, X, pins);

    // Mesh A at z=0
    build_square_mesh(ref_mesh, state, X, 1, 1, 1.0, 1.0, Vec3(0, 0, 0));
    // Mesh B at z=0.4
    build_square_mesh(ref_mesh, state, X, 1, 1, 1.0, 1.0, Vec3(0, 0, 0.4));

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    // Perturb to break symmetry
    state.deformed_positions[0] += Vec3(0.02, -0.01, 0.03);
    state.deformed_positions[1] += Vec3(-0.01, 0.03, -0.02);
    state.deformed_positions[2] += Vec3(0.01, 0.02, 0.01);
    state.deformed_positions[3] += Vec3(-0.02, -0.01, 0.02);
    state.deformed_positions[4] += Vec3(0.02, 0.01, -0.02);
    state.deformed_positions[5] += Vec3(-0.01, 0.02, 0.01);
    state.deformed_positions[6] += Vec3(0.01, -0.01, 0.02);
    state.deformed_positions[7] += Vec3(-0.02, 0.01, -0.01);

    // Pin corners of mesh B
    append_pin(pins, 6, state.deformed_positions);
    append_pin(pins, 7, state.deformed_positions);

    ref_mesh.build_lumped_mass(params.density, params.thickness);
    VertexAdjacency adj = build_vertex_adjacency(ref_mesh);

    std::vector<Vec3> xhat = state.deformed_positions;
    std::vector<Vec3> x = state.deformed_positions;

    // Contact pairs: some vertices involved, some not
    //   Vertices 2,3 from mesh A have NO barrier pairs
    //   Vertices 0,1,4,5 appear in barrier pairs
    std::vector<NodeTrianglePair> nt_pairs;
    nt_pairs.push_back({4, {0, 1, 3}});   // vertex 4 vs triangle (0,1,3)
    nt_pairs.push_back({5, {0, 1, 3}});   // vertex 5 vs triangle (0,1,3)

    std::vector<SegmentSegmentPair> ss_pairs;
    ss_pairs.push_back({{0, 1, 4, 5}});   // edge (0,1) vs edge (4,5)
    ss_pairs.push_back({{0, 3, 4, 7}});   // edge (0,3) vs edge (4,7)

    // -----------------------------------------------------------------
    //  Barrier activation check
    // -----------------------------------------------------------------
    {
        std::cout << "=== barrier activation check ===\n";
        for (int i = 0; i < (int)nt_pairs.size(); ++i) {
            const auto& p = nt_pairs[i];
            auto dr = node_triangle_distance(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]]);
            double e = node_triangle_barrier(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat);
            std::cout << "NT pair " << i << ": distance=" << dr.distance << " barrier=" << e << " region=" << to_string(dr.region) << "\n";
        }
        for (int i = 0; i < (int)ss_pairs.size(); ++i) {
            const auto& p = ss_pairs[i];
            auto dr = segment_segment_distance(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]]);
            double e = segment_segment_barrier(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat);
            std::cout << "SS pair " << i << ": distance=" << dr.distance << " barrier=" << e << " region=" << to_string(dr.region) << "\n";
        }
    }

    // -----------------------------------------------------------------
    //  Directional derivative check
    // -----------------------------------------------------------------
    {
        std::cout << "\n=== total energy directional derivative check ===\n";

        VecX q = flatten_positions(x);
        VecX dir = VecX::Random(q.size()); dir.normalize();
        double eps = 1e-6;

        auto xp = unflatten_positions(q + eps * dir);
        auto xm = unflatten_positions(q - eps * dir);
        double fd = (total_energy(ref_mesh, pins, params, xp, xhat, nt_pairs, ss_pairs)
                     - total_energy(ref_mesh, pins, params, xm, xhat, nt_pairs, ss_pairs)) / (2.0 * eps);

        VecX g(3 * x.size());
        for (int vi = 0; vi < (int)x.size(); ++vi)
            g.segment<3>(3 * vi) = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);

        double an = g.dot(dir);
        std::cout << "FD=" << fd << " analytic=" << an << " error=" << std::abs(fd - an) << "\n";
    }

    // -----------------------------------------------------------------
    //  Per-vertex gradient check (FD of total energy vs analytic gradient)
    //  Vertices 2,3: no barrier pairs => tests no_barrier path
    //  Vertices 0,1,4,5: in barrier pairs => tests barrier path
    // -----------------------------------------------------------------
    {
        std::cout << "\n=== per-vertex gradient check ===\n";
        double eps = 1e-6;
        for (int vi = 0; vi < (int)x.size(); ++vi) {
            Vec3 g = local_gradient(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
            Vec3 gfd = local_gradient_fd(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs, eps);
            std::cout << "v " << vi << " err=" << (g - gfd).norm() << "\n";
        }
    }

    // -----------------------------------------------------------------
    //  Per-vertex Hessian check (FD of gradient vs analytic Hessian)
    // -----------------------------------------------------------------
    {
        std::cout << "\n=== per-vertex Hessian check ===\n";
        double eps = 1e-6;
        for (int vi = 0; vi < (int)x.size(); ++vi) {
            Mat33 H = local_hessian(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
            Mat33 Hfd = local_hessian_fd(vi, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs, eps);
            std::cout << "v " << vi << " err=" << (H - Hfd).lpNorm<Eigen::Infinity>() << "\n";
        }
    }

    // -----------------------------------------------------------------
    //  Slope-2 checks
    //    v2: no barrier pairs (pure elastic+inertia+gravity+pin)
    //    v0: in both NT and SS pairs
    //    v4: in NT pair (query node) and SS pair
    // -----------------------------------------------------------------
    slope2_check(2, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
    slope2_check(0, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
    slope2_check(4, ref_mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);

    std::cout << "\n========================================\n"
              << "All total energy tests completed.\n"
              << "========================================\n";
    return 0;
}
