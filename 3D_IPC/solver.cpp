#include "solver.h"
#include "IPC_math.h"

void update_one_vertex(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x, const std::vector<NodeTrianglePair>& nt_pairs,
                       const std::vector<SegmentSegmentPair>& ss_pairs) {
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat);

    if (params.d_hat > 0.0) {
        const double dt2 = params.dt2();

        for (const auto& p : nt_pairs) {
            int dof = -1;
            if      (vi == p.node)      dof = 0;
            else if (vi == p.tri_v[0])  dof = 1;
            else if (vi == p.tri_v[1])  dof = 2;
            else if (vi == p.tri_v[2])  dof = 3;
            if (dof < 0) continue;
            // Combined call: one distance evaluation for both g and H
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
            g += dt2 * bg;
            H += dt2 * bH.block<3, 3>(0, 3 * dof);
        }

        for (const auto& p : ss_pairs) {
            int dof = -1;
            if      (vi == p.v[0]) dof = 0;
            else if (vi == p.v[1]) dof = 1;
            else if (vi == p.v[2]) dof = 2;
            else if (vi == p.v[3]) dof = 3;
            if (dof < 0) continue;
            // Combined call: one distance evaluation for both g and H
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, dof);
            g += dt2 * bg;
            H += dt2 * bH.block<3, 3>(0, 3 * dof);
        }
    }

    const Vec3 delta = matrix3d_inverse(H) * g;

    // CCD hook point:
    // Build a global swept-AABB candidate set using the local Newton trial displacement.
    // For now, keep step_weight as the placeholder limiter until CCD TOI is integrated.
    double step = params.step_weight;
    if (params.d_hat > 0.0) {
        std::vector<Vec3> trial_disp(x.size(), Vec3::Zero());
        trial_disp[vi] = -delta;

        BroadPhase ccd_broad_phase;
        std::vector<NodeTrianglePair> ccd_nt;
        std::vector<SegmentSegmentPair> ccd_ss;
        ccd_broad_phase.build_ccd_candidates(x, trial_disp, ref_mesh, 1.0, ccd_nt, ccd_ss);

        // Placeholder until CCD is wired into line search.
        step = params.step_weight;
    }

    x[vi] -= step * delta;
}

SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                        const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        BroadPhase& broad_phase,
                                        const std::vector<Vec3>& v,
                                        const std::vector<std::vector<int>>& color_groups,
                                        std::vector<double>* residual_history) {
    const double dt         = params.dt();
    const double dhat       = params.d_hat;
    const bool   use_barrier = dhat > 0.0;
    const double node_pad   = dhat;
    const double tri_pad    = 0.0;
    const double edge_pad   = dhat * 0.5;

    if (use_barrier) {
        broad_phase.initialize(xnew, v, ref_mesh, dt, dhat);
    }

    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return compute_global_residual(ref_mesh, adj, pins, params, xnew, xhat,
                                       broad_phase.nt_pairs(), broad_phase.ss_pairs());
    };

    SolverResult result;
    result.initial_residual = eval_residual();
    result.final_residual   = result.initial_residual;
    result.iterations       = 0;

    if (residual_history) residual_history->push_back(result.initial_residual);
    if (result.initial_residual < params.tol_abs) return result;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        for (const auto& group : color_groups) {
            if (params.use_parallel) {
                #pragma omp parallel for schedule(static) if(group.size() >= 256)
                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                    update_one_vertex(group[i], ref_mesh, adj, pins, params, xhat, xnew,
                                      broad_phase.nt_pairs(), broad_phase.ss_pairs());
                if (use_barrier) {
                    for (int vi : group)
                        broad_phase.refresh(xnew, v, ref_mesh, vi, dt, node_pad, tri_pad, edge_pad);
                }
            } else {
                for (int vi : group) {
                    update_one_vertex(vi, ref_mesh, adj, pins, params, xhat, xnew,
                                      broad_phase.nt_pairs(), broad_phase.ss_pairs());
                    if (use_barrier)
                        broad_phase.refresh(xnew, v, ref_mesh, vi, dt, node_pad, tri_pad, edge_pad);
                }
            }
        }

        result.final_residual = eval_residual();
        result.iterations     = iter;

        if (residual_history) residual_history->push_back(result.final_residual);
        if (result.final_residual < params.tol_abs) return result;
    }

    return result;
}
