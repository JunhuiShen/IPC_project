#include "solver.h"
#include "IPC_math.h"

void update_one_vertex(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                       const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x,
                       const std::vector<NodeTrianglePair>& nt_pairs,
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

    Vec3 dx = matrix3d_inverse(H) * g;
    x[vi] -= params.step_weight * dx;
}

SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                        const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        const std::vector<NodeTrianglePair>& nt_pairs,
                                        const std::vector<SegmentSegmentPair>& ss_pairs,
                                        const std::vector<std::vector<int>>& color_groups,
                                        std::vector<double>* residual_history) {
    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return compute_global_residual(ref_mesh, adj, pins, params, xnew, xhat, nt_pairs, ss_pairs);
    };

    SolverResult result;
    result.initial_residual = eval_residual();
    result.final_residual   = result.initial_residual;
    result.iterations       = 0;

    if (residual_history) residual_history->push_back(result.initial_residual);
    if (result.initial_residual < params.tol_abs) return result;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if (params.use_parallel) {
            for (const auto& group : color_groups) {
                #pragma omp parallel for schedule(static) if(group.size() >= 256)
                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                    update_one_vertex(group[i], ref_mesh, adj, pins, params, xhat, xnew, nt_pairs, ss_pairs);
            }
        } else {
            for (const auto& group : color_groups)
                for (int vi : group)
                    update_one_vertex(vi, ref_mesh, adj, pins, params, xhat, xnew, nt_pairs, ss_pairs);
        }

        result.final_residual = eval_residual();
        result.iterations     = iter;

        if (residual_history) residual_history->push_back(result.final_residual);
        if (result.final_residual < params.tol_abs) return result;
    }

    return result;
}
