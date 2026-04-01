#include "solver.h"
#include "IPC_math.h"

void update_one_vertex(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                       const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xhat, std::vector<Vec3>& x){
    Vec3 g = compute_local_gradient_no_barrier(vi, ref_mesh, lumped_mass, adj, pins, params, x, xhat);
    Mat33 H = compute_local_hessian_no_barrier(vi, ref_mesh, lumped_mass, adj, pins, params, x);
    Vec3 dx = matrix3d_inverse(H) * g;
    x[vi] -= params.step_weight * dx;
}

SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                                        const std::vector<Pin>& pins, const SimParams& params, std::vector<Vec3>& xnew,
                                        const std::vector<Vec3>& xhat, std::vector<double>* residual_history){
    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() { return compute_global_residual(ref_mesh, lumped_mass, adj, pins, params, xnew, xhat); };

    SolverResult result;
    result.initial_residual = eval_residual();
    result.final_residual = result.initial_residual;
    result.iterations = 0;

    if (residual_history) residual_history->push_back(result.initial_residual);
    if (result.initial_residual < params.tol_abs) return result;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        for (int vi = 0; vi < static_cast<int>(xnew.size()); ++vi)
            update_one_vertex(vi, ref_mesh, lumped_mass, adj, pins, params, xhat, xnew);

        result.final_residual = eval_residual();
        result.iterations = iter;

        if (residual_history) residual_history->push_back(result.final_residual);
        if (result.final_residual < params.tol_abs) return result;
    }

    return result;
}