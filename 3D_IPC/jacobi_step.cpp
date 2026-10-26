#include "jacobi_step.h"
#include "broad_phase.h"
#include "friction_energy.h"
#include "solid_ipc.h"

#include <array>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

template<class Function>
void parallel_for(std::size_t count, bool parallel, const Function& function) {
    if (count == 0) return;
    std::exception_ptr error;
    #pragma omp parallel for schedule(static) if(parallel)
    for (std::size_t i = 0; i < count; ++i) {
        try {
            function(i);
        } catch (...) {
            #pragma omp critical(ipc_jacobi_step_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

std::array<Vec3, 4> pair_positions(const NodeTrianglePair& pair, const std::vector<Vec3>& x) {
    return {x[pair.node], x[pair.tri_v[0]], x[pair.tri_v[1]], x[pair.tri_v[2]]};
}

std::array<Vec3, 4> pair_positions(const SegmentSegmentPair& pair, const std::vector<Vec3>& x) {
    return {x[pair.v[0]], x[pair.v[1]], x[pair.v[2]], x[pair.v[3]]};
}

struct JacobiWorkspace {
    BroadPhase broad_phase;
    std::vector<AABB> boxes;
    std::vector<IncidentTriangles> incident;
    std::vector<ShapeGrads> rest_gradients;
    std::vector<unsigned char> solid_nodes, surface_nodes;
    std::vector<NodeTriangleContactEvaluation> nt;
    std::vector<SegmentSegmentContactEvaluation> ss;
    std::vector<FrozenFrictionContact> nt_friction, ss_friction;
};

void add_mesh_contact(int node, const JacobiWorkspace& workspace, const SimParams& params, const std::vector<Vec3>& x, Vec3& gradient, Mat33& hessian) {
    const auto& cache = workspace.broad_phase.cache();
    const double scale = params.dt2() * params.k_barrier;
    const auto accumulate = [&](const auto& evaluation, const auto& pair, int role, const FrozenFrictionContact* friction) {
        if (!evaluation.active) return;
        const auto points = pair_positions(pair, x);
        const auto [g, H] = [&] {
            if constexpr (std::is_same_v<std::decay_t<decltype(evaluation)>, NodeTriangleContactEvaluation>)
                return node_triangle_barrier_self_gradient_and_hessian(points[0], points[1], points[2], points[3], role, evaluation);
            else
                return segment_segment_barrier_self_gradient_and_hessian(points[0], points[1], points[2], points[3], role, evaluation);
        }();
        gradient += scale * g;
        hessian += scale * H;
        if (friction) {
            const auto [fg, fH] = frozen_friction_role_gradient_and_hessian(*friction, role, params.friction_coefficient, params.dt2());
            gradient += fg;
            hessian += fH;
        }
    };
    for (const auto& entry : cache.vertex_nt[node]) accumulate(workspace.nt[entry.pair_index], cache.nt_pairs[entry.pair_index], entry.dof, params.friction_coefficient > 0 ? &workspace.nt_friction[entry.pair_index] : nullptr);
    for (const auto& entry : cache.vertex_ss[node]) accumulate(workspace.ss[entry.pair_index], cache.ss_pairs[entry.pair_index], entry.dof, params.friction_coefficient > 0 ? &workspace.ss_friction[entry.pair_index] : nullptr);
}

} // namespace

std::vector<Vec3> jacobi_step(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, const std::vector<Vec3>& v, const std::vector<Vec3>* previous_positions) {
    const std::size_t count = xnew.size();
    if (xhat.size() != count || v.size() != count || ref_mesh.mass.size() != count)
        throw std::invalid_argument("jacobi_step: state arrays must have matching sizes");
    if (!ref_mesh.rb_nodes.empty())
        throw std::invalid_argument("jacobi_step: rigid generalized coordinates require a separate block interface");
    if (!std::isfinite(params.damping) || params.damping < 0)
        throw std::invalid_argument("jacobi_step: damping must be finite and nonnegative");
    if (!std::isfinite(params.friction_coefficient) || params.friction_coefficient < 0 || (params.friction_coefficient > 0 && (!std::isfinite(params.friction_velocity_epsilon) || params.friction_velocity_epsilon <= 0)))
        throw std::invalid_argument("jacobi_step: invalid friction parameters");
    for (const auto& pin : pins)
        if (pin.vertex_index < 0 || static_cast<std::size_t>(pin.vertex_index) >= count)
            throw std::out_of_range("jacobi_step: pin vertex index");

    // Warm a private copy's lazy dt caches before parallel assembly.
    SimParams assembly_params = params;
    const double dt = assembly_params.dt();
    assembly_params.dt2();
    std::vector<Vec3> reconstructed_previous;
    if (params.friction_coefficient > 0) {
        if (!previous_positions) {
            reconstructed_previous.resize(count);
            for (std::size_t i = 0; i < count; ++i) reconstructed_previous[i] = xhat[i] - dt*v[i];
            previous_positions = &reconstructed_previous;
        }
        if (previous_positions->size() != count) throw std::invalid_argument("jacobi_step: previous-position size mismatch");
        for (const auto& position : *previous_positions)
            if (!position.allFinite()) throw std::invalid_argument("jacobi_step: nonfinite previous position");
    } else previous_positions = nullptr;
    if (count == 0) return {};

    static thread_local JacobiWorkspace storage;
    auto& workspace = storage;
    workspace.solid_nodes.assign(count, 0);
    workspace.surface_nodes.assign(count, 0);
    for (int node : ref_mesh.tets) workspace.solid_nodes[node] = 1;
    for (int node : ref_mesh.surface_nodes) workspace.surface_nodes[node] = 1;
    workspace.incident.assign(count, {});
    for (const auto& [node, row] : adj) {
        if (workspace.solid_nodes[node]) continue;
        workspace.incident[node] = row;
    }
    workspace.rest_gradients.resize(ref_mesh.Dm_inverse.size());
    parallel_for(workspace.rest_gradients.size(), params.use_parallel, [&](std::size_t i) { workspace.rest_gradients[i] = shape_function_gradients(ref_mesh.Dm_inverse[i]); });
    const auto pin_map = build_pin_map(pins, static_cast<int>(count));
    const bool mesh_contact = params.d_hat > 0 && params.k_barrier > 0;
    if (mesh_contact) {
        workspace.boxes.resize(count);
        parallel_for(count, params.use_parallel, [&](std::size_t i) { workspace.boxes[i] = AABB(xnew[i], xnew[i]); });
        workspace.broad_phase.invalidate_mesh_topology();
        if (ref_mesh.tets.empty()) workspace.broad_phase.initialize(workspace.boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::DeformableSolver);
        else workspace.broad_phase.initialize_surface_nodes(workspace.boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::DeformableSolver);
        const auto& cache = workspace.broad_phase.cache();
        workspace.nt.resize(cache.nt_pairs.size());
        workspace.ss.resize(cache.ss_pairs.size());
        if (params.friction_coefficient > 0) {
            workspace.nt_friction.resize(cache.nt_pairs.size());
            workspace.ss_friction.resize(cache.ss_pairs.size());
        }
        parallel_for(cache.nt_pairs.size(), params.use_parallel, [&](std::size_t i) {
            workspace.nt[i] = NodeTriangleContactEvaluation{};
            const auto points = pair_positions(cache.nt_pairs[i], xnew);
            if (!node_triangle_aabbs_within_distance(points[0], points[1], points[2], points[3], params.d_hat*params.d_hat)) return;
            workspace.nt[i] = make_node_triangle_contact_evaluation(points, params.d_hat, params.k_barrier, 1e-12, nullptr, params.exact_computation_fallback);
            if (params.friction_coefficient > 0 && workspace.nt[i].active) workspace.nt_friction[i] = make_node_triangle_frozen_friction_contact(points, pair_positions(cache.nt_pairs[i], *previous_positions), workspace.nt[i], dt, params.friction_velocity_epsilon);
        });
        parallel_for(cache.ss_pairs.size(), params.use_parallel, [&](std::size_t i) {
            workspace.ss[i] = SegmentSegmentContactEvaluation{};
            const auto points = pair_positions(cache.ss_pairs[i], xnew);
            if (!segment_aabbs_within_distance(points[0], points[1], points[2], points[3], params.d_hat*params.d_hat)) return;
            workspace.ss[i] = make_segment_segment_contact_evaluation(points, params.d_hat, params.k_barrier, 1e-12, nullptr, params.exact_computation_fallback);
            if (params.friction_coefficient > 0 && workspace.ss[i].active) workspace.ss_friction[i] = make_segment_segment_frozen_friction_contact(points, pair_positions(cache.ss_pairs[i], *previous_positions), workspace.ss[i], dt, params.friction_velocity_epsilon);
        });
    }

    std::vector<Vec3> displacement(count);
    parallel_for(count, params.use_parallel, [&](std::size_t i) {
        auto [gradient, hessian] = [&] {
            if (workspace.solid_nodes[i]) return compute_solid_local_gradient_and_block(static_cast<int>(i), ref_mesh, pins, assembly_params, xnew, xhat, workspace.broad_phase, &workspace.solid_nodes, &workspace.surface_nodes, &pin_map, previous_positions);
            return physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(static_cast<int>(i), ref_mesh, adj, pins, assembly_params, xnew, xhat, &pin_map, &workspace.incident[i], &workspace.rest_gradients, previous_positions);
        }();
        if (mesh_contact && !workspace.solid_nodes[i]) add_mesh_contact(static_cast<int>(i), workspace, assembly_params, xnew, gradient, hessian);
        if (!gradient.allFinite() || !hessian.allFinite()) throw std::runtime_error("jacobi_step: nonfinite local system at vertex " + std::to_string(i));
        if (params.damping == 0) displacement[i].setZero();
        else displacement[i] = -params.damping * (matrix3d_inverse(hessian) * gradient);
        if (!displacement[i].allFinite()) throw std::runtime_error("jacobi_step: nonfinite displacement at vertex " + std::to_string(i));
    });
    return displacement;
}

