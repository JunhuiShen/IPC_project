#include "solver.h"
#include "IPC_math.h"
#include "parallel_helper.h"
#include "barrier_energy.h"
#include "output.h"
#include "rigid_body_ipc.h"
#include "safe_step.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Deformable solver workspaces
// -----------------------------------------------------------------------------

namespace {

struct ElasticAdjacencyCache {
    const RefMesh* mesh = nullptr;
    const int* tris_data = nullptr;
    std::size_t tris_size = 0;
    std::size_t hinges_size = 0;
    int num_vertices = -1;
    std::vector<std::vector<int>> adjacency;

    bool matches(const RefMesh& ref_mesh, int nv) const {
        return mesh == &ref_mesh && tris_data == ref_mesh.tris.data() && tris_size == ref_mesh.tris.size() && hinges_size == ref_mesh.hinges.size() && num_vertices == nv;
    }

    const std::vector<std::vector<int>>& get(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int nv) {
        if (!matches(ref_mesh, nv)) {
            adjacency = build_elastic_adj(ref_mesh, adj, nv);
            mesh = &ref_mesh;
            tris_data = ref_mesh.tris.data();
            tris_size = ref_mesh.tris.size();
            hinges_size = ref_mesh.hinges.size();
            num_vertices = nv;
        }
        return adjacency;
    }
};

struct BasicSolverWorkspace {
    ElasticAdjacencyCache elastic_adjacency;
    const RefMesh* mesh = nullptr;
    const int* tris_data = nullptr;
    const Mat22* dm_data = nullptr;
    std::size_t tris_size = 0;
    std::size_t dm_size = 0;
    std::size_t hinges_size = 0;
    int num_vertices = -1;

    PinMap pin_map;
    std::vector<IncidentTriangles> incident_triangles;
    std::vector<ShapeGrads> rest_shape_grads;
    std::vector<double> prev_disp;
    std::vector<AABB> blue_boxes;
    std::vector<Vec3> xnew_substep_start;
    std::vector<std::vector<int>> contact_adjacency;
    std::vector<std::vector<int>> combined_adjacency;
    std::vector<std::vector<int>> color_groups;

    bool matches(const RefMesh& ref_mesh, int nv) const {
        return mesh == &ref_mesh && tris_data == ref_mesh.tris.data() && dm_data == ref_mesh.Dm_inverse.data()
            && tris_size == ref_mesh.tris.size() && dm_size == ref_mesh.Dm_inverse.size() && hinges_size == ref_mesh.hinges.size() && num_vertices == nv;
    }

    void prepare(const RefMesh& ref_mesh, const VertexTriangleMap& adj,int nv, double initial_prev_disp) {
        if (!matches(ref_mesh, nv)) {
            elastic_adjacency = ElasticAdjacencyCache{};
            incident_triangles.assign(nv, {});
            for (const auto& [vi, row] : adj) {
                if (vi >= 0 && vi < nv) incident_triangles[vi] = row;
            }

            rest_shape_grads.resize(ref_mesh.Dm_inverse.size());
            for (int ti = 0; ti < static_cast<int>(ref_mesh.Dm_inverse.size()); ++ti)
                rest_shape_grads[ti] = shape_function_gradients(ref_mesh.Dm_inverse[ti]);

            prev_disp.assign(nv, initial_prev_disp);
            contact_adjacency.clear();
            combined_adjacency.clear();
            color_groups.clear();
            mesh = &ref_mesh;
            tris_data = ref_mesh.tris.data();
            dm_data = ref_mesh.Dm_inverse.data();
            tris_size = ref_mesh.tris.size();
            dm_size = ref_mesh.Dm_inverse.size();
            hinges_size = ref_mesh.hinges.size();
            num_vertices = nv;
        }

        pin_map.assign(nv, -1);
        blue_boxes.resize(nv);
        xnew_substep_start.resize(nv);
    }
};

struct OGCSolverWorkspace {
    ElasticAdjacencyCache elastic_adjacency;
    BroadPhase broad_phase;
    const RefMesh* mesh = nullptr;
    const int* tris_data = nullptr;
    const Mat22* dm_data = nullptr;
    std::size_t tris_size = 0;
    std::size_t dm_size = 0;
    std::size_t hinges_size = 0;
    int num_vertices = -1;

    std::vector<IncidentTriangles> incident_triangles;
    std::vector<ShapeGrads> rest_shape_grads;
    std::vector<double> prev_disp;
    std::vector<AABB> bvh_node_boxes;
    std::vector<std::vector<int>> color_groups;
    std::vector<Vec3> xnew_substep_start;
    std::vector<Vec3> xnew_copy;
    std::vector<double> bounds;

    bool matches(const RefMesh& ref_mesh, int nv) const {
        return mesh == &ref_mesh && tris_data == ref_mesh.tris.data()
            && dm_data == ref_mesh.Dm_inverse.data()
            && tris_size == ref_mesh.tris.size()
            && dm_size == ref_mesh.Dm_inverse.size()
            && hinges_size == ref_mesh.hinges.size() && num_vertices == nv;
    }

    void prepare(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int nv) {
        if (matches(ref_mesh, nv)) return;

        // BroadPhase retains topology internally, so replace it when the mesh topology changes rather than reusing stale connectivity
        broad_phase = BroadPhase{};
        elastic_adjacency = ElasticAdjacencyCache{};

        incident_triangles.assign(nv, {});
        for (const auto& [vi, row] : adj) {
            if (vi >= 0 && vi < nv) incident_triangles[vi] = row;
        }

        rest_shape_grads.resize(ref_mesh.Dm_inverse.size());
        for (int ti = 0; ti < static_cast<int>(ref_mesh.Dm_inverse.size()); ++ti)
            rest_shape_grads[ti] = shape_function_gradients(ref_mesh.Dm_inverse[ti]);

        mesh = &ref_mesh;
        tris_data = ref_mesh.tris.data();
        dm_data = ref_mesh.Dm_inverse.data();
        tris_size = ref_mesh.tris.size();
        dm_size = ref_mesh.Dm_inverse.size();
        hinges_size = ref_mesh.hinges.size();
        num_vertices = nv;
    }
};

}  // namespace

// -----------------------------------------------------------------------------
// Deformable local Newton systems
// -----------------------------------------------------------------------------

// Elastic and barrier terms both read the current live GS iterate.
Vec3 gs_vertex_delta_live_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                  const std::vector<Vec3>& xhat, std::vector<Vec3>& x, const BroadPhase& broad_phase, const PinMap* pin_map,
                                  const IncidentTriangles* incident_triangles = nullptr,
                                  const std::vector<ShapeGrads>* rest_shape_grads = nullptr) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map, incident_triangles, rest_shape_grads);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;
        const double d_hat2 = params.d_hat * params.d_hat;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            if (!node_triangle_aabbs_within_distance(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], d_hat2))
                continue;
            auto [bg, bH] = node_triangle_barrier_self_gradient_and_hessian(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            if (!segment_aabbs_within_distance(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], d_hat2))
                continue;
            auto [bg, bH] = segment_segment_barrier_self_gradient_and_hessian(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }
    }

    return matrix3d_inverse(H) * g;
}

// Elastic terms read x_elastic (live, GS-style across colors); barrier terms read
// x_barrier (iteration-start snapshot, Jacobi-style). Safe to call in parallel
// within a single elastic-coloring color class.
Vec3 gs_vertex_delta_frozen_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                    const std::vector<Vec3>& xhat, const std::vector<Vec3>& x_elastic, const std::vector<Vec3>& x_barrier, const BroadPhase& broad_phase, const PinMap* pin_map,
                                    const IncidentTriangles* incident_triangles,  const std::vector<ShapeGrads>* rest_shape_grads) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x_elastic, xhat, pin_map, incident_triangles, rest_shape_grads);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;
        const double d_hat2 = params.d_hat * params.d_hat;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            if (!node_triangle_aabbs_within_distance(x_barrier[p.node], x_barrier[p.tri_v[0]], x_barrier[p.tri_v[1]], x_barrier[p.tri_v[2]], d_hat2))
                continue;
            auto [bg, bH] = node_triangle_barrier_self_gradient_and_hessian(x_barrier[p.node], x_barrier[p.tri_v[0]], x_barrier[p.tri_v[1]], x_barrier[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            if (!segment_aabbs_within_distance(x_barrier[p.v[0]], x_barrier[p.v[1]], x_barrier[p.v[2]], x_barrier[p.v[3]], d_hat2))
                continue;
            auto [bg, bH] = segment_segment_barrier_self_gradient_and_hessian(x_barrier[p.v[0]], x_barrier[p.v[1]], x_barrier[p.v[2]], x_barrier[p.v[3]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }
    }

    return matrix3d_inverse(H) * g;
}


// -----------------------------------------------------------------------------
// Deformable solver entry points
// -----------------------------------------------------------------------------

SolverResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        const std::vector<Vec3>& v,
                                        BroadPhase& broad_phase,
                                        const std::string& outdir, bool verbose) {

    //create node (blue) boxes and create broad phase (red boxes) accordingly
    const int nv = static_cast<int>(xnew.size());
    static BasicSolverWorkspace workspace;
    workspace.prepare(ref_mesh, adj, nv, params.node_box_max);

    PinMap& pm = workspace.pin_map;
    for (int pi = 0; pi < static_cast<int>(pins.size()); ++pi)
        pm[pins[pi].vertex_index] = pi;
    std::vector<double>& prev_disp = workspace.prev_disp;
    constexpr double node_box_padding = 1.2;
    const double dt = params.dt();
    auto node_box_size_fn = [&](int vi) {
        const double inertial = v[vi].norm() * dt;
        return std::clamp(std::max(prev_disp[vi], inertial) * node_box_padding, params.node_box_min, params.node_box_max);
    };
    std::vector<AABB>& blue_boxes = workspace.blue_boxes;

    // Elastic adjacency depends only on mesh topology, so reuse it across GS calls.
    const std::vector<std::vector<int>>& ea = workspace.elastic_adjacency.get(ref_mesh, adj, nv);
    std::vector<std::vector<int>>& bca = workspace.contact_adjacency;
    std::vector<std::vector<int>>& combined_adj = workspace.combined_adjacency;
    std::vector<std::vector<int>>& color_groups = workspace.color_groups;

    SolverResult result;
    // anchor for clip boxes and prev_disp
    std::vector<Vec3>& xnew_substep_start = workspace.xnew_substep_start;
    xnew_substep_start = xnew;
 
    double r1=0.;
    //gs loop
    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if((iter-1)%params.node_box_update_count==0){//rebuild node boxes and color accordingly
            if (verbose) fprintf(stderr, "  [GS] iter %d  rebuilding node boxes\n", iter);
            //create new node boxes
            for (int i = 0; i < nv; ++i) {
                const double r = node_box_size_fn(i);
                blue_boxes[i] = AABB(xnew[i] - Vec3::Constant(r), xnew[i] + Vec3::Constant(r));
            }
            //rebuild bvh and pairs
            broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat);
            build_contact_adj(broad_phase.cache(), static_cast<int>(xnew.size()), bca);
            //color
            union_adjacency(ea, bca, combined_adj);
            greedy_color_conflict_graph(combined_adj, color_groups);
        }

        if (iter == 1 && !params.fixed_iters) {
            r1 = compute_global_residual(ref_mesh,adj,pins,params,xnew,xhat,broad_phase,&pm);
            result.has_residual = true;
            result.initial_residual = r1;
            result.final_residual = r1;
            if(r1 < params.tol_rel * r1 || r1 < params.tol_abs){
                result.converged = true;
                break;
            }
        }

        std::atomic<int> clip_count{0};
        per_vertex_safe_step(broad_phase, xnew, [&](int vi) -> Vec3 {
                                             return xnew[vi] - params.damping * gs_vertex_delta_live_barrier(
                                                     vi, ref_mesh, adj, pins, params, xhat, xnew,
                                                     broad_phase, &pm,
                                                     &workspace.incident_triangles[vi],
                                                     &workspace.rest_shape_grads);
                                         },
                                         /*safety=*/0.9, /*clip_ccd=*/params.use_ogc ? false : params.use_ccd,
                                         /*use_ticcd=*/params.use_ticcd,
                                         /*use_ogc=*/params.use_ogc,
                                         params.use_parallel ? &color_groups : nullptr,
                                         verbose ? &clip_count : nullptr);

        result.iterations = iter;
        if (!params.fixed_iters){
            double residual = compute_global_residual(ref_mesh,adj,pins,params,xnew,xhat,broad_phase,&pm);
            result.final_residual = residual;
            if (verbose)
                fprintf(stderr, "  [GS] iter %d  residual = %.6e  node clips = %d\n", iter, residual, clip_count.load());
            if(residual < params.tol_rel * r1 || residual < params.tol_abs){
                result.converged = true;
                break;
            }
        }
    }

    //record displacement over sub step
    for (int i = 0; i < nv; ++i)
        prev_disp[i] = (xnew[i] - xnew_substep_start[i]).norm();

    if (params.fixed_iters) result.converged = true;

    //write substep data
    if (params.write_substeps) {
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, &color_groups);
    }

    return result;
}

SolverResult global_gauss_seidel_solver_ogc(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                            std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                            const std::vector<Vec3>& /*v*/,
                                            const std::string& outdir) {
    if (!params.fixed_iters) {
        fprintf(stderr, "global_gauss_seidel_solver_ogc: params.fixed_iters must be true\n");
        exit(1);
    }

    const int nv = static_cast<int>(xnew.size());
    const PinMap pm = build_pin_map(pins, nv);

    static OGCSolverWorkspace workspace;
    workspace.prepare(ref_mesh, adj, nv);

    std::vector<double>& prev_disp = workspace.prev_disp;
    if (static_cast<int>(prev_disp.size()) != nv)
        prev_disp.assign(nv, params.node_box_max);
    constexpr double node_box_padding = 1.2;
    auto node_box_size_fn = [&](int vi) { return std::clamp(prev_disp[vi] * node_box_padding, params.node_box_min, params.node_box_max); };

    SolverResult result;
    result.iterations = 0;

    BroadPhase& broad_phase = workspace.broad_phase;
    std::vector<Vec3>& xnew_substep_start = workspace.xnew_substep_start;
    // anchor for clip boxes and prev_disp
    xnew_substep_start = xnew; 
    const double pad = std::max(params.ogc_box_pad, params.d_hat);

    std::vector<AABB>& bvh_node_boxes = workspace.bvh_node_boxes;
    bvh_node_boxes.resize(nv);
    for (int i = 0; i < nv; ++i) {
        const double r = node_box_size_fn(i) + pad;
        bvh_node_boxes[i] = AABB(xnew[i] - Vec3::Constant(r), xnew[i] + Vec3::Constant(r));
    }
    broad_phase.initialize(bvh_node_boxes, ref_mesh, pad);

    // Color from elastic adjacency only since barrier pairs are handled by reading  a frozen snapshot (xnew_copy) inside each color, so they don't need to constrain the coloring
    const std::vector<std::vector<int>>& elastic_adj = workspace.elastic_adjacency.get(ref_mesh, adj, nv);
    std::vector<std::vector<int>>& color_groups = workspace.color_groups;
    greedy_color_conflict_graph(elastic_adj, color_groups);

    if (params.write_substeps)
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, nullptr);

    auto& bp_cache = broad_phase.mutable_cache();

    std::vector<Vec3>& xnew_copy = workspace.xnew_copy;
    std::vector<double>& bounds = workspace.bounds;
    xnew_copy.resize(nv);
    bounds.resize(nv);

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if (iter > 1) {
            for (int vi = 0; vi < nv; ++vi) {
                const double R_vi = node_box_size_fn(vi);
                incremental_refresh_vertex(bp_cache, vi, xnew, ref_mesh, pad, R_vi + pad);
            }
            broad_phase.refresh_pairs(ref_mesh);
        }

        xnew_copy = xnew;

        #pragma omp parallel for schedule(static)
        for (int vi = 0; vi < nv; ++vi) {
            double b = compute_trust_region_bound_for_vertex(vi, xnew_copy, broad_phase, 0.4);
            if (!std::isfinite(b)) b = node_box_size_fn(vi);
            bounds[vi] = b;
        }

        for (const auto& color : color_groups) {
            const int csz = static_cast<int>(color.size());
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < csz; ++idx) {
                const int vi = color[idx];
                // Elastic stencil reads live xnew (GS across colors); barrier
                // stencil reads frozen xnew_copy (Jacobi).
                const Vec3 dx = - params.damping * gs_vertex_delta_frozen_barrier(vi, ref_mesh, adj, pins, params, xhat, xnew, xnew_copy, 
                    broad_phase, &pm, &workspace.incident_triangles[vi], &workspace.rest_shape_grads);
                if (dx.squaredNorm() < 1e-28) {
                    xnew[vi] = xnew_copy[vi];
                    continue;
                }
                const double dx_norm = dx.norm();
                const double toi = (dx_norm > 0.0) ? std::min(1.0, bounds[vi] / dx_norm) : 1.0;
                xnew[vi] = xnew_copy[vi] + toi * dx;
            }
        }

        result.iterations = iter;
    }

    for (int i = 0; i < nv; ++i)
        prev_disp[i] = (xnew[i] - xnew_substep_start[i]).norm();

    result.converged = true;
    return result;
}

// -----------------------------------------------------------------------------
// Rigid-body derivative assembly
// -----------------------------------------------------------------------------

namespace rb_solver {

const Vec3& rigid_node_body_space_position(int node, const RefMesh& ref_mesh, const std::vector<int>& node_to_rb_local) {
    const int rb = owning_rb_for_node(ref_mesh.node_to_rb, node);
    return ref_mesh.ref_positions[rb][node_to_rb_local[node]];
}

void construct_current_rigid_node_positions(const RefMesh& ref_mesh, const DeformedState& state, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, double dt, std::vector<Vec3>& positions) {
    positions = state.deformed_positions;
    for (int rb = 0; rb < static_cast<int>(ref_mesh.rb_nodes.size()); ++rb) {
        const Vec4 orientation = quaternion_from_angular_velocity(state.orientations[rb], omega_new[rb], dt);
        for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local) {
            const int node = ref_mesh.rb_nodes[rb][local];
            positions[node] = world_space_position(ref_mesh.ref_positions[rb][local], x_com_new[rb], orientation);
        }
    }
}

void add_rigid_derivatives(RigidEnergyDerivatives& total, const RigidEnergyDerivatives& contribution) {
    total.translation_gradient += contribution.translation_gradient;
    total.orientation_gradient += contribution.orientation_gradient;
    total.translation_translation_hessian += contribution.translation_translation_hessian;
    total.translation_orientation_hessian += contribution.translation_orientation_hessian;
    total.orientation_orientation_hessian += contribution.orientation_orientation_hessian;
}

RigidEnergyDerivatives rigid_barrier_derivatives(int rb, const RefMesh& ref_mesh, const DeformedState& state, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& omega_new, const SimParams& params, double dt, RigidDerivativeMode mode) {
    RigidEnergyDerivatives total;
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0)
        return total;
    const double d_hat2 = params.d_hat * params.d_hat;
    QuaternionOmegaKinematics kinematics;
    const bool needs_orientation_derivatives = mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::Gradient || mode == RigidDerivativeMode::OrientationHessian;
    const QuaternionOmegaKinematics* cached_kinematics = nullptr;
    if (needs_orientation_derivatives && (!nt_pair_indices.empty() || !ss_pair_indices.empty())) {
        const bool needs_second_derivatives = mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::OrientationHessian;
        kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt, needs_second_derivatives);
        cached_kinematics = &kinematics;
    }

    for (const int pair_index : nt_pair_indices) {
        const NodeTrianglePair& pair = bp_cache.nt_pairs[pair_index];
        const int node = pair.node;
        const int v0 = pair.tri_v[0];
        const int v1 = pair.tri_v[1];
        const int v2 = pair.tri_v[2];
        const int node_rb = owning_rb_for_node(ref_mesh.node_to_rb, node);
        const int triangle_rb = owning_rb_for_node(ref_mesh.node_to_rb, v0);
        if (node_rb == triangle_rb || (node_rb != rb && triangle_rb != rb) || !node_triangle_aabbs_within_distance(positions[node], positions[v0], positions[v1], positions[v2], d_hat2))
            continue;
        if (node_rb == rb) {
            const std::array<Vec3, 4> references = {rigid_node_body_space_position(node, ref_mesh, node_to_rb_local), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
            add_rigid_derivatives(total, node_triangle_barrier_rb(positions[node], positions[v0], positions[v1], positions[v2], references, RigidBarrierSide::FirstPrimitive, state.orientations[rb], omega_new[rb], dt, params.d_hat, mode, 1.0e-12, cached_kinematics));
        } else {
            const std::array<Vec3, 4> references = {Vec3::Zero(), rigid_node_body_space_position(v0, ref_mesh, node_to_rb_local), rigid_node_body_space_position(v1, ref_mesh, node_to_rb_local), rigid_node_body_space_position(v2, ref_mesh, node_to_rb_local)};
            add_rigid_derivatives(total, node_triangle_barrier_rb(positions[node], positions[v0], positions[v1], positions[v2], references, RigidBarrierSide::SecondPrimitive, state.orientations[rb], omega_new[rb], dt, params.d_hat, mode, 1.0e-12, cached_kinematics));
        }
    }

    for (const int pair_index : ss_pair_indices) {
        const SegmentSegmentPair& pair = bp_cache.ss_pairs[pair_index];
        const int a0 = pair.v[0];
        const int a1 = pair.v[1];
        const int b0 = pair.v[2];
        const int b1 = pair.v[3];
        const int first_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, a0);
        const int second_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, b0);
        if (first_edge_rb == second_edge_rb || (first_edge_rb != rb && second_edge_rb != rb) || !segment_aabbs_within_distance(positions[a0], positions[a1], positions[b0], positions[b1], d_hat2))
            continue;
        if (first_edge_rb == rb) {
            const std::array<Vec3, 4> references = {rigid_node_body_space_position(a0, ref_mesh, node_to_rb_local), rigid_node_body_space_position(a1, ref_mesh, node_to_rb_local), Vec3::Zero(), Vec3::Zero()};
            add_rigid_derivatives(total, segment_segment_barrier_rb(positions[a0], positions[a1], positions[b0], positions[b1], references, RigidBarrierSide::FirstPrimitive, state.orientations[rb], omega_new[rb], dt, params.d_hat, mode, 1.0e-12, cached_kinematics));
        } else {
            const std::array<Vec3, 4> references = {Vec3::Zero(), Vec3::Zero(), rigid_node_body_space_position(b0, ref_mesh, node_to_rb_local), rigid_node_body_space_position(b1, ref_mesh, node_to_rb_local)};
            add_rigid_derivatives(total, segment_segment_barrier_rb(positions[a0], positions[a1], positions[b0], positions[b1], references, RigidBarrierSide::SecondPrimitive, state.orientations[rb], omega_new[rb], dt, params.d_hat, mode, 1.0e-12, cached_kinematics));
        }
    }

    return total;
}

bool rigid_sdf_min_evaluation(const SimParams& params, const Vec3& x, SDFEvaluation& result) {
    bool found = false;
    result.phi = std::numeric_limits<double>::infinity();

    const auto consider = [&](const SDFEvaluation& candidate) {
        if (!found || candidate.phi < result.phi) {
            result = candidate;
            found = true;
        }
    };

    for (const PlaneSDF& plane : params.sdf_planes)
        consider(evaluate_sdf(plane, x));
    for (const CylinderSDF& cylinder : params.sdf_cylinders)
        consider(evaluate_sdf(cylinder, x));
    for (const SphereSDF& sphere : params.sdf_spheres)
        consider(evaluate_sdf(sphere, x));

    return found;
}

void add_rigid_sdf_gradients(const std::vector<Vec3>& ref_positions, const Vec3& x_com_new, const Vec4& q_n, const Vec3& omega_new, const SimParams& params, double dt, Vec3& translation_gradient, Vec3& orientation_gradient) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(q_n, omega_new, dt);
    for (const Vec3& X_centered : ref_positions) {
        const Vec3 x = world_space_position(X_centered, x_com_new, kinematics.orientation);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        const Vec3 gx = sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
        const Mat33 J_xomega = dx_domega(X_centered, kinematics);
        translation_gradient += dt2 * gx;
        orientation_gradient += dt2 * J_xomega.transpose() * gx;
    }
}

void add_rigid_sdf_translation_terms(const std::vector<Vec3>& ref_positions, const Vec3& x_com_new, const Vec4& q_n, const Vec3& omega_new, const SimParams& params, double dt, Vec3& gradient, Mat33& hessian) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    const Vec4 orientation = quaternion_from_angular_velocity(q_n, omega_new, dt);
    for (const Vec3& X_centered : ref_positions) {
        const Vec3 x = world_space_position(X_centered, x_com_new, orientation);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        gradient += dt2 * sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
        hessian += dt2 * rigid_node_translation_hessian(sdf_penalty_hessian(sdf, params.k_sdf, params.eps_sdf, false));
    }
}

void add_rigid_sdf_orientation_terms(const std::vector<Vec3>& ref_positions, const Vec3& x_com_new, const Vec4& q_n, const Vec3& omega_new, const SimParams& params, double dt, Vec3& gradient, Mat33& hessian) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(q_n, omega_new, dt);
    for (const Vec3& X_centered : ref_positions) {
        const Vec3 x = world_space_position(X_centered, x_com_new, kinematics.orientation);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        const RigidEnergyDerivatives derivatives = sdf_penalty_derivatives_rb(sdf, X_centered, kinematics, params.k_sdf, params.eps_sdf, false, false);
        gradient += dt2 * derivatives.orientation_gradient;
        hessian += dt2 * derivatives.orientation_orientation_hessian;
    }
}

void validate_rigid_solver_state(const RefMesh& ref_mesh, const DeformedState& state, const std::vector<Vec3>& x_com_new, const std::vector<Vec4>& q_new, const std::vector<Vec3>& omega_new) {
    const std::size_t num_rbs = ref_mesh.total_mass.size();
    const bool valid = ref_mesh.I_hat.size() == num_rbs
        && ref_mesh.rb_nodes.size() == num_rbs
        && ref_mesh.ref_positions.size() == num_rbs
        && state.x_coms.size() == num_rbs
        && state.v_coms.size() == num_rbs
        && state.orientations.size() == num_rbs
        && state.omega.size() == num_rbs
        && x_com_new.size() == num_rbs
        && q_new.size() == num_rbs
        && omega_new.size() == num_rbs;
    if (!valid)
        throw std::invalid_argument("global_gauss_seidel_solver_basic_rb: inconsistent rigid-body array sizes");
}

double rigid_body_unnormalized_residual(const RefMesh& ref_mesh, const DeformedState& state, const BroadPhase::Cache& bp_cache, const std::vector<std::vector<int>>& body_nt_pair_indices, const std::vector<std::vector<int>>& body_ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const SimParams& params, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, double dt) {
    double residual = 0.0;
    const int num_rbs = static_cast<int>(ref_mesh.total_mass.size());
    const double barrier_scale = dt * dt * params.k_barrier;
    for (int rb = 0; rb < num_rbs; ++rb) {
        Vec3 com_gradient = inertia_translation_gradient(x_com_new[rb], state.x_coms[rb], state.v_coms[rb], dt, ref_mesh.total_mass[rb]);
        com_gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);

        Vec3 orientation_gradient = inertia_rotation_gradient(omega_new[rb], state.orientations[rb], state.omega[rb], dt, ref_mesh.I_hat[rb]);
        add_rigid_sdf_gradients(ref_mesh.ref_positions[rb], x_com_new[rb], state.orientations[rb], omega_new[rb], params, dt, com_gradient, orientation_gradient);
        const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, bp_cache, body_nt_pair_indices[rb], body_ss_pair_indices[rb], node_to_rb_local, positions, omega_new, params, dt, RigidDerivativeMode::Gradient);
        com_gradient += barrier_scale * barrier.translation_gradient;
        orientation_gradient += barrier_scale * barrier.orientation_gradient;
        residual += com_gradient.norm() + orientation_gradient.norm();
    }
    return residual;
}

Vec3 compute_com_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, const SimParams& params, double dt) {
    const Vec3& x_com_n = state.x_coms[rb];
    const Vec3& v_com_n = state.v_coms[rb];

    Vec3 gradient = inertia_translation_gradient(x_com_new[rb], x_com_n, v_com_n, dt, ref_mesh.total_mass[rb]);
    gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);

    Mat33 hessian = inertia_translation_hessian(ref_mesh.total_mass[rb]);
    add_rigid_sdf_translation_terms(ref_mesh.ref_positions[rb], x_com_new[rb], state.orientations[rb], omega_new[rb], params, dt, gradient, hessian);
    const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, bp_cache, nt_pair_indices, ss_pair_indices, node_to_rb_local, positions, omega_new, params, dt, RigidDerivativeMode::TranslationHessian);
    const double barrier_scale = dt * dt * params.k_barrier;
    gradient += barrier_scale * barrier.translation_gradient;
    hessian += barrier_scale * barrier.translation_translation_hessian;
    return hessian.ldlt().solve(gradient);
}

Vec3 compute_omega_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, const SimParams& params, double dt) {
    const Vec4& q_n = state.orientations[rb];
    const Vec3& omega_n = state.omega[rb];
    const Mat33& I_hat = ref_mesh.I_hat[rb];

    auto [gradient, hessian] = inertia_rotation_gradient_hessian(omega_new[rb], q_n, omega_n, dt, I_hat);
    add_rigid_sdf_orientation_terms(ref_mesh.ref_positions[rb], x_com_new[rb], q_n, omega_new[rb], params, dt, gradient, hessian);
    const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, bp_cache, nt_pair_indices, ss_pair_indices, node_to_rb_local, positions, omega_new, params, dt, RigidDerivativeMode::OrientationHessian);
    const double barrier_scale = dt * dt * params.k_barrier;
    gradient += barrier_scale * barrier.orientation_gradient;
    hessian += barrier_scale * barrier.orientation_orientation_hessian;
    return hessian.ldlt().solve(gradient);
}

} // namespace rb_solver

// -----------------------------------------------------------------------------
// Rigid-body solver workspace
// -----------------------------------------------------------------------------

namespace {

struct RigidSolverWorkspace {
    BroadPhase broad_phase;
    const RefMesh* mesh = nullptr;
    const int* tris_data = nullptr;
    const std::vector<int>* rb_nodes_data = nullptr;
    const std::vector<Vec3>* ref_positions_data = nullptr;
    std::size_t tris_size = 0;
    std::size_t num_rbs = 0;
    int num_vertices = -1;
    std::vector<double> prev_com_disp;
    std::vector<double> prev_theta_disp;
    std::vector<Vec3> substep_start_coms;
    std::vector<Vec3> com_box_anchors;
    std::vector<Vec4> orientation_box_anchors;
    std::vector<double> com_box_radii;
    std::vector<double> theta_box_radii;
    std::vector<AABB> blue_boxes;
    std::vector<int> node_to_rb_local;
    std::vector<Vec3> positions;
    std::vector<std::vector<int>> body_nt_pair_indices;
    std::vector<std::vector<int>> body_ss_pair_indices;
    std::vector<std::vector<int>> contact_adjacency;
    std::vector<std::vector<int>> color_groups;

    bool matches(const RefMesh& ref_mesh, int nv) const {
        return mesh == &ref_mesh && tris_data == ref_mesh.tris.data() && rb_nodes_data == ref_mesh.rb_nodes.data() && ref_positions_data == ref_mesh.ref_positions.data() && tris_size == ref_mesh.tris.size() && num_rbs == ref_mesh.rb_nodes.size() && num_vertices == nv;
    }

    void prepare(const RefMesh& ref_mesh, int nv, double initial_com_disp, double initial_theta_disp) {
        if (!matches(ref_mesh, nv)) {
            broad_phase = BroadPhase{};
            prev_com_disp.assign(ref_mesh.rb_nodes.size(), initial_com_disp);
            prev_theta_disp.assign(ref_mesh.rb_nodes.size(), initial_theta_disp);
            body_nt_pair_indices.clear();
            body_ss_pair_indices.clear();
            contact_adjacency.clear();
            color_groups.clear();
            node_to_rb_local.assign(nv, -1);
            for (int rb = 0; rb < static_cast<int>(ref_mesh.rb_nodes.size()); ++rb) {
                for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local)
                    node_to_rb_local[ref_mesh.rb_nodes[rb][local]] = local;
            }
            mesh = &ref_mesh;
            tris_data = ref_mesh.tris.data();
            rb_nodes_data = ref_mesh.rb_nodes.data();
            ref_positions_data = ref_mesh.ref_positions.data();
            tris_size = ref_mesh.tris.size();
            num_rbs = ref_mesh.rb_nodes.size();
            num_vertices = nv;
        }
        substep_start_coms.resize(ref_mesh.rb_nodes.size());
        com_box_anchors.resize(ref_mesh.rb_nodes.size());
        orientation_box_anchors.resize(ref_mesh.rb_nodes.size());
        com_box_radii.resize(ref_mesh.rb_nodes.size());
        theta_box_radii.resize(ref_mesh.rb_nodes.size());
        blue_boxes.resize(nv);
        positions.resize(nv);
    }
};

} // namespace

// -----------------------------------------------------------------------------
// Rigid-body solver entry point
// -----------------------------------------------------------------------------

SolverResult global_gauss_seidel_solver_basic_rb(const RefMesh& ref_mesh, const DeformedState& state, const SimParams& params, std::vector<Vec3>& x_com_new, std::vector<Vec4>& q_new, std::vector<Vec3>& omega_new, bool verbose) {
    rb_solver::validate_rigid_solver_state(ref_mesh, state, x_com_new, q_new, omega_new);

    SolverResult result;
    const int num_rbs = static_cast<int>(ref_mesh.total_mass.size());
    const double dt = params.dt();
    static RigidSolverWorkspace workspace;
    workspace.prepare(ref_mesh, static_cast<int>(state.deformed_positions.size()), params.node_box_max, params.theta_box_max);

    // The caller supplies the initial collision-free configuration, with
    // omega_new storing the rotation increment from q_n. The previous physical
    // angular velocity remains in state.omega and enters the inertial energy.
    workspace.substep_start_coms = x_com_new;
    rb_solver::construct_current_rigid_node_positions(ref_mesh, state, x_com_new, omega_new, dt, workspace.positions);

    double initial_residual = 0.0;

    auto residual_converged = [&](double value) {
        double tolerance = 0.0;
        if (params.tol_abs > 0.0)
            tolerance = std::max(tolerance, params.tol_abs);
        if (params.tol_rel > 0.0 && std::isfinite(initial_residual))
            tolerance = std::max(tolerance, params.tol_rel * initial_residual);
        return value <= tolerance;
    };

    const auto rebuild_contact_cache = [&](int iter) {
        constexpr double box_padding = 1.2;
        for (int rb = 0; rb < num_rbs; ++rb) {
            workspace.com_box_anchors[rb] = x_com_new[rb];
            workspace.orientation_box_anchors[rb] = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_new[rb], dt));
            workspace.com_box_radii[rb] = std::clamp(box_padding * std::max(workspace.prev_com_disp[rb], dt * state.v_coms[rb].norm()), params.node_box_min, params.node_box_max);
            workspace.theta_box_radii[rb] = std::clamp(box_padding * std::max(workspace.prev_theta_disp[rb], dt * state.omega[rb].norm()), params.theta_box_min, params.theta_box_max);
        }
        build_blue_boxes_rb(workspace.com_box_anchors, workspace.orientation_box_anchors, workspace.theta_box_radii, workspace.com_box_radii, ref_mesh, workspace.blue_boxes);
        workspace.broad_phase.initialize(workspace.blue_boxes, ref_mesh, params.d_hat);
        build_rb_contact_adj(workspace.broad_phase.cache(), ref_mesh.node_to_rb, num_rbs, workspace.body_nt_pair_indices, workspace.body_ss_pair_indices, workspace.contact_adjacency);
        greedy_color_conflict_graph(workspace.contact_adjacency, workspace.color_groups);
        if (verbose)
            std::fprintf(stderr, "  [RB GS] iter %d  rebuilding rigid blue boxes and coloring\n", iter);
    };

    rebuild_contact_cache(1);

    if (!params.fixed_iters) {
        initial_residual = rb_solver::rigid_body_unnormalized_residual(ref_mesh, state, workspace.broad_phase.cache(), workspace.body_nt_pair_indices, workspace.body_ss_pair_indices, workspace.node_to_rb_local, workspace.positions, params, x_com_new, omega_new, dt);
        result.has_residual = true;
        result.initial_residual = initial_residual;
        result.final_residual = initial_residual;

        if (residual_converged(initial_residual)) {
            result.converged = true;
            return result;
        }
    }

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if (iter > 1 && (iter - 1) % params.node_box_update_count == 0)
            rebuild_contact_cache(iter);

        const auto process_body = [&](int rb) {
            std::vector<Vec3>& node_positions = workspace.positions;
            const Vec3 delta_com = params.damping * rb_solver::compute_com_update(rb, state, ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], workspace.node_to_rb_local, node_positions, x_com_new, omega_new, params, dt);
            const Vec3 com_radius = Vec3::Constant(workspace.com_box_radii[rb]);
            const Vec3 com_target = (x_com_new[rb] - delta_com).cwiseMax(workspace.com_box_anchors[rb] - com_radius).cwiseMin(workspace.com_box_anchors[rb] + com_radius);
            const Vec3 proposed_com_displacement = com_target - x_com_new[rb];
            const double com_safe_step = per_rigid_body_translation_safe_step(ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], node_positions, rb, proposed_com_displacement);
            const Vec3 com_displacement = com_safe_step * proposed_com_displacement;
            x_com_new[rb] += com_displacement;
            for (const int node : ref_mesh.rb_nodes[rb])
                node_positions[node] += com_displacement;

            const Vec3 delta_omega = rb_solver::compute_omega_update(rb, state, ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], workspace.node_to_rb_local, node_positions, x_com_new, omega_new, params, dt);
            const Vec4 q_current = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_new[rb], dt));
            const Vec3 omega_target = omega_new[rb] - params.damping * delta_omega;
            const Vec4 q_target = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_target, dt));
            const Vec4 q_bounded = bound_quaternion(workspace.orientation_box_anchors[rb], q_current, q_target, workspace.theta_box_radii[rb]);
            const double rotation_safe_step = per_rigid_body_rotation_safe_step(ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], node_positions, rb, x_com_new[rb], q_current, q_bounded);
            const Vec4 q_accepted = interpolate_orientation_full_arc(q_current, q_bounded, rotation_safe_step);
            q_new[rb] = q_accepted;
            
            // const Vec4 q_n = quaternion_normalize(state.orientations[rb]);
            // const Vec4 q_dot = (q_accepted - q_n) / dt;
            // const Vec4 omega_quaternion = 2.0 * quaternion_multiply(q_dot, quaternion_inverse(q_accepted));
            // omega_new[rb] = omega_quaternion.tail<3>();
            omega_new[rb] = angular_velocity_from_orientation_full_arc(q_accepted, state.orientations[rb], dt);


            for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local)
                node_positions[ref_mesh.rb_nodes[rb][local]] = world_space_position(ref_mesh.ref_positions[rb][local], x_com_new[rb], q_accepted);
        };

        if (params.use_parallel) {
            #pragma omp parallel
            {
                for (const std::vector<int>& color : workspace.color_groups) {
                    #pragma omp for schedule(static)
                    for (int index = 0; index < static_cast<int>(color.size()); ++index)
                        process_body(color[index]);
                }
            }
        } else {
            for (int rb = 0; rb < num_rbs; ++rb)
                process_body(rb);
        }

        result.iterations = iter;
        if (!params.fixed_iters) {
            const double residual = rb_solver::rigid_body_unnormalized_residual(ref_mesh, state, workspace.broad_phase.cache(), workspace.body_nt_pair_indices, workspace.body_ss_pair_indices, workspace.node_to_rb_local, workspace.positions, params, x_com_new, omega_new, dt);
            result.final_residual = residual;
            if (verbose)
                std::fprintf(stderr, "  [RB GS] iter %d  residual = %.6e\n", iter, residual);
            if (residual_converged(residual)) {
                result.converged = true;
                break;
            }
        }
    }

    for (int rb = 0; rb < num_rbs; ++rb) {
        workspace.prev_com_disp[rb] = (x_com_new[rb] - workspace.substep_start_coms[rb]).norm();
        workspace.prev_theta_disp[rb] = dt * omega_new[rb].norm();
    }

    if (params.fixed_iters)
        result.converged = true;
    return result;
}
