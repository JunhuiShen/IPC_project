#include "solver.h"
#include "IPC_math.h"
#include "parallel_helper.h"
#include "barrier_energy.h"
#include "friction_energy.h"
#include "output.h"
#include "rigid_body_ipc.h"
#include "safe_step.h"
#include "solid_ipc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Deformable solver workspaces
// -----------------------------------------------------------------------------

namespace {

void validate_solver_friction_parameters(
    const SimParams& params, const char* caller) {
    if (!std::isfinite(params.friction_coefficient)
        || params.friction_coefficient < 0.0) {
        throw std::invalid_argument(
            std::string(caller)
            + ": friction_coefficient must be finite and nonnegative");
    }
    if (params.friction_coefficient > 0.0
        && (!std::isfinite(params.friction_velocity_epsilon)
            || params.friction_velocity_epsilon <= 0.0)) {
        throw std::invalid_argument(
            std::string(caller)
            + ": friction_velocity_epsilon must be finite and positive");
    }
}

const std::vector<Vec3>* resolve_friction_previous_positions(
    const SimParams& params, const std::vector<Vec3>& x,
    const std::vector<Vec3>& xhat, const std::vector<Vec3>& velocities,
    const std::vector<Vec3>* supplied_previous_positions,
    std::vector<Vec3>& reconstructed_previous_positions,
    const char* caller) {
    if (params.friction_coefficient == 0.0)
        return nullptr;
    if (supplied_previous_positions != nullptr) {
        if (supplied_previous_positions->size() != x.size()) {
            throw std::invalid_argument(
                std::string(caller)
                + ": previous_positions must match xnew.size()");
        }
        return supplied_previous_positions;
    }
    if (xhat.size() != x.size() || velocities.size() != x.size()) {
        throw std::invalid_argument(
            std::string(caller)
            + ": xhat and velocities must match xnew.size() to reconstruct previous_positions");
    }
    reconstructed_previous_positions.resize(x.size());
    const double dt = params.dt();
    for (std::size_t node = 0; node < x.size(); ++node)
        reconstructed_previous_positions[node] = xhat[node] - dt * velocities[node];
    return &reconstructed_previous_positions;
}

std::array<Vec3, 4> friction_node_triangle_positions(
    const NodeTrianglePair& pair, const std::vector<Vec3>& positions) {
    return {
        positions[static_cast<std::size_t>(pair.node)],
        positions[static_cast<std::size_t>(pair.tri_v[0])],
        positions[static_cast<std::size_t>(pair.tri_v[1])],
        positions[static_cast<std::size_t>(pair.tri_v[2])]};
}

std::array<Vec3, 4> friction_segment_segment_positions(
    const SegmentSegmentPair& pair, const std::vector<Vec3>& positions) {
    return {
        positions[static_cast<std::size_t>(pair.v[0])],
        positions[static_cast<std::size_t>(pair.v[1])],
        positions[static_cast<std::size_t>(pair.v[2])],
        positions[static_cast<std::size_t>(pair.v[3])]};
}

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
    std::vector<int> pinned_vertices;
    std::vector<IncidentTriangles> incident_triangles;
    std::vector<ShapeGrads> rest_shape_grads;
    std::vector<double> prev_disp;
    std::vector<double> inertial_disp;
    std::vector<AABB> blue_boxes;
    std::vector<Vec3> xnew_substep_start;
    std::vector<std::vector<int>> contact_adjacency;
    std::vector<std::vector<int>> combined_adjacency;
    std::vector<std::vector<int>> color_groups;
    std::vector<int> deformable_nodes;
    GreedyColoringWorkspace coloring_workspace;
    FrozenResidualWorkspace frozen_residual;

    bool matches(const RefMesh& ref_mesh, int nv) const {
        return mesh == &ref_mesh && tris_data == ref_mesh.tris.data() && dm_data == ref_mesh.Dm_inverse.data()
            && tris_size == ref_mesh.tris.size() && dm_size == ref_mesh.Dm_inverse.size() && hinges_size == ref_mesh.hinges.size() && num_vertices == nv;
    }

    void prepare(const RefMesh& ref_mesh, const VertexTriangleMap& adj,int nv, double initial_prev_disp) {
        const bool topology_matches = matches(ref_mesh, nv);
        if (!topology_matches) {
            elastic_adjacency = ElasticAdjacencyCache{};
            incident_triangles.assign(nv, {});
            for (const auto& [vi, row] : adj) {
                if (vi >= 0 && vi < nv) incident_triangles[vi] = row;
            }

            rest_shape_grads.resize(ref_mesh.Dm_inverse.size());
            for (int ti = 0; ti < static_cast<int>(ref_mesh.Dm_inverse.size()); ++ti)
                rest_shape_grads[ti] = shape_function_gradients(ref_mesh.Dm_inverse[ti]);

            prev_disp.assign(nv, initial_prev_disp);
            pin_map.assign(nv, -1);
            pinned_vertices.clear();
            contact_adjacency.clear();
            combined_adjacency.clear();
            color_groups.clear();
            deformable_nodes.resize(static_cast<std::size_t>(nv));
            for (int node = 0; node < nv; ++node) deformable_nodes[static_cast<std::size_t>(node)] = node;
            coloring_workspace = GreedyColoringWorkspace{};
            frozen_residual = FrozenResidualWorkspace{};
            mesh = &ref_mesh;
            tris_data = ref_mesh.tris.data();
            dm_data = ref_mesh.Dm_inverse.data();
            tris_size = ref_mesh.tris.size();
            dm_size = ref_mesh.Dm_inverse.size();
            hinges_size = ref_mesh.hinges.size();
            num_vertices = nv;
        } else {
            for (const int vertex : pinned_vertices)
                pin_map[vertex] = -1;
            pinned_vertices.clear();
        }

        inertial_disp.resize(nv);
        blue_boxes.resize(nv);
        xnew_substep_start.resize(nv);
    }
};

struct MixedAdjacencyWorkspace {
    const RefMesh* mesh = nullptr;
    const int* tris_data = nullptr;
    const int* tets_data = nullptr;
    const int* tet_nodes_data = nullptr;
    const int* surface_nodes_data = nullptr;
    const Hinge* hinges_data = nullptr;
    const int* node_to_rb_data = nullptr;
    const int* deformable_nodes_data = nullptr;
    std::size_t tris_size = 0;
    std::size_t tets_size = 0;
    std::size_t tet_nodes_size = 0;
    std::size_t surface_nodes_size = 0;
    std::size_t hinges_size = 0;
    std::size_t node_to_rb_size = 0;
    std::size_t deformable_nodes_size = 0;
    int num_vertices = -1;
    int num_rigid_bodies = -1;

    std::vector<int> cloth_nodes;
    std::vector<int> node_to_block;
    std::vector<unsigned char> solid_node_mask;
    std::vector<unsigned char> surface_node_mask;
    std::vector<std::vector<int>> conflict_adjacency;
    std::vector<std::size_t> elastic_row_sizes;
    std::vector<std::vector<int>> color_groups;
    GreedyColoringWorkspace coloring_workspace;

    bool matches(
        const RefMesh& ref_mesh,
        const std::vector<int>& deformable_nodes,
        int num_rbs,
        int nv) const {
        return mesh == &ref_mesh
            && tris_data == ref_mesh.tris.data()
            && tets_data == ref_mesh.tets.data()
            && tet_nodes_data == ref_mesh.tet_nodes.data()
            && surface_nodes_data == ref_mesh.surface_nodes.data()
            && hinges_data == ref_mesh.hinges.data()
            && node_to_rb_data == ref_mesh.node_to_rb.data()
            && deformable_nodes_data == deformable_nodes.data()
            && tris_size == ref_mesh.tris.size()
            && tets_size == ref_mesh.tets.size()
            && tet_nodes_size == ref_mesh.tet_nodes.size()
            && surface_nodes_size == ref_mesh.surface_nodes.size()
            && hinges_size == ref_mesh.hinges.size()
            && node_to_rb_size == ref_mesh.node_to_rb.size()
            && deformable_nodes_size == deformable_nodes.size()
            && num_vertices == nv
            && num_rigid_bodies == num_rbs;
    }

    void prepare(
        const RefMesh& ref_mesh,
        const std::vector<int>& deformable_nodes,
        int num_rbs,
        int nv) {
        if (matches(ref_mesh, deformable_nodes, num_rbs, nv))
            return;

        solid_node_mask.assign(static_cast<std::size_t>(nv), 0);
        for (const int node : ref_mesh.tet_nodes)
            solid_node_mask[static_cast<std::size_t>(node)] = 1;
        surface_node_mask.assign(static_cast<std::size_t>(nv), 0);
        for (const int node : ref_mesh.surface_nodes)
            surface_node_mask[static_cast<std::size_t>(node)] = 1;
        cloth_nodes.clear();
        cloth_nodes.reserve(deformable_nodes.size());
        for (const int node : deformable_nodes) {
            if (solid_node_mask[static_cast<std::size_t>(node)] == 0)
                cloth_nodes.push_back(node);
        }

        const int solid_begin = static_cast<int>(cloth_nodes.size());
        const int rigid_begin = solid_begin
            + static_cast<int>(ref_mesh.tet_nodes.size());
        node_to_block.assign(static_cast<std::size_t>(nv), -1);
        for (int cloth = 0; cloth < static_cast<int>(cloth_nodes.size()); ++cloth)
            node_to_block[static_cast<std::size_t>(cloth_nodes[cloth])] = cloth;
        for (int solid = 0; solid < static_cast<int>(ref_mesh.tet_nodes.size()); ++solid)
            node_to_block[static_cast<std::size_t>(ref_mesh.tet_nodes[solid])] = solid_begin + solid;
        for (int rb = 0; rb < num_rbs; ++rb) {
            for (const int node : ref_mesh.rb_nodes[static_cast<std::size_t>(rb)])
                node_to_block[static_cast<std::size_t>(node)] = rigid_begin + rb;
        }
        conflict_adjacency.clear();
        elastic_row_sizes.clear();
        color_groups.clear();
        coloring_workspace = GreedyColoringWorkspace{};

        mesh = &ref_mesh;
        tris_data = ref_mesh.tris.data();
        tets_data = ref_mesh.tets.data();
        tet_nodes_data = ref_mesh.tet_nodes.data();
        surface_nodes_data = ref_mesh.surface_nodes.data();
        hinges_data = ref_mesh.hinges.data();
        node_to_rb_data = ref_mesh.node_to_rb.data();
        deformable_nodes_data = deformable_nodes.data();
        tris_size = ref_mesh.tris.size();
        tets_size = ref_mesh.tets.size();
        tet_nodes_size = ref_mesh.tet_nodes.size();
        surface_nodes_size = ref_mesh.surface_nodes.size();
        hinges_size = ref_mesh.hinges.size();
        node_to_rb_size = ref_mesh.node_to_rb.size();
        deformable_nodes_size = deformable_nodes.size();
        num_vertices = nv;
        num_rigid_bodies = num_rbs;
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
                                  const IncidentTriangles* incident_triangles,
                                  const std::vector<ShapeGrads>* rest_shape_grads,
                                  const std::vector<Vec3>* previous_positions) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] =
        physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(
            vi, ref_mesh, adj, pins, params, x, xhat, pin_map,
            incident_triangles, rest_shape_grads, previous_positions);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;
        const double d_hat2 = params.d_hat * params.d_hat;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            if (!node_triangle_aabbs_within_distance(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], d_hat2))
                continue;
            if (params.friction_coefficient != 0.0) {
                const std::array<Vec3, 4> current_positions =
                    friction_node_triangle_positions(p, x);
                const NodeTriangleContactEvaluation contact_evaluation =
                    make_node_triangle_contact_evaluation(
                        current_positions, params.d_hat,
                        params.k_barrier);
                const auto [bg, bH] =
                    node_triangle_barrier_self_gradient_and_hessian(
                        current_positions[0], current_positions[1],
                        current_positions[2], current_positions[3],
                        entry.dof, contact_evaluation);
                g += dt2k * bg;
                H += dt2k * bH;
                const FrozenFrictionContact contact =
                    make_node_triangle_frozen_friction_contact(
                        current_positions,
                        friction_node_triangle_positions(
                            p, *previous_positions),
                        contact_evaluation, params.dt(),
                        params.friction_velocity_epsilon);
                const auto [fg, fH] =
                    frozen_friction_role_gradient_and_hessian(
                        contact, entry.dof,
                        params.friction_coefficient, params.dt2());
                g += fg;
                H += fH;
            } else {
                const auto [bg, bH] =
                    node_triangle_barrier_self_gradient_and_hessian(
                        x[p.node], x[p.tri_v[0]], x[p.tri_v[1]],
                        x[p.tri_v[2]], params.d_hat, entry.dof);
                g += dt2k * bg;
                H += dt2k * bH;
            }
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            if (!segment_aabbs_within_distance(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], d_hat2))
                continue;
            if (params.friction_coefficient != 0.0) {
                const std::array<Vec3, 4> current_positions =
                    friction_segment_segment_positions(p, x);
                const SegmentSegmentContactEvaluation contact_evaluation =
                    make_segment_segment_contact_evaluation(
                        current_positions, params.d_hat,
                        params.k_barrier);
                const auto [bg, bH] =
                    segment_segment_barrier_self_gradient_and_hessian(
                        current_positions[0], current_positions[1],
                        current_positions[2], current_positions[3],
                        entry.dof, contact_evaluation);
                g += dt2k * bg;
                H += dt2k * bH;
                const FrozenFrictionContact contact =
                    make_segment_segment_frozen_friction_contact(
                        current_positions,
                        friction_segment_segment_positions(
                            p, *previous_positions),
                        contact_evaluation, params.dt(),
                        params.friction_velocity_epsilon);
                const auto [fg, fH] =
                    frozen_friction_role_gradient_and_hessian(
                        contact, entry.dof,
                        params.friction_coefficient, params.dt2());
                g += fg;
                H += fH;
            } else {
                const auto [bg, bH] =
                    segment_segment_barrier_self_gradient_and_hessian(
                        x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]],
                        params.d_hat, entry.dof);
                g += dt2k * bg;
                H += dt2k * bH;
            }
        }
    }

    return matrix3d_inverse(H) * g;
}

Vec3 gs_solid_vertex_delta_live_barrier(
    const int node, const RefMesh& ref_mesh,
    const std::vector<Pin>& pins, const SimParams& params,
    const std::vector<Vec3>& xhat, const std::vector<Vec3>& x,
    const BroadPhase& broad_phase,
    const std::vector<unsigned char>& solid_node_mask,
    const std::vector<unsigned char>& surface_node_mask,
    const PinMap& pin_map,
    const std::vector<Vec3>* previous_positions) {
    const auto [gradient, block] =
        solid_ipc_detail::compute_solid_local_gradient_and_block_unchecked(
            node, ref_mesh, pins, params, x, xhat, broad_phase,
            &solid_node_mask, &surface_node_mask, &pin_map,
            previous_positions);
    return matrix3d_inverse(block) * gradient;
}

// Elastic terms read x_elastic (live, GS-style across colors); barrier terms read
// x_barrier (iteration-start snapshot, Jacobi-style). Safe to call in parallel
// within a single elastic-coloring color class.
Vec3 gs_vertex_delta_frozen_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                    const std::vector<Vec3>& xhat, const std::vector<Vec3>& x_elastic, const std::vector<Vec3>& x_barrier, const BroadPhase& broad_phase, const PinMap* pin_map,
                                    const IncidentTriangles* incident_triangles,  const std::vector<ShapeGrads>* rest_shape_grads,
                                    const std::vector<Vec3>* previous_positions) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] =
        physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(
            vi, ref_mesh, adj, pins, params, x_elastic, xhat, pin_map,
            incident_triangles, rest_shape_grads, previous_positions);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;
        const double d_hat2 = params.d_hat * params.d_hat;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            if (!node_triangle_aabbs_within_distance(
                    x_barrier[p.node], x_barrier[p.tri_v[0]],
                    x_barrier[p.tri_v[1]], x_barrier[p.tri_v[2]],
                    d_hat2)) {
                continue;
            }
            if (params.friction_coefficient != 0.0) {
                const std::array<Vec3, 4> current_positions =
                    friction_node_triangle_positions(p, x_barrier);
                const NodeTriangleContactEvaluation contact_evaluation =
                    make_node_triangle_contact_evaluation(
                        current_positions, params.d_hat,
                        params.k_barrier);
                const auto [bg, bH] =
                    node_triangle_barrier_self_gradient_and_hessian(
                        current_positions[0], current_positions[1],
                        current_positions[2], current_positions[3],
                        entry.dof, contact_evaluation);
                g += dt2k * bg;
                H += dt2k * bH;
                const FrozenFrictionContact contact =
                    make_node_triangle_frozen_friction_contact(
                        current_positions,
                        friction_node_triangle_positions(
                            p, *previous_positions),
                        contact_evaluation, params.dt(),
                        params.friction_velocity_epsilon);
                const auto [fg, fH] =
                    frozen_friction_role_gradient_and_hessian(
                        contact, entry.dof,
                        params.friction_coefficient, params.dt2());
                g += fg;
                H += fH;
            } else {
                const auto [bg, bH] =
                    node_triangle_barrier_self_gradient_and_hessian(
                        x_barrier[p.node], x_barrier[p.tri_v[0]],
                        x_barrier[p.tri_v[1]], x_barrier[p.tri_v[2]],
                        params.d_hat, entry.dof);
                g += dt2k * bg;
                H += dt2k * bH;
            }
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            if (!segment_aabbs_within_distance(
                    x_barrier[p.v[0]], x_barrier[p.v[1]],
                    x_barrier[p.v[2]], x_barrier[p.v[3]], d_hat2)) {
                continue;
            }
            if (params.friction_coefficient != 0.0) {
                const std::array<Vec3, 4> current_positions =
                    friction_segment_segment_positions(p, x_barrier);
                const SegmentSegmentContactEvaluation contact_evaluation =
                    make_segment_segment_contact_evaluation(
                        current_positions, params.d_hat,
                        params.k_barrier);
                const auto [bg, bH] =
                    segment_segment_barrier_self_gradient_and_hessian(
                        current_positions[0], current_positions[1],
                        current_positions[2], current_positions[3],
                        entry.dof, contact_evaluation);
                g += dt2k * bg;
                H += dt2k * bH;
                const FrozenFrictionContact contact =
                    make_segment_segment_frozen_friction_contact(
                        current_positions,
                        friction_segment_segment_positions(
                            p, *previous_positions),
                        contact_evaluation, params.dt(),
                        params.friction_velocity_epsilon);
                const auto [fg, fH] =
                    frozen_friction_role_gradient_and_hessian(
                        contact, entry.dof,
                        params.friction_coefficient, params.dt2());
                g += fg;
                H += fH;
            } else {
                const auto [bg, bH] =
                    segment_segment_barrier_self_gradient_and_hessian(
                        x_barrier[p.v[0]], x_barrier[p.v[1]],
                        x_barrier[p.v[2]], x_barrier[p.v[3]],
                        params.d_hat, entry.dof);
                g += dt2k * bg;
                H += dt2k * bH;
            }
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
                                        const std::string& outdir,
                                        const std::vector<Vec3>* previous_positions) {

    //create node (blue) boxes and create broad phase (red boxes) accordingly
    validate_solver_friction_parameters(
        params, "global_gauss_seidel_solver_basic");
    std::vector<Vec3> reconstructed_previous_positions;
    previous_positions = resolve_friction_previous_positions(
        params, xnew, xhat, v, previous_positions,
        reconstructed_previous_positions,
        "global_gauss_seidel_solver_basic");
    const int nv = static_cast<int>(xnew.size());
    static BasicSolverWorkspace workspace;
    workspace.prepare(ref_mesh, adj, nv, params.node_box_max);

    PinMap& pm = workspace.pin_map;
    workspace.pinned_vertices.reserve(pins.size());
    for (int pi = 0; pi < static_cast<int>(pins.size()); ++pi) {
        pm[pins[pi].vertex_index] = pi;
        workspace.pinned_vertices.push_back(pins[pi].vertex_index);
    }
    std::vector<double>& prev_disp = workspace.prev_disp;
    std::vector<double>& inertial_disp = workspace.inertial_disp;
    constexpr double node_box_padding = 1.2;
    const double dt = params.dt();
    for (int vi = 0; vi < nv; ++vi)
        inertial_disp[vi] = v[vi].norm() * dt;
    auto node_box_size_fn = [&](int vi) {
        return std::clamp(std::max(prev_disp[vi], inertial_disp[vi]) * node_box_padding, params.node_box_min, params.node_box_max);
    };
    std::vector<AABB>& blue_boxes = workspace.blue_boxes;

    // Elastic adjacency depends only on mesh topology, so reuse it across GS calls.
    const std::vector<std::vector<int>>& ea = workspace.elastic_adjacency.get(ref_mesh, adj, nv);
    std::vector<std::vector<int>>& bca = workspace.contact_adjacency;
    std::vector<std::vector<int>>& combined_adj = workspace.combined_adjacency;
    std::vector<std::vector<int>>& color_groups = workspace.color_groups;
    const auto compute_residual = [&]() {
        build_frozen_residual_workspace(
            ref_mesh, params, xnew, broad_phase,
            workspace.frozen_residual, &workspace.rest_shape_grads);
        return compute_global_deformable_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, workspace.deformable_nodes, &pm, &workspace.incident_triangles, &workspace.rest_shape_grads, &workspace.frozen_residual, previous_positions);
    };

    SolverResult result;
    // anchor for clip boxes and prev_disp
    std::vector<Vec3>& xnew_substep_start = workspace.xnew_substep_start;
    xnew_substep_start = xnew;
 
    double r1=0.;
    //gs loop
    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if((iter-1)%params.node_box_update_count==0){//rebuild node boxes and color accordingly
            if (params.verbose)
                std::fprintf(stderr, "  [GS] iter %d  rebuilding node boxes\n", iter);
            //create new node boxes
            for (int i = 0; i < nv; ++i) {
                const double r = node_box_size_fn(i);
                blue_boxes[i] = AABB(xnew[i] - Vec3::Constant(r), xnew[i] + Vec3::Constant(r));
            }
            //rebuild bvh and pairs
            broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::DeformableSolver);
            build_contact_adj(broad_phase.cache(), static_cast<int>(xnew.size()), bca);
            //color
            union_adjacency(ea, bca, combined_adj);
            greedy_color_conflict_graph(combined_adj, color_groups, &workspace.coloring_workspace);
            const BroadPhase::Cache& bp_cache = broad_phase.cache();
            // Vertices in one color share no dependencies, so process contact-heavy vertices first to avoid end-of-color stragglers.
            for (std::vector<int>& group : color_groups) std::stable_sort(group.begin(), group.end(), [&](const int a, const int b) { return bp_cache.vertex_nt[static_cast<std::size_t>(a)].size() + bp_cache.vertex_ss[static_cast<std::size_t>(a)].size() > bp_cache.vertex_nt[static_cast<std::size_t>(b)].size() + bp_cache.vertex_ss[static_cast<std::size_t>(b)].size(); });
        }

        if (iter == 1 && !params.fixed_iters) {
            r1 = compute_residual();
            result.has_residual = true;
            result.initial_residual = r1;
            result.final_residual = r1;
            if(r1 < params.tol_rel * r1 || r1 < params.tol_abs){
                result.converged = true;
                break;
            }
        }

        const auto proposed_position = [&](int vi) -> Vec3 { return xnew[vi] - params.damping * gs_vertex_delta_live_barrier(vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm, &workspace.incident_triangles[vi], &workspace.rest_shape_grads, previous_positions); };
        const auto process_vertex = [&](int vi) { per_vertex_safe_step(broad_phase, xnew, vi, proposed_position(vi), 0.9, params.use_ogc ? false : params.use_ccd, params.use_ticcd, params.use_ogc); };
        if (params.use_parallel) {
            #pragma omp parallel
            {
                for (const std::vector<int>& group : color_groups) {
                    #pragma omp for schedule(dynamic, 1)
                    for (int i = 0; i < static_cast<int>(group.size()); ++i) process_vertex(group[static_cast<std::size_t>(i)]);
                }
            }
        } else {
            for (int vi = 0; vi < nv; ++vi) process_vertex(vi);
        }

        result.iterations = iter;
        if (!params.fixed_iters){
            double residual = compute_residual();
            result.final_residual = residual;
            if (params.verbose)
                std::fprintf(stderr, "  [GS] iter %d  residual = %.6e\n", iter, residual);
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
                                            const std::vector<Vec3>& v,
                                            const std::string& outdir,
                                            const std::vector<Vec3>* previous_positions) {
    validate_solver_friction_parameters(
        params, "global_gauss_seidel_solver_ogc");
    std::vector<Vec3> reconstructed_previous_positions;
    previous_positions = resolve_friction_previous_positions(
        params, xnew, xhat, v, previous_positions,
        reconstructed_previous_positions,
        "global_gauss_seidel_solver_ogc");
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
                    broad_phase, &pm, &workspace.incident_triangles[vi], &workspace.rest_shape_grads, previous_positions);
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

// If friction_output is supplied, normal and friction derivatives are assembled
// in the same pair traversal. They retain separate accumulation order, while
// sharing one ephemeral contact evaluation from this unchanged rigid-position
// snapshot.
RigidEnergyDerivatives rigid_barrier_derivatives(int rb, const RefMesh& ref_mesh, const DeformedState& state, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& omega_new, const SimParams& params, double dt, RigidDerivativeMode mode, const QuaternionOmegaKinematics* supplied_kinematics = nullptr, const FrozenResidualWorkspace* frozen_workspace = nullptr, RigidEnergyDerivatives* friction_output = nullptr, bool assemble_barrier = true) {
    RigidEnergyDerivatives total;
    if (friction_output != nullptr)
        *friction_output = RigidEnergyDerivatives{};
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0)
        return total;
    const bool assemble_friction = friction_output != nullptr
        && params.friction_coefficient != 0.0;
    // Production solvers validate once at their entry point; the public
    // friction-only wrapper does the same before reaching this hot assembler.
    const double d_hat2 = params.d_hat * params.d_hat;
    QuaternionOmegaKinematics kinematics;
    const bool mode_requests_orientation =
        mode == RigidDerivativeMode::Full
        || mode == RigidDerivativeMode::Gradient
        || mode == RigidDerivativeMode::OrientationHessian;
    const bool needs_orientation_derivatives = mode_requests_orientation
        && (assemble_barrier
            || (assemble_friction
                && updates_rigid_orientation(
                    ref_mesh.rb_update_modes[rb])));
    const QuaternionOmegaKinematics* cached_kinematics = supplied_kinematics;
    if (needs_orientation_derivatives && cached_kinematics == nullptr && (!nt_pair_indices.empty() || !ss_pair_indices.empty())) {
        const bool needs_second_derivatives = mode == RigidDerivativeMode::Full || mode == RigidDerivativeMode::OrientationHessian;
        kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt, needs_second_derivatives);
        cached_kinematics = &kinematics;
    }
    const auto add_frozen_gradient = [&](RigidEnergyDerivatives& output, const std::array<Vec3, 4>& references, const std::array<Vec3, 4>& gradients, int first_dof, int last_dof) {
        RigidEnergyDerivatives contribution;
        for (int dof = first_dof; dof <= last_dof; ++dof) {
            contribution.translation_gradient += gradients[static_cast<std::size_t>(dof)];
            contribution.orientation_gradient += dx_domega(references[static_cast<std::size_t>(dof)], *cached_kinematics).transpose() * gradients[static_cast<std::size_t>(dof)];
        }
        add_rigid_derivatives(output, contribution);
    };
    const RigidBodyUpdateMode update_mode = ref_mesh.rb_update_modes[rb];
    const bool translation_enabled = updates_rigid_translation(update_mode);
    const bool orientation_enabled = updates_rigid_orientation(update_mode);
    const bool translation_gradient_requested = translation_enabled
        && (mode == RigidDerivativeMode::Full
            || mode == RigidDerivativeMode::Gradient
            || mode == RigidDerivativeMode::TranslationHessian);
    const bool orientation_gradient_requested = orientation_enabled
        && (mode == RigidDerivativeMode::Full
            || mode == RigidDerivativeMode::Gradient
            || mode == RigidDerivativeMode::OrientationHessian);
    const bool translation_hessian_requested = translation_enabled
        && (mode == RigidDerivativeMode::Full
            || mode == RigidDerivativeMode::TranslationHessian);
    const bool orientation_hessian_requested = orientation_enabled
        && (mode == RigidDerivativeMode::Full
            || mode == RigidDerivativeMode::OrientationHessian);
    const bool mixed_hessian_requested = mode == RigidDerivativeMode::Full
        && translation_enabled && orientation_enabled;
    const double friction_dt2 = dt * dt;
    const auto add_friction_contact = [&](RigidEnergyDerivatives& output, const std::array<int, 4>& nodes, const FrozenFrictionContact& contact) {
        if (!assemble_friction || !contact.active)
            return;

        double translation_weight = 0.0;
        Mat33 orientation_jacobian = Mat33::Zero();
        bool body_owns_role = false;
        for (int role = 0; role < 4; ++role) {
            const int node = nodes[static_cast<std::size_t>(role)];
            if (owning_rb_for_node(ref_mesh.node_to_rb, node) != rb)
                continue;
            body_owns_role = true;
            const double weight = contact.weights[static_cast<std::size_t>(role)];
            translation_weight += weight;
            if (orientation_gradient_requested
                || orientation_hessian_requested) {
                orientation_jacobian += weight * dx_domega(
                    rigid_node_body_space_position(
                        node, ref_mesh, node_to_rb_local),
                    *cached_kinematics);
            }
        }
        if (!body_owns_role)
            return;

        const Mat33 translation_jacobian =
            translation_weight * Mat33::Identity();
        const bool hessian_requested = translation_hessian_requested
            || orientation_hessian_requested || mixed_hessian_requested;
        Vec3 relative_gradient;
        Mat33 relative_hessian = Mat33::Zero();
        if (hessian_requested) {
            const auto derivatives =
                frozen_friction_relative_gradient_and_hessian(
                    contact, params.friction_coefficient, friction_dt2);
            relative_gradient = derivatives.first;
            relative_hessian = derivatives.second;
        } else {
            relative_gradient = frozen_friction_relative_gradient(
                contact, params.friction_coefficient, friction_dt2);
        }
        if (translation_gradient_requested) {
            output.translation_gradient += translation_jacobian.transpose() * relative_gradient;
        }
        if (orientation_gradient_requested) {
            output.orientation_gradient += orientation_jacobian.transpose() * relative_gradient;
        }

        if (!hessian_requested)
            return;
        if (translation_hessian_requested) {
            output.translation_translation_hessian += translation_jacobian.transpose() * relative_hessian * translation_jacobian;
        }
        if (mixed_hessian_requested) {
            output.translation_orientation_hessian += translation_jacobian.transpose() * relative_hessian * orientation_jacobian;
        }
        if (orientation_hessian_requested) {
            output.orientation_orientation_hessian += orientation_jacobian.transpose() * relative_hessian * orientation_jacobian;
        }
    };
    const auto evaluate_nt_pair = [&](const int pair_index, const bool aabb_already_active, RigidEnergyDerivatives& barrier_output, RigidEnergyDerivatives& friction_pair_output) -> bool {
        const NodeTrianglePair& pair = bp_cache.nt_pairs[static_cast<std::size_t>(pair_index)];
        const int node = pair.node;
        const int v0 = pair.tri_v[0];
        const int v1 = pair.tri_v[1];
        const int v2 = pair.tri_v[2];
        const int node_rb = owning_rb_for_node(ref_mesh.node_to_rb, node);
        const int triangle_rb = owning_rb_for_node(ref_mesh.node_to_rb, v0);
        const bool aabb_active = aabb_already_active || (frozen_workspace == nullptr ? node_triangle_aabbs_within_distance(positions[node], positions[v0], positions[v1], positions[v2], d_hat2) : frozen_workspace->nt_aabb_active[static_cast<std::size_t>(pair_index)] != 0);
        if (node_rb == triangle_rb || (node_rb != rb && triangle_rb != rb) || !aabb_active) return false;
        std::array<Vec3, 4> current_positions;
        NodeTriangleContactEvaluation contact_evaluation;
        const NodeTriangleContactEvaluation* precomputed_evaluation = nullptr;
        if (assemble_friction) {
            current_positions = friction_node_triangle_positions(pair, positions);
            contact_evaluation = make_node_triangle_contact_evaluation(current_positions, params.d_hat, params.k_barrier);
            precomputed_evaluation = &contact_evaluation;
        }
        const NodeTriangleDistanceResult* precomputed_distance = precomputed_evaluation == nullptr ? nullptr : &precomputed_evaluation->dr;
        const double* precomputed_b_prime = precomputed_evaluation == nullptr ? nullptr : &precomputed_evaluation->b_prime;
        const double* precomputed_b_double_prime = precomputed_evaluation == nullptr ? nullptr : &precomputed_evaluation->b_double_prime;
        if (assemble_barrier) {
            if (node_rb == rb) {
                const std::array<Vec3, 4> references = {rigid_node_body_space_position(node, ref_mesh, node_to_rb_local), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
                if (mode == RigidDerivativeMode::Gradient && frozen_workspace != nullptr && frozen_workspace->nt_barrier_active[static_cast<std::size_t>(pair_index)] == 0) add_rigid_derivatives(barrier_output, RigidEnergyDerivatives{});
                else if (mode == RigidDerivativeMode::Gradient && frozen_workspace != nullptr && frozen_workspace->nt_gradient_cached[static_cast<std::size_t>(pair_index)] != 0) add_frozen_gradient(barrier_output, references, frozen_workspace->nt_gradients[static_cast<std::size_t>(pair_index)], 0, 0);
                else add_rigid_derivatives(barrier_output, node_triangle_barrier_rb(positions[node], positions[v0], positions[v1], positions[v2], references, RigidBarrierSide::FirstPrimitive, state.orientations[rb], omega_new[rb], dt, params.d_hat, mode, 1.0e-12, cached_kinematics, precomputed_distance, precomputed_b_prime, precomputed_b_double_prime));
            } else {
                const std::array<Vec3, 4> references = {Vec3::Zero(), rigid_node_body_space_position(v0, ref_mesh, node_to_rb_local), rigid_node_body_space_position(v1, ref_mesh, node_to_rb_local), rigid_node_body_space_position(v2, ref_mesh, node_to_rb_local)};
                if (mode == RigidDerivativeMode::Gradient && frozen_workspace != nullptr && frozen_workspace->nt_barrier_active[static_cast<std::size_t>(pair_index)] == 0) add_rigid_derivatives(barrier_output, RigidEnergyDerivatives{});
                else if (mode == RigidDerivativeMode::Gradient && frozen_workspace != nullptr && frozen_workspace->nt_gradient_cached[static_cast<std::size_t>(pair_index)] != 0) add_frozen_gradient(barrier_output, references, frozen_workspace->nt_gradients[static_cast<std::size_t>(pair_index)], 1, 3);
                else add_rigid_derivatives(barrier_output, node_triangle_barrier_rb(positions[node], positions[v0], positions[v1], positions[v2], references, RigidBarrierSide::SecondPrimitive, state.orientations[rb], omega_new[rb], dt, params.d_hat, mode, 1.0e-12, cached_kinematics, precomputed_distance, precomputed_b_prime, precomputed_b_double_prime));
            }
        }
        if (assemble_friction) {
            const FrozenFrictionContact contact = make_node_triangle_frozen_friction_contact(current_positions, friction_node_triangle_positions(pair, state.deformed_positions), contact_evaluation, dt, params.friction_velocity_epsilon);
            add_friction_contact(friction_pair_output, {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]}, contact);
        }
        return true;
    };

    const auto evaluate_ss_pair = [&](const int pair_index, const bool aabb_already_active, RigidEnergyDerivatives& barrier_output, RigidEnergyDerivatives& friction_pair_output) -> bool {
        const SegmentSegmentPair& pair = bp_cache.ss_pairs[static_cast<std::size_t>(pair_index)];
        const int a0 = pair.v[0];
        const int a1 = pair.v[1];
        const int b0 = pair.v[2];
        const int b1 = pair.v[3];
        const int first_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, a0);
        const int second_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, b0);
        const bool aabb_active = aabb_already_active || (frozen_workspace == nullptr ? segment_aabbs_within_distance(positions[a0], positions[a1], positions[b0], positions[b1], d_hat2) : frozen_workspace->ss_aabb_active[static_cast<std::size_t>(pair_index)] != 0);
        if (first_edge_rb == second_edge_rb || (first_edge_rb != rb && second_edge_rb != rb) || !aabb_active) return false;
        std::array<Vec3, 4> current_positions;
        SegmentSegmentContactEvaluation contact_evaluation;
        const SegmentSegmentContactEvaluation* precomputed_evaluation = nullptr;
        if (assemble_friction) {
            current_positions = friction_segment_segment_positions(pair, positions);
            contact_evaluation = make_segment_segment_contact_evaluation(current_positions, params.d_hat, params.k_barrier);
            precomputed_evaluation = &contact_evaluation;
        }
        const SegmentSegmentDistanceResult* precomputed_distance = precomputed_evaluation == nullptr ? nullptr : &precomputed_evaluation->dr;
        const double* precomputed_b_prime = precomputed_evaluation == nullptr ? nullptr : &precomputed_evaluation->b_prime;
        const double* precomputed_b_double_prime = precomputed_evaluation == nullptr ? nullptr : &precomputed_evaluation->b_double_prime;
        if (assemble_barrier) {
            if (first_edge_rb == rb) {
                const std::array<Vec3, 4> references = {rigid_node_body_space_position(a0, ref_mesh, node_to_rb_local), rigid_node_body_space_position(a1, ref_mesh, node_to_rb_local), Vec3::Zero(), Vec3::Zero()};
                if (mode == RigidDerivativeMode::Gradient && frozen_workspace != nullptr && frozen_workspace->ss_barrier_active[static_cast<std::size_t>(pair_index)] == 0) add_rigid_derivatives(barrier_output, RigidEnergyDerivatives{});
                else if (mode == RigidDerivativeMode::Gradient && frozen_workspace != nullptr && frozen_workspace->ss_gradient_cached[static_cast<std::size_t>(pair_index)] != 0) add_frozen_gradient(barrier_output, references, frozen_workspace->ss_gradients[static_cast<std::size_t>(pair_index)], 0, 1);
                else add_rigid_derivatives(barrier_output, segment_segment_barrier_rb(positions[a0], positions[a1], positions[b0], positions[b1], references, RigidBarrierSide::FirstPrimitive, state.orientations[rb], omega_new[rb], dt, params.d_hat, mode, 1.0e-12, cached_kinematics, precomputed_distance, precomputed_b_prime, precomputed_b_double_prime));
            } else {
                const std::array<Vec3, 4> references = {Vec3::Zero(), Vec3::Zero(), rigid_node_body_space_position(b0, ref_mesh, node_to_rb_local), rigid_node_body_space_position(b1, ref_mesh, node_to_rb_local)};
                if (mode == RigidDerivativeMode::Gradient && frozen_workspace != nullptr && frozen_workspace->ss_barrier_active[static_cast<std::size_t>(pair_index)] == 0) add_rigid_derivatives(barrier_output, RigidEnergyDerivatives{});
                else if (mode == RigidDerivativeMode::Gradient && frozen_workspace != nullptr && frozen_workspace->ss_gradient_cached[static_cast<std::size_t>(pair_index)] != 0) add_frozen_gradient(barrier_output, references, frozen_workspace->ss_gradients[static_cast<std::size_t>(pair_index)], 2, 3);
                else add_rigid_derivatives(barrier_output, segment_segment_barrier_rb(positions[a0], positions[a1], positions[b0], positions[b1], references, RigidBarrierSide::SecondPrimitive, state.orientations[rb], omega_new[rb], dt, params.d_hat, mode, 1.0e-12, cached_kinematics, precomputed_distance, precomputed_b_prime, precomputed_b_double_prime));
            }
        }
        if (assemble_friction) {
            const FrozenFrictionContact contact = make_segment_segment_frozen_friction_contact(current_positions, friction_segment_segment_positions(pair, state.deformed_positions), contact_evaluation, dt, params.friction_velocity_epsilon);
            add_friction_contact(friction_pair_output, {pair.v[0], pair.v[1], pair.v[2], pair.v[3]}, contact);
        }
        return true;
    };

    const auto accumulate_nt_pair = [&](const int pair_index, const bool aabb_already_active) {
        RigidEnergyDerivatives pair_barrier;
        RigidEnergyDerivatives pair_friction;
        if (evaluate_nt_pair(pair_index, aabb_already_active, pair_barrier, pair_friction)) {
            add_rigid_derivatives(total, pair_barrier);
            if (friction_output != nullptr) add_rigid_derivatives(*friction_output, pair_friction);
        }
    };
    const auto accumulate_ss_pair = [&](const int pair_index, const bool aabb_already_active) {
        RigidEnergyDerivatives pair_barrier;
        RigidEnergyDerivatives pair_friction;
        if (evaluate_ss_pair(pair_index, aabb_already_active, pair_barrier, pair_friction)) {
            add_rigid_derivatives(total, pair_barrier);
            if (friction_output != nullptr) add_rigid_derivatives(*friction_output, pair_friction);
        }
    };

    for (const int pair_index : nt_pair_indices) accumulate_nt_pair(pair_index, false);
    for (const int pair_index : ss_pair_indices) accumulate_ss_pair(pair_index, false);

    return total;
}

RigidEnergyDerivatives rigid_friction_derivatives(
    int rb, const RefMesh& ref_mesh, const DeformedState& state,
    const BroadPhase::Cache& bp_cache,
    const std::vector<int>& nt_pair_indices,
    const std::vector<int>& ss_pair_indices,
    const std::vector<int>& node_to_rb_local,
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>& omega_new,
    const SimParams& params, double dt, RigidDerivativeMode mode,
    const QuaternionOmegaKinematics* supplied_kinematics,
    const FrozenResidualWorkspace* frozen_workspace) {
    RigidEnergyDerivatives friction;
    // Preserve the public helper's legacy zero-friction behavior, including
    // not inspecting previous-position data in this mode.
    if (params.friction_coefficient == 0.0)
        return friction;
    validate_solver_friction_parameters(params, "rigid_friction_derivatives");
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0)
        return friction;
    if (positions.size() != state.deformed_positions.size()) {
        throw std::invalid_argument(
            "rigid_friction_derivatives: previous positions must match current positions");
    }
    const RigidBodyUpdateMode update_mode = ref_mesh.rb_update_modes[rb];
    const bool translation_enabled = updates_rigid_translation(update_mode);
    const bool orientation_enabled = updates_rigid_orientation(update_mode);
    const bool any_derivative_requested =
        (translation_enabled
         && (mode == RigidDerivativeMode::Full
             || mode == RigidDerivativeMode::Gradient
             || mode == RigidDerivativeMode::TranslationHessian))
        || (orientation_enabled
            && (mode == RigidDerivativeMode::Full
                || mode == RigidDerivativeMode::Gradient
                || mode == RigidDerivativeMode::OrientationHessian));
    if (!any_derivative_requested)
        return friction;

    // Production solver phases request both outputs together. Keep this
    // public friction-only entry point as a compatibility wrapper around the
    // same implementation so there is only one contact assembly to maintain.
    (void)rigid_barrier_derivatives(
        rb, ref_mesh, state, bp_cache, nt_pair_indices, ss_pair_indices,
        node_to_rb_local, positions, omega_new, params, dt, mode,
        supplied_kinematics, frozen_workspace, &friction, false);
    return friction;
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

void add_rigid_sdf_gradients(
    const std::vector<Vec3>& ref_positions,
    const std::vector<int>& nodes,
    const std::vector<Vec3>& previous_positions,
    const Vec3& x_com_new, const Vec4& q_n, const Vec3& omega_new,
    const SimParams& params, double dt, Vec3& translation_gradient,
    Vec3& orientation_gradient,
    const QuaternionOmegaKinematics* supplied_kinematics = nullptr) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    const QuaternionOmegaKinematics owned_kinematics = supplied_kinematics == nullptr ? quaternion_omega_kinematics(q_n, omega_new, dt) : QuaternionOmegaKinematics{};
    const QuaternionOmegaKinematics& kinematics = supplied_kinematics == nullptr ? owned_kinematics : *supplied_kinematics;
    for (std::size_t local = 0; local < ref_positions.size(); ++local) {
        const Vec3& X_centered = ref_positions[local];
        const Vec3 x = world_space_position(X_centered, x_com_new, kinematics.orientation);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        const Vec3 gx = sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
        const Mat33 J_xomega = dx_domega(X_centered, kinematics);
        translation_gradient += dt2 * gx;
        orientation_gradient += dt2 * J_xomega.transpose() * gx;
        if (params.friction_coefficient != 0.0) {
            const FrozenFrictionContact contact =
                make_sdf_frozen_friction_contact(
                    x,
                    previous_positions[static_cast<std::size_t>(
                        nodes[local])],
                    sdf, params.k_sdf, params.eps_sdf, dt,
                    params.friction_velocity_epsilon, 1.0e-12, &gx);
            const Vec3 friction_gradient =
                frozen_friction_relative_gradient(
                    contact, params.friction_coefficient, dt2);
            translation_gradient += friction_gradient;
            orientation_gradient +=
                J_xomega.transpose() * friction_gradient;
        }
    }
}

void add_rigid_sdf_translation_terms(
    const std::vector<Vec3>& ref_positions,
    const std::vector<int>& nodes,
    const std::vector<Vec3>& previous_positions,
    const Vec3& x_com_new, const Vec4& q_n, const Vec3& omega_new,
    const SimParams& params, double dt, Vec3& gradient, Mat33& hessian,
    const QuaternionOmegaKinematics* supplied_kinematics = nullptr) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    const Vec4 orientation = supplied_kinematics == nullptr ? quaternion_from_angular_velocity(q_n, omega_new, dt) : supplied_kinematics->orientation;
    for (std::size_t local = 0; local < ref_positions.size(); ++local) {
        const Vec3& X_centered = ref_positions[local];
        const Vec3 x = world_space_position(X_centered, x_com_new, orientation);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        const Vec3 gx =
            sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
        gradient += dt2 * gx;
        const Mat33 Hx =
            sdf_penalty_hessian(
                sdf, params.k_sdf, params.eps_sdf, false);
        hessian += dt2 * rigid_node_translation_hessian(Hx);
        if (params.friction_coefficient != 0.0) {
            const FrozenFrictionContact contact =
                make_sdf_frozen_friction_contact(
                    x,
                    previous_positions[static_cast<std::size_t>(
                        nodes[local])],
                    sdf, params.k_sdf, params.eps_sdf, dt,
                    params.friction_velocity_epsilon, 1.0e-12, &gx);
            const auto [friction_gradient, friction_hessian] =
                frozen_friction_relative_gradient_and_hessian(
                    contact, params.friction_coefficient, dt2);
            gradient += friction_gradient;
            hessian += friction_hessian;
        }
    }
}

void add_rigid_sdf_orientation_terms(
    const std::vector<Vec3>& ref_positions,
    const std::vector<int>& nodes,
    const std::vector<Vec3>& previous_positions,
    const Vec3& x_com_new, const Vec4& q_n, const Vec3& omega_new,
    const SimParams& params, double dt, Vec3& gradient, Mat33& hessian,
    const QuaternionOmegaKinematics* supplied_kinematics = nullptr) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    const QuaternionOmegaKinematics owned_kinematics = supplied_kinematics == nullptr ? quaternion_omega_kinematics(q_n, omega_new, dt) : QuaternionOmegaKinematics{};
    const QuaternionOmegaKinematics& kinematics = supplied_kinematics == nullptr ? owned_kinematics : *supplied_kinematics;
    for (std::size_t local = 0; local < ref_positions.size(); ++local) {
        const Vec3& X_centered = ref_positions[local];
        const Vec3 x = world_space_position(X_centered, x_com_new, kinematics.orientation);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        const Vec3 gx =
            sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
        const Mat33 Hx =
            sdf_penalty_hessian(
                sdf, params.k_sdf, params.eps_sdf, false);
        const Mat33 J_xomega = dx_domega(X_centered, kinematics);
        const RigidEnergyDerivatives derivatives =
            sdf_penalty_derivatives_rb(
                sdf, X_centered, kinematics,
                params.k_sdf, params.eps_sdf, false, false,
                &gx, &Hx, &J_xomega);
        gradient += dt2 * derivatives.orientation_gradient;
        hessian += dt2 * derivatives.orientation_orientation_hessian;
        if (params.friction_coefficient != 0.0) {
            const FrozenFrictionContact contact =
                make_sdf_frozen_friction_contact(
                    x,
                    previous_positions[static_cast<std::size_t>(
                        nodes[local])],
                    sdf, params.k_sdf, params.eps_sdf, dt,
                    params.friction_velocity_epsilon, 1.0e-12, &gx);
            const auto [friction_gradient, friction_hessian] =
                frozen_friction_relative_gradient_and_hessian(
                    contact, params.friction_coefficient, dt2);
            gradient += J_xomega.transpose() * friction_gradient;
            hessian += J_xomega.transpose()
                * friction_hessian * J_xomega;
        }
    }
}

void validate_rigid_solver_state(const RefMesh& ref_mesh, const DeformedState& state, const std::vector<Vec3>& x_com_new, const std::vector<Vec4>& q_new, const std::vector<Vec3>& omega_new) {
    const std::size_t num_rbs = ref_mesh.total_mass.size();
    const bool valid = ref_mesh.I_hat.size() == num_rbs
        && ref_mesh.rb_nodes.size() == num_rbs
        && ref_mesh.ref_positions.size() == num_rbs
        && ref_mesh.rb_update_modes.size() == num_rbs
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

double rigid_body_unnormalized_residual(const RefMesh& ref_mesh, const DeformedState& state, const BroadPhase::Cache& bp_cache, const std::vector<std::vector<int>>& body_nt_pair_indices, const std::vector<std::vector<int>>& body_ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const SimParams& params, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, double dt, std::vector<double>& body_residuals, const std::vector<Mat33>* rotation_predictors = nullptr, const FrozenResidualWorkspace* frozen_workspace = nullptr) {
    const int num_rbs = static_cast<int>(ref_mesh.total_mass.size());
    const double barrier_scale = dt * dt * params.k_barrier;
    body_residuals.resize(static_cast<std::size_t>(num_rbs));
    // Bodies only read the frozen residual configuration. Compute their
    // contributions independently, then retain the original body-index sum
    // order so parallel execution does not change the residual value.
    const auto evaluate_body = [&](int rb) {
        const RigidBodyUpdateMode update_mode =
            ref_mesh.rb_update_modes[rb];
        const bool update_translation =
            updates_rigid_translation(update_mode);
        const bool update_orientation =
            updates_rigid_orientation(update_mode);
        if (!update_translation && !update_orientation) {
            body_residuals[static_cast<std::size_t>(rb)] = 0.0;
            return;
        }
        const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt);
        const Mat33* rotation_predictor = rotation_predictors == nullptr ? nullptr : &(*rotation_predictors)[static_cast<std::size_t>(rb)];
        Vec3 com_gradient = Vec3::Zero();
        Vec3 orientation_gradient = Vec3::Zero();
        if (update_translation) {
            com_gradient = inertia_translation_gradient(x_com_new[rb], state.x_coms[rb], state.v_coms[rb], dt, ref_mesh.total_mass[rb]);
            com_gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);
        }
        if (update_orientation) {
            orientation_gradient = inertia_rotation_gradient(omega_new[rb], state.orientations[rb], state.omega[rb], dt, ref_mesh.I_hat[rb], &kinematics, rotation_predictor);
        }
        add_rigid_sdf_gradients(
            ref_mesh.ref_positions[rb], ref_mesh.rb_nodes[rb],
            state.deformed_positions, x_com_new[rb],
            state.orientations[rb], omega_new[rb], params, dt,
            com_gradient, orientation_gradient, &kinematics);
        RigidEnergyDerivatives friction;
        const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, bp_cache, body_nt_pair_indices[rb], body_ss_pair_indices[rb], node_to_rb_local, positions, omega_new, params, dt, RigidDerivativeMode::Gradient, &kinematics, frozen_workspace, params.friction_coefficient != 0.0 ? &friction : nullptr);
        if (update_translation)
            com_gradient += barrier_scale * barrier.translation_gradient;
        if (update_orientation)
            orientation_gradient += barrier_scale * barrier.orientation_gradient;
        if (params.friction_coefficient != 0.0) {
            if (update_translation)
                com_gradient += friction.translation_gradient;
            if (update_orientation)
                orientation_gradient += friction.orientation_gradient;
        }
        body_residuals[static_cast<std::size_t>(rb)] =
            (update_translation ? com_gradient.norm() : 0.0)
            + (update_orientation ? orientation_gradient.norm() : 0.0);
    };
    if (params.use_parallel && num_rbs > 1) {
        std::exception_ptr first_exception;
        int first_exception_body = num_rbs;
        #pragma omp parallel for schedule(static)
        for (int rb = 0; rb < num_rbs; ++rb) {
            try {
                evaluate_body(rb);
            } catch (...) {
                #pragma omp critical(rigid_residual_exception)
                {
                    if (rb < first_exception_body) {
                        first_exception_body = rb;
                        first_exception = std::current_exception();
                    }
                }
            }
        }
        if (first_exception != nullptr) std::rethrow_exception(first_exception);
    } else {
        for (int rb = 0; rb < num_rbs; ++rb) evaluate_body(rb);
    }
    double residual = 0.0;
    for (const double body_residual : body_residuals)
        residual += body_residual;
    return residual;
}

Vec3 compute_com_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, const SimParams& params, double dt, const QuaternionOmegaKinematics* kinematics = nullptr) {
    const Vec3& x_com_n = state.x_coms[rb];
    const Vec3& v_com_n = state.v_coms[rb];

    Vec3 gradient = inertia_translation_gradient(x_com_new[rb], x_com_n, v_com_n, dt, ref_mesh.total_mass[rb]);
    gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);

    Mat33 hessian = inertia_translation_hessian(ref_mesh.total_mass[rb]);
    add_rigid_sdf_translation_terms(
        ref_mesh.ref_positions[rb], ref_mesh.rb_nodes[rb],
        state.deformed_positions, x_com_new[rb], state.orientations[rb],
        omega_new[rb], params, dt, gradient, hessian, kinematics);
    RigidEnergyDerivatives friction;
    const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, bp_cache, nt_pair_indices, ss_pair_indices, node_to_rb_local, positions, omega_new, params, dt, RigidDerivativeMode::TranslationHessian, kinematics, nullptr, params.friction_coefficient != 0.0 ? &friction : nullptr);
    const double barrier_scale = dt * dt * params.k_barrier;
    gradient += barrier_scale * barrier.translation_gradient;
    hessian += barrier_scale * barrier.translation_translation_hessian;
    if (params.friction_coefficient != 0.0) {
        gradient += friction.translation_gradient;
        hessian += friction.translation_translation_hessian;
    }
    return hessian.ldlt().solve(gradient);
}

Vec3 compute_omega_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, const SimParams& params, double dt, const QuaternionOmegaKinematics* supplied_kinematics = nullptr, const Mat33* rotation_predictor = nullptr) {
    const Vec4& q_n = state.orientations[rb];
    const Vec3& omega_n = state.omega[rb];
    const Mat33& I_hat = ref_mesh.I_hat[rb];

    const QuaternionOmegaKinematics owned_kinematics = supplied_kinematics == nullptr ? quaternion_omega_kinematics(q_n, omega_new[rb], dt, true) : QuaternionOmegaKinematics{};
    const QuaternionOmegaKinematics& kinematics = supplied_kinematics == nullptr ? owned_kinematics : *supplied_kinematics;
    auto [gradient, hessian] = inertia_rotation_gradient_hessian(omega_new[rb], q_n, omega_n, dt, I_hat, &kinematics, rotation_predictor);
    add_rigid_sdf_orientation_terms(
        ref_mesh.ref_positions[rb], ref_mesh.rb_nodes[rb],
        state.deformed_positions, x_com_new[rb], q_n, omega_new[rb],
        params, dt, gradient, hessian, &kinematics);
    RigidEnergyDerivatives friction;
    const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, bp_cache, nt_pair_indices, ss_pair_indices, node_to_rb_local, positions, omega_new, params, dt, RigidDerivativeMode::OrientationHessian, &kinematics, nullptr, params.friction_coefficient != 0.0 ? &friction : nullptr);
    const double barrier_scale = dt * dt * params.k_barrier;
    gradient += barrier_scale * barrier.orientation_gradient;
    hessian += barrier_scale * barrier.orientation_orientation_hessian;
    if (params.friction_coefficient != 0.0) {
        gradient += friction.orientation_gradient;
        hessian += friction.orientation_orientation_hessian;
    }
    return hessian.ldlt().solve(gradient);
}

} // namespace rb_solver

// -----------------------------------------------------------------------------
// Rigid-body solver workspace
// -----------------------------------------------------------------------------

namespace {

struct RigidSolverWorkspace {
    BroadPhase broad_phase;
    FrozenResidualWorkspace frozen_residual;
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
    GreedyColoringWorkspace coloring_workspace;
    std::vector<double> body_residuals;
    std::vector<Mat33> rotation_predictors;
    bool contact_cache_initialized = false;
    double contact_cache_d_hat = 0.0;

    bool matches(const RefMesh& ref_mesh, int nv) const {
        return mesh == &ref_mesh && tris_data == ref_mesh.tris.data() && rb_nodes_data == ref_mesh.rb_nodes.data() && ref_positions_data == ref_mesh.ref_positions.data() && tris_size == ref_mesh.tris.size() && num_rbs == ref_mesh.rb_nodes.size() && num_vertices == nv;
    }

    void prepare(const RefMesh& ref_mesh, int nv, double initial_com_disp, double initial_theta_disp) {
        if (!matches(ref_mesh, nv)) {
            broad_phase = BroadPhase{};
            contact_cache_initialized = false;
            prev_com_disp.assign(ref_mesh.rb_nodes.size(), initial_com_disp);
            prev_theta_disp.assign(ref_mesh.rb_nodes.size(), initial_theta_disp);
            body_nt_pair_indices.clear();
            body_ss_pair_indices.clear();
            contact_adjacency.clear();
            color_groups.clear();
            coloring_workspace = GreedyColoringWorkspace{};
            body_residuals.clear();
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
        body_residuals.resize(ref_mesh.rb_nodes.size());
        rotation_predictors.resize(ref_mesh.rb_nodes.size());
    }
};

} // namespace

// -----------------------------------------------------------------------------
// Rigid-body solver entry point
// -----------------------------------------------------------------------------

SolverResult global_gauss_seidel_solver_basic_rb(const RefMesh& ref_mesh, const DeformedState& state, const SimParams& params, std::vector<Vec3>& x_com_new, std::vector<Vec4>& q_new, std::vector<Vec3>& omega_new) {
    validate_solver_friction_parameters(
        params, "global_gauss_seidel_solver_basic_rb");
    if (params.friction_coefficient != 0.0
        && state.deformed_positions.size() != ref_mesh.node_to_rb.size()) {
        throw std::invalid_argument(
            "global_gauss_seidel_solver_basic_rb: previous rigid positions must match mesh node count");
    }
    rb_solver::validate_rigid_solver_state(ref_mesh, state, x_com_new, q_new, omega_new);

    SolverResult result;
    const int num_rbs = static_cast<int>(ref_mesh.total_mass.size());
    const double dt = params.dt();
    for (int rb = 0; rb < num_rbs; ++rb) {
        const RigidBodyUpdateMode update_mode =
            ref_mesh.rb_update_modes[rb];
        if (!updates_rigid_translation(update_mode))
            x_com_new[rb] = state.x_coms[rb];
        if (!updates_rigid_orientation(update_mode)) {
            omega_new[rb] = Vec3::Zero();
            q_new[rb] = state.orientations[rb];
        }
    }
    static RigidSolverWorkspace workspace;
    workspace.prepare(ref_mesh, static_cast<int>(state.deformed_positions.size()), params.node_box_max, params.theta_box_max);
    for (int rb = 0; rb < num_rbs; ++rb)
        workspace.rotation_predictors[static_cast<std::size_t>(rb)] = rigid_rotation_predictor(state.orientations[rb], state.omega[rb], dt);

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

    const auto rebuild_contact_cache = [&](int iteration) {
        constexpr double box_padding = 1.2;
        for (int rb = 0; rb < num_rbs; ++rb) {
            workspace.com_box_anchors[rb] = x_com_new[rb];
            workspace.orientation_box_anchors[rb] = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_new[rb], dt));
            workspace.com_box_radii[rb] = std::clamp(box_padding * std::max(workspace.prev_com_disp[rb], dt * state.v_coms[rb].norm()), params.node_box_min, params.node_box_max);
            workspace.theta_box_radii[rb] = std::clamp(box_padding * std::max(workspace.prev_theta_disp[rb], dt * state.omega[rb].norm()), params.theta_box_min, params.theta_box_max);
        }
        build_blue_boxes_rb(workspace.com_box_anchors, workspace.orientation_box_anchors, workspace.theta_box_radii, workspace.com_box_radii, ref_mesh, workspace.blue_boxes);
        const std::vector<AABB>& cached_boxes = workspace.broad_phase.cache().node_boxes;
        bool boxes_unchanged = workspace.contact_cache_initialized && cached_boxes.size() == workspace.blue_boxes.size() && std::memcmp(&workspace.contact_cache_d_hat, &params.d_hat, sizeof(double)) == 0;
        for (std::size_t box = 0; boxes_unchanged && box < cached_boxes.size(); ++box) boxes_unchanged = std::memcmp(cached_boxes[box].min.data(), workspace.blue_boxes[box].min.data(), 3 * sizeof(double)) == 0 && std::memcmp(cached_boxes[box].max.data(), workspace.blue_boxes[box].max.data(), 3 * sizeof(double)) == 0;
        if (boxes_unchanged) {
            if (params.verbose)
                std::fprintf(stderr, "  [RB GS] iter %d  reusing rigid contact cache\n", iteration);
            return;
        }
        workspace.broad_phase.initialize(workspace.blue_boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::RigidSolver);
        build_rb_contact_adj(workspace.broad_phase.cache(), ref_mesh.node_to_rb, num_rbs, workspace.body_nt_pair_indices, workspace.body_ss_pair_indices, workspace.contact_adjacency);
        greedy_color_conflict_graph(workspace.contact_adjacency, workspace.color_groups, &workspace.coloring_workspace);
        workspace.contact_cache_initialized = true;
        workspace.contact_cache_d_hat = params.d_hat;
        if (params.verbose)
            std::fprintf(stderr, "  [RB GS] iter %d  rebuilding rigid blue boxes and %zu block colors\n", iteration, workspace.color_groups.size());
    };

    const auto evaluate_residual = [&]() {
        const BroadPhase::Cache& broad_phase_cache =
            workspace.broad_phase.cache();
        return rb_solver::rigid_body_unnormalized_residual(
            ref_mesh, state, broad_phase_cache,
            workspace.body_nt_pair_indices,
            workspace.body_ss_pair_indices,
            workspace.node_to_rb_local, workspace.positions, params,
            x_com_new, omega_new, dt, workspace.body_residuals,
            &workspace.rotation_predictors);
    };

    rebuild_contact_cache(1);

    if (!params.fixed_iters) {
        initial_residual = evaluate_residual();
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
            const RigidBodyUpdateMode update_mode =
                ref_mesh.rb_update_modes[rb];
            std::vector<Vec3>& node_positions = workspace.positions;
            const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt, true);
            if (updates_rigid_translation(update_mode)) {
                const Vec3 delta_com = params.damping * rb_solver::compute_com_update(rb, state, ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], workspace.node_to_rb_local, node_positions, x_com_new, omega_new, params, dt, &kinematics);
                const Vec3 com_radius = Vec3::Constant(workspace.com_box_radii[rb]);
                const Vec3 com_target = (x_com_new[rb] - delta_com).cwiseMax(workspace.com_box_anchors[rb] - com_radius).cwiseMin(workspace.com_box_anchors[rb] + com_radius);
                const Vec3 proposed_com_displacement = com_target - x_com_new[rb];
                const double com_safe_step = per_rigid_body_translation_safe_step(ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], node_positions, rb, proposed_com_displacement);
                const Vec3 com_displacement = com_safe_step * proposed_com_displacement;
                x_com_new[rb] += com_displacement;
                for (const int node : ref_mesh.rb_nodes[rb])
                    node_positions[node] += com_displacement;
            }

            Vec4 q_accepted = q_new[rb];
            if (updates_rigid_orientation(update_mode)) {
                const Vec3 delta_omega = rb_solver::compute_omega_update(rb, state, ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], workspace.node_to_rb_local, node_positions, x_com_new, omega_new, params, dt, &kinematics, &workspace.rotation_predictors[static_cast<std::size_t>(rb)]);
                const Vec4 q_current = quaternion_normalize(kinematics.orientation);
                const Vec3 omega_trial = omega_new[rb] - params.damping * delta_omega;
                const Vec4 q_target = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_trial, dt));
                const Vec4 q_bounded = bound_quaternion(workspace.orientation_box_anchors[rb], q_current, q_target, workspace.theta_box_radii[rb]);
                const double rotation_safe_step = per_rigid_body_rotation_safe_step(ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], node_positions, rb, x_com_new[rb], q_current, q_bounded);
                q_accepted = interpolate_orientation_full_arc(q_current, q_bounded, rotation_safe_step);
                q_new[rb] = q_accepted;
                omega_new[rb] = angular_velocity_from_orientation_full_arc(q_accepted, state.orientations[rb], dt);
            }

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
            const double residual = evaluate_residual();
            result.final_residual = residual;
            if (params.verbose)
                std::fprintf(stderr, "  [RB GS] iter %d  residual = %.6e\n", iter, residual);
            if (residual_converged(residual)) {
                result.converged = true;
                break;
            }
        }
    }

    for (int rb = 0; rb < num_rbs; ++rb) {
        workspace.prev_com_disp[rb] = updates_rigid_translation(
            ref_mesh.rb_update_modes[rb])
            ? (x_com_new[rb] - workspace.substep_start_coms[rb]).norm()
            : 0.0;
        workspace.prev_theta_disp[rb] = updates_rigid_orientation(
            ref_mesh.rb_update_modes[rb])
            ? dt * omega_new[rb].norm() : 0.0;
    }

    if (params.fixed_iters)
        result.converged = true;
    return result;
}

// -----------------------------------------------------------------------------
// General deformable + rigid-body solver entry point
// -----------------------------------------------------------------------------
SolverResult global_gauss_seidel_solver_basic_general(
    const RefMesh& ref_mesh, const DeformedState& state,
    const VertexTriangleMap& adj, const std::vector<Pin>& pins,
    const SimParams& params, std::vector<Vec3>& xnew,
    const std::vector<Vec3>& xhat,
    std::vector<Vec3>& x_com_new, std::vector<Vec4>& q_new,
    std::vector<Vec3>& omega_new, BroadPhase& broad_phase,
    const std::string& outdir) {

    validate_solver_friction_parameters(
        params, "global_gauss_seidel_solver_basic_general");
    const int nv = static_cast<int>(xnew.size());
    if (params.friction_coefficient != 0.0
        && state.deformed_positions.size() != xnew.size()) {
        throw std::invalid_argument(
            "global_gauss_seidel_solver_basic_general: previous positions must match xnew.size()");
    }
    const std::vector<Vec3>* previous_positions =
        params.friction_coefficient == 0.0
        ? nullptr : &state.deformed_positions;
    const int num_rbs = static_cast<int>(ref_mesh.rb_nodes.size());
    const std::vector<int>& deformable_nodes = ref_mesh.deformable_nodes;
    SolverResult result;

    // A mesh without rigid bodies may predate node_to_rb. Preserve the exact
    // cloth-only path in that case.
    if (num_rbs == 0 && ref_mesh.tet_nodes.empty()) {
        return global_gauss_seidel_solver_basic(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, &state.deformed_positions);
    }

    rb_solver::validate_rigid_solver_state(ref_mesh, state, x_com_new, q_new, omega_new);

    // Preserve the exact rigid-only implementation and synchronize its proxy
    // positions before returning through the general API.
    if (deformable_nodes.empty() && ref_mesh.tet_nodes.empty()) {
        SolverResult result = global_gauss_seidel_solver_basic_rb(ref_mesh, state, params, x_com_new, q_new, omega_new);
        for (int rb = 0; rb < num_rbs; ++rb) {
            for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local) {
                xnew[ref_mesh.rb_nodes[rb][local]] = world_space_position(ref_mesh.ref_positions[rb][local], x_com_new[rb], q_new[rb]);
            }
        }
        return result;
    }

    for (int rb = 0; rb < num_rbs; ++rb) {
        const RigidBodyUpdateMode update_mode =
            ref_mesh.rb_update_modes[rb];
        if (!updates_rigid_translation(update_mode))
            x_com_new[rb] = state.x_coms[rb];
        if (!updates_rigid_orientation(update_mode)) {
            omega_new[rb] = Vec3::Zero();
            q_new[rb] = state.orientations[rb];
        }
    }

    static BasicSolverWorkspace deformable_workspace;
    static RigidSolverWorkspace rigid_workspace;
    static MixedAdjacencyWorkspace mixed_adjacency_workspace;
    deformable_workspace.prepare(ref_mesh, adj, nv, params.node_box_max);
    rigid_workspace.prepare(ref_mesh, nv, params.node_box_max, params.theta_box_max);
    const std::vector<std::vector<int>>& nodal_elastic_adj = deformable_workspace.elastic_adjacency.get(ref_mesh, adj, nv);
    mixed_adjacency_workspace.prepare(ref_mesh, deformable_nodes, num_rbs, nv);
    const std::vector<int>& cloth_nodes = mixed_adjacency_workspace.cloth_nodes;
    const std::vector<int>& solid_nodes = ref_mesh.tet_nodes;
    const int num_cloth = static_cast<int>(cloth_nodes.size());
    const int num_solid = static_cast<int>(solid_nodes.size());
    const int solid_begin = num_cloth;
    const int rigid_begin = solid_begin + num_solid;
    PinMap& pin_map = deformable_workspace.pin_map;
    deformable_workspace.pinned_vertices.reserve(pins.size());
    for (int pin = 0; pin < static_cast<int>(pins.size()); ++pin) {
        pin_map[pins[pin].vertex_index] = pin;
        deformable_workspace.pinned_vertices.push_back(pins[pin].vertex_index);
    }

    // xnew is the single live collision configuration. Its deformable entries
    // come from the caller; overwrite only rigid proxies from generalized
    // coordinates.
    for (int rb = 0; rb < num_rbs; ++rb) {
        const Vec4 orientation = updates_rigid_orientation(
                                     ref_mesh.rb_update_modes[rb])
            ? quaternion_normalize(quaternion_from_angular_velocity(
                  state.orientations[rb], omega_new[rb], params.dt()))
            : state.orientations[rb];
        q_new[rb] = orientation;
        for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local) {
            xnew[ref_mesh.rb_nodes[rb][local]] = world_space_position(ref_mesh.ref_positions[rb][local], x_com_new[rb], orientation);
        }
    }

    deformable_workspace.xnew_substep_start = xnew;
    rigid_workspace.substep_start_coms = x_com_new;
    std::vector<AABB>& blue_boxes = rigid_workspace.blue_boxes;
    const double dt = params.dt();
    for (int rb = 0; rb < num_rbs; ++rb)
        rigid_workspace.rotation_predictors[static_cast<std::size_t>(rb)] = rigid_rotation_predictor(state.orientations[rb], state.omega[rb], dt);
    for (const int node : cloth_nodes)
        deformable_workspace.inertial_disp[node] = dt * state.velocities[node].norm();
    for (const int node : solid_nodes)
        deformable_workspace.inertial_disp[node] = dt * state.velocities[node].norm();
    // SimParams caches these values lazily. Populate both caches before any
    // heterogeneous color is evaluated by multiple OpenMP workers.
    (void)params.dt2();
    constexpr double box_padding = 1.2;

    const auto rebuild_contact_cache = [&](int iteration) {
        // First fill deformable boxes. build_blue_boxes_rb then overwrites all
        // rigid proxy entries with spherical-cap plus COM bounds.
        for (const int node : cloth_nodes) {
            const double radius = std::clamp(box_padding * std::max(deformable_workspace.prev_disp[node], deformable_workspace.inertial_disp[node]),params.node_box_min, params.node_box_max);
            const Vec3 half_extent = Vec3::Constant(radius);
            blue_boxes[node] = AABB(xnew[node] - half_extent, xnew[node] + half_extent);
        }
        for (const int node : solid_nodes) {
            const double radius = std::clamp(box_padding * std::max(deformable_workspace.prev_disp[node], deformable_workspace.inertial_disp[node]),params.node_box_min, params.node_box_max);
            const Vec3 half_extent = Vec3::Constant(radius);
            blue_boxes[node] = AABB(xnew[node] - half_extent, xnew[node] + half_extent);
        }

        for (int rb = 0; rb < num_rbs; ++rb) {
            rigid_workspace.com_box_anchors[rb] = x_com_new[rb];
            rigid_workspace.orientation_box_anchors[rb] = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_new[rb], dt));
            rigid_workspace.com_box_radii[rb] = std::clamp(box_padding * std::max(rigid_workspace.prev_com_disp[rb], dt * state.v_coms[rb].norm()), params.node_box_min, params.node_box_max);
            rigid_workspace.theta_box_radii[rb] = std::clamp(box_padding * std::max(rigid_workspace.prev_theta_disp[rb], dt * state.omega[rb].norm()), params.theta_box_min, params.theta_box_max);
        }
        build_blue_boxes_rb(rigid_workspace.com_box_anchors, rigid_workspace.orientation_box_anchors, rigid_workspace.theta_box_radii, rigid_workspace.com_box_radii, ref_mesh, blue_boxes);
        if (solid_nodes.empty()) {
            broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::GeneralSolver);
        } else {
            broad_phase.initialize_surface_nodes(blue_boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::GeneralSolver);
        }
        build_rb_contact_adj(broad_phase.cache(), ref_mesh.node_to_rb, num_rbs, rigid_workspace.body_nt_pair_indices, rigid_workspace.body_ss_pair_indices, rigid_workspace.contact_adjacency);
        build_all_block_adjacency_and_contact(ref_mesh, cloth_nodes, nodal_elastic_adj, broad_phase.cache(), mixed_adjacency_workspace.conflict_adjacency, &mixed_adjacency_workspace.node_to_block, &mixed_adjacency_workspace.solid_node_mask, &mixed_adjacency_workspace.surface_node_mask, &mixed_adjacency_workspace.elastic_row_sizes, &rigid_workspace.body_nt_pair_indices, &rigid_workspace.body_ss_pair_indices);
        greedy_color_conflict_graph(mixed_adjacency_workspace.conflict_adjacency, mixed_adjacency_workspace.color_groups, &mixed_adjacency_workspace.coloring_workspace);
        if (params.verbose)
            std::fprintf(stderr, "  [General GS] iter %d  rebuilding mixed blue boxes and %zu block colors\n", iteration, mixed_adjacency_workspace.color_groups.size());
    };


    const auto update_final_residual = [&]() {
        build_frozen_residual_workspace(
            ref_mesh, params, xnew, broad_phase,
            rigid_workspace.frozen_residual,
            &deformable_workspace.rest_shape_grads);
        result.final_cloth_residual = compute_global_deformable_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, cloth_nodes, &pin_map, &deformable_workspace.incident_triangles, &deformable_workspace.rest_shape_grads, &rigid_workspace.frozen_residual, previous_positions);
        result.final_solid_residual = solid_nodes.empty() ? 0.0 : compute_global_solid_residual(ref_mesh, pins, params, xnew, xhat, broad_phase, &pin_map, &mixed_adjacency_workspace.solid_node_mask, &mixed_adjacency_workspace.surface_node_mask, &rigid_workspace.frozen_residual, previous_positions);
        result.final_rigid_residual = rb_solver::rigid_body_unnormalized_residual(ref_mesh, state, broad_phase.cache(), rigid_workspace.body_nt_pair_indices, rigid_workspace.body_ss_pair_indices, rigid_workspace.node_to_rb_local, xnew, params, x_com_new, omega_new, dt, rigid_workspace.body_residuals, &rigid_workspace.rotation_predictors, &rigid_workspace.frozen_residual);
        result.final_residual = result.final_cloth_residual + result.final_solid_residual + result.final_rigid_residual;
    };

    const auto block_residual_converged = [&](double residual, double initial) {
        double tolerance = 0.0;
        if (params.tol_abs > 0.0)
            tolerance = std::max(tolerance, params.tol_abs);
        if (params.tol_rel > 0.0 && std::isfinite(initial))
            tolerance = std::max(tolerance, params.tol_rel * initial);
        return residual <= tolerance;
    };
    const auto residual_converged = [&]() {
        return block_residual_converged(result.final_cloth_residual, result.initial_cloth_residual) && block_residual_converged(result.final_solid_residual, result.initial_solid_residual) && block_residual_converged(result.final_rigid_residual, result.initial_rigid_residual);
    };

    rebuild_contact_cache(1);
    if (!params.fixed_iters) {
        result.has_residual = true;
        result.has_residual_components = true;
        update_final_residual();
        result.initial_cloth_residual = result.final_cloth_residual;
        result.initial_solid_residual = result.final_solid_residual;
        result.initial_rigid_residual = result.final_rigid_residual;
        result.initial_residual = result.final_residual;
        if (residual_converged()) {
            result.converged = true;
            return result;
        }
    }

    for (int iteration = 1; iteration <= params.max_global_iters; ++iteration) {
        if (iteration > 1 && (iteration - 1) % params.node_box_update_count == 0) {
            rebuild_contact_cache(iteration);
        }

        const auto process_cloth_node = [&](const int cloth) {
            const int node = cloth_nodes[static_cast<std::size_t>(cloth)];
            const Vec3 proposed_position = xnew[node] - params.damping * gs_vertex_delta_live_barrier(node, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pin_map, &deformable_workspace.incident_triangles[node], &deformable_workspace.rest_shape_grads, previous_positions);
            per_vertex_safe_step(broad_phase, xnew, node, proposed_position, 0.9, params.use_ccd, params.use_ticcd, false);
        };

        const auto process_solid_node = [&](const int solid) {
            const int node = solid_nodes[static_cast<std::size_t>(solid)];
            const Vec3 proposed_position = xnew[node] - params.damping * gs_solid_vertex_delta_live_barrier(node, ref_mesh, pins, params, xhat, xnew, broad_phase, mixed_adjacency_workspace.solid_node_mask, mixed_adjacency_workspace.surface_node_mask, pin_map, previous_positions);
            per_vertex_safe_step(broad_phase, xnew, node, proposed_position, 0.9, params.use_ccd, params.use_ticcd, false);
        };

        // COM and orientation remain one indivisible update block: all proxy
        // positions are committed before another color begins.
        const auto process_body = [&](int rb) {
            const RigidBodyUpdateMode update_mode =
                ref_mesh.rb_update_modes[rb];
            const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt, true);
            if (updates_rigid_translation(update_mode)) {
                const Vec3 delta_com = params.damping * rb_solver::compute_com_update(rb, state, ref_mesh, broad_phase.cache(), rigid_workspace.body_nt_pair_indices[rb], rigid_workspace.body_ss_pair_indices[rb], rigid_workspace.node_to_rb_local, xnew, x_com_new, omega_new, params, dt, &kinematics);
                const Vec3 com_radius = Vec3::Constant(rigid_workspace.com_box_radii[rb]);
                const Vec3 com_target =(x_com_new[rb] - delta_com).cwiseMax(rigid_workspace.com_box_anchors[rb] - com_radius).cwiseMin(rigid_workspace.com_box_anchors[rb] + com_radius);
                const Vec3 proposed_com_displacement = com_target - x_com_new[rb];
                const double com_safe_step = per_rigid_body_translation_safe_step(ref_mesh, broad_phase.cache(), rigid_workspace.body_nt_pair_indices[rb], rigid_workspace.body_ss_pair_indices[rb], xnew, rb, proposed_com_displacement);
                const Vec3 com_displacement = com_safe_step * proposed_com_displacement;
                x_com_new[rb] += com_displacement;
                for (const int node : ref_mesh.rb_nodes[rb])
                    xnew[node] += com_displacement;
            }

            Vec4 q_accepted = q_new[rb];
            if (updates_rigid_orientation(update_mode)) {
                const Vec3 delta_omega = rb_solver::compute_omega_update(rb, state, ref_mesh, broad_phase.cache(), rigid_workspace.body_nt_pair_indices[rb], rigid_workspace.body_ss_pair_indices[rb], rigid_workspace.node_to_rb_local, xnew, x_com_new, omega_new, params, dt, &kinematics, &rigid_workspace.rotation_predictors[static_cast<std::size_t>(rb)]);
                const Vec4 q_current = quaternion_normalize(kinematics.orientation);
                const Vec3 omega_trial = omega_new[rb] - params.damping * delta_omega;
                const Vec4 q_target = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_trial, dt));
                const Vec4 q_bounded = bound_quaternion(rigid_workspace.orientation_box_anchors[rb], q_current,q_target, rigid_workspace.theta_box_radii[rb]);
                const double rotation_safe_step = per_rigid_body_rotation_safe_step(ref_mesh, broad_phase.cache(), rigid_workspace.body_nt_pair_indices[rb], rigid_workspace.body_ss_pair_indices[rb], xnew, rb, x_com_new[rb], q_current, q_bounded);
                q_accepted = interpolate_orientation_full_arc(q_current, q_bounded, rotation_safe_step);
                q_new[rb] = q_accepted;
                omega_new[rb] = angular_velocity_from_orientation_full_arc(q_accepted, state.orientations[rb], dt);
            }

            for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size());++local) {
                xnew[ref_mesh.rb_nodes[rb][local]] = world_space_position(ref_mesh.ref_positions[rb][local],x_com_new[rb], q_accepted);
            }
        };

        if (params.use_parallel) {
            // Block ids are [cloth nodes][solid nodes][rigid bodies]. Each
            // color is independent under cloth/tet elasticity and NT/SS
            // reads, and the omp-for barrier separates successive GS colors.
            #pragma omp parallel
            {
                for (std::size_t color_index = 0; color_index < mixed_adjacency_workspace.color_groups.size(); ++color_index) {
                    const std::vector<int>& color = mixed_adjacency_workspace.color_groups[color_index];
                    // The [cloth][solid][rigid] block ordering gives highly
                    // unequal work per block. Small dynamic chunks balance
                    // that work while each color remains conflict-free and
                    // the implicit barrier preserves the GS color order.
                    #pragma omp for schedule(dynamic, 1)
                    for (int index = 0; index < static_cast<int>(color.size()); ++index) {
                        const int block = color[index];
                        if (block < solid_begin)
                            process_cloth_node(block);
                        else if (block < rigid_begin)
                            process_solid_node(block - solid_begin);
                        else
                            process_body(block - rigid_begin);
                    }
                }
            }
        } else {
            for (int cloth = 0; cloth < num_cloth; ++cloth)
                process_cloth_node(cloth);
            for (int solid = 0; solid < num_solid; ++solid)
                process_solid_node(solid);
            for (int rb = 0; rb < num_rbs; ++rb)
                process_body(rb);
        }
        result.iterations = iteration;
        if (!params.fixed_iters) {
            update_final_residual();
            if (params.verbose) {
                std::fprintf(
                    stderr,
                    "  [General GS] iter %d  cloth residual = %.6e  solid residual = %.6e  rigid-body residual = %.6e  total residual = %.6e\n",
                    iteration, result.final_cloth_residual,
                    result.final_solid_residual,
                    result.final_rigid_residual, result.final_residual);
            }
            if (residual_converged()) {
                result.converged = true;
                break;
            }
        }
    }

    for (const int node : cloth_nodes) {
        deformable_workspace.prev_disp[node] = (xnew[node] - deformable_workspace.xnew_substep_start[node]).norm();
    }
    for (const int node : solid_nodes) {
        deformable_workspace.prev_disp[node] = (xnew[node] - deformable_workspace.xnew_substep_start[node]).norm();
    }
    for (int rb = 0; rb < num_rbs; ++rb) {
        rigid_workspace.prev_com_disp[rb] = updates_rigid_translation(
            ref_mesh.rb_update_modes[rb])
            ? (x_com_new[rb] - rigid_workspace.substep_start_coms[rb]).norm()
            : 0.0;
        rigid_workspace.prev_theta_disp[rb] = updates_rigid_orientation(
            ref_mesh.rb_update_modes[rb])
            ? dt * omega_new[rb].norm() : 0.0;
    }

    if (params.fixed_iters)
        result.converged = true;
    if (params.write_substeps)
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, nullptr);
    return result;
}
