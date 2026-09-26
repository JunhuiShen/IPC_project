#include <optional>
#include "solver.h"
#include "SIMD.h"
#include "contact_scheduling.h"
#include "grid_contact_scheduling.h"
#include "grid_coloring.h"
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
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Deformable solver workspaces
// -----------------------------------------------------------------------------

namespace {

// For setup kernels that validate inputs, propagate exceptions on the calling
// thread instead of letting them escape an OpenMP worker and terminate.
template <typename Function>
void parallel_body_setup(int count, bool parallel, const Function& function) {
    if (!parallel) {
        for (int i = 0; i < count; ++i) function(i);
        return;
    }
    std::exception_ptr failure;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        try { function(i); }
        catch (...) {
            #pragma omp critical(rigid_setup_exception)
            { if (!failure) failure = std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
}

// Task groups finish all proxy writes before the next dependent body phase.
void translate_rigid_nodes(const std::vector<int>& nodes, const Vec3 displacement,
    std::vector<Vec3>& positions, bool parallel) {
    if (parallel && nodes.size() >= 128) {
        #pragma omp taskloop grainsize(64) shared(nodes, positions) firstprivate(displacement)
        for (int local = 0; local < static_cast<int>(nodes.size()); ++local)
            positions[nodes[local]] += displacement;
    } else {
        for (const int node : nodes) positions[node] += displacement;
    }
}

void place_rigid_nodes(const std::vector<int>& nodes, const std::vector<Vec3>& material,
    const Vec3 center, const Vec4 orientation, std::vector<Vec3>& positions, bool parallel) {
    if (parallel && nodes.size() >= 128) {
        #pragma omp taskloop grainsize(64) shared(nodes, material, positions) firstprivate(center, orientation)
        for (int local = 0; local < static_cast<int>(nodes.size()); ++local)
            positions[nodes[local]] = world_space_position(material[local], center, orientation);
    } else {
        for (int local = 0; local < static_cast<int>(nodes.size()); ++local)
            positions[nodes[local]] = world_space_position(material[local], center, orientation);
    }
}

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

const std::vector<Vec3>* resolve_experimental_friction_previous_positions(
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
    #pragma omp parallel for schedule(static) if(params.use_parallel && x.size() >= 128)
    for (std::size_t node = 0; node < x.size(); ++node)
        reconstructed_previous_positions[node] = xhat[node] - dt * velocities[node];
    return &reconstructed_previous_positions;
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

// Workspace for the basic loop from 1da4c74.
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

struct ExperimentalSolverWorkspace {
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
    std::vector<std::vector<int>> elastic_color_groups;
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
            #pragma omp parallel for schedule(static) if(nv >= 128)
            for (int vi = 0; vi < nv; ++vi) {
                const auto found = adj.find(vi);
                if (found != adj.end()) incident_triangles[vi] = found->second;
            }

            rest_shape_grads.resize(ref_mesh.Dm_inverse.size());
            #pragma omp parallel for schedule(static) if(ref_mesh.Dm_inverse.size() >= 128)
            for (int ti = 0; ti < static_cast<int>(ref_mesh.Dm_inverse.size()); ++ti)
                rest_shape_grads[ti] = shape_function_gradients(ref_mesh.Dm_inverse[ti]);

            prev_disp.assign(nv, initial_prev_disp);
            pin_map.assign(nv, -1);
            pinned_vertices.clear();
            contact_adjacency.clear();
            combined_adjacency.clear();
            color_groups.clear();
            elastic_color_groups.clear();
            deformable_nodes.resize(static_cast<std::size_t>(nv));
            #pragma omp parallel for schedule(static) if(nv >= 128)
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
        #pragma omp parallel for schedule(static) if(ref_mesh.tet_nodes.size() >= 128)
        for (int solid = 0; solid < static_cast<int>(ref_mesh.tet_nodes.size()); ++solid)
            solid_node_mask[static_cast<std::size_t>(ref_mesh.tet_nodes[solid])] = 1;
        surface_node_mask.assign(static_cast<std::size_t>(nv), 0);
        #pragma omp parallel for schedule(static) if(ref_mesh.surface_nodes.size() >= 128)
        for (int surface = 0; surface < static_cast<int>(ref_mesh.surface_nodes.size()); ++surface)
            surface_node_mask[static_cast<std::size_t>(ref_mesh.surface_nodes[surface])] = 1;
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
        #pragma omp parallel for schedule(static) if(cloth_nodes.size() >= 128)
        for (int cloth = 0; cloth < static_cast<int>(cloth_nodes.size()); ++cloth)
            node_to_block[static_cast<std::size_t>(cloth_nodes[cloth])] = cloth;
        #pragma omp parallel for schedule(static) if(ref_mesh.tet_nodes.size() >= 128)
        for (int solid = 0; solid < static_cast<int>(ref_mesh.tet_nodes.size()); ++solid)
            node_to_block[static_cast<std::size_t>(ref_mesh.tet_nodes[solid])] = solid_begin + solid;
        #pragma omp parallel for schedule(static) if(num_rbs >= 8)
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
    std::vector<NodeTriangleDistanceResult> nt_distances;
    std::vector<SegmentSegmentDistanceResult> ss_distances;
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

        // Contact terms read a frozen snapshot, so only the unchanged elastic
        // topology constrains colors. Rebuild them with the topology cache.
        greedy_color_conflict_graph(elastic_adjacency.get(ref_mesh, adj, nv), color_groups);

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
// Sequential contact assembly from 1da4c74.
static Vec3 gs_vertex_delta_live_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
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
bool solver_detail::contact_boxes_separated(const std::array<Vec3, 4>& positions,
    const std::array<AABB, 4>& boxes, bool segment_segment, double d_hat) {
    if (!(d_hat > 0.0) || !std::isfinite(d_hat)) return false;
    for (int node = 0; node < 4; ++node) {
        const auto& box = boxes[node];
        if (!positions[node].allFinite() || !box.min.allFinite() || !box.max.allFinite()) return false;
        for (int axis = 0; axis < 3; ++axis) {
            if (!(box.max[axis] - box.min[axis] >= 2e-10)
                || !(box.min[axis] + 1e-10 <= box.max[axis] - 1e-10)
                || positions[node][axis] < box.min[axis] || positions[node][axis] > box.max[axis])
                return false;
        }
    }
    Vec3 direction;
    if (segment_segment) {
        const auto distance = segment_segment_distance(positions[0], positions[1], positions[2], positions[3]);
        direction = distance.closest_point_1 - distance.closest_point_2;
    } else {
        const auto distance = node_triangle_distance(positions[0], positions[1], positions[2], positions[3]);
        direction = positions[0] - distance.closest_point;
    }
    if (!direction.allFinite()) return false;
    const double scale = direction.cwiseAbs().maxCoeff();
    if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    direction /= scale;
    if (!direction.allFinite()) return false;
    double lower[4], upper[4], magnitude = 0.0;
    for (int node = 0; node < 4; ++node) {
        lower[node] = upper[node] = 0.0;
        double absolute_sum = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const auto& box = boxes[node];
            const double lo = direction[axis] >= 0.0 ? box.min[axis] : box.max[axis];
            const double hi = direction[axis] >= 0.0 ? box.max[axis] : box.min[axis];
            lower[node] += direction[axis] * lo;
            upper[node] += direction[axis] * hi;
            absolute_sum += std::abs(direction[axis]) * std::max(std::abs(lo), std::abs(hi));
        }
        if (!std::isfinite(lower[node]) || !std::isfinite(upper[node])
            || !std::isfinite(absolute_sum)) return false;
        magnitude = std::max(magnitude, absolute_sum);
    }
    const int split = segment_segment ? 2 : 1;
    double alo = lower[0], ahi = upper[0], blo = lower[split], bhi = upper[split];
    for (int node = 1; node < split; ++node) {
        alo = std::min(alo, lower[node]); ahi = std::max(ahi, upper[node]);
    }
    for (int node = split + 1; node < 4; ++node) {
        blo = std::min(blo, lower[node]); bhi = std::max(bhi, upper[node]);
    }
    const double threshold = d_hat * direction.norm();
    const double gap = std::max(alo - bhi, blo - ahi);
    // Cover projection, norm, and comparison roundoff, including subnormals.
    const double padding = 512.0 * std::numeric_limits<double>::epsilon()
        * (magnitude + threshold) + 512.0 * std::numeric_limits<double>::denorm_min();
    return std::isfinite(threshold) && std::isfinite(gap) && std::isfinite(padding)
        && gap > threshold + padding;
}

namespace {

struct SimdBoxContactCertificates {
    std::vector<unsigned char> node_triangle, segment_segment;
    std::vector<std::vector<unsigned>> vertex_clear_words;
    bool valid = false;
};

static void rebuild_simd_box_certificates(const BroadPhase::Cache& cache,
    const std::vector<Vec3>& positions, double d_hat, bool parallel,
    bool prepare_contact_masks, SimdBoxContactCertificates& certificates) {
    certificates.valid = false;
    certificates.node_triangle.resize(cache.nt_pairs.size());
    certificates.segment_segment.resize(cache.ss_pairs.size());
    const std::size_t count = cache.nt_pairs.size() + cache.ss_pairs.size();
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    #pragma omp parallel for schedule(static) if(parallel && count >= 128)
    for (std::size_t entry = 0; entry < count; ++entry) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            const bool segment = entry >= cache.nt_pairs.size();
            const std::size_t index = segment ? entry - cache.nt_pairs.size() : entry;
            std::array<int, 4> nodes;
            if (segment) {
                const auto& pair = cache.ss_pairs[index];
                nodes = {pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
            } else {
                const auto& pair = cache.nt_pairs[index];
                nodes = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
            }
            std::array<Vec3, 4> points;
            std::array<AABB, 4> boxes;
            for (int node = 0; node < 4; ++node) {
                points[node] = positions[nodes[node]];
                boxes[node] = cache.node_boxes[nodes[node]];
            }
            auto& flags = segment ? certificates.segment_segment : certificates.node_triangle;
            flags[index] = solver_detail::contact_boxes_separated(points, boxes, segment, d_hat);
        } catch (...) {
            if (!failed.exchange(true, std::memory_order_relaxed)) error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);
    if (prepare_contact_masks) {
        // Build immutable word masks once for all sweeps in this node-box block.
        // Bit zero is the local FEM contribution; contact bits retain incident order.
        certificates.vertex_clear_words.resize(cache.vertex_nt.size());
        #pragma omp parallel for schedule(static) if(parallel && cache.vertex_nt.size() >= 128)
        for (std::size_t vertex = 0; vertex < cache.vertex_nt.size(); ++vertex) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const auto& nt = cache.vertex_nt[vertex];
                const auto& ss = cache.vertex_ss[vertex];
                auto& words = certificates.vertex_clear_words[vertex];
                constexpr unsigned grain = solver_detail::contact_grain;
                words.assign((1 + nt.size() + ss.size() + grain - 1) / grain, 0u);
                for (std::size_t local = 0; local < nt.size() + ss.size(); ++local) {
                    const bool clear = local < nt.size()
                        ? certificates.node_triangle[nt[local].pair_index]
                        : certificates.segment_segment[ss[local - nt.size()].pair_index];
                    if (clear) words[(local + 1) / grain] |= 1u << ((local + 1) % grain);
                }
            } catch (...) {
                if (!failed.exchange(true, std::memory_order_relaxed)) error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
    }
    certificates.valid = true;
}

struct SimdContactBatch {
    std::array<ipc_simd::MeshContactOutput, ipc_simd::contact_tile_width> values;
    std::array<unsigned, ipc_simd::contact_tile_width> flags{};
    std::size_t count = 0;

    SimdContactBatch() {
        // Returned batches may copy every slot, including rejected contacts.
        for (auto& value : values) {
            value.gradient.setZero();
            value.hessian.setZero();
            value.friction_gradient.setZero();
            value.friction_hessian.setZero();
        }
    }
};

static unsigned gather_simd_contact(
    int vertex, std::size_t local, const BroadPhase::Cache& cache,
    const SimParams& params, const std::vector<Vec3>& x,
    const std::vector<Vec3>* previous, ipc_simd::MeshContactInput& input,
    const SimdBoxContactCertificates* certificates = nullptr) {
    const auto& nt = cache.vertex_nt[vertex];
    std::array<int, 4> nodes;
    int role;
    bool clear = false, within;
    const bool segment = local >= nt.size();
    const double distance2 = params.d_hat * params.d_hat;
    if (!segment) {
        const auto& entry = nt[local];
        if (certificates && certificates->valid && certificates->node_triangle[entry.pair_index]) return 2u;
        const auto& pair = cache.nt_pairs[entry.pair_index];
        nodes = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
        role = entry.dof;
        within = node_triangle_aabbs_within_distance(x[nodes[0]], x[nodes[1]],
            x[nodes[2]], x[nodes[3]], distance2, &clear);
    } else {
        const auto& entry = cache.vertex_ss[vertex][local - nt.size()];
        if (certificates && certificates->valid && certificates->segment_segment[entry.pair_index]) return 2u;
        const auto& pair = cache.ss_pairs[entry.pair_index];
        nodes = {pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
        role = entry.dof;
        within = segment_aabbs_within_distance(x[nodes[0]], x[nodes[1]],
            x[nodes[2]], x[nodes[3]], distance2, &clear);
    }
    if (!within) return clear ? 2u : 0u;
    input.role = role;
    input.segment_segment = segment;
    for (int node = 0; node < 4; ++node) {
        input.positions[node] = x[nodes[node]];
        if (params.friction_coefficient != 0.0)
            input.previous_positions[node] = (*previous)[nodes[node]];
    }
    return 1u;
}

// Mesh lookup and AABB certificates belong to the organizer, not the SIMD
// transpose. Compact a coarse range into local AoS tiles and emit active
// derivatives in contact order; rejected certificates can be written early.
template <class Emit>
static void evaluate_simd_contact_batch(
    int vertex, std::size_t begin, std::size_t count,
    const BroadPhase::Cache& cache, const SimParams& params,
    const std::vector<Vec3>& x, const std::vector<Vec3>* previous, const Emit& emit,
    const SimdBoxContactCertificates* certificates = nullptr) {
    if (count == 0) return;
    std::array<ipc_simd::MeshContactInput, ipc_simd::contact_tile_width> inputs;
    std::array<ipc_simd::MeshContactOutput, ipc_simd::contact_tile_width> output;
    std::array<std::size_t, ipc_simd::contact_tile_width> indices;
    std::array<unsigned char, ipc_simd::contact_tile_width> derivative_active;
    const bool use_derivative_mask = std::isfinite(params.dt2() * params.k_barrier);
    std::size_t active = 0;
    const auto flush = [&] {
        if (!active) return;
        ipc_simd::mesh_contact_derivatives_tile(inputs.data(), active,
            params.d_hat, params.k_barrier, params.friction_coefficient, params.dt(),
            params.friction_velocity_epsilon, output.data(),
            use_derivative_mask ? derivative_active.data() : nullptr);
        for (std::size_t i = 0; i < active; ++i) {
            const bool contributes = !use_derivative_mask || derivative_active[i];
            emit(indices[i], contributes ? 1u : 0u, contributes ? &output[i] : nullptr);
        }
        active = 0;
    };
    for (std::size_t i = 0; i < count; ++i) {
        const unsigned flags = gather_simd_contact(vertex, begin + i, cache,
            params, x, previous, inputs[active], certificates);
        if (!(flags & 1u)) {
            emit(i, flags, nullptr);
            continue;
        }
        indices[active++] = i;
        if (active == ipc_simd::contact_tile_width) flush();
    }
    flush();
}

static void accumulate_simd_mesh_contacts(
    int vertex, const BroadPhase::Cache& cache, const SimParams& params,
    const std::vector<Vec3>& x, const std::vector<Vec3>* previous, bool cooperative,
    safe_step_detail::VertexAabbRejections* rejections, Vec3& g, Mat33& H,
    const SimdBoxContactCertificates* certificates = nullptr) {
    const std::size_t count = cache.vertex_nt[vertex].size() + cache.vertex_ss[vertex].size();
    if (count == 0) return;
    const std::size_t width = ipc_simd::contact_tile_width;
    const int batches = static_cast<int>((count + width - 1) / width);
    const double dt2k = params.dt2() * params.k_barrier;
    const auto accumulate_value = [&](const ipc_simd::MeshContactOutput& value) {
        g += dt2k * value.gradient;
        H += dt2k * value.hessian;
        if (params.friction_coefficient != 0.0) {
            g += value.friction_gradient;
            H += value.friction_hessian;
        }
    };
    const auto evaluate = [&](int index) {
        const std::size_t begin = static_cast<std::size_t>(index) * width;
        SimdContactBatch result;
        result.count = std::min(width, count - begin);
        evaluate_simd_contact_batch(vertex, begin, result.count, cache, params, x, previous,
            [&](std::size_t i, unsigned flags, const ipc_simd::MeshContactOutput* value) {
                result.flags[i] = flags;
                if (value) result.values[i] = *value;
                if (rejections) rejections->clear[begin + i] = (flags & 2u) != 0;
            }, certificates);
        return result;
    };
    const auto accumulate = [&](const SimdContactBatch& batch) {
        for (std::size_t i = 0; i < batch.count; ++i) {
            if (!(batch.flags[i] & 1u)) continue;
            accumulate_value(batch.values[i]);
        }
    };
    if (cooperative && count >= 32 && omp_get_num_threads() > 1)
        solver_detail::parallel_contact_tasks(batches, evaluate, accumulate);
    else {
        std::array<ipc_simd::MeshContactInput, ipc_simd::contact_tile_width> inputs;
        std::array<ipc_simd::MeshContactOutput, ipc_simd::contact_tile_width> outputs;
        std::array<unsigned char, ipc_simd::contact_tile_width> derivative_active;
        const bool use_derivative_mask = std::isfinite(dt2k);
        std::size_t active = 0;
        const auto flush = [&] {
            if (!active) return;
            ipc_simd::mesh_contact_derivatives_tile(inputs.data(), active,
                params.d_hat, params.k_barrier, params.friction_coefficient,
                params.dt(), params.friction_velocity_epsilon, outputs.data(),
                use_derivative_mask ? derivative_active.data() : nullptr);
            for (std::size_t i = 0; i < active; ++i)
                if (!use_derivative_mask || derivative_active[i]) accumulate_value(outputs[i]);
            active = 0;
        };
        for (std::size_t local = 0; local < count; ++local) {
            const unsigned flags = gather_simd_contact(vertex, local, cache,
                params, x, previous, inputs[active], certificates);
            if (rejections) rejections->clear[local] = (flags & 2u) != 0;
            if ((flags & 1u) && ++active == width) flush();
        }
        flush();
    }
}

} // namespace

template <bool UseStoredMembrane = false>
Vec3 gs_vertex_delta_live_barrier_experimental(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                  const std::vector<Vec3>& xhat, std::vector<Vec3>& x, const BroadPhase& broad_phase, const PinMap* pin_map,
                                  const IncidentTriangles* incident_triangles,
                                  const std::vector<ShapeGrads>* rest_shape_grads,
                                  const std::vector<Vec3>* previous_positions, bool cooperative = false,
                                  safe_step_detail::VertexAabbRejections* rejections = nullptr,
                                  const physics_detail::MembraneDerivativeView* stored_membrane = nullptr) {
    const auto& bp_cache = broad_phase.cache();
    if (rejections) rejections->distance = 0.0;
    auto local_derivatives = [&]() {
        if constexpr (UseStoredMembrane) {
            return physics_detail::compute_local_gradient_and_hessian_with_stored_membrane_unchecked(
                vi, ref_mesh, adj, pins, params, x, xhat, pin_map,
                incident_triangles, rest_shape_grads, previous_positions,
                *stored_membrane);
        } else {
            return physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(
                vi, ref_mesh, adj, pins, params, x, xhat, pin_map,
                incident_triangles, rest_shape_grads, previous_positions);
        }
    }();
    Vec3& g = local_derivatives.first;
    Mat33& H = local_derivatives.second;

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;
        const double d_hat2 = params.d_hat * params.d_hat;

        if (rejections) {
            // Allocate before helpers run. Every contact writes a distinct
            // byte, and safe-step consumes these certificates after the join.
            rejections->clear.resize(
                bp_cache.vertex_nt[vi].size() + bp_cache.vertex_ss[vi].size());
            rejections->distance = params.d_hat;
        }
        if (params.friction_coefficient == 0.0 && !cooperative) {
            std::size_t contact_index = 0;
            // Whole-vertex work can accumulate directly in contact order,
            // without materializing optional gradient/Hessian records.
            for (const auto& entry : bp_cache.vertex_nt[vi]) {
                const auto& p = bp_cache.nt_pairs[entry.pair_index];
                bool clear = false;
                const bool within = node_triangle_aabbs_within_distance(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], d_hat2,
                    rejections ? &clear : nullptr);
                if (rejections) rejections->clear[contact_index++] = clear;
                if (!within) continue;
                const auto [bg, bH] = node_triangle_barrier_self_gradient_and_hessian(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]],
                    params.d_hat, entry.dof);
                g += dt2k * bg;
                H += dt2k * bH;
            }
            for (const auto& entry : bp_cache.vertex_ss[vi]) {
                const auto& p = bp_cache.ss_pairs[entry.pair_index];
                bool clear = false;
                const bool within = segment_aabbs_within_distance(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], d_hat2,
                    rejections ? &clear : nullptr);
                if (rejections) rejections->clear[contact_index++] = clear;
                if (!within) continue;
                const auto [bg, bH] = segment_segment_barrier_self_gradient_and_hessian(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]],
                    params.d_hat, entry.dof);
                g += dt2k * bg;
                H += dt2k * bH;
            }
            return matrix3d_inverse(H) * g;
        }

        // Share contact evaluation between direct accumulation and cooperative
        // staging so both paths use the same barrier and friction geometry.
        const auto evaluate_nt = [&](const BroadPhase::Cache::VertexPairEntry& entry,
                                     std::size_t contact_index,
                                     const auto& consume) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            bool clear = false;
            const bool within = node_triangle_aabbs_within_distance(
                x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]],
                d_hat2, rejections ? &clear : nullptr);
            if (rejections) rejections->clear[contact_index] = clear;
            if (!within) return;
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
                consume(bg, bH, fg, fH);
            } else {
                const auto [bg, bH] =
                    node_triangle_barrier_self_gradient_and_hessian(
                        x[p.node], x[p.tri_v[0]], x[p.tri_v[1]],
                        x[p.tri_v[2]], params.d_hat, entry.dof);
                consume(bg, bH, Vec3::Zero(), Mat33::Zero());
            }
        };
        const auto evaluate_ss = [&](const BroadPhase::Cache::VertexPairEntry& entry,
                                     std::size_t contact_index,
                                     const auto& consume) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            bool clear = false;
            const bool within = segment_aabbs_within_distance(
                x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]],
                d_hat2, rejections ? &clear : nullptr);
            if (rejections) rejections->clear[contact_index] = clear;
            if (!within) return;
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
                consume(bg, bH, fg, fH);
            } else {
                const auto [bg, bH] =
                    segment_segment_barrier_self_gradient_and_hessian(
                        x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]],
                        params.d_hat, entry.dof);
                consume(bg, bH, Vec3::Zero(), Mat33::Zero());
            }
        };
        const auto accumulate = [&](const Vec3& gradient, const Mat33& hessian,
                                    const Vec3& friction_gradient, const Mat33& friction_hessian) {
            g += dt2k * gradient;
            H += dt2k * hessian;
            if (params.friction_coefficient != 0.0) {
                // Friction derivatives already include their dt^2 scaling.
                g += friction_gradient;
                H += friction_hessian;
            }
        };
        const auto& nt = bp_cache.vertex_nt[vi];
        const auto& ss = bp_cache.vertex_ss[vi];
        if (!cooperative) {
            // Whole-vertex friction follows the same NT-then-SS order without
            // copying each contact's derivatives into a Contribution record.
            for (std::size_t i = 0; i < nt.size(); ++i)
                evaluate_nt(nt[i], i, accumulate);
            for (std::size_t i = 0; i < ss.size(); ++i)
                evaluate_ss(ss[i], nt.size() + i, accumulate);
        } else {
            struct Contribution {
                Vec3 gradient = Vec3::Zero(), friction_gradient = Vec3::Zero();
                Mat33 hessian = Mat33::Zero(), friction_hessian = Mat33::Zero();
            };
            const int nt_count = static_cast<int>(nt.size());
            solver_detail::ordered_contact_tasks(nt_count + static_cast<int>(ss.size()), cooperative,
                [&](int i) {
                    std::optional<Contribution> value;
                    const auto store = [&](const Vec3& gradient, const Mat33& hessian,
                                           const Vec3& friction_gradient, const Mat33& friction_hessian) {
                        value.emplace();
                        value->gradient = gradient;
                        value->hessian = hessian;
                        value->friction_gradient = friction_gradient;
                        value->friction_hessian = friction_hessian;
                    };
                    if (i < nt_count) evaluate_nt(nt[i], i, store);
                    else evaluate_ss(ss[i - nt_count], i, store);
                    return value;
                },
                [&](const std::optional<Contribution>& value) {
                    if (value) accumulate(value->gradient, value->hessian,
                                          value->friction_gradient, value->friction_hessian);
                });
        }
    }

    return matrix3d_inverse(H) * g;
}

// Keep SIMD contact assembly separate from v1's scalar local-update helper.
static Vec3 gs_vertex_delta_live_barrier_simd(
    int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj,
    const std::vector<Pin>& pins, const SimParams& params,
    const std::vector<Vec3>& xhat, const std::vector<Vec3>& x,
    const BroadPhase& broad_phase, const PinMap* pin_map,
    const IncidentTriangles* incident_triangles,
    const std::vector<ShapeGrads>* rest_shape_grads,
    const std::vector<Vec3>* previous_positions, bool cooperative = false,
    safe_step_detail::VertexAabbRejections* rejections = nullptr,
    const std::pair<Vec3, Mat33>* prepared_local = nullptr,
    const SimdBoxContactCertificates* certificates = nullptr) {
    if (rejections) rejections->distance = 0.0;
    auto local = prepared_local ? *prepared_local
        : physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(
            vi, ref_mesh, adj, pins, params, x, xhat, pin_map,
            incident_triangles, rest_shape_grads, previous_positions);
    if (params.d_hat > 0.0) {
        const auto& cache = broad_phase.cache();
        if (rejections) {
            rejections->clear.resize(cache.vertex_nt[vi].size() + cache.vertex_ss[vi].size());
            rejections->distance = params.d_hat;
        }
        accumulate_simd_mesh_contacts(vi, cache, params, x, previous_positions,
            cooperative, rejections, local.first, local.second, certificates);
    }
    return matrix3d_inverse(local.second) * local.first;
}

Vec3 gs_solid_vertex_delta_live_barrier(
    const int node, const RefMesh& ref_mesh,
    const std::vector<Pin>& pins, const SimParams& params,
    const std::vector<Vec3>& xhat, const std::vector<Vec3>& x,
    const BroadPhase& broad_phase,
    const std::vector<unsigned char>& solid_node_mask,
    const std::vector<unsigned char>& surface_node_mask,
    const PinMap& pin_map,
    const std::vector<Vec3>* previous_positions, bool cooperative = false) {
    const auto [gradient, block] =
        solid_ipc_detail::compute_solid_local_gradient_and_block_unchecked(
            node, ref_mesh, pins, params, x, xhat, broad_phase,
            &solid_node_mask, &surface_node_mask, &pin_map,
            previous_positions, cooperative);
    return matrix3d_inverse(block) * gradient;
}

// Elastic terms read x_elastic (live, GS-style across colors); barrier terms read
// x_barrier (iteration-start snapshot, Jacobi-style). Safe to call in parallel
// within a single elastic-coloring color class.
Vec3 gs_vertex_delta_frozen_barrier(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                    const std::vector<Vec3>& xhat, const std::vector<Vec3>& x_elastic, const std::vector<Vec3>& x_barrier, const BroadPhase& broad_phase, const PinMap* pin_map,
                                    const IncidentTriangles* incident_triangles,  const std::vector<ShapeGrads>* rest_shape_grads,
                                    const std::vector<Vec3>* previous_positions,
                                    const std::vector<NodeTriangleDistanceResult>& nt_distances,
                                    const std::vector<SegmentSegmentDistanceResult>& ss_distances) {
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
                        params.k_barrier, 1.0e-12, &nt_distances[entry.pair_index]);
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
                        params.d_hat, entry.dof, 1.0e-12, &nt_distances[entry.pair_index]);
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
                        params.k_barrier, 1.0e-12, &ss_distances[entry.pair_index]);
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
                        params.d_hat, entry.dof, 1.0e-12, &ss_distances[entry.pair_index]);
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
    const bool needs_mesh_contact_search =
        params.d_hat > 0.0 || params.use_ccd || params.use_ogc;
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
            if (needs_mesh_contact_search) {
                // Rebuild contact candidates, combine their dependencies with
                // elastic dependencies, and color the resulting graph.
                broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::DeformableSolver);
                build_contact_adj(broad_phase.cache(), static_cast<int>(xnew.size()), bca);
                union_adjacency(ea, bca, combined_adj);
                greedy_color_conflict_graph(combined_adj, color_groups, &workspace.coloring_workspace);
                const BroadPhase::Cache& bp_cache = broad_phase.cache();
                // Vertices in one color share no dependencies, so process contact-heavy vertices first to avoid end-of-color stragglers.
                for (std::vector<int>& group : color_groups) std::stable_sort(group.begin(), group.end(), [&](const int a, const int b) { return bp_cache.vertex_nt[static_cast<std::size_t>(a)].size() + bp_cache.vertex_ss[static_cast<std::size_t>(a)].size() > bp_cache.vertex_nt[static_cast<std::size_t>(b)].size() + bp_cache.vertex_ss[static_cast<std::size_t>(b)].size(); });
            } else {
                // Collision-free solve: keep node-box step clipping, but do no
                // primitive BVH construction, pair search, or contact-aware
                // coloring. Elastic topology alone determines the schedule.
                broad_phase.initialize_node_boxes_only(blue_boxes);
                greedy_color_conflict_graph(ea, color_groups, &workspace.coloring_workspace);
            }
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

SolverResult global_gauss_seidel_solver_basic_experimental(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        const std::vector<Vec3>& v,
                                        BroadPhase& broad_phase,
                                        const std::string& outdir,
                                        const std::vector<Vec3>* previous_positions) {
    // A direct call to v1 always selects scalar assembly, regardless of driver flags.
    if (params.use_basic_experimental_v2) {
        SimParams scalar_params = params;
        scalar_params.use_basic_experimental_v2 = false;
        scalar_params.use_simd = false;
        return global_gauss_seidel_solver_basic_experimental(ref_mesh, adj, pins, scalar_params,
            xnew, xhat, v, broad_phase, outdir, previous_positions);
    }

    //create node (blue) boxes and create broad phase (red boxes) accordingly
    validate_solver_friction_parameters(
        params, "global_gauss_seidel_solver_basic_experimental");
    std::vector<Vec3> reconstructed_previous_positions;
    previous_positions = resolve_experimental_friction_previous_positions(
        params, xnew, xhat, v, previous_positions,
        reconstructed_previous_positions,
        "global_gauss_seidel_solver_basic_experimental");
    const int nv = static_cast<int>(xnew.size());
    static ExperimentalSolverWorkspace workspace;
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
    (void)params.dt2();
    #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
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
    const bool needs_mesh_contact_search =
        params.d_hat > 0.0 || params.use_ccd || params.use_ogc;
    std::vector<std::vector<int>>& color_groups = needs_mesh_contact_search
        ? workspace.color_groups : workspace.elastic_color_groups;
    const auto compute_residual = [&]() {
        build_frozen_residual_workspace(
            ref_mesh, params, xnew, broad_phase,
            workspace.frozen_residual, &workspace.rest_shape_grads);
        return compute_global_deformable_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, workspace.deformable_nodes, &pm, &workspace.incident_triangles, &workspace.rest_shape_grads, &workspace.frozen_residual, previous_positions);
    };

    SolverResult result;
    // anchor for clip boxes and prev_disp
    std::vector<Vec3>& xnew_substep_start = workspace.xnew_substep_start;
    #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
    for (int vi = 0; vi < nv; ++vi) xnew_substep_start[vi] = xnew[vi];
 
    solver_detail::ColoredContactSweep contact_sweep;
    solver_detail::ColoredVertexSweep colored_vertex_sweep;
    const bool use_contact_sweep = params.use_parallel && omp_get_max_threads() > 1
        && params.friction_coefficient == 0.0 && params.d_hat > 0.0
        && !params.use_ogc;
    double r1=0.;
    //gs loop
    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if((iter-1)%params.node_box_update_count==0){//rebuild node boxes and color accordingly
            if (params.verbose)
                std::fprintf(stderr, "  [GS] iter %d  rebuilding node boxes\n", iter);
            //create new node boxes
            #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
            for (int i = 0; i < nv; ++i) {
                const double r = node_box_size_fn(i);
                blue_boxes[i] = AABB(xnew[i] - Vec3::Constant(r), xnew[i] + Vec3::Constant(r));
            }
            if (needs_mesh_contact_search) {
                // Rebuild contact candidates, combine their dependencies with
                // elastic dependencies, and color the resulting graph.
                broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::DeformableSolver);
                build_contact_adj(broad_phase.cache(), static_cast<int>(xnew.size()), bca);
                union_adjacency(ea, bca, combined_adj);
                greedy_color_conflict_graph(combined_adj, color_groups, &workspace.coloring_workspace);
                const BroadPhase::Cache& bp_cache = broad_phase.cache();
                // Vertices in one color share no dependencies, so process contact-heavy vertices first to avoid end-of-color stragglers.
                #pragma omp parallel for schedule(dynamic, 1) if(params.use_parallel && nv >= 128)
                for (int color = 0; color < static_cast<int>(color_groups.size()); ++color) {
                    auto& group = color_groups[color];
                    std::stable_sort(group.begin(), group.end(), [&](const int a, const int b) {
                        return bp_cache.vertex_nt[a].size() + bp_cache.vertex_ss[a].size()
                            > bp_cache.vertex_nt[b].size() + bp_cache.vertex_ss[b].size();
                    });
                }
            } else {
                // Collision-free solve: keep node-box step clipping, but do no
                // primitive BVH construction, pair search, or contact-aware
                // coloring. Elastic topology alone determines the schedule.
                broad_phase.initialize_node_boxes_only(blue_boxes);
                // The workspace invalidates these colors with elastic topology.
                if (color_groups.empty())
                    greedy_color_conflict_graph(ea, color_groups, &workspace.coloring_workspace);
            }
        }

        if (use_contact_sweep && (iter - 1) % params.node_box_update_count == 0)
            contact_sweep.prepare(color_groups, broad_phase.cache());

        if (iter == 1 && !params.fixed_iters) {
          r1 = compute_residual();
          result.has_residual = true;
          result.initial_residual = r1;
          result.final_residual = r1;
          if (r1 < params.tol_rel * r1 || r1 < params.tol_abs) {
            result.converged = true;
            break;
          }
        }

        const auto proposed_position = [&](int vi,
            safe_step_detail::VertexAabbRejections* rejections,
            bool cooperative) -> Vec3 {
          return xnew[vi] -
                 params.damping *
                     gs_vertex_delta_live_barrier_experimental(
                         vi, ref_mesh, adj, pins, params, xhat, xnew,
                         broad_phase, &pm, &workspace.incident_triangles[vi],
                         &workspace.rest_shape_grads, previous_positions,
                         cooperative, rejections);
        };
        const auto process_vertex = [&](int vi, bool cooperative = false) {
          // Scratch belongs to this worker and is consumed before updating the
          // vertex. Colors keep every incident pair fixed during these calls.
          thread_local safe_step_detail::VertexAabbRejections scratch;
          auto* rejections = params.use_ccd && !params.use_ogc
              && params.d_hat > 1e-8 ? &scratch : nullptr;
          const Vec3 proposed = proposed_position(vi, rejections, cooperative);
          per_vertex_safe_step(broad_phase, xnew, vi, proposed,
                               0.9, params.use_ogc ? false : params.use_ccd,
                               params.use_ticcd, params.use_ogc, cooperative, rejections);
        };
        if (use_contact_sweep) {
          const auto &cache = broad_phase.cache();
          const double dh2 = params.d_hat * params.d_hat,
                       dt2k = params.dt2() * params.k_barrier;
          const auto compute = [&](int vi, int local,
                                   solver_detail::ContactContribution &value) -> unsigned {
            bool aabb_clear=false;
            if (local == 0) {
              auto pair = physics_detail::
                  compute_local_gradient_and_hessian_no_barrier_unchecked(
                      vi, ref_mesh, adj, pins, params, xnew, xhat, &pm,
                      &workspace.incident_triangles[vi],
                      &workspace.rest_shape_grads, previous_positions);
              value.gradient = pair.first;
              value.hessian = pair.second;
              return 1;
            }
            --local;
            int nt = cache.vertex_nt[vi].size();
            if (local < nt) {
              const auto &entry = cache.vertex_nt[vi][local];
              const auto &p = cache.nt_pairs[entry.pair_index];
              if (!node_triangle_aabbs_within_distance(
                      xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]],
                      xnew[p.tri_v[2]], dh2, &aabb_clear)) {
                return aabb_clear?2u:0u;
              }
              auto pair = node_triangle_barrier_self_gradient_and_hessian(
                  xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]],
                  xnew[p.tri_v[2]], params.d_hat, entry.dof);
              value.gradient = pair.first;
              value.hessian = pair.second;
            } else {
              const auto &entry = cache.vertex_ss[vi][local - nt];
              const auto &p = cache.ss_pairs[entry.pair_index];
              if (!segment_aabbs_within_distance(xnew[p.v[0]], xnew[p.v[1]],
                                                 xnew[p.v[2]], xnew[p.v[3]],
                                                 dh2, &aabb_clear)) {
                return aabb_clear?2u:0u;
              }
              auto pair = segment_segment_barrier_self_gradient_and_hessian(
                  xnew[p.v[0]], xnew[p.v[1]], xnew[p.v[2]], xnew[p.v[3]],
                  params.d_hat, entry.dof);
              value.gradient = pair.first;
              value.hessian = pair.second;
            }
            return 1;
          };
          const auto apply =
              [&](int vi, const solver_detail::ContactContribution *values, const solver_detail::ContactMaskWord* mask) {
                Vec3 g = values[0].gradient;
                Mat33 H = values[0].hessian;
                int count =
                    cache.vertex_nt[vi].size() + cache.vertex_ss[vi].size();
                const auto add=[&](int j){g+=dt2k*values[j].gradient;H+=dt2k*values[j].hessian;};
                solver_detail::for_active_contact(mask,count,add);
                const Vec3 delta = matrix3d_inverse(H) * g;
                const Vec3 proposed = xnew[vi] - params.damping * delta;
                {
                  const auto &box = cache.node_boxes[vi];
                  const Vec3 lo = (box.min + Vec3::Constant(1e-10)).eval();
                  const Vec3 hi = (box.max - Vec3::Constant(1e-10)).eval();
                  const Vec3 next = proposed.cwiseMax(lo).cwiseMin(hi);
                  contact_sweep.steps[vi] = next - xnew[vi];
                  contact_sweep.nonzero_step[vi] =
                      !(contact_sweep.steps[vi].squaredNorm() < 1e-28);
                  contact_sweep.short_step[vi]=params.d_hat>1e-8 && std::isfinite(dh2) && contact_sweep.steps[vi].squaredNorm()<dh2/16.0;
                }
              };
          const auto ccd = [&](int vi, int local,
                               solver_detail::ContactContribution &value, bool aabb_clear) -> bool {
            if (!contact_sweep.nonzero_step[vi] || !params.use_ccd ||
                local == 0)
              return false;
            --local;
            int nt = cache.vertex_nt[vi].size();
            // A rejected Euclidean AABB distance exceeds d_hat, so some axis
            // gap exceeds d_hat/sqrt(3). Moving one endpoint by less than
            // d_hat/4 cannot close that gap. The original swept-AABB test
            // therefore also rejects this pair; no CCD result is approximated.
            if(aabb_clear && contact_sweep.short_step[vi]) {
                return false;
            }
            CCDResult result;
            if (local < nt) {
              const auto &entry = cache.vertex_nt[vi][local];
              result = safe_step_detail::node_triangle_vertex_ccd(
                  cache.nt_pairs[entry.pair_index], entry.dof, vi, xnew,
                  contact_sweep.steps[vi], params.use_ticcd);
            } else {
              const auto &entry = cache.vertex_ss[vi][local - nt];
              result = safe_step_detail::segment_segment_vertex_ccd(
                  cache.ss_pairs[entry.pair_index], entry.dof, vi, xnew,
                  contact_sweep.steps[vi], params.use_ticcd);
            }
            if(result.collision)value.toi=result.t;
            return result.collision;
          };
          const auto commit =
              [&](int vi, const solver_detail::ContactContribution *values, const solver_detail::ContactMaskWord* mask) {
                if (!contact_sweep.nonzero_step[vi])
                  return;
                double toi = 1.0;
                bool collision = false;
                int count =
                    cache.vertex_nt[vi].size() + cache.vertex_ss[vi].size();
                const auto consider=[&](int j){collision=true;toi=std::min(toi,values[j].toi);};
                solver_detail::for_active_contact(mask,count,consider);
                double step = collision ? 0.9 * toi : 1.0;
                xnew[vi] = xnew[vi] + step * contact_sweep.steps[vi];
              };
          contact_sweep.run(color_groups, compute, apply, process_vertex, ccd,
                            commit);
        } else if (params.use_parallel) {
          // Fixed-iteration collision-free and friction solves can reuse one
          // team until the next node-box rebuild. Keep every color barrier,
          // including the final color of each sweep, and each vertex's arithmetic.
          // Convergence-controlled solves still return after every sweep so
          // their residual checks and stopping iteration remain unchanged.
          const bool collision_free = !needs_mesh_contact_search
              && params.k_sdf == 0.0 && params.friction_coefficient == 0.0;
          const int sweeps = params.fixed_iters
              && (collision_free || params.friction_coefficient > 0.0)
              ? std::min(params.max_global_iters - iter + 1,
                         params.node_box_update_count - (iter - 1) % params.node_box_update_count)
              : 1;
          if (params.fixed_iters && params.friction_coefficient > 0.0) {
            colored_vertex_sweep.run(color_groups, sweeps, process_vertex);
          } else {
#pragma omp parallel
          {
            for (int sweep = 0; sweep < sweeps; ++sweep) {
              for (const std::vector<int> &group : color_groups) {
#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                  process_vertex(group[static_cast<std::size_t>(i)]);
              }
            }
          }
          }
          iter += sweeps - 1;
        } else {
          for (int vi = 0; vi < nv; ++vi)
            process_vertex(vi);
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
    #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
    for (int i = 0; i < nv; ++i)
        prev_disp[i] = (xnew[i] - xnew_substep_start[i]).norm();

    if (params.fixed_iters) result.converged = true;

    //write substep data
    if (params.write_substeps) {
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, &color_groups);
    }

    return result;
}

// AoS triangle records in vertex/incident order, delimited by node_offsets.
// Canonical SIMD materials use vertex indices; scalar color storage uses group
// slots. SIMD color layouts retain offsets while materials stay canonical.
struct TriangleStorage {
    std::vector<std::size_t> node_offsets;
    std::vector<int> triangle_indices;
    std::vector<int> local_corners;
    // Scalar color snapshots: [x_0^0, x_1^0, x_2^0, x_0^1, ...].
    std::vector<Vec3> positions;
    std::vector<Mat22> dm_inverse;
    std::vector<double> areas;
    std::vector<Vec2> shape_gradients;
};

// Output storage with exactly the same entry order as TriangleStorage.
// Entry e is that triangle's contribution to its active node, NOT the sum of
// all triangles incident on the node, nor a full 9x9 triangle Hessian.
// The matching input's node_offsets also delimit these output arrays.
// Scalar prepass values include rest area and dt^2. Gradient is +dE/dx;
// Hessian is the exact self block (no PSD projection). Other energies and
// contact contributions are not included.
struct ColorTriangleDerivatives {
    std::vector<Vec3> gradients;
    std::vector<Mat33> hessians;
};

struct HingeStorage {
    std::vector<std::size_t> node_offsets;
    std::vector<int> hinge_indices;
    std::vector<int> active_nodes;
    std::vector<double> coefficients;
    std::vector<double> rest_angles;
};

// Material records retain vertex/incident order independently of coloring.
// Value snapshots detect in-place rest-property changes between solver calls.
struct SimdVertexMaterials {
    const RefMesh* mesh = nullptr;
    bool valid = false, bending_prepared = false;
    int vertices = 0;
    std::vector<int> saved_triangles;
    std::vector<Mat22> saved_dm_inverse;
    std::vector<double> saved_areas;
    std::vector<Hinge> saved_hinges;
    TriangleStorage triangles;
    HingeStorage hinges;
    template <class T>
    static bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
        return a.size() == b.size() && (a.empty()
            || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
    }
    bool matches(const RefMesh& candidate, int count, bool with_bending) const {
        static_assert(sizeof(Mat22) == 4 * sizeof(double));
        static_assert(sizeof(Hinge) == 4 * sizeof(int) + 2 * sizeof(double));
        return valid && mesh == &candidate && vertices == count && (!with_bending || bending_prepared)
            && same_bytes(saved_triangles, candidate.tris)
            && same_bytes(saved_dm_inverse, candidate.Dm_inverse)
            && same_bytes(saved_areas, candidate.area)
            && same_bytes(saved_hinges, candidate.hinges);
    }
    void rebuild(const RefMesh& candidate, const std::vector<IncidentTriangles>& incident,
                 int count, bool with_bending, bool parallel) {
        valid = false;
        triangles.node_offsets.resize(count + 1);
        hinges.node_offsets.resize(count + 1);
        triangles.node_offsets[0] = hinges.node_offsets[0] = 0;
        for (int vi = 0; vi < count; ++vi) {
            triangles.node_offsets[vi + 1] = triangles.node_offsets[vi] + incident[vi].size();
            const auto found = with_bending ? candidate.hinge_adj.find(vi) : candidate.hinge_adj.end();
            hinges.node_offsets[vi + 1] = hinges.node_offsets[vi]
                + (found == candidate.hinge_adj.end() ? 0 : found->second.size());
        }
        const auto triangle_count = triangles.node_offsets.back();
        triangles.triangle_indices.resize(triangle_count);
        triangles.dm_inverse.resize(triangle_count);
        triangles.areas.resize(triangle_count);
        triangles.shape_gradients.resize(triangle_count);
        const auto hinge_count = hinges.node_offsets.back();
        hinges.hinge_indices.resize(hinge_count);
        hinges.active_nodes.resize(hinge_count);
        hinges.coefficients.resize(hinge_count);
        hinges.rest_angles.resize(hinge_count);
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        #pragma omp parallel for schedule(static) if(parallel && count >= 128)
        for (int vi = 0; vi < count; ++vi) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                auto e = triangles.node_offsets[vi];
                for (const auto& [triangle, corner] : incident[vi]) {
                    triangles.triangle_indices[e] = triangle;
                    triangles.dm_inverse[e] = candidate.Dm_inverse[triangle];
                    triangles.areas[e] = candidate.area[triangle];
                    triangles.shape_gradients[e] = shape_function_gradients(triangles.dm_inverse[e])[corner];
                    ++e;
                }
                e = hinges.node_offsets[vi];
                const auto found = with_bending ? candidate.hinge_adj.find(vi) : candidate.hinge_adj.end();
                if (found == candidate.hinge_adj.end()) continue;
                for (const auto& [index, role] : found->second) {
                    hinges.hinge_indices[e] = index;
                    hinges.active_nodes[e] = role;
                    hinges.coefficients[e] = candidate.hinges[index].c_e;
                    hinges.rest_angles[e] = candidate.hinges[index].bar_theta;
                    ++e;
                }
            } catch (...) {
                if (!failed.exchange(true, std::memory_order_relaxed)) error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
        saved_triangles = candidate.tris;
        saved_dm_inverse = candidate.Dm_inverse;
        saved_areas = candidate.area;
        saved_hinges = candidate.hinges;
        mesh = &candidate;
        vertices = count;
        bending_prepared = with_bending;
        valid = true;
    }
};

struct SimdColorStorageKey {
    const RefMesh* mesh = nullptr;
    bool valid = false, bending = false;
    std::vector<std::vector<int>> colors;
    bool matches(const RefMesh& candidate, const std::vector<std::vector<int>>& groups,
                 bool with_bending) const {
        return valid && mesh == &candidate && bending == with_bending && colors == groups;
    }
    void capture(const RefMesh& candidate, const std::vector<std::vector<int>>& groups,
                 bool with_bending) {
        mesh = &candidate;
        bending = with_bending;
        colors = groups;
        valid = true;
    }
};

// Mesh gather only. The separate SIMD kernels receive contiguous AoS records.
static void gather_color_triangles(
    const RefMesh& mesh, const std::vector<Vec3>& x, const TriangleStorage& storage,
    std::size_t begin, std::size_t count, Vec3* positions) {
    for (std::size_t e = begin; e < begin + count; ++e) {
        const std::size_t base = 3 * static_cast<std::size_t>(storage.triangle_indices[e]);
        for (int corner = 0; corner < 3; ++corner)
            positions[3 * (e-begin) + corner] = x[mesh.tris[base + corner]];
    }
}

static void gather_color_hinges(
    const RefMesh& mesh, const std::vector<Vec3>& x, const HingeStorage& storage,
    std::size_t begin, std::size_t count, Vec3* positions) {
    for (std::size_t e = begin; e < begin + count; ++e) {
        const auto& hinge = mesh.hinges[storage.hinge_indices[e]];
        for (int corner = 0; corner < 4; ++corner)
            positions[4 * (e-begin) + corner] = x[hinge.v[corner]];
    }
}

SolverResult global_gauss_seidel_solver_basic_experimental_v2(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        const std::vector<Vec3>& v,
                                        BroadPhase& broad_phase,
                                        const std::string& outdir,
                                        const std::vector<Vec3>* previous_positions) {

    //create node (blue) boxes and create broad phase (red boxes) accordingly
    validate_solver_friction_parameters(
        params, "global_gauss_seidel_solver_basic_experimental_v2");
    std::vector<Vec3> reconstructed_previous_positions;
    previous_positions = resolve_experimental_friction_previous_positions(
        params, xnew, xhat, v, previous_positions,
        reconstructed_previous_positions,
        "global_gauss_seidel_solver_basic_experimental_v2");
    const int nv = static_cast<int>(xnew.size());
    static ExperimentalSolverWorkspace workspace;
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
    const double dt2 = params.dt2();
    #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
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
    const bool needs_mesh_contact_search =
        params.d_hat > 0.0 || params.use_ccd || params.use_ogc;
    std::vector<std::vector<int>>& color_groups = needs_mesh_contact_search
        ? workspace.color_groups : workspace.elastic_color_groups;
    const auto compute_residual = [&]() {
        build_frozen_residual_workspace(
            ref_mesh, params, xnew, broad_phase,
            workspace.frozen_residual, &workspace.rest_shape_grads);
        return compute_global_deformable_residual(ref_mesh, adj, pins, params, xnew, xhat, broad_phase, workspace.deformable_nodes, &pm, &workspace.incident_triangles, &workspace.rest_shape_grads, &workspace.frozen_residual, previous_positions);
    };

    SolverResult result;
    // anchor for clip boxes and prev_disp
    std::vector<Vec3>& xnew_substep_start = workspace.xnew_substep_start;
    #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
    for (int vi = 0; vi < nv; ++vi) xnew_substep_start[vi] = xnew[vi];

    solver_detail::ColoredContactSweep contact_sweep;
    solver_detail::ColoredVertexSweep colored_vertex_sweep;
    const bool use_contact_sweep = params.use_parallel && omp_get_max_threads() > 1
        && params.friction_coefficient == 0.0 && params.d_hat > 0.0
        && !params.use_ogc;
    // SIMD gathers private AoS batches with or without contacts. Coloring
    // keeps each batch's inputs fixed while independent vertices update.
    // The scalar v2 path retains its collective color prepass.
    // Retain capacity across substeps. Static mappings/rest data are refreshed
    // when their key changes; dynamic positions/derivatives refresh on use.
    static std::vector<TriangleStorage> color_triangle_storage;
    static std::vector<ColorTriangleDerivatives> color_triangle_derivatives;
    const bool use_v2_simd = params.use_parallel && physics_detail::energy_simd_enabled(params);
    // Only clipped cloth vertices support a proof for the entire node-box block.
    static SimdBoxContactCertificates box_certificates;
    box_certificates.valid = false;
    const bool use_box_certificates = physics_detail::energy_simd_enabled(params)
        && ref_mesh.rb_nodes.empty() && ref_mesh.tets.empty() && !params.use_ogc
        && params.use_ccd && !params.use_ticcd && params.d_hat > 1e-8
        && std::isfinite(params.d_hat) && params.k_barrier >= 0.0 && std::isfinite(params.k_barrier)
        && dt > 0.0 && std::isfinite(dt) && std::isfinite(dt2)
        && std::isfinite(dt2 * params.k_barrier);
    const auto* contact_certificates = use_box_certificates ? &box_certificates : nullptr;
    static std::vector<std::size_t> vertex_color, vertex_slot;
    vertex_color.resize(use_v2_simd ? nv : 0);
    vertex_slot.resize(use_v2_simd ? nv : 0);
    static SimdColorStorageKey simd_storage_key;
    if (!use_v2_simd) simd_storage_key.valid = false;
    static SimdVertexMaterials simd_materials;
    if (use_v2_simd && !simd_materials.matches(ref_mesh, nv, params.kB > 0.0)) {
        simd_storage_key.valid = false;
        simd_materials.rebuild(ref_mesh, workspace.incident_triangles, nv, params.kB > 0.0, params.use_parallel);
    }
    static std::vector<HingeStorage> color_hinge_storage;
    static std::vector<physics_detail::SimdDerivativeView> vertex_elasticity_simd;
    vertex_elasticity_simd.resize(use_v2_simd ? nv : 0);
    static std::vector<physics_detail::SimdDerivativeView> vertex_point_simd;
    vertex_point_simd.resize(use_v2_simd ? nv : 0);
    static std::vector<physics_detail::SimdDerivativeView> vertex_bending;
    vertex_bending.resize(use_v2_simd && params.kB > 0.0 ? nv : 0);
    static std::vector<Vec3> color_rollback;
    color_rollback.resize(use_v2_simd ? nv : 0);
    // Scalar views refer to per-color arrays. SIMD views are rebound to a
    // worker's private results for each batch and consumed before its next batch.
    static std::vector<physics_detail::MembraneDerivativeView> vertex_membrane;
    vertex_membrane.resize(params.use_parallel && !use_v2_simd ? nv : 0);
    const bool use_sdf_simd = use_v2_simd
        && params.k_sdf > 0.0 && (!params.sdf_planes.empty() || !params.sdf_cylinders.empty() || !params.sdf_spheres.empty());
    static std::vector<physics_detail::SdfDerivatives> vertex_sdf;
    vertex_sdf.resize(use_sdf_simd ? nv : 0);
    std::atomic<bool> color_derivative_failed{false};
    std::exception_ptr color_derivative_error;
    // The scalar prepass gathers a color's inputs before updating its vertices.
    // Its implicit barrier joins the resulting derivative arrays. SIMD prepares
    // independent private batches on demand instead.
    // Catch geometry/derivative failures here: exceptions cannot escape an
    // OpenMP worker. All paths drain their barriers with updates disabled, then
    // the calling thread rethrows after the team has finished.
    const auto prepare_color_derivatives = [&](std::size_t color) noexcept {
        if (use_v2_simd) return;
        auto& storage = color_triangle_storage[color];
        auto& derivatives = color_triangle_derivatives[color];
        #pragma omp for schedule(static)
        for (std::size_t e = 0; e < storage.triangle_indices.size(); ++e) {
            if (color_derivative_failed.load(std::memory_order_relaxed)) continue;
            try {
                const std::size_t base = 3 * static_cast<std::size_t>(storage.triangle_indices[e]);
                for (int corner = 0; corner < 3; ++corner)
                    storage.positions[3 * e + corner] = xnew[ref_mesh.tris[base + corner]];

                Mat32 Ds;
                Ds.col(0) = storage.positions[3 * e + 1] - storage.positions[3 * e];
                Ds.col(1) = storage.positions[3 * e + 2] - storage.positions[3 * e];
                const Mat32 F = Ds * storage.dm_inverse[e];
                const CorotatedCache32 cache = buildCorotatedCache(F);
                const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
                Mat66 dPdF;
                dPdFCorotated32(cache, params.mu, params.lambda, dPdF);

                // The helpers read only the active corner's shape gradient.
                const int corner = storage.local_corners[e];
                ShapeGrads gradN{Vec2::Zero(), Vec2::Zero(), Vec2::Zero()};
                gradN[corner] = storage.shape_gradients[e];
                derivatives.gradients[e] = dt2 * corotated_node_gradient(
                    P, storage.areas[e], gradN, corner);
                derivatives.hessians[e] = dt2 * corotated_node_hessian(
                    dPdF, storage.areas[e], gradN, corner);
            } catch (...) {
                if (!color_derivative_failed.exchange(true, std::memory_order_relaxed))
                    color_derivative_error = std::current_exception();
            }
        }

    };
    // SIMD work owns complete vertex ranges. The gather, local transpose/
    // compute, and ordered accumulation stay separate, but independent batches
    // of one color need no whole-color handoff between these operations.
    const auto prepare_simd_batch = [&](std::size_t color, std::size_t first, std::size_t last) {
        // Per-worker AoS results live through the following vertex updates.
        static thread_local std::vector<Vec3> elasticity_g, bending_g, point_g;
        static thread_local std::vector<Mat33> elasticity_H, bending_H, point_H;
        std::array<Vec3, 4 * ipc_simd::tile_width> positions;
        const auto& group = color_groups[color];
        const std::size_t point_count = last - first;
        if (point_g.size() < point_count) point_g.resize(point_count);
        if (point_H.size() < point_count) point_H.resize(point_count);
        std::array<ipc_simd::PointInput, ipc_simd::tile_width> point_inputs;
        for (std::size_t offset = first; offset < last; offset += ipc_simd::tile_width) {
            const std::size_t count = std::min(ipc_simd::tile_width, last - offset);
            for (std::size_t e = 0; e < count; ++e) {
                const int vi = group[offset + e];
                auto& input = point_inputs[e];
                input.mass = ref_mesh.mass[vi];
                input.position = xnew[vi];
                input.predicted_position = xhat[vi];
                const int pin = pm[vi];
                if (pin >= 0) input.pin_target = pins[pin].target_position;
                else input.pin_target.reset();
            }
            ipc_simd::point_derivatives_tile(point_inputs.data(), count, params.gravity,
                params.kpin, dt2, point_g.data() + offset - first, point_H.data() + offset - first);
            for (std::size_t e = 0; e < count; ++e)
                vertex_point_simd[group[offset + e]] = {
                    point_g.data() + offset + e - first, point_H.data() + offset + e - first, 1};
        }
        const auto& storage = color_triangle_storage[color];
        const std::size_t triangle_begin = storage.node_offsets[first];
        const std::size_t triangle_end = storage.node_offsets[last];
        const std::size_t triangles = triangle_end-triangle_begin;
        if (elasticity_g.size() < triangles) elasticity_g.resize(triangles);
        if (elasticity_H.size() < triangles) elasticity_H.resize(triangles);
        const auto& triangle_material = simd_materials.triangles;
        if (last == first + 1) {
            const auto begin = triangle_material.node_offsets[group[first]];
            const auto end = triangle_material.node_offsets[group[first] + 1];
            for (auto e = begin; e < end; e += ipc_simd::tile_width) {
                const auto count = std::min(ipc_simd::tile_width, end - e);
                gather_color_triangles(ref_mesh, xnew, triangle_material, e, count, positions.data());
                ipc_simd::corotated_derivatives_tile(positions.data(), triangle_material.dm_inverse.data() + e,
                    triangle_material.areas.data() + e, triangle_material.shape_gradients.data() + e, count,
                    params.mu, params.lambda, elasticity_g.data() + e - begin, elasticity_H.data() + e - begin);
            }
        } else {
            std::array<Mat22, ipc_simd::tile_width> dm;
            std::array<double, ipc_simd::tile_width> areas;
            std::array<Vec2, ipc_simd::tile_width> shapes;
            std::size_t pending = 0, completed = 0;
            const auto flush = [&] {
                if (!pending) return;
                ipc_simd::corotated_derivatives_tile(positions.data(), dm.data(), areas.data(), shapes.data(),
                    pending, params.mu, params.lambda, elasticity_g.data() + completed,
                    elasticity_H.data() + completed);
                completed += pending;
                pending = 0;
            };
            for (std::size_t i = first; i < last; ++i) {
                auto e = triangle_material.node_offsets[group[i]];
                const auto end = triangle_material.node_offsets[group[i] + 1];
                while (e < end) {
                    const auto count = std::min(ipc_simd::tile_width - pending, end - e);
                    gather_color_triangles(ref_mesh, xnew, triangle_material, e, count, positions.data() + 3 * pending);
                    std::copy_n(triangle_material.dm_inverse.data() + e, count, dm.data() + pending);
                    std::copy_n(triangle_material.areas.data() + e, count, areas.data() + pending);
                    std::copy_n(triangle_material.shape_gradients.data() + e, count, shapes.data() + pending);
                    e += count;
                    pending += count;
                    if (pending == ipc_simd::tile_width) flush();
                }
            }
            flush();
        }
        for (std::size_t i = first; i < last; ++i) {
            const std::size_t offset = storage.node_offsets[i]-triangle_begin;
            const std::size_t count = storage.node_offsets[i+1]-storage.node_offsets[i];
            vertex_elasticity_simd[group[i]] = {count ? elasticity_g.data()+offset : nullptr,
                count ? elasticity_H.data()+offset : nullptr, count};
        }
        if (params.kB > 0.0) {
            const auto& hinges = color_hinge_storage[color];
            const std::size_t hinge_begin = hinges.node_offsets[first];
            const std::size_t hinge_end = hinges.node_offsets[last];
            const std::size_t count = hinge_end-hinge_begin;
            if (bending_g.size() < count) bending_g.resize(count);
            if (bending_H.size() < count) bending_H.resize(count);
            const auto& hinge_material = simd_materials.hinges;
            if (last == first + 1) {
                const auto begin = hinge_material.node_offsets[group[first]];
                const auto end = hinge_material.node_offsets[group[first] + 1];
                for (auto e = begin; e < end; e += ipc_simd::tile_width) {
                    const auto count = std::min(ipc_simd::tile_width, end - e);
                    gather_color_hinges(ref_mesh, xnew, hinge_material, e, count, positions.data());
                    ipc_simd::bending_derivatives_tile(positions.data(), hinge_material.active_nodes.data() + e,
                        hinge_material.coefficients.data() + e, hinge_material.rest_angles.data() + e, count,
                        params.kB, bending_g.data() + e - begin, bending_H.data() + e - begin);
                }
            } else {
                std::array<int, ipc_simd::tile_width> roles;
                std::array<double, ipc_simd::tile_width> coefficients, angles;
                std::size_t pending = 0, completed = 0;
                const auto flush = [&] {
                    if (!pending) return;
                    ipc_simd::bending_derivatives_tile(positions.data(), roles.data(), coefficients.data(),
                        angles.data(), pending, params.kB, bending_g.data() + completed, bending_H.data() + completed);
                    completed += pending;
                    pending = 0;
                };
                for (std::size_t i = first; i < last; ++i) {
                    auto e = hinge_material.node_offsets[group[i]];
                    const auto end = hinge_material.node_offsets[group[i] + 1];
                    while (e < end) {
                        const auto count = std::min(ipc_simd::tile_width - pending, end - e);
                        gather_color_hinges(ref_mesh, xnew, hinge_material, e, count, positions.data() + 4 * pending);
                        std::copy_n(hinge_material.active_nodes.data() + e, count, roles.data() + pending);
                        std::copy_n(hinge_material.coefficients.data() + e, count, coefficients.data() + pending);
                        std::copy_n(hinge_material.rest_angles.data() + e, count, angles.data() + pending);
                        e += count;
                        pending += count;
                        if (pending == ipc_simd::tile_width) flush();
                    }
                }
                flush();
            }
            for (std::size_t i = first; i < last; ++i) {
                const std::size_t offset = hinges.node_offsets[i]-hinge_begin;
                const std::size_t count = hinges.node_offsets[i+1]-hinges.node_offsets[i];
                vertex_bending[group[i]] = {count ? bending_g.data()+offset : nullptr,
                    count ? bending_H.data()+offset : nullptr, count};
            }
        }
        if (use_sdf_simd) {
            for (std::size_t offset = first; offset < last; offset += ipc_simd::tile_width) {
                const std::size_t count = std::min(ipc_simd::tile_width, last-offset);
                std::array<Vec3, ipc_simd::tile_width> positions, previous;
                std::array<physics_detail::SdfDerivatives, ipc_simd::tile_width> outputs;
                for (std::size_t e = 0; e < count; ++e) {
                    const int vi = group[offset+e];
                    positions[e] = xnew[vi];
                    if (params.friction_coefficient > 0.0) previous[e] = (*previous_positions)[vi];
                }
                physics_detail::compute_sdf_derivatives_tile(params, positions.data(),
                    params.friction_coefficient > 0.0 ? previous.data() : nullptr, count, outputs.data());
                for (std::size_t e = 0; e < count; ++e) vertex_sdf[group[offset+e]] = outputs[e];
            }
        }
    };
    const auto prepare_simd_vertex = [&](int vi) {
        const auto slot = vertex_slot[vi];
        prepare_simd_batch(vertex_color[vi], slot, slot + 1);
    };
    const auto prepared_local_derivatives = [&](int vi) {
        auto result = physics_detail::compute_local_simd_v2_derivatives(
            vi, ref_mesh, pins, params, xnew, xhat, pm, vertex_elasticity_simd[vi],
            params.kB > 0.0 ? vertex_bending[vi] : physics_detail::SimdDerivativeView{},
            &vertex_point_simd[vi]);
        if (use_sdf_simd) {
            result.first += dt2 * vertex_sdf[vi].gradient;
            result.second += dt2 * vertex_sdf[vi].hessian;
            if (params.friction_coefficient > 0.0) {
                result.first += vertex_sdf[vi].friction_gradient;
                result.second += vertex_sdf[vi].friction_hessian;
            }
        }
        return result;
    };
    double r1=0.;
    //gs loop
    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if((iter-1)%params.node_box_update_count==0){//rebuild node boxes and color accordingly
            if (params.verbose)
                std::fprintf(stderr, "  [GS] iter %d  rebuilding node boxes\n", iter);
            //create new node boxes
            #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
            for (int i = 0; i < nv; ++i) {
                const double r = node_box_size_fn(i);
                blue_boxes[i] = AABB(xnew[i] - Vec3::Constant(r), xnew[i] + Vec3::Constant(r));
            }
            if (needs_mesh_contact_search) {
                // Rebuild contact candidates, combine their dependencies with
                // elastic dependencies, and color the resulting graph.
                broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat, BroadPhase::InitializationMode::DeformableSolver);
                if (use_box_certificates)
                    rebuild_simd_box_certificates(broad_phase.cache(), xnew, params.d_hat,
                        params.use_parallel, use_contact_sweep && use_v2_simd, box_certificates);
                build_contact_adj(broad_phase.cache(), static_cast<int>(xnew.size()), bca);
                union_adjacency(ea, bca, combined_adj);
                greedy_color_conflict_graph(combined_adj, color_groups, &workspace.coloring_workspace);
                const BroadPhase::Cache& bp_cache = broad_phase.cache();
                // Vertices in one color share no dependencies, so process contact-heavy vertices first to avoid end-of-color stragglers.
                #pragma omp parallel for schedule(dynamic, 1) if(params.use_parallel && nv >= 128)
                for (int color = 0; color < static_cast<int>(color_groups.size()); ++color) {
                    auto& group = color_groups[color];
                    std::stable_sort(group.begin(), group.end(), [&](const int a, const int b) {
                        return bp_cache.vertex_nt[a].size() + bp_cache.vertex_ss[a].size()
                            > bp_cache.vertex_nt[b].size() + bp_cache.vertex_ss[b].size();
                    });
                }
            } else {
                // Collision-free solve: keep node-box step clipping, but do no
                // primitive BVH construction, pair search, or contact-aware
                // coloring. Elastic topology alone determines the schedule.
                broad_phase.initialize_node_boxes_only(blue_boxes);
                // The workspace invalidates these colors with elastic topology.
                if (color_groups.empty())
                    greedy_color_conflict_graph(ea, color_groups, &workspace.coloring_workspace);
            }
        }

        // Prepare shared storage before the color sweep's worker team starts.
        // Refresh static AoS mappings after layout or material changes,
        // including contact-cost reordering within a color.
        if (params.use_parallel && (iter - 1) % params.node_box_update_count == 0
            && (!use_v2_simd || !simd_storage_key.matches(ref_mesh, color_groups, params.kB > 0.0))) {
            // A partial rebuild must never become a reusable cache entry.
            simd_storage_key.valid = false;
            color_triangle_storage.resize(color_groups.size());
            if (!use_v2_simd) color_triangle_derivatives.resize(color_groups.size());
            if (use_v2_simd && params.kB > 0.0)
                color_hinge_storage.resize(color_groups.size());
            #pragma omp parallel for schedule(dynamic, 1) if(use_v2_simd && nv >= 128)
            for (std::size_t color = 0; color < color_groups.size(); ++color) {
                if (color_derivative_failed.load(std::memory_order_relaxed)) continue;
                try {
                    const auto& group = color_groups[color];
                    auto& storage = color_triangle_storage[color];
                    storage.node_offsets.resize(group.size() + 1);
                    storage.node_offsets[0] = 0;
                    if (use_v2_simd) {
                        const auto& triangles = simd_materials.triangles;
                        for (std::size_t i = 0; i < group.size(); ++i) {
                            const int vi = group[i];
                            storage.node_offsets[i + 1] = storage.node_offsets[i]
                                + triangles.node_offsets[vi + 1] - triangles.node_offsets[vi];
                            vertex_color[vi] = color;
                            vertex_slot[vi] = i;
                        }
                        if (params.kB > 0.0) {
                            auto& layout = color_hinge_storage[color];
                            const auto& hinges = simd_materials.hinges;
                            layout.node_offsets.resize(group.size() + 1);
                            layout.node_offsets[0] = 0;
                            for (std::size_t i = 0; i < group.size(); ++i) {
                                const int vi = group[i];
                                layout.node_offsets[i + 1] = layout.node_offsets[i]
                                    + hinges.node_offsets[vi + 1] - hinges.node_offsets[vi];
                            }
                        }
                    } else {
                        for (std::size_t i = 0; i < group.size(); ++i)
                            storage.node_offsets[i + 1] = storage.node_offsets[i]
                                + workspace.incident_triangles[group[i]].size();
                        const std::size_t count = storage.node_offsets.back();
                        storage.triangle_indices.resize(count);
                        storage.local_corners.resize(count);
                        storage.positions.resize(3 * count);
                        storage.dm_inverse.resize(count);
                        storage.areas.resize(count);
                        storage.shape_gradients.resize(count);
                        auto& derivatives = color_triangle_derivatives[color];
                        derivatives.gradients.resize(count);
                        derivatives.hessians.resize(count);
                        for (std::size_t i = 0; i < group.size(); ++i) {
                            const auto begin = storage.node_offsets[i];
                            const auto node_count = storage.node_offsets[i + 1] - begin;
                            vertex_membrane[group[i]] = {
                                node_count ? derivatives.gradients.data() + begin : nullptr,
                                node_count ? derivatives.hessians.data() + begin : nullptr,
                                node_count
                            };
                            auto e = begin;
                            for (const auto& [triangle, corner] : workspace.incident_triangles[group[i]]) {
                                storage.triangle_indices[e] = triangle;
                                storage.local_corners[e] = corner;
                                storage.dm_inverse[e] = ref_mesh.Dm_inverse[triangle];
                                storage.areas[e] = ref_mesh.area[triangle];
                                storage.shape_gradients[e] = shape_function_gradients(storage.dm_inverse[e])[corner];
                                ++e;
                            }
                        }
                    }
                } catch (...) {
                    if (!color_derivative_failed.exchange(true, std::memory_order_relaxed))
                        color_derivative_error = std::current_exception();
                }
            }
            if (color_derivative_error) std::rethrow_exception(color_derivative_error);
            if (use_v2_simd)
                simd_storage_key.capture(ref_mesh, color_groups, params.kB > 0.0);
        }

        if (use_contact_sweep && (iter - 1) % params.node_box_update_count == 0) {
            contact_sweep.prepare(color_groups, broad_phase.cache());
        }

        if (iter == 1 && !params.fixed_iters) {
          r1 = compute_residual();
          result.has_residual = true;
          result.initial_residual = r1;
          result.final_residual = r1;
          if (r1 < params.tol_rel * r1 || r1 < params.tol_abs) {
            result.converged = true;
            break;
          }
        }

        const auto proposed_position = [&](int vi,
            safe_step_detail::VertexAabbRejections* rejections,
            bool cooperative) -> Vec3 {
          if (use_v2_simd) {
            const auto local = prepared_local_derivatives(vi);
            if (params.d_hat <= 0.0) {
                const Vec3 delta = matrix3d_inverse(local.second) * local.first;
                return xnew[vi] - params.damping * delta;
            }
            return xnew[vi] - params.damping * gs_vertex_delta_live_barrier_simd(
                vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm,
                &workspace.incident_triangles[vi], &workspace.rest_shape_grads,
                previous_positions, cooperative, rejections, &local, contact_certificates);
          }
          if (params.use_parallel) {
            // The scalar prepass stores weighted contributions in incident order.
            return xnew[vi] -
                   params.damping *
                       gs_vertex_delta_live_barrier_experimental<true>(
                           vi, ref_mesh, adj, pins, params, xhat, xnew,
                           broad_phase, &pm, &workspace.incident_triangles[vi],
                           &workspace.rest_shape_grads, previous_positions,
                           cooperative, rejections, &vertex_membrane[vi]);
          }
          // Serial mode keeps vertex order and gathers each vertex's incident
          // AoS entries before invoking the same local SIMD tile kernels.
          if (physics_detail::energy_simd_enabled(params)) {
              return xnew[vi] - params.damping * gs_vertex_delta_live_barrier_simd(
                  vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm,
                  &workspace.incident_triangles[vi], &workspace.rest_shape_grads,
                  previous_positions, cooperative, rejections, nullptr, contact_certificates);
          }
          return xnew[vi] -
                 params.damping *
                     gs_vertex_delta_live_barrier_experimental(
                         vi, ref_mesh, adj, pins, params, xhat, xnew,
                         broad_phase, &pm, &workspace.incident_triangles[vi],
                         &workspace.rest_shape_grads, previous_positions,
                         cooperative, rejections);
        };
        const auto process_vertex = [&](int vi, bool cooperative = false, bool prepared = false) {
          if (color_derivative_failed.load(std::memory_order_relaxed)) return;
          const auto update = [&]() {
            if (use_v2_simd && !prepared) prepare_simd_vertex(vi);
            thread_local safe_step_detail::VertexAabbRejections scratch;
            auto* rejections = params.use_ccd && !params.use_ogc
                && params.d_hat > 1e-8 ? &scratch : nullptr;
            const Vec3 proposed = proposed_position(vi, rejections, cooperative);
            per_vertex_safe_step(broad_phase, xnew, vi, proposed,
                0.9, params.use_ogc ? false : params.use_ccd,
                params.use_ticcd, params.use_ogc, cooperative, rejections);
          };
          // Prepared batches have an enclosing catch which can roll back the
          // failed color. Standalone callbacks must never escape a worker.
          if (prepared) { update(); return; }
          try { update(); }
          catch (...) {
              if (!color_derivative_failed.exchange(true, std::memory_order_relaxed))
                  color_derivative_error = std::current_exception();
          }
        };
        const auto process_vertex_batch = [&](const std::vector<int>& group, int first, int last) {
            if (!use_v2_simd) {
                for (int i = first; i < last; ++i) process_vertex(group[i]);
                return;
            }
            if (first == last || color_derivative_failed.load(std::memory_order_relaxed)) return;
            try {
                prepare_simd_batch(vertex_color[group[first]], first, last);
                for (int i = first; i < last; ++i) process_vertex(group[i], false, true);
            } catch (...) {
                if (!color_derivative_failed.exchange(true, std::memory_order_relaxed))
                    color_derivative_error = std::current_exception();
            }
        };
        if (use_contact_sweep) {
          const auto &cache = broad_phase.cache();
          const double dh2 = params.d_hat * params.d_hat,
                       dt2k = params.dt2() * params.k_barrier;
          const auto compute = [&](int vi, int local,
                                   solver_detail::ContactContribution &value) -> unsigned {
            if (color_derivative_failed.load(std::memory_order_relaxed)) return 0u;
            bool aabb_clear=false;
            if (local == 0) {
              const auto pair = [&]() {
                if (use_v2_simd) {
                    prepare_simd_vertex(vi);
                    return prepared_local_derivatives(vi);
                }
                return physics_detail::compute_local_gradient_and_hessian_with_stored_membrane_unchecked(
                    vi, ref_mesh, adj, pins, params, xnew, xhat, &pm,
                    &workspace.incident_triangles[vi], &workspace.rest_shape_grads,
                    previous_positions, vertex_membrane[vi]);
              }();
              value.gradient = pair.first;
              value.hessian = pair.second;
              return 1;
            }
            --local;
            int nt = cache.vertex_nt[vi].size();
            if (local < nt) {
              const auto &entry = cache.vertex_nt[vi][local];
              const auto &p = cache.nt_pairs[entry.pair_index];
              if (!node_triangle_aabbs_within_distance(
                      xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]],
                      xnew[p.tri_v[2]], dh2, &aabb_clear)) {
                return aabb_clear?2u:0u;
              }
              auto pair = node_triangle_barrier_self_gradient_and_hessian(
                  xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]],
                  xnew[p.tri_v[2]], params.d_hat, entry.dof);
              value.gradient = pair.first;
              value.hessian = pair.second;
            } else {
              const auto &entry = cache.vertex_ss[vi][local - nt];
              const auto &p = cache.ss_pairs[entry.pair_index];
              if (!segment_aabbs_within_distance(xnew[p.v[0]], xnew[p.v[1]],
                                                 xnew[p.v[2]], xnew[p.v[3]],
                                                 dh2, &aabb_clear)) {
                return aabb_clear?2u:0u;
              }
              auto pair = segment_segment_barrier_self_gradient_and_hessian(
                  xnew[p.v[0]], xnew[p.v[1]], xnew[p.v[2]], xnew[p.v[3]],
                  params.d_hat, entry.dof);
              value.gradient = pair.first;
              value.hessian = pair.second;
            }
            return 1;
          };
          const auto compute_assigned = [&](const solver_detail::ColoredContactSweep::Assignment& assignment,
              solver_detail::ContactContribution* values, solver_detail::ContactMaskWord* masks) noexcept {
            constexpr int grain = solver_detail::contact_grain;
            const auto clear_owned_masks = [&] {
                for (int start = assignment.lane * grain; start < assignment.count;
                     start += assignment.lanes * grain)
                    masks[start / grain] = {};
            };
            if (color_derivative_failed.load(std::memory_order_relaxed)) {
                clear_owned_masks();
                return;
            }
            try {
                if (!use_v2_simd) {
                    for (int start = assignment.lane * grain; start < assignment.count;
                         start += assignment.lanes * grain) {
                        unsigned bits = 0, clear = 0;
                        for (int j = start; j < std::min(start + grain, assignment.count); ++j) {
                            const unsigned flags = compute(assignment.vertex, j, values[j]);
                            bits |= (flags & 1u) << (j - start);
                            clear |= ((flags >> 1) & 1u) << (j - start);
                        }
                        masks[start / grain] = {bits, clear};
                    }
                    return;
                }
                // Keep the scalar scheduler's fine-grained worker ownership.
                // Gather survivors across that worker's strided groups into
                // private AoS tiles, then restore each result's original index.
                std::array<ipc_simd::MeshContactInput, ipc_simd::contact_tile_width> inputs;
                std::array<ipc_simd::MeshContactOutput, ipc_simd::contact_tile_width> outputs;
                std::array<int, ipc_simd::contact_tile_width> indices;
                std::array<unsigned char, ipc_simd::contact_tile_width> derivative_active;
                const bool use_derivative_mask = std::isfinite(dt2k);
                int assembling_word = -1;
                unsigned inactive_bits = 0;
                std::size_t active = 0;
                const auto flush = [&] {
                    if (!active) return;
                    ipc_simd::mesh_contact_derivatives_tile(inputs.data(), active,
                        params.d_hat, params.k_barrier, params.friction_coefficient,
                        params.dt(), params.friction_velocity_epsilon, outputs.data(),
                        use_derivative_mask ? derivative_active.data() : nullptr);
                    for (std::size_t e = 0; e < active; ++e) {
                        if (!use_derivative_mask || derivative_active[e]) {
                            values[indices[e]].gradient = outputs[e].gradient;
                            values[indices[e]].hessian = outputs[e].hessian;
                        } else {
                            const int word = indices[e] / grain;
                            const unsigned bit = 1u << (indices[e] % grain);
                            if (word == assembling_word) inactive_bits |= bit;
                            else masks[word].bits &= ~bit;
                        }
                    }
                    active = 0;
                };
                for (int start = assignment.lane * grain; start < assignment.count;
                     start += assignment.lanes * grain) {
                    if (color_derivative_failed.load(std::memory_order_relaxed)) {
                        clear_owned_masks();
                        return;
                    }
                    assembling_word = start / grain;
                    inactive_bits = 0;
                    unsigned bits = 0;
                    unsigned clear = contact_certificates && contact_certificates->valid
                        ? contact_certificates->vertex_clear_words[assignment.vertex][start / grain] : 0u;
                    unsigned pending = ((1u << std::min(grain, assignment.count - start)) - 1u) & ~clear;
                    while (pending) {
                        const int j = start + __builtin_ctz(pending);
                        pending &= pending - 1u;
                        unsigned flags;
                        if (j == 0) {
                            flags = compute(assignment.vertex, j, values[j]);
                        } else {
                            // The word mask already removed whole-box rejections.
                            flags = gather_simd_contact(assignment.vertex, j - 1,
                                cache, params, xnew, previous_positions, inputs[active]);
                            if (flags & 1u) {
                                indices[active++] = j;
                                if (active == ipc_simd::contact_tile_width) flush();
                            }
                        }
                        bits |= (flags & 1u) << (j - start);
                        clear |= ((flags >> 1) & 1u) << (j - start);
                    }
                    // A tile can finish while this word is still being built.
                    // Preserve its inactive bits without changing CCD evidence.
                    masks[start / grain] = {bits & ~inactive_bits, clear};
                    assembling_word = -1;
                }
                flush();
            } catch (...) {
                clear_owned_masks();
                if (!color_derivative_failed.exchange(true, std::memory_order_relaxed))
                    color_derivative_error = std::current_exception();
            }
          };
          const auto apply =
              [&](int vi, const solver_detail::ContactContribution *values, const solver_detail::ContactMaskWord* mask) {
                if (color_derivative_failed.load(std::memory_order_relaxed)) return;
                Vec3 g = values[0].gradient;
                Mat33 H = values[0].hessian;
                int count =
                    cache.vertex_nt[vi].size() + cache.vertex_ss[vi].size();
                const auto add=[&](int j){g+=dt2k*values[j].gradient;H+=dt2k*values[j].hessian;};
                solver_detail::for_active_contact(mask,count,add);
                const Vec3 delta = matrix3d_inverse(H) * g;
                const Vec3 proposed = xnew[vi] - params.damping * delta;
                {
                  const auto &box = cache.node_boxes[vi];
                  const Vec3 lo = (box.min + Vec3::Constant(1e-10)).eval();
                  const Vec3 hi = (box.max - Vec3::Constant(1e-10)).eval();
                  const Vec3 next = proposed.cwiseMax(lo).cwiseMin(hi);
                  contact_sweep.steps[vi] = next - xnew[vi];
                  contact_sweep.nonzero_step[vi] =
                      !(contact_sweep.steps[vi].squaredNorm() < 1e-28);
                  contact_sweep.short_step[vi]=params.d_hat>1e-8 && std::isfinite(dh2) && contact_sweep.steps[vi].squaredNorm()<dh2/16.0;
                }
              };
          const auto ccd = [&](int vi, int local,
                               solver_detail::ContactContribution &value, bool aabb_clear) -> bool {
            if (color_derivative_failed.load(std::memory_order_relaxed)) return false;
            if (!contact_sweep.nonzero_step[vi] || !params.use_ccd ||
                local == 0)
              return false;
            --local;
            int nt = cache.vertex_nt[vi].size();
            // AABB or verified finite-primitive projection gaps cannot close
            // when one vertex moves by less than d_hat/4. Projected certificates
            // are requested only for linear CCD, preserving TICCD tolerances.
            if(aabb_clear && contact_sweep.short_step[vi]) {
                return false;
            }
            CCDResult result;
            if (local < nt) {
              const auto &entry = cache.vertex_nt[vi][local];
              // Whole-box separation covers every finite clipped step.
              if (contact_certificates && contact_certificates->valid
                  && contact_certificates->node_triangle[entry.pair_index]
                  && contact_sweep.steps[vi].allFinite()) return false;
              result = safe_step_detail::node_triangle_vertex_ccd(
                  cache.nt_pairs[entry.pair_index], entry.dof, vi, xnew,
                  contact_sweep.steps[vi], params.use_ticcd);
            } else {
              const auto &entry = cache.vertex_ss[vi][local - nt];
              if (contact_certificates && contact_certificates->valid
                  && contact_certificates->segment_segment[entry.pair_index]
                  && contact_sweep.steps[vi].allFinite()) return false;
              result = safe_step_detail::segment_segment_vertex_ccd(
                  cache.ss_pairs[entry.pair_index], entry.dof, vi, xnew,
                  contact_sweep.steps[vi], params.use_ticcd);
            }
            if(result.collision)value.toi=result.t;
            return result.collision;
          };
          const auto commit =
              [&](int vi, const solver_detail::ContactContribution *values, const solver_detail::ContactMaskWord* mask) {
                if (color_derivative_failed.load(std::memory_order_relaxed)) return;
                if (!contact_sweep.nonzero_step[vi])
                  return;
                double toi = 1.0;
                bool collision = false;
                int count =
                    cache.vertex_nt[vi].size() + cache.vertex_ss[vi].size();
                const auto consider=[&](int j){collision=true;toi=std::min(toi,values[j].toi);};
                solver_detail::for_active_contact(mask,count,consider);
                double step = collision ? 0.9 * toi : 1.0;
                xnew[vi] = xnew[vi] + step * contact_sweep.steps[vi];
              };
          if (use_v2_simd) {
              // Reuse the team while the contact set and color layout are fixed.
              const int sweeps = params.fixed_iters
                  ? std::min(params.max_global_iters - iter + 1,
                      params.node_box_update_count - (iter - 1) % params.node_box_update_count)
                  : 1;
              const auto ccd_candidates = [&](int vertex, int start, unsigned clear) {
                  if (!contact_sweep.nonzero_step[vertex] || !params.use_ccd) return 0u;
                  unsigned skipped = start == 0 ? 1u : 0u;
                  if (contact_sweep.short_step[vertex]) return ~(skipped | clear);
                  if (contact_certificates && contact_certificates->valid
                      && contact_sweep.steps[vertex].allFinite())
                      skipped |= contact_certificates->vertex_clear_words[vertex][start / solver_detail::contact_grain];
                  return ~skipped;
              };
              contact_sweep.run_assigned(color_groups, compute, apply, process_vertex, ccd,
                  commit, prepare_color_derivatives, compute_assigned, process_vertex_batch, sweeps,
                  ccd_candidates);
              iter += sweeps - 1;
          } else
              contact_sweep.run(color_groups, compute, apply, process_vertex, ccd,
                  commit, prepare_color_derivatives);
        } else if (params.use_parallel) {
          // Fixed-iteration collision-free and friction solves can reuse one
          // team until the next node-box rebuild. Keep every color barrier,
          // including the final color of each sweep, and each vertex's arithmetic.
          // Convergence-controlled solves still return after every sweep so
          // their residual checks and stopping iteration remain unchanged.
          const bool collision_free = !needs_mesh_contact_search
              && params.k_sdf == 0.0 && params.friction_coefficient == 0.0;
          const int sweeps = params.fixed_iters
              && (collision_free || params.friction_coefficient > 0.0)
              ? std::min(params.max_global_iters - iter + 1,
                         params.node_box_update_count - (iter - 1) % params.node_box_update_count)
              : 1;
          if (use_v2_simd) {
            constexpr std::size_t max_vertices_per_batch = 8;
            std::vector<std::atomic<bool>> failed_colors(color_groups.size());
            for (auto& failed : failed_colors) failed.store(false, std::memory_order_relaxed);
            #pragma omp parallel
            {
                const std::size_t team = static_cast<std::size_t>(omp_get_num_threads());
                for (int sweep = 0; sweep < sweeps; ++sweep) {
                    for (std::size_t color = 0; color < color_groups.size(); ++color) {
                        const auto& group = color_groups[color];
                        // Retain multiple requests per worker on small colors;
                        // large colors amortize dispatch without fixing owners.
                        const std::size_t vertices_per_batch = std::min(max_vertices_per_batch,
                            std::max(std::size_t(1), group.size() / (2 * team)));
                        #pragma omp for schedule(dynamic, 1)
                        for (std::size_t first = 0; first < group.size(); first += vertices_per_batch) {
                            const std::size_t last = std::min(first + vertices_per_batch, group.size());
                            // Each vertex is exclusive to this batch. Save even
                            // after a failure so the whole color can be restored.
                            for (std::size_t i = first; i < last; ++i)
                                color_rollback[group[i]] = xnew[group[i]];
                            if (color_derivative_failed.load(std::memory_order_relaxed)) continue;
                            try {
                                prepare_simd_batch(color, first, last);
                                for (std::size_t i = first; i < last; ++i)
                                    process_vertex(group[i], false, true);
                            } catch (...) {
                                failed_colors[color].store(true, std::memory_order_relaxed);
                                if (!color_derivative_failed.exchange(true, std::memory_order_relaxed))
                                    color_derivative_error = std::current_exception();
                            }
                        }
                        // Preserve v2's failed-color behavior without adding a
                        // success-path barrier between independent batches.
                        // A faster worker may already fail in the NEXT color.
                        // Only this color's flag may control its rollback join.
                        if (failed_colors[color].load(std::memory_order_relaxed)) {
                            #pragma omp for schedule(static)
                            for (std::size_t i = 0; i < group.size(); ++i)
                                xnew[group[i]] = color_rollback[group[i]];
                        }
                    }
                }
            }
          } else if (params.fixed_iters && params.friction_coefficient > 0.0) {
            colored_vertex_sweep.run(color_groups, sweeps, process_vertex,
                                     prepare_color_derivatives);
          } else {
            #pragma omp parallel
          {
            for (int sweep = 0; sweep < sweeps; ++sweep) {
              for (std::size_t color = 0; color < color_groups.size(); ++color) {
                const auto& group = color_groups[color];
                prepare_color_derivatives(color);
            #pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                  process_vertex(group[static_cast<std::size_t>(i)]);
              }
            }
          }
          }
          iter += sweeps - 1;
        } else {
          for (int vi = 0; vi < nv; ++vi)
            process_vertex(vi);
        }

        if (color_derivative_error) std::rethrow_exception(color_derivative_error);

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
    #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
    for (int i = 0; i < nv; ++i)
        prev_disp[i] = (xnew[i] - xnew_substep_start[i]).norm();

    if (params.fixed_iters) result.converged = true;

    //write substep data
    if (params.write_substeps) {
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, &color_groups);
    }

    return result;
}

// Same per-vertex numerical updates as basic; schedule independent grid cells
// instead of independent vertices, keeping each cell's vertices serial.
//
// Grid scheduling and cooperative contact work (grid_contact_scheduling.h):
// The following decisions are made at each node-box/grid rebuild, per
// conflict-free batch. Automatic dx gives one batch per occupied parity color;
// fixed dx can require multiple batches for a color. Let T be the thread count.
//
// 1. Estimate work, using candidate counts rather than measured execution time:
//      vertex_cost = 1 + node-triangle candidates + segment-segment candidates;
//      cell_cost = sum(vertex_cost); max_vertex_cost = max(vertex_cost).
//    The 1 represents the base non-barrier contribution. With no contact
//    search, cell prioritization uses vertex_cost = 1 (i.e. vertex count).
//
// 2. Sort cells in each batch by descending cell_cost. Never reorder vertices
//    within a cell: they retain ascending vertex-ID order and update serially.
//    Changing that order would change the Gauss-Seidel dependency/update order.
//
// 3. Select cells that may reserve workers for cooperative contact processing.
//    This path requires parallel execution, T > 1, zero friction, d_hat > 0,
//    and OGC disabled. Otherwise use ordinary serial-within-cell processing.
//    - Fewer than T cells: select all cells; worker budget = T.
//    - At least T cells: worker budget = floor(3*T/4), leaving the remaining
//      workers for dynamic whole-cell processing. Examine the sorted prefix,
//      selecting at most floor(budget/2) cells. Each must satisfy all of:
//        max_vertex_cost >= 128;
//        cell_cost >= 512;
//        cell_cost >= 0.75 * total_batch_cost / T.
//      Stop selecting at the FIRST failure; do not skip ahead to later cells.
//    For T = 64, the latter case has budget 48, at most 24 selected cells,
//    and at least 16 workers available for other whole cells.
//
// 4. Start with one worker per selected cell. While budget remains, consider
//    cells with max_vertex_cost >= 128 and fewer than four assigned workers.
//    Give one extra worker to the cell maximizing cell_cost / assigned_workers.
//    Repeat until the budget is exhausted or no cell can accept another worker.
//    Thus 128 is eligibility for helpers, NOT a guarantee of four workers.
//    Four means one leader plus three helpers, not four additional helpers.
//    If no helpers are allocated, fall back to dynamic whole-cell processing.
//
// 5. Execute vertices one by one within each cell, including cooperative cells:
//    - vertex_cost < 128: only the leader processes it; its helpers wait.
//    - vertex_cost >= 128: the assigned group shares contact/CCD work in chunks
//      of 16 contributions; the leader performs ordered reduction and commit.
//    Finish and join the current vertex before reading/updating the next one.
//    Unselected cells are claimed dynamically and each is handled by one worker.
//    Cell group sizes are fixed until the next rebuild; idle workers do not
//    dynamically join another active cell. A runtime OpenMP team-size mismatch
//    also falls back to ordinary whole-cell processing.
SolverResult global_gauss_seidel_solver_ambient_grid(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        const std::vector<Vec3>& v,
                                        BroadPhase& broad_phase,
                                        const std::string& outdir,
                                        const std::vector<Vec3>* previous_positions) {

    // Calling this entry point explicitly selects grid scheduling, regardless
    // of the frame driver's use_cloth_grid switch.
    SimParams grid_parameters = params;
    grid_parameters.use_cloth_grid = true;
    grid_parameters.validate_cloth_grid_parameters();
    if (!ref_mesh.rb_nodes.empty() || !ref_mesh.tet_nodes.empty()
        || !ref_mesh.tets.empty())
        throw std::invalid_argument("cloth grid is available only for the basic cloth solver");

    //create node (blue) boxes and create broad phase (red boxes) accordingly
    validate_solver_friction_parameters(
        params, "global_gauss_seidel_solver_ambient_grid");
    std::vector<Vec3> reconstructed_previous_positions;
    previous_positions = resolve_friction_previous_positions(
        params, xnew, xhat, v, previous_positions,
        reconstructed_previous_positions,
        "global_gauss_seidel_solver_ambient_grid");
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
    const bool needs_mesh_contact_search =
        params.d_hat > 0.0 || params.use_ccd || params.use_ogc;
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
    solver_detail::ClothGridSchedule grid_schedule;
    solver_detail::ClothGridContactSweep contact_sweep;
    solver_detail::ClothGridContactStepState contact_steps;
    const bool use_contact_sweep = params.use_parallel && omp_get_max_threads() > 1
        && params.friction_coefficient == 0.0 && params.d_hat > 0.0
        && !params.use_ogc;
    if (use_contact_sweep) contact_steps.resize(nv);
    std::vector<std::size_t> vertex_costs(nv, 1);

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
            if (needs_mesh_contact_search) {
                broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat,
                    BroadPhase::InitializationMode::DeformableSolver);
                build_contact_adj(broad_phase.cache(), nv, bca);
                union_adjacency(ea, bca, combined_adj);
            } else {
                broad_phase.initialize_node_boxes_only(blue_boxes);
            }
            // Cell ownership stays fixed until the next node-box rebuild.
            // Dependency-safe batches run cells in parallel. Prioritization
            // changes only cell order, never the serial vertex order inside one.
            const auto& dependencies = needs_mesh_contact_search ? combined_adj : ea;
            if (params.cloth_grid_auto_dx)
                grid_schedule.build_auto_dx(xnew, blue_boxes, dependencies, params.cloth_grid_dx);
            else
                grid_schedule.build(xnew, blue_boxes, dependencies, params.cloth_grid_dx);
            if (needs_mesh_contact_search) {
                const auto& cache = broad_phase.cache();
                for (int vi = 0; vi < nv; ++vi)
                    vertex_costs[vi] = 1 + cache.vertex_nt[vi].size()
                        + cache.vertex_ss[vi].size();
            }
            grid_schedule.prioritize_cells(vertex_costs);
            color_groups = grid_schedule.vertex_color_groups;
            if (use_contact_sweep)
                contact_sweep.prepare(grid_schedule, broad_phase.cache());
            if (params.verbose)
                std::fprintf(stderr, "  [cloth grid] dx=%.6g occupied_cells=%zu batches=%zu cooperative_cells=%zu auto_dx=%s\n",
                    grid_schedule.dx, grid_schedule.cells.size(), grid_schedule.batches.size(),
                    contact_sweep.cooperative_cells.size(), params.cloth_grid_auto_dx ? "true" : "false");
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
        if (use_contact_sweep) {
            const auto &cache = broad_phase.cache();
            const double dh2 = params.d_hat * params.d_hat,
                         dt2k = params.dt2() * params.k_barrier;
            const auto compute = [&](int vi, int local,
                                     solver_detail::ContactContribution &value) -> unsigned {
              bool aabb_clear=false;
              if (local == 0) {
                auto pair = physics_detail::
                    compute_local_gradient_and_hessian_no_barrier_unchecked(
                        vi, ref_mesh, adj, pins, params, xnew, xhat, &pm,
                        &workspace.incident_triangles[vi],
                        &workspace.rest_shape_grads, previous_positions);
                value.gradient = pair.first;
                value.hessian = pair.second;
                return 1;
              }
              --local;
              int nt = cache.vertex_nt[vi].size();
              if (local < nt) {
                const auto &entry = cache.vertex_nt[vi][local];
                const auto &p = cache.nt_pairs[entry.pair_index];
                if (!node_triangle_aabbs_within_distance(
                        xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]],
                        xnew[p.tri_v[2]], dh2, &aabb_clear)) {
                  return aabb_clear?2u:0u;
                }
                auto pair = node_triangle_barrier_self_gradient_and_hessian(
                    xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]],
                    xnew[p.tri_v[2]], params.d_hat, entry.dof);
                value.gradient = pair.first;
                value.hessian = pair.second;
              } else {
                const auto &entry = cache.vertex_ss[vi][local - nt];
                const auto &p = cache.ss_pairs[entry.pair_index];
                if (!segment_aabbs_within_distance(xnew[p.v[0]], xnew[p.v[1]],
                                                   xnew[p.v[2]], xnew[p.v[3]],
                                                   dh2, &aabb_clear)) {
                  return aabb_clear?2u:0u;
                }
                auto pair = segment_segment_barrier_self_gradient_and_hessian(
                    xnew[p.v[0]], xnew[p.v[1]], xnew[p.v[2]], xnew[p.v[3]],
                    params.d_hat, entry.dof);
                value.gradient = pair.first;
                value.hessian = pair.second;
              }
              return 1;
            };
            const auto apply =
                [&](int vi, const solver_detail::ContactContribution *values, const solver_detail::ContactMaskWord* mask) {
                  Vec3 g = values[0].gradient;
                  Mat33 H = values[0].hessian;
                  int count =
                      cache.vertex_nt[vi].size() + cache.vertex_ss[vi].size();
                  const auto add=[&](int j){g+=dt2k*values[j].gradient;H+=dt2k*values[j].hessian;};
                  solver_detail::for_active_contact(mask,count,add);
                  const Vec3 delta = matrix3d_inverse(H) * g;
                  const Vec3 proposed = xnew[vi] - params.damping * delta;
                  {
                    const auto &box = cache.node_boxes[vi];
                    const Vec3 lo = (box.min + Vec3::Constant(1e-10)).eval();
                    const Vec3 hi = (box.max - Vec3::Constant(1e-10)).eval();
                    const Vec3 next = proposed.cwiseMax(lo).cwiseMin(hi);
                    contact_steps.steps[vi] = next - xnew[vi];
                    contact_steps.nonzero_step[vi] =
                        !(contact_steps.steps[vi].squaredNorm() < 1e-28);
                    contact_steps.short_step[vi]=params.d_hat>1e-8 && std::isfinite(dh2) && contact_steps.steps[vi].squaredNorm()<dh2/16.0;
                  }
                };
            const auto ccd = [&](int vi, int local,
                                 solver_detail::ContactContribution &value, bool aabb_clear) -> bool {
              if (!contact_steps.nonzero_step[vi] || !params.use_ccd ||
                  local == 0)
                return false;
              --local;
              int nt = cache.vertex_nt[vi].size();
              // A rejected Euclidean AABB distance exceeds d_hat, so some axis
              // gap exceeds d_hat/sqrt(3). Moving one endpoint by less than
              // d_hat/4 cannot close that gap. The original swept-AABB test
              // therefore also rejects this pair; no CCD result is approximated.
              if(aabb_clear && contact_steps.short_step[vi]) {
                  return false;
              }
              CCDResult result;
              if (local < nt) {
                const auto &entry = cache.vertex_nt[vi][local];
                result = safe_step_detail::node_triangle_vertex_ccd(
                    cache.nt_pairs[entry.pair_index], entry.dof, vi, xnew,
                    contact_steps.steps[vi], params.use_ticcd);
              } else {
                const auto &entry = cache.vertex_ss[vi][local - nt];
                result = safe_step_detail::segment_segment_vertex_ccd(
                    cache.ss_pairs[entry.pair_index], entry.dof, vi, xnew,
                    contact_steps.steps[vi], params.use_ticcd);
              }
              if(result.collision)value.toi=result.t;
              return result.collision;
            };
            const auto commit =
                [&](int vi, const solver_detail::ContactContribution *values, const solver_detail::ContactMaskWord* mask) {
                  if (!contact_steps.nonzero_step[vi])
                    return;
                  double toi = 1.0;
                  bool collision = false;
                  int count =
                      cache.vertex_nt[vi].size() + cache.vertex_ss[vi].size();
                  const auto consider=[&](int j){collision=true;toi=std::min(toi,values[j].toi);};
                  solver_detail::for_active_contact(mask,count,consider);
                  double step = collision ? 0.9 * toi : 1.0;
                  xnew[vi] = xnew[vi] + step * contact_steps.steps[vi];
                };
            contact_sweep.run(grid_schedule, compute, apply, process_vertex, ccd,
                              commit);
        } else {
            grid_schedule.run(params.use_parallel, process_vertex);
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
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, &color_groups, &grid_schedule);
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
    previous_positions = resolve_experimental_friction_previous_positions(
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

    const std::vector<std::vector<int>>& color_groups = workspace.color_groups;

    if (params.write_substeps)
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, nullptr);

    auto& bp_cache = broad_phase.mutable_cache();

    std::vector<Vec3>& xnew_copy = workspace.xnew_copy;
    std::vector<double>& bounds = workspace.bounds;
    xnew_copy.resize(nv);
    bounds.resize(nv);

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        if (iter > 1) {
            #pragma omp parallel for schedule(static)
            for (int vi = 0; vi < nv; ++vi) {
                const Vec3 r = Vec3::Constant(node_box_size_fn(vi) + pad);
                bvh_node_boxes[vi] = AABB(xnew[vi] - r, xnew[vi] + r);
            }
            broad_phase.refit_boxes(bvh_node_boxes, ref_mesh, pad);
            broad_phase.refresh_pairs(ref_mesh);
        }

        xnew_copy = xnew;
        // Bounds and barrier assembly use exactly this iteration's frozen
        // geometry. Evaluate each contact once, sharing it across its vertices.
        auto& nt_distances = workspace.nt_distances;
        auto& ss_distances = workspace.ss_distances;
        nt_distances.resize(bp_cache.nt_pairs.size());
        ss_distances.resize(bp_cache.ss_pairs.size());
        #pragma omp parallel for schedule(static)
        for (std::size_t pair = 0; pair < bp_cache.nt_pairs.size(); ++pair) {
            const auto& p = bp_cache.nt_pairs[pair];
            nt_distances[pair] = node_triangle_distance(xnew_copy[p.node],
                xnew_copy[p.tri_v[0]], xnew_copy[p.tri_v[1]], xnew_copy[p.tri_v[2]]);
        }
        #pragma omp parallel for schedule(static)
        for (std::size_t pair = 0; pair < bp_cache.ss_pairs.size(); ++pair) {
            const auto& p = bp_cache.ss_pairs[pair];
            ss_distances[pair] = segment_segment_distance(xnew_copy[p.v[0]],
                xnew_copy[p.v[1]], xnew_copy[p.v[2]], xnew_copy[p.v[3]]);
        }
        #pragma omp parallel for schedule(static)
        for (int vi = 0; vi < nv; ++vi) {
            double minimum = std::numeric_limits<double>::infinity();
            for (const auto& entry : bp_cache.vertex_nt[vi]) {
                const double distance = nt_distances[entry.pair_index].distance;
                if (distance < minimum) minimum = distance;
            }
            for (const auto& entry : bp_cache.vertex_ss[vi]) {
                const double distance = ss_distances[entry.pair_index].distance;
                if (distance < minimum) minimum = distance;
            }
            double b = 0.4 * minimum;
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
                    broad_phase, &pm, &workspace.incident_triangles[vi], &workspace.rest_shape_grads,
                    previous_positions, nt_distances, ss_distances);
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
    parallel_body_setup(static_cast<int>(ref_mesh.rb_nodes.size()), ref_mesh.rb_nodes.size() >= 8, [&](int rb) {
        const Vec4 orientation = quaternion_from_angular_velocity(state.orientations[rb], omega_new[rb], dt);
        for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local) {
            const int node = ref_mesh.rb_nodes[rb][local];
            positions[node] = world_space_position(ref_mesh.ref_positions[rb][local], x_com_new[rb], orientation);
        }
    });
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
RigidEnergyDerivatives rigid_barrier_derivatives(int rb, const RefMesh& ref_mesh, const DeformedState& state, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& omega_new, const SimParams& params, double dt, RigidDerivativeMode mode, const QuaternionOmegaKinematics* supplied_kinematics = nullptr, const FrozenResidualWorkspace* frozen_workspace = nullptr, RigidEnergyDerivatives* friction_output = nullptr, bool assemble_barrier = true, bool cooperative = false, const std::function<void()>* leader_work = nullptr) {
    RigidEnergyDerivatives total;
    if (friction_output != nullptr)
        *friction_output = RigidEnergyDerivatives{};
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0) {
        if (leader_work) (*leader_work)();
        return total;
    }
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

    // A cooperative COM/rotation solve consumes one gradient and one Hessian.
    // Stage only those fields, and leave AABB-rejected records disengaged.
    // The scalar path already accumulates without inter-worker staging.
    if (cooperative && !assemble_friction
        && (mode == RigidDerivativeMode::TranslationHessian
            || mode == RigidDerivativeMode::OrientationHessian)) {
        struct BlockContribution { Vec3 gradient; Mat33 hessian; };
        const bool translation = mode == RigidDerivativeMode::TranslationHessian;
        const int nt_count = static_cast<int>(nt_pair_indices.size());
        solver_detail::ordered_contact_tasks(nt_count + static_cast<int>(ss_pair_indices.size()), true,
            [&](int i) -> std::optional<BlockContribution> {
                const bool is_nt = i < nt_count;
                const int pair_index = is_nt ? nt_pair_indices[i] : ss_pair_indices[i - nt_count];
                bool aabb_active;
                if (frozen_workspace) {
                    aabb_active = is_nt ? frozen_workspace->nt_aabb_active[pair_index] != 0
                        : frozen_workspace->ss_aabb_active[pair_index] != 0;
                } else if (is_nt) {
                    const auto& p = bp_cache.nt_pairs[pair_index];
                    aabb_active = node_triangle_aabbs_within_distance(positions[p.node],
                        positions[p.tri_v[0]], positions[p.tri_v[1]], positions[p.tri_v[2]], d_hat2);
                } else {
                    const auto& p = bp_cache.ss_pairs[pair_index];
                    aabb_active = segment_aabbs_within_distance(positions[p.v[0]], positions[p.v[1]],
                        positions[p.v[2]], positions[p.v[3]], d_hat2);
                }
                if (!aabb_active) return std::nullopt;
                RigidEnergyDerivatives derivatives;
                // Friction is disabled in this branch; its output is unused.
                const bool active = is_nt
                    ? evaluate_nt_pair(pair_index, true, derivatives, derivatives)
                    : evaluate_ss_pair(pair_index, true, derivatives, derivatives);
                if (!active) return std::nullopt;
                return BlockContribution{
                    translation ? derivatives.translation_gradient : derivatives.orientation_gradient,
                    translation ? derivatives.translation_translation_hessian : derivatives.orientation_orientation_hessian};
            }, [&](const std::optional<BlockContribution>& value) {
                if (!value) return;
                if (translation) {
                    total.translation_gradient += value->gradient;
                    total.translation_translation_hessian += value->hessian;
                } else {
                    total.orientation_gradient += value->gradient;
                    total.orientation_orientation_hessian += value->hessian;
                }
            }, leader_work);
        return total;
    }

    if (leader_work) (*leader_work)();
    struct PairDerivatives {
        RigidEnergyDerivatives barrier, friction;
        bool active = false;
    };
    const int nt_count = static_cast<int>(nt_pair_indices.size());
    solver_detail::ordered_contact_tasks(
        nt_count + static_cast<int>(ss_pair_indices.size()), cooperative,
        [&](int i) {
            PairDerivatives value;
            value.active = i < nt_count
                ? evaluate_nt_pair(nt_pair_indices[i], false, value.barrier, value.friction)
                : evaluate_ss_pair(ss_pair_indices[i - nt_count], false, value.barrier, value.friction);
            return value;
        },
        [&](const PairDerivatives& value) {
            if (!value.active) return;
            add_rigid_derivatives(total, value.barrier);
            if (friction_output) add_rigid_derivatives(*friction_output, value.friction);
        });

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
    if (params.use_parallel && num_rbs >= 8) {
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

// Inertia/SDF assembly and contact evaluation read the same fixed state.
// Cooperative updates overlap them and add the contact totals after the join.
Vec3 compute_com_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, const SimParams& params, double dt, const QuaternionOmegaKinematics* kinematics = nullptr, bool cooperative = false) {
    const Vec3& x_com_n = state.x_coms[rb];
    const Vec3& v_com_n = state.v_coms[rb];

    Vec3 gradient;
    Mat33 hessian;
    const auto compute_noncontact = [&] {
        gradient = inertia_translation_gradient(x_com_new[rb], x_com_n, v_com_n, dt, ref_mesh.total_mass[rb]);
        gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);

        hessian = inertia_translation_hessian(ref_mesh.total_mass[rb]);
        add_rigid_sdf_translation_terms(
            ref_mesh.ref_positions[rb], ref_mesh.rb_nodes[rb],
            state.deformed_positions, x_com_new[rb], state.orientations[rb],
            omega_new[rb], params, dt, gradient, hessian, kinematics);
    };
    const bool overlap_inertia = cooperative && params.friction_coefficient == 0.0;
    if (!overlap_inertia) compute_noncontact();
    const std::function<void()> leader_work = [&compute_noncontact] { compute_noncontact(); };
    RigidEnergyDerivatives friction;
    const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, bp_cache, nt_pair_indices, ss_pair_indices, node_to_rb_local, positions, omega_new, params, dt, RigidDerivativeMode::TranslationHessian, kinematics, nullptr, params.friction_coefficient != 0.0 ? &friction : nullptr, true, cooperative, overlap_inertia ? &leader_work : nullptr);
    const double barrier_scale = dt * dt * params.k_barrier;
    gradient += barrier_scale * barrier.translation_gradient;
    hessian += barrier_scale * barrier.translation_translation_hessian;
    if (params.friction_coefficient != 0.0) {
        gradient += friction.translation_gradient;
        hessian += friction.translation_translation_hessian;
    }
    return hessian.ldlt().solve(gradient);
}

Vec3 compute_omega_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<int>& node_to_rb_local, const std::vector<Vec3>& positions, const std::vector<Vec3>& x_com_new, const std::vector<Vec3>& omega_new, const SimParams& params, double dt, const QuaternionOmegaKinematics* supplied_kinematics = nullptr, const Mat33* rotation_predictor = nullptr, bool cooperative = false) {
    const Vec4& q_n = state.orientations[rb];
    const Vec3& omega_n = state.omega[rb];
    const Mat33& I_hat = ref_mesh.I_hat[rb];

    const QuaternionOmegaKinematics owned_kinematics = supplied_kinematics == nullptr ? quaternion_omega_kinematics(q_n, omega_new[rb], dt, true) : QuaternionOmegaKinematics{};
    const QuaternionOmegaKinematics& kinematics = supplied_kinematics == nullptr ? owned_kinematics : *supplied_kinematics;
    Vec3 gradient;
    Mat33 hessian;
    const auto compute_noncontact = [&] {
        const auto derivatives = inertia_rotation_gradient_hessian(omega_new[rb], q_n, omega_n, dt, I_hat, &kinematics, rotation_predictor);
        gradient = derivatives.first;
        hessian = derivatives.second;
        add_rigid_sdf_orientation_terms(
            ref_mesh.ref_positions[rb], ref_mesh.rb_nodes[rb],
            state.deformed_positions, x_com_new[rb], q_n, omega_new[rb],
            params, dt, gradient, hessian, &kinematics);
    };
    const bool overlap_inertia = cooperative && params.friction_coefficient == 0.0;
    if (!overlap_inertia) compute_noncontact();
    const std::function<void()> leader_work = [&compute_noncontact] { compute_noncontact(); };
    RigidEnergyDerivatives friction;
    const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, bp_cache, nt_pair_indices, ss_pair_indices, node_to_rb_local, positions, omega_new, params, dt, RigidDerivativeMode::OrientationHessian, &kinematics, nullptr, params.friction_coefficient != 0.0 ? &friction : nullptr, true, cooperative, overlap_inertia ? &leader_work : nullptr);
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
            #pragma omp parallel for schedule(dynamic, 1) if(ref_mesh.rb_nodes.size() >= 8)
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
    (void)params.dt2();
    #pragma omp parallel for schedule(static) if(params.use_parallel && num_rbs >= 8)
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
    parallel_body_setup(num_rbs, params.use_parallel && num_rbs >= 8, [&](int rb) {
        workspace.rotation_predictors[static_cast<std::size_t>(rb)] = rigid_rotation_predictor(state.orientations[rb], state.omega[rb], dt);
    });

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
        parallel_body_setup(num_rbs, params.use_parallel && num_rbs >= 8, [&](int rb) {
            workspace.com_box_anchors[rb] = x_com_new[rb];
            workspace.orientation_box_anchors[rb] = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_new[rb], dt));
            workspace.com_box_radii[rb] = std::clamp(box_padding * std::max(workspace.prev_com_disp[rb], dt * state.v_coms[rb].norm()), params.node_box_min, params.node_box_max);
            workspace.theta_box_radii[rb] = std::clamp(box_padding * std::max(workspace.prev_theta_disp[rb], dt * state.omega[rb].norm()), params.theta_box_min, params.theta_box_max);
        });
        build_blue_boxes_rb(workspace.com_box_anchors, workspace.orientation_box_anchors, workspace.theta_box_radii, workspace.com_box_radii, ref_mesh, workspace.blue_boxes);
        const std::vector<AABB>& cached_boxes = workspace.broad_phase.cache().node_boxes;
        bool boxes_unchanged = workspace.contact_cache_initialized && cached_boxes.size() == workspace.blue_boxes.size() && std::memcmp(&workspace.contact_cache_d_hat, &params.d_hat, sizeof(double)) == 0;
        if (boxes_unchanged) {
            #pragma omp parallel for schedule(static) reduction(&&:boxes_unchanged) if(params.use_parallel && cached_boxes.size() >= 128)
            for (std::size_t box = 0; box < cached_boxes.size(); ++box)
                boxes_unchanged = boxes_unchanged
                    && std::memcmp(cached_boxes[box].min.data(), workspace.blue_boxes[box].min.data(), 3 * sizeof(double)) == 0
                    && std::memcmp(cached_boxes[box].max.data(), workspace.blue_boxes[box].max.data(), 3 * sizeof(double)) == 0;
        }
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

        const auto process_body = [&](int rb, bool cooperative = false) {
            const RigidBodyUpdateMode update_mode =
                ref_mesh.rb_update_modes[rb];
            std::vector<Vec3>& node_positions = workspace.positions;
            const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt, true);
            if (updates_rigid_translation(update_mode)) {
                const Vec3 delta_com = params.damping * rb_solver::compute_com_update(rb, state, ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], workspace.node_to_rb_local, node_positions, x_com_new, omega_new, params, dt, &kinematics, cooperative);
                const Vec3 com_radius = Vec3::Constant(workspace.com_box_radii[rb]);
                const Vec3 com_target = (x_com_new[rb] - delta_com).cwiseMax(workspace.com_box_anchors[rb] - com_radius).cwiseMin(workspace.com_box_anchors[rb] + com_radius);
                const Vec3 proposed_com_displacement = com_target - x_com_new[rb];
                const double com_safe_step = per_rigid_body_translation_safe_step(ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], node_positions, rb, proposed_com_displacement, 0.9, cooperative);
                const Vec3 com_displacement = com_safe_step * proposed_com_displacement;
                x_com_new[rb] += com_displacement;
                translate_rigid_nodes(ref_mesh.rb_nodes[rb], com_displacement, node_positions, params.use_parallel);
            }

            Vec4 q_accepted = q_new[rb];
            if (updates_rigid_orientation(update_mode)) {
                const Vec3 delta_omega = rb_solver::compute_omega_update(rb, state, ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], workspace.node_to_rb_local, node_positions, x_com_new, omega_new, params, dt, &kinematics, &workspace.rotation_predictors[static_cast<std::size_t>(rb)], cooperative);
                const Vec4 q_current = quaternion_normalize(kinematics.orientation);
                const Vec3 omega_trial = omega_new[rb] - params.damping * delta_omega;
                const Vec4 q_target = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_trial, dt));
                const Vec4 q_bounded = bound_quaternion(workspace.orientation_box_anchors[rb], q_current, q_target, workspace.theta_box_radii[rb]);
                const double rotation_safe_step = per_rigid_body_rotation_safe_step(ref_mesh, workspace.broad_phase.cache(), workspace.body_nt_pair_indices[rb], workspace.body_ss_pair_indices[rb], node_positions, rb, x_com_new[rb], q_current, q_bounded, 0.9, cooperative);
                q_accepted = interpolate_orientation_full_arc(q_current, q_bounded, rotation_safe_step);
                q_new[rb] = q_accepted;
                omega_new[rb] = angular_velocity_from_orientation_full_arc(q_accepted, state.orientations[rb], dt);
            }

            place_rigid_nodes(ref_mesh.rb_nodes[rb], ref_mesh.ref_positions[rb], x_com_new[rb], q_accepted, node_positions, params.use_parallel);
        };

        if (params.use_parallel) {
            // Fixed solves can share a team until the next cache rebuild.
            // Residual-controlled solves still check convergence every sweep.
            const int sweeps = params.fixed_iters
                ? std::min(params.max_global_iters - iter + 1,
                    params.node_box_update_count - (iter - 1) % params.node_box_update_count)
                : 1;
            solver_detail::for_each_colored_block(workspace.color_groups,
                [&](int rb) { return workspace.body_nt_pair_indices[rb].size()
                    + workspace.body_ss_pair_indices[rb].size(); }, process_body, sweeps);
            iter += sweeps - 1;
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

    #pragma omp parallel for schedule(static) if(params.use_parallel && num_rbs >= 8)
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
        if (params.use_cloth_grid)
            return global_gauss_seidel_solver_ambient_grid(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, &state.deformed_positions);
        if (params.use_basic_experimental_v2)
            return global_gauss_seidel_solver_basic_experimental_v2(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, &state.deformed_positions);
        if (params.use_basic_experimental)
            return global_gauss_seidel_solver_basic_experimental(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, &state.deformed_positions);
        return global_gauss_seidel_solver_basic(ref_mesh, adj, pins, params, xnew, xhat, state.velocities, broad_phase, outdir, &state.deformed_positions);
    }

    rb_solver::validate_rigid_solver_state(ref_mesh, state, x_com_new, q_new, omega_new);

    // Preserve the exact rigid-only implementation and synchronize its proxy
    // positions before returning through the general API.
    if (deformable_nodes.empty() && ref_mesh.tet_nodes.empty()) {
        SolverResult result = global_gauss_seidel_solver_basic_rb(ref_mesh, state, params, x_com_new, q_new, omega_new);
        #pragma omp parallel for schedule(dynamic, 1) if(params.use_parallel && num_rbs >= 8)
        for (int rb = 0; rb < num_rbs; ++rb) {
            for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local) {
                xnew[ref_mesh.rb_nodes[rb][local]] = world_space_position(ref_mesh.ref_positions[rb][local], x_com_new[rb], q_new[rb]);
            }
        }
        return result;
    }

    #pragma omp parallel for schedule(static) if(params.use_parallel && num_rbs >= 8)
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

    static ExperimentalSolverWorkspace deformable_workspace;
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

    const double dt = params.dt();
    (void)params.dt2();

    // xnew is the single live collision configuration. Its deformable entries
    // come from the caller; overwrite only rigid proxies from generalized
    // coordinates.
    parallel_body_setup(num_rbs, params.use_parallel && num_rbs >= 8, [&](int rb) {
        const Vec4 orientation = updates_rigid_orientation(
                                     ref_mesh.rb_update_modes[rb])
            ? quaternion_normalize(quaternion_from_angular_velocity(
                  state.orientations[rb], omega_new[rb], dt))
            : state.orientations[rb];
        q_new[rb] = orientation;
        for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local) {
            xnew[ref_mesh.rb_nodes[rb][local]] = world_space_position(ref_mesh.ref_positions[rb][local], x_com_new[rb], orientation);
        }
    });

    #pragma omp parallel for schedule(static) if(params.use_parallel && nv >= 128)
    for (int node = 0; node < nv; ++node) deformable_workspace.xnew_substep_start[node] = xnew[node];
    rigid_workspace.substep_start_coms = x_com_new;
    std::vector<AABB>& blue_boxes = rigid_workspace.blue_boxes;
    parallel_body_setup(num_rbs, params.use_parallel && num_rbs >= 8, [&](int rb) {
        rigid_workspace.rotation_predictors[static_cast<std::size_t>(rb)] = rigid_rotation_predictor(state.orientations[rb], state.omega[rb], dt);
    });
    #pragma omp parallel for schedule(static) if(params.use_parallel && cloth_nodes.size() >= 128)
    for (int index = 0; index < static_cast<int>(cloth_nodes.size()); ++index) {
        const int node = cloth_nodes[index];
        deformable_workspace.inertial_disp[node] = dt * state.velocities[node].norm();
    }
    #pragma omp parallel for schedule(static) if(params.use_parallel && solid_nodes.size() >= 128)
    for (int index = 0; index < static_cast<int>(solid_nodes.size()); ++index) {
        const int node = solid_nodes[index];
        deformable_workspace.inertial_disp[node] = dt * state.velocities[node].norm();
    }
    constexpr double box_padding = 1.2;

    const auto rebuild_contact_cache = [&](int iteration) {
        // First fill deformable boxes. build_blue_boxes_rb then overwrites all
        // rigid proxy entries with spherical-cap plus COM bounds.
        #pragma omp parallel for schedule(static) if(params.use_parallel && cloth_nodes.size() >= 128)
        for (int index = 0; index < static_cast<int>(cloth_nodes.size()); ++index) {
            const int node = cloth_nodes[index];
            const double radius = std::clamp(box_padding * std::max(deformable_workspace.prev_disp[node], deformable_workspace.inertial_disp[node]),params.node_box_min, params.node_box_max);
            const Vec3 half_extent = Vec3::Constant(radius);
            blue_boxes[node] = AABB(xnew[node] - half_extent, xnew[node] + half_extent);
        }
        #pragma omp parallel for schedule(static) if(params.use_parallel && solid_nodes.size() >= 128)
        for (int index = 0; index < static_cast<int>(solid_nodes.size()); ++index) {
            const int node = solid_nodes[index];
            const double radius = std::clamp(box_padding * std::max(deformable_workspace.prev_disp[node], deformable_workspace.inertial_disp[node]),params.node_box_min, params.node_box_max);
            const Vec3 half_extent = Vec3::Constant(radius);
            blue_boxes[node] = AABB(xnew[node] - half_extent, xnew[node] + half_extent);
        }

        parallel_body_setup(num_rbs, params.use_parallel && num_rbs >= 8, [&](int rb) {
            rigid_workspace.com_box_anchors[rb] = x_com_new[rb];
            rigid_workspace.orientation_box_anchors[rb] = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_new[rb], dt));
            rigid_workspace.com_box_radii[rb] = std::clamp(box_padding * std::max(rigid_workspace.prev_com_disp[rb], dt * state.v_coms[rb].norm()), params.node_box_min, params.node_box_max);
            rigid_workspace.theta_box_radii[rb] = std::clamp(box_padding * std::max(rigid_workspace.prev_theta_disp[rb], dt * state.omega[rb].norm()), params.theta_box_min, params.theta_box_max);
        });
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

        const auto process_cloth_node = [&](const int cloth, bool cooperative = false) {
            const int node = cloth_nodes[static_cast<std::size_t>(cloth)];
            const Vec3 delta = [&]() {
                // Helper assignment follows contact work for every cloth
                // block; solver flags only select its arithmetic kernel.
                if (params.use_basic_experimental || cooperative) {
                    if (params.use_basic_experimental
                        && physics_detail::energy_simd_enabled(params))
                        return gs_vertex_delta_live_barrier_simd(node, ref_mesh, adj, pins,
                            params, xhat, xnew, broad_phase, &pin_map,
                            &deformable_workspace.incident_triangles[node],
                            &deformable_workspace.rest_shape_grads, previous_positions, cooperative);
                    return gs_vertex_delta_live_barrier_experimental(node, ref_mesh, adj, pins,
                        params, xhat, xnew, broad_phase, &pin_map,
                        &deformable_workspace.incident_triangles[node],
                        &deformable_workspace.rest_shape_grads, previous_positions, cooperative);
                }
                return gs_vertex_delta_live_barrier(node, ref_mesh, adj, pins, params,
                    xhat, xnew, broad_phase, &pin_map,
                    &deformable_workspace.incident_triangles[node],
                    &deformable_workspace.rest_shape_grads, previous_positions);
            }();
            const Vec3 proposed_position = xnew[node] - params.damping * delta;
            per_vertex_safe_step(broad_phase, xnew, node, proposed_position, 0.9, params.use_ccd, params.use_ticcd, false, cooperative);
        };

        const auto process_solid_node = [&](const int solid, bool cooperative = false) {
            const int node = solid_nodes[static_cast<std::size_t>(solid)];
            const Vec3 proposed_position = xnew[node] - params.damping * gs_solid_vertex_delta_live_barrier(node, ref_mesh, pins, params, xhat, xnew, broad_phase, mixed_adjacency_workspace.solid_node_mask, mixed_adjacency_workspace.surface_node_mask, pin_map, previous_positions, cooperative);
            per_vertex_safe_step(broad_phase, xnew, node, proposed_position, 0.9, params.use_ccd, params.use_ticcd, false, cooperative);
        };

        // COM and orientation remain one indivisible update block: all proxy
        // positions are committed before another color begins.
        const auto process_body = [&](int rb, bool cooperative = false) {
            const RigidBodyUpdateMode update_mode =
                ref_mesh.rb_update_modes[rb];
            const QuaternionOmegaKinematics kinematics = quaternion_omega_kinematics(state.orientations[rb], omega_new[rb], dt, true);
            if (updates_rigid_translation(update_mode)) {
                const Vec3 delta_com = params.damping * rb_solver::compute_com_update(rb, state, ref_mesh, broad_phase.cache(), rigid_workspace.body_nt_pair_indices[rb], rigid_workspace.body_ss_pair_indices[rb], rigid_workspace.node_to_rb_local, xnew, x_com_new, omega_new, params, dt, &kinematics, cooperative);
                const Vec3 com_radius = Vec3::Constant(rigid_workspace.com_box_radii[rb]);
                const Vec3 com_target =(x_com_new[rb] - delta_com).cwiseMax(rigid_workspace.com_box_anchors[rb] - com_radius).cwiseMin(rigid_workspace.com_box_anchors[rb] + com_radius);
                const Vec3 proposed_com_displacement = com_target - x_com_new[rb];
                const double com_safe_step = per_rigid_body_translation_safe_step(ref_mesh, broad_phase.cache(), rigid_workspace.body_nt_pair_indices[rb], rigid_workspace.body_ss_pair_indices[rb], xnew, rb, proposed_com_displacement, 0.9, cooperative);
                const Vec3 com_displacement = com_safe_step * proposed_com_displacement;
                x_com_new[rb] += com_displacement;
                translate_rigid_nodes(ref_mesh.rb_nodes[rb], com_displacement, xnew, params.use_parallel);
            }

            Vec4 q_accepted = q_new[rb];
            if (updates_rigid_orientation(update_mode)) {
                const Vec3 delta_omega = rb_solver::compute_omega_update(rb, state, ref_mesh, broad_phase.cache(), rigid_workspace.body_nt_pair_indices[rb], rigid_workspace.body_ss_pair_indices[rb], rigid_workspace.node_to_rb_local, xnew, x_com_new, omega_new, params, dt, &kinematics, &rigid_workspace.rotation_predictors[static_cast<std::size_t>(rb)], cooperative);
                const Vec4 q_current = quaternion_normalize(kinematics.orientation);
                const Vec3 omega_trial = omega_new[rb] - params.damping * delta_omega;
                const Vec4 q_target = quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega_trial, dt));
                const Vec4 q_bounded = bound_quaternion(rigid_workspace.orientation_box_anchors[rb], q_current,q_target, rigid_workspace.theta_box_radii[rb]);
                const double rotation_safe_step = per_rigid_body_rotation_safe_step(ref_mesh, broad_phase.cache(), rigid_workspace.body_nt_pair_indices[rb], rigid_workspace.body_ss_pair_indices[rb], xnew, rb, x_com_new[rb], q_current, q_bounded, 0.9, cooperative);
                q_accepted = interpolate_orientation_full_arc(q_current, q_bounded, rotation_safe_step);
                q_new[rb] = q_accepted;
                omega_new[rb] = angular_velocity_from_orientation_full_arc(q_accepted, state.orientations[rb], dt);
            }

            place_rigid_nodes(ref_mesh.rb_nodes[rb], ref_mesh.ref_positions[rb], x_com_new[rb], q_accepted, xnew, params.use_parallel);
        };

        if (params.use_parallel) {
            const int sweeps = params.fixed_iters
                ? std::min(params.max_global_iters - iteration + 1,
                    params.node_box_update_count - (iteration - 1) % params.node_box_update_count)
                : 1;
            // Block ids are [cloth nodes][solid nodes][rigid bodies]. Each
            // color is independent under cloth/tet elasticity and NT/SS
            // reads. Keep every color barrier and each body's joined proxy
            // writes while reusing the team between contact-cache rebuilds.
            solver_detail::for_each_colored_block(mixed_adjacency_workspace.color_groups,
                [&](int block) {
                    if (block >= rigid_begin) {
                        const int rb = block - rigid_begin;
                        return rigid_workspace.body_nt_pair_indices[rb].size()
                            + rigid_workspace.body_ss_pair_indices[rb].size();
                    }
                    const int node = block < solid_begin ? cloth_nodes[block]
                        : solid_nodes[block - solid_begin];
                    return broad_phase.cache().vertex_nt[node].size()
                        + broad_phase.cache().vertex_ss[node].size();
                }, [&](int block, bool cooperative) {
                    if (block < solid_begin) process_cloth_node(block, cooperative);
                    else if (block < rigid_begin) process_solid_node(block - solid_begin, cooperative);
                    else process_body(block - rigid_begin, cooperative);
                }, sweeps);
            iteration += sweeps - 1;
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

    #pragma omp parallel for schedule(static) if(params.use_parallel && cloth_nodes.size() >= 128)
    for (int index = 0; index < static_cast<int>(cloth_nodes.size()); ++index) {
        const int node = cloth_nodes[index];
        deformable_workspace.prev_disp[node] = (xnew[node] - deformable_workspace.xnew_substep_start[node]).norm();
    }
    #pragma omp parallel for schedule(static) if(params.use_parallel && solid_nodes.size() >= 128)
    for (int index = 0; index < static_cast<int>(solid_nodes.size()); ++index) {
        const int node = solid_nodes[index];
        deformable_workspace.prev_disp[node] = (xnew[node] - deformable_workspace.xnew_substep_start[node]).norm();
    }
    #pragma omp parallel for schedule(static) if(params.use_parallel && num_rbs >= 8)
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
