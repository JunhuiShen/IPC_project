#include "solver.h"
#include "IPC_math.h"
#include "parallel_helper.h"
#include "barrier_energy.h"
#include "rigid_body_ipc.h"
#include "output.h"
#include "safe_step.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

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

struct BasicSolverScratch {
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

struct OGCSolverScratch {
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

// Cheap conservative rejection for inactive barrier candidates.
// Broad-phase pairs cover each vertex's complete trust box, so many cached pairs are farther than d_hat at the current GS iterate.

// Return the squared Euclidean distance from p to the closed AABB [lo, hi].
// An axis contributes zero when p lies inside its interval; otherwise it contributes the squared gap to the nearest box face.
// Keeping the result squared avoids a square root when callers compare it with d_hat^2.
static inline double point_aabb_squared_distance(
        const Vec3& p, const Vec3& lo, const Vec3& hi) {
    double d2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        double d = 0.0;
        if (p[axis] < lo[axis]) {
            d = lo[axis] - p[axis];
        } else if (p[axis] > hi[axis]) {
            d = p[axis] - hi[axis];
        }
        d2 += d * d;
    }
    return d2;
}

static inline bool node_triangle_aabbs_within_distance(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, double distance_squared) {
    const Vec3 lo = a.cwiseMin(b).cwiseMin(c);
    const Vec3 hi = a.cwiseMax(b).cwiseMax(c);
    return point_aabb_squared_distance(p, lo, hi) <= distance_squared;
}

static inline bool segment_aabbs_within_distance(const Vec3& a0, const Vec3& a1, const Vec3& b0, const Vec3& b1, double distance_squared) {
    const Vec3 alo = a0.cwiseMin(a1);
    const Vec3 ahi = a0.cwiseMax(a1);
    const Vec3 blo = b0.cwiseMin(b1);
    const Vec3 bhi = b0.cwiseMax(b1);
    double d2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        double d = 0.0;
        if (ahi[axis] < blo[axis]) {
            d = blo[axis] - ahi[axis];
        } else if (bhi[axis] < alo[axis]) {
            d = alo[axis] - bhi[axis];
        }
        d2 += d * d;
    }
    return d2 <= distance_squared;
}

}  // namespace


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


SolverResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        const std::vector<Vec3>& v,
                                        BroadPhase& broad_phase,
                                        const std::string& outdir, bool verbose) {

    //create node (blue) boxes and create broad phase (red boxes) accordingly
    const int nv = static_cast<int>(xnew.size());
    static BasicSolverScratch scratch;
    scratch.prepare(ref_mesh, adj, nv, params.node_box_max);

    PinMap& pm = scratch.pin_map;
    for (int pi = 0; pi < static_cast<int>(pins.size()); ++pi)
        pm[pins[pi].vertex_index] = pi;
    std::vector<double>& prev_disp = scratch.prev_disp;
    constexpr double node_box_padding = 1.2;
    const double dt = params.dt();
    auto node_box_size_fn = [&](int vi) {
        const double inertial = v[vi].norm() * dt;
        return std::clamp(std::max(prev_disp[vi], inertial) * node_box_padding, params.node_box_min, params.node_box_max);
    };
    std::vector<AABB>& blue_boxes = scratch.blue_boxes;

    // Elastic adjacency depends only on mesh topology, so reuse it across GS calls.
    const std::vector<std::vector<int>>& ea = scratch.elastic_adjacency.get(ref_mesh, adj, nv);
    std::vector<std::vector<int>>& bca = scratch.contact_adjacency;
    std::vector<std::vector<int>>& combined_adj = scratch.combined_adjacency;
    std::vector<std::vector<int>>& color_groups = scratch.color_groups;

    SolverResult result;
    // anchor for clip boxes and prev_disp
    std::vector<Vec3>& xnew_substep_start = scratch.xnew_substep_start;
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
                                                     &scratch.incident_triangles[vi],
                                                     &scratch.rest_shape_grads);
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

    static OGCSolverScratch scratch;
    scratch.prepare(ref_mesh, adj, nv);

    std::vector<double>& prev_disp = scratch.prev_disp;
    if (static_cast<int>(prev_disp.size()) != nv)
        prev_disp.assign(nv, params.node_box_max);
    constexpr double node_box_padding = 1.2;
    auto node_box_size_fn = [&](int vi) { return std::clamp(prev_disp[vi] * node_box_padding, params.node_box_min, params.node_box_max); };

    SolverResult result;
    result.iterations = 0;

    BroadPhase& broad_phase = scratch.broad_phase;
    std::vector<Vec3>& xnew_substep_start = scratch.xnew_substep_start;
    // anchor for clip boxes and prev_disp
    xnew_substep_start = xnew; 
    const double pad = std::max(params.ogc_box_pad, params.d_hat);

    std::vector<AABB>& bvh_node_boxes = scratch.bvh_node_boxes;
    bvh_node_boxes.resize(nv);
    for (int i = 0; i < nv; ++i) {
        const double r = node_box_size_fn(i) + pad;
        bvh_node_boxes[i] = AABB(xnew[i] - Vec3::Constant(r), xnew[i] + Vec3::Constant(r));
    }
    broad_phase.initialize(bvh_node_boxes, ref_mesh, pad);

    // Color from elastic adjacency only since barrier pairs are handled by reading  a frozen snapshot (xnew_copy) inside each color, so they don't need to constrain the coloring
    const std::vector<std::vector<int>>& elastic_adj = scratch.elastic_adjacency.get(ref_mesh, adj, nv);
    std::vector<std::vector<int>>& color_groups = scratch.color_groups;
    greedy_color_conflict_graph(elastic_adj, color_groups);

    if (params.write_substeps)
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, nullptr);

    auto& bp_cache = broad_phase.mutable_cache();

    std::vector<Vec3>& xnew_copy = scratch.xnew_copy;
    std::vector<double>& bounds = scratch.bounds;
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
                    broad_phase, &pm, &scratch.incident_triangles[vi], &scratch.rest_shape_grads);
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


namespace rb_solver {

// Convert the triangle mesh into a unique edge list.
std::vector<std::array<int, 2>> build_unique_edges(const RefMesh& ref_mesh) {
    std::vector<std::array<int, 2>> edges;
    edges.reserve(ref_mesh.tris.size());
    for (int tri = 0; tri < num_tris(ref_mesh); ++tri) {
        const int v[3] = {tri_vertex(ref_mesh, tri, 0), tri_vertex(ref_mesh, tri, 1), tri_vertex(ref_mesh, tri, 2)};
        for (int local = 0; local < 3; ++local) {
            const int a = v[local];
            const int b = v[(local + 1) % 3];
            if (a == b)
                continue;
            edges.push_back({std::min(a, b), std::max(a, b)});
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

int owning_rb_for_node(const std::vector<int>& node_to_rb, int node) {
    if (node < 0 || node >= static_cast<int>(node_to_rb.size()))
        return -1;
    return node_to_rb[node];
}

// Map a global rigid-node index to its fixed body-space coordinate
const Vec3& rigid_node_body_space_position(int node, const RefMesh& ref_mesh) {
    const int rb = owning_rb_for_node(ref_mesh.node_to_rb, node);
    const std::vector<int>& rb_nodes = ref_mesh.rb_nodes[rb];
    const int local = static_cast<int>(std::find(rb_nodes.begin(), rb_nodes.end(), node) - rb_nodes.begin());
    return ref_mesh.ref_positions[rb][local];
}

std::vector<Vec3> construct_current_rigid_node_position(const RefMesh& ref_mesh, const DeformedState& state, const std::vector<Vec3>& x_coms, const std::vector<Vec3>& omega, double dt) {
    std::vector<Vec3> positions = state.deformed_positions;
    for (int rb = 0; rb < static_cast<int>(ref_mesh.rb_nodes.size()); ++rb) {
        for (int local = 0; local < static_cast<int>(ref_mesh.rb_nodes[rb].size()); ++local) {
            const int node = ref_mesh.rb_nodes[rb][local];
            positions[node] = world_space_position(ref_mesh.ref_positions[rb][local], x_coms[rb], state.orientations[rb], omega[rb], dt);
        }
    }
    return positions;
}

void add_rigid_derivatives(RigidEnergyDerivatives& total, const RigidEnergyDerivatives& contribution) {
    total.translation_gradient += contribution.translation_gradient;
    total.orientation_gradient += contribution.orientation_gradient;
    total.translation_translation_hessian += contribution.translation_translation_hessian;
    total.translation_orientation_hessian += contribution.translation_orientation_hessian;
    total.orientation_orientation_hessian += contribution.orientation_orientation_hessian;
}

RigidEnergyDerivatives rigid_barrier_derivatives(int rb, const RefMesh& ref_mesh, const DeformedState& state, const std::vector<std::array<int, 2>>& edges, const std::vector<Vec3>& x_coms, const std::vector<Vec3>& omega, const SimParams& params, double dt) {
    RigidEnergyDerivatives total;
    if (params.d_hat <= 0.0 || params.k_barrier <= 0.0)
        return total;

    const std::vector<Vec3> positions = construct_current_rigid_node_position(ref_mesh, state, x_coms, omega, dt);
    const int num_nodes = static_cast<int>(positions.size());

    // Node-triangle barrier derivatives involving the current rigid body
    for (int node = 0; node < num_nodes; ++node) {
        const bool node_is_current = owning_rb_for_node(ref_mesh.node_to_rb, node) == rb;
        for (int tri = 0; tri < num_tris(ref_mesh); ++tri) {
            const int v0 = tri_vertex(ref_mesh, tri, 0);
            const int v1 = tri_vertex(ref_mesh, tri, 1);
            const int v2 = tri_vertex(ref_mesh, tri, 2);
            if (node == v0 || node == v1 || node == v2)
                continue;

            const bool triangle_touches_current = owning_rb_for_node(ref_mesh.node_to_rb, v0) == rb || owning_rb_for_node(ref_mesh.node_to_rb, v1) == rb || owning_rb_for_node(ref_mesh.node_to_rb, v2) == rb;
            const bool triangle_is_current = owning_rb_for_node(ref_mesh.node_to_rb, v0) == rb && owning_rb_for_node(ref_mesh.node_to_rb, v1) == rb && owning_rb_for_node(ref_mesh.node_to_rb, v2) == rb;
            if (node_is_current && !triangle_touches_current) {
                const std::array<Vec3, 4> references = {rigid_node_body_space_position(node, ref_mesh), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
                add_rigid_derivatives(total, node_triangle_barrier_rb(positions[node], positions[v0], positions[v1], positions[v2], references, RigidBarrierSide::FirstPrimitive, state.orientations[rb], omega[rb], dt, params.d_hat));
            } else if (!node_is_current && triangle_is_current) {
                const std::array<Vec3, 4> references = {Vec3::Zero(), rigid_node_body_space_position(v0, ref_mesh), rigid_node_body_space_position(v1, ref_mesh), rigid_node_body_space_position(v2, ref_mesh)};
                add_rigid_derivatives(total, node_triangle_barrier_rb(positions[node], positions[v0], positions[v1], positions[v2], references, RigidBarrierSide::SecondPrimitive, state.orientations[rb], omega[rb], dt, params.d_hat));
            }
        }
    }

    // Segment-segment barrier derivatives involving the current rigid body
    for (int first = 0; first < static_cast<int>(edges.size()); ++first) {
        const int a0 = edges[first][0];
        const int a1 = edges[first][1];
        const bool first_touches_current = owning_rb_for_node(ref_mesh.node_to_rb, a0) == rb || owning_rb_for_node(ref_mesh.node_to_rb, a1) == rb;
        const bool first_is_current = owning_rb_for_node(ref_mesh.node_to_rb, a0) == rb && owning_rb_for_node(ref_mesh.node_to_rb, a1) == rb;
        for (int second = first + 1; second < static_cast<int>(edges.size()); ++second) {
            const int b0 = edges[second][0];
            const int b1 = edges[second][1];
            if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1)
                continue;

            const bool second_touches_current = owning_rb_for_node(ref_mesh.node_to_rb, b0) == rb || owning_rb_for_node(ref_mesh.node_to_rb, b1) == rb;
            const bool second_is_current = owning_rb_for_node(ref_mesh.node_to_rb, b0) == rb && owning_rb_for_node(ref_mesh.node_to_rb, b1) == rb;
            if (first_is_current && !second_touches_current) {
                const std::array<Vec3, 4> references = {rigid_node_body_space_position(a0, ref_mesh), rigid_node_body_space_position(a1, ref_mesh), Vec3::Zero(), Vec3::Zero()};
                add_rigid_derivatives(total, segment_segment_barrier_rb(positions[a0], positions[a1], positions[b0], positions[b1], references, RigidBarrierSide::FirstPrimitive, state.orientations[rb], omega[rb], dt, params.d_hat));
            } else if (!first_touches_current && second_is_current) {
                const std::array<Vec3, 4> references = {Vec3::Zero(), Vec3::Zero(), rigid_node_body_space_position(b0, ref_mesh), rigid_node_body_space_position(b1, ref_mesh)};
                add_rigid_derivatives(total, segment_segment_barrier_rb(positions[a0], positions[a1], positions[b0], positions[b1], references, RigidBarrierSide::SecondPrimitive, state.orientations[rb], omega[rb], dt, params.d_hat));
            }
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

void add_rigid_sdf_gradients(const std::vector<Vec3>& ref_positions, const Vec3& x_com, const Vec4& q_n, const Vec3& omega, const SimParams& params, double dt, Vec3& translation_gradient, Vec3& orientation_gradient) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    for (const Vec3& X_centered : ref_positions) {
        const Vec3 x = world_space_position(X_centered, x_com, q_n, omega, dt);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        const Vec3 gx = sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
        const Mat33 J_xomega = dx_domega(X_centered, q_n, omega, dt);
        translation_gradient += dt2 * gx;
        orientation_gradient += dt2 * J_xomega.transpose() * gx;
    }
}

void add_rigid_sdf_translation_terms(const std::vector<Vec3>& ref_positions, const Vec3& x_com, const Vec4& q_n, const Vec3& omega, const SimParams& params, double dt, Vec3& gradient, Mat33& hessian) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    for (const Vec3& X_centered : ref_positions) {
        const Vec3 x = world_space_position(X_centered, x_com, q_n, omega, dt);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        gradient += dt2 * sdf_penalty_gradient(sdf, params.k_sdf, params.eps_sdf);
        hessian += dt2 * rigid_node_translation_hessian(sdf_penalty_hessian(sdf, params.k_sdf, params.eps_sdf, false));
    }
}

void add_rigid_sdf_orientation_terms(const std::vector<Vec3>& ref_positions, const Vec3& x_com, const Vec4& q_n, const Vec3& omega, const SimParams& params, double dt, Vec3& gradient, Mat33& hessian) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    for (const Vec3& X_centered : ref_positions) {
        const Vec3 x = world_space_position(X_centered, x_com, q_n, omega, dt);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        const RigidEnergyDerivatives derivatives = sdf_penalty_derivatives_rb(sdf, X_centered, q_n, omega, dt, params.k_sdf, params.eps_sdf, false, false);
        gradient += dt2 * derivatives.orientation_gradient;
        hessian += dt2 * derivatives.orientation_orientation_hessian;
    }
}

Vec3 angular_velocity_from_orientation(const Vec4& q, const Vec4& q_n, double dt) {
    Vec4 relative = quaternion_multiply(q, quaternion_conjugate(q_n));
    relative = quaternion_normalize(relative);

    // q and -q encode the same rotation. Use the principal relative rotation
    // so the logarithm returns an angle in [0, pi].
    if (relative[0] < 0.0)
        relative = -relative;

    const Vec3 vector_part = relative.tail<3>();
    const double sin_half_angle = vector_part.norm();
    if (sin_half_angle < 1.0e-12)
        return (2.0 / dt) * vector_part;

    const double half_angle = std::atan2(sin_half_angle, relative[0]);
    return (2.0 * half_angle / (dt * sin_half_angle)) * vector_part;
}

void validate_rigid_solver_state(const RefMesh& ref_mesh, const DeformedState& state, const std::vector<Vec3>& x_coms, const std::vector<Vec4>& orientations, const std::vector<Vec3>& omega) {
    const std::size_t num_rbs = ref_mesh.total_mass.size();
    const bool valid = ref_mesh.I_hat.size() == num_rbs
        && ref_mesh.rb_nodes.size() == num_rbs
        && ref_mesh.ref_positions.size() == num_rbs
        && state.x_coms.size() == num_rbs
        && state.v_coms.size() == num_rbs
        && state.orientations.size() == num_rbs
        && state.omega.size() == num_rbs
        && x_coms.size() == num_rbs
        && orientations.size() == num_rbs
        && omega.size() == num_rbs;
    if (!valid) {
        throw std::invalid_argument("global_gauss_seidel_solver_basic_rb: inconsistent rigid-body array sizes");
    }
}

double rigid_body_unnormalized_residual(const RefMesh& ref_mesh, const DeformedState& state, const std::vector<std::array<int, 2>>& edges, const SimParams& params, const std::vector<Vec3>& x_coms, const std::vector<Vec3>& omega, double dt) {
    double residual = 0.0;
    const int num_rbs = static_cast<int>(ref_mesh.total_mass.size());
    const double barrier_scale = dt * dt * params.k_barrier;
    for (int rb = 0; rb < num_rbs; ++rb) {
        Vec3 com_gradient = inertia_translation_gradient(x_coms[rb], state.x_coms[rb], state.v_coms[rb], dt, ref_mesh.total_mass[rb]);
        com_gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);

        Vec3 orientation_gradient = inertia_rotation_gradient_hessian(omega[rb], state.orientations[rb], state.omega[rb], dt, ref_mesh.I_hat[rb]).first;
        add_rigid_sdf_gradients(ref_mesh.ref_positions[rb], x_coms[rb], state.orientations[rb], omega[rb], params, dt, com_gradient, orientation_gradient);
        const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, edges, x_coms, omega, params, dt);
        com_gradient += barrier_scale * barrier.translation_gradient;
        orientation_gradient += barrier_scale * barrier.orientation_gradient;
        residual += com_gradient.norm() + orientation_gradient.norm();
    }
    return residual;
}

Vec3 compute_com_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const std::vector<std::array<int, 2>>& edges, const std::vector<Vec3>& x_coms, const std::vector<Vec3>& omega, const SimParams& params, double dt) {
    const Vec3& x_com_n = state.x_coms[rb];
    const Vec3& v_com_n = state.v_coms[rb];

    Vec3 gradient = inertia_translation_gradient(x_coms[rb], x_com_n, v_com_n, dt, ref_mesh.total_mass[rb]);
    gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);

    Mat33 hessian = inertia_translation_hessian(ref_mesh.total_mass[rb]);
    add_rigid_sdf_translation_terms(ref_mesh.ref_positions[rb], x_coms[rb], state.orientations[rb], omega[rb], params, dt, gradient, hessian);
    const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, edges, x_coms, omega, params, dt);
    const double barrier_scale = dt * dt * params.k_barrier;
    gradient += barrier_scale * barrier.translation_gradient;
    hessian += barrier_scale * barrier.translation_translation_hessian;
    return hessian.ldlt().solve(gradient);
}

Vec3 compute_omega_update(int rb, const DeformedState& state, const RefMesh& ref_mesh, const std::vector<std::array<int, 2>>& edges, const std::vector<Vec3>& x_coms, const std::vector<Vec3>& omega, const SimParams& params, double dt) {
    const Vec4& q_n = state.orientations[rb];
    const Vec3& omega_n = state.omega[rb];
    const Mat33& I_hat = ref_mesh.I_hat[rb];

    auto [gradient, hessian] = inertia_rotation_gradient_hessian(omega[rb], q_n, omega_n, dt, I_hat);
    add_rigid_sdf_orientation_terms(ref_mesh.ref_positions[rb], x_coms[rb], q_n, omega[rb], params, dt, gradient, hessian);
    const RigidEnergyDerivatives barrier = rigid_barrier_derivatives(rb, ref_mesh, state, edges, x_coms, omega, params, dt);
    const double barrier_scale = dt * dt * params.k_barrier;
    gradient += barrier_scale * barrier.orientation_gradient;
    hessian += barrier_scale * barrier.orientation_orientation_hessian;
    return hessian.ldlt().solve(gradient);
}

} // namespace rb_solver

SolverResult global_gauss_seidel_solver_basic_rb(
    const RefMesh& ref_mesh, const DeformedState& state,
    const SimParams& params, std::vector<Vec3>& x_coms,
    std::vector<Vec4>& orientations, std::vector<Vec3>& omega,
    bool verbose) {
    rb_solver::validate_rigid_solver_state(ref_mesh, state, x_coms, orientations, omega);

    SolverResult result;
    const int num_rbs = static_cast<int>(ref_mesh.total_mass.size());
    const double dt = params.dt();
    const std::vector<std::array<int, 2>> edges = rb_solver::build_unique_edges(ref_mesh);

    // Orientation is derived from the angular-velocity solve variable. Build
    // its candidate value even if the initial residual already satisfies the
    // tolerance and no Newton update is needed.
    for (int rb = 0; rb < num_rbs; ++rb) {
        orientations[rb] = quaternion_align_sign(quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega[rb], dt)), state.orientations[rb]);
        omega[rb] = rb_solver::angular_velocity_from_orientation(orientations[rb], state.orientations[rb], dt);
    }

    double initial_residual = 0.0;

    auto residual_converged = [&](double value) {
        double tolerance = 0.0;
        if (params.tol_abs > 0.0)
            tolerance = std::max(tolerance, params.tol_abs);
        if (params.tol_rel > 0.0 && std::isfinite(initial_residual)) {
            tolerance = std::max(tolerance, params.tol_rel * initial_residual);
        }
        return value <= tolerance;
    };

    if (!params.fixed_iters) {
        initial_residual = rb_solver::rigid_body_unnormalized_residual(ref_mesh, state, edges, params, x_coms, omega, dt);
        result.has_residual = true;
        result.initial_residual = initial_residual;
        result.final_residual = initial_residual;

        if (residual_converged(initial_residual)) {
            result.converged = true;
            return result;
        }
    }

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        for (int rb = 0; rb < num_rbs; ++rb) {
            x_coms[rb] -= params.damping * rb_solver::compute_com_update(rb, state, ref_mesh, edges, x_coms, omega, params, dt);

            omega[rb] -= params.damping * rb_solver::compute_omega_update(rb, state, ref_mesh, edges, x_coms, omega, params, dt);
            orientations[rb] = quaternion_align_sign(quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega[rb], dt)), state.orientations[rb]);
            omega[rb] = rb_solver::angular_velocity_from_orientation(orientations[rb], state.orientations[rb], dt);
        }

        result.iterations = iter;
        if (!params.fixed_iters) {
            const double residual = rb_solver::rigid_body_unnormalized_residual(ref_mesh, state, edges, params, x_coms, omega, dt);
            result.final_residual = residual;
            if (verbose) {
                std::fprintf(stderr, "  [RB GS] iter %d  residual = %.6e\n", iter, residual);
            }
            if (residual_converged(residual)) {
                result.converged = true;
                break;
            }
        }
    }

    if (params.fixed_iters)
        result.converged = true;
    return result;
}
