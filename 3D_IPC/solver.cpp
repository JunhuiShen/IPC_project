#include "solver.h"
#include "IPC_math.h"
#include "parallel_helper.h"
#include "barrier_energy.h"
#include "rigid_body_ipc.h"
#include "output.h"
#include "safe_step.h"

#include <algorithm>
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


namespace {
namespace rb_solver {

bool rigid_sdf_min_evaluation(
    const SimParams& params, const Vec3& x, SDFEvaluation& result) {
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

void accumulate_rigid_sdf_terms(
    const std::vector<Vec3>& ref_positions,
    const Vec3& x_com, const Vec4& q_n, const Vec3& omega,
    const SimParams& params, double dt,
    Vec3* translation_gradient, Mat33* translation_hessian,
    Vec3* rotation_gradient, Mat33* rotation_hessian) {
    if (params.k_sdf <= 0.0)
        return;

    const double dt2 = dt * dt;
    for (const Vec3& X_centered : ref_positions) {
        const Vec3 x = world_space_position(
            X_centered, x_com, q_n, omega, dt);
        SDFEvaluation sdf;
        if (!rigid_sdf_min_evaluation(params, x, sdf))
            continue;

        if (translation_gradient != nullptr || rotation_gradient != nullptr) {
            const RigidSDFGradient sdf_gradient = sdf_penalty_gradient_rb(
                sdf, X_centered, q_n, omega, dt,
                params.k_sdf, params.eps_sdf);
            if (translation_gradient != nullptr)
                *translation_gradient += dt2 * sdf_gradient.translation;
            if (rotation_gradient != nullptr)
                *rotation_gradient += dt2 * sdf_gradient.rotation;
        }

        if (translation_hessian != nullptr || rotation_hessian != nullptr) {
            const RigidSDFHessian sdf_hessian = sdf_penalty_hessian_rb(
                sdf, X_centered, q_n, omega, dt,
                params.k_sdf, params.eps_sdf,
                /*include_sdf_curvature=*/false,
                /*include_rigid_curvature=*/false);
            if (translation_hessian != nullptr)
                *translation_hessian += dt2 * sdf_hessian.translation_translation;
            if (rotation_hessian != nullptr)
                *rotation_hessian += dt2 * sdf_hessian.rotation_rotation;
        }
    }
}

void add_rigid_sdf_translation_terms(
    const std::vector<Vec3>& ref_positions,
    const Vec3& x_com, const Vec4& q_n, const Vec3& omega,
    const SimParams& params, double dt, Vec3& gradient, Mat33& hessian) {
    accumulate_rigid_sdf_terms(
        ref_positions, x_com, q_n, omega, params, dt,
        &gradient, &hessian, nullptr, nullptr);
}

void add_rigid_sdf_rotation_terms(
    const std::vector<Vec3>& ref_positions,
    const Vec3& x_com, const Vec4& q_n, const Vec3& omega,
    const SimParams& params, double dt, Vec3& gradient, Mat33& hessian) {
    accumulate_rigid_sdf_terms(
        ref_positions, x_com, q_n, omega, params, dt,
        nullptr, nullptr, &gradient, &hessian);
}

Vec3 angular_velocity_from_orientation(
    const Vec4& q, const Vec4& q_n, double dt) {
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

void validate_rigid_solver_state(
    const RefMesh& ref_mesh, const DeformedState& state,
    const std::vector<Vec3>& x_coms,
    const std::vector<Vec4>& orientations,
    const std::vector<Vec3>& omega) {
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

double rigid_body_unnormalized_residual(
    const RefMesh& ref_mesh, const DeformedState& state,
    const SimParams& params, const std::vector<Vec3>& x_coms,
    const std::vector<Vec3>& omega, double dt) {
    double residual = 0.0;
    const int num_rbs = static_cast<int>(ref_mesh.total_mass.size());
    for (int rb = 0; rb < num_rbs; ++rb) {
        Vec3 com_gradient = inertia_translation_gradient(x_coms[rb], state.x_coms[rb], state.v_coms[rb], dt, ref_mesh.total_mass[rb]);
        com_gradient -= gravitational_potential_gradient(ref_mesh.total_mass[rb], params.gravity.y(), dt);

        Vec3 orientation_gradient = inertia_rotation_gradient_hessian(omega[rb], state.orientations[rb], state.omega[rb], dt, ref_mesh.I_hat[rb]).first;
        accumulate_rigid_sdf_terms(
            ref_mesh.ref_positions[rb], x_coms[rb], state.orientations[rb],
            omega[rb], params, dt,
            &com_gradient, nullptr, &orientation_gradient, nullptr);
        residual += com_gradient.norm() + orientation_gradient.norm();
    }
    return residual;
}

Vec3 compute_com_update(
    int rb, const DeformedState& state, const RefMesh& ref_mesh,
    const Vec3& x_com, const Vec3& omega,
    const SimParams& params, double dt) {
    const Vec3& x_com_n = state.x_coms[rb];
    const Vec3& v_com_n = state.v_coms[rb];

    Vec3 gradient = inertia_translation_gradient(
        x_com, x_com_n, v_com_n, dt, ref_mesh.total_mass[rb]);
    gradient -= gravitational_potential_gradient(
        ref_mesh.total_mass[rb], params.gravity.y(), dt);

    Mat33 hessian = inertia_translation_hessian(ref_mesh.total_mass[rb]);
    add_rigid_sdf_translation_terms(
        ref_mesh.ref_positions[rb], x_com, state.orientations[rb], omega,
        params, dt, gradient, hessian);
    return hessian.ldlt().solve(gradient);
}

Vec3 compute_omega_update(
    int rb, const DeformedState& state, const RefMesh& ref_mesh,
    const Vec3& x_com, const Vec3& omega,
    const SimParams& params, double dt) {
    const Vec4& q_n = state.orientations[rb];
    const Vec3& omega_n = state.omega[rb];
    const Mat33& I_hat = ref_mesh.I_hat[rb];

    auto [gradient, hessian] = inertia_rotation_gradient_hessian(
        omega, q_n, omega_n, dt, I_hat);
    add_rigid_sdf_rotation_terms(
        ref_mesh.ref_positions[rb], x_com, q_n, omega,
        params, dt, gradient, hessian);
    return hessian.ldlt().solve(gradient);
}

} // namespace rb_solver
} // namespace

SolverResult global_gauss_seidel_solver_basic_rb(
    const RefMesh& ref_mesh, const DeformedState& state,
    const SimParams& params, std::vector<Vec3>& x_coms,
    std::vector<Vec4>& orientations, std::vector<Vec3>& omega,
    bool verbose) {
    rb_solver::validate_rigid_solver_state(ref_mesh, state, x_coms, orientations, omega);

    SolverResult result;
    const int num_rbs = static_cast<int>(ref_mesh.total_mass.size());
    const double dt = params.dt();

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
        initial_residual = rb_solver::rigid_body_unnormalized_residual(ref_mesh, state, params, x_coms, omega, dt);
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
            x_coms[rb] -= params.damping * rb_solver::compute_com_update(
                rb, state, ref_mesh, x_coms[rb], omega[rb], params, dt);

            omega[rb] -= params.damping * rb_solver::compute_omega_update(
                rb, state, ref_mesh, x_coms[rb], omega[rb], params, dt);
            orientations[rb] = quaternion_align_sign(quaternion_normalize(quaternion_from_angular_velocity(state.orientations[rb], omega[rb], dt)), state.orientations[rb]);
            omega[rb] = rb_solver::angular_velocity_from_orientation(orientations[rb], state.orientations[rb], dt);
        }

        result.iterations = iter;
        if (!params.fixed_iters) {
            const double residual = rb_solver::rigid_body_unnormalized_residual(ref_mesh, state, params, x_coms, omega, dt);
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
