#include "initial_guess.h"

#include "broad_phase.h"
#include "ccd.h"
#include "parallel_helper.h"
#include "safe_step.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

std::vector<Vec3> collision_colored_ccd_initial_guess(
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& intended_displacement,
    const RefMesh& ref_mesh, const SimParams& params, int ccd_iterations,
    const CollisionColoredCCDObserver& observer,
    const CollisionColoredCCDColorObserver& color_observer) {
    if (x.size() != intended_displacement.size())
        throw std::invalid_argument("collision_colored_ccd_initial_guess: position/displacement size mismatch");
    if (ccd_iterations < 0)
        throw std::invalid_argument("collision_colored_ccd_initial_guess: CCD iterations must be nonnegative");
    if (!std::isfinite(params.d_hat) || params.d_hat < 0.0)
        throw std::invalid_argument("collision_colored_ccd_initial_guess: d_hat must be finite and nonnegative");
    if (x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("collision_colored_ccd_initial_guess: too many vertices");
    if (ref_mesh.tris.size() % 3 != 0)
        throw std::invalid_argument("collision_colored_ccd_initial_guess: triangle indices must come in triples");
    for (const int vertex : ref_mesh.tris) {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= x.size())
            throw std::out_of_range("collision_colored_ccd_initial_guess: triangle vertex is out of range");
    }
    for (std::size_t triangle = 0; triangle < ref_mesh.tris.size(); triangle += 3) {
        const int a = ref_mesh.tris[triangle];
        const int b = ref_mesh.tris[triangle + 1];
        const int c = ref_mesh.tris[triangle + 2];
        if (a == b || a == c || b == c)
            throw std::invalid_argument("collision_colored_ccd_initial_guess: triangle vertices must be distinct");
    }

    const int nv = static_cast<int>(x.size());
    std::vector<Vec3> targets(x.size());
    for (int vertex = 0; vertex < nv; ++vertex) {
        if (!x[vertex].allFinite() || !intended_displacement[vertex].allFinite())
            throw std::invalid_argument("collision_colored_ccd_initial_guess: positions and displacements must be finite");
        targets[vertex] = x[vertex] + intended_displacement[vertex];
        if (!targets[vertex].allFinite())
            throw std::invalid_argument("collision_colored_ccd_initial_guess: target overflow");
    }
    std::vector<Vec3> xnew = x;
    if ((ccd_iterations == 0 || nv == 0) && !observer) return xnew;

    std::vector<AABB> node_boxes(x.size());
    for (int vertex = 0; vertex < nv; ++vertex) {
        // Enclose the entire start-to-target segment, with 20% extra width
        // (10% at each end). Keep stationary axes wider than safe_step's 1e-10
        // inset, and leave a representable margin at large world coordinates.
        // Do not cap the box size: every remaining displacement must fit.
        for (int axis = 0; axis < 3; ++axis) {
            const double start = x[vertex][axis];
            const double target = targets[vertex][axis];
            const double scale = std::max(std::abs(start), std::abs(target));
            const double pad = std::max({1.0e-8,
                0.1 * std::abs(intended_displacement[vertex][axis]),
                64.0 * std::numeric_limits<double>::epsilon() * scale});
            node_boxes[vertex].min[axis] = std::min(start, target) - pad;
            node_boxes[vertex].max[axis] = std::max(start, target) + pad;
        }
        if (!node_boxes[vertex].min.allFinite() || !node_boxes[vertex].max.allFinite())
            throw std::invalid_argument("collision_colored_ccd_initial_guess: padded node box overflow");
    }

    BroadPhase broad_phase;
    // Refittable mode retains every vertex's contact incidence for both
    // conflict coloring and the subsequent per-vertex CCD sweeps.
    // Green primitive boxes add params.d_hat to the node-box unions, matching
    // the solver broad phase. This expands candidates, not the CCD thickness.
    broad_phase.initialize(node_boxes, ref_mesh, params.d_hat);
    std::vector<std::vector<int>> contact_adjacency, color_groups;
    build_contact_adj(broad_phase.cache(), nv, contact_adjacency);
    greedy_color_conflict_graph(contact_adjacency, color_groups);
    if (observer) observer(0, xnew, targets, broad_phase, color_groups);
    if (ccd_iterations == 0 || nv == 0) return xnew;

    // Every candidate's four vertices form a clique, so a same-color update
    // cannot write any other position read by a vertex's CCD query. The omp
    // for barrier is essential: all updates finish before the next color.
    // Keep one team alive across all colors/sweeps. Targets, boxes, pairs and
    // coloring stay fixed; only xnew changes toward the original targets.
    // Keep color and sweep errors separate: after the final color barrier, a
    // fast primary thread may enter the sweep callback while another worker
    // is still checking the color callback's status.
    std::exception_ptr color_observer_error, observer_error;
    #pragma omp parallel if(params.use_parallel)
    {
        for (int iteration = 0; iteration < ccd_iterations; ++iteration) {
            for (int color = 0; color < static_cast<int>(color_groups.size()); ++color) {
                const auto& group = color_groups[color];
                #pragma omp for schedule(dynamic, 1)
                for (int index = 0; index < static_cast<int>(group.size()); ++index) {
                    const int vertex = group[index];
                    per_vertex_safe_step(broad_phase, xnew, vertex, targets[vertex],
                        /*safety=*/0.9, /*clip_ccd=*/true, /*use_ticcd=*/false,
                        /*use_ogc=*/false, /*cooperative=*/false);
                }
                if (color_observer) {
                    // The for barrier finishes this color; the explicit
                    // barrier keeps the next color from changing the snapshot
                    // while the primary thread exports it.
                    #pragma omp master
                    {
                        try {
                            color_observer(iteration + 1, color, xnew, targets,
                                broad_phase, color_groups);
                        } catch (...) {
                            color_observer_error = std::current_exception();
                        }
                    }
                    #pragma omp barrier
                    if (color_observer_error) break;
                }
            }
            if (color_observer_error) break;
            if (observer) {
                // The last color's implicit barrier makes this a snapshot of
                // the actual completed sweep, not a replay or a partial update.
                #pragma omp master
                {
                    try {
                        observer(iteration + 1, xnew, targets, broad_phase, color_groups);
                    } catch (...) {
                        observer_error = std::current_exception();
                    }
                }
                #pragma omp barrier
                if (observer_error) break;
            }
        }
    }
    if (color_observer_error) std::rethrow_exception(color_observer_error);
    if (observer_error) std::rethrow_exception(observer_error);
    return xnew;
}

std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, BroadPhase* scratch_broad_phase) {
    const int nv = static_cast<int>(x.size());

    std::vector<Vec3> dx(nv);
    for (int i = 0; i < nv; ++i) dx[i] = xhat[i] - x[i];

    BroadPhase local_bp;
    BroadPhase& ccd_bp = scratch_broad_phase ? *scratch_broad_phase : local_bp;
    // CCD consumes the ordered candidate arrays, not solver incidence or node BVH.
    ccd_bp.build_ccd_candidates(x, dx, ref_mesh, 1.0, /*retain_solver_data=*/false);
    const auto& cache = ccd_bp.cache();

    double toi_min = 1.0;

    const int n_nt = static_cast<int>(cache.nt_pairs.size());
    #pragma omp parallel for reduction(min:toi_min) schedule(static)
    for (int i = 0; i < n_nt; ++i) {
        const auto& p = cache.nt_pairs[i];
        toi_min = std::min(toi_min, node_triangle_general_ccd(
            x[p.node],     dx[p.node],
            x[p.tri_v[0]], dx[p.tri_v[0]],
            x[p.tri_v[1]], dx[p.tri_v[1]],
            x[p.tri_v[2]], dx[p.tri_v[2]]));
    }

    const int n_ss = static_cast<int>(cache.ss_pairs.size());
    #pragma omp parallel for reduction(min:toi_min) schedule(static)
    for (int i = 0; i < n_ss; ++i) {
        const auto& p = cache.ss_pairs[i];
        toi_min = std::min(toi_min, segment_segment_general_ccd(
            x[p.v[0]], dx[p.v[0]],
            x[p.v[1]], dx[p.v[1]],
            x[p.v[2]], dx[p.v[2]],
            x[p.v[3]], dx[p.v[3]]));
    }

    const double omega = (toi_min >= 1.0) ? 1.0 : 0.9 * toi_min;

    std::vector<Vec3> xnew(nv);
    for (int i = 0; i < nv; ++i) xnew[i] = x[i] + omega * dx[i];

    return xnew;
}

std::vector<Vec3> verlet_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const SimParams& params, BroadPhase* scratch_broad_phase) {
    const Vec3 dt2g = params.dt2() * params.gravity;
    std::vector<Vec3> xverlet(xhat.size());
    for (int i = 0; i < static_cast<int>(xhat.size()); ++i) xverlet[i] = xhat[i] + dt2g;
    return ccd_initial_guess(x, xverlet, ref_mesh, scratch_broad_phase);
}

namespace {

bool translation_guess_sdf_min_evaluation(const SimParams& params, const Vec3& xi, SDFEvaluation& out) {
    bool any = false;
    out.phi = std::numeric_limits<double>::infinity();
    for (const PlaneSDF& p : params.sdf_planes) {
        const SDFEvaluation s = evaluate_sdf(p, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    for (const CylinderSDF& c : params.sdf_cylinders) {
        const SDFEvaluation s = evaluate_sdf(c, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    for (const SphereSDF& sp : params.sdf_spheres) {
        const SDFEvaluation s = evaluate_sdf(sp, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    return any;
}

}  // namespace

std::vector<Vec3> translation_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params) {
    std::vector<Vec3> xnew(xhat.size());
    const double dt2 = params.dt2();
    double total_mass = 0.0;
    for (double m: ref_mesh.mass) total_mass += m;

    Vec3 rhs = Vec3::Zero();
    for(int i = 0; i < (int)xhat.size(); ++i){
        rhs += ref_mesh.mass[i] * (xhat[i] - x[i]);
    }
    rhs += dt2 * total_mass * params.gravity;

    double denom = total_mass;
    if (params.kpin > 0.0) {
        for (const Pin& pin : pins) {
            rhs += dt2 * params.kpin * (pin.target_position - x[pin.vertex_index]);
            denom += dt2 * params.kpin;
        }
    }

    Vec3 C = Vec3::Zero();
    if (denom > 0.0) C = rhs / denom;

    if (params.k_sdf > 0.0) {
        Vec3 G = Vec3::Zero();
        Mat33 H = denom * Mat33::Identity();

        for(int i = 0; i < (int)xhat.size(); ++i){
            SDFEvaluation s;
            if (translation_guess_sdf_min_evaluation(params, x[i] + C, s)) {
                G += dt2 * sdf_penalty_gradient(s, params.k_sdf, params.eps_sdf);
                H += dt2 * sdf_penalty_hessian(s, params.k_sdf, params.eps_sdf, false);
            }
        }

        C -= H.ldlt().solve(G);
    }

    for(int i = 0; i < (int)xhat.size(); ++i){
        xnew[i] = x[i] + C;
    }
    return xnew;
}
