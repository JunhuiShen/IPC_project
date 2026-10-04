#include "initial_guess.h"

#include "broad_phase.h"
#include "ccd.h"
#include "node_triangle_distance.h"
#include "parallel_helper.h"
#include "safe_step.h"
#include "segment_segment_distance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// The linear CCD initial-contact tolerance is 1e-10. Repeated 0.9-TOI
// retries must not put a new iterate inside that tolerance, where even a
// separating update can return TOI=0. This is a contact gap, not a target
// offset, and is intentionally local to the colored initial guess.
static constexpr double colored_guess_min_separation = 1.0e-8;

static double guess_point_triangle_distance(
    const Vec3& point, const Vec3& a, const Vec3& b, const Vec3& c) {
    const auto result = node_triangle_distance(point, a, b, c);
    if (result.region == NodeTriangleRegion::FaceInterior)
        return result.distance;
    // Outside the face, test every finite edge: barycentric sign regions
    // alone need not select the nearest edge for an obtuse triangle.
    double t;
    return std::min({(point - segment_closest_point(point, a, b, t)).norm(),
        (point - segment_closest_point(point, b, c, t)).norm(),
        (point - segment_closest_point(point, c, a, t)).norm()});
}

template <class AcceptDistance>
static bool check_guess_contact_distances(const BroadPhase::Cache& cache,
    const std::vector<Vec3>& x, int vertex, AcceptDistance accept) {
    std::size_t index = 0;
    for (const auto& entry : cache.vertex_nt[vertex]) {
        const auto& pair = cache.nt_pairs[entry.pair_index];
        const double distance = guess_point_triangle_distance(x[pair.node],
            x[pair.tri_v[0]], x[pair.tri_v[1]], x[pair.tri_v[2]]);
        if (!std::isfinite(distance) || !accept(index++, distance)) return false;
    }
    for (const auto& entry : cache.vertex_ss[vertex]) {
        const auto& pair = cache.ss_pairs[entry.pair_index];
        const double distance = segment_segment_distance(x[pair.v[0]],
            x[pair.v[1]], x[pair.v[2]], x[pair.v[3]]).distance;
        if (!std::isfinite(distance) || !accept(index++, distance)) return false;
    }
    return true;
}

static void preserve_guess_separation(const BroadPhase::Cache& cache,
    std::vector<Vec3>& x, int vertex, const Vec3& before,
    std::vector<double>& required_distances) {
    // Leave an unobstructed/adequately separated CCD endpoint untouched.
    if (check_guess_contact_distances(cache, x, vertex,
            [](std::size_t, double distance) {
                return distance >= colored_guess_min_separation;
            })) return;

    const Vec3 accepted = x[vertex];
    const Vec3 displacement = accepted - before;
    x[vertex] = before;
    required_distances.clear();
    // A pre-existing smaller gap is not repaired or pushed apart. Preserve
    // its own value without relaxing the floor for any other contact pair.
    if (!check_guess_contact_distances(cache, x, vertex,
            [&](std::size_t, double distance) {
                required_distances.push_back(
                    std::min(colored_guess_min_separation, distance));
                return true;
            })) return;

    const auto acceptable = [&] {
        return check_guess_contact_distances(cache, x, vertex,
            [&](std::size_t index, double distance) {
                return distance >= required_distances[index];
            });
    };
    // A move away from an already small gap need not attain the full floor
    // at once. Preserve the full CCD endpoint if it meets the per-pair bounds.
    x[vertex] = accepted;
    if (acceptable()) return;

    // Keep a verified acceptable endpoint while refining the bracket, rather
    // than discarding half of an otherwise long safe move on every sweep.
    // Distances need not be globally monotone: only tested endpoints are
    // accepted, and all trials lie on the already CCD-approved segment.
    // The coloring ensures no concurrent same-color query reads this vertex.
    double lower = 0.0, upper = 1.0;
    Vec3 best = before;
    const double length = displacement.stableNorm();
    for (int attempt = 0; attempt < 48 && (upper - lower) * length > 1e-14; ++attempt) {
        const double weight = 0.5 * (lower + upper);
        x[vertex] = before + weight * displacement;
        if (acceptable()) {
            lower = weight;
            best = x[vertex];
        } else {
            upper = weight;
        }
    }
    x[vertex] = best;
}

std::vector<Vec3> collision_colored_ccd_initial_guess(
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& intended_displacement,
    const RefMesh& ref_mesh, const SimParams& params, int ccd_iterations) {
    if (x.size() != intended_displacement.size())
        throw std::invalid_argument("collision_colored_ccd_initial_guess: position/displacement size mismatch");
    if (ccd_iterations < 0)
        throw std::invalid_argument("collision_colored_ccd_initial_guess: CCD iterations must be nonnegative");
    if (!std::isfinite(params.d_hat) || params.d_hat < 0.0)
        throw std::invalid_argument("collision_colored_ccd_initial_guess: d_hat must be finite and nonnegative");
    if (x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("collision_colored_ccd_initial_guess: too many vertices");
    if ((!ref_mesh.node_to_rb.empty() && ref_mesh.node_to_rb.size() != x.size())
        || (!ref_mesh.rb_nodes.empty() && ref_mesh.node_to_rb.size() != x.size()))
        throw std::invalid_argument("collision_colored_ccd_initial_guess: inconsistent rigid-node ownership map");
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
    const auto is_rigid = [&](int vertex) {
        return !ref_mesh.node_to_rb.empty() && ref_mesh.node_to_rb[vertex] >= 0;
    };
    std::vector<Vec3> targets(x.size());
    for (int vertex = 0; vertex < nv; ++vertex) {
        if (!x[vertex].allFinite() || !intended_displacement[vertex].allFinite())
            throw std::invalid_argument("collision_colored_ccd_initial_guess: positions and displacements must be finite");
        // Rigid proxies remain collision obstacles, never independent DOFs.
        // Ignore their proposed displacement even for direct helper callers.
        if (is_rigid(vertex)) targets[vertex] = x[vertex];
        else targets[vertex] = x[vertex] + intended_displacement[vertex];
        if (!targets[vertex].allFinite())
            throw std::invalid_argument("collision_colored_ccd_initial_guess: target overflow");
    }
    std::vector<Vec3> xnew = x;
    if (ccd_iterations == 0 || nv == 0) return xnew;

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
                is_rigid(vertex) ? 0.0 : 0.1 * std::abs(intended_displacement[vertex][axis]),
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
    // Tet interiors are not contact points. Boundary triangles and edges,
    // including fixed rigid proxies, still participate in collision checks.
    if (!ref_mesh.tets.empty() || !ref_mesh.tet_nodes.empty())
        broad_phase.initialize_surface_nodes(node_boxes, ref_mesh, params.d_hat);
    else
        broad_phase.initialize(node_boxes, ref_mesh, params.d_hat);
    std::vector<std::vector<int>> contact_adjacency, color_groups;
    build_contact_adj(broad_phase.cache(), nv, contact_adjacency);
    greedy_color_conflict_graph(contact_adjacency, color_groups);

    // Every candidate's four vertices form a clique, so a same-color update
    // cannot write any other position read by a vertex's CCD query. The omp
    // for barrier is essential: all updates finish before the next color.
    // Keep one team alive across all colors/sweeps. Targets, boxes, pairs and
    // coloring stay fixed; only xnew changes toward the original targets.
    #pragma omp parallel if(params.use_parallel)
    {
        std::vector<double> required_distances; // reused, private to each worker
        for (int iteration = 0; iteration < ccd_iterations; ++iteration) {
            for (const auto& group : color_groups) {
                #pragma omp for schedule(dynamic, 1)
                for (int index = 0; index < static_cast<int>(group.size()); ++index) {
                    const int vertex = group[index];
                    if (is_rigid(vertex)) continue;
                    const Vec3 before = xnew[vertex];
                    const double step = per_vertex_safe_step(broad_phase, xnew, vertex, targets[vertex],
                        /*safety=*/0.9, /*clip_ccd=*/true, /*use_ticcd=*/false,
                        /*use_ogc=*/false, /*cooperative=*/false);
                    if (step > 0.0)
                        preserve_guess_separation(broad_phase.cache(), xnew,
                            vertex, before, required_distances);
                }
            }
        }
    }
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
