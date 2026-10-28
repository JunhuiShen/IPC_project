#include "ogc_contact.h"
#include "friction_energy.h"

#include <cmath>
#include <limits>

namespace ogc_detail {

void ContactCache::evaluate(const BroadPhase& broad_phase, const std::vector<Vec3>& positions, bool parallel, bool exact_computation_fallback) {
    const auto& cache = broad_phase.cache();
    nt_distances.resize(cache.nt_pairs.size());
    ss_distances.resize(cache.ss_pairs.size());
    #pragma omp parallel for schedule(static) if(parallel)
    for (std::size_t pair = 0; pair < cache.nt_pairs.size(); ++pair) {
        const auto points = pair_positions(cache.nt_pairs[pair], positions);
        nt_distances[pair] = ::node_triangle_distance(points[0], points[1], points[2], points[3]);
    }
    #pragma omp parallel for schedule(static) if(parallel)
    for (std::size_t pair = 0; pair < cache.ss_pairs.size(); ++pair) {
        const auto points = pair_positions(cache.ss_pairs[pair], positions);
        ss_distances[pair] = segment_segment_distance(points[0], points[1], points[2], points[3], 1.0e-12, exact_computation_fallback);
    }
}

double ContactCache::isotropic_motion_bound(const BroadPhase& broad_phase, int node, double fallback) const {
    const auto& cache = broad_phase.cache();
    double minimum = std::numeric_limits<double>::infinity();
    for (const auto& entry : cache.vertex_nt[node]) {
        const double distance = nt_distances[entry.pair_index].distance;
        if (distance < minimum) minimum = distance;
    }
    for (const auto& entry : cache.vertex_ss[node]) {
        const double distance = ss_distances[entry.pair_index].distance;
        if (distance < minimum) minimum = distance;
    }
    const double bound = 0.4 * minimum;
    return std::isfinite(bound) ? bound : fallback;
}

// Elasticity reads the live color state; contact reads its frozen snapshot.
// Entry points validate friction inputs before this solver-only fast path.
Vec3 frozen_contact_vertex_delta(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xhat, const std::vector<Vec3>& x_elastic, const std::vector<Vec3>& x_barrier, const BroadPhase& broad_phase, const PinMap* pin_map, const IncidentTriangles* incident_triangles, const std::vector<ShapeGrads>* rest_shape_grads, const std::vector<Vec3>* previous_positions, const ContactCache& contacts) {
    const auto& bp_cache = broad_phase.cache();
    const auto& nt_distances = contacts.nt_distances;
    const auto& ss_distances = contacts.ss_distances;
    auto [g, H] = physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(vi, ref_mesh, adj, pins, params, x_elastic, xhat, pin_map, incident_triangles, rest_shape_grads, previous_positions);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;
        const double d_hat2 = params.d_hat * params.d_hat;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            if (!node_triangle_aabbs_within_distance(x_barrier[p.node], x_barrier[p.tri_v[0]], x_barrier[p.tri_v[1]], x_barrier[p.tri_v[2]], d_hat2)) {
                continue;
            }
            if (params.friction_coefficient != 0.0) {
                const std::array<Vec3, 4> current_positions = pair_positions(p, x_barrier);
                const NodeTriangleContactEvaluation contact_evaluation = make_node_triangle_contact_evaluation(current_positions, params.d_hat, params.k_barrier, 1.0e-12, &nt_distances[entry.pair_index], params.exact_computation_fallback);
                const auto [bg, bH] = node_triangle_barrier_self_gradient_and_hessian(current_positions[0], current_positions[1], current_positions[2], current_positions[3], entry.dof, contact_evaluation);
                g += dt2k * bg;
                H += dt2k * bH;
                const FrozenFrictionContact contact = make_node_triangle_frozen_friction_contact(current_positions, pair_positions(p, *previous_positions), contact_evaluation, params.dt(), params.friction_velocity_epsilon);
                const auto [fg, fH] = frozen_friction_role_gradient_and_hessian(contact, entry.dof, params.friction_coefficient, params.dt2());
                g += fg;
                H += fH;
            } else {
                const auto [bg, bH] = node_triangle_barrier_self_gradient_and_hessian(x_barrier[p.node], x_barrier[p.tri_v[0]], x_barrier[p.tri_v[1]], x_barrier[p.tri_v[2]], params.d_hat, entry.dof, 1.0e-12, &nt_distances[entry.pair_index]);
                g += dt2k * bg;
                H += dt2k * bH;
            }
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            if (!segment_aabbs_within_distance(x_barrier[p.v[0]], x_barrier[p.v[1]], x_barrier[p.v[2]], x_barrier[p.v[3]], d_hat2)) {
                continue;
            }
            if (params.friction_coefficient != 0.0) {
                const std::array<Vec3, 4> current_positions = pair_positions(p, x_barrier);
                const SegmentSegmentContactEvaluation contact_evaluation = make_segment_segment_contact_evaluation(current_positions, params.d_hat, params.k_barrier, 1.0e-12, &ss_distances[entry.pair_index], params.exact_computation_fallback);
                const auto [bg, bH] = segment_segment_barrier_self_gradient_and_hessian(current_positions[0], current_positions[1], current_positions[2], current_positions[3], entry.dof, contact_evaluation);
                g += dt2k * bg;
                H += dt2k * bH;
                const FrozenFrictionContact contact = make_segment_segment_frozen_friction_contact(current_positions, pair_positions(p, *previous_positions), contact_evaluation, params.dt(), params.friction_velocity_epsilon);
                const auto [fg, fH] = frozen_friction_role_gradient_and_hessian(contact, entry.dof, params.friction_coefficient, params.dt2());
                g += fg;
                H += fH;
            } else {
                const auto [bg, bH] = segment_segment_barrier_self_gradient_and_hessian(x_barrier[p.v[0]], x_barrier[p.v[1]], x_barrier[p.v[2]], x_barrier[p.v[3]], params.d_hat, entry.dof, 1.0e-12, &ss_distances[entry.pair_index], nullptr, nullptr, params.exact_computation_fallback);
                g += dt2k * bg;
                H += dt2k * bH;
            }
        }
    }

    return matrix3d_inverse(H) * g;
}

} // namespace ogc_detail

