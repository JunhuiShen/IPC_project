#include "safe_step.h"

#include "broad_phase.h"
#include "ccd.h"
#include "node_triangle_distance.h"
#include "parallel_helper.h"
#include "quaternion_math.h"
#include "rigid_body_ipc.h"
#include "segment_segment_distance.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline bool node_triangle_single_vertex_swept_aabbs_intersect(const NodeTrianglePair& p, int moving_dof, const std::vector<Vec3>& x, const Vec3& dx) {
    AABB node_box;
    node_box.expand(x[p.node]);
    if (moving_dof == 0) node_box.expand(x[p.node] + dx);

    AABB tri_box;
    for (int role = 0; role < 3; ++role) {
        const Vec3& xi = x[p.tri_v[role]];
        tri_box.expand(xi);
        if (moving_dof == role + 1) tri_box.expand(xi + dx);
    }
    return aabb_intersects(node_box, tri_box);
}

inline bool segment_segment_single_vertex_swept_aabbs_intersect(const SegmentSegmentPair& p, int moving_dof, const std::vector<Vec3>& x, const Vec3& dx) {
    AABB first_box;
    AABB second_box;
    for (int role = 0; role < 4; ++role) {
        AABB& box = role < 2 ? first_box : second_box;
        const Vec3& xi = x[p.v[role]];
        box.expand(xi);
        if (moving_dof == role) box.expand(xi + dx);
    }
    return aabb_intersects(first_box, second_box);
}

inline bool node_triangle_swept_aabbs_intersect(const std::array<AABB, 4>& node_boxes) {
    AABB triangle_box;
    triangle_box.expand(node_boxes[1]);
    triangle_box.expand(node_boxes[2]);
    triangle_box.expand(node_boxes[3]);
    return aabb_intersects(node_boxes[0], triangle_box);
}

inline bool segment_segment_swept_aabbs_intersect(const std::array<AABB, 4>& node_boxes) {
    AABB first_box;
    first_box.expand(node_boxes[0]);
    first_box.expand(node_boxes[1]);
    AABB second_box;
    second_box.expand(node_boxes[2]);
    second_box.expand(node_boxes[3]);
    return aabb_intersects(first_box, second_box);
}

inline AABB translated_node_swept_aabb(int node, const std::vector<Vec3>& x, const std::vector<int>& node_to_rb, int rb, const Vec3& dx) {
    AABB box(x[node], x[node]);
    if (owning_rb_for_node(node_to_rb, node) == rb)
        box.expand(x[node] + dx);
    return box;
}

inline AABB rotated_node_swept_aabb(int node, const std::vector<Vec3>& x, const std::vector<int>& node_to_rb, int rb, const Vec3& x_com, const Vec4& q_current, double theta) {
    AABB box(x[node], x[node]);
    if (owning_rb_for_node(node_to_rb, node) == rb) {
        const Vec3 material_position = quaternion_inverse_rotate(q_current, x[node] - x_com);
        box.expand(spherical_cap_node_aabb(x_com, q_current, material_position, theta));
    }
    return box;
}

}  // namespace

double compute_trust_region_bound_for_vertex(int vi, const std::vector<Vec3>& x, const BroadPhase& broad_phase, double gamma_p) {
    const BroadPhase::Cache& bp_cache = broad_phase.cache();
    double d0_min = std::numeric_limits<double>::infinity();

    if (vi >= 0 && vi < static_cast<int>(bp_cache.vertex_nt.size())) {
        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            const double d0 = node_triangle_distance(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]]).distance;
            if (d0 < d0_min) d0_min = d0;
        }
    }

    if (vi >= 0 && vi < static_cast<int>(bp_cache.vertex_ss.size())) {
        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            const double d0 = segment_segment_distance(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]]).distance;
            if (d0 < d0_min) d0_min = d0;
        }
    }

    return gamma_p * d0_min;  // +inf when vi has no incident pairs
}

void per_vertex_safe_step(const BroadPhase& broad_phase, std::vector<Vec3>& x, const std::function<Vec3(int)>& x_new_fn, double safety, bool clip_ccd, bool use_ticcd, bool use_ogc, const std::vector<std::vector<int>>* color_groups, std::atomic<int>* clip_count) {
    const BroadPhase::Cache& bp_cache = broad_phase.cache();
    const int nv = static_cast<int>(x.size());

    auto process_vertex = [&](int vi) {
        const AABB& box = bp_cache.node_boxes[vi];
        assert((x[vi].array() >= box.min.array()).all() && (x[vi].array() <= box.max.array()).all() && "per_vertex_safe_step: current position is outside its cached node box");

        // Clip to the node box
        constexpr double inset = 1e-10;
        const Vec3 raw = x_new_fn(vi);
        const Vec3 lo = (box.min + Vec3::Constant(inset)).eval();
        const Vec3 hi = (box.max - Vec3::Constant(inset)).eval();
        const Vec3 x_new = raw.cwiseMax(lo).cwiseMin(hi);
        if (clip_count && (x_new - raw).squaredNorm() > 0.0) ++(*clip_count);

        const Vec3 dx = x_new - x[vi];
        if (dx.squaredNorm() < 1e-28) return;

        double toi_min = 1.0;
        bool has_collision = false;

        if (use_ogc) {
            double bound = compute_trust_region_bound_for_vertex(vi, x, broad_phase, 0.4);
            if (!std::isfinite(bound)) {
                // No-pair fallback: half min-extent of the cubic node box.
                const Vec3 e = bp_cache.node_boxes[vi].extent();
                bound = 0.5 * std::min({e.x(), e.y(), e.z()});
            }
            const double dx_norm = dx.norm();
            if (dx_norm > 0.0)
                toi_min = std::min(1.0, bound / dx_norm);
        }

        if (clip_ccd) for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            // Conservatively precheck the swept primitive AABBs to reject impossible collisions before running the more expensive exact CCD test.
            if (!node_triangle_single_vertex_swept_aabbs_intersect(p, entry.dof, x, dx))
                continue;
            CCDResult r;
            if (entry.dof == 0) {
                r = node_triangle_only_one_node_moves(x[vi], dx, x[p.tri_v[0]], Vec3::Zero(), x[p.tri_v[1]], Vec3::Zero(), x[p.tri_v[2]], Vec3::Zero(), 1e-12, use_ticcd);
            } else {
                Vec3 d0 = Vec3::Zero(), d1 = Vec3::Zero(), d2 = Vec3::Zero();
                if (entry.dof == 1) d0 = dx;
                else if (entry.dof == 2) d1 = dx;
                else d2 = dx;
                r = node_triangle_only_one_node_moves(x[p.node], Vec3::Zero(), x[p.tri_v[0]], d0, x[p.tri_v[1]], d1, x[p.tri_v[2]], d2, 1e-12, use_ticcd);
            }
            if (r.collision) {
                has_collision = true;
                toi_min = std::min(toi_min, r.t);
            }
        }

        if (clip_ccd) for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            if (!segment_segment_single_vertex_swept_aabbs_intersect(p, entry.dof, x, dx))
                continue;
            CCDResult r;
            if (entry.dof == 0)
                r = segment_segment_only_one_node_moves(x[vi], dx, x[p.v[1]], x[p.v[2]], x[p.v[3]], 1e-12, use_ticcd);
            else if (entry.dof == 1)
                r = segment_segment_only_one_node_moves(x[vi], dx, x[p.v[0]], x[p.v[2]], x[p.v[3]], 1e-12, use_ticcd);
            else if (entry.dof == 2)
                r = segment_segment_only_one_node_moves(x[vi], dx, x[p.v[3]], x[p.v[0]], x[p.v[1]], 1e-12, use_ticcd);
            else
                r = segment_segment_only_one_node_moves(x[vi], dx, x[p.v[2]], x[p.v[0]], x[p.v[1]], 1e-12, use_ticcd);
            if (r.collision) {
                has_collision = true;
                toi_min = std::min(toi_min, r.t);
            }
        }

        const double step = use_ogc ? toi_min : (has_collision ? safety * toi_min : 1.0);
        x[vi] = x[vi] + step * dx;
    };

    if (color_groups) {
        #pragma omp parallel
        {
            for (const auto& group : *color_groups) {
                #pragma omp for schedule(static)
                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                    process_vertex(group[i]);
            }
        }
    } else {
        for (int vi = 0; vi < nv; ++vi)
            process_vertex(vi);
    }
}

double per_rigid_body_translation_safe_step(const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<Vec3>& x, int rb, const Vec3& dx, double safety) {
    assert(rb >= 0);
    assert(safety >= 0.0 && safety <= 1.0);
    if (dx.squaredNorm() < 1.0e-28)
        return 1.0;

    double toi_min = 1.0;
    bool has_collision = false;
    const Vec3 zero = Vec3::Zero();
    const auto consider = [&](const CCDResult& result) {
        if (!result.collision)
            return;
        has_collision = true;
        toi_min = std::min(toi_min, result.t);
    };

    for (const int pair_index : nt_pair_indices) {
        const NodeTrianglePair& pair = bp_cache.nt_pairs[pair_index];
        const int node = pair.node;
        const int node_rb = owning_rb_for_node(ref_mesh.node_to_rb, node);
        const int triangle_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.tri_v[0]);
        if (node_rb == triangle_rb || (node_rb != rb && triangle_rb != rb))
            continue;
        const std::array<AABB, 4> node_boxes = {translated_node_swept_aabb(node, x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.tri_v[0], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.tri_v[1], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.tri_v[2], x, ref_mesh.node_to_rb, rb, dx)};
        if (!node_triangle_swept_aabbs_intersect(node_boxes))
            continue;
        if (node_rb == rb)
            consider(node_triangle_only_one_node_moves(x[node], dx, x[pair.tri_v[0]], zero, x[pair.tri_v[1]], zero, x[pair.tri_v[2]], zero, 1.0e-12, false));
        else
            consider(node_triangle_only_one_node_moves(x[node], -dx, x[pair.tri_v[0]], zero, x[pair.tri_v[1]], zero, x[pair.tri_v[2]], zero, 1.0e-12, false));
    }

    for (const int pair_index : ss_pair_indices) {
        const SegmentSegmentPair& pair = bp_cache.ss_pairs[pair_index];
        const int first_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.v[0]);
        const int second_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.v[2]);
        if (first_edge_rb == second_edge_rb || (first_edge_rb != rb && second_edge_rb != rb))
            continue;
        const std::array<AABB, 4> node_boxes = {translated_node_swept_aabb(pair.v[0], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.v[1], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.v[2], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.v[3], x, ref_mesh.node_to_rb, rb, dx)};
        if (!segment_segment_swept_aabbs_intersect(node_boxes))
            continue;
        if (first_edge_rb == rb)
            consider(segment_segment_same_displacement_linear_ccd(x[pair.v[0]], dx, x[pair.v[1]], dx, x[pair.v[2]], x[pair.v[3]], 1.0e-12));
        else
            consider(segment_segment_same_displacement_linear_ccd(x[pair.v[2]], dx, x[pair.v[3]], dx, x[pair.v[0]], x[pair.v[1]], 1.0e-12));
    }

    return has_collision ? safety * toi_min : 1.0;
}

Vec4 bound_quaternion(const Vec4& q_box_anchor, const Vec4& q_current, const Vec4& q_target, double theta_bound) {
    const Vec4 current = quaternion_normalize(q_current);
    const Vec4 target = quaternion_normalize(q_target);
    const Vec4 relative = quaternion_normalize(quaternion_multiply(target, quaternion_conjugate(current)));
    const Vec3 vector_part = relative.tail<3>();
    const double sin_half_arc = vector_part.norm();
    constexpr double full_turn_axis_tolerance = 64.0 * std::numeric_limits<double>::epsilon();
    if (relative[0] < 0.0 && sin_half_arc <= full_turn_axis_tolerance)
        throw std::invalid_argument("bound_quaternion cannot determine the axis of an exact full turn");
    if (theta_bound >= M_PI || (relative[0] >= 0.0 && sin_half_arc == 0.0))
        return target;

    const Vec4 box_anchor = quaternion_normalize(q_box_anchor);
    const double half_arc = std::atan2(sin_half_arc, relative[0]);
    const Vec4 tangent = quaternion_multiply(Vec4(0.0, vector_part.x() / sin_half_arc, vector_part.y() / sin_half_arc, vector_part.z() / sin_half_arc), current);
    const double box_dot_current = box_anchor.dot(current);
    const double sign = box_dot_current < 0.0 ? -1.0 : 1.0;
    const double a = std::abs(box_dot_current);
    const double b = sign * box_anchor.dot(tangent);
    const double cap_cosine = std::cos(0.5 * std::max(0.0, theta_bound));
    assert(a + 1.0e-12 >= cap_cosine && "bound_quaternion: current orientation is outside its cached cap");
    const double amplitude = std::hypot(a, b);
    const double phase = std::atan2(b, a);
    const double boundary_sine = std::sqrt(std::max(0.0, amplitude * amplitude - cap_cosine * cap_cosine));
    const double boundary_offset = std::atan2(boundary_sine, cap_cosine);
    const double first_exit = std::max(0.0, phase + boundary_offset);
    const double alpha = first_exit >= half_arc ? 1.0 : std::clamp(first_exit / half_arc, 0.0, 1.0);
    return interpolate_orientation_full_arc(current, target, alpha);
}

double per_rigid_body_rotation_safe_step(const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<Vec3>& x, int rb, const Vec3& x_com, const Vec4& q_current, const Vec4& q_target, double safety) {
    assert(rb >= 0);
    assert(safety >= 0.0 && safety <= 1.0);

    const Vec4 current = quaternion_normalize(q_current);
    const Vec4 proposed = quaternion_normalize(q_target);
    const Vec4 q_reverse = quaternion_normalize(quaternion_multiply(current, quaternion_conjugate(proposed)));
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 relative = quaternion_normalize(quaternion_multiply(proposed, quaternion_conjugate(current)));
    const double theta = 2.0 * std::atan2(relative.tail<3>().norm(), relative[0]);

    double toi_min = 1.0;
    bool has_collision = false;
    const auto consider = [&](bool collision, double toi) {
        if (!collision)
            return;
        has_collision = true;
        toi_min = std::min(toi_min, toi);
    };

    for (const int pair_index : nt_pair_indices) {
        const NodeTrianglePair& pair = bp_cache.nt_pairs[pair_index];
        const int node = pair.node;
        const int node_rb = owning_rb_for_node(ref_mesh.node_to_rb, node);
        const int triangle_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.tri_v[0]);
        if (node_rb == triangle_rb || (node_rb != rb && triangle_rb != rb))
            continue;
        const std::array<AABB, 4> node_boxes = {rotated_node_swept_aabb(node, x, ref_mesh.node_to_rb, rb, x_com, current, theta), rotated_node_swept_aabb(pair.tri_v[0], x, ref_mesh.node_to_rb, rb, x_com, current, theta), rotated_node_swept_aabb(pair.tri_v[1], x, ref_mesh.node_to_rb, rb, x_com, current, theta), rotated_node_swept_aabb(pair.tri_v[2], x, ref_mesh.node_to_rb, rb, x_com, current, theta)};
        if (!node_triangle_swept_aabbs_intersect(node_boxes))
            continue;
        double toi = 0.0;
        if (node_rb == rb)
            consider(point_triangle_rb_rotation_ccd(x[node], x_com, proposed, current, x[pair.tri_v[0]], x[pair.tri_v[1]], x[pair.tri_v[2]], toi), toi);
        else
            consider(point_triangle_rb_rotation_ccd(x[node], x_com, q_reverse, identity, x[pair.tri_v[0]], x[pair.tri_v[1]], x[pair.tri_v[2]], toi), toi);
    }

    for (const int pair_index : ss_pair_indices) {
        const SegmentSegmentPair& pair = bp_cache.ss_pairs[pair_index];
        const int first_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.v[0]);
        const int second_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.v[2]);
        if (first_edge_rb == second_edge_rb || (first_edge_rb != rb && second_edge_rb != rb))
            continue;
        const std::array<AABB, 4> node_boxes = {rotated_node_swept_aabb(pair.v[0], x, ref_mesh.node_to_rb, rb, x_com, current, theta), rotated_node_swept_aabb(pair.v[1], x, ref_mesh.node_to_rb, rb, x_com, current, theta), rotated_node_swept_aabb(pair.v[2], x, ref_mesh.node_to_rb, rb, x_com, current, theta), rotated_node_swept_aabb(pair.v[3], x, ref_mesh.node_to_rb, rb, x_com, current, theta)};
        if (!segment_segment_swept_aabbs_intersect(node_boxes))
            continue;
        double toi = 0.0;
        if (first_edge_rb == rb)
            consider(segment_segment_rb_rotation_ccd(x[pair.v[0]], x[pair.v[1]], x_com, proposed, current, x[pair.v[2]], x[pair.v[3]], toi), toi);
        else
            consider(segment_segment_rb_rotation_ccd(x[pair.v[0]], x[pair.v[1]], x_com, q_reverse, identity, x[pair.v[2]], x[pair.v[3]], toi), toi);
    }

    return has_collision ? safety * toi_min : 1.0;
}
