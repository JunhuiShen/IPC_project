#include "safe_step.h"

#include "broad_phase.h"
#include "ccd.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

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

double per_rigid_body_translation_safe_step(const RefMesh& ref_mesh, const std::vector<std::array<int, 2>>& edges, const std::vector<Vec3>& x, int rb, const Vec3& dx, double safety) {
    assert(rb >= 0);
    assert(safety >= 0.0 && safety <= 1.0);
    if (dx.squaredNorm() < 1.0e-28) return 1.0;

    const auto owned_by_current_body = [&](int node) {
        assert(node >= 0 && node < static_cast<int>(x.size()));
        return node < static_cast<int>(ref_mesh.node_to_rb.size()) && ref_mesh.node_to_rb[node] == rb;
    };

    double toi_min = 1.0;
    bool has_collision = false;
    const Vec3 zero = Vec3::Zero();
    const auto consider = [&](const CCDResult& result) {
        if (!result.collision) return;
        has_collision = true;
        toi_min = std::min(toi_min, result.t);
    };

    // Node-triangle pairs
    // If the node moves, use its displacement directly
    // If the triangle moves, subtract its translation from the whole scene, and the triangle is then fixed and the external node moves by -dx
    for (int node = 0; node < static_cast<int>(x.size()); ++node) {
        const bool node_is_current = owned_by_current_body(node);
        for (int tri = 0; tri < num_tris(ref_mesh); ++tri) {
            const int v0 = tri_vertex(ref_mesh, tri, 0);
            const int v1 = tri_vertex(ref_mesh, tri, 1);
            const int v2 = tri_vertex(ref_mesh, tri, 2);
            if (node == v0 || node == v1 || node == v2)
                continue;

            const bool v0_is_current = owned_by_current_body(v0);
            const bool v1_is_current = owned_by_current_body(v1);
            const bool v2_is_current = owned_by_current_body(v2);
            const bool triangle_touches_current = v0_is_current || v1_is_current || v2_is_current; // at least one triangle vertex belongs to rb
            const bool triangle_is_current = v0_is_current && v1_is_current && v2_is_current; // all three triangle vertices belong to rb

            if (node_is_current && !triangle_touches_current) { // the node moves and the entire triangle is fixed
                consider(node_triangle_only_one_node_moves(x[node], dx, x[v0], zero, x[v1], zero, x[v2], zero, /*eps=*/1.0e-12, /*use_ticcd=*/false));
            } else if (!node_is_current && triangle_is_current) { // the triangle moves and the node is fixed
                consider(node_triangle_only_one_node_moves(x[node], -dx, x[v0], zero, x[v1], zero, x[v2], zero, /*eps=*/1.0e-12, /*use_ticcd=*/false));
            }
        }
    }

    // Segment-segment pairs
    // Exactly one complete edge translates, and the other must not touch this rigid body
    // Here, the ccd must receive dx for  both endpoints because their displacement is identical
    for (int first = 0; first < static_cast<int>(edges.size()); ++first) {
        const int a0 = edges[first][0];
        const int a1 = edges[first][1];
        const bool first0_is_current = owned_by_current_body(a0);
        const bool first1_is_current = owned_by_current_body(a1);
        const bool first_touches_current = first0_is_current || first1_is_current;
        const bool first_is_current = first0_is_current && first1_is_current;

        for (int second = first + 1;
             second < static_cast<int>(edges.size()); ++second) {
            const int b0 = edges[second][0];
            const int b1 = edges[second][1];
            if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1)
                continue;

            const bool second0_is_current = owned_by_current_body(b0);
            const bool second1_is_current = owned_by_current_body(b1);
            const bool second_touches_current = second0_is_current || second1_is_current;
            const bool second_is_current = second0_is_current && second1_is_current;

            if (first_is_current && !second_touches_current) {
                consider(segment_segment_same_displacement_linear_ccd(x[a0], dx, x[a1], dx, x[b0], x[b1], /*eps=*/1.0e-12));
            } else if (!first_touches_current && second_is_current) {
                consider(segment_segment_same_displacement_linear_ccd(x[b0], dx, x[b1], dx, x[a0], x[a1], /*eps=*/1.0e-12));
            }
        }
    }

    return has_collision ? safety * toi_min : 1.0;
}
