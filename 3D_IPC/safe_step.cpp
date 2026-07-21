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
