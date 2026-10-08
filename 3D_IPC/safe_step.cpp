#include "safe_step.h"
#include "contact_scheduling.h"

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
#include <optional>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// CCD consumes an ordered minimum, unlike the derivative sums. Reduce small
// consecutive batches before staging them, preserving the first equal minimum
// (including signed zero), collision flags, and std::min's treatment of NaNs.
// This avoids a large per-contact scratch array and serial scan on the leader.
template <class Evaluate, class Accumulate>
void ordered_ccd_tasks(int count, bool cooperative,
                       const Evaluate& evaluate, const Accumulate& accumulate) {
    if (!cooperative || count < 32 || omp_get_num_threads() == 1) {
        for (int i = 0; i < count; ++i) accumulate(evaluate(i));
        return;
    }
    constexpr int batch_size = 16;
    struct alignas(64) Slot { CCDResult value; };
    static thread_local std::vector<Slot> reusable;
    std::vector<Slot> results;
    results.swap(reusable);
    results.resize(count / batch_size + (count % batch_size != 0));
    // Dispatch in units of raw queries, not batches. Otherwise the scheduler's
    // minimum grain would multiply by batch_size and leave a few large CCD
    // ranges running after their peers had finished.
    solver_detail::evaluate_contact_ranges(count, [&](int begin, int end) {
        for (int first = begin; first < end; first += batch_size) {
            CCDResult minimum;
            minimum.t = 1.0;
            for (int i = first; i < std::min(first + batch_size, end); ++i) {
                const CCDResult value = evaluate(i);
                if (value.collision) {
                    minimum.collision = true;
                    minimum.t = std::min(minimum.t, value.t);
                }
            }
            results[first / batch_size].value = minimum;
        }
    }, batch_size);
    for (const auto& result : results) accumulate(result.value);
    results.swap(reusable);
}

inline bool node_triangle_single_vertex_swept_aabbs_intersect_reference(const NodeTrianglePair& p, int moving_dof, const std::vector<Vec3>& x, const Vec3& dx) {
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

inline bool segment_segment_single_vertex_swept_aabbs_intersect_reference(const SegmentSegmentPair& p, int moving_dof, const std::vector<Vec3>& x, const Vec3& dx) {
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

// An independent non-NaN axis can reject before the remaining coordinates
// are loaded or expanded. Ordered interval min/max also supports infinities;
// only NaNs require the original Eigen min/max behavior. Check the moved
// coordinate as well, including a NaN produced by infinity plus its opposite.
inline bool node_triangle_single_vertex_swept_aabbs_intersect(const NodeTrianglePair& p, int moving_dof, const std::vector<Vec3>& x, const Vec3& dx) {
    const Vec3& node = x[p.node];
    const Vec3& a = x[p.tri_v[0]];
    const Vec3& b = x[p.tri_v[1]];
    const Vec3& c = x[p.tri_v[2]];
    const bool moves = moving_dof >= 0 && moving_dof < 4;
    const Vec3& moving = moving_dof == 0 ? node
        : moving_dof == 1 ? a : moving_dof == 2 ? b : c;
    for (int axis = 0; axis < 3; ++axis) {
        const double q = node[axis], u = a[axis], v = b[axis], w = c[axis];
        const double moved = moves ? moving[axis] + dx[axis] : 0.0;
        if (std::isnan(q) || std::isnan(u) || std::isnan(v)
            || std::isnan(w) || std::isnan(moved))
            return node_triangle_single_vertex_swept_aabbs_intersect_reference(p, moving_dof, x, dx);
        double node_lo = q, node_hi = q;
        double tri_lo = std::min({u, v, w});
        double tri_hi = std::max({u, v, w});
        if (moving_dof == 0) {
            node_lo = std::min(node_lo, moved);
            node_hi = std::max(node_hi, moved);
        } else if (moves) {
            tri_lo = std::min(tri_lo, moved);
            tri_hi = std::max(tri_hi, moved);
        }
        if (node_hi < tri_lo || tri_hi < node_lo) return false;
    }
    return true;
}

inline bool segment_segment_single_vertex_swept_aabbs_intersect(const SegmentSegmentPair& p, int moving_dof, const std::vector<Vec3>& x, const Vec3& dx) {
    const Vec3& a = x[p.v[0]];
    const Vec3& b = x[p.v[1]];
    const Vec3& c = x[p.v[2]];
    const Vec3& d = x[p.v[3]];
    const bool moves = moving_dof >= 0 && moving_dof < 4;
    const Vec3& moving = moving_dof == 0 ? a
        : moving_dof == 1 ? b : moving_dof == 2 ? c : d;
    for (int axis = 0; axis < 3; ++axis) {
        const double u = a[axis], v = b[axis], w = c[axis], z = d[axis];
        const double moved = moves ? moving[axis] + dx[axis] : 0.0;
        if (std::isnan(u) || std::isnan(v) || std::isnan(w)
            || std::isnan(z) || std::isnan(moved))
            return segment_segment_single_vertex_swept_aabbs_intersect_reference(p, moving_dof, x, dx);
        double first_lo = std::min(u, v), first_hi = std::max(u, v);
        double second_lo = std::min(w, z), second_hi = std::max(w, z);
        if (moves && moving_dof < 2) {
            first_lo = std::min(first_lo, moved);
            first_hi = std::max(first_hi, moved);
        } else if (moves) {
            second_lo = std::min(second_lo, moved);
            second_hi = std::max(second_hi, moved);
        }
        if (first_hi < second_lo || second_hi < first_lo) return false;
    }
    return true;
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

inline AABB rotated_node_swept_aabb(int node, const std::vector<Vec3>& x, const std::vector<int>& node_to_rb, int rb, const Vec3& x_com, const Vec4& q_current,
    const parallel_helper_detail::SphericalCapRotation& cap) {
    AABB box(x[node], x[node]);
    if (owning_rb_for_node(node_to_rb, node) == rb) {
        const Vec3 material_position = quaternion_inverse_rotate(q_current, x[node] - x_com);
        box.expand(parallel_helper_detail::spherical_cap_node_aabb_prepared(x_com, cap, material_position));
    }
    return box;
}

}  // namespace

namespace safe_step_detail {
CCDResult node_triangle_vertex_ccd(const NodeTrianglePair &p, int dof, int vi,
                                   const std::vector<Vec3> &x, const Vec3 &dx, bool use_ticcd, bool original) {
    if (!node_triangle_single_vertex_swept_aabbs_intersect(p, dof, x, dx))
        return {};
    if (dof == 0)
        return node_triangle_only_one_node_moves(x[vi], dx, x[p.tri_v[0]], Vec3::Zero(),
                                                 x[p.tri_v[1]], Vec3::Zero(), x[p.tri_v[2]],
                                                 Vec3::Zero(), 1e-12, use_ticcd, original);
    Vec3 d0 = Vec3::Zero(), d1 = Vec3::Zero(), d2 = Vec3::Zero();
    if (dof == 1)
        d0 = dx;
    else if (dof == 2)
        d1 = dx;
    else
        d2 = dx;
    return node_triangle_only_one_node_moves(x[p.node], Vec3::Zero(), x[p.tri_v[0]], d0,
                                             x[p.tri_v[1]], d1, x[p.tri_v[2]], d2, 1e-12,
                                             use_ticcd, original);
}
CCDResult segment_segment_vertex_ccd(const SegmentSegmentPair &p, int dof, int vi,
                                     const std::vector<Vec3> &x, const Vec3 &dx, bool use_ticcd, bool original) {
    if (!segment_segment_single_vertex_swept_aabbs_intersect(p, dof, x, dx))
        return {};
    if (dof == 0)
        return segment_segment_only_one_node_moves(x[vi], dx, x[p.v[1]], x[p.v[2]], x[p.v[3]],
                                                   1e-12, use_ticcd, original);
    if (dof == 1)
        return segment_segment_only_one_node_moves(x[vi], dx, x[p.v[0]], x[p.v[2]], x[p.v[3]],
                                                   1e-12, use_ticcd, original);
    if (dof == 2)
        return segment_segment_only_one_node_moves(x[vi], dx, x[p.v[3]], x[p.v[0]], x[p.v[1]],
                                                   1e-12, use_ticcd, original);
    return segment_segment_only_one_node_moves(x[vi], dx, x[p.v[2]], x[p.v[0]], x[p.v[1]], 1e-12,
                                               use_ticcd, original);
}

} // namespace safe_step_detail

double compute_trust_region_bound_for_vertex(int vi, const std::vector<Vec3>& x, const BroadPhase& broad_phase, double gamma_p, bool exact_computation_fallback) {
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
            const double d0 = segment_segment_distance(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], 1e-12, exact_computation_fallback).distance;
            if (d0 < d0_min) d0_min = d0;
        }
    }

    return gamma_p * d0_min;  // +inf when vi has no incident pairs
}

// This is a numerical safeguard, not the barrier activation distance or a
// contact-force modification. Match the colored guess's existing gap floor,
// and enlarge it when the world-coordinate spacing makes 1e-8 unrepresentable.
static double vertex_contact_gap_floor(const std::array<Vec3, 4>& points) {
    double scale = 0.0;
    for (const auto& point : points)
        scale = std::max(scale, point.cwiseAbs().maxCoeff());
    return std::max(1e-8, 128.0 * std::numeric_limits<double>::epsilon() * scale);
}

// The bounds contain the actual stored endpoints, hence also every position
// on their segment. Strict separation of their convex projections certifies
// the whole rounded path, independently of approximate closest-point regions.
static bool vertex_swept_axis_separated(const std::array<Vec3, 4>& start,
    const std::array<Vec3, 4>& end, bool face, double floor) {
    const int split = face ? 1 : 2;
    for (int axis = 0; axis < 3; ++axis) {
        double alo = std::numeric_limits<double>::infinity(), ahi = -alo;
        double blo = alo, bhi = -alo;
        for (int i = 0; i < 4; ++i) {
            const double lo = std::min(start[i][axis], end[i][axis]);
            const double hi = std::max(start[i][axis], end[i][axis]);
            if (i < split) { alo = std::min(alo, lo); ahi = std::max(ahi, hi); }
            else { blo = std::min(blo, lo); bhi = std::max(bhi, hi); }
        }
        if (std::max(alo - bhi, blo - ahi) > 2.0 * floor) return true;
    }
    return false;
}

static bool vertex_swept_projection_separated(const std::array<Vec3, 4>& start,
    const std::array<Vec3, 4>& end, bool face, double floor, Vec3 direction) {
    const double scale = direction.cwiseAbs().maxCoeff();
    if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    direction /= scale;
    if (!direction.allFinite()) return false;
    const int split = face ? 1 : 2;
    double alo = std::numeric_limits<double>::infinity(), ahi = -alo;
    double blo = alo, bhi = -alo, magnitude = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double a = direction.dot(start[i]), b = direction.dot(end[i]);
        const double absolute_sum = direction.cwiseAbs().dot(
            start[i].cwiseAbs().cwiseMax(end[i].cwiseAbs()));
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(absolute_sum))
            return false;
        magnitude = std::max(magnitude, absolute_sum);
        const double lo = std::min(a, b), hi = std::max(a, b);
        if (i < split) { alo = std::min(alo, lo); ahi = std::max(ahi, hi); }
        else { blo = std::min(blo, lo); bhi = std::max(bhi, hi); }
    }
    const double threshold = floor * direction.norm();
    const double gap = std::max(alo - bhi, blo - ahi);
    const double padding = 512.0 * std::numeric_limits<double>::epsilon()
        * (magnitude + threshold) + 512.0 * std::numeric_limits<double>::denorm_min();
    return std::isfinite(gap) && std::isfinite(threshold) && std::isfinite(padding)
        && gap > threshold + padding;
}

static Vec3 vertex_contact_separating_direction(const std::array<Vec3, 4>& p,
    bool face) {
    if (!face)
        return segment_segment_distance(p[0], p[1], p[2], p[3]).separation;
    const auto result = node_triangle_distance(p[0], p[1], p[2], p[3]);
    return result.region == NodeTriangleRegion::FaceInterior
        ? result.normal : Vec3(p[0] - result.closest_point);
}

// This is only a proposed projection direction, not a distance or feature
// classification. The swept projection certificate below still has to prove
// separation for all four vertices at BOTH represented endpoints. In
// particular, a rounded/ill-conditioned cross product cannot approve a step
// on its own. Trying it first avoids a robust closest-feature solve (which
// can require rational arithmetic for nearly parallel edges) merely to find
// a direction. Zero/nonfinite directions fail the certificate and fall back.
static Vec3 vertex_contact_plane_direction(const std::array<Vec3, 4>& p,
    bool face) {
    return face ? Vec3((p[2] - p[1]).cross(p[3] - p[1]))
                : Vec3((p[1] - p[0]).cross(p[3] - p[2]));
}

struct VertexContactInterval {
    double lower, upper;
};

static VertexContactInterval vertex_contact_outward(double lower, double upper) {
    const double infinity = std::numeric_limits<double>::infinity();
    if (!std::isfinite(lower) || !std::isfinite(upper))
        return {-infinity, infinity};
    return {std::nextafter(lower, -infinity), std::nextafter(upper, infinity)};
}

static VertexContactInterval vertex_contact_interval_subtract(
    const VertexContactInterval& a, const VertexContactInterval& b) {
    return vertex_contact_outward(a.lower - b.upper, a.upper - b.lower);
}

static VertexContactInterval vertex_contact_interval_add(
    const VertexContactInterval& a, const VertexContactInterval& b) {
    return vertex_contact_outward(a.lower + b.lower, a.upper + b.upper);
}

static VertexContactInterval vertex_contact_interval_multiply(
    const VertexContactInterval& a, const VertexContactInterval& b) {
    const std::array<double, 4> products{{a.lower * b.lower, a.lower * b.upper,
        a.upper * b.lower, a.upper * b.upper}};
    for (double product : products)
        if (!std::isfinite(product)) {
            const double infinity = std::numeric_limits<double>::infinity();
            return {-infinity, infinity};
        }
    const auto bounds = std::minmax_element(products.begin(), products.end());
    return vertex_contact_outward(*bounds.first, *bounds.second);
}

// Enclose the determinant of the ORIGINAL represented coordinates, including
// subtraction roundoff. Every elementary operation is rounded outwards;
// overflow, cancellation, and underflow can only make the filter inconclusive.
// No floating-point sign without an enclosing interval is used as a verdict.
static int vertex_contact_determinant_sign(const std::array<Vec3, 4>& p) {
    std::array<VertexContactInterval, 3> a, b, c;
    for (int axis = 0; axis < 3; ++axis) {
        a[axis] = vertex_contact_outward(p[1][axis] - p[0][axis], p[1][axis] - p[0][axis]);
        b[axis] = vertex_contact_outward(p[2][axis] - p[0][axis], p[2][axis] - p[0][axis]);
        c[axis] = vertex_contact_outward(p[3][axis] - p[0][axis], p[3][axis] - p[0][axis]);
    }
    VertexContactInterval determinant{0.0, 0.0};
    for (int axis = 0; axis < 3; ++axis) {
        const int j = (axis + 1) % 3, k = (axis + 2) % 3;
        const auto cross = vertex_contact_interval_subtract(
            vertex_contact_interval_multiply(a[j], b[k]),
            vertex_contact_interval_multiply(a[k], b[j]));
        determinant = vertex_contact_interval_add(determinant,
            vertex_contact_interval_multiply(cross, c[axis]));
    }
    if (!std::isfinite(determinant.lower) || !std::isfinite(determinant.upper)) return 0;
    return determinant.lower > 0.0 ? 1 : (determinant.upper < 0.0 ? -1 : 0);
}

static bool vertex_contact_filtered_path_is_clear(
    const std::array<Vec3, 4>& start, const std::array<Vec3, 4>& end) {
    // For one moving vertex the determinant is affine along the ACTUAL stored
    // endpoint segment. Equal certified nonzero signs exclude all coplanar
    // events, hence initial contact and every possible NT/SS intersection.
    // This certifies the path only; the endpoint gap is checked separately.
    const int start_sign = vertex_contact_determinant_sign(start);
    return start_sign != 0 && vertex_contact_determinant_sign(end) == start_sign;
}

// Allocated only for a backtracking update. All other vertices stay fixed
// throughout that update, so its starting floor/direction are invariant.
// In particular, do not repeat an exact SS closest-feature solve on every
// trial just to propose the same separating direction.
struct VertexContactEndpointCache {
    double floor = 0.0;
    Vec3 direction;
    bool direction_ready = false;
    std::optional<PreparedExactContactStep> exact;
};

static bool vertex_contact_endpoint_is_safe(const std::array<Vec3, 4>& start,
    const std::array<Vec3, 4>& end, bool face, int vertex, int moving_dof,
    VertexContactEndpointCache* cache, double certified_distance = 0.0) {
    if (cache && cache->floor == 0.0) cache->floor = vertex_contact_gap_floor(start);
    const double floor = cache ? cache->floor : vertex_contact_gap_floor(start);
    // The caller has bounded the ACTUAL represented displacement by d_hat/8.
    // A current barrier certificate provides initial distance > d_hat/2,
    // even allowing ample roundoff slack for its AABB-distance calculation.
    // Thus the whole path stays > 3*d_hat/8, beyond this numerical gap floor.
    if (certified_distance > 0.0 && floor <= certified_distance / 8.0
        && std::all_of(start.begin(), start.end(), [](const Vec3& p) { return p.allFinite(); })) return true;
    if (vertex_swept_axis_separated(start, end, face, floor)) return true;
    if (vertex_swept_projection_separated(start, end, face, floor,
            vertex_contact_plane_direction(start, face))) return true;
    if (vertex_swept_projection_separated(start, end, face, floor,
            vertex_contact_plane_direction(end, face))) return true;
    if (cache && !cache->direction_ready) {
        cache->direction = vertex_contact_separating_direction(start, face);
        cache->direction_ready = true;
    }
    if (vertex_swept_projection_separated(start, end, face, floor,
            cache ? cache->direction : vertex_contact_separating_direction(start, face))) return true;
    if (vertex_swept_projection_separated(start, end, face, floor,
            vertex_contact_separating_direction(end, face))) return true;
    if (vertex_contact_filtered_path_is_clear(start, end)) {
        // Even when a gap is too close to the numerical floor for a floating
        // certificate, the path proof remains useful: retain the EXACT gap
        // comparison but do not solve the certified-clear path a second time.
        if (vertex_swept_projection_separated(end, end, face, floor,
                vertex_contact_plane_direction(end, face))) return true;
        if (!cache) return contact_preserves_separation_exact(start, end, face, floor);
    }

    // Convert the represented endpoints and compute the exact starting gap
    // once, sharing them between the gap-floor and whole-path checks.
    if (cache && !cache->exact)
        cache->exact.emplace(start, face, floor, moving_dof);
    const auto result = cache ? cache->exact->test(end[moving_dof])
        : exact_contact_step_result(start, end, face, floor);
    if (result == ExactContactStepResult::InitialContact)
        throw std::runtime_error("per_vertex_safe_step: exact initial "
            + std::string(face ? "point-triangle" : "edge-edge")
            + " contact at vertex " + std::to_string(vertex)
            + "; a strictly separated state is required (no safe side history).");
    return result == ExactContactStepResult::Safe;
}

static bool vertex_contact_endpoints_are_safe(const BroadPhase::Cache& cache,
    const std::vector<Vec3>& x, int vertex, const Vec3& endpoint,
    std::vector<VertexContactEndpointCache>* scratch = nullptr,
    const safe_step_detail::VertexAabbRejections* rejections = nullptr) {
    bool reuse_rejections = rejections && rejections->distance > 0.0
        && std::isnormal(rejections->distance * rejections->distance)
        && rejections->clear.size() == cache.vertex_nt[vertex].size() + cache.vertex_ss[vertex].size();
    if (reuse_rejections) {
        // One outward rounding encloses subtraction of the two original
        // binary64 coordinates. An infinity/NaN cannot qualify. This is
        // checked again for EVERY bisection endpoint, not inferred from alpha
        // or the differently rounded proposed displacement.
        for (int axis = 0; axis < 3; ++axis) {
            const double bound = std::nextafter(std::abs(endpoint[axis] - x[vertex][axis]),
                std::numeric_limits<double>::infinity());
            reuse_rejections = reuse_rejections
                && bound < rejections->distance / 16.0;
        }
    }
    std::size_t contact = 0;
    for (const auto& entry : cache.vertex_nt[vertex]) {
        const auto& p = cache.nt_pairs[entry.pair_index];
        const std::array<Vec3, 4> start = {x[p.node], x[p.tri_v[0]],
            x[p.tri_v[1]], x[p.tri_v[2]]};
        auto end = start;
        end[entry.dof] = endpoint;
        if (!vertex_contact_endpoint_is_safe(start, end, true, vertex, entry.dof,
                scratch ? &(*scratch)[contact] : nullptr,
                reuse_rejections && rejections->clear[contact] ? rejections->distance : 0.0)) return false;
        ++contact;
    }
    for (const auto& entry : cache.vertex_ss[vertex]) {
        const auto& p = cache.ss_pairs[entry.pair_index];
        const std::array<Vec3, 4> start = {x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]]};
        auto end = start;
        end[entry.dof] = endpoint;
        if (!vertex_contact_endpoint_is_safe(start, end, false, vertex, entry.dof,
                scratch ? &(*scratch)[contact] : nullptr,
                reuse_rejections && rejections->clear[contact] ? rejections->distance : 0.0)) return false;
        ++contact;
    }
    return true;
}

double per_vertex_safe_step(
    const BroadPhase& broad_phase, std::vector<Vec3>& x, int vi,
    const Vec3& raw_proposed_position, double safety, bool clip_ccd,
    bool use_ticcd, bool use_ogc, bool cooperative,
    const safe_step_detail::VertexAabbRejections* rejections, bool exact_computation_fallback) {
    const bool original = !exact_computation_fallback && !use_ticcd;
    const BroadPhase::Cache& bp_cache = broad_phase.cache();
    const int nv = static_cast<int>(x.size());
    if (vi < 0 || vi >= nv)
        throw std::out_of_range("per_vertex_safe_step: vertex is out of range");

    const AABB& box = bp_cache.node_boxes[vi];
    assert((x[vi].array() >= box.min.array()).all() && (x[vi].array() <= box.max.array()).all() && "per_vertex_safe_step: current position is outside its cached node box");

    // Clip to the node box.
    constexpr double inset = 1e-10;
    const Vec3 lo = (box.min + Vec3::Constant(inset)).eval();
    const Vec3 hi = (box.max - Vec3::Constant(inset)).eval();
    const Vec3 x_new = raw_proposed_position.cwiseMax(lo).cwiseMin(hi);

    const Vec3 dx = x_new - x[vi];
    // A representable separating displacement can be smaller than 1e-14.
    // Only a true stored-position no-op is discarded here.
    if ((x_new.array() == x[vi].array()).all()) return 0.0;

    double toi_min = 1.0;
    bool has_collision = false;

    if (use_ogc) {
        double bound = compute_trust_region_bound_for_vertex(
            vi, x, broad_phase, 0.4, exact_computation_fallback);
        if (!std::isfinite(bound)) {
            // No-pair fallback: half min-extent of the cubic node box.
            const Vec3 e = bp_cache.node_boxes[vi].extent();
            bound = 0.5 * std::min({e.x(), e.y(), e.z()});
        }
        const double dx_norm = dx.norm();
        if (dx_norm > 0.0)
            toi_min = std::min(1.0, bound / dx_norm);
    }

    if (clip_ccd) {
        const auto& nt = bp_cache.vertex_nt[vi];
        const auto& ss = bp_cache.vertex_ss[vi];
        const int nt_count = static_cast<int>(nt.size());
        const double distance_squared = rejections
            ? rejections->distance * rejections->distance : 0.0;
        // AABB distance > d_hat gives an axis gap > d_hat/sqrt(3). A verified
        // finite-primitive projection certificate instead gives a directional
        // gap > d_hat. Moving one vertex by < d_hat/4 cannot close either gap.
        // Projected certificates are supplied only when using linear CCD.
        const bool reuse_rejections = rejections && rejections->distance > 1e-8
            && std::isfinite(distance_squared)
            && rejections->clear.size() == nt.size() + ss.size()
            && dx.squaredNorm() < distance_squared / 16.0;
        ordered_ccd_tasks(nt_count + static_cast<int>(ss.size()), cooperative,
            [&](int i) {
                if (reuse_rejections && rejections->clear[i]) return CCDResult{};
                if (i < nt_count) {
                    const auto& entry = nt[i];
                    return safe_step_detail::node_triangle_vertex_ccd(
                        bp_cache.nt_pairs[entry.pair_index], entry.dof, vi, x, dx, use_ticcd, original);
                }
                const auto& entry = ss[i - nt_count];
                return safe_step_detail::segment_segment_vertex_ccd(
                    bp_cache.ss_pairs[entry.pair_index], entry.dof, vi, x, dx, use_ticcd, original);
            }, [&](const CCDResult& r) {
                if (r.collision) {
                    has_collision = true;
                    toi_min = std::min(toi_min, r.t);
                }
            });
    }

    double step = use_ogc
        ? toi_min
        : (has_collision ? safety * toi_min : 1.0);
    const Vec3 before = x[vi];
    Vec3 accepted = before + step * dx;
    if (clip_ccd && !use_ticcd && !use_ogc && !original
        && !vertex_contact_endpoints_are_safe(bp_cache, x, vi, accepted, nullptr, rejections)) {
        // Search only within the CCD-approved prefix. Keep a verified endpoint
        // (not just a floating alpha), validating all contacts for each new
        // candidate before accepting it.
        double lower = 0.0, upper = step;
        Vec3 best = before;
        Vec3 rejected = accepted;
        std::vector<VertexContactEndpointCache> scratch(
            bp_cache.vertex_nt[vi].size() + bp_cache.vertex_ss[vi].size());
        for (int attempt = 0; attempt < 64; ++attempt) {
            const double middle = lower + 0.5 * (upper - lower);
            if (middle == lower || middle == upper) break;
            const Vec3 trial = before + middle * dx;
            if ((trial.array() == before.array()).all()) {
                lower = middle;
            } else if ((trial.array() == rejected.array()).all()) {
                // Different alphas can round to exactly the same stored
                // position. Reuse its verdict without changing the search.
                upper = middle;
            } else if ((trial.array() == best.array()).all()
                || vertex_contact_endpoints_are_safe(bp_cache, x, vi, trial, &scratch, rejections)) {
                lower = middle;
                best = trial;
            } else {
                upper = middle;
                rejected = trial;
            }
        }
        step = lower;
        accepted = best;
    }
    x[vi] = accepted;
    if (original && clip_ccd && !use_ticcd && !use_ogc) return step;
    if ((accepted.array() == before.array()).all()) return 0.0;
    return step;
}

double per_rigid_body_translation_safe_step(const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<Vec3>& x, int rb, const Vec3& dx, double safety, bool cooperative, bool original) {
    assert(rb >= 0);
    assert(safety >= 0.0 && safety <= 1.0);
    if (dx.squaredNorm() < 1.0e-28)
        return 1.0;

    double toi_min = 1.0;
    bool has_collision = false;
    const Vec3 zero = Vec3::Zero();
    const auto evaluate_nt = [&](int pair_index) {
        CCDResult value;
        const auto consider = [&](const CCDResult& result) { value = result; };

        const NodeTrianglePair& pair = bp_cache.nt_pairs[pair_index];
        const int node = pair.node;
        const int node_rb = owning_rb_for_node(ref_mesh.node_to_rb, node);
        const int triangle_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.tri_v[0]);
        if (node_rb == triangle_rb || (node_rb != rb && triangle_rb != rb))
            return value;
        const std::array<AABB, 4> node_boxes = {translated_node_swept_aabb(node, x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.tri_v[0], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.tri_v[1], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.tri_v[2], x, ref_mesh.node_to_rb, rb, dx)};
        if (!node_triangle_swept_aabbs_intersect(node_boxes))
            return value;
        if (node_rb == rb)
            consider(node_triangle_only_one_node_moves(x[node], dx, x[pair.tri_v[0]], zero, x[pair.tri_v[1]], zero, x[pair.tri_v[2]], zero, 1.0e-12, false, original));
        else
            consider(node_triangle_only_one_node_moves(x[node], -dx, x[pair.tri_v[0]], zero, x[pair.tri_v[1]], zero, x[pair.tri_v[2]], zero, 1.0e-12, false, original));
        return value;
    };
    const auto evaluate_ss = [&](int pair_index) {
        CCDResult value;
        const auto consider = [&](const CCDResult& result) { value = result; };

        const SegmentSegmentPair& pair = bp_cache.ss_pairs[pair_index];
        const int first_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.v[0]);
        const int second_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.v[2]);
        if (first_edge_rb == second_edge_rb || (first_edge_rb != rb && second_edge_rb != rb))
            return value;
        const std::array<AABB, 4> node_boxes = {translated_node_swept_aabb(pair.v[0], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.v[1], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.v[2], x, ref_mesh.node_to_rb, rb, dx), translated_node_swept_aabb(pair.v[3], x, ref_mesh.node_to_rb, rb, dx)};
        if (!segment_segment_swept_aabbs_intersect(node_boxes))
            return value;
        if (first_edge_rb == rb)
            consider(segment_segment_same_displacement_linear_ccd(x[pair.v[0]], dx, x[pair.v[1]], dx, x[pair.v[2]], x[pair.v[3]], 1.0e-12, original));
        else
            consider(segment_segment_same_displacement_linear_ccd(x[pair.v[2]], dx, x[pair.v[3]], dx, x[pair.v[0]], x[pair.v[1]], 1.0e-12, original));
        return value;
    };
    const int nt_count = static_cast<int>(nt_pair_indices.size());
    ordered_ccd_tasks(nt_count + static_cast<int>(ss_pair_indices.size()), cooperative,
        [&](int i) { return i < nt_count ? evaluate_nt(nt_pair_indices[i])
            : evaluate_ss(ss_pair_indices[i - nt_count]); },
        [&](const CCDResult& value) {
            if (value.collision) {
                has_collision = true;
                toi_min = std::min(toi_min, value.t);
            }
        });

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

double per_rigid_body_rotation_safe_step(const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<Vec3>& x, int rb, const Vec3& x_com, const Vec4& q_current, const Vec4& q_target, double safety, bool cooperative) {
    assert(rb >= 0);
    assert(safety >= 0.0 && safety <= 1.0);

    const Vec4 current = quaternion_normalize(q_current);
    const Vec4 proposed = quaternion_normalize(q_target);
    const Vec4 q_reverse = quaternion_normalize(quaternion_multiply(current, quaternion_conjugate(proposed)));
    const Vec4 identity(1.0, 0.0, 0.0, 0.0);
    const Vec4 relative = quaternion_normalize(quaternion_multiply(proposed, quaternion_conjugate(current)));
    const auto visit_body_nodes = [&](const auto& visit) {
        if (rb < static_cast<int>(ref_mesh.rb_nodes.size()) && !ref_mesh.rb_nodes[static_cast<std::size_t>(rb)].empty()) {
            for (const int node : ref_mesh.rb_nodes[static_cast<std::size_t>(rb)]) visit(node);
        } else {
            for (int node = 0; node < static_cast<int>(x.size()); ++node)
                if (owning_rb_for_node(ref_mesh.node_to_rb, node) == rb) visit(node);
        }
    };
    const bool no_pairs = nt_pair_indices.empty() && ss_pair_indices.empty();
    // Every retained NT/SS query uses one of these two quaternion paths.
    // If both hit the kernels' existing first rejection, no candidate can
    // collide. Hoist that exact predicate before swept-box construction and
    // helper dispatch, while retaining the old per-node input validation.
    const bool no_rotation = !no_pairs
        && ccd_detail::negligible_rigid_rotation(proposed, current)
        && ccd_detail::negligible_rigid_rotation(q_reverse, identity);
    if (no_pairs || no_rotation) {
        // Preserve the old cap path's validation, including overflow while
        // converting finite world positions into body space. No swept boxes
        // or angular trigonometry are needed when no pair can consume them.
        visit_body_nodes([&](int node) {
            if (owning_rb_for_node(ref_mesh.node_to_rb, node) != rb) return;
            const Vec3 material_position = quaternion_inverse_rotate(current, x[node] - x_com);
            if (!x_com.allFinite() || !material_position.allFinite())
                throw std::invalid_argument("spherical_cap_node_aabb requires finite positions");
        });
        return 1.0;
    }
    const double theta = 2.0 * std::atan2(relative.tail<3>().norm(), relative[0]);
    // The inverse rotation below still uses current. Cap evaluation retains
    // the additional normalization that the old per-node helper performed.
    const auto cap = parallel_helper_detail::prepare_spherical_cap_rotation(current, theta);

    // A rigid node participates in many candidate pairs, but its swept
    // spherical-cap box depends only on this body update. Compute it once per
    // safe-step call instead of repeating inverse rotations and trigonometry
    // for every incident candidate.
    thread_local std::vector<AABB> reusable_rotated_node_boxes;
    std::vector<AABB> rotated_node_boxes;
    rotated_node_boxes.swap(reusable_rotated_node_boxes);
    rotated_node_boxes.resize(x.size());
    const auto build_rotated_box = [&](int node) {
        rotated_node_boxes[static_cast<std::size_t>(node)] = rotated_node_swept_aabb(
            node, x, ref_mesh.node_to_rb, rb, x_com, current, cap);
    };
    if (cooperative && solver_detail::active_contact_task_group
        && rb < static_cast<int>(ref_mesh.rb_nodes.size())
        && ref_mesh.rb_nodes[rb].size() >= 128) {
        // The body's helpers otherwise wait while the leader constructs all
        // swept boxes. Boxes are independent; join every write before contact
        // evaluation, using the same per-node arithmetic as the serial path.
        const auto& nodes = ref_mesh.rb_nodes[rb];
        solver_detail::evaluate_contact_ranges(static_cast<int>(nodes.size()),
            [&](int begin, int end) {
                for (int i = begin; i < end; ++i) build_rotated_box(nodes[i]);
            }, 4);
    } else {
        visit_body_nodes(build_rotated_box);
    }
    const auto stationary_node_box = [&](const int node) { return AABB(x[static_cast<std::size_t>(node)], x[static_cast<std::size_t>(node)]); };

    double toi_min = 1.0;
    bool has_collision = false;
    const auto evaluate_nt = [&](int pair_index) {
        CCDResult value;
        const auto consider = [&](bool collision, double toi) {
            value.collision = collision; value.t = toi;
        };

        const NodeTrianglePair& pair = bp_cache.nt_pairs[pair_index];
        const int node = pair.node;
        const int node_rb = owning_rb_for_node(ref_mesh.node_to_rb, node);
        const int triangle_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.tri_v[0]);
        if (node_rb == triangle_rb || (node_rb != rb && triangle_rb != rb))
            return value;
        const std::array<AABB, 4> node_boxes = node_rb == rb ? std::array<AABB, 4>{rotated_node_boxes[static_cast<std::size_t>(node)], stationary_node_box(pair.tri_v[0]), stationary_node_box(pair.tri_v[1]), stationary_node_box(pair.tri_v[2])} : std::array<AABB, 4>{stationary_node_box(node), rotated_node_boxes[static_cast<std::size_t>(pair.tri_v[0])], rotated_node_boxes[static_cast<std::size_t>(pair.tri_v[1])], rotated_node_boxes[static_cast<std::size_t>(pair.tri_v[2])]};
        if (!node_triangle_swept_aabbs_intersect(node_boxes))
            return value;
        double toi = 0.0;
        bool collision;
        if (node_rb == rb)
            collision = point_triangle_rb_rotation_ccd(x[node], x_com, proposed, current, x[pair.tri_v[0]], x[pair.tri_v[1]], x[pair.tri_v[2]], toi);
        else
            collision = point_triangle_rb_rotation_ccd(x[node], x_com, q_reverse, identity, x[pair.tri_v[0]], x[pair.tri_v[1]], x[pair.tri_v[2]], toi);
        consider(collision, toi);
        return value;
    };
    const auto evaluate_ss = [&](int pair_index) {
        CCDResult value;
        const auto consider = [&](bool collision, double toi) {
            value.collision = collision; value.t = toi;
        };

        const SegmentSegmentPair& pair = bp_cache.ss_pairs[pair_index];
        const int first_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.v[0]);
        const int second_edge_rb = owning_rb_for_node(ref_mesh.node_to_rb, pair.v[2]);
        if (first_edge_rb == second_edge_rb || (first_edge_rb != rb && second_edge_rb != rb))
            return value;
        const std::array<AABB, 4> node_boxes = first_edge_rb == rb ? std::array<AABB, 4>{rotated_node_boxes[static_cast<std::size_t>(pair.v[0])], rotated_node_boxes[static_cast<std::size_t>(pair.v[1])], stationary_node_box(pair.v[2]), stationary_node_box(pair.v[3])} : std::array<AABB, 4>{stationary_node_box(pair.v[0]), stationary_node_box(pair.v[1]), rotated_node_boxes[static_cast<std::size_t>(pair.v[2])], rotated_node_boxes[static_cast<std::size_t>(pair.v[3])]};
        if (!segment_segment_swept_aabbs_intersect(node_boxes))
            return value;
        double toi = 0.0;
        bool collision;
        if (first_edge_rb == rb)
            collision = segment_segment_rb_rotation_ccd(x[pair.v[0]], x[pair.v[1]], x_com, proposed, current, x[pair.v[2]], x[pair.v[3]], toi);
        else
            collision = segment_segment_rb_rotation_ccd(x[pair.v[0]], x[pair.v[1]], x_com, q_reverse, identity, x[pair.v[2]], x[pair.v[3]], toi);
        consider(collision, toi);
        return value;
    };
    const int nt_count = static_cast<int>(nt_pair_indices.size());
    ordered_ccd_tasks(nt_count + static_cast<int>(ss_pair_indices.size()), cooperative,
        [&](int i) { return i < nt_count ? evaluate_nt(nt_pair_indices[i])
            : evaluate_ss(ss_pair_indices[i - nt_count]); },
        [&](const CCDResult& value) {
            if (value.collision) {
                has_collision = true;
                toi_min = std::min(toi_min, value.t);
            }
        });

    rotated_node_boxes.swap(reusable_rotated_node_boxes);
    return has_collision ? safety * toi_min : 1.0;
}
