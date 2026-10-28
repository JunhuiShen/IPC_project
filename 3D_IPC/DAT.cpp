#include "DAT.h"

#include "broad_phase.h"
#include "segment_segment_distance.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

static void dat_atomic_min(std::atomic<double>& destination, double value)
{
    double previous = destination.load(std::memory_order_relaxed);
    while (value < previous && !destination.compare_exchange_weak(
        previous, value, std::memory_order_relaxed)) {}
}

template<class Function>
static void dat_for_pairs(std::size_t count, const Function& function)
{
    std::exception_ptr error;
    #pragma omp parallel for schedule(static) if(count >= 128)
    for (std::size_t pair = 0; pair < count; ++pair) {
        try {
            function(pair);
        } catch (...) {
            // Exceptions must not escape an OpenMP worker region.
            #pragma omp critical(dat_pair_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

static Vec3 dat_capped_update(const Vec3& update, double limit)
{
    const double largest = update.cwiseAbs().maxCoeff();
    if (largest == 0) return update;
    // Avoid overflow even when the finite input's norm exceeds DBL_MAX.
    const Vec3 scaled = update / largest;
    const double length = scaled.norm();
    if (largest <= limit / length) return update;
    return scaled * (limit / length);
}

// Work relative to the query vertex instead of subtracting rounded world-space
// closest points. For an exterior projection, search all finite triangle edges:
// barycentric sign regions alone need not pick the closest edge of an obtuse tri.
static Vec3 dat_vertex_triangle_separation(
    const Vec3& vertex, const Vec3& a, const Vec3& b, const Vec3& c)
{
    std::array<Vec3, 3> local{a - vertex, b - vertex, c - vertex};
    double scale = 0;
    for (const auto& p : local) if (!p.allFinite())
        throw std::domain_error("planar_DAT: relative geometry overflow");
    for (const auto& p : local) scale = std::max(scale, p.cwiseAbs().maxCoeff());
    if (scale == 0) return Vec3::Zero();
    // Uniform scaling and stableNorm avoid underflow/overflow of face normals.
    for (auto& p : local) p /= scale;
    const Vec3 cross = (local[1] - local[0]).cross(local[2] - local[0]);
    const double normal_length = cross.stableNorm();
    if (normal_length > 0) {
        const Vec3 normal = cross / normal_length;
        const Vec3 projection = local[0].dot(normal) * normal;
        bool interior = true;
        // Oriented edge tests avoid the squared Gram determinant, which can
        // cancel to zero for a thin, but nondegenerate, triangle.
        for (int edge = 0; edge < 3; ++edge) {
            const Vec3 direction = local[(edge + 1) % 3] - local[edge];
            if (direction.cross(projection - local[edge]).dot(normal) < 0)
                interior = false;
        }
        if (interior) return projection * scale;
    }
    Vec3 closest = Vec3::Zero();
    double minimum = std::numeric_limits<double>::infinity();
    for (int edge = 0; edge < 3; ++edge) {
        double parameter;
        const Vec3 candidate = segment_closest_point(
            Vec3::Zero(), local[edge], local[(edge + 1) % 3], parameter);
        const double distance = candidate.stableNorm();
        if (distance < minimum) {
            minimum = distance;
            closest = candidate;
        }
    }
    if (!std::isfinite(minimum))
        throw std::domain_error("planar_DAT: unresolved vertex/triangle distance");
    return closest * scale;
}

// The first side_size entries form primitive A; the remainder form B.
// separation points from A to B. Every corner of both primitives contributes,
// not just the closest points, so the whole deforming primitive stays on its side.
static void dat_constrain_pair(
    const std::array<int, 4>& nodes, int side_size, const Vec3& separation,
    double gamma, double query_radius, const std::vector<Vec3>& positions,
    const std::vector<Vec3>& updates, std::vector<std::atomic<double>>& factors)
{
    const double distance = separation.stableNorm();
    if (!separation.allFinite() || !std::isfinite(distance))
        throw std::domain_error("planar_DAT: unresolved pair distance");
    if (distance >= query_radius) return;
    if (!(distance > 0))
        throw std::domain_error("planar_DAT: zero-gap pair has no separating plane");
    const Vec3 normal = separation / distance;
    const Vec3& origin = positions[nodes[0]];
    std::array<double, 4> coordinate{}, velocity{};
    double left = -std::numeric_limits<double>::infinity();
    double right = std::numeric_limits<double>::infinity();
    double approach_a = 0, approach_b = 0;
    for (int i = 0; i < 4; ++i) {
        coordinate[i] = (positions[nodes[i]] - origin).dot(normal);
        velocity[i] = updates[nodes[i]].dot(normal);
        if (!std::isfinite(coordinate[i]) || !std::isfinite(velocity[i]))
            throw std::domain_error("planar_DAT: plane projection overflow");
        if (i < side_size) {
            left = std::max(left, coordinate[i]);
            approach_a = std::max(approach_a, velocity[i]);
        } else {
            right = std::min(right, coordinate[i]);
            approach_b = std::max(approach_b, -velocity[i]);
        }
    }
    const double gap = right - left;
    if (!(gap > 0) || !std::isfinite(gap))
        throw std::domain_error("planar_DAT: closest-point normal does not separate the primitives");

    // Use the paper's proportional-gap rule (Sec. 3.3), expressed without its
    // inconsistent printed lambda/sign conventions. For A -> B normal:
    // plane = q_A + a/(a+b) * (q_B-q_A). Side A gets a/(a+b) of the gap.
    // When one side is stationary/retreating its share may be zero; that side
    // is not truncated. This avoids freezing motion toward a stationary face.
    const double largest = std::max(approach_a, approach_b);
    const double a = largest > 0 ? approach_a / largest : 1;
    const double b = largest > 0 ? approach_b / largest : 1;
    const double budget_a = gap * (a / (a + b));
    const double budget_b = gap * (b / (a + b));
    for (int i = 0; i < 4; ++i) {
        const bool side_a = i < side_size;
        const double approach = side_a ? velocity[i] : -velocity[i];
        if (approach <= 0) continue; // parallel or moving away from the plane
        const double margin = side_a
            ? (left - coordinate[i]) + budget_a
            : (coordinate[i] - right) + budget_b;
        // Eq. 16: min(1, gamma*t_hit), applying relaxation exactly once.
        // Compare first to avoid overflow for an almost tangential update.
        const double allowed = gamma * margin;
        if (approach > allowed) {
            const double factor = allowed / approach;
            dat_atomic_min(factors[nodes[i]], std::clamp(factor, 0.0, 1.0));
        }
    }
}

std::vector<Vec3> planar_DAT(
    double gamma, double query_radius, const RefMesh& mesh,
    const std::vector<Vec3>& incremental_update,
    const std::vector<Vec3>& current_positions, double node_box_padding)
{
    const std::size_t count = current_positions.size();
    if (!std::isfinite(gamma) || gamma <= 0 || gamma >= 1
        || !std::isfinite(query_radius) || query_radius <= 0
        || !std::isfinite(node_box_padding) || node_box_padding < 1)
        throw std::invalid_argument("planar_DAT: require 0 < gamma < 1, r_q > 0, and padding >= 1");
    if (incremental_update.size() != count
        || count > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || (mesh.num_positions != 0 && mesh.num_positions != count)
        || mesh.tris.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || mesh.tris.size() % 3 != 0)
        throw std::invalid_argument("planar_DAT: inconsistent vertex or triangle counts");
    const auto valid_node = [count](int node) {
        return node >= 0 && static_cast<std::size_t>(node) < count;
    };
    for (std::size_t i = 0; i < mesh.tris.size(); i += 3) {
        const int a = mesh.tris[i], b = mesh.tris[i + 1], c = mesh.tris[i + 2];
        if (!valid_node(a) || !valid_node(b) || !valid_node(c)
            || a == b || b == c || a == c)
            throw std::invalid_argument("planar_DAT: invalid triangle connectivity");
    }
    for (int node : mesh.tet_nodes) if (!valid_node(node))
        throw std::invalid_argument("planar_DAT: invalid tet-node classification");
    for (int node : mesh.surface_nodes) if (!valid_node(node))
        throw std::invalid_argument("planar_DAT: invalid surface-node classification");
    if (!mesh.tet_nodes.empty()) {
        std::vector<unsigned char> interior(count, 0);
        for (int node : mesh.tet_nodes) interior[node] = 1;
        for (int node : mesh.surface_nodes) interior[node] = 0;
        for (int node : mesh.tris) if (interior[node])
            throw std::invalid_argument("planar_DAT: triangle vertex classified as solid interior");
    }
    const double limit = (0.5 * gamma) * query_radius;
    const double radius = query_radius * node_box_padding;
    if (!(limit > 0) || !std::isfinite(radius))
        throw std::invalid_argument("planar_DAT: displacement limit or box radius out of range");

    std::vector<Vec3> updates(count);
    std::vector<AABB> boxes(count);
    std::vector<std::atomic<double>> factors(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!current_positions[i].allFinite() || !incremental_update[i].allFinite())
            throw std::invalid_argument("planar_DAT: nonfinite position or update");
        updates[i] = dat_capped_update(incremental_update[i], limit);
        boxes[i] = AABB(current_positions[i] - Vec3::Constant(radius),
                        current_positions[i] + Vec3::Constant(radius));
        if (!boxes[i].min.allFinite() || !boxes[i].max.allFinite())
            throw std::invalid_argument("planar_DAT: node box overflow");
        factors[i].store(1.0, std::memory_order_relaxed);
    }
    if (count == 0) return updates;

    // Fresh topology per call. Green triangles and red edges are unions of
    // these blue boxes. Zero additional d_hat padding: r_q already defines
    // the query neighborhood; there is no barrier stiffness/energy here.
    BroadPhase broad_phase;
    if (!mesh.tets.empty() || !mesh.tet_nodes.empty())
        broad_phase.initialize_surface_nodes(boxes, mesh, 0.0,
            BroadPhase::InitializationMode::DeformableSolver);
    else
        broad_phase.initialize(boxes, mesh, 0.0,
            BroadPhase::InitializationMode::DeformableSolver);
    const auto& cache = broad_phase.cache();
    dat_for_pairs(cache.nt_pairs.size(), [&](std::size_t i) {
        const auto& pair = cache.nt_pairs[i];
        const std::array<int, 4> nodes{
            pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
        const Vec3 separation = dat_vertex_triangle_separation(
            current_positions[nodes[0]], current_positions[nodes[1]],
            current_positions[nodes[2]], current_positions[nodes[3]]);
        dat_constrain_pair(nodes, 1, separation, gamma, query_radius,
            current_positions, updates, factors);
    });
    dat_for_pairs(cache.ss_pairs.size(), [&](std::size_t i) {
        const auto& pair = cache.ss_pairs[i];
        const std::array<int, 4> nodes{pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
        const auto distance = segment_segment_distance(
            current_positions[nodes[0]], current_positions[nodes[1]],
            current_positions[nodes[2]], current_positions[nodes[3]]);
        dat_constrain_pair(nodes, 2, -distance.separation, gamma, query_radius,
            current_positions, updates, factors);
    });
    #pragma omp parallel for schedule(static) if(count >= 128)
    for (std::size_t i = 0; i < count; ++i)
        updates[i] *= factors[i].load(std::memory_order_relaxed);
    return updates;
}
